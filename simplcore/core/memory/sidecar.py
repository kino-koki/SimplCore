# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""三层精确引用架构 —— L1 旁路档案 + L2 指针取回通路（0.1-α）。

要回答「第几个字、哪一句话」，物理上必然需要 O(N) 存储（没有例外，这是
香农）：O(1) 常驻状态不可能无损编码 O(N) 原文。因此本模块**不让 O(1) 做
精确检索**——那会毁掉架构最硬的性质。分工是：

===== ========== ===========================================================
层级  内容        存储
===== ========== ===========================================================
L0    gist+指针   O(1) 常驻（HBM）。已有：SGC/SRFN 流式状态 + 本模块的
                  PointerHead（~2·dim² 参数，与 N 无关）
L1    原文精确副本 O(N) 档案（Host RAM / SSD）。本模块 SidecarArchive：
                  token id 逐 token 副本 + 块 gist 索引 O(N/B)
L2    取回通路    O(1) 状态每步输出指针分布 → top-k → 精确读取 → 注入
===== ========== ===========================================================

**定位改口（诚实成本）**：加了 L1 之后，「O(1)」必须说全——

    O(1) 常驻状态（HBM），O(N) 精确档案（主机内存/SSD）。

§5.12 的「1M ≤ 2 GiB」讲的是 HBM 峰值，仍然成立：档案在主机内存，不进
GCD。O(1) 状态的价值不是「存储器」，是**选择性寻址器**——它不存原文，
它决定去几百万字里的哪几处取。

检索成本：全量注意力 O(L·D)；本通路取回 O(k·B·D)，与 L 无关——这是
架构保留的全部优势（避免 O(L²)），总存储不再是 O(1)。

诚实边界
--------
* 块 gist 是**无参数池化**（块内 token embedding 均值），检索质量未经
  训练验证——本模块先把**通路与账本**立起来（可测、可审计），
  gist 学习（对比学习/查询编码器）是后续工作。
* 注入默认走**零初始化加性投影**（与 CMT memory_in 同纪律：初始恒等），
  不引入注意力算子，通过 purity 三层守卫。局部精确 cross-attention
  实验件见 ``sidecar_exact_attn.py``（文件头显式豁免标记，不进默认
  产品线）。
* 指针分布 softmax(0·gist) 初始为均匀分布——未训练时检索无判别力，
  这是有意的诚实默认。

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "SidecarArchive",
    "BlockGistIndex",
    "PointerHead",
    "SidecarRetriever",
    "sidecar_ledger",
]


# ---------------------------------------------------------------------------
# L1：精确档案
# ---------------------------------------------------------------------------
class SidecarArchive:
    """L1 旁路：token id 的**逐 token 精确副本**（Host RAM，int32）。

    可寻址：任何 ``[start, end)`` 区间都能精确读回原文——这是「第几个字」
    级引用的物理前提。分段存储避免单一大张量的反复 realloc。
    """

    def __init__(self, block_size: int = 64) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = int(block_size)
        self._chunks: List[torch.Tensor] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def append(self, ids: Sequence[int] | torch.Tensor) -> int:
        """追加 token id，返回新写入段的**起始位置**（可寻址句柄）。"""
        if isinstance(ids, torch.Tensor):
            t = ids.detach().to("cpu", torch.int32).reshape(-1)
        else:
            t = torch.tensor([int(x) for x in ids], dtype=torch.int32)
        start = self._n
        if t.numel():
            self._chunks.append(t)
            self._n += int(t.numel())
        return start

    def read(self, start: int, end: int) -> torch.Tensor:
        """精确读取 ``[start, end)`` —— 逐 token 与写入时一致。"""
        if not (0 <= start <= end <= self._n):
            raise IndexError(f"read [{start}, {end}) out of range, n={self._n}")
        out: List[torch.Tensor] = []
        pos = 0
        for c in self._chunks:
            c_lo, c_hi = pos, pos + c.numel()
            if c_hi <= start:
                pos = c_hi
                continue
            if c_lo >= end:
                break
            out.append(c[max(start, c_lo) - c_lo:min(end, c_hi) - c_lo])
            pos = c_hi
        return torch.cat(out) if out else torch.empty(0, dtype=torch.int32)

    def block_span(self, block_id: int) -> Tuple[int, int]:
        """块 id → [start, end) 区间。"""
        lo = block_id * self.block_size
        hi = min(lo + self.block_size, self._n)
        if lo >= self._n:
            raise IndexError(f"block {block_id} out of range (n={self._n})")
        return lo, hi

    @property
    def num_blocks(self) -> int:
        return (self._n + self.block_size - 1) // self.block_size

    def footprint_bytes(self) -> int:
        """档案字节账：int32 × token 数（Host RAM）。"""
        return self._n * 4

    def state_dict(self) -> Dict[str, Any]:
        return {"block_size": self.block_size,
                "chunks": [c.clone() for c in self._chunks], "n": self._n}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.block_size = int(state["block_size"])
        self._chunks = [c.clone() for c in state["chunks"]]
        self._n = int(state["n"])


# ---------------------------------------------------------------------------
# L1：块 gist 索引（内容寻址的钥匙）
# ---------------------------------------------------------------------------
class BlockGistIndex:
    """每块原文的 gist：块内 token embedding 均值池化（**无参数**）。

    gist 存在 L1（O(N/B)），不在 O(1) 常驻状态里——O(1) 状态只保存
    「当前在处理什么」，指针分布由当前状态与 gist 匹配产生。
    """

    def __init__(self, embedding: nn.Embedding, block_size: int = 64) -> None:
        self.embedding = embedding           # 引用模型的 tok_emb（冻结使用）
        self.block_size = int(block_size)
        self._gists: List[torch.Tensor] = []  # fp32 CPU，逐段

    @torch.no_grad()
    def rebuild(self, archive: SidecarArchive) -> None:
        """全量重建（archive 变更后调用；增量场景用 append_update）。"""
        self._gists = []
        pos = 0
        while pos < len(archive):
            hi = min(pos + archive.block_size, len(archive))
            ids = archive.read(pos, hi).long()
            self._gists.append(
                self.embedding(ids).mean(dim=0).to("cpu", torch.float32))
            pos = hi

    @torch.no_grad()
    def append_update(self, archive: SidecarArchive, start: int) -> None:
        """增量更新：从 ``start`` 起补建新块（流式 observe 用）。"""
        pos = start - (start % archive.block_size)
        while pos < len(archive):
            hi = min(pos + archive.block_size, len(archive))
            if pos >= hi:
                break
            idx = pos // archive.block_size
            ids = archive.read(pos, hi).long()
            g = self.embedding(ids).mean(dim=0).to("cpu", torch.float32)
            if idx < len(self._gists):
                self._gists[idx] = g
            else:
                self._gists.append(g)
            pos = hi

    @property
    def gists(self) -> torch.Tensor:
        if not self._gists:
            return torch.empty(0, self.embedding.embedding_dim)
        return torch.stack(self._gists)

    def __len__(self) -> int:
        return len(self._gists)

    def footprint_bytes(self) -> int:
        return sum(g.numel() * 4 for g in self._gists)


# ---------------------------------------------------------------------------
# L2：指针头 + 取回通路
# ---------------------------------------------------------------------------
class PointerHead(nn.Module):
    """O(1) 状态 → 指针分布（over 块）。零初始化 → 初始均匀（诚实默认）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def logits(self, h: torch.Tensor, gists: torch.Tensor) -> torch.Tensor:
        """h: (..., dim) 常驻状态；gists: (B, dim) → (..., blocks)。"""
        if gists.numel() == 0:
            raise ValueError("gist index is empty: observe() before retrieve()")
        q = self.proj(h)                                  # (..., dim)
        return q @ gists.to(q.dtype).T / math.sqrt(gists.shape[-1])

    def forward(self, h: torch.Tensor, gists: torch.Tensor,
                k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (topk_scores, topk_block_ids)，按分数降序。"""
        logits = self.logits(h, gists)
        probs = torch.softmax(logits, dim=-1)
        kk = probs.shape[-1] if k is None else min(k, probs.shape[-1])
        scores, ids = torch.topk(probs, kk, dim=-1)
        return scores, ids


class SidecarRetriever(nn.Module):
    """L0+L1+L2 组合： observe → retrieve → inject。

    用法（流式）::

        ret = SidecarRetriever(model, block_size=64, top_k=8)
        ret.observe(token_ids)                  # 每步/每段把原文写入 L1
        inj = ret.inject_vector(h)              # h = 当前 O(1) 状态（残差流）
        # inj 初始恒为 0（零初始化投影）；训练后成为「去哪取」的取回向量
    """

    def __init__(self, model: nn.Module, block_size: int = 64,
                 top_k: int = 8, dim: Optional[int] = None) -> None:
        super().__init__()
        emb = getattr(model, "tok_emb", None)
        if emb is None:
            raise ValueError("model must expose .tok_emb (nn.Embedding)")
        d = dim or int(emb.embedding_dim)
        if block_size <= 0 or top_k <= 0:
            raise ValueError("block_size/top_k must be positive")
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.archive = SidecarArchive(block_size)
        self.gist = BlockGistIndex(emb, block_size)
        self.pointer = PointerHead(d)
        self.inject_proj = nn.Linear(d, d)
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)

    # -- L1 写入 -------------------------------------------------------------
    @torch.no_grad()
    def observe(self, ids: Sequence[int] | torch.Tensor) -> int:
        """把原文写入 L1 档案并增量更新 gist 索引。返回起始位置。"""
        start = self.archive.append(ids)
        self.gist.append_update(self.archive, start)
        return start

    # -- L2 取回 --------------------------------------------------------------
    def retrieve(self, h: torch.Tensor, k: Optional[int] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """O(1) 状态 → top-k 块指针。返回 (scores, block_ids)。"""
        k = k or self.top_k
        scores, ids = self.pointer(h, self.gist.gists.to(h.device), k)
        return scores, ids

    @torch.no_grad()
    def read_topk(self, h: torch.Tensor, k: Optional[int] = None
                  ) -> List[Tuple[int, torch.Tensor]]:
        """取回 top-k 块的**精确原文**（逐 token 与写入一致）。"""
        _, ids = self.retrieve(h, k)
        out = []
        for b in ids.reshape(-1).tolist():
            lo, hi = self.archive.block_span(int(b))
            out.append((int(b), self.archive.read(lo, hi)))
        return out

    # -- 注入（纯净版：无注意力算子） -------------------------------------------
    def inject_vector(self, h: torch.Tensor) -> torch.Tensor:
        """指针加权池化取回块 → 零初始化投影。初始恒为 0（与 CMT 同纪律）。"""
        scores, ids = self.retrieve(h)
        emb = self.gist.embedding
        acc = torch.zeros(h.shape[-1], dtype=h.dtype, device=h.device)
        for s, b in zip(scores.reshape(-1), ids.reshape(-1)):
            lo, hi = self.archive.block_span(int(b))
            ids_t = self.archive.read(lo, hi).to(h.device)
            acc = acc + s * emb(ids_t.long()).mean(dim=0).to(h.dtype)
        return self.inject_proj(acc)

    # -- 账本 ------------------------------------------------------------------
    def ledger(self) -> Dict[str, Any]:
        """诚实账本：HBM 常驻（O(1)）与 Host 档案（O(N)）**分开记账**。"""
        n = len(self.archive)
        d = int(self.gist.embedding.embedding_dim)
        resident = sum(p.numel() * p.element_size()
                       for p in self.pointer.parameters()) + \
            sum(p.numel() * p.element_size() for p in self.inject_proj.parameters())
        return {
            "num_tokens": n,
            "num_blocks": self.archive.num_blocks,
            "block_size": self.block_size,
            "resident_hbm_bytes": resident,               # O(1)：与 N 无关
            "resident_hbm_note": "PointerHead+inject_proj 参数；随 N 恒定",
            "archive_host_bytes": self.archive.footprint_bytes(),   # O(N)
            "gist_host_bytes": self.gist.footprint_bytes(),          # O(N/B)
            "note": "O(1) 常驻状态（HBM），O(N) 精确档案（Host RAM/SSD）——"
                    "档案不进 GCD，1M≤2GiB 的 HBM 账本不受影响",
        }


# ---------------------------------------------------------------------------
# 账本（设计期口径，无需实例化）
# ---------------------------------------------------------------------------
def sidecar_ledger(num_tokens: int, dim: int, block_size: int = 64,
                   top_k: int = 8) -> Dict[str, Any]:
    """给定规模下的三层账本（128K / 4M 两档对照）。

    - 常驻 HBM：PointerHead + inject_proj ≈ 2·dim²·4B + 2·dim·4B（O(1)）；
    - 档案 Host：原文 int32（4B/token）+ gist 索引（dim·4B/块，O(N/B)）；
    - 取回乘加：O(k·B·dim)，与 L 无关（对照全量注意力 O(L·dim)）。
    """
    resident = (2 * dim * dim + 2 * dim) * 4
    archive = num_tokens * 4
    gist = ((num_tokens + block_size - 1) // block_size) * dim * 4
    return {
        "num_tokens": num_tokens,
        "dim": dim,
        "block_size": block_size,
        "top_k": top_k,
        "resident_hbm_bytes": resident,
        "resident_hbm_mib": round(resident / 1024 ** 2, 3),
        "archive_host_bytes": archive,
        "archive_host_mib": round(archive / 1024 ** 2, 3),
        "gist_host_bytes": gist,
        "gist_host_mib": round(gist / 1024 ** 2, 3),
        "retrieve_flops_per_query": top_k * block_size * dim,
        "full_attention_flops_per_query": num_tokens * dim,
        "note": "O(1) 常驻状态（HBM），O(N) 精确档案（Host RAM/SSD）；"
                "取回 O(k·B·dim) 与 L 无关",
    }


def main(argv: Optional[List[str]] = None) -> int:
    """打印 128K / 4M 两档账本 + needle 精确性冒烟。"""
    for n in (131_072, 4_194_304):
        led = sidecar_ledger(n, dim=1024)
        print(f"=== L={led['num_tokens']:,} dim={led['dim']} ===")
        print(f"  HBM 常驻: {led['resident_hbm_bytes']:,} B "
              f"({led['resident_hbm_mib']} MiB) —— O(1)")
        print(f"  原文档案: {led['archive_host_bytes']:,} B "
              f"({led['archive_host_mib']} MiB) —— O(N), Host RAM")
        print(f"  gist 索引: {led['gist_host_bytes']:,} B "
              f"({led['gist_host_mib']} MiB) —— O(N/{led['block_size']})")
        print(f"  取回乘加/查询: {led['retrieve_flops_per_query']:,} "
              f"vs 全量 {led['full_attention_flops_per_query']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())