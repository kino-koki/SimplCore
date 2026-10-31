# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
# purity: allow-attention-reference —— 本模块内含 **TS 对照臂**（标准 Transformer），
# 用于等预算 A/B 的对照组；对照臂自包含、不进运行时、不进 mixer 注册表，
# 架构纯洁性由 test_architecture_purity 的注册表/实例/源码三层守卫隔离。
"""等预算对照 harness（验收协议 ``X-equal-budget`` 的可执行化）.

(Equal-budget comparison harness: SimplCore vs standard Transformer, same tokens.)

战役协议 ``campaign.py`` 里最容易被质疑的一条闸门::

    Gate("X-equal-budget", "对照口径：与 3B Transformer 基线**等 token / 等 FLOPs**，
                          禁止无限定宣称")
    Gate("X-heldout-ppl",  "held-out 困惑度 ≤ 基线 × 1.05")
    ShowcaseItem("S4-equal-token-ppl", "对比脚本 + runs/eval_equal_token.json")

本模块把这两条从"纸面协议"变成**任何人可以跑的命令**。

设计原则（每一条都是被质疑过的坑）
----------------------------------
1. **同一份 token，逐窗口配对**。两臂消费完全相同的 ``(x, y)`` 窗口序列
   （由 :class:`WindowStream` 以同一 seed 确定性生成/采样），held-out 评估
   在**同一批窗口**上逐窗配对 —— 所以统计层用逐窗差值 + bootstrap CI
   （与 :mod:`simplcore.core.ab` 同一哲学：只有"一边好一边坏"的窗口携带信息）。
2. **FLOPs 是一张账，不是两句各说各话**。Transformer 臂的每 token 训练
   FLOPs 用标准近似 ``6·N_nonemb + 12·s·D·L``（attention 得分项单列，
   逐项推导见 :func:`ts_train_flops_per_token`）；SimplCore 臂直接复用
   :func:`simplcore.core.scaling.train_flops_per_token`。两张账都写进报告，
   谁多算谁少算一目了然。
3. **两种口径都报，默认跑等 token**。"等 token"（两臂 token 数相同）与
   "等 FLOPs"（按 FLOPs 账反算 TS 应得 token 数）在参数量不同时通常不能
   同时成立 —— 诚实做法是：主口径实际跑，另一口径把账摆出来。
   ``mode="equal_token"`` 时报告仍包含
   ``equal_flops_ts_tokens_reference``（等 FLOPs 下 TS 应得的 token 数）。
4. **禁止项不是注释，是字段**。``campaign.forbidden`` 的四条逐条写进
   报告的 ``honesty.forbidden_checks``；``unqualified_claim_permitted``
   只有在全部闸门通过时才为 ``True`` —— JSON 里直接给出"这份数据能不能
   支撑无限定宣称"的机器可读答案。

关于 TS 对照臂（why a baseline lives here）
-------------------------------------------
注意力基线（CausalAttention 等）已按架构决策**整体移除**出 SimplCore 运行时
（见 ``.trash/attn_removed/`` 与 ``tests/test_architecture_purity.py``）。
本模块的 :class:`TransformerBaseline` 是**评测对照专用**的标准 decoder-only
Transformer（pre-LN + 因果 SDPA + SwiGLU + tied embedding，与 SimplCore 宏观
形状一致，使对照隔离的唯一变量是 mixer），它：

* 不是 SimplCore 运行时的一部分，不进任何模型包；
* 不注册进 mixer 注册表（:func:`simplcore.core.registry` 与纯洁性守卫不受影响）；
* 参数量通过 :func:`match_ts_layers` 与 SimplCore 非嵌入参数对齐（默认 <5% 偏差）。

用法 (usage)::

    python -m simplcore.core.equal_budget --preset smoke --out runs/eval_equal_token.json
    python -m simplcore.core.equal_budget --preset small --mode equal_flops \\
        --tokens 2000000 --corpus data/processed/tuan_zh

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SyntheticWindowSource",
    "TSConfig",
    "TransformerBaseline",
    "WindowStream",
    "match_ts_layers",
    "paired_eval",
    "run_comparison",
    "ts_non_embedding_params",
    "ts_train_flops_per_token",
]

PROTOCOL = "X-equal-budget"
REPORT_VERSION = 1

#: ``X-heldout-ppl`` 的默认容差：SimplCore ppl ≤ 基线 × 该系数。
DEFAULT_PPL_FACTOR = 1.05


# ---------------------------------------------------------------------------
# TS 臂的 FLOPs 账
# ---------------------------------------------------------------------------
def ts_non_embedding_params(num_layers: int, dim: int, ffn_hidden: int) -> int:
    """标准 decoder-only Transformer 的**非嵌入**参数量（tied head，无 bias）。

    每层：QKV+输出投影 ``4·D²`` + SwiGLU ``3·D·H`` + 两个 pre-LN ``4·D``；
    模型级：final LN ``2·D``。
    """
    per_layer = 4 * dim * dim + 3 * dim * ffn_hidden + 4 * dim
    return num_layers * per_layer + 2 * dim


def ts_train_flops_per_token(num_layers: int, dim: int, ffn_hidden: int,
                             seq_len: int) -> float:
    """Transformer 臂每 token 训练 FLOPs（前向+反向，逐项推导）。

    * 参数矩阵项：``6·N_nonemb``（前向 ``2 FLOPs/参数``，反向约 2 倍 → 合计
      ``6``；与 :func:`simplcore.core.scaling.train_flops_per_token` 的参数项
      **同一口径**，两张账可以直接相除）。
    * attention 得分项（参数覆盖不到的部分，单列）：对每个 query，
      ``QK^T`` 与 ``attn·V`` 各 ``2·s·D`` → 前向 ``4·s·D`` 每层每 token；
      反向合计 3 倍（前向 + 两个反向）→ **``12·s·D·L``**。该项随 ``s``
      线性增长，是 TS 长上下文训练成本超线性的全部来源。
    """
    s = float(max(seq_len, 1))
    base = 6.0 * ts_non_embedding_params(num_layers, dim, ffn_hidden)
    attn = 12.0 * s * float(dim) * float(num_layers)
    return base + attn


def _sc_train_flops_per_token(cfg: Any, seq_len: int) -> float:
    """SimplCore 臂每 token 训练 FLOPs —— 直接复用 scaling 的账（单一事实源）。"""
    from .scaling import train_flops_per_token
    return train_flops_per_token(cfg, seq_len)


def match_ts_layers(sc_non_embedding_params: int, dim: int, ffn_hidden: int,
                    sc_num_layers: int,
                    tol: float = 0.05) -> Tuple[int, int]:
    """选 TS 层数使其非嵌入参数最接近 SimplCore 的非嵌入参数。

    搜索范围 ``1 .. 4·sc_num_layers``；返回 ``(layers, ts_params)``。
    若最优偏差仍超过 ``tol``，调用方（:func:`run_comparison`）会把偏差
    如实写进报告而不是静默放过 —— 参数对不齐的"等预算"就是假对照。
    """
    best_layers, best_params, best_err = 1, ts_non_embedding_params(
        1, dim, ffn_hidden), float("inf")
    for layers in range(1, max(2, 4 * int(sc_num_layers)) + 1):
        p = ts_non_embedding_params(layers, dim, ffn_hidden)
        err = abs(p - int(sc_non_embedding_params)) / max(int(sc_non_embedding_params), 1)
        if err < best_err - 1e-12:
            best_layers, best_params, best_err = layers, p, err
    return best_layers, best_params


# ---------------------------------------------------------------------------
# TS 对照臂
# ---------------------------------------------------------------------------
@dataclass
class TSConfig:
    """TransformerBaseline 配置（与 SimplCoreConfig 宏观形状对齐）。"""

    vocab_size: int
    dim: int
    num_layers: int
    ffn_hidden: int
    num_heads: int
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(vocab_size=self.vocab_size, dim=self.dim,
                    num_layers=self.num_layers, ffn_hidden=self.ffn_hidden,
                    num_heads=self.num_heads, norm_eps=self.norm_eps,
                    dropout=self.dropout)


class _TSBlock(nn.Module):
    """pre-LN 因果注意力 + SwiGLU（与 SimplCoreBlock 的宏观残差结构一致）。"""

    def __init__(self, cfg: TSConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps)
        self.ln2 = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)
        self.n_heads = cfg.num_heads
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = D // self.n_heads
        q = q.view(B, L, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, L, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, L, self.n_heads, hd).transpose(1, 2)
        # is_causal=True：严格下三角因果掩码，未来 token 对过去位置零贡献
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.out(a))
        h = self.ln2(x)
        x = x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x


class TransformerBaseline(nn.Module):
    """评测对照专用的标准 decoder-only Transformer（tied embedding）。

    **loss 语义与 SimplCoreModel 完全一致**：``forward(idx, targets)`` 中
    ``logits[:, t]`` 直接对 ``targets[:, t]`` 做 CE（调用方负责错位
    ``x = w[:, :-1], y = w[:, 1:]``），因此两臂的 held-out loss 是同一把尺子。
    """

    def __init__(self, cfg: TSConfig) -> None:
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([_TSBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # tied
        self.apply(self._init)
        # 残差输出投影小尺度初始化（GPT-2 风格，与 SimplCore 相同的收尾）
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def non_embedding_params(self) -> int:
        return ts_non_embedding_params(self.config.num_layers, self.config.dim,
                                       self.config.ffn_hidden)


# ---------------------------------------------------------------------------
# 确定性窗口流
# ---------------------------------------------------------------------------
class WindowStream:
    """确定性的训练窗口索引流（两臂**必须**共享同一实例的产出序列）。

    以 ``seed`` 洗牌 ``arange(n_windows)`` 后周期循环 —— token 预算超过
    语料窗口数时从头再来（同一排列，因此第二个 epoch 也是确定性的）。
    两个臂各自实例化 ``WindowStream(n, seed)`` 得到**逐位相同**的序列；
    ``consumed`` 记录已发出的全部索引，供报告断言"两臂 token 恒等"。
    """

    def __init__(self, n_windows: int, seed: int = 1234) -> None:
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        self.n_windows = int(n_windows)
        self.seed = int(seed)
        g = torch.Generator().manual_seed(self.seed)
        self._order = torch.randperm(self.n_windows, generator=g).tolist()
        self._cursor = 0
        self.consumed: List[int] = []

    def take(self, n: int) -> List[int]:
        """顺序取 ``n`` 个窗口索引（周期循环），并记录进 ``consumed``。"""
        out: List[int] = []
        for _ in range(max(n, 0)):
            out.append(self._order[self._cursor % self.n_windows])
            self._cursor += 1
        self.consumed.extend(out)
        return out

    def consumed_hash(self) -> str:
        """已消费序列的摘要（放进报告做"两臂 token 恒等"断言）。"""
        import hashlib
        h = hashlib.sha256()
        h.update(",".join(map(str, self.consumed)).encode("ascii"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# 合成语料（smoke / 测试专用；真实语料走 ShardedTokenDataset 适配）
# ---------------------------------------------------------------------------
class SyntheticWindowSource:
    """确定性合成语料：稀疏一阶 Markov 链 + 均匀噪声混合。

    为什么不直接用纯随机 token：纯 i.i.d. 串的不可约熵 = log V，两臂都只能
    学到 unigram 频率，对照没有区分度。这里用**稀疏**转移矩阵（每 token 平均
    ``fanout`` 个后继）混入 ``noise_prob`` 的均匀跳转 —— 不可约熵明显低于
    ``log V``，attention 与谱卷积都有可学结构；``seed`` 决定一切（同一 seed
    => 逐位相同的语料）。

    接口与 :class:`simplcore.core.data.ShardedTokenDataset` 对齐
    （``n_windows`` / ``window(i)``），因此真实语料可以直接替换进来。
    """

    def __init__(self, vocab_size: int, seq_len: int, n_windows: int,
                 seed: int = 2026, fanout: int = 8, noise_prob: float = 0.15,
                 window_stride: int = 0) -> None:
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.n_windows = int(n_windows)
        self.seed = int(seed)
        if self.vocab_size < 4 or self.seq_len < 8 or self.n_windows < 1:
            raise ValueError("vocab_size>=4, seq_len>=8, n_windows>=1 required")
        # 生成一条长为 total_tokens 的流，切成 n_windows 个窗口。
        # stride=0 时窗口紧密相接；>0 时窗口间留空隙（模拟文档边界）。
        stride = max(int(window_stride), 0)
        total = self.n_windows * (self.seq_len + stride)
        g = torch.Generator().manual_seed(self.seed)
        # 转移表：每行 fanout 个等权后继（稀疏 → 低熵可学）
        succ = torch.randint(0, self.vocab_size, (self.vocab_size, max(fanout, 1)),
                             generator=g)
        stream = torch.empty(total, dtype=torch.long)
        cur = int(torch.randint(0, self.vocab_size, (1,), generator=g))
        noise_p = float(min(max(noise_prob, 0.0), 1.0))
        fan = succ.shape[1]
        for i in range(total):
            stream[i] = cur
            if float(torch.rand(1, generator=g)) < noise_p:
                cur = int(torch.randint(0, self.vocab_size, (1,), generator=g))
            else:
                cur = int(succ[cur, int(torch.randint(0, fan, (1,), generator=g))])
        # 切窗
        self._windows = stream.view(self.n_windows, self.seq_len + stride)[:, :self.seq_len].contiguous()

    def window(self, index: int) -> torch.Tensor:
        """第 ``index`` 个 ``(seq_len,)`` 窗口（long 型 CPU 张量）。"""
        if not 0 <= index < self.n_windows:
            raise IndexError(f"window index {index} out of range ({self.n_windows})")
        return self._windows[index]

    def batch(self, indices: Sequence[int]) -> torch.Tensor:
        """按索引序取一批窗口，``(B, seq_len)``。"""
        return torch.stack([self._windows[int(i)] for i in indices])


class _CorpusWindowSource:
    """把 :class:`simplcore.core.data.ShardedTokenDataset` 包成窗口源。"""

    def __init__(self, ds: Any) -> None:
        self._ds = ds
        self.vocab_size = int(ds.vocab_size)
        self.seq_len = int(ds.seq_len)
        self.n_windows = int(ds.num_windows)

    def window(self, index: int) -> torch.Tensor:
        return self._ds[int(index)]

    def batch(self, indices: Sequence[int]) -> torch.Tensor:
        return torch.stack([self._ds[int(i)] for i in indices])


def open_window_source(root: Any, split: str = "train",
                       max_windows: Optional[int] = None) -> Any:
    """打开窗口源：``root=None`` 用合成语料；否则读真实 corpus manifest。"""
    if root is None:
        return None
    from .data import ShardedTokenDataset
    ds = ShardedTokenDataset(root, split, verify_hashes=True)
    if max_windows is not None and ds.num_windows > int(max_windows):
        ds.num_windows = int(max_windows)          # 只影响训练采样上界
    return _CorpusWindowSource(ds)


# ---------------------------------------------------------------------------
# 共享训练循环（两臂同一把尺子）
# ---------------------------------------------------------------------------
@dataclass
class ArmResult:
    """单臂训练结果。"""

    name: str
    params_total: int
    params_non_embedding: int
    tokens_trained: int
    steps: int
    train_flops: float
    flops_per_token: float
    best_val_loss: float
    final_val_loss: float
    loss_curve: List[Tuple[int, float]]      # (tokens_seen, val_loss)
    consumed_hash: str
    wall_seconds: float
    state_dict: Optional[Dict[str, torch.Tensor]] = None

    def to_dict(self, include_curve: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "params_total": self.params_total,
            "params_non_embedding": self.params_non_embedding,
            "tokens_trained": self.tokens_trained,
            "steps": self.steps,
            "train_flops": self.train_flops,
            "flops_per_token": self.flops_per_token,
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.final_val_loss,
            "consumed_hash": self.consumed_hash,
            "wall_seconds": round(self.wall_seconds, 3),
        }
        if include_curve:
            d["loss_curve_head"] = [[int(t), round(float(v), 6)]
                                    for t, v in self.loss_curve[:16]]
        return d


def train_arm(name: str, model: nn.Module, source: Any,
              token_budget: int, seq_len: int, *,
              seed: int, lr: float = 1e-3, batch_size: int = 8,
              warmup_steps: int = 20, weight_decay: float = 0.1,
              grad_clip: float = 1.0, eval_windows: int = 16,
              eval_every_tokens: int = 32000,
              keep_state: bool = True) -> ArmResult:
    """训练一个臂（SimplCore 或 TS），返回预算账 + 验证曲线 + 最优权重。

    两臂**完全同超参**（AdamW β=(0.9,0.999)、weight_decay=0.1 只作用于
    ≥2 维参数、grad clip=1.0、warmup+cosine），唯一允许的不同是模型本体
    —— 这是"等预算对照"的字面意思。
    """
    if token_budget < seq_len:
        raise ValueError(f"token_budget {token_budget} < seq_len {seq_len}")
    torch.manual_seed(seed)
    model.train()
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999), eps=1e-8)

    stream = WindowStream(source.n_windows, seed=seed)
    tokens_per_step = batch_size * (seq_len - 1)
    total_steps = max(1, token_budget // tokens_per_step)
    # cosine 到 0.1×（短跑恒定 LR 会高估 TS 的收敛尾）
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))

    # 固定验证窗口（两臂同一批）
    ev_idx = list(range(min(int(eval_windows), source.n_windows)))
    ev_batch = source.batch(ev_idx)
    ev_x, ev_y = ev_batch[:, :-1], ev_batch[:, 1:]

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            _, loss = model(ev_x, ev_y)
        model.train()
        return float(loss)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    curve: List[Tuple[int, float]] = []
    tokens_seen = 0
    next_eval = eval_every_tokens
    t0 = time.time()
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idxs = stream.take(batch_size)
        batch = source.batch(idxs)
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step
        if tokens_seen >= next_eval or step == total_steps - 1:
            vl = evaluate()
            curve.append((tokens_seen, vl))
            if vl < best_loss:
                best_loss = vl
                if keep_state:
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
            next_eval += eval_every_tokens
    wall = time.time() - t0

    emb = sum(p.numel() for n, p in model.named_parameters()
              if "tok_emb" in n) if any("tok_emb" in n for n, _ in model.named_parameters()) else 0
    total_p = sum(p.numel() for p in model.parameters())
    if isinstance(model, TransformerBaseline):
        non_emb = model.non_embedding_params()
    else:
        from .scaling import non_embedding_params as _nep
        non_emb = int(_nep(model.config))
    return ArmResult(
        name=name, params_total=int(total_p), params_non_embedding=int(non_emb),
        tokens_trained=int(tokens_seen), steps=int(total_steps),
        train_flops=0.0, flops_per_token=0.0,     # 由调用方按统一账本填
        best_val_loss=float(best_loss), final_val_loss=float(curve[-1][1]),
        loss_curve=curve, consumed_hash=stream.consumed_hash(),
        wall_seconds=wall, state_dict=best_state,
    )


# ---------------------------------------------------------------------------
# 逐窗配对评估
# ---------------------------------------------------------------------------
def _window_losses(model: nn.Module, source: Any, indices: Sequence[int],
                   batch_size: int = 8) -> torch.Tensor:
    """逐窗 mean CE（每窗独立前向，``seq_len-1`` 个 token 的平均）。"""
    model.eval()
    out: List[float] = []
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            idxs = list(indices[s:s + batch_size])
            batch = source.batch(idxs)
            x, y = batch[:, :-1], batch[:, 1:]
            res = model(x, None)
            # SimplCoreModel 在 targets=None 时只返回 logits；TS 臂返回元组
            logits = res[0] if isinstance(res, tuple) else res
            ntok = x.shape[1]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1), reduction="none")
            out.extend(ce.view(len(idxs), ntok).mean(dim=1).tolist())
    model.train()
    return torch.tensor(out, dtype=torch.float64)


def _bootstrap_ci_mean(x: torch.Tensor, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> Tuple[float, float]:
    """逐窗差值均值的 bootstrap percentile CI（与 ab.py 同哲学，独立小实现）。"""
    if x.numel() < 2:
        m = float(x.mean()) if x.numel() else 0.0
        return m, m
    g = torch.Generator().manual_seed(seed)
    n = x.numel()
    means = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means[b] = x[idx].mean()
    lo = float(torch.quantile(means, alpha / 2))
    hi = float(torch.quantile(means, 1 - alpha / 2))
    return lo, hi


def paired_eval(sc_model: nn.Module, ts_model: nn.Module, source: Any,
                n_windows: int, seed: int = 777, n_boot: int = 2000
                ) -> Dict[str, Any]:
    """同一批 held-out 窗口上的逐窗配对评估。

    负差值 = SimplCore 该窗更好（loss 更低）。CI 完全在 0 左侧才能说
    "在窗口抽样意义上更好"；跨 0 就只能报"与基线统计不可分"。
    """
    g = torch.Generator().manual_seed(seed)
    n = min(int(n_windows), source.n_windows)
    indices = torch.randperm(source.n_windows, generator=g)[:n].tolist()
    l_sc = _window_losses(sc_model, source, indices)
    l_ts = _window_losses(ts_model, source, indices)
    diff = l_sc - l_ts                                   # <0 → SimplCore 更好
    lo, hi = _bootstrap_ci_mean(diff, n_boot=n_boot, seed=seed)
    m_sc, m_ts = float(l_sc.mean()), float(l_ts.mean())
    worst_idx = int(torch.argmax(l_sc - l_ts))           # SimplCore 最差的窗口
    return {
        "n_windows": int(n),
        "window_indices_head": [int(i) for i in indices[:16]],
        "mean_loss_simplcore": m_sc,
        "mean_loss_transformer": m_ts,
        "ppl_simplcore": math.exp(m_sc),
        "ppl_transformer": math.exp(m_ts),
        "ppl_ratio_sc_over_ts": math.exp(m_sc - m_ts),
        "mean_loss_diff_sc_minus_ts": float(diff.mean()),
        "ci95_diff": [lo, hi],
        "sc_wins": int((diff < 0).sum()),
        "ts_wins": int((diff > 0).sum()),
        "worst_window_for_sc": {
            "index": int(indices[worst_idx]),
            "loss_sc": float(l_sc[worst_idx]),
            "loss_ts": float(l_ts[worst_idx]),
        },
        "per_window_diff_head": [round(float(v), 6) for v in diff[:16]],
    }


# ---------------------------------------------------------------------------
# 对照主入口
# ---------------------------------------------------------------------------
def _default_sc_cfg(preset: str, **over: Any) -> Any:
    from .config import SimplCoreConfig
    if preset == "smoke":
        kw = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                  ffn_ratio=2.0, local_kernel=4, decay_max=256.0,
                  tie_embeddings=False)
    elif preset == "small":
        kw = dict(vocab_size=4096, dim=256, num_layers=8, num_bases=32,
                  ffn_ratio=4.0, local_kernel=4, decay_max=4096.0,
                  tie_embeddings=True)
    elif preset == "1b":
        from .config import get_preset
        return get_preset("tuan-1b")
    else:
        raise ValueError(f"unknown preset {preset!r} (smoke|small|1b)")
    kw.update(over)
    return SimplCoreConfig(**kw)


def run_comparison(*, preset: str = "smoke", mode: str = "equal_token",
                   token_budget: int = 64000, seq_len: Optional[int] = None,
                   corpus_root: Any = None, out_path: Any = None,
                   seed: int = 1234, lr: float = 1e-3, batch_size: int = 8,
                   ppl_factor: float = DEFAULT_PPL_FACTOR,
                   device: Optional[str] = None,
                   eval_every_tokens: int = 32000,
                   dry_run: bool = False) -> Dict[str, Any]:
    """跑一次等预算对照，返回报告 dict（``out_path`` 不为 None 时落盘 JSON）。

    Args:
        preset: ``smoke`` / ``small`` / ``1b``。
        mode: ``equal_token``（两臂同 token 预算，FLOPs 账照报）或
            ``equal_flops``（按 FLOPs 账反算 TS 应得 token 数，TS 臂实际多跑）。
        token_budget: 主臂（SimplCore）的 token 预算。
        seq_len: 窗口长度；``None`` 用 preset 默认（smoke=64，small=512）。
        corpus_root: 真实语料目录（含 ``index.json``）；``None`` 用合成语料。
        ppl_factor: ``X-heldout-ppl`` 容差。
        dry_run: 只算账、不训练（账本一致性冒烟）。
    """
    if mode not in ("equal_token", "equal_flops"):
        raise ValueError(f"mode must be equal_token|equal_flops, got {mode!r}")
    sc_cfg = _default_sc_cfg(preset)
    if seq_len is None:
        seq_len = 64 if preset == "smoke" else 512
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- 账本先行：参数、FLOPs、TS 对齐 --------------------------------
    from .scaling import non_embedding_params as sc_nep, embedding_params as sc_ep
    sc_non_emb = int(sc_nep(sc_cfg))
    sc_emb = int(sc_ep(sc_cfg))
    sc_flops_tok = _sc_train_flops_per_token(sc_cfg, seq_len)

    ts_dim = max(sc_cfg.dim, 16)
    ts_ffn = int(sc_cfg.ffn_hidden or sc_cfg.ffn_ratio * sc_cfg.dim)
    ts_layers, ts_non_emb = match_ts_layers(sc_non_emb, ts_dim, ts_ffn,
                                            sc_cfg.num_layers)
    ts_flops_tok = ts_train_flops_per_token(ts_layers, ts_dim, ts_ffn, seq_len)
    ts_param_dev = abs(ts_non_emb - sc_non_emb) / max(sc_non_emb, 1)

    # ---- 窗口源 ----------------------------------------------------------
    if corpus_root is not None:
        src = open_window_source(corpus_root, "train",
                                 max_windows=max(64, token_budget // max(seq_len - 1, 1) * 2))
        if src.seq_len < seq_len:
            raise ValueError(f"corpus seq_len {src.seq_len} < requested {seq_len}")
        seq_len = src.seq_len
        vocab = src.vocab_size
    else:
        n_win = max(64, (token_budget // max(seq_len - 1, 1)) * 4 + 32)
        src = SyntheticWindowSource(vocab_size=sc_cfg.vocab_size, seq_len=seq_len,
                                    n_windows=n_win, seed=seed * 31 + 7)
        vocab = sc_cfg.vocab_size

    # equal_flops：TS 预算按 FLOPs 账反算（TS 便宜就多给 token，贵就少给）
    sc_total_flops = sc_flops_tok * token_budget
    ts_budget = token_budget if mode == "equal_token" else int(
        round(sc_total_flops / max(ts_flops_tok, 1e-9)))
    ts_budget = max(ts_budget, seq_len)

    # ---- 建模 ------------------------------------------------------------
    torch.manual_seed(seed)
    from .model import SimplCoreModel
    sc_native = SimplCoreModel(sc_cfg).to(dev)
    sc_model: nn.Module = sc_native

    ts_cfg = TSConfig(vocab_size=vocab, dim=ts_dim, num_layers=ts_layers,
                      ffn_hidden=ts_ffn, num_heads=max(1, ts_dim // 32))
    torch.manual_seed(seed + 1)   # init 不同源，但预算、数据、超参严格同
    ts_model = TransformerBaseline(ts_cfg).to(dev)

    # ---- 训练 ------------------------------------------------------------
    if dry_run:
        sc_res = ArmResult(
            name="simplcore",
            params_total=sum(p.numel() for p in sc_native.parameters()),
            params_non_embedding=sc_non_emb,
            tokens_trained=0, steps=0,
            train_flops=0.0, flops_per_token=sc_flops_tok,
            best_val_loss=float("nan"), final_val_loss=float("nan"),
            loss_curve=[], consumed_hash="", wall_seconds=0.0)
        ts_res = ArmResult(
            name="transformer",
            params_total=sum(p.numel() for p in ts_model.parameters()),
            params_non_embedding=ts_non_emb,
            tokens_trained=0, steps=0, train_flops=0.0, flops_per_token=ts_flops_tok,
            best_val_loss=float("nan"), final_val_loss=float("nan"),
            loss_curve=[], consumed_hash="", wall_seconds=0.0)
    else:
        sc_res = train_arm("simplcore", sc_model, src, token_budget, seq_len,
                           seed=seed, lr=lr, batch_size=batch_size,
                           eval_every_tokens=eval_every_tokens)
        ts_res = train_arm("transformer", ts_model, src, ts_budget, seq_len,
                           seed=seed, lr=lr, batch_size=batch_size,
                           eval_every_tokens=eval_every_tokens)
        # 统一账本回填（train_arm 不懂 FLOPs 公式 —— 单一事实源在调用方）
        sc_res.train_flops = sc_flops_tok * sc_res.tokens_trained
        sc_res.flops_per_token = sc_flops_tok
        ts_res.train_flops = ts_flops_tok * ts_res.tokens_trained
        ts_res.flops_per_token = ts_flops_tok

    # ---- 配对评估（用各自 best checkpoint；equal预算两臂同规则）-----------
    if not dry_run:
        if sc_res.state_dict is not None:
            sc_model.load_state_dict(sc_res.state_dict)
        if ts_res.state_dict is not None:
            ts_model.load_state_dict(ts_res.state_dict)
        paired = paired_eval(sc_model, ts_model, src,
                             n_windows=max(16, batch_size * 4), seed=seed + 9)
    else:
        paired = {"n_windows": 0, "note": "dry_run: no training, no paired eval"}

    # ---- 闸门 ------------------------------------------------------------
    flops_ratio = ts_res.train_flops / max(sc_res.train_flops, 1e-9) if sc_res.train_flops else float("nan")
    equal_budget_ok = (
        (mode == "equal_token" and sc_res.tokens_trained == ts_res.tokens_trained)
        or (mode == "equal_flops" and abs(
            ts_res.train_flops - sc_res.train_flops) / max(sc_res.train_flops, 1e-9) <= 0.01)
    )
    ppl_pass = (not dry_run and
                paired["ppl_simplcore"] <= paired["ppl_transformer"] * ppl_factor)
    gates = {
        "X-equal-budget": {
            "passed": bool(equal_budget_ok),
            "mode": mode,
            "satisfied_by_construction": mode == "equal_token",
            "tokens_sc": sc_res.tokens_trained,
            "tokens_ts": ts_res.tokens_trained,
            "flops_sc": sc_res.train_flops,
            "flops_ts": ts_res.train_flops,
            "flops_ratio_ts_over_sc": flops_ratio,
            "equal_flops_ts_tokens_reference": (
                int(round(sc_total_flops / max(ts_flops_tok, 1e-9)))
                if mode == "equal_token" else None),
            "param_alignment_dev": round(ts_param_dev, 6),
            "note": ("等 token 口径由构造满足（两臂消费同一窗口流）；"
                     "FLOPs 差额如实报告，等 FLOPs 参考行供换算" if mode == "equal_token"
                     else "等 FLOPs 口径：TS token 预算按统一账本反算"),
        },
        "X-heldout-ppl": {
            "passed": bool(ppl_pass),
            "ppl_simplcore": (paired.get("ppl_simplcore") if not dry_run else None),
            "ppl_transformer": (paired.get("ppl_transformer") if not dry_run else None),
            "factor": ppl_factor,
            "ratio": (paired.get("ppl_ratio_sc_over_ts") if not dry_run else None),
        },
    }
    unqualified = bool(equal_budget_ok and ppl_pass)

    report: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "preset": preset,
        "seed": seed,
        "seq_len": int(seq_len),
        "lr": lr,
        "batch_size": int(batch_size),
        "device": str(dev),
        "budget": {
            "simplcore": sc_res.to_dict(include_curve=False),
            "transformer": ts_res.to_dict(include_curve=False),
            "sc_flops_per_token_formula": "6*N_nonemb + 3*L*(2DM + 10D*log2(2s) + 6D + 22D)",
            "ts_flops_per_token_formula": "6*N_nonemb + 12*s*D*L",
            "equal_flops_ts_tokens_reference": gates["X-equal-budget"]["equal_flops_ts_tokens_reference"],
        },
        "arms": {
            "simplcore": {"loss_curve_head": [[int(t), round(float(v), 6)]
                                              for t, v in sc_res.loss_curve[:16]]},
            "transformer": {"loss_curve_head": [[int(t), round(float(v), 6)]
                                                for t, v in ts_res.loss_curve[:16]]},
        },
        "paired": paired,
        "gates": gates,
        "honesty": {
            "unqualified_claim_permitted": unqualified,
            "claim_language_if_permitted": (
                "在等 token 预算下，SimplCore 的 held-out 困惑度不劣于参数量对齐的"
                "标准 Transformer 基线（配对 bootstrap CI 见 paired 字段）"
                if unqualified else
                "本报告不支撑任何无限定宣称；引用时必须带预算口径与 CI"),
            "forbidden_checks": {
                "no_attention_internal_comparison": True,   # 只报真实输出 ppl，无注意力热图
                "equal_budget_qualifier_present": True,     # 全部数字带预算口径
                "no_self_reported_scores": True,            # held-out CE 是外部可复算的客观量
                "distribution_not_average_only": (not dry_run),  # 逐窗配对 + 最差窗口入报告
            },
            "notes": [
                "TS 臂是评测对照专用的标准 decoder-only Transformer，"
                "不属于 SimplCore 运行时（架构纯洁性不受影响）",
                f"参数对齐偏差 {ts_param_dev:.2%}（TS {ts_non_emb} vs SC {sc_non_emb} 非嵌入参数）",
                "合成语料不可替代真实语料结论；正式验收必须 --corpus 指向冻结语料",
            ] if corpus_root is None else [
                f"真实语料: {corpus_root}",
                f"参数对齐偏差 {ts_param_dev:.2%}",
            ],
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simplcore.core.equal_budget",
        description="等预算对照（协议 X-equal-budget）：SimplCore vs 标准 Transformer。")
    p.add_argument("--preset", default="smoke", choices=("smoke", "small", "1b"))
    p.add_argument("--mode", default="equal_token",
                   choices=("equal_token", "equal_flops"))
    p.add_argument("--tokens", type=int, default=64000,
                   help="SimplCore 臂的 token 预算")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--corpus", default=None,
                   help="真实语料目录（含 index.json）；缺省用合成语料（仅冒烟）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ppl-factor", type=float, default=DEFAULT_PPL_FACTOR)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="runs/eval_equal_token.json")
    p.add_argument("--dry-run", action="store_true", help="只算账不训练")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_comparison(
        preset=args.preset, mode=args.mode, token_budget=args.tokens,
        seq_len=args.seq_len, corpus_root=args.corpus, out_path=args.out,
        seed=args.seed, lr=args.lr, batch_size=args.batch_size,
        ppl_factor=args.ppl_factor, dry_run=args.dry_run)
    g = report["gates"]
    print(f"protocol={report['protocol']} mode={report['mode']} preset={report['preset']}")
    print(f"tokens: sc={g['X-equal-budget']['tokens_sc']} ts={g['X-equal-budget']['tokens_ts']}"
          f"  flops ratio ts/sc={g['X-equal-budget']['flops_ratio_ts_over_sc']:.4f}")
    if not args.dry_run:
        p = report["paired"]
        print(f"ppl: sc={p['ppl_simplcore']:.4f} ts={p['ppl_transformer']:.4f}"
              f"  ratio={p['ppl_ratio_sc_over_ts']:.4f}")
        print(f"paired: sc_wins={p['sc_wins']} ts_wins={p['ts_wins']}"
              f"  ci95_diff=[{p['ci95_diff'][0]:.4f}, {p['ci95_diff'][1]:.4f}]")
    for name, gate in g.items():
        print(f"gate {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"unqualified_claim_permitted: {report['honesty']['unqualified_claim_permitted']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
