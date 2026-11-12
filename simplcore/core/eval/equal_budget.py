# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
# purity: allow-attention-reference —— 本模块内含 **TS 对照臂**（标准 Transformer），
# 用于等预算 A/B 的对照组；对照臂自包含、不进运行时、不进 mixer 注册表，
# 架构纯洁性由 test_architecture_purity 的注册表/实例/源码三层守卫隔离。
"""等预算对照 harness（验收协议 ``X-equal-budget`` 的可执行化）.

(Equal-budget comparison harness: SimplCore vs Transformer or Mamba2.)

战役协议 ``campaign.py`` 里最容易被质疑的一条闸门::

    Gate("X-equal-budget", "对照口径：与参数对齐的基线等 token（或可核算时等 FLOPs），
                          禁止无限定宣称")
    Gate("X-heldout-ppl",  "held-out 困惑度 ≤ 基线 × 1.05")
    ShowcaseItem("S4-equal-token-ppl", "对比脚本 + runs/eval_equal_token.json")

本模块把这两条从"纸面协议"变成**任何人可以跑的命令**。

设计原则（每一条都是被质疑过的坑）
----------------------------------
1. **训练与验证严格分离**。训练窗口只来自 train；验证与逐窗配对窗口只来自
   独立 val split。合成冒烟使用独立 seed 生成的验证语料。
2. **同一份 token，逐窗口配对**。两臂消费完全相同的 ``(x, y)`` 窗口序列
   （由 :class:`WindowStream` 以同一 seed 确定性生成/采样），held-out 评估
   在**同一批窗口**上逐窗配对 —— 所以统计层用逐窗差值 + bootstrap CI
   （与 :mod:`simplcore.core.ab` 同一哲学：只有"一边好一边坏"的窗口携带信息）。
3. **FLOPs 是一张账，不是两句各说各话**。Transformer 臂的每 token 训练
   FLOPs 用标准近似 ``6·N_nonemb + 12·s·D·L``（attention 得分项单列，
   逐项推导见 :func:`ts_train_flops_per_token`）；SimplCore 臂直接复用
   :func:`simplcore.core.scaling.train_flops_per_token`。两张账都写进报告，
   谁多算谁少算一目了然。
4. **两种口径都报，默认跑等 token**。"等 token"（两臂 token 数相同）与
   "等 FLOPs"（按 FLOPs 账反算 TS 应得 token 数）在参数量不同时通常不能
   同时成立 —— 诚实做法是：主口径实际跑，另一口径把账摆出来。
   ``mode="equal_token"`` 时报告仍包含
   ``equal_flops_ts_tokens_reference``（等 FLOPs 下 TS 应得的 token 数）。
5. **禁止项不是注释，是字段**。``campaign.forbidden`` 的四条逐条写进
   报告的 ``honesty.forbidden_checks``；``unqualified_claim_permitted``
   只有在全部闸门通过时才为 ``True`` —— JSON 里直接给出"这份数据能不能
   支撑无限定宣称"的机器可读答案。

关于对照臂（why baselines live here）
--------------------------------------
注意力基线（CausalAttention 等）已按架构决策**整体移除**出 SimplCore 运行时
（见 ``.trash/attn_removed/`` 与 ``tests/test_architecture_purity.py``）。
本模块的 :class:`TransformerBaseline` 与 :class:`Mamba2Baseline` 都是**评测对照专用**
基线：Transformer 使用 pre-LN + 因果 SDPA + SwiGLU；Mamba2 使用 Hugging Face
``Mamba2ForCausalLM`` 参考实现。两者：

* 不是 SimplCore 运行时的一部分，不进任何模型包；
* 不注册进 mixer 注册表（:func:`simplcore.core.registry` 与纯洁性守卫不受影响）；
* 参数量通过各自的匹配器与 SimplCore 非嵌入参数对齐；
* 不将不同架构的 FLOPs 当成精确硬件成本。Mamba2 当前仅允许等 token 模式；
  FLOPs 近似账不能替代实测吞吐。

用法 (usage)::

    simplcore-equal-budget --preset smoke --out runs/eval_equal_token.json
    simplcore-equal-budget --preset small --mode equal_flops \\
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
    "Mamba2Baseline",
    "Mamba2BudgetConfig",
    "SyntheticWindowSource",
    "TSConfig",
    "TransformerBaseline",
    "WindowStream",
    "match_ts_layers",
    "match_mamba2_config",
    "mamba2_non_embedding_params",
    "paired_eval",
    "run_comparison",
    "run_multi_seed_comparison",
    "ts_non_embedding_params",
    "ts_train_flops_per_token",
]

PROTOCOL = "X-equal-budget"
REPORT_VERSION = 2

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


@dataclass(frozen=True)
class Mamba2BudgetConfig:
    """Small, explicit subset of the Hugging Face Mamba2 configuration."""

    num_layers: int
    hidden_size: int
    state_size: int
    expand: int
    num_heads: int
    head_dim: int
    n_groups: int = 1
    conv_kernel: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "state_size": self.state_size,
            "expand": self.expand,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "n_groups": self.n_groups,
            "conv_kernel": self.conv_kernel,
            "use_bias": False,
            "use_conv_bias": True,
            "tie_word_embeddings": True,
        }


def mamba2_non_embedding_params(cfg: Mamba2BudgetConfig) -> int:
    """Count non-embedding weights in the HF Mamba2 reference model.

    This mirrors the shipped HF implementation with ``use_bias=False`` and
    ``use_conv_bias=True``. Tests compare it against instantiated state dicts.
    """
    inner = cfg.hidden_size * cfg.expand
    conv_channels = inner + 2 * cfg.n_groups * cfg.state_size
    in_proj_channels = 2 * inner + 2 * cfg.n_groups * cfg.state_size + cfg.num_heads
    per_layer = (
        cfg.hidden_size
        + 3 * cfg.num_heads
        + conv_channels * (cfg.conv_kernel + 1)
        + in_proj_channels * cfg.hidden_size
        + inner
        + cfg.hidden_size * inner
    )
    return cfg.num_layers * per_layer + cfg.hidden_size


def match_mamba2_config(sc_non_embedding_params: int, dim: int,
                        sc_num_layers: int, *, tol: float = 0.05
                        ) -> Tuple[Mamba2BudgetConfig, int]:
    """Select an HF Mamba2 shape close to SimplCore's non-embedding budget."""
    best_cfg: Optional[Mamba2BudgetConfig] = None
    best_params, best_err = 0, float("inf")
    head_dims = (64, 32, 16, 8, 4, 2, 1)
    for expand in (2,):
        inner = dim * expand
        head_dim = next((candidate for candidate in head_dims
                         if inner % candidate == 0), 1)
        heads = inner // head_dim
        for state_size in (1, 2, 4, 8, 16, 32, 64, 128):
            for layers in range(1, max(2, 4 * int(sc_num_layers)) + 1):
                cfg = Mamba2BudgetConfig(
                    num_layers=layers, hidden_size=dim,
                    state_size=state_size, expand=expand,
                    num_heads=heads, head_dim=head_dim)
                params = mamba2_non_embedding_params(cfg)
                err = abs(params - int(sc_non_embedding_params)) / max(
                    int(sc_non_embedding_params), 1)
                if err < best_err - 1e-12:
                    best_cfg, best_params, best_err = cfg, params, err
    assert best_cfg is not None
    if best_err > tol:
        raise ValueError(
            "cannot match Mamba2 non-embedding parameter budget within "
            f"{tol:.1%}: SimplCore={sc_non_embedding_params}, "
            f"nearest Mamba2={best_params} ({best_err:.1%} deviation)")
    return best_cfg, best_params


class Mamba2Baseline(nn.Module):
    """Evaluation-only wrapper around the Hugging Face Mamba2 reference model.

    Targets follow the SimplCore harness contract: the caller provides already
    shifted targets, and this wrapper computes same-position cross entropy
    without asking Transformers to shift labels a second time.
    """

    def __init__(self, vocab_size: int, cfg: Mamba2BudgetConfig) -> None:
        super().__init__()
        try:
            from transformers import Mamba2Config as HFMamba2Config
            from transformers import Mamba2ForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "baseline='mamba2' requires transformers with Mamba2 support "
                "(transformers>=4.45 or a compatible release)"
            ) from exc

        model_cfg = HFMamba2Config(
            vocab_size=vocab_size,
            hidden_size=cfg.hidden_size,
            state_size=cfg.state_size,
            num_hidden_layers=cfg.num_layers,
            conv_kernel=cfg.conv_kernel,
            expand=cfg.expand,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            n_groups=cfg.n_groups,
            chunk_size=256,
            use_bias=False,
            use_conv_bias=True,
            tie_word_embeddings=True,
        )
        self.config = model_cfg
        self.budget_config = cfg
        self.model = Mamba2ForCausalLM(model_cfg)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        result = self.model(input_ids=idx, use_cache=False)
        logits = result.logits
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100)
        return logits, loss

    def non_embedding_params(self) -> int:
        return mamba2_non_embedding_params(self.budget_config)


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
                 window_stride: int = 0,
                 transition_seed: Optional[int] = None) -> None:
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.n_windows = int(n_windows)
        self.seed = int(seed)
        self.transition_seed = (self.seed if transition_seed is None
                                else int(transition_seed))
        self.fanout = max(int(fanout), 1)
        self.noise_prob = float(min(max(noise_prob, 0.0), 1.0))
        self.window_stride = max(int(window_stride), 0)
        if self.vocab_size < 4 or self.seq_len < 8 or self.n_windows < 1:
            raise ValueError("vocab_size>=4, seq_len>=8, n_windows>=1 required")
        # 生成一条长为 total_tokens 的流，切成 n_windows 个窗口。
        # stride=0 时窗口紧密相接；>0 时窗口间留空隙（模拟文档边界）。
        stride = self.window_stride
        total = self.n_windows * (self.seq_len + stride)
        g = torch.Generator().manual_seed(self.seed)
        transition_g = (g if transition_seed is None else
                        torch.Generator().manual_seed(self.transition_seed))
        # 转移表：每行 fanout 个等权后继（稀疏 → 低熵可学）
        succ = torch.randint(0, self.vocab_size, (self.vocab_size, self.fanout),
                             generator=transition_g)
        stream = torch.empty(total, dtype=torch.long)
        cur = int(torch.randint(0, self.vocab_size, (1,), generator=g))
        noise_p = self.noise_prob
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
    from ..training.data import ShardedTokenDataset
    ds = ShardedTokenDataset(root, split, verify_hashes=True)
    if max_windows is not None and ds.num_windows > int(max_windows):
        ds.num_windows = int(max_windows)          # 只影响训练采样上界
    return _CorpusWindowSource(ds)


def window_source_fingerprint(source: Any) -> str:
    """Stable identity for the data split used by an experiment."""
    import hashlib

    metadata = {
        "vocab_size": int(source.vocab_size),
        "seq_len": int(source.seq_len),
        "n_windows": int(source.n_windows),
    }
    dataset = getattr(source, "_ds", None)
    if dataset is not None:
        split = dataset.manifest["splits"][dataset.split]
        metadata["split"] = dataset.split
        metadata["shards"] = [
            {"path": row["path"], "sha256": row.get("sha256")}
            for row in split["files"]
        ]
    else:
        metadata["seed"] = int(source.seed)
    encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_source_shard_digests(source: Any) -> frozenset[str]:
    dataset = getattr(source, "_ds", None)
    if dataset is not None:
        split = dataset.manifest["splits"][dataset.split]
        digests = [row.get("sha256") for row in split["files"]]
        if not digests or any(not digest for digest in digests):
            raise ValueError(
                f"frozen corpus split {dataset.split!r} must include SHA-256 "
                "for every shard before a parity comparison")
        return frozenset(digests)
    return frozenset()


def window_source_content_fingerprint(source: Any) -> str:
    """Fingerprint token content without split names, so copied splits collide."""
    import hashlib

    dataset = getattr(source, "_ds", None)
    if dataset is not None:
        digests = _window_source_shard_digests(source)
        payload = {
            "vocab_size": int(source.vocab_size),
            "seq_len": int(source.seq_len),
            "shard_sha256": sorted(digests),
        }
    else:
        payload = {
            "vocab_size": int(source.vocab_size),
            "seq_len": int(source.seq_len),
            "seed": int(source.seed),
            "transition_seed": int(source.transition_seed),
            "n_windows": int(source.n_windows),
            "fanout": int(source.fanout),
            "noise_prob": float(source.noise_prob),
            "window_stride": int(source.window_stride),
        }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    train_flops: Optional[float]
    flops_per_token: Optional[float]
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
              eval_source: Optional[Any] = None,
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
    effective_warmup_steps = min(warmup_steps, max(1, total_steps // 10))
    # cosine 到 0.1×（短跑恒定 LR 会高估 TS 的收敛尾）
    def lr_at(step: int) -> float:
        if step < effective_warmup_steps:
            return lr * (step + 1) / effective_warmup_steps
        t = ((step - effective_warmup_steps)
             / max(total_steps - effective_warmup_steps, 1))
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))

    # Validation must come from an independent split/source when supplied.
    validation = source if eval_source is None else eval_source
    ev_idx = list(range(min(int(eval_windows), validation.n_windows)))
    if not ev_idx:
        raise ValueError("validation source must contain at least one window")
    model_device = next(model.parameters()).device
    ev_batch = validation.batch(ev_idx).to(model_device)
    ev_x, ev_y = ev_batch[:, :-1], ev_batch[:, 1:]

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            _, loss = model(ev_x, ev_y)
        model.train()
        return float(loss)

    best_loss = evaluate()
    best_state: Optional[Dict[str, torch.Tensor]] = (
        {k: v.detach().clone() for k, v in model.state_dict().items()}
        if keep_state else None)
    curve: List[Tuple[int, float]] = [(0, best_loss)]
    tokens_seen = 0
    next_eval = eval_every_tokens
    t0 = time.time()
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idxs = stream.take(batch_size)
        batch = source.batch(idxs).to(model_device)
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
    if hasattr(model, "non_embedding_params"):
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
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            idxs = list(indices[s:s + batch_size])
            batch = source.batch(idxs).to(model_device)
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
                n_windows: int, seed: int = 777, n_boot: int = 2000,
                baseline_name: str = "transformer"
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
    m_sc, m_baseline = float(l_sc.mean()), float(l_ts.mean())
    worst_idx = int(torch.argmax(l_sc - l_ts))           # SimplCore 最差的窗口
    result = {
        "baseline_name": baseline_name,
        "n_windows": int(n),
        "window_indices_head": [int(i) for i in indices[:16]],
        "mean_loss_simplcore": m_sc,
        "mean_loss_baseline": m_baseline,
        "ppl_simplcore": math.exp(m_sc),
        "ppl_baseline": math.exp(m_baseline),
        "ppl_ratio_sc_over_baseline": math.exp(m_sc - m_baseline),
        "mean_loss_diff_sc_minus_baseline": float(diff.mean()),
        "ci95_diff": [lo, hi],
        "sc_wins": int((diff < 0).sum()),
        "baseline_wins": int((diff > 0).sum()),
        "ts_wins": int((diff > 0).sum()),
        "worst_window_for_sc": {
            "index": int(indices[worst_idx]),
            "loss_sc": float(l_sc[worst_idx]),
            "loss_baseline": float(l_ts[worst_idx]),
        },
        "per_window_diff_head": [round(float(v), 6) for v in diff[:16]],
    }
    if baseline_name == "transformer":
        result["mean_loss_diff_sc_minus_ts"] = float(diff.mean())
        result["mean_loss_transformer"] = m_baseline
        result["ppl_transformer"] = math.exp(m_baseline)
        result["ppl_ratio_sc_over_ts"] = math.exp(m_sc - m_baseline)
    return result


# ---------------------------------------------------------------------------
# 对照主入口
# ---------------------------------------------------------------------------
def _default_sc_cfg(preset: str, **over: Any) -> Any:
    from ..model.config import SimplCoreConfig
    if preset == "smoke":
        kw = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                  ffn_ratio=2.0, local_kernel=4, decay_max=256.0,
                  tie_embeddings=False)
    elif preset == "small":
        kw = dict(vocab_size=4096, dim=256, num_layers=8, num_bases=32,
                  ffn_ratio=4.0, local_kernel=4, decay_max=4096.0,
                  tie_embeddings=True)
    elif preset == "1b":
        from ..model.config import get_preset
        return get_preset("tuan-1b")
    else:
        raise ValueError(f"unknown preset {preset!r} (smoke|small|1b)")
    kw.update(over)
    return SimplCoreConfig(**kw)


def run_comparison(*, preset: str = "smoke", mode: str = "equal_token",
                   baseline: str = "transformer",
                   token_budget: int = 64000, seq_len: Optional[int] = None,
                   corpus_root: Any = None, out_path: Any = None,
                   seed: int = 1234, lr: float = 1e-3, batch_size: int = 8,
                   data_seed: int = 20261111,
                   ppl_factor: float = DEFAULT_PPL_FACTOR,
                   device: Optional[str] = None,
                   eval_every_tokens: int = 32000,
                   dry_run: bool = False) -> Dict[str, Any]:
    """跑一次等预算对照，返回报告 dict（``out_path`` 不为 None 时落盘 JSON）。

    Args:
        preset: ``smoke`` / ``small`` / ``1b``。
        baseline: ``transformer`` or evaluation-only Hugging Face ``mamba2``.
        mode: ``equal_token``（两臂同 token 预算，FLOPs 账照报）或
            ``equal_flops``（仅 Transformer；按 FLOPs 账反算基线 token 数）。
        token_budget: 主臂（SimplCore）的 token 预算。
        seq_len: 窗口长度；``None`` 用 preset 默认（smoke=64，small=512）。
        corpus_root: 真实语料目录（含 ``index.json``）；``None`` 用合成语料。
        ppl_factor: ``X-heldout-ppl`` 容差。
        dry_run: 只算账、不训练（账本一致性冒烟）。
    """
    if mode not in ("equal_token", "equal_flops"):
        raise ValueError(f"mode must be equal_token|equal_flops, got {mode!r}")
    if baseline not in ("transformer", "mamba2"):
        raise ValueError(f"baseline must be transformer|mamba2, got {baseline!r}")
    if baseline == "mamba2" and mode != "equal_token":
        raise ValueError(
            "baseline='mamba2' currently supports equal_token only; "
            "Mamba2 FLOPs are not estimated as a hardware-comparable budget")
    sc_cfg = _default_sc_cfg(preset)
    if seq_len is None:
        seq_len = 64 if preset == "smoke" else 512
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- 账本先行：参数、FLOPs、TS 对齐 --------------------------------
    from .scaling import non_embedding_params as sc_nep, embedding_params as sc_ep
    sc_non_emb = int(sc_nep(sc_cfg))
    sc_emb = int(sc_ep(sc_cfg))
    ts_dim = max(sc_cfg.dim, 16)
    ts_ffn = int(sc_cfg.ffn_hidden or sc_cfg.ffn_ratio * sc_cfg.dim)
    mamba_cfg = None
    if baseline == "transformer":
        ts_layers, ts_non_emb = match_ts_layers(sc_non_emb, ts_dim, ts_ffn,
                                                sc_cfg.num_layers)
        comparator_cfg: Any = TSConfig(
            vocab_size=sc_cfg.vocab_size, dim=ts_dim, num_layers=ts_layers,
            ffn_hidden=ts_ffn, num_heads=max(1, ts_dim // 32))
        ts_flops_tok: Optional[float] = ts_train_flops_per_token(
            ts_layers, ts_dim, ts_ffn, seq_len)
    else:
        mamba_cfg, ts_non_emb = match_mamba2_config(
            sc_non_emb, ts_dim, sc_cfg.num_layers)
        comparator_cfg = mamba_cfg
        ts_flops_tok = None
    ts_param_dev = abs(ts_non_emb - sc_non_emb) / max(sc_non_emb, 1)

    # ---- 窗口源 ----------------------------------------------------------
    if corpus_root is not None:
        src = open_window_source(corpus_root, "train",
                                 max_windows=max(64, token_budget // max(seq_len - 1, 1) * 2))
        eval_src = open_window_source(corpus_root, "val")
        if src.seq_len < seq_len:
            raise ValueError(f"corpus seq_len {src.seq_len} < requested {seq_len}")
        seq_len = src.seq_len
        vocab = src.vocab_size
        sc_cfg.vocab_size = vocab
        sc_cfg.validate()
    else:
        n_win = max(64, (token_budget // max(seq_len - 1, 1)) * 4 + 32)
        src = SyntheticWindowSource(vocab_size=sc_cfg.vocab_size, seq_len=seq_len,
                                    n_windows=n_win, seed=data_seed)
        eval_src = SyntheticWindowSource(
            vocab_size=sc_cfg.vocab_size, seq_len=seq_len,
            n_windows=max(64, batch_size * 8), seed=data_seed + 1,
            transition_seed=data_seed)
        vocab = sc_cfg.vocab_size
    if baseline == "transformer":
        ts_flops_tok = ts_train_flops_per_token(
            ts_layers, ts_dim, ts_ffn, seq_len)
    train_fingerprint = window_source_fingerprint(src)
    eval_fingerprint = window_source_fingerprint(eval_src)
    train_content_fingerprint = window_source_content_fingerprint(src)
    eval_content_fingerprint = window_source_content_fingerprint(eval_src)
    if train_content_fingerprint == eval_content_fingerprint:
        raise ValueError("training and validation token content must be disjoint")
    train_shards = _window_source_shard_digests(src)
    validation_shards = _window_source_shard_digests(eval_src)
    if train_shards.intersection(validation_shards):
        raise ValueError("training and validation splits share one or more content shards")
    sc_flops_tok = _sc_train_flops_per_token(sc_cfg, seq_len)

    # equal_flops：TS 预算按 FLOPs 账反算（TS 便宜就多给 token，贵就少给）
    sc_total_flops = sc_flops_tok * token_budget
    ts_budget = token_budget if mode == "equal_token" else int(
        round(sc_total_flops / max(ts_flops_tok or 0.0, 1e-9)))
    ts_budget = max(ts_budget, seq_len)

    # ---- 建模 ------------------------------------------------------------
    torch.manual_seed(seed)
    from ..model.model import SimplCoreModel
    sc_native = SimplCoreModel(sc_cfg).to(dev)
    sc_model: nn.Module = sc_native

    torch.manual_seed(seed + 1)   # init 不同源，但预算、数据、超参严格同
    if baseline == "transformer":
        comparator_cfg = TSConfig(
            vocab_size=vocab, dim=ts_dim, num_layers=ts_layers,
            ffn_hidden=ts_ffn, num_heads=max(1, ts_dim // 32))
        ts_model: nn.Module = TransformerBaseline(comparator_cfg).to(dev)
    else:
        assert mamba_cfg is not None
        ts_model = Mamba2Baseline(vocab, mamba_cfg).to(dev)
        actual_non_emb = sum(
            p.numel() for name, p in ts_model.model.named_parameters()
            if "embeddings.weight" not in name and "lm_head.weight" not in name)
        if actual_non_emb != ts_non_emb:
            raise RuntimeError(
                "Mamba2 parameter formula disagrees with instantiated model: "
                f"formula={ts_non_emb}, actual={actual_non_emb}")

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
            name=baseline,
            params_total=sum(p.numel() for p in ts_model.parameters()),
            params_non_embedding=ts_non_emb,
            tokens_trained=0, steps=0,
            train_flops=(0.0 if ts_flops_tok is not None else None),
            flops_per_token=ts_flops_tok,
            best_val_loss=float("nan"), final_val_loss=float("nan"),
            loss_curve=[], consumed_hash="", wall_seconds=0.0)
    else:
        sc_res = train_arm("simplcore", sc_model, src, token_budget, seq_len,
                           eval_source=eval_src,
                           seed=seed, lr=lr, batch_size=batch_size,
                           eval_every_tokens=eval_every_tokens)
        ts_res = train_arm(baseline, ts_model, src, ts_budget, seq_len,
                           eval_source=eval_src,
                           seed=seed, lr=lr, batch_size=batch_size,
                           eval_every_tokens=eval_every_tokens)
        # 统一账本回填（train_arm 不懂 FLOPs 公式 —— 单一事实源在调用方）
        sc_res.train_flops = sc_flops_tok * sc_res.tokens_trained
        sc_res.flops_per_token = sc_flops_tok
        if ts_flops_tok is not None:
            ts_res.train_flops = ts_flops_tok * ts_res.tokens_trained
            ts_res.flops_per_token = ts_flops_tok
        else:
            ts_res.train_flops = None
            ts_res.flops_per_token = None

    # ---- 配对评估（用各自 best checkpoint；equal预算两臂同规则）-----------
    if not dry_run:
        if sc_res.state_dict is not None:
            sc_model.load_state_dict(sc_res.state_dict)
        if ts_res.state_dict is not None:
            ts_model.load_state_dict(ts_res.state_dict)
        paired = paired_eval(sc_model, ts_model, eval_src,
                             n_windows=max(16, batch_size * 4), seed=seed + 9,
                             baseline_name=baseline)
    else:
        paired = {"n_windows": 0, "note": "dry_run: no training, no paired eval"}

    # ---- 闸门 ------------------------------------------------------------
    flops_ratio = (
        ts_res.train_flops / max(sc_res.train_flops, 1e-9)
        if sc_res.train_flops and ts_flops_tok is not None else None)
    equal_budget_ok = (
        (mode == "equal_token" and sc_res.tokens_trained == ts_res.tokens_trained)
        or (mode == "equal_flops" and abs(
            ts_res.train_flops - sc_res.train_flops) / max(sc_res.train_flops, 1e-9) <= 0.01)
    )
    param_alignment_ok = ts_param_dev <= 0.05
    heldout_count = min(max(16, batch_size * 4), eval_src.n_windows)
    heldout_coverage_ok = heldout_count >= 16
    ppl_pass = (not dry_run and heldout_coverage_ok and
                paired["ppl_simplcore"] <= paired["ppl_baseline"] * ppl_factor)
    gates = {
        "X-equal-budget": {
            "passed": bool(equal_budget_ok),
            "mode": mode,
            "satisfied_by_construction": mode == "equal_token",
            "tokens_sc": sc_res.tokens_trained,
            "tokens_ts": ts_res.tokens_trained,
            "flops_sc": sc_res.train_flops,
            "flops_ts": (ts_res.train_flops if ts_flops_tok is not None else None),
            "flops_ratio_ts_over_sc": flops_ratio,
            "equal_flops_ts_tokens_reference": (
                int(round(sc_total_flops / max(ts_flops_tok, 1e-9)))
                if mode == "equal_token" and ts_flops_tok is not None else None),
            "param_alignment_dev": round(ts_param_dev, 6),
            "note": ("等 token 口径由构造满足（两臂消费同一窗口流）；"
                     "FLOPs 差额如实报告，等 FLOPs 参考行供换算" if mode == "equal_token"
                     else "等 FLOPs 口径：TS token 预算按统一账本反算"),
        },
        "X-param-budget": {
            "passed": bool(param_alignment_ok),
            "max_deviation": 0.05,
            "actual_deviation": round(ts_param_dev, 6),
            "simplcore_non_embedding_params": sc_non_emb,
            "baseline_non_embedding_params": ts_non_emb,
        },
        "X-heldout-ppl": {
            "passed": bool(ppl_pass),
            "ppl_simplcore": (paired.get("ppl_simplcore") if not dry_run else None),
            "ppl_baseline": (paired.get("ppl_baseline") if not dry_run else None),
            "baseline": baseline,
            "factor": ppl_factor,
            "ratio": (paired.get("ppl_ratio_sc_over_baseline") if not dry_run else None),
            "heldout_windows": heldout_count,
            "minimum_heldout_windows": 16,
            "coverage_passed": bool(heldout_coverage_ok),
        },
    }
    real_corpus = corpus_root is not None
    unqualified = False
    arms = {"simplcore": {"loss_curve_head": [
        [int(t), round(float(v), 6)] for t, v in sc_res.loss_curve[:16]]}}
    arms[baseline] = {"loss_curve_head": [
        [int(t), round(float(v), 6)] for t, v in ts_res.loss_curve[:16]]}

    report: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "preset": preset,
        "baseline": baseline,
        "baseline_implementation": (
            "transformers.Mamba2ForCausalLM (reference implementation)"
            if baseline == "mamba2" else "local evaluation-only causal Transformer"),
        "baseline_config": (mamba_cfg.to_dict() if mamba_cfg is not None
                            else comparator_cfg.to_dict()),
        "seed": seed,
        "data_seed": (None if corpus_root is not None else data_seed),
        "seq_len": int(seq_len),
        "lr": lr,
        "batch_size": int(batch_size),
        "device": str(dev),
        "training": {
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "weight_decay": 0.1,
            "gradient_clip_norm": 1.0,
            "precision": str(next(sc_native.parameters()).dtype),
            "eval_every_tokens": int(eval_every_tokens),
            "lr_schedule": "linear warmup then cosine decay to 0.1x",
        },
        "budget": {
            "simplcore": sc_res.to_dict(include_curve=False),
            baseline: ts_res.to_dict(include_curve=False),
            "sc_flops_per_token_formula": "6*N_nonemb + 3*L*(2DM + 10D*log2(2s) + 6D + 22D)",
            "baseline_flops_per_token_formula": (
                "6*N_nonemb + 12*s*D*L" if ts_flops_tok is not None
                else "not estimated; Mamba2 state-scan costs require measured profiling"),
            "equal_flops_ts_tokens_reference": gates["X-equal-budget"]["equal_flops_ts_tokens_reference"],
        },
        "data": {
            "train_fingerprint": train_fingerprint,
            "validation_fingerprint": eval_fingerprint,
            "train_content_fingerprint": train_content_fingerprint,
            "validation_content_fingerprint": eval_content_fingerprint,
            "train_windows": int(src.n_windows),
            "validation_windows": int(eval_src.n_windows),
            "validation_split": ("val" if corpus_root is not None
                                 else "independent synthetic stream, shared transition table"),
            "synthetic_transition_seed": (
                None if corpus_root is not None else int(src.transition_seed)),
            "validation_generation": (
                None if corpus_root is not None
                else "independent stream; same seeded transition table"),
            "real_corpus": real_corpus,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": (__import__("transformers").__version__
                             if baseline == "mamba2" else None),
            "device": str(dev),
            "cuda_available": torch.cuda.is_available(),
            "device_name": (torch.cuda.get_device_name(dev)
                            if dev.type == "cuda" else "CPU"),
        },
        "arms": arms,
        "paired": paired,
        "gates": gates,
        "honesty": {
            "unqualified_claim_permitted": unqualified,
            "claim_language_if_permitted": (
                f"在等 token 预算下，SimplCore 的 held-out 困惑度不劣于参数量对齐的"
                f"{baseline} 基线（配对 bootstrap CI 见 paired 字段）"
                if unqualified else
                "本报告不支撑任何无限定宣称；引用时必须带预算口径与 CI"),
            "forbidden_checks": {
                "no_attention_internal_comparison": True,   # 只报真实输出 ppl，无注意力热图
                "equal_budget_qualifier_present": True,     # 全部数字带预算口径
                "no_self_reported_scores": True,            # held-out CE 是外部可复算的客观量
                "distribution_not_average_only": (not dry_run),  # 逐窗配对 + 最差窗口入报告
            },
            "claim_eligible_real_corpus": real_corpus,
            "multi_seed_required": True,
            "seed_count": 1,
            "notes": [
                f"{baseline} 臂是评测对照专用，不属于 SimplCore 运行时",
                f"参数对齐偏差 {ts_param_dev:.2%}（baseline {ts_non_emb} vs SC {sc_non_emb} 非嵌入参数）",
                ("合成语料不可替代真实语料结论；正式验收必须 --corpus 指向冻结语料"
                 if corpus_root is None else
                 "验证集来自独立 val split；训练集指标不用于 held-out 质量结论"),
            ] if corpus_root is None else [
                f"真实语料: {corpus_root}",
                f"参数对齐偏差 {ts_param_dev:.2%}",
                "held-out 评估只使用独立 val split，不复用 train split",
            ],
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return report


def run_multi_seed_comparison(*, seeds: Sequence[int], out_path: Any = None,
                              **kwargs: Any) -> Dict[str, Any]:
    """Run a fixed-data comparison over multiple initialization/order seeds."""
    if kwargs.get("dry_run"):
        raise ValueError("multi-seed quality evaluation cannot use --dry-run")
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 3:
        raise ValueError("multi-seed quality evaluation requires at least 3 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed evaluation requires unique seeds")
    reports = [
        run_comparison(seed=seed, out_path=None, **kwargs)
        for seed in seeds
    ]
    fixed_data = (
        reports[0]["data"]["train_content_fingerprint"],
        reports[0]["data"]["validation_content_fingerprint"],
    )
    if any((run["data"]["train_content_fingerprint"],
            run["data"]["validation_content_fingerprint"]) != fixed_data
           for run in reports[1:]):
        raise RuntimeError("multi-seed runs did not use identical train/validation content")
    if any(run["baseline"] != reports[0]["baseline"] for run in reports[1:]):
        raise RuntimeError("multi-seed runs used different baseline architectures")
    diffs = torch.tensor(
        [r["paired"]["mean_loss_diff_sc_minus_baseline"] for r in reports],
        dtype=torch.float64)
    ci = _bootstrap_ci_mean(diffs, n_boot=10000, seed=seeds[0])
    ppl_factor = float(kwargs.get("ppl_factor", DEFAULT_PPL_FACTOR))
    real_corpus = all(r["data"]["real_corpus"] for r in reports)
    equal_budget_passed = all(r["gates"]["X-equal-budget"]["passed"] for r in reports)
    parameter_budget_passed = all(r["gates"]["X-param-budget"]["passed"] for r in reports)
    quality_passed = (
        float(diffs.mean()) <= math.log(ppl_factor)
        and ci[1] <= math.log(ppl_factor)
        and all(r["gates"]["X-heldout-ppl"]["coverage_passed"] for r in reports)
    )
    gates = {
        "X-multi-seed": {"passed": len(seeds) >= 3, "seed_count": len(seeds),
                         "minimum_seeds": 3},
        "X-equal-budget": {"passed": equal_budget_passed},
        "X-param-budget": {"passed": parameter_budget_passed},
        "X-heldout-ppl": {
            "passed": bool(quality_passed),
            "baseline": reports[0]["baseline"],
            "ppl_factor": ppl_factor,
            "mean_logloss_diff_sc_minus_baseline": float(diffs.mean()),
            "ci95_logloss_diff": [ci[0], ci[1]],
            "mean_ppl_ratio_sc_over_baseline": float(torch.exp(diffs).mean()),
            "maximum_permitted_logloss_diff": math.log(ppl_factor),
        },
        "X-real-corpus": {"passed": real_corpus},
    }
    unqualified = all(gate["passed"] for gate in gates.values())
    report: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "baseline": reports[0]["baseline"],
        "seeds": list(seeds),
        "data": reports[0]["data"],
        "runtime": reports[0]["runtime"],
        "gates": gates,
        "aggregate": {
            "seed_count": len(seeds),
            "logloss_diffs_sc_minus_baseline": diffs.tolist(),
            "mean_logloss_diff_sc_minus_baseline": float(diffs.mean()),
            "ci95_logloss_diff": [ci[0], ci[1]],
            "mean_ppl_ratio_sc_over_baseline": float(torch.exp(diffs).mean()),
        },
        "runs": reports,
        "honesty": {
            "unqualified_claim_permitted": bool(unqualified),
            "claim_eligible_real_corpus": real_corpus,
            "multi_seed_required": True,
            "seed_count": len(seeds),
            "notes": [
                "每个 seed 在相同 train/val 内容上训练；seed 只改变初始化与训练窗口顺序",
                "合成语料结果不构成真实语料质量结论",
                "吞吐和显存需在相同硬件与 kernel path 上单独实测",
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
        prog="simplcore-equal-budget",
        description="等预算对照（协议 X-equal-budget）：SimplCore vs Transformer/Mamba2。")
    p.add_argument("--preset", default="smoke", choices=("smoke", "small", "1b"))
    p.add_argument("--baseline", default="transformer",
                   choices=("transformer", "mamba2"),
                   help="评测对照臂；mamba2 需要 transformers 的 Mamba2 实现")
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
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="至少 3 个 seed，跑多 seed aggregate 并生成 Parity Gate 报告")
    p.add_argument("--out", default="runs/eval_equal_token.json")
    p.add_argument("--dry-run", action="store_true", help="只算账不训练")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_kwargs = dict(
        preset=args.preset, mode=args.mode, baseline=args.baseline,
        token_budget=args.tokens, seq_len=args.seq_len,
        corpus_root=args.corpus, lr=args.lr, batch_size=args.batch_size,
        ppl_factor=args.ppl_factor, dry_run=args.dry_run)
    if args.seeds:
        report = run_multi_seed_comparison(
            seeds=args.seeds, out_path=args.out, **run_kwargs)
    else:
        report = run_comparison(
            seed=args.seed, out_path=args.out, **run_kwargs)
    g = report["gates"]
    if "aggregate" in report:
        print(f"protocol={report['protocol']} baseline={report['baseline']} seeds={report['seeds']}")
        print(f"mean logloss diff={report['aggregate']['mean_logloss_diff_sc_minus_baseline']:.4f}"
              f" ci95={report['aggregate']['ci95_logloss_diff']}")
    else:
        print(f"protocol={report['protocol']} mode={report['mode']} preset={report['preset']}")
        flops_ratio = g["X-equal-budget"]["flops_ratio_ts_over_sc"]
        ratio_text = (f"{flops_ratio:.4f}" if flops_ratio is not None
                      else "not estimated")
        print(f"tokens: sc={g['X-equal-budget']['tokens_sc']} ts={g['X-equal-budget']['tokens_ts']}"
              f"  flops ratio baseline/sc={ratio_text}")
    if not args.dry_run and "paired" in report:
        p = report["paired"]
        print(f"ppl: sc={p['ppl_simplcore']:.4f} {args.baseline}={p['ppl_baseline']:.4f}"
              f"  ratio={p['ppl_ratio_sc_over_baseline']:.4f}")
        print(f"paired: sc_wins={p['sc_wins']} ts_wins={p['ts_wins']}"
              f"  ci95_diff=[{p['ci95_diff'][0]:.4f}, {p['ci95_diff'][1]:.4f}]")
    for name, gate in g.items():
        print(f"gate {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"unqualified_claim_permitted: {report['honesty']['unqualified_claim_permitted']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
