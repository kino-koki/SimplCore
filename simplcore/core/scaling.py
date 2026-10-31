# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""参数预算与算力账本（parameter budgeting & compute ledger）。

这个模块回答的不是"我要造多大"，而是"给定参数/算力预算，怎么切分才不浪费"。
它是 30M 对话模型之所以可能的前提：**小模型最容易死的地方不是层数不够，
而是词表把预算吃光**。

为什么需要它
------------
一个 30M 模型如果沿用 150M/1.3B 的 ``vocab_size=32000, dim=512``：

===============  ==============  ================  ==================
配置              总参数           嵌入占比          每层可学习参数
===============  ==============  ================  ==================
V=32000, D=512    44.1M            37.1%            2.77M
V=8192,  D=512    31.9M            13.1%            2.77M
===============  ==============  ================  ==================

同样的深度和 FFN，只把词表从 32k 换到 8k，**从 44M 掉到 32M**：
省下的 12M 参数全部落在"永远只被用到一次"的嵌入表上。词表越大，
这个浪费越接近纯损失 —— 因为 32k 词表里的低频 token 在 30M 模型的
数据量级（<1B token）下根本训不充分，它们只是占着参数和显存。

因此本模块把"嵌入占比"提升为一等约束，并给出可搜索的配置规划器
:func:`plan_budget`。

算力账本的口径
---------------
:func:`cost_ledger` 返回的是**解析估算**（analytic），不是实测。它给的是
可复现的相对比较，用于回答：

* 同一条 4k 上下文，SGC 与 attention 每 token 差多少 FLOPs；
* 同一份显存，KV cache 能撑多长，SGC 的 O(1) 状态能撑多长；
* 固定算力预算下，两边分别能喂多少 token —— 这才是"1B 抵 3B~7B"
  唯一有意义的问法。

线性层按每参数 2 FLOPs（乘加）计，前向 ``2N``、反向 ``4N``，
合计训练 ``6N`` per token；长程项单独加：

* attention：``4 · seq · D`` 每层每 token（QKᵀ 与 AV 各 ``2·seq·D``）；
* SGC：``10 · D · log2(2·seq) + 6·D`` 每层每 token
  （长度 ``2·seq`` 的 rfft + irfft ≈ ``5·n·log2 n`` 每通道，
  摊到每 token 除以 ``seq``；频域复乘 ``seq`` 个 bin × 6 FLOPs）。

这些系数是可争论的（不同实现常数项不同），但**量级与增长阶**是对的：
attention 随 ``seq`` 线性增长，SGC 随 ``log seq`` 增长。测试
``test_scaling.py`` 对这两个增长阶做了断言，防止有人把公式改错却看不出来。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import SimplCoreConfig, analytic_params

BYTES_FP32 = 4


# ---------------------------------------------------------------------------
# 参数切分
# ---------------------------------------------------------------------------
def embedding_params(cfg: SimplCoreConfig) -> int:
    """嵌入表参数（``tie_embeddings=True`` 时输出头与它共享，不重复计）。"""
    return int(cfg.vocab_size) * int(cfg.dim)


def non_embedding_params(cfg: SimplCoreConfig) -> int:
    """除嵌入表以外的全部参数 —— 这才是"深度 + 宽度 + FFN"买到的容量。"""
    return analytic_params(cfg) - embedding_params(cfg)


def embedding_share(cfg: SimplCoreConfig) -> float:
    """嵌入表占总参数的比例。

    经验红线：**≤ 0.25**。超过它，模型名义上有 N 个参数，实际可用的
    组合容量已经明显低于同尺寸的"窄词表"模型 —— 在 <1B token 的数据量下
    尤其如此，因为大词表的低频行几乎不会被更新到。
    """
    total = analytic_params(cfg)
    return embedding_params(cfg) / max(total, 1)


def params_per_layer(cfg: SimplCoreConfig) -> float:
    """单层可学习参数（不含嵌入与 ``ln_f``），用于比较"同参数不同形状"。"""
    return (non_embedding_params(cfg) - cfg.dim) / max(cfg.num_layers, 1)


def aspect_ratio(cfg: SimplCoreConfig) -> float:
    """宽深比 ``dim / num_layers``。

    小模型的常见失败是"太宽太浅"：宽度买的是并行通道，深度买的才是
    组合与多步推理。经验区间：**20 ~ 65**（``dim`` / 层数）。
    """
    return cfg.dim / max(cfg.num_layers, 1)


# ---------------------------------------------------------------------------
# 算力账本
# ---------------------------------------------------------------------------
def _long_range_flops(cfg: SimplCoreConfig, seq_len: int) -> float:
    """每层每 token 的谱卷积路径 FLOPs（FFT 训练形式，逐项推导）。

    记 D=dim、M=num_bases、L=seq_len。这是**非参数化**的逐项账 ——
    参数矩阵（in_proj/out_proj/head/局部 FIR/coef 读出）的
    2 FLOPs/参数 已由调用方的 ``6·N`` 覆盖，这里只算参数覆盖不到的部分：

    * 核构建 ``coef(D,M) @ psi(M,L)``：``2·D·M·L``，摊到每 token **2·D·M**；
    * ``rfft + irfft``（n = 2L，实 FFT ≈ 2.5·n·log2(n) 每通道）：
      摊到每 token **10·D·log2(2L)**；
    * 频域复乘（n/2 个 bin × 6 FLOPs）：摊到每 token **6·D**；
    * 逐点激活（RMSNorm/content 门/证据门/不确定门，与 M、L 均无关）：**≈22·D**。

    v3.1 修正：旧版漏了核构建项 ``2·D·M``（L=4096、M=64 时与 FFT 项同量级），
    且把全部逐点项笼统记作 6·D。除 ``10·D·log2(2L)`` 一项随 L 对数增长外，
    其余均为常数 —— 亚线性结论不变、且更强。
    IIR 推理形式见 :func:`infer_flops_per_token`（O(1)/token，与 L 无关）。
    """
    d = float(cfg.dim)
    m = float(cfg.num_bases)
    s = float(max(seq_len, 1))
    return (2.0 * d * m                                   # 核构建，摊到每 token
            + 10.0 * d * math.log2(max(2.0 * s, 2.0))     # rfft+irfft
            + 6.0 * d                                     # 频域复乘
            + 22.0 * d)                                   # 逐点激活


def train_flops_per_token(cfg: SimplCoreConfig, seq_len: int) -> float:
    """训练期每 token FLOPs（前向 + 反向），含长程项。

    标准近似 ``6·N`` 只算线性层；谱全局卷积的长程项随 ``log seq`` 增长（FFT），
    对小模型相对差距同样决定性 —— 但相较 attention 的线性增长，仍是亚线性。
    这正是"同算力能喂更多 token"的源头（详见 ``docs/THEORETICAL_EDGE.md``）。
    """
    base = 6.0 * non_embedding_params(cfg)
    sgc_per = _long_range_flops(cfg, seq_len)
    # 反向约为前向的 2 倍，长程项同样按 3 倍计（前向 + 两个反向）
    mixer = 3.0 * sgc_per * cfg.num_layers
    return base + mixer


def infer_flops_per_token(cfg: SimplCoreConfig, seq_len: int) -> float:
    """解码期每 token FLOPs（只算前向），与 ``seq_len`` **无关** —— O(1) 流式。

    记 D=dim、M=num_bases。参数矩阵（in_proj/out_proj/head/局部 FIR/coef）
    按 2 FLOPs/参数 计入 ``2·N``；非参数化逐项账（每层）：

    * 二阶 IIR 递推 ``y = a·y1 + b·y2 + c·u`` 作用在**全部 M·D 个状态元素**上，
      每元素 3 乘 2 加 = **5·M·D**（v3.1 修正：旧版误写 6·D，漏乘基数 M）；
    * 逐点激活（RMSNorm/content 门/证据门/不确定门）：**≈22·D**。

    （einsum 读出 ``coef`` 的 2·M·D 已含在 2·N 里 —— coef 本就是 M·D 参数。）
    ``seq_len`` 形参仅为签名兼容保留；公式不含 L，这正是本架构要断言的性质。
    """
    del seq_len  # O(1) 流式的数学内容：解码成本与已见长度无关
    base = 2.0 * non_embedding_params(cfg)
    mixer = cfg.num_layers * cfg.dim * (5.0 * cfg.num_bases + 22.0)
    return base + mixer


def kv_cache_bytes_per_token(cfg: SimplCoreConfig) -> int:
    """SimplCore 没有 KV cache：注意力每 token 的 KV 缓存字节数恒为 0。

    SimplCore 用 O(1) 常驻流式状态（见 ``stream_state_bytes``）替代随 ``seq``
    线性膨胀的 KV cache，因此本函数恒返回 0，``CostLedger.is_kv_free`` 恒为 True。
    """
    return 0


def stream_state_bytes(cfg: SimplCoreConfig, batch: int = 1) -> int:
    """SimplCore 流式状态的字节数 —— **与序列长度无关**。

    按定理 3，每层状态 = 每基二阶递推状态 ``y1, y2``（``2·M·D``）
    + 局部卷积缓冲 ``D·(K−1)`` + L2 证据门前缀能量和（1 个标量）。

    **计费口径（重要）**：递推状态**恒按 fp32 计费**，与 ``cfg.compute_dtype``
    无关。理由见 ``.sgc.recurrent_dtype`` 单一事实源——低精度（bf16/fp16）计算时
    递推状态仍分配为 fp32（业界通行做法，Mamba/Mamba2 同），避免逐 token 累积
    舍入误差；因此流式显存账单以 fp32 元素大小（4 字节）为准，不随计算精度缩半。
    局部卷积缓冲与能量和同样按 fp32/恒定处理。最后加 ``4·D`` 为轻量跨层缓冲。

    注意 M 因子不可省略：M=128、D=2048 时每层 ≈2.02 MiB，
    是早期版本（漏 M，误报 ≈48 KiB/层）的 128 倍 —— 该错误由
    mathproof P12 的 ≤2 GiB 硬断言在修正时暴露。
    """
    per_layer = (2 * cfg.num_bases * cfg.dim
                 + (cfg.local_kernel - 1) * cfg.dim
                 + 1) * BYTES_FP32
    return int((per_layer * cfg.num_layers + 4 * cfg.dim) * batch)


@dataclass(frozen=True)
class CostLedger:
    """一份配置在给定上下文长度下的算力/占用账本。"""

    name: str
    params: int
    non_embedding_params: int
    embedding_share: float
    seq_len: int
    train_flops_per_token: float
    infer_flops_per_token: float
    kv_bytes_per_token: int
    state_bytes: int

    @property
    def is_kv_free(self) -> bool:
        return self.kv_bytes_per_token == 0

    def context_bytes(self, seq_len: int, batch: int = 1) -> int:
        """解码到 ``seq_len`` 时的缓存占用（attention 线性增长，SGC 恒定）。"""
        return self.kv_bytes_per_token * seq_len * batch + self.state_bytes * batch

    def train_flops(self, tokens: float) -> float:
        return self.train_flops_per_token * tokens

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "params": self.params,
            "non_embedding_params": self.non_embedding_params,
            "embedding_share": round(self.embedding_share, 4),
            "seq_len": self.seq_len,
            "train_flops_per_token": round(self.train_flops_per_token, 1),
            "infer_flops_per_token": round(self.infer_flops_per_token, 1),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "state_bytes": self.state_bytes,
            "kv_free": self.is_kv_free,
        }


def cost_ledger(cfg: SimplCoreConfig, seq_len: int = 4096,
                name: str = "cfg") -> CostLedger:
    """构造一份配置的账本。"""
    return CostLedger(
        name=name,
        params=analytic_params(cfg),
        non_embedding_params=non_embedding_params(cfg),
        embedding_share=embedding_share(cfg),
        seq_len=int(seq_len),
        train_flops_per_token=train_flops_per_token(cfg, seq_len),
        infer_flops_per_token=infer_flops_per_token(cfg, seq_len),
        kv_bytes_per_token=kv_cache_bytes_per_token(cfg),
        state_bytes=stream_state_bytes(cfg),
    )


# ---------------------------------------------------------------------------
# token / 算力预算
# ---------------------------------------------------------------------------
def chinchilla_tokens(params: int, ratio: float = 20.0) -> float:
    """Chinchilla 口径的算力最优 token 数（``ratio`` 默认 20）。"""
    return float(params) * float(ratio)


def compute_budget(params: int, tokens: float) -> float:
    """纯线性层的训练 FLOPs 下界 ``6·N·D``（不含长程项，用于对外报价）。"""
    return 6.0 * float(params) * float(tokens)


def gpu_hours(flops: float, tflops: float = 312.0, mfu: float = 0.4) -> float:
    """把 FLOPs 折算成 GPU 小时（默认 A100 bf16 峰值 312 TFLOPS，MFU 40%）。"""
    effective = float(tflops) * 1e12 * float(mfu)
    return flops / max(effective, 1.0) / 3600.0


def tokens_for_budget(flops: float, cfg: SimplCoreConfig, seq_len: int = 4096) -> float:
    """固定 FLOPs 预算下，这份配置能训多少 token。

    这是"同算力换更多 token"的直接入口：分母越小，能喂的 token 越多，
    而小模型的能力几乎完全由 token 数决定。
    """
    return float(flops) / max(train_flops_per_token(cfg, seq_len), 1.0)


def largest_config_for_flops(flops: float, tokens: float,
                             seq_len: int = 4096,
                             vocab_size: int = 32000,
                             mixer: str = "sgc",
                             ffn_ratios: Sequence[float] = (3.0, 4.0),
                             num_bases_options: Sequence[int] = (64, 128),
                             max_dim: int = 4096,
                             max_layers: int = 80) -> Optional[SimplCoreConfig]:
    """反解：给定算力预算 ``flops`` 与 token 数 ``tokens``，最大的合法模型。

    用**网格搜索**而不是解析反解：``train_flops_per_token`` 里谱全局卷积项对
    ``dim`` 是线性的、参数项对 ``dim`` 是二次的，还有 FFT 的 ``log seq``，
    解析反解容易写出看着对、实际差一半的公式。搜索慢一点，但结论可核对。

    Returns:
        满足约束的最大配置；预算连最小配置都装不下时返回 ``None``。
    """
    best: Optional[SimplCoreConfig] = None
    best_params = -1
    for r in ffn_ratios:
        for M in num_bases_options:
            for d in range(128, max_dim + 1, 64):
                heads = max(1, d // 64)
                for L in range(1, max_layers + 1):
                    cfg = SimplCoreConfig(vocab_size=vocab_size, dim=d, num_layers=L,
                                    num_bases=M, ffn_ratio=r, mixer=mixer,
                                    num_heads=heads)
                    n = analytic_params(cfg)
                    if n <= best_params:
                        continue
                    if train_flops_per_token(cfg, seq_len) * tokens <= flops:
                        best, best_params = cfg, n
                    else:
                        break          # 层数再多必然超预算，换下一组形状
    return best


# ---------------------------------------------------------------------------
# 配置规划器
# ---------------------------------------------------------------------------
_EMBED_RED_LINE = 0.25


@dataclass(frozen=True)
class BudgetPlan:
    """一个候选配置及其预算指标。"""

    cfg: SimplCoreConfig
    params: int
    embedding_share: float
    params_per_layer: float
    aspect: float

    def score(self, target: int) -> float:
        """越小越好：参数偏差 + 嵌入超红线惩罚 + 形状不合理惩罚。"""
        rel = abs(self.params - target) / max(target, 1)
        pen = 0.0
        if self.embedding_share > _EMBED_RED_LINE:
            pen += 4.0 * (self.embedding_share - _EMBED_RED_LINE)
        if self.aspect < 16:
            pen += 2.0 * (16 - self.aspect) / 16
        if self.aspect > 80:
            pen += 2.0 * (self.aspect - 80) / 80
        return rel + pen

    def as_dict(self) -> Dict[str, object]:
        return {
            "params": self.params,
            "embedding_share": round(self.embedding_share, 4),
            "params_per_layer": int(self.params_per_layer),
            "aspect": round(self.aspect, 1),
            "cfg": self.cfg.to_dict(),
        }


def plan_budget(target_params: int,
                vocab_options: Sequence[int] = (4096, 8192, 12288, 16000, 32000),
                dim_options: Optional[Iterable[int]] = None,
                ffn_ratios: Sequence[float] = (2.0, 2.5, 3.0, 4.0),
                num_bases_options: Sequence[int] = (16, 32, 64),
                local_kernel: int = 4,
                mixer: str = "sgc",
                max_layers: int = 64,
                top_k: int = 5) -> List[BudgetPlan]:
    """在参数预算约束下搜索候选配置，按 :meth:`BudgetPlan.score` 排序。

    约束（不是可选项，是本模块的核心主张）：

    * 嵌入占比 ≤ 25%，否则大词表会把预算吃穿；
    * 宽深比 ``dim / L`` 落在 16~80 —— 太浅学不会组合，太窄学不动；
    * ``dim`` 能被 ``num_heads`` 整除（attention 族），且为偶数（FFT 路径）。

    Args:
        target_params: 目标参数量（含嵌入）。
        vocab_options: 候选词表大小。**小模型请务必把 4k/8k 放进来**。
        dim_options: 候选宽度；``None`` 用 256..1536 的 64 倍数。
        max_layers: 深度上限（防止搜索退化成"极深极窄"）。

    Returns:
        按分数升序的候选列表（最佳在前）。
    """
    if dim_options is None:
        dim_options = range(256, 1537, 64)
    # SimplCore 的多头谱 mixer（GatedSpectralConv2）需要足够头数；与注意力无关。
    heads_for = lambda d: max(4, d // 64)

    plans: List[BudgetPlan] = []
    for V in vocab_options:
        for d in dim_options:
            if d % 2 or d % heads_for(d):
                continue
            for r in ffn_ratios:
                if int(d * r) % 32:
                    continue                      # FFN 宽度对齐到 32
                for M in num_bases_options:
                    # 先解出层数：总参数 ≈ V·d + L·per + d
                    probe = SimplCoreConfig(vocab_size=V, dim=d, num_layers=1,
                                      num_bases=M, ffn_ratio=r, mixer=mixer,
                                      num_heads=heads_for(d), local_kernel=local_kernel)
                    per = params_per_layer(probe)
                    L = int(round((target_params - V * d - d) / max(per, 1)))
                    for Ll in {max(1, L - 1), max(1, L), max(1, L + 1)}:
                        if Ll > max_layers:
                            continue
                        cfg = SimplCoreConfig(vocab_size=V, dim=d, num_layers=Ll,
                                        num_bases=M, ffn_ratio=r, mixer=mixer,
                                        num_heads=heads_for(d),
                                        local_kernel=local_kernel)
                        try:
                            n = analytic_params(cfg)
                        except ValueError:
                            continue
                        plans.append(BudgetPlan(
                            cfg=cfg, params=n,
                            embedding_share=embedding_share(cfg),
                            params_per_layer=params_per_layer(cfg),
                            aspect=aspect_ratio(cfg),
                        ))
    plans.sort(key=lambda p: p.score(target_params))
    # 去重（同 dim/layers/ffn 只留参数最接近的）
    seen, out = set(), []
    for p in plans:
        k = (p.cfg.vocab_size, p.cfg.dim, p.cfg.num_layers, p.cfg.ffn_ratio)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) >= top_k:
            break
    return out


def budget_table(configs: Dict[str, SimplCoreConfig], seq_len: int = 4096) -> str:
    """渲染账本对比表（CLI 入口 ``sgc-budget`` 用）。"""
    head = (f"{'name':16s} {'params':>10s} {'non-emb':>10s} {'emb%':>6s} "
            f"{'train GFLOPs/tok':>17s} {'kv B/tok':>9s} {'ctx@32k GB':>11s}")
    lines = [head, "-" * len(head)]
    for name, cfg in configs.items():
        led = cost_ledger(cfg, seq_len, name)
        lines.append(
            f"{name:16s} {led.params/1e6:9.2f}M {led.non_embedding_params/1e6:9.2f}M "
            f"{led.embedding_share*100:5.1f}% "
            f"{led.train_flops_per_token/1e9:17.3f} "
            f"{led.kv_bytes_per_token:9d} "
            f"{led.context_bytes(32768)/1e9:10.2f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="参数预算 / 算力账本")
    ap.add_argument("--target", type=int, default=30_000_000,
                    help="目标参数量，默认 30M")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="要对比的 preset 名（默认 30m/1b 段）")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    print(f"=== 目标 {args.target/1e6:.1f}M 的候选切分（嵌入占比红线 25%）")
    for i, p in enumerate(plan_budget(args.target, top_k=args.top_k), 1):
        c = p.cfg
        print(f"{i}. params={p.params/1e6:7.2f}M  emb={p.embedding_share*100:4.1f}%  "
              f"V={c.vocab_size:6d} d={c.dim:4d} L={c.num_layers:2d} "
              f"H={c.hidden:5d} M={c.num_bases:3d}  aspect={p.aspect:4.1f}")

    if args.compare:
        from .config import get_preset
        cfgs = {n: get_preset(n) for n in args.compare}
        print()
        print(budget_table(cfgs, args.seq_len))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
