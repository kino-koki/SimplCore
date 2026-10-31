# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore configuration objects and MI250-80GB scale presets."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


# ---------------------------------------------------------------------------
# 对外品牌标识
# ---------------------------------------------------------------------------
# 架构内核仍为 SGC（谱全局卷积 mixer，simplcore.core.sgc）；对外发布的模型产品
# 命名为「青小团」，英文代号 Tuan。预设规模名（tuan-150m …）即此品牌下的型号。
MODEL_NAME = "青小团"
MODEL_CODENAME = "Tuan"


@dataclass
class GateConfig:
    """抗幻觉闸门的统一配置 —— 把分散在 ``SimplCoreConfig`` 上的多个开关与阈值收敛到一处。

    默认全关。四层闸门（L2/L3 证据与不确定门、L4 拒答头）本质是**可选的监督增强**：
    只有在你提供外部证据 / grounded 标签去训练它们时才有意义。未给监督时一律关闭，
    模型退化为纯 SGC —— 这正是消除「监督缺失即空转」双刃剑的做法。

    字段映射
    --------
    * ``use_abstain`` / ``evidence_gate`` / ``uncertainty_gate`` → ``SimplCoreConfig`` 同名开关；
    * ``tau_grounded`` / ``coverage_answer`` / ``coverage_hedge`` / ``tau_retrieval`` /
      ``w_grounded`` / ``w_coverage`` / ``w_retrieval`` / ``min_coverage`` / ``min_ngram``
      → 传给 ``abstain.GroundedPolicy``（L4 四态拒答的判定阈值，详见 ``abstain.py``）。
    """

    use_abstain: bool = False
    evidence_gate: bool = False
    uncertainty_gate: bool = False
    # —— L4 拒答策略阈值（对应 abstain.GroundedPolicy 字段）——
    tau_grounded: float = 0.50
    coverage_answer: float = 0.80
    coverage_hedge: float = 0.50
    tau_retrieval: float = 0.25
    w_grounded: float = 0.45
    w_coverage: float = 0.45
    w_retrieval: float = 0.10
    min_coverage: float = 0.62
    min_ngram: float = 0.35

    def policy(self):
        """构造对应的 ``abstain.GroundedPolicy``（仅当 ``use_abstain`` 时消费）。"""
        from .abstain import GroundedPolicy
        return GroundedPolicy(
            tau_grounded=self.tau_grounded, coverage_answer=self.coverage_answer,
            coverage_hedge=self.coverage_hedge, tau_retrieval=self.tau_retrieval,
            w_grounded=self.w_grounded, w_coverage=self.w_coverage,
            w_retrieval=self.w_retrieval, min_coverage=self.min_coverage,
            min_ngram=self.min_ngram)


@dataclass
class SimplCoreConfig:
    """Configuration of a SimplCore language model (non-attention backbone).

    Note there is **no positional-embedding table and no ``max_len``**: the causal
    spectral convolution is inherently position aware, so sequences longer than
    any value seen in training remain well defined — and, unlike a Transformer
    decoder, there is **no softmax self-attention and no KV cache** that grows
    with the sequence.

    **Streaming precision invariant (MI250 / bf16):** the recurrent O(1) state
    (IIR oscillator state for SGC / gated-SGC, and the complex SSM state for SRF)
    is always held in ``float32`` when the compute dtype is ``bfloat16`` /
    ``float16`` — see :func:`simplcore.core.sgc.recurrent_dtype`. This keeps the
    headline O(1) streaming numerically honest on low-precision hardware (a bf16
    IIR / SSM recurrence would otherwise accumulate rounding error token by token),
    while all I/O stays in the compute dtype. With ``float32`` compute the state is
    ``float32`` already, so behaviour is unchanged. The policy is uniform across all
    three mixer families (SGC, SRF, GatedSpectralConv2) and the fused hot path.

    Attributes:
        vocab_size: tokenizer vocabulary size.
        dim: model width D.
        num_layers: number of blocks.
        ffn_hidden: SwiGLU hidden width; ``None`` -> ``ffn_ratio * dim``.
        ffn_ratio: FFN expansion when ``ffn_hidden`` is None.
        mixer: mixer kind — one of the non-attention families:
            ``sgc`` (content-gated spectral global conv),
            ``sgc-nogate`` (no content gate),
            ``sgc-nolocal`` (no depthwise local FIR),
            ``srf`` (Spectral Resonance Flow: selective-SSM spectral conv).
        num_bases: number of damped-oscillator bases M of the global kernel.
        decay_max: longest decay time constant (kernel initialisation scale).
        local_kernel: depthwise causal conv width.
        content_gate: input-dependent (selective) gating.
        out_gate: output gating.
        evidence_gate: L2 evidence gate (causal-prefix energy damps the long path).
            **Default OFF.** The gate is an opt-in anti-hallucination feature that only
            earns its keep when you train it with external grounded / evidence labels.
            An *untrained* gate would sit at a non-identity weight and silently attenuate
            the long-range path — the "supervision absent = idle" double-edged sword.
            Turn it on only when you actually have the supervision signal.
        uncertainty_gate: L3 uncertainty gate (long/local disagreement damps the long path).
            **Default OFF.** Same opt-in discipline as ``evidence_gate``.
        use_abstain: attach the L4 abstention head to the model. **Default OFF** — it
            needs external evidence labels to train; otherwise it stays a no-op head.
        use_memory: attach the CMT memory-injection projection (D×D).
        norm: ``rmsnorm`` or ``layernorm``.
        num_heads: head count for the multi-head spectral mixer variant
            (``GatedSpectralConv2``); ignored by SGC / SRF mixers.
        tie_embeddings: tie the output head to the token embedding.
        dropout: residual/embedding dropout.
        norm_eps: normalisation epsilon.
        fft_dtype: dtype for the FFT path (``"float32"`` recommended).
        grad_checkpoint: wrap blocks in activation checkpointing.
        init_std: std of the normal-distribution init for linear layers.
        gate_ab: enable GLA/Mamba2-style channel-independent gamma/beta content
            gates on the multi-head spectral mixer variant.
        use_mixer_stream: 启用融合 C++/CUDA 流式热路径（``simplcore.ops.mixer_stream``，
            ``SpectralGlobalConv`` mixer 专属）。默认 False；仅当 fp32 且扩展已构建
            时生效，否则自动回退纯 PyTorch。
        srf_heads: ``srf`` mixer Delta-head count (channels within a head share stride).
        srf_chunk: ``srf`` chunked-scan block length.
        codec_kind: codec type for the vocab-free mode ("char" | "byte" | "bpe").
        codec_kwargs: kwargs passed through to ``tokenization.make_codec``.
    """

    vocab_size: int = 32000
    dim: int = 768
    num_layers: int = 12
    ffn_hidden: Optional[int] = None
    ffn_ratio: float = 4.0
    mixer: str = "sgc"
    num_bases: int = 64
    decay_max: float = 8192.0
    local_kernel: int = 4
    content_gate: bool = True
    out_gate: bool = False
    evidence_gate: bool = False
    uncertainty_gate: bool = False
    use_abstain: bool = False
    use_memory: bool = False
    norm: str = "rmsnorm"
    num_heads: int = 12
    tie_embeddings: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-6
    fft_dtype: str = "float32"
    grad_checkpoint: bool = False
    init_std: float = 0.02
    gate_ab: bool = False
    use_mixer_stream: bool = False
    # -- SRFN（谱振流网络）-----------------------------------------------------
    #: ``srf`` mixer 的 Delta 头数：同一个头内的通道共享步长。
    #: 取 ``dim // 40`` 附近比较自然（head_dim≈40 时 8 个头）。
    srf_heads: int = 8
    #: ``srf`` chunked scan 的块长。32 是 CPU/GPU 折中：块越大 (C,C) 矩阵乘
    #: 越贵，但跨块串行次数越少。对话场景 T<=512，32 已经够。
    srf_chunk: int = 32
    # SimplCore 无词表模式（vocab-free）：编解码器类型（"char" | "byte" | "bpe"），
    # 及其参数字典（透传给 tokenization.make_codec）。
    # 模型 vocab_size 仍须与 codec.size() 一致（validate 不做自动校验，
    # 由调用方保证；init_simplcore_from_hf 的 cfg_override 路径推荐用
    # ``SimplCoreConfig.from_codec(codec)`` 工厂方法自动填充）。
    codec_kind: str = "char"
    codec_kwargs: Optional[dict] = None
    # -- 训练期上下文 / 序列长度课程（非模型超参，模型前向完全忽略）----------------
    #: 训练编排元信息：超长上下文课程、max_seq_len 等。仅 ``train.py`` 读取，
    #: 不影响模型结构或推理（SGC 本身无 max_len、O(1) 状态，4M 在架构上可行）。
    #: 形如 ``{"max_seq_len": 4000000, "curriculum": [{"seq_len":4096,"until_frac":0.1},
    #:        {"seq_len":16384,"until_frac":0.25}, ..., {"seq_len":4000000}]}``。
    #: 缺省 None = 固定 ``--seq-len``，行为与旧版逐位一致（零回归）。
    context: Optional[dict] = None

    # -- SGC v1.2 增强（全部默认关；开启不影响 FFT/IIR 对偶与纯洁性）----------
    #: 三路（长程/局部/瞬时）softmax 自适应混合：w_t = softmax(W h_t + b)，
    #: 逐位置因果，并行/流式逐位一致。关闭时保持固定相加。
    sgc_path_gate: bool = False
    #: 瞬时项非线性 feed·SiLU(u)（关闭时 feed·u）。
    sgc_inst_nonlin: bool = False
    #: 振荡基网格初始化："logfreq"（默认，theta 对数分布，周期与衰减窗口
    #: 尺度配对，且无 legacy 网格的公共零点）| "legacy"（theta 均匀分布，
    #: 存在 tau=2M(M+1)-1 处公共零点，显式选择需过 _assert_no_common_zero
    #: 守护）。仅影响初始值。
    sgc_init_mode: str = "logfreq"
    #: 衰减极点参数化："sigmoid"（默认，兼容旧 checkpoint，最长记忆 ~1e6）|
    #: "logspace"（``rho = exp(-softplus(raw))``，可表达 4M 所需 0.99999975，
    #: 仅 SGC 家族 mixer 支持；state_dict 自动迁移）。
    sgc_rho_param: str = "sigmoid"
    # -- SGC-R 谱专家路由（mixer="sgc-r"）------------------------------------
    #: 把 M 个振荡基按频率连续分成 G 组"谱专家"，逐 token 路由器 top-k 选择
    #: 注入组（MoE 路由，无 token↔token 成对打分，非注意力）。O(1) 流式
    #: 状态与 sgc 逐字节相同。要求 num_bases % G == 0。
    sgc_route_groups: int = 4
    #: 每 token 激活的专家数 k（1 ≤ k ≤ sgc_route_groups）。
    sgc_route_topk: int = 2
    #: switch 负载均衡辅助损失权重（0 = 关闭；训练侧显式取用，
    #: model/train 主路径不自动加）。
    sgc_route_aux: float = 1.0

    @classmethod
    def from_gate_policy(cls, gate: "GateConfig", **kw) -> "SimplCoreConfig":
        """用统一的 :class:`GateConfig` 一次性注入所有抗幻觉开关，再叠加其余模型参数。

        例::

            cfg = SimplCoreConfig.from_gate_policy(
                GateConfig(evidence_gate=True, uncertainty_gate=True,
                           use_abstain=True, tau_grounded=0.6),
                dim=768, num_layers=12)

        显式传入的同名 ``kw`` 优先于 ``gate``（``setdefault`` 语义）。
        """
        kw.setdefault("use_abstain", gate.use_abstain)
        kw.setdefault("evidence_gate", gate.evidence_gate)
        kw.setdefault("uncertainty_gate", gate.uncertainty_gate)
        return cls(**kw)

    # -- derived -------------------------------------------------------------
    @property
    def hidden(self) -> int:
        if self.ffn_hidden is not None:
            return int(self.ffn_hidden)
        # SwiGLU keeps ~ the same parameter count as a 4x GELU MLP at ratio 4.
        return int(self.ffn_ratio * self.dim)

    def resolve_fft_dtype(self):
        import torch
        return {"float32": torch.float32, "float64": torch.float64,
                "bfloat16": torch.bfloat16, "float16": torch.float16,
                "none": None, "auto": None}.get(self.fft_dtype, torch.float32)

    # -- io ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SimplCoreConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown SimplCoreConfig fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_codec(cls, codec, **kw) -> "SimplCoreConfig":
        """由编解码器构造 config：自动填充 ``vocab_size = codec.size()`` 及
        ``codec_kind`` / ``codec_kwargs``（供检查点序列化还原 codec）。

        额外 kw 透传给 ``SimplCoreConfig``（如 ``dim`` / ``num_layers`` / ``mixer``）。
        这是"舍去传统 Transformer 词表"的推荐入口：

            from simplcore.core import CharTokenizer, SimplCoreConfig
            cfg = SimplCoreConfig.from_codec(CharTokenizer(), dim=384, num_layers=6)
            m = SimplCoreModel(cfg)   # vocab_size 自动 = codec.size()
        """
        from .tokenization import Tokenizer
        assert isinstance(codec, Tokenizer)
        kind = type(codec).__name__
        kw.setdefault("vocab_size", codec.size())
        kw.setdefault("codec_kind",
                      "char" if kind == "CharTokenizer"
                      else "byte" if kind == "ByteTokenizer"
                      else "bpe")
        kw.setdefault("codec_kwargs", {})
        return cls(**kw)

    def rebuild_codec(self):
        """从 ``codec_kind`` / ``codec_kwargs`` 还原编解码器（检查点加载路径）。

        BPE 需要额外 vocab/merges 数据，由调用方通过 ``codec_kwargs`` 传入；
        数据缺失时返回 None（模型仍可按纯数字 ID 流运行，仅无法自动编解码文本）。
        """
        from .tokenization import make_codec
        if self.codec_kwargs is None:
            self.codec_kwargs = {}
        try:
            return make_codec(self.codec_kind, **self.codec_kwargs)
        except (ValueError, KeyError):
            return None

    @classmethod
    def from_json(cls, path: str) -> "SimplCoreConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> "SimplCoreConfig":
        # mixer 白名单来自注册表（simplcore/core/registry.py）的单一事实源，
        # 不再手写元组——新增 mixer 只需在注册表登记，validate 自动放行。
        from .registry import get_mixer_spec
        if self.dim % 2:
            raise ValueError("dim must be even for the FFT path")
        get_mixer_spec(self.mixer)          # 未登记 mixer 在这里抛 ValueError
        if self.mixer == "srf":
            if self.dim % self.srf_heads:
                raise ValueError("dim must be divisible by srf_heads")
            if self.srf_chunk < 1:
                raise ValueError("srf_chunk must be >= 1")
        if self.num_bases < 1:
            raise ValueError("num_bases must be >= 1")
        if self.local_kernel < 1:
            raise ValueError("local_kernel must be >= 1")
        if getattr(self, "sgc_init_mode", "legacy") not in ("legacy", "logfreq"):
            raise ValueError(
                f"sgc_init_mode must be 'legacy' or 'logfreq', "
                f"got {self.sgc_init_mode!r}")
        rho_param = getattr(self, "sgc_rho_param", "sigmoid")
        if rho_param not in ("sigmoid", "logspace"):
            raise ValueError(
                f"sgc_rho_param must be 'sigmoid' or 'logspace', "
                f"got {rho_param!r}")
        if rho_param == "logspace" and self.mixer not in (
                "sgc", "sgc-nogate", "sgc-nolocal", "sgc-r"):
            # SRF / selective 的 SSM 复极点参数化不同；静默忽略会让配置
            # "看起来开了 logspace"却仍被 1e-6 夹住 —— 直接拒绝。
            raise ValueError(
                "sgc_rho_param='logspace' 仅支持 SGC 家族 mixer "
                f"(sgc/sgc-nogate/sgc-nolocal/sgc-r)，got {self.mixer!r}")
        if self.mixer == "sgc-r":
            groups = getattr(self, "sgc_route_groups", 4)
            topk = getattr(self, "sgc_route_topk", 2)
            if groups < 1 or self.num_bases % groups:
                raise ValueError(
                    "sgc_route_groups 必须整除 num_bases "
                    f"(num_bases={self.num_bases}, sgc_route_groups={groups})")
            if not (1 <= topk <= groups):
                raise ValueError(
                    f"sgc_route_topk 必须在 [1, sgc_route_groups] 内，"
                    f"got {topk} (groups={groups})")
        if self.gate_ab:
            # gate_ab 走 GatedSpectralConv2 的多头谱基（每 head 独立谱基），
            # 因此 dim 必须能被 num_heads 整除。
            if self.num_heads < 1:
                raise ValueError("num_heads must be >= 1 when gate_ab is enabled")
            if self.dim % self.num_heads:
                raise ValueError(
                    "dim must be divisible by num_heads when gate_ab is enabled "
                    f"(dim={self.dim}, num_heads={self.num_heads}); "
                    "GatedSpectralConv2 splits the channel dim across heads")
        return self


# ----------------------------------------------------------------------------
# Presets.  Sized for the stated target compute: 1 x AMD Instinct MI250
# (128 GB HBM2e, bf16), sequence length 4096, AdamW.  Parameter counts come
# from `analytic_params()` / `simplcore-presets` (analytic, not instantiated).
# ----------------------------------------------------------------------------
def _p(**kw) -> SimplCoreConfig:
    cfg = SimplCoreConfig(**kw)
    return cfg


# 超长上下文课程（SimplCore 卖点"总上下文 4M"）：SGC 无 max_len、因果谱卷积 O(1)
# 状态，因此 4M 在**架构上**可行，唯一约束是训练显存。课程先短后长，把基础学稳
# 再拉伸到 4M 做长上下文专项收敛。这是训练编排元信息（context），模型结构 / 推理
# 完全忽略；预设不附带、且 --seq-len-curriculum 未给时退化为固定 --seq-len（零回归）。
_CTX_LONGCTX_4M = {
    "max_seq_len": 4_000_000,
    "curriculum": [
        {"seq_len": 4096,    "until_frac": 0.10},
        {"seq_len": 16384,   "until_frac": 0.25},
        {"seq_len": 65536,   "until_frac": 0.45},
        {"seq_len": 262144,  "until_frac": 0.65},
        {"seq_len": 1048576, "until_frac": 0.85},
        {"seq_len": 4000000},
    ],
}

# 2B × 30 天（单 MI250，2 GCD DDP）的上下文课程。
# 预算口径：30d × 86400 × 0.65 × 362.1 TFLOPS × 25% MFU ≈ 1.53e20 FLOPs；
# 2.02B 每 token ≈ 11.8e9 FLOPs → ~13B tokens。until_frac 直接等于 token 占比
# （global_batch_tokens 固定时，步数占比 = token 占比）。
# 4M 段占 10% ≈ 1.3B tokens ≈ 325 个 4M 窗口 —— 这是"4M 可运行 + 基础长程检索"
# 的预算，不是"4M 能力完全收敛"。要真正收敛 4M，需要再一个 30 天周期。
_CTX_2B_4M_30D = {
    "max_seq_len": 4_000_000,
    "curriculum": [
        {"seq_len": 4096,    "until_frac": 0.18},
        {"seq_len": 16384,   "until_frac": 0.36},
        {"seq_len": 65536,   "until_frac": 0.54},
        {"seq_len": 262144,  "until_frac": 0.70},
        {"seq_len": 1048576, "until_frac": 0.90},
        {"seq_len": 4000000},
    ],
}


SIMPLCORE_PRESETS = {
    # local CPU / unit-test sizes
    "debug": _p(vocab_size=1024, dim=128, num_layers=2, num_bases=16, decay_max=256,
                local_kernel=4, num_heads=4),
    "toy": _p(vocab_size=4096, dim=256, num_layers=4, num_bases=32, decay_max=1024,
              local_kernel=4, num_heads=8),
    # 1 x A100 80GB targets (bf16, seq 4096)
    "tuan-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                    decay_max=8192, local_kernel=4, num_heads=12),
    "tuan-350m": _p(vocab_size=32000, dim=1024, num_layers=20, num_bases=64,
                    decay_max=8192, local_kernel=4, num_heads=16),
    "tuan-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, num_bases=128,
                    decay_max=16384, local_kernel=4, num_heads=16, ffn_ratio=4.0,
                    context=_CTX_LONGCTX_4M),
    "tuan-3b": _p(vocab_size=32000, dim=2560, num_layers=32, num_bases=128,
                  decay_max=16384, local_kernel=4, num_heads=20, ffn_ratio=4.0,
                  context=_CTX_LONGCTX_4M),
    # ------------------------------------------------------------------
    # 对话能力阶梯：30M 起步 + 1B 效率目标
    #
    # 这一组不是"把 150m 等比缩小"的产物 —— 那样会掉进大词表陷阱：
    # V=32000, d=512, L=10 是 44.1M，其中 16.4M（37%）只在嵌入表上，
    # 而 30M 级模型的数据量（<1B token）根本训不充分 32k 行的低频尾。
    # 这里 V=12288 把嵌入压到 21%，省下的 10M 全部落到 10 层可学习参数上。
    # 见 ``simplcore/core/scaling.py`` 的 ``plan_budget``。
    #
    # A/B 配对（同词表、同宽深比、总参数差 <1%）：
    #   tuan-30m  = 30.1M total / 23.8M non-emb / d512 L10
    #   srf-30m  = 与 tuan-30m 完全同宽度/深度/词表，只换 mixer（SRFN）
    # ------------------------------------------------------------------
    "tuan-30m": _p(vocab_size=12288, dim=512, num_layers=10, num_bases=32,
                  decay_max=4096, local_kernel=4, num_heads=8, ffn_ratio=2.0),
    # 谱振流网络（SRFN）：与 tuan-30m 完全同宽度/深度/词表，只换 mixer。
    # 这是"融入 SSM 之后到底有没有变好"的唯一合法对照。
    "srf-30m": _p(vocab_size=12288, dim=512, num_layers=10, num_bases=32,
                  decay_max=4096, local_kernel=4, num_heads=8, ffn_ratio=2.0,
                  mixer="srf", srf_heads=8, srf_chunk=32),
    # 1B 效率目标（MI250 单卡满预算训练，见 docs/TRAINING_BUDGET_MI250.md）：
    # SimplCore 1B 与同等预算的传统 attention 1B 的受控 A/B（1013M vs 1005M）。
    "tuan-1b": _p(vocab_size=32000, dim=2048, num_layers=15, num_bases=128,
                 decay_max=16384, local_kernel=4, num_heads=16, ffn_ratio=4.0,
                 context=_CTX_LONGCTX_4M),
    # 2B 主干（30 天单 MI250 交付档）：2023M total / 1957M non-emb，
    # 与 tuan-1b 同宽（d=2048，kernel/调优可复用）、深度翻倍（31 层），
    # 宽深比 66，嵌入占比 3.1%。按 25% MFU 的 30 天预算约 13B tokens：
    # 这是"欠训但可用"的 2B，不是 Chinchilla-optimal（20× = 40B，需 ~92 天）。
    # 4M 课程占末段 10%（_CTX_2B_4M_30D），跑法：--train-chunk 8192 + 2 GCD DDP。
    "tuan-2b": _p(vocab_size=32000, dim=2048, num_layers=31, num_bases=128,
                  decay_max=4_000_000, local_kernel=4, num_heads=16, ffn_ratio=4.0,
                  sgc_init_mode="logfreq", sgc_rho_param="logspace",
                  context=_CTX_2B_4M_30D),
    # Sprint-30 降档档位（simplcore/core/sprint30.py）：~0.87B，每 token FLOPs
    # 更低，用于实测 MFU 低于下限时换回 ≥15B token 预算——交付出一个训完的
    # 0.87B，好过一个没训完的 1B。
    "tuan-870m": _p(vocab_size=32000, dim=1728, num_layers=18, num_bases=128,
                   decay_max=16384, local_kernel=4, num_heads=24, ffn_ratio=4.0,
                   context=_CTX_LONGCTX_4M),
    # 原生智能阈值档（simplcore/core/native_intelligence.py，纯解析 preset，
    # 单卡 MI250 不可训练——账本给出最小 GPU 数）：
    # 120B 主干 + 选择性 SRF mixer（输入相关 Δ，非注意力）+ 2B(2^31) token
    # 上下文视野（O(1) 流式状态，结构上与 L 无关）。SGC v1.2 增强全开。
    "tuan-120b": _p(vocab_size=32000, dim=8192, num_layers=118, num_bases=256,
                   decay_max=1048576.0, local_kernel=4, num_heads=64, ffn_ratio=4.0,
                   mixer="srf", srf_heads=64, srf_chunk=64,
                   sgc_init_mode="logfreq", sgc_path_gate=True, sgc_inst_nonlin=True),
}


def get_preset(name: str) -> SimplCoreConfig:
    if name not in SIMPLCORE_PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(SIMPLCORE_PRESETS)}")
    return SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[name].to_dict())


# ---------------------------------------------------------------------------
# Analytic parameter accounting
# ---------------------------------------------------------------------------
def sgc_gate_params(cfg: SimplCoreConfig) -> int:
    """四层闸门在**单个 SGC mixer** 上额外引入的标量参数量。

    * L2 证据门：``evidence_gain`` + ``evidence_pos_gain`` + ``evidence_bias``（3 个标量）
    * L3 不确定门：``uncertainty_logit``（1 个标量）

    L4 拒答门在模型层（``use_abstain``），不计入 mixer。
    """
    n = 0
    if getattr(cfg, "evidence_gate", False):
        n += 3
    if getattr(cfg, "uncertainty_gate", False):
        n += 1
    return n


def analytic_params(cfg: SimplCoreConfig) -> int:
    """Exact parameter count of :class:`~simplcore.core.model.SimplCoreModel` for ``cfg``.

    Computed from the architecture formula rather than by instantiating the model,
    so the 1.3B/3B presets can be inspected on a machine that cannot hold them.
    The count includes the token embedding (and the tied output head, which shares
    it) and every per-layer term:

    * SGC mixer: in/out projections (2·D²), optional content gate (D²),
      optional output gate (D²), oscillator bank (M·D coefficients + 2·M
      parameters), instantaneous ``feed`` (D), depthwise local conv (D·K),
      ``out_scale`` (1); SRFN replaces the oscillator bank with a selective
      SSM (2·M·D coupling + 2·M + 2·D + 1);
    * each block: SwiGLU gate/up/down (3·D·H) + two norm weights (2·D);
    * the final RMSNorm ``ln_f`` (D weight).

    ``gate_ab`` adds ``2·D² + 2·D`` to the multi-head spectral mixer term
    (gamma/beta channel gates: two ``D×D`` projections plus two length-``D``
    bias vectors).  Verified bit-exactly against
    ``SimplCoreModel(cfg).num_params()`` in ``tests/test_mixer_math.py``.

    mixer 家族分支来自注册表（``MixerSpec.family``），变体语义
    （``sgc-nogate`` 不吃 content_gate、``sgc-nolocal`` 不吃 local_conv）
    也从注册表读取——构造分支与参数公式不再各写一遍。
    """
    cfg.validate()
    d, L, V, h = cfg.dim, cfg.num_layers, cfg.vocab_size, cfg.hidden

    from .registry import get_mixer_spec
    spec = get_mixer_spec(cfg.mixer)

    def sgc_per() -> int:
        per = 2 * d * d
        if cfg.content_gate and spec.respects_content_gate:
            per += d * d
        if cfg.out_gate:
            per += d * d
        if cfg.gate_ab:
            # γ/β 通道门：两个 Linear(dim→dim) 各 D²，加上两个独立偏置向量各 D
            per += 2 * d * d + 2 * d
        if spec.family == "srf":
            # selective SSM 的耦合项：b_in 与 coef 各 M·D（SGC 只有 coef 一份），
            # 加直通 skip(D)、dt_proj(D+1)、rho/theta 各 M。
            per += 2 * cfg.num_bases * d + 2 * cfg.num_bases + 2 * d + 1
        else:
            per += cfg.num_bases * d + cfg.num_bases * 2 + d
        k = cfg.local_kernel if spec.respects_local_conv else 1
        per += d * k + 1
        per += sgc_gate_params(cfg)
        # SGC v1.2 增强：path_gate 的三路路由 W(D→3)+b(3)
        if getattr(cfg, "sgc_path_gate", False):
            per += 3 * d + 3
        # SGC-R 谱专家路由器：W(D→G)+b(G)
        if cfg.mixer == "sgc-r":
            per += cfg.sgc_route_groups * d + cfg.sgc_route_groups
        return per

    per = sgc_per()
    per += 3 * d * h + 2 * d
    total = V * d + L * per          # token embedding (tied head shares it)
    if not cfg.tie_embeddings:
        total += V * d               # separate, untied output head
    total += d                       # final RMSNorm (ln_f) weight
    if getattr(cfg, "use_memory", False):
        total += d * d               # CMT 记忆注入投影 (memory_in)
    return total


def report_presets(names=None) -> int:
    """Console entry point (``simplcore-presets``): print the preset table."""
    print(f"{'preset':14s} {'params':>14s} {'bf16 weights':>13s}  mixer  dim/depth/M")
    for name in (names or SIMPLCORE_PRESETS):
        cfg = SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[name].to_dict())
        n = analytic_params(cfg)
        print(f"{name:14s} {n:14,d} {n * 2 / 1e9:12.2f}G  {cfg.mixer:6s} "
              f"{cfg.dim}/{cfg.num_layers}/{cfg.num_bases}")
    return 0
