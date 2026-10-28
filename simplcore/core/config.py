# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore configuration objects and MI250-80GB scale presets."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class SimplCoreConfig:
    """Configuration of a SimplCore language model (non-attention backbone).

    Note there is **no positional-embedding table and no ``max_len``**: the causal
    spectral convolution is inherently position aware, so sequences longer than
    any value seen in training remain well defined — and, unlike a Transformer
    decoder, there is **no softmax self-attention and no KV cache** that grows
    with the sequence.

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
        uncertainty_gate: L3 uncertainty gate (long/local disagreement damps the long path).
        use_abstain: attach the L4 abstention head to the model.
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
    evidence_gate: bool = True
    uncertainty_gate: bool = True
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


SIMPLCORE_PRESETS = {
    # local CPU / unit-test sizes
    "debug": _p(vocab_size=1024, dim=128, num_layers=2, num_bases=16, decay_max=256,
                local_kernel=4, num_heads=4),
    "toy": _p(vocab_size=4096, dim=256, num_layers=4, num_bases=32, decay_max=1024,
              local_kernel=4, num_heads=8),
    # 1 x A100 80GB targets (bf16, seq 4096)
    "sgc-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                    decay_max=8192, local_kernel=4, num_heads=12),
    "sgc-350m": _p(vocab_size=32000, dim=1024, num_layers=20, num_bases=64,
                    decay_max=8192, local_kernel=4, num_heads=16),
    "sgc-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, num_bases=128,
                    decay_max=16384, local_kernel=4, num_heads=16, ffn_ratio=4.0),
    "sgc-3b": _p(vocab_size=32000, dim=2560, num_layers=32, num_bases=128,
                  decay_max=16384, local_kernel=4, num_heads=20, ffn_ratio=4.0),
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
    #   sgc-30m  = 30.1M total / 23.8M non-emb / d512 L10
    #   srf-30m  = 与 sgc-30m 完全同宽度/深度/词表，只换 mixer（SRFN）
    # ------------------------------------------------------------------
    "sgc-30m": _p(vocab_size=12288, dim=512, num_layers=10, num_bases=32,
                  decay_max=4096, local_kernel=4, num_heads=8, ffn_ratio=2.0),
    # 谱振流网络（SRFN）：与 sgc-30m 完全同宽度/深度/词表，只换 mixer。
    # 这是"融入 SSM 之后到底有没有变好"的唯一合法对照。
    "srf-30m": _p(vocab_size=12288, dim=512, num_layers=10, num_bases=32,
                  decay_max=4096, local_kernel=4, num_heads=8, ffn_ratio=2.0,
                  mixer="srf", srf_heads=8, srf_chunk=32),
    # 1B 效率目标（MI250 单卡满预算训练，见 docs/TRAINING_BUDGET_MI250.md）：
    # SimplCore 1B 与同等预算的传统 attention 1B 的受控 A/B（1013M vs 1005M）。
    "sgc-1b": _p(vocab_size=32000, dim=2048, num_layers=15, num_bases=128,
                 decay_max=16384, local_kernel=4, num_heads=16, ffn_ratio=4.0),
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
    """Console entry point (``sgc-presets``): print the preset table."""
    print(f"{'preset':14s} {'params':>14s} {'bf16 weights':>13s}  mixer  dim/depth/M")
    for name in (names or SIMPLCORE_PRESETS):
        cfg = SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[name].to_dict())
        n = analytic_params(cfg)
        print(f"{name:14s} {n:14,d} {n * 2 / 1e9:12.2f}G  {cfg.mixer:6s} "
              f"{cfg.dim}/{cfg.num_layers}/{cfg.num_bases}")
    return 0
