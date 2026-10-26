# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC configuration objects and A100-80GB scale presets."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class SGCConfig:
    """Configuration of an SGC language model.

    Note there is **no positional-embedding table and no ``max_len``**: the causal
    convolution is inherently position aware, so sequences longer than any value
    seen in training remain well defined (see ``docs/SGC_SPEC.md``).

    Attributes:
        vocab_size: tokenizer vocabulary size.
        dim: model width D.
        num_layers: number of blocks.
        ffn_hidden: SwiGLU hidden width; ``None`` -> ``ffn_ratio * dim``.
        ffn_ratio: FFN expansion when ``ffn_hidden`` is None.
        mixer: mixer kind: ``sgc`` | ``sgc-nogate`` | ``sgc-nolocal`` | ``attn``.
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
        num_heads: attention-baseline head count (ignored by SGC mixers).
        tie_embeddings: tie the output head to the token embedding.
        dropout: residual/embedding dropout.
        norm_eps: normalisation epsilon.
        fft_dtype: dtype for the FFT path (``"float32"`` recommended).
        grad_checkpoint: wrap blocks in activation checkpointing.
        init_std: std of the normal-distribution init for linear layers.
        attn_every: SGC Hybrid hybrid: insert a sliding-window attention layer every
            ``attn_every`` blocks (``None`` disables hybrid; ``1`` = every layer).
        attn_window: sliding-window size W of the hybrid attention layers.
        num_kv_heads: GQA kv head count for hybrid attention layers
            (``None`` = full MHA, i.e. kv heads = num_heads).
        gate_ab: enable GLA/Mamba2-style channel-independent gamma/beta
            content gates on the SGC mixer layers of hybrid stacks.
        rope_scale_strategy: attention RoPE scaling strategy:
            ``"none"`` (fixed RoPE), ``"linear"``, or ``"yarn"`` (NTK-aware).
        rope_scale_factor: scaling factor for the RoPE strategy (>1 = extrapolate).
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
    attn_every: Optional[int] = None
    attn_window: int = 256
    num_kv_heads: Optional[int] = None
    gate_ab: bool = False
    rope_scale_strategy: str = "none"
    rope_scale_factor: float = 1.0
    # SGC Hybrid 无词表模式：编解码器类型（"char" | "byte" | "bpe"），
    # 及其参数字典（透传给 tokenization.make_codec）。
    # 模型 vocab_size 仍须与 codec.size() 一致（validate 不做自动校验，
    # 由调用方保证；init_sgc_from_hf 的 cfg_override 路径推荐用
    # ``SGCConfig.from_codec(codec)`` 工厂方法自动填充）。
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
    def from_dict(cls, d: dict) -> "SGCConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown SGCConfig fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_codec(cls, codec, **kw) -> "SGCConfig":
        """由编解码器构造 config：自动填充 ``vocab_size = codec.size()`` 及
        ``codec_kind`` / ``codec_kwargs``（供检查点序列化还原 codec）。

        额外 kw 透传给 ``SGCConfig``（如 ``dim`` / ``num_layers`` / ``mixer``）。
        这是"舍去传统 Transformer 词表"的推荐入口：

            from simplcore.sgc import CharTokenizer, SGCConfig
            cfg = SGCConfig.from_codec(CharTokenizer(), dim=384, num_layers=6)
            m = SGCModel(cfg)   # vocab_size 自动 = codec.size()
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
    def from_json(cls, path: str) -> "SGCConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> "SGCConfig":
        if self.dim % 2:
            raise ValueError("dim must be even for the FFT path")
        if self.mixer not in ("sgc", "sgc-nogate", "sgc-nolocal", "attn", "hybrid",
                               "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn"):
            raise ValueError(f"unknown mixer {self.mixer!r}")
        if self.mixer in ("attn", "hybrid", "hybrid-gqa", "hybrid-yarn",
                           "hybrid-gqa-yarn") \
                and self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads for attention/hybrid mixers")
        if self.mixer in ("hybrid", "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn") \
                and (self.attn_every is not None and self.attn_every < 1):
            raise ValueError("attn_every must be >= 1 when hybrid is enabled")
        if self.num_kv_heads is not None:
            if self.num_kv_heads < 1 or self.num_kv_heads > self.num_heads \
                    or self.num_heads % self.num_kv_heads:
                raise ValueError("num_kv_heads must be a divisor of num_heads in [1, num_heads]")
        if self.mixer == "hybrid-gqa-yarn" and (
                self.num_kv_heads is None or self.num_kv_heads >= self.num_heads):
            raise ValueError("hybrid-gqa-yarn requires num_kv_heads < num_heads")
        if self.mixer == "hybrid-gqa-yarn" and \
                self.rope_scale_strategy not in ("linear", "yarn"):
            raise ValueError("hybrid-gqa-yarn requires rope_scale_strategy in ('linear','yarn')")
        if self.rope_scale_strategy not in ("none", "linear", "yarn"):
            raise ValueError(f"unknown rope_scale_strategy {self.rope_scale_strategy!r}")
        if self.rope_scale_factor <= 0:
            raise ValueError("rope_scale_factor must be > 0")
        if self.num_bases < 1:
            raise ValueError("num_bases must be >= 1")
        if self.local_kernel < 1:
            raise ValueError("local_kernel must be >= 1")
        return self


# ----------------------------------------------------------------------------
# Presets.  Sized for the stated target compute: 1 x A100 80GB, bf16,
# sequence length 4096, AdamW.  Parameter counts below in the docs come from
# `analytic_params()` / `sgc-presets` (analytic, not instantiated).
# ----------------------------------------------------------------------------
def _p(**kw) -> SGCConfig:
    cfg = SGCConfig(**kw)
    return cfg


SGC_PRESETS = {
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
    # matched attention baselines (same width/depth/FFN, only the mixer differs)
    "attn-150m": _p(vocab_size=32000, dim=768, num_layers=12, mixer="attn", num_heads=12),
    "attn-350m": _p(vocab_size=32000, dim=1024, num_layers=20, mixer="attn", num_heads=16),
    "attn-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, mixer="attn", num_heads=16,
                    ffn_ratio=4.0),
    "attn-3b": _p(vocab_size=32000, dim=2560, num_layers=32, mixer="attn", num_heads=20,
                   ffn_ratio=4.0),
    # SGC Hybrid hybrid presets: global SGC + sliding-window attention interleaved
    "hybrid-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                      decay_max=8192, local_kernel=4, num_heads=12,
                      mixer="hybrid", attn_every=4, attn_window=256),
    "hybrid-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, num_bases=128,
                      decay_max=16384, local_kernel=4, num_heads=16,
                      mixer="hybrid", attn_every=4, attn_window=512, ffn_ratio=4.0),
    "hybrid-gqa-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                           decay_max=8192, local_kernel=4, num_heads=12,
                           mixer="hybrid", attn_every=4, attn_window=256, num_kv_heads=6),
    "hybrid-gqa-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, num_bases=128,
                           decay_max=16384, local_kernel=4, num_heads=16,
                           mixer="hybrid", attn_every=4, attn_window=512,
                           num_kv_heads=8, ffn_ratio=4.0),
    "hybrid-yarn-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                            decay_max=8192, local_kernel=4, num_heads=12,
                            mixer="hybrid", attn_every=4, attn_window=256,
                            rope_scale_strategy="yarn", rope_scale_factor=2.0),
    "hybrid-gqa-yarn-150m": _p(vocab_size=32000, dim=768, num_layers=12, num_bases=64,
                                decay_max=8192, local_kernel=4, num_heads=12,
                                mixer="hybrid", attn_every=4, attn_window=256,
                                num_kv_heads=6,
                                rope_scale_strategy="yarn", rope_scale_factor=2.0),
    "hybrid-gqa-yarn-1_3b": _p(vocab_size=32000, dim=2048, num_layers=20, num_bases=128,
                                decay_max=16384, local_kernel=4, num_heads=16,
                                mixer="hybrid", attn_every=4, attn_window=512,
                                num_kv_heads=8,
                                rope_scale_strategy="yarn", rope_scale_factor=2.0,
                                ffn_ratio=4.0),
}


def get_preset(name: str) -> SGCConfig:
    if name not in SGC_PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(SGC_PRESETS)}")
    return SGCConfig.from_dict(SGC_PRESETS[name].to_dict())


# ---------------------------------------------------------------------------
# Analytic parameter accounting
# ---------------------------------------------------------------------------
def sgc_gate_params(cfg: SGCConfig) -> int:
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


def analytic_params(cfg: SGCConfig) -> int:
    """Exact parameter count of :class:`~simplcore.sgc.model.SGCModel` for ``cfg``.

    Computed from the architecture formula rather than by instantiating the model,
    so the 1.3B/3B presets can be inspected on a machine that cannot hold them.
    The count includes the token embedding (and the tied output head, which shares
    it) and every per-layer term:

    * attention baseline: 4·D² for qkv + out_proj;
    * SGC mixer: in/out projections (2·D²), optional content gate (D²),
      optional output gate (D²), oscillator bank (M·D coefficients + 2·M
      parameters), instantaneous ``feed`` (D), depthwise local conv (D·K),
      ``out_scale`` (1);
    * each block: SwiGLU gate/up/down (3·D·H) + two norm weights (2·D);
    * the final RMSNorm ``ln_f`` (D weight).

    GQA (``num_kv_heads``) shrinks the attention projection term to
    ``D² + 2·D·G·(D//H) + D² + D``; ``gate_ab`` adds ``2·D²`` to the SGC
    mixer term (gamma/beta channel gates).  Verified bit-exactly against
    ``SGCModel(cfg).num_params()`` / ``analytic_hybrid_params(cfg)`` in
    ``tests/test_mixer_math.py`` / ``tests/test_hybrid.py``.
    """
    cfg.validate()
    d, L, V, h = cfg.dim, cfg.num_layers, cfg.vocab_size, cfg.hidden
    kv = cfg.num_kv_heads or cfg.num_heads
    head_dim = d // cfg.num_heads

    def attn_per() -> int:
        return d * d + 2 * d * kv * head_dim + d * d

    def sgc_per() -> int:
        per = 2 * d * d
        if cfg.content_gate and cfg.mixer not in ("sgc-nogate",):
            per += d * d
        if cfg.out_gate:
            per += d * d
        if cfg.gate_ab:
            per += 2 * d * d
        per += cfg.num_bases * d + cfg.num_bases * 2 + d
        k = 1 if cfg.mixer == "sgc-nolocal" else cfg.local_kernel
        per += d * k + 1
        per += sgc_gate_params(cfg)
        return per

    # 与 :func:`simplcore.sgc.hybrid.analytic_hybrid_params` 同一公式口径：
    # 每层 SwiGLU + 两个 norm（3·d·h + 2·d），注意力层 mixer 核
    # 2·d² + 2·d·kv·hd（不含 norm，norm 在 2·d 里统一计）
    attn_core = 2 * d * d + 2 * d * kv * head_dim

    is_attn_family = cfg.mixer in ("attn", "hybrid", "hybrid-gqa", "hybrid-yarn",
                                   "hybrid-gqa-yarn")
    is_hybrid = cfg.mixer in ("hybrid", "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn")

    if is_hybrid:
        from .hybrid import analytic_hybrid_params
        return analytic_hybrid_params(cfg)
    if cfg.mixer == "attn":
        per = attn_per()
    else:
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
    for name in (names or SGC_PRESETS):
        cfg = SGCConfig.from_dict(SGC_PRESETS[name].to_dict())
        n = analytic_params(cfg)
        print(f"{name:14s} {n:14,d} {n * 2 / 1e9:12.2f}G  {cfg.mixer:6s} "
              f"{cfg.dim}/{cfg.num_layers}/{cfg.num_bases}")
    return 0
