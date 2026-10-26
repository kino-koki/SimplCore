# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC Hybrid —— 混合架构：全局谱卷积 + 局部滑窗注意力交替堆叠.

设计依据（当前最先进混合架构路线，如 LongNet / Jamba / Mamba 混合实践）
========================================================================
* 全局路径仍用 :class:`~simplcore.sgc.mixer.SpectralGlobalConv`
  （O(L log L) 训练 + O(1) 状态流式 + 内容门控 + 无位置嵌入）；
* 在每 ``attn_every`` 层插入一个**局部滑窗因果注意力**
  （:class:`SlidingWindowAttention`），每 head 只看 ``window`` 个 token，
  复杂度 O(L·W)（W = 滑窗大小），**不引入 O(L²) 全注意力，也不引入
  随 L 增长的 KV 缓存**；
* 默认 ``attn_every=4`` 即 1/4 的层用注意力（与 LongNet 的局部:全局
  比例同量级），可设 ``attn_every=1`` 恢复"每层都有注意力"，或
  ``attn_every=None`` 退化为纯 SGC。

为什么混合而不是纯全局
----------------------
SGC 的谱全局卷积擅长**平滑的长程结构**（谱带限），但对**尖锐的
局部 token 级路由**（检索、指代消解）不如局部注意力直接。混合架构让
局部滑窗注意力承担"近处精确交互"，全局谱卷积承担"远处平滑结构"，
二者互补 —— 这是 LongNet / Jamba 的核心结论，也是当前主流混合路线。

继承的 SimplCore 特性（全部原样保留）
--------------------------------------
* 严格因果（未来扰动不改变过去，与 ``test_mixer_math`` 守护协议一致）；
* **有界流式状态**：SGC 层 O(1)、注意力层 O(W·H·D)（滑窗 KV），
  二者都与 L 无关 —— 由 :meth:`HybridModel.stream_bytes` 精确核算；
* 内容门控（SGC 层 selective 核）；
* 无位置嵌入表（注意力用 RoPE，SGC 用因果卷积本身编码位置）。

API
---
* ``SGCConfig(mixer="hybrid", attn_every=4, attn_window=256)``；
* ``HybridModel(cfg)`` —— 前向 / 流式 / 生成；
* ``analytic_hybrid_params(cfg)`` —— 精确参数量（不实例化）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SGCConfig, sgc_gate_params
from .mixer import (
    SGCStreamState,
    SpectralGlobalConv,
    make_norm,
)
from ._rope import _apply_rope, _rope_tables
from .model import SwiGLU


# =============================================================================
# 局部滑窗因果注意力
# =============================================================================
class SlidingWindowAttention(nn.Module):
    """因果多头自注意力，每个 query 只看 ``window`` 个 key（局部滑窗）。

    复杂度 O(L·W)（W = window），无 O(L²) 项；流式推理时只保留最近
    ``window`` 个 key/value（每 head），状态形状 ``(B, H, W, head_dim)`` ×2，
    **与 L 无关**。

    与 :class:`~simplcore.sgc.model.CausalAttention` 骨架一致（QKV 投影 +
    out_proj + 残差 + RMSNorm），差异仅在于因果掩码被收紧为"因果 + 滑窗"，
    便于按同骨架做消融。
    """

    def __init__(self, dim: int, num_heads: int, window: int = 256,
                 norm: str = "rmsnorm", dropout: float = 0.0,
                 eps: float = 1e-6, rope_theta: float = 10000.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = max(1, int(window))
        self.rope_theta = float(rope_theta)
        self.norm = make_norm(norm, dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # -- 掩码 -----------------------------------------------------------------
    def _windowed_causal_mask(self, L: int, device, dtype) -> torch.Tensor:
        """(L, L) 掩码：仅当 0 <= j <= t 且 t - j < window 时可见。

        用有限大负数（而非 -inf）避免 bf16 下 -inf*0=NaN；与 SGC 主线
        因果掩码同一策略。
        """
        idx = torch.arange(L, device=device)
        j = idx.view(1, -1)
        t = idx.view(-1, 1)
        visible = (j <= t) & ((t - j) < self.window)
        neg = torch.finfo(dtype).min / 4
        mask = torch.zeros(L, L, device=device, dtype=dtype)
        return mask.masked_fill(~visible, neg)

    # -- 前向 -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))

        cos, sin = _rope_tables(self.head_dim, L, x.device, q.dtype, base=self.rope_theta)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = self._windowed_causal_mask(L, q.device, q.dtype)
        attn = F.softmax(attn + mask.unsqueeze(0).unsqueeze(0), dim=-1)
        out = torch.matmul(self.drop(attn), v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return x + self.out_proj(out)

    # -- 流式 -----------------------------------------------------------------
    def init_stream(self, batch: int, device, dtype) -> Dict:
        H = self.num_heads
        return {
            "k": torch.zeros(batch, H, self.window, self.head_dim,
                             device=device, dtype=dtype),
            "v": torch.zeros(batch, H, self.window, self.head_dim,
                             device=device, dtype=dtype),
            "n": 0,
        }

    @torch.no_grad()
    def stream_step(self, x_t: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, Dict]:
        """Consume one token (B, 1, D) or (B, D); return (B, 1, D) + new state.

        Sliding-window KV is a ring buffer (B, H, W, head_dim); state size
        O(W·H·D), independent of L. The cache stores **raw k/v (no RoPE)**;
        each step re-applies RoPE to the in-window keys by their absolute
        positions so the result matches the full-sequence forward exactly.
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        assert one == 1
        H = self.num_heads
        h = self.norm(x_t)
        qkv = self.qkv(h).reshape(B, 1, 3, H, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        # q/k/v: (B, H, 1, hd) -> take L=1
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]   # (B, H, hd)

        pos = state["n"]
        dev = x_t.device
        W = self.window
        # current query/key at absolute position pos
        cos_q, sin_q = _rope_tables(self.head_dim, pos + 1, dev, q.dtype,
                                    base=self.rope_theta)
        q_rot = _apply_rope(q.unsqueeze(2), cos_q[pos:pos + 1],
                            sin_q[pos:pos + 1])[:, :, 0]      # (B, H, hd)

        # push raw k/v into the ring (slot in the W dim, shape (B, H, hd))
        slot = pos % W
        state["k"][:, :, slot] = k
        state["v"][:, :, slot] = v
        state["n"] = pos + 1
        n = state["n"]

        # in-window keys: absolute positions [n - w .. n - 1], w = min(W, n)
        w = min(W, n)
        start_abs = n - w
        # slot of absolute position p is p % W
        slots = torch.arange(start_abs, n, device=dev) % W
        Ks = state["k"][:, :, slots]      # (B, H, w, hd)
        Vs = state["v"][:, :, slots]      # (B, H, w, hd)
        # re-apply RoPE to in-window keys by their absolute positions
        cos_k, sin_k = _rope_tables(self.head_dim, n, dev, Ks.dtype,
                                    base=self.rope_theta)
        Ks_rot = _apply_rope(Ks, cos_k[start_abs:n], sin_k[start_abs:n])  # (B, H, w, hd)

        attn = torch.matmul(q_rot.unsqueeze(2), Ks_rot.transpose(2, 3)) \
            * (self.head_dim ** -0.5)                            # (B, H, 1, w)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Vs).transpose(1, 2).reshape(B, D)
        y = x_t.squeeze(1) + self.out_proj(out)
        return y.unsqueeze(1), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        return 2 * batch * self.num_heads * self.window * self.head_dim * dtype.itemsize

    # SGCModel.stream_bytes 统一探测 ``streaming_state_bytes``，提供别名
    streaming_state_bytes = stream_bytes


# =============================================================================
# 混合块
# =============================================================================
@dataclass
class HybridLayerState:
    """一层混合块的流式状态：SGC 层的 LayerStreamState 或注意力层的滑窗 KV。"""
    sgc_state: Optional[object] = None     # SpectralGlobalConv 的 LayerStreamState
    attn_state: Optional[Dict] = None      # SlidingWindowAttention 的滑窗 KV


class HybridBlock(nn.Module):
    """单层混合块：mixer（SGC 或 滑窗注意力）+ SwiGLU FFN，各带残差。

    与 :class:`~simplcore.sgc.model.SGCBlock` 同骨架，仅把 mixer 换成
    按层切换的 SGC / 滑窗注意力。
    """

    def __init__(self, config: SGCConfig, layer_idx: int,
                 use_attn: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_attn = use_attn
        if use_attn:
            kv = getattr(config, "num_kv_heads", None) or config.num_heads
            strat = getattr(config, "rope_scale_strategy", "none")
            factor = getattr(config, "rope_scale_factor", 1.0)
            if strat in ("linear", "yarn") and kv < config.num_heads:
                # GQA + YaRN 组合：KV 按 G 头存（省 H/G 倍）+ NTK 位置缩放外推
                from .attention_yarn import GQASlidingWindowAttentionYaRN
                self.mixer = GQASlidingWindowAttentionYaRN(
                    dim=config.dim, num_heads=config.num_heads,
                    window=getattr(config, "attn_window", 256),
                    num_kv_heads=kv,
                    norm=config.norm, dropout=config.dropout, eps=config.norm_eps,
                    strategy=strat, factor=factor,
                )
            elif strat in ("linear", "yarn"):
                from .rope_scales import SlidingWindowAttentionYaRN
                self.mixer = SlidingWindowAttentionYaRN(
                    dim=config.dim, num_heads=config.num_heads,
                    window=getattr(config, "attn_window", 256),
                    norm=config.norm, dropout=config.dropout, eps=config.norm_eps,
                    strategy=strat, factor=factor,
                )
            elif kv < config.num_heads:
                from .attention import GQASlidingWindowAttention
                self.mixer = GQASlidingWindowAttention(
                    dim=config.dim, num_heads=config.num_heads,
                    window=getattr(config, "attn_window", 256),
                    num_kv_heads=kv,
                    norm=config.norm, dropout=config.dropout, eps=config.norm_eps,
                )
            else:
                self.mixer = SlidingWindowAttention(
                    dim=config.dim, num_heads=config.num_heads,
                    window=getattr(config, "attn_window", 256),
                    norm=config.norm, dropout=config.dropout, eps=config.norm_eps,
                )
        else:
            gate_ab = getattr(config, "gate_ab", False)
            if gate_ab:
                from .selective import GatedSpectralConv2
                self.mixer = GatedSpectralConv2(
                    dim=config.dim, num_bases=config.num_bases,
                    decay_max=config.decay_max, local_kernel=config.local_kernel,
                    num_heads=config.num_heads,
                    content_gate=config.content_gate, out_gate=config.out_gate,
                    gate_ab=gate_ab,
                    norm=config.norm, fft_dtype=config.resolve_fft_dtype(),
                    eps=config.norm_eps,
                )
            else:
                self.mixer = SpectralGlobalConv(
                    dim=config.dim, num_bases=config.num_bases,
                    decay_max=config.decay_max, local_kernel=config.local_kernel,
                    content_gate=config.content_gate, out_gate=config.out_gate,
                    evidence_gate=config.evidence_gate,
                    uncertainty_gate=config.uncertainty_gate,
                    norm=config.norm, fft_dtype=config.resolve_fft_dtype(),
                    eps=config.norm_eps,
                )
        self.ffn = SwiGLU(config.dim, config.hidden, norm=config.norm,
                          dropout=config.dropout, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        if memory is not None:
            # CMT 式记忆注入：残差流上的加性项（SGC 无 KV 缓存，原生注入点）
            x = x + memory.unsqueeze(1)
        if self.config.grad_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.mixer, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False)
            return x
        return self.ffn(self.mixer(x))

    def init_stream(self, batch: int, device, dtype) -> HybridLayerState:
        if self.use_attn:
            return HybridLayerState(attn_state=self.mixer.init_stream(batch, device, dtype))
        return HybridLayerState(sgc_state=self.mixer.init_state(batch, device, dtype))

    def stream_step(self, x_t: torch.Tensor, st: HybridLayerState,
                    memory: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, HybridLayerState]:
        if memory is not None:
            # CMT 式记忆注入：残差流上的加性项
            x_t = x_t + memory.unsqueeze(1)
        if self.use_attn:
            x_t, st.attn_state = self.mixer.stream_step(x_t, st.attn_state)
        else:
            x_t, st.sgc_state = self.mixer.step(x_t, st.sgc_state)
        x_t = self.ffn(x_t)
        return x_t, st


# =============================================================================
# 混合模型
# =============================================================================
def _attn_layout(num_layers: int, attn_every: Optional[int]) -> List[bool]:
    """每层是否用滑窗注意力：(i+1) % attn_every == 0。"""
    if not attn_every:
        return [False] * num_layers
    return [((i + 1) % attn_every == 0) for i in range(num_layers)]


class HybridModel(nn.Module):
    """SGC Hybrid 混合因果语言模型：全局谱卷积 + 局部滑窗注意力交替。

    * 每 ``attn_every`` 层插入一个 :class:`SlidingWindowAttention`
      （局部精确交互，O(L·W)），其余层用 :class:`SpectralGlobalConv`
      （长程平滑结构，O(L log L)）；
    * 继承 SGC 全部特性：严格因果、有界流式状态、内容门控、
      无位置嵌入表；
    * 流式状态字节数 = SGC 层 O(1) 项 + 注意力层 O(W·H·D) 滑窗项，
      二者都与 L 无关，由 :meth:`stream_bytes` 精确核算。
    """

    def __init__(self, config: SGCConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        layout = _attn_layout(config.num_layers, getattr(config, "attn_every", None))
        self.use_attn = layout
        self.blocks = nn.ModuleList([
            HybridBlock(config, i, use_attn) for i, use_attn in enumerate(layout)
        ])
        self.ln_f = make_norm(config.norm, config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # CMT 记忆注入投影：零初始化 → 初始为恒等（不改变基座行为），训练后自学
        self.memory_in = (nn.Linear(config.dim, config.dim, bias=False)
                          if getattr(config, "use_memory", False) else None)
        if self.memory_in is not None:
            with torch.no_grad():
                self.memory_in.weight.zero_()
        # 残差深度缩放（out 投影 / SwiGLU down 按 1/sqrt(2L) 缩小，
        # 与主流 LLaMA 风格 init 一致，利于深层训练稳定）。
        scale = config.init_std / math.sqrt(2 * config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith(".out_proj.weight") or name.endswith(".down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)

    # -- 前向 -----------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                return_hidden: bool = False,
                memory: Optional[torch.Tensor] = None):
        x = self.drop(self.tok_emb(idx))
        mem = self._project_memory(memory)
        for blk in self.blocks:
            x = blk(x, mem)
        x = self.ln_f(x)
        logits = self.head(x)
        if return_hidden:
            return logits, x
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return logits, loss
        return logits

    # -- 流式 -----------------------------------------------------------------
    def _project_memory(self, memory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """把检索到的压缩记忆投影到模型宽度（未启用时返回 None）。"""
        if memory is None or getattr(self, "memory_in", None) is None:
            return None
        return self.memory_in(memory)

    def init_stream(self, batch: int, device=None, dtype=None) -> List[HybridLayerState]:
        device = device or self.tok_emb.weight.device
        dtype = dtype or self.tok_emb.weight.dtype
        return [blk.init_stream(batch, device, dtype) for blk in self.blocks]

    def stream_step(self, token: torch.Tensor, states: List[HybridLayerState],
                    memory: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, List[HybridLayerState]]:
        if token.dim() == 1:
            token = token.unsqueeze(1)
        x = self.tok_emb(token)
        mem = self._project_memory(memory)
        for blk, st in zip(self.blocks, states):
            x, st = blk.stream_step(x, st, mem)
        x = self.ln_f(x)
        return self.head(x[:, -1, :]), states

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        total = 0
        for blk in self.blocks:
            if blk.use_attn:
                total += blk.mixer.stream_bytes(batch, dtype)
            else:
                total += blk.mixer.streaming_state_bytes(batch, dtype)
        return total

    # -- 生成 -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, eos_token_id: Optional[int] = None) -> torch.Tensor:
        was_training = self.training
        self.eval()
        B = idx.shape[0]
        states = self.init_stream(B, device=idx.device,
                                  dtype=self.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, states = self.stream_step(idx[:, t], states)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, states = self.stream_step(nxt, states)
        if was_training:
            self.train()
        return out

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int],
                top_p: Optional[float]) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0 < top_p < 1.0:
            srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
            cums = torch.cumsum(F.softmax(srt, dim=-1), dim=-1)
            remove = cums - F.softmax(srt, dim=-1) > top_p
            srt = srt.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, srt_idx, srt)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    @property
    def attn_layers(self) -> List[int]:
        return [i for i, u in enumerate(self.use_attn) if u]


# =============================================================================
# 参数解析（bit-exact，与 HybridModel.num_params 对照）
# =============================================================================
def analytic_hybrid_params(cfg: SGCConfig) -> int:
    """``HybridModel`` 的精确参数量（解析值，不实例化）。

    与 :func:`simplcore.sgc.config.analytic_params` 的差异仅在每层 mixer 项：
    按 ``_attn_layout`` 决定该层走 SGC 项（2·D² [+ D² 门] + M·D + 2M + D
    + D·K + 1）或注意力项。

    注意力项按配置精确区分（与 ``attention.py`` / ``hybrid.py`` /
    ``rope_scales.py`` 的实现一一对应，bit-exact）：
    * MHA（默认）：qkv 3·D² + out D² + norm D = 4·D² + D；
    * GQA（``num_kv_heads = G < H``）：q D² + k/v 各 D·(G·hd) + out D² + norm D，
      其中 ``hd = D // H``，即 ``2·D² + 2·D·G·(D//H) + D``（省 KV 不省投影
      的权重项只少 ``2·D·(H-G)·hd``）；
    * SGC 项在 ``gate_ab=True`` 时加 2·D²（gamma/beta 双门投影）。

    注意力层的滑窗 KV 是**运行时状态**而非参数，不计入。

    由 ``tests/test_hybrid.py::test_analytic_params_matches_real_model``
    做 bit-exact 守护。
    """
    cfg.validate()
    d, L, V, h = cfg.dim, cfg.num_layers, cfg.vocab_size, cfg.hidden
    layout = _attn_layout(L, getattr(cfg, "attn_every", None))
    kv_heads = getattr(cfg, "num_kv_heads", None) or cfg.num_heads
    head_dim = d // cfg.num_heads
    gate_ab = getattr(cfg, "gate_ab", False)
    use_yarn = getattr(cfg, "rope_scale_strategy", "none") in ("linear", "yarn")

    total = V * d
    for i, use_attn in enumerate(layout):
        if use_attn:
            # 注意力项精确口径：
            #   GQA+YaRN（use_yarn 且 G<H）：qkv = D + 2*G*hd → q D² + k/v 各 D*G*hd
            #   MHA+YaRN（use_yarn 且 G==H）：qkv = 3*D² → 4*D² + D
            #   GQA（非 yarn，G<H）：2*D² + 2*D*G*hd
            #   MHA（非 yarn，G==H）：4*D² + D
            # norm D 统一在下面 2*d 项里计。
            if use_yarn and kv_heads < cfg.num_heads:
                per = d * d + 2 * d * kv_heads * head_dim + d * d   # q + k/v + out
            elif use_yarn:
                per = 4 * d * d                                      # MHA + YaRN
            else:
                per = 2 * d * d + 2 * d * kv_heads * head_dim       # GQA / MHA
        else:
            per = 2 * d * d                    # in_proj + out_proj
            if cfg.content_gate:
                per += d * d
            if cfg.out_gate:
                per += d * d
            if gate_ab:
                per += 2 * d * d               # gate_gamma + gate_beta
            per += cfg.num_bases * d + cfg.num_bases * 2 + d
            per += d * cfg.local_kernel + 1
            if not gate_ab:
                # 四层闸门 L2/L3 标量。gate_ab 分支走 GatedSpectralConv2，
                # 该变体尚未实现 L2/L3（见 docs/ARCHITECTURE_PIVOT.md 的已知缺口）。
                per += sgc_gate_params(cfg)
        per += 3 * d * h + 2 * d               # SwiGLU + 两个 norm（mixer + ffn）
        total += per
    if not cfg.tie_embeddings:
        total += V * d
    total += d                                  # ln_f
    if getattr(cfg, "use_memory", False):
        total += d * d                          # CMT 记忆注入投影 (memory_in)
    return total


__all__ = [
    "SlidingWindowAttention",
    "HybridBlock",
    "HybridModel",
    "HybridLayerState",
    "analytic_hybrid_params",
    "_attn_layout",
]
