# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC model: blocks, backbone, causal LM, streaming generation."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SGCConfig
from .mixer import (SGCStreamState, SpectralGlobalConv, make_norm)
from ._rope import _rope_tables, _apply_rope
from .abstain import AbstentionHead


# ----------------------------------------------------------------------------
# Feed-forward
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """Pre-norm SwiGLU feed-forward with residual connection."""

    def __init__(self, dim: int, hidden: int, norm: str = "rmsnorm",
                 dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = make_norm(norm, dim, eps=eps)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))


# ----------------------------------------------------------------------------
# Attention baseline (identical skeleton, for honest A/B comparison only)
# ----------------------------------------------------------------------------
def _rope_tables(head_dim: int, length: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(length, device=device).float()
    freqs = torch.outer(t, inv)                       # (L, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, hd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.view(1, 1, x.shape[2], -1)
    s = sin.view(1, 1, x.shape[2], -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    """Standard causal multi-head self-attention + RoPE (fair baseline)."""

    def __init__(self, dim: int, num_heads: int, norm: str = "rmsnorm",
                 dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.norm = make_norm(norm, dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, kv_cache=None):
        B, L, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        cos, sin = _rope_tables(self.head_dim, L, x.device, q.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, L, D)
        return x + self.drop(self.out_proj(y))


# ----------------------------------------------------------------------------
# Block
# ----------------------------------------------------------------------------
class SGCBlock(nn.Module):
    """Pre-norm block: mixer -> FFN, each with its own residual."""

    def __init__(self, config: SGCConfig) -> None:
        super().__init__()
        self.config = config
        kind = config.mixer
        if kind == "attn":
            self.mixer = CausalAttention(config.dim, config.num_heads, norm=config.norm,
                                         dropout=config.dropout, eps=config.norm_eps)
        else:
            self.mixer = SpectralGlobalConv(
                dim=config.dim,
                num_bases=config.num_bases,
                decay_max=config.decay_max,
                local_kernel=1 if kind == "sgc-nolocal" else config.local_kernel,
                content_gate=config.content_gate and kind != "sgc-nogate",
                out_gate=config.out_gate,
                evidence_gate=config.evidence_gate,
                uncertainty_gate=config.uncertainty_gate,
                norm=config.norm,
                fft_dtype=config.resolve_fft_dtype(),
                eps=config.norm_eps,
            )
        self.ffn = SwiGLU(config.dim, config.hidden, norm=config.norm,
                          dropout=config.dropout, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        if memory is not None:
            # CMT 式记忆注入：把检索到的压缩记忆作为**残差流上的加性项**。
            # SGC 没有注意力、没有 KV 缓存，残差流是它原生的注入点。
            x = x + memory.unsqueeze(1)
        if self.config.grad_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.mixer, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False)
            return x
        return self.ffn(self.mixer(x))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _make_block_for_config(config: SGCConfig, layer_idx: int) -> "nn.Module":
    """Build the block for layer ``layer_idx`` given ``config``.

    ``mixer="hybrid"`` routes to a :class:`HybridBlock` (SGC or sliding-window
    attention by layout); all other mixer kinds use :class:`SGCBlock`.
    ``hybrid`` is imported lazily to avoid a circular import (``hybrid`` needs
    ``SwiGLU`` / ``_rope`` helpers from this package).
    """
    if config.mixer in ("hybrid", "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn"):
        from . import hybrid as _hybrid   # lazy: hybrid imports helpers from here
        return _hybrid.HybridBlock(config, layer_idx,
                                   _hybrid._attn_layout(config.num_layers,
                                                        config.attn_every)[layer_idx])
    return SGCBlock(config)


class SGCModel(nn.Module):
    """SGC causal language model.

    There is deliberately **no positional-embedding table**: the causal
    convolution already encodes position, which is what makes length
    extrapolation and O(1)-state streaming well defined.
    """

    def __init__(self, config: SGCConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            _make_block_for_config(config, i) for i in range(config.num_layers)
        ])
        self.ln_f = make_norm(config.norm, config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        # L4 拒答门: 可选的 grounded 预测头（默认关闭；需外部证据标签才能训练）
        self.abstain_head = AbstentionHead(config.dim) if config.use_abstain else None
        self.apply(self._init_weights)
        # CMT 记忆注入投影：零初始化 → 初始为恒等（不改变基座行为），训练后自学
        self.memory_in = nn.Linear(config.dim, config.dim, bias=False) if config.use_memory else None
        if self.memory_in is not None:
            with torch.no_grad():
                self.memory_in.weight.zero_()
        # scaled residual-output init (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.num_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)

    # -- forward -------------------------------------------------------------
    def forward(self, idx_or_x, targets: Optional[torch.Tensor] = None,
                return_hidden: bool = False,
                text: Optional[str] = None, codec=None,
                return_abstain: bool = False,
                memory: Optional[torch.Tensor] = None):
        """统一前向入口：接受 token ID 张量或文本字符串（无词表模式）.

        * ``idx_or_x`` 为张量 → 传统路径：token ID → 嵌入 → 前向；
        * ``idx_or_x`` 为字符串（且提供 ``codec``）→ 无词表路径：
          ``codec.encode(text)`` → 张量 → 嵌入 → 前向 → 返回 logit；
          若 ``targets`` 也是字符串，同 codec 编码后计算 loss。

        这使"舍去传统 Transformer 词表"成为一等公民：同一套模型权重
        既可按 HF 词表 ID 流喂入，也可按字符/字节/BPE 文本直喂，
        不重新初始化词表参数。
        """
        if isinstance(idx_or_x, str):
            if codec is None:
                raise ValueError("text input requires a codec")
            idx_or_x = torch.tensor([codec.encode(idx_or_x)],
                                     device=next(self.parameters()).device)
        if isinstance(targets, str):
            if codec is None:
                raise ValueError("text targets require a codec")
            targets = torch.tensor([codec.encode(targets)],
                                   device=next(self.parameters()).device)
        B, L = idx_or_x.shape
        x = self.drop(self.tok_emb(idx_or_x))
        mem = self._project_memory(memory)
        for blk in self.blocks:
            x = blk(x, mem)
        x = self.ln_f(x)
        logits = self.head(x)
        if return_abstain:
            if self.abstain_head is None:
                raise ValueError(
                    "return_abstain=True requires SGCConfig(use_abstain=True)"
                )
            abstain_logit = self.abstain_head(x)
            if return_hidden:
                return logits, x, abstain_logit
            return logits, abstain_logit
        if return_hidden:
            return logits, x
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return logits, loss
        return logits

    # -- streaming -----------------------------------------------------------
    def _project_memory(self, memory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """把检索到的压缩记忆投影到模型宽度。

        没有启用 ``use_memory`` 或未提供记忆时返回 None（基座行为完全不变）。
        """
        if memory is None or self.memory_in is None:
            return None
        return self.memory_in(memory)

    def init_stream(self, batch: int, device=None, dtype=None,
                    memory: Optional[torch.Tensor] = None) -> SGCStreamState:
        device = device or self.tok_emb.weight.device
        dtype = dtype or self.tok_emb.weight.dtype
        states = []
        for blk in self.blocks:
            mx = blk.mixer
            if hasattr(mx, "init_state"):
                states.append(mx.init_state(batch, device, dtype))
            elif hasattr(mx, "init_stream"):      # hybrid sliding-window attention
                states.append(mx.init_stream(batch, device, dtype))
            else:
                states.append(None)          # attention baseline: no O(1) state
        return SGCStreamState(layers=states, length=0,
                              memory=self._project_memory(memory))

    def stream_step(self, token: torch.Tensor, state: SGCStreamState) -> Tuple[torch.Tensor, SGCStreamState]:
        """Consume one token per batch element; returns logits for the *next* token.

        Supports both SGC mixers (O(1) ``init_state``/``step``) and the hybrid
        sliding-window attention layers (``init_stream``/``stream_step``).
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        x = self.tok_emb(token)                                    # (B,1,D)
        mem = state.memory
        for blk, st in zip(self.blocks, state.layers):
            mx = blk.mixer
            if st is None:
                raise RuntimeError("streaming is not supported by the attention baseline")
            if mem is not None:
                x = x + mem.unsqueeze(1)
            if hasattr(mx, "step"):
                x, _ = mx.step(x, st)
            else:
                x, st = mx.stream_step(x, st)
            x = blk.ffn(x)
        x = self.ln_f(x)
        state.length += 1
        return self.head(x[:, -1, :]), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        total = 0
        for blk in self.blocks:
            mx = blk.mixer
            if hasattr(mx, "streaming_state_bytes"):
                total += mx.streaming_state_bytes(batch, dtype)
            elif hasattr(mx, "stream_bytes"):
                total += mx.stream_bytes(batch, dtype)
        return total

    # -- streaming checkpoint/restore ----------------------------------------
    def checkpoint_stream(self, state: "SGCStreamState") -> dict:
        """把 O(1) 流式状态序列化成纯 dict（可 JSON/safetensors 持久化）.

        返回 ``{"length": int, "layers": [per-layer state dict or None]}``。
        每层的状态要么是 dict（滑窗注意力环形 KV），要么是
        :class:`LayerStreamState`（SGC IIR 二阶递推）→ 转为 plain dict。
        调用方可将结果 ``torch.save`` / 写 safetensors 以保存"断点流"，
        恢复时用 :meth:`restore_stream` 重建。
        """
        layers: list = []
        for st in state.layers:
            if st is None:
                layers.append(None)
            elif isinstance(st, dict):
                layers.append(st)
            else:
                # LayerStreamState（SGC mixer）→ 字段导出
                layers.append({
                    "y1": st.y1, "y2": st.y2,
                    "conv_buf": st.conv_buf, "length": st.length,
                })
        return {"length": state.length, "layers": layers}

    def restore_stream(self, ckpt: dict) -> "SGCStreamState":
        """从 :meth:`checkpoint_stream` 的 dict 重建 SGCStreamState.

        兼容 LayerStreamState 字段（自动还原为 LayerStreamState 实例）
        和 dict 状态（滑窗注意力）原样透传。
        """
        from .mixer import SGCStreamState, LayerStreamState
        layers = []
        for item in ckpt.get("layers", []):
            if item is None:
                layers.append(None)
            elif isinstance(item, dict) and "y1" in item:
                layers.append(LayerStreamState(
                    y1=item["y1"], y2=item["y2"],
                    conv_buf=item.get("conv_buf"),
                    length=item.get("length", 0)))
            else:
                layers.append(item)
        return SGCStreamState(layers=layers, length=ckpt.get("length", 0))

    # -- generation ----------------------------------------------------------
    @torch.no_grad()
    def generate_text(self, text: str, codec, max_new_tokens: int = 32,
                      temperature: float = 1.0, top_k: Optional[int] = None,
                      top_p: Optional[float] = None,
                      eos_token_id: Optional[int] = None) -> str:
        """无词表模式文本生成：text → codec.encode → 生成 → codec.decode → text.

        这是"舍去传统 Transformer 词表"的核心推理入口：同一套模型权重可切换
        任意编解码器（字符/字节/BPE），无需重新初始化词表参数，只需
        在推理时传入对应的 codec。

        生成在 ID 空间完成（O(1) 状态流式路径），最后统一 decode 回文本。
        若 codec 无 eos_token_id 概念（默认 CharTokenizer），传 ``eos_token_id``
        以提前停止；否则生成满 ``max_new_tokens`` 个 token。
        """
        from .tokenization import Tokenizer
        assert isinstance(codec, Tokenizer)
        idx = torch.tensor([codec.encode(text)], device=next(self.parameters()).device)
        out = self.generate(idx, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k,
                            top_p=top_p, eos_token_id=eos_token_id)
        gen_ids = out[0, idx.shape[1]:].tolist()
        return codec.decode(gen_ids)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, use_state: bool = True,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling.

        ``use_state=True`` uses the O(1) recurrent path: per-token cost and memory
        do not grow with the context length.
        """
        was_training = self.training
        self.eval()
        B = idx.shape[0]
        if not use_state:
            out = idx
            for _ in range(max_new_tokens):
                logits = self(out)[:, -1, :]
                nxt = self._sample(logits, temperature, top_k, top_p)
                out = torch.cat([out, nxt], dim=1)
                if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                    break
            if was_training:
                self.train()
            return out

        state = self.init_stream(B, device=idx.device, dtype=self.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
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
            probs = F.softmax(srt, dim=-1).cumsum(-1)
            remove = probs - F.softmax(srt, dim=-1) > top_p
            srt = srt.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, srt_idx, srt)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # -- introspection -------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def layer_summary(self) -> List[str]:
        return [repr(blk.mixer) for blk in self.blocks]


# Convenience alias: this model already carries the LM head.
SGCForCausalLM = SGCModel


if __name__ == "__main__":
    import sys

    from .config import get_preset, report_presets

    if "--report-presets" in sys.argv:
        raise SystemExit(report_presets())
    else:
        cfg = get_preset("toy")
        m = SGCModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        logits, loss = m(x, x)
        loss.backward()
        print("toy smoke OK", tuple(logits.shape), float(loss))
