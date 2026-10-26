# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""标准 Transformer baseline，用于与 SimplCoreMix 做公平消融对比。

设计原则:
- 与 SimplCoreMix 使用相同的配置字段 (vocab_size, dim, n_heads, n_kv_heads,
  max_seq_len)，保证参数量量级可比。
- 使用 GQA + RoPE + RMSNorm + SwiGLU FFN，这是当前主流 baseline 配置。
- 不包含 MBA / CCR-MoE / DMC / SPM / LTI / ACT / recurrent block，
  这样消融实验里 "baseline vs full Mix" 的差异完全归因于 SimplCore 创新模块。
- 保持与 SimplCoreMix 相同的 forward/generate 接口，方便统一评测。
"""
from __future__ import annotations

import math
from typing import Optional

import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F

from .config import SimplCoreMixConfig
from .core import (
    SCDropout,
    SCEmbedding,
    SCLinear,
    SCModule,
    SCModuleList,
    SCRMSNorm,
)
from .utils import precompute_rope_freqs, apply_rope


class GQAAttention(SCModule):
    """标准 Grouped-Query Attention (baseline 用)。

    不含 MBA 基向量场，仅做标准 Q/K/V 投影 + RoPE + causal mask。
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = SCLinear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = SCLinear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.attn_drop = SCDropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        # GQA: 扩展 kv_heads 到 q_heads
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class BaselineTransformerLayer(SCModule):
    """标准 pre-norm Transformer 层: GQA + SwiGLU FFN。"""

    def __init__(self, cfg: SimplCoreMixConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm1 = SCRMSNorm(cfg.dim)
        self.attn = GQAAttention(cfg)

        # SwiGLU FFN, 与 SimplCoreMix Expert 结构一致
        hidden_dim = cfg.dim * 4 // 3
        self.norm2 = SCRMSNorm(cfg.dim)
        self.gate = SCLinear(cfg.dim, hidden_dim, bias=False)
        self.up = SCLinear(cfg.dim, hidden_dim, bias=False)
        self.down = SCLinear(hidden_dim, cfg.dim, bias=False)
        self.dropout = SCDropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, freqs_cis, mask, kv_cache, f"baseline_{self.layer_idx}")
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = F.silu(self.gate(h)) * self.up(h)
        h = self.down(h)
        x = x + self.dropout(h)
        return x


class BaselineTransformer(SCModule):
    """标准 decoder-only Transformer (baseline)。

    层数取 prelude_layers + max_loop_iters + coda_layers，
    使其总深度与 SimplCoreMix 循环展开后的等效深度一致，
    从而参数量量级可比、计算量量级可比。
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = SCEmbedding(cfg.vocab_size, cfg.dim)

        # 等效深度 = prelude + max_loop_iters + coda
        n_layers = cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers
        self.layers = SCModuleList([
            BaselineTransformerLayer(cfg, i) for i in range(n_layers)
        ])

        self.norm_final = SCRMSNorm(cfg.dim)
        self.output = SCLinear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, SCLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, SCEmbedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        tokens: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = tokens.shape[1]
        device = tokens.device

        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + T].to(device)
        mask = self._causal_mask(T, device) if T > 1 else None

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, kv_cache)

        x = self.norm_final(x)
        return self.output(x)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 100,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        self.eval()
        prompt_len = tokens.shape[1]
        kv_cache = {}
        generated = tokens.clone()

        for step in range(max_new_tokens):
            if step == 0:
                cur_tokens = generated
                cur_start_pos = 0
            else:
                cur_tokens = generated[:, -1:]
                cur_start_pos = prompt_len + step - 1

            logits = self.forward(
                cur_tokens, kv_cache=kv_cache, start_pos=cur_start_pos
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "embedding": sum(p.numel() for p in self.tok_embeddings.parameters()),
            "layers": sum(p.numel() for p in self.layers.parameters()),
            "output": sum(p.numel() for p in self.output.parameters()),
        }

    def get_architecture_info(self) -> dict:
        n_layers = self.cfg.prelude_layers + self.cfg.max_loop_iters + self.cfg.coda_layers
        return {
            "architecture": "BaselineTransformer",
            "backend": self.sc_backend_name(),
            "self_owned_surface": True,
            "skeleton": "Standard decoder-only Transformer",
            "components": {
                "attention": "GQA + RoPE",
                "ffn": "SwiGLU",
                "norm": "RMSNorm",
                "depth": f"{n_layers} layers (prelude+loop+coda equivalent)",
            },
            "param_count": self.get_param_count()["total"],
        }


__all__ = [
    "GQAAttention",
    "BaselineTransformerLayer",
    "BaselineTransformer",
]
