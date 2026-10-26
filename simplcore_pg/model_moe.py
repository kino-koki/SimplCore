# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCoreMoEModel：基于 MoE 的完整模型 (56B 中文大模型目标架构)。

**无注意力架构 (attention-free)**：本模型**完全放弃 Transformer/自注意力**。
每个 block = SimplCore 原生局部交互层 (LocalInteraction：因果深度卷积 + 可选线性递归)
+ ResonanceField (RFP，谱全局卷积) + DynamicMemoryChannels (DMC) + MoELayer。
  * 全局依赖由 RFP (谱全局卷积, O(n log n), 可训练) 提供；
  * 局部依赖由 LocalInteraction (O(n), 无 Q/K/V/RoPE/SDPA) 提供；
  * 训练时聚合各层负载均衡损失 + router z-loss，作为辅助项加入总损失；
  * 兼容 ``torch.compile`` 安全回退与 FX 追踪（纯张量算子、无数据相关控制流）。

详见 ``docs/PAPER_SPEC.md``。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from .config import SimplCoreConfig
from .layers import RMSNorm, init_weights
from .localmix import LocalInteraction
from .moe import MoELayer
from .resonance import ResonanceField
from .memory import DynamicMemoryChannels
from .checkpointing import CheckpointMixin
from .exceptions import ModelError


class SimplCoreMoEBlock(nn.Module):
    """MoE 计算块 (无注意力)：LocalInteraction → RFP → DMC → MoE。

    生产级模型默认启用 SimplCore 自研子层 (LocalInteraction/RFP/DMC)：
      * ``use_localmix`` 为真时插入原生局部交互层（替代自注意力）；
      * ``use_rfp`` / ``use_dmc`` 为真时，分别插入共振场处理与动态记忆通道；
      * 三者均受残差 + RMSNorm 包裹，且全程可微、无数据相关控制流，兼容
        ``torch.compile`` / FX 追踪；
      * MoELayer 内部已含 pre-norm 与残差。

    顺序：``x -> LocalMix -> RFP -> DMC -> MoE``。
    """

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        H = config.hidden_size
        self.use_localmix = getattr(config, "use_localmix", True)
        self.use_rfp = getattr(config, "use_rfp", True)
        self.use_dmc = getattr(config, "use_dmc", True)
        if self.use_localmix:
            self.localmix = LocalInteraction(
                H, kernel_size=config.local_kernel_size, dilation=config.local_dilation,
                use_conv=getattr(config, "local_use_conv", True),
                use_recurrence=config.local_use_recurrence, eps=config.rms_norm_eps,
                max_seq_len=config.max_sequence_length,
            )
        else:
            self.localmix = None
        # 共振场处理 (谱全局卷积, O(n log n), 可训练)
        self.rfp = (
            ResonanceField(
                H, num_waves=config.num_waves, num_steps=config.rfp_num_steps,
                damping=config.rfp_damping, dt=config.rfp_dt, sigma=config.rfp_sigma,
                eps=config.rms_norm_eps, max_seq_len=config.max_sequence_length,
                freeze_evolution=config.rfp_freeze_evolution,
                use_checkpoint=config.rfp_use_checkpoint,
            )
            if self.use_rfp else None
        )
        # 动态记忆通道 (内容寻址外部记忆, 可训练)
        self.dmc = (
            DynamicMemoryChannels(
                H, num_channels=config.num_channels, channel_size=config.channel_size,
                num_heads=config.dmc_num_heads, eps=config.rms_norm_eps,
            )
            if self.use_dmc else None
        )
        e_inter = config.moe_intermediate_size or int(round(H * config.ffn_intermediate_ratio))
        self.moe = MoELayer(
            hidden_size=H,
            num_experts=config.num_experts,
            experts_per_tok=config.experts_per_tok,
            intermediate_size=e_inter,
            aux_loss_coeff=config.moe_aux_loss_coeff,
            z_loss_coeff=config.moe_z_loss_coeff,
            capacity_factor=config.moe_capacity_factor,
            use_shared_expert=config.moe_use_shared_expert,
            shared_intermediate_size=config.shared_expert_intermediate_size,
            dropout=config.dropout,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
        seq_len: Optional[int] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | Tuple[torch.Tensor, Optional[torch.Tensor], tuple]:
        # SimplCore 原生局部交互层 (替代自注意力)
        if self.localmix is not None:
            x = self.localmix(x)
        # 自研子层：共振场处理 (谱全局卷积)
        if self.rfp is not None:
            x = self.rfp(x, seq_len=seq_len)
        # 自研子层：动态记忆通道
        if self.dmc is not None:
            x = self.dmc(x)
        moe_out, aux = self.moe(x, training=training)
        x = moe_out
        return x, aux


class SimplCoreMoEModel(CheckpointMixin, nn.Module):
    """SimplCore MoE 完整模型。"""

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        if config.num_experts < 2:
            raise ModelError(
                "SimplCoreMoEModel 需要 num_experts >= 2。", details={"num_experts": config.num_experts}
            )
        config.validate()
        self.config = config
        H = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, H)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([SimplCoreMoEBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(H, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

        self.apply(lambda m: init_weights(m, std=0.02))
        self._compiled = None  # type: ignore[var-annotated]
        self._last_aux: Optional[torch.Tensor] = None
        self.gradient_checkpointing: bool = False

    # ------------------------------------------------------------------
    def check_inputs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> None:
        if input_ids.dim() != 2:
            raise ModelError("input_ids 应为 2D (B, L)。", details={"shape": tuple(input_ids.shape)})
        B, L = input_ids.shape
        if L > self.config.max_sequence_length:
            raise ModelError(
                f"序列长度 {L} 超过 max_sequence_length {self.config.max_sequence_length}。",
                details={"L": L, "max": self.config.max_sequence_length},
            )
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ModelError("input_ids 含越界 token id。", details={"vocab_size": self.config.vocab_size})
        if attention_mask is not None and attention_mask.shape != (B, L):
            raise ModelError("attention_mask 形状应为 (B, L)。", details={"got": tuple(attention_mask.shape)})

    # ------------------------------------------------------------------
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        x = self.emb_dropout(self.token_embedding(input_ids)) * math.sqrt(self.config.hidden_size)
        seq_len = input_ids.shape[1]
        total_aux: Optional[torch.Tensor] = None
        for block in self.blocks:
            if self.gradient_checkpointing and training:
                x, aux = _grad_checkpoint(
                    block, x, attention_mask, True, seq_len, use_reentrant=False
                )
            else:
                x, aux = block(x, attention_mask=attention_mask, training=training, seq_len=seq_len)
            if training and aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        self._last_aux = total_aux
        h = self.final_norm(x)
        return self.lm_head(h)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        self.check_inputs(input_ids, attention_mask)
        if self._compiled is not None:
            return self._compiled(input_ids, attention_mask, training)
        return self._forward_impl(input_ids, attention_mask, training)

    # ------------------------------------------------------------------
    @property
    def last_aux_loss(self) -> Optional[torch.Tensor]:
        """最近一次前向 (training=True) 聚合的辅助损失。"""
        return self._last_aux

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ModelError("temperature 必须为正。")
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, float("-inf")), logits)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 0] = False
                sorted_logits = torch.where(remove, float("-inf"), sorted_logits)
                logits = torch.gather(sorted_logits, -1, sorted_idx.argsort(-1))
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_ids], dim=-1)
            if (pad is not None and (next_ids == pad).all()) or (eos is not None and (next_ids == eos).all()):
                break
        return generated

    # ------------------------------------------------------------------
    def configure_optimizers(self, lr: float = 3e-4, weight_decay: float = 0.1) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name or "embedding" in name or "router" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))

    # ------------------------------------------------------------------
    def num_parameters(self, include_non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_non_embedding and self.config.tie_word_embeddings:
            total -= self.token_embedding.weight.numel()
        return total

    # ------------------------------------------------------------------
    def trace_fx(self, example_input_ids: Optional[torch.Tensor] = None) -> torch.fx.GraphModule:
        """FX 追踪 (临时固定容量以避免动态 one_hot)。"""
        if example_input_ids is None:
            example_input_ids = torch.zeros(1, 8, dtype=torch.long)
        B, L = example_input_ids.shape
        cap = max(
            1,
            int(math.ceil(
                self.config.moe_capacity_factor * self.config.experts_per_tok * B * L
                / self.config.num_experts
            )),
        )
        moe_layers = [m for m in self.modules() if isinstance(m, MoELayer)]
        saved = [(m, m.fixed_capacity) for m in moe_layers]
        for m in moe_layers:
            m.fixed_capacity = cap
        try:
            class _Wrap(nn.Module):
                def __init__(self, mm: "SimplCoreMoEModel"):
                    super().__init__()
                    self.m = mm
                def forward(self, input_ids):
                    return self.m._forward_impl(input_ids, None, False)
            if hasattr(torch, "export"):
                try:
                    ep = torch.export.export(_Wrap(self), (example_input_ids,), {})
                    return ep.module()
                except Exception:
                    pass
            return torch.fx.symbolic_trace(_Wrap(self), concrete_args=(example_input_ids,))
        finally:
            for m, c in saved:
                m.fixed_capacity = c

    def extra_repr(self) -> str:
        return (
            f"vocab={self.config.vocab_size}, H={self.config.hidden_size}, "
            f"layers={self.config.num_layers}, experts={self.config.num_experts}"
        )
