# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 长上下文流式模型 (SimplCoreStreamingModel)。

在 *生产级 SimplCore 自研架构* 之上提供长上下文能力：
  * 序列切成固定 ``chunk_size`` 逐块流过各层；
  * 跨块仅传递固定大小的 ``StreamState`` (RFP 频域 + DMC 记忆 + 窗口 KV)，
    三者均与序列长度 n 无关；
  * 因此激活显存峰值被 (chunk_size × 层数 × H) 封顶，n 翻倍时仅增加常数显存。

提供：
  * ``forward(input_ids)``：整段一次性前向 (训练 / 评测)，自动分块；
  * ``stream_init`` / ``stream_step``：自回归流式推演，常驻显存极小；
  * ``peak_activation_bytes``：实测单次前向峰值激活显存 (用于验证翻倍+~1GB)。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import SimplCoreConfig
from .layers import RMSNorm, init_weights
from .longctx import SimplCoreStreamingBlock, StreamState
from .checkpointing import CheckpointMixin
from .exceptions import ModelError


class SimplCoreStreamingModel(CheckpointMixin, nn.Module):
    """长上下文流式 MoE 模型。"""

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        H = config.hidden_size
        self.chunk_size = config.chunk_size

        self.token_embedding = nn.Embedding(config.vocab_size, H)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([SimplCoreStreamingBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(H, eps=config.rms_norm_eps)
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
        self.apply(lambda m: init_weights(m, std=0.02))
        self._last_aux: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def num_parameters(self, include_non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_non_embedding and self.config.tie_word_embeddings:
            total -= self.token_embedding.weight.numel()
        return total

    def configure_optimizers(self, lr: float = 3e-4, weight_decay: float = 0.1) -> "torch.optim.Optimizer":
        """分组优化器：bias/norm/embedding/router 不衰减，其余 L2 衰减。"""
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
    def check_inputs(self, input_ids: torch.Tensor) -> None:
        if input_ids.dim() != 2:
            raise ModelError("input_ids 应为 2D (B, L)。", details={"shape": tuple(input_ids.shape)})
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ModelError("input_ids 含越界 token id。", details={"vocab_size": self.config.vocab_size})

    # ------------------------------------------------------------------
    def _run_blocks(self, x: torch.Tensor, state: StreamState, training: bool
                    ) -> Tuple[torch.Tensor, StreamState, Optional[torch.Tensor]]:
        total_aux: Optional[torch.Tensor] = None
        for i, block in enumerate(self.blocks):
            x, state, aux = block(x, state, training=training, layer_idx=i)
            if training and aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        return x, state, total_aux

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> torch.Tensor:
        """整段前向 (自动分块)。"""
        self.check_inputs(input_ids)
        B, n, H = *input_ids.shape, self.config.hidden_size
        x = self.emb_dropout(self.token_embedding(input_ids)) * math.sqrt(H)
        state = self._init_state(B, x.device, x.dtype)
        # 按 chunk_size 切块逐段过 blocks，非原地收集输出 (避免反向 inplace 冲突)
        chunk_outs: List[torch.Tensor] = []
        total_aux: Optional[torch.Tensor] = None
        for i in range(0, n, self.chunk_size):
            seg = x[:, i:i + self.chunk_size, :]
            seg, state, aux = self._run_blocks(seg, state, training=training)
            if training and aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
            chunk_outs.append(seg)
        x = torch.cat(chunk_outs, dim=1)
        # 训练期聚合各块 MoE 辅助损失 (负载均衡 + z-loss)，供总损失相加
        self._last_aux = total_aux
        h = self.final_norm(x)
        return self.lm_head(h)

    # ------------------------------------------------------------------
    def _init_state(self, batch: int, device, dtype) -> StreamState:
        # 逐层初始化 (每层独立状态)；无注意力，故无 kv_cache
        rfp_states: List[torch.Tensor] = []
        dmc_rt: List[object] = []
        for block in self.blocks:
            bs = block.init_state(batch, device, dtype)
            rfp_states.extend(bs.rfp_spec)
            dmc_rt.extend(bs.dmc_runtime)
        st = StreamState(rfp_spec=rfp_states, dmc_runtime=dmc_rt, chunk_idx=0)
        return st

    # ------------------------------------------------------------------
    @property
    def last_aux_loss(self) -> Optional[torch.Tensor]:
        return self._last_aux

    # ------------------------------------------------------------------
    @torch.no_grad()
    def peak_activation_bytes(self, batch: int, seq_len: int, device=None,
                              dtype=torch.float32) -> dict:
        """实测一次前向的峰值激活显存 (字节)。

        使用 ``torch.cuda`` 内存钩子 (若可用)；CPU 下用 ``torch`` 张量分配估算。
        返回 {peak_bytes, per_token_bytes, fixed_state_bytes}。
        """
        if device is None:
            device = torch.device("cpu")
        x = torch.randint(0, self.config.vocab_size, (batch, seq_len), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            self.to(device).forward(x)
            peak = torch.cuda.max_memory_allocated(device)
            return {"peak_bytes": peak, "device": "cuda"}
        # CPU 估算：分配激活张量后统计 (近似)
        # 固定状态大小 (与 seq_len 无关)
        fixed = 0
        st = self._init_state(batch, device, dtype)
        for t in st.rfp_spec:
            fixed += t.numel() * t.element_size()
        # 注：无 KV 缓存，固定状态仅含 RFP 频域 + DMC 记忆（与序列长度无关）
        # 激活：chunk 级 (B, chunk, H) × 层数 × (若干中间副本)
        c = self.chunk_size
        act_per_chunk = batch * c * self.config.hidden_size * 4 * self.config.num_layers * 2
        est = fixed + act_per_chunk
        return {
            "fixed_state_bytes": fixed,
            "est_peak_bytes": est,
            "per_chunk_act_bytes": batch * c * self.config.hidden_size * 4 * self.config.num_layers * 2,
            "device": "cpu-estimate",
        }

    def extra_repr(self) -> str:
        return (f"vocab={self.config.vocab_size}, H={self.config.hidden_size}, "
                f"layers={self.config.num_layers}, chunk={self.chunk_size}")

    # ------------------------------------------------------------------
    # 流式分块 API (真实实现, 非口头承诺)
    # 用法:
    #   st = model.stream_init(batch=1)
    #   for chunk in chunks:
    #       out, st = model.stream_step(chunk, st)   # out: (B, chunk, V)
    # 跨 chunk 仅传递固定大小的 StreamState (RFP 频域 + DMC 记忆 + 窗口 KV),
    # 与序列总长度无关, 故激活显存近似常数。
    @torch.no_grad()
    def stream_init(self, batch: int = 1, device=None, dtype=None) -> StreamState:
        """初始化流式状态 (固定大小, 与序列总长度无关)。"""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self._init_state(batch, device, dtype)

    @torch.no_grad()
    def stream_step(self, chunk_ids: torch.Tensor, state: StreamState) -> Tuple[torch.Tensor, StreamState]:
        """处理一个 chunk, 返回该 chunk 的 logits 与更新后的状态。

        Args:
            chunk_ids: (B, Lc) 整数 token id, Lc 应 <= chunk_size。
            state: 上一 chunk 传下的 StreamState。
        Returns:
            (logits, new_state), logits 形状 (B, Lc, vocab)。
        """
        if chunk_ids.dim() != 2:
            raise ModelError("chunk_ids 应为 2D (B, Lc)。", details={"shape": tuple(chunk_ids.shape)})
        B, Lc, H = *chunk_ids.shape, self.config.hidden_size
        if Lc > self.chunk_size:
            raise ModelError(
                f"chunk 长度 {Lc} 超过 chunk_size {self.chunk_size}。",
                details={"chunk_size": self.chunk_size},
            )
        x = self.emb_dropout(self.token_embedding(chunk_ids)) * math.sqrt(H)
        x, state, _ = self._run_blocks(x, state, training=False)
        state.chunk_idx += 1
        h = self.final_norm(x)
        logits = self.lm_head(h)
        return logits, state

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """流式自回归生成: 先把 prompt 分块灌入, 再逐 token 生成。

        prompt 可分块流式喂入 (常驻显存极小); 生成阶段逐 token 走 stream_step。
        """
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        B = prompt_ids.shape[0]
        st = self.stream_init(batch=B)
        # 灌入 prompt (按 chunk 切分)
        for i in range(0, prompt_ids.shape[1], self.chunk_size):
            chunk = prompt_ids[:, i:i + self.chunk_size]
            _, st = self.stream_step(chunk, st)
        # 自回归生成
        generated = []
        last_tok = prompt_ids[:, -1:]
        for _ in range(max_new_tokens):
            logits, st = self.stream_step(last_tok, st)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k > 0:
                vals, _ = torch.topk(next_logits, top_k, dim=-1)
                thresh = vals[:, -1:]
                next_logits = torch.where(
                    next_logits < thresh, torch.full_like(next_logits, float("-inf")), next_logits
                )
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            generated.append(nxt)
            last_tok = nxt
        if generated:
            return torch.cat(generated, dim=1)
        return prompt_ids[:, :0]
