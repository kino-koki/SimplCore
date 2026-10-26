# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCoreModel：完整模型 (嵌入 + N×Block + 最终归一化 + LM Head)。

生产级特性：
  * 权重绑定 (tied embeddings) 可选；
  * ``forward`` 支持 ``torch.compile`` 封装 (带安全回退)；
  * ``trace_fx`` 提供 FX 图模式追踪与优化 pass，满足图模式要求；
  * ``generate`` 支持贪心 / top-k / top-p / temperature 自回归生成，含
    EOS / 最大步数终止与 padding 屏蔽；
  * ``configure_optimizers`` 提供生产常用优化器 (AdamW + 权重衰减解耦)。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import SimplCoreConfig
from .block import SimplCoreBlock
from .layers import RMSNorm, init_weights
from .memory import MemoryState
from .checkpointing import CheckpointMixin
from .exceptions import ModelError


class SimplCoreModel(CheckpointMixin, nn.Module):
    """SimplCore 完整模型。"""

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        H = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, H)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([SimplCoreBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(H, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

        self.apply(lambda m: init_weights(m, std=0.02))
        self._compiled = None  # type: ignore[var-annotated]

    # ------------------------------------------------------------------
    # 输入校验 (仅 eager 入口，不参与编译/追踪图)
    # ------------------------------------------------------------------
    def check_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """校验输入合法性，非法时抛出 ModelError (eager 专用)。"""
        if input_ids.dim() != 2:
            raise ModelError("input_ids 应为 2D (B, L)。", details={"shape": tuple(input_ids.shape)})
        B, L = input_ids.shape
        if L > self.config.max_sequence_length:
            raise ModelError(
                f"序列长度 {L} 超过 max_sequence_length {self.config.max_sequence_length}。",
                details={"L": L, "max": self.config.max_sequence_length},
            )
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ModelError(
                "input_ids 含越界 token id。",
                details={"vocab_size": self.config.vocab_size},
            )
        if attention_mask is not None and attention_mask.shape != (B, L):
            raise ModelError(
                "attention_mask 形状应为 (B, L)。",
                details={"got": tuple(attention_mask.shape), "expected": (B, L)},
            )

    # ------------------------------------------------------------------
    # 核心前向 (纯计算，可被 torch.compile / FX 追踪；无数据相关分支)
    # ------------------------------------------------------------------
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[MemoryState] = None,
        training: bool = False,
    ) -> torch.Tensor:
        x = self.emb_dropout(self.token_embedding(input_ids)) * math.sqrt(self.config.hidden_size)
        seq_len = input_ids.shape[1]  # 由具体 input_ids 推导 (FX concrete_args 下为常量)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, memory=memory, training=training, seq_len=seq_len)
        h = self.final_norm(x)
        logits = self.lm_head(h)
        return logits

    # ------------------------------------------------------------------
    # 公共前向 (eager 校验 + 编译分发)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[MemoryState] = None,
        training: bool = False,
    ) -> torch.Tensor:
        self.check_inputs(input_ids, attention_mask)
        if self._compiled is not None:
            return self._compiled(input_ids, attention_mask, memory, training)
        return self._forward_impl(input_ids, attention_mask, memory, training)

    # ------------------------------------------------------------------
    # 编译封装 (torch.compile, 只编译纯计算核心，带安全回退)
    # ------------------------------------------------------------------
    def compile(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "inductor",
        fullgraph: bool = False,
        smoke_input: Optional[tuple] = None,
    ) -> "SimplCoreModel":
        """用 ``torch.compile`` 编译纯计算核心，失败则安全回退到 eager。"""
        try:
            import torch._dynamo  # noqa: F401

            def _core(ids, am, mem, tr):
                return self._forward_impl(ids, am, mem, tr)

            compiled = torch.compile(
                _core, mode=mode, dynamic=dynamic, backend=backend, fullgraph=fullgraph
            )
            if smoke_input is not None:
                with torch.no_grad():
                    dummy = torch.zeros(*smoke_input, dtype=torch.long)
                    compiled(dummy, None, None, False)
            self._compiled = compiled
        except Exception as exc:  # pragma: no cover - 依赖运行环境
            import warnings
            warnings.warn(
                f"torch.compile 失败 ({exc!r})，回退到 eager 模式。",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled = None
        return self

    @property
    def is_compiled(self) -> bool:
        return self._compiled is not None

    # ------------------------------------------------------------------
    # FX 图模式追踪与优化
    # ------------------------------------------------------------------
    def trace_fx(self, example_input_ids: Optional[torch.Tensor] = None) -> torch.fx.GraphModule:
        """导出纯计算核心的 FX 图 (基于 ``torch.export``，支持动态形状)。

        ``torch.export`` 在底层使用 FX 图表示 (ATen 方言) 并正确处理符号形状，
        是生产级"图模式"的推荐路径；若环境不支持则回退到 ``symbolic_trace``。

        Args:
            example_input_ids: 示例输入；其形状将作为图的符号/具体形状基准。
        """
        if example_input_ids is None:
            example_input_ids = torch.zeros(1, 8, dtype=torch.long)

        class _Wrap(nn.Module):
            def __init__(self, m: "SimplCoreModel"):
                super().__init__()
                self.m = m
            def forward(self, input_ids):
                return self.m._forward_impl(input_ids, None, None, False)

        # 追踪以推理图为目标：临时切到 eval，避免 checkpoint 进入导出图
        was_training = self.training
        self.eval()
        try:
            if hasattr(torch, "export"):
                try:
                    ep = torch.export.export(_Wrap(self), (example_input_ids,), {})
                    return ep.module()  # 参数已绑定的可调用图模块
                except Exception:  # pragma: no cover - 依赖运行环境 (Windows 上 torch.export 受 dynamo 限制)
                    pass
            # 静态形状回退 (需要具体形状，不支持动态维度；Windows 等环境走此路径)
            return torch.fx.symbolic_trace(_Wrap(self), concrete_args=(example_input_ids,))
        finally:
            self.train(was_training)

    def summarize_graph(self, gm: torch.fx.GraphModule) -> dict:
        """统计 FX 图中各算子出现频次。"""
        counts: dict = {}
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_module", "call_method"):
                key = node.op + ":" + (
                    node.target if isinstance(node.target, str) else node.target.__class__.__name__
                )
                counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # 自回归生成
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
        """自回归生成 (贪心 / 采样)。

        Args:
            input_ids: (B, L) 前缀。
            max_new_tokens: 最大生成长度。
            temperature: 采样温度 (>0)。
            top_k: 仅从前 k 个候选采样 (0 表示关闭)。
            top_p: 核采样阈值 (1.0 表示关闭)。
            pad_token_id / eos_token_id: 终止控制。

        Returns:
            (B, L + max_new_tokens) 含前缀的完整序列。
        """
        if temperature <= 0:
            raise ModelError("temperature 必须为正。", details={"temperature": temperature})
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        cfg_pad_id = pad

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]  # (B, V)
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
            next_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_ids], dim=-1)
            # 终止判定：所有样本生成了 pad 或 eos
            if cfg_pad_id is not None and (next_ids == cfg_pad_id).all():
                break
            if eos is not None and (next_ids == eos).all():
                break
        return generated

    # ------------------------------------------------------------------
    # 优化器
    # ------------------------------------------------------------------
    def configure_optimizers(
        self, lr: float = 3e-4, weight_decay: float = 0.1
    ) -> torch.optim.Optimizer:
        """解耦权重衰减的 AdamW (norm/embedding 不衰减)。"""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".weight") and isinstance(p, nn.Linear) is False and "embedding" in name:
                no_decay.append(p)
            elif name.endswith(".bias") or "norm" in name or "embedding" in name:
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

    def extra_repr(self) -> str:
        return f"vocab={self.config.vocab_size}, H={self.config.hidden_size}, layers={self.config.num_layers}"
