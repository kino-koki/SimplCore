# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 主语言模型 (SimplCoreModel) - PyTorch 实现.

在 SimplCoreBlock 之上构建完整语言模型:

  input_ids ─► Embedding ─► RoPE ─► Dropout
                                   │
                                   ▼
                        SimplCoreBlock × N  (每块携带一份 DMC 记忆)
                                   │
                          (可选) LatentRollout 推演
                                   │
                                   ▼
                        Final LayerNorm ─► LM Head ─► logits

接口 (与 SimplCoreWorldModel 对齐):
  model(input_ids, memory_state=None) -> (logits, new_states)
    input_ids:    (B, L)
    memory_state: List[Optional[Tensor]], 长度 = num_layers, 每份 (B, C, S, D)
    logits:       (B, L, vocab_size)
    new_states:   List[Optional[Tensor]], 更新后的各块记忆

  model._text_to_hidden(input_ids) -> (B, L, H)
  model._forward_hidden(hidden, memory_state=None) -> (logits, new_states)

特性:
  - RoPE 旋转位置编码 (作用于隐状态).
  - 可选权重绑定 (tie_word_embeddings).
  - 可选梯度检查点 (gradient_checkpointing) 传给各块.
  - 可选潜空间推演 (config.use_rollout): 在 block 栈后对潜状态做多步迭代精化.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

try:  # 包内导入 (推荐)
    from .config import SimplCoreConfig
    from .simplcore_block import SimplCoreBlock
except ImportError:  # 直接以脚本运行时, 补全包路径后用绝对导入回退
    import os
    import sys

    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from simplcore.architecture.config import SimplCoreConfig  # type: ignore
    from simplcore.architecture.simplcore_block import SimplCoreBlock  # type: ignore


# ======================================================================
# RoPE - 旋转位置编码
# ======================================================================
class _RotaryPositionalEncoding(nn.Module):
    """旋转位置编码 (Rotary Position Embedding).

    将隐状态按维度对半切分为 (x1, x2), 对位置 p 施加旋转:
      [x1, x2] -> [x1*cos(pθ) - x2*sin(pθ), x1*sin(pθ) + x2*cos(pθ)]
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.half, dtype=torch.float32) / self.half)
        )                                                       # (half,)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]         # (max_seq_len, half)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape
        half = self.half
        cos = self.cos_cached[:L].to(dtype=x.dtype)             # (L, half)
        sin = self.sin_cached[:L].to(dtype=x.dtype)

        x1 = x[..., :half]
        x2 = x[..., half:2 * half]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.cat([rot1, rot2], dim=-1)
        if D % 2 == 1:                                          # 奇数维度尾部直通
            out = torch.cat([out, x[..., 2 * half:]], dim=-1)
        return out


# ======================================================================
# LatentRolloutEngine - 潜空间推演引擎 (自包含实现)
# ======================================================================
class _LatentRolloutEngine(nn.Module):
    """潜空间推演引擎 (自包含实现).

    当 config.use_rollout=True 时启用. 对 block 栈输出的潜状态做
    rollout_max_steps 步门控残差精化, 模拟在潜空间中的多步推演.

    注: architecture/rollout_engine.py 目前不可用, 故在此内联自包含实现,
    不依赖外部 RolloutEngine.

    Args:
        config: SimplCoreConfig.
    """

    def __init__(self, config: "SimplCoreConfig") -> None:
        super().__init__()
        dim = config.hidden_size
        self.max_steps = config.rollout_max_steps
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Linear(dim, 1)  # 每位置一个门控标量

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D) -> (B, L, D)."""
        for _ in range(self.max_steps):
            delta = self.refine(h)
            g = torch.sigmoid(self.gate(h))                    # (B, L, 1)
            h = h + g * delta
        return h


# ======================================================================
# SimplCoreModel - 主语言模型
# ======================================================================
class SimplCoreModel(nn.Module):
    """SimplCore 主语言模型.

    Args:
        config: SimplCoreConfig 配置.
        gradient_checkpointing: 是否对各块的 FFN 子层启用梯度检查点.
    """

    def __init__(
        self,
        config: "SimplCoreConfig",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size
        eps = config.rms_norm_eps

        # ---- 嵌入 ----
        self.embed = nn.Embedding(
            config.vocab_size, dim, padding_idx=config.pad_token_id
        )
        # 初始化为小值, 稳定训练
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[config.pad_token_id].zero_()

        # ---- 位置编码 (RoPE) ----
        self.rope = _RotaryPositionalEncoding(dim, config.max_sequence_length)

        # ---- Dropout ----
        self.embed_dropout = nn.Dropout(config.dropout)

        # ---- 核心块栈 ----
        self.blocks = nn.ModuleList(
            [
                SimplCoreBlock(config, gradient_checkpointing=gradient_checkpointing)
                for _ in range(config.num_layers)
            ]
        )

        # ---- 输出 ----
        self.final_norm = nn.LayerNorm(dim, eps=eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # 权重绑定: LM Head 与输入嵌入共享权重
            self.lm_head.weight = self.embed.weight

        # ---- 可选潜空间推演 ----
        self.use_rollout = config.use_rollout
        if self.use_rollout:
            self.rollout_engine = _LatentRolloutEngine(config)
        else:
            self.rollout_engine = None

    # ------------------------------------------------------------------
    # 文本 -> 隐状态
    # ------------------------------------------------------------------
    def _text_to_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 隐状态.

        Args:
            input_ids: (B, L) long.

        Returns:
            (B, L, H) 隐状态 (含 RoPE 与 dropout).
        """
        h = self.embed(input_ids)                              # (B, L, H)
        h = self.rope(h)                                       # RoPE
        h = self.embed_dropout(h)
        return h

    # ------------------------------------------------------------------
    # 隐状态 -> logits (核心前向)
    # ------------------------------------------------------------------
    def _forward_hidden(
        self,
        hidden: torch.Tensor,
        memory_state: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """隐状态 -> (logits, new_states).

        逐块前向, 每块携带一份 DMC 记忆; 可选在 block 栈后做潜空间推演.

        Args:
            hidden: (B, L, H).
            memory_state: List[Optional[Tensor]], 长度 = num_layers.

        Returns:
            logits:     (B, L, vocab_size).
            new_states: List[Optional[Tensor]], 更新后的各块记忆.
        """
        n = len(self.blocks)
        if memory_state is None:
            states: List[Optional[torch.Tensor]] = [None] * n
        else:
            # 兼容传入长度不足的情况
            states = list(memory_state) + [None] * max(0, n - len(memory_state))

        h = hidden
        new_states: List[Optional[torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            h, new_mem = block(h, states[i])
            new_states.append(new_mem)

        # 可选潜空间推演
        if self.rollout_engine is not None:
            h = self.rollout_engine(h)
            new_states.append(h)

        h = self.final_norm(h)
        logits = self.lm_head(h)                               # (B, L, vocab_size)
        return logits, new_states

    def gradient_checkpointing_enable(self) -> None:
        """启用各 SimplCoreBlock 的梯度检查点。"""
        for block in self.blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """关闭各 SimplCoreBlock 的梯度检查点。"""
        for block in self.blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # 主前向
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """文本前向.

        Args:
            input_ids: (B, L) token id.
            memory_state: 可选的各块 DMC 记忆状态.

        Returns:
            logits:     (B, L, vocab_size).
            new_states: 更新后的各块记忆 (长度 = num_layers).
        """
        hidden = self._text_to_hidden(input_ids)
        logits, new_states = self._forward_hidden(hidden, memory_state)
        return logits, new_states

    # ------------------------------------------------------------------
    # 参数统计
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """统计总参数量 (去重, 权重绑定只计一次)."""
        total = 0
        seen = set()
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def num_parameters_human(self) -> str:
        """以易读字符串返回参数量."""
        n = self.count_parameters()
        if n >= 1e9:
            return f"{n / 1e9:.2f} B ({n:,})"
        if n >= 1e6:
            return f"{n / 1e6:.3f} M ({n:,})"
        return f"{n / 1e3:.3f} K ({n:,})"


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("SimplCoreModel 测试")
    print("=" * 60)

    # ---- 基础配置 (无 rollout) ----
    cfg = SimplCoreConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_waves=4,
        num_channels=4,
        channel_size=8,
        num_paths=4,
        min_depth=1,
        max_depth=4,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=3,
        modality_dims=(64, 64, 64),
        tie_word_embeddings=True,
        dropout=0.0,
        sgc_num_steps=2,
        sgc_damping=0.1,
        dmc_num_heads=4,
        ccr_topk=2,
        use_rollout=False,
    )

    model = SimplCoreModel(cfg)
    print(f"模型参数量: {model.num_parameters_human()}")

    # 估算对比
    est = cfg.estimate_parameters()
    print(f"估算参数量: {est['total']:,}  (实际: {model.count_parameters():,})")

    # ---- Test 1: forward 形状 ----
    print("\n[Test 1] forward 形状")
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, states = model(input_ids)
    print(f"  input_ids={tuple(input_ids.shape)} -> logits={tuple(logits.shape)}")
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert isinstance(states, list) and len(states) == cfg.num_layers
    print(f"  states: list 长度={len(states)}, 每份 shape={tuple(states[0].shape)}")
    assert states[0].shape == (2, cfg.num_channels, cfg.channel_size, cfg.hidden_size)
    print("  [PASS]")

    # ---- Test 2: _text_to_hidden ----
    print("\n[Test 2] _text_to_hidden")
    hidden = model._text_to_hidden(input_ids)
    print(f"  input_ids={tuple(input_ids.shape)} -> hidden={tuple(hidden.shape)}")
    assert hidden.shape == (2, 16, cfg.hidden_size)
    print("  [PASS]")

    # ---- Test 3: _forward_hidden ----
    print("\n[Test 3] _forward_hidden")
    logits2, states2 = model._forward_hidden(hidden)
    assert logits2.shape == (2, 16, cfg.vocab_size)
    assert len(states2) == cfg.num_layers
    print(f"  hidden={tuple(hidden.shape)} -> logits={tuple(logits2.shape)}")
    print("  [PASS]")

    # ---- Test 4: 权重绑定 ----
    print("\n[Test 4] 权重绑定")
    assert model.lm_head.weight is model.embed.weight, "权重未绑定!"
    print(f"  lm_head.weight is embed.weight: {model.lm_head.weight is model.embed.weight}")
    print("  [PASS]")

    # ---- Test 5: 记忆状态跨步传递 ----
    print("\n[Test 5] 记忆状态跨步传递")
    _, states_a = model(input_ids)
    input_ids_b = torch.randint(0, cfg.vocab_size, (2, 12))
    logits_b, states_b = model(input_ids_b, memory_state=states_a)
    assert logits_b.shape == (2, 12, cfg.vocab_size)
    # 第二步记忆应与第一步不同 (已被写入更新)
    changed = not torch.allclose(states_a[0], states_b[0])
    print(f"  跨步后记忆已更新: {changed}")
    assert changed
    print("  [PASS]")

    # ---- Test 6: 梯度流 ----
    print("\n[Test 6] 梯度流验证")
    model.zero_grad(set_to_none=True)
    logits3, _ = model(input_ids)
    loss = logits3.pow(2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "模型无梯度!"
    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_total = len(list(model.parameters()))
    print(f"  有梯度的参数: {n_with_grad}/{n_total}")
    print("  [PASS]")

    # ---- Test 7: use_rollout=True 路径 ----
    print("\n[Test 7] use_rollout=True 路径")
    cfg_roll = SimplCoreConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_waves=4,
        num_channels=4,
        channel_size=8,
        num_paths=4,
        min_depth=1,
        max_depth=4,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=3,
        modality_dims=(64, 64, 64),
        tie_word_embeddings=True,
        dropout=0.0,
        sgc_num_steps=2,
        sgc_damping=0.1,
        dmc_num_heads=4,
        ccr_topk=2,
        use_rollout=True,
        rollout_max_steps=3,
    )
    model_roll = SimplCoreModel(cfg_roll)
    assert model_roll.rollout_engine is not None
    print(f"  rollout_engine.max_steps={model_roll.rollout_engine.max_steps}")
    logits_roll, states_roll = model_roll(input_ids)
    assert logits_roll.shape == (2, 16, cfg_roll.vocab_size)
    assert len(states_roll) == cfg_roll.num_layers
    # rollout 路径梯度流
    model_roll.zero_grad(set_to_none=True)
    logits_roll.pow(2).mean().backward()
    has_grad_roll = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_roll.rollout_engine.parameters()
    )
    assert has_grad_roll, "rollout 引擎无梯度!"
    print("  [PASS] rollout 引擎梯度流正常.")

    # ---- Test 8: tie_word_embeddings=False ----
    print("\n[Test 8] tie_word_embeddings=False")
    cfg_notie = SimplCoreConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_waves=4,
        num_channels=4,
        channel_size=8,
        num_paths=4,
        min_depth=1,
        max_depth=4,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=3,
        modality_dims=(64, 64, 64),
        tie_word_embeddings=False,
        dropout=0.0,
        sgc_num_steps=2,
        dmc_num_heads=4,
        ccr_topk=2,
        use_rollout=False,
    )
    model_notie = SimplCoreModel(cfg_notie)
    assert model_notie.lm_head.weight is not model_notie.embed.weight
    print(f"  独立 lm_head: weight is embed.weight = "
          f"{model_notie.lm_head.weight is model_notie.embed.weight}")
    print(f"  参数量(不绑定)={model_notie.count_parameters():,} > "
          f"参数量(绑定)={model.count_parameters():,}")
    assert model_notie.count_parameters() > model.count_parameters()
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("SimplCoreModel 全部测试 PASSED.")
    print("=" * 60)
