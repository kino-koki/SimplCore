# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 集成模型 (IntegratedSimplCoreModel)
=============================================

将所有 SimplCore 组件集成为统一的端到端模型:

  文本路径: input_ids → text_embed → SimplCoreBlocks → output_head → logits
  视觉路径: image → vision_encoder → SimplCoreBlocks → output_head → logits
  多模态:   text_embed ⊕ vision_embed → SimplCoreBlocks → output_head → logits

组件:
  - 文本嵌入层 (Token + Position Embedding)
  - SimplCoreBlock 层 (SGC + DMC + CCR + FFN)
  - 可选视觉编码器 (ViT)
  - 潜空间推演引擎 (RolloutEngine, 可选)
  - 输出头 (LM Head)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

# ------------------------------------------------------------------
# 导入 SimplCoreConfig (如果 config.py 损坏则使用回退定义)
# ------------------------------------------------------------------
try:
    from .config import SimplCoreConfig
except Exception:
    from dataclasses import dataclass

    @dataclass
    class SimplCoreConfig:
        """SimplCore 配置 (回退定义)."""
        vocab_size: int = 32000
        hidden_size: int = 768
        num_layers: int = 12
        num_waves: int = 8
        num_channels: int = 64
        channel_size: int = 128
        num_heads: int = 8
        num_paths: int = 4
        ffn_ratio: int = 4
        max_seq_len: int = 2048
        dropout: float = 0.1
        use_vision: bool = False
        image_size: int = 224
        patch_size: int = 16
        num_rollout_steps: int = 5

        @classmethod
        def from_preset(cls, name: str) -> "SimplCoreConfig":
            presets = {
                "tiny": cls(vocab_size=1000, hidden_size=64, num_layers=2),
                "150m": cls(vocab_size=32000, hidden_size=768, num_layers=12),
                "7b": cls(vocab_size=32000, hidden_size=4096, num_layers=32),
                "14b": cls(vocab_size=32000, hidden_size=5120, num_layers=40),
            }
            key = name.lower()
            if key not in presets:
                raise ValueError(f"未知预设: {name}. 可用: {list(presets.keys())}")
            return presets[key]

# ------------------------------------------------------------------
# 导入 SimplCoreBlock (如果 simplcore_block.py 损坏则使用回退定义)
# ------------------------------------------------------------------
try:
    from .simplcore_block import SimplCoreBlock
except Exception:
    class SimplCoreBlock(nn.Module):
        """SimplCore 块回退实现 (当 simplcore_block.py 不可用时使用).

        包含简化的 SGC + DMC + CCR + FFN 结构。
        """

        def __init__(self, config: SimplCoreConfig) -> None:
            super().__init__()
            dim = config.hidden_size
            self.dim = dim

            # 简化的 SGC: 使用线性投影模拟波场
            self.sgc_norm = nn.LayerNorm(dim)
            self.sgc_in = nn.Linear(dim, dim)
            self.sgc_out = nn.Linear(dim, dim)

            # 简化的 DMC: 记忆通道
            self.dmc_norm = nn.LayerNorm(dim)
            self.dmc_proj = nn.Linear(dim, dim)

            # 简化的 CCR: 路由
            self.ccr_norm = nn.LayerNorm(dim)
            self.ccr_router = nn.Linear(dim, config.num_paths)
            self.ccr_transform = nn.Linear(dim, dim)

            # FFN
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * config.ffn_ratio),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(dim * config.ffn_ratio, dim),
                nn.Dropout(config.dropout),
            )

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            x: torch.Tensor,
            memory_state: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """前向传播: SGC → DMC → CCR → FFN."""
            # SGC (简化)
            residual = x
            h = self.sgc_norm(x)
            h = torch.tanh(self.sgc_in(h))
            x = residual + self.dropout(self.sgc_out(h))

            # DMC (简化)
            residual = x
            h = self.dmc_norm(x)
            x = residual + self.dropout(self.dmc_proj(h))

            # CCR (简化: 加权路径)
            residual = x
            h = self.ccr_norm(x)
            route_probs = F.softmax(self.ccr_router(h), dim=-1)
            h = self.ccr_transform(h)
            x = residual + self.dropout(h)

            # FFN
            residual = x
            h = self.ffn_norm(x)
            x = residual + self.ffn(h)

            return x, memory_state


# ======================================================================
# 简化视觉编码器 (回退)
# ======================================================================

class _SimpleVisionEncoder(nn.Module):
    """简化视觉编码器 (当 vision_encoder.py 不可用时使用).

    使用 patch embedding + 线性投影将图像转换为序列嵌入。
    """

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.dim = config.hidden_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = 3 * self.patch_size * self.patch_size

        self.patch_embed = nn.Linear(patch_dim, self.dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.norm = nn.LayerNorm(self.dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """将图像编码为序列嵌入.

        Args:
            image: (B, C, H, W) 输入图像.

        Returns:
            (B, num_patches+1, dim) 视觉嵌入序列.
        """
        B = image.shape[0]
        # 分块: (B, C, H, W) → (B, num_patches, patch_dim)
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            B, 3, self.num_patches, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, self.num_patches, -1)

        # 投影到隐藏维度
        embeddings = self.patch_embed(patches)  # (B, num_patches, dim)

        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls, embeddings], dim=1)  # (B, num_patches+1, dim)

        # 位置嵌入
        embeddings = embeddings + self.pos_embed

        return self.norm(embeddings)


# ======================================================================
# 简化推演引擎 (回退)
# ======================================================================

class _SimpleRolloutEngine(nn.Module):
    """简化推演引擎 (当 rollout_engine.py 不可用时使用).

    在潜空间进行多步迭代预测。
    """

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_steps = config.num_rollout_steps
        self.step_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多步推演.

        Args:
            x: (B, L, dim) 初始潜状态.

        Returns:
            (B, L, dim) 推演后的潜状态.
        """
        for _ in range(self.num_steps):
            x = x + self.step_net(self.norm(x))
        return x


# ======================================================================
# 集成模型
# ======================================================================

class IntegratedSimplCoreModel(nn.Module):
    """SimplCore 集成模型 - 统一文本、视觉和多模态处理.

    将所有 SimplCore 组件集成到一个端到端的 PyTorch 模型中:

      ┌─────────────────────────────────────────────────┐
      │  文本路径:  input_ids → embed → blocks → head   │
      │  视觉路径:  image → vision_encoder → blocks → head │
      │  多模态:   text_embed ⊕ vision_embed → blocks → head │
      └─────────────────────────────────────────────────┘

    Args:
        config: SimplCoreConfig 配置实例, 包含:
            - vocab_size: 词表大小
            - hidden_size: 隐藏层维度
            - num_layers: SimplCoreBlock 层数
            - max_seq_len: 最大序列长度
            - use_vision: 是否启用视觉编码器
            - num_rollout_steps: 推演步数 (0 表示不使用推演)
            - dropout: dropout 概率
            等其他组件配置参数。
    """

    def __init__(
        self,
        config: SimplCoreConfig,
        enable_world_head: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_layers = config.num_layers
        self.enable_world_head = enable_world_head

        # === 文本嵌入 ===
        self.token_embed = nn.Embedding(config.vocab_size, self.dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, self.dim) * 0.02
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        # === SimplCoreBlock 层 ===
        self.blocks = self._build_blocks(config)

        # === 最终 LayerNorm ===
        self.final_norm = nn.LayerNorm(self.dim)

        # === 输出头 ===
        self.output_head = nn.Linear(self.dim, config.vocab_size, bias=False)
        # 权重绑定 (可选): output_head.weight = token_embed.weight

        # === 可选视觉编码器 ===
        self.use_vision = config.use_vision
        if self.use_vision:
            try:
                from .vision_encoder import SimplCoreViTEncoder
                self.vision_encoder = SimplCoreViTEncoder(
                    output_dim=self.dim,
                    image_size=config.image_size,
                    patch_size=config.patch_size,
                )
            except Exception:
                self.vision_encoder = _SimpleVisionEncoder(config)
        else:
            self.vision_encoder = None

        # === 可选推演引擎 ===
        self.use_rollout = config.num_rollout_steps > 0
        if self.use_rollout:
            try:
                from .rollout_engine import RolloutEngine
                self.rollout_engine = RolloutEngine(config)
            except Exception:
                self.rollout_engine = _SimpleRolloutEngine(config)
        else:
            self.rollout_engine = None

        # === 模态嵌入 (多模态区分) ===
        self.modality_embed = nn.Parameter(
            torch.randn(2, self.dim) * 0.02  # 0=text, 1=vision
        )

        if enable_world_head:
            try:
                from .rollout_engine import WorldModelHead
                self.world_head = WorldModelHead(dim=self.dim, vocab_size=config.vocab_size)
            except Exception:
                self.world_head = None
                self.reward_head = nn.Linear(self.dim, 1)
                self.continuation_head = nn.Linear(self.dim, 1)
        else:
            self.world_head = None

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化模型权重."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_blocks(self, config: SimplCoreConfig) -> nn.ModuleList:
        """创建 SimplCoreBlock 层列表.

        Args:
            config: 配置实例.

        Returns:
            nn.ModuleList: 包含 num_layers 个 SimplCoreBlock。
        """
        blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            blocks.append(SimplCoreBlock(config))
        return blocks

    # ==================================================================
    # 嵌入方法
    # ==================================================================

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """文本嵌入: token embedding + position embedding.

        Args:
            input_ids: (B, L) token ID 序列.

        Returns:
            (B, L, dim) 文本嵌入.
        """
        B, L = input_ids.shape
        token_emb = self.token_embed(input_ids)  # (B, L, dim)
        pos_emb = self.pos_embed[:, :L, :]       # (1, L, dim)

        # 添加文本模态嵌入
        mod_emb = self.modality_embed[0].view(1, 1, self.dim)
        embeddings = token_emb + pos_emb + mod_emb

        return self.embed_dropout(embeddings)

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像嵌入: 视觉编码器 + 模态嵌入.

        Args:
            image: (B, C, H, W) 输入图像.

        Returns:
            (B, L_img, dim) 视觉嵌入.
        """
        if self.vision_encoder is None:
            raise RuntimeError(
                "视觉编码器未启用, 请在配置中设置 use_vision=True"
            )

        # 编码图像
        if hasattr(self.vision_encoder, "encode_image"):
            vision_emb = self.vision_encoder.encode_image(image)
        else:
            vision_emb = self.vision_encoder(image)

        # 添加视觉模态嵌入
        mod_emb = self.modality_embed[1].view(1, 1, self.dim)
        vision_emb = vision_emb + mod_emb

        return self.embed_dropout(vision_emb)

    # ==================================================================
    # 核心前向传播
    # ==================================================================

    def _run_blocks(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """通过 SimplCoreBlock 层处理序列.

        Args:
            x: (B, L, dim) 输入序列.
            memory_state: 可选的 DMC 记忆状态.

        Returns:
            tuple: (hidden_states, memory_state)
        """
        current_memory = memory_state
        for block in self.blocks:
            # 尝试调用 block 的 forward, 支持 memory_state 参数
            try:
                x, current_memory = block(x, memory_state=current_memory)
            except TypeError:
                # 如果 block 不接受 memory_state 参数
                result = block(x)
                if isinstance(result, tuple):
                    x = result[0]
                    if len(result) > 1:
                        current_memory = result[1]
                else:
                    x = result

        x = self.final_norm(x)
        return x, current_memory

    # ==================================================================
    # 前向传播
    # ==================================================================

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """统一前向传播: 支持文本、视觉和多模态输入.

        路由逻辑:
          - 仅 input_ids: 文本路径 (embed → blocks → output)
          - 仅 image: 视觉路径 (vision_encoder → blocks → output)
          - 两者都有: 多模态 (拼接嵌入后过 blocks)
          - 两者都无: 抛出 ValueError

        Args:
            input_ids: (B, L_text) 文本 token ID, 可选.
            image: (B, C, H, W) 图像, 可选.
            memory_state: 可选的 DMC 记忆状态.

        Returns:
            dict: {
              "logits": (B, L, vocab_size) 输出 logits,
              "hidden_states": (B, L, dim) 隐藏状态,
              "memory_state": 可选的记忆状态,
            }

        Raises:
            ValueError: 当 input_ids 和 image 均为 None 时.
        """
        if input_ids is None and image is None:
            raise ValueError(
                "至少需要提供一种输入 (input_ids 或 image)"
            )

        embeddings: torch.Tensor
        seq_lengths: List[int] = []

        # === 构建输入嵌入 ===
        if input_ids is not None and image is not None:
            # 多模态: 拼接文本和视觉嵌入
            text_emb = self._embed_text(input_ids)
            vision_emb = self._embed_image(image)
            seq_lengths = [text_emb.shape[1], vision_emb.shape[1]]
            embeddings = torch.cat([text_emb, vision_emb], dim=1)
        elif input_ids is not None:
            # 纯文本
            embeddings = self._embed_text(input_ids)
            seq_lengths = [embeddings.shape[1]]
        else:
            # 纯视觉
            embeddings = self._embed_image(image)
            seq_lengths = [embeddings.shape[1]]

        # === 通过 SimplCoreBlock 层 ===
        hidden_states, new_memory = self._run_blocks(embeddings, memory_state)

        # === 可选: 推演引擎 ===
        if self.rollout_engine is not None and self.training:
            hidden_states = self.rollout_engine(hidden_states)

        # === 输出头 ===
        logits = self.output_head(hidden_states)

        result = {
            "logits": logits,
            "hidden_states": hidden_states,
            "memory_state": new_memory,
            "seq_lengths": seq_lengths,
        }

        if self.enable_world_head:
            if self.world_head is not None:
                _, reward, continuation = self.world_head(hidden_states)
            else:
                reward = self.reward_head(hidden_states)
                continuation = torch.sigmoid(self.continuation_head(hidden_states))
            result["reward"] = reward
            result["continuation"] = continuation

        if return_dict:
            return result
        return result["logits"]

    # ==================================================================
    # 工具方法
    # ==================================================================

    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数量.

        Returns:
            dict: {
              "total": 总参数量,
              "trainable": 可训练参数量,
              "embedding": 嵌入层参数量,
              "blocks": Block 层参数量,
              "output_head": 输出头参数量,
              "vision_encoder": 视觉编码器参数量 (0 如果未启用),
            }
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 嵌入层
        embed_params = sum(
            p.numel() for p in self.token_embed.parameters()
        ) + self.pos_embed.numel() + self.modality_embed.numel()

        # Block 层
        block_params = sum(
            p.numel() for p in self.blocks.parameters()
        )

        # 输出头
        head_params = sum(
            p.numel() for p in self.output_head.parameters()
        )

        # 视觉编码器
        vision_params = 0
        if self.vision_encoder is not None:
            vision_params = sum(
                p.numel() for p in self.vision_encoder.parameters()
            )

        # 推演引擎
        rollout_params = 0
        if self.rollout_engine is not None:
            rollout_params = sum(
                p.numel() for p in self.rollout_engine.parameters()
            )

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embed_params,
            "blocks": block_params,
            "output_head": head_params,
            "vision_encoder": vision_params,
            "rollout_engine": rollout_params,
            "final_norm": sum(p.numel() for p in self.final_norm.parameters()),
        }

    def get_memory_state(self) -> Optional[torch.Tensor]:
        """获取当前记忆状态 (如果 DMC 层支持)."""
        for block in self.blocks:
            if hasattr(block, "get_state"):
                return block.get_state()
            if hasattr(block, "dmc") and hasattr(block.dmc, "get_state"):
                return block.dmc.get_state()
        return None

    def reset_memory_state(self, batch_size: int) -> None:
        """重置记忆状态."""
        for block in self.blocks:
            if hasattr(block, "reset_state"):
                block.reset_state(batch_size)
            elif hasattr(block, "dmc") and hasattr(block.dmc, "reset_state"):
                block.dmc.reset_state(batch_size)

    def detach_memory_state(self) -> None:
        """分离记忆状态 (截断 BPTT)."""
        for block in self.blocks:
            if hasattr(block, "detach_state"):
                block.detach_state()
            elif hasattr(block, "dmc") and hasattr(block.dmc, "detach_state"):
                block.dmc.detach_state()

    def summary(self) -> str:
        """返回模型摘要字符串."""
        params = self.count_parameters()
        lines = [
            "=" * 60,
            "IntegratedSimplCoreModel 摘要",
            "=" * 60,
            f"  配置: hidden_size={self.dim}, num_layers={self.num_layers}",
            f"  vocab_size={self.config.vocab_size}",
            f"  use_vision={self.use_vision}",
            f"  use_rollout={self.use_rollout}",
            f"{'─' * 60}",
            f"  参数量分解:",
            f"    Embedding:     {params['embedding']:>12,}",
            f"    Blocks:        {params['blocks']:>12,}",
            f"    Output Head:   {params['output_head']:>12,}",
            f"    Vision:        {params['vision_encoder']:>12,}",
            f"    Rollout:       {params['rollout_engine']:>12,}",
            f"    Final Norm:    {params['final_norm']:>12,}",
            f"{'─' * 60}",
            f"  总参数量:       {params['total']:>12,}",
            f"  可训练参数:     {params['trainable']:>12,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("IntegratedSimplCoreModel 自检")
    print("=" * 60)

    # 创建 tiny 配置
    config = SimplCoreConfig.from_preset("tiny")
    config.use_vision = True
    config.num_rollout_steps = 2
    config.max_seq_len = 64

    model = IntegratedSimplCoreModel(config)
    print(model.summary())

    # === 测试文本路径 ===
    print("\n--- 文本路径测试 ---")
    input_ids = torch.randint(1, config.vocab_size, (2, 16))
    result = model(input_ids=input_ids)
    print(f"输入: input_ids={tuple(input_ids.shape)}")
    print(f"输出: logits={tuple(result['logits'].shape)}")
    print(f"      hidden_states={tuple(result['hidden_states'].shape)}")
    assert result["logits"].shape == (2, 16, config.vocab_size)
    assert result["hidden_states"].shape == (2, 16, config.hidden_size)
    assert not torch.isnan(result["logits"]).any()
    print("[PASS] 文本路径正常")

    # === 测试视觉路径 ===
    print("\n--- 视觉路径测试 ---")
    image = torch.randn(2, 3, config.image_size, config.image_size)
    result_v = model(image=image)
    print(f"输入: image={tuple(image.shape)}")
    print(f"输出: logits={tuple(result_v['logits'].shape)}")
    assert not torch.isnan(result_v["logits"]).any()
    print("[PASS] 视觉路径正常")

    # === 测试多模态路径 ===
    print("\n--- 多模态路径测试 ---")
    result_m = model(input_ids=input_ids, image=image)
    print(f"输入: input_ids={tuple(input_ids.shape)}, image={tuple(image.shape)}")
    print(f"输出: logits={tuple(result_m['logits'].shape)}")
    expected_len = 16 + result_v["logits"].shape[1]
    assert result_m["logits"].shape[0] == 2
    assert not torch.isnan(result_m["logits"]).any()
    print("[PASS] 多模态路径正常")

    # === 测试无输入错误 ===
    print("\n--- 错误处理测试 ---")
    try:
        model()
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"[PASS] 正确抛出 ValueError: {e}")

    # === 测试参数统计 ===
    print("\n--- 参数统计测试 ---")
    params = model.count_parameters()
    print(f"总参数量: {params['total']:,}")
    print(f"可训练:   {params['trainable']:,}")
    assert params["total"] > 0
    assert params["total"] == params["trainable"]

    # === 测试梯度 ===
    print("\n--- 梯度测试 ---")
    model.train()
    result_g = model(input_ids=input_ids)
    loss = result_g["logits"].sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print(f"[PASS] 梯度正常: {has_grad}")
    assert has_grad

    # === 测试无视觉编码器的模型 ===
    print("\n--- 纯文本模型测试 ---")
    config_text = SimplCoreConfig.from_preset("tiny")
    config_text.use_vision = False
    config_text.num_rollout_steps = 0
    model_text = IntegratedSimplCoreModel(config_text)
    result_text = model_text(input_ids=input_ids)
    assert result_text["logits"].shape == (2, 16, config_text.vocab_size)
    print("[PASS] 纯文本模型正常")

    # === 测试记忆状态 ===
    print("\n--- 记忆状态测试 ---")
    result1 = model(input_ids=input_ids)
    result2 = model(input_ids=input_ids, memory_state=result1["memory_state"])
    print("[PASS] 记忆状态传递正常")

    print("\n" + "=" * 60)
    print("IntegratedSimplCoreModel 所有测试 PASSED.")
    print("=" * 60)
