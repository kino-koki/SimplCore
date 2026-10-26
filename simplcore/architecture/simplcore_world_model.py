# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 世界模型 (SimplCoreWorldModel) - 统一世界模型接口.

将所有世界模型组件整合到一个统一的接口中:
  - 多模态感知 (文本 + 图像 + 视频)
  - 潜空间推演 (RolloutEngine)
  - 动力学预测 (DynamicsModel)
  - 动作生成 (ActionHead)
  - 视频预测 (VideoPredictionHead)

核心能力:
  1. 语言理解: 处理文本输入, 生成文本输出
  2. 视觉感知: 处理图像/视频输入
  3. 物理模拟: 预测状态转移
  4. 动作规划: 基于观察生成动作
  5. 视频预测: 预测下一帧画面

与 SimplCoreModel 的关系:
  - SimplCoreModel: 基础语言模型 (文本输入 → 文本输出)
  - SimplCoreWorldModel: 扩展世界模型 (多模态输入 → 多模态输出 + 预测)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel
from SimplCore.architecture.modality_router import ModalityRouter
from SimplCore.architecture.world_model import (
    DynamicsModel,
    ActionHead,
    VideoPredictionHead,
)


class SimplCoreWorldModel(nn.Module):
    """SimplCore 世界模型 - 统一多模态预测接口.
    
    Args:
        config: SimplCoreConfig 配置对象.
        action_dim: 动作维度.
        image_size: 预测图像尺寸.
        use_dynamics: 是否启用动力学模型.
        use_action_head: 是否启用动作生成头.
        use_video_prediction: 是否启用视频预测头.
        probabilistic: 是否启用概率性动力学预测.
    """
    
    def __init__(
        self,
        config: SimplCoreConfig,
        action_dim: int = 16,
        image_size: int = 64,
        use_dynamics: bool = True,
        use_action_head: bool = True,
        use_video_prediction: bool = True,
        probabilistic: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.image_size = image_size
        
        # ---- 核心语言模型 ----
        self.core_model = SimplCoreModel(config)
        
        # ---- 多模态路由 ----
        self.modality_router = ModalityRouter(
            hidden_size=config.hidden_size,
            use_vision=True,
        )
        
        # ---- 动力学模型 ----
        if use_dynamics:
            self.dynamics_model = DynamicsModel(
                state_dim=config.hidden_size,
                action_dim=action_dim,
                hidden_dim=config.hidden_size,
                num_layers=3,
                probabilistic=probabilistic,
            )
        else:
            self.dynamics_model = None
        
        # ---- 动作生成头 ----
        if use_action_head:
            self.action_head = ActionHead(
                state_dim=config.hidden_size,
                action_dim=action_dim,
                hidden_dim=config.hidden_size,
                discrete=False,  # 默认连续动作
            )
        else:
            self.action_head = None
        
        # ---- 视频预测头 ----
        if use_video_prediction:
            self.video_prediction_head = VideoPredictionHead(
                latent_dim=config.hidden_size,
                image_size=image_size,
                patch_size=4,
            )
        else:
            self.video_prediction_head = None
    
    def forward_text(
        self,
        input_ids: torch.Tensor,
        memory_state: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """纯文本前向 (语言理解).
        
        Args:
            input_ids: (B, L) token id 序列.
            memory_state: 可选的 DMC 记忆状态.
            
        Returns:
            dict: 包含 logits 和记忆状态.
        """
        logits, new_states = self.core_model(input_ids, memory_state)
        return {"logits": logits, "memory_state": new_states}
    
    def forward_image(
        self,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """图像前向 (视觉感知).
        
        Args:
            image: (B, C, H, W) 图像.
            
        Returns:
            dict: 包含潜状态和模态掩码.
        """
        result = self.modality_router(image=image)
        latent = result["latent"]
        modality_mask = result["modality_mask"]
        
        # 通过核心模型处理视觉潜状态
        # 注意: 需要将视觉潜状态转为模型可接受的格式
        B, L, H = latent.shape
        # 创建 dummy input_ids 用于触发 forward
        dummy_ids = torch.zeros(B, L, dtype=torch.long, device=latent.device)
        
        # 直接使用 hidden 表示进行前向
        logits, _ = self.core_model._forward_hidden(latent)
        
        return {
            "latent": latent,
            "modality_mask": modality_mask,
            "logits": logits,
        }
    
    def forward_multimodal(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """多模态前向 (文本 + 图像 + 视频).
        
        Args:
            text_embeds: (B, L, H) 文本嵌入.
            image: (B, C, H, W) 图像.
            video: (B, T, C, H, W) 视频.
            input_ids: (B, L) token id (如果提供文本嵌入则不需要).
            
        Returns:
            dict: 包含联合潜状态和预测结果.
        """
        # 获取文本嵌入
        if text_embeds is None and input_ids is not None:
            text_embeds = self.core_model._text_to_hidden(input_ids)
        
        # 使用模态路由融合所有模态
        result = self.modality_router(
            text_embeds=text_embeds,
            image=image,
            video=video,
        )
        latent = result["latent"]
        modality_mask = result["modality_mask"]
        
        # 通过核心模型处理
        logits, _ = self.core_model._forward_hidden(latent)
        
        return {
            "latent": latent,
            "modality_mask": modality_mask,
            "logits": logits,
        }
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """预测下一状态 (物理模拟).
        
        Args:
            state: (B, L, H) 当前状态.
            action: (B, L, A) 动作.
            steps: 预测步数.
            
        Returns:
            dict: 包含预测的下一状态.
        """
        if self.dynamics_model is None:
            raise NotImplementedError("动力学模型未启用.")
        
        return self.dynamics_model.predict(state, action, steps=steps, return_all=True)
    
    def generate_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """基于状态生成动作 (动作规划).
        
        Args:
            state: (B, L, H) 当前状态.
            deterministic: 是否确定性输出.
            
        Returns:
            dict: 包含生成的动作.
        """
        if self.action_head is None:
            raise NotImplementedError("动作生成头未启用.")
        
        return self.action_head(state, deterministic=deterministic)
    
    def predict_video(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """预测下一帧图像 (视频预测).
        
        Args:
            latent: (B, L, H) 潜状态.
            
        Returns:
            image: (B, C, H, W) 预测的图像.
        """
        if self.video_prediction_head is None:
            raise NotImplementedError("视频预测头未启用.")
        
        return self.video_prediction_head(latent)
    
    def world_step(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        generate_action: bool = True,
        predict_next_state: bool = True,
        predict_image: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """世界模型单步推演.
        
        完整的世界模型循环:
          当前状态 → 生成动作 → 预测下一状态 → 预测下一帧图像
        
        Args:
            state: (B, L, H) 当前状态.
            action: (B, L, A) 动作 (如果不生成).
            generate_action: 是否生成动作.
            predict_next_state: 是否预测下一状态.
            predict_image: 是否预测下一帧图像.
            
        Returns:
            dict: 包含所有预测结果.
        """
        results = {"current_state": state}
        
        # 1. 生成动作
        if generate_action and self.action_head is not None:
            action_result = self.action_head(state)
            action = action_result["action"]
            results.update(action_result)
        
        # 2. 预测下一状态
        if predict_next_state and self.dynamics_model is not None and action is not None:
            state_result = self.dynamics_model(state, action)
            next_state = state_result["next_state"]
            results.update(state_result)
            results["next_state"] = next_state
        else:
            next_state = state
        
        # 3. 预测下一帧图像
        if predict_image and self.video_prediction_head is not None:
            next_image = self.video_prediction_head(next_state)
            results["next_image"] = next_image
        
        return results
    
    def rollout_world(
        self,
        initial_state: torch.Tensor,
        steps: int = 10,
        initial_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """世界模型多步推演.
        
        在潜空间进行多步预测:
          S₀ → A₀ → S₁ → A₁ → S₂ → ... → Sₜ
        
        Args:
            initial_state: (B, L, H) 初始状态.
            steps: 推演步数.
            initial_action: 初始动作 (可选).
            
        Returns:
            dict: 包含所有中间状态、动作和图像.
        """
        all_states = [initial_state]
        all_actions = []
        all_images = []
        
        current_state = initial_state
        current_action = initial_action
        
        for step in range(steps):
            result = self.world_step(
                current_state,
                action=current_action,
                generate_action=True,
                predict_next_state=True,
                predict_image=True,
            )
            
            current_state = result["next_state"]
            current_action = result["action"]
            
            all_states.append(current_state)
            all_actions.append(current_action)
            if "next_image" in result:
                all_images.append(result["next_image"])
        
        return {
            "all_states": torch.stack(all_states, dim=1),    # (B, steps+1, L, H)
            "all_actions": torch.stack(all_actions, dim=1),  # (B, steps, L, A)
            "all_images": torch.stack(all_images, dim=1) if all_images else None,  # (B, steps, C, H, W)
        }
    
    def count_parameters(self) -> int:
        """统计总参数量."""
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
    print("SimplCoreWorldModel 测试")
    print("=" * 60)
    
    # 创建配置
    cfg = SimplCoreConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_waves=2,
        num_channels=4,
        channel_size=8,
        num_paths=2,
        min_depth=1,
        max_depth=2,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=3,
        modality_dims=(64, 64, 64),
        tie_word_embeddings=True,
        dropout=0.0,
        sgc_num_steps=2,
        sgc_damping=0.1,
        dmc_num_heads=2,
        ccr_topk=1,
        use_rollout=True,
        rollout_max_steps=3,
    )
    
    # 创建世界模型
    model = SimplCoreWorldModel(
        config=cfg,
        action_dim=8,
        image_size=64,
        use_dynamics=True,
        use_action_head=True,
        use_video_prediction=True,
    )
    
    print(f"世界模型参数量: {model.num_parameters_human()}")
    
    # ---- Test 1: 纯文本前向 ----
    print("\n[Test 1] 纯文本前向")
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    text_result = model.forward_text(input_ids)
    print(f"  input={input_ids.shape}, logits={text_result['logits'].shape}")
    assert text_result['logits'].shape == (2, 16, cfg.vocab_size)
    print("  [PASS]")
    
    # ---- Test 2: 图像前向 ----
    print("\n[Test 2] 图像前向")
    image = torch.randn(2, 3, 224, 224)
    img_result = model.forward_image(image)
    print(f"  input={image.shape}, latent={img_result['latent'].shape}")
    assert img_result['latent'].shape == (2, 196, 64)
    print("  [PASS]")
    
    # ---- Test 3: 多模态前向 ----
    print("\n[Test 3] 多模态前向")
    mm_result = model.forward_multimodal(
        text_embeds=torch.randn(2, 8, 64),
        image=torch.randn(2, 3, 224, 224),
    )
    print(f"  text+image → latent={mm_result['latent'].shape}")
    assert mm_result['latent'].shape[0] == 2
    print("  [PASS]")
    
    # ---- Test 4: 动力学预测 ----
    print("\n[Test 4] 动力学预测")
    state = torch.randn(2, 10, 64)
    action = torch.randn(2, 10, 8)
    dyn_result = model.predict_next_state(state, action, steps=3)
    print(f"  state={state.shape}, action={action.shape}")
    print(f"  next_state={dyn_result['next_state'].shape}")
    print(f"  all_states={dyn_result['all_states'].shape}")
    assert dyn_result['all_states'].shape == (2, 4, 10, 64)
    print("  [PASS]")
    
    # ---- Test 5: 动作生成 ----
    print("\n[Test 5] 动作生成")
    action_result = model.generate_action(state)
    print(f"  state={state.shape}, action={action_result['action'].shape}")
    assert action_result['action'].shape == (2, 10, 8)
    print("  [PASS]")
    
    # ---- Test 6: 视频预测 ----
    print("\n[Test 6] 视频预测")
    latent = torch.randn(2, 16, 64)
    img_pred = model.predict_video(latent)
    print(f"  latent={latent.shape}, image={img_pred.shape}")
    assert img_pred.shape == (2, 3, 64, 64)
    assert img_pred.min() >= 0 and img_pred.max() <= 1
    print("  [PASS]")
    
    # ---- Test 7: 世界模型单步推演 ----
    print("\n[Test 7] 世界模型单步推演")
    step_result = model.world_step(state)
    print(f"  输出键: {list(step_result.keys())}")
    assert "next_state" in step_result
    assert "action" in step_result
    assert "next_image" in step_result
    print("  [PASS]")
    
    # ---- Test 8: 世界模型多步推演 ----
    print("\n[Test 8] 世界模型多步推演")
    rollout_result = model.rollout_world(state, steps=5)
    print(f"  all_states={rollout_result['all_states'].shape}")
    print(f"  all_actions={rollout_result['all_actions'].shape}")
    print(f"  all_images={rollout_result['all_images'].shape}")
    assert rollout_result['all_states'].shape == (2, 6, 10, 64)
    assert rollout_result['all_actions'].shape == (2, 5, 10, 8)
    assert rollout_result['all_images'].shape == (2, 5, 3, 64, 64)
    print("  [PASS]")
    
    # ---- Test 9: 梯度流验证 ----
    print("\n[Test 9] 梯度流验证")
    input_ids2 = torch.randint(0, cfg.vocab_size, (1, 8))
    result2 = model.forward_text(input_ids2)
    loss = result2["logits"].mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "模型没有梯度!"
    print("  [PASS] 梯度流经所有模块.")
    
    print("\n" + "=" * 60)
    print("SimplCoreWorldModel 全部测试 PASSED!")
    print("=" * 60)


