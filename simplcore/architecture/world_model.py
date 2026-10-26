# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""世界模型动力学模块 (DynamicsModel).

核心思想: 模拟真实世界的物理规律, 预测状态转移。

架构设计:
  当前状态 Sₜ + 动作 Aₜ → 动力学模型 → 下一状态 Sₜ₊₁
  
  支持三种预测模式:
    1. 确定性预测 (物理模拟)
    2. 概率性预测 (不确定性建模)
    3. 多模态预测 (多个可能的未来)

与 RolloutEngine 的关系:
  - RolloutEngine: 在潜空间进行多步推演 (参数共享迭代)
  - DynamicsModel: 学习状态转移函数 (显式建模动力学)
  - 两者结合: RolloutEngine 使用 DynamicsModel 作为每步的转移函数
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


class DynamicsModel(nn.Module):
    """动力学模型 - 学习状态转移函数.
    
    Args:
        state_dim: 状态维度 (潜空间维度).
        action_dim: 动作维度.
        hidden_dim: 隐藏层维度.
        num_layers: 动力学模型层数.
        probabilistic: 是否启用概率性预测 (输出均值+方差).
        multimodal: 是否启用多模态预测 (输出多个可能的未来).
        num_modes: 多模态预测的模态数量.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        probabilistic: bool = False,
        multimodal: bool = False,
        num_modes: int = 5,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.probabilistic = probabilistic
        self.multimodal = multimodal
        self.num_modes = num_modes
        
        # 输入投影: state + action → hidden
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # 动力学核心: 多层 MLP 学习状态转移
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        self.core = nn.Sequential(*layers)
        
        if multimodal:
            # 多模态预测: 输出多个模式的状态
            self.output_proj = nn.Linear(hidden_dim, num_modes * state_dim)
            self.mode_logits = nn.Linear(hidden_dim, num_modes)
        elif probabilistic:
            # 概率性预测: 输出均值和方差
            self.output_proj = nn.Linear(hidden_dim, 2 * state_dim)
        else:
            # 确定性预测: 直接输出下一状态
            self.output_proj = nn.Linear(hidden_dim, state_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """预测下一状态.
        
        Args:
            state: (B, L, H) 当前状态序列.
            action: (B, L, A) 动作序列.
            
        Returns:
            dict: 包含预测结果的字典.
        """
        B, L, _ = state.shape
        
        # 将状态和动作拼接
        x = torch.cat([state, action], dim=-1)  # (B, L, H+A)
        x = self.input_proj(x)                    # (B, L, hidden)
        x = self.core(x)                          # (B, L, hidden)
        
        if self.multimodal:
            # 多模态预测
            modes = self.output_proj(x)           # (B, L, num_modes * H)
            modes = modes.view(B, L, self.num_modes, self.state_dim)
            logits = self.mode_logits(x)          # (B, L, num_modes)
            probs = F.softmax(logits, dim=-1)
            
            return {
                "next_state": modes,      # (B, L, num_modes, H)
                "mode_probs": probs,      # (B, L, num_modes)
            }
            
        elif self.probabilistic:
            # 概率性预测
            out = self.output_proj(x)             # (B, L, 2H)
            mean = out[:, :, :self.state_dim]
            log_var = out[:, :, self.state_dim:]
            std = torch.exp(0.5 * log_var)
            
            return {
                "next_state": mean,       # (B, L, H)
                "mean": mean,             # (B, L, H)
                "std": std,               # (B, L, H)
                "log_var": log_var,       # (B, L, H)
            }
            
        else:
            # 确定性预测
            next_state = self.output_proj(x)      # (B, L, H)
            
            return {
                "next_state": next_state,  # (B, L, H)
            }
    
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        steps: int = 1,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """多步预测.
        
        Args:
            state: (B, L, H) 初始状态.
            action: (B, L, A) 动作序列 (如果 steps>1, 需要对应步数的动作).
            steps: 预测步数.
            return_all: 是否返回所有中间状态.
            
        Returns:
            dict: 包含预测结果的字典.
        """
        current_state = state
        all_states = [current_state]
        
        for step in range(steps):
            # 获取当前步的动作
            if action.dim() == 3:
                # 动作序列: 使用整个动作序列
                if action.shape[1] == current_state.shape[1]:
                    step_action = action
                elif action.shape[1] == 1:
                    step_action = action.expand(-1, current_state.shape[1], -1)
                else:
                    step_action = action[:, step] if step < action.shape[1] else action[:, -1]
                    step_action = step_action.unsqueeze(1).expand(-1, current_state.shape[1], -1)
            elif action.dim() == 2:
                step_action = action.unsqueeze(1).expand(-1, current_state.shape[1], -1)
            else:
                step_action = action
            
            result = self.forward(current_state, step_action)
            current_state = result["next_state"]
            
            if self.multimodal:
                # 选择概率最高的模式
                mode_idx = result["mode_probs"].argmax(dim=-1)
                B, L = mode_idx.shape
                idx = mode_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, self.state_dim)
                current_state = current_state.gather(dim=2, index=idx).squeeze(2)
            
            all_states.append(current_state)
        
        if return_all:
            return {"next_state": current_state, "all_states": torch.stack(all_states, dim=1)}
        return {"next_state": current_state}


class ActionHead(nn.Module):
    """动作生成头 - 基于观察生成动作.
    
    支持两种模式:
      1. 离散动作 (分类): 输出动作类别概率分布
      2. 连续动作 (回归): 输出连续动作向量
    
    Args:
        state_dim: 状态维度.
        action_dim: 动作维度 (离散时为动作类别数).
        hidden_dim: 隐藏层维度.
        discrete: 是否为离散动作空间.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discrete: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        if not discrete:
            # 连续动作: 输出均值和方差
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """生成动作.
        
        Args:
            state: (B, L, H) 状态序列.
            deterministic: 是否确定性输出 (仅对连续动作有效).
            
        Returns:
            dict: 包含动作的字典.
        """
        x = self.mlp(state)  # (B, L, action_dim)
        
        if self.discrete:
            # 离散动作: 返回 logits 和概率
            logits = x
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1)
            
            return {
                "action": action,       # (B, L) 动作类别
                "action_logits": logits, # (B, L, action_dim)
                "action_probs": probs,   # (B, L, action_dim)
            }
        
        else:
            # 连续动作: 返回均值和采样动作
            mean = torch.tanh(x)  # 限制在 [-1, 1]
            std = torch.exp(self.log_std)
            
            if deterministic:
                action = mean
            else:
                action = mean + std * torch.randn_like(mean)
                action = torch.clamp(action, -1, 1)
            
            return {
                "action": action,       # (B, L, action_dim)
                "action_mean": mean,     # (B, L, action_dim)
                "action_std": std,       # (action_dim,)
            }


class VideoPredictionHead(nn.Module):
    """视频预测头 - 预测下一帧图像.
    
    核心思想: 将潜状态解码为图像像素.
    
    支持两种解码方式:
      1. CNN 解码器 (从潜状态逐步上采样到图像)
      2. Transformer 解码器 (类似 DiT 的像素生成)
    
    Args:
        latent_dim: 潜状态维度.
        image_size: 输出图像尺寸.
        patch_size: 图像分块大小 (用于 CNN 上采样).
        num_channels: 图像通道数 (默认 RGB=3).
    """
    
    def __init__(
        self,
        latent_dim: int,
        image_size: int = 64,
        patch_size: int = 4,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        # 计算 patch 数量和尺寸
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = num_channels * patch_size * patch_size
        
        # 潜状态 → patch 嵌入
        self.latent_to_patches = nn.Linear(latent_dim, self.num_patches * self.patch_dim)
        
        # CNN 上采样器: 将 patch 排列成图像并上采样
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 输出 [0, 1]
        )
    
    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """将潜状态解码为图像.
        
        Args:
            latent: (B, L, H) 潜状态序列.
            
        Returns:
            image: (B, C, H, W) 预测的图像.
        """
        B, L, H = latent.shape
        
        # 使用 CLS token 或最后一个 token 作为图像表示
        if L > 1:
            # 使用第一个 token (CLS)
            x = latent[:, 0, :]  # (B, H)
        else:
            x = latent.squeeze(1)  # (B, H)
        
        # 潜状态 → patch 嵌入
        patches = self.latent_to_patches(x)  # (B, num_patches * patch_dim)
        patches = patches.view(B, self.num_patches, self.patch_dim)
        
        # 排列成图像: (B, num_patches, C*H*W) → (B, C, image_size, image_size)
        num_side = int(self.num_patches ** 0.5)
        img = patches.view(
            B, num_side, num_side, self.num_channels, self.patch_size, self.patch_size
        )
        img = img.permute(0, 3, 1, 4, 2, 5).contiguous()
        img = img.view(B, self.num_channels, self.image_size, self.image_size)
        
        # CNN 上采样
        img = self.upsampler(img)
        
        return img


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 60)
    print("DynamicsModel 测试")
    print("=" * 60)
    
    # 测试确定性动力学
    dyn_determ = DynamicsModel(state_dim=128, action_dim=16, hidden_dim=256, num_layers=2)
    state = torch.randn(2, 10, 128)
    action = torch.randn(2, 10, 16)
    result = dyn_determ(state, action)
    print(f"确定性预测: input(state={state.shape}, action={action.shape})")
    print(f"             output(next_state={result['next_state'].shape})")
    assert result['next_state'].shape == (2, 10, 128)
    print("[PASS] 确定性动力学正常.")
    
    # 测试概率性动力学
    dyn_prob = DynamicsModel(state_dim=128, action_dim=16, probabilistic=True)
    result_prob = dyn_prob(state, action)
    print(f"概率性预测: mean={result_prob['mean'].shape}, std={result_prob['std'].shape}")
    assert result_prob['mean'].shape == (2, 10, 128)
    assert result_prob['std'].shape == (2, 10, 128)
    print("[PASS] 概率性动力学正常.")
    
    # 测试多模态动力学
    dyn_multi = DynamicsModel(state_dim=128, action_dim=16, multimodal=True, num_modes=5)
    result_multi = dyn_multi(state, action)
    print(f"多模态预测: next_state={result_multi['next_state'].shape}, mode_probs={result_multi['mode_probs'].shape}")
    assert result_multi['next_state'].shape == (2, 10, 5, 128)
    assert result_multi['mode_probs'].shape == (2, 10, 5)
    print("[PASS] 多模态动力学正常.")
    
    # 测试多步预测
    result_steps = dyn_determ.predict(state, action, steps=3, return_all=True)
    print(f"多步预测: all_states={result_steps['all_states'].shape}")
    assert result_steps['all_states'].shape == (2, 4, 10, 128)  # steps+1 个状态
    print("[PASS] 多步预测正常.")
    
    print("\n" + "=" * 60)
    print("ActionHead 测试")
    print("=" * 60)
    
    # 测试离散动作
    action_discrete = ActionHead(state_dim=128, action_dim=10, discrete=True)
    result_act = action_discrete(state)
    print(f"离散动作: action={result_act['action'].shape}, logits={result_act['action_logits'].shape}")
    assert result_act['action'].shape == (2, 10)
    assert result_act['action_logits'].shape == (2, 10, 10)
    print("[PASS] 离散动作头正常.")
    
    # 测试连续动作
    action_cont = ActionHead(state_dim=128, action_dim=6, discrete=False)
    result_act_cont = action_cont(state)
    print(f"连续动作: action={result_act_cont['action'].shape}, mean={result_act_cont['action_mean'].shape}")
    assert result_act_cont['action'].shape == (2, 10, 6)
    assert result_act_cont['action_mean'].shape == (2, 10, 6)
    print("[PASS] 连续动作头正常.")
    
    print("\n" + "=" * 60)
    print("VideoPredictionHead 测试")
    print("=" * 60)
    
    video_head = VideoPredictionHead(latent_dim=128, image_size=64, patch_size=4)
    latent = torch.randn(2, 16, 128)
    img = video_head(latent)
    print(f"视频预测: latent={latent.shape} → image={img.shape}")
    assert img.shape == (2, 3, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    print("[PASS] 视频预测头正常.")
    
    print("\n" + "=" * 60)
    print("所有世界模型模块测试 PASSED.")
    print("=" * 60)

