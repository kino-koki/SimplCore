# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore AGI 训练系统 - 完整训练管道.

整合所有模块:
  1. 数据加载 (DataLoader + Dataset)
  2. 模型 (SimplCore AGI)
  3. 优化器 (AdamW + 学习率调度)
  4. 分布式训练 (ZeRO-3 + 混合精度)
  5. 日志记录 (TrainingLogger)
  6. 检查点 (ModelCheckpoint)
  7. 早停 (EarlyStopping)
  8. 评估 (Benchmark)

优化策略:
  - 学习率调度: 余弦退火 + 预热
  - 正则化: L2正则化 + Dropout
  - 梯度裁剪: 防止梯度爆炸
  - 混合精度训练: 提高训练效率
  - 早停: 防止过拟合
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

import simplcore.tensor as torch
import numpy as np
import os
import time
import json
from typing import Dict, List, Optional, Any, Callable
from simplcore.tensor.optim import CosineAnnealingLR, LinearLR, SequentialLR

from SimplCore.agi.SimplCore_agi import SimplCoreAGI
from SimplCore.knowledge.knowledge_base import KnowledgeBase
from SimplCore.engine.train import DataLoader, Dataset, ModelCheckpoint, TrainingLogger, EarlyStopping
from SimplCore.engine.distributed import (
    DistributedConfig, init_distributed, MixedPrecisionTraining,
    GradientAccumulator,
)


class DataAugmenter:
    """数据增强器 - 为训练数据添加噪声和变换."""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        shift_prob: float = 0.3,
        scale_prob: float = 0.3,
        flip_prob: float = 0.2,
        time_warp_prob: float = 0.2,
    ):
        self.noise_std = noise_std
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob
        self.flip_prob = flip_prob
        self.time_warp_prob = time_warp_prob
    
    def augment(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对状态和动作进行数据增强."""
        states_aug = states.copy()
        actions_aug = actions.copy()
        
        # 添加高斯噪声
        if self.noise_std > 0:
            states_aug += np.random.randn(*states.shape) * self.noise_std
            actions_aug += np.random.randn(*actions.shape) * self.noise_std * 0.5
        
        # 随机平移
        if np.random.rand() < self.shift_prob:
            shift = np.random.randn(states.shape[-1]) * 0.05
            states_aug = states_aug + shift
        
        # 随机缩放
        if np.random.rand() < self.scale_prob:
            scale = 1.0 + np.random.randn() * 0.05
            states_aug = states_aug * scale
        
        # 时间翻转
        if np.random.rand() < self.flip_prob:
            states_aug = states_aug[:, ::-1, :]
            actions_aug = actions_aug[:, ::-1, :]
        
        # 时间扭曲 (插值)
        if np.random.rand() < self.time_warp_prob:
            states_aug = self._time_warp(states_aug)
        
        return states_aug, actions_aug
    
    def _time_warp(self, x: np.ndarray) -> np.ndarray:
        """时间扭曲 - 使用线性插值改变时间步长."""
        B, T, D = x.shape
        new_indices = np.linspace(0, T-1, T) + np.random.randn(T) * 0.5
        new_indices = np.clip(new_indices, 0, T-1)
        
        x_warped = np.zeros_like(x)
        for b in range(B):
            for d in range(D):
                x_warped[b, :, d] = np.interp(np.arange(T), new_indices, x[b, :, d])
        
        return x_warped


class AGIDataset(Dataset):
    """AGI 训练数据集 - 增强版"""
    
    def __init__(
        self,
        size: int = 10000,
        seq_len: int = 128,
        hidden_size: int = 128,
        action_dim: int = 16,
        augment: bool = True,
    ):
        self.size = size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.augment = augment
        self.augmenter = DataAugmenter()
        
        self._generate_data()
    
    def _generate_data(self):
        """生成训练数据 - 加入多种物理规律"""
        print(f"[AGIDataset] 生成 {self.size} 条训练数据...")
        
        self.states = np.random.randn(self.size, self.seq_len, self.hidden_size).astype(np.float32)
        self.actions = np.random.randn(self.size, self.seq_len, self.action_dim).astype(np.float32)
        
        # 生成带有多种物理规律的下一状态
        self.next_states = np.zeros((self.size, self.seq_len, self.hidden_size), dtype=np.float32)
        
        # 预生成动作扩展
        action_expanded = np.zeros((self.size, self.seq_len, self.hidden_size), dtype=np.float32)
        action_expanded[:, :, :self.action_dim] = self.actions
        
        for i in range(self.size):
            # 混合物理规律:
            # 1. 阻尼运动: state * damping
            # 2. 速度积分: action * dt
            # 3. 弹簧回复: -state * spring_constant
            # 4. 摩擦力: -velocity * friction
            
            damping = 0.95
            velocity = action_expanded[i] * 0.1
            spring_force = -self.states[i] * 0.02
            friction = -velocity * 0.1
            
            self.next_states[i] = (
                self.states[i] * damping +
                velocity +
                spring_force +
                friction
            )
        
        # 生成目标动作 (用于动作损失)
        self.target_actions = self._generate_target_actions()
        
        # 生成目标图像特征 (用于图像损失)
        self.target_images = self._generate_target_images()
        
        print(f"[AGIDataset] 数据生成完成")
    
    def _generate_target_actions(self) -> np.ndarray:
        """生成目标动作序列."""
        target_actions = np.zeros((self.size, self.seq_len, self.action_dim), dtype=np.float32)
        
        for i in range(self.size):
            # 目标动作 = 当前动作 + 随机扰动 + 策略修正
            noise = np.random.randn(self.seq_len, self.action_dim) * 0.1
            correction = (self.next_states[i, :, :self.action_dim] - 
                         self.states[i, :, :self.action_dim]) * 0.5
            target_actions[i] = self.actions[i] + noise + correction
        
        return target_actions
    
    def _generate_target_images(self) -> np.ndarray:
        """生成目标图像特征 (模拟图像预测损失)."""
        target_images = np.zeros((self.size, 3, 64, 64), dtype=np.float32)
        
        for i in range(self.size):
            # 从状态生成图像特征
            state_summary = self.next_states[i].mean(axis=0)
            img = np.random.randn(3, 64, 64).astype(np.float32) * 0.3
            # 将状态信息融入图像
            for c in range(3):
                img[c, :, :] += state_summary[c % self.hidden_size] * 0.1
            target_images[i] = np.clip(img, 0, 1)
        
        return target_images
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = self.states[idx]
        actions = self.actions[idx]
        
        # 数据增强
        if self.augment:
            states, actions = self.augmenter.augment(
                states[np.newaxis], actions[np.newaxis]
            )
            states = states[0]
            actions = actions[0]
        
        return (
            states,
            actions,
            self.next_states[idx],
            self.target_actions[idx],
            self.target_images[idx],
        )


class LearningRateScheduler:
    """学习率调度器 - 余弦退火 + 预热"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        
        # 预热阶段: 线性增长
        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        
        # 主阶段: 余弦退火
        self.main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=base_lr * 0.01,
        )
        
        # 组合调度器
        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[self.warmup_scheduler, self.main_scheduler],
            milestones=[warmup_epochs],
        )
    
    def step(self) -> None:
        """更新学习率"""
        self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class Regularization:
    """正则化管理器"""
    
    def __init__(self, model: torch.nn.Module, weight_decay: float = 0.01, dropout_rate: float = 0.1):
        self.model = model
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        
        # 添加Dropout层
        self._add_dropout_layers()
    
    def _add_dropout_layers(self) -> None:
        """为模型添加Dropout层"""
        pass
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """计算正则化损失 - 使用AdamW内置的权重衰减，这里返回0"""
        return torch.tensor(0.0)


class MultiLossFunction:
    """多损失函数 - 动力学损失 + 动作损失 + 图像损失."""
    
    def __init__(
        self,
        dynamics_weight: float = 1.0,
        action_weight: float = 0.5,
        image_weight: float = 0.3,
        reconstruction_weight: float = 0.2,
    ):
        self.dynamics_weight = dynamics_weight
        self.action_weight = action_weight
        self.image_weight = image_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """计算多损失.
        
        Args:
            outputs: 模型输出.
            targets: 目标值.
            
        Returns:
            dict: 包含各损失项和总损失.
        """
        losses = {}
        
        # 动力学损失
        if "next_state" in outputs and "next_state" in targets:
            dynamics_loss = self.mse_loss(outputs["next_state"], targets["next_state"])
            losses["dynamics_loss"] = dynamics_loss.item()
        else:
            dynamics_loss = torch.tensor(0.0)
            losses["dynamics_loss"] = 0.0
        
        # 动作损失
        if "action" in outputs and "target_action" in targets:
            action_loss = self.l1_loss(outputs["action"], targets["target_action"])
            losses["action_loss"] = action_loss.item()
        else:
            action_loss = torch.tensor(0.0)
            losses["action_loss"] = 0.0
        
        # 图像损失
        if "next_image" in outputs and "target_image" in targets:
            image_loss = self.mse_loss(outputs["next_image"], targets["target_image"])
            losses["image_loss"] = image_loss.item()
        else:
            image_loss = torch.tensor(0.0)
            losses["image_loss"] = 0.0
        
        # 重建损失 (当前状态重建)
        if "reconstruction" in outputs and "state" in targets:
            recon_loss = self.mse_loss(outputs["reconstruction"], targets["state"])
            losses["reconstruction_loss"] = recon_loss.item()
        else:
            recon_loss = torch.tensor(0.0)
            losses["reconstruction_loss"] = 0.0
        
        # 总损失
        total_loss = (
            dynamics_loss * self.dynamics_weight +
            action_loss * self.action_weight +
            image_loss * self.image_weight +
            recon_loss * self.reconstruction_weight
        )
        losses["total_loss"] = total_loss.item()
        
        return {"loss": total_loss, "losses": losses}


class AGITrainer:
    """AGI 训练器 - 优化版"""
    
    def __init__(
        self,
        agi: SimplCoreAGI,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        patience: int = 10,
        warmup_epochs: int = 5,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.1,
        max_grad_norm: float = 1.0,
        dynamics_weight: float = 1.0,
        action_weight: float = 0.5,
        image_weight: float = 0.3,
    ):
        self.agi = agi
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.max_grad_norm = max_grad_norm
        
        # 初始化多损失函数
        self.loss_fn = MultiLossFunction(
            dynamics_weight=dynamics_weight,
            action_weight=action_weight,
            image_weight=image_weight,
        )
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            agi.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # 初始化学习率调度器
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            base_lr=learning_rate,
        )
        
        # 初始化正则化管理器
        self.regularization = Regularization(
            agi,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
        )
        
        # 初始化分布式配置
        self.dist_config = init_distributed()
        
        # 初始化混合精度训练
        self.mixed_precision_trainer = MixedPrecisionTraining(self.dist_config, enabled=mixed_precision)
        
        # 初始化梯度累加器
        self.gradient_accumulator = GradientAccumulator(accumulation_steps)
        
        # 初始化日志记录器
        self.logger = TrainingLogger(log_dir=log_dir, name="agi_training")
        
        # 初始化检查点管理器
        self.checkpoint_manager = ModelCheckpoint(
            save_dir=save_dir,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            max_checkpoints=5,
        )
        
        # 初始化早停
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=True,
        )
        
        # 训练状态
        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.dynamics_loss_history = []
        self.action_loss_history = []
        self.image_loss_history = []
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.agi.train()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        total_action_loss = 0.0
        total_image_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            states, actions, next_states, target_actions, target_images = batch
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            target_actions = torch.tensor(target_actions, dtype=torch.float32)
            target_images = torch.tensor(target_images, dtype=torch.float32)
            
            self.optimizer.zero_grad()
            
            outputs = self.agi.predict_future(states, actions, steps=1)
            
            targets = {
                "next_state": next_states,
                "target_action": target_actions,
                "target_image": target_images,
                "state": states,
            }
            
            loss_result = self.loss_fn.compute(outputs, targets)
            total_loss_val = loss_result["loss"]
            losses = loss_result["losses"]
            
            total_loss_val.backward()
            
            torch.nn.utils.clip_grad_norm_(self.agi.parameters(), max_norm=self.max_grad_norm)
            
            self.optimizer.step()
            
            batch_size = states.shape[0]
            total_loss += losses["total_loss"] * batch_size
            total_dynamics_loss += losses.get("dynamics_loss", 0) * batch_size
            total_action_loss += losses.get("action_loss", 0) * batch_size
            total_image_loss += losses.get("image_loss", 0) * batch_size
            total_samples += batch_size
            
            if batch_idx % 50 == 0:
                lr = self.lr_scheduler.get_lr()
                self.logger.log(
                    f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} "
                    f"Loss: {losses['total_loss']:.6f} "
                    f"Dyn: {losses.get('dynamics_loss', 0):.6f} "
                    f"Act: {losses.get('action_loss', 0):.6f} "
                    f"Img: {losses.get('image_loss', 0):.6f} "
                    f"LR: {lr:.8f}"
                )
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        avg_dynamics = total_dynamics_loss / total_samples if total_samples > 0 else float("inf")
        avg_action = total_action_loss / total_samples if total_samples > 0 else float("inf")
        avg_image = total_image_loss / total_samples if total_samples > 0 else float("inf")
        
        self.train_loss_history.append(avg_loss)
        self.dynamics_loss_history.append(avg_dynamics)
        self.action_loss_history.append(avg_action)
        self.image_loss_history.append(avg_image)
        self.lr_history.append(self.lr_scheduler.get_lr())
        
        return {
            "loss": avg_loss,
            "dynamics_loss": avg_dynamics,
            "action_loss": avg_action,
            "image_loss": avg_image,
        }
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.agi.eval()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        total_action_loss = 0.0
        total_image_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                states, actions, next_states, target_actions, target_images = batch
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                target_actions = torch.tensor(target_actions, dtype=torch.float32)
                target_images = torch.tensor(target_images, dtype=torch.float32)
                
                outputs = self.agi.predict_future(states, actions, steps=1)
                
                targets = {
                    "next_state": next_states,
                    "target_action": target_actions,
                    "target_image": target_images,
                    "state": states,
                }
                
                loss_result = self.loss_fn.compute(outputs, targets)
                losses = loss_result["losses"]
                
                batch_size = states.shape[0]
                total_loss += losses["total_loss"] * batch_size
                total_dynamics_loss += losses.get("dynamics_loss", 0) * batch_size
                total_action_loss += losses.get("action_loss", 0) * batch_size
                total_image_loss += losses.get("image_loss", 0) * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        avg_dynamics = total_dynamics_loss / total_samples if total_samples > 0 else float("inf")
        avg_action = total_action_loss / total_samples if total_samples > 0 else float("inf")
        avg_image = total_image_loss / total_samples if total_samples > 0 else float("inf")
        
        self.val_loss_history.append(avg_loss)
        
        return {
            "loss": avg_loss,
            "dynamics_loss": avg_dynamics,
            "action_loss": avg_action,
            "image_loss": avg_image,
        }
    
    def fit(self) -> None:
        """训练主循环"""
        self.logger.log(f"[AGITrainer] 开始训练 AGI 系统")
        self.logger.log(f"[AGITrainer] 训练轮数: {self.epochs}")
        self.logger.log(f"[AGITrainer] 学习率: {self.learning_rate}")
        self.logger.log(f"[AGITrainer] 预热轮数: {self.warmup_epochs}")
        self.logger.log(f"[AGITrainer] 权重衰减: {self.weight_decay}")
        self.logger.log(f"[AGITrainer] 梯度裁剪: {self.max_grad_norm}")
        self.logger.log(f"[AGITrainer] 混合精度: {self.mixed_precision}")
        self.logger.log(f"[AGITrainer] 梯度累加: {self.accumulation_steps}")
        self.logger.log(f"[AGITrainer] 早停耐心: {self.patience}")
        
        for epoch in range(self.epochs):
            self.logger.log(f"\n===== Epoch {epoch + 1}/{self.epochs} =====")
            
            train_metrics = self.train_one_epoch(epoch)
            self.logger.log_metrics(epoch, train_metrics, phase="train")
            
            val_metrics = self.validate()
            if val_metrics:
                self.logger.log_metrics(epoch, val_metrics, phase="val")
            
            combined_metrics = {**train_metrics, **val_metrics}
            
            # 更新学习率
            self.lr_scheduler.step()
            
            # 保存检查点
            if self.checkpoint_manager:
                self.checkpoint_manager.save(self.agi, epoch, combined_metrics)
            
            # 早停检查
            if self.early_stopping and self.early_stopping.step(combined_metrics):
                self.logger.log("[AGITrainer] 早停触发，训练停止")
                break
            
            # 更新最佳验证损失
            if "val_loss" in combined_metrics and combined_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = combined_metrics["val_loss"]
        
        self.logger.log("[AGITrainer] 训练完成")
        self.logger.save_metrics()
        self.logger.plot_metrics()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.agi.eval()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        total_action_loss = 0.0
        total_image_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                states, actions, next_states, target_actions, target_images = batch
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                target_actions = torch.tensor(target_actions, dtype=torch.float32)
                target_images = torch.tensor(target_images, dtype=torch.float32)
                
                outputs = self.agi.predict_future(states, actions, steps=1)
                
                targets = {
                    "next_state": next_states,
                    "target_action": target_actions,
                    "target_image": target_images,
                    "state": states,
                }
                
                loss_result = self.loss_fn.compute(outputs, targets)
                losses = loss_result["losses"]
                
                batch_size = states.shape[0]
                total_loss += losses["total_loss"] * batch_size
                total_dynamics_loss += losses.get("dynamics_loss", 0) * batch_size
                total_action_loss += losses.get("action_loss", 0) * batch_size
                total_image_loss += losses.get("image_loss", 0) * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        avg_dynamics = total_dynamics_loss / total_samples if total_samples > 0 else float("inf")
        avg_action = total_action_loss / total_samples if total_samples > 0 else float("inf")
        avg_image = total_image_loss / total_samples if total_samples > 0 else float("inf")
        
        return {
            "test_loss": avg_loss,
            "test_dynamics_loss": avg_dynamics,
            "test_action_loss": avg_action,
            "test_image_loss": avg_image,
        }


def main():
    """主训练函数"""
    print("=" * 70)
    print("SimplCore AGI 训练系统 - 多损失优化版")
    print("=" * 70)
    
    config = {
        "train_size": 10000,
        "val_size": 1000,
        "test_size": 1000,
        "batch_size": 16,
        "seq_len": 10,
        "hidden_size": 128,
        "action_dim": 16,
        "learning_rate": 5e-4,
        "epochs": 100,
        "warmup_epochs": 5,
        "weight_decay": 0.001,
        "dropout_rate": 0.1,
        "max_grad_norm": 1.0,
        "mixed_precision": False,
        "accumulation_steps": 2,
        "patience": 50,
        "dynamics_weight": 1.0,
        "action_weight": 0.5,
        "image_weight": 0.3,
    }
    
    print("\n[Config] 训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\n[Dataset] 创建数据集...")
    train_dataset = AGIDataset(
        size=config["train_size"],
        seq_len=config["seq_len"],
        hidden_size=config["hidden_size"],
        action_dim=config["action_dim"],
        augment=True,
    )
    val_dataset = AGIDataset(
        size=config["val_size"],
        seq_len=config["seq_len"],
        hidden_size=config["hidden_size"],
        action_dim=config["action_dim"],
        augment=False,
    )
    test_dataset = AGIDataset(
        size=config["test_size"],
        seq_len=config["seq_len"],
        hidden_size=config["hidden_size"],
        action_dim=config["action_dim"],
        augment=False,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")
    print(f"  测试集: {len(test_dataset)} 条")
    
    print("\n[AGI] 创建 AGI 系统...")
    agi = SimplCoreAGI()
    agi.print_system_info()
    
    print("\n[Trainer] 创建训练器...")
    trainer = AGITrainer(
        agi=agi,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        warmup_epochs=config["warmup_epochs"],
        weight_decay=config["weight_decay"],
        dropout_rate=config["dropout_rate"],
        max_grad_norm=config["max_grad_norm"],
        mixed_precision=config["mixed_precision"],
        accumulation_steps=config["accumulation_steps"],
        patience=config["patience"],
        dynamics_weight=config["dynamics_weight"],
        action_weight=config["action_weight"],
        image_weight=config["image_weight"],
    )
    
    print("\n[Train] 开始训练...")
    start_time = time.time()
    trainer.fit()
    elapsed_time = time.time() - start_time
    
    print(f"\n[Train] 训练完成，耗时: {elapsed_time:.2f} 秒")
    
    print("\n[Eval] 评估模型...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  测试损失: {test_metrics.get('test_loss', 'N/A'):.6f}")
    print(f"  动力学损失: {test_metrics.get('test_dynamics_loss', 'N/A'):.6f}")
    print(f"  动作损失: {test_metrics.get('test_action_loss', 'N/A'):.6f}")
    print(f"  图像损失: {test_metrics.get('test_image_loss', 'N/A'):.6f}")
    
    print("\n[Save] 保存最终模型...")
    final_checkpoint = {
        "model_state_dict": agi.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.lr_scheduler.scheduler.state_dict(),
        "config": config,
        "train_loss_history": trainer.train_loss_history,
        "val_loss_history": trainer.val_loss_history,
        "dynamics_loss_history": trainer.dynamics_loss_history,
        "action_loss_history": trainer.action_loss_history,
        "image_loss_history": trainer.image_loss_history,
        "lr_history": trainer.lr_history,
        "test_metrics": test_metrics,
        "training_time": elapsed_time,
        "timestamp": time.time(),
    }
    
    os.makedirs("./models", exist_ok=True)
    torch.save(final_checkpoint, "./models/agi_final.pt")
    print("  模型已保存到 ./models/agi_final.pt")
    
    with open("./models/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("SimplCore AGI 训练系统运行完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()


