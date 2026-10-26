# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""视频预测模块 - 预测未来帧，添加光流估计和时序一致性约束.

核心功能:
  1. 光流估计 - 估计帧间运动信息
  2. 视频预测 - 预测下一帧或多帧图像
  3. 时序一致性约束 - 确保预测结果在时间上连续
  4. 帧插值 - 生成中间帧
  5. 运动补偿 - 基于光流进行运动补偿

设计目标:
  - 支持多步预测
  - 保持时序一致性
  - 支持光流引导的预测
  - 支持帧插值和超分辨率
"""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class OpticalFlowEstimator(nn.Module):
    """光流估计器 - 估计两帧之间的运动向量."""
    
    def __init__(self, input_channels: int = 6, hidden_channels: int = 64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """估计光流.
        
        Args:
            frame1: 前一帧 [B, 3, H, W]
            frame2: 后一帧 [B, 3, H, W]
        
        Returns:
            flow: 光流场 [B, 2, H, W]
        """
        input = torch.cat([frame1, frame2], dim=1)
        flow = self.conv_layers(input)
        return flow
    
    def warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """根据光流场扭曲图像.
        
        Args:
            frame: 输入图像 [B, 3, H, W]
            flow: 光流场 [B, 2, H, W]
        
        Returns:
            warped_frame: 扭曲后的图像 [B, 3, H, W]
        """
        B, _, H, W = frame.shape
        
        # 创建网格
        grid_x = torch.arange(W, dtype=torch.float32, device=frame.device)
        grid_y = torch.arange(H, dtype=torch.float32, device=frame.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
        
        # 添加批次维度
        grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)
        
        # 应用光流
        warped_x = grid_x + flow[:, 0, :, :]
        warped_y = grid_y + flow[:, 1, :, :]
        
        # 归一化到 [-1, 1]
        warped_x = (warped_x / (W - 1)) * 2 - 1
        warped_y = (warped_y / (H - 1)) * 2 - 1
        
        # 创建网格张量
        grid = torch.stack([warped_x, warped_y], dim=3)
        
        # 双线性插值
        warped_frame = F.grid_sample(frame, grid, mode="bilinear", padding_mode="border")
        
        return warped_frame
    
    def compute_flow_magnitude(self, flow: torch.Tensor) -> torch.Tensor:
        """计算光流幅度."""
        return torch.sqrt(flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2)


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失 - 确保预测结果在时间上连续."""
    
    def __init__(self, lambda_flow: float = 1.0, lambda_smooth: float = 0.5):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_smooth = lambda_smooth
    
    def forward(
        self,
        predicted_frames: torch.Tensor,
        flow_estimator: OpticalFlowEstimator,
    ) -> torch.Tensor:
        """计算时序一致性损失.
        
        Args:
            predicted_frames: 预测帧序列 [B, T, 3, H, W]
            flow_estimator: 光流估计器
        
        Returns:
            loss: 时序一致性损失
        """
        B, T, C, H, W = predicted_frames.shape
        
        total_loss = 0.0
        num_pairs = T - 1
        
        for t in range(num_pairs):
            frame1 = predicted_frames[:, t]
            frame2 = predicted_frames[:, t + 1]
            
            # 估计光流
            flow = flow_estimator(frame1, frame2)
            
            # 光流平滑损失
            flow_magnitude = flow_estimator.compute_flow_magnitude(flow)
            smooth_loss = torch.mean(flow_magnitude)
            
            # 运动一致性损失 - 扭曲前一帧应该接近后一帧
            warped_frame1 = flow_estimator.warp_frame(frame1, flow)
            consistency_loss = F.l1_loss(warped_frame1, frame2)
            
            total_loss += self.lambda_smooth * smooth_loss + self.lambda_flow * consistency_loss
        
        return total_loss / num_pairs


class VideoPredictionModel(nn.Module):
    """视频预测模型 - 预测未来帧."""
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 128,
        num_frames: int = 10,
        num_predicted_frames: int = 5,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_predicted_frames = num_predicted_frames
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 时序建模
        self.lstm = nn.LSTM(
            input_size=hidden_channels * 4,
            hidden_size=hidden_channels * 4,
            num_layers=2,
            batch_first=True,
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        # 光流估计器
        self.flow_estimator = OpticalFlowEstimator()
        
        # 时序一致性损失
        self.temporal_loss = TemporalConsistencyLoss()
    
    def forward(
        self,
        frames: torch.Tensor,
        predict_frames: int = None,
    ) -> Dict[str, torch.Tensor]:
        """预测未来帧.
        
        Args:
            frames: 输入帧序列 [B, T, 3, H, W]
            predict_frames: 预测帧数
        
        Returns:
            predictions: 预测帧 [B, predict_frames, 3, H, W]
            flows: 光流场列表
        """
        predict_frames = predict_frames or self.num_predicted_frames
        
        B, T, C, H, W = frames.shape
        
        # 提取特征
        features = []
        for t in range(T):
            frame = frames[:, t]
            feat = self.encoder(frame)
            # 展平
            feat = feat.view(B, -1)
            features.append(feat)
        
        features = torch.stack(features, dim=1)
        
        # LSTM时序建模
        lstm_out, _ = self.lstm(features)
        
        # 获取最后时刻的输出
        last_hidden = lstm_out[:, -1, :]
        
        # 预测多帧
        predictions = []
        flows = []
        
        current_hidden = last_hidden.unsqueeze(1)
        
        for _ in range(predict_frames):
            # 解码器生成帧
            feat_map = current_hidden.view(B, -1, 1, 1)
            pred_frame = self.decoder(feat_map)
            predictions.append(pred_frame)
            
            # 更新隐藏状态
            if len(predictions) > 1:
                # 使用光流更新
                prev_frame = predictions[-2]
                flow = self.flow_estimator(prev_frame, pred_frame)
                flows.append(flow)
                
                # 光流特征
                flow_feat = torch.mean(flow, dim=[2, 3])
                current_hidden = torch.cat([current_hidden[:, -1], flow_feat], dim=1).unsqueeze(1)
        
        predictions = torch.stack(predictions, dim=1)
        
        return {
            "predictions": predictions,
            "flows": flows,
            "features": features,
        }
    
    def compute_loss(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算损失.
        
        Args:
            input_frames: 输入帧 [B, T_in, 3, H, W]
            target_frames: 目标帧 [B, T_out, 3, H, W]
        
        Returns:
            loss: 总损失
            reconstruction_loss: 重建损失
            temporal_loss: 时序一致性损失
        """
        pred_result = self.forward(input_frames, predict_frames=target_frames.shape[1])
        predictions = pred_result["predictions"]
        
        # 重建损失
        reconstruction_loss = F.mse_loss(predictions, target_frames)
        
        # 时序一致性损失
        temporal_loss = self.temporal_loss(predictions, self.flow_estimator)
        
        # 总损失
        total_loss = reconstruction_loss + temporal_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "temporal_loss": temporal_loss,
        }


class FrameInterpolator(nn.Module):
    """帧插值器 - 生成中间帧."""
    
    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        self.flow_estimator = OpticalFlowEstimator()
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """生成插值帧.
        
        Args:
            frame1: 前一帧 [B, 3, H, W]
            frame2: 后一帧 [B, 3, H, W]
            timestamps: 插值时间点 [T] (0=frame1, 1=frame2)
        
        Returns:
            interpolated_frames: 插值帧 [B, T, 3, H, W]
        """
        B, C, H, W = frame1.shape
        
        if timestamps is None:
            timestamps = torch.linspace(0.1, 0.9, steps=5, device=frame1.device)
        
        T = len(timestamps)
        
        # 估计光流
        flow = self.flow_estimator(frame1, frame2)
        
        interpolated_frames = []
        
        for t in timestamps:
            # 线性插值光流
            flow_t = flow * t
            
            # 扭曲两帧
            warped_frame1 = self.flow_estimator.warp_frame(frame1, flow_t)
            warped_frame2 = self.flow_estimator.warp_frame(frame2, -flow * (1 - t))
            
            # 输入到网络
            input = torch.cat([warped_frame1, warped_frame2], dim=1)
            pred_frame = self.conv_layers(input)
            
            # 混合
            interpolated_frame = (1 - t) * warped_frame1 + t * warped_frame2 + pred_frame
            
            interpolated_frames.append(interpolated_frame)
        
        interpolated_frames = torch.stack(interpolated_frames, dim=1)
        
        return {
            "interpolated_frames": interpolated_frames,
            "flow": flow,
            "timestamps": timestamps,
        }


class MotionCompensatedPredictor(nn.Module):
    """运动补偿预测器 - 基于光流进行运动补偿."""
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_frames: int = 10,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 光流估计器
        self.flow_estimator = OpticalFlowEstimator()
        
        # 运动补偿模块
        self.motion_compensator = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        # 残差预测器
        self.residual_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """基于运动补偿预测下一帧.
        
        Args:
            frames: 输入帧序列 [B, T, 3, H, W]
        
        Returns:
            next_frame: 预测的下一帧 [B, 3, H, W]
            flow: 光流场 [B, 2, H, W]
            warped_frame: 运动补偿后的帧 [B, 3, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # 获取最后两帧
        frame_prev = frames[:, T - 2]
        frame_last = frames[:, T - 1]
        
        # 估计光流
        flow = self.flow_estimator(frame_prev, frame_last)
        
        # 运动补偿 - 基于光流预测下一帧
        warped_frame = self.flow_estimator.warp_frame(frame_last, flow)
        
        # 提取特征
        feat_prev = self.feature_encoder(frame_prev)
        feat_last = self.feature_encoder(frame_last)
        
        # 光流上采样到特征尺寸
        flow_feat = F.interpolate(flow, size=feat_last.shape[-2:], mode="bilinear")
        
        # 运动补偿预测
        mc_input = torch.cat([feat_last, flow_feat], dim=1)
        mc_prediction = self.motion_compensator(mc_input)
        mc_prediction = F.interpolate(mc_prediction, size=(H, W), mode="bilinear")
        
        # 残差预测
        residual_input = torch.cat([feat_prev, feat_last], dim=1)
        residual = self.residual_predictor(residual_input)
        residual = F.interpolate(residual, size=(H, W), mode="bilinear")
        
        # 最终预测
        next_frame = warped_frame + mc_prediction + residual
        next_frame = torch.clamp(next_frame, 0, 1)
        
        return {
            "next_frame": next_frame,
            "flow": flow,
            "warped_frame": warped_frame,
            "mc_prediction": mc_prediction,
            "residual": residual,
        }


class VideoPredictionPipeline:
    """视频预测管道 - 整合所有视频预测组件."""
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_frames: int = 10,
        num_predicted_frames: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        
        # 视频预测模型
        self.prediction_model = VideoPredictionModel(
            hidden_channels=hidden_channels,
            num_frames=num_frames,
            num_predicted_frames=num_predicted_frames,
        ).to(self.device)
        
        # 帧插值器
        self.frame_interpolator = FrameInterpolator(hidden_channels=hidden_channels).to(self.device)
        
        # 运动补偿预测器
        self.motion_compensator = MotionCompensatedPredictor(
            hidden_channels=hidden_channels,
            num_frames=num_frames,
        ).to(self.device)
        
        # 光流估计器
        self.flow_estimator = OpticalFlowEstimator().to(self.device)
        
        self.num_frames = num_frames
        self.num_predicted_frames = num_predicted_frames
    
    def predict_future(
        self,
        frames: torch.Tensor,
        num_frames: int = None,
        method: str = "motion_compensated",
    ) -> Dict[str, torch.Tensor]:
        """预测未来帧.
        
        Args:
            frames: 输入帧序列 [B, T, 3, H, W]
            num_frames: 预测帧数
            method: 预测方法 ('motion_compensated' | 'direct')
        
        Returns:
            predictions: 预测帧序列 [B, num_frames, 3, H, W]
            flows: 光流场列表
        """
        num_frames = num_frames or self.num_predicted_frames
        
        frames = frames.to(self.device)
        
        if method == "motion_compensated":
            return self._predict_motion_compensated(frames, num_frames)
        else:
            return self._predict_direct(frames, num_frames)
    
    def _predict_direct(self, frames: torch.Tensor, num_frames: int) -> Dict[str, torch.Tensor]:
        """直接预测."""
        result = self.prediction_model(frames, predict_frames=num_frames)
        
        return {
            "predictions": result["predictions"],
            "flows": result["flows"],
        }
    
    def _predict_motion_compensated(self, frames: torch.Tensor, num_frames: int) -> Dict[str, torch.Tensor]:
        """运动补偿预测."""
        B, T, C, H, W = frames.shape
        
        predictions = []
        flows = []
        
        current_frames = frames
        
        for _ in range(num_frames):
            result = self.motion_compensator(current_frames)
            
            next_frame = result["next_frame"]
            predictions.append(next_frame)
            flows.append(result["flow"])
            
            # 更新帧序列
            current_frames = torch.cat([current_frames[:, 1:], next_frame.unsqueeze(1)], dim=1)
        
        predictions = torch.stack(predictions, dim=1)
        
        return {
            "predictions": predictions,
            "flows": flows,
        }
    
    def interpolate_frames(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_interpolations: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """帧插值.
        
        Args:
            frame1: 前一帧 [B, 3, H, W]
            frame2: 后一帧 [B, 3, H, W]
            num_interpolations: 插值数量
        
        Returns:
            interpolated_frames: 插值帧 [B, num_interpolations, 3, H, W]
        """
        frame1 = frame1.to(self.device)
        frame2 = frame2.to(self.device)
        
        timestamps = torch.linspace(0.1, 0.9, steps=num_interpolations, device=self.device)
        
        result = self.frame_interpolator(frame1, frame2, timestamps)
        
        return {
            "interpolated_frames": result["interpolated_frames"],
            "flow": result["flow"],
            "timestamps": result["timestamps"],
        }
    
    def estimate_optical_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """估计光流.
        
        Args:
            frame1: 前一帧 [B, 3, H, W]
            frame2: 后一帧 [B, 3, H, W]
        
        Returns:
            flow: 光流场 [B, 2, H, W]
        """
        frame1 = frame1.to(self.device)
        frame2 = frame2.to(self.device)
        
        return self.flow_estimator(frame1, frame2)
    
    def train(
        self,
        dataloader: Any,
        epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """训练视频预测模型.
        
        Args:
            dataloader: 数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
        
        Returns:
            training_history: 训练历史
        """
        optimizer = torch.optim.Adam(
            self.prediction_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        training_history = {
            "loss": [],
            "reconstruction_loss": [],
            "temporal_loss": [],
        }
        
        self.prediction_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_temp_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                input_frames, target_frames = batch
                
                input_frames = input_frames.to(self.device)
                target_frames = target_frames.to(self.device)
                
                optimizer.zero_grad()
                
                loss_dict = self.prediction_model.compute_loss(input_frames, target_frames)
                
                loss_dict["loss"].backward()
                optimizer.step()
                
                epoch_loss += loss_dict["loss"].item()
                epoch_recon_loss += loss_dict["reconstruction_loss"].item()
                epoch_temp_loss += loss_dict["temporal_loss"].item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_temp_loss = epoch_temp_loss / num_batches
            
            training_history["loss"].append(avg_loss)
            training_history["reconstruction_loss"].append(avg_recon_loss)
            training_history["temporal_loss"].append(avg_temp_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Reconstruction Loss: {avg_recon_loss:.6f}")
                print(f"  Temporal Loss: {avg_temp_loss:.6f}")
        
        return training_history
    
    def save(self, filepath: str) -> bool:
        """保存模型."""
        try:
            torch.save({
                "prediction_model": self.prediction_model.state_dict(),
                "frame_interpolator": self.frame_interpolator.state_dict(),
                "motion_compensator": self.motion_compensator.state_dict(),
                "flow_estimator": self.flow_estimator.state_dict(),
            }, filepath)
            return True
        except Exception:
            return False
    
    def load(self, filepath: str) -> bool:
        """加载模型."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.prediction_model.load_state_dict(checkpoint["prediction_model"])
            self.frame_interpolator.load_state_dict(checkpoint["frame_interpolator"])
            self.motion_compensator.load_state_dict(checkpoint["motion_compensator"])
            self.flow_estimator.load_state_dict(checkpoint["flow_estimator"])
            
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("VideoPrediction 测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建测试数据
    B, T, C, H, W = 2, 10, 3, 64, 64
    test_frames = torch.rand(B, T, C, H, W).to(device)
    
    # 测试光流估计器
    print("\n--- 光流估计器测试 ---")
    flow_estimator = OpticalFlowEstimator().to(device)
    frame1 = test_frames[:, 0]
    frame2 = test_frames[:, 1]
    
    flow = flow_estimator(frame1, frame2)
    print(f"输入帧: {frame1.shape}")
    print(f"光流场: {flow.shape}")
    
    warped = flow_estimator.warp_frame(frame1, flow)
    print(f"扭曲后: {warped.shape}")
    
    flow_mag = flow_estimator.compute_flow_magnitude(flow)
    print(f"光流幅度: {flow_mag.shape}")
    
    # 测试帧插值器
    print("\n--- 帧插值器测试 ---")
    interpolator = FrameInterpolator().to(device)
    
    interp_result = interpolator(frame1, frame2, timestamps=torch.tensor([0.5]))
    print(f"插值帧: {interp_result['interpolated_frames'].shape}")
    print(f"光流: {interp_result['flow'].shape}")
    
    # 测试运动补偿预测器
    print("\n--- 运动补偿预测器测试 ---")
    motion_compensator = MotionCompensatedPredictor(num_frames=T).to(device)
    
    mc_result = motion_compensator(test_frames)
    print(f"下一帧预测: {mc_result['next_frame'].shape}")
    print(f"光流: {mc_result['flow'].shape}")
    print(f"扭曲帧: {mc_result['warped_frame'].shape}")
    
    # 测试视频预测模型
    print("\n--- 视频预测模型测试 ---")
    prediction_model = VideoPredictionModel(
        num_frames=T,
        num_predicted_frames=5,
    ).to(device)
    
    pred_result = prediction_model(test_frames, predict_frames=5)
    print(f"预测帧: {pred_result['predictions'].shape}")
    print(f"光流数量: {len(pred_result['flows'])}")
    print(f"特征: {pred_result['features'].shape}")
    
    # 测试损失计算
    target_frames = torch.rand(B, 5, C, H, W).to(device)
    loss_dict = prediction_model.compute_loss(test_frames, target_frames)
    print(f"总损失: {loss_dict['loss'].item():.6f}")
    print(f"重建损失: {loss_dict['reconstruction_loss'].item():.6f}")
    print(f"时序损失: {loss_dict['temporal_loss'].item():.6f}")
    
    # 测试视频预测管道
    print("\n--- 视频预测管道测试 ---")
    pipeline = VideoPredictionPipeline(
        num_frames=T,
        num_predicted_frames=5,
        device=device.type,
    )
    
    # 直接预测
    direct_result = pipeline.predict_future(test_frames, method="direct")
    print(f"直接预测帧: {direct_result['predictions'].shape}")
    
    # 运动补偿预测
    mc_result = pipeline.predict_future(test_frames, method="motion_compensated")
    print(f"运动补偿预测帧: {mc_result['predictions'].shape}")
    
    # 帧插值
    interp_result = pipeline.interpolate_frames(frame1, frame2, num_interpolations=3)
    print(f"插值帧: {interp_result['interpolated_frames'].shape}")
    
    # 光流估计
    flow = pipeline.estimate_optical_flow(frame1, frame2)
    print(f"光流场: {flow.shape}")
    
    # 模型保存/加载
    print("\n--- 模型保存/加载测试 ---")
    save_path = "./test_video_prediction.pth"
    pipeline.save(save_path)
    print(f"模型已保存到: {save_path}")
    
    new_pipeline = VideoPredictionPipeline(device=device.type)
    new_pipeline.load(save_path)
    print("模型加载成功!")
    
    # 清理测试文件
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n" + "=" * 60)
    print("视频预测模块测试完成!")
    print("=" * 60)
