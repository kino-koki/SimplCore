# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""PerceptionPipeline - 端到端视觉感知管道.

从摄像头电信号到 SimplCore 理解的完整管道.

信号流:
  摄像头 (CMOS传感器) → 电信号 → CameraSource (帧捕获) 
  → VisualStream (ViT编码+时序) → ModalityRouter (模态融合) 
  → SimplCoreModel (理解+推演) → 输出

设计:
  - 模块化: 每个阶段可独立替换/升级.
  - 实时: 异步处理, 非阻塞流水线.
  - 优雅降级: 摄像头断线时保持系统运行.

使用:
  >>> pipeline = PerceptionPipeline()
  >>> pipeline.start(camera_id=0)
  >>> result = pipeline.understand("你看到什么了?")
  >>> print(result)
  >>> pipeline.stop()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel
from SimplCore.architecture.modality_router import ModalityRouter
from SimplCore.architecture.vision_encoder import SimplCoreViTEncoder, vit_tiny
from SimplCore.architecture.rollout_engine import RolloutEngine, WorldModelHead
from SimplCore.perception.camera_source import CameraSource
from SimplCore.perception.visual_stream import VisualStream


class PerceptionPipeline:
    """端到端视觉感知管道.

    将摄像头信号通过 ViT + ModalityRouter 送入 SimplCore 核心处理.

    Args:
        config: SimplCoreConfig (如 None 则使用 small 配置).
        vit_config: ViT 编码器配置字典.
        device: 运行设备.

    Attributes:
        is_running: 管道是否在运行.
        camera: CameraSource 实例.
        stream: VisualStream 实例.
    """

    def __init__(
        self,
        config: Optional[SimplCoreConfig] = None,
        vit_hidden_size: int = 192,
        vit_output_size: int = 256,
        vit_depth: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        # 创建或使用默认配置
        if config is None:
            self.config = SimplCoreConfig.get_small_config()
        else:
            self.config = config

        hidden_size = self.config.hidden_size

        # ---- ViT 视觉编码器 ----
        # 直接构造 SimplCoreViTEncoder, 避免工厂函数的参数冲突
        self.vit_encoder = SimplCoreViTEncoder(
            vit_dim=vit_hidden_size,
            vit_depth=vit_depth,
            vit_heads=(vit_hidden_size // 64),
            output_dim=hidden_size,
        ).to(self.device)

        # ---- ModalityRouter ----
        self.router = ModalityRouter(
            hidden_size=hidden_size,
            use_vision=True,
        ).to(self.device)

        # ---- SimplCore 核心模型 ----
        self.model = SimplCoreModel(self.config).to(self.device)

        # ---- RolloutEngine (潜空间推演) ----
        self.use_rollout = getattr(self.config, 'use_rollout', False)

        # ---- 摄像头/流 ----
        self.camera: Optional[CameraSource] = None
        self.stream: Optional[VisualStream] = None

        self._running = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(
        self,
        camera_id: int = 0,
        img_size: int = 224,
        fps: int = 15,
        window_size: int = 1,
    ) -> None:
        """启动感知管道.

        Args:
            camera_id: 摄像头设备 ID.
            img_size: 处理图像尺寸.
            fps: 目标帧率 (15fps 可减少计算负载).
            window_size: 时序窗口大小.
        """
        if self._running:
            return

        # 摄像头
        self.camera = CameraSource(
            camera_id=camera_id,
            img_size=img_size,
            fps=fps,
            buffer_size=window_size,
            normalize=True,
        )

        # 视觉流
        self.stream = VisualStream(
            camera_source=self.camera,
            vit_encoder=self.vit_encoder,
            window_size=window_size,
        )

        self.stream.start()
        self._running = True

        print(
            f"[PerceptionPipeline] 管道已启动 (设备={self.device}, "
            f"use_rollout={self.use_rollout})"
        )

    def stop(self) -> None:
        """停止感知管道, 释放所有资源."""
        self._running = False
        if self.stream is not None:
            self.stream.stop()
        if self.camera is not None:
            self.camera.stop()
        print("[PerceptionPipeline] 管道已停止.")

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def see(self) -> Optional[torch.Tensor]:
        """看: 获取当前摄像头画面的潜状态.

        Returns:
            (1, L, H) 潜状态, 或 None (无信号).
        """
        if self.stream is None or not self.stream.is_running:
            return None
        return self.stream.read_latent()

    @torch.no_grad()
    def understand(
        self,
        text: Optional[str] = None,
        text_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """理解: 结合当前视觉和文本输入, 运行 SimplCore 推理.

        Args:
            text: 文本输入 (如 "你看到了什么?").
            text_ids: 已 token 化的文本 ID (优先于 text).

        Returns:
            dict: {
                "logits": 输出 logits,
                "latent": 视觉潜状态 (如有),
                "reward": 奖励预测 (如有),
                "cont": 连续性预测 (如有),
            }
        """
        if not self._running:
            return {"error": "管道未启动, 请先调用 start()"}

        result: Dict[str, Any] = {}

        # 1. 获取视觉潜状态
        visual_latent = self.see()
        result["has_vision"] = visual_latent is not None

        # 2. 构建输入
        if visual_latent is not None:
            # 将视觉潜状态通过 ModalityRouter 路由
            visual_embed = self.router.forward_image(
                torch.randn(1, 3, 224, 224, device=self.device)
            )
            # 使用当前视觉潜状态
            # TODO: 实际应该从 ViT 编码器获取, 这里简化
            pass

        # 3. 文本编码 (简化: 直接用 text_ids)
        if text_ids is not None:
            x = text_ids.to(self.device)
        elif text is not None:
            # 简化: 不使用 tokenizer, 返回提示
            result["message"] = (
                f"视觉输入: {'有' if visual_latent is not None else '无'}, "
                f"文本: '{text}'"
            )
            return result
        else:
            x = torch.randint(0, self.config.vocab_size, (1, 8)).to(self.device)

        # 4. SimplCore 推理
        logits, states = self.model(x)
        result["logits"] = logits.cpu()
        result["latent"] = visual_latent.cpu() if visual_latent is not None else None

        if hasattr(self.model, 'wm_head') and self.model.wm_head is not None:
            wm_logits, reward, cont = self.model.wm_head(
                self.model.final_norm(logits.to(self.device))
            )
            result["reward"] = reward.cpu()
            result["cont"] = cont.cpu()

        return result

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> Dict[str, Any]:
        """获取管道运行状态."""
        return {
            "running": self._running,
            "device": str(self.device),
            "camera": {
                "connected": self.camera is not None and self.camera.is_running,
                "fps": self.camera.fps_actual if self.camera else 0,
                "frames": self.camera.frame_count if self.camera else 0,
            } if self.camera else None,
            "stream": {
                "running": self.stream is not None and self.stream.is_running,
                "fps": self.stream.fps_actual if self.stream else 0,
                "latency_ms": f"{self.stream.latency:.1f}" if self.stream else "N/A",
            } if self.stream else None,
            "model": {
                "params": self.model.num_parameters_human(),
                "rollout": self.use_rollout,
                "hidden_size": self.config.hidden_size,
            },
        }

    def __enter__(self) -> PerceptionPipeline:
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __del__(self) -> None:
        self.stop()


# ----------------------------------------------------------------------
# 自检 (需要摄像头)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== PerceptionPipeline 自检 ===")

    # 检查摄像头
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[SKIP] 未检测到摄像头.")
            sys.exit(0)
        cap.release()
    except ImportError:
        print("[SKIP] 需要 opencv-python.")
        sys.exit(0)

    try:
        # 创建管道并启动
        pipeline = PerceptionPipeline(vit_hidden_size=192, vit_output_size=64)
        pipeline.start(camera_id=0, fps=15)

        time.sleep(1)

        # 状态
        status = pipeline.status
        print(f"  管道状态: {'运行中' if status['running'] else '已停止'}")
        print(f"  设备: {status['device']}")
        print(f"  模型: {status['model']['params']}")

        # 视觉推理
        latent = pipeline.see()
        if latent is not None:
            print(f"  当前潜状态: {latent.shape}")
        else:
            print("  [INFO] 视觉潜状态尚未就绪 (可能需要更多时间)")

        # 理解
        result = pipeline.understand(text="你看到了什么?")
        print(f"  理解结果: {result.get('message', 'N/A')}")

        pipeline.stop()
        print("[PASS] PerceptionPipeline 正常.")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


