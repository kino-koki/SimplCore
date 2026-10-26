# -*- coding: utf-8 -*-
"""VisualStream - 实时视觉流处理引擎.

从 CameraSource 接收帧, 经过 ViT 编码 + 时序聚合, 输出潜状态序列.

处理管道:
  摄像头帧 → CameraSource → 帧缓冲 → ViT 编码 → 时序窗口 → 潜状态

设计:
  - ViT 编码在独立线程/进程中进行, 不阻塞捕获.
  - 时序窗口: 保持最近 N 帧的潜状态, 支持滑动窗口推理.
  - 帧率自适应: 根据实际帧率自动调整处理策略.

使用:
  >>> from SimplCore.perception.camera_source import CameraSource
  >>> source = CameraSource(0, img_size=224)
  >>> stream = VisualStream(source, vit_encoder)
  >>> stream.start()
  >>> latent = stream.read_latent()  # (1, L, H) 当前潜状态
  >>> stream.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from SimplCore.perception.camera_source import CameraSource


class VisualStream:
    """视觉流处理器.

    Args:
        camera_source: CameraSource 实例.
        vit_encoder: SimplCoreViTEncoder 实例 (或兼容的 encode_image 方法).
        window_size: 时序窗口大小 (帧数).
        stride: 滑动步长.
        max_latent_len: 输出潜状态最大序列长度.
        auto_start: 创建时自动启动.

    Attributes:
        is_running: 处理线程是否在运行.
        fps_actual: 实际处理帧率.
        latency: 当前处理延迟 (ms).
    """

    def __init__(
        self,
        camera_source: CameraSource,
        vit_encoder: nn.Module,
        window_size: int = 1,
        stride: int = 1,
        max_latent_len: int = 512,
        auto_start: bool = False,
        use_temporal_attention: bool = True,
        quality_threshold: float = 0.3,
    ) -> None:
        self.camera = camera_source
        self.vit = vit_encoder
        self.window_size = max(1, window_size)
        self.stride = max(1, stride)
        self.max_latent_len = max_latent_len
        self.use_temporal_attention = use_temporal_attention
        self.quality_threshold = quality_threshold

        # 时序注意力模块
        if use_temporal_attention and hasattr(vit_encoder, 'output_dim'):
            self.temporal_attention = TemporalAttention(vit_encoder.output_dim)
        else:
            self.temporal_attention = None

        # 质量过滤器
        self.quality_filter = QualityFilter(quality_threshold)

        # 处理线程
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latent_buffer: list[torch.Tensor] = []
        self._quality_buffer: list[Dict[str, Any]] = []
        self._latest_latent: Optional[torch.Tensor] = None

        # 统计
        self.frame_count = 0
        self.fps_actual = 0.0
        self.latency = 0.0
        self._times: list[float] = []
        
        # 质量统计
        self._dropped_frames = 0
        self._total_processed = 0

        if auto_start:
            self.start()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动流处理线程."""
        if self._running:
            return
        if not self.camera.is_running:
            self.camera.start()

        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            name="VisualStream",
            daemon=True,
        )
        self._thread.start()
        print(f"[VisualStream] 流处理已启动 (窗口={self.window_size}, 步长={self.stride})")

    def stop(self) -> None:
        """停止流处理."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._latent_buffer.clear()
        print("[VisualStream] 流处理已停止.")

    # ------------------------------------------------------------------
    # 读取潜状态
    # ------------------------------------------------------------------
    def read_latent(self) -> Optional[torch.Tensor]:
        """读取当前潜状态.

        Returns:
            (B, L, H) 潜状态, 或 None (尚未准备好).
        """
        with self._lock:
            return self._latest_latent

    def read_latent_sequence(self) -> Optional[torch.Tensor]:
        """读取时序潜状态序列 (用于 Rollout 推演).

        Returns:
            (B, T, L, H) 时序潜状态, 或 None.
        """
        with self._lock:
            if len(self._latent_buffer) < 2:
                return None
            return torch.stack(list(self._latent_buffer), dim=1)

    # ------------------------------------------------------------------
    # 内部: 处理循环
    # ------------------------------------------------------------------
    def _process_loop(self) -> None:
        """后台处理线程主循环.

        流程:
          1. 从 CameraSource 读取帧
          2. 获取帧质量信息
          3. 质量过滤
          4. 经过 ViT 编码为潜状态
          5. 时序聚合 (时序注意力或平均池化)
          6. 更新潜状态缓冲区
        """
        while self._running:
            t_start = time.time()

            # 读取一帧
            frame = self.camera.read(as_tensor=True)
            if frame is None:
                time.sleep(0.005)
                continue

            # 获取帧质量
            quality = self.camera.get_quality()
            
            # 质量过滤
            if not self.quality_filter.should_use(quality):
                self._dropped_frames += 1
                time.sleep(0.005)
                continue

            # ViT 编码
            try:
                with torch.no_grad():
                    latent = self.vit.encode_image(frame)  # (1, L, H)
            except Exception as e:
                time.sleep(0.01)
                continue

            # 获取帧权重
            frame_weight = self.quality_filter.get_weight(quality)

            with self._lock:
                # 缓冲
                self._latent_buffer.append(latent * frame_weight)
                self._quality_buffer.append(quality)
                
                if len(self._latent_buffer) > self.window_size:
                    self._latent_buffer.pop(0)
                    self._quality_buffer.pop(0)

                # 时序聚合
                if len(self._latent_buffer) >= self.window_size:
                    if self.window_size == 1:
                        self._latest_latent = latent
                    else:
                        stacked = torch.stack(
                            list(self._latent_buffer), dim=1
                        )  # (1, T, L, H)
                        
                        if self.use_temporal_attention and self.temporal_attention is not None:
                            # 时序注意力聚合
                            aggregated = self.temporal_attention(stacked)
                        else:
                            # 滑动窗口平均池化
                            aggregated = stacked.mean(dim=1)  # (1, L, H)
                            
                        self._latest_latent = aggregated

            # 统计
            t_end = time.time()
            self.latency = (t_end - t_start) * 1000  # ms
            self.frame_count += 1
            self._total_processed += 1

            self._times.append(t_end)
            if len(self._times) > 30:
                self._times.pop(0)
            if self.frame_count % 30 == 0 and len(self._times) > 1:
                self.fps_actual = len(self._times) / (
                    self._times[-1] - self._times[0]
                )

            # 帧率自适应控制
            min_dt = 1.0 / max(1.0, getattr(self.camera, 'target_fps', 30))
            elapsed = time.time() - t_start
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息."""
        return {
            "frame_count": self.frame_count,
            "processed_frames": self._total_processed,
            "dropped_frames": self._dropped_frames,
            "fps": self.fps_actual,
            "latency_ms": self.latency,
            "avg_quality": self.camera.get_avg_quality(),
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> VisualStream:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


# ----------------------------------------------------------------------
# 自检 (需要摄像头 + ViT)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== VisualStream 自检 ===")

    # 尝试导入摄像头
    try:
        from SimplCore.architecture.vision_encoder import vit_tiny
    except ImportError as e:
        print(f"[SKIP] ViT 编码器不可用: {e}")
        sys.exit(0)

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

    enc = vit_tiny(output_dim=64)
    source = CameraSource(0, img_size=224, fps=15, buffer_size=4)
    stream = VisualStream(source, enc, window_size=1)

    stream.start()
    time.sleep(1)

    latent = stream.read_latent()
    if latent is not None:
        print(f"  潜状态: {latent.shape}")
        print(f"  处理帧数: {stream.frame_count}")
        print(f"  延迟: {stream.latency:.1f}ms")

    # 时序潜状态
    seq = stream.read_latent_sequence()
    if seq is not None:
        print(f"  时序潜状态: {seq.shape}")

    stream.stop()
    print("[PASS] VisualStream 正常.")

