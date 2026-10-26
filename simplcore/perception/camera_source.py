# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""CameraSource - 摄像头信号捕获模块.

从物理摄像头捕获原始电信号, 预处理为 ViT 接受的张量格式.

信号流:
  摄像头传感器 (CMOS/CCD) → 原始电信号 → 模数转换 → 
  OpenCV 帧 → 预处理 (去噪/归一化/尺寸) → Torch Tensor

设计原则:
  1. 非阻塞捕获: 后台线程持续拉流, 主线程随时取最新帧
  2. 断线重连: 摄像头断开后自动恢复
  3. 零拷贝: 尽可能复用内存缓冲区
  4. 纯 Python: 仅依赖 opencv-python + torch

使用:
  >>> source = CameraSource(camera_id=0, fps=30, img_size=224)
  >>> source.start()
  >>> frame_tensor = source.read()  # (1, 3, 224, 224) normalized
  >>> source.stop()
"""

from __future__ import annotations

import threading
import time
import math
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F


class FrameQualityAnalyzer:
    """帧质量分析器 - 检测模糊、过曝、欠曝、噪点."""

    def __init__(self, blur_threshold: float = 30.0, exposure_min: float = 10.0, exposure_max: float = 245.0):
        self.blur_threshold = blur_threshold
        self.exposure_min = exposure_min
        self.exposure_max = exposure_max

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """分析帧质量.
        
        Returns:
            dict: 包含质量指标.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        blur_score = self._compute_blur(gray)
        exposure_score = self._compute_exposure(gray)
        noise_score = self._compute_noise(gray)
        
        quality = self._compute_quality_score(blur_score, exposure_score, noise_score)
        
        return {
            "quality": quality,
            "blur_score": blur_score,
            "blur_status": "clear" if blur_score > self.blur_threshold else "blurry",
            "exposure_score": exposure_score,
            "exposure_status": self._classify_exposure(exposure_score),
            "noise_score": noise_score,
            "noise_status": "low" if noise_score < 10 else "high",
        }

    def _compute_blur(self, gray: np.ndarray) -> float:
        """使用拉普拉斯算子计算模糊度."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def _compute_exposure(self, gray: np.ndarray) -> float:
        """计算曝光值 (平均亮度)."""
        return np.mean(gray)

    def _compute_noise(self, gray: np.ndarray) -> float:
        """计算噪点估计."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.std()

    def _classify_exposure(self, score: float) -> str:
        """分类曝光状态."""
        if score < self.exposure_min:
            return "underexposed"
        elif score > self.exposure_max:
            return "overexposed"
        else:
            return "good"

    def _compute_quality_score(self, blur: float, exposure: float, noise: float) -> float:
        """综合质量评分 (0-1)."""
        blur_factor = min(1.0, blur / self.blur_threshold)
        exposure_factor = 1.0 - abs(exposure - 128) / 128
        noise_factor = max(0.0, 1.0 - noise / 50)
        
        return (blur_factor * 0.4 + exposure_factor * 0.4 + noise_factor * 0.2)


class CameraSource:
    """摄像头信号源.

    直接从摄像头捕获帧, 预处理为 ViT 可接受的张量.

    Args:
        camera_id: 摄像头设备 ID (0=内置, 1=外接).
        img_size: 输出图像尺寸 (img_size × img_size).
        fps: 目标捕获帧率.
        buffer_size: 帧缓冲区大小 (环形队列).
        normalize: 是否归一化到 [0, 1].
        auto_start: 创建时自动启动.

    Attributes:
        is_running: 捕获线程是否在运行.
        frame_count: 已捕获帧数.
        fps_actual: 实际帧率.
    """

    def __init__(
        self,
        camera_id: int = 0,
        img_size: int = 224,
        fps: int = 30,
        buffer_size: int = 1,
        normalize: bool = True,
        auto_start: bool = False,
        quality_enabled: bool = True,
        auto_exposure: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 60,
    ) -> None:
        if not CV2_AVAILABLE:
            raise ImportError(
                "需要安装 opencv-python: pip install opencv-python\n"
                "摄像头信号捕获依赖 OpenCV 的 VideoCapture 后端."
            )

        self.camera_id = camera_id
        self.img_size = img_size
        self.target_fps = fps
        self.buffer_size = max(1, buffer_size)
        self.normalize = normalize
        self.quality_enabled = quality_enabled
        self.auto_exposure = auto_exposure
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        # 帧质量分析器
        self.quality_analyzer = FrameQualityAnalyzer()
        self._latest_quality: Optional[Dict[str, Any]] = None

        # 曝光控制
        self._exposure_level = 0.5
        self._brightness_level = 0.0

        # 内部状态
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_buffer: list[np.ndarray] = []
        self._latest_tensor: Optional[torch.Tensor] = None

        # 重连状态
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0

        self.frame_count = 0
        self.fps_actual = 0.0
        self._last_time = 0.0
        self._frame_times: list[float] = []

        # 质量统计
        self._quality_history: list[float] = []
        self._avg_quality = 0.0

        if auto_start:
            self.start()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动摄像头捕获线程."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 #{self.camera_id}.\n"
                "请检查:\n"
                "  1. 摄像头是否已连接\n"
                "  2. 设备 ID 是否正确\n"
                "  3. 是否有其他程序占用摄像头"
            )

        # 设置分辨率 (摄像头可能不支持精确值, 会选最接近的)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # 获取实际参数
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"Camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        self._last_time = time.time()

        print(
            f"[CameraSource] 摄像头 #{self.camera_id} 已启动: "
            f"{actual_w}×{actual_h} @ {actual_fps:.1f}fps"
        )

    def stop(self) -> None:
        """停止捕获线程, 释放摄像头."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_buffer.clear()
        print(f"[CameraSource] 摄像头 #{self.camera_id} 已释放.")

    # ------------------------------------------------------------------
    # 读取帧
    # ------------------------------------------------------------------
    def read(self, as_tensor: bool = True) -> Optional[torch.Tensor]:
        """读取最新帧.

        Args:
            as_tensor: True 返回 (1, 3, H, W) 归一化 Tensor,
                       False 返回原始 OpenCV 帧 (H, W, 3) BGR.

        Returns:
            帧张量或 None (如果摄像头无信号).
        """
        with self._lock:
            if not self._frame_buffer:
                return None
            frame = self._frame_buffer[-1]  # 取最新帧

        if not as_tensor:
            return frame

        return self._frame_to_tensor(frame)

    def read_batch(self, n: int = 4, as_tensor: bool = True) -> Optional[torch.Tensor]:
        """读取最近 n 帧 (用于时序处理).

        Args:
            n: 帧数.
            as_tensor: True 返回 (1, n, 3, H, W) Tensor.

        Returns:
            帧 batch 或 None (帧数不足).
        """
        with self._lock:
            if len(self._frame_buffer) < n:
                return None
            frames = self._frame_buffer[-n:]

        if not as_tensor:
            return np.stack(frames) if frames else None

        tensors = [self._frame_to_tensor(f) for f in frames]
        return torch.stack(tensors, dim=1)  # (1, n, 3, H, W)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """后台捕获线程主循环."""

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                # 智能断线重连 - 指数退避
                self._perform_reconnection()
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # 检测到连接问题，触发重连
                self._mark_disconnected()
                time.sleep(0.01)
                continue

            # 重置重连计数器
            self._reconnect_attempts = 0

            # 帧质量分析
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.quality_enabled:
                quality = self.quality_analyzer.analyze(frame_rgb)
                with self._lock:
                    self._latest_quality = quality
                    self._quality_history.append(quality["quality"])
                    if len(self._quality_history) > 100:
                        self._quality_history.pop(0)
                    self._avg_quality = sum(self._quality_history) / max(1, len(self._quality_history))

                # 自动曝光调整
                if self.auto_exposure and self._cap is not None:
                    self._adjust_exposure(quality)

            # 帧率控制
            now = time.time()
            dt = now - self._last_time
            min_dt = 1.0 / self.target_fps
            if dt < min_dt:
                time.sleep(min_dt - dt)

            # 缓冲
            with self._lock:
                self._frame_buffer.append(frame_rgb)
                if len(self._frame_buffer) > self.buffer_size:
                    self._frame_buffer.pop(0)

            self.frame_count += 1
            self._last_time = time.time()

            # 实际帧率统计 (每30帧更新一次)
            self._frame_times.append(now)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            if self.frame_count % 30 == 0 and len(self._frame_times) > 1:
                self.fps_actual = len(self._frame_times) / (
                    self._frame_times[-1] - self._frame_times[0]
                )

    def _perform_reconnection(self) -> None:
        """执行智能重连 - 指数退避策略."""
        now = time.time()
        if now - self._last_reconnect_time < self.reconnect_interval:
            return

        self._reconnect_attempts += 1
        self._last_reconnect_time = now

        if self._reconnect_attempts > self.max_reconnect_attempts:
            print(f"[CameraSource] 摄像头 #{self.camera_id} 重连失败超过 {self.max_reconnect_attempts} 次")
            return

        # 指数退避延迟
        delay = self.reconnect_interval * (2 ** min(self._reconnect_attempts - 1, 10))
        
        print(f"[CameraSource] 尝试重连摄像头 #{self.camera_id} (第 {self._reconnect_attempts} 次, 延迟 {delay:.1f}s)...")
        
        time.sleep(delay)
        self._cap = cv2.VideoCapture(self.camera_id)
        
        if self._cap.isOpened():
            print(f"[CameraSource] 摄像头 #{self.camera_id} 重连成功!")
            self._reconnect_attempts = 0
            # 重新设置参数
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _mark_disconnected(self) -> None:
        """标记连接断开状态."""
        if self._reconnect_attempts == 0:
            print(f"[CameraSource] 摄像头 #{self.camera_id} 连接断开")

    def _adjust_exposure(self, quality: Dict[str, Any]) -> None:
        """根据帧质量自动调整曝光."""
        exposure_status = quality["exposure_status"]
        exposure_score = quality["exposure_score"]

        if exposure_status == "underexposed":
            self._exposure_level = min(1.0, self._exposure_level + 0.05)
            self._brightness_level = min(1.0, self._brightness_level + 0.02)
        elif exposure_status == "overexposed":
            self._exposure_level = max(0.0, self._exposure_level - 0.05)
            self._brightness_level = max(-1.0, self._brightness_level - 0.02)

        # 应用调整 (如果摄像头支持)
        try:
            if self._cap is not None:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_level * 100)
                self._cap.set(cv2.CAP_PROP_BRIGHTNESS, self._brightness_level * 50)
        except Exception:
            pass

    def get_quality(self) -> Optional[Dict[str, Any]]:
        """获取最新帧质量信息."""
        with self._lock:
            return self._latest_quality

    def get_avg_quality(self) -> float:
        """获取平均帧质量."""
        return self._avg_quality

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """将 OpenCV 帧转换为归一化 Tensor.

        Args:
            frame: (H, W, 3) RGB uint8 [0-255].

        Returns:
            (1, 3, img_size, img_size) float32 [0-1 或 0-255].
        """
        # 缩放
        if frame.shape[0] != self.img_size or frame.shape[1] != self.img_size:
            frame = cv2.resize(frame, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)

        # 转换为 Tensor
        tensor = torch.from_numpy(frame).float()  # (H, W, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        if self.normalize:
            tensor = tensor / 255.0  # [0-255] → [0-1]

        return tensor.contiguous()

    def __enter__(self) -> CameraSource:
        self.start()
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

    print("=== CameraSource 自检 ===")
    if not CV2_AVAILABLE:
        print("[SKIP] 需要 opencv-python: pip install opencv-python")
        sys.exit(0)

    try:
        # 尝试打开摄像头
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            print("[SKIP] 未检测到摄像头.")
            test_cap.release()
            sys.exit(0)
        test_cap.release()
    except Exception:
        print("[SKIP] 摄像头不可用.")
        sys.exit(0)

    source = CameraSource(camera_id=0, img_size=224, fps=30, normalize=True)
    source.start()
    time.sleep(1)

    print(f"  分辨率: {source.resolution}")
    print(f"  帧数: {source.frame_count}")
    print(f"  实际帧率: {source.fps_actual:.1f}")

    # 测试单帧
    tensor = source.read(as_tensor=True)
    print(f"  单帧: {tensor.shape} [{tensor.min():.3f}, {tensor.max():.3f}]")

    # 测试 batch
    batch = source.read_batch(n=4)
    if batch is not None:
        print(f"  Batch: {batch.shape}")

    source.stop()
    print("[PASS] CameraSource 正常.")

