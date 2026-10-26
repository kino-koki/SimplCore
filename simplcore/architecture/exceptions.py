# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 架构自定义异常模块
============================

定义架构层所有自定义异常类, 以及梯度裁剪工具函数。

异常层级:
  SimplCoreError (基类)
    ├── ConfigError             配置错误
    ├── ModelError              模型运行时错误
    ├── MemoryStateError        DMC 记忆状态错误
    ├── RolloutError            推演引擎错误
    ├── GradientCheckpointError 梯度检查点错误
    ├── ModalityError           多模态输入错误
    ├── StackingError           参数堆叠错误
    └── LossGuardError          损失守卫阈值违规
"""

from __future__ import annotations

from typing import Optional

import simplcore.tensor as torch


# ======================================================================
# 基础异常
# ======================================================================

class SimplCoreError(Exception):
    """SimplCore 架构基础异常.

    所有 SimplCore 自定义异常的基类, 捕获此异常可处理任意架构层错误。

    Args:
        message: 错误描述信息.
        details: 可选的附加详情字典.
    """

    def __init__(self, message: str = "", details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# ======================================================================
# 配置异常
# ======================================================================

class ConfigError(SimplCoreError):
    """配置错误.

    当 SimplCoreConfig 的参数无效、缺失或类型不正确时抛出。

    常见场景:
      - hidden_size 不是正整数
      - num_layers 为零或负数
      - 未知预设名称 (如 build_model('999b'))
      - vocab_size 与词表不匹配
    """

    pass


# ======================================================================
# 模型运行时异常
# ======================================================================

class ModelError(SimplCoreError):
    """模型运行时错误.

    当模型在前向传播、反向传播或生成过程中发生意外情况时抛出。

    常见场景:
      - 输入张量维度与模型不匹配
      - 前向传播产生 NaN / Inf
      - 生成时温度为零或负数
      - 不支持的生成模式
    """

    pass


# ======================================================================
# 记忆状态异常
# ======================================================================

class MemoryStateError(SimplCoreError):
    """DMC (动态记忆通道) 记忆状态错误.

    当记忆库状态不合法或操作失败时抛出。

    常见场景:
      - 记忆库未初始化即尝试读取
      - 记忆状态批次大小与输入不匹配
      - 记忆 detach / reset 操作失败
      - 记忆溢出 (超过 channel_size 限制)
    """

    pass


# ======================================================================
# 推演引擎异常
# ======================================================================

class RolloutError(SimplCoreError):
    """潜空间推演 (Rollout) 引擎错误.

    当 RolloutEngine 在多步推演过程中发生错误时抛出。

    常见场景:
      - 推演步数为零或负数
      - 推演过程中状态发散 (NaN)
      - 推演引擎未初始化
      - 推演步数超过最大限制
    """

    pass


# ======================================================================
# 梯度检查点异常
# ======================================================================

class GradientCheckpointError(SimplCoreError):
    """梯度检查点错误.

    当梯度检查点机制失效或梯度异常时抛出。

    常见场景:
      - 梯度检查点段数与模型层数不匹配
      - 梯度检查点重计算时产生 NaN
      - 梯度爆炸 (范数超过安全阈值)
      - 梯度消失 (范数接近零)
    """

    pass


# ======================================================================
# 多模态异常
# ======================================================================

class ModalityError(SimplCoreError):
    """多模态输入错误.

    当多模态输入处理过程中发生错误时抛出。

    常见场景:
      - 未提供任何模态输入 (文本和图像均为 None)
      - 图像尺寸与视觉编码器不匹配
      - 不支持的模态类型
      - 模态融合维度不匹配
    """

    pass


# ======================================================================
# 参数堆叠异常
# ======================================================================

class StackingError(SimplCoreError):
    """参数堆叠错误.

    当模型参数堆叠 (Stacking) 操作失败时抛出。

    常见场景:
      - 堆叠的模型参数量不匹配
      - 堆叠层数超过硬件限制
      - 堆叠配置与基准模型不兼容
      - 堆叠过程中显存不足
    """

    pass


# ======================================================================
# 损失守卫异常
# ======================================================================

class LossGuardError(SimplCoreError):
    """损失守卫阈值违规错误.

    当训练损失超过安全阈值或出现异常波动时抛出。

    常见场景:
      - 损失超过 max_loss 阈值
      - 损失为 NaN 或 Inf
      - 损失在连续多步内不下降
      - 损失波动超过安全范围
    """

    pass


# ======================================================================
# 梯度裁剪工具函数
# ======================================================================

def handle_gradient_clipping(
    grad: torch.Tensor,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """裁剪梯度并返回裁剪后的张量.

    使用 L2 范数裁剪梯度, 防止梯度爆炸。
    当梯度范数超过 max_norm 时, 按比例缩放梯度使其范数等于 max_norm。

    Args:
        grad: 待裁剪的梯度张量.
        max_norm: 最大梯度范数, 默认 1.0.

    Returns:
        裁剪后的梯度张量 (与输入形状相同).

    Raises:
        GradientCheckpointError: 当梯度包含 NaN 或 Inf 时.

    Example:
        >>> loss = model(input_ids).sum()
        >>> loss.backward()
        >>> for p in model.parameters():
        ...     if p.grad is not None:
        ...         p.grad = handle_gradient_clipping(p.grad, max_norm=0.5)
    """
    if grad is None:
        return grad

    # 检查 NaN / Inf
    if torch.isnan(grad).any():
        raise GradientCheckpointError(
            "梯度包含 NaN, 无法裁剪",
            details={"grad_shape": tuple(grad.shape)},
        )
    if torch.isinf(grad).any():
        raise GradientCheckpointError(
            "梯度包含 Inf, 无法裁剪",
            details={"grad_shape": tuple(grad.shape)},
        )

    # 计算当前梯度范数
    total_norm = grad.norm(p=2).item()

    if total_norm > max_norm and total_norm > 0:
        # 按比例缩放
        scale = max_norm / (total_norm + 1e-6)
        grad = grad * scale

    return grad


# ======================================================================
# 异常信息格式化工具
# ======================================================================

def format_exception(exc: SimplCoreError) -> str:
    """格式化 SimplCore 异常为可读字符串.

    Args:
        exc: SimplCore 异常实例.

    Returns:
        格式化后的异常描述字符串.
    """
    lines = [
        f"[{exc.__class__.__name__}] {exc.message}",
    ]
    if exc.details:
        for key, value in exc.details.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore 异常模块自检")
    print("=" * 60)

    # 测试异常层级
    assert issubclass(ConfigError, SimplCoreError)
    assert issubclass(ModelError, SimplCoreError)
    assert issubclass(MemoryStateError, SimplCoreError)
    assert issubclass(RolloutError, SimplCoreError)
    assert issubclass(GradientCheckpointError, SimplCoreError)
    assert issubclass(ModalityError, SimplCoreError)
    assert issubclass(StackingError, SimplCoreError)
    assert issubclass(LossGuardError, SimplCoreError)
    print("[PASS] 所有异常均继承自 SimplCoreError")

    # 测试异常带 details
    try:
        raise ConfigError("hidden_size 不能为零", details={"hidden_size": 0})
    except ConfigError as e:
        assert e.message == "hidden_size 不能为零"
        assert e.details["hidden_size"] == 0
        print(f"[PASS] 异常详情正常: {e}")

    # 测试梯度裁剪 - 正常情况
    grad = torch.randn(3, 4) * 10  # 大梯度
    original_norm = grad.norm(p=2).item()
    clipped = handle_gradient_clipping(grad, max_norm=1.0)
    clipped_norm = clipped.norm(p=2).item()
    assert clipped_norm <= 1.0 + 1e-5, f"裁剪后范数 {clipped_norm} 超过阈值"
    print(f"[PASS] 梯度裁剪正常: {original_norm:.4f} → {clipped_norm:.4f}")

    # 测试梯度裁剪 - 小梯度不裁剪
    small_grad = torch.randn(3, 4) * 0.01
    small_clipped = handle_gradient_clipping(small_grad, max_norm=1.0)
    assert torch.allclose(small_grad, small_clipped)
    print("[PASS] 小梯度不裁剪")

    # 测试梯度裁剪 - NaN 检测
    nan_grad = torch.tensor([1.0, float("nan"), 3.0])
    try:
        handle_gradient_clipping(nan_grad, max_norm=1.0)
        assert False, "应抛出 GradientCheckpointError"
    except GradientCheckpointError as e:
        print(f"[PASS] NaN 检测正常: {e.message}")

    # 测试梯度裁剪 - Inf 检测
    inf_grad = torch.tensor([1.0, float("inf"), 3.0])
    try:
        handle_gradient_clipping(inf_grad, max_norm=1.0)
        assert False, "应抛出 GradientCheckpointError"
    except GradientCheckpointError as e:
        print(f"[PASS] Inf 检测正常: {e.message}")

    # 测试 format_exception
    exc = ModelError("维度不匹配", details={"expected": (2, 16), "got": (2, 32)})
    formatted = format_exception(exc)
    assert "[ModelError]" in formatted
    assert "维度不匹配" in formatted
    print(f"[PASS] 异常格式化正常:\n{formatted}")

    print("\n" + "=" * 60)
    print("所有异常模块测试 PASSED.")
    print("=" * 60)
