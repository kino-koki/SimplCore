from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class LossGuardAction(str, Enum):
    PASS = "pass"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class LossGuardDecision:
    step: int
    loss: float
    action: LossGuardAction
    reason: str

    @property
    def should_continue(self) -> bool:
        return self.action != LossGuardAction.STOP

    def to_dict(self) -> dict:
        data = asdict(self)
        data["action"] = self.action.value
        data["should_continue"] = self.should_continue
        return data


@dataclass(frozen=True)
class LossGuard:
    target_loss: float = 1.5
    max_loss: float = 3.9
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.target_loss <= 0:
            raise ValueError("target_loss must be positive")
        if self.max_loss <= self.target_loss:
            raise ValueError("max_loss must be greater than target_loss")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

    def check(self, loss: float, *, step: int) -> LossGuardDecision:
        # 预热期不干预: 新模型初始 loss ≈ ln(vocab_size) (如 vocab 32000 时 ~10.4),
        # 天然高于 max_loss, 直接 STOP 会让训练永远无法开始。
        # step 为 1-based (调用方 range(1, max_steps+1)); warmup_steps=0 时
        # 任何 step 立即检查, 保持原行为。
        if step <= self.warmup_steps:
            return LossGuardDecision(
                step=step,
                loss=loss,
                action=LossGuardAction.PASS,
                reason=f"预热阶段 (step {step}/{self.warmup_steps}), 损失 {loss:.4f} 暂不干预",
            )
        if loss > self.max_loss:
            return LossGuardDecision(
                step=step,
                loss=loss,
                action=LossGuardAction.STOP,
                reason=f"训练损失 {loss:.4f} 超过最高损失上限 {self.max_loss:.4f}",
            )
        if loss >= self.target_loss:
            return LossGuardDecision(
                step=step,
                loss=loss,
                action=LossGuardAction.WARN,
                reason=f"训练损失 {loss:.4f} 尚未低于目标 {self.target_loss:.4f}",
            )
        return LossGuardDecision(
            step=step,
            loss=loss,
            action=LossGuardAction.PASS,
            reason=f"训练损失 {loss:.4f} 已低于目标 {self.target_loss:.4f}",
        )
