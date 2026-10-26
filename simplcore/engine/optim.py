# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimpOptimizer - 青小团自研优化器库
替代 torch.optim 的核心优化器
（主入口文件，从optimizers.py导入所有优化器）
"""
from __future__ import annotations

from .optimizers import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    NAdam,
    LBFGS,
    ASGD,
)

import numpy as np
from typing import List, Dict, Optional
from .tensor import SimpTensor


class LearningRateScheduler:
    """学习率调度器基类"""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self._step = 0
    
    def step(self) -> None:
        """更新学习率"""
        self._step += 1
        self.optimizer.lr = self.get_lr()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.lr


class CosineAnnealingLR(LearningRateScheduler):
    """余弦退火学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr
    
    def get_lr(self) -> float:
        if self._step >= self.T_max:
            return self.eta_min
        return self.eta_min + (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * self._step / self.T_max)) / 2


class StepLR(LearningRateScheduler):
    """阶梯式学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.lr
    
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** (self._step // self.step_size))


class ExponentialLR(LearningRateScheduler):
    """指数衰减学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, gamma: float = 0.99):
        super().__init__(optimizer)
        self.gamma = gamma
        self.initial_lr = optimizer.lr
    
    def get_lr(self) -> float:
        return self.initial_lr * (self.gamma ** self._step)


class LinearWarmupLR(LearningRateScheduler):
    """线性预热学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.initial_lr = optimizer.lr
    
    def get_lr(self) -> float:
        if self._step <= self.warmup_steps:
            return self.initial_lr * (self._step / self.warmup_steps)
        return self.initial_lr


class WarmupCosineAnnealingLR(LearningRateScheduler):
    """预热+余弦退火学习率调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr
    
    def get_lr(self) -> float:
        if self._step <= self.warmup_steps:
            return self.initial_lr * (self._step / self.warmup_steps)
        t = self._step - self.warmup_steps
        T_total = self.T_max - self.warmup_steps
        return self.eta_min + (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * t / T_total)) / 2


if __name__ == "__main__":
    print("=" * 60)
    print("SimpOptimizer 自检")
    print("=" * 60)
    
    from .tensor import randn
    
    params = [randn(3, 4, requires_grad=True)]
    
    sgd = SGD(params, lr=0.01, momentum=0.9)
    adam = Adam(params, lr=0.001)
    adamw = AdamW(params, lr=0.001, weight_decay=0.01)
    rmsprop = RMSprop(params, lr=0.01)
    adagrad = Adagrad(params, lr=0.01)
    adadelta = Adadelta(params, lr=1.0)
    adamax = Adamax(params, lr=0.002)
    nadam = NAdam(params, lr=0.002)
    lbfgs = LBFGS(params, lr=1.0)
    asgd = ASGD(params, lr=0.01)
    
    for opt_name, opt in [("SGD", sgd), ("Adam", adam), ("AdamW", adamw), ("RMSprop", rmsprop), 
                          ("Adagrad", adagrad), ("Adadelta", adadelta), ("Adamax", adamax),
                          ("NAdam", nadam), ("LBFGS", lbfgs), ("ASGD", asgd)]:
        x = randn(4, 2)
        y = params[0].matmul(x)
        loss = y.sum()
        loss.backward()
        
        opt.step()
        opt.zero_grad()
        
        print(f"{opt_name}: step completed")
    
    print("\n学习率调度器测试")
    adam_scheduler = Adam(params, lr=0.001)
    cosine_scheduler = CosineAnnealingLR(adam_scheduler, T_max=100, eta_min=1e-6)
    warmup_cosine = WarmupCosineAnnealingLR(adam_scheduler, warmup_steps=10, T_max=100)
    
    for i in range(15):
        warmup_cosine.step()
    
    print("所有测试通过!")
