# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimpOptimizer - 青小团自研优化器库
替代 torch.optim 的核心优化器
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional
from .tensor import SimpTensor


class Optimizer:
    """优化器基类"""
    
    def __init__(self, parameters: List[SimpTensor], lr: float = 1e-3):
        self.parameters = parameters
        self.lr = lr
        self._state: Dict[str, Dict] = {}
    
    def zero_grad(self) -> None:
        """清空梯度"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
    
    def step(self) -> None:
        """执行一步优化"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "momentum": np.zeros_like(param._data) if momentum != 0 else None,
            }
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            if self.momentum != 0:
                state["momentum"] = self.momentum * state["momentum"] + grad
                param._data = param._data - self.lr * state["momentum"]
            else:
                param._data = param._data - self.lr * grad


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "m": np.zeros_like(param._data),
                "v": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        self._step += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = state["m"] / (1 - self.beta1 ** self._step)
            v_hat = state["v"] / (1 - self.beta2 ** self._step)
            
            param._data = param._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "m": np.zeros_like(param._data),
                "v": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        self._step += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = state["m"] / (1 - self.beta1 ** self._step)
            v_hat = state["v"] / (1 - self.beta2 ** self._step)
            
            param._data = param._data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param._data = param._data - self.lr * self.weight_decay * param._data


class RMSprop(Optimizer):
    """RMSprop优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "square_avg": np.zeros_like(param._data),
                "momentum": np.zeros_like(param._data) if momentum != 0 else None,
            }
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["square_avg"] = self.alpha * state["square_avg"] + (1 - self.alpha) * (grad ** 2)
            
            if self.momentum != 0:
                state["momentum"] = self.momentum * state["momentum"] + grad / np.sqrt(state["square_avg"] + self.eps)
                param._data = param._data - self.lr * state["momentum"]
            else:
                param._data = param._data - self.lr * grad / np.sqrt(state["square_avg"] + self.eps)


class Adagrad(Optimizer):
    """Adagrad优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.eps = eps
        self.weight_decay = weight_decay
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "sum": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["sum"] = state["sum"] + grad ** 2
            param._data = param._data - self.lr * grad / np.sqrt(state["sum"] + self.eps)


class Adadelta(Optimizer):
    """Adadelta优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "square_avg": np.zeros_like(param._data),
                "delta_avg": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["square_avg"] = self.rho * state["square_avg"] + (1 - self.rho) * (grad ** 2)
            delta = grad * np.sqrt(state["delta_avg"] + self.eps) / np.sqrt(state["square_avg"] + self.eps)
            state["delta_avg"] = self.rho * state["delta_avg"] + (1 - self.rho) * (delta ** 2)
            
            param._data = param._data - self.lr * delta


class Adamax(Optimizer):
    """Adamax优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 2e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "m": np.zeros_like(param._data),
                "u": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        self._step += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["u"] = np.maximum(self.beta2 * state["u"], np.abs(grad))
            
            m_hat = state["m"] / (1 - self.beta1 ** self._step)
            param._data = param._data - self.lr * m_hat / (state["u"] + self.eps)


class NAdam(Optimizer):
    """NAdam优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 2e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self._step = 0
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "m": np.zeros_like(param._data),
                "v": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        self._step += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = state["m"] / (1 - self.beta1 ** self._step)
            v_hat = state["v"] / (1 - self.beta2 ** self._step)
            
            m_tilde = (1 - self.beta1) * grad + self.beta1 * m_hat
            param._data = param._data - self.lr * m_tilde / (np.sqrt(v_hat) + self.eps)


class LBFGS(Optimizer):
    """L-BFGS优化器"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
    ):
        super().__init__(parameters, lr)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "old_dirs": [],
                "old_stps": [],
                "H_diag": 1.0,
            }
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if np.linalg.norm(grad) < self.tolerance_grad:
                continue
            
            old_dirs = state["old_dirs"]
            old_stps = state["old_stps"]
            H_diag = state["H_diag"]
            
            s = -self.lr * grad / H_diag
            
            if len(old_dirs) > 0:
                rho = [1.0 / np.dot(d, y) for d, y in zip(old_dirs, old_stps)]
                alpha = [rho[j] * np.dot(old_dirs[j], s) for j in range(len(old_dirs))]
                
                for j in reversed(range(len(old_dirs))):
                    s = s - alpha[j] * old_stps[j]
                
                s = s * H_diag
                
                for j in range(len(old_dirs)):
                    beta = rho[j] * np.dot(old_stps[j], s)
                    s = s + (alpha[j] - beta) * old_dirs[j]
            
            param._data = param._data + s
            
            if len(old_dirs) >= self.history_size:
                old_dirs.pop(0)
                old_stps.pop(0)
            
            old_dirs.append(-grad)
            old_stps.append(s)
            
            if len(old_stps) > 0:
                last_y = old_stps[-1]
                last_s = old_dirs[-1]
                H_diag = np.dot(last_y, last_y) / np.dot(last_y, last_s)


class ASGD(Optimizer):
    """平均随机梯度下降"""
    
    def __init__(
        self,
        parameters: List[SimpTensor],
        lr: float = 0.01,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.weight_decay = weight_decay
        self._step = 0
        
        for i, param in enumerate(parameters):
            self._state[f"param_{i}"] = {
                "eta": lr,
                "mu": 1.0,
                "ax": np.zeros_like(param._data),
            }
    
    def step(self) -> None:
        self._step += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            state = self._state[f"param_{i}"]
            grad = param.grad._data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param._data
            
            state["eta"] = self.lr / ((1 + self.lambd * self.lr * self._step) ** self.alpha)
            
            param._data = param._data - state["eta"] * grad
            
            state["mu"] = 1.0 / max(1, self._step - self.t0 + 1)
            state["ax"] = (1 - state["mu"]) * state["ax"] + state["mu"] * param._data
