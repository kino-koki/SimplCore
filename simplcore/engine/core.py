# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCoreEngine - 青小团自研深度学习引擎核心
统一API层，支持 SimplCore/PyTorch 双后端无缝切换

核心功能:
1. 统一张量接口
2. 统一神经网络层接口
3. 统一优化器接口
4. 统一训练流程
5. 后端切换无需修改代码
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from enum import Enum


class BackendType(Enum):
    SimplCore = "SimplCore"
    PYTORCH = "pytorch"


class SimplCoreEngine:
    """青小团深度学习引擎核心"""
    
    def __init__(self, backend: Union[str, BackendType] = "SimplCore"):
        self._backend_type = BackendType(backend)
        self._torch = None
        
        if self._backend_type == BackendType.PYTORCH:
            try:
                import simplcore.tensor as torch
                self._torch = torch
            except ImportError:
                print("Warning: PyTorch not installed, falling back to SimplCore backend")
                self._backend_type = BackendType.SimplCore
        
        self._device = "cpu"
        self._dtype = np.float32
    
    @property
    def backend(self) -> str:
        return self._backend_type.value
    
    @property
    def device(self) -> str:
        return self._device
    
    @device.setter
    def device(self, value: str) -> None:
        self._device = value
    
    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = value
    
    def tensor(self, data: Union[np.ndarray, List, float, int], requires_grad: bool = False) -> Any:
        """创建张量"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.tensor(data, requires_grad=requires_grad).to(self._device)
        else:
            from .tensor import SimpTensor
            return SimpTensor(data, requires_grad=requires_grad)
    
    def randn(self, *shape: int, requires_grad: bool = False) -> Any:
        """生成正态分布随机张量"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.randn(*shape, requires_grad=requires_grad).to(self._device)
        else:
            from .tensor import randn
            return randn(*shape)
    
    def rand(self, *shape: int, requires_grad: bool = False) -> Any:
        """生成均匀分布随机张量"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.rand(*shape, requires_grad=requires_grad).to(self._device)
        else:
            from .tensor import rand
            return rand(*shape)
    
    def ones(self, *shape: int) -> Any:
        """生成全1张量"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.ones(*shape).to(self._device)
        else:
            from .tensor import ones
            return ones(*shape)
    
    def zeros(self, *shape: int) -> Any:
        """生成全0张量"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.zeros(*shape).to(self._device)
        else:
            from .tensor import zeros
            return zeros(*shape)
    
    def eye(self, n: int) -> Any:
        """生成单位矩阵"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.eye(n).to(self._device)
        else:
            from .tensor import eye
            return eye(n)
    
    def softmax(self, x: Any, dim: int = -1) -> Any:
        """softmax函数"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.softmax(x, dim=dim)
        else:
            from .tensor import softmax
            return softmax(x, dim)
    
    def relu(self, x: Any) -> Any:
        """ReLU激活函数"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.relu(x)
        else:
            from .tensor import relu
            return relu(x)
    
    def gelu(self, x: Any) -> Any:
        """GELU激活函数"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.functional.gelu(x)
        else:
            from .tensor import gelu
            return gelu(x)
    
    def layer_norm(self, x: Any, normalized_shape: int, eps: float = 1e-5) -> Any:
        """层归一化"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.functional.layer_norm(x, (normalized_shape,), eps=eps)
        else:
            from .tensor import layer_norm
            return layer_norm(x, eps)
    
    def dropout(self, x: Any, p: float = 0.5, training: bool = True) -> Any:
        """dropout"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.functional.dropout(x, p=p, training=training)
        else:
            from .tensor import dropout
            return dropout(x, p, training)
    
    def matmul(self, a: Any, b: Any) -> Any:
        """矩阵乘法"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.matmul(a, b)
        else:
            return a.matmul(b)
    
    def sum(self, x: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """求和"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.sum(x, dim=dim, keepdim=keepdim)
        else:
            return x.sum(dim=dim, keepdim=keepdim)
    
    def mean(self, x: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """求均值"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.mean(x, dim=dim, keepdim=keepdim)
        else:
            return x.mean(dim=dim, keepdim=keepdim)
    
    def cuda(self, x: Any) -> Any:
        """移动到CUDA"""
        if self._backend_type == BackendType.PYTORCH:
            return x.cuda()
        else:
            return x.cuda()
    
    def cpu(self, x: Any) -> Any:
        """移动到CPU"""
        if self._backend_type == BackendType.PYTORCH:
            return x.cpu()
        else:
            return x.cpu()
    
    def numpy(self, x: Any) -> np.ndarray:
        """转换为numpy数组"""
        if self._backend_type == BackendType.PYTORCH:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()
    
    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        """线性层"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.Linear(in_features, out_features, bias=bias).to(self._device)
        else:
            from .nn import Linear
            return Linear(in_features, out_features, bias=bias)
    
    def Embedding(self, num_embeddings: int, embedding_dim: int) -> Any:
        """嵌入层"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.Embedding(num_embeddings, embedding_dim).to(self._device)
        else:
            from .nn import Embedding
            return Embedding(num_embeddings, embedding_dim)
    
    def LayerNorm(self, normalized_shape: int, eps: float = 1e-5) -> Any:
        """层归一化层"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.LayerNorm(normalized_shape, eps=eps).to(self._device)
        else:
            from .nn import LayerNorm
            return LayerNorm(normalized_shape, eps=eps)
    
    def Dropout(self, p: float = 0.5) -> Any:
        """Dropout层"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.Dropout(p=p)
        else:
            from .nn import Dropout
            return Dropout(p=p)
    
    def MultiHeadAttention(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> Any:
        """多头注意力层"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True).to(self._device)
        else:
            from .nn import MultiHeadAttention
            return MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
    
    def Sequential(self, *modules: Any) -> Any:
        """顺序容器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.nn.Sequential(*modules)
        else:
            from .nn import Sequential
            return Sequential(*modules)
    
    def Adam(self, parameters: List[Any], lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> Any:
        """Adam优化器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            from .optim import Adam
            return Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    def AdamW(self, parameters: List[Any], lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01) -> Any:
        """AdamW优化器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            from .optim import AdamW
            return AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    
    def SGD(self, parameters: List[Any], lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0) -> Any:
        """SGD优化器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            from .optim import SGD
            return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    def RMSprop(self, parameters: List[Any], lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0) -> Any:
        """RMSprop优化器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.optim.RMSprop(parameters, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        else:
            from .optim import RMSprop
            return RMSprop(parameters, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
    
    def CosineAnnealingLR(self, optimizer: Any, T_max: int, eta_min: float = 0) -> Any:
        """余弦退火学习率调度器"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)
        else:
            from .optim import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)
    
    def WarmupCosineAnnealingLR(self, optimizer: Any, warmup_steps: int, T_max: int, eta_min: float = 0) -> Any:
        """预热+余弦退火学习率调度器"""
        if self._backend_type == BackendType.PYTORCH:
            from simplcore.tensor.optim import CosineAnnealingLR, LambdaLR
            scheduler1 = LambdaLR(optimizer, lr_lambda=lambda step: step / warmup_steps if step <= warmup_steps else 1.0)
            scheduler2 = CosineAnnealingLR(optimizer, T_max - warmup_steps, eta_min=eta_min)
            return [scheduler1, scheduler2]
        else:
            from .optim import WarmupCosineAnnealingLR
            return WarmupCosineAnnealingLR(optimizer, warmup_steps, T_max, eta_min)
    
    def fft(self, x: Any, dim: int = -1) -> Any:
        """快速傅里叶变换"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.fft.fft(x, dim=dim)
        else:
            from .fft import fft
            return fft(x, dim)
    
    def ifft(self, x: Any, dim: int = -1) -> Any:
        """逆快速傅里叶变换"""
        if self._backend_type == BackendType.PYTORCH:
            return self._torch.fft.ifft(x, dim=dim)
        else:
            from .fft import ifft
            return ifft(x, dim)
    
    def SpectralDiffusionLayer(self, dim: int, num_waves: int = 8, damping: float = 0.1, num_steps: int = 3) -> Any:
        """谱扩散层（FFT频域带通 + 热扩散迭代）"""
        from .nn import SpectralDiffusionLayer
        return SpectralDiffusionLayer(dim, num_waves, damping, num_steps)

    def ExternalMemoryLayer(self, dim: int, num_channels: int = 64, channel_size: int = 128, num_heads: int = 4) -> Any:
        """外部记忆层（可微分记忆矩阵 + 注意力读写）"""
        from .nn import ExternalMemoryLayer
        return ExternalMemoryLayer(dim, num_channels, channel_size, num_heads)

    def AdaptiveDepthRouter(self, dim: int, num_paths: int = 4, min_depth: int = 2, max_depth: int = 8, topk: int = 2, dropout: float = 0.0) -> Any:
        """自适应深度路由器（多深度GELU + top-k融合）"""
        from .nn import AdaptiveDepthRouter
        return AdaptiveDepthRouter(dim, num_paths, min_depth, max_depth, topk, dropout)

    def SimplCoreBlock(self, dim: int, num_heads: int, num_waves: int, num_channels: int, num_paths: int, **kwargs) -> Any:
        """SimplCore核心块"""
        from .nn import SimplCoreBlock
        return SimplCoreBlock(dim, num_heads, num_waves, num_channels, num_paths, **kwargs)

    # --- 向后兼容别名（deprecated）---
    def LegacySpectralMixer(self, *a, **kw) -> Any:
        """@deprecated: 请使用 SpectralDiffusionLayer"""
        return self.SpectralDiffusionLayer(*a, **kw)

    def DynamicMemoryChannel(self, *a, **kw) -> Any:
        """@deprecated: 请使用 ExternalMemoryLayer"""
        return self.ExternalMemoryLayer(*a, **kw)

    def LegacyConditionalRouter(self, *a, **kw) -> Any:
        """@deprecated: 请使用 AdaptiveDepthRouter"""
        return self.AdaptiveDepthRouter(*a, **kw)
    
    def train(self, model: Any) -> Any:
        """切换到训练模式"""
        if self._backend_type == BackendType.PYTORCH:
            model.train()
        else:
            model.train()
        return model
    
    def eval(self, model: Any) -> Any:
        """切换到评估模式"""
        if self._backend_type == BackendType.PYTORCH:
            model.eval()
        else:
            model.eval()
        return model
    
    def parameters(self, model: Any) -> List[Any]:
        """获取模型参数"""
        if self._backend_type == BackendType.PYTORCH:
            return list(model.parameters())
        else:
            return model.parameters()
    
    def named_parameters(self, model: Any) -> List[Tuple[str, Any]]:
        """获取模型参数（带名称）"""
        if self._backend_type == BackendType.PYTORCH:
            return list(model.named_parameters())
        else:
            return model.named_parameters()
    
    def backward(self, loss: Any) -> None:
        """反向传播"""
        if self._backend_type == BackendType.PYTORCH:
            loss.backward()
        else:
            loss.backward()
    
    def zero_grad(self, optimizer: Any) -> None:
        """清空梯度"""
        if self._backend_type == BackendType.PYTORCH:
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
    
    def step(self, optimizer: Any) -> None:
        """执行优化步骤"""
        if self._backend_type == BackendType.PYTORCH:
            optimizer.step()
        else:
            optimizer.step()
    
    def save_checkpoint(self, model: Any, optimizer: Any, path: str, **kwargs) -> None:
        """保存检查点"""
        if self._backend_type == BackendType.PYTORCH:
            self._torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                **kwargs
            }, path)
        else:
            from .distributed import CheckpointManager
            manager = CheckpointManager(os.path.dirname(path))
            manager.save(model, optimizer, **kwargs)
    
    def load_checkpoint(self, model: Any, optimizer: Any, path: str) -> Dict:
        """加载检查点"""
        if self._backend_type == BackendType.PYTORCH:
            checkpoint = self._torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
        else:
            from .distributed import CheckpointManager
            manager = CheckpointManager(os.path.dirname(path))
            manager.load(model, optimizer)
            return {}


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCoreEngine 自检")
    print("=" * 60)
    
    for backend in ["SimplCore"]:
        print(f"\n--- Testing {backend} backend ---")
        
        engine = SimplCoreEngine(backend=backend)
        
        x = engine.randn(2, 3)
        print(f"randn shape: {x.shape}")
        
        y = engine.ones(2, 3)
        z = x + y
        print(f"add shape: {z.shape}")
        
        mat = engine.matmul(x, engine.randn(3, 4))
        print(f"matmul shape: {mat.shape}")
        
        sm = engine.softmax(x, dim=-1)
        print(f"softmax shape: {sm.shape}")
        
        linear = engine.Linear(3, 5)
        lin_out = linear(x)
        print(f"Linear output shape: {lin_out.shape}")
        
        model = engine.Sequential(
            engine.Linear(3, 10),
            engine.gelu,
            engine.Linear(10, 5),
        )
        model_out = model(x)
        print(f"Sequential output shape: {model_out.shape}")
        
        optimizer = engine.Adam(engine.parameters(model), lr=0.001)
        
        print(f"{backend} backend PASSED")
    
    print("\nSimplCoreEngine 自检 PASSED")

