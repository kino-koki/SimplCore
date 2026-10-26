# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimpNN - 青小团自研神经网络层库
替代 torch.nn 的核心层，构建完整的神经网络模型
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List
from .tensor import SimpTensor, randn, rand, zeros, ones, eye, softmax, relu, gelu, silu, layer_norm, dropout, conv2d, max_pool2d, avg_pool2d, batch_norm2d, interpolate

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class Module:
    """神经网络模块基类"""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def register_parameter(self, name: str, param: SimpTensor) -> None:
        """注册参数"""
        self._parameters[name] = param
    
    def register_module(self, name: str, module: "Module") -> None:
        """注册子模块"""
        self._modules[name] = module
    
    def parameters(self) -> List[SimpTensor]:
        """获取所有可训练参数"""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def named_parameters(self) -> List[Tuple[str, SimpTensor]]:
        """获取所有可训练参数（带名称）"""
        params = [(name, param) for name, param in self._parameters.items()]
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                params.append((f"{module_name}.{name}", param))
        return params
    
    def eval(self) -> "Module":
        """切换到评估模式"""
        self.training = False
        for module in self._modules.values():
            module.eval()
        return self
    
    def train(self) -> "Module":
        """切换到训练模式"""
        self.training = True
        for module in self._modules.values():
            module.train()
        return self
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        """前向传播（子类实现）"""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> SimpTensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """线性层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = randn(out_features, in_features) * std
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
        
        if bias:
            self.bias = zeros(out_features)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        x_data = x._data
        weight_data = self.weight._data
        
        if x.ndim == 1:
            y_data = np.matmul(x_data.reshape(1, -1), weight_data.T).reshape(self.out_features)
        elif x.ndim == 2:
            y_data = np.matmul(x_data, weight_data.T)
        else:
            x_reshaped = x_data.reshape(-1, self.in_features)
            y_data = np.matmul(x_reshaped, weight_data.T).reshape(*x.shape[:-1], self.out_features)
        
        if self.bias is not None:
            y_data = y_data + self.bias._data
        
        y = SimpTensor(y_data, requires_grad=x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad),
                      device=x.device, dtype=x.dtype)
        
        if y.requires_grad and (self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)):
            def grad_fn(grad: SimpTensor) -> None:
                grad_data = grad._data
                
                if x.ndim == 1:
                    x_reshaped = x_data.reshape(1, -1)
                    grad_reshaped = grad_data.reshape(1, -1)
                elif x.ndim == 2:
                    x_reshaped = x_data
                    grad_reshaped = grad_data
                else:
                    x_reshaped = x_data.reshape(-1, self.in_features)
                    grad_reshaped = grad_data.reshape(-1, self.out_features)
                
                if x.requires_grad:
                    grad_x_data = np.matmul(grad_reshaped, weight_data)
                    if x.ndim != 2:
                        grad_x_data = grad_x_data.reshape(*x.shape)
                    x.backward(SimpTensor(grad_x_data))
                
                if self.weight.requires_grad:
                    grad_weight_data = np.matmul(grad_reshaped.T, x_reshaped)
                    grad_weight = SimpTensor(grad_weight_data)
                    if self.weight.grad is None:
                        self.weight._grad = grad_weight
                    else:
                        self.weight._grad = self.weight._grad + grad_weight
                
                if self.bias is not None and self.bias.requires_grad:
                    grad_bias_data = grad_reshaped.sum(axis=0)
                    grad_bias = SimpTensor(grad_bias_data)
                    if self.bias.grad is None:
                        self.bias._grad = grad_bias
                    else:
                        self.bias._grad = self.bias._grad + grad_bias
            y._grad_fn = grad_fn
        
        return y


class Embedding(Module):
    """嵌入层"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = randn(num_embeddings, embedding_dim) * 0.02
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return SimpTensor(self.weight._data[x._data.astype(np.int64)])


class LayerNorm(Module):
    """层归一化"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = ones(normalized_shape)
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
        
        self.bias = zeros(normalized_shape)
        self.bias._requires_grad = True
        self.register_parameter("bias", self.bias)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        normalized = layer_norm(x, self.eps)
        result = normalized * self.weight + self.bias
        
        if result.requires_grad and (self.weight.requires_grad or self.bias.requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                if self.weight.requires_grad:
                    grad_weight = (grad * normalized).sum()
                    if grad_weight.shape != self.weight.shape:
                        grad_weight = (grad * normalized).sum(dim=tuple(range(grad.ndim - 1)))
                    if self.weight.grad is None:
                        self.weight._grad = grad_weight
                    else:
                        self.weight._grad = self.weight._grad + grad_weight
                if self.bias.requires_grad:
                    grad_bias = grad.sum()
                    if grad_bias.shape != self.bias.shape:
                        grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))
                    if self.bias.grad is None:
                        self.bias._grad = grad_bias
                    else:
                        self.bias._grad = self.bias._grad + grad_bias
                grad_normalized = grad * self.weight
                normalized.backward(grad_normalized)
            result._grad_fn = grad_fn
        
        return result


class Dropout(Module):
    """Dropout层"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return dropout(x, self.p, self.training)


class Sequential(Module):
    """顺序容器"""
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            if isinstance(module, Module):
                self.register_module(f"module_{i}", module)
            elif callable(module):
                setattr(self, f"module_{i}", module)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        for module in self._modules.values():
            x = module(x)
        for i in range(len(self._modules), len([attr for attr in dir(self) if attr.startswith("module_")])):
            module = getattr(self, f"module_{i}")
            x = module(x)
        return x


class ReLU(Module):
    """ReLU激活层"""
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return relu(x)


class GELU(Module):
    """GELU激活层"""
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return gelu(x)


class SiLU(Module):
    """SiLU激活层"""
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return silu(x)


class Softmax(Module):
    """Softmax层"""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return softmax(x, self.dim)


class MultiHeadAttention(Module):
    """多头注意力层"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        self.dropout_layer = Dropout(dropout)
    
    def forward(
        self,
        query: SimpTensor,
        key: SimpTensor,
        value: SimpTensor,
        mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        B, L_q, D = query.shape
        _, L_k, _ = key.shape
        
        q = self.q_proj(query).reshape(B, L_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(B, L_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(B, L_k, self.num_heads, self.head_dim)
        
        attn = q.matmul(k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(1)
        
        attn = softmax(attn, dim=-1)
        
        if self.dropout > 0 and self.training:
            attn = self.dropout_layer(attn)
        
        output = attn.matmul(v).reshape(B, L_q, D)
        output = self.out_proj(output)
        
        return output


class FeedForward(Module):
    """前馈网络"""
    
    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up = Linear(dim, intermediate_dim)
        self.down = Linear(intermediate_dim, dim)
        self.dropout = Dropout(dropout)
        self.act = GELU()
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return self.down(self.dropout(self.act(self.up(x))))


class SpectralDiffusionLayer(Module):
    """谱扩散层（SimplCore核心）

    真实机制（非文档宣称的"共振场"）：
    1. 用 FFT 在频域对输入做可学习带通滤波（band-pass via learnable omega/alpha）；
    2. 用类热扩散方程（Laplacian 多步迭代）在频域演化特征。
    等价于一个频域卷积 + 迭代细化的模块，适合捕捉长程周期/平滑结构。
    """
    
    def __init__(self, dim: int, num_waves: int = 8, damping: float = 0.1, num_steps: int = 3):
        super().__init__()
        self.dim = dim
        self.num_waves = num_waves
        self.num_steps = num_steps
        self.damping = damping
        
        self.in_proj = Linear(dim, num_waves)
        self.out_proj = Linear(num_waves, dim)
        
        self.omega_raw = randn(num_waves) * 0.3
        self.omega_raw._requires_grad = True
        self.register_parameter("omega_raw", self.omega_raw)
        
        self.alpha_raw = randn(num_waves) * 0.1 + np.log(np.expm1(damping))
        self.alpha_raw._requires_grad = True
        self.register_parameter("alpha_raw", self.alpha_raw)
        
        self.gamma = ones(num_waves) * 0.5
        self.gamma._requires_grad = True
        self.register_parameter("gamma", self.gamma)
        
        self.beta = zeros(1) + damping
        self.beta._requires_grad = True
        self.register_parameter("beta", self.beta)
        
        self.out_scale = ones(1)
        self.out_scale._requires_grad = True
        self.register_parameter("out_scale", self.out_scale)
        
        self.norm = LayerNorm(dim)
    
    @property
    def omega(self) -> SimpTensor:
        return SimpTensor(np.log(1 + np.exp(self.omega_raw._data)))
    
    @property
    def alpha(self) -> SimpTensor:
        return SimpTensor(np.log(1 + np.exp(self.alpha_raw._data)))
    
    def _laplacian(self, field: SimpTensor) -> SimpTensor:
        L = field.shape[1]
        freqs = np.fft.fftfreq(L, d=1.0) * 2.0 * np.pi
        k_sq = freqs ** 2
        
        field_fft = np.fft.fft(field._data, axis=1)
        lap_fft = -k_sq.reshape(1, -1, 1) * field_fft
        lap = np.fft.ifft(lap_fft, axis=1).real
        
        return SimpTensor(lap)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        residual = x
        h = self.norm(x)
        B, L, D = h.shape
        
        I_field = self.in_proj(h)
        
        freqs = np.fft.fftfreq(L, d=1.0) * 2.0 * np.pi
        omega = self.omega._data
        
        band = np.exp(-((np.abs(freqs.reshape(-1, 1)) - omega.reshape(1, -1)) ** 2) / 0.5)
        
        F_field = np.fft.ifft(np.fft.fft(I_field._data, axis=1) * band.reshape(1, L, -1), axis=1).real
        
        alpha = self.alpha._data.reshape(1, 1, -1)
        gamma = self.gamma._data.reshape(1, 1, -1)
        beta = self.beta._data[0]
        
        for _ in range(self.num_steps):
            lap = self._laplacian(SimpTensor(F_field))
            F_field = F_field + 0.1 * (alpha * lap._data - beta * F_field + gamma * I_field._data)
        
        out = self.out_proj(SimpTensor(F_field))
        return residual + self.out_scale * out


class ExternalMemoryLayer(Module):
    """外部记忆层（SimplCore核心）

    真实机制（非文档宣称的"动态记忆通道"）：
    维护一个可微分的外部记忆矩阵 [batch, channels, channel_size, dim]，
    通过 query/key 注意力读取、通过 write_gate 写入，实现跨时间步的信息存储与检索。
    """
    
    def __init__(self, dim: int, num_channels: int = 64, channel_size: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels
        self.channel_size = channel_size
        self.num_heads = num_heads if dim % num_heads == 0 else 1
        self.head_dim = dim // self.num_heads
        
        self.query_proj = Linear(dim, dim, bias=False)
        self.key_proj = Linear(dim, dim, bias=False)
        self.write_value_proj = Linear(dim, dim)
        self.output_proj = Linear(dim, dim)
        self.write_gate = Linear(dim, num_channels)
        
        self.channel_weights = ones(num_channels)
        self.channel_weights._requires_grad = True
        self.register_parameter("channel_weights", self.channel_weights)
        
        self.norm = LayerNorm(dim)
        
        self._memory = None
    
    def reset_state(self, batch_size: int):
        self._memory = np.random.randn(batch_size, self.num_channels, self.channel_size, self.dim) * 0.02
    
    def get_state(self) -> np.ndarray:
        return self._memory
    
    def set_state(self, memory: np.ndarray):
        self._memory = memory
    
    def read(self, query: SimpTensor) -> SimpTensor:
        B = query.shape[0]
        
        if self._memory is None:
            self.reset_state(B)
        
        q = self.query_proj(query)._data.reshape(B, self.num_heads, self.head_dim)
        keys = self.key_proj(SimpTensor(self._memory))._data.reshape(B, self.num_channels, self.channel_size, self.num_heads, self.head_dim)
        
        scores = np.einsum("bhd,bcshd->bchs", q, keys) / np.sqrt(self.head_dim)
        weights = softmax(SimpTensor(scores), dim=-1)._data
        
        mem_view = self._memory.reshape(B, self.num_channels, self.channel_size, self.num_heads, self.head_dim)
        readout = np.einsum("bchs,bcshd->bchd", weights, mem_view)
        
        return SimpTensor(readout.reshape(B, self.num_channels, self.dim))
    
    def write(self, key: SimpTensor, value: SimpTensor):
        B = key.shape[0]
        
        if self._memory is None:
            self.reset_state(B)
        
        k = self.query_proj(key)._data.reshape(B, self.num_heads, self.head_dim)
        v = self.write_value_proj(value)._data
        
        gate = sigmoid(self.write_gate(key)._data)
        
        keys = self.key_proj(SimpTensor(self._memory))._data.reshape(B, self.num_channels, self.channel_size, self.num_heads, self.head_dim)
        scores = np.einsum("bhd,bcshd->bchs", k, keys) / np.sqrt(self.head_dim)
        w = softmax(SimpTensor(scores), dim=-1)._data.mean(axis=2)
        
        g = gate.reshape(B, self.num_channels, 1, 1)
        w_exp = w.reshape(B, self.num_channels, self.channel_size, 1)
        v_exp = v.reshape(B, 1, 1, self.dim)
        
        self._memory = self._memory * (1.0 - g * w_exp) + g * w_exp * v_exp
        self._memory = np.clip(self._memory, -10.0, 10.0)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        residual = x
        B, L, D = x.shape
        
        outputs = []
        cw = softmax(SimpTensor(self.channel_weights._data), dim=0)._data
        
        for t in range(L):
            xt = x[:, t, :]
            
            readout = self.read(xt)._data
            self.write(xt, xt)
            
            agg = np.einsum("c,bcd->bd", cw, readout)
            out_t = self.output_proj(SimpTensor(agg))._data
            outputs.append(out_t)
        
        output = np.stack(outputs, axis=1)
        return residual + self.norm(SimpTensor(output))


class AdaptiveDepthRouter(Module):
    """自适应深度路由器（SimplCore核心）

    真实机制（非文档宣称的"条件计算路由"）：
    1. 学习路由权重，对输入做多个不同深度（depth）的 GELU 残差变换；
    2. 按路由概率对 top-k 深度的输出做加权融合。
    本质是一个多尺度/多深度残差路由模块，与图条件计算无关。
    """
    
    def __init__(self, dim: int, num_paths: int = 4, min_depth: int = 2, max_depth: int = 8, topk: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.topk = topk
        self.dropout = dropout
        
        self.router = Linear(dim, num_paths)
        self.transform = Linear(dim, dim)
        
        self.path_depths = self._compute_path_depths()
        
        self.norm = LayerNorm(dim)
        self.out_norm = LayerNorm(dim)
        
        self.temperature = ones(1)
        self.temperature._requires_grad = True
        self.register_parameter("temperature", self.temperature)
    
    def _compute_path_depths(self) -> List[int]:
        if self.num_paths == 1:
            return [self.min_depth]
        depths = []
        for p in range(self.num_paths):
            d = self.min_depth + round(p * (self.max_depth - self.min_depth) / (self.num_paths - 1))
            depths.append(int(d))
        return depths
    
    def route(self, x: SimpTensor) -> SimpTensor:
        logits = self.router(x)
        temp = max(self.temperature._data[0], 1e-3)
        probs = softmax(SimpTensor(logits._data / temp), dim=-1)
        return probs
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        residual = x
        h = self.norm(x)
        B, L, D = h.shape
        
        probs = self.route(h)._data
        
        max_d = max(self.path_depths)
        depth_states = {0: h._data}
        current = h._data
        
        for d in range(1, max_d + 1):
            current = current + gelu(SimpTensor(self.transform(SimpTensor(current))._data))._data
            depth_states[d] = current
        
        path_states = np.stack([depth_states[d] for d in self.path_depths], axis=2)
        
        if self.topk >= self.num_paths:
            out = np.einsum("blp,blpd->bld", probs, path_states)
        else:
            topk_indices = np.argsort(-probs, axis=-1)[:, :, :self.topk]
            topk_probs = np.take_along_axis(probs, topk_indices, axis=-1)
            topk_probs = topk_probs / (topk_probs.sum(axis=-1, keepdims=True) + 1e-8)
            
            topk_states = np.zeros((B, L, self.topk, D))
            for b in range(B):
                for l in range(L):
                    for k in range(self.topk):
                        topk_states[b, l, k] = path_states[b, l, topk_indices[b, l, k]]
            
            out = np.einsum("blk,blkd->bld", topk_probs, topk_states)
        
        out = self.out_norm(SimpTensor(out))
        return residual + out


class SimplCoreBlock(Module):
    """SimplCore核心块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_waves: int,
        num_channels: int,
        num_paths: int,
        channel_size: int = 128,
        min_depth: int = 2,
        max_depth: int = 8,
        topk: int = 2,
        ffn_intermediate_ratio: float = 4.0,
        sgc_num_steps: int = 3,
        sgc_damping: float = 0.1,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        
        self.sgc = SpectralDiffusionLayer(dim, num_waves, sgc_damping, sgc_num_steps)
        self.dmc = ExternalMemoryLayer(dim, num_channels, channel_size, num_heads)
        self.ccr = AdaptiveDepthRouter(dim, num_paths, min_depth, max_depth, topk, dropout)
        
        self.ffn_norm = LayerNorm(dim, eps)
        self.ffn = FeedForward(dim, int(ffn_intermediate_ratio * dim), dropout)
        self.ffn_dropout = Dropout(dropout)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        h = self.sgc(x)
        h = self.dmc(h)
        h = self.ccr(h)
        h = h + self.ffn_dropout(self.ffn(self.ffn_norm(h)))
        return h


class Conv2d(Module):
    """二维卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = randn(*kernel_shape) * std
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
        
        if bias:
            self.bias = zeros(out_channels)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding, self.groups)


class Conv1d(Module):
    """一维卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        kernel_shape = (out_channels, in_channels // groups, kernel_size)
        std = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = randn(*kernel_shape) * std
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
        
        if bias:
            self.bias = zeros(out_channels)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C_in, L = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding
        
        x_padded = np.pad(x._data, ((0, 0), (0, 0), (P, P)), mode='constant')
        
        L_out = (L + 2 * P - K) // S + 1
        
        out = np.zeros((B, C_out, L_out))
        
        for b in range(B):
            for c_out in range(C_out):
                for l_out in range(L_out):
                    l_start = l_out * S
                    region = x_padded[b, :, l_start:l_start+K]
                    out[b, c_out, l_out] = np.sum(region * self.weight._data[c_out])
        
        if self.bias is not None:
            out = out + self.bias._data.reshape(1, -1, 1)
        
        return SimpTensor(out)


class MaxPool2d(Module):
    """二维最大池化层"""
    
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.return_indices = return_indices
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    """二维平均池化层"""
    
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class MaxPool1d(Module):
    """一维最大池化层"""
    
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C, L = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        
        x_padded = np.pad(x._data, ((0, 0), (0, 0), (P, P)), mode='constant')
        
        L_out = (L + 2 * P - K) // S + 1
        
        out = np.zeros((B, C, L_out))
        
        for b in range(B):
            for c in range(C):
                for l_out in range(L_out):
                    l_start = l_out * S
                    region = x_padded[b, c, l_start:l_start+K]
                    out[b, c, l_out] = np.max(region)
        
        return SimpTensor(out)


class BatchNorm1d(Module):
    """一维批归一化"""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = ones(num_features)
            self.weight._requires_grad = True
            self.register_parameter("weight", self.weight)
            
            self.bias = zeros(num_features)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.weight = None
            self.bias = None
        
        if track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        if self.training and self.track_running_stats:
            mean = np.mean(x._data, axis=0)
            var = np.var(x._data, axis=0)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x._data - mean) / np.sqrt(var + self.eps)
        
        if self.weight is not None:
            x_norm = x_norm * self.weight._data + self.bias._data
        
        return SimpTensor(x_norm)


class BatchNorm2d(Module):
    """二维批归一化"""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = ones(num_features)
            self.weight._requires_grad = True
            self.register_parameter("weight", self.weight)
            
            self.bias = zeros(num_features)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.weight = None
            self.bias = None
        
        if track_running_stats:
            self.running_mean = SimpTensor(np.zeros(num_features))
            self.running_var = SimpTensor(np.ones(num_features))
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        return batch_norm2d(
            x, 
            self.weight, 
            self.bias,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


class InstanceNorm2d(Module):
    """实例归一化"""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        if affine:
            self.weight = ones(num_features)
            self.weight._requires_grad = True
            self.register_parameter("weight", self.weight)
            
            self.bias = zeros(num_features)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C, H, W = x.shape
        
        mean = np.mean(x._data, axis=(2, 3), keepdims=True)
        var = np.var(x._data, axis=(2, 3), keepdims=True)
        
        x_norm = (x._data - mean) / np.sqrt(var + self.eps)
        
        if self.weight is not None:
            x_norm = x_norm * self.weight._data.reshape(1, -1, 1, 1) + self.bias._data.reshape(1, -1, 1, 1)
        
        return SimpTensor(x_norm)


class ConvTranspose2d(Module):
    """二维转置卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        kernel_shape = (in_channels, out_channels // groups, kernel_size, kernel_size)
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = randn(*kernel_shape) * std
        self.weight._requires_grad = True
        self.register_parameter("weight", self.weight)
        
        if bias:
            self.bias = zeros(out_channels)
            self.bias._requires_grad = True
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C_in, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding
        
        H_out = (H - 1) * S - 2 * P + K + OP
        W_out = (W - 1) * S - 2 * P + K + OP
        
        out = np.zeros((B, C_out, H_out, W_out))
        
        for b in range(B):
            for c_in in range(C_in):
                for h_in in range(H):
                    for w_in in range(W):
                        h_out_start = h_in * S - P
                        w_out_start = w_in * S - P
                        for k_h in range(K):
                            for k_w in range(K):
                                h_out = h_out_start + k_h
                                w_out = w_out_start + k_w
                                if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                    for c_out in range(C_out):
                                        out[b, c_out, h_out, w_out] += x._data[b, c_in, h_in, w_in] * self.weight._data[c_in, c_out, k_h, k_w]
        
        if self.bias is not None:
            out = out + self.bias._data.reshape(1, -1, 1, 1)
        
        return SimpTensor(out)


class Upsample(Module):
    """上采样层"""
    
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode: str = 'nearest',
        align_corners=None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        size = None
        scale_factor = None
        
        if self.size is not None:
            size = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        elif self.scale_factor is not None:
            scale_factor = self.scale_factor
        
        return interpolate(x, size=size, scale_factor=scale_factor, mode=self.mode)


class AdaptiveAvgPool2d(Module):
    """自适应平均池化层"""
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C, H, W = x.shape
        H_out, W_out = self.output_size
        
        out = np.zeros((B, C, H_out, W_out))
        
        for b in range(B):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = int(h_out * H / H_out)
                        h_end = int((h_out + 1) * H / H_out)
                        w_start = int(w_out * W / W_out)
                        w_end = int((w_out + 1) * W / W_out)
                        out[b, c, h_out, w_out] = np.mean(x._data[b, c, h_start:h_end, w_start:w_end])
        
        return SimpTensor(out)


class AdaptiveMaxPool2d(Module):
    """自适应最大池化层"""
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        B, C, H, W = x.shape
        H_out, W_out = self.output_size
        
        out = np.zeros((B, C, H_out, W_out))
        
        for b in range(B):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = int(h_out * H / H_out)
                        h_end = int((h_out + 1) * H / H_out)
                        w_start = int(w_out * W / W_out)
                        w_end = int((w_out + 1) * W / W_out)
                        out[b, c, h_out, w_out] = np.max(x._data[b, c, h_start:h_end, w_start:w_end])
        
        return SimpTensor(out)


class LSTM(Module):
    """长短期记忆网络"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                std = np.sqrt(1.0 / (layer_input_size + hidden_size))
                self.register_parameter(
                    f"weight_ih_l{layer}_d{direction}",
                    randn(4 * hidden_size, layer_input_size) * std
                )
                self.register_parameter(
                    f"weight_hh_l{layer}_d{direction}",
                    randn(4 * hidden_size, hidden_size) * std
                )
                
                if bias:
                    self.register_parameter(
                        f"bias_ih_l{layer}_d{direction}",
                        zeros(4 * hidden_size)
                    )
                    self.register_parameter(
                        f"bias_hh_l{layer}_d{direction}",
                        zeros(4 * hidden_size)
                    )
        
        self.dropout_layer = Dropout(dropout)
    
    def forward(
        self,
        x: SimpTensor,
        hx: Optional[Tuple[SimpTensor, SimpTensor]] = None,
    ) -> Tuple[SimpTensor, Tuple[SimpTensor, SimpTensor]]:
        if self.batch_first:
            x_data = np.transpose(x._data, (1, 0, 2))
        else:
            x_data = x._data
        
        seq_len, B, input_size = x_data.shape
        
        if hx is None:
            h_0 = np.zeros((self.num_layers * self.num_directions, B, self.hidden_size))
            c_0 = np.zeros((self.num_layers * self.num_directions, B, self.hidden_size))
        else:
            h_0, c_0 = hx
            h_0 = h_0._data
            c_0 = c_0._data
        
        outputs = []
        h_n = []
        c_n = []
        
        for layer in range(self.num_layers):
            layer_input = x_data if layer == 0 else outputs[-1]
            
            if layer > 0 and self.dropout > 0 and self.training:
                layer_input = self.dropout_layer(SimpTensor(layer_input))._data
            
            layer_outputs = []
            h_layer = h_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            c_layer = c_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            for direction in range(self.num_directions):
                h_dir = h_layer[direction]
                c_dir = c_layer[direction]
                
                w_ih = self._parameters[f"weight_ih_l{layer}_d{direction}"]._data
                w_hh = self._parameters[f"weight_hh_l{layer}_d{direction}"]._data
                
                if self.bias:
                    b_ih = self._parameters[f"bias_ih_l{layer}_d{direction}"]._data
                    b_hh = self._parameters[f"bias_hh_l{layer}_d{direction}"]._data
                else:
                    b_ih = np.zeros(4 * self.hidden_size)
                    b_hh = np.zeros(4 * self.hidden_size)
                
                dir_outputs = []
                for t in range(seq_len):
                    if direction == 0:
                        xt = layer_input[t]
                    else:
                        xt = layer_input[seq_len - 1 - t]
                    
                    gates = np.matmul(xt, w_ih.T) + np.matmul(h_dir, w_hh.T) + b_ih + b_hh
                    
                    i = sigmoid(gates[:, :self.hidden_size])
                    f = sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
                    g = np.tanh(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
                    o = sigmoid(gates[:, 3 * self.hidden_size:])
                    
                    c_dir = f * c_dir + i * g
                    h_dir = o * np.tanh(c_dir)
                    
                    dir_outputs.append(h_dir.copy())
                
                dir_outputs = np.stack(dir_outputs, axis=0)
                
                if direction == 1:
                    dir_outputs = dir_outputs[::-1]
                
                layer_outputs.append(dir_outputs)
                
                h_n.append(h_dir)
                c_n.append(c_dir)
            
            if self.num_directions == 2:
                layer_output = np.concatenate(layer_outputs, axis=-1)
            else:
                layer_output = layer_outputs[0]
            
            outputs.append(layer_output)
        
        output = outputs[-1]
        
        if self.batch_first:
            output = np.transpose(output, (1, 0, 2))
        
        h_n = np.stack(h_n, axis=0)
        c_n = np.stack(c_n, axis=0)
        
        return SimpTensor(output), (SimpTensor(h_n), SimpTensor(c_n))


class GRU(Module):
    """门控循环单元"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                std = np.sqrt(1.0 / (layer_input_size + hidden_size))
                self.register_parameter(
                    f"weight_ih_l{layer}_d{direction}",
                    randn(3 * hidden_size, layer_input_size) * std
                )
                self.register_parameter(
                    f"weight_hh_l{layer}_d{direction}",
                    randn(3 * hidden_size, hidden_size) * std
                )
                
                if bias:
                    self.register_parameter(
                        f"bias_ih_l{layer}_d{direction}",
                        zeros(3 * hidden_size)
                    )
                    self.register_parameter(
                        f"bias_hh_l{layer}_d{direction}",
                        zeros(3 * hidden_size)
                    )
        
        self.dropout_layer = Dropout(dropout)
    
    def forward(
        self,
        x: SimpTensor,
        hx: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, SimpTensor]:
        if self.batch_first:
            x_data = x._data.transpose(0, 1)
        else:
            x_data = x._data
        
        seq_len, B, input_size = x_data.shape
        
        if hx is None:
            h_0 = np.zeros((self.num_layers * self.num_directions, B, self.hidden_size))
        else:
            h_0 = hx._data
        
        outputs = []
        h_n = []
        
        for layer in range(self.num_layers):
            layer_outputs = []
            h_layer = h_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            for direction in range(self.num_directions):
                h_dir = h_layer[direction]
                
                w_ih = self._parameters[f"weight_ih_l{layer}_d{direction}"]._data
                w_hh = self._parameters[f"weight_hh_l{layer}_d{direction}"]._data
                
                if self.bias:
                    b_ih = self._parameters[f"bias_ih_l{layer}_d{direction}"]._data
                    b_hh = self._parameters[f"bias_hh_l{layer}_d{direction}"]._data
                else:
                    b_ih = np.zeros(3 * self.hidden_size)
                    b_hh = np.zeros(3 * self.hidden_size)
                
                dir_outputs = []
                for t in range(seq_len):
                    if direction == 0:
                        xt = x_data[t]
                    else:
                        xt = x_data[seq_len - 1 - t]
                    
                    gates = np.matmul(xt, w_ih.T) + np.matmul(h_dir, w_hh.T) + b_ih + b_hh
                    
                    r = sigmoid(gates[:, :self.hidden_size])
                    z = sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
                    n = np.tanh(gates[:, 2 * self.hidden_size:] + r * np.matmul(h_dir, w_hh[2 * self.hidden_size:].T))
                    
                    h_dir = (1 - z) * n + z * h_dir
                    
                    dir_outputs.append(h_dir.copy())
                
                dir_outputs = np.stack(dir_outputs, axis=0)
                
                if direction == 1:
                    dir_outputs = dir_outputs[::-1]
                
                layer_outputs.append(dir_outputs)
                h_n.append(h_dir)
            
            if self.num_directions == 2:
                layer_output = np.concatenate(layer_outputs, axis=-1)
            else:
                layer_output = layer_outputs[0]
            
            if layer < self.num_layers - 1 and self.dropout > 0 and self.training:
                layer_output = self.dropout_layer(SimpTensor(layer_output))._data
            
            outputs.append(layer_output)
            x_data = layer_output
        
        output = outputs[-1]
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        h_n = np.stack(h_n, axis=0)
        
        return SimpTensor(output), SimpTensor(h_n)


class TransformerEncoderLayer(Module):
    """Transformer编码器层"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model, layer_norm_eps)
        self.norm2 = LayerNorm(d_model, layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.activation = ReLU() if activation == 'relu' else GELU()
    
    def forward(
        self,
        src: SimpTensor,
        src_mask: Optional[SimpTensor] = None,
        src_key_padding_mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        x = src
        
        x2 = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class TransformerDecoderLayer(Module):
    """Transformer解码器层"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model, layer_norm_eps)
        self.norm2 = LayerNorm(d_model, layer_norm_eps)
        self.norm3 = LayerNorm(d_model, layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = ReLU() if activation == 'relu' else GELU()
    
    def forward(
        self,
        tgt: SimpTensor,
        memory: SimpTensor,
        tgt_mask: Optional[SimpTensor] = None,
        memory_mask: Optional[SimpTensor] = None,
        tgt_key_padding_mask: Optional[SimpTensor] = None,
        memory_key_padding_mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        x = tgt
        
        x2 = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.multihead_attn(x, memory, memory, mask=memory_mask)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        
        return x


class Transformer(Module):
    """Transformer架构"""
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        custom_encoder: Optional[Module] = None,
        custom_decoder: Optional[Module] = None,
    ):
        super().__init__()
        
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
                for _ in range(num_encoder_layers)
            ]
            self.encoder = Sequential(*encoder_layers)
        
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layers = [
                TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
                for _ in range(num_decoder_layers)
            ]
            self.decoder = Sequential(*decoder_layers)
    
    def forward(
        self,
        src: SimpTensor,
        tgt: SimpTensor,
        src_mask: Optional[SimpTensor] = None,
        tgt_mask: Optional[SimpTensor] = None,
        memory_mask: Optional[SimpTensor] = None,
        src_key_padding_mask: Optional[SimpTensor] = None,
        tgt_key_padding_mask: Optional[SimpTensor] = None,
        memory_key_padding_mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


class TransformerEncoder(Module):
    """Transformer编码器"""
    
    def __init__(
        self,
        encoder_layer: Module,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[encoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(
        self,
        src: SimpTensor,
        mask: Optional[SimpTensor] = None,
        src_key_padding_mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        output = self.layers(src)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(Module):
    """Transformer解码器"""
    
    def __init__(
        self,
        decoder_layer: Module,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[decoder_layer for _ in range(num_layers)])
        self.norm = norm
    
    def forward(
        self,
        tgt: SimpTensor,
        memory: SimpTensor,
        tgt_mask: Optional[SimpTensor] = None,
        memory_mask: Optional[SimpTensor] = None,
        tgt_key_padding_mask: Optional[SimpTensor] = None,
        memory_key_padding_mask: Optional[SimpTensor] = None,
    ) -> SimpTensor:
        output = tgt
        for layer in self.layers._modules.values():
            output = layer(output, memory)
        if self.norm is not None:
            output = self.norm(output)
        return output


class RNN(Module):
    """简单循环神经网络"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        self.activation = np.tanh if nonlinearity == 'tanh' else np.maximum
        
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                std = np.sqrt(1.0 / (layer_input_size + hidden_size))
                self.register_parameter(
                    f"weight_ih_l{layer}_d{direction}",
                    randn(hidden_size, layer_input_size) * std
                )
                self.register_parameter(
                    f"weight_hh_l{layer}_d{direction}",
                    randn(hidden_size, hidden_size) * std
                )
                
                if bias:
                    self.register_parameter(
                        f"bias_ih_l{layer}_d{direction}",
                        zeros(hidden_size)
                    )
                    self.register_parameter(
                        f"bias_hh_l{layer}_d{direction}",
                        zeros(hidden_size)
                    )
        
        self.dropout_layer = Dropout(dropout)
    
    def forward(
        self,
        x: SimpTensor,
        hx: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, SimpTensor]:
        if self.batch_first:
            x_data = x._data.transpose(0, 1)
        else:
            x_data = x._data
        
        seq_len, B, input_size = x_data.shape
        
        if hx is None:
            h_0 = np.zeros((self.num_layers * self.num_directions, B, self.hidden_size))
        else:
            h_0 = hx._data
        
        outputs = []
        h_n = []
        
        for layer in range(self.num_layers):
            layer_outputs = []
            h_layer = h_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            for direction in range(self.num_directions):
                h_dir = h_layer[direction]
                
                w_ih = self._parameters[f"weight_ih_l{layer}_d{direction}"]._data
                w_hh = self._parameters[f"weight_hh_l{layer}_d{direction}"]._data
                
                if self.bias:
                    b_ih = self._parameters[f"bias_ih_l{layer}_d{direction}"]._data
                    b_hh = self._parameters[f"bias_hh_l{layer}_d{direction}"]._data
                else:
                    b_ih = np.zeros(self.hidden_size)
                    b_hh = np.zeros(self.hidden_size)
                
                dir_outputs = []
                for t in range(seq_len):
                    if direction == 0:
                        xt = x_data[t]
                    else:
                        xt = x_data[seq_len - 1 - t]
                    
                    h_dir = self.activation(np.matmul(xt, w_ih.T) + np.matmul(h_dir, w_hh.T) + b_ih + b_hh)
                    dir_outputs.append(h_dir.copy())
                
                dir_outputs = np.stack(dir_outputs, axis=0)
                
                if direction == 1:
                    dir_outputs = dir_outputs[::-1]
                
                layer_outputs.append(dir_outputs)
                h_n.append(h_dir)
            
            if self.num_directions == 2:
                layer_output = np.concatenate(layer_outputs, axis=-1)
            else:
                layer_output = layer_outputs[0]
            
            if layer < self.num_layers - 1 and self.dropout > 0 and self.training:
                layer_output = self.dropout_layer(SimpTensor(layer_output))._data
            
            outputs.append(layer_output)
            x_data = layer_output
        
        output = outputs[-1]
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        h_n = np.stack(h_n, axis=0)
        
        return SimpTensor(output), SimpTensor(h_n)


class Flatten(Module):
    """展平层"""
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        from .tensor import flatten
        return flatten(x, self.start_dim, self.end_dim)


class Tanh(Module):
    """Tanh激活层"""
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        from .tensor import tanh
        return tanh(x)


class Sigmoid(Module):
    """Sigmoid激活层"""
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        from .tensor import sigmoid_tensor
        return sigmoid_tensor(x)


class Dropout2d(Module):
    """二维Dropout层"""
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.rand(1, x.shape[1], 1, 1) >= self.p) / (1 - self.p)
        result = x._data * mask
        
        out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
        
        if x.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                x._grad = SimpTensor(grad._data * mask)
            out._grad_fn = grad_fn
        
        return out


class Dropout1d(Module):
    """一维Dropout层"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: SimpTensor) -> SimpTensor:
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.rand(1, x.shape[1], 1) >= self.p) / (1 - self.p)
        result = x._data * mask
        
        out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
        
        if x.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                x._grad = SimpTensor(grad._data * mask)
            out._grad_fn = grad_fn
        
        return out


class BCEWithLogitsLoss(Module):
    """二元交叉熵损失（带sigmoid）"""
    
    def __init__(self, weight=None, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: SimpTensor, target: SimpTensor) -> SimpTensor:
        input_data = input._data
        target_data = target._data
        
        max_val = np.clip(input_data, 0, None)
        loss = input_data - input_data * target_data + max_val + np.log(np.exp(-max_val) + np.exp(-input_data - max_val))
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        out = SimpTensor(np.array([loss]), requires_grad=input.requires_grad)

        if input.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(input_data, -500, 500)))
                scale = 1.0 / input_data.size if self.reduction == 'mean' else 1.0
                grad_data = (sigmoid_input - target_data) * scale
                input.backward(SimpTensor(grad._data * grad_data))
            out._grad_fn = grad_fn

        return out


class MSELoss(Module):
    """均方误差损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: SimpTensor, target: SimpTensor) -> SimpTensor:
        diff = input._data - target._data
        loss = (diff ** 2)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        out = SimpTensor(np.array([loss]), requires_grad=input.requires_grad)

        if input.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                scale = 2.0 / input._data.size if self.reduction == 'mean' else 2.0
                input.backward(SimpTensor(grad._data * (scale * diff)))
            out._grad_fn = grad_fn

        return out


class L1Loss(Module):
    """L1损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: SimpTensor, target: SimpTensor) -> SimpTensor:
        loss = np.abs(input._data - target._data)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        out = SimpTensor(np.array([loss]), requires_grad=input.requires_grad)

        if input.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                scale = 1.0 / input._data.size if self.reduction == 'mean' else 1.0
                grad_data = np.sign(input._data - target._data) * scale
                input.backward(SimpTensor(grad._data * grad_data))
            out._grad_fn = grad_fn

        return out


class CrossEntropyLoss(Module):
    """交叉熵损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: SimpTensor, target: SimpTensor) -> SimpTensor:
        input_data = input._data
        target_data = target._data.astype(int)

        log_probs = -np.log(np.maximum(input_data, 1e-9))
        loss = log_probs[np.arange(len(target_data)), target_data]

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        out = SimpTensor(np.array([loss]), requires_grad=input.requires_grad)

        if input.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                grad_data = np.zeros_like(input_data)
                grad_data[np.arange(len(target_data)), target_data] = -1.0 / np.maximum(input_data[np.arange(len(target_data)), target_data], 1e-9)
                if self.reduction == 'mean':
                    grad_data /= len(target_data)
                input.backward(SimpTensor(grad._data * grad_data))
            out._grad_fn = grad_fn

        return out


class BCELoss(Module):
    """二分类交叉熵损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: SimpTensor, target: SimpTensor) -> SimpTensor:
        input_data = np.clip(input._data, 1e-9, 1 - 1e-9)
        target_data = target._data
        
        loss = -(target_data * np.log(input_data) + (1 - target_data) * np.log(1 - input_data))
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        out = SimpTensor(np.array([loss]), requires_grad=input.requires_grad)

        if input.requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                scale = 1.0 / input_data.size if self.reduction == 'mean' else 1.0
                grad_data = (input_data - target_data) / (input_data * (1 - input_data)) * scale
                input.backward(SimpTensor(grad._data * grad_data))
            out._grad_fn = grad_fn

        return out


if __name__ == "__main__":
    print("=" * 60)
    print("SimpNN 自检")
    print("=" * 60)
    
    linear = Linear(3, 5)
    x = randn(2, 3)
    y = linear(x)
    print(f"Linear: input (2,3) -> output {y.shape}")
    
    embedding = Embedding(100, 16)
    idx = SimpTensor(np.array([[1, 2, 3], [4, 5, 6]]))
    emb_out = embedding(idx)
    print(f"Embedding: input (2,3) -> output {emb_out.shape}")
    
    ln = LayerNorm(3)
    ln_out = ln(x)
    print(f"LayerNorm: input (2,3) -> output {ln_out.shape}")
    
    mha = MultiHeadAttention(16, 2)
    q = k = v = randn(2, 4, 16)
    attn_out = mha(q, k, v)
    print(f"MultiHeadAttention: input (2,4,16) -> output {attn_out.shape}")
    
    ff = FeedForward(16, 32)
    ff_out = ff(q)
    print(f"FeedForward: input (2,4,16) -> output {ff_out.shape}")
    
    rfp = SpectralDiffusionLayer(16, 4)
    sgc_out = rfp(q)
    print(f"SpectralDiffusionLayer: input (2,4,16) -> output {sgc_out.shape}")

    dmc = ExternalMemoryLayer(16, 4, 8, 2)
    dmc_out = dmc(q)
    print(f"ExternalMemoryLayer: input (2,4,16) -> output {dmc_out.shape}")

    tar = AdaptiveDepthRouter(16, 4, 2, 4, 2)
    ccr_out = tar(q)
    print(f"AdaptiveDepthRouter: input (2,4,16) -> output {ccr_out.shape}")
    
    block = SimplCoreBlock(
        dim=16,
        num_heads=2,
        num_waves=4,
        num_channels=4,
        num_paths=4,
        channel_size=8,
        min_depth=2,
        max_depth=4,
        topk=2,
        ffn_intermediate_ratio=2.0,
    )
    block_out = block(q)
    print(f"SimplCoreBlock: input (2,4,16) -> output {block_out.shape}")

    print("\nSimpNN 自检 PASSED")


# ---------------------------------------------------------------------------
# 向后兼容别名（deprecated）
# 以下名称曾使用浮夸/误导性的命名（"共振场"/"动态记忆通道"/"条件计算路由"），
# 已在 v1.x 重命名为如实描述其真实机制的 SpectralDiffusionLayer /
# ExternalMemoryLayer / AdaptiveDepthRouter。旧名保留为别名，将在未来版本移除。
# ---------------------------------------------------------------------------
LegacySpectralMixer = SpectralDiffusionLayer
DynamicMemoryChannel = ExternalMemoryLayer
LegacyConditionalRouter = AdaptiveDepthRouter


