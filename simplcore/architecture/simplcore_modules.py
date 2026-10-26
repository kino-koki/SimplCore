# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 自研架构模块
将核心架构组件从 torch 移植到 SimplCore 自研引擎
包含: LegacySpectralMixer, DynamicMemoryChannel, LegacyConditionalRouter
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List, Dict

from ..engine.tensor import (
    SimpTensor, randn, zeros, ones, softmax, gelu, layer_norm, dropout,
    sigmoid_tensor, softplus, clamp, abs as tensor_abs,
    einsum, manual_seed, allclose, topk, gather, stack
)
from ..engine.nn import Module, Linear, LayerNorm


class DynamicMemoryChannel(Module):
    """动态记忆通道 (Dynamic Memory Channel) - 自研版本.

    维护固定大小记忆库 (num_channels, channel_size, dim), 通过内容寻址读写。
    记忆复杂度 O(1) (相对于序列长度 n)。

    Args:
        dim: 特征维度 d。
        num_channels: 记忆通道数 C (并行记忆库数量)。
        channel_size: 每个通道的 slot 数 S。
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 64,
        channel_size: int = 128,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_channels > 0 and channel_size > 0
        self.dim = dim
        self.num_channels = num_channels
        self.channel_size = channel_size
        self.num_heads = num_heads if dim % max(num_heads, 1) == 0 and num_heads > 0 else 1
        self.head_dim = dim // self.num_heads

        self.query_proj = Linear(dim, dim, bias=False)
        self.register_module("query_proj", self.query_proj)
        
        self.key_proj = Linear(dim, dim, bias=False)
        self.register_module("key_proj", self.key_proj)
        
        self.write_value_proj = Linear(dim, dim, bias=True)
        self.register_module("write_value_proj", self.write_value_proj)
        
        self.output_proj = Linear(dim, dim, bias=True)
        self.register_module("output_proj", self.output_proj)
        
        self.write_gate = Linear(dim, num_channels, bias=True)
        self.register_module("write_gate", self.write_gate)
        
        self.channel_weights = ones(num_channels)
        self.channel_weights._requires_grad = True
        self.register_parameter("channel_weights", self.channel_weights)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

        self._memory: Optional[SimpTensor] = None
        self._batch_size: int = 0

    def reset_state(self, batch_size: int) -> None:
        """重置记忆库为新序列做准备."""
        self._memory = randn(batch_size, self.num_channels, self.channel_size, self.dim) * 0.02
        self._batch_size = batch_size

    def get_state(self) -> Optional[SimpTensor]:
        """获取当前记忆库状态."""
        return self._memory

    def set_state(self, memory: Optional[SimpTensor]) -> None:
        """设置记忆库状态."""
        self._memory = memory
        if memory is not None:
            self._batch_size = memory.shape[0]

    def detach_state(self) -> None:
        """截断 BPTT: 将记忆从计算图分离."""
        if self._memory is not None:
            self._memory = SimpTensor(self._memory._data)

    def _ensure_memory(self, batch_size: int) -> SimpTensor:
        """惰性确保记忆库已按批次大小分配."""
        if self._memory is None or self._batch_size != batch_size:
            self.reset_state(batch_size)
        return self._memory

    def read(self, query: SimpTensor) -> SimpTensor:
        """基于 query 的多头注意力读取."""
        B = query.shape[0]
        memory = self._ensure_memory(B)
        H, hd = self.num_heads, self.head_dim

        q = self.query_proj(query).reshape(B, H, hd)
        keys = self.key_proj(memory).reshape(B, self.num_channels, self.channel_size, H, hd)

        scores = einsum("bhd,bcshd->bchs", q, keys) / math.sqrt(hd)
        weights = softmax(scores, dim=-1)

        mem_view = memory.reshape(B, self.num_channels, self.channel_size, H, hd)
        readout = einsum("bchs,bcshd->bchd", weights, mem_view)
        return readout.reshape(B, self.num_channels, self.dim)

    def write(self, key: SimpTensor, value: SimpTensor) -> None:
        """基于门的写入 (NTM 风格: 先擦除再添加)."""
        B = key.shape[0]
        memory = self._ensure_memory(B)
        H, hd = self.num_heads, self.head_dim

        k = self.query_proj(key).reshape(B, H, hd)
        v = self.write_value_proj(value)

        gate = sigmoid_tensor(self.write_gate(key))

        keys = self.key_proj(memory).reshape(B, self.num_channels, self.channel_size, H, hd)
        scores = einsum("bhd,bcshd->bchs", k, keys) / math.sqrt(hd)
        w = softmax(scores, dim=-1).mean(dim=2)

        g = gate.unsqueeze(-1).unsqueeze(-1)
        w_exp = w.unsqueeze(-1)
        v_exp = v.unsqueeze(1).unsqueeze(2)

        self._memory = memory * (1.0 - g * w_exp) + g * w_exp * v_exp
        self._memory = clamp(self._memory, min=-10.0, max=10.0)

    def forward(self, x: SimpTensor) -> SimpTensor:
        """处理输入序列: 对每步执行 read + write, 返回融合记忆的输出."""
        residual = x
        B, L, D = x.shape
        assert D == self.dim, f"dim 不匹配: 期望 {self.dim}, 得到 {D}"

        outputs = []
        
        for t in range(L):
            xt = x[:, t, :]
            readout = self.read(xt)
            self.write(xt, xt)
            cw = softmax(self.channel_weights, dim=0)
            agg = einsum("c,bcd->bd", cw, readout)
            out_t = self.output_proj(agg)
            outputs.append(out_t)

        output = stack(outputs, dim=1)
        return residual + self.norm(output)


class LegacySpectralMixer(Module):
    """共振场处理层 (Resonance Field Processing Layer) - 自研版本.

    用可学习的波场演化替代注意力机制, 复杂度 O(n log n)。

    Args:
        dim: 输入 / 输出特征维度。
        num_waves: 波场通道数 W (每个通道一个共振频率)。
        damping: 衰减系数 β 的初始值。
        num_steps: 场演化步数 T。
        dt: 离散化时间步长。
        sigma: 共振带通高斯窗的带宽。
    """

    def __init__(
        self,
        dim: int,
        num_waves: int = 8,
        damping: float = 0.1,
        num_steps: int = 3,
        dt: float = 0.1,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_waves > 0
        self.dim = dim
        self.num_waves = num_waves
        self.num_steps = num_steps
        self.dt = dt
        self.sigma = sigma

        self.omega_raw = randn(num_waves) * 0.3
        self.omega_raw._requires_grad = True
        self.register_parameter("omega_raw", self.omega_raw)

        alpha_val = np.array([math.log(math.expm1(damping)) for _ in range(num_waves)])
        self.alpha_raw = SimpTensor(alpha_val)
        self.alpha_raw._requires_grad = True
        self.register_parameter("alpha_raw", self.alpha_raw)

        self.gamma = ones(num_waves) * 0.5
        self.gamma._requires_grad = True
        self.register_parameter("gamma", self.gamma)

        self.beta = randn(1) + damping
        self.beta._requires_grad = True
        self.register_parameter("beta", self.beta)

        self.in_proj = Linear(dim, num_waves, bias=True)
        self.out_proj = Linear(num_waves, dim, bias=True)

        self.norm = LayerNorm(dim)

        self.out_scale = ones(1)
        self.out_scale._requires_grad = True
        self.register_parameter("out_scale", self.out_scale)

    @property
    def omega(self) -> SimpTensor:
        """共振频率 ω (非负)."""
        return softplus(self.omega_raw)

    @property
    def alpha(self) -> SimpTensor:
        """扩散/阻尼系数 α (非负)."""
        return softplus(self.alpha_raw)

    def _laplacian(self, field: SimpTensor) -> SimpTensor:
        """沿序列维度用 FFT 计算一维拉普拉斯算子 ∇²F."""
        from ..engine.fft import fft, ifft
        
        L = field.shape[1]
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)
        k_sq = SimpTensor(freqs ** 2)

        field_fft = fft(field, dim=1)
        lap_fft = field_fft * (-k_sq).reshape(1, -1, 1)
        lap = ifft(lap_fft, dim=1).real

        return lap

    def forward(self, x: SimpTensor) -> SimpTensor:
        """共振场处理前向传播."""
        residual = x
        h = self.norm(x)
        B, L, _ = h.shape

        I_field = self.in_proj(h)

        from ..engine.fft import fft, ifft
        
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)
        omega = self.omega

        band = SimpTensor(np.exp(-((np.abs(freqs.reshape(-1, 1)) - omega._data.reshape(1, -1)) ** 2) / (2.0 * self.sigma ** 2)))

        F_field = ifft(fft(I_field, dim=1) * band.unsqueeze(0), dim=1).real

        alpha = self.alpha.reshape(1, 1, -1)
        gamma = self.gamma.reshape(1, 1, -1)
        beta = self.beta

        for _ in range(self.num_steps):
            lap = self._laplacian(F_field)
            F_field = F_field + self.dt * (alpha * lap - beta * F_field + gamma * I_field)

        out = self.out_proj(F_field)
        return residual + self.out_scale * out

    def resonance_complexity_analysis(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """分析本层的计算复杂度并与标准注意力对比."""
        n, d, w = seq_len, self.dim, self.num_waves
        T = self.num_steps
        fft_cost = 5.0 * n * math.log2(max(n, 2))
        proj_cost = 2.0 * n * d * w
        sgc_flops = (proj_cost + T * (fft_cost + n * w)) * batch_size
        attn_flops = 2.0 * n * n * d * batch_size
        speedup = attn_flops / max(sgc_flops, 1.0)
        return {
            "seq_len": n,
            "sgc_flops": sgc_flops,
            "attn_flops": attn_flops,
            "speedup_vs_attention": speedup,
            "sgc_big_o": "O(n log n)",
            "attn_big_o": "O(n^2)",
            "note": "SpectralMixer 用 FFT 计算拉普拉斯算子, 单层复杂度 O(n log n); 标准注意力为 O(n^2).",
        }


class LegacyConditionalRouter(Module):
    """条件计算路由器 (Conditional Compute Router) - 自研版本.

    每个 token 动态选择计算路径, 不同路径对应不同计算深度。

    Args:
        dim: 特征维度。
        num_paths: 候选路径数 P (每条路径对应一个深度)。
        min_depth: 最小计算深度。
        max_depth: 最大计算深度。
        topk: top-k 路由的 k。
        dropout: 路由 logits 的 dropout。
    """

    def __init__(
        self,
        dim: int,
        num_paths: int = 4,
        min_depth: int = 2,
        max_depth: int = 8,
        topk: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_paths > 0
        assert 1 <= min_depth <= max_depth
        assert 1 <= topk <= num_paths
        self.dim = dim
        self.num_paths = num_paths
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.topk = topk
        self.dropout = dropout

        self.router = Linear(dim, num_paths, bias=True)
        self.transform = Linear(dim, dim, bias=True)
        self.path_depths: List[int] = self._compute_path_depths()

        self.norm = LayerNorm(dim)
        self.out_norm = LayerNorm(dim)

        self.temperature = ones(1)
        self.temperature._requires_grad = True
        self.register_parameter("temperature", self.temperature)

    def _compute_path_depths(self) -> List[int]:
        """将路径均匀映射到 [min_depth, max_depth] 区间."""
        if self.num_paths == 1:
            return [self.min_depth]
        depths: List[int] = []
        for p in range(self.num_paths):
            d = self.min_depth + round(p * (self.max_depth - self.min_depth) / (self.num_paths - 1))
            depths.append(int(d))
        return depths

    def route(self, x: SimpTensor) -> SimpTensor:
        """计算每个 token 对各路径的路由概率."""
        logits = self.router(x)
        if self.training and self.dropout > 0:
            logits = dropout(logits, p=self.dropout, training=self.training)
        t = clamp(self.temperature, min=1e-3)
        probs = softmax(logits / t, dim=-1)
        return probs

    def forward(self, x: SimpTensor) -> SimpTensor:
        """路由 token 到不同计算深度路径并加权融合."""
        residual = x
        h = self.norm(x)
        B, L, D = h.shape

        probs = self.route(h)

        max_d = max(self.path_depths)
        depth_states = {0: h}
        current = h
        for d in range(1, max_d + 1):
            current = current + gelu(self.transform(current))
            depth_states[d] = current

        path_states = stack([depth_states[d] for d in self.path_depths], dim=2)

        if self.topk >= self.num_paths:
            out = einsum("blp,blpd->bld", probs, path_states)
        else:
            topk_probs, topk_idx = topk(probs, k=self.topk, dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            idx = topk_idx.unsqueeze(-1).expand((-1, -1, -1, D))
            topk_states = gather(path_states, dim=2, index=idx)
            out = einsum("blk,blkd->bld", topk_probs, topk_states)

        out = self.out_norm(out)
        return residual + out

    def expected_depth(self, x: SimpTensor) -> SimpTensor:
        """计算每个 token 的期望计算深度."""
        probs = self.route(x)
        depths = SimpTensor(np.array(self.path_depths, dtype=np.float32))
        return (probs * depths).sum(dim=-1)


if __name__ == "__main__":
    manual_seed(0)
    print("=== 测试 DynamicMemoryChannel ===")
    dmc = DynamicMemoryChannel(dim=32, num_channels=4, channel_size=8, num_heads=2)
    x = randn(2, 10, 32, requires_grad=True)
    y = dmc(x)
    print(f"output shape: {tuple(y.shape)}")
    assert y.shape == (2, 10, 32)
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and tensor_abs(p.grad).sum().item() > 0 for p in dmc.parameters())
    print(f"has gradient: {has_grad}")
    assert has_grad
    state = dmc.get_state()
    print(f"memory state shape: {tuple(state.shape)}")
    assert state.shape == (2, 4, 8, 32)
    print("DMC test PASSED\n")

    manual_seed(0)
    print("=== 测试 LegacySpectralMixer ===")
    layer = LegacySpectralMixer(dim=32, num_waves=4, damping=0.1, num_steps=3)
    x = randn(2, 16, 32, requires_grad=True)
    y = layer(x)
    print(f"output shape: {tuple(y.shape)}")
    assert y.shape == (2, 16, 32)
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and tensor_abs(p.grad).sum().item() > 0 for p in layer.parameters())
    print(f"has gradient: {has_grad}")
    assert has_grad
    analysis = layer.resonance_complexity_analysis(seq_len=4096)
    print(f"speedup vs attention: {analysis['speedup_vs_attention']:.1f}x")
    print("SpectralMixer test PASSED\n")

    manual_seed(0)
    print("=== 测试 LegacyConditionalRouter ===")
    router = LegacyConditionalRouter(dim=32, num_paths=4, min_depth=2, max_depth=8, topk=2)
    x = randn(2, 16, 32, requires_grad=True)
    y = router(x)
    print(f"output shape: {tuple(y.shape)}")
    assert y.shape == (2, 16, 32)
    probs = router.route(x)
    print(f"route probs shape: {tuple(probs.shape)}")
    assert probs.shape == (2, 16, 4)
    ed = router.expected_depth(x)
    print(f"expected depth (mean): {ed.mean()._data}")
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and tensor_abs(p.grad).sum().item() > 0 for p in router.parameters())
    print(f"has gradient: {has_grad}")
    assert has_grad
    print("CCR test PASSED\n")

    print("=== 所有测试通过 ===")


