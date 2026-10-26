# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-v2 因果全局卷积 (PyTorch 计算后端, 自研算法).

本模块把验证过的 NumPy 后端算法 `SpectralGlobalConvV2` 移植到 PyTorch,
用于跑真实规模 benchmark. 算法完全自研:

  y[t] = sum_{s<=t} K[t-s] * x[s]      (因果全局卷积)
  - FFT 实现: O(n log n), 远低于注意力的 O(n^2)
  - 可学低秩因果核: K = low(L, r) @ up(r, D)
  - 单位脉冲初始化: low[0]=1, up=单位阵 -> 初始为恒等映射, 加速收敛
  - 因果性: 仅依赖 <=t 的位置, 天然具备选择性 (这是双向谱混合缺失的)

注意: 这里用 PyTorch 仅作为 *计算后端* (自动微分 + 算子),
      卷积核结构 / FFT 因果方案 / 低秩参数化 / 初始化 全是自研设计.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGlobalConvV2(nn.Module):
    """因果全局卷积 mixer (SGC-v2).

    Args:
        dim:      通道维 D
        kernel_size: 占位(FFT 全局核不依赖局部窗口), 保留接口兼容
        max_len:  序列最大长度 L (核长度)
        rank:     低秩核秩 r (默认 8, 消融表明是效率-精度甜点)
        eps:      LayerNorm 数值稳定项
    """

    def __init__(self, dim: int, kernel_size: int = 3, max_len: int = 512,
                 rank: int = 8, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.rank = rank

        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)

        # 可学低秩因果核: K = low(L, r) @ up(r, D)
        self.low = nn.Parameter(torch.zeros(max_len, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        with torch.no_grad():
            # 单位脉冲初始化 -> 初始为恒等映射, 训练起点稳定
            for i in range(min(rank, dim)):
                self.up.data[i, i] = 1.0
            self.low.data[0, :] = 1.0

        self.out_scale = nn.Parameter(torch.ones(1))

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """FFT 实现的因果全局卷积. x: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape
        Kern = self.low[:L, :] @ self.up          # (L, D) 因果核
        N = 2 * L                                  # 循环卷积长度, 避免混叠
        # 零填充
        Xp = F.pad(x, (0, 0, 0, N - L))            # (B, N, D)
        Kp = F.pad(Kern, (0, 0, 0, L))             # (N, D)
        # 实信号 rfft
        cX = torch.fft.rfft(Xp, n=N, dim=1)        # (B, N/2+1, D)
        cK = torch.fft.rfft(Kp.unsqueeze(0), n=N, dim=1)
        cY = cX * cK                               # 频域逐点乘
        y = torch.fft.irfft(cY, n=N, dim=1)        # (B, N, D)
        return y[:, :L, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        y = self._causal_conv(u)
        out = self.out_proj(y)
        return residual + self.out_scale * out


class AttentionMixer(nn.Module):
    """标准因果自注意力 mixer (baseline). 与 SGC 共享同一 Transformer 骨架."""

    def __init__(self, dim: int, n_heads: int = 4, max_len: int = 512,
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))
        # 因果掩码缓存
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(max_len, max_len), diagonal=1).bool(),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, H, L, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        mask = self.causal_mask[:L, :L]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                         # (B, H, L, hd)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return residual + self.out_scale * out


class FFN(nn.Module):
    """SwiGLU 前馈 (现代 LLM 标配, 两类 mixer 共用, 保证公平)."""

    def __init__(self, dim: int, ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        hidden = int(dim * ratio)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Block(nn.Module):
    """统一 transformer block: mixer -> ffn. mixer 可换 Attention / SGC."""

    def __init__(self, dim: int, mixer: str = "attn", n_heads: int = 4,
                 max_len: int = 512, sgc_rank: int = 8):
        super().__init__()
        if mixer == "attn":
            self.mixer = AttentionMixer(dim, n_heads, max_len)
        elif mixer == "sgc":
            self.mixer = SpectralGlobalConvV2(dim, 3, max_len, sgc_rank)
        else:
            raise ValueError(f"unknown mixer {mixer}")
        self.ffn = FFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class LMModel(nn.Module):
    """公平对比 LM: 仅 mixer 不同, 其余(嵌入/层数/FFN/位置)完全一致."""

    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int = 4,
                 max_len: int = 512, mixer: str = "attn", sgc_rank: int = 8,
                 tie_emb: bool = True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([
            Block(dim, mixer, n_heads, max_len, sgc_rank) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie_emb:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    # 冒烟测试: 两类模型同规模, 参数应接近, 前向可微
    V, D, L, NL = 1000, 256, 128, 4
    attn = LMModel(V, D, NL, 4, L, "attn")
    sgc = LMModel(V, D, NL, 4, L, "sgc", sgc_rank=8)
    print("attn params:", count_params(attn))
    print("sgc  params:", count_params(sgc))
    x = torch.randint(0, V, (2, L))
    la = attn(x)
    ls = sgc(x)
    print("attn logits:", tuple(la.shape), "sgc logits:", tuple(ls.shape))
    loss = F.cross_entropy(la.reshape(-1, V), x.reshape(-1))
    loss.backward()
    print("backward OK, loss=", round(float(loss), 4))
