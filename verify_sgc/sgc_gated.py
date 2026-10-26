# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-v2 + 门控扩展 (GatedSGC) —— 补偿内容相关路由缺失.

背景: 论文引理3预言, SGC-v2 的位置相对核在高熵真实语言建模上因无法
      内容相关路由而泛化受限. 本模块在保持 O(n log n) 复杂度的前提下,
      引入轻量门控头:

        g_t = sigmoid(W_g x_t)          # (B, L, D) 逐点门控
        y_t = g_t * (K * u)_t           # 按内容动态调整卷积核参与度

      设计要点:
        1. 门控是逐点乘, 不破坏 FFT 因果卷积的 O(n log n) 复杂度
        2. 门控头参数 O(D^2), 远小于注意力, 参数量增长可控
        3. 初始化为 sigmoid(0)=0.5, 即初始为"半开"门控, 训练起点稳定
        4. 可随时关掉门控(use_gate=False)回到纯 SGC-v2, 便于消融对比

本文件是论文 4.4 节"门控消融"的实验代码, 与 verify_sgc/sgc_pytorch.py
完全独立, 不改动原有文件, 方便回溯.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedSGC(nn.Module):
    """门控因果全局卷积 mixer (SGC-v2 + 内容相关门控).

    Args:
        dim:       通道维 D
        max_len:   序列最大长度 L
        rank:      低秩核秩 r (默认 8)
        use_gate:  是否启用内容门控 (False 时退化为纯 SGC-v2)
        eps:       LayerNorm 数值稳定项
    """

    def __init__(self, dim: int, max_len: int = 512, rank: int = 8,
                 use_gate: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.rank = rank
        self.use_gate = use_gate

        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)

        # 可学低秩因果核: K = low(L, r) @ up(r, D)
        self.low = nn.Parameter(torch.zeros(max_len, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        with torch.no_grad():
            for i in range(min(rank, dim)):
                self.up.data[i, i] = 1.0
            self.low.data[0, :] = 1.0

        self.out_scale = nn.Parameter(torch.ones(1))

        # 内容门控头: g_t = sigmoid(W_g x_t)
        # 初始化为 0 -> sigmoid(0)=0.5, 初始半开门控, 训练稳定
        if self.use_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """FFT 实现因果全局卷积. x: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape
        Kern = self.low[:L, :] @ self.up          # (L, D) 因果核
        N = 2 * L                                  # 循环卷积长度, 避免混叠
        Xp = F.pad(x, (0, 0, 0, N - L))            # (B, N, D)
        Kp = F.pad(Kern, (0, 0, 0, L))             # (N, D)
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

        # 内容相关门控: 按当前 token 内容动态调制卷积输出
        if self.use_gate:
            g = torch.sigmoid(self.gate_proj(h))   # (B, L, D), g∈(0,1)
            y = y * g

        out = self.out_proj(y)
        return residual + self.out_scale * out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, max_len={self.max_len}, "
                f"rank={self.rank}, use_gate={self.use_gate}")


class GatedBlock(nn.Module):
    """统一 transformer block: mixer -> ffn. 支持 attn / sgc / gated_sgc."""

    def __init__(self, dim: int, mixer: str = "attn", n_heads: int = 4,
                 max_len: int = 512, sgc_rank: int = 8):
        super().__init__()
        if mixer == "attn":
            from sgc_pytorch import AttentionMixer
            self.mixer = AttentionMixer(dim, n_heads, max_len)
        elif mixer == "sgc":
            from sgc_pytorch import SpectralGlobalConvV2
            self.mixer = SpectralGlobalConvV2(dim, 3, max_len, sgc_rank)
        elif mixer == "gated_sgc":
            self.mixer = GatedSGC(dim, max_len, sgc_rank, use_gate=True)
        elif mixer == "sgc_nogate":
            self.mixer = GatedSGC(dim, max_len, sgc_rank, use_gate=False)
        else:
            raise ValueError(f"unknown mixer {mixer}")
        self.ffn = FFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class FFN(nn.Module):
    """SwiGLU 前馈 (与 sgc_pytorch 保持一致)."""

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


class GatedLMModel(nn.Module):
    """公平对比 LM: 仅 mixer 不同, 其余完全一致."""

    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int = 4,
                 max_len: int = 512, mixer: str = "gated_sgc",
                 sgc_rank: int = 8, tie_emb: bool = True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([
            GatedBlock(dim, mixer, n_heads, max_len, sgc_rank)
            for _ in range(n_layers)
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
    # 冒烟测试: 验证门控版可前向、可反向、参数合理
    V, D, L, NL = 1000, 256, 128, 4
    gated = GatedLMModel(V, D, NL, 4, L, "gated_sgc", sgc_rank=8)
    nogate = GatedLMModel(V, D, NL, 4, L, "sgc_nogate", sgc_rank=8)
    print("gated_sgc params:", count_params(gated))
    print("sgc_nogate params:", count_params(nogate))
    print("gate params added:", count_params(gated) - count_params(nogate))

    x = torch.randint(0, V, (2, L))
    lg = gated(x)
    ln = nogate(x)
    print("gated logits:", tuple(lg.shape), "nogate logits:", tuple(ln.shape))

    loss = F.cross_entropy(lg.reshape(-1, V), x.reshape(-1))
    loss.backward()
    print("backward OK, loss=", round(float(loss), 4))

    # 验证门控梯度可达
    gate_grad = gated.blocks[0].mixer.gate_proj.weight.grad
    print("gate_proj grad norm:", float(gate_grad.norm()))
    print("门控扩展验证通过 ✅")
