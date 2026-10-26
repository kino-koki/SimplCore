# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""统一序列混合算子库 (PyTorch 自研计算后端).

本模块把"序列混合"这一 LLM 核心算子做成一个**公平、同骨架**的对比实验台,
用于支撑 SGC 的颠覆性主张:

  > 所有次二次混合算子(线性注意力、SSM/Mamba)都把历史压缩进一个
  > *有界状态*, 从而在长程召回上随序列增长退化; 而 SGC 的"状态"随
  > 序列长度 n 增长(核长 = n), 无信息瓶颈, 可在 O(n log n) 下达到
  > 注意力级长程召回.

收录的 5 个 mixer (共享 Embedding/LayerNorm/FFN/位置编码, 仅 mixer 不同):
  - attn      : 标准因果自注意力        (状态 Θ(n·D), O(n^2), 无瓶颈)
  - sgc       : 因果全局卷积 (SGC-v2)   (状态 n·D,        O(n log n), 无瓶颈)
  - sgc_ag    : 自适应门控全局卷积(新)   (状态 n·D,        O(n log n), 无瓶颈, 内容自适应)
  - linear    : 线性注意力             (状态 D^2,        O(n·D), 有界瓶颈)
  - ssm       : Mamba式选择性扫描SSM    (状态 N·D 固定,   O(n·N·D), 有界瓶颈)

设计要点:
  - 自适应门控(SGC-AG)在全局卷积骨架上加一个内容依赖门控, 让全局卷积
    恢复注意力式选择性, 同时保持次二次复杂度 —— 占据"全局+自适应+次二次"
    这一此前空缺的设计点.
  - SSM 采用选择性扫描(selective scan)的*循环(recurrent)形式*而非卷积形式,
    这样才能暴露其"有界状态"的信息瓶颈; 若写成卷积等价形式就退化成全局卷积了.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用已验证的 SGC-v2 / Attention 实现与计数工具 (单一事实来源)
from sgc_pytorch import (SpectralGlobalConvV2, AttentionMixer,
                         FFN, Block, LMModel, count_params)


# --------------------------------------------------------------------------
# 共享工具: FFT 因果全局卷积 (所有基于全局卷积的 mixer 共用)
# --------------------------------------------------------------------------
def causal_conv_fft(u: torch.Tensor, low: torch.Tensor, up: torch.Tensor,
                    L: int) -> torch.Tensor:
    """u:(B,L,D) -> y:(B,L,D).  y[t]=sum_{s<=t} K[t-s] u[s], K=low[:L]@up."""
    # (L, D) 因果核
    Kern = low[:L, :] @ up
    N = 2 * L
    Xp = F.pad(u, (0, 0, 0, N - L))           # (B, N, D)
    Kp = F.pad(Kern, (0, 0, 0, L))            # (N, D)
    cX = torch.fft.rfft(Xp, n=N, dim=1)
    cK = torch.fft.rfft(Kp.unsqueeze(0), n=N, dim=1)
    cY = cX * cK
    y = torch.fft.irfft(cY, n=N, dim=1)
    return y[:, :L, :]


# --------------------------------------------------------------------------
# (1) SGC-AG: 自适应门控全局卷积  ★ 本次新增的颠覆性机制
# --------------------------------------------------------------------------
class SGCAdaptiveGating(nn.Module):
    """在 SGC-v2 因果全局卷积骨架上加入内容自适应门控.

    动机: 原始 SGC 的核是*内容无关*的(每个相对偏移用同一个核系数),
    缺少注意力"按内容选择性聚合"的能力, 因此在小数据 LM 上更易过拟合、
    质量略逊. SGC-AG 在全局卷积输出的基础上, 用一个轻量的内容依赖门控
    做"全局上下文 vs 局部残差"的自适应混合:

        v   = causal_conv(in_proj(norm(x)))      # 全历史混合, O(n log n)
        g   = sigmoid(gate_proj(in_proj(x)))      # 内容依赖门 (B,L,D)
        out = g * v + (1 - g) * in_proj(x)        # 自适应混合

    仅多一个 D->D 线性层, 复杂度仍为 O(n log n + n·D) (次二次),
    但恢复了逐 token 选择性. 这填补了"全局感受野 + 内容自适应 + 次二次"
    这一此前被注意力(二次)与 SSM/线性注意力(有界状态)共同遗漏的设计点.
    """

    def __init__(self, dim: int, max_len: int = 512, rank: int = 8, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.rank = rank
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)

        self.low = nn.Parameter(torch.zeros(max_len, rank))
        self.up = nn.Parameter(torch.zeros(rank, dim))
        with torch.no_grad():
            for i in range(min(rank, dim)):
                self.up.data[i, i] = 1.0
            self.low.data[0, :] = 1.0
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        v = causal_conv_fft(u, self.low, self.up, x.shape[1])   # 全历史
        g = torch.sigmoid(self.gate(u))                         # 内容依赖门
        out = g * v + (1.0 - g) * u                             # 自适应混合
        out = self.out_proj(out)
        return residual + self.out_scale * out


# --------------------------------------------------------------------------
# (2) Linear Attention: 有界状态 (D^2) 的次二次基线
# --------------------------------------------------------------------------
class LinearAttnMixer(nn.Module):
    """因果线性注意力 (feature-map = elu+1) 基线.

    通过前缀累积 KV 外积矩阵实现因果性:
        KV_t = Σ_{s<=t} φ(k_s) φ(v_s)^T   (D_hd × D_hd, 有界)
        Z_t  = Σ_{s<=t} φ(k_s)
        out_t = KV_t φ(q_t) / (Z_t · φ(q_t) + eps)
    历史被压缩进固定大小的 D_hd×D_hd 矩阵 -> 信息瓶颈(容量 ~ D_hd^2).
    """

    def __init__(self, dim: int, n_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.eps = eps
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        B, L, D = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)   # (B,H,L,hd)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        phi = lambda t: F.elu(t) + 1.0
        qk, kk = phi(q), phi(k)             # (B,H,L,hd)
        # 前缀累积外积 KV_t (= φ(k_s) outer v_s) 与 Z_t
        outer = kk.unsqueeze(-1) * v.unsqueeze(-2)        # (B,H,L,hd,hd)
        cum_KV = torch.cumsum(outer, dim=2)               # (B,H,L,hd,hd)
        cum_Z = torch.cumsum(kk, dim=2)                  # (B,H,L,hd)
        num = torch.einsum('bhlid,bhld->bhli', cum_KV, qk)  # (B,H,L,hd)
        den = torch.einsum('bhld,bhld->bhl', cum_Z, qk)     # (B,H,L)
        out = num / (den.unsqueeze(-1) + self.eps)          # (B,H,L,hd)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        return residual + self.out_scale * out


# --------------------------------------------------------------------------
# (3) Selective-Scan SSM (Mamba式): 有界状态 (N·D, 固定) 的次二次基线
#     采用 *循环(recurrent)* 形式以暴露其信息瓶颈.
# --------------------------------------------------------------------------
class SelectiveScanSSM(nn.Module):
    """Mamba 式选择性扫描 SSM (循环形式) —— 有界状态基线.

    对每个通道 d 与状态 n:
        δ_t = softplus(W_dt x_t)                 # 内容依赖步长
        A_bar = exp(δ_t · A[d,n])                # A:(D,state) 可学
        B_bar = δ_t · B_t[n]                     # B_t 内容依赖
        h_t = A_bar · h_{t-1} + B_bar · x_t      # 固定大小状态 h:(D,state)
        y_t = Σ_n C_t[n] · h_t                   # C_t 内容依赖
        out  = y_t · silu(z_t)                   # 门控
    状态大小 = D·state (固定, 与序列长度 n 无关) -> 信息瓶颈:
    当 n·D >> D·state 时, 历史被压缩, 长程召回退化.
    """

    def __init__(self, dim: int, state: int = 16, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.state = state
        self.eps = eps
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)   # x, z(门控)
        self.dt_proj = nn.Linear(dim, dim, bias=False)        # 逐通道 δ
        self.B_proj = nn.Linear(dim, state, bias=False)
        self.C_proj = nn.Linear(dim, state, bias=False)
        # A: 每个 (d, n) 一个实数 (初始化为负, 保证稳定衰减)
        self.A_log = nn.Parameter(torch.zeros(dim, state))
        with torch.no_grad():
            # 类似 Mamba: A = -exp(A_log), 初始在 [-1/state .. -1] 量级
            self.A_log.data = -torch.exp(torch.rand(dim, state) + 0.5)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h0 = self.norm(x)
        B, L, D = h0.shape
        xz = self.in_proj(h0)                     # (B,L,2D)
        x_in, z = xz[:, :, :D], xz[:, :, D:]
        delta = F.softplus(self.dt_proj(x_in))    # (B,L,D)
        Bm = self.B_proj(x_in)                    # (B,L,state)
        Cm = self.C_proj(x_in)                    # (B,L,state)
        A = -torch.exp(self.A_log)                # (D,state), 稳定负

        h = torch.zeros(B, D, self.state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # 离散化 (Euler)
            A_bar = torch.exp(delta[:, t, :, None] * A)          # (B,D,state)
            B_bar = delta[:, t, :, None] * Bm[:, t, None, :]     # (B,D,state)
            h = A_bar * h + B_bar * x_in[:, t, :, None]          # (B,D,state)
            y_t = torch.einsum('bds,bs->bd', h, Cm[:, t, :])     # (B,D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                  # (B,L,D)
        out = y * torch.sigmoid(z)
        out = self.out_proj(out)
        return residual + self.out_scale * out


# --------------------------------------------------------------------------
# 统一模型: 用一个 mixer *类* + kwargs 构造公平对比 LM
# --------------------------------------------------------------------------
MIXER_REGISTRY = {
    "attn": AttentionMixer,
    "sgc": SpectralGlobalConvV2,
    "sgc_ag": SGCAdaptiveGating,
    "linear": LinearAttnMixer,
    "ssm": SelectiveScanSSM,
}


class Block2(nn.Module):
    def __init__(self, dim, mixer_name, max_len=512, sgc_rank=8, n_heads=4,
                 ssm_state=16):
        super().__init__()
        m = MIXER_REGISTRY[mixer_name]
        if mixer_name == "attn":
            self.mixer = m(dim, n_heads, max_len)
        elif mixer_name in ("sgc", "sgc_ag"):
            self.mixer = m(dim, max_len, sgc_rank)
        elif mixer_name == "linear":
            self.mixer = m(dim, n_heads)
        elif mixer_name == "ssm":
            self.mixer = m(dim, ssm_state)
        else:
            raise ValueError(mixer_name)
        self.ffn = FFN(dim)

    def forward(self, x):
        return self.ffn(self.mixer(x))


class LMModelV2(nn.Module):
    """公平对比 LM: 仅 mixer 不同, 其余完全一致."""

    def __init__(self, vocab, dim, n_layers, max_len=512, mixer="attn",
                 sgc_rank=8, n_heads=4, ssm_state=16, tie_emb=True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([
            Block2(dim, mixer, max_len, sgc_rank, n_heads, ssm_state)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie_emb:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


if __name__ == "__main__":
    V, D, L, NL = 200, 64, 256, 2
    for name in ["attn", "sgc", "sgc_ag", "linear", "ssm"]:
        m = LMModelV2(V, D, NL, L, name)
        x = torch.randint(0, V, (2, L))
        # 注意: 真实 LM 训练需目标左移一位, 否则 tied-embedding 会产生
        # "回声"假象 (logits≈E·input_emb, argmax=输入自身) 使 loss 虚为 0.
        xt, yt = x[:, :-1], x[:, 1:]
        import time
        t = time.time()
        out = m(xt)
        loss = F.cross_entropy(out.reshape(-1, V), yt.reshape(-1))
        loss.backward()
        dt = time.time() - t
        print(f"{name:7s} params={count_params(m):>8d}  fwd+bwd={dt*1000:6.1f}ms  "
              f"loss={loss.item():.3f}")
