# -*- coding: utf-8 -*-
"""单步耗时测试"""
import time
import numpy as np
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from simplcore.architecture.spectral_global_conv import SpectralGlobalConvV2
from simplcore.architecture.gated_sgc import GatedSGCV2

# 配置
L, D, vocab, k, batch = 48, 64, 64, 6, 32
rng = np.random.RandomState(42)
tokens = rng.randint(0, vocab, size=(batch, L))
y_tokens = np.zeros_like(tokens)
y_tokens[:, k:] = tokens[:, :-k]
x = np.zeros((batch, L, vocab), dtype=np.float32)
y = np.zeros((batch, L, vocab), dtype=np.float32)
for b in range(batch):
    x[b, np.arange(L), tokens[b]] = 1.0
    y[b, np.arange(L), y_tokens[b]] = 1.0

xt = torch.tensor(x, requires_grad=False)
yt = torch.tensor(y, requires_grad=False)

# SGC-v2
sgc = SpectralGlobalConvV2(dim=D, max_len=L, rank=8)
opt = torch.optim.Adam(sgc.parameters(), lr=0.003)
start = time.time()
for _ in range(20):
    pred = sgc(xt)
    loss = F.mse_loss(pred, yt)
    loss.backward()
    opt.step()
    opt.zero_grad()
step_sgc = (time.time() - start) / 20
print(f'SGC-v2 单步: {step_sgc:.3f}s')

# GatedSGC
sg = GatedSGCV2(dim=D, max_len=L, rank=8)
opt2 = torch.optim.Adam(sg.parameters(), lr=0.003)
start = time.time()
for _ in range(20):
    pred = sg(xt)
    loss = F.mse_loss(pred, yt)
    loss.backward()
    opt2.step()
    opt2.zero_grad()
step_gated = (time.time() - start) / 20
print(f'GatedSGC 单步: {step_gated:.3f}s')

# Attention
class CausalSelfAttn(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dim)
        mask = torch.zeros((L, L), dtype=torch.float32)
        for i in range(L):
            for j in range(i + 1, L):
                mask.data[i, j] = float('-inf')
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))

attn = CausalSelfAttn(D, 4)
opt3 = torch.optim.Adam(attn.parameters(), lr=0.003)
start = time.time()
for _ in range(20):
    pred = attn(xt)
    loss = F.mse_loss(pred, yt)
    loss.backward()
    opt3.step()
    opt3.zero_grad()
step_attn = (time.time() - start) / 20
print(f'Attention 单步: {step_attn:.3f}s')

print(f'\n4000 步总耗时估算:')
print(f'  SGC-v2: {step_sgc * 4000:.1f}s')
print(f'  GatedSGC: {step_gated * 4000:.1f}s')
print(f'  Attention: {step_attn * 4000:.1f}s')
print(f'  合计: {(step_sgc + step_gated + step_attn) * 4000:.1f}s')
