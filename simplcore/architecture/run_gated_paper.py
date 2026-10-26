# 论文配置快速验证 (L=48, D=64, batch=32, 200步)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import numpy as np
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from simplcore.architecture.spectral_global_conv import SpectralGlobalConvV2
from simplcore.architecture.gated_sgc import GatedSGCV2

torch.manual_seed(42)
rng = np.random.RandomState(42)
L, D, vocab, k, batch = 48, 64, 64, 6, 32
steps = 200

# 数据
tokens = rng.randint(0, vocab, size=(batch, L))
y_tokens = np.zeros_like(tokens)
y_tokens[:, k:] = tokens[:, :-k]
x_np = np.zeros((batch, L, vocab), dtype=np.float32)
y_np = np.zeros((batch, L, vocab), dtype=np.float32)
for b in range(batch):
    x_np[b, np.arange(L), tokens[b]] = 1.0
    y_np[b, np.arange(L), y_tokens[b]] = 1.0
x = torch.tensor(x_np, requires_grad=False)
y = torch.tensor(y_np, requires_grad=False)

print(f"论文配置快速验证 ({steps} 步, L={L}, D={D}, batch={batch})")

class CausalAttn(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).reshape(B, L, 4, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, L, 4, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, L, 4, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dim)
        mask = torch.zeros((L, L), dtype=torch.float32)
        for i in range(L):
            for j in range(i + 1, L):
                mask.data[i, j] = float('-inf')
        s = s + mask.unsqueeze(0).unsqueeze(0)
        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, L, D)
        return self.out(o)

def train(model, x, y, steps, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    acc = float((np.argmax(pred.data, axis=-1) == np.argmax(y.data, axis=-1)).mean() * 100)
    return float(loss.data), acc

attn = CausalAttn(D, 4)
l1, a1 = train(attn, x, y, steps)
print(f"[1/3] Attention:  loss={l1:.4f} acc={a1:.2f}%")

sgc = SpectralGlobalConvV2(dim=D, max_len=L, rank=8)
l2, a2 = train(sgc, x, y, steps)
print(f"[2/3] SGC-V2:     loss={l2:.4f} acc={a2:.2f}%")

gated = GatedSGCV2(dim=D, max_len=L, rank=8)
l3, a3 = train(gated, x, y, steps)
print(f"[3/3] Gated-SGC:  loss={l3:.4f} acc={a3:.2f}%")

print(f"\nAttention: {a1:.2f}%  SGC: {a2:.2f}% (Δ={a2-a1:+.2f}pp)  Gated: {a3:.2f}% (Δ={a3-a1:+.2f}pp vs Attn, {a3-a2:+.2f}pp vs SGC)")
print(f"Loss: Attn={l1:.4f} SGC={l2:.4f} Gated={l3:.4f}")
if a3 > a1:
    print("✅ 门控修复版超越 Attention!")
elif a3 > a2:
    print("⚠️ 门控提升了 SGC 但未超越 Attention")
else:
    print("❌ 门控未提升")
