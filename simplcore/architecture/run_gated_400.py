# 400 步标准配置门控消融 (Attention vs Gated-SGC)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import time
import numpy as np
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from simplcore.architecture.gated_sgc import GatedSGCV2

torch.manual_seed(42)
rng = np.random.RandomState(42)
L, D, vocab, k, batch = 48, 64, 64, 6, 32
steps = 400

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

print(f"400 步门控消融 (L={L}, D={D}, batch={batch})")

# Attention
class CausalAttn(nn.Module):
    def __init__(self, dim, n_heads=4):
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

attn = CausalAttn(D)
opt1 = torch.optim.SGD(attn.parameters(), lr=0.01)
t0 = time.time()
for _ in range(steps):
    pred = attn(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    opt1.step()
    opt1.zero_grad()
acc1 = float((np.argmax(pred.data, axis=-1) == np.argmax(y.data, axis=-1)).mean() * 100)
print(f"[1/2] Attention:    loss={float(loss.data):.4f} acc={acc1:.2f}% time={time.time()-t0:.1f}s")

# Gated SGC
gated = GatedSGCV2(dim=D, max_len=L, rank=8)
opt2 = torch.optim.SGD(gated.parameters(), lr=0.01)
t0 = time.time()
for _ in range(steps):
    pred2 = gated(x)
    loss2 = F.mse_loss(pred2, y)
    loss2.backward()
    opt2.step()
    opt2.zero_grad()
acc2 = float((np.argmax(pred2.data, axis=-1) == np.argmax(y.data, axis=-1)).mean() * 100)
print(f"[2/2] Gated-SGC-V2:  loss={float(loss2.data):.4f} acc={acc2:.2f}% time={time.time()-t0:.1f}s")

print()
print(f"Attention acc: {acc1:.2f}%  Gated acc: {acc2:.2f}% (Δ={acc2-acc1:+.2f}pp)")
print(f"Attention loss: {float(loss.data):.4f}  Gated loss: {float(loss2.data):.4f} (Δ={float(loss2.data)-float(loss.data):+.4f})")
if acc2 > acc1:
    print("✅ 门控修复版超越 Attention!")
elif acc2 == acc1:
    print("⚠️ 持平 (需更长训练)")
else:
    print("❌ 门控未超越")
