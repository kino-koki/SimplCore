# -*- coding: utf-8 -*-
"""校验块式 bmm 的 SGC-CA == 原 einsum 线性注意力, 并对注意力测速."""
import torch, time
import torch.nn.functional as F
from sgc_v3 import ContentAdaptiveMixer, build_model, count_params
from sgc_pytorch import AttentionMixer, FFN

torch.manual_seed(1)
D, r, C, L, B = 176, 32, 32, 256, 8
mix = ContentAdaptiveMixer(D, feat=r, chunk=C)
x = torch.randn(B, L, D)

# 参考: 原始 einsum 线性注意力 (逐 token 状态)
def ref_ca(m, x):
    h = m.norm(m.in_proj(x))
    Q = F.elu(m.q_proj(h)) + 1.0
    K = F.elu(m.k_proj(h)) + 1.0
    V = m.v_proj(h)
    S = torch.zeros(B, r, D); z = torch.zeros(B, r)
    outs = []
    for t in range(L):
        kv = torch.einsum('br,bd->brd', K[:, t], V[:, t])
        S = S + kv; z = z + K[:, t]
        num = torch.einsum('br,brd->bd', Q[:, t], S)
        den = torch.einsum('br,br->b', Q[:, t], z).clamp_min(1e-6).unsqueeze(-1)
        outs.append(num / den)
    out = x + m.out_proj(torch.stack(outs, 1))
    return out, (S, z)

with torch.no_grad():
    ref, (Sr, zr) = ref_ca(mix, x)
    new, (Sn, zn) = mix.forward_stream(x)
err = (ref - new).abs().max().item()
errS = (Sr - Sn).abs().max().item()
print(f"[正确性] max|ref-new|={err:.3e}  state maxerr={errS:.3e}  (应<1e-3)")

# 测速: 同骨架 vs 注意力, 前向+反向单步
def time_step(m, x, iters=20):
    t0 = time.perf_counter()
    for _ in range(iters):
        logits = m(x)
        loss = logits.sum()
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    tot = time.perf_counter() - t0
    return tot / iters

ca = build_model(1084, D, 4, 4, L, 'ca', ca_feat=r, ca_chunk=C, dropout=0.05)
attn = build_model(1084, D, 4, 4, L, 'attn', dropout=0.05)
tok = torch.randint(0, 1084, (B, L))
t_ca = time_step(ca, tok)
t_at = time_step(attn, tok)
print(f"[测速] CA fwd+bwd={t_ca*1000:.0f}ms   Attention={t_at*1000:.0f}ms   比值 CA/Attn={t_ca/t_at:.2f}x")
print(f"      参数: CA={count_params(ca):,}  Attn={count_params(attn):,}")