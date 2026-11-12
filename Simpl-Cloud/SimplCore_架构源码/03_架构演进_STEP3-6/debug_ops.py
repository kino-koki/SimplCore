# -*- coding: utf-8 -*-
"""Per-operator finite-difference gradient checks (diagnostic, not shipped)."""
import os, sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, "_cloud_repo", "SimplCore-main",
                               "simplcore", "core", "arch"))
import nn
import simplcore_lm as A

rng = np.random.default_rng(0)
B, L, d, W = 2, 7, 5, 4


def P(shape, s=1.0):
    return nn.param(rng.standard_normal(shape) * s)


def check(name, ins, make):
    nn.clear_tape()
    for t in ins:
        t.zero_grad()
    loss = make(ins)
    loss.backward()
    ana = [t.grad.copy() for t in ins]
    worst, info = 0.0, None
    eps = 1e-6
    for ti, t in enumerate(ins):
        js = sorted(set([0, t.size // 2, t.size - 1])) if t.size >= 3 else list(range(t.size))
        for j in js:
            old = float(t.data.ravel()[j])
            t.data.ravel()[j] = old + eps
            nn.clear_tape(); lp = make(ins).data.item(); nn.clear_tape()
            t.data.ravel()[j] = old - eps
            nn.clear_tape(); lm = make(ins).data.item(); nn.clear_tape()
            t.data.ravel()[j] = old
            num = (lp - lm) / (2 * eps)
            a = float(ana[ti].ravel()[j])
            rel = abs(num - a) / (abs(num) + abs(a) + 1e-9)
            if rel > worst:
                worst, info = rel, (ti, j, round(a, 8), round(num, 8))
    print("%-18s worst=%.3e  %s" % (name, worst, info))
    return worst


WT = nn.const(rng.standard_normal((B, L, d)) * 0.3)
WTW = nn.const(rng.standard_normal((B, L, W)) * 0.3)

# ---------------- base ----------------
x, w = P((B, L, d)), P((d, d), 0.5)
check("matmul", [x, w], lambda i: nn.sum_(nn.mul(nn.matmul(i[0], i[1]), WT)))

a, b = P((B, L, d)), P((d,), 0.4)
check("add_broadcast", [a, b], lambda i: nn.sum_(nn.mul(nn.add(i[0], i[1]), WT)))

a, s = P((B, L, d)), P((), 0.4)
check("mul_scalar", [a, s], lambda i: nn.sum_(nn.mul(nn.mul(i[0], i[1]), WT)))

x, g, be = P((B, L, d)), P((d,)), P((d,))
check("layernorm", [x, g, be], lambda i: nn.sum_(nn.mul(nn.layernorm(i[0], i[1], i[2]), WT)))

x = P((B, L, d))
check("gelu", [x], lambda i: nn.sum_(nn.mul(nn.gelu(i[0]), WT)))

x = P((B, L, 1))
check("sigmoid", [x], lambda i: nn.sum_(nn.mul(nn.sigmoid(i[0]), nn.const(np.ones((B, L, 1))))))

x = P((B, L, d))
check("softmax", [x], lambda i: nn.sum_(nn.mul(nn.softmax(i[0]), WT)))

E, idx = P((11, d)), np.array([[0, 3, 2, 1, 4, 5, 6], [7, 8, 9, 10, 1, 2, 3]])
check("embedding", [E], lambda i: nn.sum_(nn.mul(nn.embedding(i[0], idx), WT)))

# ---------------- fused ----------------
q, k, v, pos = P((B, L, d)), P((B, L, d)), P((B, L, d)), P((W,))
check("window_attn", [q, k, v, pos],
      lambda i: nn.sum_(nn.mul(A.window_attn(i[0], i[1], i[2], W, i[3]), WT)))

u, wl = P((B, L, d)), P((3,))
check("toeplitz_mix", [u, wl],
      lambda i: nn.sum_(nn.mul(A.toeplitz_mix(i[0], i[1], (0.6, 0.9, 0.99), L), WT)))

q, k, v, beta = P((B, L, d)), P((B, L, d)), P((B, L, d)), P((B, L))
check("decayed_attn", [q, k, v, beta],
      lambda i: nn.sum_(nn.mul(A.decayed_attn(i[0], i[1], i[2], i[3], 0.9), WT)))

# ---------------- reference equivalence ----------------
qr, kr, vr, pr, br = (t.data.copy() for t in (q, k, v, pos, beta))
yw = A.window_attn(q, k, v, W, pos).data
yr = A.window_attn_reference(qr, kr, vr, W, pr)
print("window_attn reference max abs diff:", float(np.abs(yw - yr).max()))
yd = A.decayed_attn(q, k, v, beta, 0.9).data
ydr = A.decayed_attn_reference(qr, kr, vr, br, 0.9)
print("decayed_attn reference max abs diff:", float(np.abs(yd - ydr).max()))
