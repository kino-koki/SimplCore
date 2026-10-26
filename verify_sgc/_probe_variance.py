# -*- coding: utf-8 -*-
"""循环复用曲线波动率核查: 保留率">0.66→0.78"的上升/波动是噪声还是系统?

给每个 T 的保留率带 std 与 95% 置信区间, 并同时跑"固定一批干扰键重复打"vs
"每轮独立重采样干扰键"两种设置, 判断:
  * 若波动量级 ~ std, 则"上升"是蒙特卡洛噪声, 持平(≈0 斜率)的结论稳;
  * 若系统偏离 0(如持续爬到 0.78), 说明复用机制引入向上漂移, 需在正文解释.
"""
from __future__ import annotations
import torch, torch.nn.functional as F, numpy as np

torch.manual_seed(0)

def cos(a, b):
    return (a @ b) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)

def trial(D, r, cycle, T, reseed=False):
    g = torch.Generator().manual_seed(int(torch.rand(1).item() * 1e9) if reseed else 0)
    M = F.normalize(torch.randn(r, D, generator=g), dim=-1)
    A = torch.zeros(r, D)
    ks = torch.randn(D, generator=g); ks = ks / ks.norm()
    vs = torch.manual_seed(int(g.initial_seed())).randn if False else (torch.randn(D) / torch.randn(D).norm())
    vs = F.normalize(torch.randn(D), dim=-1)
    j0 = int(torch.argmax(ks @ M.transpose(0, 1))); M[j0], A[j0] = ks, vs
    # 固定话题池(对话复用) 或每步全新(重采样)
    if reseed:
        for t in range(T):
            k = F.normalize(torch.randn(D), dim=-1)
            v = F.normalize(torch.randn(D), dim=-1)
            j = int(torch.argmax(k @ M.transpose(0, 1))); M[j], A[j] = k, v
    else:
        pool = F.normalize(torch.randn(cycle, D, generator=g), dim=-1)
        vpool = F.normalize(torch.randn(cycle, D, generator=g), dim=-1)
        for t in range(T):
            k, v = pool[t % cycle], vpool[t % cycle]
            j = int(torch.argmax(k @ M.transpose(0, 1))); M[j], A[j] = k, v
    jr = int(torch.argmax(ks @ M.transpose(0, 1)))
    return float(cos(A[jr], vs))

def rep(D, r, cycle, Ts, reseed, n=200):
    out = {}
    for T in Ts:
        vals = np.array([trial(D, r, cycle, T, reseed) for _ in range(n)])
        out[T] = (vals.mean(), vals.std(ddof=1),
                  vals.mean() - 1.96 * vals.std(ddof=1) / np.sqrt(n),
                  vals.mean() + 1.96 * vals.std(ddof=1) / np.sqrt(n))
    return out

def main():
    D, r, cycle = 64, 64, 16
    Ts = [32, 512, 2048, 8192]
    print("=" * 86)
    print("保留率 mean±std (n=200), 判定'上升'是否超噪声")
    print("=" * 86)
    for label, reseed in [("固定复用16键", False), ("每步全新干扰(重采样)", True)]:
        print(f"\n[{label}]")
        print(f"{'T':>7} | {'mean':>8} | {'std':>7} | {'95%CI 下界':>12} | {'95%CI 上界':>12}")
        d = rep(D, r, cycle, Ts, reseed)
        for T in Ts:
            m, sd, lo, hi = d[T]
            print(f"{T:>7} | {m:>8.4f} | {sd:>7.4f} | {lo:>12.4f} | {hi:>12.4f}")

if __name__ == "__main__":
    main()