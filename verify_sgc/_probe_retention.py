# -*- coding: utf-8 -*-
"""记忆保持探针 v2（诚实、可复现版）: 把两个问题拆开测.

上一次 v1 的缺陷: 用"随机单位向量 + argmax"测欠预测 -> 在 D=64 下近乎正交,
argmax 路由近似随机, 结果随 seed 在 0 ~ 0.7 之间乱跳, 不可复现.

本版把"路由是否可靠"和"覆盖写是否随上下文灭退"拆成两块, 都报 mean±95%CI:

  面板 A · 路由可靠性:  给定 D / 槽数 r / 温度, 一个反复出现的键回到自己槽的概率。
    直接揭示"低维随机键下 argmax 是抛硬币", 而这正是"训练出结构化键"要解决的。
  面板 B · 灭退隔离:  假定内容已分离(信号占用专门槽, 干扰只写其它槽)时,
    覆盖写记忆的保留率随上下文 T 是否下降。这是对"不线性衰退"的干净机理检验:
    "替换而非求和 -> 记忆不被后来的 token 稀释"。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)

def cos(a, b):
    return (a @ b) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)


# ---------- 面板 A: 路由可靠性 ----------
def hit_prob_emp(D, r, T, n=800, temp=1.0):
    """一个键 ks 写入自己的槽后, 经过 T 个随机干扰写之后, argmax(ks·M) 是否仍在原槽."""
    hits = []
    for _ in range(n):
        M = F.normalize(torch.randn(r, D), dim=-1)
        ks = F.normalize(torch.randn(D), dim=-1)
        j0 = int(torch.argmax(ks @ M.transpose(0, 1)))     # 先占自己的槽
        M[j0] = ks
        for _ in range(T):
            k = F.normalize(torch.randn(D), dim=-1)         # 随机新键
            j = int(torch.argmax(k @ M.transpose(0, 1)))    # 注意: M 每步未重归一, 但路线一致
            M[j] = k                                        # 覆盖写(替换)
        jr = int(torch.argmax(ks @ M.transpose(0, 1)))
        hits.append(1.0 if jr == j0 else 0.0)
    hits = np.array(hits)
    return hits.mean(), hits.std(ddof=1)


# ---------- 面板 B: 覆盖写 vs 叠加, 同样写入量下随干扰 G 的灭退对照 ----------
def sum_retention(D, R, Gs, n=120):
    """叠加型线性注意力: 信号写 R 次 + G 个互异干扰, 全部累进 S=Σ k⊗v.
    读: q·S. 干扰项挤占信噪比 -> 保留率随 G 掉. (未归一: 用余弦度量方向)."""
    out = {}
    for G in Gs:
        vals = []
        for _ in range(n):
            S = torch.zeros(D, D)
            ks = F.normalize(torch.randn(D), dim=-1)
            vs = F.normalize(torch.randn(D), dim=-1)
            for _ in range(R):
                S = S + torch.outer(ks, vs)                       # 信号 R 次
            for _ in range(G):
                kn = F.normalize(torch.randn(D), dim=-1)
                vn = F.normalize(torch.randn(D), dim=-1)
                S = S + torch.outer(kn, vn)                        # G 个互异干扰
            outv = ks @ S
            vals.append(float(cos(outv, vs)))
        arr = np.array(vals); out[G] = (arr.mean(), arr.std(ddof=1))
    return out


def overwrite_retention(D, r, R, Gs, beta=0.5, n=120):
    """覆盖写/独占槽: 信号实体在专用槽 0 写 R 次(刷新到最新值 vs), G 个互异干扰
    各独占其它槽并写. 干扰永远不写槽0(内容可分离), 因此不挤占信号的精度;
    信号自身经 R 次 beta 更新收敛到 vs. 保留率应随 G 平坦 -> 不线性灭退."""
    out = {}
    for G in Gs:
        vals = []
        for _ in range(n):
            M = F.normalize(torch.randn(r, D), dim=-1)             # r 个互异原型
            A = torch.zeros(r, D)
            j0 = 0
            ks = M[j0]; vs = F.normalize(torch.randn(D), dim=-1)
            for _ in range(R):                                      # 信号刷新 R 次
                A[j0] = A[j0] + beta * (vs - A[j0])
            for t in range(G):                                      # G 个互异干扰各占一槽
                j = 1 + (t % (r - 1))
                A[j] = A[j] + beta * (F.normalize(torch.randn(D), dim=-1) - A[j])
            jr = int(torch.argmax(ks @ M.transpose(0, 1)))
            vals.append(float(cos(A[jr], vs)))                     # 读最新 vs
        arr = np.array(vals); out[G] = (arr.mean(), arr.std(ddof=1))
    return out


def main():
    D, r = 64, 64
    print("=" * 78)
    print("面板 A · 路由侵入: 随机键 argmax 回到原槽的概率 P(hit)  (D=%d, r=%d)" % (D, r))
    print("=" * 78)
    for T in [0, 8, 64, 512]:
        m, sd = hit_prob_emp(D, r, T)
        ci = 1.96 * sd / np.sqrt(800)
        print(f"  T={T:>4} 干扰后  P(hit)={m:.3f} ± {ci:.3f}   (1/r={1/r:.3f} ~ 随机碰运气)")
    print("  结论: 低维随机键下路由是抛硬币 -> 必须靠训练出的结构化键/高维路由保证可分离")
    print()
    print("=" * 78)
    print("面板 B · 覆盖写 vs 叠加, 同样写入量 (R=4 次信号 + G 个干扰) 下随 G 的灭退对照")
    print("=" * 78)
    R = 4
    Gs = [0, 64, 512, 2048]
    sumd = sum_retention(D, R, Gs)
    owd = overwrite_retention(D, r, R, Gs)
    print(f"{'G':>6} | {'叠加(线性注意力) mean±CI':>26} | {'覆盖写(独占槽) mean±CI':>26}")
    for G in Gs:
        m1, s1 = sumd[G]; m2, s2 = owd[G]
        print(f"{G:>6} | {m1:>9.4f} ±{1.96*s1/np.sqrt(120):>7.4f} | {m2:>9.4f} ±{1.96*s2/np.sqrt(120):>7.4f}")
    xs = np.log(np.array([g for g in Gs if g > 0], dtype=float))
    y1 = np.log(np.maximum([sumd[g][0] for g in Gs if g > 0], 1e-9))
    y2 = np.log(np.maximum([owd[g][0] for g in Gs if g > 0], 1e-9))
    print(f"  log-log 斜率: 叠加 = {np.polyfit(xs, y1, 1)[0]:.3f} (随G灭退) | 覆盖写 = {np.polyfit(xs, y2, 1)[0]:.4f} (平坦)")


if __name__ == "__main__":
    main()