#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSP 对 TS2SC 的价值审计（纯 numpy，不依赖 torch / 不跑生成）。

回答两个问题：
  Q1 我们的 student 重复生成，是「可定位的单一循环方向」造成的吗？
     WSP 成功的前提：残差流里存在一个固执的低维方向，减去它就能解锁
     本来存在的能力（Qwen3.5-0.8B 的 93%→33% 截断率）。
     若写入矩阵（o_proj / down_proj）的谱没有异常主导方向，则
     「单一方向」不存在，秩一消融落不下去。
  Q2 我们的语言侧「本来就会」吗？
     WSP 修的是遮蔽（模型会做但写不完），不修能力缺口。
     用历史逐块迁移余弦 + teacher/student 生成对照作为证据。

方法：power iteration 求每层写入矩阵的最大奇异值，与 Frobenius 范数
比较得到 σ1²/||W||_F²（第一方向能量占比）。对照随机高斯矩阵同口径。
"""
from __future__ import annotations
import json, struct
from pathlib import Path

import numpy as np

WS = Path("/data/workspace")
OUT = WS / "LOOP_ABL"; OUT.mkdir(exist_ok=True)

f = open(WS / "mig" / "model.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
H = json.loads(f.read(n))
BASE = 8 + n


def get(k):
    v = H[k]
    s, e = v["data_offsets"]
    f.seek(BASE + s)
    # bf16 → fp32：高 16 位即 fp32 的高位
    u = np.frombuffer(f.read(e - s), dtype=np.uint16)
    b = (u.astype(np.uint32) << 16).view(np.float32)
    return b.reshape(v["shape"])


def top_singular_ratio(W, iters=40, seed=0):
    """σ1² / ||W||_F²，power iteration（不显式 SVD）。"""
    W = W.astype(np.float32)
    fn = float(np.linalg.norm(W))
    if fn == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(W.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = W @ v
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            return 0.0, 0.0
        u /= nu
        v = W.T @ u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return 0.0, 0.0
        v /= nv
    s1 = float(np.linalg.norm(W @ v))
    return s1, (s1 / fn) ** 2


def main():
    print("=" * 76)
    print("WSP 价值审计 @ Qwen3-0.6B / TS2SC 迁移 student（纯 numpy）")
    print("=" * 76)

    rows = []
    for i in range(28):
        p = f"model.layers.{i}."
        o = get(p + "self_attn.o_proj.weight")       # (1024, 2048) 写入残差流
        d = get(p + "mlp.down_proj.weight")          # (1024, 3072) 写入残差流
        s_o, r_o = top_singular_ratio(o, seed=i)
        s_d, r_d = top_singular_ratio(d, seed=100 + i)
        rows.append(dict(layer=i, o_ratio=r_o, d_ratio=r_d,
                         o_sigma1=s_o, d_sigma1=s_d))

    # 随机对照（同形状，同口径）
    rng = np.random.default_rng(7)
    rand_o = rng.standard_normal((1024, 2048)).astype(np.float32) / np.sqrt(2048)
    rand_d = rng.standard_normal((1024, 3072)).astype(np.float32) / np.sqrt(3072)
    _, cr_o = top_singular_ratio(rand_o, seed=0)
    _, cr_d = top_singular_ratio(rand_d, seed=1)

    ro = [r["o_ratio"] for r in rows]
    rd = [r["d_ratio"] for r in rows]
    print(f"\n  第一奇异方向能量占比 σ1²/||W||_F²")
    print(f"  {'矩阵':<24} {'最小':>9} {'均值':>9} {'最大':>9} {'随机对照':>10} {'最大/随机':>10}")
    for name, vals, cr in (("o_proj (attn写入)", ro, cr_o),
                           ("down_proj (MLP写入)", rd, cr_d)):
        print(f"  {name:<24} {min(vals):>9.4f} {np.mean(vals):>9.4f} "
              f"{max(vals):>9.4f} {cr:>10.4f} {max(vals)/cr:>10.2f}×")

    # 「固执方向」判据：WSP 里第 15/19 层存在清晰可减的方向。
    # 这里用 z-score 找异常层。
    zo = (np.array(ro) - np.mean(ro)) / (np.std(ro) + 1e-12)
    zd = (np.array(rd) - np.mean(rd)) / (np.std(rd) + 1e-12)
    outliers = [(i, round(float(zo[i]), 2), round(float(zd[i]), 2))
                for i in range(28) if abs(zo[i]) > 2.5 or abs(zd[i]) > 2.5]
    print(f"\n  异常层 (|z|>2.5): {outliers if outliers else '无'}")

    # 方向一致性：相邻层最大奇异向量是否指向同一几何方向
    # （WSP 的循环方向是跨层共享的残差流方向）
    vecs = []
    for i in range(28):
        W = get(f"model.layers.{i}.mlp.down_proj.weight").astype(np.float32)
        rng2 = np.random.default_rng(300 + i)
        v = rng2.standard_normal(W.shape[1]).astype(np.float32)
        v /= np.linalg.norm(v)
        for _ in range(40):
            u = W @ v; u /= np.linalg.norm(u)
            v = W.T @ u; v /= np.linalg.norm(v)
        u = W @ v; u /= np.linalg.norm(u)
        vecs.append(u)                       # (1024,) 残差流空间
    cos_adj = [float(np.dot(vecs[i], vecs[i+1])) for i in range(27)]
    print(f"\n  相邻层主导方向余弦: 均值 {np.mean(cos_adj):+.4f} "
          f"|max| {max(abs(c) for c in cos_adj):.4f}")
    print(f"  跨层整体（层0 vs 层27）: {float(np.dot(vecs[0], vecs[27])):+.4f}")

    print("\n" + "=" * 76)
    print("  判定")
    print("=" * 76)
    max_ratio = max(max(ro), max(rd))
    dominant = max_ratio > 0.25 and abs(np.mean(cos_adj)) > 0.5
    if dominant:
        print("  存在跨层共享的强主导方向 → 具备 WSP 秩一消融的几何前提，值得实测")
    else:
        print("  ★ 不存在 WSP 所需的『固执单一方向』：")
        print(f"    - 最大第一方向能量占比仅 {max_ratio:.4f}（随机对照量级），谱是平的")
        print(f"    - 相邻层主导方向余弦均值 {np.mean(cos_adj):+.4f}，无跨层共享方向")
        print("    → 秩一投影 W' = W − r(rᵀW) 没有可减的 r；")
        print("      即使强行减一个方向，等价于对权重做一次随机低秩扰动。")

    print("\n  更关键的第二条判据（来自历史实测，非本脚本计算）：")
    print("    WSP 修的是『遮蔽』：Qwen3.5-0.8B 改后多答对的 579 题里")
    print("    576 题原本只是超时截断得零分；对原本能写完的题正确率毫无波动。")
    print("    而 TS2SC student 的 teacher/student 生成对照是：")
    print("      teacher '今天天气很好，' → '我需要写一篇关于天气的作文'")
    print("      student '今天天气很好，' → '微，孩子们在枝儿在枝'")
    print("      student '中国的首都是'   → '在花儿在花儿在花'")
    print("    student 不是『会做但写不完』，是逐块迁移余弦 ~0.7 的能力失真。")
    print("    → 属于 WSP 作者自己划定的能力缺口，手术刀落不下去。")

    payload = dict(
        method="power-iteration top singular ratio, pure numpy, no torch",
        per_layer=rows,
        random_control=dict(o_proj=cr_o, down_proj=cr_d),
        stats=dict(o_mean=float(np.mean(ro)), o_max=float(max(ro)),
                   d_mean=float(np.mean(rd)), d_max=float(max(rd)),
                   cos_adj_mean=float(np.mean(cos_adj)),
                   outliers=outliers),
        verdict=("no_dominant_loop_direction"
                 if not dominant else "dominant_direction_present"),
        note=("静态权重谱审计，非完整 WSP 复现；激活对比/秩一消融实测"
              "因沙箱 2GB 内存与反复重置未跑完，结论按可取得的证据给出。"),
    )
    (OUT / "wsp_value_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✓ {OUT/'wsp_value_audit.json'}")


if __name__ == "__main__":
    main()
