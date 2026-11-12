#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""osc_probe —— 频率网格收益的仓库内可复现前哨（上游 P1-4 / P1-5）。

动机
----
`docs/frequency_grid_fix.md` §1 / §9 明确标注：训练侧 acc / bootstrap CI /
MQAR 数字「来源：用户的白盒复现报告第 15 节，**非本仓复现**」；全仓此前
没有任何 osc 任务脚本或 raw 结果，`init_mode="linear"` 的收益是仓库外孤证。

本脚本把该主张变成可重跑的骨架：

* 任务生成器：周期信号 + 噪声（``osc`` ε=0.15）与纯周期版（``osc0``）；
* 网格扫描：``logfreq`` / ``hybrid`` / ``linear`` 多档 ``linear_floor`` /
  **非对称 θ 网格**（打破 π/2 镜像，回应 P1-5）；
* 平凡基线：解析 chance 与「复制上一个 token」；
* 配对 bootstrap：同一 seed 下逐对准确率差 + percentile CI95；
* raw json 落盘，供 §9 撤掉「非本仓复现」时引用。

⚠️ 诚实边界：默认参数是**小预算前哨**（D=32 / 1 层 / 250 步 / 3 seeds /
合成任务），只用于验证「机制方向可在本仓重跑」，**不等于**白盒报告里
1500/3000 步 + bootstrap CI [+0.0996,+0.1621] 的定量结论。要复现那个量级
请自行提高 ``--steps`` / ``--seeds`` 并保留同一 raw 口径。

CLI::

    python -m simplcore.core.eval.osc_probe --steps 250 --seeds 0 1 2 \
        --out artifacts/osc_probe_raw.json
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch.sgc import SpectralGlobalConv, _init_oscillator_grid

GEN = torch.Generator


# ---------------------------------------------------------------------------
# 任务：周期信号预测
# ---------------------------------------------------------------------------
def gen_osc(L: int, g: GEN, V: int = 16, period: int = 7,
            eps: float = 0.15, n_harm: int = 2) -> Tuple[torch.Tensor, int]:
    """多分量周期信号 + Bernoulli 替换噪声；目标 = 下一位的**干净**符号。

    每个样本重新采样 ``n_harm`` 个随机频率与相位：模型**不能靠记住某个
    固定周期**取胜，必须从当前噪声前缀里分辨并外推周期分量——这正是
    「周期 / 连续信号轴」的考验，也是频率网格该发挥作用的地方。

    ⚠️ 设计迭代记录：第一版用单一锯齿 ``floor((t mod P)·V/P)``，
    200 步内所有 7 种网格全部 1.0000（天花板效应，**区分不出网格**），
    已废弃。当前版本刻意提高难度；小预算下若仍饱和或仍接近 chance，
    raw json 会如实记录，不拿来充当「linear 收益」的证据。

    归一化只用前缀 ``[:L]`` 的 min/max，避免把目标位的信息泄漏进输入。
    ε=0 → 纯周期版 osc0。
    """
    t = torch.arange(L + 1, dtype=torch.float64)
    lo = max(3, period - 3)
    freqs = torch.randint(lo, period + 14, (n_harm,), generator=g).double()
    phases = torch.rand(n_harm, generator=g, dtype=torch.float64) * (2 * math.pi)
    sig = sum(torch.sin(2 * math.pi * t / f + p)
              for f, p in zip(freqs.tolist(), phases.tolist()))
    lo_v, hi_v = float(sig[:L].min()), float(sig[:L].max())
    norm = (sig - lo_v) / (hi_v - lo_v + 1e-9)
    clean = torch.clamp((norm * V).long(), 0, V - 1)
    seq = clean[:L].clone()
    if eps > 0:
        flip = torch.rand(L, generator=g) < eps
        n = int(flip.sum())
        if n:
            seq[flip] = torch.randint(0, V, (n,), generator=g)
    return seq.long(), int(clean[L])


def chance_and_copy_baseline(V: int = 16, period: int = 7,
                             eps: float = 0.15, n: int = 2000,
                             seed: int = 0) -> Dict[str, float]:
    """平凡基线：解析 chance；以及「直接复制上一个 token」的命中率。"""
    g = GEN().manual_seed(seed)
    hit = 0
    for _ in range(n):
        seq, tgt = gen_osc(64, g, V=V, period=period, eps=eps)
        hit += int(int(seq[-1]) == tgt)
    return {"chance": round(1.0 / V, 4),
            "copy_last": round(hit / n, 4),
            "n": n, "eps": eps, "V": V, "period": period}


# ---------------------------------------------------------------------------
# 网格工厂：包括打破 π/2 镜像的非对称线性网格（P1-5）
# ---------------------------------------------------------------------------
GRIDS: List[Tuple[str, str, float, float]] = [
    # (name, mode, linear_floor, skew)  skew≠0 → 非对称
    ("logfreq",        "logfreq", 0.3, 0.0),
    ("hybrid",         "hybrid",  0.3, 0.0),
    ("linear_f0.10",   "linear",  0.10, 0.0),
    ("linear_f0.30",   "linear",  0.30, 0.0),
    ("linear_f0.50",   "linear",  0.50, 0.0),
    ("linear_f0.30_skew+0.6", "linear", 0.30, 0.6),
    ("linear_f0.30_skew-0.6", "linear", 0.30, -0.6),
]


def _apply_asymmetric(mixer: SpectralGlobalConv, skew: float) -> None:
    """在 (m+0.5)/M 栅格上施加幂律歪斜，打破 θ 关于 π/2 的镜像对称。

    对称网格满足 sin((τ+1)(π−θ))=(−1)^τ sin((τ+1)θ)，在含 ρ 的口径 B 下
    引入冗余对（§3 末尾）。u^γ（γ=1+skew）把频率往低端或高端挤压，
    镜像关系不再成立。仅改初始化值，不改算子/状态/纯度契约。
    """
    if abs(skew) < 1e-9:
        return
    M = mixer.num_bases
    f = float(getattr(mixer, "linear_floor", 0.3))
    g = 1.0 + skew
    with torch.no_grad():
        u = (torch.arange(M, dtype=torch.float64) + 0.5) / M
        theta = (f + (math.pi - 2.0 * f) * (u ** g)).to(torch.float32)
        # raw_theta 参数化为 theta = pi * sigmoid(raw_theta)
        p = (theta / math.pi).clamp(1e-4, 1 - 1e-4)
        mixer.raw_theta.copy_(torch.log(p / (1.0 - p)))
        # 不动 _theta_floor：钳制下界只约束训练后漂移，此处所有 θ 均
        # 远大于 floor，改初始化值不涉及卫兵口径。


class OscNet(nn.Module):
    """最小可读模型：embedding → 单层 SGC → 线性头。"""

    def __init__(self, V: int, D: int, M: int, mode: str, floor: float,
                 skew: float, decay_max: float = 256.0, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, D)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mixer = SpectralGlobalConv(
                dim=D, num_bases=M, decay_max=decay_max, local_kernel=4,
                content_gate=False, evidence_gate=False,
                uncertainty_gate=False, init_mode=mode, linear_floor=floor)
        _apply_asymmetric(self.mixer, skew)
        self.head = nn.Linear(D, V)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.mixer(self.emb(idx)))


# ---------------------------------------------------------------------------
# 训练 / 评估 / 配对 bootstrap
# ---------------------------------------------------------------------------
def train_eval(name: str, mode: str, floor: float, skew: float, *,
               steps: int, seeds: List[int], V: int, D: int, M: int,
               L: int, period: int, eps: float, n_eval: int,
               batch: int, lr: float) -> Dict[str, Any]:
    accs: List[float] = []
    losses: List[float] = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = OscNet(V, D, M, mode, floor, skew, seed=sd)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        g = GEN().manual_seed(sd)
        net.train()
        last = 0.0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            ls = []
            for _ in range(batch):
                seq, tgt = gen_osc(L, g, V=V, period=period, eps=eps)
                logits = net(seq.unsqueeze(0))[:, -1, :]
                ls.append(F.cross_entropy(logits, torch.tensor([tgt])))
            loss = torch.stack(ls).mean()
            loss.backward(); opt.step()
            last = float(loss.detach())
        losses.append(round(last, 5))
        net.eval()
        ge = GEN().manual_seed(sd + 7777)
        hit = 0
        with torch.no_grad():
            for _ in range(n_eval):
                seq, tgt = gen_osc(L, ge, V=V, period=period, eps=eps)
                hit += int(net(seq.unsqueeze(0))[0, -1].argmax().item() == tgt)
        accs.append(hit / n_eval)
    return {"name": name, "mode": mode, "linear_floor": floor, "skew": skew,
            "accs": [round(a, 4) for a in accs],
            "mean": round(sum(accs) / len(accs), 4),
            "final_loss": losses,
            "params": sum(p.numel() for p in OscNet(V, D, M, mode, floor,
                                                    skew, seed=0).parameters())}


def paired_bootstrap(a: List[float], b: List[float], n_boot: int = 2000,
                     seed: int = 0) -> Dict[str, float]:
    """配对差 a−b 的 percentile CI95（同一 seed 逐对相减）。"""
    g = GEN().manual_seed(seed)
    da, db = torch.tensor(a), torch.tensor(b)
    diff = da - db
    n = len(diff)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = diff[idx].mean(1)
    lo = float(torch.quantile(means, 0.025))
    hi = float(torch.quantile(means, 0.975))
    return {"diff": round(float(diff.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def run(args) -> Dict[str, Any]:
    baselines = {}
    for eps, tag in ((0.15, "osc"), (0.0, "osc0")):
        baselines[tag] = chance_and_copy_baseline(
            V=args.V, period=args.period, eps=eps, n=args.n_baseline, seed=0)

    rows = []
    for name, mode, floor, skew in GRIDS:
        r = train_eval(name, mode, floor, skew, steps=args.steps,
                       seeds=args.seeds, V=args.V, D=args.D, M=args.M,
                       L=args.L, period=args.period, eps=args.eps,
                       n_eval=args.n_eval, batch=args.batch, lr=args.lr)
        rows.append(r)
        print(f"  {name:<26} mean={r['mean']:.4f}  accs={r['accs']}", flush=True)

    by = {r["name"]: r for r in rows}
    comparisons = {}
    base = by["logfreq"]
    for r in rows:
        if r["name"] == "logfreq":
            continue
        comparisons[f"{r['name']} - logfreq"] = paired_bootstrap(
            r["accs"], base["accs"], n_boot=args.n_boot, seed=0)

    return {
        "method": "osc period-signal probe; grid scan + paired bootstrap",
        "config": dict(steps=args.steps, seeds=args.seeds, V=args.V, D=args.D,
                       M=args.M, L=args.L, period=args.period, eps=args.eps,
                       n_eval=args.n_eval, batch=args.batch, lr=args.lr,
                       n_boot=args.n_boot),
        "honest_boundary": (
            "小预算前哨（D=32 / 1 层 / 数百步 / 3 seeds / 合成任务），"
            "只验证「脚本可在本仓重跑并产出 raw json」；不等于白盒报告 "
            "§15 的 1500/3000 步定量结论与 CI [+0.0996,+0.1621]。"
            "若全部网格饱和或全部接近 chance，本结果不构成任何一方的"
            "收益/无效证据，需加大预算并重新设计难度。"),
        "baselines": baselines,
        "rows": rows,
        "paired_vs_logfreq": comparisons,
        "p1_5_note": (
            "含 skew 的网格用于检验「非对称 θ 网格打破 π/2 镜像是否改善 "
            "含 ρ 口径下的可分性」；本脚本的 acc 是训练后下游准确率，"
            "不是 Gram 条件数，两者需分开看（见 docs/frequency_grid_fix.md §3）。"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="osc 频率网格前哨（上游 P1-4）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--V", type=int, default=16)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--period", type=int, default=7)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-baseline", type=int, default=2000)
    ap.add_argument("--out", type=str, default="artifacts/osc_probe_raw.json")
    args = ap.parse_args(argv)
    print("=" * 72)
    print(f"osc_probe  steps={args.steps} seeds={args.seeds} "
          f"D={args.D} M={args.M} L={args.L} eps={args.eps}")
    print("=" * 72, flush=True)
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  ✓ {out}")
    print("  配对差 vs logfreq:")
    for k, v in result["paired_vs_logfreq"].items():
        print(f"    {k:<34} {v['diff']:+.4f}  CI95={v['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
