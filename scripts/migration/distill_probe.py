#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""distill_probe —— 「修完迁移脚本再重新训」的判决性对照。

用户判断（2026-11-05）：「SimplCore 原生小参数模型本身没有问题，
是迁移脚本的问题」。本脚本把这句话拆成两个可证伪命题，用同一任务、
同一预算、同一 teacher 判决：

  H1「脚本问题」：修好脚本（改目标函数 + 全参数可训 + 每位置分布蒸馏）
      后，迁移学生应显著优于从零训，并达到/接近 teacher。
  H2「架构问题」：静态核在 key→value 绑定轴有函数类上限，修脚本只能
      改善收敛速度，改不了天花板；换 CAS（少一条纯粹性）才能越过。

修了什么（对照 v1~v5 与 181 步自监督那次）
------------------------------------------------
1. 目标函数：不再逼 SGC 内部隐状态/注入量像 attention（v1~v5 全部），
   改为**只对齐 teacher 的冻结输出分布**（黑盒 logits）。
2. 初始化：不再把 attention profile 的 LS 拟合当最终权重，只把它当
   **暖启动**（emb / head 逐位复制，mixer 保留 linear 网格初始化）。
3. 冻结策略：不再冻结 emb/FFN/head 只训 mixer（181 步实验证伪），
   **全参数可训**。
4. 蒸馏粒度：不再单末位 CE / T=1 / 无 teacher，改为
   **每位置 KL(T=4) + 末位 CE，α 0.7→0.3 线性退火**。
5. 主指标：不再用残差流余弦，用**任务 acc / val CE 曲线 / 收敛步数**。

四臂（同 teacher、同步数、同 seed、同评估集）
------------------------------------------------
  sgc_scratch : SGC 随机初始化 + 末位 CE 从零训（原生基线）
  sgc_warm    : SGC 暖启动 + 末位 CE（只测暖启动收益）
  sgc_kd      : SGC 暖启动 + 全位置 KD + 末位 CE（**修好的脚本 + 重训**）
  cas_kd      : CAS 暖启动 + 同一套 KD + CE（少一条路线的对照）

任务：MQAR G=4，8 类，chance 0.125，L=96。 teacher = 2 层因果
self-attention（固定 seed=0，训 TEACHER_STEPS 步，state_dict 落盘复用）。

CLI（沙箱 502 重置后按 (arm,seed) 原子续跑）::

    python3 scripts/migration/distill_probe.py \
        --steps 600 --seeds 0 1 2 --out artifacts/distill_probe_raw.json
    ONLY=sgc_kd SEED=1 python3 scripts/migration/distill_probe.py --steps 600 ...
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from simplcore.core.eval.cas_probe import (
    _CASNet, _SGCNet, _TSNet, N_CLS, VOCAB, gen_mqar, make_arm,
    paired_bootstrap, GEN,
)

ALL_ARMS = ("sgc_scratch", "sgc_warm", "sgc_kd", "cas_kd")
_only = os.getenv("ONLY", "")
ARMS = tuple(a for a in ALL_ARMS if not _only or a in _only.split(","))
STATE_PATH = Path(os.getenv("DISTILL_STATE", "artifacts/distill_probe_state.json"))
TEACHER_PATH = Path(os.getenv("TEACHER_PT", "artifacts/distill_teacher.pt"))
TEACHER_STEPS = int(os.getenv("TEACHER_STEPS", "1000"))
TEMPERATURE = 4.0
ALPHA0, ALPHA1 = 0.7, 0.3


# ---------------------------------------------------------------------------
# teacher：训一次、落盘、所有臂共用（冻结，不反传）
# ---------------------------------------------------------------------------
def get_teacher(d: int, layers: int, lr: float, L: int,
                batch: int) -> _TSNet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t, _ = make_arm("ts", d, layers, 16, d, 0)
    if TEACHER_PATH.exists():
        t.load_state_dict(torch.load(TEACHER_PATH, map_location="cpu"))
        t.eval()
        return t
    torch.manual_seed(0)
    opt = torch.optim.AdamW(t.parameters(), lr=lr)
    g = GEN().manual_seed(0)
    t.train()
    for _ in range(TEACHER_STEPS):
        opt.zero_grad(set_to_none=True)
        ls = []
        for _ in range(batch):
            seq, tgt = gen_mqar(L, g)
            ls.append(F.cross_entropy(t(seq.unsqueeze(0))[:, -1, :],
                                      torch.tensor([tgt])))
        torch.stack(ls).mean().backward()
        opt.step()
    TEACHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.state_dict(), TEACHER_PATH)
    t.eval()
    return t


# ---------------------------------------------------------------------------
# student：暖启动只搬 emb / head（维度已核对一致），mixer 保留原初始化
# ---------------------------------------------------------------------------
def build_student(arm: str, sd: int, teacher: _TSNet, d: int,
                  layers: int, M: int, feat: int):
    kind = "cas_softmax" if arm.startswith("cas_") else "sgc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model, info = make_arm(kind, d, layers, M, feat, sd)
    if arm != "sgc_scratch":                      # 暖启动
        with torch.no_grad():
            model.emb.weight.copy_(teacher.emb.weight)
            model.head.weight.copy_(teacher.head.weight)
            model.head.bias.copy_(teacher.head.bias)
    info["warm_start"] = arm != "sgc_scratch"
    info["trainable_params"] = sum(p.numel() for p in model.parameters()
                                   if p.requires_grad)
    return model, info


@torch.no_grad()
def teacher_logits(teacher: _TSNet, seq: torch.Tensor) -> torch.Tensor:
    return teacher(seq.unsqueeze(0))


@torch.no_grad()
def eval_acc(model, sd: int, L: int, n_eval: int) -> float:
    g = GEN().manual_seed(sd + 7777)
    hit = 0
    model.eval()
    for _ in range(n_eval):
        seq, tgt = gen_mqar(L, g)
        hit += int(model(seq.unsqueeze(0))[0, -1].argmax().item() == tgt)
    return hit / n_eval


# ---------------------------------------------------------------------------
def train_one_seed(arm: str, sd: int, teacher: _TSNet, *, steps: int,
                   d: int, layers: int, M: int, feat: int, L: int,
                   batch: int, lr: float, n_eval: int) -> Dict[str, Any]:
    model, info = build_student(arm, sd, teacher, d, layers, M, feat)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = GEN().manual_seed(sd + 100)               # 各臂同数据流，公平
    use_kd = arm.endswith("_kd")
    model.train()
    curve: List[float] = []
    final = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        seqs, tgts = [], []
        for _ in range(batch):
            s, y = gen_mqar(L, g)
            seqs.append(s); tgts.append(y)
        seq = torch.stack(seqs)                   # (B,L)
        tgt = torch.tensor(tgts)
        z_s = model(seq)                          # (B,L,C)
        loss_ce = F.cross_entropy(z_s[:, -1, :], tgt)
        if use_kd:
            with torch.no_grad():
                z_t = teacher(seq)                # teacher 冻结，不反传
            p_t = F.softmax(z_t / TEMPERATURE, dim=-1)
            log_p_s = F.log_softmax(z_s / TEMPERATURE, dim=-1)
            # 每位置 KL 的平均（不是只末位）× T² —— 序列级蒸馏
            loss_kd = (F.kl_div(log_p_s, p_t, reduction="sum")
                       / (seq.shape[0] * seq.shape[1])
                       * TEMPERATURE * TEMPERATURE)
            a = ALPHA0 + (ALPHA1 - ALPHA0) * (step / steps)
            loss = a * loss_kd + (1.0 - a) * loss_ce
        else:
            loss = loss_ce
        loss.backward(); opt.step()
        final = float(loss.detach())
        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                hit = 0
                ge = GEN().manual_seed(sd + 5555)
                for _ in range(64):
                    s, y = gen_mqar(L, ge)
                    hit += int(model(s.unsqueeze(0))[0, -1].argmax().item() == y)
                curve.append([step, round(hit / 64, 4)])
            model.train()
    acc = eval_acc(model, sd, L, n_eval)
    del model, opt
    return {"acc": round(acc, 4), "acc_curve": curve,
            "final_loss": round(final, 5), "info": info}


# ---------------------------------------------------------------------------
def run(args) -> Dict[str, Any]:
    teacher = get_teacher(args.d, args.layers, args.lr, args.L, args.batch)
    tacc = eval_acc(teacher, 0, args.L, args.n_eval)
    print(f"  teacher (seed0, {TEACHER_STEPS}步) acc={tacc:.4f}", flush=True)

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    seeds = [int(os.getenv("SEED"))] if os.getenv("SEED") else args.seeds
    for arm in ARMS:
        rec = rows.get(arm, {"units": {}})
        for sd in seeds:
            key = str(sd)
            if key in rec["units"]:
                print(f"    [{arm}] seed={sd} 已完成，跳过", flush=True)
                continue
            r = train_one_seed(arm, sd, teacher, steps=args.steps,
                               d=args.d, layers=args.layers, M=args.M,
                               feat=args.feat, L=args.L, batch=args.batch,
                               lr=args.lr, n_eval=args.n_eval)
            rec["units"][key] = r
            rows[arm] = rec
            tmp = STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, STATE_PATH)           # 每 (arm,seed) 原子落盘
            print(f"    [{arm}] seed={sd} acc={r['acc']:.4f} "
                  f"loss={r['final_loss']:.3f} curve={r['acc_curve']}",
                  flush=True)
        accs = [rec["units"][k]["acc"] for k in rec["units"]]
        rec["accs"] = accs
        rec["mean"] = round(sum(accs) / len(accs), 4) if accs else None
        rec["info"] = rec["units"][str(seeds[-1])]["info"]
        rows[arm] = rec
        print(f"  {arm:<14} mean={rec['mean']} accs={accs} "
              f"params={rec['info']['trainable_params']} "
              f"state={rec['info']['state_bytes']}B", flush=True)

    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)

    base = rows.get("sgc_scratch", {}).get("accs")
    comps = ({f"{a} - sgc_scratch":
              paired_bootstrap(rows[a]["accs"], base, args.n_boot)
              for a in ALL_ARMS
              if a != "sgc_scratch" and a in rows and base
              and len(rows[a]["accs"]) == len(base)})
    return {
        "task": "MQAR G=4, 8 classes, chance=0.125, L=96",
        "question": "SimplCore 原生小模型没问题 / 是迁移脚本的问题？",
        "fixes_applied": [
            "目标函数：隐状态对齐 → 只对齐 teacher 冻结输出分布",
            "初始化：profile LS 拟合当最终权重 → 只作暖启动(emb/head逐位复制)",
            "冻结策略：只训 mixer → 全参数可训",
            "蒸馏：单末位 CE/T=1 → 每位置 KL(T=4)+末位 CE，α 0.7→0.3",
            "主指标：残差流余弦 → 任务 acc / val CE / 收敛曲线",
        ],
        "config": dict(steps=args.steps, seeds=args.seeds, d=args.d,
                       layers=args.layers, M=args.M, feat_dim=args.feat,
                       L=args.L, batch=args.batch, lr=args.lr,
                       n_eval=args.n_eval, teacher_steps=TEACHER_STEPS,
                       temperature=TEMPERATURE, alpha=[ALPHA0, ALPHA1]),
        "teacher_acc": round(tacc, 4),
        "chance": 1.0 / N_CLS,
        "fairness_note": (
            "四臂同一 teacher、同步数、同 seed、同数据流、同评估集；"
            "sgc 用 init_mode='linear' 最强静态网格；参数量不等逐臂记录，"
            "不做等参断言。teacher 未调参（统一 lr）。"),
        "honest_boundary": (
            "D=32 / 2 层 / 合成 MQAR / 3 seeds 的小规模前哨，判的是"
            "「修脚本能否越过静态核函数类上限」这一机制问题；"
            "不等于 1.7B 真实语言生成。本仓从未从零训过 1.7B 语言模型，"
            "因此『原生模型在真实汉语上没问题』既未证实也未证伪。"),
        "rows": rows, "paired_vs_scratch": comps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="修好迁移脚本后重新训的判决性对照")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--feat", type=int, default=32)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/distill_probe_raw.json")
    a = ap.parse_args(argv)
    print("=" * 76)
    print(f"distill_probe  steps={a.steps} seeds={a.seeds} D={a.d} "
          f"M={a.M} F={a.feat} L={a.L} T={TEMPERATURE}")
    print("=" * 76, flush=True)
    res = run(a)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  ✓ {out}")
    for k, v in res["paired_vs_scratch"].items():
        print(f"    {k:<24} {v['diff']:+.4f} CI95={v['ci95']} "
              f"显著={v['significant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
