#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cas_probe —— 「少一条纯粹性」的真实收益：MQAR 轴判据性对照。

问题（TS2SC-Q17 v1~v5 实测）：静态、输入无关的谱核在 key→value 内容
寻址轴上撞到函数类上限（MQAR 0.1133~0.3066，TS 0.3945~0.500），
v4 已吃满静态核 oracle，加基/加门控/加适配/加步数都绕不过。

本脚本把三条路放在**同一任务、同一步数、同一 seed** 下比较：

* ``sgc``        : 静态谱核（**用 init_mode="linear" 的最强配置**，
                   不拿塌缩的 logfreq 字典当靶子）
* ``cas_elu``    : 内容写入 O(1) 叠加状态，φ=1+elu
* ``cas_softmax``: 同上，φ=特征维 softmax（可区分度更高）
* ``ts``         : 2 层因果 self-attention（含位置维 softmax，作为上界参照）

判据不是「CAS 赢了没有」这么简单，而是**成本—收益账**：

* 收益：MQAR 准确率相对静态核的提升（配对 bootstrap CI95）；
* 成本：状态字节、每 token FLOPs、以及「核输入无关」这条纯度。

CLI::

    python -m simplcore.core.eval.cas_probe --steps 1500 --seeds 0 1 2 \
        --out artifacts/cas_probe_raw.json
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

from ..arch.content_state import ContentAddressedState
from ..arch.sgc import SpectralGlobalConv

GEN = torch.Generator
VOCAB = 32
N_CLS = 8
NUM_KV = 4
CHANCE = 1.0 / N_CLS


# ---------------------------------------------------------------------------
# MQAR G=4：前缀 4 对 (key,value)，末位 query，目标 = 绑定的 value 类别
# ---------------------------------------------------------------------------
def gen_mqar(L: int, g: GEN) -> Tuple[torch.Tensor, int]:
    keys = torch.randperm(N_CLS, generator=g)[:NUM_KV]
    vals = torch.randint(0, N_CLS, (NUM_KV,), generator=g)
    pairs: List[int] = []
    for k, v in zip(keys.tolist(), vals.tolist()):
        pairs += [k, N_CLS + v]                       # value token ∈ [8,16)
    q_idx = int(torch.randint(0, NUM_KV, (1,), generator=g))
    seq = torch.randint(2 * N_CLS, VOCAB, (L,), generator=g)  # noise [16,32)
    seq[:len(pairs)] = torch.tensor(pairs, dtype=seq.dtype)
    seq[-1] = int(keys[q_idx])
    return seq, int(vals[q_idx])


# ---------------------------------------------------------------------------
# 三个臂：结构语义一致（pre-norm mixer + FFN），只有 mixer 不同
# ---------------------------------------------------------------------------
class _FFN(nn.Module):
    def __init__(self, d: int, ratio: float = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(d, int(d * ratio))
        self.fc2 = nn.Linear(int(d * ratio), d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _SGCNet(nn.Module):
    """静态核臂：SpectralGlobalConv 自带 norm+residual，外接 FFN。"""

    def __init__(self, d: int, layers: int, M: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, d)
        self.mixers = nn.ModuleList([
            SpectralGlobalConv(
                dim=d, num_bases=M, decay_max=256.0, local_kernel=4,
                init_mode="linear", linear_floor=0.3,
                content_gate=False, evidence_gate=False,
                uncertainty_gate=False)
            for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([_FFN(d) for _ in range(layers)])
        self.head = nn.Linear(d, N_CLS)
        self._state_bytes = sum(
            m.streaming_state_bytes(1) if hasattr(m, "streaming_state_bytes")
            else 0 for m in self.mixers)

    def forward(self, idx):
        h = self.emb(idx)
        for mx, ln, ff in zip(self.mixers, self.ln2, self.ffn):
            h = mx(h)                       # 内部含 norm + residual
            h = h + ff(ln(h))
        return self.head(h)


class _CASNet(nn.Module):
    """内容写入 O(1) 状态臂。"""

    def __init__(self, d: int, layers: int, feat_dim: int, phi: str,
                 seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.mixers = nn.ModuleList([
            ContentAddressedState(dim=d, feat_dim=feat_dim, value_dim=d,
                                  phi=phi, zero_init_readout=False)
            for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([_FFN(d) for _ in range(layers)])
        self.head = nn.Linear(d, N_CLS)
        self._state_bytes = sum(m.state_bytes(1) for m in self.mixers)
        self._flops = sum(m.contract()["flops_per_token"] for m in self.mixers)

    def forward(self, idx):
        h = self.emb(idx)
        for ln1, mx, ln2, ff in zip(self.ln1, self.mixers, self.ln2, self.ffn):
            h = h + mx(ln1(h))
            h = h + ff(ln2(h))
        return self.head(h)


class _TSNet(nn.Module):
    """上界参照：2 层因果 self-attention（位置维 softmax，N×N）。"""

    def __init__(self, d: int, layers: int, seed: int, heads: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * 2,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, N_CLS)

    def forward(self, idx):
        L = idx.shape[1]
        h = self.emb(idx) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        return self.head(self.enc(h, mask=mask))


ALL_ARMS = ("sgc", "cas_elu", "cas_softmax", "ts")
import os
_only = os.getenv("ONLY", "")
ARMS = tuple(a for a in ALL_ARMS if not _only or a in _only.split(","))
STATE_PATH = Path(os.getenv("CAS_STATE", "artifacts/cas_probe_state.json"))


def make_arm(arm: str, d: int, layers: int, M: int, feat: int,
             seed: int) -> Tuple[nn.Module, Dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if arm == "sgc":
            m = _SGCNet(d, layers, M, seed)
            info = {"state_bytes": m._state_bytes, "flops_per_token": M * d,
                    "no_NxN": True, "kernel_input_independent": True}
        elif arm == "cas_elu":
            m = _CASNet(d, layers, feat, "elu1", seed)
            info = {"state_bytes": m._state_bytes,
                    "flops_per_token": m._flops, "no_NxN": True,
                    "kernel_input_independent": False}
        elif arm == "cas_softmax":
            m = _CASNet(d, layers, feat, "softmax", seed)
            info = {"state_bytes": m._state_bytes,
                    "flops_per_token": m._flops, "no_NxN": True,
                    "kernel_input_independent": False}
        elif arm == "ts":
            m = _TSNet(d, layers, seed)
            info = {"state_bytes": 0, "flops_per_token": None,
                    "no_NxN": False, "kernel_input_independent": False,
                    "note": "KV cache 随 L 增长，state_bytes 记为 0 是口径占位"}
        else:
            raise ValueError(arm)
    info["params"] = sum(p.numel() for p in m.parameters())
    return m, info


# ---------------------------------------------------------------------------
def train_one_seed(arm: str, sd: int, *, steps: int, d: int,
                    layers: int, M: int, feat: int, L: int, batch: int,
                    lr: float, n_eval: int) -> Tuple[float, float, Dict]:
    """跑 (arm, seed) 一个单元；返回 (acc, final_loss, info)。"""
    torch.manual_seed(sd)
    model, info = make_arm(arm, d, layers, M, feat, sd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = GEN().manual_seed(sd)
    model.train()
    last = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ls = []
        for _ in range(batch):
            seq, tgt = gen_mqar(L, g)
            logits = model(seq.unsqueeze(0))[:, -1, :]
            ls.append(F.cross_entropy(logits, torch.tensor([tgt])))
        loss = torch.stack(ls).mean()
        loss.backward(); opt.step()
        last = float(loss.detach())
    model.eval()
    ge = GEN().manual_seed(sd + 7777)
    hit = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = gen_mqar(L, ge)
            hit += int(model(seq.unsqueeze(0))[0, -1].argmax().item() == tgt)
    acc = hit / n_eval
    del model, opt
    return acc, last, info


def paired_bootstrap(a: List[float], b: List[float], n_boot: int,
                     seed: int = 0) -> Dict[str, Any]:
    g = GEN().manual_seed(seed)
    n = min(len(a), len(b))          # 分 seed 续跑时可能尚未齐，按共有 seed 配对
    diff = torch.tensor(a[:n]) - torch.tensor(b[:n])
    idx = torch.randint(0, len(diff), (n_boot, len(diff)), generator=g)
    means = diff[idx].mean(1)
    lo, hi = float(torch.quantile(means, .025)), float(torch.quantile(means, .975))
    return {"diff": round(float(diff.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def run(args) -> Dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    for arm in ARMS:
        rec = rows.get(arm, {"accs": [], "seeds_done": [],
                             "final_loss": [], "info": None})
        if "seeds_done" not in rec:
            # 兼容旧版按臂粒度落盘的 state（已有 3 个 seed 的结果不重跑）
            rec["seeds_done"] = list(range(len(rec["accs"])))
            rec.pop("arm", None)
        for sd in args.seeds:
            if sd in rec["seeds_done"]:
                continue
            acc, loss, info = train_one_seed(
                arm, sd, steps=args.steps, d=args.d, layers=args.layers,
                M=args.M, feat=args.feat, L=args.L, batch=args.batch,
                lr=args.lr, n_eval=args.n_eval)
            rec["accs"].append(round(acc, 4))
            rec["seeds_done"].append(sd)
            rec["final_loss"].append(round(loss, 5))
            rec["info"] = rec["info"] or info
            rows[arm] = rec
            tmp = STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, STATE_PATH)     # 每个 seed 原子落盘，抗 502
            print(f"    [{arm}] seed={sd} acc={acc:.4f} loss={loss:.3f}",
                  flush=True)
        rec["mean"] = round(sum(rec["accs"]) / len(rec["accs"]), 4)
        rows[arm] = rec
        print(f"  {arm:<14} mean={rec['mean']:.4f} accs={rec['accs']} "
              f"params={rec['info']['params']} "
              f"state={rec['info']['state_bytes']}B", flush=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)
    sgc = rows.get("sgc", {}).get("accs")
    comps = ({f"{a} - sgc": paired_bootstrap(rows[a]["accs"], sgc, args.n_boot)
              for a in ALL_ARMS if a != "sgc" and a in rows}
             if sgc else {})
    return {
        "task": "MQAR G=4, 8 classes, chance=0.125, L=96",
        "config": dict(steps=args.steps, seeds=args.seeds, d=args.d,
                       layers=args.layers, M=args.M, feat_dim=args.feat,
                       L=args.L, batch=args.batch, lr=args.lr,
                       n_eval=args.n_eval, n_boot=args.n_boot),
        "chance": CHANCE,
        "fairness_note": (
            "sgc 臂使用 init_mode='linear' 的最强静态网格，不是塌缩的 "
            "logfreq 字典；三臂同结构语义（pre-norm mixer + FFN）、"
            "同步数、同 seed、同评估集。参数量不等，逐臂记录，不做等参断言。"),
        "honest_boundary": (
            "小规模合成前哨（D=32/2 层/3 seeds），只判「内容写入状态能否"
            "补上静态核的 MQAR 缺口」这一机制方向；不等于 1.7B 语言生成"
            "能力，也不证明 CAS 在真实语料上可用。"),
        "rows": rows, "paired_vs_sgc": comps,
        "arms_in_state": sorted(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CAS MQAR 判据性对照")
    ap.add_argument("--steps", type=int, default=1500)
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
    ap.add_argument("--out", default="artifacts/cas_probe_raw.json")
    a = ap.parse_args(argv)
    print("=" * 74)
    print(f"cas_probe  MQAR G=4  steps={a.steps} seeds={a.seeds} "
          f"D={a.d} M={a.M} F={a.feat} L={a.L}")
    print("=" * 74, flush=True)
    res = run(a)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  ✓ {out}")
    for k, v in res["paired_vs_sgc"].items():
        print(f"    {k:<22} {v['diff']:+.4f}  CI95={v['ci95']}  "
              f"显著={v['significant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
