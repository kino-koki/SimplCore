# Copyright 2026 (C) 六安市青奈科技有限责任公司
# purity: allow-attention-reference —— 本模块内含 **TS 对照臂**（迷你标准
# Transformer），用于轴 C/D 能力对照的对照组；对照臂自包含、不进运行时、
# 不进 mixer 注册表（先例：equal_budget.py 的 TS 臂）。定义级豁免见
# purity._SOURCE_SCAN_DEFINITION_EXEMPT_FILES，移除即恢复全量守卫。
# -*- coding: utf-8 -*-
"""轴 C/D 能力对照协议（``X-axis-probe``，0.1-α）—— 把 2026-10-31 的
独立复现固化为**可持续重跑**的对照 harness.

动机（负面结果照账，见 ``docs/THEORETICAL_EDGE.md`` 实测对照节）：
2 层精确指向 SGC 0.398 vs attention 1.000（结构性劣势，信息论上限）；
聚合 L=129 attention 0.938 vs SGC 0.875（"等效 4–7B"的通过率优势未复现，
降级为"约一半参数同等质量 + 成本优势"）。本协议把这两条前哨信号做成
**同口径、同 seed、可持续验证**的对照——§5.13 战役运行前，它是唯一的
可复现证据链。

四任务 × 四臂：

| 任务 | 能力类型 | chance |
|------|----------|--------|
| ``precise_pointer`` | 轴 D：找到 marker 并复制其 value（精确指向） | 1/16 |
| ``aggregate_count`` | 轴 C 前哨：特殊 token 计数 mod K（聚合型） | 1/K |
| ``counting`` | 近精确计数（0–8，一票之差） | 1/9 |
| ``markov`` | 一阶 Markov 语言建模（NLL，非准确率） | log V |

臂：``sgc``（默认 mixer）/ ``local``（num_bases=2 退化近局部的 SGC）/ 
``ts``（本文件内自包含迷你 Transformer 对照臂）/ chance（解析基线）。

诚实口径
--------
* 小规模前哨：D=32/64、1–2 层、数百到三千步、合成任务、**无显著性检验**
  ——足以区分数量级差异，不足以支撑任何定量的"谁更好"；
* 参数量随臂报告（同 D 下 SGC 少一个 D² 投影，对照口径对 SGC 有利）；
* chance 为解析值；TS 臂不进运行时、不构成产品线组件。
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import SimplCoreConfig
from ..model.model import SimplCoreModel

__all__ = ["TASKS", "ARMS", "run_probe", "AxisProbeReport"]

VOCAB = 32
MARKER = VOCAB - 1          # 31：needle 标记
SPECIAL = VOCAB - 2         # 30：计数目标 token
CLS_BEGIN = VOCAB           # 输出类占用 [VOCAB, VOCAB+C)（扩表头）
GEN = torch.Generator


# ---------------------------------------------------------------------------
# 任务生成器：每个任务返回 (输入序列, 目标类)。输出头类别数随任务。
# ---------------------------------------------------------------------------
def gen_precise_pointer(L: int, g: torch.Generator) -> Tuple[torch.Tensor, int]:
    """序列中段埋 (MARKER, value)，末尾再放 MARKER；目标 = value ∈ [0,16)."""
    value = int(torch.randint(0, 16, (1,), generator=g))
    noise = torch.randint(0, MARKER, (L - 3,), generator=g)
    mid = L // 2
    seq = torch.cat([noise[:mid],
                     torch.tensor([MARKER, value]),
                     noise[mid:], torch.tensor([MARKER])])
    return seq, value


def gen_aggregate_count(L: int, g: torch.Generator, k_mod: int = 4) -> Tuple[torch.Tensor, int]:
    """前缀中 SPECIAL 的出现次数 mod k_mod（聚合型，O(1) 压缩可行）。"""
    seq = torch.randint(0, SPECIAL, (L,), generator=g)
    hits = torch.rand(L, generator=g) < 0.15
    seq[hits] = SPECIAL
    return seq, int(hits.sum().item()) % k_mod


def gen_counting(L: int, g: torch.Generator) -> Tuple[torch.Tensor, int]:
    """SPECIAL 的精确个数（0..8；超出截断到 8）。"""
    seq = torch.randint(0, SPECIAL, (L,), generator=g)
    hits = torch.rand(L, generator=g) < 0.05
    seq[hits] = SPECIAL
    return seq, min(int(hits.sum().item()), 8)


def gen_markov(L: int, g: torch.Generator, V: int = 16) -> Tuple[torch.Tensor, int]:
    """一阶 Markov 链（转移矩阵随 g），目标 = 最后位置的下一 token。"""
    P = torch.softmax(torch.rand(V, V, generator=g) * 2.0, dim=1)
    seq = torch.zeros(L, dtype=torch.long)
    seq[0] = int(torch.randint(0, V, (1,), generator=g))
    for t in range(1, L):
        seq[t] = int(torch.multinomial(P[seq[t - 1]], 1, generator=g))
    return seq, int(seq[-1])


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    # name: (生成器, 类数, chance, 序列长)
    "precise_pointer": (gen_precise_pointer, 16, 1 / 16, 64),
    "aggregate_count": (gen_aggregate_count, 4, 0.25, 129),
    "counting": (gen_counting, 9, 1 / 9, 96),
    "markov": (gen_markov, 16, 1 / 16, 64),
}


# ---------------------------------------------------------------------------
# 臂
# ---------------------------------------------------------------------------
def _sgc_like(dim: int, layers: int, num_bases: int, vocab_out: int,
              seed: int) -> SimplCoreModel:
    torch.manual_seed(seed)
    cfg = SimplCoreConfig(vocab_size=VOCAB, dim=dim, num_layers=layers,
                          num_bases=num_bases, ffn_ratio=2.0,
                          decay_max=256.0, local_kernel=4,
                          tie_embeddings=False)
    model = SimplCoreModel(cfg)
    # 输出类别 > 词表时扩表头（任务类别空间与输入 vocab 解耦）
    if vocab_out > VOCAB:
        model.head = nn.Linear(cfg.dim, vocab_out)
    return model


class _TinyTS(nn.Module):
    """TS 对照臂：2 层因果 self-attention 迷你 Transformer（自包含）."""

    def __init__(self, dim: int, layers: int, vocab_out: int, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.zeros(1, 512, dim))   # TS 需要位置表
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(dim, vocab_out)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        L = idx.shape[1]
        x = self.emb(idx) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                     device=idx.device), 1)
        h = self.enc(x, mask=mask)
        return self.head(h)


def _make_arm(arm: str, dim: int, layers: int, vocab_out: int,
              seed: int) -> Tuple[nn.Module, Dict[str, int]]:
    if arm == "sgc":
        m = _sgc_like(dim, layers, num_bases=64, vocab_out=vocab_out, seed=seed)
    elif arm == "local":
        m = _sgc_like(dim, layers, num_bases=2, vocab_out=vocab_out, seed=seed)
    elif arm == "ts":
        torch.manual_seed(seed)
        m = _TinyTS(dim, layers, vocab_out)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return m, {"params": sum(p.numel() for p in m.parameters())}


def _loss_and_acc(model: nn.Module, task: str, seq: torch.Tensor,
                  target: int) -> Tuple[torch.Tensor, bool]:
    """last-position CE + argmax 命中（markov 用 NLL，命中 = top1==target）。"""
    logits = model(seq.unsqueeze(0))[:, -1, :]
    tgt = torch.tensor([target])
    hit = bool(logits.argmax(dim=-1).item() == target)
    return F.cross_entropy(logits, tgt), hit


def _train_arm(model: nn.Module, task: str, steps: int, batch: int,
               lr: float, seed: int, device) -> float:
    gen = GEN().manual_seed(seed)
    gen_fn, n_cls, _, L = TASK_SPECS[task]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    final_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(batch):
            seq, tgt = gen_fn(L, gen)
            loss, _ = _loss_and_acc(model, task, seq.to(device), tgt)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
    return final_loss


@torch.no_grad()
def _eval_arm(model: nn.Module, task: str, n_eval: int, seed: int,
              device) -> float:
    gen = GEN().manual_seed(seed + 777)
    gen_fn, n_cls, _, L = TASK_SPECS[task]
    model.eval()
    hits = 0
    for _ in range(n_eval):
        seq, tgt = gen_fn(L, gen)
        _, hit = _loss_and_acc(model, task, seq.to(device), tgt)
        hits += hit
    return hits / n_eval


@dataclass
class AxisProbeReport:
    """对照账本：任务 → 臂 → 准确率/NLL + 参数量 + 口径注记."""
    rows: Dict[str, Dict[str, Any]]
    note: str = (
        "小规模前哨（D=32/64、1–2 层、合成任务、无显著性检验）——足以区分"
        "数量级差异，不足以支撑定量的「谁更好」。真实答案等 §5.13 战役。"
        "参数量口径对 SGC 有利（同 D 少一个 D² 投影）。")

    def as_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "note": self.note}


def run_probe(tasks: Optional[List[str]] = None,
              arms: Optional[List[str]] = None,
              steps: int = 600, dim: int = 32, layers: int = 2,
              batch: int = 8, lr: float = 3e-3, n_eval: int = 256,
              seed: int = 0, device: Optional[str] = None) -> AxisProbeReport:
    """跑对照：每个 (任务, 臂) 训练 ``steps`` 步 → 评测 → 报告.

    seed 全程固定：同参数重跑结果确定性一致（可复现性由测试锁定）。
    """
    tasks = tasks or list(TASK_SPECS)
    arms = arms or ["sgc", "local", "ts"]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        gen_fn, n_cls, chance, _ = TASK_SPECS[task]
        row: Dict[str, Any] = {"chance": round(chance, 4)}
        for arm in arms:
            model, info = _make_arm(arm, dim, layers,
                                    vocab_out=VOCAB + n_cls, seed=seed)
            model.to(dev)
            _train_arm(model, task, steps, batch, lr, seed, dev)
            acc = _eval_arm(model, task, n_eval, seed, dev)
            row[arm] = {"acc": round(acc, 4), "params": info["params"]}
        rows[task] = row
    return AxisProbeReport(rows=rows)


def bootstrap_ci(values: List[float], n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """均值的 bootstrap 置信区间（percentile 法，fp64）.

    复现报告二期点名的能力：单 seed 数字在噪声内没有结论力（attention 的
    seed SD≈0.066 vs 组间差 0.0156）——**≥5 seed + bootstrap CI** 是对照
    协议的最低统计门槛。
    """
    vals = torch.tensor(values, dtype=torch.float64)
    n = len(vals)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = vals[idx].mean(dim=1)
    lo = float(torch.quantile(means, alpha / 2))
    hi = float(torch.quantile(means, 1 - alpha / 2))
    return lo, hi


def run_probe_multiseed(tasks: Optional[List[str]] = None,
                        arms: Optional[List[str]] = None,
                        seeds: Sequence[int] = (0, 1, 2, 3, 4),
                        steps: int = 600, dim: int = 32, layers: int = 2,
                        batch: int = 8, lr: float = 3e-3, n_eval: int = 256,
                        n_boot: int = 2000,
                        device: Optional[str] = None) -> AxisProbeReport:
    """多 seed 对照 + bootstrap CI95（复现报告二期的统计门槛）.

    报告结构：task → arm → {"accs": [每 seed], "mean", "ci95": (lo, hi),
    "params"}。**判定纪律**：均值差落在对方 CI 内 = 噪声内（不得宣称
    胜出）；CI 不重叠才可写"高于"。
    """
    tasks = tasks or list(TASK_SPECS)
    arms = arms or ["sgc", "local", "ts"]
    rows: Dict[str, Dict[str, Any]] = {}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for task in tasks:
        row: Dict[str, Any] = {"chance": round(TASK_SPECS[task][2], 4)}
        for arm in arms:
            accs: List[float] = []
            info = {"params": 0}
            for sd in seeds:
                model, info = _make_arm(arm, dim, layers,
                                        vocab_out=VOCAB + TASK_SPECS[task][1],
                                        seed=sd)
                model.to(dev)
                _train_arm(model, task, steps, batch, lr, sd, dev)
                accs.append(_eval_arm(model, task, n_eval, seed=sd, device=dev))
            lo, hi = bootstrap_ci(accs, n_boot=n_boot, seed=0)
            row[arm] = {"accs": [round(a, 4) for a in accs],
                        "mean": round(sum(accs) / len(accs), 4),
                        "ci95": (round(lo, 4), round(hi, 4)),
                        "params": info["params"]}
        rows[task] = row
    return AxisProbeReport(rows=rows, note=AxisProbeReport(rows={}).note +
                           " 多 seed+bootstrap CI95（percentile，2000 次）。"
                           "判定纪律：均值差落在对方 CI 内 = 噪声内。")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="轴 C/D 能力对照（X-axis-probe）")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_SPECS))
    ap.add_argument("--arms", nargs="+", default=["sgc", "local", "ts"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="多 seed 模式（如 --seeds 0 1 2 3 4）→ bootstrap CI95")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.seeds:
        rep = run_probe_multiseed(tasks=args.tasks, arms=args.arms,
                                  seeds=args.seeds, steps=args.steps,
                                  dim=args.dim, layers=args.layers,
                                  n_eval=args.n_eval, n_boot=args.n_boot)
        for task, row in rep.rows.items():
            print(f"== {task} ==")
            for arm, v in row.items():
                print(f"  {arm:>6}: {v}")
        print(f"口径: {rep.note}")
        return 0
    rep = run_probe(tasks=args.tasks, arms=args.arms, steps=args.steps,
                    dim=args.dim, layers=args.layers, n_eval=args.n_eval,
                    seed=args.seed)
    for task, row in rep.rows.items():
        print(f"== {task} ==")
        for arm, v in row.items():
            print(f"  {arm:>6}: {v}")
    print(f"\n口径: {rep.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())