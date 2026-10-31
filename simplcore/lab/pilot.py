# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Pilot：A1/A2 最小证伪实验 —— 「自主学习闭环」在小规模上跑真数据。

动机
----
Sprint-30 / 120B 账本都押在同一个未验证主张上：**自主改进闭环真的
能产生持续、不退步的学习**。回答它不需要 1B×30 天——需要的是
`dim=32` 级小模型 + 真实沙箱 + 若干轮闭环，在 CPU 上几分钟出信号。

闭环（复用现有机制，零新轮子）
------------------------------
::

    SimplCoreModel(冻结) + LearnedAdapter(可学)
        ↓ generate_text（top-1 贪心）
    PilotCodeEnvironment（code_exec 真实沙箱执行判分）
        ↓ 失败轨迹 + 正确实现
    summarize_failures → 训练样本
        ↓
    default_consolidator（只训适配器）
        ↓
    regression_gate（配对 bootstrap，A2）
        ├─ 通过 → 提交
        └─ 退化 → 逐位回滚

判定口径（诚实）
----------------
* **A1（持续改进）**：闭环前后 verifier pass_rate（逐题、真实执行）；
  同时报告蒸馏 loss_before/after——**学习信号 ≠ 生成质量**，两者分开记账。
* **A2（不失忆）**：每轮 no-regression gate 的提交/回滚计数与失败指标。
* 合成小题上的结果**只证明管线语义**，禁止外推为能力宣称
  （与 phase1_train 的 synthetic 诚实边界同口径）。
* 无生成质量提升 ≠ 机制失败：dim=32 的字符级模型本就不会写代码；
  pilot 证明的是**闭环可运行 + 账本可信**，把"规模够时是否有效"留给
  实机档位（``--scale gpu`` 说明见 README）。

用法
----
CPU 冒烟（本机即可）::

    python -m simplcore.core.pilot --rounds 2 --out runs/pilot

实机档位（MI250，配合 phase1_train 预训练后的 checkpoint）::

    python -m simplcore.core.pilot --rounds 8 --dim 256 --layers 6 \\
        --consolidate-epochs 4 --out runs/pilot-gpu
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.memory.code_exec import (
    CODE_PROMPT,
    CodeExecutionVerifier,
    CodeProblem,
    make_smoke_problems,
)
from ..core.model.config import SimplCoreConfig
from ..core.memory.dream import EvalScores
from ..core.training.envlearn import (
    EnvironmentLearner,
    EnvTrajectory,
    LearnedAdapter,
    LocalCodeEnvironment,
    RolloutResult,
    WeightBudget,
)
from ..core.model.model import SimplCoreModel
from ..core.model.tokenization import CharTokenizer

__all__ = [
    "PilotConfig",
    "PilotCodeEnvironment",
    "default_reference_problems",
    "run_pilot",
    "main",
]


# ---------------------------------------------------------------------------
# 题库：冒烟 3 题 + 参考实现（失败轨迹的蒸馏目标）
# ---------------------------------------------------------------------------
_REFERENCES: Dict[str, str] = {
    "add": "def add(a, b):\n    return a + b\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "reverse_str": "def reverse_str(s):\n    return s[::-1]\n",
    "count_vowels": (
        "def count_vowels(s):\n"
        "    return sum(1 for c in s.lower() if c in 'aeiou')\n"
    ),
    "max_of_list": (
        "def max_of_list(xs):\n"
        "    m = xs[0]\n"
        "    for x in xs[1:]:\n"
        "        if x > m:\n"
        "            m = x\n"
        "    return m\n"
    ),
}


def default_reference_problems():
    """内置题库：每题带 ``meta["reference"]``（失败时的蒸馏目标）。"""
    problems = make_smoke_problems()
    problems.append(CodeProblem(
        name="count_vowels",
        prompt="实现函数 count_vowels(s)，返回字符串 s 中元音字母（aeiou，"
               "不区分大小写）的个数。",
        tests=['assert count_vowels("hello") == 2',
               'assert count_vowels("") == 0',
               'assert count_vowels("AEIOUxyz") == 5'],
    ))
    problems[-1].meta["reference"] = _REFERENCES["count_vowels"]
    problems.append(CodeProblem(
        name="max_of_list",
        prompt="实现函数 max_of_list(xs)，返回非空整数列表 xs 中的最大值"
               "（不得使用内置 max）。",
        tests=["assert max_of_list([1, 5, 3]) == 5",
               "assert max_of_list([-2, -7, -1]) == -1",
               "assert max_of_list([42]) == 42"],
    ))
    problems[-1].meta["reference"] = _REFERENCES["max_of_list"]
    for p in problems:
        p.meta.setdefault("reference", _REFERENCES.get(p.name))
    return problems


class PilotCodeEnvironment(LocalCodeEnvironment):
    """LocalCodeEnvironment + 失败时回填参考实现（summarize_failures 需要它）。"""

    def rollout(self, generator) -> List[RolloutResult]:
        results = super().rollout(generator)
        for r, p in zip(results, self.problems):
            if not r.passed and r.correct is None:
                r.correct = p.meta.get("reference")
        return results


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class PilotConfig:
    """pilot 规模与闭环超参（CPU 冒烟默认值可直接跑）。"""
    dim: int = 32
    num_layers: int = 2
    num_bases: int = 8
    ffn_ratio: float = 2.0
    decay_max: float = 256.0
    adapter_blocks: int = 2
    rounds: int = 2
    max_new_tokens: int = 64
    consolidate_lr: float = 1e-2
    consolidate_epochs: int = 3
    batch_size: int = 2
    regression_tolerance: float = 0.0
    n_boot: int = 200
    seed: int = 7
    timeout_s: float = 5.0
    #: held-out 题集（泛化判据）。内置 5 题**同时**是训练经验来源——闭环前后
    #: 在内置题上的 pass_rate 上升只能证明**记忆（同题重放）**，不证明泛化。
    #: 泛化主张必须提供与经验不重叠的 heldout_problems，报告以它为准。
    heldout_problems: Optional[List[Any]] = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        hp = d.get("heldout_problems")
        if hp is not None:
            # CodeProblem 不可 JSON 序列化，报告里只记账题名
            d["heldout_problems"] = [getattr(p, "name", str(p)) for p in hp]
        return d


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _generator_factory(model: SimplCoreModel, codec, max_new_tokens: int):
    def _gen(instruction: str) -> str:
        prompt = CODE_PROMPT.format(instruction=instruction)
        return model.generate_text(prompt, codec,
                                   max_new_tokens=max_new_tokens,
                                   temperature=1.0, top_k=1)
    return _gen


def run_pilot(out_dir: Any = "runs/pilot",
              cfg_overrides: Optional[Dict[str, Any]] = None,
              pcfg: Optional[PilotConfig] = None) -> Dict[str, Any]:
    """跑完整 A1/A2 闭环并落盘报告。返回 report dict（JSON 可序列化）。"""
    import torch

    pc = pcfg or PilotConfig()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if not hasattr(pc, k):
                raise ValueError(f"unknown PilotConfig field: {k}")
            setattr(pc, k, v)
    torch.manual_seed(pc.seed)

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- 模型 + 适配器 --------------------------------------------------------
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(
        codec, dim=pc.dim, num_layers=pc.num_layers, num_bases=pc.num_bases,
        ffn_ratio=pc.ffn_ratio, decay_max=pc.decay_max, local_kernel=4)
    base = SimplCoreModel(cfg).eval()
    adapter = LearnedAdapter(dim=cfg.dim, num_blocks=pc.adapter_blocks)

    problems = default_reference_problems()
    env = PilotCodeEnvironment(problems, timeout_s=pc.timeout_s)
    gen = _generator_factory(base, codec, pc.max_new_tokens)
    raw_verifier = CodeExecutionVerifier(problems, gen, timeout_s=pc.timeout_s)

    # EnvironmentLearner 的 verifier 契约是 ``(gen) -> EvalScores``
    # （regression_gate 消费 .metrics）；CodeExecutionVerifier 返回 dict，
    # 这里包一层适配（分数仍来自同一个真实沙箱判分器）。
    def verifier(_gen=None) -> EvalScores:
        return EvalScores.from_any(raw_verifier())

    # -- held-out 判分器（泛化判据；与训练经验不重叠的题集） -------------------
    heldout_verifier = None
    if pc.heldout_problems:
        heldout_verifier = CodeExecutionVerifier(
            list(pc.heldout_problems), gen, timeout_s=pc.timeout_s)

    def _heldout_pass() -> Optional[float]:
        if heldout_verifier is None:
            return None
        s = heldout_verifier.evaluate()
        return (sum(s) / len(s)) if s else None

    learner = EnvironmentLearner(
        base, adapter, codec=codec,
        budget=WeightBudget(base_params=0, max_learned=10_000_000),
        env=env, verifier=verifier, generator=gen,
        regression_tolerance=pc.regression_tolerance,
        n_boot=pc.n_boot, seed=pc.seed,
        consolidate_lr=pc.consolidate_lr, epochs=pc.consolidate_epochs,
        batch_size=pc.batch_size,
    )

    ledger_path = out / "ledger.jsonl"
    ledger: List[Dict[str, Any]] = []

    def _scores() -> List[float]:
        return raw_verifier.evaluate()

    # -- 闭环前基线（A1 锚点） -------------------------------------------------
    baseline_scores = _scores()
    baseline_pass = sum(baseline_scores) / len(baseline_scores)
    baseline_heldout = _heldout_pass()
    ledger.append({"event": "baseline", "pass_rate": baseline_pass,
                   "per_problem": baseline_scores,
                   "heldout_pass_rate": baseline_heldout,
                   "detail": [r for r in raw_verifier.results]})
    with open(ledger_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(ledger[-1], ensure_ascii=False) + "\n")

    # -- 自主学习闭环（A1）+ 每轮 no-regression gate（A2） ----------------------
    stage_reports = []
    for rnd in range(1, pc.rounds + 1):
        rollouts = env.rollout(gen)
        passed = sum(1 for r in rollouts if r.passed)
        trajs = [
            EnvTrajectory(prompt=r.prompt, response=r.generated,
                          passed=r.passed, correct=r.correct,
                          meta={"detail": r.detail})
            for r in rollouts
        ]
        rep = learner.stage(trajs)
        stage_reports.append(rep)
        row = {"event": "round", "round": rnd, "raw_passed": passed,
               "raw_total": len(rollouts), **rep.as_dict()}
        ledger.append(row)
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- 闭环后终评（A1 锚点） -------------------------------------------------
    final_scores = _scores()
    final_pass = sum(final_scores) / len(final_scores)
    final_heldout = _heldout_pass()
    ledger.append({"event": "final", "pass_rate": final_pass,
                   "per_problem": final_scores,
                   "heldout_pass_rate": final_heldout,
                   "detail": [r for r in raw_verifier.results]})
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ledger[-1], ensure_ascii=False) + "\n")

    # -- 判定 ------------------------------------------------------------------
    loss_pairs = [(r.loss_before, r.loss_after) for r in stage_reports
                  if r.loss_before is not None and r.loss_after is not None]
    a1 = {
        "baseline_pass_rate": round(baseline_pass, 6),
        "final_pass_rate": round(final_pass, 6),
        "improved": bool(final_pass > baseline_pass),
        "learning_signal": [
            {"loss_before": b, "loss_after": a, "decreased": bool(a < b)}
            for b, a in loss_pairs],
        "note": ("pass_rate 是真实沙箱执行的逐题判分；learning_signal 只是蒸馏"
                 "目标的拟合程度——学习信号 ≠ 生成质量，分开记账；"
                 "内置 5 题同时是训练经验来源，其 pass_rate 上升只能证明"
                 "记忆（同题重放），不证明泛化"),
    }
    if heldout_verifier is not None:
        a1["heldout"] = {
            "num_problems": len(pc.heldout_problems),
            "baseline_pass_rate": (round(baseline_heldout, 6)
                                   if baseline_heldout is not None else None),
            "final_pass_rate": (round(final_heldout, 6)
                                if final_heldout is not None else None),
            "improved": bool(baseline_heldout is not None
                             and final_heldout is not None
                             and final_heldout > baseline_heldout),
            "note": "泛化判据：与训练经验不重叠的题集；「越用越好」以此为准",
        }
    a2 = {
        "stages": len(stage_reports),
        "committed": sum(1 for r in stage_reports if r.wrote_weights),
        "rolled_back": sum(1 for r in stage_reports if r.rolled_back),
        "failed_metrics_all": sorted({m for r in stage_reports
                                      for m in r.failed_metrics}),
        "note": "committed>0 且 rolled_back 合理 = gate 在工作；全部提交反而可疑",
    }
    # verdict：提供 held-out 时以泛化 improved 为准；否则内置题 improved
    # 只能记为"记忆证据"，verdict 最多到 inconclusive/pass（能力主张仍被
    # honesty 块关死）。
    heldout_improved = (a1.get("heldout", {}).get("improved")
                        if heldout_verifier is not None else None)
    generalized = heldout_improved if heldout_improved is not None else a1["improved"]
    verdict = "inconclusive" if (not generalized and a2["committed"] == 0) else (
        "pass" if generalized and a2["rolled_back"] >= 0 else "fail")
    if heldout_verifier is None:
        verdict = "inconclusive" if not a1["improved"] else (
            "memory-evidence-only")   # 无泛化判据：不给 pass

    report = {
        "protocol": "pilot-a1a2",
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": pc.as_dict(),
        "model_params": int(sum(p.numel() for p in base.parameters())),
        "adapter_params": int(adapter.num_params()),
        "num_problems": len(problems),
        "a1": a1,
        "a2": a2,
        "verdict": verdict,
        "honesty": {
            "claim_permitted": False,
            "reason": "内置合成小题 + 无预训练基座——本 pilot 只证明闭环管线"
                      "语义正确与账本可信，不构成任何能力宣称；能力判定"
                      "需实机预训练档位（Sprint-30 D27-D28 评测）",
            "memory_vs_generalization": (
                "内置 5 题即训练经验来源：其 pass_rate 上升只能证明记忆"
                "（同题重放）；泛化主张必须以 heldout_problems 的"
                "held-out pass_rate 为准" + (
                    "；本轮已提供 held-out 判据" if heldout_verifier is not None
                    else "；本轮未提供 heldout_problems，无泛化判据")),
        },
        "wall_seconds": round(time.time() - t0, 3),
        "artifacts": {"ledger": str(ledger_path)},
    }
    report_path = out / "report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    report["artifacts"]["report"] = str(report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pilot：A1/A2 最小证伪实验（真实沙箱闭环）")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--adapter-blocks", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--consolidate-epochs", type=int, default=3)
    ap.add_argument("--consolidate-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="runs/pilot")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    pc = PilotConfig(rounds=args.rounds, dim=args.dim, num_layers=args.layers,
                     adapter_blocks=args.adapter_blocks,
                     max_new_tokens=args.max_new_tokens,
                     consolidate_epochs=args.consolidate_epochs,
                     consolidate_lr=args.consolidate_lr, seed=args.seed)
    rep = run_pilot(out_dir=args.out, pcfg=pc)
    if not args.quiet:
        a1, a2 = rep["a1"], rep["a2"]
        print(f"=== Pilot A1/A2（verdict={rep['verdict']}）===")
        print(f"模型 {rep['model_params']:,} params + 适配器 {rep['adapter_params']:,} params，"
              f"{rep['num_problems']} 题真实沙箱判分")
        print(f"A1 pass_rate: {a1['baseline_pass_rate']} -> {a1['final_pass_rate']}"
              f"  improved={a1['improved']}  （内置题=记忆证据）")
        if "heldout" in a1:
            h = a1["heldout"]
            print(f"A1 held-out: {h['baseline_pass_rate']} -> {h['final_pass_rate']}"
                  f"  improved={h['improved']}  （{h['num_problems']} 题泛化判据）")
        for s in a1["learning_signal"]:
            print(f"  蒸馏 loss: {s['loss_before']} -> {s['loss_after']}"
                  f"  decreased={s['decreased']}")
        print(f"A2 gate: stages={a2['stages']} committed={a2['committed']} "
              f"rolled_back={a2['rolled_back']} failed={a2['failed_metrics_all']}")
        print(f"诚实边界: {rep['honesty']['reason']}")
        print(f"报告: {rep['artifacts']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
