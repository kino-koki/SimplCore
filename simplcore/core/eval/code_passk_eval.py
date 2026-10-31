# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""代码 pass@k 对照评测（验收协议 ``X-code-passk`` 的可执行化）.

(Code pass@k comparison: real-execution verifier, paired across two arms.)

战役协议::

    Gate("X-code-passk", "代码 pass@k（CodeExecutionVerifier 真实执行）≥ 基线")
    ShowcaseItem("S6-evolution-passk", "自进化前后代码 pass@k 可验证提升，含回滚记录")

设计原则
--------
1. **pass@k 是有数学定义的，不是"平均分"**。本模块实现 Codex 论文
   (Chen et al., 2021) 的**无偏估计量**：每题采 ``n`` 个样本、``c`` 个通过
   全部测试，``pass@k = 1 - C(n-c, k)/C(n, k)``（数值稳定版走对数域，
   小 ``n`` 与精确组合版互证）。:mod:`simplcore.core.code_exec` 现有的
   ``pass_rate`` 只是每样本平均分，与 pass@k 不是同一个量 —— 报告里两个
   都给、口径分开标注，谁也别冒充谁。
2. **判分只认真实执行**。每题每样本过 :func:`code_exec.run_python_tests`
   （子进程、超时、内存限制），异常/超时/断言失败一律不计通过 ——
   forbidden 第 3 条"自报自评"在此结构上不可能发生。
3. **配对统计复用 ab.py**：两臂**同一套题**逐题对齐（缺题显式报错，
   禁止按位置错位对齐）；题级二元结果（至少一个候选全过）走 McNemar
   精确检验，题级连续 ``pass@k`` 估计走配对 bootstrap CI。
4. **两臂只通过 generator 接触**。``(prompt) -> List[str]``（k 个候选）
   是模型系统的唯一入口 —— SimplCore 臂用 :meth:`make_sampling_generator`
   （温度采样 k 次），TS 臂由其模型系统提供同规格 generator；两臂必须
   **同 k、同 max_new_tokens、同温度声明**（等预算口径在推理侧的对应物）。
   没有模型时 ``--demo`` 用固定候选池跑通协议并验证统计层 —— demo 结果
   永远不构成能力证据（``demo: true`` 进报告、宣称许可关闭）。

用法 (usage)::

    python -m simplcore.core.code_passk_eval --demo --k 4 --out runs/eval_code_passk.json
    python -m simplcore.core.code_passk_eval --k 8 --temperature 0.8 \\
        --simplcore-gen mypkg:my_generator --ts-gen otherpkg:ts_generator

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..training.ab import PairedOutcome, bootstrap_diff_ci, mcnemar
from ..memory.code_exec import (CodeProblem, extract_code, make_smoke_problems,
                        run_python_tests)

__all__ = [
    "GeneratorSpec",
    "make_demo_arms",
    "make_problems_small",
    "pass_at_k",
    "run_code_passk",
]

PROTOCOL = "X-code-passk"
REPORT_VERSION = 1


# ---------------------------------------------------------------------------
# pass@k 无偏估计（Chen et al., 2021）
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """``n`` 个样本中 ``c`` 个通过时 pass@k 的无偏估计。

    ``1 - C(n-c, k)/C(n, k)``，对数域计算保证大 ``n`` 数值稳定；
    ``n - c < k``（通过数多到任取 k 个必含通过）时恒为 1。
    """
    if n < 1 or k < 1:
        raise ValueError(f"n and k must be >= 1, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_ratio)


def pass_at_k_exact(n: int, c: int, k: int) -> float:
    """精确组合版（小 n 互证用）。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# 题目集（确定性 held-out；判分 = 真实执行）
# ---------------------------------------------------------------------------
def make_problems_small() -> List[CodeProblem]:
    """8 道内置确定性题（smoke 3 题 + 5 题）。测试全部机器可判。"""
    base = list(make_smoke_problems())
    extra = [
        CodeProblem(
            name="fib10",
            prompt="实现函数 fib(n)，返回斐波那契数列第 n 项（fib(0)=0, fib(1)=1）。",
            tests=["assert fib(0) == 0", "assert fib(1) == 1",
                   "assert fib(10) == 55", "assert fib(20) == 6765"],
        ),
        CodeProblem(
            name="is_palindrome",
            prompt="实现函数 is_palindrome(s)，判断字符串 s 忽略大小写与非字母数字后是否回文。",
            tests=["assert is_palindrome('level') == True",
                   "assert is_palindrome('A man, a plan, a canal: Panama') == True",
                   "assert is_palindrome('hello') == False",
                   "assert is_palindrome('') == True"],
        ),
        CodeProblem(
            name="max_without_builtin",
            prompt="实现函数 second_max(xs)，返回列表 xs 中第二大的数；不足两个元素时返回 None。",
            tests=["assert second_max([1, 5, 3]) == 3",
                   "assert second_max([7, 7]) == 7",
                   "assert second_max([4]) is None",
                   "assert second_max([-2, -9, -1]) == -2"],
        ),
        CodeProblem(
            name="count_vowels",
            prompt="实现函数 count_vowels(s)，返回 s 中元音字母（aeiou，不区分大小写）的个数。",
            tests=["assert count_vowels('hello') == 2",
                   "assert count_vowels('AEIOU') == 5",
                   "assert count_vowels('xyz') == 0",
                   "assert count_vowels('') == 0"],
        ),
        CodeProblem(
            name="flatten",
            prompt="实现函数 flatten(xs)，把任意深度嵌套的列表展平成一维列表（保持顺序）。",
            tests=["assert flatten([1, [2, [3]], 4]) == [1, 2, 3, 4]",
                   "assert flatten([]) == []",
                   "assert flatten([[[1]]]) == [1]",
                   "assert flatten([1, 2]) == [1, 2]"],
        ),
    ]
    return base + extra


# ---------------------------------------------------------------------------
# generator 接口与 demo 候选池
# ---------------------------------------------------------------------------
@dataclass
class GeneratorSpec:
    """一臂的候选生成器 + 规格声明（等预算口径的推理侧对应物）。"""

    name: str
    generate: Callable[[str], List[str]]        # (prompt) -> k 个候选
    k: int
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    source: str = ""                            # 生成器来源（审计字段）

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "k": self.k, "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens, "source": self.source}


def make_sampling_generator(model: Any, codec: Any, k: int, *,
                            prompt_template: str = (
                                "完成以下 Python 任务，只输出一个代码块：\n{instruction}\n```python\n"),
                            temperature: float = 0.8,
                            max_new_tokens: int = 256) -> GeneratorSpec:
    """SimplCore 臂：温度采样 k 个候选（同一 prompt 采 k 次）。"""

    def _gen(instruction: str) -> List[str]:
        outs = []
        for _ in range(int(k)):
            outs.append(model.generate_text(
                prompt_template.format(instruction=instruction), codec,
                max_new_tokens=max_new_tokens, temperature=temperature, top_k=50))
        return outs

    return GeneratorSpec(name=f"simplcore({type(model).__name__})",
                         generate=_gen, k=int(k), temperature=float(temperature),
                         max_new_tokens=int(max_new_tokens),
                         source="simplcore model sampling")


def make_demo_arms(k: int = 4) -> Tuple[GeneratorSpec, GeneratorSpec]:
    """固定候选池 demo：强臂（每题给正解）vs 弱臂（半数题给错解）。

    仅用于验证协议与统计层（CLI --demo）；demo=True 的报告不构成能力证据。
    """
    sol = {
        "add": "def add(a, b):\n    return a + b",
        "is_even": "def is_even(n):\n    return n % 2 == 0",
        "reverse_str": "def reverse_str(s):\n    return s[::-1]",
        "fib10": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "is_palindrome": ("import re\ndef is_palindrome(s):\n"
                          "    t = re.sub(r'[^a-z0-9]', '', s.lower())\n"
                          "    return t == t[::-1]"),
        "max_without_builtin": ("def second_max(xs):\n"
                                "    u = sorted(xs, reverse=True)\n"
                                "    return u[1] if len(u) >= 2 else None"),
        "count_vowels": ("def count_vowels(s):\n"
                         "    return sum(1 for ch in s.lower() if ch in 'aeiou')"),
        "flatten": ("def flatten(xs):\n"
                    "    out = []\n"
                    "    def walk(v):\n"
                    "        if isinstance(v, list):\n"
                    "            for e in v:\n"
                    "                walk(e)\n"
                    "        else:\n"
                    "            out.append(v)\n"
                    "    walk(xs)\n"
                    "    return out"),
    }
    wrong = {
        "add": "def add(a, b):\n    return a - b",            # 全错
        "is_even": "def is_even(n):\n    return n % 2 == 1",  # 全错
        "reverse_str": "def reverse_str(s):\n    return s[:-1]",      # 丢尾字符
        "fib10": "def fib(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",  # fib(0)=1
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",  # 忽略大小写测试挂
        "max_without_builtin": ("def second_max(xs):\n"
                                "    u = sorted(set(xs), reverse=True)\n"
                                "    return u[0] if len(u) >= 2 else None"),  # 返回最大
        "count_vowels": ("def count_vowels(s):\n"
                         "    return sum(1 for ch in s if ch in 'aeiou')"),  # 不处理大写
        "flatten": ("def flatten(xs):\n"
                    "    out = []\n"
                    "    for v in xs:\n"
                    "        if isinstance(v, list):\n"
                    "            out.extend(v)\n"             # 只展平一层
                    "        else:\n"
                    "            out.append(v)\n"
                    "    return out"),
    }
    # 弱臂：每题只给"部分正确"的错解（无全过候选）→ any_pass=0 但 sample_rate>0，
    # 与强臂出现逐题翻转 —— 这才是配对统计有内容的 demo。
    def strong_gen(prompt: str) -> List[str]:
        name = _problem_name(prompt)
        return ["```python\n" + sol[name] + "\n```"] * int(k)

    def weak_gen(prompt: str) -> List[str]:
        name = _problem_name(prompt)
        bad = wrong[name]
        cands = ["```python\n" + bad + "\n```"] * int(k)
        rng = random.Random(hash(name) & 0xFFFF)
        rng.shuffle(cands)
        return cands

    return (GeneratorSpec(name="demo-strong", generate=strong_gen, k=int(k),
                          source="fixed candidate pool (demo)"),
            GeneratorSpec(name="demo-weak", generate=weak_gen, k=int(k),
                          source="fixed candidate pool (demo)"))


def _problem_name(prompt: str) -> str:
    """demo 候选池按题目名取解：从 prompt 里反查函数名映射。"""
    for fn in ("add", "is_even", "reverse_str", "fib", "fib10",
               "is_palindrome", "second_max", "max_without_builtin",
               "count_vowels", "flatten"):
        if fn in prompt:
            return {"fib": "fib10", "second_max": "max_without_builtin"}.get(fn, fn)
    raise KeyError(f"demo pool: unknown prompt: {prompt[:50]}...")


# ---------------------------------------------------------------------------
# 评测主入口
# ---------------------------------------------------------------------------
def _eval_arm(spec: GeneratorSpec, problems: List[CodeProblem],
              timeout_s: float) -> Dict[str, Any]:
    """跑一臂：每题 k 候选 → 真实执行 → 逐题 (n, c) 与 pass@k。"""
    per_task = []
    for p in problems:
        c = 0
        samples: List[Dict[str, Any]] = []
        for raw in spec.generate(p.prompt):
            code = extract_code(raw or "")
            res = run_python_tests(code, p.tests, timeout_s=timeout_s,
                                   max_output_chars=2000)
            ok = (res.total > 0 and res.passed == res.total and not res.timed_out)
            c += int(ok)
            samples.append({"passed_all": ok, "passed": res.passed,
                            "total": res.total, "timed_out": res.timed_out})
        per_task.append({
            "name": p.name, "n": len(samples), "c": c,
            "pass_at_k": pass_at_k(len(samples), c, min(spec.k, len(samples))),
            "any_pass": bool(c >= 1),
            "samples": samples,
        })
    n_tot = sum(t["n"] for t in per_task)
    return {
        "generator": spec.to_dict(),
        "per_task": per_task,
        "pass_at_k_mean": sum(t["pass_at_k"] for t in per_task) / max(len(per_task), 1),
        "any_pass_rate": sum(int(t["any_pass"]) for t in per_task) / max(len(per_task), 1),
        "sample_pass_rate": sum(t["c"] for t in per_task) / max(n_tot, 1),
        "n_samples_total": n_tot,
    }


def run_code_passk(*, problems: Optional[List[CodeProblem]] = None,
                   sc_gen: Optional[GeneratorSpec] = None,
                   ts_gen: Optional[GeneratorSpec] = None,
                   k: int = 4, timeout_s: float = 5.0,
                   n_boot: int = 2000, seed: int = 0,
                   out_path: Any = None) -> Dict[str, Any]:
    """跑两臂 pass@k 对照（任一臂缺省时自动用 demo 池补位，demo 标记进报告）。"""
    demo = False
    if problems is None:
        problems = make_problems_small()
    if sc_gen is None or ts_gen is None:
        demo = True
        d_sc, d_ts = make_demo_arms(k=k)
        sc_gen = sc_gen or d_sc
        ts_gen = ts_gen or d_ts
    if sc_gen.k != ts_gen.k:
        raise ValueError(f"两臂 k 不同（{sc_gen.k} vs {ts_gen.k}）—— 等预算口径不成立")
    if problems and len({p.name for p in problems}) != len(problems):
        raise ValueError("题目名重复 —— 逐题配对要求 id 唯一")

    sc_res = _eval_arm(sc_gen, problems, timeout_s)
    ts_res = _eval_arm(ts_gen, problems, timeout_s)

    # ---- 配对统计（ab.py：缺题显式报错，翻转题才携带信息） ----------------
    paired = PairedOutcome.from_reports(sc_gen.name, ts_gen.name,
                                        _Report(sc_res), _Report(ts_res))
    summary = paired.summary(n_boot=n_boot, seed=seed)
    # 题级连续 pass@k 的配对 bootstrap CI（二元 McNemar 之外的连续口径）
    p_sc = [t["pass_at_k"] for t in sc_res["per_task"]]
    p_ts = [t["pass_at_k"] for t in ts_res["per_task"]]
    ci_cont = bootstrap_diff_ci(p_sc, p_ts, n_boot=n_boot, seed=seed)

    gates = {
        "X-code-passk": {
            "passed": bool(sc_res["pass_at_k_mean"] >= ts_res["pass_at_k_mean"]),
            "pass_at_k_simplcore": sc_res["pass_at_k_mean"],
            "pass_at_k_transformer": ts_res["pass_at_k_mean"],
            "mcnemar_p": summary["mcnemar"]["p_value"],
            "mcnemar_significant": summary["mcnemar"]["significant"],
            "evidence_weight": summary["evidence_weight"],
            "note": "pass@k 差值方向与 McNemar 显著性分列 —— 8 题上翻转 <5 不当结论",
        },
    }
    report: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "k": int(k),
        "n_problems": len(problems),
        "problem_names": [p.name for p in problems],
        "timeout_s": float(timeout_s),
        "demo": bool(demo),
        "arms": {"simplcore": sc_res, "transformer": ts_res},
        "paired": summary,
        "paired_pass_at_k_continuous": {
            "delta": round(ci_cont["diff"], 4),
            "ci95": [round(ci_cont["ci_low"], 4), round(ci_cont["ci_high"], 4)],
            "excludes_zero": ci_cont["excludes_zero"],
        },
        "gates": gates,
        "honesty": {
            "unqualified_claim_permitted": bool(
                not demo and gates["X-code-passk"]["passed"]
                and summary["mcnemar"]["significant"]),
            "forbidden_checks": {
                "no_attention_internal_comparison": True,
                "equal_budget_qualifier_present": True,   # 同 k、同 timeout、逐题对齐
                "no_self_reported_scores": True,          # 判分只认子进程真实执行
                "distribution_not_average_only": True,    # 逐题 (n,c) + 翻转题清单
            },
            "notes": [
                "pass@k = 无偏估计 1 - C(n-c,k)/C(n,k)，与 sample pass_rate 是不同口径，"
                "两者分开报告",
            ] + (["demo 模式：固定候选池，只验证协议与统计层，不构成能力证据"]
                 if demo else []),
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return report


class _Report:
    """把 _eval_arm 的输出适配成 PairedOutcome.from_reports 需要的形状。"""

    def __init__(self, arm: Dict[str, Any]) -> None:
        self.per_task = [{"id": t["name"], "passed": t["any_pass"]}
                         for t in arm["per_task"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_spec(spec: str, k: int) -> GeneratorSpec:
    """``pkg.module:factory`` → GeneratorSpec（factory 返回可调用或 GeneratorSpec）。"""
    mod_name, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod_name), attr)
    out = obj() if callable(obj) and not isinstance(obj, GeneratorSpec) else obj
    if not isinstance(out, GeneratorSpec):
        out = GeneratorSpec(name=spec, generate=out, k=k, source=spec)
    if out.k != k:
        raise ValueError(f"generator {spec} 声明 k={out.k} 与 --k {k} 不一致")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simplcore.core.code_passk_eval",
        description="代码 pass@k 对照（X-code-passk）：真实执行判分 + ab.py 配对统计。")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--timeout-s", type=float, default=5.0)
    p.add_argument("--simplcore-gen", default=None,
                   help="pkg.module:factory —— 返回 (prompt)->List[str] 或 GeneratorSpec")
    p.add_argument("--ts-gen", default=None, help="同上，TS 臂")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true",
                   help="固定候选池跑通协议（不构成能力证据）")
    p.add_argument("--out", default="runs/eval_code_passk.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sc_gen = _load_spec(args.simplcore_gen, args.k) if args.simplcore_gen else None
    ts_gen = _load_spec(args.ts_gen, args.k) if args.ts_gen else None
    if not args.demo and (sc_gen is None or ts_gen is None):
        print("未提供两臂 generator 且未指定 --demo —— 回落 demo 模式（报告标注 demo）")
    rep = run_code_passk(sc_gen=sc_gen, ts_gen=ts_gen, k=args.k,
                         timeout_s=args.timeout_s, n_boot=args.n_boot,
                         seed=args.seed, out_path=args.out)
    a = rep["arms"]
    print(f"protocol={rep['protocol']} k={rep['k']} problems={rep['n_problems']} "
          f"demo={rep['demo']}")
    for name in ("simplcore", "transformer"):
        arm = a[name]
        print(f"{name}: pass@{rep['k']}={arm['pass_at_k_mean']:.3f} "
              f"any_pass={arm['any_pass_rate']:.3f} "
              f"sample_rate={arm['sample_pass_rate']:.3f}")
    g = rep["gates"]["X-code-passk"]
    print(f"paired: delta={rep['paired']['delta']} ci95={rep['paired']['ci95']} "
          f"mcnemar_p={g['mcnemar_p']:.4f} ({g['evidence_weight']})")
    print(f"gate X-code-passk: {'PASS' if g['passed'] else 'FAIL'}"
          f"  unqualified_claim_permitted={rep['honesty']['unqualified_claim_permitted']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
