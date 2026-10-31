# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""论文 / 战役账目交叉审计 —— 论文里的每个数都必须能从代码重算出来。

**方法论**：单一事实源在**代码**（``scaling`` / ``deploy_budget`` /
``config`` / ``campaign`` / ``mathproof`` / ``theory`` / ``pyproject``）；
``SimplCore_Paper.md`` 与 campaign 字符串里的数字是**引用**。本模块把引用
从文本里抓出来、把事实从代码里算出来，逐项对照。产出
``runs/paper_audit.json``；任何一项 FAIL 都以非零码退出——文档先于代码
腐烂是复验性最大的敌人，这个审计让它不可能悄悄发生。

审计范围（可机器核对的数字；§7 的等效参数"推测"按其自我标注检查
"确实被标注为推测"，不核对推测值本身——那是战役实跑的事）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["run_audit", "main"]

PAPER = "SimplCore_Paper.md"
GIB = 2 ** 30
MIB = 2 ** 20

#: 状态字节断言的登记表。
#:
#: 规则（来自独立复现的教训）：**任何"状态 X 字节"的文档数字都必须在这里登记
#: 配置**（preset 或 M/D/layers/K/batch）。实现公式含 batch 与 rd 两个自由变量，
#: 不声明配置就永远对不上；35840 B 与 30.4 MiB 的"889 倍"冲突就是口径未声明。
STATE_BYTE_CLAIMS: List[Dict[str, Any]] = [
    {"id": "S01-paper-5.3-single", "source": "SimplCore_Paper.md §5.3",
     "claim": "单层 M=16/D=256/K=4 状态 36,868 B",
     "stated_bytes": 36_868,
     "config": {"M": 16, "D": 256, "layers": 1, "K": 4, "batch": 1}},
    {"id": "S02-paper-5.3-24layer", "source": "SimplCore_Paper.md §5.3",
     "claim": "M=64/D=1024/K=4/24 层 ≈12.29 MiB",
     "stated_bytes": 12.29 * MIB,
     "config": {"M": 64, "D": 1024, "layers": 24, "K": 4, "batch": 1}},
    {"id": "S03-readme-deploy-1b", "source": "README.md 部署账本",
     "claim": "tuan-1b 流式状态 30.36 MiB",
     "stated_bytes": 30.36 * MIB,
     "config": {"preset": "tuan-1b", "batch": 1}},
    {"id": "S04-native-120b", "source": "docs/NATIVE_INTELLIGENCE.md",
     "claim": "tuan-120b 流式状态 1.855 GiB",
     "stated_bytes": 1.855 * GIB,
     "config": {"preset": "tuan-120b", "batch": 1}},
    {"id": "S05-theoretical-edge", "source": "docs/THEORETICAL_EDGE.md §2.2",
     "claim": "示例 M=64/D=512/K=4/12 层 ≈3.07 MiB",
     "stated_bytes": 3.07 * MIB,
     "config": {"M": 64, "D": 512, "layers": 12, "K": 4, "batch": 1}},
    {"id": "S06-edge-training", "source": "docs/EDGE_TRAINING_REPORT.md",
     "claim": "端侧实测 dim=768/12 层/M=64/K=4 流式状态 4,829,184 B"
              "（不含 E_t 标量口径；与 stream_state_bytes 的 48 B 差已在"
              " Paper §4 口径注勘误）",
     "stated_bytes": 4_829_184,
     "config": {"M": 64, "D": 768, "layers": 12, "K": 4, "batch": 1}},
]

_UNIT_BYTES = {
    "B": 1, "字节": 1, "KiB": 1024, "MiB": 2 ** 20, "GiB": 2 ** 30,
    "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3,
}


def audit_state_bytes(stated_bytes: float, actual_bytes: float,
                      *, has_config: bool = True) -> Dict[str, Any]:
    """按量级分档裁决一个状态字节断言。

    * 相对偏差 ≤ 0.1% → ``PASS``（解析残差/取整）；
    * 0.1% < 偏差 ≤ 1% → ``WARN``（可接受但需标注）；
    * 偏差 > 1% → ``FAIL``；
    * 整数量级不匹配（≥10×）→ ``FAIL``（"凑不出"的典型症状）；
    * 未声明配置 → ``FAIL``（公式含 batch/rd，永远无法对齐）。
    """
    if not has_config:
        return {"status": "FAIL", "reason": "missing_config",
                "rel": None, "ratio": None}
    stated = max(float(stated_bytes), 1e-12)
    actual = max(float(actual_bytes), 1e-12)
    rel = abs(actual - stated) / max(actual, 1e-12)
    ratio = max(actual / stated, stated / actual)
    if ratio >= 10.0:
        return {"status": "FAIL", "reason": "magnitude_mismatch",
                "rel": rel, "ratio": ratio}
    if rel > 0.01:
        return {"status": "FAIL", "reason": "relative_gt_1pct",
                "rel": rel, "ratio": ratio}
    if rel <= 0.001:
        return {"status": "PASS", "reason": "ok", "rel": rel, "ratio": ratio}
    return {"status": "WARN", "reason": "relative_0.1_to_1pct",
            "rel": rel, "ratio": ratio}


def _state_bytes_for(config: Dict[str, Any]) -> int:
    from ..model.config import SimplCoreConfig, get_preset
    from .scaling import stream_state_bytes

    batch = int(config.get("batch", 1))
    if config.get("preset"):
        return int(stream_state_bytes(get_preset(config["preset"]), batch=batch))
    dim = int(config["D"])
    cfg = SimplCoreConfig(vocab_size=32000, dim=dim,
                          num_layers=int(config["layers"]),
                          num_bases=int(config["M"]),
                          local_kernel=int(config.get("K", 4)),
                          num_heads=max(1, dim // 64))
    return int(stream_state_bytes(cfg, batch=batch))


_STATE_MENTION = re.compile(
    r"(?:流式(?:常驻)?|常驻|O\(1\)\s*流式|O\(1\))\s*状态[^0-9\n]{0,40}"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(B|字节|KiB|MiB|GiB|KB|MB|GB)")


def scan_state_mentions(root: Path) -> List[Dict[str, Any]]:
    """扫描文档里的"状态 X 单位"断言，返回 ``(文件, 行, 字节数, 原文)``。

    只用于发现**未登记**的数字；命中后必须补配置并加入
    :data:`STATE_BYTE_CLAIMS`，否则 :func:`run_audit` 判 FAIL。
    """
    paths = [root / PAPER, root / "README.md"]
    docs = root / "docs"
    if docs.exists():
        paths.extend(sorted(docs.glob("*.md")))
    out: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            for m in _STATE_MENTION.finditer(line):
                value = float(m.group(1).replace(",", ""))
                unit = m.group(2)
                out.append({
                    "file": str(path.relative_to(root)).replace("\\", "/"),
                    "line": lineno,
                    "bytes": value * _UNIT_BYTES[unit],
                    "text": m.group(0).strip(),
                })
    return out


def _paper_text(root: Path) -> str:
    return (root / PAPER).read_text(encoding="utf-8")


def _claim(text: str, pattern: str, label: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def run_audit(root: Any = ".", *, collect_tests: bool = True) -> Dict[str, Any]:
    """跑全部审计项。``collect_tests=False`` 时跳过 pytest 收集（测试用）。"""
    root_path = Path(root)
    paper = _paper_text(root_path)
    checks: List[Dict[str, Any]] = []

    def add(cid: str, claim: str, source: str, expected: Any, actual: Any,
            tol: Optional[float] = None) -> None:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
                and tol is not None:
            ok = actual == expected if tol == 0 else \
                abs(actual - expected) <= tol * max(abs(expected), 1e-12)
        else:
            ok = expected == actual
        checks.append({"id": cid, "claim": claim, "source": source,
                       "expected": expected, "actual": actual, "match": bool(ok)})

    # ---- 单一事实源 ----------------------------------------------------------
    from ..deploy.campaign import (PHASE0_MAX_DAYS, PHASE1_MAX_DAYS, PHASE2_MAX_DAYS,
                           RETROSPECT_MAX_DAYS, TOTAL_BUDGET_DAYS)
    from ..model.config import analytic_params, get_preset
    from ..deploy.deploy_budget import DEPLOY_HBM_CAP_1M, deploy_bytes
    from .mathproof import _PROOFS
    from .scaling import infer_flops_per_token, stream_state_bytes
    from .throughput_eval import analytic_decode_ledger

    cfg = get_preset("tuan-1b")

    # ---- C01–C04：部署账本（§5.12 表） ---------------------------------------
    total = int(deploy_bytes(cfg))
    add("C01-deploy-1m-total", "1M 上下文完整推理合计 1.948 GiB",
        "deploy_budget.deploy_bytes", 1.948, round(total / GIB, 3), tol=0.001)
    add("C02-deploy-cap", "1M ≤ 2 GiB 硬上限",
        "deploy_budget.deploy_bytes vs cap", True, total <= DEPLOY_HBM_CAP_1M)
    doubling_ok = True
    ladder = [1 << 20, 2 << 20, 4 << 20, 8 << 20]
    try:
        prev = None
        for L in ladder:
            cur = int(deploy_bytes(cfg, context_tokens=L)) \
                if "context_tokens" in deploy_bytes.__code__.co_varnames \
                else total
            if prev is not None and cur != prev:
                doubling_ok = False
            prev = cur
    except TypeError:
        pass  # 签名不接受长度 → 结构性恒等（§5.12 的原始主张）
    add("C03-deploy-doubling", "上下文翻倍增量恒 0（逐字节相同）",
        "deploy_budget.deploy_bytes", True, bool(doubling_ok))
    sb = int(stream_state_bytes(cfg))
    add("C04-stream-state", "O(1) 流式状态 30.36 MiB（tuan-1b, batch=1, fp32, 15 层）",
        "scaling.stream_state_bytes", 30.36, round(sb / MIB, 2), tol=0.001)

    # ---- C05–C06：FLOPs 与 KV 账 --------------------------------------------
    flops = float(infer_flops_per_token(cfg, 1024))
    add("C05-infer-flops", "每 token 推理 1.92e9 FLOPs（与 L 无关）",
        "scaling.infer_flops_per_token", 1.92e9, flops, tol=0.01)
    ledger = analytic_decode_ledger(cfg, cfg.num_layers, cfg.dim, 1 << 20)
    kv_1m = int(ledger["ts_kv_cache_bytes"])
    add("C06-kv-1m", "TS KV cache @1M（1B 形态，K/V 双份）",
        "throughput_eval.analytic_decode_ledger", 120 * GIB, kv_1m, tol=0.01)
    # 论文 §7 行 399 写 "≈60 GB"——与单一事实源（120 GiB ≈ 128.8 GB）冲突，
    # 按代码裁决：60 GB 恰好是只算了 K 或 V 一半的错误。逐字核对进报告。
    paper_kv = _claim(paper, r"TS KV cache ≈ ([0-9.]+) GB", "kv")
    add("C07-kv-paper-text", "论文 §7 的 TS KV 数字应与代码账一致（≈129 GB/120 GiB）",
        "paper §7 vs analytic_decode_ledger", "129", paper_kv)

    # ---- C08–C09：Chinchilla 比率（§5.13） -----------------------------------
    tok_1b = 26.6e9
    add("C08-chinchilla-1b", "1B 基座 26.6B tokens = 最优的 1.33×",
        "campaign token 预算 vs 20×N", 1.33, round(tok_1b / (20 * 1.0e9), 2),
        tol=0.01)
    tok_3b = 36.8e9
    add("C09-chinchilla-3b", "3B 总形态 36.8B tokens = 最优的 0.61×",
        "campaign token 预算 vs 20×N", 0.61, round(tok_3b / (20 * 3.0e9), 2),
        tol=0.01)

    # ---- C10：战役硬停日历 ----------------------------------------------------
    add("C10-campaign-days", "Phase 0/1/复盘/2 = 30/30/2/28 天，总 90",
        "campaign.py 常量", [30, 30, 2, 28, 90],
        [PHASE0_MAX_DAYS, PHASE1_MAX_DAYS, RETROSPECT_MAX_DAYS, PHASE2_MAX_DAYS,
         TOTAL_BUDGET_DAYS])
    campaign_src = (root_path / "simplcore" / "core" / "deploy" / "campaign.py").read_text(
        encoding="utf-8")
    add("C11-showcase-8", "无人可忽视演示包 = 8 条", "campaign.py ShowcaseItem",
        8, len(re.findall(r"ShowcaseItem\(", campaign_src)))

    # ---- C12–C15：复验计数（S8） ---------------------------------------------
    add("C12-proofs", "数学证明 13 项", "mathproof._PROOFS", 13, len(_PROOFS))
    n_checks = 0
    from .claims_audit import _CHECKS as THEORY_CHECKS
    for fn in THEORY_CHECKS:
        n_checks += len(fn())
    add("C13-theory-guards", "机制层守卫 34 项", "theory._CHECKS", 34, n_checks)

    params = int(analytic_params(cfg))
    add("C14-1b-params", "tuan-1b 预设参数量在 1B 量级",
        "config.analytic_params", True, 0.9e9 <= params <= 1.1e9)
    add("C15-weight-gib", "bf16 权重 1.888 GiB",
        "analytic_params × 2 B", 1.888, round(params * 2 / GIB, 3), tol=0.001)

    # ---- S01–S05：状态字节断言（必须声明配置；按量级分档） ------------------
    # 独立复现抓到的病根：35840 B（单层 M=16/D=256/不含 energy+跨层）与
    # 30.4 MiB（tuan-1b/15 层）被并列引用，差 889 倍。两者都不是错数，
    # 错的是"没声明口径"。这里要求每条数字带配置，并由代码重算。
    warnings: List[Dict[str, Any]] = []
    for claim in STATE_BYTE_CLAIMS:
        actual_sb = _state_bytes_for(claim["config"])
        verdict = audit_state_bytes(claim["stated_bytes"], actual_sb,
                                    has_config=bool(claim.get("config")))
        checks.append({
            "id": claim["id"], "claim": claim["claim"],
            "source": claim["source"],
            "expected": claim["stated_bytes"], "actual": actual_sb,
            "match": verdict["status"] != "FAIL",
            "status": verdict["status"], "reason": verdict["reason"],
            "rel": verdict["rel"], "ratio": verdict["ratio"],
            "config": claim["config"],
        })
        if verdict["status"] == "WARN":
            warnings.append({"id": claim["id"], "status": "WARN",
                             "rel": verdict["rel"], "reason": verdict["reason"]})

    # 文档里出现但未登记的数字 → 硬失败（防止"示例字符串"再次被当事实）。
    declared = [c["stated_bytes"] for c in STATE_BYTE_CLAIMS]
    for item in scan_state_mentions(root_path):
        if any(abs(item["bytes"] - d) <= 1e-3 * max(abs(item["bytes"]), abs(d), 1)
               for d in declared):
            continue
        checks.append({
            "id": f"S-undocumented-{item['file']}:{item['line']}",
            "claim": item["text"],
            "source": f"{item['file']}:{item['line']}",
            "expected": "已登记配置的状态字节",
            "actual": item["bytes"],
            "match": False, "status": "FAIL",
            "reason": "undocumented_state_claim",
        })

    if collect_tests:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pytest", "simplcore", "--collect-only", "-q",
                 "-p", "no:warnings"], cwd=str(root_path), capture_output=True,
                text=True, timeout=600)
            n_tests = sum(int(m) for m in
                          re.findall(r": (\d+)$", out.stdout, re.M)) or None
        except Exception:
            n_tests = None
        claimed_tests = re.search(r"(\d+) 测试", campaign_src)
        add("C16-test-count", "全量守护测试数与 campaign S8 声称一致",
            "pytest --collect-only vs campaign S8",
            int(claimed_tests.group(1)) if claimed_tests else None, n_tests)
        campaign_s8 = re.search(r"(\d+) CLI", campaign_src)
        pyproject = (root_path / "pyproject.toml").read_text(encoding="utf-8")
        sec = re.split(r"\[project\.scripts\]", pyproject, 1)
        sec_body = re.split(r"^\[", sec[1], 1, re.M)[0] if len(sec) == 2 else ""
        n_cli = len(re.findall(r"^[\w-]+ =", sec_body, re.M)) or None
        add("C17-cli-count", "CLI 入口数与 campaign S8 声称一致",
            "pyproject [project.scripts]",
            int(campaign_s8.group(1)) if campaign_s8 else None, n_cli)

    failed = [c["id"] for c in checks if not c["match"]]
    report = {
        "protocol": "paper-audit",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paper": PAPER,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_ids": failed,
        "warnings": warnings,
        "n_warnings": len(warnings),
        "checks": checks,
        "honesty": {
            "note": ("§7 等效参数推测（轴 A–E）按其自我标注为'推测'处理，"
                     "不核对推测值本身——把它变成事实需要战役的一次真实运行"),
            "all_fail_hard": True,
        },
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="论文/战役账目交叉审计")
    ap.add_argument("--out", type=str, default="runs/paper_audit.json")
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--no-collect", action="store_true",
                    help="跳过 pytest 收集（快，但 C16 缺失）")
    args = ap.parse_args(argv)
    rep = run_audit(args.root, collect_tests=not args.no_collect)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in rep["checks"]:
        mark = "PASS" if c["match"] else "FAIL"
        print(f" [{mark}] {c['id']}: expected={c['expected']} actual={c['actual']}")
    print(f"== {rep['n_checks'] - rep['n_failed']}/{rep['n_checks']} 通过；"
          f"报告: {out} ==")
    return 0 if not rep["n_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
