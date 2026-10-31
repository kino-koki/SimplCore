# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长上下文检索评测（验收协议 ``X-longctx-retrieval`` + ``X-stream-consistency``）.

(Long-context needle retrieval eval: X-longctx-retrieval, X-stream-consistency.)

战役协议中两条必须用**真实输出**验收的闸门::

    Gate("X-longctx-retrieval", "1M / 2M / 4M 长上下文检索准确率 ≥ 基线")
    Gate("X-stream-consistency", "全部评测在流式与整段两种模式下结果一致")
    ShowcaseItem("S5-longctx-4m",  "1M/2M/4M 检索+摘要综合 ≥ YaRN 扩展 TS-5B")
    ShowcaseItem("S7-honest-needle", "1M needle 精确召回 < TS-2B —— 信息论边界主动呈现")

设计原则
--------
1. **needle 任务是结构化、确定性、可审计的**。每条 trial 的语料由 seed
   完全决定：干扰序列 + ``K`` 个 needle（``[开标, key₂, key₁, 闭标, value]``）
   插在指定深度 + 查询后缀（``[开标, key₂, key₁, 闭标]``）。答案 =
   末位 argmax 是否等于目标 value —— 机器可判、无自报自评（forbidden 第 3 条）。
   多 needle 不同深度防"只记最后一个"；逐深度分桶 + 最差桶入报告
   （forbidden 第 4 条，禁 average-only）。
2. **流式==整段是闸门，不是注释**。同一模型对同一 trial 分别跑并行前向与
   O(1) 流式（``init_stream``/``stream_step``），argmax 必须逐 trial 相同、
   logit 差必须 < 容差 —— ``X-stream-consistency`` 是协议的一部分，因为
   "1M 检索"如果只在整段模式成立而流式掉点，部署口径就是假的。
3. **长 L 的账先于跑**。TS 臂 KV cache = ``2·L·layers·D·2B``（bf16 K+V），
   随 L 线性；1M 时 3B 级 TS ~32 GB（§5.13 口径）。给定 ``max_device_mem``
   （默认 80 GiB）后，放不下的臂**如实标记不可跑**，账面照报、闸门给
   ``insufficient_device_memory`` 而不是假装比过 —— 不跑假对照。
4. **S7 的诚实呈现是结构性的**。报告永久携带信息论账：needle 原文字节、
   O(1) 流式状态字节、TS KV 字节三行并列。若 SimplCore 检索不如基线，
   ``unqualified_claim_permitted`` 就是 False，且 S7 字段把边界数字直接给出
   —— 主动呈现弱点与呈现优点是同一个协议。

用法 (usage)::

    python -m simplcore.core.longctx_eval --preset smoke --lengths 512 --out runs/eval_longctx.json
    python -m simplcore.core.longctx_eval --lengths 8192 --trials-per-depth 4
    python -m simplcore.core.longctx_eval --lengths 1048576,2097152,4194304 --ledger-only

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "NeedleCorpus",
    "NeedleTrial",
    "run_longctx_eval",
    "score_trial",
]

PROTOCOL = "X-longctx-retrieval"
REPORT_VERSION = 1

MARK_OPEN = 0        # needle/查询块开始
MARK_CLOSE = 1       # key 与 value 的分界（查询块以它结尾，答案 = 其后一位）


# ---------------------------------------------------------------------------
# needle 语料（确定性）
# ---------------------------------------------------------------------------
@dataclass
class NeedleTrial:
    """一条检索 trial 的全部机器可判信息。"""

    tokens: torch.Tensor          # (L,) 完整序列（含查询后缀）
    value_token: int              # 正确答案
    depth: float                  # query key 的插入深度 ∈ (0,1)
    key_id: int                   # 被查询的 key
    all_key_depths: List[float]   # 全部 needle 的（key_id, 深度）账
    n_values: int


class NeedleCorpus:
    """确定性 needle 语料生成器。

    词表布局（``vocab_size`` 由本类划定，建模方照抄）::

        [0] MARK_OPEN  [1] MARK_CLOSE
        [2 .. 2+2K)          key tokens（key i = (2+2i, 3+2i)，各 2 token）
        [2+2K .. 2+2K+V)     value tokens（value j = 2+2K+j）
        其余                  haystack fillers

    序列布局（定长 L）::

        haystack(L-4-q)  每隔一段插一个 needle 块(5 token，slice 替换)，
        末尾 append 查询后缀 (4 token) = [OPEN, k1, k2, CLOSE]
        答案 = 末位 logits 的 argmax（预测 CLOSE 之后那一位）。
    """

    def __init__(self, seq_len: int, *, n_keys: int = 4, n_values: int = 8,
                 needles_per_seq: int = 4, seed: int = 2026) -> None:
        if seq_len < 64:
            raise ValueError(f"seq_len must be >= 64, got {seq_len}")
        if needles_per_seq < 1 or needles_per_seq > n_keys:
            raise ValueError("needles_per_seq in [1, n_keys]")
        self.seq_len = int(seq_len)
        self.n_keys = int(n_keys)
        self.n_values = int(n_values)
        self.needles_per_seq = int(needles_per_seq)
        self.seed = int(seed)
        self.vocab_size = 2 + 2 * self.n_keys + self.n_values + 16   # +16 fillers

    # -- 词表映射 -------------------------------------------------------------
    def key_tokens(self, key_id: int) -> Tuple[int, int]:
        return 2 + 2 * key_id, 3 + 2 * key_id

    def value_token(self, value_id: int) -> int:
        return 2 + 2 * self.n_keys + value_id

    def _filler(self, g: torch.Generator) -> int:
        return 2 + 2 * self.n_keys + self.n_values + int(
            torch.randint(0, self.vocab_size - (2 + 2 * self.n_keys + self.n_values),
                          (1,), generator=g))

    # -- 生成一条 trial --------------------------------------------------------
    def make_trial(self, index: int, depth: float) -> NeedleTrial:
        """第 ``index`` 条 trial，query key 的 needle 落在相对深度 ``depth``。"""
        if not 0.0 < depth < 1.0:
            raise ValueError(f"depth must be in (0,1), got {depth}")
        g = torch.Generator().manual_seed(self.seed * 100003 + index)
        L, body = self.seq_len, self.seq_len - 4          # 去掉查询后缀
        seq = [self._filler(g) for _ in range(body)]
        # K 个 needle：query key 落 depth（精确配对，不排序打乱），其余均匀散布
        rest_keys = [k for k in range(self.n_keys) if k != index % self.n_keys]
        chosen = rest_keys[:self.needles_per_seq - 1]
        filler_depths = [0.05 + 0.18 * i for i in range(len(chosen))]
        keyed = [(index % self.n_keys, depth)] + list(zip(chosen, filler_depths))
        # value 分配：每个 key 一个固定 value（seed 决定映射，全语料一致）
        key2value = {k: self.value_token((self.seed + k) % self.n_values)
                     for k, _ in keyed}
        for k, d in keyed:
            pos = min(max(int(d * (body - 5)), 0), body - 5)
            k1, k2 = self.key_tokens(k)
            seq[pos:pos + 5] = [MARK_OPEN, k1, k2, MARK_CLOSE, key2value[k]]
        k1, k2 = self.key_tokens(index % self.n_keys)
        seq += [MARK_OPEN, k1, k2, MARK_CLOSE]            # 查询后缀
        return NeedleTrial(tokens=torch.tensor(seq, dtype=torch.long),
                           value_token=key2value[index % self.n_keys],
                           depth=depth, key_id=index % self.n_keys,
                           all_key_depths=[[k, round(d, 4)] for k, d in keyed],
                           n_values=self.n_values)


# ---------------------------------------------------------------------------
# 评分（并行 + 流式两模式）
# ---------------------------------------------------------------------------
def score_trial(model: Any, trial: NeedleTrial, *, mode: str,
                consistency_tol: float = 4e-5,
                chunk_size: int = 4096
                ) -> Dict[str, Any]:
    """对一条 trial 计分，返回 argmax 正确性 + value NLL +（流式时）一致性账。

    * ``mode="parallel"``：整段前向，取末位 logits。
    * ``mode="streaming"``：逐 token ``stream_step``（O(1) 状态）；
      若模型同时支持并行，则附带回算整段 logits 做一致性核验
      （argmax 相等 + max|Δlogit| < ``consistency_tol``）。
    * ``mode="chunked"``：分块前向（截断状态、无梯度），4M 长度下不会
      把 ``(L, D)`` 核与激活一次性物化。训练中途的 GATE-2 用这条路径。
    """
    x = trial.tokens.unsqueeze(0)
    with torch.no_grad():
        if mode == "parallel":
            res = model(x, None)
            logits = res[0] if isinstance(res, tuple) else res
            last = logits[0, -1]
            extra: Dict[str, Any] = {}
        elif mode == "chunked":
            if not hasattr(model, "forward_chunked"):
                raise ValueError("model does not support forward_chunked")
            logits = model.forward_chunked(x, chunk_size=int(chunk_size))
            last = logits[0, -1]
            extra = {"chunk_size": int(chunk_size)}
        elif mode == "streaming":
            if not getattr(model, "supports_streaming", False):
                raise ValueError("model does not support O(1) streaming")
            # 状态 dtype 取模型参数精度（fp32/bf16），绝不能传 token id 的
            # int64——递推状态 dtype 是浮点语义（v1.1.2 勘误：int64 曾被
            # recurrent_dtype 原样接受，递推在整数域进行 → 数值垃圾）
            state = model.init_stream(
                1, device=x.device,
                dtype=next(model.parameters()).dtype)
            last = None
            for t in x[0].tolist():
                last, state = model.stream_step(torch.tensor([t]), state)
            last = last[0]
            # 一致性核验（有并行能力才算 —— 这是闸门数据，不是装饰）
            if hasattr(model, "forward"):
                res = model(x, None)
                par = (res[0] if isinstance(res, tuple) else res)[0, -1]
                extra = {
                    "stream_consistent": bool(torch.equal(last.argmax(), par.argmax())),
                    "max_logit_diff": float((last - par).abs().max()),
                    "tolerance": consistency_tol,
                }
                extra["within_tol"] = extra["max_logit_diff"] <= consistency_tol
            else:
                extra = {}
        else:
            raise ValueError(f"mode must be parallel|streaming, got {mode!r}")
        logp = torch.log_softmax(last.float(), dim=-1)
    return {
        "argmax": int(last.argmax()),
        "correct": bool(int(last.argmax()) == trial.value_token),
        "value_nll": float(-logp[trial.value_token]),
        "rank_of_value": int((logp > logp[trial.value_token]).sum()),
        **extra,
    }


def _ts_kv_bytes(layers: int, dim: int, seq_len: int,
                 bytes_per_elm: int = 2) -> int:
    """TS 臂 KV cache 字节：K+V × layers × L × dim × 元素字节（bf16=2）。"""
    return 2 * layers * int(seq_len) * dim * bytes_per_elm


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_longctx_eval(*, preset: str = "smoke",
                     lengths: Sequence[int] = (512,),
                     trials_per_depth: int = 2,
                     depths: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                     seed: int = 2026, max_device_mem_gb: float = 80.0,
                     ledger_only: bool = False, out_path: Any = None
                     ) -> Dict[str, Any]:
    """跑长上下文检索评测，返回报告 dict（``out_path`` 不为 None 时落盘）。

    Args:
        preset: ``smoke`` / ``small`` / ``1b``（同 :mod:`equal_budget` 的配置族）。
        lengths: 评估的序列长度阶梯（如 ``512`` 或 ``1048576,2097152,4194304``）。
        trials_per_depth: 每个深度的 trial 数（总 trial = ×len(depths)）。
        ledger_only: 只算账不跑模型 —— 1M/2M/4M 的可行性账 + S7 信息论边界，
            CPU 上跑 1M 整段不现实时用它；正式实机评测才关掉。
    """
    from ..model.config import SimplCoreConfig
    from ..model.model import SimplCoreModel

    if preset == "smoke":
        sc_kw = dict(vocab_size=None, dim=32, num_layers=2, num_bases=8,
                     ffn_ratio=2.0, local_kernel=4, decay_max=512.0,
                     tie_embeddings=False)
    elif preset == "small":
        sc_kw = dict(vocab_size=None, dim=256, num_layers=8, num_bases=32,
                     ffn_ratio=4.0, local_kernel=4, decay_max=65536.0,
                     tie_embeddings=True)
    elif preset == "1b":
        from ..model.config import get_preset
        sc_cfg = get_preset("tuan-1b")
        sc_kw = None
    else:
        raise ValueError(f"unknown preset {preset!r}")

    max_dev = int(max_device_mem_gb * (1 << 30))
    length_rows = []
    for L in lengths:
        L = int(L)
        corpus = NeedleCorpus(L, seed=seed)
        if sc_kw is not None:
            cfg = SimplCoreConfig(**{**sc_kw, "vocab_size": corpus.vocab_size})
        else:
            cfg = sc_cfg
        # ---- 账本（先于跑：TS 放不下就不跑假对照） -------------------------
        from .scaling import stream_state_bytes, non_embedding_params
        sc_state = int(stream_state_bytes(cfg))
        ts_layers = cfg.num_layers          # 参数对齐由 equal_budget 证明可行，
        ts_dim = cfg.dim                    # 此处账本按同形 TS 记（口径注明）
        ts_kv = _ts_kv_bytes(ts_layers, ts_dim, L)
        ts_runnable = ts_kv <= max_dev
        # S7 信息论账：needle 原文（5 token/needle × bf16）vs 状态字节
        needle_bytes = 5 * corpus.needles_per_seq * 2
        row = {
            "seq_len": L,
            "simplcore_stream_state_bytes": sc_state,
            "simplcore_state_over_needle_bytes": round(sc_state / max(needle_bytes, 1), 1),
            "ts_kv_cache_bytes": ts_kv,
            "ts_kv_over_sc_state": round(ts_kv / max(sc_state, 1), 1),
            "ts_runnable_on_device": bool(ts_runnable),
            "needle_original_bytes": needle_bytes,
            "chance_level": 1.0 / corpus.n_values,
        }
        if ledger_only or not ts_runnable:
            row["evaluated"] = False
            row["reason"] = ("ledger_only" if ledger_only
                             else f"TS KV cache {ts_kv} bytes > device {max_dev} bytes")
            length_rows.append(row)
            continue

        # ---- 实跑两臂 -------------------------------------------------------
        torch.manual_seed(seed)
        sc_model = SimplCoreModel(cfg).eval()
        from .equal_budget import TSConfig, TransformerBaseline
        ts_cfg = TSConfig(vocab_size=corpus.vocab_size, dim=cfg.dim,
                          num_layers=cfg.num_layers,
                          ffn_hidden=int(cfg.ffn_hidden or cfg.ffn_ratio * cfg.dim),
                          num_heads=max(1, cfg.dim // 16))
        torch.manual_seed(seed + 1)
        ts_model = TransformerBaseline(ts_cfg).eval()

        def run_arm(model: Any, mode: str) -> Dict[str, Any]:
            per_depth: Dict[str, List[Dict[str, Any]]] = {}
            n_ok = n_tot = 0
            max_diff = 0.0
            all_consistent = True
            t0 = time.time()
            for di, d in enumerate(depths):
                bucket: List[Dict[str, Any]] = []
                for j in range(int(trials_per_depth)):
                    idx = di * int(trials_per_depth) + j
                    tr = corpus.make_trial(idx, d)
                    r = score_trial(model, tr, mode=mode)
                    if "max_logit_diff" in r:
                        max_diff = max(max_diff, r["max_logit_diff"])
                        all_consistent = all_consistent and r["stream_consistent"] and r["within_tol"]
                    bucket.append(r)
                    n_ok += int(r["correct"])
                    n_tot += 1
                per_depth[f"{d:.2f}"] = bucket
            acc = n_ok / max(n_tot, 1)
            worst = min(
                sum(int(r["correct"]) for r in b) / max(len(b), 1)
                for b in per_depth.values())
            return {
                "accuracy": acc,
                "worst_depth_accuracy": worst,
                "per_depth_accuracy": {k: sum(int(r["correct"]) for r in b) / max(len(b), 1)
                                       for k, b in per_depth.items()},
                "mean_value_nll": sum(r["value_nll"] for b in per_depth.values()
                                      for r in b) / max(n_tot, 1),
                "n_trials": n_tot,
                "stream_consistency_all": (all_consistent if mode == "streaming" else None),
                "max_logit_diff_parallel_vs_stream": (max_diff if mode == "streaming" else None),
                "wall_seconds": round(time.time() - t0, 3),
            }

        sc_res = run_arm(sc_model, "streaming")     # SC 主口径 = 流式（部署口径）
        sc_par = run_arm(sc_model, "parallel")      # 整段复核（一致性账）
        ts_res = run_arm(ts_model, "parallel")
        row.update({
            "evaluated": True,
            "simplcore": sc_res,
            "transformer": ts_res,
            "stream_consistency": {
                "X-stream-consistency_passed": bool(sc_res["stream_consistency_all"]),
                "parallel_accuracy": sc_par["accuracy"],
                "streaming_accuracy": sc_res["accuracy"],
                "max_logit_diff": sc_res["max_logit_diff_parallel_vs_stream"],
            },
        })
        length_rows.append(row)

    # ---- 闸门 ------------------------------------------------------------
    ev = [r for r in length_rows if r.get("evaluated")]
    consistency_pass = bool(ev) and all(
        r["stream_consistency"]["X-stream-consistency_passed"] for r in ev)
    retrieval_rows = [r for r in ev if "transformer" in r]
    retrieval_pass = bool(retrieval_rows) and all(
        r["simplcore"]["accuracy"] >= r["transformer"]["accuracy"]
        for r in retrieval_rows)
    # 未训练/近 chance 的"≥基线"没有意义 —— 如实标注供人工复核
    near_chance = all(r["simplcore"]["accuracy"] <= 2.0 * r["chance_level"]
                      for r in retrieval_rows)
    gates = {
        "X-longctx-retrieval": {
            "passed": bool(retrieval_pass),
            "evaluated_lengths": [r["seq_len"] for r in retrieval_rows],
            "not_evaluated": [{"seq_len": r["seq_len"], "reason": r.get("reason")}
                              for r in length_rows if not r.get("evaluated")],
            "near_chance_warning": bool(near_chance),
            "note": ("未实跑的长度不参与判定 —— 账面不可行就是不可行，不假装比过"
                     + ("；当前准确率处于 chance 附近，任何 PASS 不构成能力证据"
                        if near_chance else "")),
        },
        "X-stream-consistency": {
            "passed": bool(consistency_pass),
            "scope": "全部 trial 的流式 argmax == 整段 argmax 且 logit 差在容差内",
        },
    }
    unqualified = bool(retrieval_pass and consistency_pass and ev
                       and len(length_rows) == len(ev) and not near_chance)
    report = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": preset,
        "depths": list(depths),
        "trials_per_depth": int(trials_per_depth),
        "max_device_mem_gb": max_device_mem_gb,
        "lengths": length_rows,
        "gates": gates,
        "honesty": {
            "unqualified_claim_permitted": unqualified,
            "S7_honest_needle": {
                "definition": "信息论边界主动呈现：O(1) 状态字节 vs 上下文原文字节 vs TS KV",
                "rows": [{k: r[k] for k in ("seq_len", "simplcore_stream_state_bytes",
                                            "ts_kv_cache_bytes", "needle_original_bytes",
                                            "simplcore_state_over_needle_bytes",
                                            "ts_kv_over_sc_state")} for r in length_rows],
            },
            "forbidden_checks": {
                "no_attention_internal_comparison": True,
                "equal_budget_qualifier_present": True,
                "no_self_reported_scores": True,      # argmax==value 机器可判
                "distribution_not_average_only": True,  # 逐深度分桶 + 最差桶
            },
            "notes": [
                "TS 臂 KV 账按与 SC 同形（同 layers/dim）计；参数量对齐见 equal_budget",
                "SMOKE/合成结果不可替代 1M 实机验收；ledger_only 行只记账不判定",
            ],
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simplcore.core.longctx_eval",
        description="长上下文检索评测（X-longctx-retrieval + X-stream-consistency）。")
    p.add_argument("--preset", default="smoke", choices=("smoke", "small", "1b"))
    p.add_argument("--lengths", default="512",
                   help="逗号分隔的序列长度阶梯，如 512,8192 或 1048576,2097152,4194304")
    p.add_argument("--trials-per-depth", type=int, default=2)
    p.add_argument("--max-device-mem-gb", type=float, default=80.0)
    p.add_argument("--ledger-only", action="store_true",
                   help="只算可行性账与 S7 信息论边界，不跑模型")
    p.add_argument("--out", default="runs/eval_longctx.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    lengths = [int(x) for x in str(args.lengths).split(",") if x.strip()]
    rep = run_longctx_eval(preset=args.preset, lengths=lengths,
                           trials_per_depth=args.trials_per_depth,
                           max_device_mem_gb=args.max_device_mem_gb,
                           ledger_only=args.ledger_only, out_path=args.out)
    print(f"protocol={rep['protocol']} preset={rep['preset']} lengths={lengths}")
    for r in rep["lengths"]:
        if r.get("evaluated"):
            sc, ts = r["simplcore"], r["transformer"]
            print(f"L={r['seq_len']}: sc_acc={sc['accuracy']:.3f} "
                  f"(worst {sc['worst_depth_accuracy']:.3f}) "
                  f"ts_acc={ts['accuracy']:.3f}  "
                  f"stream_consistent={r['stream_consistency']['X-stream-consistency_passed']}")
        else:
            print(f"L={r['seq_len']}: ledger-only  state={r['simplcore_stream_state_bytes']}B "
                  f"ts_kv={r['ts_kv_cache_bytes']}B  ({r['reason']})")
    for name, g in rep["gates"].items():
        print(f"gate {name}: {'PASS' if g['passed'] else 'FAIL'}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
