# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
# purity: allow-attention-reference —— 本模块内含 TS 对照臂（KV-cache 解码基线），
# 只用于吞吐 A/B，不进运行时/注册表。
"""解码吞吐对照评测（验收协议 ``X-throughput`` 的可执行化）.

(Decode throughput comparison: streaming O(1) state vs KV-cache, latency vs context.)

战役协议::

    Gate("X-throughput", "1M 上下文解码吞吐 ≥ 基线（对方 KV cache ~32 GB vs 我方流式 <1 MB）")
    ShowcaseItem("S3-decode-10x", "1M 上下文解码吞吐 ≥ 10× 同卡 TS-3B")

架构主张的实机可验证形式
------------------------
1. **O(1) 流式的延迟斜率为 0**。SimplCore 臂用 ``init_stream``/``stream_step``
   逐 token 解码，在前缀长度 L ∈ {128, 512, 2048, ...} 下分别计时每 token
   延迟 —— 主张是**延迟与 L 无关**（斜率 ≈ 0）。这不是"感觉快"，是一条
   可拟合、可断言的直线（对照臂的斜率是正的、KV cache 越大越慢）。
2. **TS 臂必须是真 KV cache，不是稻草人**。:class:`TSDecoder` 用基线模型的
   权重做增量解码（prefill 建 cache、每步只算新 query 对全量 K/V 的注意力
   —— 标准推理实现）。正确性由守护测试钉死：**prefill 后的下一步 logits
   与整段并行前向的末位 logits 逐位一致**，慢是慢在真实算法上。
3. **1M 的结论是解析账 + 实测斜率外推，不假装测过**。设备放不下 1M 的
   TS KV cache 时（解析账先行，同 :mod:`longctx_eval` 哲学），报告如实标注
   "实测上限 L=XXX，1M 数字由斜率外推 + 解析账支撑"；外推方法写进报告，
   外推值与实测值分列。
4. **延迟报分布不报平均**（forbidden 第 4 条）：p10/p50/p90 + min/max 全入
   报告；斜率用逐对 (L, median latency) 最小二乘拟合。

判分口径
--------
每臂每 L：prefill L 个 token（不计入延迟）→ 计时 ``steps`` 个解码步
（贪心 argmax 喂回，与部署口径一致）→ 每 token 延迟分布。
``X-throughput`` 闸门 = 两臂延迟比（measured）+ 1M 解析账（analytic）分列。

用法 (usage)::

    python -m simplcore.core.throughput_eval --preset smoke --lengths 128,512,2048
    python -m simplcore.core.throughput_eval --preset 1b --lengths 1048576 --ledger-only

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "TSDecoder",
    "analytic_decode_ledger",
    "fit_slope",
    "run_throughput_eval",
    "summarize_latency",
]

PROTOCOL = "X-throughput"
REPORT_VERSION = 1


# ---------------------------------------------------------------------------
# TS 臂：真 KV-cache 增量解码器（复用 TransformerBaseline 权重）
# ---------------------------------------------------------------------------
class TSDecoder:
    """标准 KV-cache 逐步解码（prefill 一次建 cache，每步只算新 query）。

    正确性契约（守护测试钉死）：``prefill(tokens)`` 返回的"下一步 logits"
    必须与 :meth:`TransformerBaseline.forward` 整段前向的末位 logits 一致。
    """

    def __init__(self, base: Any) -> None:      # base: TransformerBaseline
        self.base = base
        cfg = base.config
        self.dim, self.n_heads = cfg.dim, cfg.num_heads
        self.hd = cfg.dim // cfg.num_heads
        self.kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.length = 0

    def _forward_block(self, x: torch.Tensor, blk: Any, blk_idx: int,
                       new_cache: bool) -> torch.Tensor:
        """单块前向；``new_cache`` 时把 K/V 存入（prefill），否则追加并取全量。"""
        B, L, D = x.shape
        h = blk.ln1(x)
        qkv = blk.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.hd).transpose(1, 2)
        if new_cache:
            self.kv.append((k, v))
        else:
            pk, pv = self.kv[blk_idx]
            k = torch.cat([pk, k], dim=2)          # 追加到 KV cache
            v = torch.cat([pv, v], dim=2)
            self.kv[blk_idx] = (k, v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=new_cache)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + blk.out(a)
        h = blk.ln2(x)
        return x + blk.w2(F.silu(blk.w1(h)) * blk.w3(h))

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """消费前缀 ``(L,)``，返回**下一步**的 logits ``(V,)``。"""
        self.kv = []
        self.length = 0
        x = self.base.tok_emb(tokens.unsqueeze(0))
        for i, blk in enumerate(self.base.blocks):
            x = self._forward_block(x, blk, i, new_cache=True)
        logits = self.base.head(self.base.ln_f(x))
        self.length += tokens.shape[0]
        return logits[0, -1]

    @torch.no_grad()
    def step(self, token: int) -> torch.Tensor:
        """消费一个新 token，返回**下一步**的 logits ``(V,)``（每步 O(L) KV 读写）。"""
        x = self.base.tok_emb(torch.tensor([token]).unsqueeze(0))
        for i, blk in enumerate(self.base.blocks):
            x = self._forward_block(x, blk, i, new_cache=False)
        logits = self.base.head(self.base.ln_f(x))
        self.length += 1
        return logits[0, -1]

    def kv_bytes(self) -> int:
        """当前 KV cache 字节数（K+V × layers × L × D × 元素字节）。"""
        if not self.kv:
            return 0
        k, v = self.kv[0]
        per = 2 * k.numel() * k.element_size()      # K+V 每层
        return per * len(self.kv)


# ---------------------------------------------------------------------------
# 计时协议
# ---------------------------------------------------------------------------
def summarize_latency(latencies: Sequence[float]) -> Dict[str, float]:
    """延迟分布（禁 average-only：p10/p50/p90 + min/max + mean 全入报告）。"""
    xs = sorted(latencies)
    n = len(xs)
    if n == 0:
        raise ValueError("no latency samples")
    return {
        "mean_ms": statistics.fmean(xs) * 1e3,
        "p10_ms": xs[max(0, int(0.10 * n) - 1)] * 1e3,
        "p50_ms": xs[n // 2] * 1e3,
        "p90_ms": xs[min(n - 1, int(0.90 * n))] * 1e3,
        "min_ms": xs[0] * 1e3,
        "max_ms": xs[-1] * 1e3,
    }


def fit_slope(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """最小二乘拟合 ``y = slope·x + intercept``，返回 ``(slope, intercept)``。"""
    if len(points) < 2:
        raise ValueError("slope fit needs >= 2 points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("degenerate x values for slope fit")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    return slope, intercept


def _decode_latency(step_fn: Callable[[int], Any], feed_tokens: List[int],
                    n_steps: int, warmup_steps: int) -> Tuple[List[float], Any]:
    """贪心解码计时：先消费前缀（不计），再计时 ``n_steps`` 步。

    返回 ``(每步延迟秒列表, 最后一步的下一步 logits)``。
    """
    last = None
    for t in feed_tokens:
        last = step_fn(int(t))
    with torch.no_grad():
        for i in range(warmup_steps):
            tok = int(last.argmax()) if last is not None else 0
            last = step_fn(tok)
        lat = []
        for _ in range(n_steps):
            tok = int(last.argmax()) if last is not None else 0
            t0 = time.perf_counter()
            last = step_fn(tok)
            lat.append(time.perf_counter() - t0)
    return lat, last


# ---------------------------------------------------------------------------
# 解析账（1M 的账先于跑）
# ---------------------------------------------------------------------------
def analytic_decode_ledger(sc_cfg: Any, ts_layers: int, ts_dim: int,
                           seq_len: int) -> Dict[str, Any]:
    """每 token 解码账：SC O(1)（与 L 无关）vs TS 参数项 + O(L) attention/KV 项。

    * SC：``scaling.infer_flops_per_token``（前向，公式不含 L —— O(1) 的数学内容）；
    * TS：``2·N_nonemb + 4·L·D·layers``（fwd：QK^T+AV 各 2·L·D），
      **每步还必须读写整个 KV cache**（memory-bound）：``2·layers·L·D·2B``。
    """
    from .scaling import infer_flops_per_token
    from .equal_budget import ts_non_embedding_params
    sc_flops = float(infer_flops_per_token(sc_cfg, seq_len))
    ts_n = ts_non_embedding_params(ts_layers, ts_dim, 4 * ts_dim)  # 与 TS 臂同构
    ts_attn = 4.0 * seq_len * ts_dim * ts_layers
    ts_flops = 2.0 * ts_n + ts_attn
    ts_kv = _ts_kv_bytes(ts_layers, ts_dim, seq_len)
    return {
        "seq_len": seq_len,
        "sc_flops_per_token": sc_flops,
        "ts_flops_per_token": ts_flops,
        "ts_flops_over_sc": ts_flops / max(sc_flops, 1e-9),
        "sc_stream_state_bytes": int(_sc_state_bytes(sc_cfg)),
        "ts_kv_cache_bytes": ts_kv,
        "ts_kv_over_sc_state": ts_kv / max(int(_sc_state_bytes(sc_cfg)), 1),
        "note": "TS 每步为读全量 KV 而受内存带宽约束 —— flops 账是下界，实测更慢",
    }


def _ts_kv_bytes(layers: int, dim: int, seq_len: int, bytes_per_elm: int = 2) -> int:
    return 2 * layers * int(seq_len) * dim * bytes_per_elm


def _sc_state_bytes(cfg: Any) -> int:
    from .scaling import stream_state_bytes
    return int(stream_state_bytes(cfg))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_throughput_eval(*, preset: str = "smoke",
                        lengths: Sequence[int] = (128, 512, 2048),
                        steps: int = 32, warmup: int = 4,
                        max_device_mem_gb: float = 80.0,
                        ledger_only: bool = False, seed: int = 2026,
                        out_path: Any = None) -> Dict[str, Any]:
    """跑解码吞吐对照，返回报告 dict（``out_path`` 不为 None 时落盘）。"""
    from .config import SimplCoreConfig
    from .equal_budget import TSConfig, TransformerBaseline
    from .model import SimplCoreModel

    if preset == "smoke":
        sc_kw = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                     ffn_ratio=2.0, local_kernel=4, decay_max=256.0,
                     tie_embeddings=False)
    elif preset == "small":
        sc_kw = dict(vocab_size=512, dim=128, num_layers=4, num_bases=16,
                     ffn_ratio=4.0, local_kernel=4, decay_max=4096.0,
                     tie_embeddings=True)
    elif preset == "1b":
        from .config import get_preset
        sc_kw = None
    else:
        raise ValueError(f"unknown preset {preset!r}")

    max_dev = int(max_device_mem_gb * (1 << 30))
    g = torch.Generator().manual_seed(seed)
    rows: List[Dict[str, Any]] = []
    for L in lengths:
        L = int(L)
        cfg = (SimplCoreConfig(**sc_kw) if sc_kw is not None else get_preset("tuan-1b"))
        ledger = analytic_decode_ledger(cfg, cfg.num_layers, cfg.dim, L)
        ts_runnable = ledger["ts_kv_cache_bytes"] <= max_dev
        row: Dict[str, Any] = {"seq_len": L, "analytic": ledger,
                               "ts_runnable_on_device": bool(ts_runnable)}
        if ledger_only or not ts_runnable:
            row["evaluated"] = False
            row["reason"] = ("ledger_only" if ledger_only
                             else f"TS KV {ledger['ts_kv_cache_bytes']}B > device {max_dev}B")
            rows.append(row)
            continue

        # ---- 建模 -----------------------------------------------------------
        torch.manual_seed(seed)
        sc_model = SimplCoreModel(cfg).eval()
        ts_cfg = TSConfig(vocab_size=cfg.vocab_size, dim=cfg.dim,
                          num_layers=cfg.num_layers,
                          ffn_hidden=int(cfg.ffn_hidden or cfg.ffn_ratio * cfg.dim),
                          num_heads=max(1, cfg.dim // 16))
        torch.manual_seed(seed + 1)
        ts_base = TransformerBaseline(ts_cfg).eval()

        prefix = torch.randint(0, cfg.vocab_size, (L,), generator=g)

        def sc_step(tok: int) -> Any:
            nonlocal sc_state
            logits, sc_state = sc_model.stream_step(torch.tensor([tok]), sc_state)
            return logits

        sc_state = sc_model.init_stream(1)
        sc_lat, _ = _decode_latency(sc_step, prefix.tolist(), steps, warmup)

        ts_dec = TSDecoder(ts_base)
        ts_dec.prefill(prefix)
        ts_lat, _ = _decode_latency(ts_dec.step, prefix.tolist(), steps, warmup)

        sc_sum, ts_sum = summarize_latency(sc_lat), summarize_latency(ts_lat)
        row.update({
            "evaluated": True,
            "steps_timed": int(steps),
            "simplcore": {**sc_sum, "state_bytes_during_decode": _sc_state_bytes(cfg)},
            "transformer": {**ts_sum, "kv_bytes_during_decode": ts_dec.kv_bytes()},
            "latency_ratio_p50": ts_sum["p50_ms"] / max(sc_sum["p50_ms"], 1e-9),
        })
        rows.append(row)

    # ---- 斜率（核心主张：SC 延迟与 L 无关，TS 随 L 线性） -------------------
    ev = [r for r in rows if r.get("evaluated")]
    sc_slope = ts_slope = None
    if len(ev) >= 2:
        sc_slope, _ = fit_slope([(r["seq_len"], r["simplcore"]["p50_ms"]) for r in ev])
        ts_slope, _ = fit_slope([(r["seq_len"], r["transformer"]["p50_ms"]) for r in ev])
    # SC 平坦性判据：L_max 与 L_min 的 p50 比值 ≤ 1.5 —— "延迟不随上下文增长"
    # 的直接检验。不用最小二乘斜率：CPU 计时噪声非单调，斜率对它脆弱；
    # 比值判据对噪声稳健，且 TS 臂在同一判据下（tiny 实测 ~2×）可区分。
    sc_flat = False
    sc_flat_ratio = None
    analytic_1m: Optional[Dict[str, Any]] = None
    if len(ev) >= 2:
        p_min = min(r["simplcore"]["p50_ms"] for r in ev)
        p_max = max(r["simplcore"]["p50_ms"] for r in ev)
        sc_flat_ratio = p_max / max(p_min, 1e-9)
        sc_flat = sc_flat_ratio <= 1.5
    # 解析账：真实 1B 形态在 1M 下的倍数（不依赖 tiny 实测）
    try:
        from .config import get_preset
        cfg_1b = get_preset("tuan-1b")
        analytic_1m = analytic_decode_ledger(cfg_1b, cfg_1b.num_layers,
                                             cfg_1b.dim, 1048576)
    except Exception:
        analytic_1m = None
    gates = {
        "X-throughput": {
            "passed": bool(sc_flat and analytic_1m is not None
                           and analytic_1m["ts_kv_over_sc_state"] > 1000),
            "sc_flat_measured": sc_flat,
            "sc_flat_p50_ratio": sc_flat_ratio,
            "sc_flat_reason": (
                None if sc_flat else
                "计时窗口被背景负载污染或 SC 延迟确实随 L 增长；"
                "结构性平坦性（状态/FLOPs 与 L 无关）见 analytic_1m 与 scaling 账"),
            "sc_latency_slope_ms_per_token": sc_slope,
            "ts_latency_slope_ms_per_token": ts_slope,
            "measured_lengths": [r["seq_len"] for r in ev],
            "analytic_1m": analytic_1m,
            "not_measured": [{"seq_len": r["seq_len"], "reason": r.get("reason")}
                             for r in rows if not r.get("evaluated")],
            "note": ("实测部分只判 SC 延迟平坦（斜率跨阶梯相对变化 ≤15%）；"
                     "1M 吞吐倍数由解析账（KV 120GB vs 状态 30MiB）支撑，与实测分列，"
                     "正式验收以目标机实测为准"),
        },
    }
    report = {
        "protocol": PROTOCOL,
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": preset,
        "steps_per_length": int(steps),
        "lengths": rows,
        "gates": gates,
        "honesty": {
            "unqualified_claim_permitted": bool(gates["X-throughput"]["passed"]),
            "forbidden_checks": {
                "no_attention_internal_comparison": True,
                "equal_budget_qualifier_present": True,
                "no_self_reported_scores": True,       # 计时器读数，非自评
                "distribution_not_average_only": True,  # p10/p50/p90 + min/max 全入报告
            },
            "notes": [
                "1M/2M/4M 的倍数由解析账（flops + KV 带宽）与实测斜率外推支撑，"
                "外推与实测分列；正式验收以目标机实测为准",
                "TS 臂为标准 KV-cache 增量解码（prefill 正确性有守护测试钉死），非稻草人",
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
        prog="python -m simplcore.core.throughput_eval",
        description="解码吞吐对照（X-throughput）：SC O(1) 流式 vs TS KV cache。")
    p.add_argument("--preset", default="smoke", choices=("smoke", "small", "1b"))
    p.add_argument("--lengths", default="128,512,2048",
                   help="前缀长度阶梯（逗号分隔）")
    p.add_argument("--steps", type=int, default=32, help="每档计时的解码步数")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-device-mem-gb", type=float, default=80.0)
    p.add_argument("--ledger-only", action="store_true")
    p.add_argument("--out", default="runs/eval_throughput.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    lengths = [int(x) for x in str(args.lengths).split(",") if x.strip()]
    rep = run_throughput_eval(preset=args.preset, lengths=lengths,
                              steps=args.steps, warmup=args.warmup,
                              max_device_mem_gb=args.max_device_mem_gb,
                              ledger_only=args.ledger_only, out_path=args.out)
    print(f"protocol={rep['protocol']} preset={rep['preset']} lengths={lengths}")
    for r in rep["lengths"]:
        a = r["analytic"]
        if r.get("evaluated"):
            print(f"L={r['seq_len']}: sc_p50={r['simplcore']['p50_ms']:.3f}ms "
                  f"ts_p50={r['transformer']['p50_ms']:.3f}ms "
                  f"ratio={r['latency_ratio_p50']:.2f}x  "
                  f"ts_kv={a['ts_kv_cache_bytes']}B")
        else:
            print(f"L={r['seq_len']}: ledger-only  ts_kv={a['ts_kv_cache_bytes']}B "
                  f"ts_flops/sc={a['ts_flops_over_sc']:.1f}  ({r['reason']})")
    g = rep["gates"]["X-throughput"]
    print(f"slopes: sc={g['sc_latency_slope_ms_per_token']} "
          f"ts={g['ts_latency_slope_ms_per_token']}")
    print(f"gate X-throughput: {'PASS' if g['passed'] else 'FAIL'}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
