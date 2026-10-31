# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""1B→3B 战役计划 —— 把"1B 打 3B Transformer，再学 2B 附加参数"的硬指标
变成可执行、可审计、带硬停闸门的阶段计划。

用户给定的硬指标（原文口径）
----------------------------
1. 1B 模型 = 1B 参数 + **4M 上下文用于学习**；上下文要验证"是否真正 1M 起"，
   且**上下文翻倍增加 ≤ 1 GiB HBM 占用**。
2. 目标：1B 模型性能打 3B Transformer —— 用**真实输出结果**验收
   （held-out 困惑度 / 长上下文检索 / 代码 pass@k / 解码吞吐），
   **不接受"注意力层面"的机制内对比**。
3. 阶段一：1B 基座，单张 MI250（128 GB HBM = 2×64 GB GCD），**最长 30 天**。
4. 基座完成 → **立即复盘自身**（失败轨迹总结为训练样本），随后开始学习
   **2B 附加参数，最长 28 天，多了等不了**（硬停，无延期）。

本模块只做计划与账本（纯解析计算，不建模型、不占显存、可在任何机器审计）：

* **HBM 账本**：优化器状态 / O(1) 流式状态 / 分块流式训练 / 朴素整段 FFT
  四条路径的显存公式，并逐级验证"翻倍 +1 GiB"主张在哪些路径成立——
  结论（解析）：流式与分块路径翻倍增量 ≈ 0（≤1 GiB 成立），
  朴素整段 FFT 随 L 线性增长（4M 时约 1.5 TB，不可行）——
  因此**战役必须用分块流式训练**，这条是硬闸门不是建议。
* **阶段计划**（v2：**全程 90 天总预算，零余量**）：
  Phase 0（≤30 天，硬停，含后端验收）→ Phase 1（≤30 天，硬停）→
  复盘（≤2 天，硬停）→ Phase 2（≤28 天，硬停，多了等不了）。
  30+30+2+28 = 90：**任何阶段超时都触发降档预案，不挪用训练期**。
* **「无人可忽视」演示包**：8 条可独立复验的主张/边界，每条配复验命令——
  轰动的前置投资是可复验性，不是形容词。
* **验收协议**：与 3B Transformer 的真实输出对照（等 token / 等 FLOPs 口径），
  附"禁止的对比方式"清单。

诚实边界
--------
本模块是**计划与守门**，不是训练结果。"1B 打 3B Transformer"是**目标**，
不是已达成的事实；所有质量数字以实机实测为准
（见 ``docs/TRAINING_BUDGET_MI250.md``：30 天有效算力 ~1.5e20 FLOPs @25% MFU，
1B 只喂得动 20~35B tokens）。

**运行时**：自研 CUDA/C++ 后端（已就绪），PyTorch 退役出战役路径；
torch 参考实现保留为冻结数值规格 oracle。后端必须实现的算子、精度与逐位
一致性见 ``backend_contract.py``（Phase 0 闸门自动从它生成）；CUDA 端实现
本身不在本仓范围。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import SIMPLCORE_PRESETS, SimplCoreConfig, get_preset
from .backend_contract import conformance_checks
from .envlearn import WeightBudget
from .scaling import (
    _long_range_flops,
    embedding_params,
    non_embedding_params,
    stream_state_bytes,
    train_flops_per_token,
)

__all__ = [
    "GIB",
    "HBM_MI250_BYTES",
    "OPTIMIZER_BYTES_PER_PARAM",
    "TOTAL_BUDGET_DAYS",
    "PHASE0_MAX_DAYS",
    "PHASE1_MAX_DAYS",
    "PHASE2_MAX_DAYS",
    "RETROSPECT_MAX_DAYS",
    "CONTEXT_LADDER",
    "CONTEXT_CLAIM_GIB",
    "DoublingRow",
    "Gate",
    "Phase",
    "ShowcaseItem",
    "CampaignPlan",
    "optimizer_state_bytes",
    "naive_fft_train_bytes",
    "chunked_train_bytes",
    "adapter_train_flops_per_token",
    "effective_flops",
    "doubling_report",
    "claim_verdict",
    "build_plan",
    "main",
]

# ---------------------------------------------------------------------------
# 硬件与战役常量
# ---------------------------------------------------------------------------
GIB = 1024 ** 3
#: MI250 OAM = 2 GCD × 64 GB HBM2e。ROCm 暴露为两张卡，须 FSDP/ZeRO 切分，
#: 不能当单设备 128 GB 用（见 docs/TRAINING_BUDGET_MI250.md §1）。
HBM_MI250_BYTES = 128 * GIB
#: Adam 混合精度优化器状态：bf16 权重 2 + bf16 梯度 2 + fp32 master 4
#: + Adam m 4 + Adam v 4 = 16 B/参数（同 TRAINING_BUDGET_MI250.md §4）。
OPTIMIZER_BYTES_PER_PARAM = 16
BYTES_BF16 = 2
#: 阶段时间预算（用户原话："最长一个月" / "最长28天时间，多了等不了"）。
PHASE1_MAX_DAYS = 30
PHASE2_MAX_DAYS = 28
#: v2 硬指标：**全程 3 个月（90 天）必须完全搞定**——含 Phase 0 后端验收。
#: 分配：Phase 0 ≤30 + Phase 1 ≤30 + 复盘 ≤2 + Phase 2 ≤28 = 90，零余量。
TOTAL_BUDGET_DAYS = 90
PHASE0_MAX_DAYS = 30
RETROSPECT_MAX_DAYS = 2
#: 上下文阶梯：验证"是否真正 1M 起"，逐级翻倍到 8M。
CONTEXT_LADDER: Tuple[int, ...] = (
    512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024,
)
#: 用户主张：上下文翻倍，HBM 占用增加 ≤ 1 GiB。
CONTEXT_CLAIM_GIB = 1.0
MI250_PEAK_TFLOPS = 362.1  # bf16 矩阵峰值


# ---------------------------------------------------------------------------
# HBM 账本（四条路径）
# ---------------------------------------------------------------------------
def optimizer_state_bytes(params: int) -> int:
    """Adam 混合精度优化器状态 = 16 B/参数（1B → ~15.2 GiB）。"""
    return int(params) * OPTIMIZER_BYTES_PER_PARAM


def stream_bytes(cfg: SimplCoreConfig, batch: int = 1) -> int:
    """O(1) 流式状态：**与序列长度 L 无关**（复用 scaling.stream_state_bytes）。"""
    return int(stream_state_bytes(cfg, batch))


def naive_fft_train_bytes(
    cfg: SimplCoreConfig, params: int, seq_len: int, act_tensors: int = 6,
) -> int:
    """朴素路径：整段 L 一次 FFT 前向 + 一次反传。

    激活随 L **线性**增长（每层每 token 存 ~act_tensors 个 D 维 bf16 张量）。
    这是"不可行路径"，列在这里只为给分块路径做对照——4M 时约 1.5 TB。
    """
    act = cfg.num_layers * int(seq_len) * cfg.dim * act_tensors * BYTES_BF16
    return optimizer_state_bytes(params) + act


def chunked_train_bytes(
    cfg: SimplCoreConfig, params: int, chunk: int = 8192, act_tensors: int = 8,
) -> int:
    """分块流式训练：把长上下文切成 chunk 顺序喂入，流式状态跨块携带。

    显存 = 优化器状态 + **单个 chunk** 的激活，**与总上下文 L 无关**——
    这是"翻倍 +1 GiB"主张成立的训练路径，也是 Phase 1 的硬性要求。
    """
    act = cfg.num_layers * int(chunk) * cfg.dim * act_tensors * BYTES_BF16
    return optimizer_state_bytes(params) + act


def adapter_train_flops_per_token(
    cfg: SimplCoreConfig, learned_params: int, seq_len: int,
) -> float:
    """Phase 2 每 token FLOPs：基座冻结只需**前向**（2·N_base），
    适配器 fwd+bwd（6·N_learned），长程项按前向 1 倍计（梯度不穿基座参数）。"""
    base_fwd = 2.0 * non_embedding_params(cfg)
    learned = 6.0 * float(learned_params)
    long_range = _long_range_flops(cfg, seq_len) * cfg.num_layers
    return base_fwd + learned + long_range


def effective_flops(
    days: float, mfu: float = 0.25, util: float = 0.65,
    peak_tflops: float = MI250_PEAK_TFLOPS,
) -> float:
    """有效训练 FLOPs = 天数 × 86400 × 有效占比 × 峰值 × MFU。

    ``util=0.65`` 来自 TRAINING_BUDGET_MI250.md（扣调试/评测/断点重启）。
    """
    return days * 86400.0 * util * peak_tflops * 1e12 * mfu


# ---------------------------------------------------------------------------
# 翻倍验证（用户主张："翻倍增加 1GB HBM 占用"）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DoublingRow:
    """上下文阶梯上的一行显存账。"""

    seq_len: int
    stream_bytes: int          # O(1) 流式状态（推理/分块携带）
    stream_delta: int          # 相对上一级的增量（首行为 0）
    chunked_bytes: int         # 分块流式训练峰值
    chunked_delta: int
    naive_bytes: int           # 朴素整段 FFT（对照，预期爆掉）
    naive_fits_hbm: bool


def doubling_report(
    cfg: SimplCoreConfig, params: int,
    ladder: Sequence[int] = CONTEXT_LADDER, chunk: int = 8192,
) -> List[DoublingRow]:
    """沿上下文阶梯逐级算三条路径的显存，并给出翻倍增量。"""
    rows: List[DoublingRow] = []
    prev_stream = prev_chunked = 0
    for seq_len in ladder:
        sb = stream_bytes(cfg)
        cb = chunked_train_bytes(cfg, params, chunk=chunk)
        nb = naive_fft_train_bytes(cfg, params, seq_len)
        rows.append(DoublingRow(
            seq_len=int(seq_len),
            stream_bytes=sb, stream_delta=sb - prev_stream if prev_stream else 0,
            chunked_bytes=cb, chunked_delta=cb - prev_chunked if prev_chunked else 0,
            naive_bytes=nb, naive_fits_hbm=nb <= HBM_MI250_BYTES,
        ))
        prev_stream, prev_chunked = sb, cb
    return rows


def claim_verdict(rows: List[DoublingRow]) -> Dict[str, object]:
    """对"翻倍增加 ≤ 1 GiB"主张逐路径裁决。

    返回 ``{"stream": {...}, "chunked": {...}, "naive": {...}, "claim_holds": bool}``。
    主张只在**可行路径**（stream / chunked）上裁决；naive 只作对照。
    """
    limit = int(CONTEXT_CLAIM_GIB * GIB)
    max_stream_delta = max((r.stream_delta for r in rows), default=0)
    max_chunked_delta = max((r.chunked_delta for r in rows), default=0)
    naive_all_fit = all(r.naive_fits_hbm for r in rows)
    naive_overflows_from = next((r.seq_len for r in rows if not r.naive_fits_hbm), None)
    verdict = {
        "claim": f"上下文翻倍 HBM 增量 <= {CONTEXT_CLAIM_GIB} GiB",
        "limit_bytes": limit,
        "stream": {"max_delta": max_stream_delta,
                   "holds": max_stream_delta <= limit,
                   "note": "O(1) 流式状态与 L 无关，翻倍增量恒为 0"},
        "chunked": {"max_delta": max_chunked_delta,
                    "holds": max_chunked_delta <= limit,
                    "note": "分块训练显存只随 chunk 增长，与总 L 无关，增量恒为 0"},
        "naive": {"fits_hbm_anywhere": naive_all_fit,
                  "overflows_from": naive_overflows_from,
                  "note": "整段 FFT 随 L 线性增长（见 overflows_from 起点）—— 不可行，仅作对照"},
    }
    verdict["claim_holds"] = bool(verdict["stream"]["holds"] and verdict["chunked"]["holds"])
    return verdict


# ---------------------------------------------------------------------------
# 阶段与闸门
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    """验收闸门。``kind="hard"`` 不达标不得进入下一阶段。"""

    name: str
    criterion: str
    kind: str = "hard"  # "hard" | "soft" | "honest"


@dataclass(frozen=True)
class ShowcaseItem:
    """「无人可忽视」演示包条目：一条可独立复验的主张。

    ``verify`` 必须是**任何人拿到仓库就能跑**的命令或脚本——
    轰动的前置投资是可复验性，不是形容词。
    ``kind="honest"`` 为诚实边界条目（主动展示，不等别人拆穿）。
    """

    name: str
    claim: str
    verify: str
    kind: str = "hard"      # "hard" | "honest"
    owner: str = ""         # 产出的阶段


@dataclass(frozen=True)
class Phase:
    """战役阶段：目标、步骤、硬停天数、闸门、峰值显存估算。"""

    name: str
    objective: str
    max_days: int                  # 0 = 无独立时间预算（准备/复盘）
    hard_stop: bool                # True = 到点强制冻结，无延期
    steps: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    hbm_bytes: int = 0             # 峰值显存解析估算
    token_budget: Optional[float] = None  # 该阶段可喂的 token 上限（解析）

    def days_ok(self, used_days: float) -> bool:
        if self.max_days <= 0:
            return True
        return used_days <= self.max_days


@dataclass
class CampaignPlan:
    """完整战役：预算 + 阶段 + 最终验收协议 + 禁止项 + 演示包。"""

    preset: str
    base_params: int
    max_learned: int
    total_cap: int
    budget: WeightBudget
    phases: List[Phase]
    protocol: List[Gate]           # "1B(+2B) 打 3B Transformer" 的真实输出验收
    forbidden: List[str]           # 禁止的"对比"方式
    doubling: Dict[str, object]    # claim_verdict 结果
    showcase: List[ShowcaseItem] = field(default_factory=list)

    # -- 守门 ---------------------------------------------------------------
    def schedule_check(self) -> Dict[str, object]:
        """90 天总预算裁决：各阶段天数之和 ≤ TOTAL_BUDGET_DAYS，零余量。"""
        alloc = {p.name: p.max_days for p in self.phases if p.max_days > 0}
        total = sum(alloc.values())
        return {
            "cap_days": TOTAL_BUDGET_DAYS,
            "allocations": alloc,
            "total_days": total,
            "holds": total <= TOTAL_BUDGET_DAYS,
            "note": "Phase 0/复盘也有硬停——全程 90 天含后端验收，不是只有训练",
        }

    def gate_names(self) -> List[str]:
        names: List[str] = []
        for p in self.phases:
            names.extend(g.name for g in p.gates)
        names.extend(g.name for g in self.protocol)
        return names

    def hard_gate_names(self) -> List[str]:
        out: List[str] = []
        for p in self.phases:
            out.extend(g.name for g in p.gates if g.kind == "hard")
        out.extend(g.name for g in self.protocol if g.kind == "hard")
        return out

    def phase(self, name: str) -> Phase:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)

    def as_dict(self) -> Dict[str, object]:
        return {
            "preset": self.preset,
            "base_params": self.base_params,
            "max_learned": self.max_learned,
            "total_cap": self.total_cap,
            "budget": self.budget.as_dict(),
            "doubling": self.doubling,
            "schedule": self.schedule_check(),
            "phases": [
                {
                    "name": p.name,
                    "objective": p.objective,
                    "max_days": p.max_days,
                    "hard_stop": p.hard_stop,
                    "hbm_gib": round(p.hbm_bytes / GIB, 2),
                    "token_budget": p.token_budget,
                    "steps": p.steps,
                    "gates": [
                        {"name": g.name, "kind": g.kind, "criterion": g.criterion}
                        for g in p.gates
                    ],
                }
                for p in self.phases
            ],
            "protocol": [
                {"name": g.name, "kind": g.kind, "criterion": g.criterion}
                for g in self.protocol
            ],
            "showcase": [
                {"name": s.name, "kind": s.kind, "owner": s.owner,
                 "claim": s.claim, "verify": s.verify}
                for s in self.showcase
            ],
            "forbidden": list(self.forbidden),
        }

    def format(self) -> str:
        lines: List[str] = []
        b = self.budget
        sched = self.schedule_check()
        lines.append(f"=== 1B→3B 战役计划（preset={self.preset}） ===")
        lines.append(
            f"权重预算: base={b.base_params/1e9:.0f}B(冻结) + learned<={b.max_learned/1e9:.0f}B"
            f" | 总上限={b.total_cap/1e9:.0f}B | 显存只算可学={b.vram_params()/1e9:.2f}B")
        lines.append(
            f"总时间预算: {sched['total_days']} / {sched['cap_days']} 天"
            f"（{'成立' if sched['holds'] else '超支！'}）"
            f" | 分配: " + " + ".join(f"{k.split(' ')[0]}≤{v}d" for k, v in sched['allocations'].items()))
        d = self.doubling
        lines.append(f"翻倍主张: {d['claim']} -> {'成立' if d['claim_holds'] else '不成立'}")
        lines.append(f"  stream : max_delta={d['stream']['max_delta']/GIB:.4f} GiB  holds={d['stream']['holds']}")
        lines.append(f"  chunked: max_delta={d['chunked']['max_delta']/GIB:.4f} GiB  holds={d['chunked']['holds']}")
        lines.append(f"  naive  : {d['naive']['note']}")
        for p in self.phases:
            stop = f"硬停 {p.max_days} 天" if p.hard_stop else (
                f"{p.max_days} 天" if p.max_days else "无独立预算")
            lines.append(f"\n-- {p.name}（{stop}）: {p.objective}")
            lines.append(f"   峰值显存估算: {p.hbm_bytes/GIB:.1f} GiB / {HBM_MI250_BYTES/GIB:.0f} GiB")
            if p.token_budget:
                lines.append(f"   token 预算: ~{p.token_budget/1e9:.1f}B tokens")
            for s in p.steps:
                lines.append(f"   · {s}")
            for g in p.gates:
                tag = "[硬]" if g.kind == "hard" else "[软]"
                lines.append(f"   {tag} {g.name}: {g.criterion}")
        lines.append("\n== 最终验收协议（真实输出，1B+2B vs 3B Transformer）==")
        for g in self.protocol:
            tag = "[硬]" if g.kind == "hard" else "[软]"
            lines.append(f" {tag} {g.name}: {g.criterion}")
        lines.append("\n== 「无人可忽视」演示包（每条可独立复验）==")
        for s_ in self.showcase:
            tag = "[主张]" if s_.kind == "hard" else "[边界]"
            lines.append(f" {tag} {s_.name}（{s_.owner}）: {s_.claim}")
            lines.append(f"        复验: {s_.verify}")
        lines.append("== 禁止的对比方式 ==")
        for f_ in self.forbidden:
            lines.append(f" ✗ {f_}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 组装战役
# ---------------------------------------------------------------------------
def build_plan(
    preset: str = "tuan-1b",
    cfg: Optional[SimplCoreConfig] = None,
    base_params: int = 1_000_000_000,
    max_learned: int = 2_000_000_000,
    total_cap: int = 3_000_000_000,
    seq_len: int = 4 * 1024 * 1024,
    chunk: int = 8192,
    mfu: float = 0.25,
) -> CampaignPlan:
    """把用户硬指标组装成可审计的战役计划（纯解析，不建模型）。"""
    if cfg is None:
        cfg = get_preset(preset)
    analytic_total = embedding_params(cfg) + non_embedding_params(cfg)
    if not (0.9e9 <= analytic_total <= 1.2e9):
        raise ValueError(
            f"preset={preset} 解析参数量 {analytic_total/1e9:.3f}B 不在 1B 口径内，"
            "战役指标按 1B 基座给定，请换 tuan-1b")

    budget = WeightBudget(
        base_params=base_params, learned_params=0,
        max_learned=max_learned, total_cap=total_cap)

    # -- 显存账 ---------------------------------------------------------------
    p1_hbm = chunked_train_bytes(cfg, base_params, chunk=chunk)
    # Phase 2：基座 bf16 常驻（冻结，只前向）+ 2B 适配器全套优化器状态 + 单 chunk 激活
    p2_hbm = (base_params * BYTES_BF16
              + optimizer_state_bytes(max_learned)
              + chunked_train_bytes(cfg, max_learned, chunk=chunk))

    # -- token 预算（解析） ----------------------------------------------------
    p1_tokens = effective_flops(PHASE1_MAX_DAYS, mfu=mfu) / train_flops_per_token(cfg, seq_len)
    p2_tokens = (effective_flops(PHASE2_MAX_DAYS, mfu=mfu)
                 / adapter_train_flops_per_token(cfg, max_learned, seq_len))

    rows = doubling_report(cfg, base_params, chunk=chunk)
    verdict = claim_verdict(rows)

    # -- 阶段 -----------------------------------------------------------------
    # Phase 0 闸门自动从后端契约生成（单一事实源：backend_contract.py）。
    backend_gates = [Gate(c.name, c.criterion, kind=c.kind) for c in conformance_checks()]

    phase0 = Phase(
        name="Phase 0 冻结",
        objective="自研 CUDA/C++ 后端一致性验收（BackendContract）+ 数据 / 分词器 / 配置冻结"
                  "（运行时 = 自研后端；PyTorch 退役出战役路径，torch 实现仅作冻结数值 oracle）",
        max_days=PHASE0_MAX_DAYS, hard_stop=True,
        steps=[
            "后端逐项通过 BackendContract 一致性检查（下闸门自动生成自 conformance_checks）",
            "语料与词表冻结（32000 vocab 口径），文档级无泄漏切分",
            f"后端在 512K 上下文跑通 {chunk}-chunk 分块流式训练；实测吞吐/MFU，修正 token 预算",
            "90 天总预算下 Phase 0 不得挤占训练期：第 30 天未过 conformance 即战役重排（砍 8M 档，保 4M）",
        ],
        gates=backend_gates + [
            Gate("P0-budget-frozen", "数据/分词器/配置冻结并记录哈希，之后不得更改"),
            Gate("P0-hard-stop-30d", "到 30 天硬停：conformance 未全过即触发降档预案（8M→4M），训练期不动", kind="hard"),
        ],
        hbm_bytes=p1_hbm,
    )

    phase1 = Phase(
        name="Phase 1 · 1B 基座",
        objective="1B 基座训练完成（4M 上下文学习），单张 MI250，最长 30 天，到点硬停",
        max_days=PHASE1_MAX_DAYS, hard_stop=True,
        steps=[
            "上下文课程：1M → 2M → 4M 逐级翻倍，每级稳定后升级",
            "每周 checkpoint + held-out loss 曲线留档",
            "第 4 周起转入 4M 长上下文主训",
        ],
        gates=[
            Gate("P1-chunked-4m-nooom",
                 f"4M 上下文分块训练在 {HBM_MI250_BYTES/GIB:.0f} GB HBM 内 OOM-free"
                 f"（解析估算 {p1_hbm/GIB:.1f} GiB）"),
            Gate("P1-doubling-claim",
                 f"实测上下文翻倍 HBM 增量 ≤ {CONTEXT_CLAIM_GIB} GiB（解析预期 0）"),
            Gate("P1-stream-bitexact-4m", "4M 长度下 流式==并行 逐位一致"),
            Gate("P1-loss-stable", "held-out loss 曲线稳定无发散"),
            Gate("P1-token-budget", f"喂满解析 token 预算 ~{p1_tokens/1e9:.1f}B（MFU 实测后修正）",
                 kind="soft"),
            Gate("P1-hard-stop-30d", "到 30 天强制冻结最优 checkpoint，无条件进入复盘", kind="hard"),
        ],
        hbm_bytes=p1_hbm, token_budget=p1_tokens,
    )

    retrospect = Phase(
        name="复盘（立即）",
        objective="基座完成后**立即**复盘自身：失败轨迹 → 总结为训练样本（envlearn.summarize_failures 语义）",
        max_days=RETROSPECT_MAX_DAYS, hard_stop=True,
        steps=[
            "收集 Phase 1 全部失败/低分轨迹（长上下文检索、代码执行）",
            "总结为少量高价值训练样本（不是原样回放）",
            "产出复盘报告：哪类能力缺失、2B 附加参数优先学什么",
        ],
        gates=[
            Gate("R-failures-summarized", "失败轨迹已总结为训练样本并有清单"),
            Gate("R-report-written", "复盘报告落盘，明确 2B 阶段的学习优先级"),
            Gate("R-hard-stop-2d", "复盘 ≤2 天（90 天总预算零余量），超时砍样本量不砍 Phase 2", kind="hard"),
        ],
        hbm_bytes=0,
    )

    phase2 = Phase(
        name="Phase 2 · 2B 附加参数",
        objective="基座冻结，学习 ≤2B 适配器（总上限 3B），分阶段蒸馏，最长 28 天，多了等不了",
        max_days=PHASE2_MAX_DAYS, hard_stop=True,
        steps=[
            "LearnedAdapter.grow 按复盘优先级分配 2B 预算（WeightBudget 硬拒超额）",
            "分阶段蒸馏：环境交互 → 总结 → 蒸馏 → 环境验证 → no-regression gate → 退化回滚",
            "每 stage 留档：loss_before/after、pass_rate、是否回滚",
        ],
        gates=[
            Gate("P2-base-frozen", "基座权重逐位未变（快照对比）"),
            Gate("P2-budget-cap", f"适配器 ≤ {max_learned/1e9:.0f}B 且总权重 ≤ {total_cap/1e9:.0f}B（硬拒超额）"),
            Gate("P2-no-regression", "每个 stage 通过 no-regression gate，退化即回滚"),
            Gate("P2-hard-stop-28d", "到 28 天硬停，多了等不了，无延期", kind="hard"),
        ],
        hbm_bytes=p2_hbm, token_budget=p2_tokens,
    )

    protocol = [
        Gate("X-equal-budget", "对照口径：与 3B Transformer 基线**等 token / 等 FLOPs**，禁止无限定宣称"),
        Gate("X-heldout-ppl", "held-out 困惑度 ≤ 基线 × 1.05（目标值，非承诺；实测为准）"),
        Gate("X-longctx-retrieval", "1M / 2M / 4M 长上下文检索准确率 ≥ 基线"),
        Gate("X-code-passk", "代码 pass@k（CodeExecutionVerifier 真实执行）≥ 基线"),
        Gate("X-throughput", "1M 上下文解码吞吐 ≥ 基线（对方 KV cache ~32 GB vs 我方流式 <1 MB）"),
        Gate("X-stream-consistency", "全部评测在流式与整段两种模式下结果一致"),
    ]
    forbidden = [
        "注意力图 / 注意力热图等**机制内部**的对比（用户原话：真实输出结果，而不是注意力层面）",
        "无等 token / 等 FLOPs 口径的无限定'1B 打 3B'宣称",
        "自报自评（无独立 verifier 的自我打分）",
        "只报平均、不报分布与最差 case 的 cherry-picking",
    ]

    # -- 「无人可忽视」演示包 ---------------------------------------------------
    # 轰动的前置投资是可复验性：每条主张配一条任何人拿到仓库就能跑的复验命令。
    showcase = [
        ShowcaseItem(
            "S1-deploy-2gb", "1M 上下文完整推理 ≤ 2 GiB（1.948 GiB，含权重）",
            "python -m simplcore.core.deploy_budget", owner="Phase 0（账本）+ 实机一条命令"),
        ShowcaseItem(
            "S2-doubling-zero", "上下文 1M→8M 翻倍 HBM 增量恒为 0（结构性，非调参）",
            "python -m simplcore.core.campaign  # 阶梯表 delta 全 0", owner="Phase 1"),
        ShowcaseItem(
            "S3-decode-10x", "1M 上下文解码吞吐 ≥ 10× 同卡 TS-3B（KV cache 60 GB vs 流式 30 MiB）",
            "python -m simplcore.core.scaling  + 后端实机 benchmark", owner="Phase 1"),
        ShowcaseItem(
            "S4-equal-token-ppl", "等 token 预算 held-out ppl ≤ TS-2B × 1.05",
            "对比脚本 + runs/eval_equal_token.json（协议 X-equal-budget）", owner="Phase 1"),
        ShowcaseItem(
            "S5-longctx-4m", "1M/2M/4M 检索+摘要综合 ≥ YaRN 扩展 TS-5B（§7 轴 C 中位预测）",
            "评测脚本 + 原始输出留档", owner="Phase 1+2"),
        ShowcaseItem(
            "S6-evolution-passk", "自进化前后代码 pass@k 可验证提升，含 no-regression 回滚记录",
            "envlearn stage 日志（loss_before/after、pass_rate、rollback）", owner="Phase 2"),
        ShowcaseItem(
            "S7-honest-needle", "1M needle 精确召回 < TS-2B —— 信息论边界主动呈现（30.4 MiB 状态 vs ≥2 GB 原文）",
            "同评测脚本，结果如实入演示包", kind="honest", owner="Phase 1"),
        ShowcaseItem(
            "S8-reproducible", "任何人 5 分钟复验数学与账本：848 测试 + 13 证明 + 34 守卫 + 21 CLI",
            "python -m pytest simplcore -q && python -m simplcore.core.mathproof", owner="全程"),
    ]

    return CampaignPlan(
        preset=preset, base_params=base_params, max_learned=max_learned,
        total_cap=total_cap, budget=budget,
        phases=[phase0, phase1, retrospect, phase2],
        protocol=protocol, forbidden=forbidden, doubling=verdict,
        showcase=showcase,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI：``python -m simplcore.core.campaign`` —— 打印完整战役计划与账本。"""
    ap = argparse.ArgumentParser(description="SimplCore 1B→3B 战役计划（解析账本，不占显存）")
    ap.add_argument("--preset", default="tuan-1b", choices=sorted(SIMPLCORE_PRESETS))
    ap.add_argument("--mfu", type=float, default=0.25, help="假设的 MFU（0-1）")
    ap.add_argument("--chunk", type=int, default=8192, help="分块训练 chunk 长度")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args(argv)

    plan = build_plan(preset=args.preset, chunk=args.chunk, mfu=args.mfu)
    if args.json:
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(plan.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
