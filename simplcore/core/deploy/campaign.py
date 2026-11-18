# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""2B×24h 战役计划 —— 把"单张 RTX 4090D-24GB 在 24h 以内训练完整 2B 参数"
的硬指标变成可执行、可审计、带硬停闸门的阶段计划。

用户给定的硬指标（原文口径）
----------------------------
1. 目标硬件：**单张 RTX 4090D（24 GB GDDR6X）**，24 小时以内；
2. 训练对象：**完整 2B 参数**（tuan-2b 预设，2023M total / 1957M non-emb）；
3. 到点硬停：24h 强制冻结最优 checkpoint，无延期。

本模块只做计划与账本（纯解析计算，不建模型、不占显存、可在任何机器审计）：

* **显存账本**：AdamW 混合精度（16 B/参数）/ Adafactor 因子化（≈8 B/参数）/
  O(1) 流式状态 / 分块流式训练 / 朴素整段 FFT 五条路径的显存公式——
  结论（解析）：2B 全参在 24GB 上**只有 Adafactor + 分块流式**可行
  （AdamW 16 B/参数 = 32 GiB 直接爆），这是硬闸门不是建议；
* **阶段计划**（总预算 2 天，零余量）：
  Phase 0（≤1 天，栈验收：dry-run OOM-free + 实测吞吐修正 token 预算 + 数据冻结）
  → Phase 1（≤1 天 = **24h 训练硬停**）→ 复盘（立即，0 天）；
* **「无人可忽视」演示包**：8 条可独立复验的主张/边界，每条配复验命令——
  轰动的前置投资是可复验性，不是形容词；
* **验收协议**：与等参数 Transformer 的真实输出对照（等 token / 等 FLOPs 口径），
  附"禁止的对比方式"清单。

诚实边界
--------
本模块是**计划与守门**，不是训练结果。24h × 4090D 在诚实 MFU 下只喂得动
~0.3B tokens（tokens/param ≈ 0.15）——这是**欠训档**：训的是"完整 2B 参数"
（全参参与更新、loss 稳定下降），不是 Chinchilla-optimal 2B（40B tokens
需 ~55 个 24h）。所有质量数字以实机实测为准，见 ``docs/RTX4090D_TRAINING.md``。

**运行时**：``simplcore/core/training/train``（PyTorch + bf16 autocast + TF32 +
torch.compile + CUDA Graph + Adafactor）是默认训练路径；自研 CUDA 后端为可选
加速（契约见 ``backend_contract.py``）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..model.config import SIMPLCORE_PRESETS, SimplCoreConfig, get_preset
from ..backend.backend_contract import conformance_checks
from ..training.envlearn import WeightBudget
from ..eval.scaling import (
    _long_range_flops,
    embedding_params,
    non_embedding_params,
    stream_state_bytes,
    train_flops_per_token,
)

__all__ = [
    "GIB",
    "HBM_4090D_BYTES",
    "OPTIMIZER_BYTES_PER_PARAM",
    "ADAFACTOR_BYTES_PER_PARAM",
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
#: RTX 4090D = 24 GB GDDR6X，单设备单卡（CUDA 上下文自身占 ~0.5 GiB，
#: 账本以 24 GiB 为硬上限、闸门按 23 GiB 实测线执行）。
HBM_4090D_BYTES = 24 * GIB
#: Adam 混合精度优化器状态：bf16 权重 2 + bf16 梯度 2 + fp32 master 4
#: + Adam m 4 + Adam v 4 = 16 B/参数（mathproof P12 的 AdamW 口径）。
OPTIMIZER_BYTES_PER_PARAM = 16
#: Adafactor 因子化路径：fp32 参数 4 + fp32 梯度 4 + 行/列因子状态
#: （O(m+n)，2B 时 <0.1 GiB，按 0 计）≈ 8 B/参数——**24GB 上 2B 全参的
#: 唯一可行组合**（AdamW 16 B/参数 = 32 GiB 直接爆）。
ADAFACTOR_BYTES_PER_PARAM = 8
BYTES_BF16 = 2
#: 阶段时间预算（用户原话："24h 以内训练完整的 2B 参数"）。
#: 分配：Phase 0 栈验收 ≤1 天 + Phase 1 训练 ≤1 天（=24h 硬停）= 2 天，零余量。
TOTAL_BUDGET_DAYS = 2
PHASE0_MAX_DAYS = 1
PHASE1_MAX_DAYS = 1
PHASE2_MAX_DAYS = 0     # 本战役无 Phase 2
RETROSPECT_MAX_DAYS = 0  # 复盘是"立即"，不占独立天数
#: 上下文阶梯：分块流式路径显存与 L 无关的结构性验证，逐级翻倍到 8M。
CONTEXT_LADDER: Tuple[int, ...] = (
    512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024,
)
#: 结构性主张：上下文翻倍，训练显存增加 ≤ 1 GiB。
CONTEXT_CLAIM_GIB = 1.0
#: RTX 4090D bf16 稠密峰值（tensor core，fp32 累加口径）。4090 公称 165.2，
#: 4090D ≈ 0.95×；按规格表估算，实机以 Phase 0 实测为准。
RTX4090D_PEAK_TFLOPS = 157.0


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
    bytes_per_param: Optional[int] = None,
) -> int:
    """分块流式训练：把长上下文切成 chunk 顺序喂入，流式状态跨块携带。

    显存 = 优化器状态 + **单个 chunk** 的激活，**与总上下文 L 无关**——
    这是"翻倍 +1 GiB"主张成立的训练路径，也是 Phase 1 的硬性要求。
    ``bytes_per_param`` 缺省用 AdamW 16 B/参数口径；24GB 的 2B 战役
    传 ``ADAFACTOR_BYTES_PER_PARAM``（8 B/参数）。
    """
    bpp = int(bytes_per_param) if bytes_per_param is not None else OPTIMIZER_BYTES_PER_PARAM
    act = cfg.num_layers * int(chunk) * cfg.dim * act_tensors * BYTES_BF16
    return int(params) * bpp + act


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
    days: float, mfu: float = 0.30, util: float = 0.65,
    peak_tflops: float = RTX4090D_PEAK_TFLOPS,
) -> float:
    """有效训练 FLOPs = 天数 × 86400 × 有效占比 × 峰值 × MFU。

    ``util=0.65`` 扣调试/评测/断点重启（24h 战役同理）；MFU 以 Phase 0
    实测为准，0.30 是 torch.compile + bf16 在 Ada 卡上的诚实预设。
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
    bytes_per_param: Optional[int] = None,
) -> List[DoublingRow]:
    """沿上下文阶梯逐级算三条路径的显存，并给出翻倍增量。"""
    rows: List[DoublingRow] = []
    prev_stream = prev_chunked = 0
    for seq_len in ladder:
        sb = stream_bytes(cfg)
        cb = chunked_train_bytes(cfg, params, chunk=chunk,
                                 bytes_per_param=bytes_per_param)
        nb = naive_fft_train_bytes(cfg, params, seq_len)
        rows.append(DoublingRow(
            seq_len=int(seq_len),
            stream_bytes=sb, stream_delta=sb - prev_stream if prev_stream else 0,
            chunked_bytes=cb, chunked_delta=cb - prev_chunked if prev_chunked else 0,
            naive_bytes=nb, naive_fits_hbm=nb <= HBM_4090D_BYTES,
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
            "note": "Phase 0 栈验收与 Phase 1 训练都有硬停——2 天总预算含预检，不是只有训练",
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
        lines.append(f"=== 2B×24h·4090D 战役计划（preset={self.preset}） ===")
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
            lines.append(f"   峰值显存估算: {p.hbm_bytes/GIB:.1f} GiB / {HBM_4090D_BYTES/GIB:.0f} GiB")
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
    preset: str = "tuan-2b",
    cfg: Optional[SimplCoreConfig] = None,
    base_params: int = 2_000_000_000,
    max_learned: int = 0,
    total_cap: int = 2_000_000_000,
    seq_len: int = 4096,
    chunk: int = 4096,
    mfu: float = 0.30,
) -> CampaignPlan:
    """把"2B×24h×单张 4090D"硬指标组装成可审计的战役计划（纯解析，不建模型）。"""
    if cfg is None:
        cfg = get_preset(preset)
    analytic_total = embedding_params(cfg) + non_embedding_params(cfg)
    if not (1.8e9 <= analytic_total <= 2.2e9):
        raise ValueError(
            f"preset={preset} 解析参数量 {analytic_total/1e9:.3f}B 不在 2B 口径内，"
            "战役指标按 2B 全参给定，请换 tuan-2b")

    budget = WeightBudget(
        base_params=base_params, learned_params=0,
        max_learned=max_learned, total_cap=total_cap)

    # -- 显存账（Adafactor 路径：fp32 参数+梯度+因子状态 ≈ 8 B/参数）----------
    p1_hbm = chunked_train_bytes(cfg, base_params, chunk=chunk,
                                 bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)

    # -- token 预算（解析） ----------------------------------------------------
    p1_tokens = effective_flops(PHASE1_MAX_DAYS, mfu=mfu) / train_flops_per_token(cfg, seq_len)

    rows = doubling_report(cfg, base_params, chunk=chunk,
                           bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    verdict = claim_verdict(rows)

    # -- 阶段 -----------------------------------------------------------------
    # Phase 0 闸门自动从后端契约生成（单一事实源：backend_contract.py）。
    backend_gates = [Gate(c.name, c.criterion, kind=c.kind) for c in conformance_checks()]

    phase0 = Phase(
        name="Phase 0 栈验收",
        objective="训练栈 dry-run 验收（OOM-free + 实测吞吐修正 token 预算）+ 数据冻结",
        max_days=PHASE0_MAX_DAYS, hard_stop=True,
        steps=[
            "--dry-run 随机 token 100 步：显存峰值实测 <= 23 GiB（解析估算见下）",
            "真实数据小跑 >=200 步：实测 tokens/s 与 MFU，修正 24h token 预算",
            "torch.compile / CUDA Graph / Adafactor 三开关逐一实测收益，锁定配置",
            "语料与词表冻结（记录 SHA-256），文档级无泄漏切分",
        ],
        gates=backend_gates + [
            Gate("P0-budget-frozen", "数据/分词器/配置冻结并记录哈希，之后不得更改"),
            Gate("P0-oom-free-dryrun", "dry-run 100 步 OOM-free，显存峰值 <= 23 GiB", kind="hard"),
        ],
        hbm_bytes=p1_hbm,
    )

    phase1 = Phase(
        name="Phase 1 · 2B 基座（24h）",
        objective="单张 RTX 4090D-24GB 训练完整 2B 参数，24h 硬停（用户口径：24h 以内）",
        max_days=PHASE1_MAX_DAYS, hard_stop=True,
        steps=[
            "bf16 autocast 前向 + fp32 参数/梯度 + Adafactor 因子化二阶矩"
            "（24GB 上 2B 全参的唯一可行组合）",
            f"chunked 流式训练（chunk={chunk}，激活 O(chunk)）；OOM 即降 chunk 至 2048",
            "课程校准（NT46）：--max-steps = ceil(token 预算 / 全局 batch)"
            "（≈3358 步），使 _CTX_2B_4M 的 until_frac 阶梯在 24h 内展开；"
            "汇报口径 = 各阶段 traversed 步数/token 数，不宣称逐阶段收敛",
            "每 15 分钟 checkpoint + 每 30 分钟 eval；--time-budget-min 1440 硬停",
            "训完立即 held-out ppl / 长上下文检索评测并留档",
        ],
        gates=[
            Gate("P1-oom-free",
                 f"24GB 内 OOM-free（解析估算 {p1_hbm/GIB:.1f} GiB / "
                 f"{HBM_4090D_BYTES/GIB:.0f} GiB）"),
            Gate("P1-doubling-claim",
                 f"实测上下文翻倍显存增量 <= {CONTEXT_CLAIM_GIB} GiB（解析预期 0）"),
            Gate("P1-stream-consistency", "流式与整段评测结果一致"),
            Gate("P1-loss-stable", "held-out loss 曲线稳定无发散"),
            Gate("P1-token-budget",
                 f"24h 喂满解析 token 预算 ~{p1_tokens/1e9:.2f}B（MFU 实测后修正）",
                 kind="soft"),
            Gate("P1-hard-stop-24h", "到 24h 强制冻结最优 checkpoint，无条件停止", kind="hard"),
        ],
        hbm_bytes=p1_hbm, token_budget=p1_tokens,
    )

    retrospect = Phase(
        name="复盘（立即）",
        objective="训练完成后立即评测与复盘：token 预算达成率、loss 曲线、能力快照",
        max_days=RETROSPECT_MAX_DAYS, hard_stop=True,
        steps=[
            "汇总 metrics.jsonl：吞吐/MFU 实测 vs 解析预算、loss 曲线",
            "held-out ppl + 长上下文检索快照，与等 token Transformer 基线对照",
            "诚实记录欠训档：tokens/param 实测值与下一阶段建议",
        ],
        gates=[
            Gate("R-metrics-summarized", "吞吐/MFU/token 预算达成率已汇总入档"),
            Gate("R-eval-snapshot", "held-out ppl + 检索快照已留档"),
            Gate("R-hard-stop-0d", "复盘不占训练窗口（立即完成，不占 24h）", kind="hard"),
        ],
        hbm_bytes=0,
    )

    phase2 = Phase(
        name="Phase 2（本战役无）",
        objective="单卡 24h 只交付 2B 基座；多阶段蒸馏/适配器战役不在本指标内",
        max_days=PHASE2_MAX_DAYS, hard_stop=False,
        steps=["（预留：如需适配器/多卡战役，另行立项）"],
        gates=[],
        hbm_bytes=0,
    )

    protocol = [
        Gate("X-equal-budget",
             "对照口径：与等参数 Transformer 基线**等 token / 等 FLOPs**，禁止无限定宣称"),
        Gate("X-heldout-ppl", "held-out 困惑度报告实测值与基线比值（不承诺领先；欠训档如实呈现）"),
        Gate("X-longctx-retrieval", "长上下文检索准确率 >= 等预算基线（chunked 流式口径）"),
        Gate("X-stream-consistency", "全部评测在流式与整段两种模式下结果一致"),
        Gate("X-throughput", "解码吞吐 >= 基线（对方 KV cache GB 级 vs 我方流式 <1 MB）"),
    ]
    forbidden = [
        "注意力图 / 注意力热图等**机制内部**的对比（用户原话：真实输出结果，而不是注意力层面）",
        "无等 token / 等 FLOPs 口径的无限定宣称",
        "自报自评（无独立 verifier 的自我打分）",
        "只报平均、不报分布与最差 case 的 cherry-picking",
        "把 24h 欠训档说成训练充分（tokens/param 实测值必须随结果一起报告）",
    ]

    # -- 「无人可忽视」演示包 ---------------------------------------------------
    # 轰动的前置投资是可复验性：每条主张配一条任何人拿到仓库就能跑的复验命令。
    showcase = [
        ShowcaseItem(
            "S1-deploy-2gb", "1M 上下文完整推理 <= 2 GiB（1.948 GiB，含权重）",
            "python -m simplcore.core.deploy_budget", owner="Phase 0（账本）+ 实机一条命令"),
        ShowcaseItem(
            "S2-doubling-zero", "上下文 512K→8M 翻倍训练显存增量恒为 0（结构性，非调参）",
            "python -m simplcore.core.campaign  # 阶梯表 delta 全 0", owner="Phase 1"),
        ShowcaseItem(
            "S3-2b-fits-24gb",
            "2B 全参训练显存 < 24 GB（Adafactor + 分块流式；AdamW 32 GiB 不可行）",
            "python -m simplcore.core.campaign  # 显存账 + P1-oom-free 闸门", owner="Phase 0"),
        ShowcaseItem(
            "S4-equal-token-ppl", "等 token 预算 held-out ppl 对照等参数 Transformer（协议 X-equal-budget）",
            "对比脚本 + runs/eval_equal_token.json", owner="Phase 1"),
        ShowcaseItem(
            "S5-longctx-chunked", "分块流式长上下文评测 OOM-free（chunk 级激活，与总 L 无关）",
            "python -m simplcore.core.longctx_eval --help", owner="Phase 1"),
        ShowcaseItem(
            "S6-stream-consistency", "流式与整段评测逐位一致（NT34/NT36 的操作形式）",
            "python -m pytest simplcore/core/tests/test_stream_precision.py -q", owner="Phase 1"),
        ShowcaseItem(
            "S7-honest-token-budget",
            "24h 单卡 4090D 只喂得动 ~0.3B tokens（tokens/param ≈ 0.15）——欠训档主动呈现，不宣称 SOTA",
            "python -m simplcore.core.campaign  # token 预算行", kind="honest", owner="Phase 1"),
        ShowcaseItem(
            "S8-reproducible", "任何人 5 分钟复验数学与账本：全仓 pytest + 13 项数学证明",
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
    ap = argparse.ArgumentParser(description="SimplCore 2B×24h·4090D 战役计划（解析账本，不占显存）")
    ap.add_argument("--preset", default="tuan-2b", choices=sorted(SIMPLCORE_PRESETS))
    ap.add_argument("--mfu", type=float, default=0.30, help="假设的 MFU（0-1）")
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
