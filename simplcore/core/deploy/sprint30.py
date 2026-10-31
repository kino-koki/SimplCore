# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""30 天冲刺计划（Sprint-30）——「1 个月内**真实确保**完成训练」的可执行账本。

与 ``campaign.py``（90 天 v2 战役）的区别
------------------------------------------
campaign.py 回答的是"按 25% MFU 能喂多少 token"；本模块回答的是
**在保守 MFU 下账本仍然闭合**的 30 天硬保证，手段有三：

1. **测后锁预算（measure-then-lock）**：第 1–2 天用 512K 上下文分块冒烟
   实测 MFU，随即把 token 预算锁定为 ``min(目标, 实测有效算力 / 每 token FLOPs)``。
   训练中段不再依赖任何纸面假设——这是"真实确保"的第一根支柱。
2. **保守下限闭合**：整个计划在 ``MFU_FLOOR=0.12``（自研后端 + rocFFT 的
   现实下限）仍然闭合；低于下限触发**降档阶梯**，而不是延期。
3. **降档阶梯（fallback ladder）**：实测 MFU 下 1B 档 token 预算 < 15B 时
   自动切换 ``tuan-870m``（~0.87B，每 token FLOPs 更低）**最大化** token
   预算（MFU≥0.15 时恢复 ≥15B；更低时两档均 <15B，账本照实呈现）——
   交付出一个**训完的模型**，好过一个**没训完的 1B**。

运行时前提（解除 Phase 0 阻塞）
--------------------------------
campaign v2 的 Phase 0（≤30 天自研 HIP 后端验收）被 sprint 移出关键路径：
训练主路径走 **PyTorch + torch.compile → ROCm CK**（README §运行时口径），
自研 HIP 后端降级为训练期并行任务、不阻塞开训。因此 30 天里
**第 1 天就可以开始喂数据**。

两种模式
--------
* ``backbone30``（默认）：1B 基座 30 天内训完（用户原话"最长一个月"口径）；
  2B 学习分支在 D29–D30 起步蒸馏、到点冻结最优 checkpoint，完整学习
  按 campaign v2 续期 28 天。
* ``allin30``：**含 2B 学习分支在内全部 30 天**：基座 22 天 + 切换 1 天 +
  2B 蒸馏 7 天。代价是基座 token 预算缩水（见账本输出）——这是
  "一个月全要"的真实价格，账本照实打印，不做乐观伪装。

诚实边界
--------
本模块是**计划与守门**，不是训练结果。它保证的是"时间账本闭合 +
预算锁定机制"，不保证"1B 打 3B Transformer"——后者仍以实测输出验收
（campaign.py 协议 X-*）。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..model.config import SIMPLCORE_PRESETS, SimplCoreConfig, get_preset
from ..eval.scaling import non_embedding_params, train_flops_per_token

__all__ = [
    "MI250_PEAK_TFLOPS",
    "SPRINT_UTIL",
    "SPRINT_DAYS",
    "MFU_FLOOR",
    "TARGET_TOKENS_1B",
    "MIN_TOKENS_KEEP_1B",
    "CHUNK",
    "SEQ_LEN_4M",
    "MfuRow",
    "SprintGate",
    "SprintPlan",
    "effective_flops",
    "scenario_table",
    "locked_token_budget",
    "build_sprint",
    "main",
]

# ---------------------------------------------------------------------------
# 常量（与 campaign.py 同源：MI250 bf16 矩阵峰值 / 预算利用率口径）
# ---------------------------------------------------------------------------
MI250_PEAK_TFLOPS = 362.1   # bf16 矩阵峰值（同 campaign.MI250_PEAK_TFLOPS）
SPRINT_UTIL = 0.65          # 扣调试/评测/断点重启（同 campaign.effective_flops）
SPRINT_DAYS = 30            # 用户硬指标："1 月必须完全做好训练"
#: 保守 MFU 下限：自研融合核 + rocFFT 在 MI250 上的现实下限估计。
#: 低于它说明实现有缺陷（或硬件被共享），触发降档而不是硬撑。
MFU_FLOOR = 0.12
#: Chinchilla≈20 token/参数：1B 基座的质量目标预算（非承诺，账本会裁剪）。
TARGET_TOKENS_1B = 20.0e9
#: 保 1B 不降档的最低 token 预算：低于它换 tuan-870m（0.87B）换预算。
MIN_TOKENS_KEEP_1B = 15.0e9
CHUNK = 8192
SEQ_LEN_4M = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# 解析账本
# ---------------------------------------------------------------------------
def effective_flops(days: float, mfu: float, util: float = SPRINT_UTIL,
                    peak_tflops: float = MI250_PEAK_TFLOPS) -> float:
    """30 天单卡有效算力 = 天数 × 86400 s × util × 峰值 × MFU（同 campaign 口径）。"""
    return days * 86400.0 * util * peak_tflops * 1e12 * mfu


def scenario_table(cfg: SimplCoreConfig, seq_len: int = SEQ_LEN_4M,
                   days: float = SPRINT_DAYS,
                   mfus: Sequence[float] = (0.10, 0.12, 0.15, 0.20, 0.25)) -> List["MfuRow"]:
    """MFU 情景表：每个 MFU 档位下 30 天可喂 token 数与是否达标。"""
    fpt = train_flops_per_token(cfg, seq_len)
    rows = []
    for mfu in mfus:
        eff = effective_flops(days, mfu)
        tokens = eff / fpt
        rows.append(MfuRow(
            mfu=mfu, eff_flops=eff, tokens=tokens, flops_per_token=fpt,
            meets_target=tokens >= TARGET_TOKENS_1B,
            meets_floor=tokens >= MIN_TOKENS_KEEP_1B,
        ))
    return rows


@dataclass(frozen=True)
class MfuRow:
    """情景表一行。"""
    mfu: float
    eff_flops: float
    tokens: float
    flops_per_token: float
    meets_target: bool   # ≥ 20B（1B 模型的 Chinchilla 目标）
    meets_floor: bool    # ≥ 15B（保 1B 不降档的下限）


def locked_token_budget(measured_mfu: float, remaining_days: float,
                        cfg: SimplCoreConfig, seq_len: int = SEQ_LEN_4M) -> float:
    """测后锁预算：``min(20B 目标, 实测有效算力 / 每 token FLOPs)``。

    第 2 天实测 MFU 后调用一次，输出即本 sprint 的 token 硬预算；
    训练中段不再改（改了就不是"真实确保"）。
    """
    eff = effective_flops(remaining_days, measured_mfu)
    return min(TARGET_TOKENS_1B, eff / train_flops_per_token(cfg, seq_len))


# ---------------------------------------------------------------------------
# 冲刺计划
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SprintGate:
    """冲刺闸门。``hard=True`` 未过即触发预案，不挪用后续天数。"""
    name: str
    day: str
    criterion: str
    hard: bool = True


@dataclass
class SprintPlan:
    """30 天冲刺计划：模式 + 降档阶梯 + 锁定账本 + 逐周日程 + 闸门。"""

    mode: str                       # "backbone30" | "allin30"
    primary_preset: str
    fallback_preset: str
    mfu_floor: float
    days_phase1: int
    days_switch: int                # 复盘/切换（allin30 用）
    days_phase2: int                # 2B 蒸馏窗口（allin30=7；backbone30=2 起步）
    primary_cfg: SimplCoreConfig
    fallback_cfg: SimplCoreConfig
    seq_len: int = SEQ_LEN_4M
    measured_mfu: Optional[float] = None   # None = 尚未实测（D2 锁定）
    gates: List[SprintGate] = field(default_factory=list)

    # -- 账本 ----------------------------------------------------------------
    def _cfg_for(self, mfu: Optional[float]):
        """按实测 MFU 裁决用哪个 preset（降档阶梯的裁决点）。

        未实测（``mfu=None``）时返回主档并按 MFU 下限演算其预算——降档决定
        属于 D2 实测时刻，不由纸面假设代做。实测后：1B 档保留条件是
        ``预算 ≥ 15B``；否则切 870m 档**最大化** token 预算（注意：MFU<
        ~0.14 时两档都到不了 15B，照实呈现，不做乐观伪装）。
        """
        if mfu is None:
            return self.primary_cfg, locked_token_budget(
                self.mfu_floor, self.days_phase1, self.primary_cfg, self.seq_len)
        tok_p = locked_token_budget(mfu, self.days_phase1, self.primary_cfg, self.seq_len)
        if tok_p >= MIN_TOKENS_KEEP_1B:
            return self.primary_cfg, tok_p
        return self.fallback_cfg, locked_token_budget(
            mfu, self.days_phase1, self.fallback_cfg, self.seq_len)

    def ledger(self) -> Dict[str, object]:
        """当前裁决下的完整账本（闭合性在此断言）。"""
        cfg, tokens = self._cfg_for(self.measured_mfu)
        fpt = train_flops_per_token(cfg, self.seq_len)
        mfu = self.measured_mfu if self.measured_mfu is not None else self.mfu_floor
        eff = effective_flops(self.days_phase1, mfu)
        closes = tokens * fpt <= eff * 1.001   # 闭合性：预算 ≤ 有效算力
        derated = cfg is self.fallback_cfg
        total_days = self.days_phase1 + self.days_switch + self.days_phase2
        out = {
            "mode": self.mode,
            "active_preset": self._preset_name(cfg),
            "derated": derated,
            "derate_reason": (None if not derated else
                              f"实测 MFU={mfu:.2f} 下 1B 档预算 < "
                              f"{MIN_TOKENS_KEEP_1B/1e9:.0f}B -> 切降档档位最大化 token 预算"
                              "（MFU<~0.14 时两档均 <15B，照实呈现）"),
            "params_total": cfg.dim * 0 + _analytic_total(cfg),
            "flops_per_token": fpt,
            "days_phase1": self.days_phase1,
            "days_switch": self.days_switch,
            "days_phase2": self.days_phase2,
            "total_days": total_days,
            "schedule_fits": total_days <= SPRINT_DAYS,
            "mfu_assumed": mfu,
            "effective_flops": eff,
            "locked_tokens": tokens,
            "meets_chinchilla_20B": tokens >= TARGET_TOKENS_1B,
            "closes_at_floor": None,   # 下面补算：floor MFU 下也闭合
        }
        # 闭合性下限复核：换算到 MFU_FLOOR 的预算仍 ≤ floor 有效算力
        cfg_f, tok_f = self._cfg_for(None)
        fpt_f = train_flops_per_token(cfg_f, self.seq_len)
        eff_f = effective_flops(self.days_phase1, self.mfu_floor)
        out["closes_at_floor"] = bool(tok_f * fpt_f <= eff_f * 1.001)
        out["floor_tokens"] = tok_f
        out["closes"] = bool(closes and out["closes_at_floor"] and out["schedule_fits"])
        return out

    def _preset_name(self, cfg: SimplCoreConfig) -> str:
        for name in SIMPLCORE_PRESETS:
            if SIMPLCORE_PRESETS[name].to_dict() == cfg.to_dict():
                return name
        return "custom"

    def as_dict(self) -> Dict[str, object]:
        return {
            "ledger": self.ledger(),
            "scenarios_primary": [vars(r) | {"tokens": r.tokens} for r in
                                  scenario_table(self.primary_cfg, self.seq_len,
                                                 days=self.days_phase1)],
            "gates": [vars(g) | {"hard": g.hard} for g in self.gates],
            "schedule": [f"{g.day} {g.name}: {g.criterion}" for g in self.gates],
        }

    def format(self) -> str:
        d = self.ledger()
        lines = [f"=== Sprint-30（mode={self.mode}）==="]
        lines.append(
            f"active_preset={d['active_preset']}  derated={d['derated']}"
            f"  params={d['params_total']/1e9:.2f}B  fpt={d['flops_per_token']/1e9:.2f}G")
        lines.append(
            f"日程: 基座{d['days_phase1']}d + 切换{d['days_switch']}d + 2B蒸馏{d['days_phase2']}d"
            f" = {d['total_days']}d (≤30 {'OK' if d['schedule_fits'] else '超支!'})")
        lines.append(
            f"锁定 token 预算: {d['locked_tokens']/1e9:.1f}B"
            f"（Chinchilla 20B 目标: {'达标' if d['meets_chinchilla_20B'] else '未达标-照实呈现'}）"
            f" | MFU 假设 {d['mfu_assumed']:.2f}")
        lines.append(
            f"闭合性: 30天账本闭合={d['closes']}"
            f"（MFU 下限 {self.mfu_floor:.2f} 处 token 预算 {d['floor_tokens']/1e9:.1f}B）")
        if d["derated"]:
            lines.append(f"降档: {d['derate_reason']}")
        lines.append("\n-- MFU 情景表（基座窗口）--")
        lines.append(f"{'MFU':>5} {'有效FLOPs':>14} {'tokens':>9}  ≥20B  ≥15B")
        for r in scenario_table(self.primary_cfg, self.seq_len, days=self.days_phase1):
            lines.append(f"{r.mfu:5.2f} {r.eff_flops:14.3e} {r.tokens/1e9:9.2f}B"
                         f"  {'✓' if r.meets_target else '✗'}    {'✓' if r.meets_floor else '✗'}")
        lines.append("\n-- 日程与闸门 --")
        for g in self.gates:
            tag = "[硬]" if g.hard else "[软]"
            lines.append(f"{tag} {g.day} {g.name}: {g.criterion}")
        lines.append("\n诚实边界: 本计划保证时间账本闭合与预算锁定机制；"
                     "「1B 打 3B」仍以 campaign 协议 X-* 的实测输出验收。")
        return "\n".join(lines)


def _analytic_total(cfg: SimplCoreConfig) -> int:
    from ..model.config import analytic_params
    return analytic_params(cfg)


def build_sprint(
    mode: str = "backbone30",
    primary: str = "tuan-1b",
    fallback: str = "tuan-870m",
    seq_len: int = SEQ_LEN_4M,
    measured_mfu: Optional[float] = None,
    mfu_floor: float = MFU_FLOOR,
) -> SprintPlan:
    """组装 30 天冲刺计划（纯解析，不建模型、不占显存）。"""
    if mode not in ("backbone30", "allin30"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "backbone30":
        days_p1, days_sw, days_p2 = 30, 0, 0    # 2B 完整学习按 campaign v2 续期
    else:
        days_p1, days_sw, days_p2 = 22, 1, 7    # 全月压缩档的真实价格

    if mode == "backbone30":
        # P1 D1-30；2B 完整学习按 campaign v2 续期 28 天，这里只起步
        day_specs = [
            ("S-freeze", "D1", "语料/分词器/配置冻结并记录哈希；torch.compile→ROCm CK 路径编译通过"),
            ("S-measure-mfu", "D2", "512K 分块冒烟实测 MFU，锁定 token 预算"),
            ("S-course-1m", "D3-D6", "1M 上下文主训；held-out loss 曲线留档"),
            ("S-course-2m", "D7-D12", "2M 上下文主训；翻倍 HBM 增量实测 ≤1 GiB"),
            ("S-main-4m", "D13-D26", "4M 主训窗口（占基座窗口 ~46%）"),
            ("S-eval-harvest", "D27-D28", "held-out ppl / 长上下文检索 / 代码 pass@k；失败轨迹收集为复盘样本"),
            ("S-adapter-boot", "D29-D30", "2B 学习分支起步蒸馏（verifier 门控 + no-regression 回滚）"),
            ("S-hard-stop-30d", "D30", "30 天硬停：冻结最优 checkpoint——『1 月必须完全做好训练』的执行口径"),
        ]
    else:
        # allin30：P1 D1-22 → 切换/复盘 D23 → 2B 蒸馏 D24-30
        day_specs = [
            ("S-freeze", "D1", "语料/分词器/配置冻结并记录哈希；torch.compile→ROCm CK 路径编译通过"),
            ("S-measure-mfu", "D2", "512K 分块冒烟实测 MFU，锁定 token 预算"),
            ("S-course-1m", "D3-D5", "1M 上下文主训；held-out loss 曲线留档"),
            ("S-course-2m", "D6-D9", "2M 上下文主训；翻倍 HBM 增量实测 ≤1 GiB"),
            ("S-main-4m", "D10-D20", "4M 主训窗口（压缩档）"),
            ("S-eval-harvest", "D21-D22", "held-out 评测 + 失败轨迹收集"),
            ("S-retrospect", "D23", "复盘 1 天：失败轨迹总结为训练样本，确定 2B 预算分配"),
            ("S-adapter-distill", "D24-D30", "2B 学习分支蒸馏 7 天（verifier 门控 + no-regression 回滚）"),
            ("S-hard-stop-30d", "D30", "30 天硬停：全部训练冻结——『一个月全要』的执行口径"),
        ]
    gates = [SprintGate(n, day, c, hard=(n == "S-hard-stop-30d"))
             for n, day, c in day_specs]

    return SprintPlan(
        mode=mode, primary_preset=primary, fallback_preset=fallback,
        mfu_floor=mfu_floor, days_phase1=days_p1, days_switch=days_sw,
        days_phase2=days_p2, primary_cfg=get_preset(primary),
        fallback_cfg=get_preset(fallback), seq_len=seq_len,
        measured_mfu=measured_mfu, gates=gates,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SimplCore Sprint-30（30 天硬保证训练账本）")
    ap.add_argument("--mode", default="backbone30", choices=["backbone30", "allin30"],
                    help="backbone30=基座30天（默认）; allin30=含2B分支全月压缩22+1+7")
    ap.add_argument("--measured-mfu", type=float, default=None,
                    help="D2 实测 MFU；不传则按下限 %.2f 演算" % MFU_FLOOR)
    ap.add_argument("--mfu-floor", type=float, default=MFU_FLOOR)
    ap.add_argument("--primary", default="tuan-1b")
    ap.add_argument("--fallback", default="tuan-870m")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    plan = build_sprint(mode=args.mode, primary=args.primary, fallback=args.fallback,
                        measured_mfu=args.measured_mfu, mfu_floor=args.mfu_floor)
    d = plan.ledger()
    if not d["closes"]:
        print("账本不闭合（不应发生；请检查 preset 或 floor）", file=__import__("sys").stderr)
        return 1
    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2)
          if args.json else plan.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
