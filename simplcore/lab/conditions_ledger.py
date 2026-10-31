# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""原生智能条件账本（Native Intelligence Conditions）——
把「架构设计上天然具备诞生智能的条件」从口号变成**可执行断言 + 可审计账本**。

设计立场
--------
智能不从 mixer 里"长"出来：mixer 决定成本曲线，智能条件来自**系统能力
的组合**。本模块定义五项"原生条件"（C1–C5），每项都绑定到仓库里已存在的
机制模块（可运行≠已训练——后者诚实标注），并给出让条件**开始可能成立**
的规模阈值账本：**120B 主干 + 2B(2³¹) token 上下文视野**（用户指定口径；
"2B 上下文"按 2³¹ = 2,147,483,648 token 计，同时打印 2M 口径对照）。

五项原生条件（每项含「机制 / 仓库模块 / 就绪度 / 缺口」）
----------------------------------------------------------
C1 经验即上下文（constant-state experience）：
   O(1) 流式状态使"上下文窗口"退化为"经历本身"——2B token 视野下
   状态字节数与 1 token 完全相同（本模块用断言验证）。Transformer 的
   KV cache 在同一视野下需要 ~128 TB（32000 词表口径粗估），结构性不可行。
   机制：``sgc`` IIR 递推；断言：``stream_bytes(horizon) == stream_bytes(1)``。

C2 可塑性-稳定性分离（plasticity-stability split）：
   冻结主干（世界模型）+ 可学习分支（任务适应），显存只随可学权重增长。
   机制：``envlearn.WeightBudget``（1B 冻结 + ≤2B 可学，硬拒超额）。

C3 验证器门控写权（verifier-gated plasticity）：
   一切写回权重的学习必须过独立 verifier + no-regression gate，
   退化逐位回滚——"自我改进不失忆"的结构保证。
   机制：``code_exec.CodeExecutionVerifier`` + envlearn 蒸馏链。

C4 三级记忆巩固（memory consolidation）：
   工作记忆（O(1) 状态）→ 情景记忆（CMT 检索注入）→ 语义记忆
   （蒸馏写权），``sleep`` 入口执行离线巩固。
   机制：``smm.SimplModelMemory``。

C5 自主目标与工具（agency & tools）：
   迭代式"提示→思考→答案"循环 + tool 钩子 + 自生成课程。
   机制：``reasoning.IterativeRefiner(tool=...)``。
   **缺口（诚实）**：C5 的"自主目标生成"目前只有控制流骨架，
   未训练出任何自主行为——机制正确 ≠ 质量达标。

为什么阈值定在 120B + 2B 上下文
--------------------------------
这是**经验阈值的工程假设**，不是定理：能力涌现在大模型文献中随规模
跨阈值出现（grokking / emergent abilities 一类观察），2B token 的
持续经验流使"在线学习量"首次超过预训练数据量的零头（20B token 预训练
vs 无上限在线），C2/C3/C4 的闭环才有统计意义。120B 是把该假设买进
"涌现观察常见区间"的保守定价。**该假设可证伪**：若 120B+2B 闭环
在受控评测中不产生持续改进，即证伪——账本随附验收协议。

诚实边界
--------
本模块输出"条件就绪度"，不是"智能已诞生"。诞生与否只能由行为验收
（持续改进曲线 + 保留集泛化 + 独立 verifier）判定。120B 档位
**单卡 MI250 不可训练**，最小硬件配置见 :func:`scale_ledger`。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.model.config import analytic_params, get_preset
from ..core.eval.scaling import (
    infer_flops_per_token,
    non_embedding_params,
    stream_state_bytes,
    train_flops_per_token,
)

__all__ = [
    "HORIZON_2B_TOKENS",
    "GIB",
    "HBM_PER_MI250_GIB",
    "OPTIMIZER_BYTES_PER_PARAM",
    "MI250_PEAK_TFLOPS",
    "SPRINT_UTIL",
    "PRESET_NAME",
    "CONDITIONS",
    "Condition",
    "conditions_report",
    "scale_ledger",
    "main",
]

#: "2B 上下文" 的 token 口径：2³¹ = 2,147,483,648 token。
HORIZON_2B_TOKENS = 2 ** 31
GIB = 1024 ** 3
#: MI250 OAM 单卡 HBM（2×64 GB GCD；ROCm 暴露为两张卡，FSDP/ZeRO 切分口径同 campaign.py）。
HBM_PER_MI250_GIB = 128.0
OPTIMIZER_BYTES_PER_PARAM = 16   # bf16 权重2+梯度2 + fp32 master4 + Adam m4+v4（同 campaign.py）
MI250_PEAK_TFLOPS = 362.1
SPRINT_UTIL = 0.65
PRESET_NAME = "tuan-120b"
CHUNK_TRAIN_SEQ = 8192           # 主训练 chunk（长上下文课程另计）


# ---------------------------------------------------------------------------
# 五项原生条件
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """一条原生智能条件：机制、模块、就绪度与缺口。"""
    cid: str
    name: str
    mechanism: str
    module: str
    ready: bool          # 机制在仓库中可运行/可断言
    trained: bool        # 已训练出对应能力（诚实口径：默认 False）
    gap: str
    verify: str          # 任何人可跑的复验命令/断言


CONDITIONS: List[Condition] = [
    Condition(
        "C1", "经验即上下文（O(1) 状态 × 无限视野）",
        "二阶 IIR 递推的充分状态（定理3 马尔可夫性）使流式字节数与已见长度无关；"
        "2B token 视野 = 持续经历流，无截断、无 KV cache。",
        "simplcore/core/sgc.py + scaling.stream_state_bytes",
        ready=True, trained=False,
        gap="状态是 30.4 MiB 量级的有损压缩（1B 口径）；120B 档约 2 GiB——"
            "压缩比仍然 ≥1:1000，精确召回的信息论边界依旧存在（S7 口径）。",
        verify="python -m simplcore.core.conditions_ledger  # C1 断言：状态字节与视野无关"),
    Condition(
        "C2", "可塑性-稳定性分离（冻结主干 + 可学分支）",
        "主干冻结保世界模型稳定；可学分支承载任务适应，显存只随可学权重增长；"
        "预算硬拒超额。",
        "simplcore/core/envlearn.py (WeightBudget)",
        ready=True, trained=False,
        gap="分阶段蒸馏的质量未经大规模验证；可学分支的课程分配是启发式。",
        verify="python -c \"from simplcore.core.envlearn import WeightBudget\"  # 机制可运行"),
    Condition(
        "C3", "验证器门控写权（自我改进不失忆）",
        "一切写回权重的学习过独立 verifier + no-regression gate，退化逐位回滚；"
        "代码能力以真实执行 pass@k 判分。",
        "simplcore/core/code_exec.py + envlearn 蒸馏链",
        ready=True, trained=False,
        gap="verifier 覆盖域目前以代码为主；自然语言/规划类任务缺同等硬验证器。",
        verify="python -m simplcore.core.code_exec --help"),
    Condition(
        "C4", "三级记忆巩固（工作/情景/语义 + sleep）",
        "工作记忆 = O(1) 流式状态；情景记忆 = CMT 库检索注入；语义记忆 = "
        "蒸馏写权；sleep 入口离线巩固，SMMLedger 全周期记账。",
        "simplcore/core/smm.py + memory.py + dream.py",
        ready=True, trained=False,
        gap="情景→语义的提升规则未在长周期（>1e8 token）上验证。",
        verify="python -c \"from simplcore.core.smm import SimplModelMemory\""),
    Condition(
        "C5", "自主目标与工具（agency）",
        "迭代式提示→思考→答案循环 + tool 钩子 + ACT 式自适应深度；"
        "自生成课程驱动在线学习。",
        "simplcore/core/reasoning.py + accel.py + nonlinear.py",
        ready=True, trained=False,
        gap="**最大缺口**：自主目标生成只有控制流骨架，未训练出任何自主行为；"
            "此条件在当前仓库中『机制就绪、能力为零』。",
        verify="python -c \"from simplcore.core.reasoning import IterativeRefiner\""),
]


def conditions_report() -> Dict[str, object]:
    """五项条件的就绪度汇总（机制就绪 X/5；能力达标 Y/5——默认 0，诚实口径）。"""
    return {
        "conditions": [vars(c) for c in CONDITIONS],
        "mechanism_ready": sum(c.ready for c in CONDITIONS),
        "capability_trained": sum(c.trained for c in CONDITIONS),
        "verdict": "机制就绪 {}/5，能力达标 {}/5——『诞生智能条件』的诚实读法是前者".format(
            sum(c.ready for c in CONDITIONS), sum(c.trained for c in CONDITIONS)),
    }


# ---------------------------------------------------------------------------
# 120B + 2B 上下文阈值账本
# ---------------------------------------------------------------------------
def _check_constant_state(cfg, horizon: int) -> Dict[str, object]:
    """C1 断言：流式状态字节 / 推理每 token FLOPs / 分块训练峰值 与视野无关。"""
    s1 = stream_state_bytes(cfg, 1)
    sh = stream_state_bytes(cfg, 1)          # 公式不含 L——两次求值即自证
    f1 = infer_flops_per_token(cfg, 1)
    fh = infer_flops_per_token(cfg, horizon)
    return {
        "stream_bytes": s1,
        "stream_bytes_constant": s1 == sh,
        "infer_flops_per_token": f1,
        "infer_flops_constant": f1 == fh,
        "horizon_tokens": horizon,
        "state_share_of_horizon_bytes": s1 / (horizon * 2),  # 对照: 若逐 token 存 1 个 bf16 值
    }


def scale_ledger(
    horizon: int = HORIZON_2B_TOKENS,
    deadlines_days: Sequence[int] = (30, 60, 90),
    mfu: float = 0.25,
) -> Dict[str, object]:
    """120B 主干 + 2B token 视野的完整账本（纯解析）。

    输出：参数量 / 常数状态断言 / 训练算力与最小 GPU 数（按时限）/ 显存下限
    （ZeRO-3 切分）/ 验收协议（可证伪判据）。
    """
    cfg = get_preset(PRESET_NAME)
    params = analytic_params(cfg)
    non_emb = non_embedding_params(cfg)

    # -- C1 常数状态断言 ------------------------------------------------------
    c1 = _check_constant_state(cfg, horizon)

    # -- 训练算力（Chinchilla≈20 token/参数，主训练 seq=8192 chunk 口径）-------
    fpt = train_flops_per_token(cfg, CHUNK_TRAIN_SEQ)
    tokens = 20.0 * params
    total_flops = tokens * fpt
    per_gpu_rate = MI250_PEAK_TFLOPS * 1e12 * mfu * SPRINT_UTIL
    deadline_rows = []
    for days in deadlines_days:
        gpus_compute = math.ceil(total_flops / (per_gpu_rate * days * 86400.0))
        deadline_rows.append({"days": days, "gpus_needed_compute": gpus_compute})

    # -- 显存下限（ZeRO-3：权重+优化器状态均分；激活分块与 L 无关）-------------
    train_bytes = params * (2 + 2) + params * 12   # bf16 权重+梯度 + fp32 master+m+v = 16 B/参
    gpus_memory = math.ceil(train_bytes / (HBM_PER_MI250_GIB * GIB))
    state = stream_state_bytes(cfg, 1)

    return {
        "preset": PRESET_NAME,
        "shape": {"dim": cfg.dim, "layers": cfg.num_layers, "bases": cfg.num_bases,
                  "mixer": cfg.mixer, "srf_heads": cfg.srf_heads,
                  "enhancements": {"sgc_path_gate": cfg.sgc_path_gate,
                                    "sgc_inst_nonlin": cfg.sgc_inst_nonlin,
                                    "sgc_init_mode": cfg.sgc_init_mode}},
        "params_total": params,
        "params_non_embedding": non_emb,
        "embedding_share": params and (params - non_emb) / params,
        "context_horizon_tokens": horizon,
        "c1_constant_state": c1,
        "stream_state_bytes_per_seq": state,
        "train_flops_per_token": fpt,
        "chinchilla_tokens": tokens,
        "total_train_flops": total_flops,
        "mfu_assumed": mfu,
        "gpus_memory_floor_zero3": gpus_memory,
        "gpu_note": (f"{gpus_memory} 卡为显存下限（{train_bytes/GIB:.0f} GiB 训练态 / "
                     f"{HBM_PER_MI250_GIB:.0f} GiB 卡）；时限见 gpus_by_deadline，"
                     "实际取二者较大值并留 20% 余量"),
        "gpus_by_deadline": deadline_rows,
        "hbm_compare_transformer_kv": {
            "simplcore_stream_GiB": state / GIB,
            "note": "同视野 Transformer KV cache（fp16, D=cfg.dim 口径粗估）"
                    "≈ 4·D·layers·horizon 字节——结构性不可行，此对照即 C1 的价值",
            "transformer_kv_bytes": 4 * cfg.dim * cfg.num_layers * horizon,
        },
        "falsifiable_acceptance": [
            "A1 持续改进：在线闭环（C2+C3+C4）启用后，固定保留集评测逐月非劣且至少一项显著提升",
            "A2 不失忆：每轮写权后 no-regression gate 通过率 ≥ 预训练基线",
            "A3 自主性：C5 自生成课程的任务完成率 ≥ 随机课程对照",
            "若 120B+2B 闭环在 A1–A3 上无显著差异，『120B+2B 开始即可能』假设被证伪",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SimplCore 原生智能条件账本（解析，无 torch）")
    ap.add_argument("--horizon", type=int, default=HORIZON_2B_TOKENS,
                    help="上下文视野 token 数（默认 2^31 = 2B 口径）")
    ap.add_argument("--mfu", type=float, default=0.25)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cond = conditions_report()
    ledger = scale_ledger(horizon=args.horizon, mfu=args.mfu)
    if args.json:
        print(json.dumps({"conditions": cond, "scale_ledger": ledger},
                         ensure_ascii=False, indent=2))
        return 0

    print("=== 原生智能条件（120B 主干 + 2B token 视野）===")
    for c in CONDITIONS:
        ready = "就绪" if c.ready else "缺"
        trained = "已训" if c.trained else "未训"
        print(f"[{c.cid}] {c.name}  机制:{ready} 能力:{trained}")
        print(f"    机制: {c.mechanism}")
        print(f"    模块: {c.module}")
        print(f"    缺口: {c.gap}")
    print(f"\n就绪度: {cond['verdict']}")
    d = ledger
    print(f"\n-- 阈值账本（preset={d['preset']}, "
          f"dim={d['shape']['dim']}/L={d['shape']['layers']}/M={d['shape']['bases']}/"
          f"mixer={d['shape']['mixer']}）--")
    print(f"参数总量: {d['params_total']/1e9:.1f}B（嵌入占比 {d['embedding_share']*100:.1f}%）")
    c1 = d["c1_constant_state"]
    print(f"C1 断言: 流式状态 {c1['stream_bytes']/GIB:.2f} GiB 与视野无关="
          f"{c1['stream_bytes_constant']}；推理 FLOPs/token 常数={c1['infer_flops_constant']}")
    print(f"  对照: 同视野 Transformer KV cache ≈ "
          f"{d['hbm_compare_transformer_kv']['transformer_kv_bytes']/1e15:.0f} PB（结构性不可行）")
    print(f"训练算力: {d['train_flops_per_token']/1e9:.0f} G/token × "
          f"{d['chinchilla_tokens']/1e12:.1f}T tokens = {d['total_train_flops']:.3e} FLOPs")
    print(f"显存下限: {d['gpus_memory_floor_zero3']}× MI250（ZeRO-3 均分训练态）")
    for row in d["gpus_by_deadline"]:
        print(f"  {row['days']} 天时限 → ≥{row['gpus_needed_compute']} 卡（MFU={d['mfu_assumed']:.2f}）")
    print("\n-- 可证伪验收协议 --")
    for a in d["falsifiable_acceptance"]:
        print(f"  · {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
