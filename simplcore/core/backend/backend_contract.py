# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""自研 CUDA/C++ 后端算子契约 —— SimplCore 跑上非 PyTorch 运行时的**单一事实源**。

背景
----
战役运行时切换为**自研 CUDA·HIP/C++ 后端**（fp32 CPU 参考实现已就绪，
见 ``docs/REF_BACKEND.md``；生产 HIP 后端为 Phase 0 主体工作——目标硬件
MI250/ROCm，融合流式核以 hipcc 编译 ``.cu``），PyTorch 退役出战役路径。
仓库内的 torch 参考实现**保留为冻结的数值规格（oracle）**——它是后端要逐位
对齐的数学定义，不参与训练部署。本文件是后端团队与架构之间的接口契约：

* 必须实现哪些内核（``SIMPLCORE_BACKEND_OPS``）；
* 什么精度策略（bf16 计算 / fp32 累加与优化器状态）；
* 哪些逐位一致性必须达标（FFT↔IIR 双形式、分块状态携带、回滚逐位恢复）；
* 什么**禁止**进入内核（注意力家族 / KV cache / 位置嵌入表）。

用法
----
后端每完成一批内核，跑 ``python -m simplcore.core.backend_contract`` 打印契约
与一致性检查清单，逐项打钩；全部硬项通过才算 Phase 0 验收
（见 ``campaign.py`` 的 ``build_plan`` —— Phase 0 闸门自动从这里生成）。

诚实边界
--------
本文件是**契约**，不是后端实现本身；"后端已就绪"的判定权在这 8 项
一致性检查（``conformance_checks``）上，不在任何口头声明上。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "BackendOp",
    "ConformanceCheck",
    "BackendContract",
    "SIMPLCORE_BACKEND_OPS",
    "DEFAULT_CONTRACT",
    "conformance_checks",
    "main",
]


# ---------------------------------------------------------------------------
# 契约数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendOp:
    """后端必须提供的一个内核。"""

    name: str
    kind: str            # "compute" | "state" | "optimizer" | "collective" | "io"
    shape_contract: str  # 形状/语义约定（紧凑记法）
    dtype_compute: str   # 计算精度（一律 bf16）
    dtype_accum: str     # 累加/统计精度（一律 fp32）
    notes: str = ""


@dataclass(frozen=True)
class ConformanceCheck:
    """后端一致性检查项（Phase 0 验收闸门的来源）。"""

    name: str
    criterion: str
    kind: str = "hard"  # 全部为 hard：任何一项不过，不得开训


@dataclass
class BackendContract:
    """一份完整契约：算子 + 精度 + 逐位一致性 + 禁止项。"""

    name: str
    version: str
    ops: List[BackendOp]
    precision_policy: List[str]
    bitexact_requirements: List[str]
    forbidden: List[str]

    def op_names(self) -> List[str]:
        return [o.name for o in self.ops]

    def ops_by_kind(self, kind: str) -> List[BackendOp]:
        return [o for o in self.ops if o.kind == kind]

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "ops": [
                {"name": o.name, "kind": o.kind, "shape": o.shape_contract,
                 "compute": o.dtype_compute, "accum": o.dtype_accum, "notes": o.notes}
                for o in self.ops
            ],
            "precision_policy": list(self.precision_policy),
            "bitexact_requirements": list(self.bitexact_requirements),
            "forbidden": list(self.forbidden),
        }

    def format(self) -> str:
        lines = [f"=== 后端算子契约 {self.name} v{self.version} ===",
                 f"算子数: {len(self.ops)}"]
        for o in self.ops:
            lines.append(f"  [{o.kind:<10}] {o.name:<28} {o.shape_contract}")
        lines.append("\n-- 精度策略 --")
        for p in self.precision_policy:
            lines.append(f"  · {p}")
        lines.append("\n-- 逐位一致性（硬性）--")
        for b in self.bitexact_requirements:
            lines.append(f"  · {b}")
        lines.append("\n-- 禁止进入内核 --")
        for f_ in self.forbidden:
            lines.append(f"  ✗ {f_}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SimplCore 后端算子清单（与 torch 参考实现逐一对齐）
# ---------------------------------------------------------------------------
SIMPLCORE_BACKEND_OPS: List[BackendOp] = [
    # -- 嵌入与归一化 ---------------------------------------------------------
    BackendOp("emb_lookup", "compute", "V×D 查表, [B,L]->[B,L,D]", "bf16", "fp32",
              "词表 32000 口径；embedding 梯度 scatter-add fp32 累加"),
    BackendOp("rmsnorm", "compute", "[B,L,D]->[B,L,D]", "bf16", "fp32",
              "均值/方差在 fp32 累加后一次性 round 回 bf16（与参考实现同口径）"),
    # -- 线性层与 FFN ---------------------------------------------------------
    BackendOp("gemm", "compute", "[..,D_in]×[D_in,D_out]", "bf16", "fp32",
              "in/out 投影、head、FFN 全部走它；支持 bf16 输入 fp32 输出模式"),
    BackendOp("swiglu", "compute", "[B,L,H]->[B,L,H/2]×sigmoid+silu", "bf16", "fp32",
              "SwiGLU FFN 激活"),
    BackendOp("residual_add", "compute", "[B,L,D]+[B,L,D]", "bf16", "fp32",
              "残差；与 CMT 记忆注入共用（残差流加性项）"),
    # -- mixer 三件套（非注意力核心） ------------------------------------------
    BackendOp("causal_depthwise_conv", "compute", "[B,L,D]×K=4 逐通道因果", "bf16", "fp32",
              "局部路径；左填充 K-1，严禁未来泄漏"),
    BackendOp("spectral_causal_conv", "compute", "[B,L,D] FFT 因果全局卷积", "bf16", "fp32",
              "长程路径训练形式，O(L log L)；或等效直接卷积，须过 C-dual-form 检查"),
    BackendOp("chunked_iir_scan", "compute+state", "[B,C,D]×M 基 二阶 IIR 递推 →y(B,C,M,D)[,coef→long(B,C,D)]", "bf16", "fp32",
              "流式/分块训练核心：y=a·y1+b·y2+sinθ·u；状态 2·M·D/层，跨块携带；\n              coef 融合投影 epilogue（Σ_m y·coef，真实后端融进 scan 输出）"),
    BackendOp("content_gate", "compute", "sigmoid(W_g·norm(x)) 逐元素", "bf16", "fp32",
              "L1 选择门，逐位置瞬时乘法，不进核"),
    BackendOp("evidence_uncertainty_gate", "compute", "前缀能量 cumsum + 分歧统计", "bf16", "fp32",
              "L2/L3 闸门；cumsum 必须 fp32（并行==流式逐位一致的前提）"),
    # -- 输出与损失 -----------------------------------------------------------
    BackendOp("ce_loss", "compute", "[B,L,V] vs labels（错位一格）", "bf16", "fp32",
              "只在目标段计损失（SFT/蒸馏掩码语义）"),
    # -- 流式状态管理 ----------------------------------------------------------
    BackendOp("stream_state_init", "state", "2·M·D/层 + 局部缓冲 K·D/层", "fp32", "fp32",
              "O(1) 常驻状态，与 L 无关；init/copy/serialize 三件套逐位往返"),
    BackendOp("stream_state_serialize", "state", "state ↔ bytes", "fp32", "fp32",
              "checkpoint/恢复与 no-regression 快照的底座"),
    # -- 优化器（预算语义在掩码里） ---------------------------------------------
    BackendOp("adamw_frozen_mask", "optimizer", "参数级冻结掩码", "bf16", "fp32",
              "master/m/v 全 fp32；冻结位零更新（基座冻结/适配器训练的唯一入口）"),
    BackendOp("weight_decay_exempt", "optimizer", "振荡基参数豁免 decay", "bf16", "fp32",
              "对 ρ/θ 基参数不做 weight decay（掐死长程记忆的坑，参考实现已修）"),
    # -- 双 GCD 并行 ----------------------------------------------------------
    BackendOp("shard_allreduce", "collective", "2×GCD FSDP/ZeRO 切分", "bf16", "fp32",
              "MI250 = 2×64 GB；梯度 all-reduce fp32，切分后与单卡数值一致"),
]

DEFAULT_CONTRACT = BackendContract(
    name="simplcore-cuda-backend",
    version="1.0",
    ops=SIMPLCORE_BACKEND_OPS,
    precision_policy=[
        "计算 bf16；GEMM/卷积/归一化统计/cumsum/损失 一律 fp32 累加",
        "优化器 master 权重与 m/v 全 fp32（16 B/参数口径，见 TRAINING_BUDGET_MI250.md §4）",
        "O(1) 流式状态 fp32 常驻；与序列长度无关",
        "误差预算：任一内核 fp32 参考对照 max|Δ| ≤ 1e-2（bf16 计算口径）",
    ],
    bitexact_requirements=[
        "C-dual-form：FFT 并行形式 vs IIR 流式形式，fp32 下 max|Δ| < 4e-5（4M 长度）",
        "C-chunked-carry：分块训练的跨块状态携带 与 整段参考 逐位一致",
        "C-rollback：no-regression gate 回滚后权重逐位恢复（byte-exact 快照）",
        "C-stream-serialize：流式状态 serialize→deserialize 往返逐位一致",
    ],
    forbidden=[
        "softmax(QKᵀ) / 注意力 / KV cache 内核（purity.py 同款红线）",
        "位置嵌入表（位置由因果核编码）",
        "任何注意力家族的融合算子（fused-attention、flash-attention 等）",
    ],
)


def conformance_checks(contract: Optional[BackendContract] = None) -> List[ConformanceCheck]:
    """Phase 0 验收的一致性检查清单（campaign.build_plan 的闸门来源）。"""
    c = contract or DEFAULT_CONTRACT
    n_ops = len(c.ops)
    return [
        ConformanceCheck(
            "P0-backend-conformance",
            f"契约 {c.name} v{c.version} 全部 {n_ops} 个内核有实现并通过单元测试"),
        ConformanceCheck(
            "P0-stream-bitexact",
            c.bitexact_requirements[0].split("：", 1)[-1]),
        ConformanceCheck(
            "P0-chunked-carry",
            c.bitexact_requirements[1].split("：", 1)[-1]),
        ConformanceCheck(
            "P0-frozen-mask",
            "adamw_frozen_mask 冻结位零更新（基座冻结/适配器预算语义的前提）"),
        ConformanceCheck(
            "P0-bf16-accum",
            "抽样内核验证 fp32 累加（GEMM / rmsnorm / cumsum / ce_loss）"),
        ConformanceCheck(
            "P0-shard-2gcd",
            "2 GCD 切分下训练数值与单卡一致（all-reduce fp32）"),
        ConformanceCheck(
            "P0-no-attention",
            "内核清单结构审查：零注意力 / 零 KV cache / 零位置表"),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """CLI：``python -m simplcore.core.backend_contract`` —— 打印契约与检查清单。"""
    ap = argparse.ArgumentParser(description="SimplCore 自研后端算子契约")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args(argv)
    if args.json:
        print(json.dumps(DEFAULT_CONTRACT.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(DEFAULT_CONTRACT.format())
        print("\n-- 一致性检查（Phase 0 验收闸门）--")
        for c in conformance_checks():
            tag = "[硬]" if c.kind == "hard" else "[软]"
            print(f" {tag} {c.name}: {c.criterion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
