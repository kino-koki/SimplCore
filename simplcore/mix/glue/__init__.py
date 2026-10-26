# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 模型粘合 (Model Glue) — 修复版

原始设计意图:
  1. 训练多个小模型 (150M级别)，每个在不同数据/任务上专精
  2. 将每个小模型的FFN层提取为"专家"
  3. 共享基础层 (embedding + attention + norm + output)
  4. 通过路由网络动态选择top-K专家激活

架构:
  Input
    ↓
  [Shared Embedding]
    ↓
  [Shared Attention Layers]
    ↓
  [Expert Router] → [Expert 0] [Expert 1] ... [Expert N-1]
    ↓                    ↓         ↓             ↓
  [Weighted Sum] ←  weights   weights        weights
    ↓
  [Shared Output]
    ↓
  Logits

修复说明 (2026-08-02):
  本模块此前 import 了不存在的 config / expert_model / glue_model / trainer，
  导致 `import simplcore.mix.glue` 直接 ModuleNotFoundError（堆叠命门 0）。
  这些文件从未被实现。真实的"多模型合并"机制已由
  `simplcore.mix.stacking.stacker.structural_stack` 承接并落地:
    - 多源 150M checkpoint 结构感知堆叠到目标模型
    - expert 槽位交错映射 + router 权重同步扩展
    - 对称破缺噪声 + 逐参数策略报告
  本模块现在转导出 stacking 的真实实现，保证 `import simplcore.mix.glue`
  可用，并保留原始架构意图文档。
"""
from simplcore.mix.stacking.mapper import (
    copy_matching_tensors_strict,
    copy_or_expand_tensors,
    expand_tensor_to_shape,
)
from simplcore.mix.stacking.manifest import SourceCheckpoint, StackManifest
from simplcore.mix.stacking.stacker import StackReport, structural_stack
from simplcore.mix.stacking.build_targets import build_target_config

__all__ = [
    # 原始 glue 概念（现由 stacking 承接）
    "structural_stack",
    "StackReport",
    "SourceCheckpoint",
    "StackManifest",
    "build_target_config",
    # 形状工具（保留向后兼容）
    "copy_matching_tensors_strict",
    "copy_or_expand_tensors",
    "expand_tensor_to_shape",
]
