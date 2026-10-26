# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""结构感知多模型堆叠器（Structural Stacker）v2。

修复命门 0 / 1 / 2：
- 命门 0: 真正的"多个 150M checkpoint → 目标模型"的可执行合并逻辑。
- 命门 1: 弃用盲目 `tile_expand` 做初始化；结构感知扩展 + 对称破缺噪声。
- 命门 2: expert 槽位交错映射 + router 权重按槽位同步扩展。

v2 关键修正（由测试抓出）：
  state_dict 中 MoE expert 是 `...experts.N.xxx` 每专家一个 key。
  因此 expert 槽位映射是**跨 key** 的（source 的 experts.N → target 的 experts.M），
  不能按 key 一一对应。本版重构为:
    1) 收集各 source 的 expert 块 {prefix: {idx: {suffix: tensor}}}
    2) 按槽位映射把 source expert 权重复制到 target expert 槽位
    3) 对 expert 权重剩余维度做 dim 扩展 + 对称破缺噪声

设计原则：
  1. 只堆叠"参数"（model.named_parameters），buffers（RoPE 频率等）保持目标
     模型原有初始化。
  2. 每个目标张量选择策略后记录进报告，堆叠全程可诊断、可追溯。
  3. 无法可靠映射的槽位**保留目标随机初始化**并在报告中标注 fallback。
  4. 噪声用调用方 `torch.manual_seed(seed)` 控制，保证可复现。

用法：
    torch.manual_seed(seed)
    result = structural_stack(
        source_states=[ckpt1_state, ckpt2_state, ...],
        target_model=target_model,   # 已实例化（meta/cpu）的目标模型
        noise_std=0.02,
        expert_mode="interleave",
    )
    target_model.load_state_dict(result["state_dict"])
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import simplcore.tensor as torch

# 匹配 "...experts.N" 或 "...experts.N.xxx"（N 为专家索引）
_EXPERT_KEY_RE = re.compile(r"^(.+\.experts\.)(\d+)(\..+)?$")

_ROUTER_SUFFIXES = ("expert_prototypes", "prototype_scale", "route_bias")


@dataclass
class StackReport:
    """每个参数采用的堆叠策略。"""
    strategies: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategies": self.strategies,
            "warnings": self.warnings,
        }


def split_expert_key(key: str) -> Optional[Tuple[str, int, str]]:
    """拆分 expert key -> (prefix, expert_idx, suffix)。

    "recurrent.moe.experts.5.gate.weight" -> ("recurrent.moe.experts.", 5, "gate.weight")
    非 expert key 返回 None。
    """
    m = _EXPERT_KEY_RE.match(key)
    if not m:
        return None
    return m.group(1), int(m.group(2)), (m.group(3) or "")[1:]


def dimensional_expand(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
    *,
    noise_std: float = 0.02,
) -> Tuple[torch.Tensor, List[int]]:
    """按维度扩展张量到目标形状 + 对称破缺噪声。

    与旧 `mapper.expand_tensor_to_shape` 的区别：
      - 扩展后加高斯噪声打破周期性 aliasing（修复命门 1）
      - 返回"哪些维度扩展过"以便报告
    """
    src = tensor
    if tuple(src.shape) == tuple(target_shape):
        return src.clone(), []
    if src.dim() != len(target_shape):
        raise ValueError(
            f"rank mismatch: source {tuple(src.shape)} target {target_shape}"
        )

    expanded_dims: List[int] = []
    for dim, target_size in enumerate(target_shape):
        source_size = src.shape[dim]
        if source_size == target_size:
            continue
        if source_size < 1:
            raise ValueError(f"cannot expand empty dimension {dim}")
        expanded_dims.append(dim)
        repeats = [1] * src.dim()
        repeats[dim] = (target_size + source_size - 1) // source_size
        src = src.repeat(*repeats)

    slices = tuple(slice(0, size) for size in target_shape)
    out = src[slices].clone()

    # 对称破缺：扩展过的维度加小高斯噪声
    if expanded_dims and noise_std > 0:
        out = out + torch.randn_like(out) * noise_std

    return out, expanded_dims


def expert_slot_map(
    n_sources: int,
    src_experts: int,
    dst_experts: int,
    mode: str,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """生成 expert 槽位映射表: [(source_idx, expert_idx) or (None,None), ...]。

    返回长度恒为 dst_experts；(None, None) 表示保留目标随机初始化。
    mode="interleave": 多 source 交错填充（槽位 = s + e*n_sources）
    mode="tile":       单 source 循环重复
    """
    mapping: List[Tuple[Optional[int], Optional[int]]] = []
    if mode == "tile":
        return [(0, dst % src_experts) for dst in range(dst_experts)]
    for dst in range(dst_experts):
        s = dst % n_sources
        e = (dst // n_sources) % src_experts
        mapping.append((s, e))
    return mapping


def _collect_expert_blocks(
    state: Dict[str, torch.Tensor],
) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
    """收集 state_dict 中的 expert 块。

    {prefix: {expert_idx: {suffix: tensor}}}
    """
    blocks: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    for key, val in state.items():
        split = split_expert_key(key)
        if split is None:
            continue
        prefix, idx, suffix = split
        blocks.setdefault(prefix, {}).setdefault(idx, {})[suffix] = val
    return blocks


def _stack_expert_slot(
    prefix: str,
    dst_idx: int,
    suffix: str,
    target_shape: Tuple[int, ...],
    mapping: List[Tuple[Optional[int], Optional[int]]],
    source_states: List[Dict[str, torch.Tensor]],
    noise_std: float,
) -> Tuple[Optional[torch.Tensor], str]:
    """堆叠单个 target expert 槽位（跨 key 映射）。"""
    s, e = mapping[dst_idx]
    if s is None:
        return None, "expert_random_keep"
    src_key = f"{prefix}{e}" if not suffix else f"{prefix}{e}.{suffix}"
    src_t = source_states[s].get(src_key)
    if src_t is None:
        return None, "expert_missing"
    out, expanded = dimensional_expand(src_t, target_shape, noise_std=noise_std)
    return out, ("expert_expand" if expanded else "expert_copy")


def _stack_router(
    source_tensors: List[torch.Tensor],
    target_shape: Tuple[int, ...],
    n_sources: int,
    expert_mode: str,
    noise_std: float,
) -> Tuple[torch.Tensor, str]:
    """router 权重（expert_prototypes/scale/bias）：第一维槽位映射 + 其余维度扩展。"""
    src_n = source_tensors[0].shape[0]
    dst_n = target_shape[0]
    mapping = expert_slot_map(n_sources, src_n, dst_n, expert_mode)
    rows = []
    for dst in range(dst_n):
        s, e = mapping[dst]
        if s is None:
            rows.append(torch.zeros(source_tensors[0].shape[1:], dtype=source_tensors[0].dtype))
        else:
            rows.append(source_tensors[s][e])
    sel = torch.stack(rows, dim=0)  # (dst_n, *rest)
    out, expanded = dimensional_expand(sel, target_shape, noise_std=noise_std)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out, f"router_{expert_mode}"


def _stack_single_tensor(
    key: str,
    source_tensors: List[torch.Tensor],
    target_shape: Tuple[int, ...],
    *,
    noise_std: float,
    n_sources: int,
) -> Tuple[torch.Tensor, str]:
    """非 expert、非 router 权重：多源平均或通用维度扩展。"""
    # 同形张量：多源平均（保留共同知识）
    if all(tuple(t.shape) == tuple(target_shape) for t in source_tensors):
        stacked = torch.stack([t.float() for t in source_tensors], dim=0).mean(dim=0)
        return stacked, "mean_sources"
    # 通用维度扩展（注意力 / FFN / norm / embedding）
    out, expanded = dimensional_expand(source_tensors[0], target_shape, noise_std=noise_std)
    return out, ("dim_expand" if expanded else "copy_exact")


def structural_stack(
    source_states: List[Dict[str, torch.Tensor]],
    target_model: torch.nn.Module,
    *,
    noise_std: float = 0.02,
    expert_mode: str = "interleave",
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """结构感知多模型堆叠主入口。

    Args:
        source_states: 多个 checkpoint 的 state_dict（每个一个 dict）。
        target_model: 目标模型（7B 配置），meta/cpu 实例化皆可。
        noise_std: 对称破缺噪声标准差。0 = 不加噪声（诊断用）。
        expert_mode: "interleave"（多源交错）或 "tile"（单源循环）。
        device: 结果张量设备（默认 target_model 参数设备）。

    Returns:
        {"state_dict": 填好的目标 state_dict, "report": StackReport}
    """
    if not source_states:
        raise ValueError("source_states 不能为空")
    if expert_mode not in ("interleave", "tile"):
        raise ValueError(f"expert_mode 必须是 interleave 或 tile，got {expert_mode}")

    n_sources = len(source_states)
    target_state = dict(target_model.state_dict())
    param_names = {name for name, _ in target_model.named_parameters()}
    report = StackReport()
    result_state: Dict[str, torch.Tensor] = {}

    # 自研后端 state_dict 返回原始 ndarray；统一包装为 Tensor 以保证
    # .device / .to() / .repeat() 等接口一致（与 torch 后端行为对齐）。
    def _to_tensor(v):
        import numpy as np
        if isinstance(v, torch.Tensor):
            return v
        return torch.tensor(np.asarray(v))
    target_state = {k: _to_tensor(v) for k, v in target_state.items()}
    source_states = [{k: _to_tensor(v) for k, v in s.items()} for s in source_states]

    # 收集各 source 的 expert 块（跨 key 映射用）
    src_expert_blocks = [_collect_expert_blocks(s) for s in source_states]

    # 预计算各 prefix 的 source / target expert 总数（槽位映射用）
    src_expert_counts: Dict[str, int] = {}
    for block in src_expert_blocks:
        for prefix, idxs in block.items():
            src_expert_counts[prefix] = max(
                src_expert_counts.get(prefix, 0), max(idxs) + 1
            )
    tgt_expert_counts: Dict[str, int] = {}
    for key in target_state:
        sp = split_expert_key(key)
        if sp is not None:
            prefix, idx, _ = sp
            tgt_expert_counts[prefix] = max(tgt_expert_counts.get(prefix, 0), idx + 1)

    for key, target_tensor in target_state.items():
        # buffers 保持目标模型原有初始化
        if key not in param_names:
            result_state[key] = target_tensor
            continue

        target_shape = tuple(target_tensor.shape)
        split = split_expert_key(key)
        dst_device = target_tensor.device

        if split is not None:
            # ── expert 权重：跨 key 槽位映射 ──
            prefix, dst_idx, suffix = split
            mapping = expert_slot_map(
                n_sources,
                src_expert_counts.get(prefix, 0),
                tgt_expert_counts.get(prefix, 0),
                expert_mode,
            )
            filled, strat = _stack_expert_slot(
                prefix, dst_idx, suffix, target_shape, mapping,
                source_states, noise_std,
            )
            if filled is None:
                report.warnings.append(f"{key}: 无匹配 source expert，保留目标随机初始化")
                result_state[key] = target_tensor
            else:
                result_state[key] = filled.to(
                    dtype=target_tensor.dtype, device=dst_device
                )
            report.strategies[key] = strat
            continue

        # ── router 权重（expert_prototypes/scale/bias）──
        if any(t in key for t in _ROUTER_SUFFIXES):
            src_tensors = [s[key] for s in source_states if key in s]
            if not src_tensors:
                report.warnings.append(f"{key}: 所有 source 均无此参数，保留目标随机初始化")
                result_state[key] = target_tensor
                report.strategies[key] = "random_keep"
                continue
            filled, strat = _stack_router(
                src_tensors, target_shape, n_sources, expert_mode, noise_std
            )
            result_state[key] = filled.to(dtype=target_tensor.dtype, device=dst_device)
            report.strategies[key] = strat
            continue

        # ── 普通参数 ──
        src_tensors = [s[key] for s in source_states if key in s]
        if not src_tensors:
            report.warnings.append(f"{key}: 所有 source 均无此参数，保留目标随机初始化")
            result_state[key] = target_tensor
            report.strategies[key] = "random_keep"
            continue
        filled, strat = _stack_single_tensor(
            key, src_tensors, target_shape, noise_std=noise_std, n_sources=n_sources
        )
        result_state[key] = filled.to(dtype=target_tensor.dtype, device=dst_device)
        report.strategies[key] = strat

    return {"state_dict": result_state, "report": report}
