from typing import Dict, List, Tuple

import simplcore.tensor as torch


def expand_tensor_to_shape(source_tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """把较小权重按维度平铺到目标形状，再裁剪到精确大小。"""
    if tuple(source_tensor.shape) == tuple(target_shape):
        return source_tensor.clone()
    if source_tensor.dim() != len(target_shape):
        raise ValueError(
            f"rank mismatch: source rank {source_tensor.dim()} target rank {len(target_shape)}"
        )

    expanded = source_tensor
    for dim, target_size in enumerate(target_shape):
        source_size = expanded.shape[dim]
        if source_size < 1:
            raise ValueError(f"cannot expand empty tensor dimension {dim}")
        repeats = [1] * expanded.dim()
        repeats[dim] = (target_size + source_size - 1) // source_size
        expanded = expanded.repeat(*repeats)

    slices = tuple(slice(0, size) for size in target_shape)
    return expanded[slices].clone()


def copy_matching_tensors_strict(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> List[str]:
    copied: List[str] = []
    for name, source_tensor in source_state.items():
        if name not in target_state:
            continue
        target_tensor = target_state[name]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"shape mismatch for {name}: "
                f"source {tuple(source_tensor.shape)} target {tuple(target_tensor.shape)}"
            )
        target_tensor.copy_(source_tensor)
        copied.append(name)
    return copied


def copy_or_expand_tensors(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """复制同形权重；形状变大时使用平铺扩展，保留可追踪策略名称。"""
    strategies: Dict[str, str] = {}
    for name, source_tensor in source_state.items():
        if name not in target_state:
            continue
        target_tensor = target_state[name]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_tensor.copy_(source_tensor)
            strategies[name] = "copy_exact"
            continue
        expanded = expand_tensor_to_shape(source_tensor, tuple(target_tensor.shape))
        target_tensor.copy_(expanded.to(dtype=target_tensor.dtype, device=target_tensor.device))
        strategies[name] = "tile_expand"
    return strategies
