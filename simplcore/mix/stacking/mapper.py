from typing import Dict, List

import torch


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
