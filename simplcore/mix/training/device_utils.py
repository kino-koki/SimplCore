"""设备抽象层：AMD 680M (DirectML) 与 NVIDIA V100 (CUDA) 统一入口。

设计目标：
  同一份训练代码，在 680M 核显和 V100 集群上都能跑。
  设备选择优先级：CUDA（V100）→ DirectML（AMD 680M）→ 明确失败（不回退 CPU）。

  - ``resolve_train_device()``: 自动探测并返回 (device, backend_name, is_distributed)
  - ``empty_device_cache()``: 按后端清理显存缓存
  - ``tensor_to_cpu()`` / ``move_to_device()``: 统一 checkpoint 保存/加载
  - ``save_checkpoint_tensors()``: 把 state_dict 里的所有张量搬 CPU（两种后端通用）

用法：
    from simplcore.mix.training.device_utils import resolve_train_device
    device, backend, is_ddp = resolve_train_device()
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import simplcore.tensor as torch


# torch_directml 已替换为自研引擎（CPU模式）：DirectML 不可用时回退到 CPU 语义
class _DirectMLCompat:
    def device(self, idx=0): return 'cpu'
    def device_count(self): return 0
    def empty_cache(self): pass


def _try_import_directml():
    try:
        torch_directml = _DirectMLCompat()
        return torch_directml
    except ImportError:
        return None


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _directml_available() -> bool:
    dml = _try_import_directml()
    if dml is None:
        return False
    try:
        dev = dml.device()
        return str(dev) != "cpu"
    except Exception:
        return False


def resolve_train_device(
    *,
    prefer_cuda: bool = True,
) -> Tuple[torch.device, str, bool]:
    """返回 (device, backend_name, is_distributed)。

    backend_name: "cuda" | "directml" | "cpu"
    is_distributed: 仅 CUDA 且 WORLD_SIZE>1 时为 True
    """
    # 1) CUDA (V100)
    if _cuda_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        world = int(os.environ.get("WORLD_SIZE", 1))
        is_ddp = world > 1
        return device, "cuda", is_ddp

    # 2) DirectML (AMD 680M 核显)
    if _directml_available():
        dml = _try_import_directml()
        device = dml.device()
        return device, "directml", False

    # 3) 无 GPU: 允许单卡调试（模型很小），但打印醒目警告
    if os.environ.get("SIMPLCORE_ALLOW_CPU", "0") == "1":
        return torch.device("cpu"), "cpu", False

    raise RuntimeError(
        "未检测到可用 GPU。V100 请装 CUDA 版 torch；AMD 680M 请装 torch_directml。"
        "如需 CPU 调试（仅小模型），设 SIMPLCORE_ALLOW_CPU=1。"
    )


def empty_device_cache(backend: str) -> None:
    """按后端清理显存缓存。"""
    if backend == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif backend == "directml":
        dml = _try_import_directml()
        if dml is not None:
            try:
                dml.empty_cache()
            except Exception:
                pass
    # cpu: 无操作


def tensor_to_cpu(t: torch.Tensor) -> torch.Tensor:
    """张量搬 CPU；meta/非张量安全。"""
    if isinstance(t, torch.Tensor):
        if t.device.type == "meta":
            return t
        return t.detach().cpu()
    return t


def state_dict_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    """state_dict 全部张量搬 CPU（DirectML/CUDA 通用）。"""
    return {k: tensor_to_cpu(v) for k, v in state.items()}


def state_dict_to_device(
    state: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """state_dict 全部张量搬到目标设备。"""
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """模型搬到目标设备（兼容 DirectML 的 device 对象）。"""
    return model.to(device)


def describe_device(device: torch.device, backend: str) -> str:
    if backend == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        return f"cuda[{name}]"
    return str(device)
