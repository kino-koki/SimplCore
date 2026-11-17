# 必须在任何子模块导入 torch / tempfile 之前执行：C 盘容量紧张，
# 缓存一律写非系统盘（自动选仓库同盘等，不依赖固定盘符）。
# 详见 simplcore/_cache_env.py 里的原因说明。
from . import _cache_env as _cache_env

_cache_env.apply()

__author__ = "青奈科技青小团团队"
__version__ = "0.1-α"
PROJECT_SUMMARY = "SimplCore: a research prototype for attention-free sequence modeling."

__all__ = [
    "__author__",
    "__version__",
    "PROJECT_SUMMARY",
    "cache_root",
    "report_environment",
]


def cache_root() -> str:
    """返回当前仓库强制使用的缓存根目录。"""
    return _cache_env.cache_root()


def report_environment() -> dict[str, object]:
    """收集最小开发诊断信息，便于快速确认本机环境。"""
    info: dict[str, object] = {
        "project": "SimplCore",
        "version": __version__,
        "author": __author__,
        "cache_root": cache_root(),
    }

    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception as exc:  # pragma: no cover - does not affect import path
        info["numpy_error"] = str(exc)

    try:
        import torch
        info["torch_version"] = torch.__version__
    except Exception as exc:  # pragma: no cover - does not affect import path
        info["torch_error"] = str(exc)

    return info

