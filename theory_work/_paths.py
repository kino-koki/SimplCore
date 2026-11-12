# theory_work 脚本的共享路径解析。
#
# superposition_probe 的原始结果目录与统一落盘目录原本位于外部工作区
# /data/workspace（Linux 训练机）。为让本目录的定理脚本在任意机器可运行，
# 工作区根可通过环境变量 SIMPLCORE_THEORY_WORKSPACE 覆盖；默认值保持
# 原路径，行为与历史运行一致。
from __future__ import annotations

import os

DEFAULT_WORKSPACE = "/data/workspace"


def workspace() -> str:
    """工作区根目录：环境变量 SIMPLCORE_THEORY_WORKSPACE 优先，否则原默认。"""
    return os.environ.get("SIMPLCORE_THEORY_WORKSPACE", DEFAULT_WORKSPACE)


def probe_results() -> str:
    """superposition_probe 原始结果目录（定理脚本的只读输入）。"""
    return workspace() + "/train/superposition_probe/results"


def out_dir() -> str:
    """theory_work/results 统一落盘目录。"""
    return workspace() + "/theory_work/results"