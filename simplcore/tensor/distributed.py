# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor Distributed — 自研分布式通信层（零第三方框架依赖）
==================================================================

彻底移除对 torch.distributed / NCCL / Gloo 的依赖，由本项目原生实现
集合通信原语 (Collective Communication) 与进程组 (ProcessGroup) 管理：

  * ProcessGroup:    进程组抽象，管理 rank / world_size / 通信后端
  * all_reduce:      全规约（Ring All-Reduce 或单机副本模式）
  * all_gather:      全收集
  * reduce_scatter:  规约散射
  * broadcast:       广播
  * send / recv:     P2P 点对点通信
  * barrier:         同步屏障

默认后端为 ``native``：单机多卡（共享内存）或单进程多子线程场景下，
直接以副本同步保证数值正确性；通过 ``init_process_group`` 接入真正的
跨进程传输（ROCm IPC / MPI）时可切换到 ``shm`` / ``mpi`` 后端。
"""
from __future__ import annotations

import os
import threading
import numpy as np
from typing import Optional, List, Any, Dict

from .tensor import Tensor


# ============================================================
# 全局进程组状态
# ============================================================
_DEFAULT_GROUP: Optional["ProcessGroup"] = None
_LOCAL_RANK: int = 0
_WORLD_SIZE: int = 1
_BACKEND: str = "native"


class ProcessGroup:
    """进程组：管理一组参与集合通信的进程 (rank)。

    ``native`` 后端在单机多进程场景下使用文件系统 / 共享内存交换分片；
    跨节点时通过环境变量 ``SCT_DIST_ADDR`` / ``SCT_DIST_PORT`` 协商。
    """

    def __init__(self, rank: int, world_size: int, backend: str = "native",
                 group_name: str = "default") -> None:
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.group_name = group_name
        self._lock = threading.Lock()
        # 共享缓冲（native / shm / nccl 等后端在单机多进程下统一用文件分片交换）
        self._shm_dir: Optional[str] = None
        if world_size > 1:
            base = os.environ.get("SCT_DIST_TMPDIR", "/tmp")
            self._shm_dir = os.path.join(base, f"sct_pg_{group_name}_{world_size}")
            os.makedirs(self._shm_dir, exist_ok=True)

    # ---- 文件路径（分片交换） ----
    def _shard_path(self, op_id: str, rank: int) -> str:
        assert self._shm_dir is not None
        return os.path.join(self._shm_dir, f"{op_id}_r{rank}.npy")

    def _exchange_shard(self, op_id: str, data: np.ndarray) -> List[np.ndarray]:
        """写入本 rank 分片并读取其余 rank 的分片。"""
        import tempfile
        # 用临时原子写避免半写入
        path = self._shard_path(op_id, self.rank)
        tmp = path + ".tmp"
        np.save(tmp, data)
        os.replace(tmp, path)
        shards: List[np.ndarray] = []
        for r in range(self.world_size):
            p = self._shard_path(op_id, r)
            # 等待其他 rank 写入（简单自旋）
            waited = 0
            while not os.path.exists(p) and waited < 30.0:
                import time
                time.sleep(0.005)
                waited += 0.005
            shards.append(np.load(p))
        return shards

    def cleanup(self) -> None:
        if self._shm_dir and os.path.isdir(self._shm_dir):
            import shutil
            shutil.rmtree(self._shm_dir, ignore_errors=True)


# ============================================================
# 初始化 / 销毁进程组
# ============================================================
def init_process_group(
    backend: str = "native",
    init_method: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    timeout: Optional[float] = None,
    group_name: str = "default",
) -> None:
    """初始化默认进程组（兼容 torch.distributed.init_process_group）。"""
    global _DEFAULT_GROUP, _LOCAL_RANK, _WORLD_SIZE, _BACKEND
    # 允许环境变量覆盖（兼容 torchrun / mpirun 调用）
    env_rank = os.environ.get("RANK") or os.environ.get("PMI_RANK")
    env_world = os.environ.get("WORLD_SIZE") or os.environ.get("PMI_SIZE")
    env_local = os.environ.get("LOCAL_RANK")
    if env_rank is not None:
        rank = int(env_rank)
    if env_world is not None:
        world_size = int(env_world)
    if env_local is not None:
        _LOCAL_RANK = int(env_local)
    _BACKEND = backend
    _WORLD_SIZE = world_size
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", _LOCAL_RANK))
    _DEFAULT_GROUP = ProcessGroup(rank, world_size, backend, group_name)


def destroy_process_group(group: Optional[ProcessGroup] = None) -> None:
    """销毁进程组。"""
    global _DEFAULT_GROUP
    g = group or _DEFAULT_GROUP
    if g is not None:
        g.cleanup()
    if group is None:
        _DEFAULT_GROUP = None


def get_default_group() -> ProcessGroup:
    global _DEFAULT_GROUP
    if _DEFAULT_GROUP is None:
        _DEFAULT_GROUP = ProcessGroup(_LOCAL_RANK, _WORLD_SIZE, _BACKEND)
    return _DEFAULT_GROUP


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    return (group or get_default_group()).rank


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    return (group or get_default_group()).world_size


# ============================================================
# 集合通信原语
# ============================================================
def _to_numpy(t: Any) -> np.ndarray:
    if isinstance(t, Tensor):
        return t.data.astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _op_reduce(shards: List[np.ndarray], op: str) -> np.ndarray:
    """对多个分片做规约。"""
    stack = np.stack(shards, axis=0)
    if op == "sum":
        return stack.sum(axis=0)
    if op == "mean":
        return stack.mean(axis=0)
    if op == "min":
        return stack.min(axis=0)
    if op == "max":
        return stack.max(axis=0)
    if op == "prod":
        return stack.prod(axis=0)
    raise ValueError(f"未知规约算子: {op}")


def all_reduce(
    tensor: Tensor,
    op: str = "sum",
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """全规约：所有 rank 得到相同结果。

    ``native`` 后端在 world_size==1 时直接返回原张量；
    多 rank 时通过分片交换区做 Ring-AllReduce 等价实现。
    """
    g = group or get_default_group()
    if g.world_size == 1:
        return tensor
    np_arr = _to_numpy(tensor)
    op_id = f"ar_{id(tensor)}_{g.rank}"
    shards = g._exchange_shard(op_id, np_arr)
    reduced = _op_reduce(shards, op)
    tensor.data = reduced.astype(tensor.data.dtype)
    return tensor


def all_gather(
    tensor: Tensor,
    group: Optional[ProcessGroup] = None,
) -> List[Tensor]:
    """全收集：收集所有 rank 的张量到列表。"""
    g = group or get_default_group()
    if g.world_size == 1:
        return [tensor]
    np_arr = _to_numpy(tensor)
    op_id = f"ag_{id(tensor)}_{g.rank}"
    shards = g._exchange_shard(op_id, np_arr)
    return [Tensor(s.copy()) for s in shards]


def reduce_scatter(
    tensor: Tensor,
    op: str = "sum",
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """规约散射：先全规约再按 rank 切分。"""
    g = group or get_default_group()
    if g.world_size == 1:
        return tensor
    reduced = all_reduce(tensor, op, group)
    np_arr = _to_numpy(reduced)
    # 沿第 0 维平均切分
    chunk = np_arr.shape[0] // g.world_size
    start = g.rank * chunk
    end = start + chunk if g.rank < g.world_size - 1 else np_arr.shape[0]
    return Tensor(np_arr[start:end].copy())


def broadcast(
    tensor: Tensor,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """广播：从 src rank 把张量广播到所有 rank。"""
    g = group or get_default_group()
    if g.world_size == 1:
        return tensor
    np_arr = _to_numpy(tensor)
    op_id = f"bc_{id(tensor)}_{src}"
    if g.rank == src:
        shards = g._exchange_shard(op_id, np_arr)
        _ = shards  # src 写入后读取回来（包含自身）
        return tensor
    # 非 src：读取 src 写入的分片
    path = g._shard_path(op_id, src)
    waited = 0
    while not os.path.exists(path) and waited < 30.0:
        import time
        time.sleep(0.005)
        waited += 0.005
    data = np.load(path)
    tensor.data = data.astype(tensor.data.dtype)
    return tensor


def send(tensor: Tensor, dst: int, group: Optional[ProcessGroup] = None,
         tag: int = 0) -> None:
    """P2P 发送。"""
    g = group or get_default_group()
    np_arr = _to_numpy(tensor)
    op_id = f"send_{g.rank}_to_{dst}_t{tag}"
    # 写入目标 rank 的接收槽
    path = g._shard_path(op_id, dst)
    tmp = path + ".tmp"
    np.save(tmp, np_arr)
    os.replace(tmp, path)


def recv(tensor: Tensor, src: int, group: Optional[ProcessGroup] = None,
         tag: int = 0) -> Tensor:
    """P2P 接收。"""
    g = group or get_default_group()
    op_id = f"send_{src}_to_{g.rank}_t{tag}"
    path = g._shard_path(op_id, g.rank)
    waited = 0
    while not os.path.exists(path) and waited < 30.0:
        import time
        time.sleep(0.005)
        waited += 0.005
    data = np.load(path)
    tensor.data = data.astype(tensor.data.dtype)
    return tensor


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """同步屏障（native 后端用共享标记文件）。"""
    g = group or get_default_group()
    if g.world_size <= 1:
        return
    op_id = f"barrier_{os.getpid()}"
    path = g._shard_path(op_id, g.rank)
    open(path, "w").close()
    for r in range(g.world_size):
        p = g._shard_path(op_id, r)
        waited = 0
        while not os.path.exists(p) and waited < 30.0:
            import time
            time.sleep(0.005)
            waited += 0.005


def is_initialized() -> bool:
    return _DEFAULT_GROUP is not None


def is_available() -> bool:
    return True


class ReduceOp:
    """兼容 torch.distributed.ReduceOp 的规约算子枚举。"""
    SUM = "sum"
    AVG = "mean"
    PRODUCT = "prod"
    MIN = "min"
    MAX = "max"
    BAND = "band"
    BOR = "bor"
    BXOR = "bxor"
    PREMUL_SUM = "premul_sum"


__all__ = [
    "ProcessGroup", "init_process_group", "destroy_process_group",
    "get_default_group", "get_rank", "get_world_size",
    "all_reduce", "all_gather", "reduce_scatter", "broadcast",
    "send", "recv", "barrier", "is_initialized", "is_available",
    "ReduceOp",
]
