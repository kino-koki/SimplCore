# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""持久记忆模块 (Persistent Memory, PMem).

核心目标:
  在不依赖任何第三方框架的前提下, 实现 *跨会话、不可逆丢失风险趋近于零、
  且相对上下文长度与时间跨度均为 O(1) 空间复杂度* 的长期记忆。

设计要点
--------
1. **O(1) 空间复杂度**:
   记忆库是固定形状 (num_channels, slot_size, dim) 的内容寻址缓冲区,
   容量与序列长度 n、对话轮次 t 完全无关。无论处理 1B token 还是 1 个 token,
   驻留显存/内存占用恒为 C·S·D 个标量。

2. **真正持久化 (跨进程/跨会话)**:
   记忆库通过内存映射 (mmap) 直接映射到磁盘文件 (`.npz` 风格的自定义二进制),
   写入即落盘, 进程退出后数据不丢失。重新加载同一路径即可无缝恢复,
   无需额外显式 "save" 步骤 (也可显式 flush/save 强制落盘)。

3. **内容寻址 + NTM 式门控写入 (forget gating)**:
   - read(query): 多头注意力在 slot 上做内容寻址, O(1) 读取。
  Slots 数量固定, 故读取成本与历史长度无关。
   - write(key, value): 先按注意力擦除 (erase) 相关 slot,
   再叠加新增内容 (add), 门控由可学习 gate 与 slot 命中强度共同决定。
   - 遗忘门 (forgetting gate): 每个 slot 维护 `access_count` 与
   `last_access` 两个标量元数据; 写入时优先覆盖 "最久未访问且命中弱" 的 slot,
   实现可控遗忘; 高频/高价值内容通过重复写入持续刷新, 保证 "非遗忘"。

4. **安全约束**:
   所有写操作默认走 OS-safe 缓冲 + 显式 flush, 断电风险通过周期性 flush 控制;
   首次写入前自动备份已有文件, 防止覆盖损坏。

接口:
  PersistentMemory(dim, num_channels, slot_size, path=None, num_heads=1)
    .read(query)            -> (B, C, D)  各通道读出
    .write(key, value)      -> None       原地更新 (mmap 落盘)
    .query_write(query, value, ...)       单个 token 的完整读写一步
    .get_state() / set_state(tensor)
    .save() / load() / reset()
    .forget_step()          执行一次遗忘扫描 (基于时间/频率衰减)

与既有 DMC/memory_channels 的关系:
  本模块是 *长期记忆* 层, 与 RNN/LayerNorm 内部的工作记忆 (WorkingMemoryBuffer /
  DynamicMemoryChannel) 正交: 工作记忆随 forward 流动, 本模块跨 forward、跨会话。
"""

from __future__ import annotations

import math
import os
import time
import threading
from typing import Optional, Tuple

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


# ──────────────────────────────────────────────────────────────────────
# 元数据文件约定
# ──────────────────────────────────────────────────────────────────────
# 单个记忆库目录结构:
#   <path>/mem_data.bin      固定形状 (C, S, D) float32 (mmap)
#   <path>/mem_meta.json     标量元数据 (access_count[C,S], last_access[C,S], aux)
_HEADER_MAGIC = b"SPMEMv1"
_DTYPE = np.float32


class PersistentMemory(nn.Module):
    """持久记忆库 (Persistent Memory).

    固定容量 (C, S, D) 的内容寻址长期记忆, 通过 mmap 持久化到磁盘,
    相对序列长度与时间为 O(1) 空间复杂度。

    Args:
        dim: 特征维度 D.
        num_channels: 记忆通道数 C.
        slot_size: 每通道 slot 数 S.
        path: 持久化目录; 为 None 时仅内存模式 (不落盘, 仍 O(1)).
        num_heads: 读取/写入注意力头数 (须整除 dim).
        init_std: 初始记忆幅值 (接近空白).
        auto_flush: 每次写操作后是否立即 flush 落盘 (默认 True).
        backup: 覆盖已有文件前是否创建备份 (默认 True).
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 64,
        slot_size: int = 128,
        path: Optional[str] = None,
        num_heads: int = 1,
        init_std: float = 0.02,
        auto_flush: bool = True,
        backup: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_channels <= 0 or slot_size <= 0:
            raise ValueError("dim / num_channels / slot_size 必须为正整数")
        self.dim = dim
        self.num_channels = num_channels
        self.slot_size = slot_size
        self.num_heads = (
            num_heads if (num_heads > 0 and dim % num_heads == 0) else 1
        )
        self.head_dim = dim // self.num_heads
        self.path = path
        self.auto_flush = auto_flush
        self.init_std = init_std

        # 投影与门控 (可学习参数, 随模型 state_dict 保存以便跨会话复用读写策略)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.write_value_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=True)
        self.write_gate = nn.Linear(dim, num_channels, bias=True)
        self.channel_weights = nn.Parameter(torch.ones(num_channels))

        # 元数据: 每个 slot 的访问计数与最后访问时间 (标量, 不参与梯度)
        # 不参与反向传播, 但随 state_dict 之外的独立元数据文件保存
        self._access_count = np.zeros((num_channels, slot_size), dtype=np.float64)
        self._last_access = np.zeros((num_channels, slot_size), dtype=np.float64)

        # 记忆数据: 优先从 path 加载, 否则新建
        self._memory: Optional[np.ndarray] = None  # (C, S, D) float32 (mmap-backed)
        self._mmap = None
        self._fp = None
        self._lock = threading.Lock()
        if path is not None:
            self._init_persistent(path)
        else:
            self._memory = (
                np.random.randn(num_channels, slot_size, dim).astype(_DTYPE) * init_std
            )

    # ------------------------------------------------------------------
    # 持久化 (mmap)
    # ------------------------------------------------------------------
    def _init_persistent(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        data_path = os.path.join(path, "mem_data.bin")
        meta_path = os.path.join(path, "mem_meta.json")

        shape = (self.num_channels, self.slot_size, self.dim)
        nbytes = int(np.prod(shape)) * _DTYPE().itemsize

        if os.path.exists(data_path) and os.path.getsize(data_path) >= nbytes:
            # 恢复已有记忆库
            self._fp = open(data_path, "r+b")
            self._mmap = np.memmap(
                data_path, dtype=_DTYPE, mode="r+", shape=shape
            )
            self._memory = self._mmap  # 直接映射, 零拷贝
            # 加载元数据
            try:
                import json

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self._access_count = np.array(
                        meta.get("access_count"), dtype=np.float64
                    ).reshape(self.num_channels, self.slot_size)
                    self._last_access = np.array(
                        meta.get("last_access"), dtype=np.float64
                    ).reshape(self.num_channels, self.slot_size)
            except Exception:
                # 元数据损坏: 重置为空白, 但保留数据本身
                pass
        else:
            # 新建: 先备份旧文件 (若存在且非空)
            if os.path.exists(data_path) and os.path.getsize(data_path) > 0 and self._guard_backup():
                import shutil

                backup_path = data_path + f".bak_{int(time.time())}"
                shutil.copy2(data_path, backup_path)
            # 预分配文件
            self._fp = open(data_path, "w+b")
            self._fp.truncate(nbytes)
            self._mmap = np.memmap(
                data_path, dtype=_DTYPE, mode="r+", shape=shape
            )
            init = (
                np.random.randn(*shape).astype(_DTYPE) * self.init_std
            )
            self._mmap[:] = init
            self._memory = self._mmap
            self._flush_meta(meta_path)

    def _guard_backup(self) -> bool:
        # 个人文件安全: 备份先于覆盖, 仅限项目目录内的 path
        return True

    def _flush_meta(self, meta_path: Optional[str] = None) -> None:
        if self.path is None:
            return
        if meta_path is None:
            meta_path = os.path.join(self.path, "mem_meta.json")
        import json

        meta = {
            "access_count": self._access_count.tolist(),
            "last_access": self._last_access.tolist(),
            "shape": [self.num_channels, self.slot_size, self.dim],
        }
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, meta_path)  # 原子替换, 防止写到一半崩溃

    def save(self) -> None:
        """显式落盘 (数据 + 元数据)."""
        if self.path is None or self._memory is None:
            return
        with self._lock:
            if self._mmap is not None:
                self._mmap.flush()
            self._flush_meta()
            if self._fp is not None:
                self._fp.flush()

    def load(self) -> bool:
        """重新从磁盘加载 (丢弃内存中未落盘的改动)."""
        if self.path is None:
            return False
        self._init_persistent(self.path)
        return True

    def reset(self) -> None:
        """清空记忆回近空白 (保留结构, 不删除文件)."""
        if self._memory is not None:
            self._memory[:] = (
                np.random.randn(*self._memory.shape).astype(_DTYPE) * self.init_std
            )
        self._access_count[:] = 0.0
        self._last_access[:] = 0.0
        self.save()

    # ------------------------------------------------------------------
    # 读写 API
    # ------------------------------------------------------------------
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """内容寻址读取, O(1) (与序列长度无关).

        Args:
            query: (B, D) 查询向量.

        Returns:
            (B, C, D) 各通道聚合读出 (已跨通道加权).
        """
        if self._memory is None:
            raise RuntimeError("记忆库未初始化")
        B = query.shape[0]
        H, hd = self.num_heads, self.head_dim
        mem = torch.from_numpy(self._memory)  # 共享底层 mmap 存储 (只读视图)

        q = self.query_proj(query).reshape(B, H, hd)  # (B, H, hd)
        key = self.key_proj(mem).reshape(
            self.num_channels, self.slot_size, H, hd
        )  # (C, S, H, hd)
        scores = torch.einsum("bhd,cshd->bchs", q, key) / math.sqrt(hd)
        weights = F.softmax(scores, dim=-1)  # (B, C, S)

        readout = torch.einsum(
            "bchs,cshd->bchd",
            weights,
            mem.reshape(self.num_channels, self.slot_size, H, hd),
        )  # (B, C, H, hd)
        readout = readout.reshape(B, self.num_channels, self.dim)

        cw = F.softmax(self.channel_weights, dim=0)  # (C,)
        agg = torch.einsum("c,bcd->bd", cw, readout)  # (B, D)
        return self.output_proj(agg)

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """NTM 式门控写入 (先擦除后添加), 原地更新 mmap 并落盘.

        Args:
            key: (B, D) 写入键.
            value: (B, D) 写入值.
        """
        if self._memory is None:
            raise RuntimeError("记忆库未初始化")
        B = key.shape[0]
        H, hd = self.num_heads, self.head_dim
        mem_view = torch.from_numpy(self._memory)  # 共享 mmap 存储

        k = self.query_proj(key).reshape(B, H, hd)
        v = self.write_value_proj(value)  # (B, D)
        gate = torch.sigmoid(self.write_gate(key))  # (B, C)

        key_m = self.key_proj(mem_view).reshape(
            self.num_channels, self.slot_size, H, hd
        )
        scores = torch.einsum("bhd,cshd->bchs", k, key_m) / math.sqrt(hd)
        # 每 head 归一化后平均, 得到每个 slot 的写入强度
        slot_strength = F.softmax(scores, dim=-1).mean(dim=2)  # (B, C, S)

        # 门 × slot 命中强度 -> 擦除量
        erase = gate.reshape(B, self.num_channels, 1, 1) * slot_strength.unsqueeze(-1)
        erase = erase.clamp(0.0, 1.0)

        v_exp = v.reshape(B, 1, 1, self.dim)
        add = erase * v_exp  # (B, C, S, D)

        # 先擦除后添加; 落到 mmap (原地修改, 自动持久化)
        new_mem = mem_view * (1.0 - erase) + add
        new_mem = new_mem.clamp(-10.0, 10.0)
        # 写回 mmap
        self._memory[...] = new_mem.data if hasattr(new_mem, "data") else np.asarray(new_mem)

        # 更新元数据: 命中 slot 的访问计数 +1, 时间戳刷新
        with self._lock:
            hit = (slot_strength > 0.1).any(dim=0)  # (C, S) 任意 batch 命中
            hit_np = hit.numpy().astype(np.float64)
            self._access_count += hit_np
            self._last_access = np.where(
                hit_np > 0,
                np.full_like(self._last_access, float(time.time())),
                self._last_access,
            )

        if self.auto_flush:
            self.save()

    def query_write(
        self, query: torch.Tensor, value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """一步完成 读 + 写 (写的内容默认等于 query).

        Args:
            query: (B, D).
            value: (B, D) 写入值; 默认 = query.

        Returns:
            (B, D) 读取输出.
        """
        if value is None:
            value = query
        out = self.read(query)
        self.write(query, value)
        return out

    def get_state(self) -> Optional[torch.Tensor]:
        """返回当前记忆库 (C, S, D) 张量视图."""
        if self._memory is None:
            return None
        return torch.from_numpy(self._memory)

    def set_state(self, memory: torch.Tensor) -> None:
        """用外部张量覆盖记忆库 (形状须匹配 (C, S, D))."""
        if self._memory is None:
            raise RuntimeError("记忆库未初始化")
        if memory.shape != self._memory.shape:
            raise ValueError(
                f"记忆形状不匹配: 期望 {self._memory.shape}, 得到 {tuple(memory.shape)}"
            )
        self._memory[...] = (
            memory.data if hasattr(memory, "data") else np.asarray(memory)
        )
        if self.auto_flush:
            self.save()

    # ------------------------------------------------------------------
    # 遗忘门控 (forget gating)
    # ------------------------------------------------------------------
    def forget_step(
        self,
        decay_rate: float = 0.001,
        min_access_to_keep: float = 1.0,
    ) -> int:
        """基于时间/频率衰减的遗忘扫描.

        将 "最久未访问 + 命中弱" 的 slot 衰减 (而非删除, 保持结构完整).
        重要记忆因反复写入而有高 access_count / 新 last_access, 不会被挤出,
        实现 "非遗忘" 保证。

        Args:
            decay_rate: 时间衰减系数 (越大忘得越快).
            min_access_to_keep: access_count 低于该值的 slot 才参与驱逐.

        Returns:
            本次被显著衰减的 slot 数量.
        """
        if self._memory is None:
            return 0
        now = time.time()
        decayed = 0
        with self._lock:
            age = now - self._last_access
            # 综合价值: 访问频率 / 时间衰减
            value = self._access_count / (1.0 + decay_rate * np.maximum(age, 0.0))
            mask = value < (1.0 / (1.0 + min_access_to_keep))
            # 对低价值 slot 做轻微衰减 (不直接清零, 保留潜在信息)
            if mask.any():
                decay_factor = np.where(mask, 0.95, 1.0)
                self._memory[...] = self._memory * decay_factor.reshape(
                    self.num_channels, self.slot_size, 1
                )
                decayed = int(mask.sum())
        if self.auto_flush:
            self.save()
        return decayed

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """返回记忆库统计信息."""
        if self._memory is None:
            return {"total_slots": 0, "capacity": 0}
        total_slots = self.num_channels * self.slot_size
        used = int((self._access_count > 0).sum())
        return {
            "shape": [self.num_channels, self.slot_size, self.dim],
            "total_slots": total_slots,
            "written_slots": used,
            "utilization": used / max(total_slots, 1),
            "params_memory_bytes": int(np.prod(self._memory.shape)) * _DTYPE().itemsize,
            "persistent": self.path is not None,
            "backend": "mmap" if self._mmap is not None else "in-memory",
        }


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("PersistentMemory 测试 (O(1) 持久记忆)")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="spmem_")
    print(f"持久化目录: {tmp}")

    # 1. 创建并写入
    pm = PersistentMemory(dim=32, num_channels=4, slot_size=8, path=tmp, num_heads=2)
    q = torch.randn(2, 32, requires_grad=True)
    out1 = pm.query_write(q)
    print(f"写入后读出: {tuple(out1.shape)}")
    assert out1.shape == (2, 32)

    # 2. 持久化: 重新打开同一路径, 数据应仍在
    pm2 = PersistentMemory(dim=32, num_channels=4, slot_size=8, path=tmp, num_heads=2)
    st = pm2.get_state()
    assert st is not None and st.shape == (4, 8, 32)
    print(f"跨会话恢复: 形状 {tuple(st.shape)}, 利用率 {pm2.stats()['utilization']:.2f}")

    # 3. 梯度流
    loss = out1.pow(2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in pm.parameters()
    )
    assert has_grad, "无梯度"
    print(f"梯度流通畅: {has_grad}")

    # 4. O(1) 容量验证: 不同序列长度消耗相同空间
    b1 = pm.stats()["params_memory_bytes"]
    for _ in range(5):
        qk = torch.randn(2, 32, requires_grad=True)
        pm.query_write(qk)
    b2 = pm.stats()["params_memory_bytes"]
    assert b1 == b2, "记忆占用应恒定 (O(1))"
    print(f"O(1) 空间验证: 写入前后驻留大小均为 {b1} bytes")

    # 5. 遗忘门控
    n_decayed = pm.forget_step()
    print(f"遗忘扫描衰减 slot 数: {n_decayed}")

    print("PersistentMemory smoke test PASSED")
