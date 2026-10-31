# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""定容绑定记忆（R1，0.1-α 实验件）—— 修复轴 D 的 key→value 绑定缺陷.

背景（2026-10-31 复现报告，``docs/REPRODUCTION_REPORT.md``）：
* MQAR 对照（G=4，2 层，3000 步）：attention 0.500 vs SGC 0.3125 vs
  chance 0.125，加上精确指向 1.000 vs 0.398——**SGC 缺的不是感受野，
  是 key→value 绑定能力**；
* 第一版定容记忆原型失败的**真卡点在读取**：槽内把 h_t 同时当寻址 key
  和读出 value——查询是 h_k、值存在 h_v 槽，配对关系只存在于上下文内，
  逐 token 写入从原理上造不出绑定（这正是 attention 用独立 Q/K/V +
  induction head 做到的事）。

本模块的修复（两条都对着卡点）：
1. **槽内 key/value 分开存**：``addr_key`` / ``addr_value`` 两个独立投影；
2. **绑定写入单元从"单 token"改为"局部窗口"**：宽度 ``bind_window`` 的
   相邻窗口**前半池化进 key 侧、后半池化进 value 侧**——自然语言里
   「键 值」恰恰是相邻结构（"key_token value_token"），depthwise 局部
   卷积的窗口天然覆盖这种绑定；流式下每步写一个绑定单元（key 侧来自
   滑窗历史、value 侧来自当前 token），FIFO 环形槽。

一致性判据（架构忠诚度的硬标准，**不是**"有没有 softmax"）：
* 状态与 L 无关：槽容 O(S·D)，写满后 FIFO 覆盖，``slot_bytes()`` 恒定；
* 无 O(L²) 两两交互：寻址是 softmax over **S 个槽**（O(S·D)），不是
  softmax over L 个位置——定容检索不背叛架构。
* 命名说明：与 attention 的机制类比成立（分离的寻址/读出投影），但键集
  是**定容可变内容的槽**而非逐位置键值——它是记忆组件，不是 mixer；
  纯 attention-free 叙事不受影响（无 ``q/k/v_proj``、无 SDPA，通过
  purity 三层守卫）。

诚实边界
--------
* ``write_mode="single"`` 是**故意保留的失败对照**（复现原型的卡点），
  供消融；默认 ``"bound"``；
* 读出注入默认**零初始化投影**（与 CMT memory_in 同纪律，初始恒等）；
* 梯度可达性全链断言（复现报告的 mem_Wq 梯度恒 0 教训）；
* 未在真实数据训练验证——本模块锁机制与口径，不锁检索质量数值；
* 单序列状态语义（与 ``stream_step`` 的 state 同构）：多序列并行由
  调用方持有多个实例，本模块不做 batch 内多状态。
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["BoundedBindingMemory"]


class BoundedBindingMemory(nn.Module):
    """定容绑定记忆：S 槽、槽内 (key, value) 分开、局部窗口绑定写入.

    Args:
        dim: 通道数。
        n_slots: 槽数 S（定容；FIFO 环形覆盖）。
        bind_window: 绑定窗口宽度 W（偶数；前半 key 侧、后半 value 侧）。
            W=2 即 (h_{t-1} | h_t) 最小绑定——覆盖「键 值」相邻结构。
        write_mode: ``"bound"``（默认，跨位置绑定）/ ``"single"``
            （失败对照：同 token 既当 key 又当 value——复现原型卡点）。
        zero_init_inject: 读出投影零初始化（默认 True，注入初始恒等）。

    状态语义：**单序列**（与 ``stream_step`` 的 state 同构）。两条路径：
    * 推理：``reset`` → 逐步 ``write_step(h)`` → ``retrieve(q)``（no_grad）；
    * 训练：``reset`` → ``retrieve_sequence(H)``（整段，边写边读，
      因果：位置 t 只读 [0, t) 的写入；全链 autograd 可导）。
    """

    def __init__(self, dim: int, n_slots: int = 64, bind_window: int = 2,
                 write_mode: str = "bound", zero_init_inject: bool = True) -> None:
        super().__init__()
        if dim <= 0 or n_slots <= 0:
            raise ValueError("dim/n_slots must be positive")
        if bind_window < 2 or bind_window % 2:
            raise ValueError(f"bind_window must be even and >= 2, got {bind_window}")
        if write_mode not in ("bound", "single", "soft"):
            raise ValueError(
                f"write_mode must be 'bound'|'single'|'soft', got {write_mode}")
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.bind_window = int(bind_window)
        self.write_mode = write_mode
        self.half = self.bind_window // 2

        # 寻址/读出投影：**key 与 value 分开**（修复第一版原型的卡点）
        self.addr_key = nn.Linear(dim, dim, bias=False)
        self.addr_value = nn.Linear(dim, dim, bias=False)
        self.addr_query = nn.Linear(dim, dim, bias=False)
        self.read_out = nn.Linear(dim, dim, bias=False)
        if zero_init_inject:
            nn.init.zeros_(self.read_out.weight)
        # soft 写入门（R1-可微化二期）：硬 argmin/FIFO 的离散写入不可微——
        # 长训练槽位分配坍塌后无梯度信号修复（复现报告二期：3000 步退化）。
        # 门控写入全程可微，逐槽 sigmoid 混合，状态仍定容 O(S·D)。
        self.write_mode_soft = write_mode == "soft"
        if self.write_mode_soft:
            self.write_gate = nn.Linear(2 * dim, n_slots)

        # 定容槽状态（buffer：随 state_dict 走、不进参数）
        self.register_buffer("slot_k", torch.zeros(n_slots, dim))
        self.register_buffer("slot_v", torch.zeros(n_slots, dim))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        # **写入位置与填充计数必须分离**：write_ptr 是环形物理槽位（回绕后
        # 变小），retrieve 的 filled 掩码必须用 n_written（累计写入数）——
        # 二者混用是「回绕后静默只读 1 槽」缺陷的根源（0.1-α 调试实录）。
        self.register_buffer("n_written", torch.zeros(1, dtype=torch.long))
        self._window: deque = deque(maxlen=self.bind_window)

    # -- 状态管理 -------------------------------------------------------------
    def reset(self, device=None, dtype=None) -> None:
        dev = device or self.slot_k.device
        dt = dtype or self.slot_k.dtype
        self.slot_k = torch.zeros(self.n_slots, self.dim, device=dev, dtype=dt)
        self.slot_v = torch.zeros(self.n_slots, self.dim, device=dev, dtype=dt)
        self.write_ptr = torch.zeros(1, dtype=torch.long, device=dev)
        self.n_written = torch.zeros(1, dtype=torch.long, device=dev)
        self._window.clear()

    def slot_bytes(self) -> int:
        """定容账本：槽状态字节数——**与已写入步数无关**（O(1) 判据）。"""
        return (self.slot_k.numel() + self.slot_v.numel()) * self.slot_k.element_size()

    # -- 绑定单元 ---------------------------------------------------------------
    def _bind_pair(self, win: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """局部窗口 → (key, value)：前半池化进 key 侧、后半进 value 侧.

        win: (W, dim)（单序列）或 (W, B, dim)（批量）。
        """
        k = self.addr_key(win[:self.half].mean(dim=0))
        v = self.addr_value(win[self.half:].mean(dim=0))
        return k, v

    # -- 推理路径（no_grad）------------------------------------------------------
    @torch.no_grad()
    def write_step(self, h: torch.Tensor) -> bool:
        """流式写入一步（h: (dim,)）。窗口满则写入一个绑定单元.

        返回是否发生写入。``write_mode="single"`` 时每步都写（失败对照）。
        """
        h = h.detach().to(self.slot_k.dtype).reshape(-1)
        if self.write_mode == "single":
            k, v = self.addr_key(h), self.addr_value(h)
            i = int(self.write_ptr.item()) % self.n_slots
            self.slot_k[i], self.slot_v[i] = k, v
            self.write_ptr[0] = i + 1
            self.n_written[0] += 1
            return True
        self._window.append(h)
        if len(self._window) < self.bind_window:
            return False                       # 绑定需要完整窗口
        k, v = self._bind_pair(torch.stack(list(self._window)))
        if self.write_mode_soft:
            # soft：逐槽门控混合（buffer 版与训练递推同语义）
            gate_in = torch.cat([k, v], dim=-1)
            w_col = torch.sigmoid(self.write_gate(gate_in)).unsqueeze(-1)
            self.slot_k = (1.0 - w_col) * self.slot_k + w_col * k.reshape(1, -1)
            self.slot_v = (1.0 - w_col) * self.slot_v + w_col * v.reshape(1, -1)
            self.n_written[0] += 1
            return True
        i = int(self.write_ptr.item()) % self.n_slots
        self.slot_k[i], self.slot_v[i] = k, v
        self.write_ptr[0] = i + 1
        self.n_written[0] += 1
        return True

    @torch.no_grad()
    def write_sequence(self, H: torch.Tensor) -> None:
        """buffer 路径整段写入：H (L, dim)（单序列；批量由调用方循环）.

        与 ``retrieve_sequence`` 同因果语义：位置 t 的查询读到的槽 =
        截至 t-1 的写入（先读后写由调用方排序）。与训练路径（图外槽容器）
        的数值语义一致，由测试锁定。
        """
        if H.dim() == 3 and H.shape[0] == 1:
            H = H[0]
        if H.dim() != 2:
            raise ValueError(f"expect (L,dim) or (1,L,dim), got {tuple(H.shape)}")
        for t in range(H.shape[0]):
            self.write_step(H[t])

    @torch.no_grad()
    def retrieve(self, q: torch.Tensor,
                 top_slots: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """查询 → (读出向量, softmax 分布)。O(S·D)，与历史长度无关.

        q: (dim,) 或 (..., dim)。寻址 softmax over **S 个槽**；未填槽掩掉；
        ``top_slots`` 可选稀疏化。无已填槽时返回**零向量**（不产生 NaN）。
        """
        q = q.to(self.slot_k.dtype)
        filled = min(int(self.n_written.item()), self.n_slots)
        if filled == 0:
            zero = torch.zeros(q.shape[:-1] + (self.dim,),
                               dtype=q.dtype, device=q.device)
            return zero, torch.zeros(q.shape[:-1] + (self.n_slots,),
                                     dtype=q.dtype, device=q.device)
        scores = self.addr_query(q) @ self.slot_k.T / math.sqrt(self.dim)
        if filled < self.n_slots:
            mask = torch.full_like(scores, float("-inf"))
            mask[..., :filled] = scores[..., :filled]
            scores = mask
        if top_slots is not None and top_slots < self.n_slots:
            kth = torch.topk(scores, top_slots, dim=-1).values[..., -1:]
            scores = torch.where(scores < kth,
                                 torch.full_like(scores, float("-inf")), scores)
        probs = torch.softmax(scores, dim=-1)
        return probs @ self.slot_v, probs

    def inject(self, q: torch.Tensor) -> torch.Tensor:
        """读出 → 零初始化投影（推理便捷口）。初始恒为 0。"""
        out, _ = self.retrieve(q)
        return self.read_out(out)

    # -- 训练路径（整段状态递推，全链 autograd）-----------------------------------
    def retrieve_sequence(self, H: torch.Tensor,
                          slot_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                          return_state: bool = False):
        """H (L, dim) → 读出 (L, dim)：位置 t 的查询 = h_t，槽 = 截至 t 的写入.

        **统一状态递推**（0.1-α 二期重构）：槽状态 (slot_k, slot_v) 是显式
        递推变量，每步产生**新张量**——无 in-place、无 CopySlices 版本冲突
        （复现报告 mem_Wq 教训的根治），梯度经递推自然流回全部历史写入投影。
        写入掩码按 write_mode 分派：

        * ``bound``：one-hot 写掩码（确定性 FIFO 位置；掩码是常量，无梯度）；
        * ``soft``：**可微门控写掩码**（w = σ(write_gate([k_new; v_new]))，
          逐槽 sigmoid 混合）——2026-10-31 二期报告的 3000 步退化结构性
          修复：硬 argmin 不可微，槽位坍塌后无梯度信号，门控全程可微；
        * ``single``：失败对照（同 token 既当 key 又当 value）。

        因果：位置 t 的读出使用**写入前**的槽（先读后写）。
        ``slot_state`` 传入上段终态（跨 chunk BPTT，与 stream_step 的 state
        语义同构）；``return_state=True`` 返回终态。
        """
        if H.dim() != 2:
            raise ValueError(f"expect (L,dim), got {tuple(H.shape)}")
        L = H.shape[0]
        dev, dt = H.device, H.dtype
        if slot_state is not None:
            slot_k, slot_v = slot_state
        else:
            slot_k = torch.zeros(self.n_slots, self.dim, device=dev, dtype=dt)
            slot_v = torch.zeros(self.n_slots, self.dim, device=dev, dtype=dt)
        outs: List[torch.Tensor] = []
        ptr = int(self.write_ptr.item())
        filled = min(int(self.n_written.item()), self.n_slots)
        window: deque = deque(maxlen=self.bind_window)
        for t in range(L):
            ht = H[t]
            # -- 先读（因果：只用已写入的槽）--
            if filled == 0:
                # 无已填槽：零读出（整行 -inf 的 softmax 是 NaN，推理路径
                # retrieve 的同名分支已处理，递推必须同口径）
                outs.append(torch.zeros(self.dim, device=dev, dtype=dt))
            else:
                qk = self.addr_query(ht)
                scores = qk @ slot_k.T / math.sqrt(self.dim)
                if filled < self.n_slots:
                    # 未填槽（零行）不得参与 softmax：score=0 → exp=1 会稀释
                    # 命中槽（推理路径 retrieve 有 filled 掩码，同口径）
                    mask = torch.full_like(scores, float("-inf"))
                    mask[:filled] = scores[:filled]
                    scores = mask
                probs = torch.softmax(scores, dim=-1)
                outs.append(probs @ slot_v)
            # -- 后写（按 write_mode 分派写掩码 w）--
            w, k_new, v_new = None, None, None
            if self.write_mode == "single":
                k_new, v_new = self.addr_key(ht), self.addr_value(ht)
                w = torch.zeros(self.n_slots, device=dev, dtype=dt)
                w[ptr % self.n_slots] = 1.0
                ptr += 1
            else:
                window.append(ht)
                if len(window) == self.bind_window:
                    k_new, v_new = self._bind_pair(torch.stack(list(window)))
                    if self.write_mode_soft:
                        gate_in = torch.cat([k_new, v_new], dim=-1)
                        w = torch.sigmoid(self.write_gate(gate_in))   # (S,) 可微
                    else:  # bound：one-hot FIFO 位置（确定性，无梯度需求）
                        w = torch.zeros(self.n_slots, device=dev, dtype=dt)
                        w[ptr % self.n_slots] = 1.0
                        ptr += 1
                    window.popleft()      # 滑动一格：保留最近 W-1 个
            if w is not None:
                filled = min(filled + 1, self.n_slots)
                slot_k = (1.0 - w).unsqueeze(-1) * slot_k + w.unsqueeze(-1) * k_new
                slot_v = (1.0 - w).unsqueeze(-1) * slot_v + w.unsqueeze(-1) * v_new
        if return_state:
            return torch.stack(outs, dim=0), (slot_k, slot_v)
        return torch.stack(outs, dim=0)
