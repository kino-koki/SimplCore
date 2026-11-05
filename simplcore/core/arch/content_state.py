# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""内容寻址的 O(1) 叠加状态（CAS）——「少一条纯粹性」那条路的最小实现。

为什么有这个模块
----------------
TS2SC-Q17 v1~v5 的实测边界（`docs/TS2SC_Q17_v5_direct_weight.md`）：
静态、输入无关的谱核与 attention 的核 ``K(i,j)=f(q_i,k_j)`` **函数类不
匹配**。精确核 oracle 显示：核形状 100% 正确时中层注入 cos 仍 ≈0，因为
真实 attention 的方向由 ``q_i·k_j`` 内容匹配决定；v4 已把静态核函数类内的
上限吃满（L27 残差 cos 0.2484 ≈ oracle 0.2476），再加适配、加步数都绕不过。

本模块用**一阶叠加状态**直接对准这个缺口::

    φ(·) = 1 + elu(·)                      # 正值特征映射，不涉及位置维 softmax
    S_t  = S_{t-1} + φ(k_t) ⊗ v_t          # (F, V) 矩阵
    z_t  = z_{t-1} + φ(k_t)                # (F,) 归一化
    y_t  = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ z_t + ε)

其中 ``k,v,q`` 都是当前输入 ``u_t`` 的线性投影（分离的寻址/读出投影，
与 `memory/binding_memory.py` R1 修复第一版原型卡点的结论一致）。

契约账（**这是关键，别含糊**）
------------------------------
保持：

* **O(1) 常驻状态**：``S`` 是 (F,V)、``z`` 是 (F,)，形状与 L 无关；
* **无 N×N 打分**：全程没有位置对 (i,j) 的权重矩阵；
* **无 KV cache**：不保存历史 token；
* **严格因果**：位置 t 只读 s≤t 的写入；改未来不影响过去；
* **双形式**：流式一阶累加 ↔ 并行 prefix cumsum（fp64 一致，fp32 容差内，
  偏差随 L 增长——与 SGC 的 FFT 路径同口径，不是逐位）。

**少掉的一条**：核（历史聚合模式）从「完全输入无关」变成「输入的函数」。
``S_t`` 由内容写入，读出的等效核 ``K_t(s) ∝ φ(q_t)·φ(k_s)`` 依赖当前
query 与历史 key 的内容——这正是 SimplCore 原先最硬的那条纯度。

**另一处口径变化（必须一并说明，不能只说"少一条"）**：并行形式不再是
FFT 卷积，而是 associative prefix-sum；IIR 二阶振荡递推变成一阶累加。
即「FFT ↔ IIR 双形式」这一族数学让位给「prefix-sum ↔ 累加」，O(1) 状态
本身不受影响。论文 §4 定理 1 的分路径口径不适用于本模块。

**不属于本模块**：没有 softmax over positions、没有 q·k 逐对打分、没有
KV 载体、没有 O(L²)。它是线性注意力族的内容写入状态，不是 attention
模块；`purity.py` 的三层守卫按标识符扫描，本文件不含任何被禁标识符，
也不在豁免白名单里——不需要靠改名或豁免通过。

诚实边界
--------
* 这是一个**实验件**，锁机制与口径，不锁「1.7B 上能说正常汉语」；
* 状态开销 O(F·V) 随 D² 增长（1.7B / D=2048 全尺寸约 16 MiB/层）；
  可用低秩 ``feat_dim=r`` 压到 O(r·D)。每 token FLOPs 是 O(F·V)，
  比 SGC 的 O(M·D) 贵——**成本不是零**，见 `docs/CAS_少一条纯粹性的真实收益.md`；
* 迁移到 Qwen3-1.7B 的完整适配尚未做；`eval/cas_probe.py` 只给
  MQAR 轴上的判据性对照。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContentAddressedState", "CASState"]


class CASState:
    """流式状态：与 L 无关，只装两块常驻张量。"""

    __slots__ = ("S", "z")

    def __init__(self, S: torch.Tensor, z: torch.Tensor):
        self.S = S
        self.z = z


class ContentAddressedState(nn.Module):
    """内容写入的 O(1) 叠加状态 mixer（单序列或 batch 均可）。

    Args:
        dim: 残差流通道 D。
        feat_dim: 特征维 F（低秩可用 r≪D 压状态）。
        value_dim: 值维 V（默认 = D）。
        eps: 读出归一化分母的下限。
        zero_init_readout: 读出投影零初始化（初始注入恒等，与 CMT/R1
            同纪律），避免一上来就破坏残差流。
        phi: ``"elu1"``（1+elu，默认）/ ``"relu1"``（1+relu）/
            ``"softmax"``（特征维 softmax，φ 落在概率单纯形上，可区分度
            更高）。三者都**只对特征维归一化，不涉及位置维 softmax**。
    """

    def __init__(self, dim: int, feat_dim: Optional[int] = None,
                 value_dim: Optional[int] = None, eps: float = 1e-6,
                 zero_init_readout: bool = True,
                 phi: str = "elu1") -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if phi not in ("elu1", "relu1", "softmax"):
            raise ValueError(
                f"phi must be 'elu1'|'relu1'|'softmax', got {phi!r}")
        self.dim = int(dim)
        self.feat_dim = int(feat_dim or dim)
        self.value_dim = int(value_dim or dim)
        self.eps = float(eps)
        self.phi_kind = phi

        # 分离的寻址 / 读出投影（R1 的卡点修复：key 与 value 不能同源）
        self.write_key = nn.Linear(self.dim, self.feat_dim, bias=False)
        self.write_value = nn.Linear(self.dim, self.value_dim, bias=False)
        self.read_query = nn.Linear(self.dim, self.feat_dim, bias=False)
        self.read_out = nn.Linear(self.value_dim, self.dim, bias=False)
        if zero_init_readout:
            nn.init.zeros_(self.read_out.weight)

    # -- 特征映射 -------------------------------------------------------------
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """正值特征映射，**不涉及位置维 softmax**（无 N×N 的来源）。"""
        if self.phi_kind == "relu1":
            return F.relu(x) + 1.0
        if self.phi_kind == "softmax":
            # 乘 F 让 φ 的元素和恒为 F（与 elu1 在 x≈0 处的尺度一致），
            # 避免归一化分母随特征维塌缩。这是特征维 softmax，不是
            # attention 的位置维 softmax —— 无 N×N 打分矩阵。
            return F.softmax(x, dim=-1) * x.shape[-1]
        return F.elu(x) + 1.0

    # -- 状态 -----------------------------------------------------------------
    def init_state(self, batch: int = 1, device=None,
                   dtype=None) -> CASState:
        """O(1) 状态：形状只由 (F,V) 决定，与序列长度无关。"""
        p = next(self.parameters())
        dev = device or p.device
        dt = dtype or p.dtype
        return CASState(
            S=torch.zeros(batch, self.feat_dim, self.value_dim,
                          device=dev, dtype=dt),
            z=torch.zeros(batch, self.feat_dim, device=dev, dtype=dt))

    def state_bytes(self, batch: int = 1) -> int:
        """状态字节数（**与 L 无关**，O(1) 判据的机器可验证形式）。"""
        p = next(self.parameters())
        return (batch * (self.feat_dim * self.value_dim + self.feat_dim)
                * p.element_size())

    # -- 双形式 ---------------------------------------------------------------
    def forward(self, u: torch.Tensor,
                state: Optional[CASState] = None) -> torch.Tensor:
        """并行路径：prefix cumsum（associative scan），O(L·F·V) 训练内存。

        ``u``: (B, L, D)。位置 t 只读 s≤t 的写入（含自身），严格因果。
        返回注入量 (B, L, D)，由调用方做残差相加。
        """
        k = self.phi(self.write_key(u))                  # (B,L,F)
        v = self.write_value(u)                         # (B,L,V)
        q = self.phi(self.read_query(u))                # (B,L,F)
        kv = torch.einsum("blf,blv->blfv", k, v)        # (B,L,F,V)
        S = kv.cumsum(dim=1)                            # prefix 状态
        z = k.cumsum(dim=1)                             # (B,L,F)
        if state is not None:
            # 允许带初始状态（长上下文分块）：S[:,0] 已含 s=0，整体加前缀
            S = S + state.S.unsqueeze(1)
            z = z + state.z.unsqueeze(1)
        num = torch.einsum("blf,blfv->blv", q, S)       # (B,L,V)
        den = (q * z).sum(dim=-1, keepdim=True) + self.eps
        y = num / den
        return self.read_out(y)

    def step(self, u_t: torch.Tensor,
             state: CASState) -> Tuple[torch.Tensor, CASState]:
        """流式路径：一阶累加递推，O(1) 状态、O(F·V) / token。

        ``u_t``: (B, D)。返回 (注入量 (B,D), 新状态)。
        """
        k = self.phi(self.write_key(u_t))               # (B,F)
        v = self.write_value(u_t)                       # (B,V)
        q = self.phi(self.read_query(u_t))              # (B,F)
        kv = torch.einsum("bf,bv->bfv", k, v)
        S = state.S + kv
        z = state.z + k
        num = torch.einsum("bf,bfv->bv", q, S)
        den = (q * z).sum(dim=-1, keepdim=True) + self.eps
        y = num / den
        return self.read_out(y), CASState(S, z)

    # -- 审计 -----------------------------------------------------------------
    def contract(self) -> Dict[str, object]:
        """给守护测试/CI 读的契约账，避免"少一条"被悄悄扩大。"""
        return {
            "state_bytes_L_independent": True,
            "state_shape": [self.feat_dim, self.value_dim],
            "state_scalar_count": self.feat_dim * self.value_dim + self.feat_dim,
            "no_NxN_scores": True,
            "no_kv_cache": True,
            "no_position_softmax": True,
            "strictly_causal": True,
            "kernel_input_independent": False,          # ← 少掉的那一条
            "dual_form": "prefix-cumsum <-> first-order accumulator",
            "fft_iir_dual_form": False,                 # ← 口径变化，不是"少一条"能概括的
            "flops_per_token": self.feat_dim * self.value_dim,
        }
