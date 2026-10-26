# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
LatticeModel —— 点阵谐振模型 (非 Transformer / 非注意力 架构)
==============================================================

设计目标
--------
彻底脱离传统 Transformer / 注意力范式: 不使用 self-attention、softmax 打分、
QK^T、positional encoding、FFT 或任何注意力变体。整个序列混合与记忆访问
走两条物理直觉驱动的机制:

  1. 谐振耦合层 (ResonantCoupling, RC)
     在时域用可学习的高斯带通卷积核实现"局部-全局"频率选择,
     复杂度 O(n · k) (k 为固定核长), 与序列长度 n 近线性。
     不使用 FFT: 直接时域卷积, 避免对缺失算子的依赖。

  2. 哈希寻址记忆 (HashMemory, HM)
     用可学习投影 + 最近邻定位 (argmax) 把每个位置映射到固定记忆槽,
     读取走稀疏 gather、写入走稀疏 scatter-add, 全程无任何 softmax 打分。
     复杂度 O(n · C · S · D) 且 C·S 固定, 相对 n 为 O(1) 级。

两个机制都不含 attention, 因此即便在 1B 上下文下, 显存与算力都不随 n 平方增长。

依赖: 仅依赖自研 simplcore.tensor (NumPy 后端), 不 import torch / deepspeed / tensorflow。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


# ======================================================================
# 1. 谐振耦合层 —— 时域高斯带通卷积 (无 FFT, 无可学习注意力)
# ======================================================================
class ResonantCoupling(nn.Module):
    """谐振耦合层.

    用可学习的高斯带通卷积核替代注意力进行序列混合. 每个通道携带一个中心偏移
    与带宽, 卷积核直接在时域构造 (而非常规 FFT 频谱), 因此对后端无 FFT 依赖.

    前向:
        y = x + out_scale · Conv1d(x, W)   (W 为 (num_bands, 1, kernel_size) 可学习)
    复杂度: O(B · L · D · K), K 固定 → 相对 n 近似 O(n).
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_bands > 0 and kernel_size > 0
        self.dim = dim
        self.num_bands = num_bands
        self.kernel_size = kernel_size

        # 可学习带通中心偏移 (相对位置, 决定每个频带捕捉的尺度)
        self.center = nn.Parameter(torch.randn(num_bands) * 0.3)
        # 可学习带宽 (softplus 保证非负)
        self.width_raw = nn.Parameter(torch.ones(num_bands) * 2.0)
        # 每通道一个 1×K 卷积核 (跨特征共享权重, 减少参数量)
        self.kernel = nn.Parameter(
            torch.randn(num_bands, 1, kernel_size) * 0.02
        )
        self.out_scale = nn.Parameter(torch.ones(1) * 0.5)
        # 可训练投影: 把 (num_bands * D) 融合回 D (替代原先无参 mean, 提升表达力)
        self.band_proj = nn.Linear(num_bands * dim, dim, bias=True)
        self.norm = nn.RMSNorm(dim)
        self.dropout = dropout

    def widths(self) -> torch.Tensor:
        # 注意: 必须为普通方法而非 @property, 否则会被 nn.Module.__getattr__ 拦截
        # (它会优先把 self.width_raw 当作已注册参数查找, 导致 @property 失效).
        return F.softplus(self.width_raw)

    def _build_bandpass(self) -> torch.Tensor:
        """时域高斯带通核 (num_bands, 1, K)."""
        K = self.kernel_size
        t = torch.arange(K, dtype=torch.float32) - (K - 1) / 2.0  # (K,)
        t = t.reshape(1, 1, K)
        c = self.center.reshape(-1, 1, 1)        # (Bn,1,1)
        w = self.widths().reshape(-1, 1, 1)      # (Bn,1,1)
        # 高斯带通: exp(-((t-c)^2)/(2 w^2)), 能量归一
        raw = torch.exp(-((t - c) ** 2) / (2.0 * w ** 2))
        norm = raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return raw / norm  # (Bn,1,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D).

        正确实现: 用滑动窗口 (unfold) 把序列展开为 (B, L', K), 再与每个 band
        的高斯带通核做 einsum 卷积, 最后用可训练投影把 (D·Bn) 压回 D。
        """
        residual = x
        h = self.norm(x)
        B, L, D = h.shape
        K = self.kernel_size
        band = self._build_bandpass()  # (Bn, 1, K)
        band_k = band.reshape(self.num_bands, K)  # (Bn, K)

        # --- 滑动窗口: 构造 (B, D, L', K) ---
        Lp = L - K + 1
        if Lp <= 0:
            # 序列比核还短: 直接返回残差 (避免越界)
            return residual
        idx = torch.arange(L, dtype=torch.long)
        win = torch.arange(K, dtype=torch.long).reshape(1, K) + idx[:Lp].reshape(Lp, 1)  # (L',K)
        # h: (B, L, D) -> gather 到 (B, L', K, D)
        h_win = h[:, win.data, :]  # (B, L', K, D)
        # 与 band 核卷积: einsum "bikd,bk->bid"  (对每个 band)
        # 先扩展到 band 维度: (B, L', K, D, 1) * (Bn, K, 1) -> sum over K
        conv = torch.einsum("blcd,ic->blid", h_win, band_k)  # (B, L', Bn, D)
        conv = conv.reshape(B, Lp, self.num_bands * D)  # (B, L', Bn*D)
        # 可训练投影 (Bn*D -> D), 复用 out_scale 之外的专用权重
        out = self.band_proj(conv)  # (B, L', D)

        # 长度对齐 (valid 卷积后 L' < L): 两端镜像填充回 L
        if out.shape[1] != L:
            left = (L - Lp) // 2
            right = L - Lp - left
            if left > 0:
                out = torch.cat([out[:, :1, :].repeat_interleave(left, dim=1), out], dim=1)
            if right > 0:
                out = torch.cat([out, out[:, -1:, :].repeat_interleave(right, dim=1)], dim=1)
        out = self.out_scale * out
        return residual + out

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容旧接口 (当前 forward 已内联投影, 此方法保留供测试/复用)
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_bands, self.dim).mean(dim=2)

    def complexity(self, seq_len: int, batch: int = 1) -> Dict[str, float]:
        n, d, K, Bn = seq_len, self.dim, self.kernel_size, self.num_bands
        # 每 band 手卷: B·D·(L-K+1)·K 次乘加
        rc_flops = batch * d * (n - K + 1) * K * Bn
        attn_flops = 2.0 * n * n * d * batch
        return {
            "rc_flops": rc_flops,
            "attn_flops": attn_flops,
            "speedup_vs_attention": attn_flops / max(rc_flops, 1.0),
            "rc_big_o": "O(n)",
            "attn_big_o": "O(n^2)",
        }


# ======================================================================
# 2. 哈希寻址记忆 —— 最近邻定位 (无 softmax, 无 QK^T)
# ======================================================================
class HashMemory(nn.Module):
    """哈希寻址记忆 (HashMemory).

    不用注意力打分: 通过可学习投影把查询映射到哈希码, 取 argmax 定位到固定记忆槽,
    读取走稀疏 gather, 写入走稀疏 scatter-add. 全程无 softmax / 无 QK^T.

    复杂度: 读取与写入均为 O(B · L · C · S · D), 其中 C,S 固定 ->
            单步计算相对 n 为 O(n) (无 n^2 项); 记忆容量形状 (B,C,S,D) 与 n 解耦 (O(1) 空间)。

    Args:
        dim: 特征维度 D.
        num_tables: 哈希表数 C.
        slots_per_table: 每表槽数 S.
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 64,
        slots_per_table: int = 128,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_tables > 0 and slots_per_table > 0
        self.dim = dim
        self.num_tables = num_tables
        self.slots_per_table = slots_per_table

        # 可学习投影: 把 D 维投影到 (C * S) 的离散哈希空间
        self.addr_proj = nn.Linear(dim, num_tables * slots_per_table, bias=False)
        self.write_val_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.RMSNorm(dim, eps=eps)

    def _init_memory(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            batch, self.num_tables, self.slots_per_table, self.dim,
            device=ref.device, dtype=ref.dtype,
        ) * 0.02

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, D); memory_state: (B, C, S, D) or None."""
        residual = x
        B, L, D = x.shape
        if memory_state is None:
            memory = self._init_memory(B, x)
        else:
            memory = memory_state

        # ---- 读取: 最近邻哈希定位 (无 softmax) ----
        logits = self.addr_proj(x)                       # (B, L, C*S)
        logits = logits.reshape(B, L, self.num_tables, self.slots_per_table)
        # 直接取 argmax 定位槽位 (确定性, 非概率)
        addr = torch.argmax(logits, dim=-1)              # (B, L, C)
        # 用 one-hot 稀疏 gather 读出 (避免 softmax 加权的连续注意力)
        oh = F.one_hot(addr, num_classes=self.slots_per_table)  # (B,L,C,S)
        readout = torch.einsum("blcs,bcsd->bld", oh, memory)    # (B, L, D)
        out = self.output_proj(readout)

        # ---- 写入: 同一地址 scatter-add (NTM 风格, 无擦除概率) ----
        wv = self.write_val_proj(x)                     # (B, L, D)
        # 每个 (b, l, c) 对应一个待写入向量: 用地址处的写向量, 形状 (B, L, C, D)
        val = wv.reshape(B, L, 1, D).repeat_interleave(self.num_tables, dim=2) \
            .reshape(-1, D)                              # (B*L*C, D)
        # 扁平槽索引 (B*L*C,): idx = b*(C*S) + c*S + addr[b,l,c]
        ar_b = torch.arange(B).reshape(B, 1, 1)         # (B,1,1)
        ar_c = torch.arange(self.num_tables).reshape(1, 1, self.num_tables)  # (1,1,C)
        base_b = ar_b * (self.num_tables * self.slots_per_table)  # (B,1,1)
        base_c = ar_c * self.slots_per_table            # (1,1,C)
        addr3 = addr.reshape(B, L, self.num_tables)     # (B, L, C)
        flat_idx = (base_b + base_c + addr3).reshape(-1)  # (B*L*C,)
        flat_mem = memory.reshape(
            B * self.num_tables * self.slots_per_table, self.dim
        ).clone()
        # 累加写入 (scatter-add, 仅聚合, 不删除)
        flat_mem.index_add_(0, flat_idx, val)
        new_memory = flat_mem.reshape(B, self.num_tables, self.slots_per_table, self.dim)

        return residual + self.norm(out), new_memory

    def complexity(self, seq_len: int, batch: int = 1) -> Dict[str, float]:
        n, C, S, d = seq_len, self.num_tables, self.slots_per_table, self.dim
        # 地址投影 + gather + 写入: 近似 O(B·L·C·S·d)
        hm_flops = batch * n * C * S * d
        attn_flops = 2.0 * n * n * d * batch
        return {
            "hm_flops": hm_flops,
            "attn_flops": attn_flops,
            "speedup_vs_attention": attn_flops / max(hm_flops, 1.0),
            "hm_big_o": "O(n) wrt n; capacity O(1) wrt n",
            "attn_big_o": "O(n^2)",
        }


# ======================================================================
# 3. 前馈 + 点阵块
# ======================================================================
class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.fc2 = nn.Linear(dim * ratio, dim)
        self.norm = nn.RMSNorm(dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class LatticeBlock(nn.Module):
    """一个点阵块: 谐振耦合 + 哈希记忆 + 前馈, 无任何注意力.

    Args:
        cfg: LatticeConfig
        use_checkpoint: 是否对前向做梯度检查点（recompute）。
            开启后，反向时重新计算块内前向，激活内存从 O(L·D·layers)
            降到 O(L·D)，显存随层数增长从线性变为常数级，适合 0.966B~1B
            专家在 32GB 显存内训练（配合 fp16）。
    """

    def __init__(self, cfg, use_checkpoint: bool = False):
        super().__init__()
        d = cfg.hidden_size
        self.rc = ResonantCoupling(d, num_bands=cfg.num_bands, kernel_size=cfg.kernel_size)
        self.hm = HashMemory(d, num_tables=cfg.num_tables, slots_per_table=cfg.slots_per_table)
        self.ffn = FeedForward(d, ratio=cfg.ffn_ratio)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x, memory_state=None):
        x = self.rc(x)
        x, memory_state = self.hm(x, memory_state)
        x = x + self.ffn(x)
        return x, memory_state

    def forward(self, x, memory_state=None):
        if self.use_checkpoint and x.requires_grad:
            # 梯度检查点：用闭包捕获子模块，反向时重算
            from simplcore.tensor.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, memory_state)
        return self._forward_impl(x, memory_state)


# ======================================================================
# 4. 完整语言模型
# ======================================================================
class LatticeConfig:
    """80B 级纯中文语料模型的默认配置 (可调)."""

    def __init__(
        self,
        vocab_size: int = 150000,      # 纯中文: 大字表 (汉字 + 词 + 子词)
        hidden_size: int = 8192,       # 80B 量级隐藏维度
        num_layers: int = 104,
        num_bands: int = 8,
        kernel_size: int = 31,
        num_tables: int = 64,
        slots_per_table: int = 128,
        ffn_ratio: int = 4,
        max_seq_len: int = 1_048_576,  # 1B 上下文目标
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.num_tables = num_tables
        self.slots_per_table = slots_per_table
        self.ffn_ratio = ffn_ratio
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class LatticeLM(nn.Module):
    """点阵谐振语言模型 (Lattice Resonance Language Model).

    完全不含 self-attention / softmax 注意力 / positional encoding / FFT。

    Args:
        cfg: LatticeConfig
        use_checkpoint: 是否对所有 LatticeBlock 启用梯度检查点（省显存）。
        dtype: 权重 dtype（torch.float32 / torch.float16）。
            用 float16 时，权重以 fp16 存储（省一半显存/带宽），
            优化器维护 fp32 主副本（混合精度训练，见 ExpertTrainer）。
    """

    def __init__(self, cfg: LatticeConfig, use_checkpoint: bool = False,
                 dtype=None):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.use_checkpoint = use_checkpoint
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList(
            [LatticeBlock(cfg, use_checkpoint=use_checkpoint)
             for _ in range(cfg.num_layers)]
        )
        self.final_norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if dtype is not None:
            self.to(dtype)

    def forward(self, input_ids, memory_state=None):
        B, L = input_ids.shape
        x = self.embed(input_ids)
        for block in self.blocks:
            x, memory_state = block(x, memory_state)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "memory_state": memory_state}

    # ── HashMemory 记忆态存取（容错续训用）──
    def get_memory_state(self) -> Optional[torch.Tensor]:
        """聚合各 block 的 HashMemory 状态（当前为 None，HM 在 forward 内重建）。

        若未来 HM 改为跨步持久记忆，此处返回真实状态；当前返回 None
        表示每步独立初始化，续训无需恢复记忆。
        """
        return None

    def set_memory_state(self, state) -> None:
        """设置 HashMemory 记忆态（当前为 no-op，见 get_memory_state）。"""
        pass

    def half(self):
        """转为 fp16 权重（混合精度训练入口）。"""
        return self.to(torch.float16)

    def to(self, dtype):
        """递归转换所有参数 dtype。"""
        from simplcore.tensor.nn import Parameter
        for p in self.parameters():
            if isinstance(p, Parameter):
                p.data = p.data.astype(dtype)
        return self


def build_lattice_80b() -> LatticeLM:
    """构建 80B 级模型实例 (参数规模由 config 决定, 此处仅构造结构)."""
    return LatticeLM(LatticeConfig())


def parameter_count(cfg: LatticeConfig) -> int:
    """粗略估计参数量 (不含 lm_head 与 embedding 共享视角的重复统计)."""
    d = cfg.hidden_size
    per_block = (
        # ResonantCoupling: center + width + kernel(num_bands,1,K) + out_scale
        cfg.num_bands + cfg.num_bands + cfg.num_bands * cfg.kernel_size + 1
        # HashMemory: addr_proj(D, C*S) + write_val_proj(D,D) + output_proj(D,D)
        + d * (cfg.num_tables * cfg.slots_per_table) + d * d + d * d
        # FFN: D*(D*ratio) + D*ratio*D
        + d * (d * cfg.ffn_ratio) + (d * cfg.ffn_ratio) * d
    )
    total = (
        cfg.vocab_size * d            # embedding
        + cfg.num_layers * per_block
        + d                           # final_norm
        + d * cfg.vocab_size          # lm_head
    )
    return total


if __name__ == "__main__":
    cfg = LatticeConfig()
    model = build_lattice_80b()
    print("LatticeLM 构建成功, 估算参数量: %.3f B" % (parameter_count(cfg) / 1e9))
    rc = ResonantCoupling(64, num_bands=4, kernel_size=15)
    hm = HashMemory(64, num_tables=8, slots_per_table=16)
    print("RC @4096 speedup:", rc.complexity(4096)["speedup_vs_attention"])
    print("HM @4096 speedup:", hm.complexity(4096)["speedup_vs_attention"])
    print("OK")
