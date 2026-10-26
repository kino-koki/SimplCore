# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""复数位置编码 (Complex Positional Encoder, CPE).

核心思想:
  一种基于复数表示的绝对位置编码。通过相位编码 (复指数) 与可学习
  酉/耦合线性变换, 生成位置相关的复数特征, 并投影到模型维度。

三步建模 (均为线性代数, 非量子计算):
  1. 相位编码: 将位置编码为复指数 exp(i · φ(pos)), 并按可学习的每基频频率
     构造初始复振幅;
  2. 可学习酉变换: 施加可学习酉矩阵 U = exp(i · H), 其中 H 为 Hermite 矩阵;
  3. 可学习耦合: 基频位置间的成对线性相互作用 (可学习耦合矩阵)。

随后将复态 (实部 + 虚部) 投影到模型维度 D, 与输入做残差相加。
提供 amplitude_distribution() 给出特征幅值平方归一化分布 (去除伪科学表述)。
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from simplcore.tensor.tensor import _add_grad


class ComplexPositionalEncoder(nn.Module):
    """复数位置编码 (Complex Positional Encoder).

    基于复数表示的绝对位置编码: 在 2^n_basis 维复数空间中演化位置相关的复振幅,
    再投影回模型维度 D 并以残差方式注入输入。

    Args:
        dim: 模型特征维度 D.
        num_basis: 基频数量 n (复数空间维度 = 2^n).
        num_layers: 变换层数 (酉变换 + 耦合 的重复次数).
        dropout: 输出编码的 dropout 概率.
    """

    def __init__(
        self,
        dim: int,
        num_basis: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_basis >= 1
        self.dim = dim
        self.num_basis = num_basis
        self.basis_dim = 2 ** num_basis
        self.num_layers = num_layers
        H = self.basis_dim

        # 可学习相位频率 (每基频一个)
        self.phase_freqs = nn.Parameter(torch.randn(num_basis) * 0.1)

        # 可学习酉变换: 每层一个可学习酉矩阵 U = exp(i · H)
        # 以实部/虚部参数化 Hermite 矩阵 (实部对称, 虚部反对称由 Hermite 化保证)
        # 合并为单一张量 (num_layers, H, H) 以兼容无 ParameterList 的自研后端
        self.unitary_re = nn.Parameter(torch.randn(num_layers, H, H) * 0.02)
        self.unitary_im = nn.Parameter(torch.randn(num_layers, H, H) * 0.02)

        # 耦合: 每层一个可学习耦合矩阵 (成对线性相互作用)
        self.coupling = nn.Parameter(torch.randn(num_layers, H, H) * 0.01)

        # 复态 (实部 + 虚部) -> 模型维度
        self.proj = nn.Linear(2 * H, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    # 内部算子
    # ------------------------------------------------------------------
    def _learnable_unitary(
        self, re: torch.Tensor, im: torch.Tensor
    ) -> torch.Tensor:
        """构造酉矩阵 U = exp(i · H), 其中 H 为 Hermite 矩阵 (线性变换, 非量子门).

        注: 自研 NumPy 后端无复数自动微分, 故用 numpy 计算 (前向, 不接 autograd)。

        H = (W + W†) / 2 保证 Hermite; U = exp(i·H) 为酉矩阵。

        Args:
            re: (H, H) 实部参数 (Tensor).
            im: (H, H) 虚部参数 (Tensor).

        Returns:
            (H, H) 复酉矩阵 (numpy complex64).
        """
        W = (re.data + 1j * im.data).astype(np.complex64)  # (H, H) complex
        H_mat = (W + W.conj().T) / 2.0  # Hermite 化
        # 酉矩阵 U = exp(i·H): 通过特征分解实现 (numpy 2.x 已移除 np.linalg.matrix_exp)
        # H = V · diag(λ) · V^{-1}  =>  exp(i·H) = V · diag(exp(i·λ)) · V^{-1}
        eigvals, eigvecs = np.linalg.eig(H_mat)
        exp_diag = np.exp(1j * eigvals)
        U = eigvecs @ np.diag(exp_diag) @ np.linalg.inv(eigvecs)
        return U.astype(np.complex64)  # 酉矩阵

    def _coupling_matrix(self, layer: int) -> np.ndarray:
        """构造归一化的可学习耦合矩阵 (线性相互作用).

        用 numpy 计算 (前向, 不接 autograd)。

        Returns:
            (H, H) 实耦合矩阵 (行归一化为概率)。
        """
        E = self.coupling[layer].data.astype(np.float32)
        E_sym = (E + E.T) / 2.0  # 对称化
        # 行 softmax
        E_sym = E_sym - E_sym.max(axis=-1, keepdims=True)
        ex = np.exp(E_sym)
        return ex / ex.sum(axis=-1, keepdims=True)

    def _build_freq_vector(self) -> torch.Tensor:
        """由每基频频率构造各基态的频率向量.

        基态 |k⟩ 的频率 = Σ_j phase_freqs[j] · bit_j(k),
        即每个置位基频贡献一个可学习频率。

        Returns:
            (H,) 频率向量。
        """
        n = self.num_basis
        H = self.basis_dim
        # 频率向量是常数 (由可学习 phase_freqs 组合), 用 numpy 整数运算构造后转 Tensor
        # (自研后端 Tensor 不支持 >> 位运算, 且此处不依赖输入梯度)
        freqs_np = np.zeros(H, dtype=np.float32)
        k_np = np.arange(H, dtype=np.int64)
        for j in range(n):
            bit = (k_np >> j) & 1
            freqs_np = freqs_np + float(self.phase_freqs[j].data) * bit.astype(np.float32)
        freq = torch.tensor(freqs_np)  # (H,)
        return freq

    def _phase_encode(self, positions: torch.Tensor) -> np.ndarray:
        """位置相关的复指数相位编码 (numpy 复数, 前向).

        Args:
            positions: (L,) 位置索引 (浮点 Tensor).

        Returns:
            (L, H) 复振幅态 (numpy complex64, 归一化)。
        """
        H = self.basis_dim
        pos = positions.data.astype(np.float32)  # (L,)
        k_idx = np.arange(1, H + 1, dtype=np.float32)  # (H,)
        freq = self._build_freq_vector().data.astype(np.float32)  # (H,)

        # 基础复指数相位: φ = 2π · pos / 2^n_basis
        phi = (2.0 * math.pi * pos / H).reshape(-1, 1)  # (L, 1)
        # 角度 = φ · (1 + freq) · k
        angle = phi * (1.0 + freq).reshape(1, -1) * k_idx.reshape(1, -1)  # (L, H)

        # 复振幅: exp(i · angle) / sqrt(H)
        psi = (np.cos(angle) + 1j * np.sin(angle)).astype(np.complex64) / math.sqrt(H)
        return psi  # (L, H) complex

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """为输入添加复数绝对位置编码.

        实现说明 (诚实): 自研 NumPy 后端暂无复数自动微分, 因此复数链路
        (相位编码 → 酉变换 → 耦合) 用 numpy 计算, 仅前向可运行; 返回
        的编码张量不携带梯度图。参数 (phase_freqs / unitary_* / coupling)
        仍可定义, 端到端训练需接入复数 autograd 后方可启用反向。

        Args:
            x: (B, L, D) 输入特征.
            positions: (L,) 位置索引; 若为 None 则使用 arange(L).

        Returns:
            (B, L, D) 复数位置编码后的特征 (含残差, 不接梯度图).
        """
        B, L, D = x.shape
        if positions is None:
            positions = torch.arange(L, dtype=torch.float32)
        else:
            positions = positions.to(dtype=torch.float32)
            if positions.dim() == 0:
                positions = positions.unsqueeze(0)

        # 1. 相位编码 (numpy 复数)
        psi = self._phase_encode(positions)  # (L, H) complex64

        # 2 & 3. 逐层施加酉变换 (叠加) 与耦合
        for layer in range(self.num_layers):
            U = self._learnable_unitary(
                self.unitary_re[layer], self.unitary_im[layer]
            )  # (H, H) complex, 酉
            psi = psi @ U.T  # 叠加: 对每行状态施加酉变换
            C = self._coupling_matrix(layer)  # (H, H) 实
            psi = psi @ (C + 0j)  # 耦合: 成对线性相互作用
            # 重归一化为单位模长
            norm = np.sqrt((np.abs(psi) ** 2).sum(axis=-1, keepdims=True)) + 1e-8
            psi = psi / norm

        # 投影到模型维度: 复态 (实部 + 虚部) -> D
        feat = np.concatenate([psi.real, psi.imag], axis=-1).astype(np.float32)  # (L, 2H)
        feat_t = torch.tensor(feat)  # (L, 2H)
        encoding = self.proj(feat_t)  # (L, D)
        encoding = encoding.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        encoding = self.norm(encoding)

        # 返回不接梯度图 (复数链路未经 autograd)
        out = (x.data + self.dropout(encoding).data).astype(np.float32)
        return torch.tensor(out)

    # ------------------------------------------------------------------
    # 幅值分布
    # ------------------------------------------------------------------
    def amplitude_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """幅值平方分布: 计算特征幅值平方归一化分布 (统计用途, 非量子测量).

        将输入视为复振幅的实/虚表示, 计算 p_i ∝ |x_i|² 并归一化,
        用于分析各维度的激活强度分布。

        Args:
            x: (..., D) 输入特征 (视为振幅).

        Returns:
            dict:
              - "probabilities": (..., D) 归一化概率分布;
              - "collapsed": (...,) 最大幅值维度索引 (按概率采样)。
        """
        amp = x.abs()  # 振幅
        probs = amp ** 2  # p ∝ |x|²
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        # 取最大幅值维度作为代表索引
        flat = probs.reshape(-1, probs.shape[-1])
        idx = torch.argmax(flat, dim=-1).reshape(probs.shape[:-1])
        return {"probabilities": probs, "index": idx}

    # ------------------------------------------------------------------
    # 参数统计
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """统计可训练参数数量.

        Returns:
            可训练参数总数。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _PrimitiveComposer(nn.Module):
    """CPE 原语组合层。

    通过一组可学习 primitive 原语库和 top-k 组合门，把小参数库组合成
    等效更宽的表示，再投影回模型维度。
    """

    def __init__(
        self,
        d_model: int,
        n_primitives: int,
        primitive_size: int,
        k_compose: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if k_compose < 1 or k_compose > n_primitives:
            raise ValueError("k_compose 必须在 [1, n_primitives] 内。")
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.primitive_size = primitive_size
        self.k_compose = k_compose

        self.norm = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_primitives)
        self.primitives = nn.Parameter(torch.randn(n_primitives, primitive_size) * 0.02)
        self.in_proj = nn.Linear(d_model, primitive_size)
        self.out_proj = nn.Linear(primitive_size, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        logits = self.router(h)  # (B, L, P)
        top_values, top_indices = logits.topk(self.k_compose, dim=-1)  # (B, L, k)

        # 自研后端 topk 不可微: 重建可微 top_values (父节点 logits) 并手动反向
        Bs, Ls, ks = top_values.shape
        lg_data = logits.data
        ti_data = top_indices.data.astype("int64")
        topk_vals = logits._make_out(top_values.data, "cpe_topk", (logits,))

        def _topk_backward():
            if topk_vals.grad is None:
                return
            g = np.asarray(topk_vals.grad, dtype=lg_data.dtype)  # (B, L, k)
            grad_lg = np.zeros_like(lg_data)
            for b in range(Bs):
                for l in range(Ls):
                    for j in range(ks):
                        p = int(ti_data[b, l, j])
                        grad_lg[b, l, p] += g[b, l, j]
            _add_grad(logits, grad_lg)

        topk_vals._backward = _topk_backward

        # 自研后端 F.softmax 不接 autograd, 用 exp/sum/div 实现可微 softmax
        weights = (topk_vals.exp() / topk_vals.exp().sum(dim=-1, keepdim=True))  # (B, L, k)
        idx_data = top_indices.data.astype("int64")
        ps_data = self.primitives.data  # (P, F)
        buf = np.zeros((Bs, Ls, ks, ps_data.shape[-1]), dtype=ps_data.dtype)
        for b in range(Bs):
            for l in range(Ls):
                for j in range(ks):
                    p = int(idx_data[b, l, j])
                    buf[b, l, j, :] = ps_data[p, :]
        selected = self.primitives._make_out(buf, "cpe_gather", (self.primitives,))

        # 手动反向: 把 selected 的梯度按 topk_idx 加回 primitives (稀疏 scatter-add)
        def _gather_backward():
            if selected.grad is None:
                return
            g = np.asarray(selected.grad, dtype=ps_data.dtype)  # (B, L, k, F)
            grad_ps = np.zeros_like(ps_data)
            for b in range(Bs):
                for l in range(Ls):
                    for j in range(ks):
                        p = int(idx_data[b, l, j])
                        grad_ps[p, :] += g[b, l, j, :]
            _add_grad(self.primitives, grad_ps)

        selected._backward = _gather_backward

        composed = (weights.unsqueeze(-1) * selected).sum(dim=2)  # (B, L, F)
        local = self.in_proj(h)
        mixed = torch.tanh(local + composed)
        x = residual + self.dropout(self.out_proj(mixed))
        return x + self.dropout(self.ffn(x))


class SimplCoreWithCPE(nn.Module):
    """带 CPE 复数位置编码与原语组合的轻量语言模型。

    该模型用于验证“参数效率 + 可训练”的 CPE 路线：
    token 嵌入先注入 CPE 位置/相位编码，再经过多层 primitive composer，
    最后输出语言模型 logits。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_primitives: int = 256,
        primitive_size: int = 64,
        k_compose: int = 4,
        max_seq_len: int = 2048,
        num_basis: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        if num_basis is None:
            num_basis = max(2, min(8, math.ceil(math.log2(max(max_seq_len, 2))) - 1))

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.cpe = ComplexPositionalEncoder(d_model, num_basis=num_basis, num_layers=2, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                _PrimitiveComposer(
                    d_model=d_model,
                    n_primitives=n_primitives,
                    primitive_size=primitive_size,
                    k_compose=k_compose,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        h = self.token_embed(input_ids)
        h = self.cpe(h)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
            )
        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        cur = input_ids
        for _ in range(max_new_tokens):
            out = self(cur[:, -self.max_seq_len :])
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


def compute_parameter_efficiency(
    d_model: int,
    n_layers: int,
    n_primitives: int,
    primitive_size: int,
    k_compose: int,
    vocab_size: int = 32000,
) -> Dict[str, float]:
    """估算 CPE 原语组合相对标准 Transformer FFN 的参数效率。"""
    std_layer = 12 * d_model * d_model
    std_total = vocab_size * d_model + n_layers * std_layer

    primitive_bank = n_primitives * primitive_size
    composer_layer = (
        d_model * n_primitives
        + d_model * primitive_size
        + primitive_size * d_model
        + 8 * d_model * d_model
    )
    cpe_total = vocab_size * d_model + primitive_bank + n_layers * composer_layer
    equivalent_capacity = vocab_size * d_model + n_layers * (
        std_layer + k_compose * n_primitives * primitive_size
    )

    return {
        "std_transformer_params_B": std_total / 1e9,
        "cpe_SimplCore_params_B": cpe_total / 1e9,
        "equivalent_capacity_B": equivalent_capacity / 1e9,
        "parameter_efficiency_ratio": equivalent_capacity / max(cpe_total, 1),
        "compression_ratio": std_total / max(cpe_total, 1),
    }


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("ComplexPositionalEncoder 测试")
    print("=" * 60)

    enc = ComplexPositionalEncoder(dim=16, num_basis=4, num_layers=2, dropout=0.0)
    print(f"复数空间维度: {enc.basis_dim}, 可训练参数: {enc.count_parameters()}")

    x = torch.randn(2, 8, 16, requires_grad=True)
    y = enc(x)
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y.shape)}")
    assert y.shape == (2, 8, 16)
    assert not bool(np.isnan(y.data).any()), "输出含 NaN"

    # 带显式 positions
    pos = torch.arange(8)
    y2 = enc(x, positions=pos)
    assert y2.shape == (2, 8, 16)
    assert not bool(np.isnan(y2.data).any())

    # 幅值分布
    res = enc.amplitude_distribution(y)
    print(
        f"幅值分布: 概率 {tuple(res['probabilities'].shape)}, "
        f"代表索引 {tuple(res['index'].shape)}"
    )
    assert res["probabilities"].shape == (2, 8, 16)
    assert res["index"].shape == (2, 8)
    assert np.allclose(
        res["probabilities"].sum(dim=-1).data, np.ones((2, 8)), atol=1e-5
    )

    # 注: 复数链路在自研 NumPy 后端不接 autograd, 此处不验证复数参数梯度;
    # 仅验证前向形状与无 NaN (诚实反映后端能力边界).
    print("CPE 前向可运行 (复数链路不接 autograd, 详见模块说明)")

    print("CPE smoke test PASSED")
