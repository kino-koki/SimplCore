# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""mathproof —— 架构数学主张的机器可验证证明（符号 + fp64 数值）。

与 ``theory.py`` 的分工
-----------------------
``theory.py`` 守护"模块行为符合文档主张"（工程断言，fp32 口径）；
本模块证明"数学本身成立"（derivations，fp64 / 符号口径）：

* 闭式阻尼振荡核 ↔ 二阶 IIR 递推的**等价性**（sympy 符号 + fp64）；
* 伴随矩阵幂 = 闭式解；特征值模长 = ρ（稳定性谱）；
* **分块携带律**（``chunked_iir_scan`` 的数学基础）与叠加原理；
* FFT 零填充 N=2L 的**无混叠**精确性（对照 O(L²) 直接卷积）；
* BIBO 稳定性与核 ℓ₁ 界 ``Σ|ψ| ≤ 1/(1−ρ)``；
* 流式 == 并行（mixer 级，fp64，gates 开/关两种口径）；
* 证据门 cumsum / 运行和恒等式；闸门有界性（三角不等式）；
* 时移协变性（门关）与非协变性（门开，设计使然）；
* O(1) 状态的**充分性**（马尔可夫性：同状态 ⇒ 未来逐位一致）；
* 复杂度公式的逐项重推导审计；HBM 账本代数；冻结基座梯度隔离。

全部证明**不依赖任何硬件**：符号部分用 sympy（缺省时退化为 fp64 数值），
数值部分统一 fp64、容差按推导给出。CLI：``python -m simplcore.core.mathproof``。

诚实边界
--------
数值证明在 fp64 与小尺寸上进行 —— 它证明**数学结构**正确，
不证明 1B 实机精度/吞吐（那需要卡，见 ``campaign.py`` Phase 0）。
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from ..deploy.campaign import OPTIMIZER_BYTES_PER_PARAM, chunked_train_bytes, optimizer_state_bytes
from ..model.config import SimplCoreConfig, get_preset
from ..deploy.deploy_budget import (DEPLOY_HBM_CAP_1M, DEFAULT_CHUNK, DEFAULT_WORKSPACE_TENSORS,
                            deploy_bytes)
from ..training.envlearn import WeightBudget
from ..arch.gates import evidence_weight_step, evidence_weights, joint_disagreement, uncertainty_weights
from .scaling import (_long_range_flops, embedding_params, infer_flops_per_token,
                      non_embedding_params, stream_state_bytes)
from ..arch.sgc import (RMSNorm, SpectralGlobalConv, fft_causal_conv,
                        theta_clamp_floor)

GIB = 1024 ** 3

__all__ = [
    "ProofResult",
    "run_all",
    "main",
]


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProofResult:
    """一项数学证明的结果。"""

    pid: str            # 证明编号，如 "P03"
    claim: str          # 被证明的数学命题
    method: str         # symbolic / numeric-fp64 / algebraic / numeric-fp32-impl
    passed: bool
    residual: str       # 残差或关键量（人类可读）
    note: str = ""


def _frac(a: float, b: float) -> float:
    """相对残差。"""
    return abs(a) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# P01 闭式核 ↔ 二阶递推（sympy 符号 + fp64 数值）
# ---------------------------------------------------------------------------
def _proof_closed_form() -> ProofResult:
    """ψ_τ = ρ^τ·sin((τ+1)θ) 满足 y_τ = 2ρcosθ·y_{τ−1} − ρ²·y_{τ−2} + sinθ·δ_τ。

    初条件：ψ_0 = sinθ（脉冲驱动），ψ_1 = 2ρcosθ·ψ_0 = ρ·sin2θ。
    符号部分：令 x = e^{iθ}，sin(kθ) = (x^k − x^{−k})/(2i)，cosθ = (x + x^{−1})/2，
    则递推残差是 (ρ, x) 的 Laurent 多项式 —— 对 t = 1..12 逐个 ``cancel`` 到恒 0。
    """
    # --- 符号证明（整数 t，符号 rho/x） -------------------------------------
    symbolic = "unavailable"
    sym_ok = True
    try:
        import sympy as sp
        rho, x = sp.symbols("rho x", positive=True)
        iu = sp.I

        def psi(t: int):
            return rho ** t * (x ** (t + 1) - x ** (-(t + 1))) / (2 * iu)

        a = 2 * rho * (x + 1 / x) / 2
        b = -rho ** 2
        # 初条件 ψ_1 = a·ψ_0（t=1 时 y_{-1}:=0，脉冲在 t=0）
        r1 = sp.cancel(sp.together(psi(1) - a * psi(0)))
        sym_ok = sym_ok and (sp.simplify(r1) == 0)
        # 递推 t = 2..12（无输入）
        for t in range(2, 13):
            res = psi(t) - a * psi(t - 1) - b * psi(t - 2)
            sym_ok = sym_ok and (sp.simplify(sp.cancel(sp.together(res))) == 0)
        symbolic = "proved" if sym_ok else "FAILED"
    except ImportError:
        sym_ok = True   # 无 sympy 时由数值部分独立承担

    # --- fp64 数值证明（网格 × 长序列） -------------------------------------
    torch.manual_seed(11)
    grid = [(0.30, math.pi / 7), (0.70, math.pi / 3), (0.95, 2 * math.pi / 3),
            (0.99, 5 * math.pi / 6), (0.999, math.pi / 2)]
    worst = 0.0
    for r, th in grid:
        a = 2.0 * r * math.cos(th)
        b = -(r ** 2)
        c = math.sin(th)
        T = 4096
        u = torch.zeros(T, dtype=torch.float64)
        u[0] = 1.0                                   # 单位脉冲
        y1 = y2 = 0.0
        for t in range(T):
            y_new = a * y1 + b * y2 + c * u[t].item()
            y2, y1 = y1, y_new
            closed = (r ** t) * math.sin((t + 1) * th)
            worst = max(worst, abs(y_new - closed))
    ok = sym_ok and worst < 1e-9
    method = "symbolic(sympy)+numeric-fp64" if symbolic == "proved" else "numeric-fp64"
    return ProofResult("P01", "闭式阻尼振荡核满足二阶 IIR 递推（含初条件）",
                       method, ok, f"符号={symbolic}, fp64 最大残差={worst:.3e}",
                       "grid ρ∈{0.3..0.999}, θ∈(0,π), T=4096")


# ---------------------------------------------------------------------------
# P02 伴随矩阵幂 = 闭式解；特征值模长 = ρ
# ---------------------------------------------------------------------------
def _proof_companion_matrix() -> ProofResult:
    """A = [[2ρcosθ, −ρ²],[1,0]]，s_t=[y_t; y_{t−1}]。

    (a) A^n·[sinθ; 0] = [ρⁿ·sin((n+1)θ); ρ^{n−1}·sin(nθ)]（n ≤ 128）；
    (b) eig(A) = ρ·e^{±iθ}，模长恰为 ρ < 1（谱半径 = 衰减率）。
    这是分块扫描状态转移 A^c 的数学基础（P03 直接使用）。
    """
    torch.manual_seed(12)
    grid = [(0.30, math.pi / 7), (0.70, math.pi / 3), (0.95, 2 * math.pi / 3),
            (0.99, 5 * math.pi / 6), (0.999, math.pi / 2)]
    worst_pow, worst_eig = 0.0, 0.0
    for r, th in grid:
        a, b = 2.0 * r * math.cos(th), -(r ** 2)
        A = torch.tensor([[a, b], [1.0, 0.0]], dtype=torch.float64)
        s0 = torch.tensor([math.sin(th), 0.0], dtype=torch.float64)
        for n in range(1, 129):
            lhs = torch.matrix_power(A, n) @ s0
            rhs = torch.tensor([r ** n * math.sin((n + 1) * th),
                                r ** (n - 1) * math.sin(n * th)], dtype=torch.float64)
            worst_pow = max(worst_pow, (lhs - rhs).abs().max().item())
        eigs = torch.linalg.eigvals(A)
        worst_eig = max(worst_eig, (eigs.abs() - r).abs().max().item())
    ok = worst_pow < 1e-9 and worst_eig < 1e-12
    return ProofResult("P02", "伴随矩阵幂给出闭式解；谱半径 |eig(A)| = ρ < 1",
                       "numeric-fp64", ok,
                       f"矩阵幂最大残差={worst_pow:.3e}, 特征值模长残差={worst_eig:.3e}",
                       "grid 5×(ρ,θ), n=1..128")


# ---------------------------------------------------------------------------
# P03 分块携带律 + 叠加原理（chunked_iir_scan 的数学基础）
# ---------------------------------------------------------------------------
def _proof_chunk_carry_law() -> ProofResult:
    """对常系数二阶递推 s_t = A·s_{t−1} + [c·u_t; 0]：

    (a) 闭式分块计算 y_t = (A^t·s₀)₁ + Σ_{j≤t} (A^{t−j}·[c·u_j; 0])₁
        == 逐步模拟（fp64, 残差 ~1e-12）；
    (b) **叠加原理**：out(s₀, u) − out(0, u) == 齐次响应 (A^t·s₀)₁（线性）；
    (c) **两块组合**：state(c₁+c₂) = state(c₂) ∘ state(c₁)，
        输出逐位一致（同一算子序列 → torch.equal）。

    这就是后端 ``chunked_iir_scan`` + ``stream_state_*`` 必须满足的合同数学。
    """
    torch.manual_seed(13)
    grid = [(0.85, math.pi / 5), (0.98, 2.3), (0.60, 0.7)]
    worst_closed, worst_super = 0.0, 0.0
    bit_exact = True
    for r, th in grid:
        a, b, c = 2.0 * r * math.cos(th), -(r ** 2), math.sin(th)
        A = torch.tensor([[a, b], [1.0, 0.0]], dtype=torch.float64)
        Bv = torch.tensor([c, 0.0], dtype=torch.float64)
        T1, T2 = 13, 9
        u1 = torch.randn(T1, dtype=torch.float64)
        u2 = torch.randn(T2, dtype=torch.float64)
        s0 = torch.randn(2, dtype=torch.float64)

        def run(inputs: torch.Tensor, st: torch.Tensor):
            ys = []
            for t in range(inputs.shape[0]):
                st = A @ st + Bv * inputs[t].item()
                ys.append(st[0].item())
            return ys, st

        # (i) 逐步模拟（携带 s0）
        ys_seq, s_end = run(torch.cat([u1, u2]), s0.clone())
        # (ii) 闭式分块：齐次 + 特解
        ys_closed = []
        uu = torch.cat([u1, u2])
        for t in range(uu.shape[0]):
            y = (torch.matrix_power(A, t + 1) @ s0)[0].item()
            for j in range(t + 1):
                y += (torch.matrix_power(A, t - j) @ (Bv * uu[j].item()))[0].item()
            ys_closed.append(y)
        worst_closed = max(worst_closed, max(abs(p - q) for p, q in zip(ys_seq, ys_closed)))
        # (iii) 叠加：out(s0,u) − out(0,u) == 齐次响应
        ys_zero, _ = run(torch.cat([u1, u2]), torch.zeros(2, dtype=torch.float64))
        hom = [(torch.matrix_power(A, t + 1) @ s0)[0].item() for t in range(uu.shape[0])]
        sup = [p - q for p, q in zip(ys_seq, ys_zero)]
        worst_super = max(worst_super, max(abs(p - q) for p, q in zip(sup, hom)))
        # (iv) 两块组合：逐位一致（第二块的输出与拼接运行的后 T2 个位置比较）
        _, s1 = run(u1, s0.clone())
        ys_c2, s_c2 = run(u2, s1.clone())
        ys_cat, s_cat = run(torch.cat([u1, u2]), s0.clone())
        if ys_c2 != ys_cat[T1:] or not torch.equal(s_c2, s_cat):
            bit_exact = False
    ok = worst_closed < 1e-9 and worst_super < 1e-9 and bit_exact
    return ProofResult("P03", "分块携带律：闭式分块==逐步模拟；叠加原理成立；两块组合逐位一致",
                       "numeric-fp64", ok,
                       f"闭式残差={worst_closed:.3e}, 叠加残差={worst_super:.3e}, "
                       f"组合逐位一致={bit_exact}",
                       "这是 backend_contract.chunked_iir_scan 的合同数学")


# ---------------------------------------------------------------------------
# P04 FFT 因果卷积无混叠（N=2L 零填充）
# ---------------------------------------------------------------------------
def _proof_fft_no_aliasing() -> ProofResult:
    """fft_causal_conv == 直接 O(L²) 因果卷积（fp64），对 L∈{1,2,3,5,8,17,64,257}。

    命题：长度 2L 的圆周卷积在"输入长 L、核长 ≤ L、均因果"时无回绕污染，
    故截断前 L 个位置给出**精确**线性卷积 —— 零填充倍数 2 是充分的（紧下界）。
    """
    torch.manual_seed(14)
    worst = 0.0
    for L in (1, 2, 3, 5, 8, 17, 64, 257):
        u = torch.randn(2, L, 3, dtype=torch.float64)
        k = torch.randn(L, 3, dtype=torch.float64)
        y_fft = fft_causal_conv(u, k, torch.float64)
        y_ref = torch.zeros_like(u)
        for t in range(L):
            # y[t] = Σ_{s≤t} k[t−s]·u[s]
            y_ref[:, t] = (k[:t + 1].flip(0).unsqueeze(0) * u[:, :t + 1]).sum(dim=1)
        worst = max(worst, (y_fft - y_ref).abs().max().item())
    ok = worst < 1e-9
    return ProofResult("P04", "FFT 因果卷积（N=2L 零填充）== 直接卷积，无混叠",
                       "numeric-fp64", ok, f"最大残差={worst:.3e}",
                       "L∈{1,2,3,5,8,17,64,257}（含素数与非 2 幂）")


# ---------------------------------------------------------------------------
# P05 BIBO 稳定性与核 ℓ₁ 界
# ---------------------------------------------------------------------------
def _proof_bibo_stability() -> ProofResult:
    """ρ = σ(raw) < 1（fp64 数学层对任意有限 raw 严格成立）；
    |ψ_τ| ≤ ρ^τ（包络）；Σ_τ |ψ_τ| ≤ 1/(1−ρ)（|sin| ≤ 1 的几何和）→ BIBO 稳定。

    **实现层卫兵（0.1-α，阈值口径对齐）**：torch fp32 的 sigmoid 在
    raw ≳ **16.636**（= 24·ln2，``exp(x)/(1+exp(x))`` 实现路径**实测**，
    见 ``test_audit_guards.py`` 的饱和测试）时精确饱和到 1.0，极点会落到
    单位圆上（递推失去衰减）；**25·ln2 ≈ 17.33 是解析舍入的充分界**
    （证明按此口径放缩，两数字关系：实测紧界 16.636 < 充分界 17.33）。
    ``SpectralGlobalConv.oscillator_params`` 已加 ``clamp(max=1−1e-6)`` ——
    此处验证：把 raw_rho 推到 ±30，ρ 仍严格 < 1。

    **θ 卫兵（P1，0.1-α 口径修正）**：命题 5 修正后的误差界含
    1/|sinθ_m|，θ→0/π 时界发散。``oscillator_params`` 的**上界**是绝对的
    ``π−1e-3``；**下界**由 ``theta_clamp_floor(decay_max) = min(1e-3,
    π/decay_max)`` 给出——它随 decay_max 让步（logfreq 最慢基的合法初值
    θ_min=π/decay_max 不被钳掉），**不提供固定 1e-3 保证**。

    上游待修清单 P0-2：旧版只在 ``decay_max=64``（floor 恰为 1e-3）上断言
    ``theta_lo >= 1e-3``，把小配置的保证误当成生产配置的保证。这里改为按
    ``(64, 8192, 16384, 4e6)`` 逐档扫描、按各档真实 floor 断言，并在
    ``detail`` 里如实打印每个档位的 ``1/|sin floor|``（默认档约 2608×、
    tuan-2b 档约 1.27e6×）——这些是误差界的实际放大量，不是卫兵失效。
    """
    # (a) fp64 数学层：sigmoid 严格开区间
    raw = torch.tensor([0.0, 5.0, -5.0, 20.0, -20.0], dtype=torch.float64)
    sig = torch.sigmoid(raw)
    math_ok = bool((sig < 1.0).all() and (sig > 0.0).all())
    # (b) 包络与 ℓ₁ 界
    worst_env, worst_l1 = 0.0, 0.0
    tau = torch.arange(65536, dtype=torch.float64)
    for r in (0.3, 0.7, 0.95, 0.99, 0.999):
        for th in (math.pi / 7, math.pi / 3, 2 * math.pi / 3):
            psi = (r ** tau) * torch.sin((tau + 1) * th)
            worst_env = max(worst_env, (psi.abs() - r ** tau).max().item())
            l1 = psi.abs().sum().item()
            worst_l1 = max(worst_l1, l1 - 1.0 / (1.0 - r))
    # (c) 实现层：极端 raw_rho 下 ρ 仍严格 < 1（卫兵生效）
    # (d) θ 卫兵按 decay_max 扫描（P0-2）——旧版只测 64，误把小配置保证
    #     当成生产配置保证。每档按 theta_clamp_floor(dm) 断言，并如实记录
    #     1/|sin floor| 这个误差界放大量。
    DECAY_CASES = (64.0, 8192.0, 16384.0, 4_000_000.0)
    guard_ok = True
    theta_ok = True
    theta_cases = []
    for dm in DECAY_CASES:
        mx = SpectralGlobalConv(dim=4, num_bases=4, decay_max=dm, local_kernel=1,
                                content_gate=False, evidence_gate=False,
                                uncertainty_gate=False)
        floor = theta_clamp_floor(dm)
        with torch.no_grad():
            mx.raw_rho.fill_(30.0)
            rho_hi = mx.oscillator_params()[0]
            mx.raw_rho.fill_(-30.0)
            rho_lo = mx.oscillator_params()[0]
            mx.raw_theta.fill_(30.0)
            theta_hi = mx.oscillator_params()[1]
            mx.raw_theta.fill_(-30.0)
            theta_lo = mx.oscillator_params()[1]
        guard_ok &= bool((rho_hi < 1.0).all() and (rho_lo > 0.0).all()
                         and (rho_hi <= 1.0 - 1e-6).all())
        # 上界绝对；下界随 decay_max 让步，按各档真实 floor 断言
        case_ok = bool((theta_hi <= math.pi - 1e-3 + 1e-9).all()
                       and (theta_lo >= floor - 1e-9).all()
                       and torch.sin(theta_hi).abs().min() >= math.sin(1e-3) - 1e-9)
        theta_ok &= case_ok
        amp = 1.0 / math.sin(floor)
        theta_cases.append(
            f"dm={dm:g}: floor={floor:.3e}, 1/|sin floor|={amp:.1f}×, ok={case_ok}")
        del mx
    ok = math_ok and worst_env <= 0.0 and worst_l1 <= 1e-9 and guard_ok and theta_ok
    return ProofResult("P05", "BIBO 稳定：|ψ_τ| ≤ ρ^τ, Σ|ψ| ≤ 1/(1−ρ)；fp32 饱和与 θ 漂移均被卫兵拦截",
                       "numeric-fp64 + fp32-impl", ok,
                       f"fp64 sigmoid 严格={math_ok}, 包络违反={worst_env:.1e}, "
                       f"ℓ₁ 超界={worst_l1:.1e}, raw=±30 时 ρ<1={guard_ok}, "
                       f"θ 上界绝对/下界随 decay_max 让步，逐档={theta_ok}; "
                       + " | ".join(theta_cases),
                       "饱和阈值：实测 16.636（24·ln2）为紧界、25·ln2≈17.33 为解析充分界；"
                       "卫兵阈值 1−1e-6 高于 ρ₀≈0.99988（decay_max=8192），不影响初始化。"
                       "口径修正：下界不是固定 1e-3；默认档误差界实际放大约 2608×、"
                       "tuan-2b(4e6) 档约 1.27e6×，论文/§3.1 的「固定 1e-3 保证」表述已撤回")


# ---------------------------------------------------------------------------
# P06 流式 == 并行（mixer 级，fp64）
# ---------------------------------------------------------------------------
def _mixer(gates: bool) -> SpectralGlobalConv:
    torch.manual_seed(15)
    return SpectralGlobalConv(dim=16, num_bases=32, decay_max=256, local_kernel=4,
                              content_gate=True, evidence_gate=gates,
                              uncertainty_gate=gates,
                              fft_dtype=torch.float64).double()


def _proof_stream_equals_parallel() -> ProofResult:
    """SpectralGlobalConv 整段前向（FFT 形式）与逐 token 流式（IIR 形式）一致。

    门关：纯线性路径，fp64 下残差应 ~1e-12（FFT 与递推的舍入差）；
    门开：证据门能量和在 fp32 内做 cumsum vs 运行和（**结合顺序不同**），
    残差抬到 ~1e-6 —— 数学上仍是同一和，工程上由 theory.py 的 4e-5 口径守护。
    """
    out = []
    for gates, tol in ((False, 1e-8), (True, 1e-4)):
        mx = _mixer(gates)
        x = torch.randn(2, 65, 16, dtype=torch.float64)   # 奇数长度
        with torch.no_grad():
            y_par = mx(x)
            st = mx.init_state(2, x.device, torch.float64)
            ys = []
            for t in range(x.shape[1]):
                yt, st = mx.step(x[:, t:t + 1, :], st)
                ys.append(yt)
            y_seq = torch.cat(ys, dim=1)
        rel = _frac((y_par - y_seq).abs().max().item(), y_par.abs().max().item())
        out.append((gates, rel, rel < tol))
    ok = all(p for _, _, p in out)
    det = "; ".join(f"gates={'on' if g else 'off'} rel={r:.2e}" for g, r, _ in out)
    return ProofResult("P06", "mixer 级 流式(IIR) == 并行(FFT)（fp64）",
                       "numeric-fp64", ok, det,
                       "门开时残差来自 fp32 能量和的结合顺序，非数学差异")


# ---------------------------------------------------------------------------
# P07 证据门 cumsum / 运行和恒等式
# ---------------------------------------------------------------------------
def _proof_evidence_identity() -> ProofResult:
    """c_t = (1/t)·Σ_{s≤t}‖u_s‖² 的两种计算是**同一个和**：

    (a) fp64 数学层：cumsum vs 逐次折叠，残差 ≤ 1e-12（纯结合顺序舍入）；
    (b) fp64 下闸门权重逐位置相等；
    (c) 实现级（gates.py，fp32 口径）：并行 evidence_weights 与流式
        evidence_weight_step 循环一致（≤ 1e-5）。
    """
    torch.manual_seed(16)
    u = torch.randn(2, 128, 8, dtype=torch.float64)
    e = (u ** 2).sum(dim=-1)
    cs = e.cumsum(dim=1)
    tvec = torch.arange(1, 129, dtype=torch.float64).unsqueeze(0)
    c_par = cs / tvec
    # 逐次折叠
    run = torch.zeros(2, dtype=torch.float64)
    worst = 0.0
    for t in range(128):
        run = run + e[:, t]
        worst = max(worst, (run / (t + 1) - c_par[:, t]).abs().max().item())
    # fp64 闸门权重逐位置相等（手推流式公式对照并行实现）
    g = torch.tensor(1.3, dtype=torch.float64)
    pg = torch.tensor(0.2, dtype=torch.float64)
    bs = torch.tensor(-0.1, dtype=torch.float64)
    w_par = torch.sigmoid(g * torch.log1p(c_par) + pg * torch.log1p(tvec) + bs)
    worst_w = 0.0
    run = torch.zeros(2, dtype=torch.float64)
    for t in range(128):
        run = run + e[:, t]
        c_t = run / (t + 1)
        w_t = torch.sigmoid(g * torch.log1p(c_t) + pg * math.log1p(t + 1) + bs)
        worst_w = max(worst_w, (w_t - w_par[:, t]).abs().max().item())
    # 实现级 fp32 一致性
    u32 = u.float()
    g32, pg32, bs32 = g.float(), pg.float(), bs.float()
    w_impl_par = evidence_weights(u32, g32, pg32, bs32)          # (B,L,1)
    esum = None
    worst_impl = 0.0
    for t in range(128):
        w_t, esum = evidence_weight_step(u32[:, t:t + 1, :], esum, t, g32, pg32, bs32)
        worst_impl = max(worst_impl, (w_t - w_impl_par[:, t:t + 1, :]).abs().max().item())
    ok = worst < 1e-12 and worst_w < 1e-12 and worst_impl < 1e-5
    return ProofResult("P07", "证据门 cumsum==运行和（fp64 恒等；fp32 实现一致）",
                       "numeric-fp64 + fp32-impl", ok,
                       f"数学残差={worst:.2e}, 权重残差={worst_w:.2e}, "
                       f"实现残差={worst_impl:.2e}",
                       "FFT↔IIR 对偶形式在闸门层保持的前提")


# ---------------------------------------------------------------------------
# P08 闸门有界性与 RMSNorm 归一化
# ---------------------------------------------------------------------------
def _proof_gate_bounds() -> ProofResult:
    """三角不等式 ⇒ d_t = ‖a−b‖₁/(‖a‖₁+‖b‖₁+ε) ≤ 1，且在精确算术下**严格** < 1：

    (a) fp64 数学层：d < 1 严格（含对抗情形 local = −long，d = 2L/(2L+ε)）；
    (b) fp32 实现层：‖a‖₁ ≫ ε 时 d 可舍入到**精确 1.0**（2L/(2L+ε) 与 1
        之差低于 fp32 半精度）—— 闭界 [0,1] 仍成立且无 NaN 风险：
        w = 1 − σ(λ)·d ∈ [1−σ(λ), 1]，对一切 λ 有界。故实现层断言闭界。
    另：d 关于 (a,b) 对称；evidence ∈ (0,1)；RMSNorm 输出均方 ≈ 1。
    """
    torch.manual_seed(17)
    # (a) fp64 严格界
    long64 = torch.randn(4, 64, 32, dtype=torch.float64)
    d64 = joint_disagreement(long64, torch.randn(4, 64, 32, dtype=torch.float64))
    d64_adv = joint_disagreement(long64, -long64)
    strict64 = bool((d64 >= 0).all() and (d64 < 1).all()
                    and (d64_adv > 0.99).all() and (d64_adv < 1).all())
    # (b) fp32 实现层：闭界 + 对称 + 对抗到达上界
    long = long64.float()
    local = torch.randn(4, 64, 32)
    d = joint_disagreement(long, local)
    d_rev = joint_disagreement(local, long)
    sym = bool(torch.equal(d, d_rev))
    bounded32 = bool((d >= 0).all() and (d <= 1).all())
    d_adv = joint_disagreement(long, -long)
    adv = bool((d_adv > 0.99).all() and (d_adv <= 1).all())
    ev = evidence_weights(torch.randn(2, 32, 8), torch.tensor(1.5), torch.tensor(0.3),
                          torch.tensor(-0.2))
    ev_ok = bool((ev > 0).all() and (ev < 1).all())
    w = uncertainty_weights(long, local, torch.tensor(5.0))
    w_ok = bool((w > 0).all() and (w <= 1).all())
    n = RMSNorm(64)
    x = torch.randn(8, 7, 64)
    rms = n(x).pow(2).mean(-1)
    rms_ok = bool((rms - 1).abs().max() < 1e-4)
    ok = strict64 and sym and bounded32 and adv and ev_ok and w_ok and rms_ok
    return ProofResult("P08", "闸门有界性：fp64 严格 d<1（三角不等式）；fp32 实现闭界 [0,1]",
                       "numeric-fp64 + fp32-impl", ok,
                       f"fp64严格={strict64}, 对称={sym}, fp32闭界={bounded32}, "
                       f"对抗d→1⁻={adv}, evidence∈(0,1)={ev_ok}, w∈(0,1]={w_ok}, "
                       f"rms偏差={(rms - 1).abs().max().item():.2e}",
                       "fp32 下 d 精确到 1.0 是舍入而非越界：w 仍有界、无 NaN")


# ---------------------------------------------------------------------------
# P09 时移协变性（门关）/ 非协变性（门开，设计使然）
# ---------------------------------------------------------------------------
def _proof_shift_covariance() -> ProofResult:
    """闸门全关时 mixer 是（逐点 + LTI）因果系统：输入右移 s ⇒ 输出右移 s
    （t ≥ s 处逐位相等，fp64 ≤1e-10）。无位置嵌入表的时序感全部来自
    因果核本身 —— 这是"位置由因果核编码"主张的可验证形式。
    证据门开启时系统**故意**依赖绝对位置 t（c_t 的 1/t 归一化、log1p(t) 项），
    此时不再协变 —— 如实验证并记录该边界。
    """
    mx = _mixer(gates=False)
    x = torch.randn(1, 48, 16, dtype=torch.float64)
    s = 7
    x2 = torch.cat([torch.zeros(1, s, 16, dtype=torch.float64), x[:, :-s, :]], dim=1)
    with torch.no_grad():
        y1 = mx(x)
        y2 = mx(x2)
    off = (y2[:, s:] - y1[:, :-s]).abs().max().item()
    mx_on = _mixer(gates=True)
    with torch.no_grad():
        z1 = mx_on(x)
        z2 = mx_on(x2)
    on = (z2[:, s:] - z1[:, :-s]).abs().max().item()
    ok = off < 1e-10 and on > 1e-4
    return ProofResult("P09", "门关：时移协变（位置由因果核编码）；门开：故意非协变",
                       "numeric-fp64", ok,
                       f"门关残差={off:.2e}, 门开差异={on:.2e}",
                       "非协变来自证据门的绝对位置项（1/t 归一化），非缺陷")


# ---------------------------------------------------------------------------
# P10 O(1) 状态充分性（马尔可夫性）
# ---------------------------------------------------------------------------
def _proof_state_sufficiency() -> ProofResult:
    """两条**不同**历史只要收敛到同一 (y1, y2, conv_buf, energy_sum, length)，
    一切未来输出**逐位相同** —— O(1) 状态是未来的充分统计量，
    "无 KV cache 不丢信息"的严格形式。"""
    mx = _mixer(gates=True)
    fut = torch.randn(1, 30, 16, dtype=torch.float64)
    with torch.no_grad():
        states = []
        for seed in (0, 1):
            torch.manual_seed(100 + seed)
            pre = torch.randn(1, 20, 16, dtype=torch.float64)
            st = mx.init_state(1, fut.device, torch.float64)
            for t in range(20):
                _, st = mx.step(pre[:, t:t + 1, :], st)
            states.append(st)
        st_a, st_b = states
        # 把 B 的状态整体替换为 A 的（历史不同，状态相同）
        st_b.y1 = st_a.y1.clone()
        st_b.y2 = st_a.y2.clone()
        st_b.conv_buf = st_a.conv_buf.clone()
        st_b.energy_sum = st_a.energy_sum.clone()
        st_b.length = st_a.length
        outs = []
        for st in (st_a, st_b):
            ys = []
            for t in range(fut.shape[1]):
                yt, st = mx.step(fut[:, t:t + 1, :], st)
                ys.append(yt)
            outs.append(torch.cat(ys, dim=1))
    same = bool(torch.equal(outs[0], outs[1]))
    return ProofResult("P10", "O(1) 状态充分性：同状态 ⇒ 未来逐位一致（马尔可夫性）",
                       "numeric-fp64", same,
                       f"逐位一致={same}",
                       "两条随机历史在状态替换后对相同未来输入输出完全相同")


# ---------------------------------------------------------------------------
# P11 复杂度公式逐项重推导审计
# ---------------------------------------------------------------------------
def _proof_complexity_audit() -> ProofResult:
    """(a) ``_long_range_flops`` 与独立逐项推导一致（不同结合顺序）；
    (b) attention 每 token 随 L **精确翻倍**，SGC 长程项同倍增下 < +5%（亚线性）；
    (c) ``infer_flops_per_token`` 与 L 无关，且 mixer 项恰为 layers·D·(5M+22)
        —— v3.1 修正（旧版 6·D 漏乘基数 M）的防回归断言。"""
    cfg = SimplCoreConfig(vocab_size=8192, dim=256, num_layers=4, num_bases=16,
                          local_kernel=4, num_heads=4, ffn_ratio=2.0)
    d, m = float(cfg.dim), float(cfg.num_bases)

    def derive_long_range(s: float) -> float:
        """独立逐项推导（每 token 摊销：FFT/复乘项除以 s）。"""
        n = 2.0 * s
        return (2.0 * d * m                              # 核构建（本就是每 token）
                + 2.5 * n * math.log2(n) * d / s         # rfft，摊到每 token
                + 2.5 * n * math.log2(n) * d / s         # irfft，摊到每 token
                + 6.0 * (n / 2.0) * d / s                # 频域复乘，摊到每 token
                + 22.0 * d)                              # 逐点激活

    worst = 0.0
    for s in (512, 4096, 65536, 1048576):
        code = _long_range_flops(cfg, int(s))
        worst = max(worst, abs(code - derive_long_range(s)) / max(abs(code), 1e-300))
    # (b) 增长口径
    att = lambda L: 4.0 * float(L) * d
    att_ratio = att(2 ** 20) / att(2 ** 19)
    sgc_ratio = _long_range_flops(cfg, 2 ** 20) / _long_range_flops(cfg, 2 ** 19)
    growth_ok = (att_ratio == 2.0) and (sgc_ratio < 1.05)
    # (c) infer：与 L 无关 + mixer 项精确等式
    L_free = infer_flops_per_token(cfg, 1024) == infer_flops_per_token(cfg, 4 * 1024 * 1024)
    mixer_term = infer_flops_per_token(cfg, 4096) - 2.0 * non_embedding_params(cfg)
    mixer_exact = mixer_term == cfg.num_layers * cfg.dim * (5.0 * cfg.num_bases + 22.0)
    ok = worst < 1e-9 and growth_ok and L_free and mixer_exact
    return ProofResult("P11", "复杂度公式逐项重推导一致；attention×2 vs SGC 亚线性；infer 含基数 M",
                       "algebraic", ok,
                       f"推导残差={worst:.2e}, att比={att_ratio:.1f}, sgc比={sgc_ratio:.4f}, "
                       f"L无关={L_free}, mixer项精确={mixer_exact}",
                       "v3.1 修正：infer mixer 6·D → (5M+22)·D；训练式补核构建 2·D·M")


# ---------------------------------------------------------------------------
# P12 HBM 账本代数
# ---------------------------------------------------------------------------
def _proof_ledger_algebra() -> ProofResult:
    """部署/训练账本从组件公式**独立重推**并逐字节相等：

    (a) deploy：权重(bf16) + 流式状态 + 单chunk workspace + logits == deploy_bytes；
        总量 ≤ 2 GiB（1M 指标）；公式结构上不含 L（签名审计）。
    (b) campaign 分块训练：16 B/参数 × P + 单 chunk 激活 == chunked_train_bytes。
    (c) WeightBudget：allocate 守卫（2B 可学 / 3B 总上限）+ 显存只算可学。
    """
    cfg = get_preset("tuan-1b")
    P = embedding_params(cfg) + non_embedding_params(cfg)
    total = deploy_bytes(cfg, params=P)
    # v1.1.1：融合单缓冲——激活 workspace 与 logits 复用同一块（峰值取较大者）；
    # 状态公式含 M 因子（2·M·D/层）
    buffer = max(cfg.num_layers * DEFAULT_CHUNK * cfg.dim * DEFAULT_WORKSPACE_TENSORS * 2,
                 DEFAULT_CHUNK * cfg.vocab_size * 2)
    mine = (P * 2
            + stream_state_bytes(cfg, 1)
            + buffer)
    deploy_exact = mine == total
    deploy_cap = total <= DEPLOY_HBM_CAP_1M
    no_l = "seq" not in str(inspect.signature(deploy_bytes))
    cb = chunked_train_bytes(cfg, P, chunk=8192)
    cb_mine = OPTIMIZER_BYTES_PER_PARAM * P + cfg.num_layers * 8192 * cfg.dim * 8 * 2
    chunked_exact = cb == cb_mine
    opt_ok = optimizer_state_bytes(1_000_000_000) == 16 * 1_000_000_000
    b = WeightBudget()
    vram0 = b.vram_params() == b.learned_params == 0
    b.allocate(1_500_000_000)
    vram1 = b.vram_params() == 1_500_000_000
    refuse = not b.can_allocate(600_000_000)          # 1.5B + 0.6B > 2B 可学上限
    try:
        b.allocate(600_000_000)
        raises = False
    except ValueError:
        raises = True
    ok = (deploy_exact and deploy_cap and no_l and chunked_exact and opt_ok
          and vram0 and vram1 and refuse and raises)
    return ProofResult("P12", "HBM 账本代数：deploy/campaign 逐字节重推一致；预算守卫正确",
                       "algebraic", ok,
                       f"deploy精确={deploy_exact}({total / GIB:.3f} GiB), ≤2GiB={deploy_cap}, "
                       f"无L={no_l}, chunked精确={chunked_exact}, opt16B={opt_ok}, "
                       f"vram只算可学={vram0 and vram1}, 拒绝超2B={refuse and raises}",
                       "1M ≤ 2 GiB 与翻倍增量恒 0 的代数根源：四项均与 L 无关")


# ---------------------------------------------------------------------------
# P13 冻结基座梯度隔离
# ---------------------------------------------------------------------------
def _proof_frozen_base_gradient() -> ProofResult:
    """ForwardAdapter 反传一次：基座（含 head）所有参数 .grad 均为 None
    （梯度在结构上不可能流向基座），适配器参数全部有梯度且非全零。"""
    from ..training.envlearn import ForwardAdapter, LearnedAdapter
    from ..model.model import SimplCoreModel
    from ..model.tokenization import CharTokenizer

    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcd")
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_bases=8,
                                     decay_max=128, local_kernel=3, num_heads=2,
                                     ffn_ratio=2.0, mixer="sgc")
    base = SimplCoreModel(cfg)
    adapter = LearnedAdapter(dim=32, num_blocks=2)
    fa = ForwardAdapter(base, adapter, codec)
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    _, loss = fa(idx, targets=idx)
    loss.backward()
    base_clean = all(p.grad is None for p in fa.base.parameters())
    grads = [p.grad for p in fa.adapter.parameters()]
    adapter_has = all(g is not None for g in grads)
    adapter_live = any(g.abs().sum().item() > 0 for g in grads)
    ok = base_clean and adapter_has and adapter_live
    n_base = sum(1 for _ in fa.base.parameters())
    return ProofResult("P13", "梯度隔离：基座 .grad 全 None，梯度只流向适配器",
                       "numeric-fp32-impl", ok,
                       f"基座参数 {n_base} 个 grad 全 None={base_clean}, "
                       f"适配器有梯度={adapter_has and adapter_live}",
                       "Phase 2 \"显存只吃 2B、不碰 1B\" 的结构保证")


# ---------------------------------------------------------------------------
# 汇总与 CLI
# ---------------------------------------------------------------------------
_PROOFS: Tuple[Callable[[], ProofResult], ...] = (
    _proof_closed_form,
    _proof_companion_matrix,
    _proof_chunk_carry_law,
    _proof_fft_no_aliasing,
    _proof_bibo_stability,
    _proof_stream_equals_parallel,
    _proof_evidence_identity,
    _proof_gate_bounds,
    _proof_shift_covariance,
    _proof_state_sufficiency,
    _proof_complexity_audit,
    _proof_ledger_algebra,
    _proof_frozen_base_gradient,
)


def run_all() -> List[ProofResult]:
    """跑完全部数学证明，返回逐项结果。"""
    return [fn() for fn in _PROOFS]


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    results = run_all()
    print("SimplCore 数学证明（symbolic / fp64，无硬件依赖）")
    print("=" * 78)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"[{flag}] {r.pid} {r.claim}")
        print(f"       方法={r.method}  {r.residual}")
        if r.note:
            print(f"       注: {r.note}")
    n_ok = sum(1 for r in results if r.passed)
    print("=" * 78)
    print(f"{n_ok}/{len(results)} 项证明通过")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
