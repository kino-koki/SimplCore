# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""后端一致性跑批器 —— 把 ``backend_contract.py`` 从清单变成**可执行验收**。

这是 Phase 0 的对接脚本：自研 CUDA/C++ 后端只要把 16 个内核包装成一个
:class:`BackendAdapter` 子类（torch 张量进、torch 张量出，包装层可薄到
pybind11 直通），运行::

    python -m simplcore.core.conformance_runner --adapter mypkg.backend:MyAdapter

跑批器会对它执行 9 项一致性检查（7 项来自 ``conformance_checks()``，
外加契约 ``bitexact_requirements`` 的 C-rollback 与 C-stream-serialize）。
全部通过 → ``exit 0``，Phase 0 闸门 ``P0-backend-conformance`` 达成；
任何一项失败 → ``exit 1`` 并打印失败详情。

与 torch 参考实现的关系
------------------------
``TorchReferenceAdapter`` 是**数值规格 oracle 的适配器化**：它逐算子镜像
``sgc.py``/``gates.py`` 的参考数学（FFT 因果卷积、二阶 IIR、证据/不确定门、
冻结掩码 AdamW）。它通过跑批只证明"跑批器本身与参考数学自洽"；
**后端的验收永远是后端适配器 vs 参考数学的对照**，不 vs 参考适配器。

判定口径
--------
* C-dual-form：并行(FFT) vs 流式(IIR) 完整 mixer 输出，fp32，max|Δ| < 4e-5
  （契约原文口径；CI profile 用短序列同口径跑，full profile 用 4M）。
* C-chunked-carry：分块扫描 vs 整段扫描，输出与终态**逐位一致**。
* P0-bf16-accum：bf16 输入 / fp32 累加 vs fp64 真值，相对误差 ≤ 1e-2；
  累加长度 4096 时，纯 bf16 累加的期望相对误差 ~9%，无法混过该容差
  （``test_conformance_runner.py`` 有数值演示）。
* 其余检查以逐位/结构判定，详见各 ``_check_*`` docstring。

诚实边界
--------
跑批器判定的是**适配器暴露的算子语义**；吞吐、MFU、多卡扩展性不在此范围
（那是 Phase 1 bring-up 的事）。CI profile 在笔记本 CPU 上 ~秒级。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .backend_contract import DEFAULT_CONTRACT, conformance_checks
from .gates import evidence_weight_step, evidence_weights, uncertainty_weights
from .sgc import depthwise_causal_conv, fft_causal_conv

__all__ = [
    "BackendAdapter",
    "TorchReferenceAdapter",
    "ConformanceResult",
    "ConformanceReport",
    "run_conformance",
    "main",
]


# ---------------------------------------------------------------------------
# 适配器接口：16 个契约内核的 Python 包装面
# ---------------------------------------------------------------------------
# 契约算子 → 适配器方法（一个算子可对应多个方法，如 evidence 的并行/流式两形式）。
OP_TO_METHODS: Dict[str, Tuple[str, ...]] = {
    "emb_lookup": ("emb_lookup",),
    "rmsnorm": ("rmsnorm",),
    "gemm": ("gemm",),
    "swiglu": ("swiglu",),
    "residual_add": ("residual_add",),
    "causal_depthwise_conv": ("causal_depthwise_conv",),
    "spectral_causal_conv": ("spectral_causal_conv",),
    "chunked_iir_scan": ("chunked_iir_scan",),
    "content_gate": ("content_gate",),
    "evidence_uncertainty_gate": (
        "evidence_gate_parallel", "evidence_gate_chunk", "uncertainty_gate"),
    "ce_loss": ("ce_loss",),
    "stream_state_init": ("stream_state_init",),
    "stream_state_serialize": ("stream_state_serialize", "stream_state_deserialize"),
    "adamw_frozen_mask": ("adamw_frozen_mask",),
    "weight_decay_exempt": ("weight_decay_exempt",),
    "shard_allreduce": ("shard_allreduce",),
}
REQUIRED_METHODS: Tuple[str, ...] = tuple(
    sorted({m for ms in OP_TO_METHODS.values() for m in ms}))


class BackendAdapter:
    """后端算子适配器基类。

    子类覆盖全部 :data:`REQUIRED_METHODS`。约定：

    * 张量进、张量出（torch 张量是跑批器的通用语；pybind 包装层直通即可）；
    * 计算语义遵循契约：bf16 计算、fp32 累加；流式状态 fp32 常驻；
    * 不得引入契约 ``forbidden`` 清单中的任何内核名。
    """

    name: str = "unnamed-adapter"

    def ops_registry(self) -> Dict[str, Callable]:
        """适配器暴露的算子注册表（名称 → 可调用）。

        默认按 :data:`REQUIRED_METHODS` 内省；后端如需暴露额外算子可覆盖，
        但任何含契约禁止关键词的名字都会被 P0-no-attention 抓住。
        """
        return {m: getattr(self, m) for m in REQUIRED_METHODS}

    # -- 以下全部需要子类实现（签名即规格） ------------------------------------
    def emb_lookup(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor: ...
    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor: ...
    def gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
    def residual_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def causal_depthwise_conv(self, u: torch.Tensor, weight: torch.Tensor,
                              buf: Optional[torch.Tensor]
                              ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def spectral_causal_conv(self, u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor: ...
    def chunked_iir_scan(self, u: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def content_gate(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...
    def evidence_gate_parallel(self, u: torch.Tensor, gain: torch.Tensor,
                               pos_gain: torch.Tensor, bias: torch.Tensor) -> torch.Tensor: ...
    def evidence_gate_chunk(self, u: torch.Tensor, energy_sum: Optional[torch.Tensor],
                            length: int, gain: torch.Tensor, pos_gain: torch.Tensor,
                            bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def uncertainty_gate(self, long_: torch.Tensor, local: torch.Tensor,
                         logit: torch.Tensor, eps: float) -> torch.Tensor: ...
    def ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
    def stream_state_init(self, spec: Dict[str, List[int]]) -> Dict[str, torch.Tensor]: ...
    def stream_state_serialize(self, state: Dict[str, torch.Tensor]) -> bytes: ...
    def stream_state_deserialize(self, blob: bytes) -> Dict[str, torch.Tensor]: ...
    def adamw_frozen_mask(self, params: torch.Tensor, grads: torch.Tensor,
                          m: torch.Tensor, v: torch.Tensor, frozen: torch.Tensor,
                          lr: float, beta1: float, beta2: float, eps: float,
                          wd: float, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def weight_decay_exempt(self, param_names: List[str]) -> List[bool]: ...
    def shard_allreduce(self, grads: List[torch.Tensor]) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# torch 参考适配器（数值规格 oracle 的适配器化）
# ---------------------------------------------------------------------------
class TorchReferenceAdapter(BackendAdapter):
    """逐算子镜像 ``sgc.py`` / ``gates.py`` 参考数学。仅作 oracle，不参与训练。"""

    name = "torch-reference"

    def emb_lookup(self, weight, ids):
        return weight[ids]

    def rmsnorm(self, x, weight, eps):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (y * weight.float()).to(x.dtype)

    def gemm(self, a, b):
        # bf16 输入 → fp32 累加 → 按输入精度回写
        return (a.float() @ b.float()).to(a.dtype)

    def swiglu(self, gate, up):
        return (F.silu(gate.float()) * up.float()).to(gate.dtype)

    def residual_add(self, a, b):
        return (a.float() + b.float()).to(a.dtype)

    def causal_depthwise_conv(self, u, weight, buf):
        return depthwise_causal_conv(u, weight, buf)

    def spectral_causal_conv(self, u, kernel):
        return fft_causal_conv(u, kernel, torch.float32)

    def chunked_iir_scan(self, u, y1, y2, a, b, c, coef=None):
        """逐 token 二阶递推（与 ``SpectralGlobalConv.step`` 同一数学）。

        ``coef`` 给定时执行融合投影 epilogue（真实后端把 einsum 融进 scan）：
        返回 long (B,C,D)；否则返回原始振荡器态 y (B,C,M,D)。
        """
        uf = u.float()
        ys = []
        for t in range(uf.shape[1]):
            u_t = uf[:, t].unsqueeze(1)                       # (B,1,D)
            y_new = (a.view(1, -1, 1) * y1 + b.view(1, -1, 1) * y2
                     + c.view(1, -1, 1) * u_t)                # (B,M,D)
            y2, y1 = y1, y_new
            ys.append(y_new)
        y = torch.stack(ys, dim=1)                            # (B,C,M,D)
        if coef is None:
            return y, y1, y2
        long_ = torch.einsum("bcmd,md->bcd", y, coef.float())
        return long_, y1, y2

    def content_gate(self, h, w):
        return torch.sigmoid(h @ w.t())

    def evidence_gate_parallel(self, u, gain, pos_gain, bias):
        return evidence_weights(u, gain, pos_gain, bias)

    def evidence_gate_chunk(self, u, energy_sum, length, gain, pos_gain, bias):
        ws, es = [], energy_sum
        for t in range(u.shape[1]):
            w_t, es = evidence_weight_step(u[:, t:t + 1], es, length + t,
                                           gain, pos_gain, bias)
            ws.append(w_t)
        return torch.cat(ws, dim=1), es

    def uncertainty_gate(self, long_, local, logit, eps):
        return uncertainty_weights(long_, local, logit, eps)

    def ce_loss(self, logits, labels):
        V = logits.shape[-1]
        return F.cross_entropy(logits.float().reshape(-1, V), labels.reshape(-1),
                               ignore_index=-100)

    def stream_state_init(self, spec):
        return {k: torch.zeros(shape, dtype=torch.float32) for k, shape in spec.items()}

    def stream_state_serialize(self, state):
        names = sorted(state)
        shapes = [list(state[k].shape) for k in names]
        dtypes = [str(state[k].dtype) for k in names]
        header = json.dumps({"names": names, "shapes": shapes,
                             "dtypes": dtypes}).encode("utf-8")
        blobs = [state[k].contiguous().flatten().view(torch.uint8).numpy().tobytes()
                 for k in names]
        return len(header).to_bytes(8, "little") + header + b"".join(blobs)

    def stream_state_deserialize(self, blob):
        n = int.from_bytes(blob[:8], "little")
        meta = json.loads(blob[8:8 + n].decode("utf-8"))
        out, off = {}, 8 + n
        for name, shape, dt in zip(meta["names"], meta["shapes"], meta["dtypes"]):
            count = 1
            for s in shape:
                count *= s
            nbytes = count * _dtype_itemsize(dt)
            t = torch.frombuffer(bytearray(blob[off:off + nbytes]), dtype=_dtype(dt))
            out[name] = t.reshape(shape).clone()
            off += nbytes
        return out

    def adamw_frozen_mask(self, params, grads, m, v, frozen, lr, beta1, beta2, eps,
                          wd, step):
        """冻结位 p/m/v 全部零更新（``torch.where`` 逐位保护）。"""
        g = torch.where(frozen, torch.zeros_like(grads), grads)
        m_new = torch.where(frozen, m, beta1 * m + (1 - beta1) * g)
        v_new = torch.where(frozen, v, beta2 * v + (1 - beta2) * g * g)
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        upd = (m_new / bc1) / ((v_new / bc2).sqrt() + eps) + wd * params
        p_new = torch.where(frozen, params, params - lr * upd)
        return p_new, m_new, v_new

    def weight_decay_exempt(self, param_names):
        return [any(s in n for s in ("rho", "theta")) for n in param_names]

    def shard_allreduce(self, grads):
        out = grads[0].float().clone()
        for g in grads[1:]:
            out = out + g.float()
        return out


def _dtype(dt: str):
    return getattr(torch, dt.split(".")[-1])


def _dtype_itemsize(dt: str) -> int:
    return torch.empty(0, dtype=_dtype(dt)).element_size()


# ---------------------------------------------------------------------------
# 跑批器：composite mixer（runner 用固定权重把算子拼成完整 mixer）
# ---------------------------------------------------------------------------
@dataclass
class ConformanceResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ConformanceReport:
    adapter_name: str
    profile: Dict[str, object]
    results: List[ConformanceResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def as_dict(self) -> Dict[str, object]:
        return {"adapter": self.adapter_name, "profile": self.profile,
                "all_passed": self.all_passed,
                "results": [{"name": r.name, "passed": r.passed, "detail": r.detail}
                            for r in self.results]}

    def format(self) -> str:
        lines = [f"=== 后端一致性跑批：{self.adapter_name} ===",
                 f"profile: {self.profile}"]
        for r in self.results:
            lines.append(f" {'✓' if r.passed else '✗'} {r.name}: {r.detail}")
        lines.append(f"== 结论: {'通过（可进入 Phase 0 验收）' if self.all_passed else '失败（不得开训）'} ==")
        return "\n".join(lines)


def _mixer_weights(dim: int, num_bases: int, local_kernel: int, seed: int):
    """固定种子的 mixer 参数（fp32，runner 侧持有，不算适配器算子）。"""
    g = torch.Generator().manual_seed(seed)
    coef = torch.randn(num_bases, dim, generator=g) * 0.02
    lam = torch.exp(torch.linspace(math.log(1.0), math.log(64.0), num_bases))
    rho = torch.exp(-1.0 / lam)
    theta = math.pi * (torch.arange(num_bases, dtype=torch.float32) + 0.5) / num_bases
    w_loc = torch.randn(dim, 1, local_kernel, generator=g) * 0.1
    feed = torch.ones(dim)
    gain, pos_gain, bias = torch.tensor(1.0), torch.tensor(0.3), torch.tensor(-0.2)
    lam_logit = torch.tensor(0.7)
    w_in = torch.randn(dim, dim, generator=g) * 0.05
    w_gate = torch.randn(dim, dim, generator=g) * 0.05
    return dict(coef=coef, rho=rho, theta=theta, w_loc=w_loc, feed=feed,
                gain=gain, pos_gain=pos_gain, bias=bias, lam_logit=lam_logit,
                w_in=w_in, w_gate=w_gate)


def _mixer_parallel(ad: BackendAdapter, W: dict, u: torch.Tensor) -> torch.Tensor:
    """并行（FFT）形式，完整四层闸门口径。fp32。"""
    L = u.shape[1]
    tau = torch.arange(L, dtype=torch.float32).unsqueeze(0)
    psi = torch.exp(torch.log(W["rho"]).unsqueeze(1) * tau) * \
        torch.sin((tau + 1.0) * W["theta"].unsqueeze(1))
    kernel = (W["coef"].t() @ psi).t().contiguous()               # (L, D)
    long_ = ad.spectral_causal_conv(u, kernel)
    local, _ = ad.causal_depthwise_conv(u, W["w_loc"], None)
    long_ = long_ * ad.evidence_gate_parallel(u, W["gain"], W["pos_gain"], W["bias"])
    long_ = long_ * ad.uncertainty_gate(long_, local, W["lam_logit"], 1e-6)
    return long_ + local + W["feed"] * u


def _mixer_streaming(ad: BackendAdapter, W: dict, u: torch.Tensor, chunk: int,
                     state: Optional[Dict[str, torch.Tensor]] = None):
    """流式（分块 IIR）形式，同一数学的 scan 口径。fp32。"""
    B, L, D = u.shape
    M = W["rho"].shape[0]
    K = W["w_loc"].shape[-1]
    if state is None:
        state = dict(
            y1=torch.zeros(B, M, D), y2=torch.zeros(B, M, D),
            conv_buf=torch.zeros(B, K - 1, D) if K > 1 else torch.zeros(B, 0, D),
            energy_sum=torch.zeros(B, 1), length=torch.zeros(1, dtype=torch.int64))
    a = 2.0 * W["rho"] * torch.cos(W["theta"])
    b = -(W["rho"] ** 2)
    c = torch.sin(W["theta"])
    outs = []
    length = int(state["length"].item())
    for s in range(0, L, chunk):
        uc = u[:, s:s + chunk]
        yscan, state["y1"], state["y2"] = ad.chunked_iir_scan(
            uc, state["y1"], state["y2"], a, b, c)
        long_ = torch.einsum("bcmd,md->bcd", yscan, W["coef"])
        loc, state["conv_buf"] = ad.causal_depthwise_conv(uc, W["w_loc"], state["conv_buf"])
        ev, state["energy_sum"] = ad.evidence_gate_chunk(
            uc, state["energy_sum"], length, W["gain"], W["pos_gain"], W["bias"])
        long_ = long_ * ev
        long_ = long_ * ad.uncertainty_gate(long_, loc, W["lam_logit"], 1e-6)
        outs.append(long_ + loc + W["feed"] * uc)
        length += uc.shape[1]
    state["length"] = torch.tensor([length], dtype=torch.int64)
    return torch.cat(outs, dim=1), state


# ---------------------------------------------------------------------------
# 9 项检查
# ---------------------------------------------------------------------------
def _exercise_all_ops(ad: BackendAdapter, cfg: dict) -> List[str]:
    """逐算子冒烟调用（形状/有限性），返回错误清单（空 = 全部可运行）。"""
    D, M, K, V, B, L = cfg["dim"], cfg["num_bases"], cfg["local_kernel"], 97, 2, 16
    errs: List[str] = []
    x = torch.randn(B, L, D)
    w2 = torch.randn(D, D)

    def call(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 —— 跑批器必须报告任何算子异常
            errs.append(f"{name}: {type(e).__name__}: {e}")

    call("emb_lookup", lambda: ad.emb_lookup(torch.randn(V, D), torch.randint(0, V, (B, L))))
    call("rmsnorm", lambda: ad.rmsnorm(x, torch.randn(D), 1e-6))
    call("gemm", lambda: ad.gemm(torch.randn(B * L, D), w2))
    call("swiglu", lambda: ad.swiglu(torch.randn(B, L, 2 * D), torch.randn(B, L, 2 * D)))
    call("residual_add", lambda: ad.residual_add(x, x))
    call("causal_depthwise_conv", lambda: ad.causal_depthwise_conv(
        x, torch.randn(D, 1, K), None))
    call("spectral_causal_conv", lambda: ad.spectral_causal_conv(
        x, torch.randn(L, D)))
    call("chunked_iir_scan", lambda: ad.chunked_iir_scan(
        x, torch.randn(B, M, D), torch.randn(B, M, D),
        torch.rand(M) * 2 - 1, -torch.rand(M), torch.rand(M),
        torch.randn(M, D)))
    call("content_gate", lambda: ad.content_gate(x, w2))
    call("evidence_gate_parallel", lambda: ad.evidence_gate_parallel(
        x, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)))
    call("evidence_gate_chunk", lambda: ad.evidence_gate_chunk(
        x, None, 0, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)))
    call("uncertainty_gate", lambda: ad.uncertainty_gate(x, x, torch.tensor(0.0), 1e-6))
    call("ce_loss", lambda: ad.ce_loss(
        torch.randn(B, L, V), torch.randint(0, V, (B, L))))
    call("stream_state_init", lambda: ad.stream_state_init({"y1": [B, M, D]}))
    st = {"y1": torch.randn(B, M, D)}
    call("stream_state_serialize", lambda: ad.stream_state_serialize(st))
    call("stream_state_deserialize", lambda: ad.stream_state_deserialize(
        ad.stream_state_serialize(st)))
    call("adamw_frozen_mask", lambda: ad.adamw_frozen_mask(
        torch.randn(8), torch.randn(8), torch.zeros(8), torch.zeros(8),
        torch.tensor([True, False] * 4), 1e-3, 0.9, 0.999, 1e-8, 0.01, 1))
    call("weight_decay_exempt", lambda: ad.weight_decay_exempt(["raw_rho", "coef"]))
    call("shard_allreduce", lambda: ad.shard_allreduce([torch.randn(4), torch.randn(4)]))
    return errs


def _check_conformance(ad: BackendAdapter, cfg: dict) -> ConformanceResult:
    """P0-backend-conformance：16 契约算子全部有实现且可运行。"""
    missing = [m for m in REQUIRED_METHODS
               if not callable(getattr(ad, m, None))]
    if missing:
        return ConformanceResult("P0-backend-conformance", False,
                                 f"缺失 {len(missing)} 个方法: {missing}")
    errs = _exercise_all_ops(ad, cfg)
    if errs:
        return ConformanceResult("P0-backend-conformance", False,
                                 f"{len(errs)} 个算子冒烟失败: " + "; ".join(errs[:3]))
    n_ops = len(DEFAULT_CONTRACT.ops)
    return ConformanceResult("P0-backend-conformance", True,
                             f"契约全部 {n_ops} 算子有实现且冒烟通过")


def _check_dual_form(ad: BackendAdapter, cfg: dict, W: dict) -> ConformanceResult:
    """P0-stream-bitexact：并行(FFT) vs 流式(IIR)，fp32，max|Δ| < 4e-5。"""
    g = torch.Generator().manual_seed(cfg["seed"])
    u = torch.randn(cfg["batch"], cfg["length"], cfg["dim"], generator=g)
    y_par = _mixer_parallel(ad, W, u)
    y_stream, _ = _mixer_streaming(ad, W, u, cfg["chunk"])
    dev = (y_par - y_stream).abs().max().item()
    ok = dev < 4e-5
    return ConformanceResult("P0-stream-bitexact", ok,
                             f"max|Δ|={dev:.3e} (阈值 4e-5, L={cfg['length']})")


def _check_chunked_carry(ad: BackendAdapter, cfg: dict, W: dict) -> ConformanceResult:
    """P0-chunked-carry：分块 vs 整段扫描，输出与终态逐位一致。"""
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    u = torch.randn(cfg["batch"], cfg["length"], cfg["dim"], generator=g)
    y_chunked, st_c = _mixer_streaming(ad, W, u, cfg["chunk"])
    y_whole, st_w = _mixer_streaming(ad, W, u, cfg["length"])
    bit_y = torch.equal(y_chunked, y_whole)
    bit_s = all(torch.equal(st_c[k], st_w[k])
                for k in ("y1", "y2", "conv_buf", "energy_sum"))
    ok = bit_y and bit_s
    return ConformanceResult("P0-chunked-carry", ok,
                             f"输出逐位={bit_y}, 终态逐位={bit_s} "
                             f"(chunk={cfg['chunk']}, L={cfg['length']})")


def _check_frozen_mask(ad: BackendAdapter, cfg: dict) -> ConformanceResult:
    """P0-frozen-mask：冻结位 p/m/v 逐位不动，可训位与手工 AdamW 一致。"""
    g = torch.Generator().manual_seed(cfg["seed"] + 2)
    n = 4096
    p = torch.randn(n, generator=g)
    gr = torch.randn(n, generator=g) * 0.1
    m0, v0 = torch.zeros(n), torch.zeros(n)
    frozen = torch.zeros(n, dtype=torch.bool)
    frozen[::2] = True                                            # 一半冻结
    lr, b1, b2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 5
    p1, m1, v1 = ad.adamw_frozen_mask(p, gr, m0, v0, frozen, lr, b1, b2, eps, wd, step)
    bit_frozen = (torch.equal(p1[frozen], p[frozen]) and torch.equal(m1[frozen], m0[frozen])
                  and torch.equal(v1[frozen], v0[frozen]))
    # 可训位对照手工 AdamW
    tr = ~frozen
    gt = torch.where(tr, gr, torch.zeros_like(gr))
    m_ref = b1 * m0 + (1 - b1) * gt
    v_ref = b2 * v0 + (1 - b2) * gt * gt
    upd = (m_ref / (1 - b1 ** step)) / ((v_ref / (1 - b2 ** step)).sqrt() + eps) + wd * p
    p_ref = torch.where(tr, p - lr * upd, p)
    ok_train = torch.allclose(p1[tr], p_ref[tr], atol=1e-6, rtol=1e-5)
    ok = bool(bit_frozen and ok_train)
    return ConformanceResult("P0-frozen-mask", ok,
                             f"冻结位逐位不动={bit_frozen}, 可训位≈手工AdamW={ok_train}")


def _check_bf16_accum(ad: BackendAdapter, cfg: dict) -> ConformanceResult:
    """P0-bf16-accum：bf16 输入 / fp32 累加 vs fp64，相对误差 ≤ 1e-2。

    K=4096：纯 bf16 顺序累加的期望相对误差 ~9%，混不过该容差。
    """
    g = torch.Generator().manual_seed(cfg["seed"] + 3)
    K = 4096
    a = (torch.randn(32, K, generator=g) * 0.1).to(torch.bfloat16)
    w = (torch.randn(K, 32, generator=g) * 0.1).to(torch.bfloat16)
    worst = 0.0

    def rel_err(out, ref):
        ref = ref.double()
        return ((out.double() - ref).abs() / (1.0 + ref.abs())).max().item()

    out = ad.gemm(a, w)
    worst = max(worst, rel_err(out, a.double() @ w.double()))
    x = (torch.randn(8, 512, 64, generator=g)).to(torch.bfloat16)
    wt = torch.randn(64, generator=g).abs() + 0.5
    out = ad.rmsnorm(x, wt, 1e-6)
    xf = x.double()
    ref = (xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * wt.double()
    worst = max(worst, rel_err(out, ref))
    # ce_loss 抽样（契约点名的四个 fp32 累加内核之一）
    logits = (torch.randn(4, 128, 97, generator=g) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 97, (4, 128), generator=g)
    out = ad.ce_loss(logits, labels)
    lf = logits.double()
    logz = torch.logsumexp(lf, dim=-1, keepdim=True)
    nll = -(lf.gather(-1, labels.unsqueeze(-1)) - logz).squeeze(-1)
    ref_ce = nll[nll == nll].mean()          # ignore_index 语义下全有效
    worst = max(worst, abs(out.item() - ref_ce.item()) / max(1.0, abs(ref_ce.item())))
    ok = worst <= 1e-2
    return ConformanceResult("P0-bf16-accum", ok,
                             f"gemm+rmsnorm+ce_loss vs fp64 最差相对误差={worst:.2e} (≤1e-2)")


def _check_shard(ad: BackendAdapter, cfg: dict) -> ConformanceResult:
    """P0-shard-2gcd：2 分片 all-reduce 与单卡和逐位一致。"""
    g = torch.Generator().manual_seed(cfg["seed"] + 4)
    g0 = torch.randn(1024, generator=g)
    g1 = torch.randn(1024, generator=g)
    out = ad.shard_allreduce([g0, g1])
    ok = torch.equal(out, g0 + g1)
    # 4 分片：浮点可结合性容差
    gs = [torch.randn(1024, generator=g) for _ in range(4)]
    out4 = ad.shard_allreduce(gs)
    seq = gs[0] + gs[1] + gs[2] + gs[3]
    ok4 = (out4 - seq).abs().max().item() <= 1e-6 * (1 + seq.abs().max().item())
    return ConformanceResult("P0-shard-2gcd", bool(ok and ok4),
                             f"2 分片逐位={ok}, 4 分片容差内={ok4}")


def _check_no_attention(ad: BackendAdapter, cfg: dict) -> ConformanceResult:
    """P0-no-attention：注册表结构审查 —— 零注意力 / 零 KV cache / 零位置表。"""
    reg = ad.ops_registry()
    forbidden = ("attention", "softmax_qk", "kv_cache", "kvcache", "pos_emb",
                 "rope", "alibi", "flash")
    bad = [n for n in reg if any(f in n.lower() for f in forbidden)]
    extra = sorted(set(reg) - set(REQUIRED_METHODS))
    ok = not bad
    detail = f"注册表 {len(reg)} 算子, 禁止名命中: {bad or '无'}"
    if extra:
        detail += f"; 额外算子(需人工审查): {extra}"
    return ConformanceResult("P0-no-attention", ok, detail)


def _check_serialize(ad: BackendAdapter, cfg: dict, W: dict) -> ConformanceResult:
    """C-stream-serialize：状态 serialize→deserialize 往返逐位一致。"""
    g = torch.Generator().manual_seed(cfg["seed"] + 5)
    u = torch.randn(cfg["batch"], 256, cfg["dim"], generator=g)
    _, st = _mixer_streaming(ad, W, u, 128)
    blob = ad.stream_state_serialize(st)
    st2 = ad.stream_state_deserialize(blob)
    ok = all(torch.equal(st[k], st2[k]) for k in st)
    return ConformanceResult("C-stream-serialize", ok,
                             f"{len(st)} 个状态张量往返逐位={ok}, {len(blob)} bytes")


def _check_rollback(ad: BackendAdapter, cfg: dict, W: dict) -> ConformanceResult:
    """C-rollback：快照→继续→回滚→续跑 与 不间断续跑 逐位一致。"""
    g = torch.Generator().manual_seed(cfg["seed"] + 6)
    u1 = torch.randn(cfg["batch"], 128, cfg["dim"], generator=g)
    u2 = torch.randn(cfg["batch"], 128, cfg["dim"], generator=g)
    # 不间断：u1 + u2 一路流式（沿序列维拼接）
    y_ref, _ = _mixer_streaming(ad, W, torch.cat([u1, u2], dim=1), cfg["chunk"])
    # 快照 → 回滚 → 续跑
    _, st_snap = _mixer_streaming(ad, W, u1, cfg["chunk"])
    blob = ad.stream_state_serialize(st_snap)                     # 快照
    st_restored = ad.stream_state_deserialize(blob)               # 回滚恢复
    y_cont, _ = _mixer_streaming(ad, W, u2, cfg["chunk"], state=st_restored)
    ok = torch.equal(y_cont, y_ref[:, 128:])
    return ConformanceResult("C-rollback", ok,
                             f"快照→回滚→续跑 逐位一致={ok}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_conformance(adapter: BackendAdapter, *, length: int = 4096, chunk: int = 512,
                    batch: int = 2, dim: int = 16, num_bases: int = 8,
                    local_kernel: int = 4, seed: int = 2026) -> ConformanceReport:
    """对一个适配器执行全部 9 项一致性检查。"""
    cfg = dict(length=length, chunk=chunk, batch=batch, dim=dim,
               num_bases=num_bases, local_kernel=local_kernel, seed=seed)
    W = _mixer_weights(dim, num_bases, local_kernel, seed)
    rep = ConformanceReport(adapter_name=getattr(adapter, "name", type(adapter).__name__),
                            profile=dict(cfg))
    rep.results.append(_check_conformance(adapter, cfg))
    rep.results.append(_check_dual_form(adapter, cfg, W))
    rep.results.append(_check_chunked_carry(adapter, cfg, W))
    rep.results.append(_check_frozen_mask(adapter, cfg))
    rep.results.append(_check_bf16_accum(adapter, cfg))
    rep.results.append(_check_shard(adapter, cfg))
    rep.results.append(_check_no_attention(adapter, cfg))
    rep.results.append(_check_serialize(adapter, cfg, W))
    rep.results.append(_check_rollback(adapter, cfg, W))
    return rep


def _load_adapter(spec: str) -> BackendAdapter:
    mod_name, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    obj = cls()
    if not isinstance(obj, BackendAdapter):
        raise TypeError(f"{spec} 不是 BackendAdapter 子类")
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SimplCore 后端一致性跑批（Phase 0 验收）")
    ap.add_argument("--adapter", default=None,
                    help="后端适配器 '包.模块:类名'；缺省跑 torch 参考自检")
    ap.add_argument("--length", type=int, default=4096, help="dual-form 检查长度")
    ap.add_argument("--chunk", type=int, default=512, help="分块长度")
    ap.add_argument("--full", action="store_true",
                    help="契约全口径：L=4194304（慢，建议在后端机上跑）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args(argv)
    kwargs = {}
    if args.full:
        kwargs["length"] = 4 * 1024 * 1024
    adapter = _load_adapter(args.adapter) if args.adapter else TorchReferenceAdapter()
    rep = run_conformance(adapter, **kwargs)
    if args.json:
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(rep.format())
    return 0 if rep.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
