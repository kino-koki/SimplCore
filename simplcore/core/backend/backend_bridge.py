# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""后端 C-ABI 桥接层 —— :class:`BackendAdapter` 16 算子的设备内核调用约定。

**定位**：自研 CUDA/C++ 后端编译出 ``.so`` / ``.dll`` 后，无需改一行驱动
代码即可接入——用 :func:`load_backend` 加载，先过
``simplcore.core.conformance_runner`` 的 9 项检查，再进训练/推理。

**ABI v1 调用约定**（设备内核；宿主侧簿记不进 ABI）
------------------------------------------------------
* 符号名 ``sc_<op>``，全部返回 ``int32`` 错误码（0 = 成功，非 0 由
  ``CtypesBackendAdapter`` 转成带算子名的 RuntimeError）；
* 每个张量参数固定展开为 ``(void* ptr, const int64_t* shape, int32_t rank)``，
  行主序连续；dtype 由各算子约定固定（见 :data:`ABI_SPEC`），不传 dtype——
  传错 dtype 属于调用方 bug，桥接层在编组时已经校验；
* 输出缓冲由**调用方**分配（桥接层按 :data:`ABI_SPEC` 的形状规则分配），
  内核只写不分配——CUDA 侧因此可以自由使用 stream / workspace；
* 可空张量（如 ``chunked_iir_scan`` 的 ``coef``、``causal_depthwise_conv``
  的 ``buf``）以 ``NULL`` 指针表示，配套的标志参数显式告知；
* 标量参数一律 ``double``（fp64 传参，内核内自行降到计算精度）；
* 额外输出（如 new_buf 的实际长度）通过 ``int64_t*`` 出参带回；
* 宿主侧算子（``stream_state_init/serialize/deserialize``、
  ``weight_decay_exempt``）**不进 ABI**，桥接层原生实现且与参考实现
  byte 兼容——它们是簿记不是内核，让后端团队免写无意义的序列化代码。

**测试策略**：真实后端未到位期间，用 ``FakeCLib``（Python 函数按同一
签名经 **真实 ctypes 指针** 读写缓冲，数学委托参考适配器）端到端验证
编组层——指针、dtype、形状、contiguity、输出回填全走真路径；后端到位
后同一套测试换成加载真实库即得复用。
"""
from __future__ import annotations

import ctypes
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .conformance_runner import BackendAdapter

__all__ = [
    "ABI_VERSION",
    "ABI_SPEC",
    "CtypesBackendAdapter",
    "FakeCLib",
    "load_backend",
    "symbol_report",
]

ABI_VERSION = 1
DT_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.int64: 2, torch.int32: 3}

# ---------------------------------------------------------------------------
# ABI 规范：op → (输入张量名, 输出张量名, 标量, 出参, 输出形状规则)
# 形状规则收到"输入形状 dict"返回每个输出的形状；这是桥接层分配缓冲的依据。
# ---------------------------------------------------------------------------
ABI_SPEC: Dict[str, Dict[str, Any]] = {
    "emb_lookup": {
        "tensors_in": ["weight", "ids"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: tuple(s["ids"]) + (s["weight"][1],),
        "note": "weight(V,D) fp32; ids(...,) int64（任意秩）→ out(ids.shape + (D,)) fp32",
    },
    "rmsnorm": {
        "tensors_in": ["x", "weight"],
        "tensors_out": ["out"],
        "scalars": ["eps"], "extra_outs": [],
        "out_shapes": lambda s: tuple(s["x"]),
        "note": "x(...,D) fp32/bf16; weight(D); out 同 x（bf16 计算 fp32 累加）",
    },
    "gemm": {
        "tensors_in": ["a", "b"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: tuple(s["a"][:-1]) + (s["b"][1],),
        "note": "a(...,K), b(K,N) → out(a.shape[:-1] + (N,))；fp32 累加回写 a.dtype",
    },
    "swiglu": {
        "tensors_in": ["gate", "up"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: {"out": tuple(s["gate"])},
        "note": "elementwise 同形",
    },
    "residual_add": {
        "tensors_in": ["a", "b"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: {"out": tuple(s["a"])},
        "note": "elementwise 同形",
    },
    "spectral_causal_conv": {
        "tensors_in": ["u", "kernel"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: {"out": tuple(s["u"])},
        "note": "u(B,T,D), kernel(T,D) 核库 → out(B,T,D)（闭式谱核，fp32）",
    },
    "causal_depthwise_conv": {
        "tensors_in": ["u", "weight", "buf"],
        "tensors_out": ["out", "new_buf"],
        "scalars": [], "extra_outs": ["new_buf_len"],
        "out_shapes": lambda s: {"out": tuple(s["u"]),
                                 "new_buf": (s["u"][0], max(s["weight"][2] - 1, 0),
                                             s["u"][2])},
        "note": "weight(D,1,K); buf(B,P,D) 可空(NULL)；new_buf 实际长度写出参",
    },
    "chunked_iir_scan": {
        "tensors_in": ["u", "y1", "y2", "a", "b", "c", "coef"],
        "tensors_out": ["out", "out_y1", "out_y2"],
        "scalars": ["use_coef"], "extra_outs": [],
        "out_shapes": lambda s: (
            {"out": (s["u"][0], s["u"][1], s["u"][2]),
             "out_y1": tuple(s["y1"]), "out_y2": tuple(s["y2"])}
            if s.get("_use_coef") else
            {"out": (s["u"][0], s["u"][1], s["y1"][1], s["u"][2]),
             "out_y1": tuple(s["y1"]), "out_y2": tuple(s["y2"])}),
        "note": "coef(M,D) 可空；use_coef=1 时 out(B,C,D)（融合投影 epilogue）",
    },
    "content_gate": {
        "tensors_in": ["h", "w"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: tuple(s["h"][:-1]) + (s["w"][0],),
        "note": "h(...,Dh), w(K,Dh) → sigmoid(h·wᵀ) (...,K)",
    },
    "evidence_gate_parallel": {
        "tensors_in": ["u"],
        "tensors_out": ["out"],
        "scalars": ["gain", "pos_gain", "bias"], "extra_outs": [],
        "out_shapes": lambda s: {"out": (s["u"][0], s["u"][1], 1)},
        "note": "u(B,L,D) → 权重 (B,L,1)",
    },
    "evidence_gate_chunk": {
        "tensors_in": ["u", "energy_sum"],
        "tensors_out": ["out", "out_es"],
        "scalars": ["gain", "pos_gain", "bias"], "extra_outs": [],
        "out_shapes": lambda s: {"out": (s["u"][0], s["u"][1], 1),
                                 "out_es": (s["u"][0], 1)},
        "extra_scalars": ["length"],
        "note": "energy_sum(B,1) 可空(NULL=0)；length 为此前 token 数（int64）",
    },
    "uncertainty_gate": {
        "tensors_in": ["long_", "local"],
        "tensors_out": ["out"],
        "scalars": ["logit", "eps"], "extra_outs": [],
        "out_shapes": lambda s: {"out": (s["long_"][0], s["long_"][1], 1)},
        "note": "long_/local 同形 (B,L,D) → 阻尼 (B,L,1)",
    },
    "ce_loss": {
        "tensors_in": ["logits", "labels"],
        "tensors_out": ["out"],
        "scalars": [], "extra_outs": [],
        "out_shapes": lambda s: {"out": ()},
        "note": "logits(...,V) fp32/bf16; labels(...) int64（-100 忽略）→ 标量 mean CE",
    },
    "adamw_frozen_mask": {
        "tensors_in": ["params", "grads", "m", "v", "frozen"],
        "tensors_out": ["p_new", "m_new", "v_new"],
        "scalars": ["lr", "beta1", "beta2", "eps", "wd"],
        "extra_outs": [],
        "extra_scalars": ["step"],
        "out_shapes": lambda s: {"p_new": tuple(s["params"]),
                                 "m_new": tuple(s["params"]),
                                 "v_new": tuple(s["params"])},
        "note": "frozen(uint8 bool) 同形；step 为 int64 bias-correction 步数",
    },
}

REQUIRED_SYMBOLS = tuple(f"sc_{k}" for k in ABI_SPEC)


# ---------------------------------------------------------------------------
# ctypes 编组
# ---------------------------------------------------------------------------
def _shape_arr(shape: Sequence[int]) -> ctypes.Array:
    arr = (ctypes.c_int64 * max(len(shape), 1))(*[int(v) for v in shape])
    return arr


def _as_contig(t: torch.Tensor, name: str, want: Optional[torch.dtype] = None,
               allow_none: bool = False) -> Optional[torch.Tensor]:
    if t is None:
        if allow_none:
            return None
        raise ValueError(f"{name} 不允许为空")
    if want is not None and t.dtype != want:
        raise TypeError(f"{name} dtype 应为 {want}, 得到 {t.dtype}")
    return t.contiguous()


def _to_fp32(t: torch.Tensor, name: str) -> torch.Tensor:
    """统一 fp32 单一契约：非 fp32 输入在此升精度（与 ops/mixer_stream.py 同哲学）。

    内核（无论真实 C/CUDA/HIP 还是 FakeCLib）永远收到 fp32；低精度的
    归一化/回写由本桥接层负责。这同时修复了 FakeCLib 按 fp32 解读 bf16
    字节的潜在误读 bug——bf16 输入现在不可能到达内核。
    """
    ct = _as_contig(t, name)
    assert ct is not None
    return ct if ct.dtype == torch.float32 else ct.float()


class CtypesBackendAdapter(BackendAdapter):
    """经 C-ABI v1 调用设备内核的适配器（宿主侧算子原生实现）。

    ``lib`` 是带 ``sc_*`` 属性的可调用集合：真实场景来自
    ``ctypes.CDLL(path)``；测试可注入 :class:`FakeCLib`。构造即校验
    符号完备——缺符号在加载时报，不在训练第 400 步报。
    """

    name = "c-abi-backend"

    def __init__(self, lib: Any, *, strict_symbols: bool = True) -> None:
        missing = [s for s in REQUIRED_SYMBOLS if not hasattr(lib, s)]
        if missing and strict_symbols:
            raise RuntimeError(
                f"后端缺少 ABI v1 符号: {missing}（实现 {len(REQUIRED_SYMBOLS) - len(missing)}"
                f"/{len(REQUIRED_SYMBOLS)}）——对照 simplcore.core.backend_bridge.ABI_SPEC")
        self._lib = lib
        # 宿主侧算子：簿记不进 ABI，与参考实现 byte 兼容
        self._host = TorchHostOps()

    # -- 通用调用 --------------------------------------------------------------
    def _tensor_args(self, t: Optional[torch.Tensor], name: str,
                     allow_none: bool = False
                     ) -> Tuple[Any, ...]:
        if t is None and allow_none:
            return (ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int64)(),
                    ctypes.c_int32(0))
        ct = _as_contig(t, name, allow_none=allow_none)
        assert ct is not None
        shape = list(ct.shape) or [1]        # 0-dim 标量张量按 rank-1 处理
        return (ctypes.c_void_p(ct.data_ptr()), _shape_arr(shape),
                ctypes.c_int32(len(shape)))

    def _alloc(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty([int(v) for v in shape], dtype=dtype)

    def _call(self, op: str, tensors: Dict[str, Optional[torch.Tensor]],
              outs: Dict[str, torch.Tensor], scalars: Sequence[float] = (),
              iargs: Sequence[int] = (), extra_out_refs: Sequence[Any] = ()) -> None:
        spec = ABI_SPEC[op]
        fn = getattr(self._lib, f"sc_{op}")
        args: List[Any] = []
        keep: List[torch.Tensor] = []        # 悬指针防护： contiguous 临时量保活
        for name in spec["tensors_in"]:
            t = tensors.get(name)
            allow_none = name in ("buf", "coef", "energy_sum")
            if t is not None:
                keep.append(_as_contig(t, name, allow_none=allow_none))
                t = keep[-1]
            args.extend(self._tensor_args(t, name, allow_none=allow_none))
        for name in spec["tensors_out"]:
            ot = outs[name].contiguous()
            keep.append(ot)
            args.extend((ctypes.c_void_p(ot.data_ptr()), _shape_arr(ot.shape),
                         ctypes.c_int32(len(ot.shape))))
        args.extend(ctypes.c_double(v) for v in scalars)
        args.extend(ctypes.c_int64(v) for v in iargs)
        args.extend(extra_out_refs)
        rc = fn(*args)
        rc = int(rc)
        if rc != 0:
            raise RuntimeError(f"后端内核 {op} 失败: code={rc}")

    # -- 设备内核（经 ABI） -----------------------------------------------------
    def emb_lookup(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        w = _as_contig(weight, "weight", torch.float32)
        i = _as_contig(ids, "ids", torch.int64)
        out = self._alloc(tuple(i.shape) + (w.shape[1],), torch.float32)
        self._call("emb_lookup", {"weight": w, "ids": i}, {"out": out})
        return out

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig = x.dtype
        xc = _to_fp32(x, "x")
        w = _to_fp32(weight, "weight")
        out = self._alloc(xc.shape, torch.float32)
        self._call("rmsnorm", {"x": xc, "weight": w}, {"out": out},
                   scalars=[float(eps)])
        return out.to(orig)

    def gemm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        orig = a.dtype
        ac = _to_fp32(a, "a")
        bc = _to_fp32(b, "b")
        out = self._alloc(tuple(ac.shape[:-1]) + (bc.shape[1],), torch.float32)
        self._call("gemm", {"a": ac, "b": bc}, {"out": out})
        return out.to(orig)

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig = gate.dtype
        g = _to_fp32(gate, "gate")
        u = _to_fp32(up, "up")
        out = self._alloc(g.shape, torch.float32)
        self._call("swiglu", {"gate": g, "up": u}, {"out": out})
        return out.to(orig)

    def residual_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        orig = a.dtype
        ac = _to_fp32(a, "a")
        bc = _to_fp32(b, "b")
        out = self._alloc(ac.shape, torch.float32)
        self._call("residual_add", {"a": ac, "b": bc}, {"out": out})
        return out.to(orig)

    def spectral_causal_conv(self, u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        uc = _as_contig(u, "u", torch.float32)
        kc = _as_contig(kernel, "kernel", torch.float32)
        out = self._alloc(uc.shape, torch.float32)
        self._call("spectral_causal_conv", {"u": uc, "kernel": kc}, {"out": out})
        return out

    def causal_depthwise_conv(self, u: torch.Tensor, weight: torch.Tensor,
                              buf: Optional[torch.Tensor]
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig = u.dtype
        uc = _to_fp32(u, "u")
        wc = _to_fp32(weight, "weight")
        if buf is not None:
            buf = _to_fp32(buf, "buf")
        k = wc.shape[-1]
        out = self._alloc(uc.shape, torch.float32)
        new_buf = self._alloc((uc.shape[0], max(k - 1, 0), uc.shape[2]),
                              torch.float32)
        new_len = ctypes.c_int64(0)
        self._call("causal_depthwise_conv",
                   {"u": uc, "weight": wc, "buf": buf},
                   {"out": out, "new_buf": new_buf},
                   extra_out_refs=[ctypes.pointer(new_len)])
        return (out.to(orig),
                new_buf[:, :int(new_len.value), :].to(orig))

    def chunked_iir_scan(self, u: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                         coef: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uc = _as_contig(u, "u", torch.float32)
        y1c = _as_contig(y1, "y1", torch.float32)
        y2c = _as_contig(y2, "y2", torch.float32)
        ac = _as_contig(a, "a", torch.float32)
        bc = _as_contig(b, "b", torch.float32)
        cc = _as_contig(c, "c", torch.float32)
        use_coef = coef is not None
        co = _as_contig(coef, "coef", torch.float32, allow_none=True) \
            if use_coef else None
        main_shape = ((uc.shape[0], uc.shape[1], uc.shape[2])
                      if use_coef else
                      (uc.shape[0], uc.shape[1], y1c.shape[1], uc.shape[2]))
        out = self._alloc(main_shape, torch.float32)
        oy1 = self._alloc(y1c.shape, torch.float32)
        oy2 = self._alloc(y2c.shape, torch.float32)
        self._call("chunked_iir_scan",
                   {"u": uc, "y1": y1c, "y2": y2c, "a": ac, "b": bc, "c": cc,
                    "coef": co},
                   {"out": out, "out_y1": oy1, "out_y2": oy2},
                   iargs=[1 if use_coef else 0])
        return out, oy1, oy2

    def content_gate(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        hc = _to_fp32(h, "h")
        wc = _to_fp32(w, "w")
        out = self._alloc(tuple(hc.shape[:-1]) + (wc.shape[0],), torch.float32)
        self._call("content_gate", {"h": hc, "w": wc}, {"out": out})
        return out

    def evidence_gate_parallel(self, u: torch.Tensor, gain, pos_gain, bias) -> torch.Tensor:
        uc = _as_contig(u, "u", torch.float32)
        out = self._alloc((uc.shape[0], uc.shape[1], 1), torch.float32)
        self._call("evidence_gate_parallel", {"u": uc}, {"out": out},
                   scalars=[float(gain), float(pos_gain), float(bias)])
        return out

    def evidence_gate_chunk(self, u: torch.Tensor, energy_sum: Optional[torch.Tensor],
                            length: int, gain, pos_gain, bias
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        uc = _as_contig(u, "u", torch.float32)
        es = None if energy_sum is None else \
            _as_contig(energy_sum, "energy_sum", torch.float32)
        out = self._alloc((uc.shape[0], uc.shape[1], 1), torch.float32)
        out_es = self._alloc((uc.shape[0], 1), torch.float32)
        self._call("evidence_gate_chunk", {"u": uc, "energy_sum": es},
                   {"out": out, "out_es": out_es},
                   scalars=[float(gain), float(pos_gain), float(bias)],
                   iargs=[int(length)])
        return out, out_es

    def uncertainty_gate(self, long_: torch.Tensor, local: torch.Tensor,
                         logit, eps: float) -> torch.Tensor:
        lc = _as_contig(long_, "long_", torch.float32)
        oc = _as_contig(local, "local", torch.float32)
        out = self._alloc((lc.shape[0], lc.shape[1], 1), torch.float32)
        self._call("uncertainty_gate", {"long_": lc, "local": oc}, {"out": out},
                   scalars=[float(logit), float(eps)])
        return out

    def ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        lg = _to_fp32(logits, "logits")
        lb = _as_contig(labels, "labels", torch.int64)
        out = self._alloc((), torch.float32)
        self._call("ce_loss", {"logits": lg, "labels": lb}, {"out": out})
        return out

    def adamw_frozen_mask(self, params: torch.Tensor, grads: torch.Tensor,
                          m: torch.Tensor, v: torch.Tensor, frozen: torch.Tensor,
                          lr: float, beta1: float, beta2: float, eps: float,
                          wd: float, step: int
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pc = _as_contig(params, "params", torch.float32)
        gc = _as_contig(grads, "grads", torch.float32)
        mc = _as_contig(m, "m", torch.float32)
        vc = _as_contig(v, "v", torch.float32)
        fc = _as_contig(frozen, "frozen", torch.bool)
        f_u8 = fc.view(torch.uint8)
        p_new = self._alloc(pc.shape, torch.float32)
        m_new = self._alloc(pc.shape, torch.float32)
        v_new = self._alloc(pc.shape, torch.float32)
        self._call("adamw_frozen_mask",
                   {"params": pc, "grads": gc, "m": mc, "v": vc, "frozen": f_u8},
                   {"p_new": p_new, "m_new": m_new, "v_new": v_new},
                   scalars=[float(lr), float(beta1), float(beta2),
                            float(eps), float(wd)],
                   iargs=[int(step)])
        return p_new, m_new, v_new

    # -- 宿主侧算子（簿记，不进 ABI） -------------------------------------------
    def shard_allreduce(self, grads: List[torch.Tensor]) -> torch.Tensor:
        out = grads[0].float().clone()
        for g in grads[1:]:
            out = out + g.float()
        return out

    def stream_state_init(self, spec: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        return self._host.stream_state_init(spec)

    def stream_state_serialize(self, state: Dict[str, torch.Tensor]) -> bytes:
        return self._host.stream_state_serialize(state)

    def stream_state_deserialize(self, blob: bytes) -> Dict[str, torch.Tensor]:
        return self._host.stream_state_deserialize(blob)

    def weight_decay_exempt(self, param_names: List[str]) -> List[bool]:
        return self._host.weight_decay_exempt(param_names)


class TorchHostOps:
    """宿主侧算子的原生实现（与参考适配器 byte 兼容；测试钉死）。"""

    def stream_state_init(self, spec: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros(shape, dtype=torch.float32) for k, shape in spec.items()}

    def stream_state_serialize(self, state: Dict[str, torch.Tensor]) -> bytes:
        names = sorted(state)
        shapes = [list(state[k].shape) for k in names]
        dtypes = [str(state[k].dtype) for k in names]
        header = json.dumps({"names": names, "shapes": shapes,
                             "dtypes": dtypes}).encode("utf-8")
        blobs = [state[k].contiguous().flatten().view(torch.uint8).numpy().tobytes()
                 for k in names]
        return len(header).to_bytes(8, "little") + header + b"".join(blobs)

    def stream_state_deserialize(self, blob: bytes) -> Dict[str, torch.Tensor]:
        from .conformance_runner import _dtype, _dtype_itemsize
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

    def weight_decay_exempt(self, param_names: List[str]) -> List[bool]:
        return [any(s in n for s in ("rho", "theta")) for n in param_names]


# ---------------------------------------------------------------------------
# 加载与符号校验
# ---------------------------------------------------------------------------
def symbol_report(path: Any) -> Dict[str, Any]:
    """加载库并列出 ABI 符号完备性（不抛异常，出报告）。"""
    lib = ctypes.CDLL(str(path))
    present = {s: hasattr(lib, s) for s in REQUIRED_SYMBOLS}
    has_ver = hasattr(lib, "sc_abi_version")
    version = int(lib.sc_abi_version()) if has_ver else None
    return {"path": str(path), "symbols_present": sum(present.values()),
            "symbols_required": len(REQUIRED_SYMBOLS), "missing": [s for s, ok in
                                                                   present.items() if not ok],
            "abi_version_symbol": has_ver, "abi_version": version,
            "compatible": bool(has_ver and version == ABI_VERSION and not
                               any(present[s] is False for s in REQUIRED_SYMBOLS))}


def load_backend(path: Any) -> CtypesBackendAdapter:
    """加载真实后端库 → CtypesBackendAdapter。缺符号/版本不符即拒绝。"""
    rep = symbol_report(path)
    if rep["missing"]:
        raise RuntimeError(f"后端 {path} 缺少 ABI v1 符号: {rep['missing']}")
    if not rep["abi_version_symbol"]:
        raise RuntimeError(f"后端 {path} 未导出 sc_abi_version")
    if rep["abi_version"] != ABI_VERSION:
        raise RuntimeError(f"ABI 版本不符: 库={rep['abi_version']}, "
                           f"驱动={ABI_VERSION}")
    return CtypesBackendAdapter(ctypes.CDLL(str(path)))


# ---------------------------------------------------------------------------
# 测试替身：按 ABI 签名经真实指针读写缓冲（数学委托参考适配器）
# ---------------------------------------------------------------------------
class FakeCLib:
    """与 C-ABI v1 同签名的 Python 替身——编组路径是真的，内核是参考数学。

    每个方法收到 ctypes 指针/形状数组，用 ``torch.frombuffer`` 解引用——
    与真实 C 库读同一块内存的方式一致，因此指针/dtype/形状/contiguity
    的编组正确性被真实锻炼，而不是被 mock 掉。
    """

    name = "fake-c-lib"

    def __init__(self) -> None:
        from .conformance_runner import TorchReferenceAdapter
        self._ref = TorchReferenceAdapter()
        for op in ABI_SPEC:
            setattr(self, f"sc_{op}", getattr(self, f"_op_{op}"))

    # -- 缓冲解引用 ------------------------------------------------------------
    @staticmethod
    def _in(args: List[Any], idx: int, dtype: torch.dtype) -> torch.Tensor:
        ptr, shape_arr, rank = args[idx * 3], args[idx * 3 + 1], args[idx * 3 + 2]
        shape = []
        for i in range(int(rank.value if hasattr(rank, "value") else rank)):
            shape.append(int(shape_arr[i]))
        if ptr.value is None or ptr.value == 0:
            if shape:                      # 非空指针不可能为 0 值——防御
                raise ValueError("收到空指针但声明了非空形状")
            return None  # type: ignore[return-value]
        n = 1
        for v in shape:
            n *= v
        buf = (ctypes.c_char * (n * torch.empty((), dtype=dtype).element_size())
               ).from_address(ptr.value)
        return torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape).clone()

    @staticmethod
    def _write_out(args: List[Any], idx: int, values: torch.Tensor) -> None:
        ptr = args[idx * 3]
        flat = values.contiguous().flatten()
        nbytes = flat.numel() * flat.element_size()
        dst = (ctypes.c_char * nbytes).from_address(ptr.value)
        dst[:] = flat.view(torch.uint8).numpy().tobytes()

    def _dec(self, args: List[Any], start: int, count: int,
             dtypes: List[torch.dtype]) -> List[Optional[torch.Tensor]]:
        return [self._in(args, start + i, dtypes[i]) for i in range(count)]

    # -- 各内核：解缓冲 → 参考数学 → 写回 ----------------------------------------
    # 参数索引布局（与 CtypesBackendAdapter._call 一致）：
    #   [0, 3*n_in)  输入张量（每个 = ptr, shape*, rank）
    #   [3*n_in, 3*n_in+3*n_out)  输出缓冲
    #   随后 scalars(double)，再 iargs(int64)，最后 extra_out_refs
    # Fake 只覆盖 fp32 内核路径；bf16 路径的编组由桥接层 dtype 校验守护、
    # 数值由真实后端的 conformance 验收。
    def _op_emb_lookup(self, *args: Any) -> int:
        w = self._in(args, 0, torch.float32)
        ids = self._in(args, 1, torch.int64)
        self._write_out(args, 2, self._ref.emb_lookup(w, ids))
        return 0

    def _op_rmsnorm(self, *args: Any) -> int:
        x = self._in(args, 0, torch.float32)
        w = self._in(args, 1, torch.float32)
        eps = float(args[9].value)
        self._write_out(args, 2, self._ref.rmsnorm(x, w, eps))
        return 0

    def _op_gemm(self, *args: Any) -> int:
        a = self._in(args, 0, torch.float32)
        b = self._in(args, 1, torch.float32)
        self._write_out(args, 2, self._ref.gemm(a, b))
        return 0

    def _op_swiglu(self, *args: Any) -> int:
        g = self._in(args, 0, torch.float32)
        u = self._in(args, 1, torch.float32)
        self._write_out(args, 2, self._ref.swiglu(g, u))
        return 0

    def _op_residual_add(self, *args: Any) -> int:
        a = self._in(args, 0, torch.float32)
        b = self._in(args, 1, torch.float32)
        self._write_out(args, 2, self._ref.residual_add(a, b))
        return 0

    def _op_spectral_causal_conv(self, *args: Any) -> int:
        u = self._in(args, 0, torch.float32)
        k = self._in(args, 1, torch.float32)
        self._write_out(args, 2, self._ref.spectral_causal_conv(u, k))
        return 0

    def _op_causal_depthwise_conv(self, *args: Any) -> int:
        u = self._in(args, 0, torch.float32)
        w = self._in(args, 1, torch.float32)
        buf = self._in(args, 2, torch.float32)
        y, nb = self._ref.causal_depthwise_conv(u, w, buf)
        self._write_out(args, 3, y)
        self._write_out(args, 4, nb)
        args[15][0] = int(nb.shape[1])             # 经指针写 *out_new_buf_len
        return 0

    def _op_chunked_iir_scan(self, *args: Any) -> int:
        u = self._in(args, 0, torch.float32)
        y1 = self._in(args, 1, torch.float32)
        y2 = self._in(args, 2, torch.float32)
        a = self._in(args, 3, torch.float32)
        b = self._in(args, 4, torch.float32)
        c = self._in(args, 5, torch.float32)
        coef = self._in(args, 6, torch.float32)
        use_coef = int(args[30].value)             # iargs[0]，位于 7*3+3*3=30
        out, oy1, oy2 = self._ref.chunked_iir_scan(
            u, y1, y2, a, b, c, coef=coef if use_coef else None)
        self._write_out(args, 7, out)
        self._write_out(args, 8, oy1)
        self._write_out(args, 9, oy2)
        return 0

    def _op_content_gate(self, *args: Any) -> int:
        h = self._in(args, 0, torch.float32)
        w = self._in(args, 1, torch.float32)
        self._write_out(args, 2, self._ref.content_gate(h, w))
        return 0

    def _op_evidence_gate_parallel(self, *args: Any) -> int:
        u = self._in(args, 0, torch.float32)
        sc = [float(args[i].value) for i in (6, 7, 8)]
        self._write_out(args, 1, self._ref.evidence_gate_parallel(u, *sc))
        return 0

    def _op_evidence_gate_chunk(self, *args: Any) -> int:
        u = self._in(args, 0, torch.float32)
        es = self._in(args, 1, torch.float32)
        sc = [float(args[i].value) for i in (12, 13, 14)]
        length = int(args[15].value)               # iargs[0]
        out, oes = self._ref.evidence_gate_chunk(u, es, length, *sc)
        self._write_out(args, 2, out)
        self._write_out(args, 3, oes)
        return 0

    def _op_uncertainty_gate(self, *args: Any) -> int:
        lg = self._in(args, 0, torch.float32)
        lc = self._in(args, 1, torch.float32)
        sc = [float(args[i].value) for i in (9, 10)]
        # 参考数学要求 logit 为张量（ABI 传参是 double，宿主侧包回 0-dim）
        self._write_out(args, 2, self._ref.uncertainty_gate(
            lg, lc, torch.tensor(sc[0]), sc[1]))
        return 0

    def _op_ce_loss(self, *args: Any) -> int:
        logits = self._in(args, 0, torch.float32)
        labels = self._in(args, 1, torch.int64)
        self._write_out(args, 2, self._ref.ce_loss(logits, labels))
        return 0

    def _op_adamw_frozen_mask(self, *args: Any) -> int:
        p = self._in(args, 0, torch.float32)
        g = self._in(args, 1, torch.float32)
        m = self._in(args, 2, torch.float32)
        v = self._in(args, 3, torch.float32)
        frozen = self._in(args, 4, torch.uint8).bool()
        sc = [float(args[i].value) for i in (24, 25, 26, 27, 28)]
        step = int(args[29].value)                 # iargs[0]
        p_new, m_new, v_new = self._ref.adamw_frozen_mask(
            p, g, m, v, frozen, *sc, step)
        self._write_out(args, 5, p_new)
        self._write_out(args, 6, m_new)
        self._write_out(args, 7, v_new)
        return 0
