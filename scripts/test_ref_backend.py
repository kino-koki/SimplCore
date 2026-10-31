# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""C 参考后端（sc_ref_backend）的本地数值自测 —— 纯 stdlib + ctypes，无 torch 依赖。

对每个 ABI 内核：用固定种子的随机小张量，经**真实 ctypes 指针编组**调用
共享库，与写在测试里的 Python 参考数学（逐式镜像
``conformance_runner.TorchReferenceAdapter`` / ``gates.py`` / ``sgc.py``）比对。

另含两项关键逐位检查：
* P0-chunked-carry 同构：chunked_iir_scan 分块 vs 整段，输出与终态 **字节级一致**；
* evidence 并行 vs 分块口径一致性（容差 1e-6）。

用法：``python scripts/test_ref_backend.py``（先跑 scripts/build_ref_backend.sh）
退出码 0 = 全部通过。torch 集成跑批（conformance_runner）在装 torch 的目标机上执行。
"""
from __future__ import annotations

import ctypes
import math
import os
import random
import struct
import sys
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

F32, I64 = ctypes.c_float, ctypes.c_int64

FAILURES: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f" [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a: Sequence[float], b: Sequence[float], rtol: float = 1e-5,
          atol: float = 1e-6) -> Tuple[bool, float]:
    worst = 0.0
    for x, y in zip(a, b):
        d = abs(x - y) / max(1.0, abs(y))
        worst = max(worst, d)
        if abs(x - y) > atol + rtol * abs(y):
            return False, worst
    return True, worst


# ---------------------------------------------------------------------------
# ABI 编组（镜像 backend_bridge.CtypesBackendAdapter._call 的参数布局）
# ---------------------------------------------------------------------------
class Lib:
    def __init__(self, path: str) -> None:
        self.lib = ctypes.CDLL(path)

    def _tensor(self, data: Optional[List[float]], shape: Sequence[int],
                ctype=I64) -> Tuple[Any, ...]:  # noqa: F821
        raise NotImplementedError

    @staticmethod
    def _pack_tensor(buf: Optional[ctypes.Array], shape: Sequence[int]):
        if buf is None:
            return (ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int64)(),
                    ctypes.c_int32(0))
        n = 1
        for s in shape:
            n *= s
        assert len(buf) == n, f"buf len {len(buf)} != numel {n}"
        shape_arr = (ctypes.c_int64 * max(len(shape), 1))(*[int(v) for v in shape])
        return (ctypes.c_void_p(ctypes.cast(buf, ctypes.c_void_p).value),
                ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)),
                ctypes.c_int32(len(shape)))

    @staticmethod
    def f32(vals: Sequence[float]) -> ctypes.Array:
        return (F32 * len(vals))(*[float(v) for v in vals])

    @staticmethod
    def i64(vals: Sequence[int]) -> ctypes.Array:
        return (I64 * len(vals))(*[int(v) for v in vals])

    def call(self, op: str, tensors_in, tensors_out, scalars: Sequence[float] = (),
             iargs: Sequence[int] = (), extra_outs: Optional[List[Any]] = None) -> int:  # noqa: F821
        args: List[Any] = []
        for buf, shape in tensors_in:
            args.extend(self._pack_tensor(buf, shape))
        for buf, shape in tensors_out:
            args.extend(self._pack_tensor(buf, shape))
        args.extend(ctypes.c_double(v) for v in scalars)
        args.extend(ctypes.c_int64(v) for v in iargs)
        args.extend(extra_outs or [])
        fn = getattr(self.lib, f"sc_{op}")
        fn.restype = ctypes.c_int32
        return int(fn(*args))


# ---------------------------------------------------------------------------
# Python 参考数学（逐式镜像 TorchReferenceAdapter / gates / sgc）
# ---------------------------------------------------------------------------
def ref_rmsnorm(x, w, eps):
    out = []
    for row in x:
        mean = sum(v * v for v in row) / len(row)
        inv = 1.0 / math.sqrt(mean + eps)
        out.append([v * inv * w[d] for d, v in enumerate(row)])
    return out


def ref_gemm(a, b):
    K, N = len(b), len(b[0])
    out = []
    for row in a:
        orow = []
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += row[k] * b[k][n]
            orow.append(acc)
        out.append(orow)
    return out


def ref_spectral_conv(u, k):
    B, T, D = len(u), len(u[0]), len(u[0][0])
    out = []
    for b in range(B):
        rows = []
        for t in range(T):
            row = []
            for d in range(D):
                acc = 0.0
                for s in range(t + 1):
                    acc += k[t - s][d] * u[b][s][d]
                row.append(acc)
            rows.append(row)
        out.append(rows)
    return out


def ref_depthwise(u, w, buf):
    """w: D×K（D,1,K 展平）。语义镜像 sgc.depthwise_causal_conv。"""
    B, T, D = len(u), len(u[0]), len(u[0][0])
    K = len(w[0])
    if K == 1:
        y = [[[u[b][t][d] * w[d][0] for d in range(D)] for t in range(T)]
             for b in range(B)]
        return y, [[] for _ in range(B)]
    uu = [(buf[b] if buf is not None else []) + u[b] for b in range(B)]
    P = len(uu[0]) - T
    out = []
    for b in range(B):
        rows = []
        for t in range(T):
            row = []
            for d in range(D):
                acc = 0.0
                for k in range(K):
                    j = P + t + k - (K - 1)
                    if j < 0:
                        continue
                    acc += w[d][k] * uu[b][j][d]
                row.append(acc)
            rows.append(row)
        out.append(rows)
    plen = min(P + T, K - 1)
    new_buf = [uu[b][P + T - plen:] for b in range(B)] if plen > 0 \
        else [[] for _ in range(B)]
    return out, new_buf


def ref_iir_scan(u, y1, y2, a, b, c):
    """返回 (y(B,C,M,D), y1_final, y2_final)；逐 token 顺序递推。"""
    B, C, D = len(u), len(u[0]), len(u[0][0])
    M = len(a)
    y1c = [row[:] for row in y1]
    y2c = [row[:] for row in y2]
    ys = []
    for t in range(C):
        ynew = []
        for bi in range(B):
            for m in range(M):
                row = []
                for d in range(D):
                    row.append(a[m] * y1c[bi * M + m][d]
                               + b[m] * y2c[bi * M + m][d] + c[m] * u[bi][t][d])
                ynew.append(row)
        for bi in range(B * M):
            y2c[bi] = y1c[bi][:]
            y1c[bi] = ynew[bi][:]
        ys.append(ynew)
    y = [[ys[t][bi * M:(bi + 1) * M] for t in range(C)] for bi in range(B)]
    return y, y1c, y2c


def ref_evidence(u, gain, pos_gain, bias, chunked: bool, length: int = 0,
                 es0: Optional[List[float]] = None):
    B, L, D = len(u), len(u[0]), len(u[0][0])
    ws, es_final = [], [es0[b] if es0 else 0.0 for b in range(B)]
    for b in range(B):
        row = []
        es = es0[b] if es0 else 0.0
        for t in range(L):
            es += sum(u[b][t][d] ** 2 for d in range(D))
            cnt = (t + 1) if not chunked else (length + t + 1)
            c = es / cnt
            logit = gain * math.log1p(c) + pos_gain * math.log1p(cnt) + bias
            row.append(1.0 / (1.0 + math.exp(-logit)))
        ws.append(row)
        es_final[b] = es
    return ws, es_final


def ref_uncertainty(lg, lc, logit, eps):
    B, L, D = len(lg), len(lg[0]), len(lg[0][0])
    out = []
    sig = 1.0 / (1.0 + math.exp(-logit))
    for b in range(B):
        row = []
        for t in range(L):
            num = sum(abs(lg[b][t][d] - lc[b][t][d]) for d in range(D))
            den = sum(abs(lg[b][t][d]) + abs(lc[b][t][d]) for d in range(D)) + eps
            row.append(1.0 - sig * (num / den))
        out.append(row)
    return out


def ref_ce(logits, labels):
    V = len(logits[0])
    tot, cnt = 0.0, 0
    for r, y in enumerate(labels):
        if y == -100:
            continue
        x = logits[r]
        mx = max(x)
        lse = mx + math.log(sum(math.exp(v - mx) for v in x))
        tot += lse - x[y]
        cnt += 1
    return tot / cnt


def ref_adamw(p, g, m, v, frozen, lr, b1, b2, eps, wd, step):
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    pn, mn, vn = [], [], []
    for i in range(len(p)):
        if frozen[i]:
            pn.append(p[i]); mn.append(m[i]); vn.append(v[i])
        else:
            mnew = b1 * m[i] + (1 - b1) * g[i]
            vnew = b2 * v[i] + (1 - b2) * g[i] * g[i]
            upd = (mnew / bc1) / (math.sqrt(vnew / bc2) + eps) + wd * p[i]
            pn.append(p[i] - lr * upd)
            mn.append(mnew)
            vn.append(vnew)
    return pn, mn, vn


def flat3(x):
    return [v for plane in x for row in plane for v in row]


def flat4(x):
    return [v for cube in x for plane in cube for row in plane for v in row]


def flat2(x):
    return [v for row in x for v in row]


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
def main() -> int:
    dll = os.path.join(ROOT, "simplcore", "csrc", "build",
                       "sc_ref_backend.dll" if os.name == "nt"
                       else "libsc_ref_backend.so")
    if not os.path.exists(dll):
        print(f"未找到 {dll}，先跑 scripts/build_ref_backend.sh")
        return 2
    L = Lib(dll)

    v = ctypes.c_int32()
    L.lib.sc_abi_version.restype = ctypes.c_int32
    check("sc_abi_version == 1", L.lib.sc_abi_version() == 1)

    random.seed(2026)

    def rnd(n, s=1.0):
        return [random.gauss(0, s) for _ in range(n)]

    # -- emb_lookup ----------------------------------------------------------
    V, D, B, T = 7, 3, 2, 4
    w = rnd(V * D)
    ids = [random.randrange(V) for _ in range(B * T)]
    wbuf, ibuf = L.f32(w), L.i64(ids)
    obuf = L.f32([0.0] * (B * T * D))
    rc = L.call("emb_lookup", [(wbuf, (V, D)), (ibuf, (B, T))], [(obuf, (B, T, D))])
    exp = [w[i * D + d] for i in ids for d in range(D)]
    ok, err = close(list(obuf), exp)
    check("emb_lookup", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- rmsnorm -------------------------------------------------------------
    rows, D = 4, 5
    x = [rnd(D) for _ in range(rows)]
    wt = [abs(v) + 0.5 for v in rnd(D)]
    eps = 1e-6
    xbuf, wbuf = L.f32(flat2(x)), L.f32(wt)
    obuf = L.f32([0.0] * rows * D)
    rc = L.call("rmsnorm", [(xbuf, (rows, D)), (wbuf, (D,))], [(obuf, (rows, D))],
                scalars=[eps])
    exp = flat2(ref_rmsnorm(x, wt, eps))
    ok, err = close(list(obuf), exp)
    check("rmsnorm", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- gemm ----------------------------------------------------------------
    M_, K, N = 3, 4, 2
    a = [rnd(K) for _ in range(M_)]
    b = [rnd(N) for _ in range(K)]
    abuf, bbuf = L.f32(flat2(a)), L.f32(flat2(b))
    obuf = L.f32([0.0] * M_ * N)
    rc = L.call("gemm", [(abuf, (M_, K)), (bbuf, (K, N))], [(obuf, (M_, N))])
    exp = flat2(ref_gemm(a, b))
    ok, err = close(list(obuf), exp)
    check("gemm", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- swiglu / residual_add -------------------------------------------------
    n = 6
    g, u = rnd(n), rnd(n)
    gbuf, ubuf = L.f32(g), L.f32(u)
    obuf = L.f32([0.0] * n)
    rc = L.call("swiglu", [(gbuf, (2, 3)), (ubuf, (2, 3))], [(obuf, (2, 3))])
    exp = [gv * (1 / (1 + math.exp(-gv))) * uv for gv, uv in zip(g, u)]
    ok, err = close(list(obuf), exp)
    check("swiglu", rc == 0 and ok, f"worst_rel={err:.2e}")
    obuf = L.f32([0.0] * n)
    rc = L.call("residual_add", [(gbuf, (2, 3)), (ubuf, (2, 3))], [(obuf, (2, 3))])
    ok, err = close(list(obuf), [gv + uv for gv, uv in zip(g, u)])
    check("residual_add", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- spectral_causal_conv --------------------------------------------------
    B, T, D = 1, 6, 2
    u = [[rnd(D) for _ in range(T)] for _ in range(B)]
    k = [rnd(D) for _ in range(T)]
    ubuf, kbuf = L.f32(flat3(u)), L.f32(flat2(k))
    obuf = L.f32([0.0] * B * T * D)
    rc = L.call("spectral_causal_conv", [(ubuf, (B, T, D)), (kbuf, (T, D))],
                [(obuf, (B, T, D))])
    exp = flat3(ref_spectral_conv(u, k))
    ok, err = close(list(obuf), exp)
    check("spectral_causal_conv(直接时域)", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- causal_depthwise_conv：带 buf / 不带 buf / T < K-1 边界 ------------------
    for tag, (B, T, D, K, P) in {
        "带buf(P=K-1)": (1, 4, 2, 3, 2),
        "无buf": (2, 5, 2, 4, 0),
        "短段边界(T=1,P=0,K=4)": (1, 1, 2, 4, 0),
    }.items():
        uu_t = [[rnd(D) for _ in range(T)] for _ in range(B)]
        w_t = [[rnd(1)[0] for _ in range(K)] for _ in range(D)]
        buf_t = [[rnd(D) for _ in range(P)] for _ in range(B)] if P else None
        ubuf = L.f32(flat3(uu_t))
        wbuf = L.f32(flat2(w_t))
        bbuf = L.f32(flat3(buf_t)) if buf_t else None
        nb_len = ctypes.c_int64(0)
        obuf = L.f32([1e9] * B * T * D)          # 预填垃圾：验证内核有清零
        nbbuf = L.f32([0.0] * B * max(K - 1, 0) * D)
        rc = L.call("causal_depthwise_conv",
                    [(ubuf, (B, T, D)), (wbuf, (D, 1, K)),
                     (bbuf, (B, P, D)) if bbuf is not None else (None, ())],
                    [(obuf, (B, T, D)), (nbbuf, (B, max(K - 1, 0), D))],
                    extra_outs=[ctypes.pointer(nb_len)])
        exp_y, exp_nb = ref_depthwise(uu_t, w_t, buf_t)
        ok, err = close(list(obuf), flat3(exp_y))
        ok2, err2 = close(list(nbbuf)[:B * len(exp_nb[0]) * D],
                          flat3(exp_nb)) if exp_nb[0] else (True, 0.0)
        check(f"causal_depthwise_conv[{tag}]", rc == 0 and ok and ok2
              and int(nb_len.value) == len(exp_nb[0]),
              f"y_rel={err:.2e} nb_rel={err2:.2e} new_len={int(nb_len.value)}"
              f"(exp {len(exp_nb[0])})")

    # -- chunked_iir_scan：数值 + 分块/整段字节级一致 -----------------------------
    B, C, D, M = 1, 5, 2, 3
    uu = [[rnd(D) for _ in range(C)] for _ in range(B)]
    y1_0 = [rnd(D) for _ in range(B * M)]
    y2_0 = [rnd(D) for _ in range(B * M)]
    a = [rnd(1)[0] * 1.5 for _ in range(M)]
    b = [-abs(v) for v in rnd(M)]
    c = [abs(v) for v in rnd(M)]
    uubuf, y1buf, y2buf = L.f32(flat3(uu)), L.f32(flat2(y1_0)), L.f32(flat2(y2_0))
    abuf, bbuf, cbuf = L.f32(a), L.f32(b), L.f32(c)
    obuf = L.f32([0.0] * B * C * M * D)
    oy1, oy2 = L.f32([0.0] * B * M * D), L.f32([0.0] * B * M * D)
    rc = L.call("chunked_iir_scan",
                [(uubuf, (B, C, D)), (y1buf, (B, M, D)), (y2buf, (B, M, D)),
                 (abuf, (M,)), (bbuf, (M,)), (cbuf, (M,)), (None, ())],
                [(obuf, (B, C, M, D)), (oy1, (B, M, D)), (oy2, (B, M, D))],
                iargs=[0])
    exp_y, exp_y1, exp_y2 = ref_iir_scan(uu, y1_0, y2_0, a, b, c)
    ok, err = close(list(obuf), flat4(exp_y))
    ok1, err1 = close(list(oy1), flat2(exp_y1))
    ok2, err2 = close(list(oy2), flat2(exp_y2))
    check("chunked_iir_scan(数值)", rc == 0 and ok and ok1 and ok2,
          f"y={err:.2e} y1={err1:.2e} y2={err2:.2e}")

    # 分块 vs 整段：字节级一致（P0-chunked-carry 同构检查）
    def run_chunk(chunks):
        y1c, y2c = L.f32(flat2(y1_0)), L.f32(flat2(y2_0))
        outs = []
        off = 0
        for ch in chunks:
            uc = L.f32(flat2([uu[0][off + i] for i in range(ch)]))
            o = L.f32([0.0] * B * ch * M * D)
            n1, n2 = L.f32([0.0] * B * M * D), L.f32([0.0] * B * M * D)
            rc2 = L.call("chunked_iir_scan",
                         [(uc, (B, ch, D)), (y1c, (B, M, D)), (y2c, (B, M, D)),
                          (abuf, (M,)), (bbuf, (M,)), (cbuf, (M,)), (None, ())],
                         [(o, (B, ch, M, D)), (n1, (B, M, D)), (n2, (B, M, D))],
                         iargs=[0])
            assert rc2 == 0
            outs.append(bytes(o))
            y1c, y2c = n1, n2
            off += ch
        return b"".join(outs), bytes(y1c), bytes(y2c)

    whole_y, whole_1, whole_2 = run_chunk([C])
    part_y, part_1, part_2 = run_chunk([2, 3])
    check("chunked-carry 字节级一致",
          whole_y == part_y and whole_1 == part_1 and whole_2 == part_2)

    # -- evidence_gate parallel vs chunk ---------------------------------------
    B, Tn, D = 2, 6, 3
    uu = [[rnd(D) for _ in range(Tn)] for _ in range(B)]
    gain, pos_gain, bias = 1.0, 0.3, -0.2
    ubuf = L.f32(flat3(uu))
    obuf = L.f32([0.0] * B * Tn)
    rc = L.call("evidence_gate_parallel", [(ubuf, (B, Tn, D))], [(obuf, (B, Tn, 1))],
                scalars=[gain, pos_gain, bias])
    exp, _ = ref_evidence(uu, gain, pos_gain, bias, chunked=False)
    ok, err = close(list(obuf), flat2(exp))
    check("evidence_gate_parallel", rc == 0 and ok, f"worst_rel={err:.2e}")

    length = 10
    es0 = [5.0 * random.random() for _ in range(B)]
    esbuf = L.f32(es0)
    obuf = L.f32([0.0] * B * Tn)
    oes = L.f32([0.0] * B)
    rc = L.call("evidence_gate_chunk", [(ubuf, (B, Tn, D)), (esbuf, (B, 1))],
                [(obuf, (B, Tn, 1)), (oes, (B, 1))],
                scalars=[gain, pos_gain, bias], iargs=[length])
    exp, exp_es = ref_evidence(uu, gain, pos_gain, bias, chunked=True,
                               length=length, es0=es0)
    ok, err = close(list(obuf), flat2(exp))
    oke, erre = close(list(oes), exp_es)
    check("evidence_gate_chunk", rc == 0 and ok and oke,
          f"w={err:.2e} es={erre:.2e}")

    # -- uncertainty_gate -------------------------------------------------------
    lg = [[rnd(D) for _ in range(Tn)] for _ in range(B)]
    lc = [[rnd(D) for _ in range(Tn)] for _ in range(B)]
    lgb, lcb = L.f32(flat3(lg)), L.f32(flat3(lc))
    obuf = L.f32([0.0] * B * Tn)
    rc = L.call("uncertainty_gate", [(lgb, (B, Tn, D)), (lcb, (B, Tn, D))],
                [(obuf, (B, Tn, 1))], scalars=[0.7, 1e-6])
    exp = flat2(ref_uncertainty(lg, lc, 0.7, 1e-6))
    ok, err = close(list(obuf), exp)
    check("uncertainty_gate", rc == 0 and ok, f"worst_rel={err:.2e}")

    # -- ce_loss -----------------------------------------------------------------
    R, V = 3, 5
    logits = [rnd(V, 2.0) for _ in range(R)]
    labels = [1, -100, 4]
    lgb, lbb = L.f32(flat2(logits)), L.i64(labels)
    obuf = L.f32([0.0])
    rc = L.call("ce_loss", [(lgb, (R, V)), (lbb, (R,))], [(obuf, ())])
    exp = ref_ce(logits, labels)
    ok, err = close(list(obuf), [exp])
    check("ce_loss(-100 忽略)", rc == 0 and ok, f"rel={err:.2e}")

    # -- adamw_frozen_mask ---------------------------------------------------------
    n = 8
    p, g = rnd(n), [v * 0.1 for v in rnd(n)]
    m0, v0 = [0.0] * n, [0.0] * n
    frozen = [1 if i % 2 == 0 else 0 for i in range(n)]
    pbuf, gbuf = L.f32(p), L.f32(g)
    mbuf, vbuf = L.f32(m0), L.f32(v0)
    fbuf = (ctypes.c_uint8 * n)(*frozen)
    pnb, mnb, vnb = L.f32([0.0] * n), L.f32([0.0] * n), L.f32([0.0] * n)
    rc = L.call("adamw_frozen_mask",
                [(pbuf, (n,)), (gbuf, (n,)), (mbuf, (n,)), (vbuf, (n,)),
                 (fbuf, (n,))],
                [(pnb, (n,)), (mnb, (n,)), (vnb, (n,))],
                scalars=[1e-3, 0.9, 0.999, 1e-8, 0.01], iargs=[5])
    ep, em, ev = ref_adamw(p, g, m0, v0, frozen, 1e-3, 0.9, 0.999, 1e-8, 0.01, 5)
    okp, errp = close(list(pnb), ep)
    okm, errm = close(list(mnb), em)
    okv, errv = close(list(vnb), ev)
    frozen_bit = all(list(pnb)[i] == pbuf[i] and list(mnb)[i] == mbuf[i]
                     and list(vnb)[i] == vbuf[i]
                     for i in range(n) if frozen[i])
    check("adamw_frozen_mask", rc == 0 and okp and okm and okv and frozen_bit,
          f"p={errp:.2e} m={errm:.2e} v={errv:.2e} 冻结位逐位={frozen_bit}")

    print()
    if FAILURES:
        print(f"结果：{len(FAILURES)} 项失败 → {FAILURES}")
        return 1
    print("结果：全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
