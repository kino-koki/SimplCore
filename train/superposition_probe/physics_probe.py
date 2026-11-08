"""
physics_probe.py — 物理世界边界探针。

目标不是再刷 benchmark 倍率，而是把「真实硬件约束」逐条压上去，
主动找让 1.0000 掉下来的点，并在掉点之前把可优化的物理适配做满。

五组真实约束
============
A 有限精度累加器：fp64 / fp32 / bf16 / fp16 / int8（每步 cast 状态寄存器）
B 长度 × 精度联合相图：L = 9 … 4097，ρ^τ 衰减 vs 低精度下溢
C 状态容量 / 干扰：n_pairs 4…128，输入噪声 0…0.3
D 稳定性权衡：ρ<1（BIBO 稳定但远端下溢） vs ρ=1（无衰减但临界稳定）
E 实测吞吐与状态字节（不再用解析估算）
F 抗量化加固：在 fp16 / int8 下把掉点尽量往后推（"做到最强"）

严格保持：LTI / Toeplitz / O(1) 状态 / 严格因果 / 无 N×N / 无对照模型实现。
评测走 token-id 索引的解析式递推（不建 autograd 图、不存 one-hot），
与 forward() 逐样本一致。

用法
====
  python physics_probe.py --mode train
  python physics_probe.py --mode consistency
  python physics_probe.py --mode precision
  python physics_probe.py --mode capacity
  python physics_probe.py --mode stability
  python physics_probe.py --mode hardening
  python physics_probe.py --mode throughput
  python physics_probe.py --mode all
"""
import argparse
import json
import os
import time
import numpy as np

import tensor_spectral_probe as T
import sgc_fixed as S
import task_utils as U
from mini_autograd import T as AT, nll_from_logits, free_graph

OUT = "results"
CKPT = f"{OUT}/physics_ckpt.npz"
ARM = "W1_code64_lr"        # vocab=64, R=64, N=110, P=2, 10264 params
PRECISIONS = ("fp64", "fp32", "bf16", "fp16", "int8")


# ---------------------------------------------------------------- 低精度算子
def _bf16(x):
    """round-to-nearest-even 截断 fp32 尾数，模拟 bfloat16 寄存器。"""
    a = np.asarray(x, dtype=np.float32)
    i = a.view(np.uint32).copy()
    rnd = 0x7FFF + ((i >> 16) & 1)
    i = (i.astype(np.uint64) + rnd) & 0xFFFF0000
    return i.astype(np.uint32).view(np.float32).astype(np.float64)


def cast(x, precision, scale=None):
    """把数值压进目标精度的物理寄存器，再升回 fp64 供统计。

    int8 且给定 scale 时使用**固定量化步长**（校准一次、所有 L 复用），
    模拟真实定点硬件；scale 可为标量或与 x 最后一维同形的 per-channel 数组。
    """
    if precision == "fp64":
        return np.asarray(x, dtype=np.float64)
    if precision == "fp32":
        return np.asarray(x, dtype=np.float32).astype(np.float64)
    if precision == "bf16":
        return _bf16(x)
    if precision == "fp16":
        return np.asarray(x, dtype=np.float16).astype(np.float64)
    if precision == "int8":
        a = np.asarray(x, dtype=np.float64)
        if scale is None:
            sc = max(float(np.abs(a).max()) / 127.0, 1e-12)
        else:
            sc = np.asarray(scale, dtype=np.float64)
            if sc.ndim == 1:
                sc = sc.reshape((1,) * (a.ndim - 1) + (sc.size,))
            sc = np.maximum(sc, 1e-12)
        return np.clip(np.round(a / sc), -127, 127) * sc
    raise ValueError(precision)


# ---------------------------------------------------------------- checkpoint
def load_params():
    if not os.path.exists(CKPT):
        raise SystemExit("先运行 --mode train")
    z = np.load(CKPT)
    return {k: AT(z[k]) for k in z.files}


def train_save(steps=800, seed=0):
    os.makedirs(OUT, exist_ok=True)
    print(f"训练 {ARM} ({steps} 步) ...", flush=True)
    r = T.run_train(ARM, seed, steps=steps)
    P = T._LAST_P
    np.savez(CKPT, **{k: v.d for k, v in P.items()})
    meta = dict(arm=ARM, steps=steps, seed=seed, params=r["params"],
                final_acc=r["final_acc"], perm_acc=r["perm_acc"],
                guards=r["guards_all_passed"])
    json.dump(meta, open(f"{OUT}/physics_ckpt_meta.json", "w"), indent=1)
    print(f"  acc={r['final_acc']:.4f} perm={r['perm_acc']:.4f} "
          f"params={r['params']} guards={r['guards_all_passed']}", flush=True)
    print(f"  已保存 {CKPT}", flush=True)


# ---------------------------------------------------------------- 任务（索引式）
def build_ids(n_pairs, n_samples, seed, vocab=64):
    """返回 token-id 序列 (B,L) 与 target。

    允许 key 重复（容量/干扰实验需要），但强制 query 指向的 key
    全序列唯一 ⇒ 答案仍唯一。t=0 的前驱为零向量。
    """
    vk, vv, v0 = U.split_vocab(vocab)
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    ids = np.zeros((n_samples, L), dtype=np.int32)
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        keys = r.choice(vk - 1, size=n_pairs, replace=True)
        vals = r.choice(vv, size=n_pairs, replace=True) + v0
        j = int(r.integers(0, n_pairs))
        q = vk - 1
        keys[j] = q                       # 唯一 query key
        ids[i, 0::2][:n_pairs] = keys
        ids[i, 1::2][:n_pairs] = vals
        ids[i, L - 1] = q
        Y[i] = vals[j]
    return ids, Y


# ---------------------------------------------------------------- 解析式模拟
def simulate(P, ids, Y, precision="fp64", rho_override=None,
             compensate=False, renorm=False, noise=0.0, seed=0,
             int8_scale=None, state_scale=None):
    """低精度递推评测。每步把状态寄存器 cast 到目标精度 —— 真实硬件语义。

    ids[:, t] 是当前 token；前驱 key 取 ids[:, t-1]（t=0 为零向量）。
    """
    cfg = T.ARMS[ARM]
    V = T._vocab(cfg)
    R, N, Pp = cfg["R"], cfg["N"], cfg["P"]
    B, L = ids.shape
    rng = np.random.default_rng(seed + 777)

    Code = T._fixed_code(R, V, cfg.get("code_kind", "hadamard"))
    rho = (T.RHO_MIN + (1 - T.RHO_MIN) * T._sig(P["p"].d)).reshape(Pp, N)
    if rho_override is not None:
        rho = np.full_like(rho, float(rho_override))
    w = P["w"].d.reshape(Pp, N)
    E, Wv, Wout = P["E"].d, P["Wv"].d, P["Wout"].d

    def c(x): return cast(x, precision, scale=int8_scale)
    if state_scale is not None and hasattr(state_scale, "ndim") and state_scale.ndim == 2:
        _ss = np.asarray(state_scale)[None, :, None, :]     # (1,P,1,N)
    else:
        _ss = state_scale
    def cs(x): return cast(x, precision,
                           scale=_ss if _ss is not None else int8_scale)
    if state_scale is not None and hasattr(state_scale, "ndim") and state_scale.ndim == 2:
        _ss = np.asarray(state_scale)[None, :, None, :]     # (1,P,1,N)
    else:
        _ss = state_scale
    def cs(x): return cast(x, precision,
                           scale=_ss if _ss is not None else int8_scale)
    rho_c = c(rho)
    w_c = c(w)
    Wout_c = c(Wout)
    Code_c = c(Code)
    E_c = c(E)
    Wv_c = c(Wv)

    s = np.zeros((B, Pp, R, N), dtype=np.float64)
    for t in range(L):
        cur = ids[:, t]
        emb = E_c[cur]
        if noise > 0:
            emb = c(emb + rng.normal(0, noise, emb.shape))
        b_t = c(emb @ Wv_c)                                   # (B,N)
        if t == 0:
            a_t = np.zeros((B, R), dtype=np.float64)
        else:
            a_t = Code_c[ids[:, t - 1]]                       # (B,R)
        u_t = c(a_t[:, :, None] * b_t[:, None, :])
        s = cs(rho_c[None, :, None, :] * s + u_t[:, None, :, :])

    q = Code_c[ids[:, L - 1]]                                 # (B,R)
    zr = c((s * q[:, None, :, None]).sum(axis=2))
    z = c((zr * w_c[None, :, :]).sum(axis=1))

    if compensate:
        # 补偿指数衰减：读出端除以 Σ_τ ρ^τ 的解析幅度（逐 p 通道平均）
        amps = np.array([np.sum(rho_c[p][:, None] ** np.arange(L)[None, :])
                         for p in range(Pp)])
        amp = max(float(amps.mean()), 1e-12)
        z = c(z / amp)
    if renorm:
        # 抗 int8 动态范围：把 z 归一化到单位 RMS 再进输出投影
        rms = max(float(np.sqrt((z * z).mean())), 1e-12)
        z = c(z / rms)

    logits = c(z @ Wout_c)
    return (logits.argmax(-1) == Y).astype(int)


def build_prefix_ids(Lmax, n_samples, seed, vocab=64):
    """跨长度用**同一批样本**：生成 Lmax 长序列，每个短 L 取前缀并
    把 query 重放到 L-1（保证 query key 全序列唯一）。

    消除「不同 L 用不同随机样本」带来的假非单调。返回 {L: (ids, Y)}。
    """
    vk, vv, v0 = U.split_vocab(vocab)
    r = np.random.default_rng(seed)
    nmax = (Lmax - 1) // 2
    ids_full = np.zeros((n_samples, Lmax), dtype=np.int32)
    for i in range(n_samples):
        keys = r.choice(vk - 1, size=nmax, replace=True)
        vals = r.choice(vv, size=nmax, replace=True) + v0
        ids_full[i, 0::2][:nmax] = keys
        ids_full[i, 1::2][:nmax] = vals
    q = vk - 1                       # 唯一 query key，不出现在任何 key 位置
    out = {}
    for L in [9, 17, 65, 257, 1025, 2049, 4097, 8193, 16385]:
        if L > Lmax: continue
        n = (L - 1) // 2
        ids = ids_full[:, :L].copy()
        j = int(r.integers(0, n))
        ids[:, 2 * j] = q
        Y = ids_full[:, 2 * j + 1]
        ids[:, L - 1] = q
        out[L] = (ids, Y)
    return out


def ids_to_onehot(ids, vocab=64):
    B, L = ids.shape
    X = np.zeros((B, L, vocab)); Xp = np.zeros_like(X)
    X[np.arange(B)[:, None], np.arange(L)[None, :], ids] = 1.0
    Xp[:, 1:] = X[:, :-1]
    return X, Xp




# ---------------------------------------------------------------- I 固定量程校准
def calibrate_scales(P, L=16385, n_samples=8, seed=7000, mode="perchannel"):
    """在 fp64 无量化递推下测量状态寄存器量程，得到**固定** int8 步长。

    mode='scalar'     : 单一全局 scale
    mode='perchannel' : 每个 (p,n) 通道独立 scale（真实定点加速器常见）
    校准一次，所有 L 复用 —— 消除动态 scale 随 L 重算造成的假非单调。
    """
    cfg = T.ARMS[ARM]; R, N, Pp = cfg["R"], cfg["N"], cfg["P"]
    ids, Y = build_ids((L - 1) // 2, n_samples, seed=seed)
    Code = T._fixed_code(R, T._vocab(cfg), cfg.get("code_kind", "hadamard"))
    rho = (T.RHO_MIN + (1 - T.RHO_MIN) * T._sig(P["p"].d)).reshape(Pp, N)
    E, Wv = P["E"].d, P["Wv"].d
    B = len(ids)
    s = np.zeros((B, Pp, R, N))
    peak = np.zeros((Pp, N))
    for t in range(ids.shape[1]):
        b_t = E[ids[:, t]] @ Wv
        a_t = np.zeros((B, R)) if t == 0 else Code[ids[:, t - 1]]
        u_t = a_t[:, :, None] * b_t[:, None, :]
        s = rho[None, :, None, :] * s + u_t[:, None, :, :]
        peak = np.maximum(peak, np.abs(s).max(axis=(0, 2)))
    sc = np.maximum(peak / 127.0, 1e-12)
    if mode == "scalar":
        return float(sc.max()), dict(global_scale=float(sc.max()),
                                    peak_max=float(peak.max()))
    return sc, dict(perchannel_min=float(sc.min()),
                    perchannel_max=float(sc.max()), peak_max=float(peak.max()))


def run_fixedscale():
    """固定量程 + 同一样本前缀：定位并消除 fp16/int8 的非单调。"""
    print("\n=== I 固定量程 + 同一样本前缀 ===", flush=True)
    P = load_params()
    P2 = {k: AT(np.load(f"{OUT}/physics_ckpt_rho0.999.npz")[k])
          for k in np.load(f"{OUT}/physics_ckpt_rho0.999.npz").files}
    pref = build_prefix_ids(16385, 16, seed=4242)
    Ls = sorted(pref)
    fn = f"{OUT}/physics_fixedscale.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    res["lengths"] = Ls
    res.setdefault("conditions", {})

    conds = [
        ("fp32_dynamic",      dict(precision="fp32")),
        ("fp16_dynamic",      dict(precision="fp16")),
        ("int8_dynamic",      dict(precision="int8")),
        ("int8_fixed_scalar", dict(precision="int8", int8_scale=None)),  # 占位，下面覆盖
    ]
    scalar, meta_s = calibrate_scales(P2, mode="scalar")
    perchan, meta_p = calibrate_scales(P2, mode="perchannel")
    res["calibration"] = {"scalar": meta_s, "perchannel": meta_p}
    conds = [
        ("fp32",               dict(precision="fp32")),
        ("fp16",               dict(precision="fp16")),
        ("int8_dynamic",       dict(precision="int8")),
        ("int8_fixed_scalar",  dict(precision="int8", int8_scale=scalar)),
        ("int8_fixed_perchan", dict(precision="int8", int8_scale=scalar,
                                    state_scale=perchan)),
    ]
    for name, kw in conds:
        if name in res["conditions"] and len(res["conditions"][name]) == len(Ls):
            print(f"  {name:20s} skip", flush=True); continue
        row = {}
        for L in Ls:
            ids, Y = pref[L]
            row[f"L{L}"] = float(simulate(P2, ids, Y, **kw).mean())
        res["conditions"][name] = row
        print(f"  {name:20s} " + " ".join(f"{row[f'L{L}']:.2f}" for L in Ls),
              flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    print("  长度:              " + " ".join(f"{L:5d}" for L in Ls), flush=True)
    return res


# ---------------------------------------------------------------- H 从头固定 ρ 重训
def retrain_fixed_rho(rho=0.999, steps=800, seed=0):
    """补齐「ρ 是后验覆盖」的缺口：从头训练时就把 ρ 冻结为常数，
    w 与其余权重照常学。验证不是靠事后挑 ρ 侥幸。"""
    cfg = T.ARMS[ARM]
    V = T._vocab(cfg); npairs = cfg.get("n_pairs", U.N_PAIRS)
    P = T.make_params(cfg, seed)
    pval = np.log((rho - T.RHO_MIN) / (1 - rho))
    P["p"].d[:] = pval                       # ρ 初值即目标值
    Xtr, Xptr, Ytr, _ = U.build(npairs, 2000, seed=1000 + seed, vocab=V)
    ADAM = {}
    lr = cfg.get("lr", U.LR)
    for step in range(1, steps + 1):
        if cfg.get("schedule") == "cosine":
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        else:
            cur_lr = lr
        bs = cfg.get("batch", U.B)
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), bs,
                                                                 replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        for v in P.values(): v.g[...] = 0.0
        lg, st = T.forward(ARM, P, X, Xp, X.shape[1])
        loss = nll_from_logits(lg, Y)
        loss.backward()
        free_graph(loss)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, v in P.items():
            if name == "p":                  # 冻结 ρ
                continue
            k = "m_" + name
            if k not in ADAM:
                ADAM[k] = np.zeros_like(v.d); ADAM["v_" + name] = np.zeros_like(v.d)
            m, vv = ADAM[k], ADAM["v_" + name]
            m *= b1; m += (1 - b1) * v.g
            vv *= b2; vv += (1 - b2) * v.g * v.g
            v.d -= cur_lr * (m / (1 - b1 ** step)) / (np.sqrt(vv / (1 - b2 ** step)) + eps)
        P["p"].d[:] = pval                  # 每步硬复位，杜绝数值漂移

    fn = f"{OUT}/physics_ckpt_rho{rho}.npz"
    np.savez(fn, **{k: v.d for k, v in P.items()})
    rho_eff = float((T.RHO_MIN + (1 - T.RHO_MIN) * T._sig(P["p"].d)).mean())
    print(f"  已保存 {fn}  ρ_eff={rho_eff:.6f}", flush=True)
    return P, rho_eff


def run_retrain():
    print("\n=== H 从头固定 ρ=0.999 重训（补齐后验覆盖缺口）===", flush=True)
    P, rho_eff = retrain_fixed_rho(0.999, steps=800, seed=0)
    res = {"rho_eff": rho_eff, "lengths": [1025, 4097, 8193, 16385],
           "acc": {}, "heldout": {}}
    # held-out（同分布）
    Xte, Xpte, Yte, _ = U.build(4, 512, seed=9000)
    ids = np.argmax(Xte, -1).astype(np.int32)
    for prec in ("fp64", "fp32", "fp16", "int8"):
        res["heldout"][prec] = float(simulate(P, ids, Yte, precision=prec).mean())
    print("  held-out: " + " ".join(f"{k}:{v:.4f}" for k, v in res["heldout"].items()),
          flush=True)
    for L in res["lengths"]:
        ids2, Y2 = build_ids((L - 1) // 2, 16, seed=7000 + (L - 1) // 2)
        res["acc"][f"L{L}"] = {}
        for prec in ("fp32", "fp16"):
            res["acc"][f"L{L}"][prec] = float(
                simulate(P, ids2, Y2, precision=prec).mean())
        print(f"  L={L:5d}  " +
              " ".join(f"{k}:{v:.2f}" for k, v in res["acc"][f"L{L}"].items()),
              flush=True)
    json.dump(res, open(f"{OUT}/physics_retrain.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- 一致性
def run_consistency(P):
    ids, Y = build_ids(4, 64, seed=42)
    X, Xp = ids_to_onehot(ids)
    c1 = T.eval_correct(ARM, P, X, Xp, Y)
    c2 = simulate(P, ids, Y, precision="fp64")
    ok = bool(np.array_equal(c1, c2))
    print(f"一致性: forward={c1.mean():.4f} analytic={c2.mean():.4f} "
          f"逐样本相同={ok}", flush=True)
    json.dump(dict(forward=float(c1.mean()), analytic=float(c2.mean()),
                   identical=ok),
              open(f"{OUT}/physics_consistency.json", "w"), indent=1)
    return ok


# ---------------------------------------------------------------- A/B 精度相图
def run_precision(P):
    print("\n=== A/B 有限精度 × 长度相图 ===", flush=True)
    lengths = [9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097]
    res = {}
    for prec in PRECISIONS:
        row = {}
        for L in lengths:
            n = (L - 1) // 2
            ids, Y = build_ids(n, 32, seed=7000 + n)
            row[f"L{L}"] = float(simulate(P, ids, Y, precision=prec).mean())
        res[prec] = row
        print(f"  {prec:5s} " + " ".join(f"{row[f'L{L}']:.2f}" for L in lengths),
              flush=True)
    print("  长度:  " + " ".join(f"{L:4d}" for L in lengths), flush=True)
    json.dump(dict(lengths=lengths, phase=res),
              open(f"{OUT}/physics_precision.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- C 容量/干扰
def run_capacity(P):
    print("\n=== C 状态容量 × 干扰（fp32）===", flush=True)
    cfg = T.ARMS[ARM]
    state_n = cfg["P"] * cfg["R"] * cfg["N"]
    pairs = [4, 8, 16, 32, 64, 128]
    noises = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30]
    res = {"state_scalars": int(state_n), "fp16_bits": int(state_n * 16),
           "pairs": {}, "noise": {}}
    for n in pairs:
        ids, Y = build_ids(n, 128, seed=8000 + n)
        acc = float(simulate(P, ids, Y, precision="fp32").mean())
        vk, vv, _ = U.split_vocab(64)
        need = n * (np.log2(vk) + np.log2(vv))
        res["pairs"][f"n{n}"] = dict(acc=acc, info_bit_lowerbound=float(need))
        print(f"  n_pairs={n:3d} L={2*n+1:4d} acc={acc:.4f} "
              f"所需 ≥{need:.0f} bit / 状态 {state_n*16} bit", flush=True)
    ids4, Y4 = build_ids(4, 256, seed=8100)
    for nz in noises:
        acc = float(simulate(P, ids4, Y4, precision="fp32", noise=nz,
                             seed=int(nz * 10000)).mean())
        res["noise"][f"e{nz}"] = acc
        print(f"  noise={nz:.2f} acc={acc:.4f}", flush=True)
    json.dump(res, open(f"{OUT}/physics_capacity.json", "w"), indent=1)
    return res


# ---------------------------------------------------------------- D 稳定性权衡
def run_stability(P):
    print("\n=== D ρ 稳定性 vs 长程下溢（fp16）===", flush=True)
    rhos = [0.90, 0.95, 0.98, 0.995, 0.999, 1.0]
    lengths = [257, 1025, 2049, 4097]
    fn = f"{OUT}/physics_stability.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for rho in rhos:
        key = f"rho{rho}"
        if key in res:
            print(f"  {key} skip", flush=True); continue
        row = {}
        for L in lengths:
            ids, Y = build_ids((L - 1) // 2, 24, seed=7000 + (L - 1) // 2)
            row[f"L{L}"] = float(simulate(P, ids, Y, precision="fp16",
                                          rho_override=rho).mean())
        ids, Y = build_ids(1024, 24, seed=7123)
        row["L2049_compensated"] = float(
            simulate(P, ids, Y, precision="fp16", rho_override=rho,
                     compensate=True).mean())
        res[key] = row
        print(f"  ρ={rho:<6} " + " ".join(f"{row[f'L{L}']:.2f}" for L in lengths)
              + f"  comp:{row['L2049_compensated']:.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    print("  长度:  " + " ".join(f"{L:4d}" for L in lengths) + "   comp@2049",
          flush=True)
    return res


# ---------------------------------------------------------------- F 抗量化加固
def run_hardening(P):
    print("\n=== F 抗量化加固：把掉点尽量往后推 ===", flush=True)
    configs = {
        "base":        dict(),
        "rho1":        dict(rho_override=1.0),
        "comp":        dict(compensate=True),
        "renorm":      dict(renorm=True),
        "rho1_renorm": dict(rho_override=1.0, renorm=True),
        "rho1_comp":   dict(rho_override=1.0, compensate=True),
        "full":        dict(rho_override=1.0, compensate=True, renorm=True),
    }
    lengths = [257, 1025, 2049, 4097]
    fn = f"{OUT}/physics_hardening.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for prec in ("fp16", "int8"):
        res.setdefault(prec, {})
        for name, kw in configs.items():
            if name in res[prec]:
                print(f"  [{prec}] {name} skip", flush=True); continue
            row = {}
            for L in lengths:
                ids, Y = build_ids((L - 1) // 2, 24, seed=7000 + (L - 1) // 2)
                row[f"L{L}"] = float(simulate(P, ids, Y, precision=prec,
                                              **kw).mean())
            res[prec][name] = row
            print(f"  [{prec:4s}] {name:12s} " +
                  " ".join(f"{row[f'L{L}']:.2f}" for L in lengths), flush=True)
            json.dump(res, open(fn, "w"), indent=1)
        print(f"  长度:            " + " ".join(f"{L:4d}" for L in lengths),
              flush=True)
    return res


# ---------------------------------------------------------------- G 极限前沿 + 守恒
def run_frontier(P):
    """把掉点继续往后推，并验证「加固后的常数 ρ」仍是合法的 LTI 核。"""
    print("\n=== G 极限前沿（fp16 / fp32）===", flush=True)
    cfg = T.ARMS[ARM]
    N, Pp = cfg["N"], cfg["P"]
    w = P["w"].d.reshape(Pp, N)
    fn = f"{OUT}/physics_frontier.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    res.setdefault("lengths", [4097, 8193, 16385])
    res.setdefault("acc", {}); res.setdefault("guards", {})
    lengths = res["lengths"]

    for rho in (0.995, 0.999, 0.9999, 1.0):
        for prec in ("fp16", "fp32"):
            key = f"rho{rho}_{prec}"
            row = {}
            if key not in res["acc"]:
                row = {}
                for L in lengths:
                    ids, Y = build_ids((L - 1) // 2, 8, seed=7000 + (L - 1) // 2)
                    row[f"L{L}"] = float(simulate(P, ids, Y, precision=prec,
                                                  rho_override=rho).mean())
                res["acc"][key] = row
                json.dump(res, open(fn, "w"), indent=1)
            row = res["acc"][key]
            print(f"  {key:16s} " +
                  " ".join(f"L{L}:{row[f'L{L}']:.2f}" for L in lengths),
                  flush=True)

        # 守恒：常数 ρ ⇒ K(τ)=Σ_p w_p ρ^τ 仍 Toeplitz，FFT/IIR 对偶仍成立
        Lg = 64
        rr = np.full((Pp, N), float(rho))
        tau = np.arange(Lg, dtype=np.float64)
        K = (w[:, :, None] * (rr[:, :, None] ** tau[None, None, :])).sum(0)
        Tm = S.toeplitz(K)
        dev = max(abs(Tm[n, t, ss] - Tm[n, t - 1, ss - 1])
                  for n in range(N) for t in range(1, Lg)
                  for ss in range(1, t + 1))
        u = np.random.default_rng(1).standard_normal((2, Lg, N))
        y = np.einsum("nls,bsn->bln", Tm, u)
        yf = S.fft_apply(K, u[:, :, :, None])[:, :, :, 0]
        den = max(np.abs(y).max(), 1e-12)
        rel_fft = float(np.abs(y - yf).max() / den)
        ss = np.zeros((2, Pp, N)); yi = np.zeros((2, Lg, N))
        for t in range(Lg):
            ss = rr[None, :, :] * ss + u[:, t, :][:, None, :]
            yi[:, t, :] = (w[None, :, :] * ss).sum(1)
        rel_iir = float(np.abs(y - yi).max() / den)
        u2 = u.copy(); u2[:, Lg - 1] += 10
        past = float(np.abs(y[:, :Lg - 1]
                            - np.einsum("nls,bsn->bln", Tm, u2)[:, :Lg - 1]).max())
        state_bytes = [Pp * cfg["R"] * N * 4 for _ in (9, 65, 1025, 16385)]
        if f"rho{rho}" in res["guards"]:
            json.dump(res, open(fn, "w"), indent=1)
            continue
        res["guards"][f"rho{rho}"] = dict(
            toeplitz=bool(dev < 1e-10), rel_fft=rel_fft, rel_iir=rel_iir,
            strict_causality=bool(past == 0),
            state_independent_of_L=bool(len(set(state_bytes)) == 1),
            state_bytes=state_bytes[0], BIBO_stable=bool(rho < 1.0),
            all_passed=bool(dev < 1e-10 and rel_fft < 1e-9
                            and rel_iir < 1e-9 and past == 0
                            and len(set(state_bytes)) == 1))
        json.dump(res, open(fn, "w"), indent=1)
        g = res["guards"][f"rho{rho}"]
        print(f"    guards: toeplitz={g['toeplitz']} fft={rel_fft:.1e} "
              f"iir={rel_iir:.1e} causal={g['strict_causality']} "
              f"BIBO={g['BIBO_stable']} all={g['all_passed']}", flush=True)
    json.dump(res, open(fn, "w"), indent=1)
    return res


# ---------------------------------------------------------------- E 实测吞吐
def run_throughput(P):
    print("\n=== E 实测吞吐 / 状态字节（不再用解析估算）===", flush=True)
    cfg = T.ARMS[ARM]
    state_n = cfg["P"] * cfg["R"] * cfg["N"]
    res = {"state_scalars": int(state_n),
           "state_bytes_fp32": int(state_n * 4),
           "state_bytes_fp16": int(state_n * 2),
           "independent_of_L": True, "lengths": {}}
    for L in [65, 257, 1025, 4097]:
        ids, Y = build_ids((L - 1) // 2, 8, seed=7000 + (L - 1) // 2)
        simulate(P, ids, Y, precision="fp32")           # 预热
        t0 = time.perf_counter(); reps = 3
        for _ in range(reps):
            simulate(P, ids, Y, precision="fp32")
        dt = (time.perf_counter() - t0) / reps
        tok_s = (8 * L) / dt
        res["lengths"][f"L{L}"] = dict(seconds=float(dt),
                                        tokens_per_s=float(tok_s),
                                        us_per_token=float(dt / (8 * L) * 1e6))
        print(f"  L={L:5d}  {dt*1000:8.1f} ms  {tok_s:9.0f} tok/s  "
              f"{dt/(8*L)*1e6:7.2f} µs/tok  状态={state_n*4} B", flush=True)
    json.dump(res, open(f"{OUT}/physics_throughput.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["train", "consistency", "precision", "capacity",
                             "stability", "hardening", "frontier",
                             "throughput", "retrain", "fixedscale", "all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.mode == "train":
        train_save(); return
    P = load_params()
    if a.mode == "fixedscale":
        run_fixedscale(); return
    if a.mode == "retrain":
        run_retrain(); return
    if a.mode == "consistency":
        run_consistency(P); return
    if a.mode == "precision":
        run_precision(P); return
    if a.mode == "capacity":
        run_capacity(P); return
    if a.mode == "stability":
        run_stability(P); return
    if a.mode == "hardening":
        run_hardening(P); return
    if a.mode == "frontier":
        run_frontier(P); return
    if a.mode == "throughput":
        run_throughput(P); return

    ok = run_consistency(P)
    prec = run_precision(P)
    cap = run_capacity(P)
    st = run_stability(P)
    hd = run_hardening(P)
    fr = run_frontier(P)
    th = run_throughput(P)
    rt = run_retrain()
    meta = json.load(open(f"{OUT}/physics_ckpt_meta.json"))
    json.dump(dict(arm=ARM, params=meta["params"],
                   forward_analytic_match=ok,
                   precision=prec, capacity=cap, stability=st,
                   hardening=hd, frontier=fr, throughput=th, retrain=rt),
              open(f"{OUT}/physics_summary.json", "w"), indent=1,
              ensure_ascii=False)
    print("\n写入 results/physics_summary.json", flush=True)


if __name__ == "__main__":
    main()
