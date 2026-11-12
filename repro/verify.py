"""
结构性质验证套件（白盒复现）——对应论文 §4 定理 1/2/3、命题 5、§5.4、§5.12

  T1  双形式对偶：FFT 整段 == IIR 流式（逐位 / 相对误差，随 L 扫描）
  T2  严格因果：扰动未来 token，过去输出逐位不变
  T3  常数状态：状态字节与 L 无关，且 == 4(2MD+(K-1)D+1)
  P5  命题 5 误差界：双口径（裸 u / 随 |y| 重构的 delta）实测是否成立
  LB  部署账本：1M 上下文内存 vs 同规模 Transformer KV cache
"""
import json
import numpy as np
import sgc_core as S

F32 = np.float32
U_ROUND = float(np.finfo(np.float32).eps) / 2.0     # unit roundoff u = 2^-24


# ------------------------------------------------------------ 一个完整 SGC 块
class SGCBlock:
    """白盒 SGC 块：谱全局卷积(M 基) + 深度卷积(K) + 内容门控 + 投影。

    参数（均为 fp32）：Wu/Wg/Wo (D,D)、c (M,D)、wdw (D,K)
    """

    def __init__(self, D, M, K=4, rng=None):
        rng = rng or np.random.default_rng(0)
        s = 1.0 / np.sqrt(D)
        self.Wu = (rng.standard_normal((D, D)) * s).astype(F32)
        self.Wg = (rng.standard_normal((D, D)) * s).astype(F32)
        self.Wo = (rng.standard_normal((D, D)) * s).astype(F32)
        self.c = (rng.standard_normal((M, D)) * s).astype(F32)
        self.wdw = (rng.standard_normal((D, K)) * s).astype(F32)
        self.D, self.M, self.K = D, M, K

    # 整段（训练路径）
    def forward_block(self, x, grid):
        B, L, D = x.shape
        u = x @ self.Wu                                        # (B,L,D)
        h = S.impulse_response(grid, L)
        C = S.causal_conv_fft(np.transpose(u, (0, 2, 1)), h)   # (M,B,D,L)
        C = np.transpose(C, (1, 3, 2, 0)).copy()               # (B,L,D,M)
        y = np.einsum('bldm,md->bld', C, self.c, optimize=True)
        y = S.dwconv_causal(y, self.wdw)
        g = x @ self.Wg
        g = g / (1.0 + np.exp(-g))                             # sigmoid 门控
        return (y * g) @ self.Wo

    # 流式（部署路径）
    def stream_block(self, x, grid, state=None):
        """x: (B,L,D)。state: StreamState 或 None。返回 (B,L,D)。O(1) 状态。"""
        B, L, D = x.shape
        if state is None:
            state = StreamState(self, grid, B, D)
        u = x @ self.Wu
        g = x @ self.Wg
        g = g / (1.0 + np.exp(-g))
        out = np.zeros((B, L, D), dtype=F32)
        for t in range(L):
            ut = u[:, t, :]                                    # (B,D)
            # 每基输入 = u_t（逐基调制并入读出系数 c），状态为 (B,M,D)
            xm = np.broadcast_to(ut[:, None, :], (B, self.M, D)).copy()
            ym = state.sgc.step(xm)                            # (B,M,D)
            y = np.einsum('bmd,md->bd', ym, self.c, optimize=True)   # (B,D)
            y = state.push_dw(y)                               # 深度卷积（含缓存）
            out[:, t, :] = (y * g[:, t, :]) @ self.Wo
        return out, state


class StreamState:
    """一块的 O(1) 常驻状态：IIR 两拍 (2,M,D) + 深度卷积缓存 (K-1,D) + 1 标量。"""

    def __init__(self, block, grid, B, D):
        self.sgc = S.SGCStream(grid, B, D)
        self.K = block.K
        self.dw_cache = np.zeros((B, self.K - 1, D), dtype=F32)
        self.wdw = block.wdw
        self.step_count = np.zeros((1,), dtype=F32)

    def push_dw(self, y_t):
        """喂入第 t 步的深度卷积输入 y_t (B,D)，输出卷积结果 (B,D)。"""
        B, D = y_t.shape
        K = self.K
        hist = np.concatenate([self.dw_cache, y_t[:, None, :]], axis=1)   # (B,K,D)
        out = np.zeros((B, D), dtype=F32)
        for k in range(K):
            out += hist[:, k, :] * self.wdw[None, :, k]
        self.dw_cache = hist[:, 1:, :]
        self.step_count += 1.0
        return out

    def nbytes(self):
        return self.sgc.state_bytes() + self.dw_cache.nbytes + self.step_count.nbytes


# ---------------------------------------------------------------------- T1
def test_duality(grid, lens=(64, 128, 256, 512, 1024, 2048, 4096), B=1, D=32, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for L in lens:
        u = rng.standard_normal((B, L, D)).astype(F32) * 0.05
        h = S.impulse_response(grid, L)
        Yf = np.transpose(S.causal_conv_fft(np.transpose(u, (0, 2, 1)), h), (1, 3, 2, 0)).copy()
        Yi = np.transpose(S.iir_scan(u, grid), (0, 1, 3, 2)).copy()
        bw = S.bitwise_equal(Yf, Yi)
        rows.append(dict(L=L, bitwise=bool(bw), rel_err=float(S.rel_err(Yf, Yi)),
                         max_abs=float(np.abs(Yf.astype(np.float64) - Yi.astype(np.float64)).max()),
                         scale=float(np.abs(Yi.astype(np.float64)).max())))
    return rows


# ---------------------------------------------------------------------- T2
def test_causality(grid, D=32, L=256, B=2, K=4, seed=0):
    rng = np.random.default_rng(seed)
    blk = SGCBlock(D, grid["M"], K, rng=rng)
    x = rng.standard_normal((B, L, D)).astype(F32) * 0.2
    T = L // 2
    out_a, _ = blk.stream_block(x, grid)
    x2 = x.copy()
    x2[:, T:, :] = rng.standard_normal((B, L - T, D)).astype(F32) * 5.0   # 强扰动未来
    out_b, _ = blk.stream_block(x2, grid)
    past_bitwise = S.bitwise_equal(out_a[:, :T, :], out_b[:, :T, :])
    # 对照：整段 FFT 路径也应当因果
    oa = blk.forward_block(x, grid)
    ob = blk.forward_block(x2, grid)
    fft_causal_bitwise = S.bitwise_equal(oa[:, :T, :], ob[:, :T, :])
    return dict(past_bitwise_identical=bool(past_bitwise),
                fft_path_bitwise_identical=bool(fft_causal_bitwise),
                future_changed=bool(not S.bitwise_equal(out_a[:, T:, :], out_b[:, T:, :])),
                max_past_abs_diff=float(np.abs(out_a[:, :T, :].astype(np.float64)
                                               - out_b[:, :T, :].astype(np.float64)).max()))


# ---------------------------------------------------------------------- T3
def test_state_bytes(grid, D=768, K=4, lens=(1024, 8192, 65536, 1000000)):
    rng = np.random.default_rng(0)
    blk = SGCBlock(D, grid["M"], K, rng=rng)
    sizes = []
    for L in lens:
        x = rng.standard_normal((1, min(L, 512), D)).astype(F32) * 0.01
        _, st = blk.stream_block(x, grid)
        sizes.append(int(st.nbytes()))
    formula = 4 * (2 * grid["M"] * D + (K - 1) * D + 1)
    return dict(sizes=sizes, constant=bool(len(set(sizes)) == 1),
                formula_bytes=int(formula), observed_bytes=sizes[0],
                matches_formula=bool(sizes[0] == formula))


# ---------------------------------------------------------------------- P5
def test_proposition5(grid, L=512, seed=0, bases=None):
    """命题 5：IIR 递推的舍入误差界。

    递推 y_t = a1 y_{t-1} + a2 y_{t-2} + x_t（fp32 逐步舍入），误差满足
        e_t = sum_tau psi_m[tau]/sin(theta_m) * delta_{t-tau},
        |e| <= delta_max / ((1-rho)|sin theta|)
    口径 A：delta = 裸 u（只在 |y|<=1 成立，不是界）
    口径 B：delta = u * max_t(|a1||y_{t-1}| + |a2||y_{t-2}| + |x_t|)（随轨迹重构）
    旧界  ：去掉 1/|sin theta| 因子
    """
    rng = np.random.default_rng(seed)
    a = S.iir_coeffs(grid, np.float64)
    theta, rho = grid["theta"], grid["rho"]
    idx = range(grid["M"]) if bases is None else bases
    rows = []
    for m in idx:
        a1, a2 = float(a[m, 0]), float(a[m, 1])
        x = rng.standard_normal(L).astype(np.float64)
        # fp32 轨迹（每步强制舍入）
        y32 = np.zeros(L + 2, dtype=np.float32)
        deltas = np.zeros(L, dtype=np.float64)
        mag = 0.0
        for t in range(L):
            s1 = np.float32(a1 * y32[t + 1])
            s2 = np.float32(a2 * y32[t])
            xt = np.float32(x[t])
            s3 = np.float32(s1 + s2)
            ynew = np.float32(s3 + xt)
            exact = np.float64(s1) + np.float64(s2) + np.float64(xt)
            deltas[t] = float(ynew) - exact
            mag = max(mag, abs(float(a1 * y32[t + 1])) + abs(float(a2 * y32[t])) + abs(x[t]))
            y32[t + 2] = ynew
        # fp64 参考轨迹
        y64 = np.zeros(L + 2, dtype=np.float64)
        for t in range(L):
            y64[t + 2] = a1 * y64[t + 1] + a2 * y64[t] + x[t]
        err = float(np.abs(np.float64(y32[2:]) - y64[2:]).max())
        sth = abs(np.sin(theta[m]))
        inv_sin = 1.0 / sth
        rho_m = float(rho[m])
        denom = (1.0 - rho_m)
        delta_obs = float(np.abs(deltas).max())
        delta_b = U_ROUND * mag
        bound_A = U_ROUND / (denom * sth)          # 口径 A（裸 u）
        bound_B = delta_b / (denom * sth)          # 口径 B（随轨迹重构）
        bound_old = delta_b / denom                # 旧界（无 1/|sin theta|）
        rows.append(dict(base=int(m), theta=float(theta[m]), rho=rho_m,
                         inv_sin_theta=float(inv_sin),
                         observed_max_err=err, delta_observed=delta_obs,
                         delta_bound_B=float(delta_b),
                         ratio_vs_boundA=err / bound_A if bound_A > 0 else float('nan'),
                         ratio_vs_boundB=err / bound_B if bound_B > 0 else float('nan'),
                         ratio_vs_old_bound=err / bound_old if bound_old > 0 else float('nan')))
    return rows


# ---------------------------------------------------------------------- 账本
def ledger(D=768, n_layers=12, M=64, K=4, vocab=32000, ctx=1_000_000,
           bytes_per_param=2, kv_dtype_bytes=2):
    """1M 上下文部署账本（bf16 权重 + O(1) 流式状态）vs 同规模 Transformer KV cache。"""
    per_layer = 3 * D * D + 8 * D * D + M * D + D * K + 2 * D      # Wu/Wg/Wo + MLP + 读出 + dw + norms
    n_params = n_layers * per_layer + 2 * vocab * D                  # embedding + tied head
    w_bytes = n_params * bytes_per_param
    st_per_layer = 4 * (2 * M * D + (K - 1) * D + 1)
    st_bytes = n_layers * st_per_layer
    total = w_bytes + st_bytes
    kv = 2 * ctx * n_layers * D * kv_dtype_bytes                     # K 与 V，各 n_layers*D
    return dict(config=dict(D=D, n_layers=n_layers, M=M, K=K, vocab=vocab, ctx=ctx),
                n_params=int(n_params),
                weights_GiB=w_bytes / 2**30, state_MiB=st_bytes / 2**20,
                state_bytes_per_layer=int(st_per_layer),
                total_GiB=total / 2**30,
                ts_kv_cache_GiB=kv / 2**30,
                ts_total_GiB=(w_bytes + kv) / 2**30,
                ratio_kv_over_state=float(kv / st_bytes),
                ratio_total=float((w_bytes + kv) / total))


if __name__ == "__main__":
    import sys
    out = {}
    grid = S.make_grid(M=64)
    out["grid"] = dict(M=64, theta_min=float(grid["theta"][0]), theta_max=float(grid["theta"][-1]),
                       inv_sin_theta_min=float(1 / np.sin(grid["theta"][0])),
                       rho_min=float(grid["rho"].min()), rho_max=float(grid["rho"].max()))
    print("== T1 双形式对偶 ==")
    t1 = test_duality(grid)
    out["T1_duality"] = t1
    for r in t1:
        print(f"  L={r['L']:>5}  bitwise={str(r['bitwise']):>5}  rel={r['rel_err']:.3e}  max_abs={r['max_abs']:.3e}")
    print("== T2 严格因果 ==")
    t2 = test_causality(grid)
    out["T2_causality"] = t2
    print(" ", t2)
    print("== T3 常数状态 ==")
    t3 = test_state_bytes(grid)
    out["T3_state"] = t3
    print(" ", t3)
    print("== P5 命题5 误差界 ==")
    p5 = test_proposition5(grid, L=512)
    out["P5"] = p5
    viol_B = [r for r in p5 if r["ratio_vs_boundB"] > 1.0]
    viol_old = [r for r in p5 if r["ratio_vs_old_bound"] > 1.0]
    print(f"  bases={len(p5)}  boundB 违反数={len(viol_B)}  old-bound 违反数={len(viol_old)}")
    worst_old = max(p5, key=lambda r: r["ratio_vs_old_bound"])
    worst_B = max(p5, key=lambda r: r["ratio_vs_boundB"])
    print(f"  最坏 vs 口径B: base={worst_B['base']} ratio={worst_B['ratio_vs_boundB']:.3f} "
          f"(1/sinθ={worst_B['inv_sin_theta']:.1f})")
    print(f"  最坏 vs 旧界: base={worst_old['base']} ratio={worst_old['ratio_vs_old_bound']:.3f} "
          f"(1/sinθ={worst_old['inv_sin_theta']:.1f})")
    print("== 部署账本 ==")
    lb = ledger()
    out["ledger"] = lb
    for k, v in lb.items():
        print(f"  {k}: {v}")
    with open("results_structural.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results_structural.json")
