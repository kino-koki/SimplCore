// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 后端算子契约 v1 的 **参考实现**（纯 C99，无任何依赖，fp32 单一契约）。
//
// 定位（诚实边界）
// ----------------
//   * 目的：让 ``backend_bridge.load_backend`` 路径首次跑通一个**真实编译的
//     共享库**（而非 FakeCLib 替身），使 conformance_runner 的 9 项检查第一次
//     锻炼真实的 dlopen / 符号校验 / ctypes 编组链路。
//   * 它**不是** MI250 生产后端：fp32 CPU、spectral_causal_conv 用直接时域
//     因果卷积（契约允许"或等效直接卷积"，O(T²·D)），仅供 CI profile
//     （L≈4096）规模的一致性验收；full profile（L=4M）必须用 FFT 后端。
//   * 数学逐式对齐 ``conformance_runner.TorchReferenceAdapter`` 与
//     ``gates.py``——那是数值规格 oracle。
//
// ABI v1 调用约定（见 backend_bridge.ABI_SPEC）
// ---------------------------------------------
//   * 每个张量 = (void* ptr, const int64_t* shape, int32_t rank)，行主序连续；
//   * 输出缓冲由调用方分配，内核只写不分配；
//   * 可空张量（buf / coef / energy_sum）以 NULL 指针 + rank 0 表示；
//   * 标量 double 按值传；iargs 为 int64_t；extra_out 为指针出参；
//   * 全部返回 int32 错误码（0 = 成功）。
//   * fp32 单一契约：非 fp32 输入由桥接层（CtypesBackendAdapter）转换，
//     本实现永远收到 fp32——与 ops/mixer_stream.py 的既有哲学一致。
//
// 数值纪律
// --------
//   * 编译必须带 -ffp-contract=off：禁止 FMA 收缩，保证分块 vs 整段
//     chunked_iir_scan 逐位一致（P0-chunked-carry）。
//   * 求和一律顺序累加（与参考实现的 cumsum / 递推同序）。
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#define SC_EXPORT __declspec(dllexport)
#else
#define SC_EXPORT __attribute__((visibility("default")))
#endif

static float sc_sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static int64_t sc_numel(const int64_t* s, int32_t r) {
    int64_t n = 1;
    for (int32_t i = 0; i < r; ++i) n *= s[i];
    return n;
}

SC_EXPORT int32_t sc_abi_version(void) { return 1; }

// ---------------------------------------------------------------------------
// emb_lookup: weight(V,D) fp32, ids(...) int64 -> out(ids.shape + (D,)) fp32
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_emb_lookup(
        const void* w_, const int64_t* wsh, int32_t wr,
        const void* ids_, const int64_t* ish, int32_t ir,
        void* out_, const int64_t* osh, int32_t orank) {
    (void)wr; (void)osh; (void)orank;
    const float* w = (const float*)w_;
    const int64_t* ids = (const int64_t*)ids_;
    float* out = (float*)out_;
    const int64_t D = wsh[1];
    const int64_t n = sc_numel(ish, ir);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t id = ids[i];
        const float* row = w + id * D;
        float* o = out + i * D;
        for (int64_t d = 0; d < D; ++d) o[d] = row[d];
    }
    return 0;
}

// ---------------------------------------------------------------------------
// rmsnorm: x(...,D) fp32, weight(D) -> out 同形
//   y = x * rsqrt(mean(x^2) + eps); out = y * w   （统计 fp32，与参考同序）
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_rmsnorm(
        const void* x_, const int64_t* xsh, int32_t xr,
        const void* w_, const int64_t* wsh, int32_t wr_,
        void* out_, const int64_t* osh_, int32_t orank_,
        double eps) {
    (void)wsh; (void)wr_; (void)osh_; (void)orank_;
    const float* x = (const float*)x_;
    const float* w = (const float*)w_;
    float* out = (float*)out_;
    const int64_t D = xsh[xr - 1];
    const int64_t rows = sc_numel(xsh, xr) / D;
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr_ = x + r * D;
        float* orow = out + r * D;
        float ss = 0.0f;
        for (int64_t d = 0; d < D; ++d) ss += xr_[d] * xr_[d];
        const float inv = 1.0f / sqrtf(ss / (float)D + (float)eps);
        for (int64_t d = 0; d < D; ++d) orow[d] = xr_[d] * inv * w[d];
    }
    return 0;
}

// ---------------------------------------------------------------------------
// gemm: a(...,K) fp32, b(K,N) fp32 -> out(...,N)；fp32 顺序累加
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_gemm(
        const void* a_, const int64_t* ash, int32_t ar,
        const void* b_, const int64_t* bsh, int32_t br,
        void* out_, const int64_t* osh, int32_t orank_) {
    (void)ar; (void)br; (void)osh; (void)orank_;
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* out = (float*)out_;
    const int64_t K = ash[ar - 1];
    const int64_t N = bsh[1];
    const int64_t M = sc_numel(ash, ar) / K;
    for (int64_t m = 0; m < M; ++m) {
        const float* arow = a + m * K;
        float* orow = out + m * N;
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += arow[k] * b[k * N + n];
            orow[n] = acc;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// swiglu: out = silu(gate) * up，逐元素
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_swiglu(
        const void* g_, const int64_t* gsh, int32_t gr,
        const void* u_, const int64_t* ush, int32_t ur,
        void* out_, const int64_t* osh, int32_t orank_) {
    (void)gsh; (void)gr; (void)ush; (void)ur; (void)osh; (void)orank_;
    const float* g = (const float*)g_;
    const float* u = (const float*)u_;
    float* out = (float*)out_;
    const int64_t n = sc_numel(gsh, gr);
    for (int64_t i = 0; i < n; ++i) out[i] = g[i] * sc_sigm(g[i]) * u[i];
    return 0;
}

// ---------------------------------------------------------------------------
// residual_add: out = a + b
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_residual_add(
        const void* a_, const int64_t* ash, int32_t ar,
        const void* b_, const int64_t* bsh, int32_t br,
        void* out_, const int64_t* osh, int32_t orank_) {
    (void)ash; (void)ar; (void)bsh; (void)br; (void)osh; (void)orank_;
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* out = (float*)out_;
    const int64_t n = sc_numel(ash, ar);
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    return 0;
}

// ---------------------------------------------------------------------------
// spectral_causal_conv（参考实现 = 直接时域因果卷积，契约允许的等效形式）：
//   u(B,T,D), kernel(T,D) -> out(B,T,D)
//   out[b,t,d] = sum_{s<=t} kernel[t-s,d] * u[b,s,d]
// 复杂度 O(B·T²·D)：仅适用于 CI profile（L≈4096）；full profile 须 FFT 后端。
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_spectral_causal_conv(
        const void* u_, const int64_t* ush, int32_t ur,
        const void* k_, const int64_t* ksh, int32_t kr,
        void* out_, const int64_t* osh, int32_t orank_) {
    (void)ksh; (void)kr; (void)osh; (void)orank_;
    const float* u = (const float*)u_;
    const float* k = (const float*)k_;
    float* out = (float*)out_;
    const int64_t T = ush[1], D = ush[2];
    const int64_t B = sc_numel(ush, ur) / (T * D);
    memset(out, 0, (size_t)(sc_numel(ush, ur)) * sizeof(float));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float* orow = out + (b * T + t) * D;
            for (int64_t s = 0; s <= t; ++s) {
                const float* urow = u + (b * T + s) * D;
                const float* krow = k + (t - s) * D;
                for (int64_t d = 0; d < D; ++d) orow[d] += krow[d] * urow[d];
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// causal_depthwise_conv: u(B,T,D), weight(D,1,K), buf(B,P,D) 可空
//   语义对齐 sgc.depthwise_causal_conv：在 cat(buf,u) 上左填充 K-1 做逐通道
//   因果卷积，只返回新 T 个位置的输出；new_buf = 末尾 min(P+T, K-1) 行。
//   *out_new_buf_len = 实际行数。
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_causal_depthwise_conv(
        const void* u_, const int64_t* ush, int32_t ur,
        const void* w_, const int64_t* wsh, int32_t wr_,
        const void* buf_, const int64_t* bsh, int32_t br_,
        void* out_, const int64_t* osh_, int32_t orank__,
        void* nb_, const int64_t* nbsh, int32_t nbr_,
        int64_t* out_new_buf_len) {
    (void)wr_; (void)br_; (void)osh_; (void)orank__; (void)nbsh; (void)nbr_;
    const float* u = (const float*)u_;
    const float* w = (const float*)w_;
    const float* buf = (const float*)buf_;
    float* out = (float*)out_;
    float* nb = (float*)nb_;
    const int64_t T = ush[1], D = ush[2];
    const int64_t B = sc_numel(ush, ur) / (T * D);
    const int64_t K = wsh[2];
    const int64_t P = (br_ > 0) ? bsh[1] : 0;      // buf 行数（NULL → 0）
    memset(out, 0, (size_t)(sc_numel(ush, ur)) * sizeof(float));  // 累加前清零（缓冲可能未初始化）
    const int64_t TT = P + T;
    for (int64_t b = 0; b < B; ++b) {
        const float* ub = u + b * T * D;
        const float* bb = buf ? buf + b * P * D : NULL;
        for (int64_t t = 0; t < T; ++t) {
            float* orow = out + (b * T + t) * D;
            for (int64_t k = 0; k < K; ++k) {
                // torch 语义：y_full[i] = Σ_k w[k]·uu[i+k-(K-1)]，取末 T 个位置
                // ⇒ i = P+t ⇒ uu 下标 j = P+t+k-(K-1)（<0 为左填充，≥P 落在 u）
                const int64_t j = P + t + k - (K - 1);
                if (j < 0) continue;
                const float* src = (j < P) ? (bb ? bb + j * D : NULL)
                                           : ub + (j - P) * D;
                if (!src) continue;
                const float* wk = w + k;             // w 布局 (D,1,K): w[d*K+k]
                for (int64_t d = 0; d < D; ++d) orow[d] += wk[d * K] * src[d];
            }
        }
        // new_buf：uu 的末尾 min(TT, K-1) 行
        const int64_t plen = (K > 1) ? (TT < K - 1 ? TT : K - 1) : 0;
        if (plen > 0) {
            float* nbrow = nb + b * (K - 1) * D;
            for (int64_t i = 0; i < plen; ++i) {
                const int64_t j = TT - plen + i;
                const float* src = (j < P) ? (bb ? bb + j * D : NULL)
                                           : ub + (j - P) * D;
                if (src) memcpy(nbrow + i * D, src, (size_t)D * sizeof(float));
                else     memset(nbrow + i * D, 0, (size_t)D * sizeof(float));
            }
        }
    }
    *out_new_buf_len = (K > 1) ? (TT < K - 1 ? TT : K - 1) : 0;
    return 0;
}

// ---------------------------------------------------------------------------
// chunked_iir_scan: u(B,C,D), y1/y2(B,M,D), a/b/c(M,)
//   逐 token 二阶递推（与 SpectralGlobalConv.step 同一数学，顺序 fp32）：
//     y_new = a·y1 + b·y2 + c·u_t;  (y2, y1) <- (y1_old, y_new)
//   use_coef=0: out(B,C,M,D) = 振荡器态；use_coef=1: out(B,C,D) = Σ_m y·coef。
//   输出 out_y1/out_y2 为终态；输入 y1/y2 只读。
//   同序顺序累加 ⇒ 分块 vs 整段逐位一致（P0-chunked-carry）。
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_chunked_iir_scan(
        const void* u_, const int64_t* ush, int32_t ur,
        const void* y1_, const int64_t* y1sh, int32_t y1r,
        const void* y2_, const int64_t* y2sh, int32_t y2r,
        const void* a_, const int64_t* ash_, int32_t ar_,
        const void* b_, const int64_t* bsh_, int32_t br_,
        const void* c_, const int64_t* csh_, int32_t cr_,
        const void* coef_, const int64_t* csh2, int32_t cr2,
        void* out_, const int64_t* osh, int32_t orank_,
        void* oy1_, const int64_t* o1sh, int32_t o1r_,
        void* oy2_, const int64_t* o2sh, int32_t o2r_,
        int64_t use_coef) {
    (void)ur; (void)y2sh; (void)y1r; (void)y2r; (void)ash_; (void)ar_; (void)bsh_; (void)br_;
    (void)csh_; (void)cr_; (void)csh2; (void)cr2; (void)osh; (void)orank_;
    (void)o1sh; (void)o1r_; (void)o2sh; (void)o2r_;
    const float* u = (const float*)u_;
    const float* y1in = (const float*)y1_;
    const float* y2in = (const float*)y2_;
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    const float* c = (const float*)c_;
    const float* coef = (const float*)coef_;
    float* out = (float*)out_;
    float* oy1 = (float*)oy1_;
    float* oy2 = (float*)oy2_;
    const int64_t C = ush[1], D = ush[2];
    const int64_t B = sc_numel(ush, ur) / (C * D);
    const int64_t M = y1sh[1];
    const size_t st_bytes = (size_t)(B * M * D) * sizeof(float);
    float* y1c = (float*)malloc(st_bytes);
    float* y2c = (float*)malloc(st_bytes);
    if (!y1c || !y2c) { free(y1c); free(y2c); return 2; }
    memcpy(y1c, y1in, st_bytes);
    memcpy(y2c, y2in, st_bytes);
    for (int64_t t = 0; t < C; ++t) {
        for (int64_t bi = 0; bi < B; ++bi) {
            const float* urow = u + (bi * C + t) * D;
            for (int64_t m = 0; m < M; ++m) {
                const int64_t idx = (bi * M + m) * D;
                for (int64_t d = 0; d < D; ++d) {
                    const float yn = a[m] * y1c[idx + d] + b[m] * y2c[idx + d]
                                   + c[m] * urow[d];
                    y2c[idx + d] = y1c[idx + d];
                    y1c[idx + d] = yn;
                    if (!use_coef) {
                        out[((bi * C + t) * M + m) * D + d] = yn;
                    } else {
                        float* orow = out + (bi * C + t) * D + d;
                        if (m == 0) orow[0] = 0.0f;
                        orow[0] += yn * coef[m * D + d];
                    }
                }
            }
        }
    }
    memcpy(oy1, y1c, st_bytes);
    memcpy(oy2, y2c, st_bytes);
    free(y1c);
    free(y2c);
    return 0;
}

// ---------------------------------------------------------------------------
// content_gate: h(...,Dh), w(K,Dh) -> sigmoid(h·wᵀ) (...,K)
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_content_gate(
        const void* h_, const int64_t* hsh, int32_t hr,
        const void* w_, const int64_t* wsh, int32_t wr_,
        void* out_, const int64_t* osh_, int32_t orank_) {
    (void)wr_; (void)osh_; (void)orank_;
    const float* h = (const float*)h_;
    const float* w = (const float*)w_;
    float* out = (float*)out_;
    const int64_t Dh = hsh[hr - 1];
    const int64_t K = wsh[0];
    const int64_t rows = sc_numel(hsh, hr) / Dh;
    for (int64_t r = 0; r < rows; ++r) {
        const float* hrow = h + r * Dh;
        float* orow = out + r * K;
        for (int64_t k = 0; k < K; ++k) {
            const float* wk = w + k * Dh;
            float acc = 0.0f;
            for (int64_t d = 0; d < Dh; ++d) acc += hrow[d] * wk[d];
            orow[k] = sc_sigm(acc);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// evidence_gate_parallel: u(B,L,D) -> (B,L,1)
//   c_t = (Σ_{s≤t} ||u_s||²) / (t+1)（顺序累加 fp32）
//   w_t = sigmoid(gain·log1p(c_t) + pos_gain·log1p(t+1) + bias)
//   （对齐 gates.evidence_weights：并行 cumsum 口径）
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_evidence_gate_parallel(
        const void* u_, const int64_t* ush, int32_t ur,
        void* out_, const int64_t* osh_, int32_t orank_,
        double gain, double pos_gain, double bias) {
    (void)osh_; (void)orank_;
    const float* u = (const float*)u_;
    float* out = (float*)out_;
    const int64_t L = ush[1], D = ush[2];
    const int64_t B = sc_numel(ush, ur) / (L * D);
    for (int64_t b = 0; b < B; ++b) {
        float es = 0.0f;
        for (int64_t t = 0; t < L; ++t) {
            const float* urow = u + (b * L + t) * D;
            for (int64_t d = 0; d < D; ++d) es += urow[d] * urow[d];
            const float c = es / (float)(t + 1);
            const float logit = (float)gain * log1pf(c)
                              + (float)pos_gain * log1pf((float)(t + 1))
                              + (float)bias;
            out[(b * L + t)] = sc_sigm(logit);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// evidence_gate_chunk: u(B,C,D), energy_sum(B,1) 可空, length=此前 token 数
//   逐步 es += ||u_t||²; c = es/(length+t+1);
//   w = sigmoid(gain·log1p(c) + pos_gain·log1p(length+t+1) + bias)
//   out_es = 终态 es（B,1）。（对齐 gates.evidence_weight_step）
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_evidence_gate_chunk(
        const void* u_, const int64_t* ush, int32_t ur,
        const void* es_, const int64_t* esh_, int32_t esr_,
        void* out_, const int64_t* osh_, int32_t orank__,
        void* oes_, const int64_t* oesh, int32_t oesr_,
        double gain, double pos_gain, double bias,
        int64_t length) {
    (void)ur; (void)esh_; (void)esr_; (void)osh_; (void)orank__; (void)oesh; (void)oesr_;
    const float* u = (const float*)u_;
    const float* esin = (const float*)es_;
    float* out = (float*)out_;
    float* oes = (float*)oes_;
    const int64_t C = ush[1], D = ush[2];
    const int64_t B = sc_numel(ush, ur) / (C * D);
    for (int64_t b = 0; b < B; ++b) {
        float es = esin ? esin[b] : 0.0f;
        for (int64_t t = 0; t < C; ++t) {
            const float* urow = u + (b * C + t) * D;
            for (int64_t d = 0; d < D; ++d) es += urow[d] * urow[d];
            const float cnt = (float)(length + t + 1);
            const float c = es / cnt;
            const float logit = (float)gain * log1pf(c)
                              + (float)pos_gain * log1pf(cnt)
                              + (float)bias;
            out[(b * C + t)] = sc_sigm(logit);
        }
        oes[b] = es;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// uncertainty_gate: long_/local (B,L,D) -> 阻尼 (B,L,1)
//   d = Σ|l-lc| / (Σ|l| + Σ|lc| + eps);  w = 1 - sigmoid(logit)·d
//   （对齐 gates.uncertainty_weights / joint_disagreement）
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_uncertainty_gate(
        const void* lg_, const int64_t* lsh, int32_t lr_,
        const void* lc_, const int64_t* lcsh, int32_t lcr_,
        void* out_, const int64_t* osh_, int32_t orank_,
        double logit, double eps) {
    (void)lr_; (void)lcsh; (void)lcr_; (void)osh_; (void)orank_;
    const float* lg = (const float*)lg_;
    const float* lc = (const float*)lc_;
    float* out = (float*)out_;
    const int64_t D = lsh[2];
    const int64_t BL = sc_numel(lsh, lr_) / D;
    for (int64_t i = 0; i < BL; ++i) {
        const float* lrow = lg + i * D;
        const float* crow = lc + i * D;
        float num = 0.0f, den = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            num += fabsf(lrow[d] - crow[d]);
            den += fabsf(lrow[d]) + fabsf(crow[d]);
        }
        const float dd = num / (den + (float)eps);
        out[i] = 1.0f - sc_sigm((float)logit) * dd;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// ce_loss: logits(...,V) fp32, labels(...) int64（-100 忽略）-> 标量 mean CE
//   逐位置 lse = max + log(Σ exp(x-max))（fp32，max-shift 与 torch 同构）
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_ce_loss(
        const void* lg_, const int64_t* gsh, int32_t gr,
        const void* lb_, const int64_t* lbsh, int32_t lbr,
        void* out_, const int64_t* osh_, int32_t orank_) {
    (void)lbsh; (void)lbr; (void)osh_; (void)orank_;
    const float* lg = (const float*)lg_;
    const int64_t* lb = (const int64_t*)lb_;
    float* out = (float*)out_;
    const int64_t V = gsh[gr - 1];
    const int64_t rows = sc_numel(gsh, gr) / V;
    float sum = 0.0f;
    int64_t cnt = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t y = lb[r];
        if (y == -100) continue;
        const float* xrow = lg + r * V;
        float mx = xrow[0];
        for (int64_t v = 1; v < V; ++v) if (xrow[v] > mx) mx = xrow[v];
        float z = 0.0f;
        for (int64_t v = 0; v < V; ++v) z += expf(xrow[v] - mx);
        const float lse = mx + logf(z);
        sum += lse - xrow[y];
        ++cnt;
    }
    out[0] = (cnt > 0) ? sum / (float)cnt : (0.0f / 0.0f);
    return 0;
}

// ---------------------------------------------------------------------------
// adamw_frozen_mask: params/grads/m/v fp32, frozen(uint8 0/1)
//   冻结位 p/m/v 逐位不动；可训位与参考 AdamW 同式（bias-correction）。
//   bc 在 double 里算完再降 fp32（参考实现：Python 标量 double → fp32 张量）。
// ---------------------------------------------------------------------------
SC_EXPORT int32_t sc_adamw_frozen_mask(
        const void* p_, const int64_t* psh, int32_t pr,
        const void* g_, const int64_t* gsh_, int32_t gr_,
        const void* m_, const int64_t* msh_, int32_t mr_,
        const void* v_, const int64_t* vsh_, int32_t vr_,
        const void* f_, const int64_t* fsh_, int32_t fr_,
        void* pn_, const int64_t* pnsh_, int32_t pnr_,
        void* mn_, const int64_t* mnsh_, int32_t mnr_,
        void* vn_, const int64_t* vnsh_, int32_t vnr_,
        double lr, double beta1, double beta2, double eps, double wd,
        int64_t step) {
    (void)psh; (void)pr; (void)gsh_; (void)gr_; (void)msh_; (void)mr_;
    (void)vsh_; (void)vr_; (void)fsh_; (void)fr_; (void)pnsh_; (void)pnr_;
    (void)mnsh_; (void)mnr_; (void)vnsh_; (void)vnr_;
    const float* p = (const float*)p_;
    const float* g = (const float*)g_;
    const float* m = (const float*)m_;
    const float* v = (const float*)v_;
    const uint8_t* fz = (const uint8_t*)f_;
    float* pn = (float*)pn_;
    float* mn = (float*)mn_;
    float* vn = (float*)vn_;
    const int64_t n = sc_numel(psh, pr);
    const float bc1 = (float)(1.0 - pow(beta1, (double)step));
    const float bc2 = (float)(1.0 - pow(beta2, (double)step));
    for (int64_t i = 0; i < n; ++i) {
        if (fz[i]) {
            pn[i] = p[i];
            mn[i] = m[i];
            vn[i] = v[i];
        } else {
            const float gt = g[i];
            const float mnew = (float)beta1 * m[i] + (1.0f - (float)beta1) * gt;
            const float vnew = (float)beta2 * v[i] + (1.0f - (float)beta2) * gt * gt;
            const float upd = (mnew / bc1) / (sqrtf(vnew / bc2) + (float)eps)
                            + (float)wd * p[i];
            pn[i] = p[i] - (float)lr * upd;
            mn[i] = mnew;
            vn[i] = vnew;
        }
    }
    return 0;
}
