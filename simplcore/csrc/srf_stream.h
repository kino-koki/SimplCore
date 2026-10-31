// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 谱振流（SRF）mixer 的融合流式热路径：把原本在 PyTorch 里拆成十几个
// 小算子（RMSNorm -> in_proj -> 内容门控 -> selective SSM 复数递推 -> 局部卷积
// -> 证据门 -> 不确定门 -> out_proj -> 残差）融合成单遍，O(1) 状态原地更新。
//
// 与 simplcore/core/srf.py 的 SpectralResonanceFlow.step 数学逐位一致（fp32 守护
// 阈值见 simplcore/core/tests/test_srf_stream.py）。本文件零第三方依赖（仅
// <cmath>/<cstring>），可被 torch C++ 扩展 include，也可在无 torch 环境用 g++
// 单独编译核验。device 元素函数带 __host__ __device__ 供 .cu 复用。
//
// 与 mixer_stream.h 的关系：二者融合的是不同 mixer 家族（SGC=time-invariant IIR；
// SRF=input-dependent selective SSM），数学不同，故分文件、分扩展模块，各对应
// 各自的纯 PyTorch step 守护测试。命名均诚实反映"融合的是整个 mixer 的流式 step"。
#ifndef SRF_STREAM_H
#define SRF_STREAM_H

#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define SRF_STREAM_HD __host__ __device__
#else
#define SRF_STREAM_HD
#endif

namespace srf {

/// 数值稳定的 softplus(x) = log(1 + exp(x))（selective SSM 的步长）。
SRF_STREAM_HD inline float srf_softplus(float x) {
    if (x > 20.0f) return x;            // log1p(exp(x)) ≈ x
    if (x < -20.0f) return 0.0f;        // exp(x) → 0
    return log1pf(expf(x));
}

/// SRF 融合算子的全部权重与形状/开关。所有指针均按 fp32、行优先、contiguous 解释。
struct SrfStreamWeights {
    int B = 1;
    int D = 1;
    int M = 1;            ///< SSM 振荡基数量（状态维度）
    int K = 1;            ///< 深度局部卷积宽度（K==1 时无 conv_buf）
    int length = 0;       ///< 本步之前已消费的 token 数（证据门分母 t = length+1）
    bool content_gate = true;
    bool evidence_gate = true;
    bool uncertainty_gate = true;
    bool out_gate = false;
    float eps = 1e-6f;           ///< RMSNorm epsilon
    float gate_eps = 1e-6f;       ///< L3 不确定门分母 epsilon
    float out_scale = 1.0f;       ///< 残差输出缩放
    float ev_gain = 1.0f;         ///< L2 证据项强度
    float ev_pos_gain = 0.0f;     ///< L2 位置计数项强度
    float ev_bias = 0.0f;         ///< L2 偏置
    float unc_logit = 0.0f;       ///< L3 可学标量 logit
    const float* wnorm = nullptr;     ///< [D]   RMSNorm 权重
    const float* in_w = nullptr;      ///< [D*D] in_proj 权重
    const float* gate_w = nullptr;    ///< [D*D] 内容门控权重（content_gate 时非空）
    const float* out_w = nullptr;     ///< [D*D] out_proj 权重
    const float* outgate_w = nullptr; ///< [D*D] 输出门权重（out_gate 时非空）
    const float* b_in = nullptr;      ///< [M*D] 输入耦合 b[m,d]
    const float* coef = nullptr;      ///< [M*D] 输出耦合 coef[m,d]
    const float* skip = nullptr;      ///< [D]   SSM 直通项 D
    const float* dt_w = nullptr;      ///< [D]   dt_proj 权重（Linear(D,1)，行向量）
    float dt_bias = 0.0f;             ///< [1]   dt_proj 偏置
    const float* rho = nullptr;       ///< [M]   衰减率（0,1）
    const float* theta = nullptr;     ///< [M]   角频率（0,pi）
    const float* local_w = nullptr;   ///< [D*K] 局部卷积权重
};

/// 单个 (b,d) 的 selective SSM 复数递推 + 局部卷积 + 状态原地更新。
///
/// 计算 long = sum_m coef[m,d]*new_re[m,d] + skip[d]*u[d]，其中
///   lambda_m = rho_m * exp(i*theta_m)；delta = softplus(dt_w·u + dt_bias)；
///   amp_m = exp(delta*log(rho_m))；phase_m = delta*theta_m；
///   ar_m = amp_m*cos(phase_m)；ai_m = amp_m*sin(phase_m)；
///   new_re[m,d] = ar_m*h_re[m,d] - ai_m*h_im[m,d] + u[d]*b_in[m,d]；
///   new_im[m,d] = ai_m*h_re[m,d] + ar_m*h_im[m,d]；
/// 同时把复数状态原地推进：h_re[m,d] <- new_re，h_im[m,d] <- new_im。
///
/// 每个 (b,m,d) 只读写自己的状态元素，不同 d / 不同 m 之间不互相干扰，因此
/// 多线程/多 token 并行安全（同一 (b,m,d) 由单一线程完整读写）。
SRF_STREAM_HD inline void srf_recurrence_and_local(
    int b, int d, int B, int D, int M, int K,
    const float* u,          ///< [B*D] 门控后的输入
    const float* b_in,       ///< [M*D]
    const float* coef,       ///< [M*D]
    const float* skip,       ///< [D]
    const float* rho,        ///< [M]
    const float* theta,      ///< [M]
    const float* local_w,    ///< [D*K]
    float delta,             ///< 本 token 的标量步长（per-b 广播）
    float* h_re, float* h_im,/// [B*M*D] 原地读写（复数状态原地推进）
    const float* conv_buf,   ///< [B*(K-1)*D] 可 nullptr（K==1）
    float* long_out,
    float* loc_out) {
    const float* ub = u + (size_t)b * D;
    const size_t base = (size_t)b * M * D;
    float long_acc = skip[d] * ub[d];    // SSM 直通项 D
    for (int m = 0; m < M; ++m) {
        float r = rho[m];
        float th = theta[m];
        float log_rho = logf(r);
        if (log_rho > -1e-6f) log_rho = -1e-6f;   // clamp_max(-1e-6)
        float amp = expf(delta * log_rho);
        float ph = delta * th;
        float ar = amp * cosf(ph);
        float ai = amp * sinf(ph);
        size_t off = (size_t)m * D + d;
        float hre = h_re[base + off];
        float him = h_im[base + off];
        float uvd = ub[d];
        float nre = ar * hre - ai * him + uvd * b_in[off];
        float nim = ai * hre + ar * him;
        h_re[base + off] = nre;
        h_im[base + off] = nim;
        long_acc += coef[off] * nre;
    }
    *long_out = long_acc;

    float lc = 0.0f;
    if (K == 1) {
        lc = local_w[d] * ub[d];
    } else {
        const float* cb = conv_buf + (size_t)b * (K - 1) * D;
        for (int k = 0; k < K; ++k) {
            const float* src = (k < K - 1) ? (cb + (size_t)k * D) : ub;
            lc += local_w[(size_t)d * K + k] * src[d];
        }
    }
    *loc_out = lc;
}

/// CPU 参考实现：整批一步 SRF mixer（与 SpectralResonanceFlow.step 逐位一致）。
inline void srf_stream_step_host(
    const float* x, float* y_t,
    float* h_re, float* h_im, float* conv_buf, float* energy,
    float* workspace,
    const SrfStreamWeights& w) {
    const int B = w.B, D = w.D, M = w.M, K = w.K;
    const size_t BD = (size_t)B * D;
    float* h = workspace;
    float* u = workspace + BD;
    float* long_arr = workspace + 2 * BD;
    float* loc_arr = workspace + 3 * BD;

    for (int b = 0; b < B; ++b) {
        const float* xb = x + b * D;
        float* hb = h + b * D;
        float* ub = u + b * D;

        // 1) RMSNorm
        float ss = 0.0f;
        for (int d = 0; d < D; ++d) ss += xb[d] * xb[d];
        float rn = 1.0f / sqrtf(ss / D + w.eps);
        for (int d = 0; d < D; ++d) hb[d] = xb[d] * rn * w.wnorm[d];

        // 2) in_proj
        for (int dd = 0; dd < D; ++dd) {
            float s = 0.0f;
            const float* iw = w.in_w + (size_t)dd * D;
            for (int j = 0; j < D; ++j) s += iw[j] * hb[j];
            ub[dd] = s;
        }
        // 3) 内容门控（selective）
        if (w.content_gate) {
            for (int dd = 0; dd < D; ++dd) {
                float s = 0.0f;
                const float* gw = w.gate_w + (size_t)dd * D;
                for (int j = 0; j < D; ++j) s += gw[j] * hb[j];
                ub[dd] *= 1.0f / (1.0f + expf(-s));
            }
        }

        // 4) selective 步长 delta = softplus(dt_w·u + dt_bias)（per-token 标量）
        float dt = w.dt_bias;
        for (int j = 0; j < D; ++j) dt += w.dt_w[j] * ub[j];
        float delta = srf_softplus(dt);

        // 5) SSM 复数递推 + 局部卷积（原地推进 h_re/h_im）
        for (int d = 0; d < D; ++d) {
            float lo, lc;
            srf_recurrence_and_local(b, d, B, D, M, K, u, w.b_in, w.coef, w.skip,
                                     w.rho, w.theta, w.local_w, delta,
                                     h_re, h_im, conv_buf, &lo, &lc);
            long_arr[b * D + d] = lo;
            loc_arr[b * D + d] = lc;
        }

        // 6) L2 证据门：c_t = (1/t) * sum_{s<=t} ||u_s||^2，t = length+1
        float en = 0.0f;
        for (int d = 0; d < D; ++d) en += ub[d] * ub[d];
        float newen = energy[b] + en;
        float t = static_cast<float>(w.length + 1);
        float c = newen / t;
        float we = w.evidence_gate
            ? (1.0f / (1.0f + expf(-(w.ev_gain * log1pf(c)
                                 + w.ev_pos_gain * log1pf(t) + w.ev_bias))))
            : 1.0f;
        for (int d = 0; d < D; ++d) long_arr[b * D + d] *= we;

        // 7) L3 不确定门：d = ||long-loc||_1 / (||long||_1 + ||loc||_1 + eps)
        float l1 = 0.0f, l2 = 0.0f, num = 0.0f;
        for (int d = 0; d < D; ++d) {
            float l = long_arr[b * D + d];
            float lc = loc_arr[b * D + d];
            l1 += fabsf(l);
            l2 += fabsf(lc);
            num += fabsf(l - lc);
        }
        float dd = (l1 + l2 > 0.0f) ? (num / (l1 + l2 + w.gate_eps)) : 0.0f;
        float wu = w.uncertainty_gate
            ? (1.0f - (1.0f / (1.0f + expf(-w.unc_logit))) * dd)
            : 1.0f;
        for (int d = 0; d < D; ++d) long_arr[b * D + d] *= wu;

        // 8) 合成 y、out_proj、输出门、残差
        for (int d = 0; d < D; ++d) {
            float yv = long_arr[b * D + d] + loc_arr[b * D + d];
            float o = 0.0f;
            const float* ow = w.out_w + (size_t)d * D;
            for (int j = 0; j < D; ++j) o += ow[j] * yv;
            if (w.out_gate) {
                float s = 0.0f;
                const float* og = w.outgate_w + (size_t)d * D;
                for (int j = 0; j < D; ++j) s += og[j] * hb[j];
                o *= 1.0f / (1.0f + expf(-s));
            }
            y_t[b * D + d] = xb[d] + w.out_scale * o;
        }

        // 9) 更新 conv_buf：新缓冲 = [旧缓冲[1..K-2], u]
        if (K > 1) {
            float* bufb = conv_buf + (size_t)b * (K - 1) * D;
            if (K > 2) {
                std::memmove(bufb, bufb + D,
                             (size_t)(K - 2) * D * sizeof(float));
            }
            for (int d = 0; d < D; ++d) bufb[(size_t)(K - 2) * D + d] = ub[d];
        }
        // 10) 更新能量运行和
        energy[b] = newen;
    }
}

}  // namespace srf

#endif  // SRF_STREAM_H
