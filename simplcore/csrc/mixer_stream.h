// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 流式热路径：融合的逐 token mixer 算子（CPU 参考实现 + 可复用的
// device 元素函数）。
//
// 这是 SGC / SRF 之外的**推理热路径**：把原本在 Python/PyTorch 里拆成十几个
// 小算子（RMSNorm → in_proj → 内容门控 → IIR 振荡递推 → 局部卷积 → L2 证据门
// → L3 不确定门 → out_proj → 残差）**融合成单遍**计算，O(1) 状态原地更新。
//
// 设计约束（与 simplcore/core/sgc.py、gates.py 严格一致）：
//   * 无位置嵌入表、无 KV cache；位置由因果卷积核本身编码。
//   * 流式 == 整段：本文件产出的 y_t 必须与 simplcore.core.sgc.SpectralGlobalConv
//     的纯 PyTorch step() 在 fp32 下逐位近似（仓库守护阈值 4e-5）。
//   * 严格因果：evidence 统计量只依赖 <= t 的前缀能量。
//
// 本文件**零第三方依赖**（只用到 <cmath>/<cstring>），因此：
//   * CPU 路径可被 torch C++ 扩展直接 include；
//   * 也可在无 torch 环境下用 g++ 单独编译核验（见 test_mixer_stream.cpp）；
//   * device 元素函数（resonator_conv_and_local）带 __host__ __device__，供 .cu 复用。
#ifndef MIXER_STREAM_H
#define MIXER_STREAM_H

#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define MIXER_STREAM_HD __host__ __device__
#else
#define MIXER_STREAM_HD
#endif

namespace mixer_stream {

static MIXER_STREAM_HD inline float mixer_stream_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/// 融合算子的全部权重与形状/开关。所有指针均按 fp32、行优先、contiguous 解释。
struct MixerStreamWeights {
    int B = 1;        ///< batch
    int D = 1;        ///< 模型宽度
    int M = 1;        ///< 阻尼振荡基数量
    int K = 1;        ///< 深度局部卷积宽度（K==1 时无 conv_buf）
    int length = 0;   ///< 本步之前已消费的 token 数（证据门分母 t = length+1）
    bool content_gate = true;       ///< 输入相关门控
    bool evidence_gate = true;      ///< L2 证据门
    bool uncertainty_gate = true;   ///< L3 不确定门
    bool out_gate = false;          ///< 输出门
    float eps = 1e-6f;             ///< RMSNorm epsilon
    float gate_eps = 1e-6f;         ///< L3 不确定门分母 epsilon
    float out_scale = 1.0f;         ///< 残差输出缩放（GPT-2 式）
    float ev_gain = 1.0f;           ///< L2：证据项强度
    float ev_pos_gain = 0.0f;       ///< L2：位置计数项强度
    float ev_bias = 0.0f;           ///< L2：偏置
    float unc_logit = 0.0f;         ///< L3：可学标量 logit
    const float* wnorm = nullptr;    ///< [D]   RMSNorm 权重
    const float* in_w = nullptr;     ///< [D*D] in_proj 权重
    const float* gate_w = nullptr;   ///< [D*D] 内容门控权重（content_gate 时非空）
    const float* out_w = nullptr;    ///< [D*D] out_proj 权重
    const float* outgate_w = nullptr;///< [D*D] 输出门权重（out_gate 时非空）
    const float* coef = nullptr;     ///< [M*D] 振荡基混合系数
    const float* rho = nullptr;      ///< [M]   衰减率（0,1）
    const float* theta = nullptr;    ///< [M]   角频率（0,pi）
    const float* local_w = nullptr;  ///< [D*K] 局部卷积权重
    const float* feed = nullptr;     ///< [D]   逐通道瞬时项（init 1）
};

/// 单个 (b,d) 的振荡基递推 + 局部卷积 + 状态原地更新。
///
/// 计算 long = sum_m coef[m,d] * (a_m*y1[m,d] + b_m*y2[m,d] + c_m*u[d])，
/// 其中 a_m=2*rho_m*cos(theta_m), b_m=-rho_m^2, c_m=sin(theta_m)；
/// 同时把 IIR 状态原地推进：y2 <- y1(旧), y1 <- y_new。
/// 局部卷积 loc 用 [conv_buf(最旧..最新), u] 为窗口（左填充因果）。
///
/// 注意：每个 (m,d) 只读写自己的状态元素，不同 d 之间不互相干扰，因此
/// 多线程/多 token 并行安全（同一 m 的不同 d 写入不同地址）。
MIXER_STREAM_HD inline void resonator_conv_and_local(
    int b, int d, int B, int D, int M, int K,
    const float* u,          ///< [B*D] 门控后的输入（gated input）
    const float* coef,       ///< [M*D]
    const float* rho,        ///< [M]
    const float* theta,      ///< [M]
    const float* local_w,    ///< [D*K]
    float* y1, float* y2,    ///< [B*M*D] 原地读写（y1 也原地推进）
    const float* conv_buf,   ///< [B*(K-1)*D] 可以为 nullptr（K==1 时）
    float* long_out,
    float* loc_out) {
    float* y1b = y1 + (size_t)b * M * D;
    float* y2b = y2 + (size_t)b * M * D;
    const float* ub = u + (size_t)b * D;
    const float* cb = (K > 1) ? (conv_buf + (size_t)b * (K - 1) * D) : nullptr;

    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        float r = rho[m];
        float th = theta[m];
        float a = 2.0f * r * std::cos(th);
        float bb = -r * r;
        float c = std::sin(th);
        size_t off = (size_t)m * D + d;
        float y1old = y1b[off];
        float yn = a * y1old + bb * y2b[off] + c * ub[d];
        y2b[off] = y1old;   ///< 二阶 IIR：y2 接管上一拍的 y1
        y1b[off] = yn;       ///< y1 接管新值
        acc += coef[off] * yn;
    }
    *long_out = acc;

    float lc = 0.0f;
    if (K == 1) {
        lc = local_w[d] * ub[d];
    } else {
        for (int k = 0; k < K; ++k) {
            const float* src = (k < K - 1) ? (cb + (size_t)k * D) : ub;
            lc += local_w[(size_t)d * K + k] * src[d];
        }
    }
    *loc_out = lc;
}

/// CPU 参考实现：整批一步流式 mixer。
///
/// 输入：
///   x[b*D + d]             当前 token 的输入（残差流输入，未含记忆注入）
///   y1/y2[b*M*D + m*D+d]   IIR 二阶状态（原地更新）
///   conv_buf[b*(K-1)*D + ...] 局部卷积历史缓冲（K>1 时；原地更新）
///   energy[b]              因果前缀能量运行和（原地更新）
///   workspace[B*D*4]        临时：h / u / long / loc 各 B*D
/// 输出：
///   y_t[b*D + d] = x + out_scale * out   （经 RMSNorm→proj→门→振荡→局部→门→out）
///
/// 与 simplcore.core.sgc.SpectralGlobalConv.step 数学逐位一致（fp32 阈值 4e-5）。
inline void mixer_stream_step_host(
    const float* x, float* y_t,
    float* y1, float* y2, float* conv_buf, float* energy,
    float* workspace,
    const MixerStreamWeights& w) {
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

        // 1) RMSNorm（无均值减、无偏置）
        float ss = 0.0f;
        for (int d = 0; d < D; ++d) ss += xb[d] * xb[d];
        float rn = 1.0f / std::sqrt(ss / D + w.eps);
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
                ub[dd] *= mixer_stream_sigmoid(s);
            }
        }

        // 4) L2 证据门：c_t = (1/t) * sum_{s<=t} ||u_s||^2，t = length+1
        float en = 0.0f;
        for (int d = 0; d < D; ++d) en += ub[d] * ub[d];
        float newen = energy[b] + en;
        float t = static_cast<float>(w.length + 1);
        float c = newen / t;
        float we = w.evidence_gate
            ? mixer_stream_sigmoid(w.ev_gain * std::log1p(c)
                          + w.ev_pos_gain * std::log1p(t)
                          + w.ev_bias)
            : 1.0f;

        // 5) 振荡基 + 局部卷积（同时原地推进 y1/y2）
        for (int d = 0; d < D; ++d) {
            float lo, lc;
            resonator_conv_and_local(b, d, B, D, M, K, u, w.coef, w.rho, w.theta,
                             w.local_w, y1, y2, conv_buf, &lo, &lc);
            long_arr[b * D + d] = lo;
            loc_arr[b * D + d] = lc;
        }

        // 6) L3 不确定门：d = ||long-loc||_1 / (||long||_1 + ||loc||_1 + eps)
        float l1 = 0.0f, l2 = 0.0f, num = 0.0f;
        for (int d = 0; d < D; ++d) {
            float l = long_arr[b * D + d];
            float lc = loc_arr[b * D + d];
            l1 += std::fabs(l);
            l2 += std::fabs(lc);
            num += std::fabs(l - lc);
        }
        float dd = (l1 + l2 > 0.0f) ? (num / (l1 + l2 + w.gate_eps)) : 0.0f;
        float wu = w.uncertainty_gate
            ? (1.0f - mixer_stream_sigmoid(w.unc_logit) * dd)
            : 1.0f;

        // 7) 合成 y、out_proj、输出门、残差
        for (int d = 0; d < D; ++d) {
            float yv = long_arr[b * D + d] * we * wu
                     + loc_arr[b * D + d]
                     + w.feed[d] * ub[d];
            float o = 0.0f;
            const float* ow = w.out_w + (size_t)d * D;
            for (int j = 0; j < D; ++j) o += ow[j] * yv;
            if (w.out_gate) {
                float s = 0.0f;
                const float* og = w.outgate_w + (size_t)d * D;
                for (int j = 0; j < D; ++j) s += og[j] * hb[j];
                o *= mixer_stream_sigmoid(s);
            }
            y_t[b * D + d] = xb[d] + w.out_scale * o;
        }

        // 8) 更新 conv_buf：新缓冲 = [旧缓冲[1..K-2], u]（保留最近 K-1 个）
        if (K > 1) {
            float* bufb = conv_buf + (size_t)b * (K - 1) * D;
            if (K > 2) {
                std::memmove(bufb, bufb + D,
                             (size_t)(K - 2) * D * sizeof(float));
            }
            for (int d = 0; d < D; ++d) bufb[(size_t)(K - 2) * D + d] = ub[d];
        }
        // 9) 更新能量运行和
        energy[b] = newen;
    }
}

}  // namespace mixer_stream

#endif  // MIXER_STREAM_H
