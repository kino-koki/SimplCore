// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 谱振流（SRF）mixer 融合热路径的 CUDA kernel。
//
// 设计取舍（与 mixer_stream_cuda.cu 一致，并沿用仓库"诚实纪律"）：
//   * 本文件需 nvcc 编译；在当前开发机（无 CUDA 工具链）无法实测，必须在装有
//     torch + CUDA/ROCm + nvcc/hipcc 的目标机上由 test_srf_stream.py 验证 fp32
//     等价性（与 simplcore.core.srf.SpectralResonanceFlow.step 的纯 PyTorch 实现）。
//   * 正确性优先：所有 per-b 归约（RMSNorm 平方和、dt_w·u 点积、证据能量和、
//     L3 长/短分歧的 l1/l2/num）改用 warp-shuffle + 共享内存的并行归约。
//     并行归约改变求和顺序，与串行 CPU 参考有 ≤ 1e-6 量级相对差，远在守护阈值内。
//   * 复数 SSM 递推在 phase 2 逐 d 并行：每个 (b,m,d) 由单一线程完整读写，状态
//     地址互不重叠，无竞争；conv_buf 移位亦逐 d 并行。
#include <torch/extension.h>
#include "srf_stream.h"

constexpr int SRF_STREAM_BLOCK = 256;   // 必须等于 8 个 warp
static_assert(SRF_STREAM_BLOCK == 256, "srf kernel block must be 256 (=8 warps)");

__device__ float block_reduce(float partial, float* warp_buf) {
    float sum = partial;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;            // 0..7
    if (lane == 0) warp_buf[wid] = sum;
    __syncthreads();
    float total = 0.0f;
    if (wid == 0) {
        total = (lane < 8) ? warp_buf[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
        if (lane == 0) warp_buf[0] = total;
    }
    __syncthreads();
    return warp_buf[0];
}

__global__ void srf_kernel(
    const float* x, float* y_t, float* h_re, float* h_im, const float* conv_buf,
    float* energy,
    const float* wnorm, const float* in_w, const float* gate_w, const float* out_w,
    const float* outgate_w, const float* b_in, const float* coef, const float* skip,
    const float* dt_w, float dt_bias, const float* rho, const float* theta,
    const float* local_w,
    int B, int D, int M, int K, int length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    float out_scale, float eps, float gate_eps, float ev_gain, float ev_pos_gain,
    float ev_bias, float unc_logit,
    float* u_g, float* h_g, float* long_g, float* loc_g) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float warp_buf[8];

    const float* xb = x + (size_t)b * D;
    float* hre_b = h_re + (size_t)b * M * D;
    float* him_b = h_im + (size_t)b * M * D;
    const float* cb = (K > 1) ? (conv_buf + (size_t)b * (K - 1) * D) : nullptr;
    float* ub = u_g + (size_t)b * D;
    float* hb = h_g + (size_t)b * D;
    float* longb = long_g + (size_t)b * D;
    float* locb = loc_g + (size_t)b * D;

    // phase 0: RMSNorm 归一化系数（并行归约平方和）
    float ss_part = 0.0f;
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) ss_part += xb[d] * xb[d];
    float ss = block_reduce(ss_part, warp_buf);
    float rn = 1.0f / sqrtf(ss / D + eps);

    // phase 1a: 先把 h 全部写出（跨线程同步后再读，消除竞争）
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK)
        hb[d] = xb[d] * rn * wnorm[d];
    __syncthreads();

    // phase 1b: in_proj + 内容门控（读已写满的 hb）
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) {
        float s = 0.0f;
        for (int j = 0; j < D; ++j) s += in_w[(size_t)d * D + j] * hb[j];
        float uv = s;
        if (content_gate) {
            float g = 0.0f;
            for (int j = 0; j < D; ++j) g += gate_w[(size_t)d * D + j] * hb[j];
            uv /= (1.0f + expf(-g));
        }
        ub[d] = uv;
    }
    __syncthreads();

    // phase 1c: delta = softplus(dt_w·u + dt_bias)（per-token 标量，块内归约点积）
    float dt_part = 0.0f;
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) dt_part += dt_w[d] * ub[d];
    float dt_sum = block_reduce(dt_part, warp_buf);
    float delta = srf::srf_softplus(dt_sum + dt_bias);

    // phase 2: selective SSM 复数递推 + 局部卷积（逐 d 并行，状态按 (b,m,d) 互不干扰）
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) {
        float lo, lc;
        srf::srf_recurrence_and_local(b, d, B, D, M, K, ub, b_in, coef, skip,
                                      rho, theta, local_w, delta, hre_b, him_b,
                                      cb, &lo, &lc);
        longb[d] = lo;
        locb[d] = lc;
    }
    __syncthreads();

    // phase 3: 并行归约 L2 证据能量和 / L3 长-短分歧
    float en_part = 0.0f, l1_part = 0.0f, l2_part = 0.0f, num_part = 0.0f;
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) {
        float uu = ub[d];
        float l = longb[d];
        float lc = locb[d];
        en_part += uu * uu;
        l1_part += fabsf(l);
        l2_part += fabsf(lc);
        num_part += fabsf(l - lc);
    }
    float en = block_reduce(en_part, warp_buf);
    float l1 = block_reduce(l1_part, warp_buf);
    float l2 = block_reduce(l2_part, warp_buf);
    float num = block_reduce(num_part, warp_buf);

    float newen = energy[b] + en;
    float t = static_cast<float>(length + 1);
    float c = newen / t;
    float we = evidence_gate
        ? (1.0f / (1.0f + expf(-(ev_gain * log1pf(c)
                                + ev_pos_gain * log1pf(t) + ev_bias))))
        : 1.0f;
    float dd = (l1 + l2 > 0.0f) ? (num / (l1 + l2 + gate_eps)) : 0.0f;
    float wu = uncertainty_gate
        ? (1.0f - (1.0f / (1.0f + expf(-unc_logit))) * dd)
        : 1.0f;

    // phase 4: 合成 y、out_proj、输出门、残差（逐 d 并行）
    for (int d = tid; d < D; d += SRF_STREAM_BLOCK) {
        float yv = longb[d] * we * wu + locb[d];    // long 已含证据+不确定两道门
        float o = 0.0f;
        for (int j = 0; j < D; ++j) o += out_w[(size_t)d * D + j] * yv;
        if (out_gate) {
            float s = 0.0f;
            for (int j = 0; j < D; ++j) s += outgate_w[(size_t)d * D + j] * hb[j];
            o /= (1.0f + expf(-s));
        }
        y_t[(size_t)b * D + d] = xb[d] + out_scale * o;
    }
    __syncthreads();

    // phase 5: conv_buf 并行移位（逐 d 独立，无竞争）：new = [old[1..K-2], ub]
    if (K > 1) {
        float* bufb = const_cast<float*>(conv_buf) + (size_t)b * (K - 1) * D;
        for (int d = tid; d < D; d += SRF_STREAM_BLOCK) {
            for (int k = 0; k < K - 1; ++k) {
                float v = (k < K - 2) ? bufb[(size_t)(k + 1) * D + d] : ub[d];
                bufb[(size_t)k * D + d] = v;
            }
        }
        __syncthreads();
    }

    // 更新能量运行和（单一写者，避免竞争）
    if (tid == 0) energy[b] = newen;
}

torch::Tensor srf_stream_step_cuda(
    torch::Tensor x, torch::Tensor h_re, torch::Tensor h_im, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor b_in, torch::Tensor coef,
    torch::Tensor skip, torch::Tensor dt_w, double dt_bias, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    const int64_t B = x.size(0), D = x.size(1), M = coef.size(0), K = local_w.size(1);
    auto y_t = torch::empty_like(x);
    auto u_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto h_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto long_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto loc_g = torch::empty({(int64_t)(B * D)}, x.options());

    srf_kernel<<<(unsigned)B, SRF_STREAM_BLOCK>>>(
        x.data_ptr<float>(), y_t.data_ptr<float>(), h_re.data_ptr<float>(),
        h_im.data_ptr<float>(), conv_buf.data_ptr<float>(), energy.data_ptr<float>(),
        wnorm.data_ptr<float>(), in_w.data_ptr<float>(), gate_w.data_ptr<float>(),
        out_w.data_ptr<float>(), outgate_w.data_ptr<float>(), b_in.data_ptr<float>(),
        coef.data_ptr<float>(), skip.data_ptr<float>(), dt_w.data_ptr<float>(),
        (float)dt_bias, rho.data_ptr<float>(), theta.data_ptr<float>(),
        local_w.data_ptr<float>(), (int)B, (int)D, (int)M, (int)K, (int)length,
        content_gate, evidence_gate, uncertainty_gate, out_gate,
        (float)out_scale, (float)eps, (float)gate_eps, (float)ev_gain,
        (float)ev_pos_gain, (float)ev_bias, (float)unc_logit,
        u_g.data_ptr<float>(), h_g.data_ptr<float>(),
        long_g.data_ptr<float>(), loc_g.data_ptr<float>());
    return y_t;
}
