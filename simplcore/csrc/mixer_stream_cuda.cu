// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 流式热路径的 CUDA kernel。
//
// 设计取舍（与仓库"诚实纪律"一致）：
//   * 本文件需 nvcc 编译；在当前开发机（无 CUDA 工具链）无法实测，必须在装有
//     torch + CUDA + nvcc 的目标机上由 test_mixer_stream.py 验证 fp32 等价性。
//   * 正确性优先，但**不再用 thread0 串行**——所有 per-b 归约（RMSNorm 平方和、
//     证据能量和、L3 长/短分歧的 l1/l2/num）改用 warp-shuffle + 共享内存的并行
//     归约；conv_buf 移位也逐 d 并行。并行归约改变了求和顺序，与串行 CPU 参考
//     会有 ≤ 1e-6 量级的相对差，远在守护阈值 1e-4 之内（详见 docs/MIXER_STREAM.md §5）。
//
// 关键正确性修复（v2）：
//   初版 phase-1 在同一个循环里"写 hb[d] 同时读 hb[j]（j 属其它线程）"，而
//   __syncthreads() 在循环之后——hb 是 torch::empty（未初始化）的全局 scratch，
//   跨线程读到的是尚未写过的垃圾，CUDA 路径会产出错误结果。纯 PyTorch / CPU 串行
//   参考无此问题；而本机无 nvcc 导致 pytest 一直 skip，该 bug 从未暴露。
//   现把 h 的写与矩阵的读拆成两趟 + 同步，彻底消除竞争。
#include <torch/extension.h>
#include "mixer_stream.h"

constexpr int MIXER_STREAM_BLOCK = 256;   // 必须等于 8 个 warp（见 block_reduce）
static_assert(MIXER_STREAM_BLOCK == 256, "kernel block must be 256 (=8 warps)");

/// 块内全归约：把每个线程的 partial 求和后返回给所有线程。
/// 用 warp-shuffle 做 warp 内归约，再用 8 个 warp 的部分和做第二级归约。
__device__ float block_reduce(float partial, float* warp_buf) {
    // 1) warp 内 shuffle 归约（每个 warp 32 线程）
    float sum = partial;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    // lane 0 持有本 warp 的局部和
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;            // 0..7
    if (lane == 0) warp_buf[wid] = sum;
    __syncthreads();
    // 2) warp 0 归约 8 个 warp 局部和（warp_buf 仅 8 项）
    float total = 0.0f;
    if (wid == 0) {
        total = (lane < 8) ? warp_buf[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
        if (lane == 0) warp_buf[0] = total;     // 仅 lane 0 写最终结果
    }
    __syncthreads();
    return warp_buf[0];                          // 广播给全块
}

__global__ void mixer_kernel(
    const float* x, float* y_t, float* y1, float* y2, const float* conv_buf, float* energy,
    const float* wnorm, const float* in_w, const float* gate_w, const float* out_w,
    const float* outgate_w, const float* coef, const float* rho, const float* theta,
    const float* local_w, const float* feed,
    int B, int D, int M, int K, int length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    float out_scale, float eps, float gate_eps, float ev_gain, float ev_pos_gain,
    float ev_bias, float unc_logit,
    float* u_g, float* h_g, float* long_g, float* loc_g) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float warp_buf[8];

    const float* xb = x + (size_t)b * D;
    float* y1b = y1 + (size_t)b * M * D;
    float* y2b = y2 + (size_t)b * M * D;
    const float* cb = (K > 1) ? (conv_buf + (size_t)b * (K - 1) * D) : nullptr;
    float* ub = u_g + (size_t)b * D;
    float* hb = h_g + (size_t)b * D;
    float* longb = long_g + (size_t)b * D;
    float* locb = loc_g + (size_t)b * D;

    // phase 0: RMSNorm 归一化系数（并行归约平方和）
    float ss_part = 0.0f;
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) ss_part += xb[d] * xb[d];
    float ss = block_reduce(ss_part, warp_buf);
    float rn = 1.0f / sqrtf(ss / D + eps);

    // phase 1a: 先把 h 全部写出（跨线程同步后再读，消除竞争）
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK)
        hb[d] = xb[d] * rn * wnorm[d];
    __syncthreads();

    // phase 1b: in_proj + 内容门控（读已写满的 hb）
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) {
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

    // phase 2: 振荡基 + 局部卷积 + 状态原地更新（逐 d 并行，状态按 (m,d) 互不干扰）
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) {
        float lo, lc;
        mixer_stream::resonator_conv_and_local(b, d, B, D, M, K, ub, coef, rho, theta, local_w,
                         y1b, y2b, cb, &lo, &lc);
        longb[d] = lo;
        locb[d] = lc;
    }
    __syncthreads();

    // phase 3: 并行归约 L2 证据能量和 / L3 长-短分歧
    float en_part = 0.0f, l1_part = 0.0f, l2_part = 0.0f, num_part = 0.0f;
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) {
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

    // phase 4: 合成 y、out_proj、输出门、残差（逐 d 并行；long/loc/u 已写满）
    for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) {
        float yv = longb[d] * we * wu + locb[d] + feed[d] * ub[d];
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
        for (int d = tid; d < D; d += MIXER_STREAM_BLOCK) {
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

torch::Tensor mixer_stream_step_cuda(
    torch::Tensor x, torch::Tensor y1, torch::Tensor y2, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor coef, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, torch::Tensor feed, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    const int64_t B = x.size(0), D = x.size(1), M = coef.size(0), K = local_w.size(1);
    auto y_t = torch::empty_like(x);
    auto u_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto h_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto long_g = torch::empty({(int64_t)(B * D)}, x.options());
    auto loc_g = torch::empty({(int64_t)(B * D)}, x.options());

    mixer_kernel<<<(unsigned)B, MIXER_STREAM_BLOCK>>>(
        x.data_ptr<float>(), y_t.data_ptr<float>(), y1.data_ptr<float>(),
        y2.data_ptr<float>(), conv_buf.data_ptr<float>(), energy.data_ptr<float>(),
        wnorm.data_ptr<float>(), in_w.data_ptr<float>(), gate_w.data_ptr<float>(),
        out_w.data_ptr<float>(), outgate_w.data_ptr<float>(), coef.data_ptr<float>(),
        rho.data_ptr<float>(), theta.data_ptr<float>(), local_w.data_ptr<float>(),
        feed.data_ptr<float>(), (int)B, (int)D, (int)M, (int)K, (int)length,
        content_gate, evidence_gate, uncertainty_gate, out_gate,
        (float)out_scale, (float)eps, (float)gate_eps, (float)ev_gain,
        (float)ev_pos_gain, (float)ev_bias, (float)unc_logit,
        u_g.data_ptr<float>(), h_g.data_ptr<float>(),
        long_g.data_ptr<float>(), loc_g.data_ptr<float>());
    return y_t;
}
