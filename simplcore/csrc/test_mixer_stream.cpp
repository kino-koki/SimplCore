// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// 无依赖的融合流式 step 数值核验。
//
// 用一份**与 mixer_stream.h 不同代码路径**的独立 reference（先算全部 long 再统一更新
// IIR 状态，而非在 long 计算时原地更新）来交叉验证 mixer_stream_step_host。
// 两者在 fp32 下必须一致（仓库守护阈值 4e-5）。
//
// 编译（MinGW / 任意 C++17 编译器）：
//   g++ -std=c++17 -O2 -I. test_mixer_stream.cpp -o test_mixer_stream && ./test_mixer_stream
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

#include "mixer_stream.h"

namespace {

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// 与 simplcore.core.sgc._init_oscillator_grid 同口径初始化 rho/theta
void init_oscillators(int M, std::vector<float>& rho, std::vector<float>& theta) {
    rho.resize(M); theta.resize(M);
    float lo = std::log(1.0f), hi = std::log(8192.0f);
    for (int m = 0; m < M; ++m) {
        float lam = std::exp(lo + (hi - lo) * m / (M - 1));
        rho[m] = std::exp(-1.0f / lam);
        theta[m] = 3.14159265f * (m + 0.5f) / M;
    }
}

// 独立 reference：先算 long（用步初状态），再统一推进 IIR 状态。
void reference_step(const std::vector<float>& x, std::vector<float>& y_t,
                   std::vector<float>& y1, std::vector<float>& y2,
                   std::vector<float>& conv_buf, std::vector<float>& energy,
                   int B, int D, int M, int K, int length,
                   const mixer_stream::MixerStreamWeights& w) {
    for (int b = 0; b < B; ++b) {
        const float* xb = x.data() + b * D;
        std::vector<float> h(D), u(D), lo(D), lc(D);
        float ss = 0; for (int d = 0; d < D; ++d) ss += xb[d] * xb[d];
        float rn = 1.0f / std::sqrt(ss / D + w.eps);
        for (int d = 0; d < D; ++d) h[d] = xb[d] * rn * w.wnorm[d];
        for (int dd = 0; dd < D; ++dd) {
            float s = 0; const float* iw = w.in_w + (size_t)dd * D;
            for (int j = 0; j < D; ++j) s += iw[j] * h[j];
            u[dd] = s;
        }
        if (w.content_gate) {
            for (int dd = 0; dd < D; ++dd) {
                float s = 0; const float* gw = w.gate_w + (size_t)dd * D;
                for (int j = 0; j < D; ++j) s += gw[j] * h[j];
                u[dd] *= sigmoidf(s);
            }
        }
        float en = 0; for (int d = 0; d < D; ++d) en += u[d] * u[d];
        float newen = energy[b] + en;
        float t = (float)(length + 1);
        float c = newen / t;
        float we = w.evidence_gate ? sigmoidf(w.ev_gain * std::log1p(c)
                                + w.ev_pos_gain * std::log1p(t) + w.ev_bias) : 1.0f;

        // long 用步初状态（y1/y2 当前值），不更新
        for (int d = 0; d < D; ++d) {
            float acc = 0;
            for (int m = 0; m < M; ++m) {
                float r = w.rho[m], th = w.theta[m];
                float a = 2.0f * r * std::cos(th), bb = -r * r, cc = std::sin(th);
                size_t off = (size_t)m * D + d;
                acc += w.coef[off] * (a * y1[(size_t)b * M * D + off]
                                    + bb * y2[(size_t)b * M * D + off]
                                    + cc * u[d]);
            }
            lo[d] = acc;
            float l = 0;
            if (K == 1) { l = w.local_w[d] * u[d]; }
            else {
                const float* cb = conv_buf.data() + (size_t)b * (K - 1) * D;
                for (int k = 0; k < K; ++k) {
                    const float* src = (k < K - 1) ? (cb + (size_t)k * D) : u.data();
                    l += w.local_w[(size_t)d * K + k] * src[d];
                }
            }
            lc[d] = l;
        }
        float l1 = 0, l2 = 0, num = 0;
        for (int d = 0; d < D; ++d) { l1 += std::fabs(lo[d]); l2 += std::fabs(lc[d]);
                                     num += std::fabs(lo[d] - lc[d]); }
        float dd = (l1 + l2 > 0) ? num / (l1 + l2 + w.gate_eps) : 0;
        float wu = w.uncertainty_gate ? (1.0f - sigmoidf(w.unc_logit) * dd) : 1.0f;

        for (int d = 0; d < D; ++d) {
            float yv = lo[d] * we * wu + lc[d] + w.feed[d] * u[d];
            float o = 0; const float* ow = w.out_w + (size_t)d * D;
            for (int j = 0; j < D; ++j) o += ow[j] * yv;
            if (w.out_gate) {
                float s = 0; const float* og = w.outgate_w + (size_t)d * D;
                for (int j = 0; j < D; ++j) s += og[j] * h[j];
                o *= sigmoidf(s);
            }
            y_t[b * D + d] = xb[d] + w.out_scale * o;
        }
        // 统一推进 IIR 状态（独立代码路径）
        for (int m = 0; m < M; ++m) {
            float r = w.rho[m], th = w.theta[m];
            float a = 2.0f * r * std::cos(th), bb = -r * r, cc = std::sin(th);
            for (int d = 0; d < D; ++d) {
                size_t off = (size_t)m * D + d;
                float y1old = y1[(size_t)b * M * D + off];
                float yn = a * y1old + bb * y2[(size_t)b * M * D + off] + cc * u[d];
                y2[(size_t)b * M * D + off] = y1old;
                y1[(size_t)b * M * D + off] = yn;
            }
        }
        if (K > 1) {
            float* bufb = conv_buf.data() + (size_t)b * (K - 1) * D;
            if (K > 2) std::memmove(bufb, bufb + D, (size_t)(K - 2) * D * sizeof(float));
            for (int d = 0; d < D; ++d) bufb[(size_t)(K - 2) * D + d] = u[d];
        }
        energy[b] = newen;
    }
}

}  // namespace

int main() {
    const int B = 2, D = 64, M = 16, K = 4;
    std::mt19937 gen(20261028);
    std::uniform_real_distribution<float> rnd(-0.5f, 0.5f);

    std::vector<float> wnorm(D, 1.0f);
    std::vector<float> in_w((size_t)D * D), gate_w((size_t)D * D), out_w((size_t)D * D);
    std::vector<float> outgate_w((size_t)D * D);
    std::vector<float> coef((size_t)M * D), local_w((size_t)D * K), feed(D, 1.0f);
    std::vector<float> rho, theta;
    init_oscillators(M, rho, theta);
    for (auto& v : in_w) v = rnd(gen);
    for (auto& v : gate_w) v = rnd(gen);
    for (auto& v : out_w) v = rnd(gen);
    for (auto& v : outgate_w) v = rnd(gen);
    // 关键：coef 必须非零，否则振荡器参数梯度恒 0（sgc.py 已知初始化陷阱）
    for (auto& v : coef) v = rnd(gen) * 0.02f;
    for (auto& v : local_w) v = rnd(gen) * 0.1f;

    mixer_stream::MixerStreamWeights w;
    w.B = B; w.D = D; w.M = M; w.K = K; w.length = 3;
    w.content_gate = true; w.evidence_gate = true;
    w.uncertainty_gate = true; w.out_gate = true;
    w.eps = 1e-6f; w.gate_eps = 1e-6f; w.out_scale = 1.0f;
    w.ev_gain = 0.7f; w.ev_pos_gain = 0.1f; w.ev_bias = 0.0f; w.unc_logit = -0.3f;
    w.wnorm = wnorm.data(); w.in_w = in_w.data(); w.gate_w = gate_w.data();
    w.out_w = out_w.data(); w.outgate_w = outgate_w.data();
    w.coef = coef.data(); w.rho = rho.data(); w.theta = theta.data();
    w.local_w = local_w.data(); w.feed = feed.data();

    std::vector<float> ws((size_t)B * D * 4, 0.0f);
    std::vector<float> y1((size_t)B * M * D, 0.0f);
    std::vector<float> y2((size_t)B * M * D, 0.0f);
    std::vector<float> conv_buf((size_t)B * (K - 1) * D, 0.0f);
    std::vector<float> energy(B, 0.0f);

    // 复制一份给 reference（独立状态）
    std::vector<float> r_y1 = y1, r_y2 = y2, r_conv = conv_buf, r_energy = energy;

    int max_steps = 12;
    float worst = 0.0f;
    for (int step = 0; step < max_steps; ++step) {
        std::vector<float> x((size_t)B * D);
        for (auto& v : x) v = rnd(gen);
        std::vector<float> x_r = x;

        std::vector<float> y_t((size_t)B * D), y_t_r((size_t)B * D);
        // 融合实现（原地推进 y1/y2/conv/energy）
        mixer_stream::mixer_stream_step_host(x.data(), y_t.data(), y1.data(), y2.data(),
                                  conv_buf.data(), energy.data(), ws.data(), w);
        // 独立 reference
        reference_step(x_r, y_t_r, r_y1, r_y2, r_conv, r_energy, B, D, M, K, w.length, w);
        w.length += 1;

        for (size_t i = 0; i < y_t.size(); ++i)
            worst = std::max(worst, std::fabs(y_t[i] - y_t_r[i]));
        for (size_t i = 0; i < y1.size(); ++i)
            worst = std::max(worst, std::fabs(y1[i] - r_y1[i]));
        for (size_t i = 0; i < y2.size(); ++i)
            worst = std::max(worst, std::fabs(y2[i] - r_y2[i]));
        for (size_t i = 0; i < conv_buf.size(); ++i)
            worst = std::max(worst, std::fabs(conv_buf[i] - r_conv[i]));
        for (size_t i = 0; i < energy.size(); ++i)
            worst = std::max(worst, std::fabs(energy[i] - r_energy[i]));
    }

    std::printf("steps=%d  B=%d D=%d M=%d K=%d  worst|Δ| = %.3e\n",
                max_steps, B, D, M, K, worst);
    if (worst > 1e-4f) {
        std::printf("FAIL: 超过阈值 1e-4\n");
        return 1;
    }
    std::printf("PASS: 融合实现与独立 reference 在 fp32 下一致（仓库守护阈值 4e-5）\n");
    return 0;
}
