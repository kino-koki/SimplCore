// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 流式热路径的 torch C++ 扩展入口。
//
// 暴露一个自定义算子 mixer_stream_step：把整批一步的融合 mixer 计算下沉到 C++，
// CPU 后端直接调用 mixer_stream.h 的 mixer_stream_step_host；CUDA 后端（见
// mixer_stream_cuda.cu，仅当以 -DMIXER_STREAM_BUILD_CUDA 编译时链接）走 GPU kernel。
//
// 约定：
//   * 所有张量 fp32、contiguous；x/y_t 形状 (B, D)，状态 y1/y2 (B,M,D)、
//     conv_buf (B,K-1,D)、energy (B,) 或 (B,1)。
//   * y1/y2/conv_buf/energy 原地更新（推理路径，无 autograd）。
//   * 可选权重（gate_w / outgate_w / conv_buf）以 undefined 张量表示"无"。
#include <torch/extension.h>
#include "mixer_stream.h"

#ifdef MIXER_STREAM_BUILD_CUDA
torch::Tensor mixer_stream_step_cuda(
    torch::Tensor x, torch::Tensor y1, torch::Tensor y2, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor coef, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, torch::Tensor feed, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit);
#endif

static torch::Tensor mixer_stream_step_cpu(
    torch::Tensor x, torch::Tensor y1, torch::Tensor y2, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor coef, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, torch::Tensor feed, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    // 内核与 CUDA 路径一律按 fp32 契约接收数据：bf16/fp16 的归一化（x、权重、状态
    // 全部 .to(kFloat32)）由 Python 层 ops/mixer_stream.mixer_stream_step 完成，故此处
    // 永远收到 fp32，无需为每种低精度各写一份实现。
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mixer_stream_step: 仅支持 fp32");
    TORCH_CHECK(x.is_contiguous() && y1.is_contiguous() && y2.is_contiguous(),
                "mixer_stream_step: 张量必须 contiguous");
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t M = coef.size(0);
    const int64_t K = local_w.size(1);

    auto y_t = torch::empty_like(x);
    auto ws = torch::empty({(int64_t)(B * D * 4)}, x.options());

    mixer_stream::MixerStreamWeights w;
    w.B = (int)B; w.D = (int)D; w.M = (int)M; w.K = (int)K;
    w.length = (int)length;
    w.content_gate = content_gate; w.evidence_gate = evidence_gate;
    w.uncertainty_gate = uncertainty_gate; w.out_gate = out_gate;
    w.eps = (float)eps; w.gate_eps = (float)gate_eps; w.out_scale = (float)out_scale;
    w.ev_gain = (float)ev_gain; w.ev_pos_gain = (float)ev_pos_gain; w.ev_bias = (float)ev_bias;
    w.unc_logit = (float)unc_logit;
    w.wnorm = wnorm.data_ptr<float>();
    w.in_w = in_w.data_ptr<float>();
    w.gate_w = gate_w.defined() ? gate_w.data_ptr<float>() : nullptr;
    w.out_w = out_w.data_ptr<float>();
    w.outgate_w = outgate_w.defined() ? outgate_w.data_ptr<float>() : nullptr;
    w.coef = coef.data_ptr<float>();
    w.rho = rho.data_ptr<float>();
    w.theta = theta.data_ptr<float>();
    w.local_w = local_w.data_ptr<float>();
    w.feed = feed.data_ptr<float>();

    float* conv_ptr = conv_buf.defined() ? conv_buf.data_ptr<float>() : nullptr;

    mixer_stream::mixer_stream_step_host(
        x.data_ptr<float>(), y_t.data_ptr<float>(),
        y1.data_ptr<float>(), y2.data_ptr<float>(),
        conv_ptr, energy.data_ptr<float>(), ws.data_ptr<float>(), w);
    return y_t;
}

torch::Tensor mixer_stream_step(
    torch::Tensor x, torch::Tensor y1, torch::Tensor y2, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor coef, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, torch::Tensor feed, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    if (x.device().is_cuda()) {
#ifdef MIXER_STREAM_BUILD_CUDA
        return mixer_stream_step_cuda(x, y1, y2, conv_buf, energy, wnorm, in_w, gate_w,
                                    out_w, outgate_w, coef, rho, theta, local_w, feed,
                                    length, content_gate, evidence_gate,
                                    uncertainty_gate, out_gate, out_scale, eps, gate_eps,
                                    ev_gain, ev_pos_gain, ev_bias, unc_logit);
#else
        TORCH_CHECK(false, "mixer_stream_step: 该扩展未以 CUDA 构建");
#endif
    }
    return mixer_stream_step_cpu(x, y1, y2, conv_buf, energy, wnorm, in_w, gate_w,
                              out_w, outgate_w, coef, rho, theta, local_w, feed,
                              length, content_gate, evidence_gate, uncertainty_gate,
                              out_gate, out_scale, eps, gate_eps, ev_gain, ev_pos_gain,
                              ev_bias, unc_logit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mixer_stream_step", &mixer_stream_step,
          "SimplCore 融合流式 mixer step（CPU/CUDA）");
}
