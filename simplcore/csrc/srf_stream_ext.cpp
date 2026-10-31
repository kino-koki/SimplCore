// Copyright 2026 (C) 六安市青奈科技有限责任公司
// -*- coding: utf-8 -*-
//
// SimplCore 谱振流（SRF）mixer 融合热路径的 torch C++ 扩展入口。
//
// 暴露自定义算子 srf_stream_step：把整批一步的融合 SRF mixer 计算下沉到 C++，
// CPU 后端调用 srf_stream.h 的 srf_stream_step_host；CUDA 后端（见
// srf_stream_cuda.cu，仅当 -DSRF_STREAM_BUILD_CUDA 编译时链接）走 GPU kernel。
//
// 约定（与 mixer_stream_ext.cpp 一致）：
//   * 所有张量 fp32、contiguous；x/y_t 形状 (B, D)，复数状态 h_re/h_im (B,M,D)、
//     conv_buf (B,K-1,D)、energy (B,) 或 (B,1)。
//   * h_re/h_im/conv_buf/energy 原地更新（推理路径，无 autograd）。
//   * 可选权重（gate_w / outgate_w / conv_buf）以 undefined 张量表示"无"。
//   * 内核与 CUDA 路径一律按 fp32 契约接收数据：bf16/fp16 的归一化由 Python 层
//     ops/srf_stream.srf_stream_step 完成，故此处永远收到 fp32。
#include <torch/extension.h>
#include "srf_stream.h"

#ifdef SRF_STREAM_BUILD_CUDA
torch::Tensor srf_stream_step_cuda(
    torch::Tensor x, torch::Tensor h_re, torch::Tensor h_im, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor b_in, torch::Tensor coef,
    torch::Tensor skip, torch::Tensor dt_w, double dt_bias, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit);
#endif

static torch::Tensor srf_stream_step_cpu(
    torch::Tensor x, torch::Tensor h_re, torch::Tensor h_im, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor b_in, torch::Tensor coef,
    torch::Tensor skip, torch::Tensor dt_w, double dt_bias, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "srf_stream_step: 仅支持 fp32");
    TORCH_CHECK(x.is_contiguous() && h_re.is_contiguous() && h_im.is_contiguous(),
                "srf_stream_step: 张量必须 contiguous");
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t M = coef.size(0);
    const int64_t K = local_w.size(1);

    auto y_t = torch::empty_like(x);
    auto ws = torch::empty({(int64_t)(B * D * 4)}, x.options());

    srf::SrfStreamWeights w;
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
    w.b_in = b_in.data_ptr<float>();
    w.coef = coef.data_ptr<float>();
    w.skip = skip.data_ptr<float>();
    w.dt_w = dt_w.data_ptr<float>();
    w.dt_bias = (float)dt_bias;
    w.rho = rho.data_ptr<float>();
    w.theta = theta.data_ptr<float>();
    w.local_w = local_w.data_ptr<float>();

    float* conv_ptr = conv_buf.defined() ? conv_buf.data_ptr<float>() : nullptr;

    srf::srf_stream_step_host(
        x.data_ptr<float>(), y_t.data_ptr<float>(),
        h_re.data_ptr<float>(), h_im.data_ptr<float>(), conv_ptr,
        energy.data_ptr<float>(), ws.data_ptr<float>(), w);
    return y_t;
}

torch::Tensor srf_stream_step(
    torch::Tensor x, torch::Tensor h_re, torch::Tensor h_im, torch::Tensor conv_buf,
    torch::Tensor energy, torch::Tensor wnorm, torch::Tensor in_w, torch::Tensor gate_w,
    torch::Tensor out_w, torch::Tensor outgate_w, torch::Tensor b_in, torch::Tensor coef,
    torch::Tensor skip, torch::Tensor dt_w, double dt_bias, torch::Tensor rho,
    torch::Tensor theta, torch::Tensor local_w, int64_t length,
    bool content_gate, bool evidence_gate, bool uncertainty_gate, bool out_gate,
    double out_scale, double eps, double gate_eps, double ev_gain, double ev_pos_gain,
    double ev_bias, double unc_logit) {
    if (x.device().is_cuda()) {
#ifdef SRF_STREAM_BUILD_CUDA
        // K==1 时 conv_buf 可能为 undefined；构造一个零长占位，kernel 不会读取它。
        torch::Tensor conv_arg = conv_buf.defined()
            ? conv_buf
            : torch::zeros({0}, x.options());
        return srf_stream_step_cuda(x, h_re, h_im, conv_arg, energy, wnorm, in_w,
                                   gate_w, out_w, outgate_w, b_in, coef, skip, dt_w,
                                   dt_bias, rho, theta, local_w, length, content_gate,
                                   evidence_gate, uncertainty_gate, out_gate, out_scale,
                                   eps, gate_eps, ev_gain, ev_pos_gain, ev_bias,
                                   unc_logit);
#else
        TORCH_CHECK(false, "srf_stream_step: 该扩展未以 CUDA 构建");
#endif
    }
    return srf_stream_step_cpu(x, h_re, h_im, conv_buf, energy, wnorm, in_w, gate_w,
                              out_w, outgate_w, b_in, coef, skip, dt_w, dt_bias, rho,
                              theta, local_w, length, content_gate, evidence_gate,
                              uncertainty_gate, out_gate, out_scale, eps, gate_eps,
                              ev_gain, ev_pos_gain, ev_bias, unc_logit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("srf_stream_step", &srf_stream_step,
          "SimplCore 谱振流(SRF) 融合流式 mixer step（CPU/CUDA）");
}
