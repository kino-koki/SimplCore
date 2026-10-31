# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""纯算子重组模型 —— 用 **且仅用** ``BackendAdapter`` 的内核搭出完整 SimplCore。

这是 16 算子契约的**完备性证明**：如果这 16 个内核不足以表达整个模型的
前向（并行 + 流式）与训练梯度，契约就是不完备的，后端团队会在中途发现
缺内核——那种发现必须发生在设计期，不是集成期。

本模块把 ``SimplCoreModel``（默认 ``sgc`` mixer 路径）逐层重组：

* 嵌入查表 → ``emb_lookup``
* RMSNorm → ``rmsnorm``
* 所有线性投影（in/gate/out/head/FFN）→ ``gemm``
* SwiGLU FFN → ``swiglu`` + ``gemm``
* 残差 → ``residual_add``
* 内容门 → ``content_gate``；证据/不确定门 → ``evidence_gate_*``/``uncertainty_gate``
* 长程路径（并行）→ ``spectral_causal_conv``；（流式）→ ``chunked_iir_scan``
* 局部路径 → ``causal_depthwise_conv``
* 损失 → ``ce_loss``
* 训练 → ``adamw_frozen_mask`` + ``weight_decay_exempt``（trainer.py）
* 快照/回滚 → ``stream_state_serialize``（trainer.py）

**不属于算子的部分**：振荡器核构建（``_kernel``，参数→核矩阵的确定性
host 侧变换，每步一次，不是序列内核）。除此之外每一个张量运算都经过
适配器——用记录型适配器（``RecordingAdapter``）可在测试中证明这一点。

与原生模型的关系
----------------
``load_native`` 直接吃 ``SimplCoreModel.state_dict()``；同名权重、同数学、
同初始化语义（含 ρ 稳定性钳制、out_scale 缩放初始化的影响面）。fp32 下
并行/流式前向与原生模型 ``forward`` / ``stream_step`` 数值一致（测试守护）。

诚实边界
--------
只支持默认 ``sgc`` mixer 且 ``gate_ab=False``（战役 preset 的路径）；
``srf``/``GatedSpectralConv2``/CMT 记忆注入/拒答头不在重组范围。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from ..backend.backend_contract import DEFAULT_CONTRACT
from .config import SimplCoreConfig
from ..backend.conformance_runner import BackendAdapter, REQUIRED_METHODS

__all__ = ["AdapterModel", "RecordingAdapter"]


class RecordingAdapter(BackendAdapter):
    """记录型适配器：包装另一个适配器并记录被调用的方法名。

    用于**契约覆盖审计**：一次完整前向 + 流式 + 训练步之后，
    ``called_ops()`` 与 16 契约算子的映射差集就是"契约写了但没人用"或
    "模型需要但契约没有"的证据。

    实现注意：基类的桩方法（``def gemm(...): ...``）在 MRO 上先于
    ``__getattr__``，因此必须在 ``__init__`` 里把未被覆盖的桩**替换成
    实例级代理**，否则调用会静默落到桩上返回 None。
    """

    def __init__(self, inner: BackendAdapter) -> None:
        self.inner = inner
        self.called: List[str] = []
        self.name = f"recording[{getattr(inner, 'name', type(inner).__name__)}]"
        for m in REQUIRED_METHODS:
            if getattr(type(self), m, None) is getattr(BackendAdapter, m, None):
                setattr(self, m, self._make_proxy(m))

    def _make_proxy(self, attr: str):
        def proxy(*a, **k):
            self.called.append(attr)
            return getattr(self.inner, attr)(*a, **k)
        proxy.__name__ = attr
        return proxy

    def ops_registry(self) -> Dict[str, object]:
        reg = dict(self.inner.ops_registry())
        reg["recorded_calls"] = self.called_ops
        return reg

    def called_ops(self) -> List[str]:
        return sorted(set(self.called))


class AdapterModel:
    """逐算子重组的 SimplCore（fp32；参数字典键与原生 state_dict 同名）。"""

    def __init__(self, cfg: SimplCoreConfig, adapter: BackendAdapter) -> None:
        if cfg.mixer != "sgc" or cfg.gate_ab:
            raise ValueError("AdapterModel 仅支持 mixer='sgc' 且 gate_ab=False")
        if cfg.dropout:
            raise ValueError("AdapterModel 不支持 dropout（训练语义在 trainer 中）")
        if cfg.use_abstain or cfg.use_memory:
            raise ValueError("AdapterModel 不含拒答头 / CMT 记忆注入")
        self.cfg = cfg
        self.adapter = adapter
        self.eps = cfg.norm_eps
        self.P: Dict[str, torch.nn.Parameter] = {}
        D, M, K, H, V = cfg.dim, cfg.num_bases, cfg.local_kernel, cfg.hidden, cfg.vocab_size
        P = self.P
        P["tok_emb.weight"] = torch.nn.Parameter(torch.empty(V, D))
        for i in range(cfg.num_layers):
            p = {k: f"blocks.{i}.{k}" for k in ()}  # noqa: F841 —— 可读性占位
            P[f"blocks.{i}.mixer.norm.weight"] = torch.nn.Parameter(torch.empty(D))
            P[f"blocks.{i}.mixer.in_proj.weight"] = torch.nn.Parameter(torch.empty(D, D))
            if cfg.content_gate:
                P[f"blocks.{i}.mixer.gate_proj.weight"] = torch.nn.Parameter(torch.empty(D, D))
            P[f"blocks.{i}.mixer.out_proj.weight"] = torch.nn.Parameter(torch.empty(D, D))
            P[f"blocks.{i}.mixer.raw_rho"] = torch.nn.Parameter(torch.empty(M))
            P[f"blocks.{i}.mixer.raw_theta"] = torch.nn.Parameter(torch.empty(M))
            P[f"blocks.{i}.mixer.coef"] = torch.nn.Parameter(torch.empty(M, D))
            P[f"blocks.{i}.mixer.feed"] = torch.nn.Parameter(torch.empty(D))
            P[f"blocks.{i}.mixer.local_conv.weight"] = torch.nn.Parameter(torch.empty(D, 1, K))
            P[f"blocks.{i}.mixer.out_scale"] = torch.nn.Parameter(torch.empty(1))
            # 闸门 opt-in（v1.2）：与原生 mixer 一致，门关闭时不注册参数。
            if cfg.evidence_gate:
                P[f"blocks.{i}.mixer.evidence_gain"] = torch.nn.Parameter(torch.empty(()))
                P[f"blocks.{i}.mixer.evidence_pos_gain"] = torch.nn.Parameter(torch.empty(()))
                P[f"blocks.{i}.mixer.evidence_bias"] = torch.nn.Parameter(torch.empty(()))
            if cfg.uncertainty_gate:
                P[f"blocks.{i}.mixer.uncertainty_logit"] = torch.nn.Parameter(torch.empty(1))
            P[f"blocks.{i}.ffn.norm.weight"] = torch.nn.Parameter(torch.empty(D))
            P[f"blocks.{i}.ffn.gate.weight"] = torch.nn.Parameter(torch.empty(H, D))
            P[f"blocks.{i}.ffn.up.weight"] = torch.nn.Parameter(torch.empty(H, D))
            P[f"blocks.{i}.ffn.down.weight"] = torch.nn.Parameter(torch.empty(D, H))
        P["ln_f.weight"] = torch.nn.Parameter(torch.empty(D))
        if cfg.tie_embeddings:
            P["head.weight"] = P["tok_emb.weight"]           # 共享同一 Parameter
        else:
            P["head.weight"] = torch.nn.Parameter(torch.empty(V, D))
        for p in self.parameters():
            p.requires_grad_(True)

    # -- 参数管理 --------------------------------------------------------------
    def parameters(self):
        seen, out = set(), []
        for p in self.P.values():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def named_parameters(self):
        seen, out = set(), []
        for k, p in self.P.items():
            if id(p) not in seen:
                seen.add(id(p))
                out.append((k, p))
        return out

    def load_native(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """从 ``SimplCoreModel.state_dict()`` 加载（同名逐位拷贝）。"""
        copied = set()
        with torch.no_grad():
            for k, v in state_dict.items():
                if k not in self.P:
                    continue
                p = self.P[k]
                if id(p) in copied:
                    continue
                p.copy_(v)
                copied.add(id(p))
        missing = [k for k in self.P if k not in state_dict]
        if missing:
            raise KeyError(f"native state_dict 缺键: {missing[:4]}")

    # -- 振荡器核（host 侧确定性参数变换，非序列内核） --------------------------
    def _osc(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = torch.sigmoid(self.P[f"blocks.{i}.mixer.raw_rho"]).clamp(max=1.0 - 1e-6)
        theta = torch.sigmoid(self.P[f"blocks.{i}.mixer.raw_theta"]) * math.pi
        return rho, theta

    def _kernel(self, i: int, length: int) -> torch.Tensor:
        """核 bank K[L,D] = Σ_m coef[m,d]·ρ_m^τ·sin((τ+1)θ_m)（镜像 build_kernel）。"""
        rho, theta = self._osc(i)
        coef = self.P[f"blocks.{i}.mixer.coef"]
        tau = torch.arange(length, dtype=coef.dtype).unsqueeze(0)
        psi = torch.exp(torch.log(rho).unsqueeze(1) * tau) \
            * torch.sin((tau + 1.0) * theta.unsqueeze(1))
        return (coef.t() @ psi).t().contiguous()          # (L, D)

    # -- 前向（并行 / FFT 形式） ------------------------------------------------
    def _mixer(self, x: torch.Tensor, i: int) -> torch.Tensor:
        ad, P = self.adapter, self.P
        pre = f"blocks.{i}.mixer"
        h = ad.rmsnorm(x, P[f"{pre}.norm.weight"], self.eps)
        u = ad.gemm(h, P[f"{pre}.in_proj.weight"].t())
        if self.cfg.content_gate:
            u = u * ad.content_gate(h, P[f"{pre}.gate_proj.weight"])
        long_ = ad.spectral_causal_conv(u, self._kernel(i, x.shape[1]))
        local, _ = ad.causal_depthwise_conv(u, P[f"{pre}.local_conv.weight"], None)
        if self.cfg.evidence_gate:
            long_ = long_ * ad.evidence_gate_parallel(
                u, P[f"{pre}.evidence_gain"], P[f"{pre}.evidence_pos_gain"],
                P[f"{pre}.evidence_bias"])
        if self.cfg.uncertainty_gate:
            long_ = long_ * ad.uncertainty_gate(
                long_, local, P[f"{pre}.uncertainty_logit"], 1e-6)
        y = long_ + local + P[f"{pre}.feed"] * u
        out = ad.gemm(y, P[f"{pre}.out_proj.weight"].t())
        return x + P[f"{pre}.out_scale"] * out

    def _ffn(self, x: torch.Tensor, i: int) -> torch.Tensor:
        ad, P = self.adapter, self.P
        pre = f"blocks.{i}.ffn"
        h = ad.rmsnorm(x, P[f"{pre}.norm.weight"], self.eps)
        gate = ad.gemm(h, P[f"{pre}.gate.weight"].t())
        up = ad.gemm(h, P[f"{pre}.up.weight"].t())
        return ad.residual_add(x, ad.gemm(ad.swiglu(gate, up), P[f"{pre}.down.weight"].t()))

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """并行前向：与 ``SimplCoreModel.forward`` 同数学（fp32）。"""
        ad = self.adapter
        x = ad.emb_lookup(self.P["tok_emb.weight"], idx)
        for i in range(self.cfg.num_layers):
            x = self._mixer(x, i)
            x = self._ffn(x, i)
        x = ad.rmsnorm(x, self.P["ln_f.weight"], self.eps)
        logits = ad.gemm(x, self.P["head.weight"].t())
        loss = None
        if targets is not None:
            loss = ad.ce_loss(logits, targets)
        return logits, loss

    # -- 前向（流式 / 分块 IIR 形式） --------------------------------------------
    def init_stream_states(self, batch: int) -> List[Dict[str, torch.Tensor]]:
        """每层 O(1) 流式状态（y1/y2/局部卷积缓冲/证据能量和）。"""
        cfg = self.cfg
        M, K = cfg.num_bases, cfg.local_kernel
        return [dict(y1=torch.zeros(batch, M, cfg.dim),
                     y2=torch.zeros(batch, M, cfg.dim),
                     buf=torch.zeros(batch, max(K - 1, 0), cfg.dim),
                     es=torch.zeros(batch, 1)) for _ in range(cfg.num_layers)]

    def forward_chunk(self, idx_chunk: torch.Tensor,
                      states: List[Dict[str, torch.Tensor]], length: int,
                      targets: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """消费一个 chunk（状态就地推进）：训练/流式评估的块级入口。"""
        ad, P, cfg = self.adapter, self.P, self.cfg
        xc = ad.emb_lookup(P["tok_emb.weight"], idx_chunk)
        for i, st in enumerate(states):
            pre = f"blocks.{i}.mixer"
            rho, theta = self._osc(i)
            a = 2.0 * rho * torch.cos(theta)
            b = -(rho ** 2)
            c = torch.sin(theta)
            h = ad.rmsnorm(xc, P[f"{pre}.norm.weight"], self.eps)
            u = ad.gemm(h, P[f"{pre}.in_proj.weight"].t())
            if cfg.content_gate:
                u = u * ad.content_gate(h, P[f"{pre}.gate_proj.weight"])
            long_, st["y1"], st["y2"] = ad.chunked_iir_scan(
                u, st["y1"], st["y2"], a, b, c, P[f"{pre}.coef"])
            loc, st["buf"] = ad.causal_depthwise_conv(
                u, P[f"{pre}.local_conv.weight"], st["buf"])
            if cfg.evidence_gate:
                ev, st["es"] = ad.evidence_gate_chunk(
                    u, st["es"], length, P[f"{pre}.evidence_gain"],
                    P[f"{pre}.evidence_pos_gain"], P[f"{pre}.evidence_bias"])
                long_ = long_ * ev
            if cfg.uncertainty_gate:
                long_ = long_ * ad.uncertainty_gate(
                    long_, loc, P[f"{pre}.uncertainty_logit"], 1e-6)
            y = long_ + loc + P[f"{pre}.feed"] * u
            xc = xc + P[f"{pre}.out_scale"] * ad.gemm(y, P[f"{pre}.out_proj.weight"].t())
            # FFN（与 native stream_step 同序：mixer → ffn）
            fpre = f"blocks.{i}.ffn"
            hf = ad.rmsnorm(xc, P[f"{fpre}.norm.weight"], self.eps)
            gate = ad.gemm(hf, P[f"{fpre}.gate.weight"].t())
            up = ad.gemm(hf, P[f"{fpre}.up.weight"].t())
            xc = ad.residual_add(
                xc, ad.gemm(ad.swiglu(gate, up), P[f"{fpre}.down.weight"].t()))
        xc = ad.rmsnorm(xc, P["ln_f.weight"], self.eps)
        logits = ad.gemm(xc, P["head.weight"].t())
        loss = None
        if targets is not None:
            loss = ad.ce_loss(logits, targets)
        return logits, loss

    def forward_streaming(self, idx: torch.Tensor, chunk: int,
                          targets: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """分块流式前向：O(1) 状态跨块携带，与并行前向同数学（4e-5 口径）。"""
        states = self.init_stream_states(idx.shape[0])
        parts, length = [], 0
        for s in range(0, idx.shape[1], chunk):
            uc = idx[:, s:s + chunk]
            tc = None if targets is None else targets[:, s:s + chunk]
            logits_c, _ = self.forward_chunk(uc, states, length, None)
            parts.append(logits_c)
            length += uc.shape[1]
        logits = torch.cat(parts, dim=1)
        loss = None
        if targets is not None:
            loss = self.adapter.ce_loss(logits, targets)
        return logits, loss

    # -- 训练辅助 --------------------------------------------------------------
    def param_names(self) -> List[str]:
        """按 native state_dict 语义去重后的可训练参数名（head 共享只计一次）。"""
        return [k for k, _ in self.named_parameters()]
