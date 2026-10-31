# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""生物计算主义（Biological Computationalism）—— 把 Milinković & Aru (2026,
Neurosci. Biobehav. Rev. 181, 106524) 的三条核心计算原则做成**真实可训练**的
训练增强，并接入 SimplCore 训练闭环（opt-in，默认关闭，零回归）。

论文主张当前数字 AI 与生物计算之间存在「尺度可分 / 离散化 / 与底物解耦」的鸿沟，
并给出生物计算区别于数字计算的三条**必要**计算特征（作者明确说这不限于碳基，
只要非生物系统能复现这些计算属性即可）：

1. 混合计算（Hybrid computation）
   离散事件（神经脉冲）嵌入连续值过程（电场 / 离子流 / 局部场电位 / 树突整合）
   之中，二者**相互构成**、不可抽象剥离。本架构里：SGC 是连续谱卷积（连续底物），
   token 是离散事件——我们用一个**常驻 O(1) 连续场**把二者耦合起来。

2. 尺度不可分（Scale-inseparability / heterarchy）
   离子通道→树突→环路→全脑动力学的各个尺度双向耦合：低尺度内生地生成高尺度，
   高尺度同时约束低尺度（连续共决定），不存在「真正发生计算的特权尺度」。
   本架构里原本分离的 micro（token / 路由）/ meso（连续场）/ macro（seq-len 课程）
   三尺度，被一个**可训练的耦合器**双向连结。

3. 代谢约束（Metabolic / energetic constraints）
   代谢预算（脑占体重 2% 却耗 20% 能量）主动塑造学习、稳定性与信息流。
   我们用一个**代谢账本 + 能量预算**把「能耗」作为可被反向传播塑形的信号：
   超额能耗产生代谢效率损失，预算紧张时自动收缩场增益（稳定性 / 信息流被代谢塑造）。

诚实边界
--------
* 这是把论文的**计算风格**做成机制实验件，绝非声称模型「有意识」。
  论文本身也只是论证这些是意识的*可能*必要条件，且明确可被非生物系统复现。
* 默认关闭；关闭时训练行为与基座逐位一致（零回归）。
* 连续场状态 O(1)、与序列长度无关，与 SGC 流式不变量一致（长度无关、无 max_len）。
* 与全架构纯洁性约束一致：不引入注意力，全部为线性 / 谱 / 连续递推。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BioStreamState:
    """生物计算主义流式状态：包裹底层（基座 / Unity）的 O(1) 流式状态。

    连续场 ``phi`` 本身作为 ``ContinuousField`` 的 buffer 常驻（O(1)），
    不进状态对象；这里只携带底层递推状态与序列长度，供 checkpoint/restore。
    """
    base: object
    length: int = 0


def _sample_logits(logits: torch.Tensor, temperature: float,
                   top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    """采样下一 token（与 ``SimplCoreModel._sample`` 同语义）。"""
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(srt, dim=-1).cumsum(-1)
        remove = probs - F.softmax(srt, dim=-1) > top_p
        srt = srt.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, srt_idx, srt)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _infer_dim(model: nn.Module) -> int:
    """从基座推断隐藏宽度 D（与 UnityModel._infer_dim 同口径）。"""
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "dim", None) is not None:
        return int(cfg.dim)
    emb = getattr(model, "tok_emb", None)
    if emb is not None:
        return int(emb.weight.shape[1])
    raise ValueError("无法推断基座维度：model 既无 .config.dim 也无 .tok_emb")


# ---------------------------------------------------------------------------
# 支柱 1：混合计算 —— 离散脉冲 + 常驻连续场（相互构成）
# ---------------------------------------------------------------------------
class ContinuousField(nn.Module):
    """常驻 O(1) 连续场 ``phi``：离散 token 处理嵌入连续动力之中。

    每个前向调用对序列 L 个位置展开（unroll）：

    * 离散 → 连续：每个 token 隐状态 ``h_t`` 向场注入「脉冲」 ``spike(h_t)``；
    * 连续演化：``phi <- phi + dt·tanh(W·phi + b) + spike``（Euler 步，神经场动力学）；
    * 连续 → 离散：场以可学习的逐维门 ``g=σ(gain(phi))·scale`` 调制 token 处理
      ``h_t' = h_t + g ⊗ phi``。

    连续与离散**相互构成**（论文原话："mutually constitutive"）。序列内梯度在
    unroll 中贯通（场动力学可训）；chunk 边界处 ``phi`` 仅以 O(1) 常驻态携带
    （detach 一次，内存与序列长度无关——与 SGC 流式不变量一致）。
    """

    def __init__(self, dim: int, dt: float = 0.5, scale: float = 1.0,
                 leak: float = 0.05) -> None:
        super().__init__()
        self.dim = int(dim)
        self.dt = float(dt)
        self.scale = float(scale)
        # 耗散项（leak）：生物连续场是耗散系统（有阻尼），leak 保证数值稳定、
        # 防止 4M 上下文下场无限累积，也让代谢预算有意义（场能量有界）。
        self.leak = float(leak)
        # 连续动力：线性递推 + tanh 非线性（神经场 / 局部场电位类比）
        self.W = nn.Linear(dim, dim, bias=True)
        # 离散 → 连续：脉冲注入
        self.spike = nn.Linear(dim, dim, bias=False)
        # 连续 → 离散：场→token 调制门
        self.gain = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()
        # 常驻 O(1) 连续场状态（population-level），跨 chunk 携带
        self.register_buffer("phi", torch.zeros(1, dim), persistent=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.W.weight, std=0.02)
        nn.init.zeros_(self.W.bias)
        nn.init.normal_(self.spike.weight, std=0.02)
        nn.init.normal_(self.gain.weight, std=0.02)

    def _ensure(self, B: int, device, dtype) -> torch.Tensor:
        phi = self.phi
        if phi.shape[0] != B or phi.device != device:
            phi = torch.zeros(B, self.dim, device=device, dtype=dtype)
            self.phi = phi
        return phi[:B]

    def forward_sequence(self, h: torch.Tensor, dt: Optional[torch.Tensor] = None,
                        scale: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``h``: (B, L, D)。返回 (调制后隐状态 (B,L,D), 末位场 (B,D))。

        ``dt`` / ``scale`` 可由尺度耦合器动态注入（macro→meso、meso→micro）。
        """
        B, L, D = h.shape
        phi = self._ensure(B, h.device, h.dtype)          # 携带自上一 chunk（已 detach）
        # 注意：dt / scale 以张量传入时保留梯度（耦合器权重可训），不可 float() 剥离。
        dt_v = self.dt if dt is None else dt
        scale_v = self.scale if scale is None else scale
        mods: list = []
        for t in range(L):
            h_t = h[:, t, :]
            spike = self.spike(h_t)                        # 离散 → 连续
            # 连续演化（Euler，耗散）：phi = (1 - dt·leak)·phi + dt·tanh(W·phi) + spike
            phi = (1.0 - dt_v * self.leak) * phi \
                + dt_v * torch.tanh(self.W(phi)) + spike
            g = torch.sigmoid(self.gain(phi)) * scale_v    # 连续 → 离散门
            h_mod = h_t + g * phi
            mods.append(h_mod)
        phi = phi.detach()                                 # O(1) 跨 chunk 携带
        self.phi = phi
        return torch.stack(mods, dim=1), phi

    def field_energy(self) -> torch.Tensor:
        """当前场能量（范数）——meso 尺度的可观测标量。"""
        return self.phi.norm(dim=-1).mean()

    def reset(self, B: int, device, dtype) -> None:
        """重置常驻场为全 0（序列 / 段开始；生成 init_stream 调用）。"""
        self.phi = torch.zeros(B, self.dim, device=device, dtype=dtype)

    def step(self, h_t: torch.Tensor,
             dt: Optional[torch.Tensor] = None,
             scale: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单 token 流式步进（生成期用），与 ``forward_sequence`` 的逐位置内循环
        逐位一致（段内 phi 连续、无 detach）。

        ``h_t``: (B, D) 单 token 的 ln_f 隐状态。返回 (调制后隐状态 (B,D), 末位场 (B,D))。
        """
        phi = self._ensure(h_t.shape[0], h_t.device, h_t.dtype)
        dt_v = self.dt if dt is None else dt
        scale_v = self.scale if scale is None else scale
        spike = self.spike(h_t)                                   # 离散 → 连续
        # 连续演化（Euler，耗散）：与 forward_sequence 同式
        phi = (1.0 - dt_v * self.leak) * phi \
            + dt_v * torch.tanh(self.W(phi)) + spike
        g = torch.sigmoid(self.gain(phi)) * scale_v                # 连续 → 离散门
        h_mod = h_t + g * phi
        # 跨 token 常驻（O(1)）；流式在 no_grad 下运行，无需段边界 detach。
        self.phi = phi
        return h_mod, phi


# ---------------------------------------------------------------------------
# 支柱 2：尺度不可分 —— micro / meso / macro 双向耦合（heterarchy）
# ---------------------------------------------------------------------------
class MultiscaleCoupling(nn.Module):
    """把原本分离的 micro/meso/macro 三尺度用**可训练权重**双向连结。

    具体耦合（论文："lower scales endogenously generate higher scales; higher
    scales simultaneously constrain lower scales through continuous
    co-determination"）：

    * macro → meso：``curriculum_frac``（训练课程进度，0→1）调制场的连续演化步长 dt
      —— 训练编排（高尺度）连续地约束连续动力（中尺度）；
    * meso → micro：上一 chunk 的场能量调制 token 调制门增益 scale
      —— 连续态（中尺度）约束离散处理（微尺度）；
    * micro → meso：token 脉冲注入场（已在 ``ContinuousField`` 内）
      —— 离散事件（微尺度）内生地生成连续场（中尺度）；
    * meso → macro：场「就绪度」（场能量）作为指标暴露给训练循环（中尺度反馈高尺度）。

    所有耦合系数均可训练，因此「算法即底物」：尺度纠缠进权重，而非数字系统那样
    把算法与物理实现干净分层。
    """

    def __init__(self, dim: int, base_dt: float = 0.5) -> None:
        super().__init__()
        self.base_dt = float(base_dt)
        # macro → meso：课程进度 → 场 dt（偏置初始化为 base_dt，保证默认行为一致）
        self.macro_to_dt = nn.Linear(1, 1, bias=True)
        # meso → micro：场能量 → token 门增益（偏置初始化为 1.0）
        self.meso_to_gain = nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.macro_to_dt.weight)
            self.macro_to_dt.bias.fill_(base_dt)
            nn.init.zeros_(self.meso_to_gain.weight)
            self.meso_to_gain.bias.fill_(1.0)

    def dt(self, curriculum_frac: float) -> torch.Tensor:
        """macro → meso：由课程进度得到场演化步长（标量张量，可训）。"""
        f = torch.tensor([[float(curriculum_frac)]], dtype=torch.float32)
        return torch.clamp(self.macro_to_dt(f), min=0.01, max=2.0)

    def gain_scale(self, field_energy: torch.Tensor) -> torch.Tensor:
        """meso → micro：由场能量得到 token 门增益缩放（标量张量，可训）。"""
        e = field_energy.reshape(-1, 1)
        return torch.clamp(self.meso_to_gain(e), min=0.0, max=4.0)


# ---------------------------------------------------------------------------
# 支柱 3：代谢约束 —— 能量预算 + 代谢效率损失（塑形学习/稳定性/信息流）
# ---------------------------------------------------------------------------
class MetabolicMeter:
    """代谢账本：把「能耗」做成可被反向传播塑形的信号。

    * 每个 token 的归一化代谢能耗 = 基座底价 + 连续场动力能耗 +（若 unity）专家激活能耗；
    * 超过 ``budget`` 的部分产生**代谢效率损失** ``w·relu(E - budget)``，引导模型
      在预算内完成预测（"metabolism structures and guides information flow"）；
    * 预算紧张（场能耗越界）时自动收缩 token 门增益 → 稳定性 / 信息流被代谢塑造。
    """

    def __init__(self, budget: float = 1.5, energy_weight: float = 0.01) -> None:
        self.budget = float(budget)
        self.energy_weight = float(energy_weight)
        self.reset()

    def reset(self) -> None:
        self.total_energy = 0.0
        self.field_energy = 0.0
        self.waste = 0.0
        self.last_energy = 0.0

    def energy(self, dim: int, field_energy: torch.Tensor,
               expert_frac: float = 0.0) -> torch.Tensor:
        """可微的归一化每 token 代谢能耗（标量）。"""
        field_cost = field_energy / max(dim, 1)          # 连续动力能耗
        base_cost = torch.tensor(1.0)                     # 基座底价（归一化）
        return base_cost + field_cost + float(expert_frac)

    def efficiency_loss(self, energy: torch.Tensor) -> torch.Tensor:
        """代谢效率损失：超额能耗受罚。"""
        return self.energy_weight * F.relu(energy - self.budget)

    def record(self, energy: float, field_energy: float) -> None:
        self.last_energy = float(energy)
        self.total_energy += float(energy)
        self.field_energy = float(field_energy)
        self.waste = max(self.waste, float(energy) - self.budget)

    def as_dict(self) -> Dict[str, float]:
        return {
            "budget": self.budget,
            "energy_weight": self.energy_weight,
            "last_energy": round(self.last_energy, 4),
            "total_energy": round(self.total_energy, 4),
            "field_energy": round(self.field_energy, 4),
            "waste": round(self.waste, 4),
        }


# ---------------------------------------------------------------------------
# 集成器：把三支柱接入训练前向（nn.Module，可 DDP 包裹、进优化器）
# ---------------------------------------------------------------------------
class BioEnhancer(nn.Module):
    """生物计算主义训练增强。

    包裹基座（或 SimplUnity 模型），在前向里：

    1. 取基座 / Unity 的 ln_f 隐状态 ``h``；
    2. macro 尺度（``curriculum_frac``，由 train.py 每步注入）经耦合器得到场 dt；
    3. 上一 chunk 场能量经耦合器得到 token 门增益缩放；
    4. ``ContinuousField`` 跑混合计算，得到调制隐状态 ``h2``；
    5. 若包裹 Unity，则先由 meso/macro 调节路由温度（meso/macro → micro 耦合）；
    6. 头映射 ``h2 → logits``；CE + 代谢效率损失（+ Unity 的 lb/z 辅助项）；
    7. ``MetabolicMeter`` 记账。

    作为 ``nn.Module``，``parameters()`` 含基座（可训）+ 场/耦合器（可训）；
    包裹 Unity 时还含路由/专家（可训），基座在 Unity 内冻结。
    """

    def __init__(self, model: Optional[nn.Module], cfg=None, unity=None,
                 energy_weight: float = 0.01, metabolic_budget: float = 1.0,
                 field_dt: float = 0.5, field_scale: float = 1.0) -> None:
        super().__init__()
        self.unity = unity
        if unity is not None:
            # 包裹 Unity：基座在 UnityModel 内已冻结，为避免重复登记基座参数，
            # 这里不把 base 作为子模块；前向只走 self.unity。
            self.model = None
            self.head = unity.base.head
            self.dim = _infer_dim(unity.base)
        else:
            if model is None:
                raise ValueError("BioEnhancer 在 unity=None 时必须提供 model")
            self.model = model
            self.head = model.head
            self.dim = _infer_dim(model)
        self.cfg = cfg
        # 三支柱
        self.field = ContinuousField(self.dim, dt=field_dt, scale=field_scale)
        self.coupling = MultiscaleCoupling(self.dim, base_dt=field_dt)
        self.meter = MetabolicMeter(budget=metabolic_budget,
                                    energy_weight=energy_weight)
        # macro 尺度输入（train.py 每步写入；默认 0 = 训练早期）
        self.curriculum_frac = 0.0
        # 场能量 EMA（跨 step）——用于 meso→macro 就绪度反馈
        self._energy_ema: Optional[float] = None

    # -- 尺度耦合辅助 -------------------------------------------------------
    def _route_temperature(self, field_energy: torch.Tensor) -> float:
        """meso/macro → micro：场能量与课程进度共同决定路由温度。

        场越活跃、课程越推进 → 路由越锐化（更像「专用专家」），体现中/高尺度
        连续地约束微尺度离散路由。默认（bio 关、frac=0、场≈0）温度≈1.0 → 零回归。
        """
        e = float(field_energy) / max(self.dim, 1)
        temp = 1.0 + 0.5 * e - float(self.curriculum_frac)
        return float(min(max(temp, 0.3), 3.0))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                return_aux: bool = False):
        # macro → meso：课程进度决定场演化步长
        dt = self.coupling.dt(self.curriculum_frac)
        # meso → micro：上一 chunk 场能量决定 token 门增益缩放
        prev_field_energy = self.field.field_energy().detach()
        gain_scale = self.coupling.gain_scale(prev_field_energy)
        # 代谢感知：预算紧张时收缩增益（稳定性被代谢塑造）
        field_norm = float(prev_field_energy) / max(self.dim, 1)
        if field_norm > self.meter.budget:
            gain_scale = gain_scale * (self.meter.budget / max(field_norm, 1e-6))

        # 取隐状态：包裹 Unity 时拿「专家调制后」隐状态 + 路由 aux；
        # 否则拿基座 ln_f 隐状态。
        if self.unity is not None:
            # 先由 meso/macro 调节路由温度（meso/macro → micro 耦合）
            self.unity.router.temperature = self._route_temperature(prev_field_energy)
            out = self.unity(idx, targets, return_hidden=True,
                            return_expert_hidden=True)
            if targets is not None:
                _, h, aux = out
            else:
                _, h = out
        else:
            out = self.model(idx, return_hidden=True)
            _, h = out

        # 支柱 1 + 2：混合计算 + 尺度耦合（动态 dt / scale）
        h2, phi = self.field.forward_sequence(h, dt=dt, scale=self.field.scale * gain_scale)
        field_energy = phi.norm(dim=-1).mean()

        logits = self.head(h2)

        if targets is None:
            if return_aux:
                aux = getattr(self, "_last_aux", {"lb": torch.tensor(0.0),
                                                 "z": torch.tensor(0.0)})
                return logits, aux
            return logits

        # 支柱 3：代谢能耗 + 效率损失
        expert_frac = 0.0
        aux_loss = torch.tensor(0.0)
        if self.unity is not None:
            expert_frac = self.unity.router.top_k / max(self.unity.router.num_experts, 1)
            aux_loss = (self.unity.cfg.load_balance_loss * aux["lb"]
                        + self.unity.cfg.router_z_loss * aux["z"])
        energy = self.meter.energy(self.dim, field_energy, expert_frac=expert_frac)
        meta_loss = self.meter.efficiency_loss(energy)
        self.meter.record(float(energy.detach()), float(field_energy.detach()))
        self._last_aux = aux if self.unity is not None else {"lb": torch.tensor(0.0),
                                                            "z": torch.tensor(0.0)}

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1), ignore_index=-100)
        loss = ce + meta_loss + aux_loss
        if return_aux:
            return logits, loss, self._last_aux
        return logits, loss

    # -- meso → macro 反馈：连续场就绪度 ----------------------------------
    def readiness(self) -> float:
        """连续场「就绪度」∈ [0,1]（1 = 当前长度已被充分吸收）。

        这是论文「尺度不可分」里 **meso → macro** 的那条边：中尺度的连续场状态
        反馈给高尺度的训练编排（序列长度课程）。它回答的问题是——*模型现在能
        承受更长的上下文了吗？*

        * 场能量超过代谢天花板（``budget × D``）→ 0.0：系统代谢失衡（场在暴涨），
          说明当前长度已超出模型的吸收能力 ⇒ 课程应**推迟**拉长；
        * 场稳定（瞬时能量 ≈ EMA）且代谢余量充足 → 接近 1.0 ⇒ 可以推进；
        * 场仍在快速增长 → 稳定度低 → 就绪度低。

        注意：这是**有状态**探针（内部维护场能量 EMA），训练循环每步调用一次。
        """
        e = float(self.field.field_energy())
        ema = self._energy_ema
        if ema is None:
            self._energy_ema = e
            ema = e
        else:
            self._energy_ema = 0.9 * ema + 0.1 * e
        ceiling = max(self.meter.budget * self.dim, 1e-6)
        if e > ceiling:
            return 0.0
        # 稳定度：|e - ema| 相对 EMA 越小越稳（1.0 = 完全稳定）
        dev = abs(e - ema) / (ema + 1e-6)
        stability = 1.0 / (1.0 + dev)
        # 代谢余量：场能量离天花板越远，余量越大
        margin = min(max((ceiling - e) / ceiling, 0.0), 1.0)
        return float(min(1.0, stability * (0.5 + 0.5 * margin)))

    # -- 优化器 / 账本接口（与其他增强对齐）--------------------------------
    def param_filter(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def num_params_learned(self) -> int:
        return sum(p.numel() for p in self.param_filter())

    def budget_dict(self) -> Dict[str, object]:
        d = self.meter.as_dict()
        d["num_experts_unity"] = (self.unity.router.num_experts
                                  if self.unity is not None else 0)
        d["learned_params"] = self.num_params_learned()
        return d

    def as_dict(self) -> Dict[str, object]:
        return self.budget_dict()

    # -- 流式 / 生成（让混合计算在生成期也生效）----------------------------
    @property
    def supports_streaming(self) -> bool:
        return True

    def _base(self):
        """底层被包裹的模型（Unity 时是其基座）。"""
        return self.unity.base if self.unity is not None else self.model

    def _device(self):
        base = self._base()
        emb = getattr(base, "tok_emb", None)
        if emb is not None:
            return emb.weight.device
        return next(self.parameters()).device

    def _dtype(self):
        """基座的 dtype —— 流式状态必须与基座递推状态同 dtype（bf16 训练时）。"""
        base = self._base()
        emb = getattr(base, "tok_emb", None)
        if emb is not None:
            return emb.weight.dtype
        return next(self.parameters()).dtype

    def init_stream(self, batch, device=None, dtype=None) -> "BioStreamState":
        """初始化流式状态：底层基座 / Unity 的 O(1) 递推状态 + 复位连续场。

        耦合 ``dt`` / ``gain`` 在「段」（chunk / 序列）粒度固定 —— 与训练前向
        一致（训练每 chunk 只算一次），由段首的课程进度与上一 carry 场能量决定。
        """
        device = device or self._device()
        dtype = dtype or self._dtype()
        if self.unity is not None:
            base_state = self.unity.init_stream(batch, device=device, dtype=dtype)
        else:
            base_state = self.model.init_stream(batch, device=device, dtype=dtype)
        self.field.reset(batch, device, dtype)
        dt = self.coupling.dt(self.curriculum_frac)
        prev_fe = self.field.field_energy().detach()
        gain_scale = self.coupling.gain_scale(prev_fe)
        field_norm = float(prev_fe) / max(self.dim, 1)
        if field_norm > self.meter.budget:
            gain_scale = gain_scale * (self.meter.budget / max(field_norm, 1e-6))
        self._dt_cache = dt
        self._gain_cache = gain_scale
        return BioStreamState(base=base_state, length=0)

    def _base_stream_hidden(self, token: torch.Tensor, state):
        """基座 O(1) 流式取 ln_f 隐状态（pre-head），与 SimplCoreModel.stream_step 同序。"""
        model = self.model
        if token.dim() == 1:
            token = token.unsqueeze(1)
        x = model.tok_emb(token)                                # (B,1,D)
        mem = state.memory
        for blk, st in zip(model.blocks, state.layers):
            mx = blk.mixer
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, _ = mx.step(x, st)
            x = blk.ffn(x)
        state.length += 1
        return model.ln_f(x), state                              # (B,1,D)

    @torch.no_grad()
    def stream_step(self, token: torch.Tensor, state: "BioStreamState"
                    ) -> Tuple[torch.Tensor, "BioStreamState"]:
        """消费一个 token，返回下一 token 的 logits（混合计算已调制）。

        与训练前向同语义：底层的 ln_f 隐状态（Unity 时为专家调制后隐状态）→
        连续场单步调制 → 头。段首已缓存耦合 dt/gain，段内恒定（与训练一致）。
        """
        if self.unity is not None:
            h, st = self.unity.stream_step_hidden(token, state.base)
        else:
            h, st = self._base_stream_hidden(token, state.base)
        state.base = st
        state.length += 1
        dt = getattr(self, "_dt_cache", self.coupling.dt(self.curriculum_frac))
        gain_scale = getattr(self, "_gain_cache", torch.tensor(1.0))
        scale = self.field.scale * gain_scale
        h2, _ = self.field.step(h[:, -1, :], dt=dt, scale=scale)
        logits = self.head(h2.unsqueeze(1))[:, -1, :]
        return logits, state

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, use_state: bool = True,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """自回归采样；O(1) 流式路径也跑混合计算。

        与 ``SimplCoreModel.generate`` 同签名，但经 ``BioEnhancer`` 的
        ``stream_step`` —— 连续场在每个生成 token 上持续演化（生成期也跑混合计算）。
        """
        was_training = self.training
        self.eval()
        B = idx.shape[0]
        if not use_state:
            # 全序列回退（仍过 bio 连续场）；每段前向前复位场，避免跨步污染。
            out = idx
            for _ in range(max_new_tokens):
                self.field.reset(B, idx.device, self._dtype())
                logits = self.forward(out)
                logits = logits[:, -1, :]
                nxt = _sample_logits(logits, temperature, top_k, top_p)
                out = torch.cat([out, nxt], dim=1)
                if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                    break
            if was_training:
                self.train()
            return out

        state = self.init_stream(B, device=idx.device, dtype=self._dtype())
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = _sample_logits(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def generate_text(self, text: str, codec, max_new_tokens: int = 32,
                      temperature: float = 1.0, top_k: Optional[int] = None,
                      top_p: Optional[float] = None,
                      eos_token_id: Optional[int] = None) -> str:
        """无词表模式文本生成（同 ``SimplCoreModel.generate_text`` 语义）。"""
        from .model.tokenization import Tokenizer
        assert isinstance(codec, Tokenizer)
        idx = torch.tensor([codec.encode(text)], device=self._device())
        out = self.generate(idx, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k,
                            top_p=top_p, eos_token_id=eos_token_id)
        gen_ids = out[0, idx.shape[1]:].tolist()
        return codec.decode(gen_ids)

    # -- 流式状态持久化（与 SimplCoreModel 对齐）----------------------------
    def checkpoint_stream(self, state: "BioStreamState") -> dict:
        base = self.model if self.unity is None else self.unity.base
        return {"length": state.length,
                "base": base.checkpoint_stream(state.base),
                "phi": self.field.phi.detach().clone()}

    def restore_stream(self, ckpt: dict) -> "BioStreamState":
        base = self.model if self.unity is None else self.unity.base
        base_state = base.restore_stream(ckpt["base"])
        self.field.phi = ckpt["phi"].clone()
        return BioStreamState(base=base_state, length=ckpt.get("length", 0))
