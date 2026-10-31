# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""训练增强集成 —— 把六个「可运行但未接入训练」的控制器真正接入训练闭环。

本模块把 ``accel`` / ``nonlinear`` / ``reasoning`` / ``envlearn`` / ``smm`` /
``dream`` 从「孤立 demo」变成训练流水线里的**可选、默认关闭**的真实组件。
每一个增强都是 opt-in：不开任何开关时，训练行为与原来逐位一致（零风险）。

各增强的训练角色（从「死代码」变成「流水线一环」）
----------------------------------------------
* nonlinear (PonderEnhancer)：用 ``LatentRefiner.forward`` 做训练前向，
  CE + ponder 正则一起反传 —— 思考算力成为可训练信号，不再是摆设。
  它是 ``nn.Module``（内含基座 + thinker），可被 DDP 包裹、进优化器。
* envlearn (EnvLearnEnhancer)：冻结 ``SimplCoreModel`` 基座，只训
  ``LearnedAdapter``（包成 ``ForwardAdapter``）；``WeightBudget`` 守卫 3B 上限。
  同样是 ``nn.Module``，参数只含适配器 → 优化器只更新它们。
* accel (AccelEval)：评估期跑 ``AdaptiveDepth`` 自适应深度，记录
  ``ComputeLedger``（加速比/平均深度）—— 作为 eval-hook，不改动主 val_loss。
* smm (MemoryEnhancer)：评估期把采样窗口解码成文本、写入 ``SimplModelMemory``
  情景记忆库（训练/交互期经验采集的一环）；生成期可 ``answer`` 注入记忆。
* dream (DreamConsolidator)：训练后跑 ``ReflectionLoop`` 自进化巩固
  （no-regression gate 逐位回滚）—— 作为 post-hook。
* reasoning (ReasoningEnhancer)：评估期跑 ``IterativeRefiner`` 迭代式思考 —— eval-hook。

诚实边界
--------
* 机制已接入、可被测试、默认关闭；但「训出质量优势」仍需 MI250 战役验证
  （见 ``campaign.py``）。
* 所有增强都不引入注意力，与全架构纯洁性约束一致（``purity.py``）。
* 本模块只做编排；张量运算全部委托给各控制器自身的实现，不另造轮子。
* ``smm``/``dream``/``reasoning`` 需要编解码器（codec）。未显式传入时，
  自动用 ``CharTokenizer(limit=vocab_size)`` 兜底，保证接入点永远可跑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import SimplCoreConfig

__all__ = [
    "ENHANCEMENTS",
    "ENHANCEMENT_SPECS",
    "EnhancementSpec",
    "resolve_enhancements",
    "PonderEnhancer",
    "EnvLearnEnhancer",
    "AccelEval",
    "MemoryEnhancer",
    "DreamConsolidator",
    "ReasoningEnhancer",
    "UnityEnhancer",
    "BioEnhancer",
    "build_enhanced_model",
    "EnhancedModel",
]


# ---------------------------------------------------------------------------
# 1. nonlinear —— latent 思考进入训练前向（CE + ponder 正则）
# ---------------------------------------------------------------------------
class PonderEnhancer(nn.Module):
    """把 ``LatentRefiner`` 接到训练前向：思考算力成为可训练信号。

    它是 ``nn.Module``（内含基座 ``self.model`` 与 ``self.thinker``），因此
    基座 + 思考单元的参数**都**进入优化器、都能被 DDP 包裹、都能被梯度裁剪。

    ``max_iters=0`` 时退化为原模型（逐位一致），便于对照。
    """

    def __init__(self, model: nn.Module, max_iters: int = 2,
                 halt_threshold: float = 0.5, ponder_weight: float = 0.01) -> None:
        super().__init__()
        from ..model.nonlinear import LatentRefiner
        self.model = model
        self.thinker = LatentRefiner(
            model, max_iters=max_iters, halt_threshold=halt_threshold,
            ponder_weight=ponder_weight, enabled=max_iters > 0)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                memory=None):
        return self.thinker(idx, targets, memory=memory)

    @property
    def ledger(self):
        return self.thinker.ledger


# ---------------------------------------------------------------------------
# 2. envlearn —— 冻结基座 + 只训适配器（WeightBudget 守卫）
# ---------------------------------------------------------------------------
class EnvLearnEnhancer(nn.Module):
    """冻结基座、只训适配器的训练封装（显存只随可学权重增长）。

    * 基座（``SimplCoreModel``）在 ``ForwardAdapter`` 内 ``requires_grad_(False)``；
    * 新增 ``LearnedAdapter``，参数计入 ``WeightBudget`` 的「可学」部分；
    * 作为 ``nn.Module``，``parameters()`` 含基座（冻结）+ 适配器（可学），
      优化器经 ``build_param_groups`` 自动只更新适配器矩阵。
    """

    def __init__(self, model: nn.Module, dim: int, num_blocks: int = 2,
                 max_learned: int = 2_000_000_000, base_params: int = 1_000_000_000,
                 codec=None) -> None:
        super().__init__()
        from .envlearn import LearnedAdapter, WeightBudget, ForwardAdapter
        self.base = model
        self.adapter = LearnedAdapter(dim, num_blocks=num_blocks)
        self.budget = WeightBudget(base_params=base_params, max_learned=max_learned)
        cost = int(self.adapter.num_params())
        if not self.budget.can_allocate(cost):
            raise ValueError(f"适配器 {cost} 参数超预算上限 {max_learned}")
        self.budget.allocate(cost)
        self.fa = ForwardAdapter(model, self.adapter, codec=codec)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        return self.fa(idx, targets)

    def budget_dict(self) -> Dict[str, int]:
        return self.budget.as_dict()


# ---------------------------------------------------------------------------
# 3. accel —— 评估期自适应深度 + 算力账本（eval-hook）
# ---------------------------------------------------------------------------
class AccelEval:
    """评估期用 ``AdaptiveDepth`` 跑自适应深度，记录 ``ComputeLedger``。

    ``skip_threshold=-inf`` 时应与原模型逐位一致（speedup==1）；
    提高阈值 → 易 token 走浅路径 → speedup>1。让加速比成为训练期可观测指标。
    """

    def __init__(self, model: nn.Module, skip_threshold: float = float("-inf"),
                 min_depth: int = 1) -> None:
        from ..arch.accel import AdaptiveDepth
        self.model = model
        self.accel = AdaptiveDepth(
            model, skip_threshold=skip_threshold, min_depth=min_depth)

    @torch.no_grad()
    def eval_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> float:
        state = self.accel.init_stream(idx.shape[0], device=idx.device,
                                       dtype=self.model.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.accel.stream_step(idx[:, t], state)
        loss = F.cross_entropy(logits, targets[:, -1])
        return float(loss)

    @torch.no_grad()
    def eval_dataset(self, loader, device, amp_dtype, max_batches: int = 8) -> Dict[str, float]:
        """对 val 集跑自适应深度评估，返回末位 CE（代理）与算力账本。"""
        self.accel.ledger.reset()
        total = 0.0
        n = 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            total += self.eval_loss(x, y)
            n += 1
        led = self.accel.ledger
        depth_mean = (led.blocks_run / max(led.tokens, 1)) if led.tokens else float("nan")
        return {
            "loss_proxy": total / max(n, 1),
            "tokens": led.tokens,
            "depth_mean": depth_mean,
            "speedup": led.speedup,
            "frac_compute": led.frac_compute,
        }

    @property
    def ledger(self):
        return self.accel.ledger


# ---------------------------------------------------------------------------
# 4. smm —— 评估期采集经验写入压缩记忆（eval-hook）
# ---------------------------------------------------------------------------
class MemoryEnhancer:
    """把 ``SimplModelMemory`` 接到训练闭环：评估期记录经验、注入检索记忆。

    它的真实训练角色是**经验采集**：评估期把采样窗口解码成文本、作为
    ``(prompt, response)`` 写入情景记忆库；生成期通过 ``answer`` 把检索到的
    压缩记忆注入残差流（需模型 ``use_memory``）。它不侵入 LM 前向损失。
    """

    def __init__(self, model: nn.Module, codec, config=None) -> None:
        from ..memory.smm import SimplModelMemory
        self.model = model
        self.codec = codec
        self.memory = SimplModelMemory(model, codec, config=config)

    def experience(self, prompt: str, response: str, **kw):
        return self.memory.experience(prompt, response, **kw)

    def recall(self, query: str):
        return self.memory.recall(query)

    @torch.no_grad()
    def answer(self, idx: torch.Tensor, max_new_tokens: int = 32, query: str = "",
               **kw):
        return self.memory.answer(idx, max_new_tokens=max_new_tokens,
                                  inject=bool(query), query=query, **kw)

    def num_experiences(self) -> int:
        led = getattr(self.memory, "ledger", None)
        return int(getattr(led, "experiences", 0)) if led is not None else 0

    @property
    def ledger(self):
        return getattr(self.memory, "ledger", None)


# ---------------------------------------------------------------------------
# 5. dream —— 训练后自进化巩固（no-regression gate，post-hook）
# ---------------------------------------------------------------------------
class DreamConsolidator:
    """训练后跑 ``ReflectionLoop`` 巩固：更好→提交，退化超容差→逐位回滚。

    这是把 ReflectionLoop 从「孤立 demo」接入训练闭环的**后处理阶段**：训练完成后
    把经验喂入 ``ReflectionLoop`` 的回放池并 ``run()``，复用模型自带快照/回滚。
    """

    def __init__(self, model: nn.Module, codec=None, tol: float = 0.02) -> None:
        from ..memory.dream import ReflectionLoop
        self.model = model
        self.codec = codec
        self.dream = ReflectionLoop(model, codec=codec, config=None)
        self._tol = tol

    def consolidate(self, experiences=None, val_loader=None,
                    max_rounds: int = 1) -> Dict[str, object]:
        from ..memory.dream import Trajectory
        if experiences:
            for e in experiences:
                if isinstance(e, Trajectory):
                    self.dream.buffer.add(e)
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    self.dream.buffer.add(Trajectory(prompt=str(e[0]),
                                                    response=str(e[1])))
        if val_loader is not None and self.codec is not None:
            for i, (x, y) in enumerate(val_loader):
                if i >= 4:
                    break
                prompt = self.codec.decode(x[0].tolist())
                response = self.codec.decode(y[0].tolist())
                self.dream.buffer.add(Trajectory(prompt=prompt, response=response))
        reports = self.dream.run(max_rounds=max_rounds)
        rollbacks = sum(1 for r in reports if getattr(r, "rolled_back", False))
        return {"stages": len(reports), "rollbacks": rollbacks,
                "reports": [getattr(r, "as_dict", lambda: {})() for r in reports]}


# ---------------------------------------------------------------------------
# 6. reasoning —— 评估期迭代式深度思考（eval-hook）
# ---------------------------------------------------------------------------
class ReasoningEnhancer:
    """评估期用 ``IterativeRefiner`` 跑迭代式「提示→思考→答案」循环。"""

    def __init__(self, model: nn.Module, codec, max_steps: int = 3,
                 policy: str = "stabilize", tool: Optional[Callable] = None) -> None:
        from ..dialogue.reasoning import IterativeRefiner
        self.model = model
        self.reasoner = IterativeRefiner(model, codec, max_steps=max_steps,
                                     policy=policy, tool=tool)

    @torch.no_grad()
    def reason(self, prompt: str):
        return self.reasoner.reason(prompt)


# ---------------------------------------------------------------------------
# 7. unity —— 冻结基座 + 路由门 + 多个小专家（MoE 效果，训练数据按专家分域）
# ---------------------------------------------------------------------------
class UnityEnhancer(nn.Module):
    """把 ``UnityModel``（冻结 SGC 基座 + 路由 + 多个小专家）接到训练闭环。

    * 基座在 ``UnityModel`` 内冻结；梯度只流向路由门 + 专家；
    * 作为 ``nn.Module``，``parameters()`` 含冻结基座（被 ``build_param_groups`` 跳过）
      与可学路由/专家（进优化器）；
    * ``spawn_expert`` 提供「模型在真实环境学到新域时后台并行训练新专家」的自进化入口。

    与 ``envlearn`` 是**替代型**适配器策略（两者都替换 ``em.model``），同时开启时
    ``unity`` 在后、以 unity 为准。
    """

    def __init__(self, model: nn.Module, num_experts: int = 4,
                 top_k: int = 2) -> None:
        super().__init__()
        from ...unity.config import UnityConfig
        from ...unity.model import UnityModel
        ucfg = UnityConfig(num_experts=num_experts, top_k=top_k)
        self.unity = UnityModel(model, ucfg)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        return self.unity(idx, targets)

    def num_params_learned(self) -> int:
        return self.unity.num_params_learned()

    def spawn_expert(self, *args, **kwargs):
        return self.unity.spawn_expert(*args, **kwargs)

    def budget_dict(self) -> Dict[str, object]:
        return self.unity.as_dict()


def warmup_unity_domains(unity_model, domain_loaders, steps: int = 200,
                         lr: float = 1e-2, device=None, amp_dtype=None) -> None:
    """按域隔离 warm-up：第 ``i`` 个专家只在第 ``i`` 个域数据上训练。

    直接落实用户硬约束「训练数据必须不同」——每个专家喂不同域 loader，
    路由强制锁到该专家（``train_expert(domain=i)``），其余专家与基座冻结。
    域数多于初始专家时自动 ``spawn_expert`` 增长（受预算守卫）。
    """
    for i, loader in enumerate(domain_loaders):
        if i >= len(unity_model.experts):
            unity_model.spawn_expert()
        opt = torch.optim.AdamW(unity_model.experts[i].parameters(), lr=lr)
        unity_model.train_expert(i, loader, opt, steps=steps,
                                device=device, amp_dtype=amp_dtype)


# ---------------------------------------------------------------------------
# 注册表 + 工厂：把增强接到 train.py 的训练闭环
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnhancementSpec:
    """一种训练增强的静态契约。

    ``kind`` 决定它在训练闭环里的挂载位置，而不是由调用方靠 if/else 猜：

    * ``wrapper`` —— 替换训练前向（``nn.Module``），进入优化器与 DDP；
    * ``eval`` —— 评估期 hook，不改变主 ``val_loss``；
    * ``post`` —— 主循环结束后的 post-hook。

    ``supersedes`` 声明"同位置替代"关系：多个 wrapper 同时开启时，后者覆盖
    前者是**显式语义**，被覆盖项会进入 :attr:`EnhancedModel.superseded` 并由
    ``train.py`` 打印，不再静默生效。``bio`` 是唯一可以**消费**另一个 wrapper
    （``unity``）的组合，因此单独用 ``consumes`` 标注。
    """

    name: str
    kind: str
    description: str
    requires: Tuple[str, ...] = ()
    supersedes: Tuple[str, ...] = ()
    consumes: Tuple[str, ...] = ()


ENHANCEMENT_SPECS: Dict[str, EnhancementSpec] = {
    "ponder": EnhancementSpec(
        "ponder", "wrapper",
        "nonlinear —— latent 思考进入训练前向（CE + ponder 正则），nn.Module"),
    "envlearn": EnhancementSpec(
        "envlearn", "wrapper",
        "envlearn —— 冻结基座 + 只训适配器（WeightBudget 守卫），nn.Module",
        supersedes=("ponder",)),
    "unity": EnhancementSpec(
        "unity", "wrapper",
        "unity —— 冻结基座 + 路由门 + 多个小专家（MoE 效果，训练数据按专家分域），nn.Module",
        supersedes=("ponder", "envlearn")),
    "bio": EnhancementSpec(
        "bio", "wrapper",
        "bio —— 生物计算主义：混合计算(连续场×离散脉冲)+尺度不可分"
        "(micro/meso/macro 耦合)+代谢约束(能量预算)，nn.Module",
        supersedes=("ponder", "envlearn"),
        consumes=("unity",)),
    "accel": EnhancementSpec(
        "accel", "eval",
        "accel —— 评估期自适应深度 + 算力账本（eval-hook）"),
    "smm": EnhancementSpec(
        "smm", "eval",
        "smm —— 评估期采集经验写入压缩记忆（eval-hook）"),
    "dream": EnhancementSpec(
        "dream", "post",
        "dream —— 训练后自进化巩固（no-regression gate，post-hook）"),
    "reasoning": EnhancementSpec(
        "reasoning", "eval",
        "reasoning —— 评估期迭代式深度思考（eval-hook）"),
}

#: 构建顺序（wrapper 必须按此顺序包裹，eval/post 只影响挂载顺序）。
_BUILD_ORDER: Tuple[str, ...] = (
    "ponder", "envlearn", "unity", "bio", "accel", "smm", "dream", "reasoning")


def resolve_enhancements(names: Sequence[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """把用户输入解析成：生效列表、被替代项、显式组合项。

    返回 ``(active, superseded, consumed)``：

    * ``active`` 按构建顺序排列；
    * ``superseded`` 是 ``{被替代项: 替代者}``，供调用方打印/审计；
    * ``consumed`` 是"被另一个 wrapper 消费但仍保留"的项（当前仅
      ``bio`` 消费 ``unity``）。

    未知名称**直接报错**：旧实现静默忽略拼错的增强名，用户以为开了，
    实际训练行为逐位不变 —— 这是最难发现的一类配置缺陷。
    """
    requested: List[str] = []
    for raw in names or []:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in ENHANCEMENT_SPECS:
            raise ValueError(
                f"unknown enhancement {name!r}; valid: {sorted(ENHANCEMENT_SPECS)}")
        if name not in requested:
            requested.append(name)

    # requires 自动补齐（当前没有条目使用，保留为契约扩展点）。
    for name in list(requested):
        for req in ENHANCEMENT_SPECS[name].requires:
            if req not in requested:
                requested.append(req)

    superseded: Dict[str, str] = {}
    consumed: List[str] = []
    active: List[str] = []
    for name in _BUILD_ORDER:
        if name not in requested:
            continue
        spec = ENHANCEMENT_SPECS[name]
        # 先看"消费"关系：bio 保留 unity，而不是把 unity 标成被替代。
        if any(c in active for c in spec.consumes):
            for c in spec.consumes:
                if c in active:
                    consumed.append(c)
            active.append(name)
            continue
        for prev in list(active):
            if prev in spec.supersedes:
                superseded[prev] = name
                active.remove(prev)
        active.append(name)

    active = [n for n in _BUILD_ORDER if n in active]
    consumed = [n for n in _BUILD_ORDER if n in consumed]
    return active, superseded, consumed


#: 兼容旧接口：``ENHANCEMENTS[name] -> description``。
ENHANCEMENTS = {k: v.description for k, v in ENHANCEMENT_SPECS.items()}


@dataclass
class EnhancedModel:
    """train.py 拿到的增强结果。

    Attributes:
        model: 实际参与前向/反向、被 DDP 包裹、被存盘的模块
            （ponder/envlearn 时为对应封装；否则为基座）。
        train_forward: ``(idx, targets) -> (logits, loss)``（= model.forward）。
        eval_loss: 可选的主评估覆盖 ``(idx, targets) -> float``；None 用默认评估。
        eval_hooks: 评估期调用的 ``(em, val_loader, device, amp) -> dict`` 列表。
        post_hooks: 主循环后调用的 ``(em, val_loader) -> dict`` 列表。
        extras: 其余增强对象（accel/smm/dream/reasoning），供调试/生成调用。
    """

    model: nn.Module
    train_forward: Callable
    eval_loss: Optional[Callable] = None
    eval_hooks: List[Callable] = field(default_factory=list)
    post_hooks: List[Callable] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)
    #: 实际生效的增强（按构建顺序）。
    active: List[str] = field(default_factory=list)
    #: ``{被替代项: 替代者}`` —— 不再静默覆盖，供 train.py 打印与审计。
    superseded: Dict[str, str] = field(default_factory=dict)
    #: 被另一个 wrapper 消费但仍保留的项（当前仅 ``bio`` 消费 ``unity``）。
    consumed: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, object]:
        """机器可读的增强挂载状态（训练日志/审计用）。"""
        return {
            "active": list(self.active),
            "superseded": dict(self.superseded),
            "consumed": list(self.consumed),
            "model": type(self.model).__name__,
            "eval_hooks": len(self.eval_hooks),
            "post_hooks": len(self.post_hooks),
        }


# ---------------------------------------------------------------------------
# eval / post hook 工厂
# ---------------------------------------------------------------------------
def _make_accel_hook(enh: AccelEval):
    @torch.no_grad()
    def hook(em, val_loader, device, amp_dtype):
        if val_loader is None:
            return {}
        res = enh.eval_dataset(val_loader, device, amp_dtype, max_batches=8)
        return {"accel": res}
    return hook


def _make_smm_hook(enh: MemoryEnhancer, collected: list):
    @torch.no_grad()
    def hook(em, val_loader, device, amp_dtype):
        if val_loader is None or enh.codec is None:
            return {}
        new = 0
        for i, (x, y) in enumerate(val_loader):
            if i >= 4:
                break
            prompt = enh.codec.decode(x[0].tolist())
            response = enh.codec.decode(y[0].tolist())
            enh.experience(prompt, response)
            collected.append((prompt, response))
            new += 1
        return {"smm_experiences_total": enh.num_experiences(), "smm_new": new}
    return hook


def _make_reasoning_hook(enh: ReasoningEnhancer, prompts: Sequence[str]):
    @torch.no_grad()
    def hook(em, val_loader, device, amp_dtype):
        out = enh.reason(prompts[0])
        return {"reasoning_steps": getattr(out, "steps", None),
                "reasoning_answer": getattr(out, "answer", None)}
    return hook


def _make_dream_hook(enh: DreamConsolidator, collected: list):
    def hook(em, val_loader):
        res = enh.consolidate(experiences=list(collected), val_loader=val_loader,
                              max_rounds=1)
        return {"dream": res}
    return hook


@dataclass
class _BuildContext:
    """构建增强时共享的只读上下文（避免每个 builder 重复拼参数）。"""

    base: nn.Module
    cfg: SimplCoreConfig
    codec: object
    eval_prompts: Optional[Sequence[str]]
    unity_experts: int
    unity_topk: int
    bio_energy_weight: float
    bio_metabolic_budget: float
    bio_field_dt: float
    bio_field_scale: float
    collected: list


def _build_ponder(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = PonderEnhancer(ctx.base)
    em.model = enh
    em.train_forward = enh.forward
    em.extras["ponder"] = enh


def _build_envlearn(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = EnvLearnEnhancer(ctx.base, dim=ctx.cfg.dim)
    em.model = enh
    em.train_forward = enh.forward
    em.extras["envlearn"] = enh


def _build_unity(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = UnityEnhancer(ctx.base, num_experts=ctx.unity_experts,
                        top_k=ctx.unity_topk)
    em.model = enh
    em.train_forward = enh.forward
    em.extras["unity"] = enh


def _build_bio(em: EnhancedModel, ctx: _BuildContext) -> None:
    from ..bio_enhance import BioEnhancer
    unity_model = em.extras["unity"].unity if "unity" in em.extras else None
    enh = BioEnhancer(model=(None if unity_model else ctx.base), cfg=ctx.cfg,
                      unity=unity_model,
                      energy_weight=ctx.bio_energy_weight,
                      metabolic_budget=ctx.bio_metabolic_budget,
                      field_dt=ctx.bio_field_dt, field_scale=ctx.bio_field_scale)
    em.model = enh
    em.train_forward = enh.forward
    em.extras["bio"] = enh


def _build_accel(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = AccelEval(ctx.base)
    em.extras["accel"] = enh
    em.eval_hooks.append(_make_accel_hook(enh))


def _build_smm(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = MemoryEnhancer(ctx.base, ctx.codec, config=None)
    em.extras["smm"] = enh
    em.eval_hooks.append(_make_smm_hook(enh, ctx.collected))


def _build_dream(em: EnhancedModel, ctx: _BuildContext) -> None:
    enh = DreamConsolidator(ctx.base, codec=ctx.codec)
    em.extras["dream"] = enh
    em.post_hooks.append(_make_dream_hook(enh, ctx.collected))


def _build_reasoning(em: EnhancedModel, ctx: _BuildContext) -> None:
    prompts = list(ctx.eval_prompts) if ctx.eval_prompts else [
        "你好，请做一道简单的加法：3+4=？"]
    enh = ReasoningEnhancer(ctx.base, ctx.codec, max_steps=3)
    em.extras["reasoning"] = enh
    em.eval_hooks.append(_make_reasoning_hook(enh, prompts))


#: 名称 → builder。新增增强只改 :data:`ENHANCEMENT_SPECS` 与本表，
#: 不再在 ``build_enhanced_model`` 里堆 if/elif。
_BUILDERS: Dict[str, Callable[[EnhancedModel, _BuildContext], None]] = {
    "ponder": _build_ponder,
    "envlearn": _build_envlearn,
    "unity": _build_unity,
    "bio": _build_bio,
    "accel": _build_accel,
    "smm": _build_smm,
    "dream": _build_dream,
    "reasoning": _build_reasoning,
}


def build_enhanced_model(model: nn.Module, cfg: SimplCoreConfig,
                         enhance: Sequence[str],
                         codec=None, eval_prompts: Optional[Sequence[str]] = None,
                         unity_experts: int = 4, unity_topk: int = 2,
                         bio_energy_weight: float = 0.01,
                         bio_metabolic_budget: float = 1.5,
                         bio_field_dt: float = 0.5, bio_field_scale: float = 1.0
                         ) -> EnhancedModel:
    """按 ``enhance`` 列表把控制器接到 ``model`` 上，返回训练闭环所需组件。

    编排由 :data:`ENHANCEMENT_SPECS` + :func:`resolve_enhancements` 驱动：
    未知名称直接报错，同位置替代显式记录到 ``em.superseded``，不再静默
    last-wins。未列在 ``enhance`` 中的控制器一律不动；空列表 → 原模型逐位不变。
    ``smm``/``dream``/``reasoning`` 需要 codec，未传入时自动用
    ``CharTokenizer(limit=vocab_size)`` 兜底。
    """
    active, superseded, consumed = resolve_enhancements(enhance)
    em = EnhancedModel(
        model=model,
        train_forward=lambda idx, tgt: model(idx, tgt),
        eval_loss=None,
        extras={},
        active=active,
        superseded=superseded,
        consumed=consumed,
    )
    needs_codec = bool(set(active) & {"smm", "dream", "reasoning"})
    if needs_codec and codec is None:
        from ..model.tokenization import CharTokenizer
        codec = CharTokenizer(limit=int(cfg.vocab_size))

    ctx = _BuildContext(
        base=model, cfg=cfg, codec=codec, eval_prompts=eval_prompts,
        unity_experts=unity_experts, unity_topk=unity_topk,
        bio_energy_weight=bio_energy_weight,
        bio_metabolic_budget=bio_metabolic_budget,
        bio_field_dt=bio_field_dt, bio_field_scale=bio_field_scale,
        collected=[],
    )
    for name in active:
        _BUILDERS[name](em, ctx)
    return em
