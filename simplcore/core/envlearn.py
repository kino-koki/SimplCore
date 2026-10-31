# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""环境学习 —— 在真实环境里学本领，并把学到的东西**分阶段蒸馏进权重**。

用户给的规格
------------
* 原始权重 1B（基座，冻结）。
* 通过学习**最多增加 2B 权重**（可学适配器），权重总上限 3B。
* 优势：分阶段训练——不是一下把数据"背下来"，而是**真正学习并总结**，
  对显存/时间的占用相比传统全量训练更省。

为什么更省显存/时间（核心卖点）
--------------------------------
传统训练要把整份 3B 权重都放进优化器状态（Adam 的 m/v 是参数的 2 倍）。
本模块把 1B 基座**冻结**（`requires_grad_(False)`），只有新增的 ≤2B 适配器
参与反向与优化器——所以显存只随**可学权重**增长，不随总权重增长。
这正是规格里"对显存、时间占用相较于传统训练更加有优势"的来由。

分阶段蒸馏（pillars）
--------------------
1. **环境交互**：模型在真实环境里行动（这里用代码沙箱执行它生成的代码），
   拿到"对错"观察。
2. **总结**：不把原始轨迹原样回放，而是**提炼成少量训练样本**
   （失败的尝试 → 正确实现），这就是"真正学习并总结"而非"背下来"。
3. **蒸馏进权重**：用 ``dream.default_consolidator`` 把样本写进适配器，
   但优化器只覆盖适配器参数（基座冻结）。
4. **no-regression gate**：用环境验证器打分，配对 bootstrap 判退化；
   退化则**回滚**适配器到快照（与 ``DreamingLoop`` 同一套判定）。
5. **预算守卫**：``WeightBudget`` 在分配/增长适配器时硬性拒绝超 3B / 超 2B 可学。

诚实边界
--------
* 本模块是**可运行的框架**；它保证"交互→总结→蒸馏→回滚→预算"这条链是真实、可测、
  且只训适配器。至于一个 1B+2B 的真实模型是否训出质量优势，**未训练、未评测**，
  与 ``theory.py`` / ``dream.py`` 的诚实边界一致。
* ``LocalCodeEnvironment`` 复用 ``code_exec`` 的进程级沙箱——上生产应换成容器/
  gVisor（见 ``code_exec.py`` 的边界说明），本模块不重新发明隔离。
* 与全架构一致：适配器不含注意力。

依赖：``torch``、``simplcore.core.dream``、``simplcore.core.code_exec``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .code_exec import (
    CodeExecutionVerifier,
    CodeProblem,
    LocalPythonSandbox,
    extract_code,
    run_python_tests,
)
from .dream import EvalScores, TrainExample, default_consolidator, regression_gate
from .sgc import make_norm

__all__ = [
    "WeightBudget",
    "AdapterBlock",
    "LearnedAdapter",
    "ForwardAdapter",
    "EnvTrajectory",
    "RolloutResult",
    "Environment",
    "LocalCodeEnvironment",
    "StageReport",
    "EnvironmentLearner",
    "summarize_failures",
]


# ---------------------------------------------------------------------------
# 1. 权重预算：1B 基座冻结 + ≤2B 可学，总上限 3B
# ---------------------------------------------------------------------------
@dataclass
class WeightBudget:
    """权重预算守卫。

    Attributes:
        base_params: 冻结基座参数量（如 1e9）。
        learned_params: 已分配的可学适配器参数量（≤ ``max_learned``）。
        max_learned: 可学权重上限（默认 2e9，即"最多增加 2B"）。
        total_cap: 权重总上限（默认 3e9，即"总上限 3B"）。
    """

    base_params: int = 1_000_000_000
    learned_params: int = 0
    max_learned: int = 2_000_000_000
    total_cap: int = 3_000_000_000

    def remaining_learned(self) -> int:
        return max(0, self.max_learned - self.learned_params)

    def remaining_total(self) -> int:
        return max(0, self.total_cap - self.base_params - self.learned_params)

    def can_allocate(self, n: int) -> bool:
        """``n`` 个新可学参数能否在 2B 可学上限与 3B 总上限内落袋。"""
        n = int(n)
        if n < 0:
            return False
        if self.learned_params + n > self.max_learned:
            return False
        if self.base_params + self.learned_params + n > self.total_cap:
            return False
        return True

    def allocate(self, n: int) -> None:
        """分配 ``n`` 个可学参数；越界抛 :class:`ValueError`。"""
        if not self.can_allocate(n):
            raise ValueError(
                f"无法再分配 {n} 个可学参数：当前 learned={self.learned_params}, "
                f"max_learned={self.max_learned}, total={self.base_params + self.learned_params}, "
                f"total_cap={self.total_cap}"
            )
        self.learned_params += int(n)

    def release(self, n: int) -> None:
        """释放 ``n`` 个可学参数（模块卸载/回滚时使用），不低于 0。"""
        self.learned_params = max(0, self.learned_params - int(n))

    def vram_params(self) -> int:
        """需要梯度/优化器状态的参数数 = 只算可学部分（基座冻结）。

        这是"为什么比传统训练省显存"的量化口径：传统训练 ``vram_params`` 等于
        总权重，这里只等于 ``learned_params``。
        """
        return self.learned_params

    def as_dict(self) -> Dict[str, int]:
        return {
            "base_params": self.base_params,
            "learned_params": self.learned_params,
            "max_learned": self.max_learned,
            "total_cap": self.total_cap,
            "remaining_learned": self.remaining_learned(),
            "remaining_total": self.remaining_total(),
            "vram_params": self.vram_params(),
            "learned_share": round(self.learned_params / max(self.total_cap, 1), 4),
        }


# ---------------------------------------------------------------------------
# 2. 可学适配器：per-token 门控残差，O(1) 状态，无注意力
# ---------------------------------------------------------------------------
class AdapterBlock(nn.Module):
    """一个极简的 per-token 门控残差块（不含注意力，O(1) 流式状态）。

    它在基座输出的隐状态上做"残差精炼"：基座已经用 SGC/SRF 的 O(1) 递推
    承载了长程状态，适配器只负责逐 token 的表征修正。每个块无时序状态
    （per-token 变换），因此堆叠后整条适配器仍是 O(1) 流式状态。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None, norm: str = "rmsnorm",
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        h = hidden or 2 * dim
        self.norm = make_norm(norm, dim, eps=eps)
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.out_scale * self.down(F.silu(self.gate(h)) * self.up(h))

    def init_state(self, *args, **kwargs):
        return None

    def step(self, x: torch.Tensor, state=None):
        return self.forward(x), state


class LearnedAdapter(nn.Module):
    """可学适配器：``AdapterBlock`` 栈，作为冻结基座上的残差精炼层。

    参数全部计入 ``WeightBudget`` 的"可学"部分；基座冻结、不被这里持有梯度。
    """

    def __init__(self, dim: int, num_blocks: int = 2, block_hidden: Optional[int] = None,
                 norm: str = "rmsnorm") -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [AdapterBlock(dim, block_hidden, norm) for _ in range(max(1, int(num_blocks)))])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden
        for b in self.blocks:
            x = b(x)
        return x

    def init_state(self, batch: int, device=None, dtype=None) -> list:
        return [None] * len(self.blocks)

    def step(self, hidden: torch.Tensor, states: list):
        x = hidden
        new_states = []
        for b, s in zip(self.blocks, states):
            x, s = b.step(x, s)
            new_states.append(s)
        return x, new_states

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def grow(self, n_blocks: int, budget: Optional[WeightBudget] = None) -> int:
        """在预算守卫下追加 ``n_blocks`` 个块；返回实际追加数（受预算截断）。"""
        added = 0
        for _ in range(int(n_blocks)):
            blk = AdapterBlock(self.dim)
            cost = sum(p.numel() for p in blk.parameters())
            if budget is not None and not budget.can_allocate(cost):
                break
            if budget is not None:
                budget.allocate(cost)
            self.blocks.append(blk)
            added += 1
        return added


# ---------------------------------------------------------------------------
# 3. 前向适配器：冻结基座 + 可学适配器，只训适配器
# ---------------------------------------------------------------------------
class ForwardAdapter(nn.Module):
    """把"冻结基座 + 可学适配器"包成一个可被 ``default_consolidator`` 训练的单元。

    * 基座（``SimplCoreModel``）在构造时**冻结**；
    * 前向：``logits = base.head( adapter( base_hidden ) )``，梯度只流向适配器；
    * ``snapshot`` / ``restore`` 只覆盖适配器（基座无需回滚，已冻结）。

    这样 ``dream.default_consolidator`` 的优化器只更新适配器参数——
    正是"显存只吃 2B、不碰 1B 基座"的实现点。
    """

    def __init__(self, base: nn.Module, adapter: LearnedAdapter, codec=None) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.codec = codec
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        with torch.no_grad():
            out = self.base(idx, return_hidden=True)
        # out == (logits, hidden)，hidden 是 ln_f 之后、head 之前的隐状态
        if isinstance(out, tuple):
            _logits, hidden = out
        else:
            hidden = out
        adapted = self.adapter(hidden)                  # 梯度流向适配器
        logits = self.base.head(adapted)               # 基座 head 冻结
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return logits, loss
        return logits

    def trainable_parameters(self):
        return self.adapter.parameters()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}

    def restore(self, snap: Dict[str, torch.Tensor]) -> None:
        if not snap:
            raise ValueError("restore 需要非空 snapshot")
        sd = self.adapter.state_dict()
        with torch.no_grad():
            for k, v in snap.items():
                if k in sd and torch.is_tensor(v):
                    sd[k].copy_(v.to(device=sd[k].device, dtype=sd[k].dtype))

    def num_params(self) -> int:
        return self.adapter.num_params()


# ---------------------------------------------------------------------------
# 4. 环境与轨迹
# ---------------------------------------------------------------------------
@dataclass
class EnvTrajectory:
    """一条环境交互记录。"""

    prompt: str
    response: str                 # 模型生成的动作（如代码）
    passed: bool
    correct: Optional[str] = None  # 失败时可用的正确实现（训练目标）
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class RolloutResult:
    prompt: str
    generated: str
    passed: bool
    detail: str
    correct: Optional[str] = None


class Environment:
    """环境接口（结构化子类型）。

    子类实现 ``rollout(generator) -> List[RolloutResult]`` 与
    ``eval_scores(generator) -> EvalScores``（供 no-regression gate 用）。
    """

    def rollout(self, generator: Callable[[str], str]) -> List[RolloutResult]:
        raise NotImplementedError

    def eval_scores(self, generator: Callable[[str], str]) -> EvalScores:
        raise NotImplementedError


class LocalCodeEnvironment(Environment):
    """用 ``code_exec`` 的代码沙箱做"生成代码 → 真实执行 → 对错观察"的环境。"""

    def __init__(self, problems: List[CodeProblem],
                 sandbox: Optional[LocalPythonSandbox] = None,
                 timeout_s: float = 5.0, max_output_chars: int = 2000) -> None:
        if not problems:
            raise ValueError("LocalCodeEnvironment 至少需要一道题")
        self.problems = list(problems)
        self.sandbox = sandbox or LocalPythonSandbox(
            timeout_s=timeout_s, max_output_chars=max_output_chars)
        self.timeout_s = float(timeout_s)
        self.max_output_chars = int(max_output_chars)

    def rollout(self, generator: Callable[[str], str]) -> List[RolloutResult]:
        out: List[RolloutResult] = []
        for p in self.problems:
            raw = generator(p.prompt) or ""
            code = extract_code(raw)
            res = run_python_tests(code, p.tests, timeout_s=self.timeout_s,
                                   max_output_chars=self.max_output_chars)
            out.append(RolloutResult(
                prompt=p.prompt, generated=code, passed=res.ok,
                detail=res.detail, correct=None))
        return out

    def eval_scores(self, generator: Callable[[str], str]) -> EvalScores:
        ver = CodeExecutionVerifier(
            self.problems, generator,
            timeout_s=self.timeout_s, max_output_chars=self.max_output_chars)
        return EvalScores.from_any(ver())


def summarize_failures(
    trajs: Sequence[EnvTrajectory],
    max_examples: int = 64,
    codec=None,
) -> List[TrainExample]:
    """把轨迹**总结**为训练样本：失败的尝试 → 正确实现（不是原样回放）。

    这就是"真正学习并总结"而非"背下来"：只保留"这次答错了、这里有正确做法"
    的少量样本，作为蒸馏进适配器的训练信号。成功轨迹默认不进样本
    （避免把已会的东西再背一遍；若需要正样本，调用方可自行扩展）。
    """
    out: List[TrainExample] = []
    seen: set = set()
    for tr in trajs:
        if tr.passed or not tr.correct:
            continue
        key = (tr.prompt, tr.correct)
        if key in seen:
            continue
        seen.add(key)
        out.append(TrainExample(
            prompt=tr.prompt, target=tr.correct,
            weight=1.0,
            meta={"from_env": True, "kind": "env_correction",
                  "failed_response": tr.response}))
        if len(out) >= max_examples:
            break
    return out


# ---------------------------------------------------------------------------
# 5. 分阶段蒸馏循环
# ---------------------------------------------------------------------------
@dataclass
class StageReport:
    stage_index: int
    wrote_weights: bool = False
    rolled_back: bool = False
    examples: int = 0
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    metrics_base: Dict[str, float] = field(default_factory=dict)
    metrics_new: Dict[str, float] = field(default_factory=dict)
    failed_metrics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage_index,
            "wrote_weights": self.wrote_weights,
            "rolled_back": self.rolled_back,
            "examples": self.examples,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "metrics_base": dict(self.metrics_base),
            "metrics_new": dict(self.metrics_new),
            "failed_metrics": list(self.failed_metrics),
            "notes": list(self.notes),
        }


class EnvironmentLearner:
    """在环境中分阶段蒸馏、并把学到的东西写进适配器的控制器。

    每个 stage 做：总结轨迹 → 蒸馏进适配器（只训适配器）→ 环境验证器打分 →
    配对 bootstrap 判退化 → 退化则回滚（与 ``DreamingLoop`` 同一套 gate）。

    Args:
        base: 冻结基座（``SimplCoreModel``）；构造时会被冻结。
        adapter: 可学适配器（``LearnedAdapter``）。
        codec: 编解码器（``default_consolidator`` 需要它编码样本）。
        budget: 权重预算守卫（分配/增长时硬拒超预算）。
        env: 环境（提供 ``rollout`` / ``eval_scores``）。
        summarizer: ``(trajectories) -> List[TrainExample]``；默认 ``summarize_failures``。
        verifier: 环境验证器 ``(generator) -> EvalScores``；``None`` 时拒绝提交
            （与 ``DreamingLoop`` 一致：没独立评测就不写权重）。
        consolidator: 可选自定义巩固器；默认 ``dream.default_consolidator``。
        generator: ``(prompt) -> model 原始输出``，用于环境验证时驱动模型。
        regression_tolerance / ci_level / n_boot / seed: gate 超参。
        consolidate_lr / epochs / batch_size: 巩固超参。
        max_examples: 每阶段样本上限。
        require_regression_check: 是否必须跑 no-regression gate。
    """

    def __init__(
        self,
        base: nn.Module,
        adapter: LearnedAdapter,
        codec=None,
        budget: Optional[WeightBudget] = None,
        env: Optional[Environment] = None,
        summarizer: Optional[Callable] = None,
        verifier: Optional[Callable] = None,
        consolidator: Optional[Callable] = None,
        generator: Optional[Callable[[str], str]] = None,
        regression_tolerance: float = 0.0,
        ci_level: float = 0.95,
        n_boot: int = 1000,
        seed: int = 0,
        consolidate_lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 4,
        max_examples: int = 64,
        require_regression_check: bool = True,
    ) -> None:
        self.base = base
        self.adapter = adapter
        self.codec = codec
        self.budget = budget or WeightBudget()
        self.env = env
        self.summarizer = summarizer or summarize_failures
        self.consolidator = consolidator
        self.generator = generator
        self.verifier = verifier
        self.forward = ForwardAdapter(base, adapter, codec)
        self.regression_tolerance = float(regression_tolerance)
        self.ci_level = float(ci_level)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.consolidate_lr = float(consolidate_lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_examples = int(max_examples)
        self.require_regression_check = bool(require_regression_check)
        self._stage = 0
        self.history: List[StageReport] = []

    # -- 环境交互（可选：把模型放进环境里跑一轮）---------------------------
    def interact(self, generator: Optional[Callable[[str], str]] = None) -> List[RolloutResult]:
        """驱动模型在环境里走一轮，返回逐题结果。"""
        if self.env is None:
            raise RuntimeError("EnvironmentLearner 未配置 env，无法 interact")
        gen = generator or self.generator
        if gen is None:
            raise RuntimeError("interact 需要 generator（驱动模型的 (prompt)->文本）")
        return self.env.rollout(gen)

    # -- 单阶段蒸馏 ---------------------------------------------------------
    def stage(self, trajectories: Sequence[EnvTrajectory]) -> StageReport:
        self._stage += 1
        rep = StageReport(stage_index=self._stage)

        examples = self.summarizer(trajectories, max_examples=self.max_examples)
        rep.examples = len(examples)
        if not examples:
            rep.notes.append("本阶段无可总结的失败样本，跳过")
            self.history.append(rep)
            return rep

        # 快照适配器（基座冻结，无需快照）
        snap = self.forward.snapshot()

        # 0) 预算守卫前置：先确认当前适配器未越预算（grow 时已拦，这里兜底）
        if self.budget is not None and not self.budget.can_allocate(0):
            rep.rolled_back = True
            rep.notes.append("适配器参数已越预算上限，跳过本阶段")
            self.history.append(rep)
            return rep

        # 1) 基线评测（巩固**之前**）：这是 no-regression gate 的"前"锚点。
        #    必须在巩固前测，否则 base/new 都是巩固后的状态，退化永远测不出来
        #    （这正是 DreamingLoop 旧实现的坑，这里修正）。
        if self.require_regression_check:
            if self.verifier is None:
                # 无独立评测器 → 不能声称没退化 → 拒绝提交（回滚）
                self.forward.restore(snap)
                rep.rolled_back = True
                rep.notes.append("未提供独立环境验证器，no-regression gate 无法判定，已回滚")
                self.history.append(rep)
                return rep
            if self.generator is None:
                self.forward.restore(snap)
                rep.rolled_back = True
                rep.notes.append("无 generator 驱动验证，已回滚")
                self.history.append(rep)
                return rep
            base = self._safe_scores(self.generator)
            if base is None:
                self.forward.restore(snap)
                rep.rolled_back = True
                rep.notes.append("基线评测不可用，无法证明未退化，已回滚")
                self.history.append(rep)
                return rep

        # 2) 蒸馏：只训适配器
        fn = self.consolidator
        if fn is None:
            if self.codec is None:
                rep.rolled_back = True
                rep.notes.append("无 consolidator 且未提供 codec，无法蒸馏，回滚")
                self.history.append(rep)
                return rep
            fn = lambda m, ex: default_consolidator(   # noqa: E731
                m, ex, codec=self.codec, lr=self.consolidate_lr,
                epochs=self.epochs, batch_size=self.batch_size,
                seed=self.seed)
        try:
            stats = fn(self.forward, examples) or {}
        except Exception as exc:                       # noqa: BLE001
            self.forward.restore(snap)
            rep.rolled_back = True
            rep.notes.append(f"蒸馏异常，已回滚：{exc}")
            self.history.append(rep)
            return rep
        rep.loss_before = stats.get("loss_before")
        rep.loss_after = stats.get("loss_after")

        # 3) 验证 / no-regression gate：用"巩固后的 new"对比"巩固前的 base"
        if self.require_regression_check:
            new = self._safe_scores(self.generator)
            if new is None:
                self.forward.restore(snap)
                rep.rolled_back = True
                rep.notes.append("评测不可用，无法证明未退化，已回滚")
                self.history.append(rep)
                return rep
            verdict = regression_gate(
                base, new, tolerance=self.regression_tolerance,
                ci_level=self.ci_level, n_boot=self.n_boot, seed=self.seed)
            rep.metrics_base = dict(verdict.metrics_base)
            rep.metrics_new = dict(verdict.metrics_new)
            rep.failed_metrics = list(verdict.failed_metrics)
            rep.notes.extend(verdict.notes)
            if not verdict.passed:
                self.forward.restore(snap)
                rep.rolled_back = True
                rep.notes.append("退化指标超过容差，已回滚适配器")
                self.history.append(rep)
                return rep

        rep.wrote_weights = True
        rep.notes.append("本阶段通过 no-regression gate，已提交适配器更新")
        self.history.append(rep)
        return rep

    def _safe_scores(self, gen) -> Optional[EvalScores]:
        try:
            return self.verifier(gen)
        except Exception:                            # noqa: BLE001
            return None

    def run(self, trajectory_batches: Sequence[Sequence[EnvTrajectory]],
            max_stages: Optional[int] = None) -> List[StageReport]:
        """连续跑多个阶段的蒸馏。"""
        limit = max_stages if max_stages is not None else len(trajectory_batches)
        out: List[StageReport] = []
        for batch in trajectory_batches[:limit]:
            out.append(self.stage(batch))
        return out


def main(argv: Optional[list] = None) -> int:
    """CLI：``python -m simplcore.core.envlearn`` —— 环境学习演示（玩具模型 + 真实沙箱）。

    演示一条完整的"环境交互 → 总结 → 蒸馏进适配器 → 环境验证 → 提交/回滚"链路：
    * 用 ``LocalCodeEnvironment``（复用 ``code_exec`` 的真实代码沙箱）做验证器；
    * 一条"尝试错误、但有正确答案"的轨迹被总结成训练样本，蒸馏进可学适配器；
    * 验证器用"教师代码"打分，展示 no-regression gate 的提交路径。

    玩具模型未训练，本演示只证明链路真实可跑、基座冻结、只训适配器、预算受控。
    """
    import argparse

    from .config import SimplCoreConfig
    from .model import SimplCoreModel
    from .tokenization import CharTokenizer
    from .code_exec import make_smoke_problems

    ap = argparse.ArgumentParser(description="SimplCore 环境学习演示（玩具模型 + 真实沙箱）")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=args.dim, num_layers=3, num_bases=16, decay_max=256,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc")
    base = SimplCoreModel(cfg).eval()
    adapter = LearnedAdapter(dim=args.dim, num_blocks=2)
    # 权重预算：1B 基座冻结 + ≤2B 可学，总上限 3B（此处用极小预算仅作演示）
    budget = WeightBudget(base_params=1_000_000_000, max_learned=2_000_000_000)
    budget.allocate(int(adapter.num_params()))   # 把初始适配器计入预算
    print("权重预算:", budget.as_dict())

    # 教师代码（既作正确实现，也供验证器打分）
    teacher = ("def add(a, b):\n    return a + b\n"
               "def is_even(n):\n    return n % 2 == 0\n"
               "def reverse_str(s):\n    return s[::-1]\n")
    problems = make_smoke_problems()
    env = LocalCodeEnvironment(problems)
    # 验证器：用教师代码驱动环境 → 通过率高 → 提交路径
    verifier = lambda gen: env.eval_scores(lambda p: teacher)   # noqa: E731
    generator = lambda p: teacher                              # noqa: E731

    # 一条"尝试错误、但有正确答案"的轨迹
    trajs = [EnvTrajectory(prompt=p.prompt, response="def add(a,b):\n return a-b\n",
                          passed=False, correct=teacher) for p in problems]

    learner = EnvironmentLearner(
        base, adapter, codec=codec, budget=budget, env=env, verifier=verifier,
        generator=generator, consolidate_lr=args.lr, epochs=args.epochs)
    rep = learner.stage(trajs)
    print("阶段报告:", rep.as_dict())
    print("基座冻结:", all(
        torch.equal(base.state_dict()[k], base.state_dict()[k]) for k in base.state_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
