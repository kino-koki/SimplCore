"""序列长度课程（opt-in，零回归）。

SGC 无 ``max_len``、因果谱卷积状态 O(1)，因此 4M 这类超长上下文在**架构上**
可行——唯一约束是训练显存。本模块把「seq_len 随训练推进逐步拉长」做成可调度
的课程，让小序列先把基础学好、再拉伸到目标长度做长上下文专项收敛。

设计原则（与项目一贯的 opt-in 纪律一致）：
- 没有课程时（``--seq-len-curriculum`` 未给、且预设 ``context`` 无 ``curriculum``）
  ``SeqLenCurriculum`` 为 falsy，``train.py`` 退化为固定 ``--seq-len``，行为逐位不变。
- 课程由各阶段 ``seq_len`` + 截止点（绝对 step 或 ``max_steps`` 的比例）组成；
  最后阶段无截止 = 跑到训练结束。
- 长度翻倍 / 拉到 4M 这类极端阶段仅改变数据窗口与累计步数，**模型权重零改动**
  （长度无关），因此切换安全。

CLI 紧凑写法（逗号分隔，``seq_len:截止比例``；末段比例可省，默认 1.0）::

    --seq-len-curriculum 4096:0.1,16384:0.25,65536:0.45,262144:0.65,1048576:0.85,4000000

预设 JSON 写法（``context.curriculum`` 与上面等价，比例用 ``until_frac``）::

    "context": {"max_seq_len": 4000000,
                "curriculum": [
                  {"seq_len": 4096,  "until_frac": 0.10},
                  {"seq_len": 16384, "until_frac": 0.25},
                  {"seq_len": 65536, "until_frac": 0.45},
                  {"seq_len": 262144,"until_frac": 0.65},
                  {"seq_len": 1048576,"until_frac": 0.85},
                  {"seq_len": 4000000}
                ]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class _Stage:
    seq_len: int
    until_step: Optional[int] = None   # None = 跑到结束（最后阶段）

    def as_tuple(self) -> Tuple[int, Optional[int]]:
        return (self.seq_len, self.until_step)


@dataclass
class SeqLenCurriculum:
    """给定 step 返回当前应使用的 seq_len；并暴露 max_seq_len 供显存预算断言。

    自适应模式（``adaptive=True``）：阶段推进由外部**就绪度**信号门控——
    调度边界是*最早*可能推进的时刻；就绪度不足则**推迟**拉长（模型尚未消化
    当前长度时硬拉到 4M 只会把表征练废）。越过边界超过 ``grace_steps`` 仍不
    达标则强制推进，避免课程死锁。
    """
    stages: List[_Stage] = field(default_factory=list)
    max_steps: int = 1
    adaptive: bool = False
    achieved_stage: int = 0        # 已达成的阶段序号（单调不减）

    # ---- 构造 ---------------------------------------------------------------
    @classmethod
    def from_cli(cls, spec: str, max_steps: int) -> "SeqLenCurriculum":
        """解析 ``4096:0.1,16384:0.25,...,4000000``（末段比例可省）。"""
        if not spec or not spec.strip():
            return cls([], max_steps)
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        raw: List[Tuple[int, Optional[float]]] = []
        for i, p in enumerate(parts):
            if ":" in p:
                sl_s, frac_s = p.split(":", 1)
                sl, frac = int(sl_s), float(frac_s)
            else:
                sl = int(p)
                # 末段省比例默认 1.0；中间段省比例则等分到后续（这里直接报错更稳）
                frac = 1.0 if i == len(parts) - 1 else None
                if frac is None:
                    raise ValueError(
                        f"中间阶段必须给截止比例：'{p}'（末段可省）")
            raw.append((sl, frac))
        return cls._from_raw(raw, max_steps)

    @classmethod
    def from_context(cls, context: Optional[dict], max_steps: int) -> "SeqLenCurriculum":
        if not context:
            return cls([], max_steps)
        cur = context.get("curriculum")
        if not cur:
            # 仅给了 max_seq_len 而无课程：不启用课程（保持固定 seq_len）
            return cls([], max_steps)
        raw: List[Tuple[int, Optional[float]]] = []
        for i, st in enumerate(cur):
            sl = int(st["seq_len"])
            if "until_step" in st:
                until = float(st["until_step"])
                raw.append((sl, until if until < 1 else until))  # 兼容绝对 step
            elif "until_frac" in st:
                raw.append((sl, float(st["until_frac"])))
            else:
                # 末段：无截止
                if i != len(cur) - 1:
                    raise ValueError(
                        f"非末阶段缺少 until_step/until_frac：{st}")
                raw.append((sl, None))
        return cls._from_raw(raw, max_steps)

    @classmethod
    def _from_raw(cls, raw: List[Tuple[int, Optional[float]]],
                  max_steps: int) -> "SeqLenCurriculum":
        if not raw:
            return cls([], max_steps)
        stages: List[_Stage] = []
        prev_until = 0
        prev_frac = 0.0
        for sl, frac in raw:
            if sl <= 0:
                raise ValueError(f"seq_len 必须为正：{sl}")
            if frac is None:
                stages.append(_Stage(sl, None))
                continue
            if frac <= 0 or frac > 1.0:
                raise ValueError(f"截止比例必须在 (0,1]：{frac}")
            # 配置笔误（比例不递增）仍然直接拒绝。
            if frac <= prev_frac:
                raise ValueError(
                    f"课程阶段截止必须递增：{prev_frac} >= {frac}")
            until = int(round(frac * max_steps))
            # 但**短 run 的取整碰撞**不是配置错误：max_steps=1 时 0.1/0.25
            # 都会 round 到 0。此时单调化到 prev+1，让短跑只走到它能走到的
            # 阶段，而不是在启动时抛一个与用户输入无关的异常。
            if until <= prev_until:
                until = prev_until + 1
            prev_until, prev_frac = until, frac
            stages.append(_Stage(sl, until))
        # 末段强制 until_step=None（跑到结束）
        if stages[-1].until_step is not None:
            stages[-1].until_step = None
        # 单调性兜底：保证 until_step 严格递增（防止配置笔误）
        for i in range(len(stages) - 1):
            nxt = stages[i + 1].until_step
            if nxt is not None and stages[i].until_step is not None and nxt <= stages[i].until_step:
                raise ValueError(
                    f"课程阶段截止必须递增：stage {i} until={stages[i].until_step} "
                    f">= stage {i+1} until={nxt}")
        return cls(stages, max_steps)

    # ---- 查询 ---------------------------------------------------------------
    def __bool__(self) -> bool:
        return len(self.stages) > 0

    @property
    def max_seq_len(self) -> int:
        return max((s.seq_len for s in self.stages), default=0)

    @property
    def initial_seq_len(self) -> int:
        return self.stages[0].seq_len if self.stages else 0

    def _scheduled_index(self, step: int) -> int:
        """调度表给出的阶段序号（不考虑就绪度门控）。"""
        if not self.stages:
            return -1
        for i, st in enumerate(self.stages):
            if st.until_step is None or step < st.until_step:
                return i
        return len(self.stages) - 1

    def promote(self, step: int, readiness: Optional[float] = None,
                gate_threshold: float = 0.5, grace_steps: int = 0) -> bool:
        """尝试推进阶段（仅自适应模式生效）。返回是否真的推进了。

        * 调度边界是**最早**可能推进的时刻（绝不提前，避免长度失控）；
        * ``readiness < gate_threshold`` → 保持当前阶段：模型还没消化当前长度，
          此时硬拉到下一档只会把长上下文表征练废（这正是 4M 课程的真实风险）；
        * 越过当前阶段边界超过 ``grace_steps`` 步仍不达标 → **强制推进**，
          避免课程死锁（生物体的发育也不能无限期停滞）。
        """
        if not self.adaptive or not self.stages:
            return False
        target = self._scheduled_index(step)
        if target <= self.achieved_stage:
            return False
        cur_until = self.stages[self.achieved_stage].until_step
        overdue = 0 if cur_until is None else max(0, step - cur_until)
        if (readiness is not None and readiness < gate_threshold
                and overdue <= grace_steps):
            return False
        self.achieved_stage = target
        return True

    def seq_len_at(self, step: int) -> int:
        """返回 step 当前所处阶段的 seq_len。"""
        if not self.stages:
            return 0
        if self.adaptive:
            return self.stages[min(self.achieved_stage, len(self.stages) - 1)].seq_len
        for st in self.stages:
            if st.until_step is None or step < st.until_step:
                return st.seq_len
        return self.stages[-1].seq_len

    def stage_index_at(self, step: int) -> int:
        if not self.stages:
            return -1
        if self.adaptive:
            return min(self.achieved_stage, len(self.stages) - 1)
        for i, st in enumerate(self.stages):
            if st.until_step is None or step < st.until_step:
                return i
        return len(self.stages) - 1

    def to_dict(self) -> dict:
        return {
            "max_seq_len": self.max_seq_len,
            "stages": [{"seq_len": s.seq_len,
                        "until_step": s.until_step} for s in self.stages],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return json.dumps(self.to_dict(), ensure_ascii=False)


def build_curriculum(args, cfg, max_steps: int) -> SeqLenCurriculum:
    """从 CLI 优先、其次预设 ``context`` 构造课程。

    ``--bio-adaptive-curriculum`` 打开时进入自适应模式：阶段推进由连续场就绪度
    （meso→macro 反馈）门控——需同时启用 ``--enhance bio`` 才有就绪度信号，
    否则就绪度视为中性（按调度推进，行为与自适应关闭时一致）。
    """
    adaptive = bool(getattr(args, "bio_adaptive_curriculum", False))
    if getattr(args, "seq_len_curriculum", None):
        cur = SeqLenCurriculum.from_cli(args.seq_len_curriculum, max_steps)
    else:
        ctx = getattr(cfg, "context", None)
        cur = SeqLenCurriculum.from_context(ctx, max_steps) if ctx \
            else SeqLenCurriculum([], max_steps)
    cur.adaptive = adaptive
    return cur
