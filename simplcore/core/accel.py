# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""思考加速 —— 在不引入注意力的前提下，按 token 难度动态削减推理算力。

为什么需要它
------------
SimplCore 的每个 mixer 都是 O(1) 流式状态（无 KV cache、无位置表），所以
"解码每 token 的算力与上下文长度无关"。但**逐层**算力是固定的：每个 token 都要
走完所有 ``num_layers`` 个 block。``ThinkAccelerator`` 要削掉的是这一层冗余——

* 易 token（模型已经很确信）只需要浅层就能给出高质量预测，深层是浪费；
* 难 token（熵高、长程/局部路径分歧大）才值得跑满全部层。

机制：逐 token **自适应深度**（adaptive depth）
-------------------------------------------
每个 token 先过第 0 层（锚点），算一个"必要性分数"：

* ``necessity`` 高（高熵 / 低置信）→ 跑满 ``max_depth`` 层（难 token 走全算力）；
* ``necessity`` 低（低熵 / 高置信）→ 只跑 ``min_depth`` 层（易 token 走浅算力）。

``skip_threshold``、``min_depth``、``max_depth`` 是三个可调旋钮：

* ``skip_threshold = -inf``、``min_depth = num_layers`` → 退化为恒全算力，
  **与 ``SimplCoreModel.stream_step`` 逐位一致**（这是理论守卫要验证的不变式）；
* 调高 ``skip_threshold`` / 调低 ``min_depth`` → 越激进的加速，质量损失越大。

诚实边界
--------
1. 自适应深度是**有损近似**：被跳过的层不处理该 token，其常驻状态保持上一帧——
   这与 Transformer 的 layer-drop / 早退是同一类近似，不是无损加速。
2. 它**不缩短** O(1) 流式状态（状态始终维护），加速来自"少跑几层"，不是"少存状态"。
3. 加速的代价是质量：``skip_threshold`` 越大、``min_depth`` 越小，速度越快但
   退化风险越高。本模块只负责把"算多少"这件事变成可调、可测、可回滚的旋钮，
   **不保证**某个阈值下质量不退化——那需要 ``DreamingLoop`` 的 no-regression gate。
4. 本模块**不引入任何注意力**（与全架构的纯洁性约束一致，见 ``purity.py``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ThinkAccelerator", "ComputeLedger"]


# ---------------------------------------------------------------------------
# 算力账本
# ---------------------------------------------------------------------------
@dataclass
class ComputeLedger:
    """逐 token 记录"本可跑的层数"与"实际跑的层数"，算出加速比。

    加速比是**结构性**的（每 token 少跑的层），不依赖具体硬件计时——
    这样在 CI 里也能稳定复现，而不是看 GPU 抖不抖。
    """

    blocks_budget: int = 0            # Σ 每 token 的层预算（= num_layers）
    blocks_run: int = 0               # Σ 每 token 实际跑的层数
    tokens: int = 0

    def record(self, run: int, budget: int) -> None:
        self.blocks_run += int(run)
        self.blocks_budget += int(budget)
        self.tokens += 1

    @property
    def speedup(self) -> float:
        """单位算力的加速比（budget / run）。恒 ≥ 1。"""
        return self.blocks_budget / max(self.blocks_run, 1)

    @property
    def frac_compute(self) -> float:
        """实际算力占全算力的比例（0..1）。"""
        return self.blocks_run / max(self.blocks_budget, 1)

    def reset(self) -> None:
        self.blocks_budget = 0
        self.blocks_run = 0
        self.tokens = 0

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "blocks_budget": self.blocks_budget,
            "blocks_run": self.blocks_run,
            "speedup": round(self.speedup, 4),
            "frac_compute": round(self.frac_compute, 4),
        }


# ---------------------------------------------------------------------------
# 必要性打分
# ---------------------------------------------------------------------------
def _necessity_entropy(model: "SimplCoreModelLike", hidden: torch.Tensor) -> float:
    """以锚点层输出的预测熵作为必要性：熵越高越"难"，越该跑满层。"""
    with torch.no_grad():
        h = model.ln_f(hidden)
        logits = model.head(h[:, -1, :])                 # (B, V)
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * (probs + 1e-12).log()).sum(-1).mean()
    return float(ent)


def _necessity_confidence(model: "SimplCoreModelLike", hidden: torch.Tensor) -> float:
    """以 1 - 最大概率 作为必要性（置信越低越"难"）。"""
    with torch.no_grad():
        h = model.ln_f(hidden)
        logits = model.head(h[:, -1, :])
        conf = F.softmax(logits, dim=-1).max(-1).values.mean()
    return float(1.0 - conf)


# 前向引用，仅用于类型提示；运行时用鸭子类型访问 model.ln_f / model.head。
class SimplCoreModelLike:  # pragma: no cover - 仅类型占位
    pass


# ---------------------------------------------------------------------------
# 加速器
# ---------------------------------------------------------------------------
class ThinkAccelerator(nn.Module):
    """逐 token 自适应深度的思考加速器。

    包裹一个 ``SimplCoreModel``，在**流式**解码路径上按 token 难度动态决定
    跑几层。保留原模型的 O(1) 流式状态语义——被跳过的层只是"不处理这个 token"，
    其常驻状态维持上一帧（与 layer-drop 同类的近似）。

    Args:
        model: 被加速的 ``SimplCoreModel``。
        skip_threshold: 必要性低于此值就走浅路径（``min_depth`` 层）。
            设 ``-inf`` 则永不跳过（恒全算力，等价于原模型）。
        min_depth: 浅路径跑的层数（易 token 的最低算力）。必须 ≥ 1。
        max_depth: 深路径跑的层数上限（难 token 的最高算力）。默认 = 全部层。
            设成 < num_layers 可对"思考"设置硬性算力上限。
        necessity: 必要性打分方式：``"entropy"`` / ``"confidence"`` 或自定义
            ``(model, hidden) -> float``。
    """

    def __init__(
        self,
        model: nn.Module,
        skip_threshold: float = float("-inf"),
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        necessity: str | Callable = "entropy",
    ) -> None:
        super().__init__()
        self.model = model
        if min_depth < 1:
            raise ValueError("min_depth 必须 >= 1（至少过锚点层）")
        self.skip_threshold = float(skip_threshold)
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth) if max_depth is not None else len(model.blocks)
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth 不能大于 max_depth")
        self._necessity_kind = necessity
        if necessity == "entropy":
            self._necessity_fn: Callable = lambda m, h: _necessity_entropy(m, h)
        elif necessity == "confidence":
            self._necessity_fn = lambda m, h: _necessity_confidence(m, h)
        elif callable(necessity):
            self._necessity_fn = necessity
        else:
            raise ValueError(f"未知 necessity: {necessity!r}")
        self.ledger = ComputeLedger()

    # -- 流式 ----------------------------------------------------------------
    def init_stream(self, batch: int, device=None, dtype=None,
                    memory: Optional[torch.Tensor] = None):
        return self.model.init_stream(batch, device=device, dtype=dtype, memory=memory)

    @torch.no_grad()
    def stream_step(self, token: torch.Tensor,
                    state, skip_override: Optional[bool] = None) -> Tuple[torch.Tensor, object]:
        """消费一个 token，按难度决定跑几层，返回下一 token 的 logits。

        与 ``SimplCoreModel.stream_step`` 的关系：``skip_threshold=-inf`` 且
        ``min_depth=num_layers`` 时应逐位一致——这是 ``theory.py`` 的守卫项。

        Args:
            token: ``(B,)`` 或 ``(B, 1)`` 的 token id。
            state: ``SGCStreamState``。
            skip_override: 强制跳过决策（``True``=浅路径，``False``=深路径），
                ``None``=按 necessity 自动决策。测试用。
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        model = self.model
        mem = state.memory
        x = model.tok_emb(token)                              # (B,1,D)

        # 第 0 层（锚点）—— 永远要跑
        blk0, st0 = model.blocks[0], state.layers[0]
        if mem is not None:
            x = x + mem.unsqueeze(1)
        x, _ = blk0.mixer.step(x, st0)
        x = blk0.ffn(x)
        anchor = x

        # 决策
        if skip_override is None:
            nec = self._necessity_fn(model, anchor)
            skip = nec < self.skip_threshold
        else:
            skip = bool(skip_override)
        depth = self.min_depth if skip else self.max_depth

        # 跑剩余层 1..depth-1
        for i in range(1, depth):
            blk, st = model.blocks[i], state.layers[i]
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, _ = blk.mixer.step(x, st)
            x = blk.ffn(x)

        x = model.ln_f(x)
        state.length += 1
        self.ledger.record(run=depth, budget=len(model.blocks))
        return model.head(x[:, -1, :]), state

    # -- 生成（用自适应深度替换模型的全算力生成）----------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """与 ``SimplCoreModel.generate`` 同签名，但用自适应深度跑流式路径。"""
        was_training = self.model.training
        self.model.eval()
        B = idx.shape[0]
        state = self.init_stream(B, device=idx.device, dtype=self.model.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
        if was_training:
            self.model.train()
        return out

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int],
                top_p: Optional[float]) -> torch.Tensor:
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
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    # -- 自检 ----------------------------------------------------------------
    def attention_like_modules(self) -> List[Tuple[str, str]]:
        """返回加速器及其包裹模型里所有类名含 attention/attn 的模块（应为空）。"""
        from .purity import attention_like_modules as _attn
        hits = list(_attn(self))
        return hits
