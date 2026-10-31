# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""深度推理 —— 在 SimplCore 流式状态上做"迭代式思考"的控制器。

为什么是"迭代式"而不是"更深一层"
----------------------------------
o1 / R1 一类的深度推理，本质是把"先想清楚再答"显式化：模型生成若干**思考步**
（thought），每一步都更新流式状态，最后才吐出答案。SimplCore 的 O(1) 流式状态
天然适合这个循环——思考步只是不断往状态里喂 token，状态始终是常驻大小，
不会因为"想了 10 步"就爆显存（对比注意力 KV cache 会随步数线性膨胀）。

本模块提供
----------
* :class:`DeepReasoner`：包裹一个 ``SimplCoreModel``（可选接 ``ThinkAccelerator``），
  跑"提示 → 多步思考 → 答案"的循环。
* 收敛策略（``policy``）：
  * ``"fixed_steps"`` —— 固定走 ``max_steps`` 步；
  * ``"stabilize"``   —— 当连续 ``patience`` 步生成的"候选答案"不再变化即提前停止
    （这是**最大的加速器**：很多简单题想 1~2 步就够了，不必烧满 ``max_steps``）。
* 环境工具钩子（``tool``）：每一步思考后可以把文本交给一个外部工具
  （例如 ``envlearn.LocalCodeEnvironment``），把观察结果重新流入状态——
  这正是"深度推理"与"环境学习"的接缝。
* :class:`ReasoningTrace`：结构化的推理轨迹（思考列表、答案、步数、工具调用、收敛点）。

诚实边界
--------
1. 本模块是**控制流骨架**，不是"已训好的推理模型"。它保证循环正确、步数可控、
   流式 O(1) 状态不变、可插工具；至于模型有没有被训练出真正的多步推理能力，
   和 ``theory.py`` 的诚实边界一致——**机制正确 ≠ 质量达标**，未训的玩具模型
   只会吐出无意义的循环文本。
2. ``stabilize`` 早停依赖"候选答案相等"的启发式：在噪声采样下可能多走几步，
   在贪心（``temperature=0``）确定生成下是稳的。它**缩短**的是思考步数，
   不是单步算力（单步算力由 ``ThinkAccelerator`` 负责）。
3. 与全架构一致：本模块不引入注意力。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

__all__ = ["ReasoningTrace", "DeepReasoner", "stabilized"]


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------
@dataclass
class ReasoningTrace:
    """一次深度推理的结构化结果。"""

    prompt: str = ""
    thoughts: List[str] = field(default_factory=list)
    answer: str = ""
    steps_used: int = 0
    tool_calls: List[dict] = field(default_factory=list)
    #: 在 ``policy="stabilize"`` 下，第几步开始候选答案稳定（``None``=没稳定）。
    stabilized_at: Optional[int] = None
    tokens_used: int = 0

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "thoughts": list(self.thoughts),
            "answer": self.answer,
            "steps_used": self.steps_used,
            "tool_calls": list(self.tool_calls),
            "stabilized_at": self.stabilized_at,
            "tokens_used": self.tokens_used,
        }


# ---------------------------------------------------------------------------
# 早停判定（纯函数，便于单测）
# ---------------------------------------------------------------------------
def stabilized(history: Sequence[str], patience: int) -> bool:
    """``history`` 末尾连续 ``patience`` 个候选答案都相等则视为已收敛。

    ``history`` 是各步生成的候选答案序列；长度不足 ``patience`` 时返回 ``False``。
    """
    if patience < 1 or len(history) < patience:
        return False
    window = list(history[-patience:])
    return all(w == window[0] for w in window)


# ---------------------------------------------------------------------------
# 推理器
# ---------------------------------------------------------------------------
class DeepReasoner:
    """在流式状态上做迭代式深度推理的控制器。

    Args:
        model: 被推理的 ``SimplCoreModel``。
        codec: 文本编解码器（``Tokenizer``），用于把生成 token 解回文本。
        accelerator: 可选 ``ThinkAccelerator``，用于给每一步推理加速。
        max_steps: 思考步数上限。
        max_thought_tokens: 每步思考最多生成的 token 数。
        max_answer_tokens: 最终答案最多生成的 token 数。
        policy: ``"fixed_steps"``（走满）或 ``"stabilize"``（候选答案稳定即停）。
        patience: ``stabilize`` 下连续相等的步数阈值。
        temperature / top_k / top_p: 生成采样参数；测试建议 ``temperature=0``。
        eos_token_id: 遇到即停止当前段生成（``None``=只看长度上限）。
        sep_token_id: 可选的分隔 token；遇到它当前思考段即结束（即使未到长度上限）。
        tool: 可选 ``(thought_text) -> Optional[observation_text]`` 钩子；
            返回非 ``None`` 时，观察文本会被重新流入状态，供下一步参考。
    """

    def __init__(
        self,
        model,
        codec,
        accelerator=None,
        max_steps: int = 4,
        max_thought_tokens: int = 48,
        max_answer_tokens: int = 48,
        policy: str = "stabilize",
        patience: int = 2,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        tool: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.model = model
        self.codec = codec
        self.accelerator = accelerator
        self.max_steps = int(max_steps)
        self.max_thought_tokens = int(max_thought_tokens)
        self.max_answer_tokens = int(max_answer_tokens)
        if policy not in ("fixed_steps", "stabilize"):
            raise ValueError(f"未知 policy: {policy!r}")
        self.policy = policy
        self.patience = int(patience)
        self.temperature = float(temperature)
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.tool = tool

    # -- 内部：单 token 推进 ------------------------------------------------
    def _step(self, token: torch.Tensor, state):
        if self.accelerator is not None:
            return self.accelerator.stream_step(token, state)
        return self.model.stream_step(token, state)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        if self.temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        l = logits / self.temperature
        if self.top_k is not None and self.top_k > 0:
            kth = torch.topk(l, min(self.top_k, l.size(-1)), dim=-1).values[:, -1:]
            l = l.masked_fill(l < kth, float("-inf"))
        return torch.multinomial(F.softmax(l, dim=-1), num_samples=1)

    # -- 内部：从给定 logits 起生成一段文本，直到 eos/sep/长度上限 ----------
    @torch.no_grad()
    def _generate_segment(self, logits, state, max_tokens: int) -> tuple:
        if logits is None:
            return "", 0
        text_parts: List[str] = []
        n = 0
        cur = logits
        for _ in range(int(max_tokens)):
            nxt = self._sample(cur)
            tid = int(nxt.reshape(-1)[0])
            if self.eos_token_id is not None and tid == self.eos_token_id:
                break
            if self.sep_token_id is not None and tid == self.sep_token_id:
                break
            cur, state = self._step(nxt, state)
            tok = self.codec.decode([tid])
            if tok:
                text_parts.append(tok)
            n += 1
        return "".join(text_parts), n

    # -- 主入口 -------------------------------------------------------------
    @torch.no_grad()
    def reason(self, prompt: str) -> ReasoningTrace:
        """对 ``prompt`` 跑深度推理，返回 :class:`ReasoningTrace`。"""
        was_training = self.model.training
        self.model.eval()
        trace = ReasoningTrace(prompt=prompt)

        # 1) 把提示流入状态，并拿到首个 logits
        ids = self.codec.encode(prompt)
        if not ids:
            ids = [0]
        idx = torch.tensor([ids], device=next(self.model.parameters()).device)
        state = (self.accelerator.init_stream if self.accelerator is not None
                 else self.model.init_stream)(1, device=idx.device,
                                              dtype=self.model.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self._step(idx[:, t], state)

        candidates: List[str] = []
        for step in range(self.max_steps):
            # 2) 生成这一步的"思考"
            thought, n_th = self._generate_segment(logits, state, self.max_thought_tokens)
            trace.tokens_used += n_th
            if thought:
                trace.thoughts.append(thought)

            # 3) 可选：把思考交给环境工具，把观察流回状态
            if self.tool is not None and thought.strip():
                obs = self.tool(thought)
                if obs:
                    trace.tool_calls.append({"thought": thought, "observation": obs})
                    for oid in self.codec.encode(obs):
                        logits, state = self._step(
                            torch.tensor([[oid]], device=idx.device), state)
                        trace.tokens_used += 1

            # 4) 生成候选答案（作为收敛判据 / 最终答案来源）
            ans, n_ans = self._generate_segment(logits, state, self.max_answer_tokens)
            trace.tokens_used += n_ans
            candidates.append(ans)

            if self.policy == "stabilize" and stabilized(candidates, self.patience):
                trace.stabilized_at = step - self.patience + 1
                trace.answer = candidates[-1]
                trace.steps_used = step + 1
                break

            # 用候选答案的尾部推一步，作为下一轮思考的起点 logits
            if ans:
                aids = self.codec.encode(ans)
                if aids:
                    logits, state = self._step(
                        torch.tensor([[aids[-1]]], device=idx.device), state)
            trace.steps_used = step + 1
        else:
            # 走满 max_steps（for 正常结束不触发 break）
            trace.answer = candidates[-1]

        if was_training:
            self.model.train()
        return trace


def main(argv: Optional[list] = None) -> int:
    """CLI：``python -m simplcore.core.reasoning`` —— 深度推理演示（玩具模型）。

    用一个小随机模型跑一段"提示 → 多步思考 → 答案"，打印结构化轨迹。
    玩具模型未训练，输出只是机制演示、非真实推理能力。
    """
    import argparse

    from .config import SimplCoreConfig
    from .model import SimplCoreModel
    from .tokenization import CharTokenizer

    ap = argparse.ArgumentParser(description="SimplCore 深度推理演示（玩具模型）")
    ap.add_argument("prompt", nargs="?", default="ab")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--policy", default="stabilize", choices=("fixed_steps", "stabilize"))
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--thought-tokens", type=int, default=16)
    ap.add_argument("--answer-tokens", type=int, default=16)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=args.dim, num_layers=3, num_bases=16, decay_max=256,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc")
    m = SimplCoreModel(cfg).eval()
    r = DeepReasoner(
        m, codec, max_steps=args.max_steps, policy=args.policy,
        patience=args.patience, max_thought_tokens=args.thought_tokens,
        max_answer_tokens=args.answer_tokens, temperature=0.0)
    tr = r.reason(args.prompt)

    print(f"prompt   : {tr.prompt!r}")
    for i, t in enumerate(tr.thoughts):
        print(f"thought[{i}]: {t!r}")
    print(f"answer   : {tr.answer!r}")
    print(f"steps_used={tr.steps_used}, stabilized_at={tr.stabilized_at}, "
          f"tokens_used={tr.tokens_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
