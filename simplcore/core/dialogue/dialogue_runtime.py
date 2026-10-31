# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话运行时：把 SimplCore + CMT 记忆 + 四层闸门串成一条可用管线。

为什么需要这一层
----------------
前面几层各自独立可测，但"能力"只发生在它们**串起来**的时候：

* ``memory.py`` 有压缩记忆库，但没人调用它；
* ``abstain.py`` 有四态决策，但没人喂它覆盖率；
* ``dialogue_eval.py`` 能生成，但生成完就直接返回原文。

每一层单独看都是对的，串起来才发现"记忆写了从不被召回"、"拒答门永远
收不到输入"这类问题。本模块就是那条串起来的线，并且刻意让每个环节的
中间量都出现在返回值里 —— 出问题时能定位到具体一环，而不是"答得不好"。

三类输出的区别（必须分清）
--------------------------
===================  ==========================================================
``ANSWER``            检索到的记忆**逐句**都能归因 → 返回原文
``HEDGE``             部分能归因 → 返回原文 + 限定语，且**标注**哪句没依据
``CLARIFY``/``ABSTAIN``  归因不上 → **不返回原文**，返回模板话术
===================  ==========================================================

最后一行是关键：拒答时把原文一起返回，等于"说了不该说的话，再加一句
免责声明"。用户会读到那句编造。所以拒答时原文只留在 ``raw_text`` 里，
供调试和审计，不进对话。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from ..arch.abstain import (AbstentionState, GroundedPolicy, evidence_aware_decision)
from .chat import ChatTemplate, SpecialTokenCodec
from .dialogue_eval import generate_reply, normalize_text
from ..arch.grounding import EvidenceIndex, split_claims
from ..memory.memory import CompressedMemoryBank, TextKeyEncoder


# ---------------------------------------------------------------------------
# 记忆
# ---------------------------------------------------------------------------
@dataclass
class MemoryHit:
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "score": round(self.score, 4),
                "meta": {k: v for k, v in self.meta.items() if k != "vector"}}


class DialogMemory:
    """跨会话记忆：写入「一条事实」，将来按语义相近的查询召回。

    与 :class:`~simplcore.core.memory.CompressedMemoryBank` 的分工：
    后者只管向量存取与 LRU 驱逐，本类负责**文本 ↔ 向量**的转换和
    "召回结果怎么变成证据"。

    键用 ``TextKeyEncoder``（字面 n-gram 哈希），**不是语义嵌入**。
    这意味着"营业总收入"和"营业收入"未必能互相召回 —— 这是当前实现的
    真实边界，不是可以忽略的细节。
    """

    def __init__(self, dim: int, capacity: int = 4096,
                 encoder: Optional[TextKeyEncoder] = None) -> None:
        self.dim = int(dim)
        self._encoder = encoder or TextKeyEncoder(self.dim)
        self._bank = CompressedMemoryBank(dim=self.dim, capacity=capacity)

    # -- 写 ---------------------------------------------------------------
    def add(self, text: str, key: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None) -> bool:
        """写入一条记忆。返回是否真的写入（空文本 / 纯标点会被拒绝）。

        空文本的编码向量为零，写进去只会让"任意查询都召回一条空记忆"，
        然后把召回分数整体拉低 —— 这种污染不会报错，只会让检索慢慢失效。
        """
        body = (text or "").strip()
        if not body:
            return False
        k = self._encoder((key or body).strip())
        v = self._encoder(body)
        if float(k.norm()) <= 0.0 or float(v.norm()) <= 0.0:
            return False
        self._bank.add(k.unsqueeze(0), v.unsqueeze(0),
                       [dict(meta or {}, text=body)])
        return True

    def extend(self, texts: Sequence[str],
               metas: Optional[Sequence[Dict[str, Any]]] = None) -> int:
        n = 0
        for i, t in enumerate(texts):
            m = metas[i] if metas and i < len(metas) else None
            n += int(self.add(t, meta=m))
        return n

    # -- 读 ---------------------------------------------------------------
    def search(self, query: str, top_k: int = 4,
               min_score: float = 0.0) -> List[MemoryHit]:
        """返回 top-k 命中，**按分数降序**。

        这里自己做一次 ``topk`` 而不是复用 ``bank.retrieve``：retrieve 只回
        向量和分数、不回下标，而 ``metas`` 的顺序是**写入顺序**。两者的
        顺序不一致，靠"第 k 个分数对应第 k 条 meta"去对齐会静默错位 ——
        表现是"召回了正确的句子，但记录了错误的来源"，排查时极难发现。
        """
        body = (query or "").strip()
        if not body or len(self._bank) == 0:
            return []
        q = self._encoder(body)
        if float(q.norm()) <= 0.0:
            return []
        keys = self._bank.keys
        if keys.numel() == 0:
            return []
        sim = torch.nn.functional.normalize(q, dim=0) @ \
            torch.nn.functional.normalize(keys, dim=-1).T
        k = min(int(top_k), int(sim.numel()))
        scores, idxs = torch.topk(sim, k)
        metas = self._bank.metas()
        out: List[MemoryHit] = []
        for s, i in zip(scores.tolist(), idxs.tolist()):
            if s < min_score:
                continue
            m = metas[i] if 0 <= i < len(metas) else {}
            out.append(MemoryHit(text=str(m.get("text", "")), score=float(s),
                                 meta=m))
        return out

    def aggregate(self, hits: Sequence[MemoryHit]) -> Optional[torch.Tensor]:
        """把命中的记忆聚合成一个注入向量（按分数加权平均）。

        不做加权平均而是直接拼接是错的：注入点只有一个 ``(1, dim)``
        向量，拼接会改变维度。加权平均保留了"哪条更相关"的信息。
        """
        if not hits:
            return None
        vecs, ws = [], []
        for h in hits:
            v = self._encoder(h.text)
            if float(v.norm()) <= 0.0:
                continue
            vecs.append(v)
            ws.append(max(h.score, 0.0) + 1e-6)
        if not vecs:
            return None
        m = torch.stack(vecs)
        w = torch.tensor(ws, dtype=m.dtype).unsqueeze(1)
        agg = (m * w).sum(0) / w.sum()
        n = float(agg.norm())
        return agg / n if n > 0 else agg

    # -- 状态 -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._bank)

    def snapshot(self):
        return self._bank.state_dict()

    def restore(self, state) -> None:
        self._bank.load_state_dict(state)

    def stats(self) -> Dict[str, Any]:
        return {"size": len(self._bank), "dim": self.dim}


# ---------------------------------------------------------------------------
# 一轮回复
# ---------------------------------------------------------------------------
@dataclass
class Reply:
    text: str
    state: str
    reason: str
    raw_text: str = ""
    hits: List[MemoryHit] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    coverage: float = 0.0
    n_unsupported: int = 0
    stopped: bool = False
    leaked: bool = False
    suppressed: bool = False       # 是否因为拒答而**没有**返回原文

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "state": self.state,
            "reason": self.reason,
            "coverage": round(self.coverage, 4),
            "n_unsupported": self.n_unsupported,
            "stopped": self.stopped,
            "leaked": self.leaked,
            "suppressed": self.suppressed,
            "memory_hits": [h.as_dict() for h in self.hits],
            "claims": self.claims,
            "raw_text": self.raw_text,
        }


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
#: 记忆召回的最低余弦相似度。
#:
#: 这个值刻意很低（0.05）而不是 0.2+：``TextKeyEncoder`` 是**字面 3-gram
#: 哈希**，不是语义嵌入。"我叫什么名字" 与 "用户的名字是小林" 的 3-gram
#: 交集为空（"名字"只是 2-gram），相似度就是 0 —— 真实数据上期望值本来就
#: 接近 0，卡高阈值等于把记忆检索整个关掉。
#:
#: 真正"够不够用"的判断交给 :class:`GroundedPolicy.tau_retrieval`：
#: 检索层只负责"按字面重叠排序"，决策层负责"这个分数算不算有依据"。
MIN_RECALL_SCORE = 0.05

_HEDGE_PREFIX = "（不确定）"
_CLARIFY_TEXT = "我需要更多信息才能回答这个问题。"
_ABSTAIN_TEXT = "这个问题我没有可靠依据，不能给出确定答案。"


class ChatSession:
    """多轮会话：历史 + 跨会话记忆 + 证据感知的拒答。

    Args:
        model: ``SimplCoreModel``（``use_memory=True`` 时记忆才真正生效）。
        codec / template: 与训练时**完全一致**的那一套；不一致时生成的
            是乱码而不是"答得差"，前者会让人误判模型能力。
        memory: 跨会话记忆；``None`` 时退化为纯上下文对话。
        policy: 四态策略阈值。
        use_evidence_gate: 关掉可以对照"有闸门 vs 无闸门"的差异，
            **不是**给线上用的开关。
    """

    def __init__(self, model, codec: SpecialTokenCodec,
                 template: Optional[ChatTemplate] = None,
                 memory: Optional[DialogMemory] = None,
                 policy: Optional[GroundedPolicy] = None,
                 max_new_tokens: int = 96,
                 use_evidence_gate: bool = True,
                 use_state: bool = True,
                 generator: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        self.model = model
        self.codec = codec
        self.template = template or ChatTemplate()
        self.memory = memory
        self.policy = policy or GroundedPolicy()
        self.max_new_tokens = int(max_new_tokens)
        self.use_evidence_gate = bool(use_evidence_gate)
        self.use_state = bool(use_state)
        # ``generator`` 是给测试用的注入点：把"生成"和"归因/四态"解耦，
        # 才能用确定的文本验证闸门行为，而不必先训一个模型。
        # 生产路径永远用默认的 ``generate_reply``。
        self._generate = generator or generate_reply
        self.history: List[Dict[str, str]] = []
        self.turns: List[Reply] = []

    # -- 记忆 -------------------------------------------------------------
    def remember(self, text: str, key: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> bool:
        if self.memory is None:
            return False
        return self.memory.add(text, key=key, meta=meta)

    def recall(self, query: str, top_k: int = 4,
               min_score: float = 0.0) -> List[MemoryHit]:
        if self.memory is None:
            return []
        return self.memory.search(query, top_k=top_k, min_score=min_score)

    def reset(self) -> None:
        """清空**会话历史**。跨会话记忆不受影响 —— 这正是两者该有的区别。"""
        self.history.clear()
        self.turns.clear()

    # -- 主流程 -----------------------------------------------------------
    def ask(self, user_text: str, evidence: Optional[str] = None,
            top_k: int = 4, min_score: float = MIN_RECALL_SCORE) -> Reply:
        """一轮问答。四步：检索 → 生成 → 归因 → 四态处置。

        Args:
            evidence: 额外证据（检索文档、工具返回值）。与记忆命中合并成
                同一个证据文本，一起用于归因 —— 否则"有检索但不在记忆里"
                会被误判成无依据。
        """
        hits = self.recall(user_text, top_k=top_k, min_score=min_score)
        mem_vec = self.memory.aggregate(hits) if (self.memory and hits) else None

        messages = list(self.history) + [{"role": "user", "content": user_text}]
        prompt = self.template.prompt_ids(self.codec, messages)
        gen = self._generate(self.model, prompt, self.codec,
                             max_new_tokens=self.max_new_tokens,
                             temperature=0.0, use_state=self.use_state,
                             memory=mem_vec)
        raw = gen["text"]

        reply = self._adjudicate(raw, user_text, hits, evidence)
        reply.stopped = bool(gen["stopped"])
        reply.leaked = bool(gen["leaked"])

        # 只有 ANSWER / HEDGE 才把这一轮写进历史。
        # 把被拒答的那句塞回历史，下一轮模型会把它当成"自己说过的事实"，
        # 然后在此基础上继续编 —— 拒答必须在上下文里也是干净的。
        if not reply.suppressed:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply.text})
        else:
            self.history.append({"role": "user", "content": user_text})
        self.turns.append(reply)
        return reply

    # -- 归因 + 四态 ------------------------------------------------------
    def _adjudicate(self, raw: str, question: str, hits: Sequence[MemoryHit],
                    evidence: Optional[str]) -> Reply:
        spans: List[str] = [h.text for h in hits if h.text]
        if evidence:
            spans.append(evidence)
        corpus = "\n".join(spans).strip()

        claims = split_claims(raw)
        if not self.use_evidence_gate:
            return Reply(text=raw, state=AbstentionState.ANSWER.value,
                         reason="证据闸门已关闭", raw_text=raw,
                         hits=list(hits),
                         claims=[c.as_dict() for c in claims])

        if normalize_text(raw) == "":
            # 空回复**永远**要拦，与有没有证据无关。放行的话会返回空串，
            # 而空串在下游会被当成"格式干净"（stop=Y、leak=n、empty=n
            # 如果只有一句"没内容"是不算 leak 的），指标全部虚高。
            return Reply(text=_CLARIFY_TEXT,
                         state=AbstentionState.CLARIFY.value,
                         reason="模型没有生成任何内容",
                         raw_text=raw, hits=list(hits), suppressed=True,
                         claims=[])

        if not corpus:
            # 一条证据都没有：这是"资料不足"，不是"答得不对"。
            # 30M 模型在无证据时最常见的失败是编一个像样的数字，
            # 所以这里必须是 CLARIFY 而不是放行。
            return Reply(text=_CLARIFY_TEXT,
                         state=AbstentionState.CLARIFY.value,
                         reason="没有任何可用证据（记忆与外部证据都为空）",
                         raw_text=raw, hits=list(hits), suppressed=True,
                         claims=[c.as_dict() for c in claims])

        index = EvidenceIndex.from_text(corpus, source="memory+evidence")
        bindings = index.bind_all(claims)
        n_total = len(bindings)
        n_unsupported = sum(1 for b in bindings if not b.supported)
        # 归因覆盖率按**句**算，不按字符长度算。
        # 按长度加权会让一句短的编造被两句长的真话稀释到阈值以上，
        # 评测里实测过这个缺陷：多句拦截率只有 0.88。
        # 没有句子时候 coverage 必须是 0 而不是 1 —— 否则"什么都没说"
        # 会拿到满分归因，直接判 ANSWER。
        coverage = (1.0 - n_unsupported / n_total) if n_total > 0 else 0.0

        # L4 头的输出在这里没有意义：30M 模型在本仓库里从未在带引用的
        # 数据上训练过这个头。用覆盖率作为 grounded 的可信替代，
        # 并在文档里写清这是"代理分数"。
        grounded_prob = coverage
        retrieval_conf = (max((h.score for h in hits), default=0.0)
                          if hits else 0.0)

        dec = evidence_aware_decision(
            grounded_prob=grounded_prob, coverage=coverage,
            retrieval_confidence=retrieval_conf, policy=self.policy,
            n_unsupported=n_unsupported)

        claim_dicts = [
            {"text": b.claim.text, "supported": bool(b.supported),
             "coverage": round(float(b.coverage), 4),
             "matched_span": (b.span.text[:40] if b.span is not None else None)}
            for b in bindings
        ]

        if dec.state is AbstentionState.ANSWER:
            text, suppressed = raw, False
        elif dec.state is AbstentionState.HEDGE:
            text, suppressed = f"{_HEDGE_PREFIX}{raw}", False
        elif dec.state is AbstentionState.CLARIFY:
            text, suppressed = _CLARIFY_TEXT, True
        else:
            text, suppressed = _ABSTAIN_TEXT, True

        return Reply(text=text, state=dec.state.value, reason=dec.reason,
                     raw_text=raw, hits=list(hits), claims=claim_dicts,
                     coverage=coverage, n_unsupported=n_unsupported,
                     suppressed=suppressed)

    # -- 报告 -------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        if not self.turns:
            return {"turns": 0}
        n = len(self.turns)
        return {
            "turns": n,
            "answer_rate": round(sum(1 for r in self.turns
                                     if r.state == AbstentionState.ANSWER.value) / n, 4),
            "hedge_rate": round(sum(1 for r in self.turns
                                    if r.state == AbstentionState.HEDGE.value) / n, 4),
            "suppressed_rate": round(sum(1 for r in self.turns
                                         if r.suppressed) / n, 4),
            "mean_coverage": round(sum(r.coverage for r in self.turns) / n, 4),
            "memory": (self.memory.stats() if self.memory is not None else None),
        }
