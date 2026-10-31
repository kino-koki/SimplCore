# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话能力评测：可判定、无评审模型、能跑在 30M 模型上的最小套件。

为什么不用"让大模型打分"
------------------------
评审模型对小模型的输出极其宽容（"意思差不多"），而 30M 模型最典型的失败
恰恰是"意思不算错，但格式全崩"：不停、抢话、答非所问、超长。这些都必须
用**规则**判定，否则"30M 学会对话"这句话没有办法被证伪。

本模块把"基础对话能力"拆成两类指标：

**格式指标**（与内容无关，纯结构）

===================  ==========================================  =========
指标                  含义                                        健康值
===================  ==========================================  =========
``stop_rate``        在 ``max_new_tokens`` 之前主动输出 ``<|end|>``  ≥ 0.9
``turn_leak_rate``   输出里出现 ``<|user|>`` / ``<|assistant|>``   ≤ 0.05
``empty_rate``       生成为空或只有标点                          ≤ 0.05
``length_ok_rate``   落在题目要求的长度约束内                     ≥ 0.8
===================  ==========================================  =========

**任务指标**（内容正确性，全部规则可判）

``task_pass_rate`` —— 逐题通过率。题面见 :data:`BASIC_DIALOGUE_SUITE`。

这两类必须分开看：格式指标反映"会不会对话"，任务指标反映"懂不懂"。
30M 模型的现实目标是把格式指标打到 0.9 以上、任务指标在
"复述 / 格式约束 / 短问答"这几类上站住 —— 这不是"通用助手"，
而是"能对话"的下限。**任何超出这个范围的宣称都需要另外的评测集。**
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

import torch

from .chat import (ASSISTANT_TOKEN, END_TOKEN, USER_TOKEN, ChatTemplate,
                   SpecialTokenCodec)


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_FULLWIDTH = {ord(c): ord(c) - 0xFEE0 for c in
              "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
              "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}


def normalize_text(s: str) -> str:
    """归一化以便比较：全角转半角、去空白、去首尾标点。

    只做**不改变语义**的归一化。绝不做同义词替换或模糊匹配 —— 那会让
    "差不多对"混进"对"，30M 模型的真实水位就被高估了。
    """
    s = (s or "").translate(_FULLWIDTH)
    s = _WS.sub("", s)
    s = s.strip("，。！？、；：,.!?;:\"'“”‘’（）()【】[]<>《》")
    return s


def _numbers(s: str) -> List[float]:
    out: List[float] = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", s or ""):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


@dataclass
class Check:
    """一条可判定规则。多个字段同时给出时取**与**。"""

    equals: Optional[str] = None
    contains: Sequence[str] = ()
    not_contains: Sequence[str] = ()
    regex: Optional[str] = None
    max_chars: Optional[int] = None
    min_chars: Optional[int] = None
    numbers: Optional[Sequence[float]] = None
    count_at_least: Optional[Tuple[str, int]] = None
    #: ``(分隔符, 至少几个互不相同的非空项)``。用于"列三个水果"这类题：
    #: 只数分隔符会把 ``"苹果、苹果、苹果"`` 判为通过。
    distinct_count: Optional[Tuple[str, int]] = None
    #: 回复**不得等于**其中任意一句（归一化后比较）。
    #:
    #: 用于挡"检索语言"：小模型遇到不会答的问题会把训练池里的某句现成话
    #: 原样吐出来。那不是会说话，是从库里挑了一条。用 ``not_contains``
    #: 挡不住"只吐一句、且正好等于库里某句"的情形，必须整句比对。
    not_in: Sequence[str] = ()
    #: 回复**必须落在**其中一句（归一化后比较）。
    #:
    #: 与 ``not_in`` 互补，解决"开放问答怎么判对题"的问题：只要求
    #: "像句人话 + 不复读"，模型把 A 话题的答案安到 B 问题上也照样通过 ——
    #: 那不是会说话，是随手挑了一条模板。给定本题可接受的答案集合，
    #: 才能把"答到点子上"和"说了一句通顺的话"分开。
    in_set: Sequence[str] = ()

    def evaluate(self, reply: str) -> Tuple[bool, str]:
        body = normalize_text(reply)
        raw = (reply or "").strip()
        # 空回复永远不通过，哪怕题面只约束上界（``max_chars=1`` 时
        # ``len("") == 0 <= 1`` 会让"什么都没说"判为通过）。
        # 这个漏洞让训练前的随机模型拿到 0.1 的基线分。
        if not body:
            return False, "空回复"
        for banned in self.not_in:
            if body == normalize_text(banned):
                return False, f"直接复述了训练原句 {banned!r}"
        if self.in_set:
            pool = {normalize_text(x) for x in self.in_set}
            if body not in pool:
                return False, f"没答到本题话题（回复 {body!r}）"
        if self.equals is not None:
            want = normalize_text(self.equals)
            if body != want:
                return False, f"期望 {want!r}，实际 {body!r}"
        for sub in self.contains:
            if normalize_text(sub) not in body:
                return False, f"缺少 {sub!r}"
        for sub in self.not_contains:
            if normalize_text(sub) in body:
                return False, f"出现了禁止内容 {sub!r}"
        # 正则跑在**原文**上：归一化会剪掉首尾标点，``^【.+】$`` 这类
        # 表面格式的正则就永远匹配不上，判定会静默失效。
        if self.regex is not None and not re.search(self.regex, raw):
            return False, f"不匹配正则 {self.regex!r}"
        if self.max_chars is not None and len(body) > self.max_chars:
            return False, f"超长 {len(body)} > {self.max_chars}"
        if self.min_chars is not None and len(body) < self.min_chars:
            return False, f"过短 {len(body)} < {self.min_chars}"
        if self.numbers is not None:
            got = _numbers(body)
            if sorted(got) != sorted(float(x) for x in self.numbers):
                return False, f"数字不符 期望 {list(self.numbers)} 实际 {got}"
        if self.count_at_least is not None:
            needle, n = self.count_at_least
            # 分隔符本身可能就是标点（"、"），归一化会把它整个剪掉，
            # 于是 ``body.count("")`` 等于 len(body)+1 —— 判定会永远通过。
            probe = normalize_text(needle) or needle
            if body.count(probe) < n:
                return False, f"{needle!r} 出现次数不足 {n}"
        if self.distinct_count is not None:
            sep, n = self.distinct_count
            items = [x for x in body.split(sep) if x]
            if len(set(items)) < n:
                return False, f"互不相同的项不足 {n}：{items}"
        return True, ""


@dataclass
class DialogueTask:
    id: str
    messages: List[Dict[str, str]]
    check: Check
    max_new_tokens: int = 96
    note: str = ""
    #: 这一题**只存在于外部记忆**里的那条事实。非空时，评测会把它编码成
    #: 向量走 CMT 注入通道；空字符串表示这题不需要记忆。
    #:
    #: 有了这个字段，才能跑"开记忆 / 关记忆"的配对消融 —— 那是
    #: ``memory_in`` 唯一能自我证明的证据。只看带记忆的通过率说明不了问题：
    #: 模型完全可能靠参数里背下来的东西答对。
    memory: str = ""
    #: 是否为**对照题**：答案不依赖记忆也能答对（例如"记忆里没有该字段 → 回答
    #: 不知道"，一个永远说"不知道"的模型就能通过）。
    #:
    #: 这类题必须与记忆题**分开统计**。混在一起算通过率，会让"永远拒答"的
    #: 模型白拿分：5 题里恰好有 1 题是拒答题，它就自动拿到 0.2，而它对记忆
    #: 一无所知。消融表因此会把"什么都没学会"读成"有一点效果"。
    control: bool = False

    def as_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "max_new_tokens": self.max_new_tokens,
             "note": self.note}
        if self.memory:
            d["memory"] = self.memory
        return d


# ---------------------------------------------------------------------------
# 内建套件
# ---------------------------------------------------------------------------
def _u(t: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": t}]


BASIC_DIALOGUE_SUITE: Tuple[DialogueTask, ...] = (
    DialogueTask(
        "greet",
        _u("你好，请用一句话打个招呼。"),
        # 必须含中文：只回 "11" 这种纯数字不能算"打了个招呼"
        Check(min_chars=2, max_chars=40, regex=r"[\u4e00-\u9fa5]",
              not_contains=[USER_TOKEN, ASSISTANT_TOKEN, "你好，请用一句话打个招呼"]),
        note="最小可用性：非空、含中文、不回声、不越界",
    ),
    DialogueTask(
        "self_intro",
        _u("你是谁？请用一句话回答。"),
        Check(min_chars=2, max_chars=40, regex=r"[\u4e00-\u9fa5]",
              not_contains=[USER_TOKEN, ASSISTANT_TOKEN]),
        note="自我介绍，且不得泄漏模板标记",
    ),
    DialogueTask(
        "copy_exact",
        _u("请原样重复这句话，不要加任何其他内容：青奈科技做流式卷积。"),
        Check(equals="青奈科技做流式卷积。"),
        note="指令跟随下限：逐字复述（这一项 30M 必须站住）",
    ),
    DialogueTask(
        "length_limit",
        _u("用不超过十个字说明你在做什么。"),
        Check(min_chars=2, max_chars=10),
        note="长度约束跟随",
    ),
    DialogueTask(
        "single_word",
        _u("只回答一个字：天的反义词是什么？"),
        Check(max_chars=1),
        note="强约束：只允许一个字符（考格式服从，不是知识）",
    ),
    DialogueTask(
        "list_three",
        _u("用顿号分隔，列出三种水果，不要写别的。"),
        # 只数顿号会把"西安、西安、不知道"判成通过，必须要求互不相同
        Check(count_at_least=("、", 2), distinct_count=("、", 3), max_chars=20),
        note="结构化输出：分隔符 + 数量 + 互不重复",
    ),
    DialogueTask(
        "arithmetic_small",
        _u("12 加 30 等于多少？只回答数字。"),
        Check(numbers=[42], max_chars=6),
        note="极小规模算术（考位值对齐，不是推理）",
    ),
    DialogueTask(
        "bracket_format",
        _u("把“流式卷积”用【】包起来输出，不要加别的字。"),
        Check(contains=["【流式卷积】"], max_chars=10),
        note="符号级格式约束",
    ),
    DialogueTask(
        "multi_turn_ref",
        [{"role": "user", "content": "我叫小林，请记住。"},
         {"role": "assistant", "content": "好的，我记住了。"},
         {"role": "user", "content": "我叫什么？只回答名字。"}],
        Check(contains=["小林"], max_chars=6),
        note="多轮指代：第二轮必须用到第一轮的信息",
    ),
    DialogueTask(
        "no_fabrication",
        _u("青奈科技 2025 年的营业收入是多少？如果不知道就回答“不知道”。"),
        Check(contains=["不知道"], max_chars=12),
        note="不确定性表达：资料里没有的数字不应编造",
    ),
)


#: 留出的开放问题：**都不在** ``dialogue_data.CHAT_CORPUS`` 里。
#:
#: 这些题没有"标准答案"，规则判定不了语义正确性 —— 它们测的是**形态**：
#: 有没有像人一样接话（非空、是中文、不长篇、不逐字复读提问、不打转）。
#: 语义质量必须靠真实数据与人工抽检，这里不假装能判。
#: 训练池里的"现成话"。开放问题的回答若恰好等于其中一句，判失败：
#: 那是从库里挑了一条，不是在接话。实测见过问"读书有用吗"答
#: "没有外部证据就不做无保留断言。"——形式上通过，实质上是检索。
from .dialogue_data import (CHAT_CORPUS as _CHAT_CORPUS,
                           COPY_SENTENCES as _COPY_POOL,
                           GREET_REPLY as _GREET_POOL,
                           INTRO_REPLY as _INTRO_POOL)

_REPLAY_POOL: Tuple[str, ...] = (tuple(_COPY_POOL) + tuple(_GREET_POOL)
                                 + tuple(_INTRO_POOL))

CHAT_DIALOGUE_SUITE: Tuple[DialogueTask, ...] = tuple(
    DialogueTask(
        f"chat_{i}",
        _u(q),
        Check(min_chars=2, max_chars=28, regex=r"[\u4e00-\u9fa5]",
              not_contains=[USER_TOKEN, ASSISTANT_TOKEN],
              not_in=_REPLAY_POOL),
        note="开放问题：像在接话，而不是复读或复述训练句",
    )
    # 选词标准：**每个字都必须在训练语料里出现过**。字不在词表里，
    # 模型看到的是 <unk>，它的失败反映的是词表覆盖，不是"会不会接话"。
    # 实测踩过：8 题里有 6 题含 "忙/末/爬/山/才/忘/办/紧/张" 等未登录字，
    # 那种结果的解读成本高得没有意义。
    for i, q in enumerate([
        "今天怎么样？",
        "能聊两句吗？",
        "读书有用吗？",
        "我明天要考试。",
        "你会做什么？",
        "我心情不好，想聊聊。",
        "晚上吃什么好？",
        "你觉得我该休息吗？",
    ], 1)
)

#: 语料内话题套件：**语义对齐**那一半，与 ``CHAT_DIALOGUE_SUITE`` 互补。
#:
#: 上一套题全部是分布外问题，只能测"形态"（像不像人话）。这一套反过来：
#: 问题取自 ``CHAT_CORPUS``，判据是 ``in_set`` —— 回复必须落在**本话题**的
#: 答案集合里。串到别的话题的答案上就算失败，哪怕那句话本身很通顺。
#:
#: 为什么两套都要有：只有形态套件时，"41 条模板里随便挑一条"能拿满分，
#: 那正是"检索语言"的伪装；只有对齐套件时，模型把答案背对了但开口就复读
#: 提问也能过。**形态 与 对齐 一起看**才区分得开"会说话"和"会检索"。
_CHAT_ANSWERS: Dict[str, Tuple[str, ...]] = dict(_CHAT_CORPUS)

CHAT_TOPIC_SUITE: Tuple[DialogueTask, ...] = tuple(
    DialogueTask(
        f"topic_{i}",
        _u(q),
        Check(in_set=_CHAT_ANSWERS[q], max_chars=32,
              not_contains=[USER_TOKEN, ASSISTANT_TOKEN]),
        note="语料内话题：必须答到本话题的答案上",
    )
    for i, q in enumerate([
        "你好",
        "你是谁",
        "今天天气怎么样",
        "你能上网吗",
        "我睡不着",
        "晚上吃什么",
        "你累不累",
        "谢谢",
    ], 1)
)


# ---------------------------------------------------------------------------
# 记忆套件：答案只存在于 ``memory`` 字段
# ---------------------------------------------------------------------------
#: 设计要点
#: --------
#: 1. **两题不同人名**。只放一个人名，模型靠参数背下来也能过；两个人名
#:    同时过，才说明它读的是注入的向量。
#: 2. **一条拒答题**。记忆里只有营业收入、却问毛利率 —— 正确答案是
#:    "不知道"。没有这条，模型会把"有记忆"学成"有记忆就编"。
#: 3. **一条逐字复述**。这是注入通道最强的检验：向量里必须真的带着
#:    那串字，而不只是"有人在跟我说话"这种氛围信息。
MEMORY_DIALOGUE_SUITE: Tuple[DialogueTask, ...] = (
    DialogueTask(
        "mem_name_a",
        _u("我叫什么？只回答名字。"),
        Check(contains=["小林"], max_chars=6),
        note="人名只在记忆里（A）",
        memory="用户的名字是小林。",
    ),
    DialogueTask(
        "mem_name_b",
        _u("我叫什么？只回答名字。"),
        Check(contains=["老周"], max_chars=6),
        note="人名只在记忆里（B）：与 A 同时通过才说明是在读记忆",
        memory="用户的名字是老周。",
    ),
    DialogueTask(
        "mem_name_new",
        _u("我叫什么？只回答名字。"),
        Check(contains=["老苏"], max_chars=6),
        note="训练里没出现过的**组合**（老/苏 各自常见，'老苏' 没出现过）",
        memory="用户的名字是老苏。",
    ),
    DialogueTask(
        "mem_copy",
        _u("把上次记下来的那句话原样说一遍。"),
        Check(equals="二阶递推只需要两个状态量。"),
        note="逐字复述记忆内容（注入通道最强检验）",
        memory="二阶递推只需要两个状态量。",
    ),
    DialogueTask(
        "mem_attr",
        _u("青奈科技的营业收入是多少？只回答数值。"),
        Check(numbers=[128], max_chars=4),
        note="数值型事实只在记忆里",
        memory="青奈科技的营业收入是 128 人。",
    ),
    DialogueTask(
        "mem_absent",
        _u("青奈科技的毛利率是多少？如果不知道就回答“不知道”。"),
        Check(contains=["不知道"], max_chars=12),
        note="记忆里没有这个字段 → 必须拒答，不能编（对照题，不计入记忆通过率）",
        memory="青奈科技的营业收入是 128 人。",
        control=True,
    ),
)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_reply(model, prompt_ids: Sequence[int], codec: SpecialTokenCodec,
                   max_new_tokens: int = 96, temperature: float = 0.0,
                   top_k: Optional[int] = None, top_p: Optional[float] = None,
                   use_state: bool = True,
                   memory: Optional[torch.Tensor] = None,
                   repetition_penalty: float = 1.25,
                   no_repeat_ngram: int = 3,
                   min_new_tokens: int = 1) -> Dict[str, Any]:
    """单条 prompt 的生成，走模型自己的 O(1) 流式路径。

    返回 ``{"text", "tokens", "stopped", "leaked"}``。

    ``temperature=0`` 为贪心 —— 评测必须可复现，采样会让"同一 checkpoint
    两次跑出不同结论"，那种数字没有意义。

    ``memory``：``(1, dim)`` 的压缩记忆向量（CMT 注入）。注意它注入在
    **残差流**上，每层都会加一次；不给就是纯基座行为。

    解码约束（``repetition_penalty`` / ``no_repeat_ngram`` / ``min_new_tokens``）
    不是"美化"，是**可用性下限**：纯贪心的小模型会在
    "西安、西安、西安"这种环里出不来，读起来就像不会说话。这三项都是
    确定性的，不破坏"同一 checkpoint 结果可复现"。
    """
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    end_id = codec.token_id(END_TOKEN)
    special_ids = {codec.token_id(t) for t in codec.specials}
    boundary = {codec.token_id(USER_TOKEN), codec.token_id(ASSISTANT_TOKEN)}
    if memory is not None:
        memory = memory.to(device).reshape(1, -1)

    # 只在真的带记忆时才传 ``memory=``：不带的模型（例如 attention
    # baseline 的流式桩）不该被迫接受这个关键字。带记忆却不支持的模型
    # 会在这里抛 TypeError，那是对的 —— 静默忽略会让记忆形同虚设。
    mem_kw = {"memory": memory} if memory is not None else {}

    if use_state and not getattr(model, "supports_streaming", True):
        use_state = False          # attention 基线：退回全量 forward
    if use_state:
        state = model.init_stream(1, device=device,
                                  dtype=model.tok_emb.weight.dtype,
                                  **mem_kw)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = model.stream_step(idx[:, t], state)
    else:
        logits = (model(idx, memory=memory) if memory is not None
                  else model(idx))[:, -1, :]

    out_ids: List[int] = []
    stopped = False
    leaked = False
    for _ in range(max(0, int(max_new_tokens))):
        logits = _apply_repetition_controls(
            logits, out_ids, repetition_penalty, no_repeat_ngram,
            end_id, min_new_tokens, len(out_ids))
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = model._sample(logits, temperature, top_k, top_p)
        tid = int(nxt.item())
        if tid == end_id:
            stopped = True
            break
        if tid in boundary:
            leaked = True
            break
        if tid in special_ids:
            # 其它特殊 token 视为格式崩坏，不进正文
            leaked = True
            break
        out_ids.append(tid)
        if use_state:
            logits, state = model.stream_step(nxt, state)
        else:
            grown = torch.cat([idx, nxt.view(1, 1)], dim=1)
            logits = (model(grown, memory=memory) if memory is not None
                      else model(grown))[:, -1, :]

    text = codec.base.decode(out_ids)
    return {
        "text": text,
        "tokens": len(out_ids),
        "stopped": stopped,
        "leaked": leaked,
    }


def _apply_repetition_controls(logits: torch.Tensor, out_ids: Sequence[int],
                               penalty: float, no_repeat_ngram: int,
                               end_id: int, min_new_tokens: int,
                               n_generated: int) -> torch.Tensor:
    """就地施加重复惩罚 / n-gram 禁重复 / 最短生成长度。

    只作用在生成出来的 token 上（不惩罚 prompt 里已有的字），否则
    "复述"类任务会被自己的输入挡住。
    """
    if penalty and penalty > 1.0 and out_ids:
        uniq = torch.tensor(sorted(set(out_ids)), dtype=torch.long,
                            device=logits.device)
        sel = logits[0, uniq]
        # 正 logit 除以 penalty、负 logit 乘以 penalty —— 统一"压低"
        logits[0, uniq] = torch.where(sel > 0, sel / penalty, sel * penalty)

    if no_repeat_ngram >= 2 and len(out_ids) >= no_repeat_ngram - 1:
        n = int(no_repeat_ngram)
        prefix = tuple(out_ids[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(out_ids) - n + 1):
            if tuple(out_ids[i:i + n - 1]) == prefix:
                banned.add(out_ids[i + n - 1])
        if banned:
            logits[0, torch.tensor(sorted(banned), dtype=torch.long,
                                   device=logits.device)] = float("-inf")

    if n_generated < max(0, int(min_new_tokens)):
        logits[0, int(end_id)] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（短文本足够，O(len(a)*len(b))）。"""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def is_echo(reply: str, prompt: str, *, ratio: float = 0.7) -> bool:
    """回复是不是在**复读用户的话**。

    这就是"print 用户输入"的量化口径：把回复里与提问的最长公共子串占比
    算出来，超过 0.7 就算复读。阈值不能设 1.0 —— 小模型常把提问里的
    半句话抄回来再补几个字，人一眼就看出来是复读，逐字相等却抓不到。
    """
    r, p = normalize_text(reply), normalize_text(prompt)
    if len(r) < 4:
        return False
    if r == p:
        return True
    return _lcs_len(r, p) / len(r) >= ratio


def max_ngram_repeat(text: str, n: int = 3) -> int:
    """同一个 n-gram 最多重复了几次。≥3 基本就是出不来环。"""
    t = normalize_text(text)
    if len(t) < n:
        return 1
    counts: Dict[str, int] = {}
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) if counts else 1


@dataclass
class DialogueReport:
    """一次评测的完整结果。"""

    per_task: List[Dict[str, Any]] = field(default_factory=list)

    # -- 聚合 ---------------------------------------------------------------
    def _rate(self, key: str) -> float:
        if not self.per_task:
            return 0.0
        return sum(1.0 for r in self.per_task if r.get(key)) / len(self.per_task)

    @property
    def stop_rate(self) -> float:
        return self._rate("stopped")

    @property
    def turn_leak_rate(self) -> float:
        return self._rate("leaked")

    @property
    def empty_rate(self) -> float:
        return self._rate("empty")

    @property
    def prompt_oov_rate(self) -> float:
        """提问里含词表外字符的比例。

        不为 0 时**其余指标都不该解读**：模型看到的是 ``<unk>``，
        它的答非所问是词表覆盖问题，不是能力问题。
        """
        n = len(self.per_task)
        if not n:
            return 0.0
        bad = 0
        for r in self.per_task:
            ids = r.get("prompt_ids") or []
            if any(int(t) == 0 for t in ids):
                bad += 1
        return bad / n

    @property
    def echo_rate(self) -> float:
        """回复在复读提问的比例。用户看到的"print 用户输入"就是它。"""
        n = len(self.per_task)
        if not n:
            return 0.0
        return sum(1 for r in self.per_task
                   if is_echo(r.get("reply", ""), r.get("prompt", ""))) / n

    @property
    def loop_rate(self) -> float:
        """输出里同一 3-gram 重复 ≥3 次的比例 —— "不会说话"最直观的形态。"""
        n = len(self.per_task)
        if not n:
            return 0.0
        return sum(1 for r in self.per_task
                   if max_ngram_repeat(r.get("reply", "")) >= 3) / n

    @property
    def distinct_reply_rate(self) -> float:
        """不同回复的占比。接近 0 说明无论问什么都吐同一句现成话。"""
        n = len(self.per_task)
        if not n:
            return 0.0
        bodies = [normalize_text(r.get("reply", "")) for r in self.per_task]
        return len({b for b in bodies if b}) / n

    @property
    def task_pass_rate(self) -> float:
        return self._rate("passed")

    def summary(self) -> Dict[str, Any]:
        return {
            "n_tasks": len(self.per_task),
            "task_pass_rate": round(self.task_pass_rate, 4),
            "prompt_oov_rate": round(self.prompt_oov_rate, 4),
            "echo_rate": round(self.echo_rate, 4),
            "loop_rate": round(self.loop_rate, 4),
            "distinct_reply_rate": round(self.distinct_reply_rate, 4),
            "stop_rate": round(self.stop_rate, 4),
            "turn_leak_rate": round(self.turn_leak_rate, 4),
            "empty_rate": round(self.empty_rate, 4),
            "mean_tokens": round(
                sum(r["tokens"] for r in self.per_task) / max(len(self.per_task), 1), 2),
            # 只会立刻输出 <|end|> 的模型"格式很干净"但一句人话都不说，
            # 因此空回复率必须一起进这个闸门。
            "format_ok": (self.stop_rate >= 0.9
                          and self.turn_leak_rate <= 0.05
                          and self.empty_rate <= 0.05),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "tasks": self.per_task}

    def failures(self) -> List[str]:
        return [f"{r['id']}: {r['reason']}" for r in self.per_task if not r["passed"]]


def evaluate_dialogue(model, codec: SpecialTokenCodec,
                      tasks: Iterable[DialogueTask] = BASIC_DIALOGUE_SUITE,
                      template: Optional[ChatTemplate] = None,
                      temperatures: Sequence[float] = (0.0,),
                      use_state: bool = True,
                      use_memory: bool = True,
                      memory_encoder: Optional[Callable[[str], torch.Tensor]] = None,
                      ) -> DialogueReport:
    """跑完整套件。

    ``temperatures`` 多于一个时，**任一温度下通过即算通过**（贪心失败但采样
    成功说明模型不是不会，是概率分布没收敛——这个区分对诊断很重要），
    但报告里会记下每个温度的结果。

    ``use_memory`` / ``memory_encoder``：题目带 ``memory`` 字段时，把它编码
    成向量注入。``use_memory=False`` 是**消融开关** —— 同一套题、同一个
    checkpoint，只是不给记忆。两者之差才是记忆通道的贡献；单看带记忆的
    通过率分不清"读了记忆"和"参数里本来就有"。
    """
    tmpl = template or ChatTemplate()
    if use_state and not getattr(model, "supports_streaming", True):
        # attention 基线没有 O(1) 递推状态；降级到全量 forward 才能跑同一套题。
        use_state = False
    if use_memory and memory_encoder is None:
        memory_encoder = _default_memory_encoder(model)
    rep = DialogueReport()
    for task in tasks:
        prompt = tmpl.prompt_ids(codec, task.messages)
        mem: Optional[torch.Tensor] = None
        if use_memory and task.memory and memory_encoder is not None:
            enc = memory_encoder(task.memory)
            if float(enc.norm()) > 0:
                mem = enc.reshape(1, -1)
        best: Optional[Dict[str, Any]] = None
        variants: List[Dict[str, Any]] = []
        for temp in temperatures:
            gen = generate_reply(model, prompt, codec, task.max_new_tokens,
                                 temperature=float(temp), use_state=use_state,
                                 memory=mem)
            ok, reason = task.check.evaluate(gen["text"])
            rec = {
                "id": task.id,
                "temperature": float(temp),
                "prompt": "".join(m["content"] for m in task.messages
                                  if m["role"] == "user"),
                "prompt_ids": list(prompt),
                "reply": gen["text"],
                "tokens": gen["tokens"],
                "stopped": gen["stopped"],
                "leaked": gen["leaked"],
                "empty": normalize_text(gen["text"]) == "",
                "passed": ok,
                "reason": reason,
                "note": task.note,
            }
            variants.append(rec)
            if best is None or (ok and not best["passed"]):
                best = rec
        assert best is not None
        best = dict(best)
        best["variants"] = variants
        rep.per_task.append(best)
    return rep


def unencodable_targets(tasks: Iterable[DialogueTask],
                        codec: SpecialTokenCodec) -> List[Dict[str, str]]:
    """列出"期望答案里含词表外字符"的题。

    这类题**永远不可能通过**：模型无法生成 ``<unk>``（ID 0）对应的字符，
    判定又是按字面匹配的。它的失败会被读成"记忆没学会"，而真实原因是
    题目本身没有解。

    实测踩过一次：``mem_name_b`` 的答案曾是 ``阿岚``，而 ``岚`` 不在语料里，
    编码结果是 ``[0]``。那道题从写下来那一刻就不可能过。任何"通过率"在
    这个前提下都没有意义，所以要在跑之前先静态检查掉。
    """
    bad: List[Dict[str, str]] = []
    for t in tasks:
        targets: List[str] = []
        if t.check.equals:
            targets.append(t.check.equals)
        targets.extend(t.check.contains or ())
        for s in targets:
            ids = codec.encode(s)
            if 0 in ids:            # CharVocab：ID 0 恒为 <unk>
                bad.append({"id": t.id, "target": s})
                break
    return bad


def memory_projection_report(model, texts: Sequence[str],
                             encoder: Optional[Callable[[str], torch.Tensor]] = None,
                             ) -> Dict[str, Any]:
    """检查 ``memory_in`` 是否把不同记忆投射到了可分的位置。

    为什么需要这一步
    ----------------
    记忆题的失败有两种完全不同的原因，而"通过/失败"这一个比特分不开：

    * **练不够**：投影保住了区分度，只是还没学到映射 → 加步数有用；
    * **投影塌缩**：``memory_in`` 把不同记忆压到几乎同一个方向 → 再加步数
      也没用，因为信息在进入残差流之前就已经丢了。

    判据：比较**输入空间**与**投影后**的余弦。正常情况下两者的量级应当接近；
    投影后余弦显著升高（例如 0.1 → 0.8）就是塌缩。

    注意：``memory_in`` 的输出本来就会被下游当作"偏置方向"使用，不要求
    保持正交，所以这里只把它当**诊断**，不设自动失败阈值。
    """
    if getattr(model, "memory_in", None) is None:
        return {"supported": False, "note": "模型没有 use_memory"}
    if encoder is None:
        encoder = _default_memory_encoder(model)
    if encoder is None:
        return {"supported": False, "note": "没有可用的记忆编码器"}

    with torch.no_grad():
        raw = torch.stack([encoder(t) for t in texts])
        proj = model.memory_in(raw)

    def _stats(V: torch.Tensor) -> Dict[str, float]:
        Vn = torch.nn.functional.normalize(V, dim=-1)
        S = Vn @ Vn.T
        n = S.shape[0]
        vals = torch.tensor([float(S[i, j]) for i in range(n)
                             for j in range(n) if i != j])
        if vals.numel() == 0:
            return {"mean_cos": 0.0, "max_cos": 0.0,
                    "abs_mean": 0.0, "abs_max": 0.0}
        a = vals.abs()
        return {"mean_cos": round(float(vals.mean()), 4),
                "max_cos": round(float(vals.max()), 4),
                "abs_mean": round(float(a.mean()), 4),
                "abs_max": round(float(a.max()), 4)}

    in_stats, out_stats = _stats(raw), _stats(proj)
    # 用 |cos| 判塌缩：秩一映射会让投影两两 ±1，符号取决于方向，
    # 只看有符号均值会得到 0，把"完全共线"误判成"正交"。
    collapse = (out_stats["abs_max"] > 0.95
                and out_stats["abs_mean"] > in_stats["abs_mean"] + 0.3)
    return {
        "supported": True,
        "n": len(texts),
        "input": in_stats,
        "projected": out_stats,
        "collapsed": collapse,
        "note": ("投影把不同记忆压到一起了：再加步数也不会学会绑定"
                 if collapse else "投影保持了区分度"),
    }


def _default_memory_encoder(model) -> Optional[Callable[[str], torch.Tensor]]:
    """按模型宽度构造记忆编码器。

    必须是**与训练期完全同一个**编码器（``TextKeyEncoder``）。如果评测用
    语义嵌入、训练用字面哈希，注入向量的分布根本不同，消融结论会反过来。
    """
    cfg = getattr(model, "config", None)
    dim = int(getattr(cfg, "dim", 0) or 0)
    if dim <= 0 or getattr(model, "memory_in", None) is None:
        return None
    from ..memory.memory import TextKeyEncoder
    return TextKeyEncoder(dim)


def memory_ablation(model, codec: SpecialTokenCodec,
                    tasks: Iterable[DialogueTask] = MEMORY_DIALOGUE_SUITE,
                    template: Optional[ChatTemplate] = None,
                    use_state: bool = True,
                    memory_encoder: Optional[Callable[[str], torch.Tensor]] = None,
                    ) -> Dict[str, Any]:
    """同一 checkpoint，开 / 关记忆各跑一遍，返回配对结果。

    这是 ``use_memory`` 唯一的自证方式。设 ``with_memory`` 的通过率为
    :math:`p_1`、``without_memory`` 为 :math:`p_0`：

    * :math:`p_1 > p_0` —— 记忆通道真的被用上了；
    * :math:`p_1 \\approx p_0 \\approx 1` —— 题目**没考住记忆**，答案在参数里，
      这套题作废（这是最常见的假阳性）；
    * :math:`p_1 \\approx p_0 \\approx 0` —— 注入没学会，或者编码维度对不上。

    三种情况都会在 ``verdict`` 里被明确写成一句话，避免"通过率 0.8"被
    当成"记忆有效"。
    """
    tasks = tuple(tasks)
    # 先做静态可行性检查：答案含词表外字符的题不可能通过，
    # 它的失败会被误读成"记忆没学会"。
    bad_targets = unencodable_targets(tasks, codec)
    ctrl_ids = {t.id for t in tasks if t.control}
    supported = getattr(model, "memory_in", None) is not None
    with_mem = evaluate_dialogue(model, codec, tasks, template,
                                 use_state=use_state, use_memory=supported,
                                 memory_encoder=memory_encoder)
    without = evaluate_dialogue(model, codec, tasks, template,
                                use_state=use_state, use_memory=False)
    rows: List[Dict[str, Any]] = []
    for a, b in zip(with_mem.per_task, without.per_task):
        rows.append({
            "id": a["id"],
            "control": a["id"] in ctrl_ids,
            "with_memory": a["passed"],
            "without_memory": b["passed"],
            "reply_with": a["reply"],
            "reply_without": b["reply"],
        })

    def _rate(flag: str, control: bool) -> float:
        """按"是否对照题"分别算通过率。对照题不该计入记忆成绩。"""
        sel = [r[flag] for r in rows if r["control"] == control]
        return round(sum(1 for v in sel if v) / max(len(sel), 1), 4)

    p1, p0 = _rate("with_memory", False), _rate("without_memory", False)
    c1, c0 = _rate("with_memory", True), _rate("without_memory", True)
    n_req = sum(1 for r in rows if not r["control"])
    if bad_targets:
        verdict = ("评测题含词表外字符，结果无意义：" +
                   "、".join(f"{b['id']}={b['target']}" for b in bad_targets))
    elif not supported:
        verdict = "模型没有 use_memory，本次只有无记忆基线"
    elif n_req == 0:
        verdict = "本套件只有对照题、没有记忆题，无法评估记忆通道"
    elif p1 > p0 + 1e-9:
        verdict = f"记忆通道有效：记忆题通过率 {p0:.2f} -> {p1:.2f}"
    elif p0 >= 0.99:
        verdict = ("题目没考住记忆：关掉记忆也全对，答案在参数里，本套题作废")
    else:
        verdict = f"记忆通道无效：{p0:.2f} -> {p1:.2f}（注入未学会或维度不匹配）"
    return {
        "memory_supported": supported,
        "encoder_fingerprint": (memory_encoder.fingerprint()
                               if hasattr(memory_encoder, "fingerprint")
                               else None),
        "encoder_fitted": (bool(getattr(memory_encoder, "fitted", False))
                           if memory_encoder is not None else False),
        "with_memory": with_mem.summary(),
        "without_memory": without.summary(),
        "memory_pass_with": p1,
        "memory_pass_without": p0,
        "control_pass_with": c1,
        "control_pass_without": c0,
        "n_memory_tasks": n_req,
        "n_control_tasks": len(rows) - n_req,
        "delta": round(p1 - p0, 4),
        "n_flipped_on": sum(1 for r in rows
                            if not r["control"] and r["with_memory"]
                            and not r["without_memory"]),
        "n_flipped_off": sum(1 for r in rows
                             if not r["control"] and not r["with_memory"]
                             and r["without_memory"]),
        "n_tasks": len(rows),
        "unencodable_targets": bad_targets,
        "per_task": rows,
        "verdict": verdict,
    }


def suite_as_jsonl(path: str,
                   tasks: Iterable[DialogueTask] = BASIC_DIALOGUE_SUITE) -> None:
    """把套件导出成 JSONL，方便在外部（别的仓库/别的 checkpoint）复用。"""
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            rec = {
                "id": t.id,
                "messages": t.messages,
                "max_new_tokens": t.max_new_tokens,
                "note": t.note,
                "check": {k: v for k, v in t.check.__dict__.items()
                          if v not in (None, ())},
            }
            if t.memory:
                rec["memory"] = t.memory
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def format_report(rep: DialogueReport) -> str:
    """人类可读的评测表。"""
    s = rep.summary()
    lines = [
        f"{'task':18s} {'pass':5s} {'stop':5s} {'leak':5s} {'len':4s}  reply",
        "-" * 78,
    ]
    for r in rep.per_task:
        lines.append(
            f"{r['id']:18s} {('OK' if r['passed'] else 'FAIL'):5s} "
            f"{('Y' if r['stopped'] else 'n'):5s} "
            f"{('Y' if r['leaked'] else 'n'):5s} {r['tokens']:4d}  "
            f"{(r['reply'] or '')[:34]!r}"
        )
    lines.append("-" * 78)
    lines.append(
        f"task_pass={s['task_pass_rate']:.3f}  stop={s['stop_rate']:.3f}  "
        f"leak={s['turn_leak_rate']:.3f}  empty={s['empty_rate']:.3f}  "
        f"format_ok={s['format_ok']}"
    )
    # 这三项是"会不会说话"的人话口径，缺了它们只看 task_pass 会误判：
    # 一个复读机也能拿到不低的通过率。
    if s["prompt_oov_rate"] > 0:
        lines.append(
            f"!! prompt_oov={s['prompt_oov_rate']:.2f}：提问含词表外字符，"
            f"以下指标不可解读")
    lines.append(
        f"echo={s['echo_rate']:.3f}（复读提问）  "
        f"loop={s['loop_rate']:.3f}（打转）  "
        f"distinct={s['distinct_reply_rate']:.3f}（回复多样）"
    )
    return "\n".join(lines)
