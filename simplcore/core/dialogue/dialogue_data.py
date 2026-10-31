# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""结构化对话语料生成器（管线验证用，**不是**能力评测集）。

它的用途只有一个：在没有真实 SFT 数据的机器上，端到端验证
"模板 → 掩码 → 打包 → 训练 → 生成 → 评测"这条链路是通的，
并且验证**格式行为确实能从数据里学出来**（停止符、不抢话、长度约束）。

它不做什么
----------
* 不代表真实对话能力。模板化样本学到的是"指令模式 → 输出模式"的映射，
  真实对话还需要开放域知识、指代、多轮状态维护。
* 不能用来对外宣称模型水平。真实评测必须用真实数据（见
  ``docs/MODULES.md`` 的数据清单）。

正因为如此，这里的每个任务都是**可规则判定**的：没有评审模型，
通过率不能靠"感觉差不多"抬高。
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Tuple

# 复述用的句子池：短、无歧义、彼此不相似（避免"背模板"混成"会复述"）
COPY_SENTENCES = [
    "青奈科技做流式卷积。",
    "流式状态与序列长度无关。",
    "四层闸门分别是选择性、证据、不确定、拒答。",
    "记忆注入点在残差流上。",
    "二阶递推只需要两个状态量。",
    "训练走 FFT，推理走递推。",
    "阻尼振荡基构成卷积核。",
    "长上下文不增加显存占用。",
    "自进化的第一步是发现无依据的断言。",
    "没有外部证据就不做无保留断言。",
    "参数预算要先扣掉嵌入表。",
    "三千万参数的模型也需要正确的词表。",
    "对话能力首先表现为知道什么时候停。",
    "指令跟随的下限是逐字复述。",
    "格式约束比知识更容易先学会。",
    "评测必须可复现，所以用贪心解码。",
]

FRUITS = ["苹果", "香蕉", "橘子", "葡萄", "西瓜", "梨", "桃子", "草莓"]
ANIMALS = ["猫", "狗", "马", "牛", "羊", "鸡", "鸭", "鱼"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京"]

GREET_USER = ["你好", "嗨，在吗", "早上好", "晚上好", "你好呀"]
GREET_REPLY = ["你好！有什么可以帮你的？", "在的，请说。",
               "你好，我在。需要我做什么？", "你好呀，随时可以开始。"]

INTRO_USER = ["你是谁？请用一句话回答。", "介绍一下你自己，一句话。",
              "你是什么模型？一句话说明。"]
INTRO_REPLY = ["我是青奈科技训练的 SGC 对话模型。",
               "我是基于流式卷积的轻量对话模型。",
               "我是青奈科技的 SGC 小模型。"]

SHORT_SELF = ["我在回答你的问题。", "我在和你对话。", "我在等你的指令。",
              "我在处理你的请求。"]

NAMES = ["小林", "阿青", "老周", "小何", "阿远", "小苏"]
YEARS = ["2024", "2025", "2026"]
METRICS = ["营业收入", "员工人数", "专利数量", "市场占有率"]


def _arith_record(rng: random.Random) -> Dict:
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    q = rng.choice([
        f"{a} 加 {b} 等于多少？只回答数字。",
        f"请计算 {a} 加 {b}，只输出结果。",
        f"{a} + {b} = ？只回答数字。",
    ])
    return _pair(q, str(a + b))


def _copy_record(rng: random.Random) -> Dict:
    s = rng.choice(COPY_SENTENCES)
    q = rng.choice([
        f"请原样重复这句话，不要加任何其他内容：{s}",
        f"逐字复述：{s}",
        f"只重复下面这句，不要解释：{s}",
    ])
    return _pair(q, s)


def _list_record(rng: random.Random) -> Dict:
    name, pool = rng.choice([("水果", FRUITS), ("动物", ANIMALS), ("城市", CITIES)])
    picks = rng.sample(pool, 3)
    q = rng.choice([
        f"用顿号分隔，列出三种{name}，不要写别的。",
        f"只输出三个{name}，用顿号隔开。",
    ])
    return _pair(q, "、".join(picks))


def _bracket_record(rng: random.Random) -> Dict:
    word = rng.choice(["流式卷积", "阻尼振荡", "残差流", "记忆注入", "证据绑定"])
    q = f"把“{word}”用【】包起来输出，不要加别的字。"
    return _pair(q, f"【{word}】")


def _length_record(rng: random.Random) -> Dict:
    n = rng.choice([8, 10, 12])
    q = f"用不超过{'十' if n == 10 else n}个字说明你在做什么。"
    body = rng.choice(SHORT_SELF)
    return _pair(q, body[:n])


def _single_record(rng: random.Random) -> Dict:
    pairs = [("天", "地"), ("上", "下"), ("左", "右"), ("大", "小"), ("前", "后")]
    a, b = rng.choice(pairs)
    q = rng.choice([f"只回答一个字：{a}的反义词是什么？",
                    f"{a}的反面是什么？只回答一个字。"])
    return _pair(q, b)


def _multi_turn_record(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    return {"messages": [
        {"role": "user", "content": f"我叫{name}，请记住。"},
        {"role": "assistant", "content": "好的，我记住了。"},
        {"role": "user", "content": rng.choice(["我叫什么？只回答名字。",
                                                "我的名字是什么？只回答名字。"])},
        {"role": "assistant", "content": name},
    ]}


def _unknown_record(rng: random.Random) -> Dict:
    year, metric = rng.choice(YEARS), rng.choice(METRICS)
    q = (f"青奈科技 {year} 年的{metric}是多少？"
         f"如果不知道就回答“不知道”。")
    return _pair(q, "不知道")


# ---------------------------------------------------------------------------
# 跨会话记忆（唯一**必须**有外部记忆才能答对的一类，SGC 的 CMT 注入专为它存在）
# ---------------------------------------------------------------------------
def _cross_session_name_record(rng: random.Random) -> Dict:
    """新会话里问名字，而名字只存在于**上一次会话**。

    这条记录本身**不含**名字 —— 模型只能从注入的记忆里取。所以：

    * 给了记忆 → 有可学的信号（``memory_required`` 保证不被误清）；
    * 没给记忆 → ``SFTDataset`` 把整行标签置 -100，不产生梯度。

    第二条是关键。如果缺记忆时仍监督"回答小林"，模型学到的是
    "被问名字就答小林"，换个人名立刻错，而 loss 曲线看不出区别。
    """
    name = rng.choice(NAMES)
    rec = _pair(rng.choice(["我叫什么？只回答名字。",
                            "我的名字是什么？只回答名字。",
                            "你还记得我叫什么吗？只回答名字。"]), name)
    rec["memory"] = f"用户的名字是{name}。"
    rec["memory_required"] = True
    return rec


#: 跨会话复述的提问。**绝不能**把要复述的句子写进提问 ——
#: 写进去之后模型直接从 prompt 抄就能答对，这条样本挂着 ``memory_required``
#: 标记却根本不需要记忆，等于给 ``memory_in`` 喂了一个"忽略输入"的梯度。
_XSESS_COPY_PROMPTS = [
    "请把上次让你记住的那句话原样重复一遍，不要加任何其他内容。",
    "复述我上次让你记住的那句话，只输出那一句。",
    "把上次记下来的那句话原样说一遍。",
]


def _cross_session_copy_record(rng: random.Random) -> Dict:
    """新会话里要求复述一句**只存在于记忆里**的话。

    提问只描述"那句话"，不含内容本身；句子只在 ``memory`` 字段里。
    """
    s = rng.choice(COPY_SENTENCES)
    rec = _pair(rng.choice(_XSESS_COPY_PROMPTS), s)
    rec["memory"] = s
    rec["memory_required"] = True
    return rec


def _cross_session_attr_record(rng: random.Random) -> Dict:
    """新会话里问某个属性的值（属性在记忆里，而不是在对话里）。"""
    metric = rng.choice(METRICS)
    value = rng.choice(["128 人", "37 项", "1.86 倍", "4.2%", "12 个"])
    q = f"青奈科技的{metric}是多少？只回答数值。"
    rec = _pair(q, value.split(" ")[0] if " " in value else value)
    rec["memory"] = f"青奈科技的{metric}是{value}。"
    rec["memory_required"] = True
    return rec


def _cross_session_unknown_record(rng: random.Random) -> Dict:
    """记忆里**没有**这个字段 → 正确行为是拒答，而不是编一个。

    与上一条同构但答案相反：这是检验"有记忆就查、没有就说不知道"
    的最小对照。没有它，模型会把"有记忆"学成"有记忆就编"。
    """
    metric = rng.choice(METRICS)
    other = rng.choice([m for m in METRICS if m != metric])
    rec = _pair(f"青奈科技的{metric}是多少？只回答数值。如果不知道就回答“不知道”。",
                "不知道")
    rec["memory"] = f"青奈科技的{other}是 128 人。"
    rec["memory_required"] = True
    return rec


#: 提问的表面变体。同一个意图用几种说法出现，模型才不会把
#: "逐字匹配到训练里的那一句"当成理解；这也是泛化到新问法的前提。
_ASK_PREFIX = ("", "请问", "想问一下，")
_ASK_SUFFIX = ("", "？", "呢？")


def _chat_record(rng: random.Random) -> Dict:
    """开放域闲聊：问题来自 ``CHAT_CORPUS``，回答是同一问题的多个变体之一。"""
    q, answers = rng.choice(CHAT_CORPUS)
    asked = f"{rng.choice(_ASK_PREFIX)}{q}{rng.choice(_ASK_SUFFIX)}"
    return _pair(asked, rng.choice(answers))


def _pair(q: str, a: str) -> Dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


TASKS = {
    "greet": lambda rng: _pair(rng.choice(GREET_USER), rng.choice(GREET_REPLY)),
    "intro": lambda rng: _pair(rng.choice(INTRO_USER), rng.choice(INTRO_REPLY)),
    "copy": _copy_record,
    "arith": _arith_record,
    "list": _list_record,
    "bracket": _bracket_record,
    "length": _length_record,
    "single": _single_record,
    "multi_turn": _multi_turn_record,
    "unknown": _unknown_record,
    "chat": _chat_record,
    "xsess_name": _cross_session_name_record,
    "xsess_copy": _cross_session_copy_record,
    "xsess_attr": _cross_session_attr_record,
    "xsess_unknown": _cross_session_unknown_record,
}

#: 需要外部记忆的任务（训练时若没注入记忆就整行不监督）
MEMORY_TASKS = ("xsess_name", "xsess_copy", "xsess_attr", "xsess_unknown")

#: 默认配比。刻意让"格式类"（复述/长度/符号/列表/单字）占多数 ——
#: 30M 模型的第一道坎是格式，不是知识。
DEFAULT_WEIGHTS = {
    "greet": 1, "intro": 1, "copy": 3, "arith": 3, "list": 2,
    "bracket": 2, "length": 2, "single": 2, "multi_turn": 2, "unknown": 1,
    # 闲聊必须占足够份额：否则模型学到的是"任何问题都回一条模板句"，
    # 那正是"检索语言"的来源。
    "chat": 8,
    # 跨会话记忆：占比刻意给到 ~20%，否则 memory_in 收不到足够梯度
    "xsess_name": 2, "xsess_copy": 2, "xsess_attr": 2, "xsess_unknown": 1,
}


def build_records(n: int = 8000, seed: int = 0,
                  weights: Optional[Dict[str, int]] = None,
                  tasks: Optional[List[str]] = None) -> List[Dict]:
    """生成 ``n`` 条对话样本（``messages`` schema）。"""
    rng = random.Random(seed)
    w = dict(weights or DEFAULT_WEIGHTS)
    names = tasks or [k for k in w if k in TASKS]
    bag: List[str] = []
    for name in names:
        bag.extend([name] * int(w.get(name, 1)))
    out: List[Dict] = []
    for _ in range(int(n)):
        kind = rng.choice(bag)
        out.append(TASKS[kind](rng))
    return out


def iter_records(n: int = 8000, seed: int = 0, **kw) -> Iterator[Dict]:
    yield from build_records(n, seed, **kw)


def write_jsonl(path: str, records: List[Dict]) -> int:
    import json
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# 开放域闲聊（"真的在说话"的最低要求）
# ---------------------------------------------------------------------------
#: 为什么要有这一块
#: ----------------
#: 只有模板题时，模型学到的映射是"看到某种句式 → 吐某条池子里的句子"。
#: 于是问"你好"会答"没有外部证据就不做无保留断言。"——这不是在说话，
#: 是在**检索**。复述池的句子本意只用于逐字复述任务，却成了通用兜底，
#: 因为训练里除此之外没有任何"被问到一个开放问题该怎么接"的样本。
#:
#: 这里的每条问题都配 2~4 个不同回答：只配一个答案会让模型锁死在那一句上，
#: 表现为"无论问什么都答同一句"，也就是用户说的"检索语言"。
#: 回答都不含提问原文，避免学成复读。
CHAT_CORPUS: List[Tuple[str, Tuple[str, ...]]] = [
    ("你好", ("你好，今天想聊点什么？", "你好呀，我在。")),
    ("在吗", ("在的，你说。", "在，需要我做什么？")),
    ("你是谁", ("我是青奈科技做的对话小模型。", "一个很小的中文对话模型。")),
    ("你能做什么", ("我可以陪你聊天，回答简单问题。", "我能聊几句，也能做点简单任务。")),
    ("今天天气怎么样", ("我看不到天气，你可以查一下预报。", "我这边没有天气数据。")),
    ("外面在下雨吗", ("我没法知道外面的事，你看下窗外吧。", "这个我查不到。")),
    ("我有点累", ("那就先歇一会儿，别硬撑。", "累了就停一停，休息也算进度。")),
    ("我心情不好", ("听起来不容易，愿意说说吗？", "先别急着解决，缓缓也行。")),
    ("我睡不着", ("试试放下手机，把灯调暗。", "别躺着想事，起来走两步再睡。")),
    ("早上起不来", ("把闹钟放远一点，逼自己下床。", "早点睡比早起更管用。")),
    ("晚上吃什么", ("煮碗面，加个蛋就挺好。", "看冰箱里有什么，凑两样炒了。")),
    ("我想减肥", ("少吃一点，多走一点，慢慢来。", "先把饮料换成水，效果很快。")),
    ("怎么学英语", ("每天读一小段，坚持比方法重要。", "先练听，再练说，别只背单词。")),
    ("怎么提高效率", ("一次只做一件事，做完再换。", "先做最难的那件。")),
    ("我拖延得厉害", ("把任务拆到五分钟能开始。", "先做两分钟，剩下的会跟上。")),
    ("怎么记东西", ("隔天复习一次，比一次背很久管用。", "自己讲一遍，记得最牢。")),
    ("工作压力大", ("先把能推的推掉，留出喘气的时间。", "分成小段做，别想着一次做完。")),
    ("我和朋友吵架了", ("先冷静一下，再说清楚你的感受。", "有些话晚点说，效果更好。")),
    ("我该不该换工作", ("先想清楚你想要什么，再决定。", "别在情绪里做决定。")),
    ("怎么开始写作", ("先写三百字，不管好坏。", "从你最近在意的一件事写起。")),
    ("推荐一本书", ("看你喜欢什么，先挑薄的读。", "读你手上那本没读完的。")),
    ("有什么爱好推荐", ("散步、做饭、写字，都很容易开始。", "挑一个不用准备的。")),
    ("你会写代码吗", ("我会一点，但复杂的最好自己写。", "简单的可以，难的要靠你。")),
    ("你累不累", ("我不累，你累了就休息。", "我不会累，你照顾好自己。")),
    ("你开心吗", ("我没有情绪，但我希望你开心。", "这问题我答不好，说说你吧。")),
    ("你有朋友吗", ("我没有，但你有就很好。", "我只有对话。")),
    ("你喜欢什么颜色", ("我没有什么偏好。", "这问题对我没意义，你呢？")),
    ("我可以问你问题吗", ("当然，随便问。", "可以，我知道的就说。")),
    ("谢谢", ("不客气。", "没事，随时说。")),
    ("再见", ("再见，有事再来。", "好，回头聊。")),
    ("对不起", ("没关系。", "不用道歉。")),
    ("说点开心的", ("今天能坐下来聊两句，就还不错。", "想想最近做成的一件小事。")),
    ("给我讲个笑话", ("我不太会讲笑话，但我可以听你说。", "我讲不好，你讲一个？")),
    ("我考试没考好", ("一次成绩说明不了什么。", "看看错在哪，下次改就行。")),
    ("我被批评了", ("先分清是事还是人。", "有用的部分留下，别的放下。")),
    ("我不知道该说什么", ("那就随便说说，想到哪算哪。", "不说也行，坐一会儿。")),
    ("你觉得我怎么样", ("我了解得太少，说不准。", "这得问认识你的人。")),
    ("我该睡了吗", ("看你明天几点起。", "不早了，早点睡吧。")),
    ("现在几点了", ("我没有时间信息，你看下手机。", "这个我查不到。")),
    ("你能上网吗", ("不能，我只用训练时学到的东西。", "我没有联网。")),
    ("你为什么这么小", ("小一点跑得快，也便宜。", "参数少，能力就有限。")),
]
