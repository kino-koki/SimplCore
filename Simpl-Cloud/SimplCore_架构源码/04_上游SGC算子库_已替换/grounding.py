# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""证据绑定层（evidence binding）—— L4 拒答门的标签来源。

为什么需要这一层
----------------
四层闸门的 L4 需要一个标签：**这句话有没有外部证据支撑**。
``abstain.py`` 的契约写得很清楚 —— 标签必须来自外部证据，不能由模型自己生成，
否则"我认为我有依据"只是置信度估计的平方，闸门白做。

但在本轮之前，仓库里**没有任何东西能产出这个标签**：``abstention_loss`` 收
``grounded`` 张量，却没有人构造它。于是 L4 头永远训不起来，抗幻觉只有结构没有监督。
本模块补上这一环。

judge
-----
给定「一句话」与「一组候选证据片段」，本模块做**逐字归因**，用两个互补判据：

* ``coverage``：归一化后**字符级最长公共子序列**占比 ``LCS(claim, span)/len(claim)``；
* ``ngram``：claim 的**二元组（bigram）落在 span 里的比例**；
* ``numbers``：claim 里出现的**每一个数字**都必须在 span 里出现；
* ``polarity``：claim 与 span 的**否定词计数必须相等**。

只看 LCS 会被"共享汉字"抬高 —— 一句纯编造的中文与任意中文片段常有 0.3–0.5 的
LCS 覆盖率（"的""在""规模""数据"这类高频片段贡献的）。后三条判据补的是
文本重叠**结构性看不见**的三类篡改：

* 单字/双字替换（bigram 判据）；
* **数字篡改**（``1.86×`` → ``2.86×``，覆盖率仍有 0.95，但这是最危险的一类编造）；
* **否定翻转**（"与长度无关" → "与长度有关"，一字之差、语义相反）。

因此：

    supported = exact  or  (coverage >= min_coverage and ngram >= min_ngram
                            and numbers_ok and polarity_ok)

默认 ``min_coverage=0.62``、``min_ngram=0.35``。阈值是**先验**，
由 ``scripts/eval_grounding.py`` 用构造标签实测精确率/召回率后写入 JSON。

诚实边界（必须写在任何引用本模块的结论旁边）
--------------------------------------------
1. 它验证的是**文本层面的可归因性**，不是**语义蕴含**。一句话可以逐字出现在
   证据里而在语境上被误用（反讽、否定、条件句）。本模块**不能**发现这类错误。
2. 因此它生产的标签是**必要不充分**条件：``supported=True`` 不等于"这句话正确"，
   只等于"这句话的文本依据在证据里找得到"。
3. 覆盖率阈值是**先验**，不是调出来的最优值；``scripts/eval_grounding.py``
   用构造标签（逐字复制 / 数字篡改 / 跨文档拼接）实测它的精确率与召回率，
   数字写在产出的 JSON 里，不写在这里。

本模块零第三方依赖（stdlib only），可脱离 torch 单独使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "EvidenceSpan",
    "Claim",
    "Binding",
    "EvidenceIndex",
    "normalize",
    "lcs_length",
    "coverage_of",
    "bigram_coverage_of",
    "extract_numbers",
    "numbers_supported",
    "negation_count",
    "polarity_matches",
    "bind_claim",
    "bind_claims",
    "split_claims",
    "attribution_coverage",
    "unsupported_claims",
    "grounded_labels",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MIN_NGRAM",
]

#: 默认覆盖率阈值。先验取值，非调参结果（见模块 docstring 第 3 条）。
DEFAULT_MIN_COVERAGE = 0.62
#: 默认二元组命中率阈值（与 coverage 同时满足才算 supported）。
DEFAULT_MIN_NGRAM = 0.35

# 归一化时丢弃的字符：空白 + 常见中英标点（保留字母/数字/CJK）
_PUNCT = " \t\r\n\u3000，。、；：？！“”‘’（）《》〈〉【】…—·,.;:?!\"'()[]{}<>-_/\\|`~@#$%^&*+="
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT) + "]+")

# 断句：中文句末标点 + 英文句末标点 + 换行
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """归一化：去空白与标点，ASCII 转小写。

    中文不做繁简/异体归一（那需要词表，会引入不可复核的隐式行为），
    因此"繁简混写"会被判为低覆盖 —— 这是**已知的保守偏差**。
    """
    if not text:
        return ""
    return _PUNCT_RE.sub("", text).lower()


def lcs_length(a: str, b: str) -> int:
    """字符级最长公共子序列长度（滚动数组，O(len(a)·len(b)) 时间 / O(len(b)) 空间）。"""
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if ca == cb else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[-1]


def coverage_of(claim_text: str, span_text: str,
                max_claim_chars: int = 400) -> Tuple[float, bool]:
    """``claim`` 被 ``span`` 覆盖的比例，以及是否构成**逐字包含**。

    Returns:
        ``(coverage, exact)``，coverage ∈ [0, 1]，exact 为归一化后子串命中。
    """
    c = normalize(claim_text)[:max_claim_chars]
    s = normalize(span_text)
    if not c or not s:
        return 0.0, False
    if c in s:
        return 1.0, True
    return lcs_length(c, s) / float(len(c)), False


def bigram_coverage_of(claim_text: str, span_text: str,
                       max_claim_chars: int = 400) -> float:
    """claim 的**二元组**落在 span 中的比例 ∈ [0, 1]。

    这是对 LCS 的补充判据：LCS 会因为共享单字（中文高频字）而虚高，
    而连续两字同时命中是更强的信号。claim 归一化后短于 2 字符时返回 1.0
    （没有二元组可比，视为不设限，由 LCS 判据独自决定）。
    """
    c = normalize(claim_text)[:max_claim_chars]
    s = normalize(span_text)
    if len(c) < 2:
        return 1.0
    if not s:
        return 0.0
    grams = [c[i:i + 2] for i in range(len(c) - 1)]
    hit = sum(1 for g in grams if g in s)
    return hit / float(len(grams))


# 数字：整数/小数（含千分位与小数点；不处理中文数字，属已知保守边界）
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# 否定标记：单字 + 常见双字。（"非常""无论"这类会被计入，属已知过检，见 eval）
_NEG_CUES = ("不", "没", "无", "未", "非", "否", "勿", "免", "莫", "别")


def extract_numbers(text: str) -> List[str]:
    """抽取文本中的数字 token（去掉千分位逗号）。"""
    return [m.replace(",", "") for m in _NUMBER_RE.findall(text or "")]


def numbers_supported(claim_text: str, span_text: str) -> bool:
    """claim 里**每一个**数字都必须在 span 中出现。

    文本重叠对数字篡改几乎无感：把一个 ``1.86`` 改成 ``2.86``，字符覆盖率只掉
    百分之几，LCS/bigram 判据都拦不住，但这是最有杀伤力的一类编造 —— 读者
    记住的正是那个数字。因此数字必须逐值核对。
    """
    # 注意：不能用 normalize() —— 它会把小数点当标点去掉（"5.64" → "564"），
    # 与 extract_numbers 产出的 token 永远对不上。这里只去掉空白与千分位逗号。
    span_key = re.sub(r"[\s,，]", "", span_text or "")
    if not span_key:
        return False
    for n in extract_numbers(claim_text):
        if n not in span_key:
            return False
    return True


def negation_count(text: str) -> int:
    """否定标记出现次数（用于检测"一字之差、语义相反"的翻转）。"""
    t = normalize(text)
    return sum(t.count(cue) for cue in _NEG_CUES)


def polarity_matches(claim_text: str, span_text: str) -> bool:
    """claim 与 span 的否定词计数是否一致。

    保守判据：计数不等即视为极性不一致。这会把"无关"（含"无"）与"有关"
    判成极性相反 —— 正是想要的行为；副作用是少量改写句会被误判为不一致
    （召回代价换精确率，实测数字见 artifacts/grounding_eval.json）。
    """
    return negation_count(claim_text) == negation_count(span_text)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceSpan:
    """一条外部证据片段。``source`` 必须能追到原文（文档名 / URL / 检索命中 id）。"""

    span_id: str
    text: str
    source: str = ""
    start: int = -1
    end: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {"span_id": self.span_id, "text": self.text,
                "source": self.source, "start": self.start, "end": self.end}


@dataclass
class Claim:
    """被检查的一句话（通常来自模型输出，按句切分）。

    ``start`` / ``end`` 是这句话在**原回答**里的字符跨度（由 :func:`split_claims`
    填充，-1 表示未定位）。它们存在的唯一目的是把句子级标签铺到 token 位置 ——
    ``abstention_loss`` 要的是 (B, L) 的逐位置标签，没有跨度就无从对齐。
    """

    text: str
    index: int = 0
    start: int = -1
    end: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "index": self.index,
                "start": self.start, "end": self.end}


@dataclass
class Binding:
    """一句话与它最匹配的证据片段之间的归因结果。"""

    claim: Claim
    span: Optional[EvidenceSpan]
    coverage: float
    exact: bool
    supported: bool
    ngram: float = 0.0
    numbers_ok: bool = True
    polarity_ok: bool = True
    threshold: float = DEFAULT_MIN_COVERAGE
    ngram_threshold: float = DEFAULT_MIN_NGRAM

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.text,
            "claim_index": self.claim.index,
            "span_id": self.span.span_id if self.span else None,
            "source": self.span.source if self.span else None,
            "coverage": round(self.coverage, 4),
            "ngram": round(self.ngram, 4),
            "numbers_ok": self.numbers_ok,
            "polarity_ok": self.polarity_ok,
            "exact": self.exact,
            "supported": self.supported,
            "threshold": self.threshold,
            "ngram_threshold": self.ngram_threshold,
        }


# ---------------------------------------------------------------------------
# 绑定
# ---------------------------------------------------------------------------
def bind_claim(claim: Claim, spans: Sequence[EvidenceSpan],
               min_coverage: float = DEFAULT_MIN_COVERAGE,
               min_ngram: float = DEFAULT_MIN_NGRAM) -> Binding:
    """为单条 claim 找最佳证据片段。

    选择规则（确定性）：先按 ``exact`` 优先，再按 ``coverage`` 降序，再按
    ``span_id`` 字典序 —— 保证同一输入永远选出同一片段（可复现，不依赖 dict 顺序）。
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if not 0.0 <= min_ngram <= 1.0:
        raise ValueError(f"min_ngram must be in [0, 1], got {min_ngram}")
    best: Optional[Binding] = None
    for sp in spans:
        cov, exact = coverage_of(claim.text, sp.text)
        ng = bigram_coverage_of(claim.text, sp.text)
        if exact:
            # claim 逐字包含在 span 里：数字必然在，极性也必然同源。
            # 若仍与整段 span 比否定词计数，claim 是长句的一部分时会误判
            # （span 里多出来的否定词会把它算成"极性不一致"）。
            num_ok = pol_ok = True
        else:
            num_ok = numbers_supported(claim.text, sp.text)
            pol_ok = polarity_matches(claim.text, sp.text)
        b = Binding(claim=claim, span=sp, coverage=cov, exact=exact,
                    # exact 也要求数字/极性一致：逐字包含却篡改了数字是不可能的，
                    # 但 normalize 会丢掉标点，这里统一走同一套判据更安全
                    supported=bool((exact and num_ok and pol_ok) or
                                   (cov >= min_coverage and ng >= min_ngram
                                    and num_ok and pol_ok)),
                    ngram=ng, numbers_ok=num_ok, polarity_ok=pol_ok,
                    threshold=min_coverage, ngram_threshold=min_ngram)
        if best is None:
            best = b
            continue
        if (b.exact, b.coverage, b.ngram) > (best.exact, best.coverage, best.ngram):
            best = b
        elif ((b.exact, b.coverage, b.ngram) == (best.exact, best.coverage, best.ngram)
              and sp.span_id < (best.span.span_id if best.span else "")):
            best = b
    if best is None:
        return Binding(claim=claim, span=None, coverage=0.0, exact=False,
                       supported=False, ngram=0.0, numbers_ok=False,
                       polarity_ok=False, threshold=min_coverage,
                       ngram_threshold=min_ngram)
    return best


def bind_claims(claims: Iterable[Claim], spans: Sequence[EvidenceSpan],
                min_coverage: float = DEFAULT_MIN_COVERAGE,
                min_ngram: float = DEFAULT_MIN_NGRAM) -> List[Binding]:
    return [bind_claim(c, spans, min_coverage, min_ngram) for c in claims]


def split_claims(answer: str, max_claims: int = 64,
                 min_chars: int = 4) -> List[Claim]:
    """把一段回答切成"可核验的句子"，并记录每句在原文中的字符跨度。

    过短的片段（归一化后 < ``min_chars``）直接丢弃：像"是的""好的"这类
    不含事实主张的片段参与归因会稀释覆盖率，也会让拒答门误判。

    跨度（``Claim.start`` / ``Claim.end``）与 :func:`token_labels_from_bindings`
    配合，把句子级标签铺到 token 位置。
    """
    out: List[Claim] = []
    cursor = 0
    for m in _SENT_SPLIT_RE.finditer(answer or ""):
        raw = answer[cursor:m.start()]
        cursor = m.end()
        if not raw or not raw.strip():
            continue
        if len(normalize(raw)) < min_chars:
            continue
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        start = m.start() - len(raw) + lead
        out.append(Claim(text=body, index=len(out),
                         start=start, end=start + len(body)))
        if len(out) >= max_claims:
            break
    # 结尾没有句末标点的残余段
    if cursor < len(answer or "") and len(out) < max_claims:
        raw = answer[cursor:]
        if raw.strip() and len(normalize(raw)) >= min_chars:
            lead = len(raw) - len(raw.lstrip())
            body = raw.strip()
            start = cursor + lead
            out.append(Claim(text=body, index=len(out),
                             start=start, end=start + len(body)))
    return out


def char_token_offsets(text: str) -> List[Tuple[int, int]]:
    """字符级 tokenizer 的 `(start, end)` 偏移表（每个字符一个 token）。

    用于没有真实 tokenizer 的场景（测试、纯字符模型）。字符级模型的
    ``abstention_loss`` 直接消费本函数的结果。
    """
    return [(i, i + 1) for i in range(len(text or ""))]


def token_labels_from_bindings(
    bindings: Sequence[Binding],
    token_offsets: Sequence[Tuple[int, int]],
    default: float = 0.0,
) -> List[float]:
    """把**句子级**标签铺到 **token 位置**，供 ``abstention_loss`` 使用。

    规则：一个 token 的字符区间若与某个 claim 的字符区间**有交集**，就取该
    claim 的 ``supported`` 标签；不属于任何 claim（标点、句间空隙、被
    ``min_chars`` 过滤掉的短片段）取 ``default``。

    为什么需要这一步
    ----------------
    ``abstention_loss`` 收 ``(B, L)`` 的逐位置标签，而证据归因天然是**句子级**的。
    缺了这一步，L4 头就只能拿整段标签（全 1 或全 0）训练 —— 那等于没训练。
    多个 claim 与同一 token 重叠时按 ``claim.index`` 顺序取第一个，保证确定性。
    """
    labels = [float(default)] * len(token_offsets)
    for b in bindings:
        c = b.claim
        if c.start < 0 or c.end <= c.start:
            continue
        value = 1.0 if b.supported else 0.0
        for i, (ts, te) in enumerate(token_offsets):
            if te > c.start and ts < c.end:          # 区间相交
                labels[i] = value
    return labels


def grounded_label_tensor(bindings: Sequence[Binding],
                          token_offsets: Sequence[Tuple[int, int]],
                          default: float = 0.0):
    """同 :func:`token_labels_from_bindings`，但返回 ``(L,)`` 的 torch 张量。

    torch 只在调用时导入 —— 本模块的其余部分保持零第三方依赖。
    """
    import torch

    return torch.tensor(token_labels_from_bindings(bindings, token_offsets, default),
                        dtype=torch.float32)


# ---------------------------------------------------------------------------
# 聚合指标
# ---------------------------------------------------------------------------
def attribution_coverage(bindings: Sequence[Binding],
                         weight_by_length: bool = True) -> float:
    """归因覆盖率 ∈ [0, 1]：被证据支撑的 claim 占比。

    ``weight_by_length=True`` 时按 claim 归一化长度加权 —— 长句承载更多事实，
    一句长编造不该和一句短寒暄等价。
    """
    if not bindings:
        return 0.0
    if not weight_by_length:
        return sum(1.0 for b in bindings if b.supported) / len(bindings)
    num = 0.0
    den = 0.0
    for b in bindings:
        w = max(len(normalize(b.claim.text)), 1)
        den += w
        if b.supported:
            num += w
    return num / den if den else 0.0


def unsupported_claims(bindings: Sequence[Binding]) -> List[Binding]:
    return [b for b in bindings if not b.supported]


def grounded_labels(bindings: Sequence[Binding],
                    as_float: bool = True) -> List[float]:
    """给 L4 头用的逐 claim 标签（1.0 = 有外部证据支撑）。

    注意：这是**句子级**标签。位置级对齐由训练器按 claim 的 token 跨度完成
    （``abstention_loss`` 接受任意 (B, L) 形状）。
    """
    return [1.0 if b.supported else 0.0 for b in bindings]


# ---------------------------------------------------------------------------
# 证据索引
# ---------------------------------------------------------------------------
@dataclass
class EvidenceIndex:
    """从上下文/检索结果构建的可查询证据集合。"""

    spans: List[EvidenceSpan] = field(default_factory=list)
    source: str = ""

    # -- 构建 ---------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, source: str = "context",
                  max_span_chars: int = 240) -> "EvidenceIndex":
        """把一段上下文切成句子级片段（过长的按窗口硬切，保留原始偏移）。"""
        idx = cls(source=source)
        if not text:
            return idx
        cursor = 0
        for raw in _SENT_SPLIT_RE.split(text):
            if not raw or not raw.strip():
                cursor += len(raw) + 1
                continue
            seg = raw.strip()
            start = text.find(seg, cursor)
            cursor = (start + len(seg)) if start >= 0 else cursor
            if len(seg) <= max_span_chars:
                idx.add(seg, start=start, end=start + len(seg) if start >= 0 else -1)
            else:
                for off in range(0, len(seg), max_span_chars):
                    piece = seg[off:off + max_span_chars]
                    idx.add(piece, start=(start + off) if start >= 0 else -1)
        return idx

    @classmethod
    def from_retrieval(cls, hits: Sequence[Any],
                       text_field: str = "text",
                       id_field: str = "id",
                       source_field: str = "source") -> "EvidenceIndex":
        """从检索命中列表构建（dict 或对象均可）。``source`` 缺失时用命中 id。"""
        idx = cls(source="retrieval")
        for i, h in enumerate(hits):
            if isinstance(h, dict):
                text = str(h.get(text_field, ""))
                sid = str(h.get(id_field, f"hit-{i}"))
                src = str(h.get(source_field, sid))
            else:
                text = str(getattr(h, text_field, ""))
                sid = str(getattr(h, id_field, f"hit-{i}"))
                src = str(getattr(h, source_field, sid))
            if text:
                idx.spans.append(EvidenceSpan(span_id=sid, text=text, source=src))
        return idx

    def add(self, text: str, source: Optional[str] = None,
            start: int = -1, end: int = -1) -> EvidenceSpan:
        sp = EvidenceSpan(span_id=f"s{len(self.spans):04d}", text=text,
                          source=source or self.source, start=start, end=end)
        self.spans.append(sp)
        return sp

    # -- 查询 ---------------------------------------------------------------
    def bind(self, claim: Claim,
             min_coverage: float = DEFAULT_MIN_COVERAGE,
             min_ngram: float = DEFAULT_MIN_NGRAM) -> Binding:
        return bind_claim(claim, self.spans, min_coverage, min_ngram)

    def bind_all(self, claims: Iterable[Claim],
                 min_coverage: float = DEFAULT_MIN_COVERAGE,
                 min_ngram: float = DEFAULT_MIN_NGRAM) -> List[Binding]:
        return bind_claims(claims, self.spans, min_coverage, min_ngram)

    def coverage(self, answer: str,
                 min_coverage: float = DEFAULT_MIN_COVERAGE,
                 min_ngram: float = DEFAULT_MIN_NGRAM) -> float:
        """便捷入口：一段回答 → 归因覆盖率。"""
        return attribution_coverage(
            self.bind_all(split_claims(answer), min_coverage, min_ngram))

    def attribute(self, answer: str,
                  min_coverage: float = DEFAULT_MIN_COVERAGE,
                  min_ngram: float = DEFAULT_MIN_NGRAM) -> Dict[str, Any]:
        """完整归因报告（JSON 可序列化），供拒答策略与审计使用。"""
        claims = split_claims(answer)
        bindings = self.bind_all(claims, min_coverage, min_ngram)
        return {
            "source": self.source,
            "n_spans": len(self.spans),
            "n_claims": len(claims),
            "coverage": round(attribution_coverage(bindings), 4),
            "n_unsupported": len(unsupported_claims(bindings)),
            "bindings": [b.as_dict() for b in bindings],
        }

    def __len__(self) -> int:
        return len(self.spans)
