# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""文本清洗：NFKC、控制字符剥离、代码/日志噪声行剔除、长度门限。"""

from .constants import _LOG, MIN_SYMBOL_RATIO_CHARS
from .types import CleanConfig

import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple




# ---------------------------------------------------------------------------
# 正则 / 噪声判定
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

#: 常见"日志/代码"特征。刻意写得保守, 只抓高置信度的形状。
_NOISE_PATTERNS: Tuple[re.Pattern, ...] = (
    # 时间戳: 2024-01-31 12:00:00 / [12:00:00] / 12:00:00.123
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"),
    re.compile(r"\[\d{2}:\d{2}:\d{2}\]"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}[.,]\d{3}\b"),
    # 日志级别
    re.compile(r"\b(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE|CRITICAL)\b\s*[:：]"),
    # 异常与栈
    re.compile(r"Traceback\s*\(most recent call last\)"),
    re.compile(r"^\s*(?:File|文件)\s*\"[^\"]+\",\s*line\s+\d+"),
    re.compile(r"^\s*at\s+[\w.$<>]+\([\w.$<>]*\.java:\d+\)"),
    re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception)\b.*:"),
    # 代码外壳
    re.compile(r"^\s*(?:def|class|function|public|private|static|import|from|export|const|let|var|#include|using\s+namespace|package)\b"),
    re.compile(r"^\s*[{}()\[\];]+\s*$"),
    re.compile(r"^\s*(?:\+\+\+|---)\s"),
    re.compile(r"^\s*(?:>>>|\.\.\.)\s"),
    re.compile(r"^\s*(?:rm|cd|ls|cp|mv|sudo|pip|npm|git|python|conda)\s+-{0,2}\S"),
    re.compile(r"^\s*<\?xml|^\s*<!DOCTYPE"),
    re.compile(r"^\s*[A-Za-z0-9_./\\-]+\.(?:py|js|ts|java|cpp|h|go|rs|json|yaml|yml|xml|html|css|sh|ps1|bat)\b.*:\d+"),
)

#: 分隔线: 8 个以上同类符号组成的行 (``====``/``----``/``####``/``****``)。
_SEPARATOR_RE = re.compile(r"^\s*([=\-*#~_+=])\1{7,}\s*$")

#: 控制字符 / 零宽字符 / 双向控制符 / 私用区 —— 一律剥离。
#: ``\n`` 有意排除在外: 行结构需要保留。
_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    r"|[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufff9-\ufffb]"
    r"|[\ue000-\uf8ff]"
)
_WHITESPACE_RE = re.compile(r"[\t\v\f\r\u00a0\u3000 \u2000-\u200a\u2028\u2029]+")


def _is_cjk(ch: str) -> bool:
    """是否属于主要 CJK 区段 (中日韩统一表意文字及其扩展/兼容区、假名、谚文)。"""
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF       # CJK 统一表意文字
        or 0x3400 <= o <= 0x4DBF    # 扩展 A
        or 0xF900 <= o <= 0xFAFF    # 兼容表意文字
        or 0x20000 <= o <= 0x2FA1F  # 扩展 B-F + 兼容补充
        or 0x3040 <= o <= 0x30FF    # 平假名 / 片假名
        or 0xAC00 <= o <= 0xD7AF    # 谚文音节
        or 0x1100 <= o <= 0x11FF    # 谚文字母
    )


def _symbol_ratio(line: str) -> float:
    """一行里"既非 CJK 又非字母数字"的字符占比 (空行返回 0)。

    中文写作里正常的标点占比通常 < 0.2; 代码/日志/十六进制转储会明显超过 0.4。
    """
    stripped = line.strip()
    if not stripped:
        return 0.0
    non_text = 0
    for ch in stripped:
        if ch.isalnum() or _is_cjk(ch):
            continue
        non_text += 1
    return non_text / len(stripped)


def looks_like_noise(
    line: str,
    noise_ratio_threshold: float = 0.40,
    min_ratio_chars: int = MIN_SYMBOL_RATIO_CHARS,
) -> bool:
    """启发式判断一行是不是代码/日志噪声。

    判据 (任一命中即算噪声):
      * 匹配 :data:`_NOISE_PATTERNS` 中的日志/代码形状;
      * 是 8 个以上同类符号组成的分隔线;
      * 整行**只有符号**(非 CJK 非字母数字)且长度 >= 3;
      * 长度 >= ``min_ratio_chars`` 且 :func:`_symbol_ratio` 超过 ``noise_ratio_threshold``。

    为什么要有 ``min_ratio_chars``: 占比在短行上方差极大 —— 中文里
    ``"前。"`` 的符号占比是 0.5, 远超 0.4 的阈值, 纯按占比判会把大量
    "短句成段"的正文整行删掉 (这是真实踩到的 bug)。所以占比判据只对
    长度 >= 8 个字符的行生效; 短行仍会被上面的模式判据和"纯符号"判据拦住。

    Args:
        line: 单行文本。
        noise_ratio_threshold: 符号占比阈值, 默认 0.40。
        min_ratio_chars: 占比判据生效的最小行长, 默认 8。
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _SEPARATOR_RE.match(stripped):
        return True
    for pat in _NOISE_PATTERNS:
        if pat.search(stripped):
            return True
    has_cjk = any(_is_cjk(ch) for ch in stripped)
    has_alnum = any(ch.isalnum() for ch in stripped)
    if not has_cjk and not has_alnum and len(stripped) >= 3:
        return True
    if len(stripped) < int(min_ratio_chars):
        return False
    return _symbol_ratio(stripped) > float(noise_ratio_threshold)


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def classify_text(text: Any, cfg: Optional[CleanConfig] = None) -> Tuple[str, str]:
    """清洗一段文档, 同时给出**保留/丢弃的原因**。

    Args:
        text: 原始文本 (``str``; 其他类型一律按空文档处理)。
        cfg: :class:`CleanConfig`; ``None`` 用默认值。

    Returns:
        ``(cleaned, reason)``。``reason == "kept"`` 表示保留,
        其余取值: ``"empty"`` / ``"no_renderable_text"`` / ``"too_short"`` /
        ``"too_long"`` / ``"too_many_lines"`` / ``"too_noisy"``。
    """
    cfg = cfg or CleanConfig()
    if not isinstance(text, str) or not text:
        return "", "empty"

    original_chars = len(text)

    # 1) Unicode NFKC 归一化: 全角/兼容字符折叠, 不让同形异码撑大词表。
    s = unicodedata.normalize("NFKC", text)

    # 2) 剥离控制字符/零宽字符/私用区; \x1f (EOS 字面量) 一并去掉。
    s = _CTRL_RE.sub("", s)
    s = s.replace("\x1f", "")

    # 3) URL / 邮箱折叠成占位符 (先折叠再逐行处理, 否则行内符号占比会被 URL 拉高)。
    if cfg.strip_urls:
        s = _URL_RE.sub("<url>", s)
    if cfg.strip_emails:
        s = _EMAIL_RE.sub("<email>", s)

    # 4) 逐行: 行内空白压缩 -> 噪声行剔除 -> 连续空行折叠。
    #    折叠语义: 一个空行连续段最多保留 ``max_blank_lines`` 个空行, 其中
    #    ``max_blank_lines = blank_line_collapse_threshold + collapse_blank_lines - 1``
    #    (默认 2 + 1 - 1 = 2, 即最多两行空行); 设 collapse_blank_lines=0 就是最多一行。
    out_lines: List[str] = []
    blank_run = 0
    threshold = max(int(cfg.blank_line_collapse_threshold), 1)
    max_blank_lines = max(threshold + max(int(cfg.collapse_blank_lines), 0) - 1, 0)
    for raw_line in s.split("\n"):
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            blank_run += 1
            if blank_run <= max_blank_lines:
                out_lines.append("")
            continue
        if cfg.drop_noise_lines and looks_like_noise(
            line, cfg.noise_ratio_threshold, cfg.min_symbol_ratio_chars
        ):
            # 被丢掉的行**不重置** blank_run: 一行噪声夹在两段空行之间时, 它两侧的
            # 空行属于同一个空行段, 否则被删掉的噪声行会把 3 行空白"洗成"合法空白。
            continue
        blank_run = 0
        out_lines.append(line)

    cleaned = "\n".join(out_lines).strip()
    if not cleaned:
        return "", "no_renderable_text"

    keep_ratio = len(cleaned) / max(original_chars, 1)
    n_lines = cleaned.count("\n") + 1
    if len(cleaned) < cfg.min_chars:
        return cleaned, "too_short"
    if len(cleaned) > cfg.max_chars:
        return cleaned, "too_long"
    if n_lines > cfg.max_lines:
        return cleaned, "too_many_lines"
    if keep_ratio < cfg.min_keep_ratio:
        return cleaned, "too_noisy"
    return cleaned, "kept"


def clean_text(text: Any, cfg: Optional[CleanConfig] = None) -> str:
    """清洗单篇文档, 返回清洗结果; 被丢弃时返回 ``""``。

    处理顺序: NFKC 归一化 -> 剥离控制字符/零宽字符/私用区 -> URL、邮箱折叠为占位符 ->
    行内空白压缩 -> 剔除代码/日志噪声行 -> 折叠连续空行 ->
    最小/最大字符数与行数门限 -> 保留率门限。

    Args:
        text: 原始文本。
        cfg: :class:`CleanConfig`。

    Returns:
        清洗后的文本, 或 ``""`` (该文档被判为不可用)。
    """
    cleaned, reason = classify_text(text, cfg)
    return cleaned if reason == "kept" else ""

