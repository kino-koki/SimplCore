import re
from typing import Any, Dict, Optional


_TRADITIONAL_MARKERS = set("體臺灣後裏麼為與這個們來時會說國學習資訊變處")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def is_simplified_chinese_text(
    text: str,
    min_cjk_ratio: float = 0.45,
    max_latin_ratio: float = 0.25,
) -> bool:
    if not text:
        return False
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < 12:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if cjk_count / max(len(chars), 1) < min_cjk_ratio:
        return False
    if latin_count / max(len(chars), 1) > max_latin_ratio:
        return False
    traditional_hits = sum(1 for ch in text if ch in _TRADITIONAL_MARKERS)
    if traditional_hits >= 3:
        return False
    return True


def clean_record(record: Dict[str, Any], text_key: str = "text") -> Optional[Dict[str, str]]:
    raw = record.get(text_key, "")
    if not isinstance(raw, str):
        return None
    text = normalize_text(raw)
    if not is_simplified_chinese_text(text):
        return None
    return {"text": text}
