# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 真实数据管道 —— 清洗、打包、分片与内存映射数据集.

(Real-data pipeline for SimplCore: cleaning, packing, sharding, memmapped dataset.)

本模块把磁盘上的 parquet 原始文本变成训练可以直接消费的 token 流,
一共四步, 每一步都可单独调用、可单独测试:

1. :func:`iter_documents` —— 流式读 parquet (自动识别文本列), 跳过下载未完成的文件;
2. :func:`clean_text`     —— NFKC + 去控制字符 + 代码/日志噪声行剔除 + 长度门限;
3. :func:`pack_documents` —— 分词、文档间插 EOS、拼成一条长流、
   **按文档**划分 train/val/test 后切成 ``seq_len`` 定长窗口, 落盘 ``*.bin`` + ``index.json``;
4. :class:`ShardedTokenDataset` —— 对 ``.bin`` 做内存映射 (mmap) 的 ``torch.utils.data.Dataset``。

为什么必须"按文档"划验证集 (why document-level splitting)
--------------------------------------------------------
如果先把全部文本拼成一条 token 流再按窗口切分, 同一个文档会同时出现在训练集和
验证集里 —— 验证 loss 会因此被系统性低估 (同文续写), 模型选择随之失真。
本实现先洗牌**文档**, 再把整篇文档整体分派给某一个 split, 因此同一文档的
token 只会出现在一个 split 中; ``tests/test_data.py`` 会显式断言这一点。

关于 EOS (about the separator)
------------------------------
文档之间插入 1 个 EOS id。EOS 的字面量默认是 ASCII 单元分隔符 ``\\x1f``
(见 :mod:`simplcore.core.tokenizer`), 它在 :func:`clean_text` 里已被全部剥离,
因此不可能和正文冲突 (用 ``<|eos|>`` 这类字面量做 EOS 会劫持真实网页文本)。

CLI 用法 (usage)::

    python -m simplcore.core.data --help
    python -m simplcore.core.data --parquet-dir data/raw/fineweb_zh \\
        --out data/processed/tuan_zh --tokenizer data/processed/tuan_zh --seq-len 4096

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "CleanConfig",
    "DataPipelineError",
    "PackConfig",
    "PackResult",
    "ShardedTokenDataset",
    "classify_text",
    "clean_text",
    "compute_sha256",
    "count_tokens",
    "encode_document",
    "iter_documents",
    "load_shard",
    "load_split",
    "looks_like_noise",
    "pack_documents",
    "pick_text_column",
    "read_manifest",
    "shard_paths",
]

_LOG = logging.getLogger("simplcore.core.data")

MANIFEST_NAME = "index.json"
DEFAULT_TEXT_COLUMNS: Tuple[str, ...] = ("text", "content", "raw_content")

#: 下载未完成的痕迹; 名字里带这些子串的文件一律跳过。
SKIP_MARKERS: Tuple[str, ...] = (".incomplete", ".lock", ".part", ".tmp")

#: ``data/raw/fineweb_zh`` 这类镜像仓库布局里, 这些目录只是 HF 的下载缓存,
#: 不是语料本体, 一律不扫。
SKIP_DIR_NAMES: Tuple[str, ...] = (".cache", "__pycache__", ".git")

UINT32_MAX = 2 ** 32 - 1

#: :func:`_symbol_ratio` 判据生效的最小行长。短行 (例如 "前。") 的符号占比
#: 天然很高, 必须用长度门限挡住, 否则会误删正文里的短句成段。
#: 定义在 :class:`CleanConfig` 之前, 因为它被用作该 dataclass 的默认值。
MIN_SYMBOL_RATIO_CHARS = 8


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class CleanConfig:
    """文本清洗参数 (cleaning knobs), 会**原样**写进 manifest 以保证可复现.

    Attributes:
        min_chars: 清洗后短于该字符数的文档整篇丢弃 (默认 200)。
        max_chars: 清洗后长于该字符数的文档整篇丢弃 (默认 200000)。
            注意这是"整篇丢弃"而不是截断 —— 截断会在文档尾部制造伪 EOS 边界。
        max_lines: 行数上限, 超出整篇丢弃 (防住"整本字典"这类退化长文)。
        blank_line_collapse_threshold: 从第几个连续空行起开始按
            ``collapse_blank_lines`` 折叠 (默认 2; 比它短的空行段原样保留)。
        collapse_blank_lines: 每个空行连续段**额外**保留几个空行 (默认 1,
            即最多两行空行; 设 0 则最多一行空行) —— 用于在某些语料上压掉段落间空隙。
        noise_ratio_threshold: 一行里"非中日韩且非字母数字"字符占比超过该值即判为噪声行 (默认 0.40)。
        min_symbol_ratio_chars: 占比判据生效的最小行长 (默认 8)。
            短行占比噪声大 (``"前。"`` 就是 0.5), 必须靠长度门限保护。
        min_keep_ratio: 清洗后保留字符数 / 原字符数 低于该值整篇丢弃 (默认 0.10);
            它挡住"整篇几乎都是代码/日志"的文档。
        strip_urls: 是否把 URL 折叠成 ``<url>`` 占位符 (默认 True)。
        strip_emails: 是否把邮箱折叠成 ``<email>`` 占位符 (默认 True)。
        drop_noise_lines: 关掉它可以把噪声行**原样保留** (用于消融对比)。
    """

    min_chars: int = 200
    max_chars: int = 200_000
    max_lines: int = 20_000
    collapse_blank_lines: int = 1
    blank_line_collapse_threshold: int = 2
    noise_ratio_threshold: float = 0.40
    min_symbol_ratio_chars: int = MIN_SYMBOL_RATIO_CHARS
    min_keep_ratio: float = 0.10
    strip_urls: bool = True
    strip_emails: bool = True
    drop_noise_lines: bool = True

    def to_dict(self) -> dict:
        """JSON-able view written verbatim into the manifest."""
        return asdict(self)


@dataclass
class PackConfig:
    """打包参数 (packing knobs).

    Attributes:
        seq_len: 训练窗口长度, 也是切窗的粒度。
        val_frac: **文档级**验证集比例 (默认 0.005, 与 CLI 默认一致)。
        test_frac: **文档级**测试集比例 (默认 0.005)。
        seed: 文档洗牌种子; 同一份输入 + 同一 seed => 同一份切分。
        eos_id: 文档间分隔符 id; 传 ``None`` 时从 tokenizer 上取 ``eos_id``。
        dtype: ``"uint32"`` (默认, 安全) 或 ``"uint16"`` (vocab < 65535 时省一半磁盘)。
        shard_tokens: 单个 ``*.bin`` 的最大 token 数; 超出即滚动到下一个分片。
            ``None`` 表示每个 split 只写一个文件。
        drop_last: 丢弃不足 ``seq_len`` 的尾部窗口 (默认 True; 该尾部仍计入 token 统计)。
        append_eos_after_last: 最后一篇文档之后是否再补一个 EOS (默认 True)。
        min_train_windows: 训练集窗口数低于该值直接报错, 避免悄悄产出空语料。
    """

    seq_len: int = 4096
    val_frac: float = 0.005
    test_frac: float = 0.005
    seed: int = 1234
    eos_id: Optional[int] = None
    dtype: str = "uint32"
    shard_tokens: Optional[int] = 200_000_000
    drop_last: bool = True
    append_eos_after_last: bool = True
    min_train_windows: int = 1

    def to_dict(self) -> dict:
        """JSON-able view written verbatim into the manifest."""
        d = asdict(self)
        d["eos_id"] = None if self.eos_id is None else int(self.eos_id)
        return d


@dataclass
class PackResult:
    """一次 :func:`pack_documents` 的结果 (summary of one packing run).

    Attributes:
        out_dir: 输出目录。
        manifest_path: ``index.json`` 路径。
        n_docs_in: 输入文档总数。
        n_docs_kept: 通过 :func:`clean_text` 的文档数。
        n_docs_dropped: 被丢弃的文档数。
        drop_reasons: ``原因 -> 次数``。原因取自 :func:`classify_text`:
            ``too_short`` / ``too_long`` / ``too_many_lines`` / ``too_noisy``。
        n_docs_empty: 迭代器产出空串 (非 ``str`` 或空文本) 的次数。
        splits: ``split -> 统计``, 含文档数 / token 数 / 窗口数 / 分片路径 / SHA-256。
        total_tokens: 所有 split **写进 .bin** 的 token 总数 (与磁盘字节数严格一致)。
        total_tokens_encoded: 编码出的 token 总数 (含被 ``drop_last`` 丢掉的尾巴)。
        chars: 保留文档的总字符数 (用于算压缩比)。
        compression_ratio: 实测 chars per token = ``chars / total_tokens_encoded``。
        elapsed_seconds: 挂钟耗时。
        source_files: 读取过的 parquet 文件名。
    """

    out_dir: str
    manifest_path: str
    n_docs_in: int
    n_docs_kept: int
    n_docs_dropped: int
    drop_reasons: Dict[str, int]
    n_docs_empty: int
    splits: Dict[str, Dict[str, Any]]
    total_tokens: int
    total_tokens_encoded: int
    chars: int
    compression_ratio: float
    elapsed_seconds: float
    source_files: List[str] = field(default_factory=list)

    def summary_table(self) -> str:
        """固定宽度文本表, 便于 CLI 直接打印 (Windows 上用 ``Out-String -Width 200``)."""
        rows = [("split", "docs", "tokens", "windows", "files", "bytes")]
        for name in ("train", "val", "test"):
            s = self.splits.get(name)
            if s is None:
                continue
            rows.append((
                name,
                str(s["n_docs"]),
                str(s["n_tokens"]),
                str(s["n_windows"]),
                str(len(s["files"])),
                str(s["bytes"]),
            ))
        rows.append((
            "TOTAL",
            str(self.n_docs_kept),
            str(self.total_tokens),
            str(sum(s["n_windows"] for s in self.splits.values())),
            str(sum(len(s["files"]) for s in self.splits.values())),
            str(sum(s["bytes"] for s in self.splits.values())),
        ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = []
        for idx, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if idx == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)


class DataPipelineError(RuntimeError):
    """数据管道里的致命错误 (fatal pipeline error, e.g. an empty corpus)."""


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


# ---------------------------------------------------------------------------
# parquet 读取
# ---------------------------------------------------------------------------
def _iter_parquet_files(parquet_dir: Any, skip_markers: Sequence[str] = SKIP_MARKERS) -> List[Path]:
    """列出目录下所有可用 parquet 文件 (递归, 排序, 跳过缓存目录与未完成文件)。

    镜像仓库布局 (``<repo>/<subset>/xxxxx.parquet``) 与平铺布局都支持。
    ``.cache`` 下 HF 正在写的 ``*.incomplete`` 会被排除。
    """
    root = Path(parquet_dir)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"parquet dir not found: {root}")
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            if not name.endswith(".parquet"):
                continue
            if any(marker in name for marker in skip_markers):
                continue
            files.append(Path(dirpath) / name)
    files.sort()
    return files


def pick_text_column(available: Sequence[str], candidates: Sequence[str] = DEFAULT_TEXT_COLUMNS) -> Optional[str]:
    """在可用列名里挑文本列 (优先 ``text`` -> ``content`` -> ``raw_content``)。

    比较时忽略大小写与前后空白; 都找不到返回 ``None``。
    """
    lowered = {str(c).strip().lower(): str(c) for c in available}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def iter_documents(
    parquet_dir: Any,
    *,
    text_column: Optional[str] = None,
    max_docs: Optional[int] = None,
    max_docs_per_file: Optional[int] = None,
    skip_files: Sequence[str] = (),
    strict: bool = False,
    skip_markers: Sequence[str] = SKIP_MARKERS,
) -> Iterator[str]:
    """惰性读取 parquet 目录, 逐篇产出**原始**文本 (未清洗)。

    实现要点:
      * 按 row group / batch 读取, 不把整个文件读进内存;
      * 只物化文本列, 其它列直接跳过 (fineweb 的 parquet 往往带若干元数据列);
      * 文件名含 ``.incomplete`` / ``.lock`` 的一律跳过 —— 这是下载未完成的记号;
      * 单个文件读失败 (典型场景: 下载被中断导致 footer 缺失) 默认**记录并跳过**,
        不中断整条流水线; ``strict=True`` 时改为抛出。

    Args:
        parquet_dir: 目录或单个 ``.parquet`` 文件。
        text_column: 指定文本列名; ``None`` 时按
            ``text`` -> ``content`` -> ``raw_content`` 自动识别。
        max_docs: 全局最多产出多少篇文档。
        max_docs_per_file: 每个文件最多产出多少篇文档。
        skip_files: 需要额外跳过的文件名或路径子串。
        strict: 读文件出错时是否直接抛出。
        skip_markers: 文件名里出现即跳过的子串。

    Yields:
        原始文档字符串。

    Raises:
        FileNotFoundError: 目录里一个可用的 parquet 都没有。
        KeyError: 自动识别不出文本列。
    """
    files = _iter_parquet_files(parquet_dir, skip_markers=skip_markers)
    if skip_files:
        files = [f for f in files if not any(s in str(f) for s in skip_files)]
    if not files:
        raise FileNotFoundError(
            f"no usable .parquet files under {parquet_dir} (skipped markers: {list(skip_markers)})"
        )

    emitted = 0
    # 列名问题属于配置错误, 不是"单个坏分片", 所以无论 strict 与否都立刻报错 ——
    # 否则整条流水线会静默地产出 0 篇文档。
    if text_column is not None:
        import pyarrow.parquet as _pq

        for path in files:
            names = _pq.ParquetFile(str(path)).schema_arrow.names
            if text_column not in names:
                raise KeyError(f"column {text_column!r} not in {path}: {list(names)}")

    for path in files:
        _LOG.info("reading parquet shard: %s", path)
        remaining = None if max_docs is None else max(int(max_docs) - emitted, 0)
        try:
            for value in _iter_one_parquet(
                path,
                text_column=text_column,
                max_docs=remaining,
                max_docs_per_file=max_docs_per_file,
            ):
                yield value
                emitted += 1
        except Exception as exc:  # noqa: BLE001 - 单个坏分片不应拖垮整条流水线
            if strict:
                raise
            _LOG.warning("skipping unreadable parquet shard %s: %r", path, exc)
            continue
        if max_docs is not None and emitted >= int(max_docs):
            return


def _iter_one_parquet(
    path: Path,
    *,
    text_column: Optional[str],
    max_docs: Optional[int],
    max_docs_per_file: Optional[int],
) -> Iterator[str]:
    """读一个 parquet 文件, 按批次逐条产出文本。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    column = text_column
    if column is None:
        column = pick_text_column(pf.schema_arrow.names)
        if column is None:
            raise KeyError(
                f"no text column among {DEFAULT_TEXT_COLUMNS} in {path}; "
                f"available columns: {list(pf.schema_arrow.names)}"
            )
    elif column not in pf.schema_arrow.names:
        raise KeyError(f"column {column!r} not in {path}: {list(pf.schema_arrow.names)}")

    local = 0
    batch_size = 1024 if max_docs_per_file is None else min(1024, int(max_docs_per_file))
    for batch in pf.iter_batches(columns=[column], batch_size=max(batch_size, 1)):
        for value in batch.column(0).to_pylist():
            if value is None:
                continue
            yield value
            local += 1
            if max_docs_per_file is not None and local >= max_docs_per_file:
                return
            if max_docs is not None and local >= max_docs:
                return


# ---------------------------------------------------------------------------
# 分词辅助
# ---------------------------------------------------------------------------
def _eos_of(tokenizer: Any) -> Optional[int]:
    """尽量从分词器对象上取 EOS id (兼容我们自己挂的属性与 HF 的原生属性)。"""
    eos = getattr(tokenizer, "eos_id", None)
    if eos is not None:
        return int(eos)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return None


class _TokenizerAdapter:
    """把 HF ``tokenizers.Tokenizer`` 或自定义分词器包成统一接口。

    提供 :meth:`encode_all` / :meth:`count_tokens` / :attr:`eos_id`。
    HF 的 Rust 实现有 ``encode_batch``, 批编码比逐条 Python 循环快一个量级,
    打包 1.5 GB 语料时这个差别是分钟级的。
    """

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.raw = tokenizer
        self.eos_id = _eos_of(tokenizer)
        enc = getattr(tokenizer, "encode", None)
        if not callable(enc):
            raise TypeError(f"tokenizer of type {type(tokenizer).__name__} has no encode()")
        self._enc = enc

    def encode_all(self, texts: Sequence[str]) -> List[List[int]]:
        """批量编码; 优先走 HF 的 Rust ``encode_batch``。"""
        if not texts:
            return []
        batch_fn = getattr(self.raw, "encode_batch", None)
        if callable(batch_fn):
            return [list(enc.ids) for enc in batch_fn(list(texts))]
        return [list(self._enc(t)) for t in texts]

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """每篇文档的 ``len(encode(text)) + 1`` (那个 +1 是文档间追加的 EOS)。"""
        if not texts:
            return []
        batch_fn = getattr(self.raw, "encode_batch", None)
        if callable(batch_fn):
            return [len(enc.ids) + 1 for enc in batch_fn(list(texts))]
        return [len(self._enc(t)) + 1 for t in texts]


def encode_document(tokenizer: Any, text: str, eos_id: Optional[int] = None) -> List[int]:
    """编码单篇文档并在尾部追加 EOS (返回 ``encode(text) + [eos_id]``)。"""
    if eos_id is None:
        eos_id = _eos_of(tokenizer)
    if eos_id is None:
        raise ValueError("eos_id is required (pass it explicitly or put it on the tokenizer)")
    return list(tokenizer.encode(text)) + [int(eos_id)]


def count_tokens(tokenizer: Any, text: str) -> int:
    """单篇文档编码后的 token 数 (含 1 个 EOS)。"""
    return len(tokenizer.encode(text)) + 1


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def compute_sha256(path: Any, chunk_bytes: int = 1 << 20) -> str:
    """流式计算文件 SHA-256 (不把整个文件读进内存)。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokens_from_docs(adapter: _TokenizerAdapter, docs: Sequence[str], batch: int = 256) -> Iterator[List[int]]:
    """把文档列表编码成一条 token 流: **每篇文档产出一个 list[int]**。

    契约是"一个 list 就是一篇文档的完整 token 序列 (含尾 EOS)", 调用方
    (:func:`_write_split`) 依赖这个粒度做拼接与切窗。早期版本写成 ``yield from``
    把每个 int 单独 yield, 结果 ``np.asarray(5)`` 变成 0 维、被 ``np.atleast_1d``
    悄悄压成 1 个元素, EOS 因此永远不会落盘 —— 这个 bug 由
    ``test_splits_are_document_disjoint`` (检查 EOS 数量) 抓出来。
    """
    for i in range(0, len(docs), batch):
        for ids in adapter.encode_all(docs[i:i + batch]):
            # 每篇文档**自己**带上尾 EOS: 这是"文档之间可见的硬边界"的唯一实现,
            # 少了这一行, 相邻两篇文档就会被拼成一个连续上下文 (真实的早期 bug)。
            yield list(ids) + [int(adapter.eos_id)]


def _plan_document_splits(n_kept: int, val_frac: float, test_frac: float) -> Dict[str, int]:
    """按文档数把 kept 文档切成 train/val/test, 返回各 split 的文档数。

    先把 val/test 向下取整, 剩余全给 train; 只要 kept 文档数 >= 3,
    每个 split 至少 1 篇 —— 否则小语料上 val/test 会静默为空。
    """
    if n_kept <= 0:
        return {"train": 0, "val": 0, "test": 0}
    n_val = int(n_kept * max(float(val_frac), 0.0))
    n_test = int(n_kept * max(float(test_frac), 0.0))
    if n_kept >= 3:
        if val_frac > 0 and n_val == 0:
            n_val = 1
        if test_frac > 0 and n_test == 0:
            n_test = 1
    # 不能把训练集挤空
    while (n_val + n_test) >= n_kept and (n_val + n_test) > 0:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return {"train": n_kept - n_val - n_test, "val": n_val, "test": n_test}


def _write_split(
    split: str,
    docs: Sequence[str],
    adapter: _TokenizerAdapter,
    seq_len: int,
    out_dir: Path,
    dtype: str,
    shard_tokens: Optional[int],
    drop_last: bool,
    append_eos_after_last: bool,
) -> Dict[str, Any]:
    """把某个 split 的文档切成 ``seq_len`` 窗口并落盘, 返回统计字典。

    文档边界被**显式**保留: 每篇文档以 EOS 结尾, 因此窗口里不会出现
    "上一篇文章的尾巴直接续上下一篇开头" 的伪上下文 (EOS 是窗口内可见的硬边界)。
    切窗严格按 token 流顺序, 不做随机洗牌 —— 顺序切窗的确定性使
    "同一输入 + 同一 seed => 同一份语料" 成立, 也让 manifest 里的窗口数可被独立复算。
    """
    np_dtype = np.dtype(dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    single_file = shard_tokens is None or int(shard_tokens) <= 0

    files: List[Dict[str, Any]] = []
    total_encoded = 0
    total_windows = 0
    total_written = 0          # 写进磁盘的 token 数 (被 drop_last 丢掉的不算)
    remainder = 0
    buf = np.empty(0, dtype=np_dtype)
    fh = None
    fname: Optional[str] = None
    written_in_shard = 0
    n_shards = 0

    def _open_shard() -> None:
        nonlocal fh, fname, written_in_shard, n_shards
        n_shards += 1
        fname = f"{split}.bin" if single_file else f"{split}-{n_shards:05d}.bin"
        fh = open(out_dir / fname, "wb")
        written_in_shard = 0

    def _close_shard() -> None:
        nonlocal fh, fname, written_in_shard
        if fh is None:
            return
        fh.flush()
        fh.close()
        fh = None
        path = out_dir / fname
        size = path.stat().st_size
        files.append({
            "path": fname,
            "n_tokens": int(written_in_shard),
            "windows": int(written_in_shard // seq_len),
            "bytes": int(size),
            "sha256": compute_sha256(path),
        })
        fname = None
        written_in_shard = 0

    _open_shard()
    try:
        for ids in _tokens_from_docs(adapter, docs):
            if not ids:
                continue
            # np.asarray(5) 是 0 维数组 (单 token 文档), 必须升到 1 维才能拼接。
            arr = np.atleast_1d(np.asarray(ids, dtype=np_dtype))
            total_encoded += int(arr.size)
            buf = np.concatenate((buf, arr)) if buf.size else arr
            n_win = int(buf.size // seq_len)
            if n_win <= 0:
                continue
            whole = buf[: n_win * seq_len]
            if (not single_file) and written_in_shard + whole.size > int(shard_tokens) > 0 and written_in_shard > 0:
                _close_shard()
                _open_shard()
            fh.write(whole.tobytes(order="C"))
            written_in_shard += int(whole.size)
            total_written += int(whole.size)
            total_windows += n_win
            buf = np.ascontiguousarray(buf[n_win * seq_len:])
            if (not single_file) and written_in_shard >= int(shard_tokens):
                _close_shard()
                _open_shard()

        # 流结束: 补一个尾部 EOS, 再处理不足一窗的残留
        if append_eos_after_last and docs:
            tail = np.atleast_1d(np.asarray([int(adapter.eos_id)], dtype=np_dtype))
            buf = np.concatenate((buf, tail)) if buf.size else tail
            total_encoded += 1
        remainder = int(buf.size)
        if (not drop_last) and buf.size >= seq_len:
            n_win = int(buf.size // seq_len)
            whole = buf[: n_win * seq_len]
            if (not single_file) and written_in_shard + whole.size > int(shard_tokens) > 0 and written_in_shard > 0:
                _close_shard()
                _open_shard()
            fh.write(whole.tobytes(order="C"))
            written_in_shard += int(whole.size)
            total_written += int(whole.size)
            total_windows += n_win
            remainder = int(buf.size - whole.size)
        # 无论有没有写满一窗, 都必须关闭当前分片 —— 否则最后一个分片永远
        # 不会进入 files 列表 (未被 flush/stat/hash), 严重点就是"数据写进了磁盘
        # 却没有出现在 manifest 里"。
        _close_shard()
    except BaseException:
        if fh is not None:
            fh.close()
        raise

    # 分片轮换可能在尾部留下 0 token 的空分片: numpy 无法 mmap 空文件,
    # 而且它会把 manifest 的窗口数算错, 所以直接删掉。
    empty = [f for f in files if f["n_tokens"] <= 0]
    if empty:
        for f in empty:
            (out_dir / f["path"]).unlink(missing_ok=True)
        files = [f for f in files if f["n_tokens"] > 0]

    written_total = int(sum(f["n_tokens"] for f in files))
    if written_total != total_written:
        raise DataPipelineError(
            f"{split}: internal accounting mismatch, shards hold {written_total} tokens "
            f"but {total_written} were written"
        )
    return {
        "n_docs": len(docs),
        # n_tokens 只统计**真正写进 .bin** 的 token, 因此恒等于 bytes / itemsize。
        # 被 drop_last 丢掉的尾巴单独记在 remainder_tokens 里, 免得 manifest 出现
        # "声称 1363 个 token、磁盘上只有 1344 个" 这种账实不符。
        "n_tokens": written_total,
        "n_tokens_encoded": int(total_encoded),
        "n_windows": int(total_windows),
        "remainder_tokens": int(remainder),
        "files": files,
        "bytes": int(sum(f["bytes"] for f in files)),
        "dtype": np_dtype.name,
    }


def pack_documents(
    docs: Iterable[str],
    tokenizer: Any,
    seq_len: int,
    out_dir: Any,
    val_frac: float = 0.005,
    test_frac: float = 0.005,
    seed: int = 1234,
    *,
    clean_cfg: Optional[CleanConfig] = None,
    eos_id: Optional[int] = None,
    dtype: str = "uint32",
    shard_tokens: Optional[int] = 200_000_000,
    drop_last: bool = True,
    append_eos_after_last: bool = True,
    min_train_windows: int = 1,
    tokenizer_info: Optional[Any] = None,
    source_files: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every: int = 2000,
) -> PackResult:
    """清洗 -> 分词 -> 按文档切分 -> 切窗 -> 落盘 ``*.bin`` + ``index.json``。

    流程 (以及为什么这样排序):
      1. 先全量 :func:`clean_text`, 丢弃不合格文档, 按原因分别计数;
      2. 用 ``seed`` 洗牌**文档**列表, 按 ``val_frac`` / ``test_frac`` 切出
         val/test 整篇文档, 剩下是 train —— 这一步保证了文档级不泄漏;
      3. 每个 split 内单独拼流: 文档末尾追加 EOS, 顺序切成 ``seq_len`` 定长窗口,
         以 little-endian 无符号整型写入 ``*.bin``;
      4. 计算每个 ``*.bin`` 的 SHA-256, 连同词表大小、token 数、清洗配置一起
         写进 ``index.json`` (manifest)。

    Args:
        docs: 原始文档迭代器 (通常来自 :func:`iter_documents`)。清洗后的文本会留在
            内存里以便洗牌, 因此 ``CleanConfig.max_chars`` 是内存上界的主要旋钮。
        tokenizer: 任何提供 ``encode`` (可选 ``encode_batch`` / ``eos_id``) 的对象。
        seq_len: 窗口长度。
        out_dir: 输出目录 (不存在则创建)。
        val_frac: 验证集文档比例 (文档级)。
        test_frac: 测试集文档比例 (文档级)。
        seed: 文档洗牌种子。
        clean_cfg: :class:`CleanConfig`。
        eos_id: 文档分隔 id; ``None`` 时取 ``tokenizer.eos_id``。
        dtype: ``"uint32"`` (默认) 或 ``"uint16"`` (vocab < 65535 时省一半磁盘)。
        shard_tokens: 单分片最大 token 数; ``None``/``<=0`` 表示每 split 单文件。
        drop_last: 是否丢弃末尾不足一窗的 token。
        append_eos_after_last: 最后一篇文档后是否再补 EOS。
        min_train_windows: 训练窗口数下限, 低于它抛 :class:`DataPipelineError`。
        tokenizer_info: 分词器信息对象 (有 ``to_dict()``) 或 dict, 写进 manifest。
        source_files: 数据来源文件列表, 写进 manifest 便于追溯。
        progress_cb: 每 ``progress_every`` 篇文档回调一次 ``dict`` 进度。
        progress_every: 进度回调频率 (文档数); ``<=0`` 关闭。

    Returns:
        :class:`PackResult`。

    Raises:
        DataPipelineError: 清洗后无文档, 或训练窗口数不足。
        ValueError: ``seq_len`` / 比例 / ``dtype`` / ``eos_id`` 非法。
    """
    t_start = time.time()
    cfg = clean_cfg or CleanConfig()
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if not (0.0 <= float(val_frac) < 1.0) or not (0.0 <= float(test_frac) < 1.0):
        raise ValueError(f"val_frac/test_frac must be in [0, 1); got {val_frac}/{test_frac}")
    if float(val_frac) + float(test_frac) >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    np_dtype = np.dtype(dtype)
    if np_dtype.kind != "u":
        raise ValueError(f"dtype must be an unsigned integer type, got {dtype!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = _TokenizerAdapter(tokenizer)
    if eos_id is not None:
        adapter.eos_id = int(eos_id)
    if adapter.eos_id is None or int(adapter.eos_id) < 0:
        raise ValueError("eos_id could not be determined from the tokenizer; pass eos_id explicitly")
    adapter.eos_id = int(adapter.eos_id)

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size == 0:
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab_size = len(get_vocab())
    if vocab_size and adapter.eos_id >= vocab_size:
        raise ValueError(f"eos_id {adapter.eos_id} outside vocab (size {vocab_size})")
    if np_dtype == np.dtype("uint16") and (vocab_size == 0 or vocab_size >= 65535):
        raise ValueError("uint16 is unsafe: vocabulary size must be known and < 65535")

    # ---- 1) 清洗 ---------------------------------------------------------
    _LOG.info("cleaning documents (min_chars=%d max_chars=%d)", cfg.min_chars, cfg.max_chars)
    kept: List[str] = []
    reasons: Dict[str, int] = {}
    n_in = 0
    n_empty = 0
    chars = 0
    for raw in docs:
        n_in += 1
        if not isinstance(raw, str) or not raw:
            n_empty += 1
        else:
            cleaned, reason = classify_text(raw, cfg)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "kept":
                kept.append(cleaned)
                chars += len(cleaned)
        if progress_cb is not None and progress_every > 0 and n_in % progress_every == 0:
            progress_cb({"stage": "clean", "seen": n_in, "kept": len(kept), "dropped": n_in - len(kept)})

    n_kept = len(kept)
    dropped = {k: int(v) for k, v in reasons.items() if k != "kept"}
    if n_kept == 0:
        raise DataPipelineError(
            f"no documents survived cleaning (input={n_in}, reasons={dropped}). "
            "Check --min-chars / the parquet text column."
        )

    # ---- 2) 文档级切分 ---------------------------------------------------
    rng = random.Random(int(seed))
    rng.shuffle(kept)
    plan = _plan_document_splits(n_kept, float(val_frac), float(test_frac))
    split_docs: Dict[str, List[str]] = {}
    cursor = 0
    for name in ("val", "test", "train"):
        n = plan[name]
        split_docs[name] = kept[cursor:cursor + n]
        cursor += n
    _LOG.info("document-level split: %s (from %d kept docs)", plan, n_kept)

    # ---- 3) 打包 ---------------------------------------------------------
    split_stats: Dict[str, Dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        t0 = time.time()
        stats = _write_split(
            name, split_docs[name], adapter, int(seq_len), out_dir,
            np_dtype.name, shard_tokens, bool(drop_last), bool(append_eos_after_last),
        )
        stats["elapsed_seconds"] = round(time.time() - t0, 3)
        split_stats[name] = stats
        _LOG.info(
            "wrote %s: docs=%d tokens=%d windows=%d files=%d (%.1fs)",
            name, stats["n_docs"], stats["n_tokens"], stats["n_windows"],
            len(stats["files"]), stats["elapsed_seconds"],
        )

        if split_stats[name]["n_windows"] <= 0:
            raise DataPipelineError(
                f"split {name!r} has {split_stats[name]['n_docs']} document(s) but no full "
                f"{seq_len}-token window, so there is nothing to write; "
                f"lower --min-chars / --seq-len, or feed more documents"
            )

    if split_stats["train"]["n_windows"] < int(min_train_windows):
        raise DataPipelineError(
            f"train split produced {split_stats['train']['n_windows']} windows, "
            f"below min_train_windows={min_train_windows}; lower --min-chars, "
            "raise the document budget, or feed more documents"
        )

    total_tokens = int(sum(s["n_tokens"] for s in split_stats.values()))
    total_encoded = int(sum(s["n_tokens_encoded"] for s in split_stats.values()))
    total_remainder = int(sum(s["remainder_tokens"] for s in split_stats.values()))
    # 压缩比用**编码出的** token 数做分母才诚实 (写进文件的少了被丢掉的尾巴),
    # 否则改一个 drop_last 就能让压缩比看起来变好。
    ratio = (chars / total_encoded) if total_encoded else 0.0

    # ---- 4) manifest -----------------------------------------------------
    tok_info: Optional[dict] = None
    if tokenizer_info is not None:
        tok_info = tokenizer_info.to_dict() if hasattr(tokenizer_info, "to_dict") else dict(tokenizer_info)
    manifest = {
        "format": "simplcore.core.corpus",
        "format_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "eos_id": int(adapter.eos_id),
        "dtype": np_dtype.name,
        "byteorder": "little",
        "tokenizer": tok_info,
        "cleaning": cfg.to_dict(),
        "packing": {
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "seed": int(seed),
            "shard_tokens": None if shard_tokens is None else int(shard_tokens),
            "drop_last": bool(drop_last),
            "append_eos_after_last": bool(append_eos_after_last),
            "split_unit": "document",
        },
        "documents": {
            "n_in": int(n_in),
            "n_kept": int(n_kept),
            "n_dropped": int(sum(dropped.values())),
            "n_empty": int(n_empty),
            "drop_reasons": dropped,
            "chars": int(chars),
            "per_split": {k: int(v["n_docs"]) for k, v in split_stats.items()},
        },
        "total_tokens": int(total_tokens),
        "total_tokens_encoded": int(total_encoded),
        "dropped_tail_tokens": int(total_remainder),
        "compression_ratio": round(float(ratio), 6),
        "compression_basis": "chars of kept documents / encoded tokens (incl. dropped tail)",
        "splits": {
            k: {
                "n_docs": v["n_docs"],
                "n_tokens": v["n_tokens"],
                "n_tokens_encoded": v["n_tokens_encoded"],
                "n_windows": v["n_windows"],
                "remainder_tokens": v["remainder_tokens"],
                "bytes": v["bytes"],
                "elapsed_seconds": v["elapsed_seconds"],
                "files": v["files"],
            }
            for k, v in split_stats.items()
        },
        "source_files": [str(s) for s in (source_files or [])],
        "elapsed_seconds": round(time.time() - t_start, 3),
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return PackResult(
        out_dir=str(out_dir),
        manifest_path=str(manifest_path),
        n_docs_in=int(n_in),
        n_docs_kept=int(n_kept),
        n_docs_dropped=int(sum(dropped.values())),
        drop_reasons=dropped,
        n_docs_empty=int(n_empty),
        splits=manifest["splits"],
        total_tokens=int(total_tokens),
        total_tokens_encoded=int(total_encoded),
        chars=int(chars),
        compression_ratio=float(ratio),
        elapsed_seconds=round(time.time() - t_start, 3),
        source_files=[str(s) for s in (source_files or [])],
    )


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def read_manifest(path: Any) -> dict:
    """读取 corpus manifest。

    Args:
        path: ``index.json`` 路径, 或包含它的目录。
    """
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"corpus manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def shard_paths(manifest: dict, out_dir: Any, split: str = "train") -> List[Path]:
    """从 manifest 取出某个 split 的全部分片绝对路径 (按写入顺序)。"""
    out_dir = Path(out_dir)
    entry = (manifest.get("splits") or {}).get(split)
    if not entry:
        raise KeyError(f"split {split!r} not in manifest (have: {list((manifest.get('splits') or {}).keys())})")
    return [out_dir / f["path"] for f in entry["files"]]


def _load_memmap(path: Path, dtype: Any, expected_tokens: Optional[int] = None) -> np.ndarray:
    """把一个 ``.bin`` 映射成只读 numpy memmap, 并可选校验 token 数。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token shard missing: {path}")
    dt = np.dtype(dtype)
    size = path.stat().st_size
    if size % dt.itemsize:
        raise DataPipelineError(
            f"{path} size {size} is not a multiple of itemsize {dt.itemsize}; the file is truncated"
        )
    n = size // dt.itemsize
    if expected_tokens is not None and n != int(expected_tokens):
        raise DataPipelineError(f"{path} holds {n} tokens, manifest says {expected_tokens}")
    if n == 0:
        # numpy 无法 mmap 0 字节文件 ("cannot mmap an empty file"), 直接给空数组。
        return np.empty(0, dtype=dt)
    return np.memmap(str(path), dtype=dt, mode="r", shape=(n,))


class ShardedTokenDataset(Dataset):
    """按 ``seq_len`` 切窗的定长语言建模数据集 (mmap 分片, 常驻内存约等于 0)。

    每个样本是 ``(seq_len,)`` 的 ``torch.long`` 窗口 —— 对 causal LM 来说
    ``input_ids`` 与 ``labels`` 是同一个窗口错位对齐, 因此默认只返回输入窗口,
    由 collate / 模型侧做 ``labels = input_ids`` 的 shift。

    数据布局::

        train-00000.bin  train-00001.bin ...   # uint32 little-endian token 流

    窗口 ``i`` 就是 ``tokens[i * seq_len : (i + 1) * seq_len]``, 与
    ``pack_documents(drop_last=True)`` 的写入约定一致: 不足一窗的尾部根本不会
    写进文件, 因此**不存在**需要在这一层丢弃的孤儿 token。

    Args:
        root: 语料目录 (含 ``index.json`` 与 ``*.bin``), 或直接用 ``index.json`` 路径。
        split: ``"train"`` / ``"val"`` / ``"test"``。
        seq_len: 窗口长度; ``None`` 时用 manifest 里的值。
        dtype: ``None`` 时用 manifest 里的 ``dtype`` (默认 uint32)。
        verify_hashes: 加载时校验每个分片的 SHA-256 (默认 True; 换来"读到坏盘或
            半截下载文件"能被立刻发现, 代价是启动时多读一遍)。
        return_labels: 是否额外返回 ``labels`` (同一窗口; 便于直接在训练循环里 unpack)。
        token_dtype: 返回张量的 dtype, 默认 ``torch.long``。
        transform: 可选的一元变换, 作用在返回的 tensor 上。

    Attributes:
        manifest: 原始 manifest dict。
        vocab_size: 词表大小。
        eos_id: EOS id。
        num_tokens: 该 split 的 token 总数。
        num_windows: 窗口数。
        shards: 每个分片的 ``{path, n_tokens, offset, bytes}``。
    """

    def __init__(
        self,
        root: Any,
        split: str = "train",
        *,
        seq_len: Optional[int] = None,
        dtype: Optional[str] = None,
        verify_hashes: bool = True,
        return_labels: bool = False,
        token_dtype: torch.dtype = torch.long,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_dir() else root_path.parent
        self.manifest = read_manifest(root_path)
        self.split = str(split)
        if self.split not in (self.manifest.get("splits") or {}):
            raise KeyError(f"split {self.split!r} not in manifest {self.root / MANIFEST_NAME}")

        entry = self.manifest["splits"][self.split]
        self.seq_len = int(seq_len if seq_len is not None else self.manifest["seq_len"])
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        self.dtype = np.dtype(dtype or self.manifest.get("dtype", "uint32"))
        if self.dtype.kind != "u":
            raise ValueError(f"manifest dtype must be unsigned, got {self.dtype}")
        self.vocab_size = int(self.manifest.get("vocab_size", 0) or 0)
        self.eos_id = int(self.manifest.get("eos_id", -1))
        self.return_labels = bool(return_labels)
        self.token_dtype = token_dtype
        self.transform = transform

        files = list(entry["files"])
        if not files:
            raise DataPipelineError(f"split {self.split!r} has no shards in the manifest")

        self.shards: List[Dict[str, Any]] = []
        offsets: List[int] = []
        running = 0
        for f in files:
            path = self.root / f["path"]
            if verify_hashes and f.get("sha256"):
                actual = compute_sha256(path)
                if actual != f["sha256"]:
                    raise DataPipelineError(
                        f"sha256 mismatch for {path}: manifest={f['sha256'][:16]}... actual={actual[:16]}..."
                    )
            offsets.append(running)
            self.shards.append({
                "path": str(path),
                "n_tokens": int(f.get("n_tokens", 0)),
                "offset": int(running),
                "bytes": int(path.stat().st_size),
            })
            running += int(f.get("n_tokens", 0))
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._arrays: List[np.ndarray] = [
            _load_memmap(Path(s["path"]), self.dtype, expected_tokens=s["n_tokens"]) for s in self.shards
        ]
        self.num_tokens = int(running)
        # 窗口数始终按"实际 token 数 // seq_len"计算: 文件里最后一个 token 永远
        # 可被某个窗口用到 (写入时已保证 token 数正好是整数个 seq_len 窗口)。
        # manifest 里的 n_windows 只是同一算式在 seq_len 下的快照, 仅作交叉校验。
        self.num_windows = self.num_tokens // self.seq_len
        declared = int(entry.get("n_windows", self.num_windows))
        if declared != self.num_windows:
            _LOG.warning(
                "%s: manifest declares %d windows of %d, the %d tokens on disk give %d; using the disk value",
                self.split, declared, self.seq_len, self.num_tokens, self.num_windows,
            )
        if self.num_windows <= 0:
            raise DataPipelineError(
                f"split {self.split!r} has {self.num_tokens} tokens, fewer than one window of {self.seq_len}"
            )

    # -- 读取 ---------------------------------------------------------------
    def _window(self, index: int) -> np.ndarray:
        """取第 ``index`` 个窗口的 numpy 视图 (跨分片时做一次拷贝)。"""
        start = index * self.seq_len
        shard_idx = int(np.searchsorted(self._offsets, start, side="right") - 1)
        shard_idx = max(shard_idx, 0)
        arr = self._arrays[shard_idx]
        local = start - int(self._offsets[shard_idx])
        if local + self.seq_len <= arr.size:
            return arr[local:local + self.seq_len]
        head = arr[local:]
        need = self.seq_len - head.size
        if shard_idx + 1 >= len(self._arrays):
            raise DataPipelineError(f"window {index} runs past the last shard of split {self.split!r}")
        return np.concatenate((head, self._arrays[shard_idx + 1][:need]))

    def __len__(self) -> int:
        """窗口数 (``num_tokens // seq_len``)。"""
        return int(self.num_windows)

    def __getitem__(self, index: int):
        """返回 ``(seq_len,)`` 的窗口; ``return_labels=True`` 时返回 ``(x, y)``。"""
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"index {index} out of range for split {self.split!r} ({n} windows)")
        # mmap 是只读视图, torch.from_numpy 会为"不可写数组"发警告, 并让张量
        # 直接引用磁盘页; 这里拷一份 (每次一个窗口, 开销可忽略) 换取干净的语义。
        window = np.asarray(self._window(index)).copy()
        x = torch.from_numpy(window).to(self.token_dtype)
        if self.transform is not None:
            x = self.transform(x)
        if self.return_labels:
            return x, x.clone()
        return x

    # -- 内省 ---------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ShardedTokenDataset(split={self.split!r}, seq_len={self.seq_len}, "
            f"windows={self.num_windows}, tokens={self.num_tokens}, "
            f"dtype={self.dtype.name}, shards={len(self.shards)})"
        )

    def token_stream(self) -> Iterator[int]:
        """顺序产出该 split 的全部 token (调试/统计用; 大语料上很慢)。"""
        for arr in self._arrays:
            yield from arr.tolist()

    def num_bytes(self) -> int:
        """该 split 在磁盘上的字节数。"""
        return int(sum(s["bytes"] for s in self.shards))

    def close(self) -> None:
        """释放 mmap 句柄 (Windows 上删除/覆盖文件前必须先调用)。"""
        self._arrays = []

    def __getstate__(self) -> dict:
        """支持 ``DataLoader(num_workers>0)``: 不把 mmap 对象跨进程序列化。"""
        state = self.__dict__.copy()
        state["_arrays"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        """反序列化后按需重新打开 mmap (每个 worker 自己持有句柄)。"""
        self.__dict__.update(state)
        self._arrays = [
            _load_memmap(Path(s["path"]), self.dtype, expected_tokens=s["n_tokens"]) for s in self.shards
        ]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def load_shard(path: Any, start: int = 0, count: Optional[int] = None, dtype: str = "uint32") -> np.ndarray:
    """读一个 token 分片的一段 (默认全部), 返回 numpy 数组。

    这是"``.bin`` 能否解析"最直接的抓手: 它不依赖 manifest,
    只按字节数推断 token 数并检查是否整除了 itemsize。
    """
    arr = _load_memmap(Path(path), np.dtype(dtype))
    if count is None:
        return np.asarray(arr[start:])
    return np.asarray(arr[start:start + int(count)])


def load_split(
    root: Any,
    split: str = "train",
    *,
    seq_len: Optional[int] = None,
    verify_hashes: bool = True,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """把某个 split 读成 ``(n_windows, seq_len)`` 数组 (调试用, 会占内存)。"""
    ds = ShardedTokenDataset(root, split, seq_len=seq_len, verify_hashes=verify_hashes)
    n = len(ds) if max_windows is None else min(len(ds), int(max_windows))
    out = np.empty((n, ds.seq_len), dtype=ds.dtype)
    for i in range(n):
        out[i] = ds[i]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """构造 ``python -m simplcore.core.data --help`` 用的参数解析器。"""
    p = argparse.ArgumentParser(
        prog="python -m simplcore.core.data",
        description="SGC 真实数据管道: parquet -> 清洗 -> 分词 -> 定长窗口 token 分片。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet-dir", required=True, help="原始 parquet 目录 (递归; 跳过 .incomplete/.lock)")
    p.add_argument("--out", required=True, help="输出目录 (train/val/test .bin + index.json)")
    p.add_argument("--tokenizer", default=None, help="已训练好的分词器目录或 tokenizer.json")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--test-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=("uint32", "uint16"), default="uint32")
    p.add_argument("--shard-tokens", type=int, default=200_000_000,
                   help="单分片最大 token 数 (0 = 每 split 单文件)")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--min-keep-ratio", type=float, default=0.10)
    p.add_argument("--noise-ratio", type=float, default=0.40)
    p.add_argument("--keep-noise-lines", action="store_true", help="保留疑似代码/日志行 (消融用)")
    p.add_argument("--strip-urls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--text-column", default=None, help="文本列名; 默认自动识别 text/content/raw_content")
    p.add_argument("--max-docs", type=int, default=None, help="最多处理多少篇文档 (冒烟用)")
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m simplcore.core.data`` 的入口。

    单独跑它时**不会**训练分词器: 必须给 ``--tokenizer``, 否则报错退出。
    完整流水线 (训练分词器 + 打包 + 汇总表) 请用 ``scripts/build_corpus.py``。
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.tokenizer:
        _LOG.error("--tokenizer is required here; use scripts/build_corpus.py for the full pipeline")
        return 2

    from . import tokenizer as _tok

    tok = _tok.load_tokenizer(args.tokenizer)
    cfg = CleanConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_keep_ratio=args.min_keep_ratio,
        noise_ratio_threshold=args.noise_ratio,
        drop_noise_lines=not args.keep_noise_lines,
        strip_urls=bool(args.strip_urls),
    )
    files = _iter_parquet_files(args.parquet_dir)
    result = pack_documents(
        iter_documents(args.parquet_dir, text_column=args.text_column, max_docs=args.max_docs),
        tok,
        args.seq_len,
        args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        clean_cfg=cfg,
        dtype=args.dtype,
        shard_tokens=None if not args.shard_tokens else int(args.shard_tokens),
        drop_last=bool(args.drop_last),
        source_files=[str(f) for f in files],
    )
    print(result.summary_table())
    print(f"\nmanifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
