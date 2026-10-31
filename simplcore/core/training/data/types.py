# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""配置 dataclass 与流水线结果类型。"""
from __future__ import annotations


from .constants import MIN_SYMBOL_RATIO_CHARS

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple




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

