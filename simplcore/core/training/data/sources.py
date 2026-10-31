# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""语料读取：流式迭代 parquet、文本列识别、未完成文件跳过。"""
from __future__ import annotations


from .constants import _LOG, DEFAULT_TEXT_COLUMNS, SKIP_DIR_NAMES, SKIP_MARKERS

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple




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

