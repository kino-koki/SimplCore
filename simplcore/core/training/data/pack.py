# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""打包落盘：分词、EOS 插入、按文档划分 split、定长窗口切片与 index.json。"""
from __future__ import annotations


from .clean import classify_text
from .constants import _LOG, MANIFEST_NAME, UINT32_MAX
from .encode import _TokenizerAdapter, count_tokens, encode_document
from .types import CleanConfig, DataPipelineError, PackConfig, PackResult

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np



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

