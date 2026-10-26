#!/usr/bin/env python
# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""端到端语料构建 CLI (end-to-end corpus build for SGC).

一条命令走完: 扫描 parquet -> 采样 -> 训练分词器 -> 清洗打包 -> 切窗落盘 -> 汇总表::

    python scripts/build_corpus.py \\
        --parquet-dir data/raw/fineweb_zh \\
        --out data/processed/sgc_zh \\
        --vocab-size 32000 --seq-len 4096 --val-frac 0.005

产出 (all inside ``--out``)::

    tokenizer.json        分词器本体 (ByteLevel BPE)
    tokenizer_meta.json   分词器配置 + 实测压缩比
    train-00000.bin ...   uint32 little-endian token 分片
    val-00000.bin
    test-00000.bin
    index.json            manifest: 词表大小 / token 数 / seq_len / 清洗配置 / 每个 .bin 的 SHA-256

设计取舍 (why it looks like this)
--------------------------------
* **分词器只在有界样本上训练**: ``--tokenizer-sample-mb`` (默认 200 MB) 决定采样上限。
  全量训练 BPE 对 32k 词表几乎没有收益, 却把耗时从分钟级推到小时级。
* **打包分两遍读 parquet**: 采样阶段顺带把样本留在内存里复用给压缩比量测,
  打包阶段重新流式读盘 (内存占用与语料总量无关)。
* **分词器缓存**: ``--out`` 下已有 ``tokenizer.json`` 且未给 ``--force`` 时直接复用,
  重跑打包不会重复训练。
* **失败可读**: 单个 parquet 分片损坏 (下载中断) 只告警跳过, 不中断整条流水线;
  真到了"一篇文档都没留下"才报错退出。

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 允许 "python scripts/build_corpus.py" 直接运行 (不依赖已安装的包)。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simplcore.sgc import data as sgc_data  # noqa: E402
from simplcore.sgc import tokenizer as sgc_tok  # noqa: E402
from simplcore.sgc.data import CleanConfig, DataPipelineError  # noqa: E402

_LOG = logging.getLogger("build_corpus")

MB = 1024 * 1024


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器 (``--help`` 的全部内容都在这里)。"""
    p = argparse.ArgumentParser(
        prog="python scripts/build_corpus.py",
        description="SGC 端到端语料构建: parquet -> 分词器 -> 清洗 -> 定长窗口 token 分片。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("输入 / 输出 (input / output)")
    src.add_argument("--parquet-dir", default="data/raw/fineweb_zh",
                     help="原始 parquet 目录 (递归; 自动跳过 .incomplete/.lock)")
    src.add_argument("--out", default="data/processed/sgc_zh", help="输出目录")
    src.add_argument("--text-column", default=None,
                     help="文本列名; 默认自动识别 text/content/raw_content")
    src.add_argument("--force", action="store_true", help="即使 --out 下已有分词器也重新训练")

    tok = p.add_argument_group("分词器 (tokenizer)")
    tok.add_argument("--vocab-size", type=int, default=32000, help="词表大小")
    tok.add_argument("--tokenizer-kind", choices=("bpe", "char"), default="bpe",
                     help="bpe = 字节级 BPE; char = 字符表回退 (秒级完成)")
    tok.add_argument("--tokenizer-sample-mb", type=float, default=200.0,
                     help="训练分词器时最多读取的原始文本 MB (0 = 不限, 慎用)")
    tok.add_argument("--tokenizer-sample-docs", type=int, default=200_000,
                     help="训练分词器时最多读取的文档数")
    tok.add_argument("--min-frequency", type=int, default=2, help="BPE merge 的最小频次")

    pack = p.add_argument_group("打包 (packing)")
    pack.add_argument("--seq-len", type=int, default=4096, help="训练窗口长度")
    pack.add_argument("--val-frac", type=float, default=0.005, help="验证集**文档**比例")
    pack.add_argument("--test-frac", type=float, default=0.005, help="测试集**文档**比例")
    pack.add_argument("--seed", type=int, default=1234, help="文档洗牌种子")
    pack.add_argument("--dtype", choices=("uint32", "uint16"), default="uint32",
                      help="token 存储宽度 (uint16 省一半磁盘, 要求 vocab < 65535)")
    pack.add_argument("--shard-tokens", type=int, default=200_000_000,
                      help="单个 .bin 的最大 token 数 (0 = 每个 split 只写一个文件)")
    pack.add_argument("--max-docs", type=int, default=None, help="最多处理多少篇文档 (冒烟用)")

    cln = p.add_argument_group("清洗 (cleaning)")
    cln.add_argument("--min-chars", type=int, default=200, help="清洗后文档的最少字符数")
    cln.add_argument("--max-chars", type=int, default=200_000, help="清洗后文档的最多字符数")
    cln.add_argument("--max-lines", type=int, default=20_000, help="文档最大行数 (超出整篇丢弃)")
    cln.add_argument("--min-keep-ratio", type=float, default=0.10, help="清洗后保留率下限")
    cln.add_argument("--noise-ratio", type=float, default=0.40, help="行内符号占比的噪声阈值")
    cln.add_argument("--keep-noise-lines", action="store_true",
                     help="保留疑似代码/日志行 (消融对比用)")
    cln.add_argument("--strip-urls", action=argparse.BooleanOptionalAction, default=True,
                     help="把 URL/邮箱折叠成 <url>/<email> 占位符")

    misc = p.add_argument_group("其它 (misc)")
    misc.add_argument("--log-level", default="INFO")
    misc.add_argument("--json-summary", default=None, help="把最终汇总额外写成 JSON 到这个路径")
    return p


# ---------------------------------------------------------------------------
# 采样与训练
# ---------------------------------------------------------------------------
def sample_for_tokenizer(parquet_dir: str, *, max_bytes: float, max_docs: int,
                         text_column: Optional[str]) -> Dict[str, object]:
    """在有界预算内采样文档, 用于训练分词器。

    采样在**原始字节**上封顶 (``max_bytes`` 是 UTF-8 字节数), 因为磁盘/网络代价
    是按字节算的, 而不是按文档数。返回的文本列表会被复用给压缩比量测,
    因此不需要二次读盘。

    Args:
        parquet_dir: parquet 目录。
        max_bytes: 文本字节上限; ``<= 0`` 表示不限。
        max_docs: 文档数上限。
        text_column: 文本列名或 ``None`` (自动识别)。

    Returns:
        ``{"texts": [...], "chars": int, "bytes": int, "docs": int, "seconds": float}``。
    """
    t0 = time.time()
    texts: List[str] = []
    chars = 0
    nbytes = 0
    for doc in sgc_data.iter_documents(parquet_dir, text_column=text_column):
        if not isinstance(doc, str) or not doc:
            continue
        texts.append(doc)
        chars += len(doc)
        nbytes += len(doc.encode("utf-8"))
        if len(texts) >= int(max_docs):
            break
        if max_bytes and max_bytes > 0 and nbytes >= max_bytes:
            break
    return {
        "texts": texts,
        "chars": chars,
        "bytes": nbytes,
        "docs": len(texts),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    """执行完整流水线, 返回进程退出码 (0 = 成功)。"""
    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_json = out_dir / "tokenizer.json"

    files = sgc_data._iter_parquet_files(args.parquet_dir)
    if not files:
        _LOG.error("no usable .parquet files under %s (download still running?)", args.parquet_dir)
        return 3
    total_bytes = sum(f.stat().st_size for f in files)
    _LOG.info(
        "found %d parquet file(s), %.1f MB total, under %s",
        len(files), total_bytes / MB, args.parquet_dir,
    )
    for f in files:
        _LOG.info("  %s  (%.1f MB)", f, f.stat().st_size / MB)

    # ---- 1) 分词器 -------------------------------------------------------
    reuse = tok_json.exists() and not args.force
    tokenizer_info = None
    if reuse:
        _LOG.info("reusing existing tokenizer: %s", tok_json)
        try:
            tokenizer_info = sgc_tok.load_tokenizer(out_dir)
            _LOG.info("  vocab_size=%d eos_id=%d", tokenizer_info.vocab_size, tokenizer_info.eos_id)
        except Exception as exc:  # noqa: BLE001 - 缓存坏了就重训
            _LOG.warning("existing tokenizer unusable (%r); retraining", exc)
            reuse = False

    if not reuse:
        sample = sample_for_tokenizer(
            args.parquet_dir,
            max_bytes=args.tokenizer_sample_mb * MB,
            max_docs=args.tokenizer_sample_docs,
            text_column=args.text_column,
        )
        if sample["docs"] == 0:
            _LOG.error("tokenizer sample is empty; nothing to train on")
            return 3
        _LOG.info(
            "tokenizer sample: %d docs, %.1f MB text, %.1f M chars (%.1fs)",
            sample["docs"], sample["bytes"] / MB, sample["chars"] / 1e6, sample["seconds"],
        )
        t0 = time.time()
        _LOG.info("training %s tokenizer (vocab_size=%d) ...", args.tokenizer_kind, args.vocab_size)
        tokenizer_info = sgc_tok.train_tokenizer(
            (iter(sample["texts"]), sample["texts"]),
            args.vocab_size,
            out_dir,
            kind=args.tokenizer_kind,
            sample_chars=None,
            min_frequency=args.min_frequency,
            measure_ratio=True,
        )
        _LOG.info(
            "tokenizer done: vocab=%d eos_id=%d compression=%.4f chars/token (%.1fs)",
            tokenizer_info.vocab_size, tokenizer_info.eos_id,
            tokenizer_info.compression_ratio or 0.0, time.time() - t0,
        )

    # ---- 2) 清洗 + 打包 --------------------------------------------------
    clean_cfg = CleanConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_lines=args.max_lines,
        min_keep_ratio=args.min_keep_ratio,
        noise_ratio_threshold=args.noise_ratio,
        drop_noise_lines=not args.keep_noise_lines,
        strip_urls=bool(args.strip_urls),
        strip_emails=bool(args.strip_urls),
    )
    _LOG.info("packing: seq_len=%d val_frac=%g test_frac=%g dtype=%s shard_tokens=%s",
              args.seq_len, args.val_frac, args.test_frac, args.dtype, args.shard_tokens)
    t0 = time.time()

    def _progress(info: dict) -> None:
        _LOG.info(
            "clean progress: seen=%d kept=%d dropped=%d (%.0f%%)",
            info["seen"], info["kept"], info["dropped"],
            100.0 * info["kept"] / max(info["seen"], 1),
        )

    try:
        result = sgc_data.pack_documents(
            sgc_data.iter_documents(args.parquet_dir, text_column=args.text_column,
                                     max_docs=args.max_docs),
            sgc_tok.load_tokenizer(out_dir),
            args.seq_len,
            out_dir,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            clean_cfg=clean_cfg,
            dtype=args.dtype,
            shard_tokens=None if not args.shard_tokens else int(args.shard_tokens),
            tokenizer_info=tokenizer_info,
            source_files=[str(f) for f in files],
            progress_cb=_progress,
            progress_every=20_000,
        )
    except DataPipelineError as exc:
        _LOG.error("packing failed: %s", exc)
        return 4

    packing_seconds = time.time() - t0

    # ---- 3) 汇总 ---------------------------------------------------------
    manifest = sgc_data.read_manifest(out_dir)
    elapsed = time.time() - t_start
    print("")
    print("=" * 78)
    print("SGC corpus build summary / 语料构建汇总")
    print("=" * 78)
    print(result.summary_table())
    print("")
    tok_ratio = tokenizer_info.compression_ratio if tokenizer_info is not None else None
    rows = [
        ("parquet files read", f"{len(files)}  ({total_bytes / MB:.1f} MB on disk)"),
        ("output dir", str(out_dir)),
        ("tokenizer kind", getattr(tokenizer_info, "kind", "?")),
        ("vocab size", f"{manifest['vocab_size']} (eos_id={manifest['eos_id']})"),
        ("tokenizer compression", "n/a" if tok_ratio is None else f"{tok_ratio:.4f} chars/token (sampled)"),
        ("corpus compression", f"{result.compression_ratio:.4f} chars/token (kept docs / all tokens)"),
        ("documents in", f"{result.n_docs_in}"),
        ("documents kept", f"{result.n_docs_kept} ({100.0 * result.n_docs_kept / max(result.n_docs_in, 1):.1f}%)"),
        ("documents dropped", f"{result.n_docs_dropped}  {result.drop_reasons}"),
        ("empty / non-str", f"{result.n_docs_empty}"),
        ("kept chars", f"{result.chars} ({result.chars / 1e6:.2f} M)"),
        ("total tokens", f"{result.total_tokens} ({result.total_tokens / 1e6:.2f} M)"),
        ("train windows", f"{result.splits['train']['n_windows']} x {manifest['seq_len']}"),
        ("val windows", f"{result.splits['val']['n_windows']} x {manifest['seq_len']}"),
        ("test windows", f"{result.splits['test']['n_windows']} x {manifest['seq_len']}"),
        ("disk written", f"{sum(s['bytes'] for s in result.splits.values()) / MB:.1f} MB"),
        ("tokenize+pack time", f"{packing_seconds:.1f} s"),
        ("total wall time", f"{elapsed:.1f} s"),
        ("manifest", result.manifest_path),
    ]
    width = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)} : {v}")
    print("=" * 78)

    if args.json_summary:
        import json

        payload = {
            "out_dir": str(out_dir),
            "parquet_files": [str(f) for f in files],
            "parquet_bytes": int(total_bytes),
            "tokenizer": tokenizer_info.to_dict() if hasattr(tokenizer_info, "to_dict") else None,
            "manifest": manifest,
            "elapsed_seconds": round(elapsed, 3),
            "packing_seconds": round(packing_seconds, 3),
        }
        Path(args.json_summary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\njson summary -> {args.json_summary}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口: 解析参数 -> 跑流水线 -> 返回退出码。"""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
