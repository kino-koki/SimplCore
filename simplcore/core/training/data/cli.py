# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""命令行入口。"""

from .constants import _LOG
from .pack import pack_documents
from .sources import _iter_parquet_files, iter_documents
from .types import CleanConfig

import argparse
import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple




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

    from ..model import tokenizer as _tok

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

