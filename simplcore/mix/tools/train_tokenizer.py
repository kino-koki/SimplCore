# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix — BPE tokenizer 一键训练脚本。

在服务器上从真实中文语料训练一个可用的 BPE tokenizer，输出 tokenizer.json。
这就是堆叠命门 4 的修复工具：把字符哈希换成真 BPE，中文能力的地基。

输入（三选一）:
  --corpus-jsonl <file>    JSONL，每行含 text 字段
  --corpus-dir <dir>       目录，递归读 *.jsonl / *.txt
  --corpus-parquet <dir>   目录，读 *.parquet（text 列）

采样控制（训练 BPE 只需代表性样本，别喂全量）:
  --max-bytes <int>        采样文本上限（默认 500MB；够训练 32k vocab）
  --max-docs <int>         最多读入文档数

用法（服务器上一键）:
  python -m simplcore.mix.tools.train_tokenizer \
      --corpus-dir /path/to/raw/Fineweb-Edu-Chinese-V2.2/2_3 \
      --vocab-size 32000 --max-bytes 536870912 \
      --output /path/to/tokenizer.json --chinese
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List

# 让 `python -m simplcore.mix.tools.train_tokenizer` 在仓库根可用
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simplcore.mix.tokenizer import BPETokenizer  # noqa: E402


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                yield text


def iter_dir_texts(dir_path: Path) -> Iterator[str]:
    for fp in sorted(dir_path.rglob("*.jsonl")):
        yield from iter_jsonl_texts(fp)
    for fp in sorted(dir_path.rglob("*.txt")):
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            yield from f


def iter_parquet_texts(dir_path: Path, text_column: str = "text") -> Iterator[str]:
    import pyarrow.parquet as pq

    for fp in sorted(dir_path.rglob("*.parquet")):
        try:
            table = pq.read_table(fp, columns=[text_column])
            for value in table[text_column].to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value
        except Exception as exc:  # 单分片失败不中断
            print(f"[train_tokenizer] 跳过分片 {fp}: {exc}")


def sample_stream(
    it: Iterator[str], *, max_bytes: int, max_docs: int
) -> Iterator[str]:
    """采样文本流：控制总字节与文档数，防止全量喂爆内存。"""
    seen_bytes = 0
    seen_docs = 0
    for text in it:
        if max_docs and seen_docs >= max_docs:
            break
        seen_docs += 1
        b = len(text.encode("utf-8"))
        if seen_bytes + b > max_bytes:
            break
        seen_bytes += b
        yield text
    print(f"[train_tokenizer] 采样: {seen_docs} 文档, {seen_bytes / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="SimplCore BPE tokenizer 训练")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus-jsonl", type=Path)
    src.add_argument("--corpus-dir", type=Path)
    src.add_argument("--corpus-parquet", type=Path)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-bytes", type=int, default=500 * 1024 * 1024)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--chinese", action="store_true", default=True,
                        help="中文友好预分词（默认开）")
    parser.add_argument("--no-chinese", action="store_true")
    args = parser.parse_args()

    # 数据流
    if args.corpus_jsonl:
        raw = iter_jsonl_texts(args.corpus_jsonl)
    elif args.corpus_dir:
        raw = iter_dir_texts(args.corpus_dir)
    else:
        raw = iter_parquet_texts(args.corpus_parquet)

    texts = sample_stream(raw, max_bytes=args.max_bytes, max_docs=args.max_docs)

    chinese = args.chinese and not args.no_chinese
    tok = BPETokenizer(vocab_size=args.vocab_size, chinese=chinese)
    print(f"[train_tokenizer] 开始训练 BPE: vocab={args.vocab_size} chinese={chinese}")
    t0 = time.time()
    tok.train(list(texts), verbose=True)
    print(f"[train_tokenizer] 训练耗时 {time.time() - t0:.1f}s, 最终 vocab={len(tok)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.output))
    print(f"[train_tokenizer] 已保存: {args.output}")
    print(f"    布局: pad={tok.pad_token_id} eos={tok.eos_token_id} "
          f"字节={2}..257 merge=258..{258 + len(tok.merges) - 1}")

    # 快速自检：round-trip
    probe = "安徽省六安市皋城中学，训练深度学习模型。"
    ids = tok.encode(probe)
    back = tok.decode(ids)
    ok = back == probe
    print(f"[train_tokenizer] 自检: {probe!r} -> {len(ids)} tokens "
          f"-> {back!r} {'✓' if ok else '✗ 不一致！'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
