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
from .constants import (DEFAULT_TEXT_COLUMNS, MANIFEST_NAME,
                        MIN_SYMBOL_RATIO_CHARS, SKIP_DIR_NAMES,
                        SKIP_MARKERS, UINT32_MAX)
from .clean import classify_text, clean_text, looks_like_noise
from .dataset import ShardedTokenDataset, load_shard, load_split, read_manifest, shard_paths
from .encode import count_tokens, encode_document
from .pack import compute_sha256, pack_documents, _plan_document_splits
from .sources import _iter_parquet_files, iter_documents, pick_text_column
from .types import CleanConfig, DataPipelineError, PackConfig, PackResult

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
