# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
from __future__ import annotations

"""共享常量与 logger（拆分自原 data.py 头部）。"""

import logging
from typing import Tuple

_LOG = logging.getLogger("simplcore.core.data")

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

