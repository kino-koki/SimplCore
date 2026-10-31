# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""把缓存 / 临时目录固定到 E 盘。

为什么需要它
------------
C 盘只剩 7.8G（94% 已用），而 Python 生态默认把临时文件、torch 权重、
HF 缓存全写在 ``C:\\Users\\<name>\\AppData\\Local\\Temp`` 与 ``~/.cache``。
一次带记忆的训练就会在 C 盘留下几十兆临时文件，跑几次就把系统盘填满。

为什么不能只靠环境变量 / sitecustomize
-------------------------------------
1. 已运行的进程不会重读环境变量；不同入口（pytest / 子进程 / IDE）未必继承。
2. venv 里的 ``sitecustomize`` 会被宿主 CLI 自带的同名模块抢先加载
   （``sys.path`` 里 CLI 的 vendor 目录排在 site-packages 之前）——实测如此。
3. ``tempfile.tempdir`` 是惰性计算后**缓存**的。上游只要在导入我们之前调过
   一次 ``gettempdir()``，之后再改环境变量也没用，临时文件照样落 C 盘。

所以：显式调用 + 清缓存。:func:`apply` 幂等，重复调用无副作用。
"""
from __future__ import annotations

import os
import sys
from typing import Dict

#: 默认缓存根目录。可用 ``SIMPLCORE_CACHE_ROOT`` 覆盖（例如换机器）。
DEFAULT_CACHE_ROOT = r"E:\SimplCore\cache"

_LAYOUT: Dict[str, str] = {
    # 候选列表的第一项就是 TMPDIR：漏掉它，前面全白改。
    "TMPDIR": "tmp",
    "TEMP": "tmp",
    "TMP": "tmp",
    "TORCH_HOME": "torch",
    "HF_HOME": "hf",
    "HUGGINGFACE_HUB_CACHE": os.path.join("hf", "hub"),
    "TRANSFORMERS_CACHE": os.path.join("hf", "transformers"),
    "HF_DATASETS_CACHE": os.path.join("hf", "datasets"),
    "PIP_CACHE_DIR": "pip",
    "XDG_CACHE_HOME": "xdg",
    "MPLCONFIGDIR": "mpl",
}

_applied = False


def apply(root: str | None = None) -> str:
    """设置环境变量并清掉 ``tempfile`` 的缓存，返回实际使用的根目录。"""
    global _applied
    base = root or os.environ.get("SIMPLCORE_CACHE_ROOT") or DEFAULT_CACHE_ROOT

    for key, sub in _LAYOUT.items():
        path = os.path.join(base, sub)
        # 强制覆盖：TEMP/TMP 本来就有值且指向 C 盘，setdefault 等于没改。
        os.environ[key] = path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass

    # 清掉已缓存的结果，让 gettempdir() 按新环境重算。
    mod = sys.modules.get("tempfile")
    if mod is not None:
        mod.tempdir = None

    _applied = True
    return base


def cache_root() -> str:
    return os.environ.get("SIMPLCORE_CACHE_ROOT") or DEFAULT_CACHE_ROOT
