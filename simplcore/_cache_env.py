# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""把缓存 / 临时目录固定到非系统盘。

为什么需要它
------------
C 盘容量紧张（90%+ 已用），而 Python 生态默认把临时文件、torch 权重、
HF 缓存全写在 ``C:\\Users\\<name>\\AppData\\Local\\Temp`` 与 ``~/.cache``。
一次带记忆的训练就会在系统盘留下几十兆临时文件，跑几次就把系统盘填满。

为什么不能只靠环境变量 / sitecustomize
-------------------------------------
1. 已运行的进程不会重读环境变量；不同入口（pytest / 子进程 / IDE）未必继承。
2. venv 里的 ``sitecustomize`` 会被宿主 CLI 自带的同名模块抢先加载
   （``sys.path`` 里 CLI 的 vendor 目录排在 site-packages 之前）——实测如此。
3. ``tempfile.tempdir`` 是惰性计算后**缓存**的。上游只要在导入我们之前调过
   一次 ``gettempdir()``，之后再改环境变量也没用，临时文件照样落系统盘。

所以：显式调用 + 清缓存。:func:`apply` 幂等，重复调用无副作用。

缓存根目录的选取（按顺序，第一个可创建者胜出）
----------------------------------------------
1. ``SIMPLCORE_CACHE_ROOT`` 环境变量（显式覆盖，例如换机器）；
2. **仓库所在盘** ``<repo-drive>:\\SimplCore\\cache``——缓存随工作区同盘，
   换机器 / 换盘符自动跟随（本模块的历史默认 E:\\ 已随机器变更失效，
   这是不再硬编码盘符的原因）；
3. 其余非 C 盘依次兜底。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

#: 历史默认根目录（兼容旧部署；仅在它真的可写时才会被选中）。
LEGACY_CACHE_ROOT = r"E:\SimplCore\cache"

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
_active_root: str = ""


def _repo_drive() -> str:
    """仓库（本文件所属包）所在的盘符，如 ``I:``；取不到则返回空串."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.splitdrive(repo_root)[0]


#: 仓库同盘的默认缓存根目录（换机器自动跟随工作区盘符）。
DEFAULT_CACHE_ROOT = os.path.join(_repo_drive() + os.sep, "SimplCore", "cache")


def candidate_roots() -> List[str]:
    """缓存根目录候选列表（按优先级；不含重复项）."""
    cands: List[str] = []
    env = os.environ.get("SIMPLCORE_CACHE_ROOT")
    if env:
        cands.append(env)
    if DEFAULT_CACHE_ROOT:
        cands.append(DEFAULT_CACHE_ROOT)
    cands.append(LEGACY_CACHE_ROOT)
    for d in "DFGHIJKLMNOPQRSTUVWXYZ":
        cands.append(d + ":\\SimplCore\\cache")
    seen, out = set(), []
    for c in cands:
        key = c.upper()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _try_base(base: str) -> bool:
    """试探 base：所有布局子目录都能创建才返回 True."""
    for sub in _LAYOUT.values():
        try:
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        except OSError:
            return False
    return True


def apply(root: str | None = None) -> str:
    """设置环境变量并清掉 ``tempfile`` 的缓存，返回实际使用的根目录.

    ``root`` 显式指定时只试它；否则按 :func:`candidate_roots` 顺序挑第一个
    可创建的。全部失败时抛 :class:`RuntimeError`（带完整候选清单），绝不
    静默回落到系统盘——那是本模块要防的事故。
    """
    global _applied, _active_root
    if _applied and _active_root and root is None:
        return _active_root

    tried: List[str] = []
    for base in ([root] if root else candidate_roots()):
        if not base:
            continue
        tried.append(base)
        if _try_base(base):
            for key, sub in _LAYOUT.items():
                # 强制覆盖：TEMP/TMP 本来就有值且指向系统盘，setdefault 等于没改。
                os.environ[key] = os.path.join(base, sub)
            # 清掉已缓存的结果，让 gettempdir() 按新环境重算。
            mod = sys.modules.get("tempfile")
            if mod is not None:
                mod.tempdir = None
            _applied = True
            _active_root = base
            return base
    raise RuntimeError(
        "无可用缓存根目录（候选均不可创建）："
        + "; ".join(tried)
        + "。请用 SIMPLCORE_CACHE_ROOT 指定一个可写目录。"
    )


def cache_root() -> str:
    """当前生效（或将被 :func:`apply` 选中）的缓存根目录."""
    if _active_root:
        return _active_root
    return os.environ.get("SIMPLCORE_CACHE_ROOT") or DEFAULT_CACHE_ROOT
