# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity —— SGC 栈上的「模型粘合 / MoE 效果」模块。

把多个**训练数据不同**的小参数专家（``Expert``）经 token 级 top-k 路由门
（``ExpertRouter``）粘合到冻结的 SGC 基座（``UnityModel``）之上，得到 MoE 级别的
动态容量，而显存只随可学权重（路由 + 专家）增长。支持 ``spawn_expert`` 后台自进化。

参考设计文档：``docs/SIMPLUNITY_SGC.md``。
"""
from __future__ import annotations

from .config import (DEFAULT_UNITY, UNITY_PRESETS, UnityConfig,
                     ResolvedUnityConfig, for_preset)
from .expert import Expert, ExpertMLP
from .router import ExpertRouter
from .model import UnityModel
from .evolution import UnitySelfEvolution
from .bundle import ModuleBundle, base_fingerprint
from .experience import ExperienceLoader

__all__ = [
    "UnityConfig",
    "ResolvedUnityConfig",
    "UNITY_PRESETS",
    "DEFAULT_UNITY",
    "for_preset",
    "Expert",
    "ExpertMLP",
    "ExpertRouter",
    "UnityModel",
    "UnitySelfEvolution",
    "ModuleBundle",
    "base_fingerprint",
    "ExperienceLoader",
]
