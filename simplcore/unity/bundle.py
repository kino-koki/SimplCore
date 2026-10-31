# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 模块包：专家/适配器的独立导出、跨模型拼接、版本与来源审计。

为什么要单独的 bundle
----------------------
`UnityModel.save_learned` 存的是"整套路由 + 全部专家"，适合整机恢复，不适合：

* 把一个域专家从 2B 主干搬到另一个同构主干；
* 后台训练完成后只交换某一个模块；
* 记录"这个模块是在哪个基座、多少 token、什么 val loss 上训出来的"。

``ModuleBundle`` 是模块级的最小可搬运单元：只含一个专家的 state_dict + 构造元数据 +
来源审计。加载时校验 ``dim/kind`` 与基座指纹；预算由 ``UnityModel.spawn_expert``
守卫，不会因为拼接而绕过 3B 上限。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

__all__ = ["ModuleBundle", "base_fingerprint"]


def base_fingerprint(cfg: Any) -> str:
    """基座配置指纹：用于判断模块是否来自同构主干。

    只取结构字段（dim/layers/num_bases/local_kernel/mixer），不取词汇表与训练超参，
    因为专家只依赖 ``dim`` 与残差流语义；不同词表的主干仍可拼接。
    """
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    keys = ("dim", "num_layers", "num_bases", "local_kernel", "mixer", "norm")
    payload = {k: cfg.get(k) for k in keys if isinstance(cfg, dict) and k in cfg}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class ModuleBundle:
    """一个可独立搬运的 Unity 模块（当前支持 ``unity_expert``）。"""

    name: str
    kind: str = "unity_expert"
    dim: int = 0
    hidden: int = 0
    layers: int = 1
    dropout: float = 0.0
    state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault(
            "created_utc",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.metadata["version"] = int(self.version)
        self.metadata.setdefault("kind", self.kind)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "dim": self.dim,
            "hidden": self.hidden, "layers": self.layers,
            "dropout": self.dropout, "state_dict": self.state_dict,
            "metadata": self.metadata, "version": self.version,
        }

    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.as_dict(), p)

    @classmethod
    def load(cls, path: Any) -> "ModuleBundle":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "state_dict" not in state:
            raise ValueError(f"{path} is not a ModuleBundle")
        return cls(
            name=str(state.get("name", Path(path).stem)),
            kind=str(state.get("kind", "unity_expert")),
            dim=int(state.get("dim", 0)),
            hidden=int(state.get("hidden", 0)),
            layers=int(state.get("layers", 1)),
            dropout=float(state.get("dropout", 0.0)),
            state_dict=state["state_dict"],
            metadata=dict(state.get("metadata", {})),
            version=int(state.get("version", 1)),
        )

    def compatible_with(self, unity) -> bool:
        if self.kind != "unity_expert":
            return False
        if int(self.dim) != int(unity.dim):
            return False
        want = self.metadata.get("base_fingerprint")
        have = getattr(unity, "base_fingerprint", None)
        if want and have and want != have:
            return False
        return True

    def audit(self) -> Dict[str, Any]:
        """可审计摘要（不含权重）。"""
        return {
            "name": self.name, "kind": self.kind, "dim": self.dim,
            "hidden": self.hidden, "layers": self.layers,
            "dropout": self.dropout,
            "num_params": int(sum(v.numel() for v in self.state_dict.values())),
            "metadata": dict(self.metadata),
        }