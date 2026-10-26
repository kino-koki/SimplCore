# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""堆叠来源与清单（增强版：修复命门 8）。

命门 8 修复:
  - `SourceCheckpoint.data_range` 此前永远为 None，且 role 无合法性校验、
    无 seed/device/数据指纹记录，无法追溯"这个 7B 用什么堆的"。
  - 本增强版加入 `seed`、`device`、`data_fingerprint`、`params_fingerprint`，
    并在构建清单时校验 role 合法性，保证堆叠可复现、可审计。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# 合法的来源角色（150M 变体在堆叠中的定位）
VALID_ROLES = {"prelude", "recurrent", "coda", "expert", "generic"}


@dataclass(frozen=True)
class SourceCheckpoint:
    path: str
    role: str
    shard_id: int
    loss: Optional[float] = None
    step: Optional[int] = None
    data_range: Optional[str] = None
    # ---- 命门 8 增强字段 ----
    seed: Optional[int] = None          # 训练随机种子（多样性指纹）
    device: Optional[str] = None        # 训练设备（directml/cuda）
    data_fingerprint: Optional[str] = None  # 数据子集指纹（领域/分片哈希）
    params_fingerprint: Optional[str] = None  # 参数量指纹（校验同构）

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"非法 role {self.role!r}，合法值: {sorted(VALID_ROLES)}"
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StackManifest:
    target_name: str
    sources: List[SourceCheckpoint]
    stack_params: Optional[Dict[str, object]] = None  # 命门 8：记录堆叠超参

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_name": self.target_name,
            "sources": [source.to_dict() for source in self.sources],
            "stack_params": self.stack_params or {},
        }

    def diversity_report(self) -> Dict[str, object]:
        """命门 8：检查 150M 变体的多样性（seed / 数据指纹）。"""
        seeds = {s.seed for s in self.sources if s.seed is not None}
        data_fps = {s.data_fingerprint for s in self.sources if s.data_fingerprint}
        return {
            "n_sources": len(self.sources),
            "distinct_seeds": len(seeds),
            "distinct_data_fingerprints": len(data_fps),
            "warning": (
                None
                if len(seeds) >= 2 and len(data_fps) >= 2
                else "多样性不足：若所有 150M 用相同 seed + 相同数据训练，堆叠 ≈ 复制同一模型"
            ),
        }
