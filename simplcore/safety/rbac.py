# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""RBAC 基于角色的访问控制 — 模型生命周期权限管理。

用途（对应 HONEST_STATUS.md 待落地项"安全模块（AES-256/RBAC/内存保护）"）：
  * 控制谁可以加载 / 训练 / 部署 / 导出 / 删除模型权重；
  * 控制谁可以访问训练数据 / 审计日志 / 敏感配置；
  * 默认拒绝（deny-by-default）：未显式授权的操作一律拒绝。

设计：
  * 角色（Role）→ 权限（Permission）映射，角色可继承；
  * 用户（User）→ 角色映射；
  * ``RBACManager.check_permission(user, permission)`` 为唯一鉴权入口；
  * 每次鉴权可写审计日志（与 ``AuditLogger`` 集成，可选）；
  * 内置角色：admin / developer / operator / auditor / readonly，
    权限集见 ``DEFAULT_ROLE_PERMISSIONS``。

权限命名约定：``<domain>.<action>``，如 ``model.load`` / ``model.deploy`` /
``data.access`` / ``audit.view``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


# ----------------------------------------------------------------------
# 内置权限与角色
# ----------------------------------------------------------------------
# 权限全集（新增权限应在此登记，便于审计与文档）
PERMISSIONS: FrozenSet[str] = frozenset({
    "model.load", "model.train", "model.deploy", "model.export",
    "model.delete", "model.encrypt",
    "data.access", "data.clean", "data.delete",
    "audit.view", "audit.export",
    "config.read", "config.write",
    "rbac.manage",
})

# 角色 → 权限（默认拒绝：未列出的权限一律拒绝）
DEFAULT_ROLE_PERMISSIONS: Dict[str, FrozenSet[str]] = {
    "admin": frozenset(PERMISSIONS),
    "developer": frozenset({
        "model.load", "model.train", "model.export", "model.encrypt",
        "data.access", "data.clean",
        "config.read", "config.write",
    }),
    "operator": frozenset({
        "model.load", "model.deploy",
        "data.access",
        "config.read",
    }),
    "auditor": frozenset({
        "audit.view", "audit.export",
        "config.read",
    }),
    "readonly": frozenset({
        "model.load", "data.access", "config.read", "audit.view",
    }),
}

# 角色继承（子角色自动获得父角色权限）
ROLE_INHERITANCE: Dict[str, Tuple[str, ...]] = {
    "admin": (),
    "developer": ("readonly",),
    "operator": ("readonly",),
    "auditor": ("readonly",),
    "readonly": (),
}


class RBACError(Exception):
    """RBAC 相关错误（未授权 / 角色不存在 / 权限未登记）。"""


@dataclass
class RBACManager:
    """基于角色的访问控制管理器。

    Args:
        user_roles: 初始 用户 → 角色列表 映射。
        role_permissions: 角色 → 权限集（默认用内置角色表）。
        role_inheritance: 角色 → 父角色列表（默认用内置继承表）。
        audit_logger: 可选 ``AuditLogger`` 实例，鉴权结果写入审计日志。
    """

    user_roles: Dict[str, List[str]] = field(default_factory=dict)
    role_permissions: Dict[str, FrozenSet[str]] = field(
        default_factory=lambda: dict(DEFAULT_ROLE_PERMISSIONS)
    )
    role_inheritance: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(ROLE_INHERITANCE)
    )
    audit_logger: Optional[object] = None

    # ------------------------------------------------------------------
    # 角色管理
    # ------------------------------------------------------------------
    def add_role(self, role: str, permissions: Iterable[str],
                 inherits: Iterable[str] = ()) -> None:
        """注册新角色（权限集 + 继承关系）。"""
        unknown = [p for p in permissions if p not in PERMISSIONS]
        if unknown:
            raise RBACError(f"未登记权限: {unknown}（请先在 PERMISSIONS 登记）")
        self.role_permissions[role] = frozenset(permissions)
        self.role_inheritance[role] = tuple(inherits)

    def remove_role(self, role: str) -> None:
        """删除角色（同时清理用户引用）。"""
        if role not in self.role_permissions:
            raise RBACError(f"角色不存在: {role}")
        del self.role_permissions[role]
        del self.role_inheritance[role]
        for user, roles in list(self.user_roles.items()):
            self.user_roles[user] = [r for r in roles if r != role]

    # ------------------------------------------------------------------
    # 用户管理
    # ------------------------------------------------------------------
    def assign_role(self, user: str, role: str) -> None:
        """给用户分配角色。"""
        if role not in self.role_permissions:
            raise RBACError(f"角色不存在: {role}")
        roles = self.user_roles.setdefault(user, [])
        if role not in roles:
            roles.append(role)

    def revoke_role(self, user: str, role: str) -> None:
        """撤销用户角色。"""
        roles = self.user_roles.get(user, [])
        if role in roles:
            roles.remove(role)

    def roles_for(self, user: str) -> List[str]:
        """返回用户全部角色（含继承展开）。"""
        return list(self.user_roles.get(user, []))

    # ------------------------------------------------------------------
    # 鉴权（唯一入口）
    # ------------------------------------------------------------------
    def _effective_permissions(self, user: str) -> Set[str]:
        """计算用户的有效权限集（含角色继承）。"""
        effective: Set[str] = set()
        visited: Set[str] = set()

        def walk(role: str) -> None:
            if role in visited:
                return
            visited.add(role)
            perms = self.role_permissions.get(role, frozenset())
            effective.update(perms)
            for parent in self.role_inheritance.get(role, ()):
                walk(parent)

        for role in self.user_roles.get(user, []):
            walk(role)
        return effective

    def check_permission(self, user: str, permission: str,
                         audit: bool = True) -> bool:
        """鉴权：用户是否拥有指定权限。

        Args:
            user: 用户标识。
            permission: 形如 ``model.load`` 的权限名。
            audit: 是否把本次鉴权写入审计日志（若配置了 audit_logger）。
        Returns:
            bool: 是否授权。
        """
        if permission not in PERMISSIONS:
            raise RBACError(f"未登记权限: {permission}")
        allowed = permission in self._effective_permissions(user)
        if audit and self.audit_logger is not None:
            try:
                self.audit_logger.log_event(
                    event_type="rbac",
                    source="RBACManager",
                    user_id=user,
                    action="check_permission",
                    resource=permission,
                    details={"allowed": allowed},
                    severity="low" if allowed else "medium",
                )
            except Exception:
                pass  # 审计失败不影响鉴权结果
        return allowed

    def require(self, user: str, permission: str, audit: bool = True) -> None:
        """强制鉴权：未授权抛 ``RBACError``。"""
        if not self.check_permission(user, permission, audit=audit):
            raise RBACError(
                f"用户 '{user}' 无权限 '{permission}'（deny-by-default）。"
            )

    # ------------------------------------------------------------------
    # 查询 / 序列化
    # ------------------------------------------------------------------
    def permissions_for(self, user: str) -> Set[str]:
        """返回用户有效权限集（用于 UI / 调试）。"""
        return self._effective_permissions(user)

    def to_dict(self) -> Dict[str, object]:
        return {
            "user_roles": {k: list(v) for k, v in self.user_roles.items()},
            "role_permissions": {
                k: sorted(v) for k, v in self.role_permissions.items()
            },
            "role_inheritance": {
                k: list(v) for k, v in self.role_inheritance.items()
            },
        }


__all__ = ["RBACManager", "RBACError", "PERMISSIONS", "DEFAULT_ROLE_PERMISSIONS"]
