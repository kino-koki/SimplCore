# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""RBAC 基于角色的访问控制测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simplcore.safety.rbac import RBACManager, RBACError, PERMISSIONS


@pytest.fixture
def rbac():
    mgr = RBACManager()
    mgr.assign_role("alice", "admin")
    mgr.assign_role("bob", "developer")
    mgr.assign_role("carol", "operator")
    mgr.assign_role("dave", "auditor")
    mgr.assign_role("eve", "readonly")
    return mgr


def test_admin_all_permissions(rbac):
    for p in PERMISSIONS:
        assert rbac.check_permission("alice", p)


def test_developer_permissions(rbac):
    assert rbac.check_permission("bob", "model.train")
    assert rbac.check_permission("bob", "model.export")
    assert not rbac.check_permission("bob", "model.deploy")
    assert not rbac.check_permission("bob", "model.delete")
    assert not rbac.check_permission("bob", "rbac.manage")


def test_operator_permissions(rbac):
    assert rbac.check_permission("carol", "model.load")
    assert rbac.check_permission("carol", "model.deploy")
    assert not rbac.check_permission("carol", "model.train")
    assert not rbac.check_permission("carol", "model.delete")


def test_auditor_only_view(rbac):
    assert rbac.check_permission("dave", "audit.view")
    assert rbac.check_permission("dave", "audit.export")
    assert not rbac.check_permission("dave", "model.train")
    assert not rbac.check_permission("dave", "model.deploy")


def test_readonly_inheritance(rbac):
    # developer 继承 readonly → 拥有 model.load / data.access / config.read
    assert rbac.check_permission("bob", "model.load")
    assert rbac.check_permission("bob", "data.access")
    assert rbac.check_permission("bob", "config.read")


def test_unknown_user_denied(rbac):
    assert not rbac.check_permission("mallory", "model.load")


def test_require_raises(rbac):
    with pytest.raises(RBACError, match="无权限"):
        rbac.require("eve", "model.deploy")
    rbac.require("alice", "model.deploy")  # admin 不抛


def test_unregistered_permission_raises(rbac):
    with pytest.raises(RBACError, match="未登记权限"):
        rbac.check_permission("alice", "model.hack")


def test_revoke_role(rbac):
    rbac.assign_role("bob", "operator")
    assert rbac.check_permission("bob", "model.deploy")
    rbac.revoke_role("bob", "operator")
    assert not rbac.check_permission("bob", "model.deploy")


def test_custom_role(rbac):
    rbac.add_role("ml_researcher", ["model.load", "model.train", "data.access"])
    rbac.assign_role("frank", "ml_researcher")
    assert rbac.check_permission("frank", "model.train")
    assert not rbac.check_permission("frank", "model.deploy")


def test_deny_by_default_for_new_permission():
    mgr = RBACManager()
    mgr.assign_role("alice", "admin")
    # admin 拥有 PERMISSIONS 全集；新增权限后默认拒绝直到登记
    assert "model.future" not in PERMISSIONS
    with pytest.raises(RBACError):
        mgr.check_permission("alice", "model.future")


def test_to_dict_roundtrip(rbac):
    d = rbac.to_dict()
    assert "alice" in d["user_roles"]
    assert "admin" in d["role_permissions"]
