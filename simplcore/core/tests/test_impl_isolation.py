# -*- coding: utf-8 -*-
"""非 Python 平行实现与记忆分层的**结构隔离**测试。

这里不比较数值（那需要 JVM/CUDA 实跑，属于实测），只固化三条结构性事实：

1. Java 平行实现是 NON-NORMATIVE：不进包、不被产品线 import、不参与 CI；
2. Python 侧才是唯一规范来源，其源码纯洁性由 AST 扫描守护；
3. ``memory → dream → smm`` 是单向分层：SMM 只能**编排**已有实现，
   不得再定义一套记忆库/巩固循环（防止同一概念两套代码漂移）。
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
JAVA = ROOT / "java-high-performance"
CORE = ROOT / "simplcore" / "core"


# ---------------------------------------------------------------------------
# 1. Java 平行实现的隔离
# ---------------------------------------------------------------------------
def test_java_tree_is_marked_non_normative() -> None:
    text = (JAVA / "README.md").read_text(encoding="utf-8")
    assert "NON-NORMATIVE" in text
    assert "不参与 CI" in text
    assert "唯一规范来源" in text
    assert "不得作为 SimplCore 架构或性能证据" in text


def test_java_tree_is_not_packaged() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = data["tool"]["setuptools"]["packages"]
    assert not any("java" in p for p in packages)


def test_product_line_never_imports_the_java_tree() -> None:
    offenders = []
    for path in (ROOT / "simplcore").rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "java-high-performance" in text or "com.qingxiaotuan" in text:
            offenders.append(str(path))
    assert offenders == [], f"产品线引用了非规范 Java 实现: {offenders}"


def test_java_attention_remains_outside_python_product_line() -> None:
    """Java 侧确实还有注意力类——这正是它必须被隔离的原因。"""
    assert (JAVA / "src/main/java/com/qingxiaotuan/simpcore/components/"
            "MultiHeadAttention.java").exists()
    from simplcore.core.purity import scan_source_attention_free
    assert scan_source_attention_free(ROOT / "simplcore") == []


# ---------------------------------------------------------------------------
# 2. 记忆分层：memory → dream → smm 单向
# ---------------------------------------------------------------------------
def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod.startswith("simplcore.core."):
                found.add(mod.rsplit(".", 1)[-1])
            elif mod.startswith("."):
                found.add(mod.lstrip(".").split(".")[0])
            elif "." not in mod:
                # Python 3.11 的相对导入在 AST 里已去掉前导点（``.dream`` → ``dream``）。
                found.add(mod)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("simplcore.core."):
                    found.add(alias.name.rsplit(".", 1)[-1])
    return found


def _class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def test_memory_does_not_depend_on_higher_layers() -> None:
    imports = _module_imports(CORE / "memory.py")
    assert not (imports & {"dream", "smm"}), f"memory 反向依赖: {imports}"


def test_dream_does_not_depend_on_smm() -> None:
    imports = _module_imports(CORE / "dream.py")
    assert "smm" not in imports, f"dream 反向依赖 smm: {imports}"


def test_smm_reuses_instead_of_reimplementing() -> None:
    """SMM 只能编排 memory/dream，不得复制它们的核心类。"""
    imports = _module_imports(CORE / "smm.py")
    assert {"memory", "dream"} <= imports, f"SMM 未复用底层实现: {imports}"
    own = _class_names(CORE / "smm.py")
    forbidden = {"CompressedMemoryBank", "TextKeyEncoder", "ExperienceBuffer",
                 "DreamingLoop", "NoRegressionGate"}
    assert own & forbidden == set(), f"SMM 重复定义了底层类: {own & forbidden}"


def test_memory_and_dream_own_the_primitives() -> None:
    assert {"CompressedMemoryBank", "TextKeyEncoder"} <= _class_names(CORE / "memory.py")
    assert {"ExperienceBuffer", "DreamingLoop"} <= _class_names(CORE / "dream.py")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))