# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""参考后端（fp32 CPU）的 torch 集成验收：真实 load_backend + 9 项一致性检查。

本机无 torch / 无编译器时 skip；在目标机（MI250 或任意装 torch 的机器）上
执行 ``pytest simplcore/core/tests/test_ref_backend_conformance.py``，
全过即 P0-backend-conformance 闸门在**真实编译库**口径下达成。
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")  # noqa: F841 —— 无 torch 环境 skip

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _ensure_built() -> str:
    """确保参考后端已构建，返回库路径。"""
    name = "sc_ref_backend.dll" if os.name == "nt" else "libsc_ref_backend.so"
    path = os.path.join(ROOT, "simplcore", "csrc", "build", name)
    if not os.path.exists(path):
        script = os.path.join(ROOT, "scripts", "build_ref_backend.sh")
        subprocess.run(["bash", script], check=True, cwd=ROOT)
    assert os.path.exists(path), f"参考后端构建失败：{path}"
    return path


def test_ref_backend_full_conformance() -> None:
    from simplcore.core.backend_bridge import load_backend
    from simplcore.core.conformance_runner import run_conformance

    lib_path = _ensure_built()
    adapter = load_backend(lib_path)
    # CI profile 口径（默认 L=4096）；--full 需生产 FFT 后端，参考实现不适用
    rep = run_conformance(adapter)
    print(rep.format())
    failed = [r.name for r in rep.results if not r.passed]
    assert rep.all_passed, f"一致性检查失败项: {failed}"


def test_ref_backend_bf16_gemm_bridge() -> None:
    """桥接层 fp32 单一契约：bf16 输入不得误读（R8 回归钉子）。"""
    from simplcore.core.backend_bridge import load_backend

    lib_path = _ensure_built()
    ad = load_backend(lib_path)
    g = torch.Generator().manual_seed(7)
    a = torch.randn(4, 8, generator=g).to(torch.bfloat16)
    b = torch.randn(8, 3, generator=g).to(torch.bfloat16)
    out = ad.gemm(a, b)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, ref), "bf16 gemm 必须与参考数学逐位一致"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
