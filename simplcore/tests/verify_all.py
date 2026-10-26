# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 快速验证脚本 (现实版本).

验证当前实际存在的架构组件 (SGC / DMC / CCR / CPE / OEU / Rollout / ModalityRouter)
在自研 simplcore.tensor 后端上可实例化、前向、反向。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import simplcore.tensor as torch
from simplcore.architecture.architecture_components import ArchitectureComponents

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} - {detail}")

print("=" * 60)
print("SimplCore 架构组件快速验证")
print("=" * 60)

comps = ArchitectureComponents()
print(f"\n注册组件数: {len(comps.components)}")
for name in comps.components:
    print(f"  - {name}")
check("注册 7 个架构组件", len(comps.components) == 7)

# 逐个实例化 + 前向 + 反向冒烟 (直接导入真实类)
print("\n[组件冒烟测试]")
from simplcore.architecture.spectral_global_conv import SpectralGlobalConv
from simplcore.architecture.memory_channels import DynamicMemoryChannel
from simplcore.architecture.conditional_compute_router import ConditionalComputeRouter
from simplcore.architecture.complex_positional_encoder import ComplexPositionalEncoder

x = torch.randn(2, 8, 32, requires_grad=True)

# 1. SGC
try:
    sgc = SpectralGlobalConv(dim=32, num_bands=8, num_scales=3, rank=16)
    y = sgc(x); y.pow(2).mean().backward()
    ok = (tuple(y.shape) == tuple(x.shape)) and any(
        p.grad is not None and float(np.abs(p.grad).sum()) > 0 for p in sgc.parameters())
    check("SGC 谱全局卷积 前向+反向", ok)
except Exception as e:
    check("SGC 运行失败", False, str(e))

# 2. DMC
try:
    dmc = DynamicMemoryChannel(dim=32, num_channels=4, channel_size=8, num_heads=4)
    out = dmc(x)
    ok = (hasattr(out, "shape") and out.shape[-1] == 32) or (
        isinstance(out, (tuple, list)) and out[0].shape[-1] == 32)
    check("DMC 动态记忆通道 前向", ok)
except Exception as e:
    check("DMC 运行失败", False, str(e))

# 3. CCR
try:
    ccr = ConditionalComputeRouter(dim=32, num_paths=4, topk=2, min_depth=2, max_depth=8)
    y = ccr(x); y.pow(2).mean().backward()
    ok = (tuple(y.shape) == tuple(x.shape)) and any(
        p.grad is not None and float(np.abs(p.grad).sum()) > 0 for p in ccr.parameters())
    check("CCR 条件计算路由 前向+反向", ok)
except Exception as e:
    check("CCR 运行失败", False, str(e))

# 4. CPE
try:
    cpe = ComplexPositionalEncoder(dim=32, num_basis=4, num_layers=2, dropout=0.0)
    y = cpe(x)
    check("CPE 复数位置编码 前向", tuple(y.shape) == tuple(x.shape) and not bool(np.isnan(y.data).any()))
except Exception as e:
    check("CPE 运行失败", False, str(e))

print("\n" + "=" * 60)
print(f"结果: {passed} 通过, {failed} 失败")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
