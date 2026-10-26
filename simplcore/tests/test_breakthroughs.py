# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 架构组件验证测试 (现实版本)
=====================================
验证当前实际存在的架构组件在自研 simplcore.tensor 后端上的正确性。

命名与描述遵循「技术中性、可被代码验证」原则, 去除伪科学修辞
(如「谐振场 / 量子 / 拓扑感知 / 自发塑性」等)。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import simplcore.tensor as torch
import unittest


class TestSpectralGlobalConv(unittest.TestCase):
    """组件一：谱全局卷积 (SGC) 测试"""

    def setUp(self):
        from simplcore.architecture.spectral_global_conv import SpectralGlobalConv
        self.dim = 32
        self.sgc = SpectralGlobalConv(dim=self.dim, num_bands=8, num_scales=3, rank=16)
        self.x = torch.randn(2, 16, self.dim, requires_grad=True)

    def test_output_shape(self):
        y = self.sgc(self.x)
        self.assertEqual(tuple(y.shape), tuple(self.x.shape))

    def test_not_nan(self):
        y = self.sgc(self.x)
        self.assertFalse(bool(np.isnan(y.data).any()))

    def test_grad_flow(self):
        y = self.sgc(self.x)
        y.pow(2).mean().backward()
        self.assertTrue(
            any(p.grad is not None and float(np.abs(p.grad).sum()) > 0
                for p in self.sgc.parameters())
        )

    def test_complexity_claim(self):
        info = self.sgc.complexity_analysis(seq_len=4096)
        self.assertEqual(info["sgc_big_o"], "O(n log n + F*r)")


class TestDynamicMemoryChannel(unittest.TestCase):
    """组件二：动态记忆通道 (DMC) 测试"""

    def setUp(self):
        from simplcore.architecture.memory_channels import DynamicMemoryChannel
        self.dim = 32
        self.dmc = DynamicMemoryChannel(dim=self.dim, num_channels=4, channel_size=8, num_heads=4)
        self.x = torch.randn(2, 8, self.dim)

    def test_output_shape(self):
        out = self.dmc(self.x)
        self.assertEqual(len(out), 2)
        y, mem = out
        # 输出最后一维为模型维度 D
        self.assertEqual(y.shape[-1], self.dim)
        self.assertEqual(mem.shape[-1], self.dim)

    def test_memory_update(self):
        _, mem1 = self.dmc(self.x)
        _, mem2 = self.dmc(self.x)
        self.assertEqual(mem1.shape[-1], self.dim)
        self.assertEqual(mem2.shape[-1], self.dim)


class TestConditionalComputeRouter(unittest.TestCase):
    """组件三：条件计算路由 (CCR) 测试"""

    def setUp(self):
        from simplcore.architecture.conditional_compute_router import ConditionalComputeRouter
        self.dim = 32
        self.ccr = ConditionalComputeRouter(dim=self.dim, num_paths=4, topk=2, min_depth=2, max_depth=8)
        self.x = torch.randn(2, 8, self.dim, requires_grad=True)

    def test_output_shape(self):
        y = self.ccr(self.x)
        self.assertEqual(tuple(y.shape), tuple(self.x.shape))

    def test_grad_flow(self):
        y = self.ccr(self.x)
        y.pow(2).mean().backward()
        self.assertTrue(
            any(p.grad is not None and float(np.abs(p.grad).sum()) > 0
                for p in self.ccr.parameters())
        )

    def test_expected_depth(self):
        ed = self.ccr.expected_depth(self.x.detach())
        self.assertEqual(tuple(ed.shape), (2, 8))


class TestComplexPositionalEncoder(unittest.TestCase):
    """组件四：复数位置编码 (CPE) 测试"""

    def setUp(self):
        from simplcore.architecture.complex_positional_encoder import ComplexPositionalEncoder
        self.dim = 32
        self.cpe = ComplexPositionalEncoder(dim=self.dim, num_basis=4, num_layers=2, dropout=0.0)
        self.x = torch.randn(2, 8, self.dim)

    def test_output_shape(self):
        y = self.cpe(self.x)
        self.assertEqual(tuple(y.shape), tuple(self.x.shape))

    def test_not_nan(self):
        y = self.cpe(self.x)
        self.assertFalse(bool(np.isnan(y.data).any()))


class TestRolloutEngine(unittest.TestCase):
    """组件六：潜空间推演 (Rollout) 测试"""

    def test_rollout_instantiates(self):
        from simplcore.architecture.simplcore_block import SimplCoreBlock
        from simplcore.architecture.rollout_engine import RolloutEngine
        block = SimplCoreBlock(dim=32, num_waves=2, num_channels=4, channel_size=4, num_paths=2)
        rollout = RolloutEngine(block, max_steps=3, adaptive=False)
        self.assertEqual(rollout.max_steps, 3)


class TestModalityRouter(unittest.TestCase):
    """组件七：多模态路由 (ModalityRouter) 测试"""

    def test_router_runs(self):
        from simplcore.architecture.modality_router import ModalityRouter
        router = ModalityRouter(hidden_size=32)
        x = torch.randn(2, 8, 32)
        y = router(x)
        self.assertIsInstance(y, dict)
        self.assertIn("latent", y)
        self.assertEqual(y["latent"].shape[-1], 32)


class TestArchitectureComponentsRegistry(unittest.TestCase):
    """组件登记一致性测试"""

    def test_registry(self):
        from simplcore.architecture.architecture_components import ArchitectureComponents
        comps = ArchitectureComponents()
        self.assertEqual(len(comps.components), 7)
        # SGC 复杂度声明诚实 (含 n)
        sgc = comps.get_component("谱全局卷积")
        self.assertIn("n", sgc["complexity"])
        # DMC 已修正为「相对 n 不增长」而非 O(1)
        dmc = comps.get_component("动态记忆通道")
        self.assertNotIn("O(1)", dmc["complexity"])


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore 架构组件验证测试套件 (现实版本)")
    print("=" * 60)
    unittest.main(verbosity=2)
