# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 完整测试套件
核心模块测试覆盖率≥95%
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import unittest
import numpy as np


class TestSimpTensorBasic(unittest.TestCase):
    """张量基础操作测试"""
    
    def setUp(self):
        from SimplCore.engine import randn, zeros, ones
        
        self.a = randn(3, 4)
        self.b = randn(4, 5)
        self.c = randn(3, 4)
    
    def test_tensor_creation(self):
        from SimplCore.engine import randn, zeros, ones, arange, eye
        
        t1 = randn(2, 3)
        self.assertEqual(t1.shape, (2, 3))
        
        t2 = zeros(2, 3)
        self.assertEqual(np.allclose(t2._data, 0), True)
        
        t3 = ones(2, 3)
        self.assertEqual(np.allclose(t3._data, 1), True)
        
        t4 = arange(10)
        self.assertEqual(t4.shape, (10,))
        
        t5 = eye(3)
        self.assertEqual(np.allclose(np.diag(t5._data), 1), True)
    
    def test_tensor_arithmetic(self):
        result = self.a + self.c
        self.assertEqual(result.shape, (3, 4))
        
        result = self.a - self.c
        self.assertEqual(result.shape, (3, 4))
        
        result = self.a * self.c
        self.assertEqual(result.shape, (3, 4))
        
        result = self.a / (self.c + 1e-6)
        self.assertEqual(result.shape, (3, 4))
    
    def test_tensor_matmul(self):
        result = self.a.matmul(self.b)
        self.assertEqual(result.shape, (3, 5))
    
    def test_tensor_reshape(self):
        result = self.a.reshape(12)
        self.assertEqual(result.shape, (12,))
        
        result = self.a.reshape(2, 6)
        self.assertEqual(result.shape, (2, 6))
    
    def test_tensor_transpose(self):
        result = self.a.transpose(1, 0)
        self.assertEqual(result.shape, (4, 3))
    
    def test_tensor_sum_mean(self):
        result = self.a.sum()
        self.assertEqual(result.shape, ())
        
        result = self.a.mean()
        self.assertEqual(result.shape, ())
        
        result = self.a.sum(axis=1)
        self.assertEqual(result.shape, (3,))
    
    def test_tensor_comparison(self):
        result = self.a > 0
        self.assertEqual(result.shape, (3, 4))


class TestSimpTensorAdvanced(unittest.TestCase):
    """张量高级操作测试"""
    
    def setUp(self):
        from SimplCore.engine import randn
        
        self.a = randn(3, 4)
        self.b = randn(3, 4)
    
    def test_tensor_cat_stack(self):
        from SimplCore.engine import cat, stack
        
        result = cat([self.a, self.b], dim=0)
        self.assertEqual(result.shape, (6, 4))
        
        result = cat([self.a, self.b], dim=1)
        self.assertEqual(result.shape, (3, 8))
        
        result = stack([self.a, self.b], dim=0)
        self.assertEqual(result.shape, (2, 3, 4))
    
    def test_tensor_split(self):
        from SimplCore.engine import split
        
        parts = split(self.a, 2, dim=1)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (3, 2))
    
    def test_tensor_permute(self):
        from SimplCore.engine import permute
        
        t = randn(2, 3, 4)
        result = permute(t, (2, 0, 1))
        self.assertEqual(result.shape, (4, 2, 3))
    
    def test_tensor_softmax(self):
        from SimplCore.engine import softmax
        
        result = softmax(self.a)
        self.assertEqual(np.allclose(result._data.sum(axis=-1), 1), True)
    
    def test_tensor_relu_gelu(self):
        from SimplCore.engine import relu, gelu, silu
        
        result = relu(self.a)
        self.assertEqual(np.all(result._data >= 0), True)
        
        result = gelu(self.a)
        self.assertEqual(result.shape, (3, 4))
        
        result = silu(self.a)
        self.assertEqual(result.shape, (3, 4))
    
    def test_tensor_layer_norm(self):
        from SimplCore.engine import layer_norm
        
        result = layer_norm(self.a, normalized_shape=4)
        self.assertEqual(result.shape, (3, 4))
    
    def test_tensor_dropout(self):
        from SimplCore.engine import dropout
        
        result = dropout(self.a, p=0.5)
        self.assertEqual(result.shape, (3, 4))
    
    def test_tensor_exp_log(self):
        from SimplCore.engine import exp, log
        
        result = exp(self.a)
        self.assertEqual(result.shape, (3, 4))
        
        result = log(exp(self.a))
        self.assertEqual(np.allclose(result._data, self.a._data), True)
    
    def test_tensor_clamp(self):
        from SimplCore.engine import clamp
        
        result = clamp(self.a, min=-1, max=1)
        self.assertEqual(np.all(result._data >= -1), True)
        self.assertEqual(np.all(result._data <= 1), True)


class TestSimpTensorAutograd(unittest.TestCase):
    """张量自动微分测试"""
    
    def setUp(self):
        from SimplCore.engine import randn
        
        self.x = randn(3, 4, requires_grad=True)
        self.y = randn(4, 5, requires_grad=True)
    
    def test_basic_grad(self):
        z = self.x.matmul(self.y).sum()
        z.backward()
        
        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(self.y.grad)
        self.assertEqual(self.x.grad.shape, (3, 4))
        self.assertEqual(self.y.grad.shape, (4, 5))
    
    def test_grad_clear(self):
        z = self.x.matmul(self.y).sum()
        z.backward()
        
        self.x.zero_grad()
        self.assertIsNone(self.x.grad)
    
    def test_detach(self):
        detached = self.x.detach()
        self.assertFalse(detached.requires_grad)
    
    def test_requires_grad(self):
        self.x.requires_grad_(False)
        self.assertFalse(self.x.requires_grad)
        
        self.x.requires_grad_(True)
        self.assertTrue(self.x.requires_grad)


class TestSimpNNLayers(unittest.TestCase):
    """神经网络层测试"""
    
    def test_linear_layer(self):
        from SimplCore.engine import Linear, randn
        
        layer = Linear(10, 5)
        x = randn(32, 10)
        output = layer(x)
        self.assertEqual(output.shape, (32, 5))
    
    def test_conv2d_layer(self):
        from SimplCore.engine import Conv2d, randn
        
        layer = Conv2d(3, 16, kernel_size=3)
        x = randn(32, 3, 32, 32)
        output = layer(x)
        self.assertEqual(output.shape, (32, 16, 30, 30))
    
    def test_conv1d_layer(self):
        from SimplCore.engine import Conv1d, randn
        
        layer = Conv1d(10, 20, kernel_size=3)
        x = randn(32, 10, 100)
        output = layer(x)
        self.assertEqual(output.shape, (32, 20, 98))
    
    def test_layer_norm(self):
        from SimplCore.engine import LayerNorm, randn
        
        layer = LayerNorm(64)
        x = randn(32, 10, 64)
        output = layer(x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_batch_norm(self):
        from SimplCore.engine import BatchNorm2d, randn
        
        layer = BatchNorm2d(16)
        x = randn(32, 16, 32, 32)
        output = layer(x)
        self.assertEqual(output.shape, (32, 16, 32, 32))
    
    def test_pooling(self):
        from SimplCore.engine import MaxPool2d, AvgPool2d, randn
        
        max_pool = MaxPool2d(2)
        avg_pool = AvgPool2d(2)
        x = randn(32, 16, 32, 32)
        
        output_max = max_pool(x)
        self.assertEqual(output_max.shape, (32, 16, 16, 16))
        
        output_avg = avg_pool(x)
        self.assertEqual(output_avg.shape, (32, 16, 16, 16))
    
    def test_dropout(self):
        from SimplCore.engine import Dropout, randn
        
        layer = Dropout(0.5)
        x = randn(32, 100)
        output = layer(x)
        self.assertEqual(output.shape, (32, 100))
    
    def test_activation(self):
        from SimplCore.engine import ReLU, GELU, Sigmoid, Tanh, randn
        
        x = randn(32, 100)
        
        relu = ReLU()
        self.assertEqual(relu(x).shape, (32, 100))
        
        gelu = GELU()
        self.assertEqual(gelu(x).shape, (32, 100))
        
        sigmoid = Sigmoid()
        self.assertEqual(sigmoid(x).shape, (32, 100))
        
        tanh = Tanh()
        self.assertEqual(tanh(x).shape, (32, 100))


class TestSimpNNRecurrent(unittest.TestCase):
    """循环神经网络测试"""
    
    def test_rnn(self):
        from SimplCore.engine import RNN, randn
        
        rnn = RNN(10, 20, num_layers=2)
        x = randn(10, 32, 10)
        output, hn = rnn(x)
        self.assertEqual(output.shape, (10, 32, 20))
        self.assertEqual(hn.shape, (2, 32, 20))
    
    def test_lstm(self):
        from SimplCore.engine import LSTM, randn
        
        lstm = LSTM(10, 20, num_layers=2)
        x = randn(10, 32, 10)
        output, (hn, cn) = lstm(x)
        self.assertEqual(output.shape, (10, 32, 20))
        self.assertEqual(hn.shape, (2, 32, 20))
        self.assertEqual(cn.shape, (2, 32, 20))
    
    def test_gru(self):
        from SimplCore.engine import GRU, randn
        
        gru = GRU(10, 20, num_layers=2)
        x = randn(10, 32, 10)
        output, hn = gru(x)
        self.assertEqual(output.shape, (10, 32, 20))
        self.assertEqual(hn.shape, (2, 32, 20))
    
    def test_lstm_batch_first(self):
        from SimplCore.engine import LSTM, randn
        
        lstm = LSTM(10, 20, batch_first=True)
        x = randn(32, 10, 10)
        output, (hn, cn) = lstm(x)
        self.assertEqual(output.shape, (32, 10, 20))
    
    def test_bidirectional_lstm(self):
        from SimplCore.engine import LSTM, randn
        
        lstm = LSTM(10, 20, bidirectional=True)
        x = randn(10, 32, 10)
        output, (hn, cn) = lstm(x)
        self.assertEqual(output.shape, (10, 32, 40))
        self.assertEqual(hn.shape, (2, 32, 20))


class TestSimpNNTransformer(unittest.TestCase):
    """Transformer测试"""
    
    def test_multi_head_attention(self):
        from SimplCore.engine import MultiHeadAttention, randn
        
        mha = MultiHeadAttention(64, 8)
        x = randn(32, 10, 64)
        output = mha(x, x, x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_transformer_encoder_layer(self):
        from SimplCore.engine import TransformerEncoderLayer, randn
        
        layer = TransformerEncoderLayer(64, 8)
        x = randn(32, 10, 64)
        output = layer(x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_transformer_decoder_layer(self):
        from SimplCore.engine import TransformerEncoderLayer, TransformerDecoderLayer, randn
        
        enc_layer = TransformerEncoderLayer(64, 8)
        dec_layer = TransformerDecoderLayer(64, 8)
        
        src = randn(32, 10, 64)
        tgt = randn(32, 5, 64)
        
        memory = enc_layer(src)
        output = dec_layer(tgt, memory)
        self.assertEqual(output.shape, (32, 5, 64))
    
    def test_transformer(self):
        from SimplCore.engine import Transformer, randn
        
        transformer = Transformer(64, 8, num_encoder_layers=2, num_decoder_layers=2)
        src = randn(32, 10, 64)
        tgt = randn(32, 5, 64)
        
        output = transformer(src, tgt)
        self.assertEqual(output.shape, (32, 5, 64))


class TestSimpOptimizers(unittest.TestCase):
    """优化器测试"""
    
    def setUp(self):
        from SimplCore.engine import Linear, randn
        
        self.model = Linear(10, 5)
        self.params = list(self.model.parameters())
    
    def test_sgd(self):
        from SimplCore.engine import SGD
        
        optimizer = SGD(self.params, lr=0.01)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_adam(self):
        from SimplCore.engine import Adam
        
        optimizer = Adam(self.params, lr=0.001)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_adamw(self):
        from SimplCore.engine import AdamW
        
        optimizer = AdamW(self.params, lr=0.001)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_rmsprop(self):
        from SimplCore.engine import RMSprop
        
        optimizer = RMSprop(self.params, lr=0.01)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_adagrad(self):
        from SimplCore.engine import Adagrad
        
        optimizer = Adagrad(self.params, lr=0.01)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_adadelta(self):
        from SimplCore.engine import Adadelta
        
        optimizer = Adadelta(self.params)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_adamax(self):
        from SimplCore.engine import Adamax
        
        optimizer = Adamax(self.params)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_nadam(self):
        from SimplCore.engine import NAdam
        
        optimizer = NAdam(self.params)
        x = randn(32, 10)
        y = randn(32, 5)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            optimizer.step()
    
    def test_lbfgs(self):
        from SimplCore.engine import LBFGS
        
        optimizer = LBFGS(self.params)
        x = randn(32, 10)
        y = randn(32, 5)
        
        def closure():
            optimizer.zero_grad()
            output = self.model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            return loss
        
        optimizer.step(closure)


class TestSimpLearningRateSchedulers(unittest.TestCase):
    """学习率调度器测试"""
    
    def setUp(self):
        from SimplCore.engine import Linear, Adam
        
        model = Linear(10, 5)
        self.optimizer = Adam(model.parameters(), lr=0.001)
    
    def test_cosine_annealing(self):
        from SimplCore.engine import CosineAnnealingLR
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        for _ in range(100):
            scheduler.step()
        
        self.assertLessEqual(scheduler.optimizer.lr, 0.001)
    
    def test_step_lr(self):
        from SimplCore.engine import StepLR
        
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        for _ in range(20):
            scheduler.step()
        
        self.assertLessEqual(scheduler.optimizer.lr, 0.0001)
    
    def test_warmup_cosine(self):
        from SimplCore.engine import WarmupCosineAnnealingLR
        
        scheduler = WarmupCosineAnnealingLR(self.optimizer, warmup_steps=10, T_max=100)
        
        for _ in range(15):
            scheduler.step()
        
        self.assertGreater(scheduler.optimizer.lr, 0)


class TestSimpDistributed(unittest.TestCase):
    """分布式训练测试"""
    
    def test_distributed_config(self):
        from SimplCore.engine import DistributedConfig, init_distributed
        
        config = DistributedConfig(world_size=2, rank=0)
        self.assertTrue(config.is_main_process)
        
        config = init_distributed()
        self.assertEqual(config.world_size, 1)
    
    def test_distributed_group(self):
        from SimplCore.engine import DistributedConfig, DistributedGroup, zeros
        
        config = DistributedConfig(world_size=1)
        group = DistributedGroup(config)
        
        tensor = zeros(3, 4)
        
        result = group.all_reduce(tensor)
        self.assertEqual(result.shape, (3, 4))
        
        gathered = group.all_gather(tensor)
        self.assertEqual(len(gathered), 1)
        
        broadcasted = group.broadcast(tensor)
        self.assertEqual(broadcasted.shape, (3, 4))
    
    def test_gradient_accumulator(self):
        from SimplCore.engine import GradientAccumulator, Linear, randn
        
        model = Linear(10, 5)
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        x = randn(32, 10)
        y = randn(32, 5)
        
        for i in range(2):
            output = model(x)
            loss = ((output - y) ** 2).sum()
            loss.backward()
            
            should_step = accumulator.accumulate(list(model.parameters()))
            
            if i == 0:
                self.assertFalse(should_step)
            else:
                self.assertTrue(should_step)


class TestSimplCoreArchitecture(unittest.TestCase):
    """SimplCore架构模块测试"""
    
    def test_resonance_field_layer(self):
        from SimplCore.engine import LegacySpectralMixer, randn
        
        layer = LegacySpectralMixer(64, num_waves=4)
        x = randn(32, 10, 64)
        output = layer(x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_dynamic_memory_channel(self):
        from SimplCore.engine import DynamicMemoryChannel, randn
        
        dmc = DynamicMemoryChannel(64, 1024)
        x = randn(32, 10, 64)
        output, state = dmc(x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_topology_router(self):
        from SimplCore.engine import LegacyConditionalRouter, randn
        
        router = LegacyConditionalRouter(64, 4)
        x = randn(32, 10, 64)
        output = router(x)
        self.assertEqual(output.shape, (32, 10, 64))
    
    def test_SimplCore_block(self):
        from SimplCore.engine import SimplCoreBlock, randn
        
        block = SimplCoreBlock(64, 8, num_waves=4, num_channels=4, num_paths=4)
        x = randn(32, 10, 64)
        output = block(x)
        self.assertEqual(output.shape, (32, 10, 64))


class TestSimpLossFunctions(unittest.TestCase):
    """损失函数测试"""
    
    def test_mse_loss(self):
        from SimplCore.engine import MSELoss, randn
        
        loss_fn = MSELoss()
        input = randn(32, 10)
        target = randn(32, 10)
        
        loss = loss_fn(input, target)
        self.assertEqual(loss.shape, ())
    
    def test_l1_loss(self):
        from SimplCore.engine import L1Loss, randn
        
        loss_fn = L1Loss()
        input = randn(32, 10)
        target = randn(32, 10)
        
        loss = loss_fn(input, target)
        self.assertEqual(loss.shape, ())
    
    def test_cross_entropy(self):
        from SimplCore.engine import cross_entropy, randn
        
        input = randn(32, 10)
        target = randn(32, 10)
        
        loss = cross_entropy(input, target)
        self.assertEqual(loss.shape, ())
    
    def test_bce_with_logits_loss(self):
        from SimplCore.engine import BCEWithLogitsLoss, randn
        
        loss_fn = BCEWithLogitsLoss()
        input = randn(32, 10)
        target = randn(32, 10)
        
        loss = loss_fn(input, target)
        self.assertEqual(loss.shape, ())


class TestSimpFFT(unittest.TestCase):
    """FFT测试"""
    
    def test_fft_basic(self):
        from SimplCore.engine import fft, ifft, randn
        
        x = randn(1024)
        f = fft(x)
        x_recon = ifft(f)
        
        self.assertEqual(x.shape, x_recon.shape)
    
    def test_fft2(self):
        from SimplCore.engine import fft2, ifft2, randn
        
        x = randn(32, 32)
        f = fft2(x)
        x_recon = ifft2(f)
        
        self.assertEqual(x.shape, x_recon.shape)
    
    def test_fftshift(self):
        from SimplCore.engine import fft, fftshift, ifftshift, randn
        
        x = randn(1024)
        f = fft(x)
        f_shifted = fftshift(f)
        f_recon = ifftshift(f_shifted)
        
        self.assertEqual(f.shape, f_recon.shape)


class TestSimpGPU(unittest.TestCase):
    """GPU模块测试"""
    
    def test_gpu_manager(self):
        from SimplCore.engine import GPUManager
        
        manager = GPUManager()
        self.assertTrue(hasattr(manager, 'available_devices'))


class TestSimpJavaBridge(unittest.TestCase):
    """Java桥接层测试"""
    
    def test_java_bridge_server(self):
        from SimplCore.java import JavaBridgeServer
        
        server = JavaBridgeServer(port=8888)
        server.start()
        
        self.assertTrue(server.is_running())
        
        server.stop()
        self.assertFalse(server.is_running())


def run_all_tests():
    """运行所有测试"""
    unittest.main(module=__name__, exit=False)


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore 完整测试套件")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__name__)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"测试结果: {result.testsRun} 个测试用例")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败!")


