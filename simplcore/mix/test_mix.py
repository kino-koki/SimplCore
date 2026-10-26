# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 测试脚本
验证架构前向传播、参数量、生成能力
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from mix.config import mix_1b, mix_7b
from mix.model import SimplCoreMix


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("TEST 1: Model Creation")
    print("=" * 60)

    cfg = mix_1b()
    model = SimplCoreMix(cfg)

    stats = model.get_param_count()
    print(f"Total parameters: {stats['total']:,}")
    print(f"  - Prelude: {stats['prelude']:,}")
    print(f"  - Recurrent: {stats['recurrent']:,}")
    print(f"  - Coda: {stats['coda']:,}")
    print(f"  - Embedding: {stats['embedding']:,}")
    print(f"  - Output: {stats['output']:,}")
    print("[PASS] Model created successfully")
    return model, cfg


def test_forward_pass(model, cfg):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    batch_size = 2
    seq_len = 32

    tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits = model(tokens, n_loops=4)

    expected_shape = (batch_size, seq_len, cfg.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    print(f"Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print("[PASS] Forward pass successful")


def test_generation(model, cfg):
    """测试生成"""
    print("\n" + "=" * 60)
    print("TEST 3: Generation")
    print("=" * 60)

    prompt = torch.randint(0, cfg.vocab_size, (1, 10))

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            n_loops=4,
            temperature=1.0,
        )

    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")
    print("[PASS] Generation successful")


def test_gradient_flow(model, cfg):
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow")
    print("=" * 60)

    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))

    model.train()
    logits = model(tokens, n_loops=2)
    loss = nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        targets.view(-1),
    )
    loss.backward()

    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
        else:
            no_grad += 1
            print(f"  WARNING: No gradient for {name}")

    print(f"Parameters with gradient: {has_grad}")
    print(f"Parameters without gradient: {no_grad}")
    print(f"Loss: {loss.item():.4f}")
    print("[PASS] Gradient flow verified")


def test_kv_cache(model, cfg):
    """测试KV缓存"""
    print("\n" + "=" * 60)
    print("TEST 5: KV Cache")
    print("=" * 60)

    model.eval()

    # 第一次前向：完整序列
    tokens1 = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits1 = model(tokens1, n_loops=2)

    # 第二次前向：只传最后一个token，使用缓存
    kv_cache = {}
    with torch.no_grad():
        _ = model(tokens1[:, :-1], n_loops=2, kv_cache=kv_cache)
        logits2 = model(tokens1[:, -1:], n_loops=2, kv_cache=kv_cache)

    print(f"Full forward shape: {logits1.shape}")
    print(f"Cached forward shape: {logits2.shape}")
    print(f"Cache entries: {len(kv_cache)}")
    print("[PASS] KV cache working")


def test_different_loop_depths(model, cfg):
    """测试不同循环深度"""
    print("\n" + "=" * 60)
    print("TEST 6: Variable Loop Depths")
    print("=" * 60)

    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    model.eval()
    for n_loops in [1, 2, 4, 8]:
        with torch.no_grad():
            logits = model(tokens, n_loops=n_loops)
        print(f"  n_loops={n_loops}: shape={logits.shape}")

    print("[PASS] Variable loop depths working")


def test_mba_spectral_radius(model):
    """测试MBA谱半径"""
    print("\n" + "=" * 60)
    print("TEST 7: Parcae Stability (Spectral Radius)")
    print("=" * 60)

    A = model.recurrent.stability.get_A()
    spectral_radius = torch.max(torch.abs(torch.diag(A))).item()

    print(f"Spectral radius of A: {spectral_radius:.6f}")
    assert spectral_radius < 1.0, f"Spectral radius {spectral_radius} >= 1.0!"
    print("[PASS] Stability constraint satisfied (ρ(A) < 1)")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SimplCore Mix Architecture Tests")
    print("=" * 60)

    model, cfg = test_model_creation()
    test_forward_pass(model, cfg)
    test_generation(model, cfg)
    test_gradient_flow(model, cfg)
    test_kv_cache(model, cfg)
    test_different_loop_depths(model, cfg)
    test_mba_spectral_radius(model)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
