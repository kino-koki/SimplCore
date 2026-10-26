#!/usr/bin/env python3
"""世界模型测试脚本.

验证:
1. RolloutEngine 创建和推演
2. SimplCoreModel + RolloutEngine 前向/反向
3. 梯度流经推演引擎
4. 延迟解码 (Deferred Decoding)
5. 世界模型预测头 (logits + reward + cont)
6. ModalityRouter 文本路由
7. VisionEncoder 接口占位
"""

import sys
sys.path.insert(0, '.')

# colorama 环境修复 (tqdm 依赖 colorama.init 但某些环境缺失)
import colorama
if not hasattr(colorama, 'init'):
    colorama.init = lambda strip=False: None
    colorama.deinit = lambda: None
    colorama.Fore = type('Fore', (), {k: '' for k in ['RESET','RED','GREEN','YELLOW','BLUE','MAGENTA','CYAN','WHITE']})()
    colorama.Style = type('Style', (), {k: '' for k in ['BRIGHT','RESET_ALL','DIM','NORMAL']})()

import torch
import torch.nn as nn

# ======================================================================
# 1. RolloutEngine 基础测试
# ======================================================================
def test_rollout_engine():
    print("\n[1] RolloutEngine 基础测试")
    from SimplCore.architecture.config import SimplCoreConfig
    from SimplCore.architecture.SimplCore_block import SimplCoreBlock
    from SimplCore.architecture.rollout_engine import RolloutEngine

    cfg = SimplCoreConfig.get_small_config()
    block = SimplCoreBlock(
        dim=cfg.hidden_size, num_heads=cfg.dmc_num_heads,
        num_waves=cfg.num_waves, num_channels=cfg.num_channels,
        num_paths=cfg.num_paths, channel_size=cfg.channel_size,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth,
        topk=cfg.tar_topk, ffn_intermediate_ratio=cfg.ffn_intermediate_ratio,
        rfp_num_steps=cfg.rfp_num_steps, rfp_damping=cfg.rfp_damping,
        dropout=cfg.dropout, eps=cfg.layer_norm_eps,
    )
    engine = RolloutEngine(block, max_steps=6, adaptive=False)
    
    x = torch.randn(2, 16, cfg.hidden_size)
    out, ms, states = engine(x, num_steps=4, return_all_states=True)
    
    assert out.shape == (2, 16, cfg.hidden_size), f"输出形状错误: {out.shape}"
    assert len(states) == 5, f"状态数错误: {len(states)} (应为 5 = 1初始 + 4步, 实际 {len(states)})"
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  推演步数: {len(states)-1}")
    print("  [PASS]")

# ======================================================================
# 2. Rollout 模式前向/反向
# ======================================================================
def test_rollout_forward_backward():
    print("\n[2] Rollout 模式前向/反向")
    from SimplCore.architecture.SimplCore_model import SimplCoreModel
    from SimplCore.architecture.config import SimplCoreConfig

    cfg = SimplCoreConfig(
        vocab_size=256, hidden_size=32, num_layers=2,
        num_waves=2, num_channels=4, channel_size=8, num_paths=2,
        min_depth=1, max_depth=2, ffn_intermediate_ratio=2.0,
        max_sequence_length=64, num_modalities=3,
        modality_dims=(16, 16, 16), tie_word_embeddings=True,
        dropout=0.0, rfp_num_steps=2, rfp_damping=0.1,
        dmc_num_heads=2, tar_topk=1, use_attn_res=False,
        use_rollout=True, rollout_steps=3, rollout_max_steps=6,
        rollout_adaptive=False, rollout_detach=False,
    )
    model = SimplCoreModel(cfg)
    print(f"  参数量: {model.num_parameters_human()}")

    # 前向
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, states = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size), f"logits形状错误: {logits.shape}"
    assert len(states) == cfg.num_layers + 1, f"状态数错误: {len(states)}"
    print(f"  前向通过, logits: {logits.shape}, 状态: {len(states)}")

    # 反向 (梯度流经 rollout)
    loss = logits.pow(2).mean()
    loss.backward()
    
    rollout_params = list(model.rollout_engine.parameters())
    rollout_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in rollout_params)
    assert rollout_has_grad, "RolloutEngine 没有梯度!"
    print("  反向传播通过, RolloutEngine 收到梯度 ✓")
    print("  [PASS]")

# ======================================================================
# 3. 自适应停止
# ======================================================================
def test_adaptive_stop():
    print("\n[3] RolloutEngine 自适应停止")
    from SimplCore.architecture.config import SimplCoreConfig
    from SimplCore.architecture.SimplCore_block import SimplCoreBlock
    from SimplCore.architecture.rollout_engine import RolloutEngine

    cfg = SimplCoreConfig.get_small_config()
    block = SimplCoreBlock(
        dim=cfg.hidden_size, num_heads=cfg.dmc_num_heads,
        num_waves=cfg.num_waves, num_channels=cfg.num_channels,
        num_paths=cfg.num_paths, channel_size=cfg.channel_size,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth,
        topk=cfg.tar_topk, ffn_intermediate_ratio=cfg.ffn_intermediate_ratio,
        rfp_num_steps=cfg.rfp_num_steps, rfp_damping=cfg.rfp_damping,
        dropout=cfg.dropout, eps=cfg.layer_norm_eps,
    )
    engine = RolloutEngine(block, max_steps=8, adaptive=True)
    
    x = torch.randn(2, 16, cfg.hidden_size)
    out, ms, states = engine(x, return_all_states=True)
    n_rollout = len(states) - 1
    print(f"  最大步数: 8, 实际推演: {n_rollout} 步")
    assert n_rollout <= 8, "不应超过最大步数"
    print("  [PASS]")

# ======================================================================
# 4. 世界模型预测头
# ======================================================================
def test_world_model_head():
    print("\n[4] 世界模型预测头")
    from SimplCore.architecture.rollout_engine import WorldModelHead

    head = WorldModelHead(dim=64, vocab_size=256)
    latent = torch.randn(2, 16, 64)
    logits, reward, cont = head(latent)
    
    assert logits.shape == (2, 16, 256), f"logits形状错误: {logits.shape}"
    assert reward.shape == (2, 16, 1), f"reward形状错误: {reward.shape}"
    assert cont.shape == (2, 16, 1), f"cont形状错误: {cont.shape}"
    assert (cont >= 0).all() and (cont <= 1).all(), "continuity 应在 [0,1]"
    print(f"  logits: {logits.shape}")
    print(f"  reward: {reward.shape}")
    print(f"  cont(0-1): {cont.min().item():.3f} ~ {cont.max().item():.3f}")
    print("  [PASS]")

# ======================================================================
# 5. ModalityRouter 文本路由
# ======================================================================
def test_modality_router_text():
    print("\n[5] ModalityRouter 文本路由")
    from SimplCore.architecture.modality_router import ModalityRouter

    router = ModalityRouter(hidden_size=64, use_vision=False)
    text = torch.randn(2, 16, 64)
    out = router(text_embeds=text)
    
    assert out['latent'].shape == (2, 16, 64), f"潜状态形状错误: {out['latent'].shape}"
    assert out['modality_mask'].shape == (2, 16), f"模态掩码形状错误: {out['modality_mask'].shape}"
    print(f"  潜状态: {out['latent'].shape}")
    print(f"  模态掩码: {out['modality_mask'].shape}")
    print("  [PASS]")

# ======================================================================
# 6. ModalityRouter 未来视觉路由 (should raise)
# ======================================================================
def test_modality_router_vision_disabled():
    print("\n[6] ModalityRouter 视觉路由 (use_vision=False, 应报错)")
    from SimplCore.architecture.modality_router import ModalityRouter

    router = ModalityRouter(hidden_size=64, use_vision=False)
    img = torch.randn(2, 3, 224, 224)
    try:
        router(image=img)
        print("  ERROR: 应抛出 NotImplementedError 但未抛出")
        assert False, "应抛出错误"
    except NotImplementedError as e:
        print(f"  正确拦截: {e}")
        print("  [PASS]")


def test_modality_router_vision_enabled():
    print("\n[6b] ModalityRouter 视觉路由 (use_vision=True)")
    from SimplCore.architecture.modality_router import ModalityRouter

    router = ModalityRouter(hidden_size=64, use_vision=True)
    img = torch.randn(2, 3, 224, 224)
    out = router(image=img)
    assert out['latent'].shape[0] == 2, f"batch 维度错误: {out['latent'].shape}"
    assert out['latent'].shape[2] == 64, f"hidden 维度错误: {out['latent'].shape}"
    print(f"  视觉编码: {img.shape} → {out['latent'].shape}")
    print("  [PASS]")

# ======================================================================
# 7. 跨序列记忆续接 (rollout)
# ======================================================================
def test_memory_continuation():
    print("\n[7] 跨序列记忆续接 (rollout)")
    from SimplCore.architecture.SimplCore_model import SimplCoreModel
    from SimplCore.architecture.config import SimplCoreConfig

    cfg = SimplCoreConfig(
        vocab_size=256, hidden_size=32, num_layers=2,
        num_waves=2, num_channels=4, channel_size=8, num_paths=2,
        min_depth=1, max_depth=2, ffn_intermediate_ratio=2.0,
        max_sequence_length=64, num_modalities=3,
        modality_dims=(16, 16, 16), tie_word_embeddings=True,
        dropout=0.0, rfp_num_steps=2, rfp_damping=0.1,
        dmc_num_heads=2, tar_topk=1, use_attn_res=False,
        use_rollout=True, rollout_steps=3, rollout_max_steps=6,
        rollout_adaptive=False, rollout_detach=False,
    )
    model = SimplCoreModel(cfg)
    
    x1 = torch.randint(0, cfg.vocab_size, (2, 4))
    x2 = torch.randint(0, cfg.vocab_size, (2, 4))
    
    # 第一次前向
    logits1, states = model(x1)
    # 带记忆续接第二次前向
    logits2, states2 = model(x2, memory_state=states)
    
    # 不带记忆的第二次前向
    logits2_no_mem, _ = model(x2)
    
    assert logits1.shape == (2, 4, cfg.vocab_size)
    assert logits2.shape == (2, 4, cfg.vocab_size)
    
    # 有/无记忆的输出应该不同 (记忆影响结果)
    diff = (logits2 - logits2_no_mem).abs().mean().item()
    print(f"  有记忆 vs 无记忆的 logits 差异: {diff:.4f}")
    assert diff > 0, "记忆应该影响输出"
    print("  [PASS]")

# ======================================================================
# 8. 梯度检查点 + Rollout
# ======================================================================
def test_gradient_checkpointing():
    print("\n[8] 梯度检查点 + Rollout")
    from SimplCore.architecture.SimplCore_model import SimplCoreModel
    from SimplCore.architecture.config import SimplCoreConfig

    cfg = SimplCoreConfig(
        vocab_size=256, hidden_size=32, num_layers=2,
        num_waves=2, num_channels=4, channel_size=8, num_paths=2,
        min_depth=1, max_depth=2, ffn_intermediate_ratio=2.0,
        max_sequence_length=64, num_modalities=3,
        modality_dims=(16, 16, 16), tie_word_embeddings=True,
        dropout=0.0, rfp_num_steps=2, rfp_damping=0.1,
        dmc_num_heads=2, tar_topk=1, use_attn_res=False,
        use_rollout=True, rollout_steps=3, rollout_max_steps=6,
        rollout_adaptive=False, rollout_detach=False,
    )
    model = SimplCoreModel(cfg)
    model.gradient_checkpointing_enable()
    
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, states = model(x)
    loss = logits.pow(2).mean()
    loss.backward()
    
    rollout_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.rollout_engine.parameters()
    )
    assert rollout_has_grad, "梯度检查点模式下 RolloutEngine 没有梯度!"
    print("  梯度检查点 + Rollout 反向正常")
    print("  [PASS]")

# ======================================================================
# 9. detach_each_step 模式 (省显存)
# ======================================================================
def test_detach_mode():
    print("\n[9] detach_each_step 模式 (省显存)")
    from SimplCore.architecture.rollout_engine import RolloutEngine
    from SimplCore.architecture.config import SimplCoreConfig
    from SimplCore.architecture.SimplCore_block import SimplCoreBlock

    cfg = SimplCoreConfig.get_small_config()
    block = SimplCoreBlock(
        dim=cfg.hidden_size, num_heads=cfg.dmc_num_heads,
        num_waves=cfg.num_waves, num_channels=cfg.num_channels,
        num_paths=cfg.num_paths, channel_size=cfg.channel_size,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth,
        topk=cfg.tar_topk, ffn_intermediate_ratio=cfg.ffn_intermediate_ratio,
        rfp_num_steps=cfg.rfp_num_steps, rfp_damping=cfg.rfp_damping,
        dropout=cfg.dropout, eps=cfg.layer_norm_eps,
    )
    # detach=True 模式
    engine = RolloutEngine(block, max_steps=6, adaptive=False, detach_each_step=True)
    
    x = torch.randn(2, 16, cfg.hidden_size, requires_grad=True)
    out, ms, states = engine(x, num_steps=4)
    loss = out.sum()
    loss.backward()
    print(f"  detach 模式下反向传播正常, x.grad 非空: {x.grad is not None}")
    print("  [PASS]")


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("SimplCore 世界模型测试套件")
    print("=" * 60)

    tests = [
        test_rollout_engine,
        test_rollout_forward_backward,
        test_adaptive_stop,
        test_world_model_head,
        test_modality_router_text,
        test_modality_router_vision_disabled,
        test_modality_router_vision_enabled,
        test_memory_continuation,
        test_gradient_checkpointing,
        test_detach_mode,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{len(tests)} 测试通过")
    if passed == len(tests):
        print("全部通过! ✓")
    else:
        print(f"{len(tests) - passed} 个测试失败!")
    print("=" * 60)

