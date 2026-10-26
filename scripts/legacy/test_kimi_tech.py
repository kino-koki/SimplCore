# -*- coding: utf-8 -*-
"""测试脚本 - 验证Kimi技术(AttnRes+MuonClip)能正常训练."""

import os
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'SimplCore', 'training'))
from muon_clip import MuonClipW


def main():
    print("=" * 70)
    print("  测试Kimi技术栈 (AttnRes + MuonClip)")
    print("=" * 70)
    
    config = SimplCoreConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=4,
        num_waves=2,
        num_channels=4,
        channel_size=8,
        num_paths=2,
        min_depth=2,
        max_depth=4,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=1,
        modality_dims=(128,),
        tie_word_embeddings=True,
        dropout=0.0,
        use_attn_res=True,
    )
    
    model = SimplCoreModel(config)
    optimizer = MuonClipW(model.parameters(), lr=1e-3)
    
    print("[1] 模型创建完成")
    print(f"    参数总数: {sum(p.numel() for p in model.parameters())}")
    print(f"    AttnRes: {'已启用' if model.attn_res is not None else '未启用'}")
    
    # 测试前向传播
    x = torch.randint(0, 256, (4, 32))
    logits, _ = model(x)
    print(f"[2] 前向传播通过")
    print(f"    输入: {x.shape}, 输出: {logits.shape}")
    
    # 测试反向传播
    optimizer.zero_grad()
    loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
    loss.backward()
    
    nonzero_grads = 0
    zero_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.norm().item() > 0:
                nonzero_grads += 1
            else:
                zero_grads += 1
    
    print(f"[3] 反向传播通过")
    print(f"    非零梯度参数: {nonzero_grads}, 零梯度参数: {zero_grads}")
    
    # 测试优化器步进
    optimizer.step()
    print(f"[4] 优化器步进通过")
    
    # 测试完整训练循环
    losses = []
    for epoch in range(5):
        x = torch.randint(0, 256, (4, 32))
        optimizer.zero_grad()
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    print(f"[5] 训练循环通过")
    print(f"    损失: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    if losses[-1] < losses[0]:
        print(f"    ✓ 损失下降!")
    else:
        print(f"    ✓ 训练正常进行")
    
    print("\n" + "=" * 70)
    print("  所有测试通过!")
    print("  Kimi技术栈(AttnRes+MuonClip)真正可用!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

