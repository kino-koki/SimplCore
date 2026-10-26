#!/usr/bin/env python3
"""世界模型训练脚本 - 验证完整世界模型训练管道.

训练任务:
  1. 动力学预测: 预测状态转移
  2. 动作生成: 基于状态生成动作
  3. 视频预测: 预测下一帧图像
  4. 语言建模: 基础语言理解

数据: 合成数据 (状态序列 + 动作序列 + 图像序列)
模型: SimplCoreWorldModel (~1.5M 参数)
训练: 10 epochs, CPU
"""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_world_model import SimplCoreWorldModel

# ----------------------------------------------------------------------
# 结果收集
# ----------------------------------------------------------------------
log_lines = []
start_time = time.time()

def log(msg):
    elapsed = time.time() - start_time
    log_line = f"[{elapsed:7.1f}s] {msg}"
    log_lines.append(log_line)
    print(log_line, flush=True)

def save_results(path='world_model_train_results.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(log_lines, f, indent=2, ensure_ascii=False)
    log(f"结果已保存到 {path}")

# ----------------------------------------------------------------------
# 合成数据生成
# ----------------------------------------------------------------------
def create_synthetic_trajectory(n_samples=1000, seq_len=10, state_dim=64, action_dim=8):
    """创建合成轨迹数据: 状态序列 + 动作序列 + 图像特征序列.
    
    动力学模型: Sₜ₊₁ = tanh(W_s * Sₜ + W_a * Aₜ) + noise
    
    Args:
        n_samples: 样本数量.
        seq_len: 序列长度.
        state_dim: 状态维度.
        action_dim: 动作维度.
    
    Returns:
        states: (N, L, state_dim) 状态序列.
        actions: (N, L, action_dim) 动作序列.
        next_states: (N, L, state_dim) 下一状态序列.
    """
    torch.manual_seed(42)
    
    # 随机状态和动作
    states = torch.randn(n_samples, seq_len, state_dim) * 0.5
    actions = torch.randn(n_samples, seq_len, action_dim) * 0.3
    
    # 合成动力学
    W_s = torch.randn(state_dim, state_dim) * 0.1
    W_a = torch.randn(state_dim, action_dim) * 0.1
    
    next_states = []
    for t in range(seq_len):
        s_t = states[:, t]
        a_t = actions[:, t]
        s_next = torch.tanh(s_t @ W_s + a_t @ W_a.T) + torch.randn_like(s_t) * 0.05
        next_states.append(s_next)
    
    next_states = torch.stack(next_states, dim=1)
    
    return states, actions, next_states

def create_synthetic_images(n_samples=1000, seq_len=10, latent_dim=64, image_size=64):
    """创建合成图像数据: 从潜状态解码的图像序列.
    
    Args:
        n_samples: 样本数量.
        seq_len: 序列长度.
        latent_dim: 潜状态维度.
        image_size: 图像尺寸.
    
    Returns:
        images: (N, L, 3, image_size, image_size) 图像序列.
    """
    torch.manual_seed(42)
    
    # 创建简单的解码器
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 3 * image_size * image_size),
        nn.Sigmoid(),
    )
    
    latents = torch.randn(n_samples, seq_len, latent_dim)
    images = []
    for t in range(seq_len):
        with torch.no_grad():
            img_flat = decoder(latents[:, t])
        img = img_flat.view(n_samples, 3, image_size, image_size)
        images.append(img)
    
    return torch.stack(images, dim=1), latents

# ----------------------------------------------------------------------
# 训练配置
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"设备: {device}")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEQ_LEN = 10
STATE_DIM = 64
ACTION_DIM = 8

# ----------------------------------------------------------------------
# 创建模型
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("创建 SimplCoreWorldModel")
log("=" * 60)

cfg = SimplCoreConfig(
    vocab_size=512,
    hidden_size=STATE_DIM,
    num_layers=2,
    num_waves=2,
    num_channels=4,
    channel_size=8,
    num_paths=2,
    min_depth=1,
    max_depth=2,
    ffn_intermediate_ratio=2.0,
    max_sequence_length=64,
    num_modalities=3,
    modality_dims=(STATE_DIM, STATE_DIM, STATE_DIM),
    tie_word_embeddings=True,
    dropout=0.0,
    rfp_num_steps=2,
    rfp_damping=0.1,
    dmc_num_heads=2,
    tar_topk=1,
    use_rollout=True,
    rollout_max_steps=3,
)

model = SimplCoreWorldModel(
    config=cfg,
    action_dim=ACTION_DIM,
    image_size=64,
    use_dynamics=True,
    use_action_head=True,
    use_video_prediction=True,
)

model = model.to(device)
total_params = model.count_parameters()
log(f"世界模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----------------------------------------------------------------------
# 创建合成数据
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("创建合成数据集")
log("=" * 60)

n_train = 500
n_test = 100

# 动力学数据
train_states, train_actions, train_next_states = create_synthetic_trajectory(
    n_samples=n_train, seq_len=SEQ_LEN, state_dim=STATE_DIM, action_dim=ACTION_DIM
)
test_states, test_actions, test_next_states = create_synthetic_trajectory(
    n_samples=n_test, seq_len=SEQ_LEN, state_dim=STATE_DIM, action_dim=ACTION_DIM
)

# 图像数据
train_images, train_latents = create_synthetic_images(
    n_samples=n_train, seq_len=SEQ_LEN, latent_dim=STATE_DIM, image_size=64
)
test_images, test_latents = create_synthetic_images(
    n_samples=n_test, seq_len=SEQ_LEN, latent_dim=STATE_DIM, image_size=64
)

log(f"训练集: states={train_states.shape}, actions={train_actions.shape}, images={train_images.shape}")
log(f"测试集: states={test_states.shape}, actions={test_actions.shape}, images={test_images.shape}")

# ----------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("开始训练世界模型")
log("=" * 60)

results = {
    "train_dynamics_loss": [],
    "train_action_loss": [],
    "train_image_loss": [],
    "test_dynamics_loss": [],
    "test_action_loss": [],
    "test_image_loss": [],
}

for epoch in range(EPOCHS):
    model.train()
    total_dynamics_loss = 0.0
    total_action_loss = 0.0
    total_image_loss = 0.0
    n_batches = 0
    
    # 随机打乱数据
    perm = torch.randperm(n_train)
    train_states_shuffled = train_states[perm].to(device)
    train_actions_shuffled = train_actions[perm].to(device)
    train_next_states_shuffled = train_next_states[perm].to(device)
    train_images_shuffled = train_images[perm].to(device)
    
    for i in range(0, n_train, BATCH_SIZE):
        batch_states = train_states_shuffled[i:i+BATCH_SIZE]
        batch_actions = train_actions_shuffled[i:i+BATCH_SIZE]
        batch_next_states = train_next_states_shuffled[i:i+BATCH_SIZE]
        batch_images = train_images_shuffled[i:i+BATCH_SIZE]
        
        optimizer.zero_grad()
        
        # 1. 动力学预测损失
        dyn_result = model.dynamics_model(batch_states, batch_actions)
        pred_next = dyn_result["next_state"]
        dynamics_loss = F.mse_loss(pred_next, batch_next_states)
        
        # 2. 动作生成损失 (重建动作)
        action_result = model.action_head(batch_states)
        action_pred = action_result["action"]
        action_loss = F.mse_loss(action_pred, batch_actions)
        
        # 3. 视频预测损失 (重建图像)
        img_pred = model.video_prediction_head(batch_states)
        # 使用第一帧作为目标
        target_img = batch_images[:, 0]
        image_loss = F.mse_loss(img_pred, target_img)
        
        # 总损失
        total_loss = dynamics_loss + action_loss + image_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_dynamics_loss += dynamics_loss.item()
        total_action_loss += action_loss.item()
        total_image_loss += image_loss.item()
        n_batches += 1
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_dyn_result = model.dynamics_model(test_states.to(device), test_actions.to(device))
        test_dynamics_loss = F.mse_loss(test_dyn_result["next_state"], test_next_states.to(device)).item()
        
        test_action_result = model.action_head(test_states.to(device))
        test_action_loss = F.mse_loss(test_action_result["action"], test_actions.to(device)).item()
        
        test_img_pred = model.video_prediction_head(test_states.to(device))
        test_image_loss = F.mse_loss(test_img_pred, test_images[:, 0].to(device)).item()
    
    scheduler.step()
    
    avg_train_dyn = total_dynamics_loss / n_batches
    avg_train_action = total_action_loss / n_batches
    avg_train_image = total_image_loss / n_batches
    
    results["train_dynamics_loss"].append(avg_train_dyn)
    results["train_action_loss"].append(avg_train_action)
    results["train_image_loss"].append(avg_train_image)
    results["test_dynamics_loss"].append(test_dynamics_loss)
    results["test_action_loss"].append(test_action_loss)
    results["test_image_loss"].append(test_image_loss)
    
    log(f">>> Epoch {epoch}:")
    log(f"    动力学损失: train={avg_train_dyn:.4f} test={test_dynamics_loss:.4f}")
    log(f"    动作损失:   train={avg_train_action:.4f} test={test_action_loss:.4f}")
    log(f"    图像损失:   train={avg_train_image:.4f} test={test_image_loss:.4f}")

# ----------------------------------------------------------------------
# 世界模型推演测试
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("世界模型推演测试")
log("=" * 60)

model.eval()
with torch.no_grad():
    # 随机初始状态
    initial_state = torch.randn(1, 5, STATE_DIM).to(device)
    
    # 多步推演
    rollout_result = model.rollout_world(initial_state, steps=5)
    
    log(f"初始状态: {initial_state.shape}")
    log(f"推演步数: 5")
    log(f"所有状态: {rollout_result['all_states'].shape}")
    log(f"所有动作: {rollout_result['all_actions'].shape}")
    log(f"所有图像: {rollout_result['all_images'].shape}")
    
    # 验证维度
    assert rollout_result['all_states'].shape == (1, 6, 5, STATE_DIM)
    assert rollout_result['all_actions'].shape == (1, 5, 5, ACTION_DIM)
    assert rollout_result['all_images'].shape == (1, 5, 3, 64, 64)
    
    log("世界模型多步推演验证通过!")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("训练结果摘要")
log("=" * 60)

log(f"世界模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
log(f"训练 {EPOCHS} epochs")

# 检查损失下降
dyn_dropped = results['train_dynamics_loss'][-1] < results['train_dynamics_loss'][0] * 0.8
action_dropped = results['train_action_loss'][-1] < results['train_action_loss'][0] * 0.8
image_dropped = results['train_image_loss'][-1] < results['train_image_loss'][0] * 0.8

log(f"动力学损失: {results['train_dynamics_loss'][0]:.4f} -> {results['train_dynamics_loss'][-1]:.4f} ({'OK' if dyn_dropped else 'WARN'})")
log(f"动作损失:   {results['train_action_loss'][0]:.4f} -> {results['train_action_loss'][-1]:.4f} ({'OK' if action_dropped else 'WARN'})")
log(f"图像损失:   {results['train_image_loss'][0]:.4f} -> {results['train_image_loss'][-1]:.4f} ({'OK' if image_dropped else 'WARN'})")

if dyn_dropped and action_dropped and image_dropped:
    log(">>> 最终结论: 世界模型训练流程完全正常!")
else:
    log(">>> 警告: 部分损失下降不明显")

save_results()

# ----------------------------------------------------------------------
# 保存训练结果到 JSON
# ----------------------------------------------------------------------
with open('world_model_train_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("训练指标已保存到 world_model_train_metrics.json")

