#!/usr/bin/env python3
"""高效世界模型训练脚本 - 使用快速数据生成器."""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim

from SimplCore.architecture.improved_world_model import (
    ImprovedDynamicsModel,
    ImprovedActionHead,
    ImprovedVideoPredictionHead,
)
from SimplCore.data.fast_dataset import FastDataset

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

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"设备: {device}")

STATE_DIM = 64
ACTION_DIM = 8
IMAGE_SIZE = 64
SEQ_LEN = 16

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

N_TRAIN = 20000
N_TEST = 5000

# ----------------------------------------------------------------------
# 创建模型
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("创建改进的世界模型")
log("=" * 60)

class CompactWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics_model = ImprovedDynamicsModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
        )
        
        self.action_head = ImprovedActionHead(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            dropout=0.3,
        )
        
        self.video_prediction_head = ImprovedVideoPredictionHead(
            latent_dim=STATE_DIM,
            image_size=IMAGE_SIZE,
            dropout=0.3,
        )

model = CompactWorldModel()
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
log(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ----------------------------------------------------------------------
# 创建数据集
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log(f"创建数据集 ({N_TRAIN}训练 + {N_TEST}测试)")
log("=" * 60)

train_dataset = FastDataset(
    n_samples=N_TRAIN,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    seq_len=SEQ_LEN,
    image_size=IMAGE_SIZE,
)

test_dataset = FastDataset(
    n_samples=N_TEST,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    seq_len=SEQ_LEN,
    image_size=IMAGE_SIZE,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ----------------------------------------------------------------------
# 优化器
# ----------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("开始训练")
log("=" * 60)

best_dyn = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_dyn = 0.0
    total_act = 0.0
    total_img = 0.0
    n_batches = 0
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 动力学损失
        dyn_result = model.dynamics_model(batch["state"], batch["action"])
        dyn_loss = nn.functional.mse_loss(dyn_result["next_state"], batch["next_state"])
        
        # 动作损失
        act_result = model.action_head(batch["state"])
        act_loss = nn.functional.mse_loss(act_result["action"], batch["action"])
        
        # 图像损失
        img_pred = model.video_prediction_head(batch["state"])
        img_loss = nn.functional.mse_loss(img_pred, batch["image"][:, 0])
        
        loss = dyn_loss + act_loss + img_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_dyn += dyn_loss.item()
        total_act += act_loss.item()
        total_img += img_loss.item()
        n_batches += 1
    
    scheduler.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_dyn = 0.0
        test_act = 0.0
        test_img = 0.0
        test_batches = 0
        
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            dyn_result = model.dynamics_model(batch["state"], batch["action"])
            test_dyn += nn.functional.mse_loss(dyn_result["next_state"], batch["next_state"]).item()
            
            act_result = model.action_head(batch["state"])
            test_act += nn.functional.mse_loss(act_result["action"], batch["action"]).item()
            
            img_pred = model.video_prediction_head(batch["state"])
            test_img += nn.functional.mse_loss(img_pred, batch["image"][:, 0]).item()
            
            test_batches += 1
        
        test_dyn /= test_batches
        test_act /= test_batches
        test_img /= test_batches
    
    if test_dyn < best_dyn:
        best_dyn = test_dyn
    
    log(f">>> Epoch {epoch}:")
    log(f"    训练: Dyn={total_dyn/n_batches:.4f} Act={total_act/n_batches:.4f} Img={total_img/n_batches:.4f}")
    log(f"    测试: Dyn={test_dyn:.4f} Act={test_act:.4f} Img={test_img:.4f}")
    
    # 检查目标
    if test_dyn < 0.1 and test_act < 0.1 and test_img < 0.01:
        log("🎉 所有目标达成!")
        break

# ----------------------------------------------------------------------
# 最终评估
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("最终评估")
log("=" * 60)

model.eval()
with torch.no_grad():
    final_dyn = 0.0
    final_act = 0.0
    final_img = 0.0
    test_batches = 0
    
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        dyn_result = model.dynamics_model(batch["state"], batch["action"])
        final_dyn += nn.functional.mse_loss(dyn_result["next_state"], batch["next_state"]).item()
        
        act_result = model.action_head(batch["state"])
        final_act += nn.functional.mse_loss(act_result["action"], batch["action"]).item()
        
        img_pred = model.video_prediction_head(batch["state"])
        final_img += nn.functional.mse_loss(img_pred, batch["image"][:, 0]).item()
        
        test_batches += 1
    
    final_dyn /= test_batches
    final_act /= test_batches
    final_img /= test_batches

log(f"最终动力学损失: {final_dyn:.4f}")
log(f"最终动作损失:   {final_act:.4f}")
log(f"最终图像损失:   {final_img:.4f}")

# 保存模型
torch.save(model.state_dict(), 'world_model_optimized.pt')
log("模型已保存")

# 保存结果
with open('world_model_optimized_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        "dynamics_loss": final_dyn,
        "action_loss": final_act,
        "image_loss": final_img,
        "params": total_params,
    }, f, indent=2)

log("\n训练完成!")

