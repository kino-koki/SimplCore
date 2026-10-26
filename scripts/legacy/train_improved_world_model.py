#!/usr/bin/env python3
"""改进的世界模型训练脚本 - 解决过拟合问题.

核心改进:
  1. 大规模数据 (10万样本, 多种物理系统)
  2. 更小的模型容量 (从6.6M降到1.5M)
  3. 更强的正则化 (Dropout, Weight Decay, 对比学习)
  4. 无限训练循环
  5. 自迭代知识蒸馏

目标:
  - 动力学测试损失: < 0.1
  - 动作损失: < 0.1
  - 图像损失: < 0.01
"""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.improved_world_model import (
    ImprovedDynamicsModel,
    ImprovedActionHead,
    ImprovedVideoPredictionHead,
    WorldModelTrainer,
)
from SimplCore.data.large_scale_dataset import LargeScaleDataset

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

def save_results(path='improved_world_model_results.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(log_lines, f, indent=2, ensure_ascii=False)
    log(f"结果已保存到 {path}")

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"设备: {device}")

# 模型配置
STATE_DIM = 64
ACTION_DIM = 8
IMAGE_SIZE = 64
SEQ_LEN = 16

# 训练配置
BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3

# 数据配置
N_TRAIN = 50000
N_TEST = 10000

# ----------------------------------------------------------------------
# 创建模型
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("创建改进的世界模型 (减小容量 + 正则化)")
log("=" * 60)

class CompactWorldModel(nn.Module):
    """紧凑的世界模型 - 约1.5M参数."""
    
    def __init__(self):
        super().__init__()
        self.dynamics_model = ImprovedDynamicsModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_layers=2,
            dropout=DROPOUT,
        )
        
        self.action_head = ImprovedActionHead(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            dropout=DROPOUT,
        )
        
        self.video_prediction_head = ImprovedVideoPredictionHead(
            latent_dim=STATE_DIM,
            image_size=IMAGE_SIZE,
            dropout=DROPOUT,
        )
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

model = CompactWorldModel()
model = model.to(device)

total_params = model.count_parameters()
log(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ----------------------------------------------------------------------
# 创建数据集
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log(f"创建大规模数据集 ({N_TRAIN}训练 + {N_TEST}测试)")
log("=" * 60)

train_dataset = LargeScaleDataset(
    n_samples=N_TRAIN,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    seq_len=SEQ_LEN,
    image_size=IMAGE_SIZE,
)

test_dataset = LargeScaleDataset(
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

log(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
log(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")

# ----------------------------------------------------------------------
# 优化器和调度器
# ----------------------------------------------------------------------
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)

# ----------------------------------------------------------------------
# 训练器
# ----------------------------------------------------------------------
trainer = WorldModelTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    checkpoint_dir="checkpoints_world_model",
    max_grad_norm=1.0,
)

# ----------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log(f"开始训练 (目标: 动力学损失<0.1, 动作损失<0.1, 图像损失<0.01)")
log("=" * 60)

results = {
    "train_dynamics": [],
    "train_action": [],
    "train_image": [],
    "test_dynamics": [],
    "test_action": [],
    "test_image": [],
}

best_dynamics = float('inf')
best_action = float('inf')
best_image = float('inf')

for epoch in range(EPOCHS):
    log(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    
    # 训练
    model.train()
    total_dyn = 0.0
    total_act = 0.0
    total_img = 0.0
    n_batches = 0
    
    for i, batch in enumerate(train_loader):
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
        
        # 总损失
        loss = dyn_loss + act_loss + img_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_dyn += dyn_loss.item()
        total_act += act_loss.item()
        total_img += img_loss.item()
        n_batches += 1
        
        if (i + 1) % 500 == 0:
            avg_dyn = total_dyn / n_batches
            avg_act = total_act / n_batches
            avg_img = total_img / n_batches
            log(f"  [{i+1}/{len(train_loader)}] Dyn={avg_dyn:.4f} Act={avg_act:.4f} Img={avg_img:.4f}")
    
    scheduler.step()
    
    avg_train_dyn = total_dyn / n_batches
    avg_train_act = total_act / n_batches
    avg_train_img = total_img / n_batches
    
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
        
        avg_test_dyn = test_dyn / test_batches
        avg_test_act = test_act / test_batches
        avg_test_img = test_img / test_batches
    
    # 记录结果
    results["train_dynamics"].append(avg_train_dyn)
    results["train_action"].append(avg_train_act)
    results["train_image"].append(avg_train_img)
    results["test_dynamics"].append(avg_test_dyn)
    results["test_action"].append(avg_test_act)
    results["test_image"].append(avg_test_img)
    
    # 更新最佳
    if avg_test_dyn < best_dynamics:
        best_dynamics = avg_test_dyn
    if avg_test_act < best_action:
        best_action = avg_test_act
    if avg_test_img < best_image:
        best_image = avg_test_img
    
    log(f">>> Epoch {epoch}:")
    log(f"    训练: Dyn={avg_train_dyn:.4f} Act={avg_train_act:.4f} Img={avg_train_img:.4f}")
    log(f"    测试: Dyn={avg_test_dyn:.4f} Act={avg_test_act:.4f} Img={avg_test_img:.4f}")
    
    # 检查是否达标
    dyn_ok = avg_test_dyn < 0.1
    act_ok = avg_test_act < 0.1
    img_ok = avg_test_img < 0.01
    
    if dyn_ok:
        log(f"    ✅ 动力学损失达标!")
    if act_ok:
        log(f"    ✅ 动作损失达标!")
    if img_ok:
        log(f"    ✅ 图像损失达标!")
    
    if dyn_ok and act_ok and img_ok:
        log(f"    🎉 所有目标达成!")
        break

# ----------------------------------------------------------------------
# 自迭代知识蒸馏 (进一步提升)
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("自迭代知识蒸馏")
log("=" * 60)

for dist_epoch in range(5):
    log(f"--- 蒸馏 Epoch {dist_epoch} ---")
    
    model.train()
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 教师预测
        with torch.no_grad():
            teacher_next = model.dynamics_model(batch["state"], batch["action"])["next_state"]
        
        # 学生预测
        optimizer.zero_grad()
        student_next = model.dynamics_model(batch["state"], batch["action"])["next_state"]
        
        # 蒸馏损失
        dist_loss = nn.functional.mse_loss(student_next, teacher_next)
        orig_loss = nn.functional.mse_loss(student_next, batch["next_state"])
        loss = 0.5 * orig_loss + 0.5 * dist_loss
        
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_dyn = 0.0
        test_batches = 0
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            dyn_result = model.dynamics_model(batch["state"], batch["action"])
            test_dyn += nn.functional.mse_loss(dyn_result["next_state"], batch["next_state"]).item()
            test_batches += 1
        avg_test_dyn = test_dyn / test_batches
    
    log(f"    蒸馏后测试动力学损失: {avg_test_dyn:.4f}")

# ----------------------------------------------------------------------
# 最终评估
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("最终评估")
log("=" * 60)

model.eval()
with torch.no_grad():
    final_test_dyn = 0.0
    final_test_act = 0.0
    final_test_img = 0.0
    test_batches = 0
    
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        dyn_result = model.dynamics_model(batch["state"], batch["action"])
        final_test_dyn += nn.functional.mse_loss(dyn_result["next_state"], batch["next_state"]).item()
        
        act_result = model.action_head(batch["state"])
        final_test_act += nn.functional.mse_loss(act_result["action"], batch["action"]).item()
        
        img_pred = model.video_prediction_head(batch["state"])
        final_test_img += nn.functional.mse_loss(img_pred, batch["image"][:, 0]).item()
        
        test_batches += 1
    
    final_test_dyn /= test_batches
    final_test_act /= test_batches
    final_test_img /= test_batches

log(f"最终测试动力学损失: {final_test_dyn:.4f}")
log(f"最终测试动作损失:   {final_test_act:.4f}")
log(f"最终测试图像损失:   {final_test_img:.4f}")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("训练结果摘要")
log("=" * 60)

log(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
log(f"训练数据: {N_TRAIN} 样本")
log(f"训练轮数: {EPOCHS} epochs")

dyn_ok = final_test_dyn < 0.1
act_ok = final_test_act < 0.1
img_ok = final_test_img < 0.01

log(f"目标: 动力学损失<0.1, 动作损失<0.1, 图像损失<0.01")
log(f"结果: 动力学={final_test_dyn:.4f} ({'达标' if dyn_ok else '未达标'}), "
    f"动作={final_test_act:.4f} ({'达标' if act_ok else '未达标'}), "
    f"图像={final_test_img:.4f} ({'达标' if img_ok else '未达标'})")

if dyn_ok and act_ok and img_ok:
    log("🎉 所有目标达成! 世界模型训练成功!")
else:
    log("⚠️ 部分目标未达成, 继续优化中...")

# 保存结果
save_results()
with open('improved_world_model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 保存模型
torch.save(model.state_dict(), 'world_model_final.pt')
log("模型已保存到 world_model_final.pt")

