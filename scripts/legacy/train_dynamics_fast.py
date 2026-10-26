#!/usr/bin/env python3
"""专注于动力学模型训练 - 快速验证."""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------------------------------------------------
# 快速数据生成
# ----------------------------------------------------------------------
def generate_physics_data(n_samples, seq_len=16, state_dim=64, action_dim=8):
    """批量生成物理数据."""
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    next_states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    
    # 弹簧振子参数
    k = np.random.uniform(0.5, 2.0, n_samples)
    m = np.random.uniform(0.5, 2.0, n_samples)
    b = np.random.uniform(0.0, 0.2, n_samples)
    
    x = np.random.uniform(-2.0, 2.0, n_samples)
    v = np.random.uniform(-1.0, 1.0, n_samples)
    
    for t in range(seq_len):
        force = np.random.uniform(-0.5, 0.5, n_samples)
        
        a = (-k * x - b * v + force) / m
        v_new = v + a * 0.1
        x_new = x + v_new * 0.1
        
        states[:, t, 0] = x
        states[:, t, 1] = v
        states[:, t, 2] = k
        states[:, t, 3] = m
        
        next_states[:, t, 0] = x_new
        next_states[:, t, 1] = v_new
        next_states[:, t, 2] = k
        next_states[:, t, 3] = m
        
        actions[:, t, 0] = force
        
        x, v = x_new, v_new
    
    states += np.random.randn(*states.shape) * 0.01
    next_states += np.random.randn(*next_states.shape) * 0.01
    actions += np.random.randn(*actions.shape) * 0.01
    
    return (
        torch.from_numpy(states),
        torch.from_numpy(actions),
        torch.from_numpy(next_states),
    )

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

STATE_DIM = 64
ACTION_DIM = 8
SEQ_LEN = 16

BATCH_SIZE = 256
EPOCHS = 100
LR = 5e-4

N_TRAIN = 100000
N_TEST = 20000

# ----------------------------------------------------------------------
# 创建模型
# ----------------------------------------------------------------------
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        self.core = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.output_proj = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        residual = x
        x = self.core(x)
        x = x + residual
        delta = self.output_proj(x)
        return state + delta

class ActionHead(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        return torch.tanh(self.mlp(state))

model = nn.Sequential(
    nn.Identity(),
).to(device)

dynamics_model = DynamicsModel(STATE_DIM, ACTION_DIM).to(device)
action_head = ActionHead(STATE_DIM, ACTION_DIM).to(device)

total_params = sum(p.numel() for p in dynamics_model.parameters()) + sum(p.numel() for p in action_head.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ----------------------------------------------------------------------
# 生成数据
# ----------------------------------------------------------------------
print(f"\n生成训练数据 ({N_TRAIN}样本)...")
start = time.time()
train_states, train_actions, train_next_states = generate_physics_data(N_TRAIN, SEQ_LEN)
print(f"耗时: {time.time() - start:.2f}s")

print(f"生成测试数据 ({N_TEST}样本)...")
start = time.time()
test_states, test_actions, test_next_states = generate_physics_data(N_TEST, SEQ_LEN)
print(f"耗时: {time.time() - start:.2f}s")

# 创建 DataLoader
train_dataset = torch.utils.data.TensorDataset(train_states, train_actions, train_next_states)
test_dataset = torch.utils.data.TensorDataset(test_states, test_actions, test_next_states)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------------------------------
# 优化器
# ----------------------------------------------------------------------
optimizer = optim.AdamW(
    list(dynamics_model.parameters()) + list(action_head.parameters()),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

start = time.time()

for epoch in range(EPOCHS):
    dynamics_model.train()
    action_head.train()
    
    total_dyn = 0.0
    total_act = 0.0
    n_batches = 0
    
    for states, actions, next_states in train_loader:
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        
        optimizer.zero_grad()
        
        # 动力学损失
        pred_next = dynamics_model(states, actions)
        dyn_loss = nn.functional.mse_loss(pred_next, next_states)
        
        # 动作损失
        pred_action = action_head(states)
        act_loss = nn.functional.mse_loss(pred_action, actions)
        
        loss = dyn_loss + act_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(dynamics_model.parameters()) + list(action_head.parameters()), 1.0
        )
        optimizer.step()
        
        total_dyn += dyn_loss.item()
        total_act += act_loss.item()
        n_batches += 1
    
    scheduler.step()
    
    # 评估
    dynamics_model.eval()
    action_head.eval()
    
    with torch.no_grad():
        test_dyn = 0.0
        test_act = 0.0
        test_batches = 0
        
        for states, actions, next_states in test_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            pred_next = dynamics_model(states, actions)
            test_dyn += nn.functional.mse_loss(pred_next, next_states).item()
            
            pred_action = action_head(states)
            test_act += nn.functional.mse_loss(pred_action, actions).item()
            
            test_batches += 1
        
        test_dyn /= test_batches
        test_act /= test_batches
    
    elapsed = time.time() - start
    
    print(f"[Epoch {epoch}/{EPOCHS} {elapsed:.1f}s] "
          f"Dyn={total_dyn/n_batches:.4f} -> {test_dyn:.4f} "
          f"Act={total_act/n_batches:.4f} -> {test_act:.4f}")
    
    if test_dyn < 0.05 and test_act < 0.1:
        print("🎉 目标达成!")
        break

# ----------------------------------------------------------------------
# 最终结果
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)

dynamics_model.eval()
action_head.eval()

with torch.no_grad():
    final_dyn = 0.0
    final_act = 0.0
    for states, actions, next_states in test_loader:
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        
        pred_next = dynamics_model(states, actions)
        final_dyn += nn.functional.mse_loss(pred_next, next_states).item()
        
        pred_action = action_head(states)
        final_act += nn.functional.mse_loss(pred_action, actions).item()
    
    final_dyn /= len(test_loader)
    final_act /= len(test_loader)

print(f"最终动力学损失: {final_dyn:.4f}")
print(f"最终动作损失:   {final_act:.4f}")

# 保存
torch.save({
    'dynamics_model': dynamics_model.state_dict(),
    'action_head': action_head.state_dict(),
}, 'dynamics_model_final.pt')

results = {
    'dynamics_loss': final_dyn,
    'action_loss': final_act,
    'params': total_params,
    'epochs': epoch + 1,
}

with open('dynamics_train_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n训练完成!")
