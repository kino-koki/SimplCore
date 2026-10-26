#!/usr/bin/env python3
"""改进训练 - 增加模型容量 + 更多物理系统."""

import sys, os, time, json
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----------------------------------------------------------------------
# 多种物理系统数据生成
# ----------------------------------------------------------------------
def generate_spring_data(n_samples, seq_len=16, state_dim=64, action_dim=8):
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    next_states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    
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
    
    return states, actions, next_states

def generate_gravity_data(n_samples, seq_len=16, state_dim=64, action_dim=8):
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    next_states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    
    g = np.random.uniform(8.0, 12.0, n_samples)
    
    x = np.random.uniform(-5.0, 5.0, n_samples)
    y = np.random.uniform(5.0, 15.0, n_samples)
    vx = np.random.uniform(-2.0, 2.0, n_samples)
    vy = np.random.uniform(-5.0, 0.0, n_samples)
    
    for t in range(seq_len):
        fx = np.random.uniform(-1.0, 1.0, n_samples)
        fy = np.random.uniform(-1.0, 1.0, n_samples)
        
        vx_new = vx + fx * 0.1
        vy_new = vy + (fy - g) * 0.1
        x_new = x + vx_new * 0.1
        y_new = y + vy_new * 0.1
        
        mask = y_new < 0
        y_new[mask] = -y_new[mask] * 0.8
        vy_new[mask] = -vy_new[mask] * 0.8
        
        states[:, t, 0] = x
        states[:, t, 1] = y
        states[:, t, 2] = vx
        states[:, t, 3] = vy
        states[:, t, 4] = g
        
        next_states[:, t, 0] = x_new
        next_states[:, t, 1] = y_new
        next_states[:, t, 2] = vx_new
        next_states[:, t, 3] = vy_new
        next_states[:, t, 4] = g
        
        actions[:, t, 0] = fx
        actions[:, t, 1] = fy
        
        x, y, vx, vy = x_new, y_new, vx_new, vy_new
    
    return states, actions, next_states

def generate_damped_data(n_samples, seq_len=16, state_dim=64, action_dim=8):
    states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((n_samples, seq_len, action_dim), dtype=np.float32)
    next_states = np.zeros((n_samples, seq_len, state_dim), dtype=np.float32)
    
    omega = np.random.uniform(1.0, 3.0, n_samples)
    zeta = np.random.uniform(0.1, 0.5, n_samples)
    
    x = np.random.uniform(-1.0, 1.0, n_samples)
    v = np.random.uniform(-0.5, 0.5, n_samples)
    
    for t in range(seq_len):
        u = np.random.uniform(-0.3, 0.3, n_samples)
        a = -2 * zeta * omega * v - omega**2 * x + u
        v_new = v + a * 0.05
        x_new = x + v_new * 0.05
        
        states[:, t, 0] = x
        states[:, t, 1] = v
        states[:, t, 2] = omega
        states[:, t, 3] = zeta
        
        next_states[:, t, 0] = x_new
        next_states[:, t, 1] = v_new
        next_states[:, t, 2] = omega
        next_states[:, t, 3] = zeta
        
        actions[:, t, 0] = u
        
        x, v = x_new, v_new
    
    return states, actions, next_states

def generate_mixed_data(n_samples, seq_len=16, state_dim=64, action_dim=8):
    n_per_system = n_samples // 3
    
    s1, a1, ns1 = generate_spring_data(n_per_system, seq_len, state_dim, action_dim)
    s2, a2, ns2 = generate_gravity_data(n_per_system, seq_len, state_dim, action_dim)
    s3, a3, ns3 = generate_damped_data(n_per_system, seq_len, state_dim, action_dim)
    
    # 补齐剩余样本
    remaining = n_samples - n_per_system * 3
    s4, a4, ns4 = generate_spring_data(remaining, seq_len, state_dim, action_dim)
    
    states = np.concatenate([s1, s2, s3, s4], axis=0)
    actions = np.concatenate([a1, a2, a3, a4], axis=0)
    next_states = np.concatenate([ns1, ns2, ns3, ns4], axis=0)
    
    # 打乱
    perm = np.random.permutation(n_samples)
    return states[perm], actions[perm], next_states[perm]

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
LR = 3e-4

N_TRAIN = 200000
N_TEST = 50000

# ----------------------------------------------------------------------
# 创建更大的模型
# ----------------------------------------------------------------------
class ImprovedDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))
        self.core = nn.Sequential(*layers)
        
        self.output_proj = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        residual = x
        for i in range(0, len(self.core), 4):
            x = self.core[i:i+4](x)
        x = x + residual
        delta = self.output_proj(x)
        return state + delta

model = ImprovedDynamicsModel(STATE_DIM, ACTION_DIM, hidden_dim=256, num_layers=3).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ----------------------------------------------------------------------
# 生成数据
# ----------------------------------------------------------------------
print(f"\n生成训练数据 ({N_TRAIN}样本)...")
start = time.time()
train_states, train_actions, train_next_states = generate_mixed_data(N_TRAIN, SEQ_LEN)
print(f"耗时: {time.time() - start:.2f}s")

print(f"生成测试数据 ({N_TEST}样本)...")
start = time.time()
test_states, test_actions, test_next_states = generate_mixed_data(N_TEST, SEQ_LEN)
print(f"耗时: {time.time() - start:.2f}s")

# 添加噪声
train_states += np.random.randn(*train_states.shape) * 0.01
train_next_states += np.random.randn(*train_next_states.shape) * 0.01
train_actions += np.random.randn(*train_actions.shape) * 0.01

test_states += np.random.randn(*test_states.shape) * 0.01
test_next_states += np.random.randn(*test_next_states.shape) * 0.01
test_actions += np.random.randn(*test_actions.shape) * 0.01

# 创建 DataLoader
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(train_states),
    torch.from_numpy(train_actions),
    torch.from_numpy(train_next_states),
)

test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(test_states),
    torch.from_numpy(test_actions),
    torch.from_numpy(test_next_states),
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------------------------------
# 优化器
# ----------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for states, actions, next_states in train_loader:
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        
        optimizer.zero_grad()
        pred_next = model(states, actions)
        loss = nn.functional.mse_loss(pred_next, next_states)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_batches = 0
        for states, actions, next_states in test_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            pred_next = model(states, actions)
            test_loss += nn.functional.mse_loss(pred_next, next_states).item()
            test_batches += 1
        test_loss /= test_batches
    
    print(f"[Epoch {epoch}] 训练={total_loss/n_batches:.6f} 测试={test_loss:.6f}")
    
    if test_loss < 0.005:
        print("🎉 目标达成!")
        break

# ----------------------------------------------------------------------
# 保存模型
# ----------------------------------------------------------------------
torch.save({'dynamics_model': model.state_dict()}, 'dynamics_model_improved.pt')
print("\n模型已保存到 dynamics_model_improved.pt")
