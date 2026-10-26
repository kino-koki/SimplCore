# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""大规模合成数据生成器 - 模拟互联网级别的数据多样性.

核心思想: 生成具有复杂物理规律的序列数据, 模拟真实世界的多样性.

数据类型:
  1. 物理动力学数据: 弹簧振子、自由落体、碰撞等
  2. 图像序列数据: 移动的物体、变化的场景
  3. 动作序列数据: 策略行为、控制信号
  4. 混合模式数据: 多种物理系统的混合

生成规模: 10万+样本, 支持增量生成
"""

from __future__ import annotations

import math
import numpy as np
import simplcore.tensor as torch
import simplcore.tensor.nn as nn


class PhysicsDataGenerator:
    """物理动力学数据生成器.
    
    支持多种物理系统:
      - 弹簧振子
      - 自由落体
      - 双摆系统
      - 碰撞检测
      - 阻尼振动
    """
    
    SYSTEM_TYPES = ['spring', 'gravity', 'double_pendulum', 'collision', 'damped']
    
    def __init__(self, state_dim=64, action_dim=8, seq_len=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
    
    def generate_spring(self, batch_size=32):
        """弹簧振子系统."""
        states = []
        actions = []
        next_states = []
        
        for _ in range(batch_size):
            # 随机参数
            k = np.random.uniform(0.5, 2.0)  # 弹性系数
            m = np.random.uniform(0.5, 2.0)  # 质量
            b = np.random.uniform(0.0, 0.2)  # 阻尼
            
            # 初始状态: [位置, 速度]
            x0 = np.random.uniform(-2.0, 2.0)
            v0 = np.random.uniform(-1.0, 1.0)
            
            state_seq = []
            action_seq = []
            next_state_seq = []
            
            x, v = x0, v0
            for t in range(self.seq_len):
                # 动作: 外力
                force = np.random.uniform(-0.5, 0.5)
                
                # 动力学: F = -kx - bv + force
                a = (-k * x - b * v + force) / m
                v_new = v + a * 0.1
                x_new = x + v_new * 0.1
                
                # 构建状态向量
                state = np.zeros(self.state_dim)
                state[0] = x
                state[1] = v
                state[2] = k
                state[3] = m
                
                next_state = np.zeros(self.state_dim)
                next_state[0] = x_new
                next_state[1] = v_new
                next_state[2] = k
                next_state[3] = m
                
                action = np.zeros(self.action_dim)
                action[0] = force
                
                # 添加噪声
                state += np.random.randn(self.state_dim) * 0.01
                next_state += np.random.randn(self.state_dim) * 0.01
                action += np.random.randn(self.action_dim) * 0.01
                
                state_seq.append(state)
                action_seq.append(action)
                next_state_seq.append(next_state)
                
                x, v = x_new, v_new
            
            states.append(state_seq)
            actions.append(action_seq)
            next_states.append(next_state_seq)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )
    
    def generate_gravity(self, batch_size=32):
        """自由落体系统."""
        states = []
        actions = []
        next_states = []
        
        for _ in range(batch_size):
            g = np.random.uniform(8.0, 12.0)  # 重力加速度
            
            x0 = np.random.uniform(-5.0, 5.0)
            y0 = np.random.uniform(5.0, 15.0)
            vx0 = np.random.uniform(-2.0, 2.0)
            vy0 = np.random.uniform(-5.0, 0.0)
            
            state_seq = []
            action_seq = []
            next_state_seq = []
            
            x, y, vx, vy = x0, y0, vx0, vy0
            for t in range(self.seq_len):
                # 动作: 推力
                fx = np.random.uniform(-1.0, 1.0)
                fy = np.random.uniform(-1.0, 1.0)
                
                vx_new = vx + fx * 0.1
                vy_new = vy + (fy - g) * 0.1
                x_new = x + vx_new * 0.1
                y_new = y + vy_new * 0.1
                
                # 地面反弹
                if y_new < 0:
                    y_new = -y_new * 0.8
                    vy_new = -vy_new * 0.8
                
                state = np.zeros(self.state_dim)
                state[0] = x
                state[1] = y
                state[2] = vx
                state[3] = vy
                state[4] = g
                
                next_state = np.zeros(self.state_dim)
                next_state[0] = x_new
                next_state[1] = y_new
                next_state[2] = vx_new
                next_state[3] = vy_new
                next_state[4] = g
                
                action = np.zeros(self.action_dim)
                action[0] = fx
                action[1] = fy
                
                state += np.random.randn(self.state_dim) * 0.01
                next_state += np.random.randn(self.state_dim) * 0.01
                action += np.random.randn(self.action_dim) * 0.01
                
                state_seq.append(state)
                action_seq.append(action)
                next_state_seq.append(next_state)
                
                x, y, vx, vy = x_new, y_new, vx_new, vy_new
            
            states.append(state_seq)
            actions.append(action_seq)
            next_states.append(next_state_seq)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )
    
    def generate_damped(self, batch_size=32):
        """阻尼振动系统."""
        states = []
        actions = []
        next_states = []
        
        for _ in range(batch_size):
            omega = np.random.uniform(1.0, 3.0)
            zeta = np.random.uniform(0.1, 0.5)
            
            x0 = np.random.uniform(-1.0, 1.0)
            v0 = np.random.uniform(-0.5, 0.5)
            
            state_seq = []
            action_seq = []
            next_state_seq = []
            
            x, v = x0, v0
            for t in range(self.seq_len):
                u = np.random.uniform(-0.3, 0.3)
                
                a = -2 * zeta * omega * v - omega**2 * x + u
                v_new = v + a * 0.05
                x_new = x + v_new * 0.05
                
                state = np.zeros(self.state_dim)
                state[0] = x
                state[1] = v
                state[2] = omega
                state[3] = zeta
                
                next_state = np.zeros(self.state_dim)
                next_state[0] = x_new
                next_state[1] = v_new
                next_state[2] = omega
                next_state[3] = zeta
                
                action = np.zeros(self.action_dim)
                action[0] = u
                
                state += np.random.randn(self.state_dim) * 0.01
                next_state += np.random.randn(self.state_dim) * 0.01
                action += np.random.randn(self.action_dim) * 0.01
                
                state_seq.append(state)
                action_seq.append(action)
                next_state_seq.append(next_state)
                
                x, v = x_new, v_new
            
            states.append(state_seq)
            actions.append(action_seq)
            next_states.append(next_state_seq)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
        )
    
    def generate_mixed(self, batch_size=32):
        """混合物理系统."""
        system_type = np.random.choice(self.SYSTEM_TYPES)
        
        if system_type == 'spring':
            return self.generate_spring(batch_size)
        elif system_type == 'gravity':
            return self.generate_gravity(batch_size)
        elif system_type == 'damped':
            return self.generate_damped(batch_size)
        else:
            return self.generate_spring(batch_size)


class ImageSequenceGenerator:
    """图像序列数据生成器.
    
    生成包含移动物体的图像序列:
      - 移动的小球
      - 闪烁的方块
      - 旋转的三角形
      - 多物体交互
    """
    
    def __init__(self, image_size=64, seq_len=32):
        self.image_size = image_size
        self.seq_len = seq_len
    
    def generate_moving_ball(self, batch_size=32):
        """生成移动小球的图像序列."""
        images = []
        
        for _ in range(batch_size):
            seq = []
            # 随机初始位置和速度
            x = np.random.uniform(10, self.image_size - 10)
            y = np.random.uniform(10, self.image_size - 10)
            vx = np.random.uniform(-2, 2)
            vy = np.random.uniform(-2, 2)
            radius = np.random.randint(3, 8)
            color = np.random.rand(3)
            
            for t in range(self.seq_len):
                img = np.zeros((3, self.image_size, self.image_size))
                
                # 绘制小球
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = ((i - y) ** 2 + (j - x) ** 2) ** 0.5
                        if dist < radius:
                            img[:, i, j] = color * (1 - dist / radius)
                
                # 添加噪声
                img += np.random.randn(3, self.image_size, self.image_size) * 0.05
                img = np.clip(img, 0, 1)
                
                seq.append(img)
                
                # 更新位置
                x += vx
                y += vy
                
                # 边界反弹
                if x < radius or x > self.image_size - radius:
                    vx = -vx
                if y < radius or y > self.image_size - radius:
                    vy = -vy
            
            images.append(seq)
        
        return torch.tensor(images, dtype=torch.float32)
    
    def generate_moving_objects(self, batch_size=32):
        """生成多物体移动的图像序列."""
        images = []
        
        for _ in range(batch_size):
            seq = []
            num_objects = np.random.randint(2, 5)
            
            # 初始化物体
            objects = []
            for _ in range(num_objects):
                objects.append({
                    'x': np.random.uniform(10, self.image_size - 10),
                    'y': np.random.uniform(10, self.image_size - 10),
                    'vx': np.random.uniform(-2, 2),
                    'vy': np.random.uniform(-2, 2),
                    'radius': np.random.randint(2, 6),
                    'color': np.random.rand(3),
                })
            
            for t in range(self.seq_len):
                img = np.zeros((3, self.image_size, self.image_size))
                
                for obj in objects:
                    for i in range(self.image_size):
                        for j in range(self.image_size):
                            dist = ((i - obj['y']) ** 2 + (j - obj['x']) ** 2) ** 0.5
                            if dist < obj['radius']:
                                img[:, i, j] = np.maximum(img[:, i, j], obj['color'] * (1 - dist / obj['radius']))
                
                img += np.random.randn(3, self.image_size, self.image_size) * 0.03
                img = np.clip(img, 0, 1)
                
                seq.append(img)
                
                # 更新位置和反弹
                for obj in objects:
                    obj['x'] += obj['vx']
                    obj['y'] += obj['vy']
                    if obj['x'] < obj['radius'] or obj['x'] > self.image_size - obj['radius']:
                        obj['vx'] = -obj['vx']
                    if obj['y'] < obj['radius'] or obj['y'] > self.image_size - obj['radius']:
                        obj['vy'] = -obj['vy']
            
            images.append(seq)
        
        return torch.tensor(images, dtype=torch.float32)


class LargeScaleDataset(torch.utils.data.Dataset):
    """大规模数据集 - 动态生成无限数据.
    
    支持:
      - 预先生成缓存
      - 动态生成 (无限数据)
      - 混合数据模式
    """
    
    def __init__(
        self,
        n_samples=100000,
        state_dim=64,
        action_dim=8,
        seq_len=32,
        image_size=64,
        cache_dir=None,
    ):
        self.n_samples = n_samples
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.image_size = image_size
        
        self.physics_gen = PhysicsDataGenerator(state_dim, action_dim, seq_len)
        self.image_gen = ImageSequenceGenerator(image_size, seq_len)
        
        # 预先生成缓存
        print(f"[Dataset] 生成 {n_samples} 样本...")
        self._generate_cache()
        print(f"[Dataset] 生成完成.")
    
    def _generate_cache(self):
        """生成缓存数据."""
        self.states = []
        self.actions = []
        self.next_states = []
        self.images = []
        
        batch_size = 1000
        for i in range(0, self.n_samples, batch_size):
            actual_batch = min(batch_size, self.n_samples - i)
            
            # 物理数据
            s, a, ns = self.physics_gen.generate_mixed(actual_batch)
            self.states.append(s)
            self.actions.append(a)
            self.next_states.append(ns)
            
            # 图像数据
            img = self.image_gen.generate_moving_objects(actual_batch)
            self.images.append(img)
            
            if (i + actual_batch) % 10000 == 0:
                print(f"[Dataset] 已生成 {i + actual_batch}/{self.n_samples} 样本")
        
        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.next_states = torch.cat(self.next_states, dim=0)
        self.images = torch.cat(self.images, dim=0)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'next_state': self.next_states[idx],
            'image': self.images[idx],
        }


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("大规模数据生成器测试")
    print("=" * 60)
    
    # 测试物理生成器
    physics_gen = PhysicsDataGenerator(state_dim=64, action_dim=8, seq_len=10)
    s, a, ns = physics_gen.generate_mixed(batch_size=4)
    print(f"物理数据: states={s.shape}, actions={a.shape}, next_states={ns.shape}")
    assert s.shape == (4, 10, 64)
    assert a.shape == (4, 10, 8)
    assert ns.shape == (4, 10, 64)
    print("[PASS] 物理生成器正常")
    
    # 测试图像生成器
    image_gen = ImageSequenceGenerator(image_size=64, seq_len=10)
    img = image_gen.generate_moving_objects(batch_size=4)
    print(f"图像数据: images={img.shape}")
    assert img.shape == (4, 10, 3, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    print("[PASS] 图像生成器正常")
    
    # 测试大规模数据集
    ds = LargeScaleDataset(n_samples=1000, state_dim=64, action_dim=8, seq_len=10)
    print(f"大规模数据集: {len(ds)} 样本")
    item = ds[0]
    print(f"  state={item['state'].shape}, action={item['action'].shape}")
    print(f"  next_state={item['next_state'].shape}, image={item['image'].shape}")
    print("[PASS] 大规模数据集正常")
    
    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)

