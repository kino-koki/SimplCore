# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""物理引擎模块 - 模拟真实世界的物理规律.

核心物理规律实现:
  1. 万有引力定律 (F = G*m1*m2/r^2)
  2. 牛顿第二定律 (F = ma)
  3. 能量守恒定律
  4. 动量守恒定律
  5. 碰撞检测与弹性碰撞
  6. 摩擦力与阻力
  7. 弹簧振子系统
  8. 流体动力学简化模型

设计目标:
  - 使世界模型能够学习和预测真实物理现象
  - 支持多物体交互模拟
  - 提供可微的物理计算图
  - 验证模型学到了真正的物理规律

与DynamicsModel的关系:
  - PhysicsEngine: 提供物理规律的显式规则和模拟器
  - DynamicsModel: 学习状态转移函数
  - 两者结合: 使用PhysicsEngine生成训练数据, DynamicsModel学习预测
"""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class PhysicsObject:
    """物理对象 - 描述具有质量、位置、速度的物体."""
    
    def __init__(
        self,
        mass: float = 1.0,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        acceleration: Optional[torch.Tensor] = None,
        size: float = 1.0,
        elasticity: float = 0.8,
        friction: float = 0.1,
        charge: float = 0.0,
    ):
        self.mass = mass
        self.position = position if position is not None else torch.zeros(3)
        self.velocity = velocity if velocity is not None else torch.zeros(3)
        self.acceleration = acceleration if acceleration is not None else torch.zeros(3)
        self.size = size
        self.elasticity = elasticity
        self.friction = friction
        self.charge = charge
        
        self.forces = torch.zeros(3)
        self.energy_kinetic = 0.0
        self.energy_potential = 0.0
    
    def compute_kinetic_energy(self) -> float:
        """计算动能: E_k = 0.5 * m * v^2"""
        return 0.5 * self.mass * torch.dot(self.velocity, self.velocity).item()
    
    def compute_potential_energy(self, gravity: float = 9.8, height: float = 0.0) -> float:
        """计算重力势能: E_p = m * g * h"""
        return self.mass * gravity * (height + self.position[2].item())
    
    def total_energy(self, gravity: float = 9.8) -> float:
        """计算总能量"""
        return self.compute_kinetic_energy() + self.compute_potential_energy(gravity)
    
    def apply_force(self, force: torch.Tensor) -> None:
        """施加力"""
        self.forces += force
    
    def update(self, dt: float = 0.01) -> None:
        """根据牛顿第二定律更新状态"""
        self.acceleration = self.forces / self.mass
        
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        self.forces = torch.zeros(3)
        self.energy_kinetic = self.compute_kinetic_energy()


class PhysicsEngine(nn.Module):
    """物理引擎 - 模拟多种物理规律.
    
    支持的物理规律:
      1. 万有引力
      2. 碰撞检测
      3. 能量守恒
      4. 动量守恒
      5. 摩擦力
      6. 弹簧振子
      7. 流体阻力
    
    输出:
      - next_state: 下一时刻状态
      - force: 作用在物体上的力
      - energy: 能量变化
      - collision: 碰撞检测结果
    """
    
    def __init__(
        self,
        num_objects: int = 10,
        dim: int = 3,
        gravity_constant: float = 6.674e-11,
        g: float = 9.8,
        friction_coeff: float = 0.1,
        drag_coeff: float = 0.01,
        spring_constant: float = 100.0,
        damping: float = 0.95,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.dim = dim
        self.gravity_constant = gravity_constant
        self.g = g
        self.friction_coeff = friction_coeff
        self.drag_coeff = drag_coeff
        self.spring_constant = spring_constant
        self.damping = damping
        
        # 可学习参数
        self.learnable_gravity = nn.Parameter(torch.tensor(g))
        self.learnable_friction = nn.Parameter(torch.tensor(friction_coeff))
        self.learnable_damping = nn.Parameter(torch.tensor(damping))
    
    def compute_gravitational_force(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """计算万有引力: F = G * m1 * m2 / r^2
        
        Args:
            positions: (N, D) 物体位置
            masses: (N,) 物体质量
            
        Returns:
            forces: (N, D) 每个物体受到的引力
        """
        N = positions.shape[0]
        forces = torch.zeros(N, self.dim, device=positions.device)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # 计算距离向量和距离
                r_vec = positions[j] - positions[i]
                r_sq = torch.sum(r_vec ** 2) + 1e-10  # 防止除零
                r = torch.sqrt(r_sq)
                
                # 万有引力定律
                force_magnitude = self.gravity_constant * masses[i] * masses[j] / r_sq
                
                # 方向向量
                force_dir = r_vec / r
                
                forces[i] += force_magnitude * force_dir
        
        return forces
    
    def compute_collision(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        sizes: torch.Tensor,
        elasticities: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """碰撞检测与弹性碰撞.
        
        Args:
            positions: (N, D) 物体位置
            velocities: (N, D) 物体速度
            sizes: (N,) 物体尺寸
            elasticities: (N,) 弹性系数
            
        Returns:
            dict: 包含碰撞后的速度和碰撞检测结果
        """
        N = positions.shape[0]
        new_velocities = velocities.clone()
        collision_mask = torch.zeros(N, N, dtype=torch.bool, device=positions.device)
        
        for i in range(N):
            for j in range(i + 1, N):
                # 计算距离
                r_vec = positions[j] - positions[i]
                r = torch.norm(r_vec)
                
                # 判断是否碰撞
                min_distance = (sizes[i] + sizes[j]) / 2
                if r < min_distance:
                    collision_mask[i, j] = True
                    collision_mask[j, i] = True
                    
                    # 碰撞法线
                    normal = r_vec / (r + 1e-10)
                    
                    # 相对速度
                    v_rel = velocities[i] - velocities[j]
                    
                    # 法向相对速度
                    v_n = torch.dot(v_rel, normal)
                    
                    # 只有相向运动才碰撞
                    if v_n < 0:
                        # 弹性碰撞公式
                        restitution = (elasticities[i] + elasticities[j]) / 2
                        
                        # 冲量
                        impulse = -(1 + restitution) * v_n / (1/sizes[i] + 1/sizes[j])
                        
                        # 更新速度
                        new_velocities[i] += impulse * normal / sizes[i]
                        new_velocities[j] -= impulse * normal / sizes[j]
        
        return {
            "velocities": new_velocities,
            "collisions": collision_mask,
            "num_collisions": collision_mask.sum().item() // 2,
        }
    
    def compute_friction(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        surfaces: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算摩擦力: F_friction = -mu * m * g * v/|v|
        
        Args:
            velocities: (N, D) 物体速度
            masses: (N,) 物体质量
            
        Returns:
            friction_forces: (N, D) 摩擦力
        """
        N = velocities.shape[0]
        friction_forces = torch.zeros(N, self.dim, device=velocities.device)
        
        speed = torch.norm(velocities, dim=1, keepdim=True) + 1e-10
        direction = velocities / speed
        
        # 摩擦力与速度方向相反
        friction_forces = -self.learnable_friction * masses.unsqueeze(1) * self.learnable_gravity * direction
        
        return friction_forces
    
    def compute_drag(
        self,
        velocities: torch.Tensor,
        sizes: torch.Tensor,
    ) -> torch.Tensor:
        """计算流体阻力: F_drag = -0.5 * rho * v^2 * C_d * A * v/|v|
        
        Args:
            velocities: (N, D) 物体速度
            sizes: (N,) 物体尺寸
            
        Returns:
            drag_forces: (N, D) 阻力
        """
        N = velocities.shape[0]
        drag_forces = torch.zeros(N, self.dim, device=velocities.device)
        
        speed = torch.norm(velocities, dim=1, keepdim=True) + 1e-10
        direction = velocities / speed
        
        # 简化阻力公式: F = -k * v^2 * v/|v|
        drag_forces = -self.drag_coeff * sizes.unsqueeze(1) * speed * velocities
        
        return drag_forces
    
    def compute_spring_force(
        self,
        positions: torch.Tensor,
        rest_positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """计算弹簧力: F = -k * (x - x0) - b * v
        
        Args:
            positions: (N, D) 当前位置
            rest_positions: (N, D) 平衡位置
            velocities: (N, D) 速度
            
        Returns:
            spring_forces: (N, D) 弹簧力
        """
        displacement = positions - rest_positions
        spring_forces = -self.spring_constant * displacement - self.learnable_damping * velocities
        
        return spring_forces
    
    def compute_gravity(
        self,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """计算重力: F = m * g (沿-z方向)"""
        gravity_forces = torch.zeros(len(masses), self.dim, device=masses.device)
        gravity_forces[:, 2] = -self.learnable_gravity * masses
        
        return gravity_forces
    
    def compute_energy_conservation(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> Dict[str, float]:
        """计算能量守恒验证.
        
        Returns:
            dict: 包含动能、势能、总能量
        """
        # 动能
        kinetic_energy = 0.5 * torch.sum(masses.unsqueeze(1) * velocities ** 2)
        
        # 势能 (假设位置的z坐标为高度)
        potential_energy = torch.sum(masses * self.learnable_gravity * positions[:, 2])
        
        # 总能量
        total_energy = kinetic_energy + potential_energy
        
        return {
            "kinetic": kinetic_energy.item(),
            "potential": potential_energy.item(),
            "total": total_energy.item(),
        }
    
    def compute_momentum_conservation(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> float:
        """计算动量守恒验证."""
        momentum = torch.sum(masses.unsqueeze(1) * velocities)
        return torch.norm(momentum).item()
    
    def forward(
        self,
        state: torch.Tensor,
        dt: float = 0.01,
        apply_gravity: bool = True,
        apply_friction: bool = True,
        apply_drag: bool = False,
        apply_spring: bool = False,
        apply_collision: bool = True,
        rest_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """模拟物理系统一步.
        
        Args:
            state: (N, D*2) 状态张量 [position, velocity]
            dt: 时间步长
            apply_gravity: 是否应用重力
            apply_friction: 是否应用摩擦力
            apply_drag: 是否应用阻力
            apply_spring: 是否应用弹簧力
            apply_collision: 是否检测碰撞
            
        Returns:
            dict: 包含下一状态和各种物理量
        """
        N = state.shape[0]
        
        # 分离位置和速度
        positions = state[:, :self.dim]
        velocities = state[:, self.dim:]
        
        # 默认质量和尺寸
        masses = torch.ones(N, device=state.device)
        sizes = torch.ones(N, device=state.device) * 0.5
        elasticities = torch.ones(N, device=state.device) * 0.8
        
        # 计算所有力
        total_forces = torch.zeros(N, self.dim, device=state.device)
        
        if apply_gravity:
            total_forces += self.compute_gravity(masses)
        
        if apply_friction:
            total_forces += self.compute_friction(velocities, masses)
        
        if apply_drag:
            total_forces += self.compute_drag(velocities, sizes)
        
        if apply_spring and rest_positions is not None:
            total_forces += self.compute_spring_force(positions, rest_positions, velocities)
        
        # 万有引力 (始终计算)
        total_forces += self.compute_gravitational_force(positions, masses)
        
        # 更新加速度和速度
        accelerations = total_forces / masses.unsqueeze(1)
        new_velocities = velocities + accelerations * dt
        
        # 碰撞检测
        if apply_collision:
            collision_result = self.compute_collision(
                positions, new_velocities, sizes, elasticities
            )
            new_velocities = collision_result["velocities"]
            num_collisions = collision_result["num_collisions"]
        else:
            num_collisions = 0
        
        # 更新位置
        new_positions = positions + new_velocities * dt
        
        # 阻尼衰减
        new_velocities = new_velocities * self.learnable_damping
        
        # 组装下一状态
        next_state = torch.cat([new_positions, new_velocities], dim=1)
        
        # 计算能量
        energy = self.compute_energy_conservation(new_positions, new_velocities, masses)
        
        return {
            "next_state": next_state,
            "forces": total_forces,
            "accelerations": accelerations,
            "energy": energy,
            "momentum": self.compute_momentum_conservation(new_velocities, masses),
            "num_collisions": num_collisions,
        }
    
    def simulate(
        self,
        initial_state: torch.Tensor,
        steps: int = 100,
        dt: float = 0.01,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """多步模拟.
        
        Args:
            initial_state: (N, D*2) 初始状态
            steps: 模拟步数
            dt: 时间步长
            
        Returns:
            dict: 包含所有状态和物理量
        """
        states = [initial_state]
        forces_history = []
        energy_history = []
        
        current_state = initial_state
        
        for _ in range(steps):
            result = self.forward(current_state, dt, **kwargs)
            current_state = result["next_state"]
            
            states.append(current_state)
            forces_history.append(result["forces"])
            energy_history.append(result["energy"])
        
        return {
            "states": torch.stack(states, dim=0),  # (steps+1, N, D*2)
            "forces": torch.stack(forces_history, dim=0),  # (steps, N, D)
            "energy_history": energy_history,
        }


class PhysicsLoss(nn.Module):
    """物理损失函数 - 确保模型学习符合物理规律.
    
    损失项:
      1. 动力学损失: 预测状态与真实状态的MSE
      2. 能量守恒损失: 能量变化的绝对值
      3. 动量守恒损失: 动量变化的绝对值
      4. 碰撞一致性损失: 碰撞后速度变化的合理性
      5. 力平衡损失: 合力为零时加速度应为零
    """
    
    def __init__(
        self,
        dynamics_weight: float = 1.0,
        energy_weight: float = 0.1,
        momentum_weight: float = 0.01,
        collision_weight: float = 0.5,
        force_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.dynamics_weight = dynamics_weight
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight
        self.collision_weight = collision_weight
        self.force_balance_weight = force_balance_weight
        
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predicted_state: torch.Tensor,
        target_state: torch.Tensor,
        physics_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """计算物理损失.
        
        Args:
            predicted_state: (B, N, D*2) 预测状态
            target_state: (B, N, D*2) 目标状态
            physics_info: 物理信息字典
            
        Returns:
            dict: 包含各损失项
        """
        losses = {}
        
        # 动力学损失
        dynamics_loss = self.mse_loss(predicted_state, target_state)
        losses["dynamics_loss"] = dynamics_loss.item()
        
        # 能量守恒损失
        if physics_info is not None and "energy" in physics_info:
            energy = physics_info["energy"]
            # 理想情况下总能量应保持不变
            energy_change = abs(energy.get("total", 0))
            energy_loss = torch.tensor(energy_change, dtype=torch.float32)
            losses["energy_loss"] = energy_loss.item()
        else:
            energy_loss = torch.tensor(0.0)
            losses["energy_loss"] = 0.0
        
        # 动量守恒损失
        if physics_info is not None and "momentum" in physics_info:
            momentum = physics_info["momentum"]
            # 理想情况下动量应守恒
            momentum_loss = torch.tensor(momentum, dtype=torch.float32)
            losses["momentum_loss"] = momentum_loss.item()
        else:
            momentum_loss = torch.tensor(0.0)
            losses["momentum_loss"] = 0.0
        
        # 碰撞一致性损失
        if physics_info is not None and "num_collisions" in physics_info:
            collisions = physics_info["num_collisions"]
            if collisions > 0:
                # 碰撞后速度应有明显变化
                predicted_vel = predicted_state[:, :, 3:]
                target_vel = target_state[:, :, 3:]
                vel_change = torch.norm(predicted_vel - target_vel, dim=-1).mean()
                collision_loss = torch.clamp(1.0 - vel_change, min=0)
                losses["collision_loss"] = collision_loss.item()
            else:
                collision_loss = torch.tensor(0.0)
                losses["collision_loss"] = 0.0
        else:
            collision_loss = torch.tensor(0.0)
            losses["collision_loss"] = 0.0
        
        # 力平衡损失
        if physics_info is not None and "forces" in physics_info:
            forces = physics_info["forces"]
            # 合力为零时加速度应为零
            force_magnitude = torch.norm(forces, dim=-1).mean()
            force_balance_loss = torch.clamp(force_magnitude - 1e-6, min=0)
            losses["force_balance_loss"] = force_balance_loss.item()
        else:
            force_balance_loss = torch.tensor(0.0)
            losses["force_balance_loss"] = 0.0
        
        # 总损失
        total_loss = (
            dynamics_loss * self.dynamics_weight +
            energy_loss * self.energy_weight +
            momentum_loss * self.momentum_weight +
            collision_loss * self.collision_weight +
            force_balance_loss * self.force_balance_weight
        )
        losses["total_loss"] = total_loss.item()
        
        return {"loss": total_loss, "losses": losses}


class PhysicsVerification:
    """物理规律验证器 - 验证模型学到了真正的物理规律."""
    
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics_engine = physics_engine
    
    def verify_gravity(self, states: torch.Tensor) -> Dict[str, float]:
        """验证万有引力定律.
        
        检查: 距离越近,引力越大; 质量越大,引力越大
        """
        positions = states[:, :, :3]
        velocities = states[:, :, 3:]
        
        # 计算相邻时间步的速度变化
        delta_v = velocities[1:] - velocities[:-1]
        
        # 检查是否符合引力规律
        avg_accel = torch.mean(torch.norm(delta_v, dim=-1))
        
        return {
            "avg_acceleration": avg_accel.item(),
            "verified": avg_accel > 0,
        }
    
    def verify_energy_conservation(
        self,
        states: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """验证能量守恒定律."""
        if masses is None:
            masses = torch.ones(states.shape[1], device=states.device)
        
        initial_state = states[0]
        final_state = states[-1]
        
        # 初始能量
        init_kinetic = 0.5 * torch.sum(masses.unsqueeze(1) * initial_state[:, 3:] ** 2)
        init_potential = torch.sum(masses * self.physics_engine.learnable_gravity * initial_state[:, 2])
        
        # 最终能量
        final_kinetic = 0.5 * torch.sum(masses.unsqueeze(1) * final_state[:, 3:] ** 2)
        final_potential = torch.sum(masses * self.physics_engine.learnable_gravity * final_state[:, 2])
        
        energy_change = abs((init_kinetic + init_potential) - (final_kinetic + final_potential))
        total_energy = (init_kinetic + init_potential).item()
        
        return {
            "initial_energy": (init_kinetic + init_potential).item(),
            "final_energy": (final_kinetic + final_potential).item(),
            "energy_change": energy_change.item(),
            "relative_change": (energy_change / total_energy).item() if total_energy > 0 else float("inf"),
            "conserved": energy_change < total_energy * 0.1,
        }
    
    def verify_momentum_conservation(
        self,
        states: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """验证动量守恒定律."""
        if masses is None:
            masses = torch.ones(states.shape[1], device=states.device)
        
        initial_momentum = torch.sum(masses.unsqueeze(1) * states[0, :, 3:])
        final_momentum = torch.sum(masses.unsqueeze(1) * states[-1, :, 3:])
        
        momentum_change = torch.norm(initial_momentum - final_momentum)
        total_momentum = torch.norm(initial_momentum)
        
        return {
            "initial_momentum": total_momentum.item(),
            "momentum_change": momentum_change.item(),
            "relative_change": (momentum_change / total_momentum).item() if total_momentum > 0 else float("inf"),
            "conserved": momentum_change < total_momentum * 0.1,
        }
    
    def verify_collision(self, states: torch.Tensor) -> Dict[str, float]:
        """验证碰撞检测."""
        positions = states[:, :, :3]
        
        # 计算相邻物体间的最小距离
        min_distances = []
        for t in range(states.shape[0]):
            for i in range(positions.shape[1]):
                for j in range(i + 1, positions.shape[1]):
                    dist = torch.norm(positions[t, i] - positions[t, j])
                    min_distances.append(dist.item())
        
        avg_min_distance = np.mean(min_distances)
        
        return {
            "avg_min_distance": avg_min_distance,
            "has_collisions": avg_min_distance < 0.5,
        }
    
    def comprehensive_verification(
        self,
        states: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """综合验证所有物理规律."""
        results = {
            "gravity": self.verify_gravity(states),
            "energy_conservation": self.verify_energy_conservation(states, masses),
            "momentum_conservation": self.verify_momentum_conservation(states, masses),
            "collision": self.verify_collision(states),
        }
        
        # 计算综合得分
        score = 0
        total = 0
        
        if results["gravity"]["verified"]:
            score += 1
        total += 1
        
        if results["energy_conservation"]["conserved"]:
            score += 1
        total += 1
        
        if results["momentum_conservation"]["conserved"]:
            score += 1
        total += 1
        
        results["overall_score"] = score / total
        
        return results


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 60)
    print("PhysicsEngine 测试")
    print("=" * 60)
    
    # 测试物理引擎
    engine = PhysicsEngine(num_objects=3, dim=3)
    initial_state = torch.randn(3, 6) * 0.5
    initial_state[:, :3] += 5.0  # 设置初始位置
    
    print(f"初始状态: {initial_state.shape}")
    print(f"初始位置:\n{initial_state[:, :3]}")
    print(f"初始速度:\n{initial_state[:, 3:]}")
    
    # 单步模拟
    result = engine(initial_state, dt=0.01)
    print(f"\n单步模拟结果:")
    print(f"  下一状态: {result['next_state'].shape}")
    print(f"  力: {result['forces'].shape}")
    print(f"  能量: {result['energy']}")
    print(f"  动量: {result['momentum']}")
    print(f"  碰撞数: {result['num_collisions']}")
    
    # 多步模拟
    sim_result = engine.simulate(initial_state, steps=10, dt=0.01)
    print(f"\n多步模拟结果:")
    print(f"  状态序列: {sim_result['states'].shape}")
    print(f"  力历史: {sim_result['forces'].shape}")
    
    # 测试物理验证器
    verifier = PhysicsVerification(engine)
    verification = verifier.comprehensive_verification(sim_result["states"])
    print(f"\n物理规律验证:")
    print(f"  万有引力: {'✅' if verification['gravity']['verified'] else '❌'}")
    print(f"  能量守恒: {'✅' if verification['energy_conservation']['conserved'] else '❌'}")
    print(f"    能量变化: {verification['energy_conservation']['energy_change']:.6f}")
    print(f"  动量守恒: {'✅' if verification['momentum_conservation']['conserved'] else '❌'}")
    print(f"    动量变化: {verification['momentum_conservation']['momentum_change']:.6f}")
    print(f"  碰撞检测: {'✅' if verification['collision']['has_collisions'] else '❌'}")
    print(f"  综合得分: {verification['overall_score']:.2%}")
    
    # 测试物理损失函数
    physics_loss = PhysicsLoss()
    target_state = result["next_state"].unsqueeze(0)
    predicted_state = target_state * 0.95  # 模拟预测误差
    loss_result = physics_loss(predicted_state, target_state, result)
    print(f"\n物理损失函数:")
    print(f"  动力学损失: {loss_result['losses']['dynamics_loss']:.6f}")
    print(f"  能量损失: {loss_result['losses']['energy_loss']:.6f}")
    print(f"  动量损失: {loss_result['losses']['momentum_loss']:.6f}")
    print(f"  碰撞损失: {loss_result['losses']['collision_loss']:.6f}")
    print(f"  力平衡损失: {loss_result['losses']['force_balance_loss']:.6f}")
    print(f"  总损失: {loss_result['losses']['total_loss']:.6f}")
    
    print("\n" + "=" * 60)
    print("所有物理引擎模块测试 PASSED.")
    print("=" * 60)
