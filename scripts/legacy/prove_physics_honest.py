#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物理规律学习证明（诚实版，结论由实测指标自动生成，绝不写死成功文案）。

与早期 ``prove_physics_learning.py`` 的区别：
  * ``final_conclusion`` / ``statement`` 不再写死"成功学习了物理规律"；
  * 所有结论字段均由真实误差指标推导（通过阈值见 ``PASS_THRESHOLD``）；
  * 若指标不达标，报告如实写"未学会"，并给出具体误差。

验证三类系统：
  1. 弹簧振子 (spring)
  2. 自由落体 (gravity)
  3. 未见参数组合的泛化 (generalization)

判定阈值（相对误差）：
  * 单系统 pass：相对误差 < 10%
  * 泛化 pass：平均 RMSE < 0.05（与状态尺度归一后）
"""
from __future__ import annotations

import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
PASS_REL_ERR = 10.0      # 单系统相对误差通过阈值 (%)
PASS_GEN_RMSE = 0.05     # 泛化平均 RMSE 通过阈值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 模型（与仓库 dynamics 模型一致，便于加载已训练权重）
# ----------------------------------------------------------------------
class DynamicsModel(nn.Module):
    def __init__(self, state_dim=64, action_dim=8, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        self.core = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
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


# ----------------------------------------------------------------------
# 真实物理模拟器
# ----------------------------------------------------------------------
def simulate_spring(k, m, b, x0, v0, forces, dt=0.1):
    x, v, states = x0, v0, []
    for f in forces:
        a = (-k * x - b * v + f) / m
        v = v + a * dt
        x = x + v * dt
        states.append([x, v])
    return np.array(states, dtype=np.float32)


def simulate_gravity(g, y0, vy0, forces, dt=0.1):
    # y: 高度; 受外力 f（向上为正）
    y, vy, states = y0, vy0, []
    for f in forces:
        a = -g + f
        vy = vy + a * dt
        y = y + vy * dt
        states.append([y, vy])
    return np.array(states, dtype=np.float32)


# ----------------------------------------------------------------------
# 用模型预测（state_dim=64 中前4维编码 x, v, k, m）
# ----------------------------------------------------------------------
STATE_DIM, ACTION_DIM = 64, 8


def model_rollout(model, init, params, forces):
    """init: (x0, v0); params: (p2, p3) 写入 state[2:4]; forces: (T,)"""
    state = np.zeros((1, 1, STATE_DIM), dtype=np.float32)
    state[0, 0, 0] = init[0]
    state[0, 0, 1] = init[1]
    state[0, 0, 2] = params[0]
    state[0, 0, 3] = params[1]
    st = torch.from_numpy(state).to(DEVICE)
    preds = []
    with torch.no_grad():
        for f in forces:
            act = np.zeros((1, 1, ACTION_DIM), dtype=np.float32)
            act[0, 0, 0] = f
            st = model(st, torch.from_numpy(act).to(DEVICE))
            preds.append([float(st[0, 0, 0].cpu()), float(st[0, 0, 1].cpu())])
    return np.array(preds, dtype=np.float32)


# ----------------------------------------------------------------------
# 相对误差（基于真实轨迹幅值归一）
# ----------------------------------------------------------------------
def rel_err(pred, true):
    err = np.abs(pred[:, 0] - true[:, 0])
    rng = float(np.max(true[:, 0]) - np.min(true[:, 0]))
    mean_err = float(np.mean(err))
    rel = mean_err / rng * 100.0 if rng > 1e-6 else 0.0
    rmse = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)))
    return rmse, rel, mean_err


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    print(f"[INFO] device = {DEVICE}")
    model = DynamicsModel(STATE_DIM, ACTION_DIM).to(DEVICE)

    # 尝试加载已有训练权重；无权重则如实报告
    ckpt_path = "dynamics_model_final.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "dynamics_model" in ckpt:
            model.load_state_dict(ckpt["dynamics_model"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f"[OK] 已加载权重: {ckpt_path}")
        weight_loaded = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 未加载权重 ({e})；将以随机初始化模型评测（预期不通过）。")
        weight_loaded = False

    seq_len = 50
    results = {}

    # --- 弹簧 ---
    k = float(np.random.uniform(0.8, 1.8))
    m = float(np.random.uniform(0.8, 1.8))
    b = float(np.random.uniform(0.05, 0.15))
    x0 = float(np.random.uniform(-1.5, 1.5))
    v0 = float(np.random.uniform(-0.8, 0.8))
    forces = np.random.uniform(-0.4, 0.4, seq_len).astype(np.float32)
    true_spring = simulate_spring(k, m, b, x0, v0, forces)
    pred_spring = model_rollout(model, (x0, v0), (k, m), forces)
    rmse_s, rel_s, _ = rel_err(pred_spring, true_spring)
    results["spring_test"] = {
        "k": k, "m": m, "b": b, "rmse": rmse_s,
        "relative_error_percent": rel_s, "passed": int(rel_s < PASS_REL_ERR),
    }
    print(f"[SPRING] k={k:.3f} m={m:.3f} rel_err={rel_s:.2f}% rmse={rmse_s:.4f} "
          f"-> {'PASS' if rel_s < PASS_REL_ERR else 'FAIL'}")

    # --- 重力 ---
    g = float(np.random.uniform(8.0, 11.0))
    y0 = float(np.random.uniform(0.5, 2.0))
    vy0 = float(np.random.uniform(-0.5, 0.5))
    forces_g = np.random.uniform(-0.3, 0.3, seq_len).astype(np.float32)
    true_grav = simulate_gravity(g, y0, vy0, forces_g)
    pred_grav = model_rollout(model, (y0, vy0), (g, 1.0), forces_g)
    rmse_g, rel_g, _ = rel_err(pred_grav, true_grav)
    results["gravity_test"] = {
        "g": g, "rmse": rmse_g,
        "relative_error_percent": rel_g, "passed": int(rel_g < PASS_REL_ERR),
    }
    print(f"[GRAVITY] g={g:.3f} rel_err={rel_g:.2f}% rmse={rmse_g:.4f} "
          f"-> {'PASS' if rel_g < PASS_REL_ERR else 'FAIL'}")

    # --- 泛化（100 组未见参数）---
    gen_rmses = []
    for _ in range(100):
        kk = float(np.random.uniform(0.5, 2.0))
        mm = float(np.random.uniform(0.5, 2.0))
        bb = float(np.random.uniform(0.0, 0.2))
        xx0 = float(np.random.uniform(-2.0, 2.0))
        vv0 = float(np.random.uniform(-1.0, 1.0))
        ff = np.random.uniform(-0.5, 0.5, seq_len).astype(np.float32)
        tr = simulate_spring(kk, mm, bb, xx0, vv0, ff)
        pr = model_rollout(model, (xx0, vv0), (kk, mm), ff)
        gen_rmses.append(float(np.sqrt(np.mean((pr[:, 0] - tr[:, 0]) ** 2))))
    gen_rmses = np.array(gen_rmses)
    mean_rmse = float(np.mean(gen_rmses))
    results["generalization_test"] = {
        "n_tests": 100, "mean_rmse": mean_rmse,
        "min_rmse": float(np.min(gen_rmses)),
        "max_rmse": float(np.max(gen_rmses)),
        "pct_below_0.05": float(np.sum(gen_rmses < 0.05) / 100 * 100),
        "passed": int(mean_rmse < PASS_GEN_RMSE),
    }
    print(f"[GEN] mean_rmse={mean_rmse:.4f} pct<0.05={results['generalization_test']['pct_below_0.05']:.1f}% "
          f"-> {'PASS' if mean_rmse < PASS_GEN_RMSE else 'FAIL'}")

    # --- 结论由指标自动生成（不写死）---
    all_passed = (results["spring_test"]["passed"] and results["gravity_test"]["passed"]
                  and results["generalization_test"]["passed"])
    conclusion = (
        f"基于实测：spring 相对误差 {results['spring_test']['relative_error_percent']:.2f}%，"
        f"gravity 相对误差 {results['gravity_test']['relative_error_percent']:.2f}%，"
        f"泛化平均 RMSE {mean_rmse:.4f}。"
        + ("三项均达到通过阈值，判定为已学习物理规律且泛化良好。"
           if all_passed else
           "存在未达通过阈值的项目，判定为尚未充分学习物理规律，结论需以真实误差为准。")
    )
    results["final_conclusion"] = conclusion
    results["all_passed"] = bool(all_passed)
    results["weight_loaded"] = weight_loaded
    results["pass_thresholds"] = {
        "relative_error_percent": PASS_REL_ERR, "generalization_mean_rmse": PASS_GEN_RMSE,
    }

    with open("final_physics_proof.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("物理规律学习证明（诚实版）")
    print("=" * 64)
    print(conclusion)
    print(f"权重加载: {weight_loaded} | 总体通过: {all_passed}")
    print("=" * 64)


if __name__ == "__main__":
    main()
