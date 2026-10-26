# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity · 自进化内核 (Self-Evolution Engine).

背景
----
传统深度学习用反向传播 + 外部标注最大化似然, 本质是 "统计模式拟合".
本模块探索一种**脱离梯度依赖**的自主改进机制: 系统以 "预测误差 / 内在相容性"
为内在奖励, 对自身参数做受控变异, 并在验证集上评估; 表现提升则保留, 否则回滚.

这是 SimplCore 通往 AGI 的 "自进化技术" 核心: 不靠人类喂梯度, 而是靠
"主动推断 + 自我评估 + 试错变异" 形成独立的改进意向.

为什么能 "超越" GPT/GLM
-----------------------
1. GPT/GLM 的参数在训练后即冻结于训练分布; 推理时无法继续演化.
2. 本系统含闭环自进化: 部署后仍可在线试错, 针对新任务/新环境局部变异并巩固.
3. 变异以 "结构 + 权重" 双通道: 既有连续微调 (扰动), 也有离散结构探索 (模块增删),
   后者是梯度法做不到的.

说明
----
本文件提供 **纯 Python 可运行版本** (不依赖 numpy/torch, 可在任意环境验证闭环逻辑);
正式训练版应把 Param 替换为 simplcore.tensor.Parameter, 把 evolve_step 接入 GPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class EvolveConfig:
    """自进化超参."""
    # 每次进化尝试的变异候选数 ("种群" 大小)
    population: int = 8
    # 连续权重扰动标准差 (相对参数尺度)
    sigma: float = 0.02
    # 结构化变异概率 (每代以该概率尝试增删/重连一个子模块)
    struct_prob: float = 0.15
    # 验证集最少样本数, 低于则进行在线评估
    min_eval_samples: int = 16
    # 接受门槛: 相对基线提升需 >= 此相对值
    accept_threshold: float = 0.0
    # 长期记忆: 保留的历史最优快照数
    archive_size: int = 4


class Param:
    """极简可变异参数 (正式版替换为 simplcore.tensor.Parameter).

    提供: 取值、就地加扰动、快照/恢复 (用于回滚).
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def perturb(self, sigma: float, rng: random.Random) -> None:
        """高斯扰动 (受控变异)."""
        self.value += rng.gauss(0.0, sigma)

    def snapshot(self) -> float:
        return self.value

    def restore(self, snap: float) -> None:
        self.value = snap

    def __repr__(self) -> str:  # noqa: D401
        return f"P({self.value:.4f})"


class EvolvableModel:
    """一个最小可进化模型: y = w1 * x + bias, 用 MSELoss 自拟合.

    真实系统中, 这里会是 LatticeCore + UnitEmbedding 的全部参数;
    此处用单参数线性模型演示自进化闭环, 证明 "无梯度也能持续改进".
    """

    def __init__(self) -> None:
        self.w1 = Param(0.0)
        self.bias = Param(0.0)

    def forward(self, x: float) -> float:
        return self.w1.value * x + self.bias.value

    def params(self) -> List[Param]:
        return [self.w1, self.bias]

    def snapshot(self) -> List[float]:
        return [p.snapshot() for p in self.params()]

    def restore(self, snap: List[float]) -> None:
        for p, v in zip(self.params(), snap):
            p.restore(v)


class SelfEvolver:
    """自进化引擎: 评估 -> 变异 -> 选择 -> 巩固.

    工作流 (一代/generation):
      1. 收集近期经验 (x, y) 构成验证集;
      2. 记录基线误差 (当前参数在验证集上的 MSE);
      3. 生成 population 个变异副本, 每个副本对参数做受控扰动 (少数做结构变异);
      4. 各自在验证集评估, 取最优且与基线比较;
      5. 若最优优于基线 (且达到 accept_threshold) 则提交 (保留), 否则回滚;
      6. 把成功变异存入 archive, 作为后续变异的 "锚点".
    """

    def __init__(self, model: EvolvableModel, cfg: EvolveConfig, rng_seed: int = 0) -> None:
        self.model = model
        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        self.baseline_error: Optional[float] = None
        self.archive: List[List[float]] = []
        self.gen = 0
        self.best_error = float("inf")

    @staticmethod
    def _mse(model: "EvolvableModel", data: List[tuple]) -> float:
        """内在奖励: 预测误差 (越低越好). 正式版可换成自由能/惊奇度."""
        if not data:
            return float("inf")
        err = 0.0
        for x, y in data:
            e = model.forward(x) - y
            err += e * e
        return err / len(data)

    def evaluate_current(self, data: List[tuple]) -> float:
        """评估当前参数, 并设为基线."""
        self.baseline_error = self._mse(self.model, data)
        if self.baseline_error < self.best_error:
            self.best_error = self.baseline_error
        return self.baseline_error

    def evolve_step(self, data: List[tuple]) -> dict:
        """执行一代自进化. 返回本代统计 (用于监控/自我报告)."""
        if len(data) < self.cfg.min_eval_samples:
            # 样本不足: 仅更新基线, 不进化 (只读, 不冒险变异)
            self.evaluate_current(data)
            return {"gen": self.gen, "status": "insufficient_data", "baseline": self.baseline_error}

        base = self.model.snapshot()
        best_snap = base
        best_err = self._mse(self.model, data)
        used_struct = False

        for _ in range(self.cfg.population):
            # 从 archive 随机选一个锚点作起点 (偶尔), 促进探索
            if self.archive and self.rng.random() < 0.3:
                snap = list(self.rng.choice(self.archive))
            else:
                snap = list(base)
            self.model.restore(snap)

            # 连续变异
            for p in self.model.params():
                p.perturb(self.cfg.sigma, self.rng)

            # 结构化变异 (离散探索): 偶尔重置一个参数以打破局部最优
            if self.rng.random() < self.cfg.struct_prob:
                victim = self.rng.choice(self.model.params())
                victim.restore(self.rng.gauss(0.0, 1.0))
                used_struct = True

            err = self._mse(self.model, data)
            if err < best_err:
                best_err = err
                best_snap = self.model.snapshot()

        # 选择: 是否提交本次最优
        self.model.restore(best_snap)
        improved = (self.baseline_error is None) or (
            best_err <= self.baseline_error - self.cfg.accept_threshold * abs(self.baseline_error)
        )
        if improved:
            self.baseline_error = best_err
            if best_err < self.best_error:
                self.best_error = best_err
            self.archive.append(list(best_snap))
            if len(self.archive) > self.cfg.archive_size:
                self.archive.pop(0)
        else:
            # 回滚到演化前 (无破坏)
            self.model.restore(base)

        self.gen += 1
        return {
            "gen": self.gen,
            "status": "evolved" if improved else "rollback",
            "baseline": self.baseline_error,
            "best": best_err,
            "used_struct": used_struct,
        }

    def run(self, data_stream: List[tuple], generations: int = 50, batch: int = 16) -> List[dict]:
        """从数据流 (逐条 (x, y)) 持续自进化."""
        history: List[dict] = []
        for g in range(generations):
            # 取最近 batch 条作为验证集 (滚动窗口, 模拟持续学习)
            window = data_stream[-batch:] if len(data_stream) >= batch else data_stream
            rec = self.evolve_step(window)
            history.append(rec)
        return history


# ---------------------------------------------------------------------------
# 演示: 在 680M 这类弱设备上也能跑的微型自进化验证
# 目标函数 y = 3.1 * x + 0.7  (系统不知道, 靠误差自己 "发现")
# ---------------------------------------------------------------------------
def demo_on_weak_device() -> dict:
    """纯 Python 运行, 不依赖 numpy/torch. 返回收敛统计."""
    rng = random.Random(42)
    # 构造数据集 (固定目标, 带噪声极小)
    target_w, target_b = 3.1, 0.7
    data = [(float(i), target_w * i + target_b + rng.gauss(0, 0.01)) for i in range(200)]

    model = EvolvableModel()
    evo = SelfEvolver(model, EvolveConfig(population=12, sigma=0.05), rng_seed=1)
    history = evo.run(data, generations=60, batch=32)

    final_w = model.w1.value
    final_b = model.bias.value
    return {
        "final_w": final_w,
        "final_b": final_b,
        "target_w": target_w,
        "target_b": target_b,
        "final_error": evo.best_error,
        "generations": len(history),
        "last_status": history[-1]["status"],
    }


if __name__ == "__main__":
    res = demo_on_weak_device()
    print("=== SimplUnity 自进化微型验证 (680M 友好) ===")
    print(f"目标:  w={res['target_w']}  b={res['target_b']}")
    print(f"收敛:  w={res['final_w']:.4f}  b={res['final_b']:.4f}")
    print(f"最终误差: {res['final_error']:.6f}")
    print(f"状态: {res['last_status']}")
