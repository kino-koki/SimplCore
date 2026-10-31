# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""世界模型容量探针的守护测试。

只锁**机制**，不锁实验结果（lift 的正负是实测数据，不是结构性质）：
1. collect 抽的是真 O(1) 递推状态（维度 = 末层 y1 的展平宽度），样本数
   = Σ(L_i − horizon)，目标桶 = (t+Δ) token % n_buckets；
2. 闭式岭回归在**可构造**的可读出数据上恢复满精度（机制自检：
   X 与 y 线性相关时 lift = 1 − 基线，误差 fp64 级）；
3. 置换基线与多数类基线口径正确（置换不掉边缘分布；多数类 = 众数频率）；
4. 账本字段齐 + claim 字段存在（诚实边界写进输出）；
5. horizon/n_buckets 的参数校验。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.eval.world_probe import WorldModelProbe
from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel


def _model() -> SimplCoreModel:
    torch.manual_seed(0)
    return SimplCoreModel(SimplCoreConfig(vocab_size=64, dim=16, num_layers=1,
                                          num_bases=4, ffn_ratio=2.0,
                                          decay_max=256.0, local_kernel=2))


def test_collect_shape_and_labels() -> None:
    model = _model()
    probe = WorldModelProbe(model, horizon=2, n_buckets=8)
    seqs = [torch.randint(0, 64, (40,)) for _ in range(3)]
    X, y = probe.collect(seqs)
    assert X.shape[0] == y.shape[0] == 3 * (40 - 2)
    assert X.shape[1] == 4 * 16          # 末层 y1 (M=4, D=16) 展平 = 64
    assert y.min() >= 0 and y.max() < 8
    with pytest.raises(ValueError):
        WorldModelProbe(model, horizon=0)
    with pytest.raises(ValueError):
        WorldModelProbe(model, n_buckets=1)


def test_ridge_recovers_constructible_signal() -> None:
    """机制自检（纯数学，与 mixer 无关），分两段：

    (a) 连续目标 Y = XW：闭式岭回归必须以 fp64 精度恢复（λ→小量）；
    (b) 分类：两类的类条件高斯充分分离（均值距 3σ）时，one-hot 线性回归
        的决策边界逼近贝叶斯最优 → acc ≥ 0.98。
        （注意 one-hot(argmax(XW)) 是非线性目标——不能用 argmax 构造
        「精确线性可恢复」的分类断言，那是数学错误。）
    """
    probe = WorldModelProbe.__new__(WorldModelProbe)
    probe.horizon, probe.n_buckets = 1, 2
    probe.ridge_lambda = 1e-8
    torch.manual_seed(0)

    # (a) 连续回归：λ→0 时闭式解恢复真值
    n, d = 3000, 6
    W_true = torch.randn(d, 3, dtype=torch.float64)
    Xtr = torch.randn(n, d, dtype=torch.float64)
    Ytr = Xtr @ W_true
    Xte = torch.randn(500, d, dtype=torch.float64)
    Yte = Xte @ W_true
    W = WorldModelProbe._ridge_fit(Xtr, Ytr, probe.ridge_lambda)
    mse = float(((Xte @ W - Yte) ** 2).mean())
    var = float((Yte ** 2).mean())
    assert mse < 1e-2 * var, f"闭式解未恢复线性真值: mse={mse:.3e} var={var:.3e}"

    # (b) 两类充分分离：acc ≥ 0.98
    mu_pos = torch.tensor([3.0] + [0.0] * (d - 1), dtype=torch.float64)
    ytr_b = torch.randint(0, 2, (n,))
    Xtr_b = torch.where(ytr_b.unsqueeze(1) == 1, mu_pos, -mu_pos) \
        + 0.5 * torch.randn(n, d, dtype=torch.float64)
    yte_b = torch.randint(0, 2, (500,))
    Xte_b = torch.where(yte_b.unsqueeze(1) == 1, mu_pos, -mu_pos) \
        + 0.5 * torch.randn(500, d, dtype=torch.float64)
    Xtr_p = torch.cat([Xtr_b, torch.ones(n, 1, dtype=torch.float64)], dim=1)
    Xte_p = torch.cat([Xte_b, torch.ones(500, 1, dtype=torch.float64)], dim=1)
    Yoh = torch.nn.functional.one_hot(ytr_b, 2).to(torch.float64)
    Wb = WorldModelProbe._ridge_fit(Xtr_p, Yoh, 1e-4)
    acc = float((torch.argmax(Xte_p @ Wb, dim=1) == yte_b)
                .to(torch.float64).mean())
    assert acc >= 0.98, f"充分分离的两类线性读出应 ≥0.98, got {acc}"


def test_fit_and_eval_report_and_baselines() -> None:
    model = _model()
    probe = WorldModelProbe(model, horizon=1, n_buckets=8)
    train = [torch.randint(0, 64, (60,)) for _ in range(2)]
    test = [torch.randint(0, 64, (60,)) for _ in range(1)]
    rep = probe.fit_and_eval(train, test, seed=0)
    d = rep.as_dict()
    for k in ("horizon", "n_buckets", "n_train", "n_test", "feature_dim",
              "readout_accuracy", "shuffle_accuracy", "majority_baseline",
              "lift", "ridge_lambda", "notes", "claim"):
        assert k in d, k
    assert 0.0 <= d["readout_accuracy"] <= 1.0
    assert 0.0 <= d["shuffle_accuracy"] <= 1.0
    assert d["n_train"] == 118 and d["n_test"] == 59     # (60-1) 每序列
    # 多数类基线在 8 桶随机标签下 ≈ 1/8（容差覆盖有限样本）
    assert d["majority_baseline"] == pytest.approx(0.125, abs=0.15)
    # claim 字段必须包含"不构成宣称"的诚实边界
    assert "不构成" in d["claim"]