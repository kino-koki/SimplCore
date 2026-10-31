# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""世界模型容量探针（0.1-α）—— 测量，不宣称.

回答的问题：**O(1) 流式状态里到底编码了多少"可预测未来"的信息？**

「世界模型」若只写在白皮书 §3.5 里就只是一段话（``claims_audit`` 的同一
立场）。本模块把它变成一个**可证伪的测量**：

1. 流式跑若干序列，在每个位置 t 抽取 O(1) 递推状态（末层 IIR 的 ``y1``，
   真实常驻状态，非隐藏层输出）；
2. 冻结状态，拟合一个**线性读出头**（fp64 闭式岭回归，无隐藏层——
   只测状态里**线性可读出**的信息），预测 t+Δ 处 token 的桶类别；
3. 对照基线三件套：**行置换基线**（破坏 (状态, 目标) 对应、保留边缘分布
   ——掉点说明信息真在状态里而非读出头过拟合）、**多数类基线**、
   **样本量账本**。

输出 ``lift = acc − shuffle_acc``：> 0 且显著才支持「状态携带可预测未来」
的弱主张；接近 0 则说明该任务上状态不含可读出的未来信息——**负结果照实
记账**（这正是与 retrieval_bound 相同的测量纪律）。

诚实边界
--------
* 测的是**线性可读出**：非线性解码器能读出的信息不计入（下界，不是上界）；
* 桶类别（``id % n_buckets``）是合成任务——真实世界任务的探针（因果
  物理量、对话状态）需要对应数据集，本模块只提供机制与口径；
* 未在预训练模型上实测：架构随机权重下的 lift **不构成任何能力宣称**。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..model.config import SimplCoreConfig
from ..model.model import SimplCoreModel

__all__ = ["WorldModelProbe", "WorldProbeReport"]


@dataclass
class WorldProbeReport:
    """探针测量账本（JSON 可序列化）。"""
    horizon: int
    n_buckets: int
    n_train: int
    n_test: int
    feature_dim: int
    readout_accuracy: float
    shuffle_accuracy: float
    majority_baseline: float
    lift: float
    ridge_lambda: float
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon, "n_buckets": self.n_buckets,
            "n_train": self.n_train, "n_test": self.n_test,
            "feature_dim": self.feature_dim,
            "readout_accuracy": round(self.readout_accuracy, 6),
            "shuffle_accuracy": round(self.shuffle_accuracy, 6),
            "majority_baseline": round(self.majority_baseline, 6),
            "lift": round(self.lift, 6),
            "ridge_lambda": self.ridge_lambda,
            "notes": list(self.notes),
            "claim": ("仅当 lift>0 且显著时支持「状态线性携带可预测未来」"
                      "的弱主张；本测量不构成「世界模型已实现」宣称"),
        }


class WorldModelProbe:
    """O(1) 状态 → t+Δ 未来信息的线性可读出测量（fp64 闭式岭回归）。

    Args:
        model: 冻结使用（探针不训练模型本身）。
        horizon: 预测多远的未来（Δ，token 数）。
        n_buckets: 目标桶数（``bucket = token_id % n_buckets``）。
        ridge_lambda: 岭回归正则（fp64 闭式解）。
        layer: 取哪一层的递推状态（默认末层）。
    """

    def __init__(self, model: SimplCoreModel, horizon: int = 1,
                 n_buckets: int = 8, ridge_lambda: float = 1e-2,
                 layer: int = -1) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if n_buckets <= 1:
            raise ValueError("n_buckets must be > 1")
        self.model = model
        self.horizon = int(horizon)
        self.n_buckets = int(n_buckets)
        self.ridge_lambda = float(ridge_lambda)
        self.layer = int(layer)
        self.was_training = model.training
        model.eval()

    # -- 状态抽取 ---------------------------------------------------------------
    @torch.no_grad()
    def collect(self, sequences: Sequence[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """流式跑序列，收集 (状态特征, t+Δ 桶标签) 对.

        特征 = 末层 IIR 的 ``y1``（真 O(1) 常驻状态，拉平）；每步一个样本，
        目标 = t+Δ 处 token 的桶（序列末端 Δ 步无目标，丢弃）。
        """
        device = next(self.model.parameters()).device
        dtype = self.model.tok_emb.weight.dtype
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for seq in sequences:
            seq = seq.reshape(1, -1).to(device)
            state = self.model.init_stream(1, device=device, dtype=dtype)
            L = seq.shape[1]
            states: List[torch.Tensor] = []
            for t in range(L):
                _, state = self.model.stream_step(seq[:, t], state)
                layer = state.layers[self.layer]
                y1 = layer.y1 if hasattr(layer, "y1") else layer[0]
                states.append(y1.reshape(-1).to("cpu", torch.float64))
            states = torch.stack(states)                     # (L, F)
            lab = (seq[0] % self.n_buckets).to(torch.int64).cpu()
            # 样本 t 的目标 = t+Δ 的桶；末端 Δ 步丢弃
            feats.append(states[:L - self.horizon])
            labels.append(lab[self.horizon:])
        return torch.cat(feats), torch.cat(labels)

    # -- 闭式岭回归 --------------------------------------------------------------
    @staticmethod
    def _ridge_fit(X: torch.Tensor, Y: torch.Tensor,
                   lam: float) -> torch.Tensor:
        """fp64 闭式解 W = (XᵀX + λI)⁻¹ XᵀY（X 已含偏置列由调用方处理）。"""
        d = X.shape[1]
        A = X.T @ X + lam * torch.eye(d, dtype=torch.float64)
        return torch.linalg.solve(A, X.T @ Y)

    def fit_and_eval(self, train_seqs: Sequence[torch.Tensor],
                     test_seqs: Sequence[torch.Tensor],
                     max_train_samples: int = 20000,
                     seed: int = 0) -> WorldProbeReport:
        """收集 → 闭式拟合 → 三基线对照评估，返回账本."""
        Xtr, ytr = self.collect(train_seqs)
        Xte, yte = self.collect(test_seqs)
        notes: List[str] = []
        if len(Xtr) > max_train_samples:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(Xtr), generator=g)[:max_train_samples]
            Xtr, ytr = Xtr[idx], ytr[idx]
            notes.append(f"训练样本按 {max_train_samples} 截断")
        if len(Xtr) < 10 or len(Xte) < 10:
            raise ValueError(
                f"样本过少（train={len(Xtr)}, test={len(Xte)}）："
                f"序列太短或 horizon 过大")

        # 特征标准化 + 偏置列（岭回归口径）
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-8)
        def _prep(X: torch.Tensor, stats: bool) -> torch.Tensor:
            z = (X - mu) / sd
            return torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], dim=1)
        Xtr_p, Xte_p = _prep(Xtr, True), _prep(Xte, False)
        Ytr = torch.nn.functional.one_hot(
            ytr, self.n_buckets).to(torch.float64)

        W = self._ridge_fit(Xtr_p, Ytr, self.ridge_lambda)
        acc = float((torch.argmax(Xte_p @ W, dim=1) == yte)
                    .to(torch.float64).mean())

        # 行置换基线：破坏 (状态, 目标) 对应、保留边缘分布
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(Xte_p), generator=g)
        acc_shuffle = float((torch.argmax(Xte_p[perm] @ W, dim=1) == yte)
                            .to(torch.float64).mean())
        # 多数类基线
        counts = torch.bincount(yte, minlength=self.n_buckets).to(torch.float64)
        acc_majority = float(counts.max() / counts.sum())

        rep = WorldProbeReport(
            horizon=self.horizon, n_buckets=self.n_buckets,
            n_train=len(Xtr), n_test=len(Xte), feature_dim=Xtr.shape[1],
            readout_accuracy=acc, shuffle_accuracy=acc_shuffle,
            majority_baseline=acc_majority,
            lift=acc - acc_shuffle, ridge_lambda=self.ridge_lambda,
            notes=notes)
        if rep.lift <= 0:
            rep.notes.append(
                "lift≤0：该任务上线性读出未超过置换基线 —— 负结果照实记账，"
                "不构成「状态携带可预测未来」的证据")
        return rep


def main(argv: Optional[List[str]] = None) -> int:
    """合成周期序列冒烟：确定性输入下状态必然携带相位信息（机制自检）。"""
    torch.manual_seed(0)
    model = SimplCoreModel(SimplCoreConfig(vocab_size=256, dim=16,
                                           num_layers=1, num_bases=8,
                                           ffn_ratio=2.0, decay_max=256.0,
                                           local_kernel=2))
    period = 16
    seqs = [torch.arange(period * 4) % 256 for _ in range(4)]
    test = [torch.arange(period * 4) % 256 for _ in range(2)]
    probe = WorldModelProbe(model, horizon=1, n_buckets=8)
    rep = probe.fit_and_eval(seqs, test)
    for k, v in rep.as_dict().items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())