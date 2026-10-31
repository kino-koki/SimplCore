# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 路由门：token 级 top-k 软选择 + 负载均衡 / z-loss 辅助项。

路由是线性投影（可选一层隐藏），O(1) 状态、可流式，与 SGC 非注意力哲学一致。
前向返回：
* ``w``：稠密 (B, L, E) 权重矩阵（未选中专家处为 0，选中的 top-k 处为归一化权重）；
* ``idxs``：被选中的专家索引 (B, L, top_k)；
* ``aux``：``{"lb": 负载不均衡标量, "z": 路由 logit 平方均值}``，供模型按系数合成 aux loss。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, router_hidden: int = None,
                 top_k: int = 2, z_loss: float = 1e-3, lb_loss: float = 1e-2) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.z_loss = float(z_loss)
        self.lb_loss = float(lb_loss)
        # 路由温度：默认 1.0（逐位不变）。被生物计算主义耦合器动态调节，
        # 实现 meso/macro（连续场能量 / 课程进度）→ micro（离散路由）的跨尺度约束。
        self.temperature = 1.0
        if router_hidden and int(router_hidden) != dim:
            self.proj = nn.Sequential(
                nn.Linear(dim, int(router_hidden), bias=False),
                nn.Linear(int(router_hidden), self.num_experts, bias=False),
            )
        else:
            self.proj = nn.Linear(dim, self.num_experts, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, h: torch.Tensor, domain=None):
        """``h``: (B, L, D)。``domain``: None（MoE 路由）或 int / (B,) / (B,L) 长整型
        （强制把 token 路由到指定专家——满足「每个专家训练数据必须不同」的隔离训练）。"""
        logits = self.proj(h)  # (B, L, E)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if domain is not None:
            w, idxs = self._domain_weights(logits, domain)
            aux = self._aux(logits, w)
            return w, idxs, aux
        topk = torch.topk(logits, self.top_k, dim=-1)
        vals = topk.values
        idxs = topk.indices
        w = torch.zeros_like(logits)
        w.scatter_(-1, idxs, F.softmax(vals, dim=-1))
        aux = self._aux(logits, w)
        return w, idxs, aux

    def _domain_weights(self, logits: torch.Tensor, domain):
        w = torch.zeros_like(logits)
        if isinstance(domain, int):
            w[..., domain] = 1.0
            idxs = torch.full(logits.shape[:-1] + (1,),
                             int(domain), dtype=torch.long, device=logits.device)
        else:
            domain = domain.long()
            ones = torch.ones_like(domain, dtype=logits.dtype).unsqueeze(-1)
            w.scatter_(-1, domain.unsqueeze(-1), ones)
            idxs = domain.unsqueeze(-1)
        return w, idxs

    def _aux(self, logits: torch.Tensor, w: torch.Tensor):
        # 负载均衡：每个专家被激活的 token 占比（期望均匀）
        f = w.mean(dim=(0, 1))                       # (E,)
        lb = ((f - f.mean()) ** 2).mean()            # 不均衡度
        z = (logits ** 2).mean()                     # 防熵崩
        return {"lb": lb, "z": z}
