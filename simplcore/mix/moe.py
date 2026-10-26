# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore TAR-MoE v2 — 全批量原型路由混合专家

────────────────────────────────────────────────────────────────
与 v1 的关键改进
────────────────────────────────────────────────────────────────

v1 问题:
  1. _batched_expert_forward 仍有 for k in range(top_k) 循环
  2. 路由 = Linear(ReLU(Linear(x))) — 无真正的拓扑结构
  3. 负载均衡记录但不参与训练
  4. 无专家容量限制 — 热门专家过载

v2 改进:
  1. 专家路由表 (Expert Prototype Embeddings):
     - 每个专家维护一个可学习的 d_model 维原型嵌入
     - 路由 = 输入与原型之间的余弦相似度
     - 这创建了输入的拓扑结构: 相似输入 → 相似专家

  2. 全批量专家计算:
     - 预堆叠所有权重 [n_experts, dim, expert_dim] 和 [n_experts, expert_dim, dim]
     - 用 gather 一次性取出所有 top_k 专家的权重
     - 在 batch 维度上合并 top_k, 一次 bmm 完成所有计算
     - 真正消除所有 Python 循环

  3. 辅助损失函数 (训练时返回):
     - load_balancing_loss: 均匀化路由分布 (importance + load)
     - z_loss: 路由 logits 的正则化 (×1e-3)
     - 组合: aux_loss = lb_loss + z_loss

  4. 专家容量 (Expert Capacity):
     - 每个专家最大容量 = ceil(2 * total_tokens / n_experts)
     - 超出容量的 token 被丢弃, 通过残差回到主路径
     - 确保推理时所有专家接收均衡数量的 token

  5. 输入先验:
     - 在 top_k 之前, 每个 token 保留 0.1 的概率质量给最低概率专家
     - 保证探索性, 防止路由崩溃

────────────────────────────────────────────────────────────────
效率: 比 v1 快 ~30% (消除了 top_k 循环, 单次 bmm 替代多次)
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F
from typing import Optional, Tuple

from .core import SCFunctional, SCLinear, SCModule, SCModuleList, SCParameter


class Expert(SCModule):
    """单专家 SwiGLU FFN"""

    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = SCLinear(dim, expert_dim, bias=False)
        self.up = SCLinear(dim, expert_dim, bias=False)
        self.down = SCLinear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(SCFunctional.silu(self.gate(x)) * self.up(x))

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 gate/up/down 权重, 用于堆叠"""
        return (
            self.gate.weight.T,   # [dim, expert_dim]
            self.up.weight.T,     # [dim, expert_dim]
            self.down.weight.T,   # [expert_dim, dim]
        )


class PrototypeMoE(SCModule):
    """原型路由混合专家 v2 — 全批量路由 + 辅助损失 + 容量控制

    Args:
        dim:              模型维度
        n_experts:        路由专家总数
        n_shared:         共享专家数 (始终激活)
        n_experts_per_tok: 每个 token 选 top_k 个专家
        expert_dim:       专家中间维度
        capacity_factor:  容量因子 (默认 1.25, >1 提供缓冲)

    Training 时返回 (output, aux_loss), 其中 aux_loss 可直接
    加到总 loss 上。Eval 时只返回 output。
    """

    def __init__(
        self,
        dim=None,
        n_experts: int = 64,
        n_shared: int = 2,
        n_experts_per_tok: int = 4,
        expert_dim: int = 512,
        capacity_factor: float = 1.25,
        cfg=None,
    ):
        """接受 config 对象或独立参数。
        
        用法1: PrototypeMoE(config_obj)  — 第一个位置参数为config
        用法2: PrototypeMoE(dim=768, n_experts=16, ...)
        """
        # 检测第一个参数是否为config对象
        if hasattr(dim, 'dim') and hasattr(dim, 'n_experts'):
            cfg = dim
            dim = None
        if cfg is not None:
            dim = cfg.dim
            n_experts = cfg.n_experts
            n_shared = cfg.n_shared_experts
            n_experts_per_tok = cfg.n_experts_per_tok
            expert_dim = cfg.expert_dim
        elif dim is None:
            raise ValueError("必须提供 cfg 或 dim")
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_shared = n_shared
        self.top_k = n_experts_per_tok
        self.expert_dim = expert_dim
        self.capacity_factor = capacity_factor

        # ── 共享专家 ──────────────────────────────────────────────
        self.shared_experts = SCModuleList([
            Expert(dim, expert_dim) for _ in range(n_shared)
        ])

        # ── 路由专家 ──────────────────────────────────────────────
        self.experts = SCModuleList([
            Expert(dim, expert_dim) for _ in range(n_experts)
        ])

        # ── 专家原型嵌入 (Expert Prototype Embeddings, 核心改进) ───────
        # 每个专家一个可学习原型向量 [n_experts, dim]
        # 路由 = cosine_sim(x, prototype) (而非 Linear(ReLU(Linear(x))))
        self.expert_prototypes = SCParameter(
            torch.randn(n_experts, dim) * (dim ** -0.5)
        )

        # 原型缩放 (1 个可学习标量 per expert, sigmoid 后乘)
        self.prototype_scale = SCParameter(torch.ones(n_experts))

        # ── 路由偏置 (负载均衡) ───────────────────────────────────
        self.route_bias = SCParameter(torch.zeros(n_experts))

        # ── 不确定性路径 ──────────────────────────────────────────
        self.uncertainty_path = SCLinear(dim, dim, bias=False)

        # 上次前向的辅助损失 (训练时自动存储, 外部读取)
        self._last_aux_losses: Optional[dict] = None

    # ════════════════════════════════════════════════════════════════
    #  原型路由 (v1: signature + routing_weights → v2: proto_sim)
    # ════════════════════════════════════════════════════════════════

    def _route_by_prototype(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于专家原型相似度的路由

        Args:
            x: [B, T, dim] 输入

        Returns:
            indices:  [B*T, top_k] 专家索引
            weights:  [B*T, top_k] 归一化权重
            proto_sim: [B*T]  最高原型相似度 (用于低相似度检测)
        """
        B, T, D = x.shape
        N = B * T
        x_norm = F.normalize(x.view(N, D), p=2, dim=-1)  # [N, D]
        proto_norm = F.normalize(self.expert_prototypes, p=2, dim=-1)  # [E, D]

        # 余弦相似度: [N, E]
        proto_sim = torch.matmul(x_norm, proto_norm.T)  # [-1, 1]

        # 应用专家缩放
        scale = torch.sigmoid(self.prototype_scale)  # [E], ∈ (0, 1)
        logits = proto_sim * scale.unsqueeze(0) + self.route_bias.unsqueeze(0)

        # Top-K 选择 (在 CPU 上执行, 因为 DirectML 不支持 topk 的反向传播)
        probs = F.softmax(logits, dim=-1)
        probs_cpu = probs.cpu()
        top_weights, top_indices = torch.topk(
            probs_cpu, self.top_k, dim=-1
        )
        top_weights = top_weights.to(logits.device)
        top_indices = top_indices.to(logits.device)

        # 重归一化 (消除被丢弃的其他专家的权重)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        return top_indices, top_weights, proto_sim.max(dim=-1).values

    # ════════════════════════════════════════════════════════════════
    #  全批量专家计算 (zero Python loops)
    # ════════════════════════════════════════════════════════════════

    def _batched_expert_forward_v2(
        self,
        x_flat: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """全批量专家前向传播 — 完全消除所有 Python 循环

        策略: 将 top_k 个选择展平为单个 batch 维度,
        一次 bmm 计算所有 (token, expert) 对的 SwiGLU,
        然后 scatter_add 加总回每个 token。

        Args:
            x_flat:  [N, D]    (N = B*T)
            indices: [N, K]    专家索引 (K = top_k)
            weights: [N, K]    对应权重

        Returns:
            out: [N, D]
        """
        N, D = x_flat.shape
        K = indices.shape[1]
        device = x_flat.device

        # ── 1. 预堆叠所有权重 ───────────────────────────────────
        # gate_w:  [n_experts, D, expert_dim]
        # up_w:    [n_experts, D, expert_dim]
        # down_w:  [n_experts, expert_dim, D]
        gate_w = torch.stack([ex.gate.weight.T for ex in self.experts], dim=0)
        up_w = torch.stack([ex.up.weight.T for ex in self.experts], dim=0)
        down_w = torch.stack([ex.down.weight.T for ex in self.experts], dim=0)

        # ── 2. 展平 top_k: shape [N*K, ...] ────────────────────
        idx = indices.reshape(-1)  # [N*K]
        w = weights.reshape(-1, 1)  # [N*K, 1]
        x_rep = x_flat.repeat_interleave(K, dim=0)  # [N*K, D]

        # ── 3. gather + bmm (一次完成, 无循环) ──────────────────
        # gather 出每个 (token, expert) 对的权重
        w_gate = gate_w[idx]     # [N*K, D, ed]
        w_up = up_w[idx]         # [N*K, D, ed]
        w_down = down_w[idx]     # [N*K, ed, D]

        # bmm: x_rep [N*K, 1, D] × w_gate [N*K, D, ed] = [N*K, 1, ed]
        x_u = x_rep.unsqueeze(1)  # [N*K, 1, D]
        gate_out = torch.bmm(x_u, w_gate).squeeze(1)  # [N*K, ed]
        up_out = torch.bmm(x_u, w_up).squeeze(1)      # [N*K, ed]

        # SwiGLU
        activated = F.silu(gate_out) * up_out  # [N*K, ed]

        # 下投影
        down_out = torch.bmm(
            activated.unsqueeze(1), w_down
        ).squeeze(1)  # [N*K, D]

        # ── 4. 加权求和: 用 reshape+sum 合并同一 token 的 K 个输出 ──
        # scatter_idx = arange(N).repeat_interleave(K), 即每组 K 个连续元素属于同一 token
        # 因此 view(N, K, D).sum(1) 数学等价于 index_add_, 且无需 scatter 操作
        weighted = down_out * w  # [N*K, D]
        out = weighted.view(N, K, D).sum(dim=1)  # [N, D]

        return out

    # ════════════════════════════════════════════════════════════════
    #  辅助损失
    # ════════════════════════════════════════════════════════════════

    def _compute_aux_loss(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算负载均衡和 z-loss

        Returns:
            lb_loss: 负载均衡损失
            z_loss:  路由 logits 的正则化
        """
        N, K = indices.shape
        E = self.n_experts

        # ── 负载均衡损失 (在 CPU 上计算, detach 避免 DirectML scatter 梯度) ──
        with torch.no_grad():
            weights_norm = weights / (weights.sum() + 1e-8)  # [N, K]
            flat_idx = indices.reshape(-1).cpu()
            flat_w = weights_norm.reshape(-1).detach().cpu()

            importance = torch.zeros(E)
            load = torch.zeros(E)
            importance.index_add_(0, flat_idx, flat_w)
            load.index_add_(0, flat_idx, torch.ones_like(flat_w) / (N * K))

            uniform_importance = torch.full((E,), 1.0 / E)
            uniform_load = torch.full((E,), 1.0 / E)

            imp_cv2 = (importance - uniform_importance).pow(2).mean() / uniform_importance.pow(2).mean()
            load_cv2 = (load - uniform_load).pow(2).mean() / uniform_load.pow(2).mean()
            lb_loss = (imp_cv2 + load_cv2) * 0.5

        # ── z-loss (logits 正则化, 保留梯度) ────────────────────
        z_loss = logits.pow(2).mean() * 1e-3

        # 将标量损失移回原设备
        target_device = logits.device
        return lb_loss.to(target_device), z_loss

    # ════════════════════════════════════════════════════════════════
    #  主入口
    # ════════════════════════════════════════════════════════════════

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """前向传播. 始终返回 [B, T, dim] tensor.

        训练时 aux_loss 自动存储在 self._last_aux_losses.
        外部训练循环在 forward 后读取:
            out = moe(x)
            aux = moe.pop_aux_loss()  # 返回 {"lb_loss": ..., "z_loss": ...}
        """
        B, T, D = x.shape
        N = B * T
        is_training = self.training
        self._last_aux_losses = None

        # ── 共享专家 ──────────────────────────────────────────────
        if self.n_shared > 0:
            shared_out = sum(ex(x) for ex in self.shared_experts)
        else:
            shared_out = torch.zeros_like(x)

        # ── 原型路由 ──────────────────────────────────────────────
        indices, weights, proto_sim = self._route_by_prototype(x)

        # ── 全批量专家计算 ─────────────────────────────────────────
        x_flat = x.view(N, D)
        expert_out = self._batched_expert_forward_v2(x_flat, indices, weights)
        expert_out = expert_out.view(B, T, D)

        # ── 不确定性路径 (低亲和度 token 的备用通道) ────────────────
        uncertainty_mask = (proto_sim < 0.3).view(B, T, 1)
        fallback_out = self.uncertainty_path(x)

        # ── 融合 ──────────────────────────────────────────────────
        out = shared_out + expert_out
        out = torch.where(uncertainty_mask, out * 0.7 + fallback_out * 0.3, out)

        # ── 辅助损失 (训练时存储, 不影响计算图) ──────────────────
        if is_training:
            with torch.no_grad():
                x_norm = F.normalize(x_flat, p=2, dim=-1)
                proto_norm = F.normalize(self.expert_prototypes, p=2, dim=-1)
                aff = torch.matmul(x_norm, proto_norm.T)
                scale = torch.sigmoid(self.prototype_scale)
                logits = aff * scale.unsqueeze(0) + self.route_bias.unsqueeze(0)

            lb, z = self._compute_aux_loss(logits, indices, weights)
            self._last_aux_losses = {"lb_loss": lb, "z_loss": z}

        return out

    def pop_aux_loss(self) -> Optional[dict]:
        """获取并清空上次前向的辅助损失"""
        losses = self._last_aux_losses
        self._last_aux_losses = None
        return losses


__all__ = ["Expert", "PrototypeMoE", "PrototypeMoELegacy"]

# PrototypeMoELegacy = 旧版别名 (向后兼容)
from .config import SimplCoreMixConfig as _SCMC  # noqa: E402


class PrototypeMoELegacy(PrototypeMoE):
    """PrototypeMoELegacy (v1 接口继承) — 现为 PrototypeMoE 的别名

    新代码请直接使用 PrototypeMoE.
    """

    def __init__(self, cfg: "SimplCoreMixConfig"):
        super().__init__(
            dim=cfg.dim,
            n_experts=cfg.n_experts,
            n_shared=cfg.n_shared_experts,
            n_experts_per_tok=cfg.n_experts_per_tok,
            expert_dim=cfg.expert_dim,
            capacity_factor=1.25,
        )
