# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
[SimplCore 核心创新] SGI — 结构化门控干预

─────────────────────────────────────────────────
三大创新机制
─────────────────────────────────────────────────

1. 分组有向注意力 (Grouped Directed Attention, GDA)
   - 将隐藏维度划分为分组变量组 (n_groups 个分组变量)
   - 通过双线性「因果注意力」机制动态发现组间因果关系
   - 上三角掩码 + 谱正则化保证有向无环性
   - 有向权重图随输入内容变化 (输入依赖)

2. 门控父节点替换 (Gated Parent Replacement, GPR)
   - 在表征空间近似 Pearl do-算子: do(X=x)
   - 每组从其前置分组预测"自然"值, 替换原值
   - 干预强度为输入依赖的可学习门控

3. 上下文残差减除 (Context Residual Subtraction, CRS)
   - 从序列上下文聚合全局上下文残差
   - 近似后门公式: 在表征空间减去混杂分量
   - 对混杂影响较大的分组变量更强调整

─────────────────────────────────────────────────
理论框架
─────────────────────────────────────────────────

在 Pearl 因果层次中:
  P(Y|X)       — 相关性 (association)
  P(Y|do(X=x)) — 干预 (intervention)
  P(Y|X=x', X=x)  — 反事实 (counterfactual)

标准 Transformer 学习 P(Y|X), 受上下文残差和虚假相关影响。
SGI 在表征空间实现 do-算子的可微近似:

  do(X=x) ≈ (1 − λ)·x + λ·f(Pa(X))

其中:
  Pa(X) — X 的前置分组
  f     — 从父节点到 X 的替换函数
  λ     — 输入依赖的干预强度, λ ∈ (0,1)
  λ=0   — 保持原值 (无干预)
  λ=1   — 完全由替换函数决定 (强干预)

─────────────────────────────────────────────────
效率分析
─────────────────────────────────────────────────

干预空间维度 intervention_dim 约为主维度 dim 的 1/4,
分组变量数 n_groups = 8 (小常数), 每组维度 group_dim = intervention_dim/8.

计算复杂度:
  - 干预投影:    O(dim · intervention_dim)
  - 有向权重图:      O(T · n_groups² · group_dim) = O(T · intervention_dim · n_groups)
  - 干预 :       O(intervention_dim · n_groups)
  - 混杂调整:    O(T · intervention_dim²)
  - 反向投影:    O(intervention_dim · dim)

总复杂度约为 O(dim · intervention_dim + T · intervention_dim²), 约为自注意力的 1/4。
"""
import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F
from typing import Optional, Tuple

from .core import SCLinear, SCModule, SCParameter, SCFunctional, SCRMSNorm


class StructuredGatedIntervention(SCModule):
    """结构化门控干预 (SGI)

    在表征空间实现可微的因果干预, 去除上下文残差, 增强模型鲁棒性。

    Args:
        dim:               输入/输出维度
        n_groups:          分组变量数量 (默认 8)
        top_k_parents:     每个变量最多保留的父节点数 (默认 3)
        scale:             残差缩放系数 (默认 0.1)
        use_confounder:    是否启用上下文残差减除 (默认 True)
    """

    def __init__(
        self,
        dim: int,
        n_groups: int = 8,
        top_k_parents: int = 3,
        scale: float = 0.1,
        use_confounder: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k_parents
        self.scale = scale
        self.use_confounder = use_confounder

        # ── 干预空间维度 ───────────────────────────────────────────
        # intervention_dim ≈ dim/4 (不小于 64, 不大於 dim)
        self.intervention_dim = max(min(dim // 4, dim), 64) if dim > 128 else dim
        # 调整 n_groups 使 group_dim >= 2
        while self.intervention_dim // max(n_groups, 1) < 2:
            n_groups = max(n_groups // 2, 2)
        self.n_groups = max(n_groups, 2)
        self.group_dim = self.intervention_dim // self.n_groups

        # ── 1. 干预投影: dim ↔ intervention_dim ──────────────────────────
        self.to_intervention = SCLinear(dim, self.intervention_dim, bias=False)
        self.from_intervention = SCLinear(self.intervention_dim, dim, bias=False)
        self.intervention_norm = SCRMSNorm(self.intervention_dim)

        # 组嵌入 — 每个分组变量一个可学习偏置
        self.group_embed = SCParameter(
            torch.randn(1, 1, self.n_groups, self.group_dim) * 0.02
        )

        # ── 2. 动态有向权重图 (DCGD) ───────────────────────────────────
        # 双线性因果注意力: A[i,j] = σ(q(g_i)·W·k(g_j))
        self.directed_q = SCLinear(self.group_dim, self.group_dim, bias=False)
        self.directed_k = SCLinear(self.group_dim, self.group_dim, bias=False)
        self.directed_bias = SCParameter(torch.zeros(1, 1, self.n_groups, self.n_groups))

        # 上三角掩码 — 保证无环性 (i < j = 允许, i ≥ j = 阻断)
        # 使用有限大负数避免 -inf × 0 = NaN (IEEE 754)
        _NEG = -1e4
        triu_mask = torch.zeros(self.n_groups, self.n_groups)
        tril_mask = torch.tril(torch.ones(self.n_groups, self.n_groups), diagonal=0)
        triu_mask = torch.where(tril_mask.bool(), _NEG, triu_mask)  # lower+diag = 阻断
        triu_mask = triu_mask + torch.triu(
            torch.ones(self.n_groups, self.n_groups), diagonal=1
        )  # 上三角偏置 +1, 允许且增强梯度
        self.register_buffer("triu_mask", triu_mask)

        # 谱正则化投影 (预留, 后续可引入 DAG 谱惩罚)
        # self.spectral_proj = SCLinear(self.group_dim, self.group_dim, bias=False)

        # ── 3. 干预 (DDI) ───────────────────────────────────────────
        # 父节点到子节点的替换函数
        self.parent_proj = SCLinear(self.group_dim, self.group_dim, bias=False)
        # 干预强度门控: λ_i = σ(MLP(g_i))
        self.replacement_gate = SCLinear(self.group_dim, 1, bias=True)

        # ── 4. 上下文残差减除 (BCA) ──────────────────────────────────
        if use_confounder:
            # 从上下文估计上下文残差
            self.ctx_q = SCLinear(self.intervention_dim, self.intervention_dim, bias=False)
            self.ctx_k = SCLinear(self.intervention_dim, self.intervention_dim, bias=False)
            self.ctx_v = SCLinear(self.intervention_dim, self.n_groups * self.group_dim, bias=False)
            self.ctx_gate = SCLinear(self.n_groups * self.group_dim, 1, bias=True)
            self.ctx_norm = SCRMSNorm(self.n_groups * self.group_dim)

    # =================================================================
    #  内部方法
    # =================================================================

    def _compute_directed_attention(
        self, groups: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DCGD: 分组有向注意力

        对每组分组变量计算它与其他组的因果影响强度,
        上三角掩码保证无环, Top-K稀疏化保证效率。

        Args:
            groups: [B, T, n_g, g_dim] 分组表征

        Returns:
            directed_weights: [B, T, n_g, n_g] 有向权重矩阵 (上三角, 仅 i < j)
            sparsity:   稀疏度 (标量, 用于日志)
        """
        B, T, ng, gd = groups.shape

        # 双线性因果注意力
        q = self.directed_q(groups)  # [B, T, ng, gd]
        k = self.directed_k(groups)  # [B, T, ng, gd]

        # A[i,j] = q(g_i)·k(g_j) / sqrt(gd)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (gd ** -0.5)  # [B, T, ng, ng]
        attn = attn + self.directed_bias + self.triu_mask  # 上三角偏置

        # Top-K 稀疏化: 每组只保留 top_k 个最强父节点
        if self.top_k > 0 and self.top_k < ng:
            # attn[b, t, j, :]: 对组 j, 各候选父节点的因果强度
            k = min(self.top_k, ng - 1)  # 最多 ng-1 个父节点
            values = torch.topk(attn, k, dim=-1, largest=True).values  # [B,T,ng,k]
            threshold = values[..., -1:]  # [B,T,ng,1] 第k大的值作为阈值
            attn = attn * (attn >= threshold).float()

        directed_weights = F.sigmoid(attn)  # [B, T, ng, ng]
        sparsity = (directed_weights > 0.5).float().mean().item()

        return directed_weights, sparsity

    def _gated_parent_replacement(
        self, groups: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """DDI: 门控父节点替换

        对每组分组变量:
          1. 从父节点预测: f(Pa_i) = Σ_j adj[j,i] · proj(g_j)
          2. 干预: g_i' = (1-λ_i)·g_i + λ_i·f(Pa_i)

        Args:
            groups: [B, T, n_g, g_dim]  原始分组
            adj:    [B, T, n_g, n_g]    因果邻接 (上三角)

        Returns:
            intervened: [B, T, n_g, g_dim]
        """
        # 父节点贡献: adj^T @ parent_proj(groups)
        # 因为 adj[i,j] 表示 i → j, 所以对 j 的父贡献是 Σ_i adj[i,j] · proj(g_i)
        parent_input = self.parent_proj(groups)  # [B, T, ng, gd]
        # matmul over group dim: [B, T, ng, gd] = [B, T, ng, ng] × [B, T, ng, gd]
        parent_contrib = torch.matmul(adj.transpose(-2, -1), parent_input)

        # 干预强度: λ_i = σ(MLP(g_i))
        lam = torch.sigmoid(self.replacement_gate(groups))  # [B, T, ng, 1]

        # 软干预: g' = (1 - λ) * g + λ * parent_pred
        intervened = (1 - lam) * groups + lam * parent_contrib

        return intervened

    def _context_residual_subtraction(
        self, groups: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """BCA: 上下文残差减除

        从上下文序列中提取全局上下文残差, 对每个分组变量的
        表征减去其混杂分量。

        Args:
            groups:  [B, T, n_g, g_dim] 干预后的分组
            context: [B, T, dim]        序列上下文 (可选)

        Returns:
            adjusted: [B, T, n_g * g_dim] 调整后的拼接表征
        """
        if not self.use_confounder:
            return groups.reshape(groups.shape[0], groups.shape[1], -1)

        B, T, ng, gd = groups.shape
        intervention_h = groups.reshape(B, T, -1)  # [B, T, n_g * g_dim]

        # 从上下文估计上下文残差: 用注意力池化聚合序列
        ctx_proj = self.intervention_norm(self.to_intervention(context))  # [B, T, intervention_dim]
        ctx_q = self.ctx_q(ctx_proj)  # [B, T, intervention_dim]
        ctx_k = self.ctx_k(ctx_proj)  # [B, T, intervention_dim]

        # 自注意力: 每个 token 的混杂分量 = 加权聚合所有 token
        ctx_attn = torch.matmul(ctx_q, ctx_k.transpose(-2, -1)) * (
            self.intervention_dim ** -0.5
        )
        # 因果掩码 (2026-10 审计修复): 位置 t 只能聚合 s <= t 的上下文,
        # 否则全行 softmax 会把未来 token 的混杂信号泄漏进过去位置, 破坏
        # 自回归 LM 的因果性 (与 HONEST_STATUS 第 0.2 节一致)。
        # 有限大负数 -1e4 避免 -inf * 0 = NaN (IEEE 754), 与 DCGD triu_mask 同策略。
        if T > 1:
            _causal = torch.full((T, T), -1e4, device=ctx_attn.device)
            _causal = torch.triu(_causal, diagonal=1)
            ctx_attn = ctx_attn + _causal.unsqueeze(0).unsqueeze(0)
        ctx_attn = F.softmax(ctx_attn, dim=-1)  # [B, T, T]

        # 混杂值: [B, T, n_g * g_dim]
        ctx_values = self.ctx_v(ctx_proj)  # [B, T, intervention_dim → n_g*g_dim]
        context_residual = torch.matmul(
            ctx_attn, ctx_values
        )  # [B, T, n_g * g_dim]

        # 门控: 每个位置的混杂调整强度
        gamma = torch.sigmoid(self.ctx_gate(context_residual))  # [B, T, 1]

        # 调整: y' = y - γ · confounder
        adjusted = intervention_h - gamma * context_residual
        adjusted = self.ctx_norm(adjusted)

        return adjusted

    # =================================================================
    #  主入口
    # =================================================================

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """执行因果干预与调整

        Args:
            x:       [B, T, dim] 隐藏状态
            context: [B, T, dim] 序列上下文 (默认使用 x 自身)

        Returns:
            y:       [B, T, dim] 因果干预后的输出
        """
        B, T, D = x.shape
        ctx = context if context is not None else x

        # ── 步骤 1: 投影到干预空间 ────────────────────────────────
        intervention_h = self.intervention_norm(self.to_intervention(x))  # [B, T, intervention_dim]
        # 重塑为分组: [B, T, n_g, g_dim] + 组嵌入
        groups = (
            intervention_h.reshape(B, T, self.n_groups, self.group_dim)
            + self.group_embed
        )

        # ── 步骤 2: 分组有向注意力 ─────────────────────────────────
        directed_weights, sparsity = self._compute_directed_attention(groups)

        # ── 步骤 3: 可微干预 ───────────────────────────────────────
        intervened = self._gated_parent_replacement(groups, directed_weights)

        # ── 步骤 4: 混杂调整 ───────────────────────────────────────
        adjusted = self._context_residual_subtraction(intervened, ctx)

        # ── 步骤 5: 投影回原空间 ───────────────────────────────────
        # adjusted: [B, T, n_g * g_dim] → [B, T, intervention_dim] → [B, T, dim]
        causal_out = adjusted.reshape(B, T, self.intervention_dim)
        out = self.from_intervention(causal_out)

        # 残差连接 (缩放保持稳定)
        return x + out * self.scale

    # =================================================================
    #  辅助方法
    # =================================================================

    def get_directed_weights(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """获取当前输入的有向权重图 (用于分析与可视化)

        Args:
            x: [B, T, dim] 输入

        Returns:
            adj: [n_groups, n_groups] 平均有向权重矩阵
        """
        self.eval()
        with torch.no_grad():
            intervention_h = self.intervention_norm(self.to_intervention(x))
            groups = intervention_h.reshape(
                intervention_h.shape[0], intervention_h.shape[1], self.n_groups, self.group_dim
            ) + self.group_embed
            adj, _ = self._compute_directed_attention(groups)
        return adj.mean(dim=(0, 1))  # [n_g, n_g]

    def get_gate_values(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """获取每组干预强度 λ (用于分析)

        Returns:
            lam: [n_groups] 平均干预强度
        """
        self.eval()
        with torch.no_grad():
            intervention_h = self.intervention_norm(self.to_intervention(x))
            groups = intervention_h.reshape(
                intervention_h.shape[0], intervention_h.shape[1], self.n_groups, self.group_dim
            ) + self.group_embed
            lam = torch.sigmoid(self.replacement_gate(groups))
        return lam.mean(dim=(0, 1, 2))  # [n_groups]


__all__ = ["StructuredGatedIntervention"]
