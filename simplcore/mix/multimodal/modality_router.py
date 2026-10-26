# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
多模态路由器 (ModalityRouter)
==============================
根据输入嵌入的类型 (模态) 将其路由到不同的专家通路,
并输出该输入的模态分类 logits 与路由融合后的特征。

支持模态: text / image / video / audio (共 ``router_num_modalities`` 种)。

工作流程:
  1. 对输入嵌入序列 [B, T, dim] 做均值池化得到句向量 [B, dim];
  2. 轻量级分类器判断输入模态, 输出 logits [B, num_modalities];
  3. 经 softmax 得到软路由权重 (可选可学习门控缩放);
  4. 每个模态对应一个可学习专家投影 (Linear), 对输入分别变换;
  5. 以路由权重对各专家输出做加权融合, 再与原输入残差相加并归一化;
  6. 返回 (modality_logits, routed_features)。

路由权重可学习:
  - 每个专家 (Linear) 的权重矩阵;
  - 每个模态一个可学习门控标量 ``router_gate`` (调节该模态路由强度);
  - 每个模态一个可学习类型嵌入 ``modality_embed``。

底层张量库为 ``simplcore.tensor`` (双后端: 自研 NumPy 引擎 / PyTorch),
完全脱离对 PyTorch 的硬依赖, 与 SimplCore Mix 其余模块保持一致。
"""
from __future__ import annotations

from typing import Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from .config import MultimodalConfig


# ============================================================
# 轻量级模态分类器
# ============================================================
class ModalityClassifier(nn.Module):
    """轻量级模态分类器。

    将句向量 [B, dim] 映射为模态 logits [B, num_modalities]。
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout。

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cfg.dim, cfg.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.router_dropout),
            nn.Linear(cfg.router_hidden_dim, cfg.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.router_dropout),
            nn.Linear(cfg.router_hidden_dim, cfg.router_num_modalities),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            pooled: [B, dim] 句向量
        Returns:
            [B, num_modalities] 模态 logits
        """
        return self.classifier(pooled)


# ============================================================
# 多模态路由器
# ============================================================
class ModalityRouter(nn.Module):
    """多模态路由器 — 按输入模态路由到不同专家通路。

    对每个模态维护一个可学习专家 (Linear 投影), 并以轻量级分类器
    输出的软路由权重对各专家结果做加权融合, 实现端到端可微分路由。

    Args:
        cfg: MultimodalConfig 配置对象

    Note:
        ``forward(input_embeddings)`` 返回 ``(modality_logits, routed_features)``:
          - modality_logits:  [B, num_modalities] 模态分类 logits
          - routed_features:  [B, T, dim] 路由融合后的特征序列
    """

    #: 支持的模态名称 (顺序与 logits 维度对应)
    MODALITY_NAMES: Tuple[str, ...] = ("text", "image", "video", "audio")

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.num_modalities = cfg.router_num_modalities

        # 确保模态名称数量与配置一致
        if len(self.MODALITY_NAMES) != cfg.router_num_modalities:
            # 自动截断或补齐名称 (保持健壮性)
            if cfg.router_num_modalities <= len(self.MODALITY_NAMES):
                self.MODALITY_NAMES = self.MODALITY_NAMES[: cfg.router_num_modalities]
            else:
                self.MODALITY_NAMES = tuple(
                    list(self.MODALITY_NAMES)
                    + [f"modality_{i}" for i in range(
                        len(self.MODALITY_NAMES), cfg.router_num_modalities
                    )]
                )

        # ── 轻量级模态分类器 ──
        self.classifier = ModalityClassifier(cfg)

        # ── 每个模态一个可学习专家投影 (路由权重主体) ──
        self.experts = nn.ModuleList(
            [nn.Linear(cfg.dim, cfg.dim) for _ in range(cfg.router_num_modalities)]
        )

        # ── 可学习路由门控: 每个模态一个标量, 调节该模态路由强度 ──
        self.router_gate = nn.Parameter(torch.ones(cfg.router_num_modalities))

        # ── 模态类型嵌入: 路由后按预测模态叠加 ──
        self.modality_embed = nn.Parameter(
            torch.randn(cfg.router_num_modalities, cfg.dim) * 0.02
        )

        # ── 输出归一化与 dropout ──
        self.output_norm = nn.RMSNorm(cfg.dim)
        self.output_drop = nn.Dropout(cfg.router_dropout)

        # 输入投影 (稳定路由前表征)
        self.input_proj = nn.Linear(cfg.dim, cfg.dim)

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------
    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        """将 [B, T, D] 均值池化为 [B, D]; 若已是 2D 则原样返回。"""
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    def _compute_routing(
        self, pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算模态 logits 与软路由权重。

        Args:
            pooled: [B, dim] 句向量
        Returns:
            (modality_logits, routing_probs):
              - modality_logits: [B, num_modalities]
              - routing_probs:  [B, num_modalities] (softmax, 经门控缩放)
        """
        modality_logits = self.classifier(pooled)            # [B, M]
        # 门控缩放 logits 后做 softmax, 使路由强度可学习
        routing_probs = torch.softmax(
            modality_logits * self.router_gate, dim=-1
        )  # [B, M]
        return modality_logits, routing_probs

    def _route(
        self,
        features: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """对各专家输出做加权融合。

        Args:
            features:      [B, T, dim] 输入序列
            routing_probs: [B, num_modalities] 软路由权重
        Returns:
            [B, T, dim] 融合后的特征
        """
        B, T, D = features.shape
        M = self.num_modalities

        # 各专家处理: 每个模态对完整序列做一次投影
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(features))  # [B, T, D]
        # 堆叠: [B, M, T, D]
        stacked = torch.stack(expert_outputs, dim=1)

        # 路由权重广播: [B, M] -> [B, M, 1, 1]
        weights = routing_probs.unsqueeze(-1).unsqueeze(-1)
        # 加权求和: [B, M, T, D] * [B, M, 1, 1] -> sum(dim=1) -> [B, T, D]
        routed = (stacked * weights).sum(dim=1)
        return routed

    # --------------------------------------------------------
    # 前向接口
    # --------------------------------------------------------
    def forward(
        self,
        input_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算: 分类模态并路由融合。

        Args:
            input_embeddings: [B, T, dim] 输入嵌入序列
                             (也接受 [B, dim], 内部按 2D 处理)
        Returns:
            (modality_logits, routed_features):
              - modality_logits: [B, num_modalities] 模态分类 logits
              - routed_features: [B, T, dim] 路由融合后的特征序列
                (输入为 2D 时输出为 [B, dim])
        """
        is_2d = input_embeddings.ndim == 2
        if is_2d:
            # [B, dim] -> [B, 1, dim] 以复用序列处理路径
            features = input_embeddings.unsqueeze(1)
        else:
            features = input_embeddings

        # 1. 输入投影 (稳定表征)
        features = self.input_proj(features)

        # 2. 池化 + 分类 + 路由权重
        pooled = self._pool(features)                       # [B, dim]
        modality_logits, routing_probs = self._compute_routing(pooled)

        # 3. 专家加权融合
        routed = self._route(features, routing_probs)       # [B, T, dim]

        # 4. 按预测模态叠加类型嵌入 (取期望模态嵌入)
        #    E[modality | input] = sum_m p_m * embed_m  -> [B, dim]
        expected_modality_embed = torch.matmul(
            routing_probs, self.modality_embed
        )  # [B, dim]
        # 广播到序列维度: [B, dim] -> [B, 1, dim]
        routed = routed + expected_modality_embed.unsqueeze(1)

        # 5. 残差 + dropout + 归一化
        routed = self.output_norm(features + self.output_drop(routed))

        if is_2d:
            routed = routed.squeeze(1)  # [B, dim]
        return modality_logits, routed

    # --------------------------------------------------------
    # 辅助接口
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_modality(
        self,
        input_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测输入的模态 (argmax)。

        Args:
            input_embeddings: [B, T, dim] 或 [B, dim]
        Returns:
            (modality_index, modality_name):
              - modality_index: [B] 预测模态索引 (int)
              - modality_name:  预测模态名称张量 [B] (按索引映射)
        """
        pooled = self._pool(input_embeddings)
        logits = self.classifier(pooled)
        modality_index = logits.argmax(dim=-1)  # [B]
        # 名称映射: 用索引从名称列表取值
        names = self.MODALITY_NAMES
        # 构建名称查找表并按索引取值 (双后端兼容)
        # 注意: 自研引擎的 Tensor 不支持直接存字符串, 这里返回索引张量与
        #       Python 名称列表, 由调用方自行映射。
        return modality_index, names

    def route_by_modality(
        self,
        input_embeddings: torch.Tensor,
        modality_index: int,
    ) -> torch.Tensor:
        """显式按指定模态路由 (硬路由, 用于已知模态的场景)。

        Args:
            input_embeddings: [B, T, dim] 或 [B, dim]
            modality_index:   模态索引 (0=text, 1=image, 2=video, 3=audio)
        Returns:
            [B, T, dim] 路由后的特征 (单一专家 + 模态嵌入 + 残差归一化)
        """
        assert 0 <= modality_index < self.num_modalities, (
            f"modality_index 超出范围 [0, {self.num_modalities})"
        )
        is_2d = input_embeddings.ndim == 2
        features = input_embeddings.unsqueeze(1) if is_2d else input_embeddings

        features = self.input_proj(features)
        routed = self.experts[modality_index](features)
        routed = routed + self.modality_embed[modality_index].unsqueeze(0).unsqueeze(0)
        routed = self.output_norm(features + self.output_drop(routed))

        if is_2d:
            routed = routed.squeeze(1)
        return routed


__all__ = [
    "ModalityClassifier",
    "ModalityRouter",
]
