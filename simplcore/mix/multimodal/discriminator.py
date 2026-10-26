# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
多模态判别器模块
================
用于 GAN 式训练与生成质量评估。

核心组件:
  - DiscriminatorBlock: 含多头自注意力 + FFN 的判别块 (Pre-Norm 结构)
  - Discriminator:       完整判别器, 输入多模态特征序列, 输出两路信号
  - compute_alignment_loss: 真实/生成特征对齐损失 (对比学习用)

Discriminator 输出 (DiscriminatorOutput):
  1. logits   : [B, dis_output_dim]  真假判别 logits
                (经 ``torch.sigmoid(logits)`` 即为真假概率;
                 GAN 训练推荐配合 BCEWithLogitsLoss 直接使用 logits 以保证数值稳定)
  2. features : [B, dis_feature_dim] 特征向量 (用于对比学习 / 特征匹配)

底层张量库为 ``simplcore.tensor`` (双后端: 自研 NumPy 引擎 / PyTorch),
完全脱离对 PyTorch 的硬依赖, 与 SimplCore Mix 其余模块保持一致。
"""
from __future__ import annotations

from collections import namedtuple
from typing import Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from .config import MultimodalConfig


# 判别器前向输出容器 (自文档化, 避免 tuple 顺序混淆)
DiscriminatorOutput = namedtuple(
    "DiscriminatorOutput", ["logits", "features"]
)


# ============================================================
# 多头自注意力 (判别器专用, 双向)
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力 (双向, 无因果掩码)

    判别器需要对完整序列做全局建模, 因此不使用因果掩码。
    可选 padding mask 用于忽略变长序列的填充位置。

    Args:
        hidden_dim: 模型隐藏维度
        n_heads:    注意力头数
        dropout:    注意力权重 dropout 概率
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert n_heads > 0, "注意力头数必须大于 0"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim > 0, (
            f"hidden_dim({hidden_dim}) 必须大于等于 n_heads({n_heads})"
        )
        self.inner_dim = self.n_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Q/K/V 独立投影 (双后端兼容)
        self.wq = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向计算

        Args:
            x:    [B, T, hidden_dim]
            mask: [B, T] 可选, 1 表示有效 token, 0 表示填充 (将被忽略)
        Returns:
            [B, T, hidden_dim]
        """
        B, T, _ = x.shape

        # 分别计算 Q/K/V: [B, T, inner_dim] -> [B, n_heads, T, head_dim]
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: [B, n_heads, T, T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用 padding mask (屏蔽 key 侧的填充位置)
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T]
            key_mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(key_mask < 0.5, -1e4)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和: [B, n_heads, T, head_dim] -> 合并头 [B, T, inner_dim]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, self.inner_dim)
        return self.wo(out)


# ============================================================
# FFN (前馈网络, GELU 激活)
# ============================================================
class FeedForward(nn.Module):
    """两层前馈网络: hidden_dim -> mlp_dim -> hidden_dim, 使用 GELU 激活

    Args:
        hidden_dim: 输入/输出维度
        mlp_dim:    中间扩展维度
        dropout:    dropout 概率
    """

    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.drop(h)


# ============================================================
# DiscriminatorBlock
# ============================================================
class DiscriminatorBlock(nn.Module):
    """判别器基础块: Pre-Norm 多头自注意力 + Pre-Norm FFN

    结构::
        x = x + attn(norm1(x), mask)
        x = x + ffn(norm2(x))

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        hidden_dim = cfg.dis_hidden_dim
        # FFN 扩展比默认 4.0 (配置未提供专用字段)
        mlp_dim = hidden_dim * 4

        # 多头自注意力子层
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            n_heads=cfg.dis_n_heads,
            dropout=cfg.dis_dropout,
        )

        # 前馈子层
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=cfg.dis_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向计算

        Args:
            x:    [B, T, hidden_dim]
            mask: [B, T] 可选 padding mask
        Returns:
            [B, T, hidden_dim]
        """
        # 自注意力 + 残差
        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        x = x + h

        # FFN + 残差
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ============================================================
# Discriminator
# ============================================================
class Discriminator(nn.Module):
    """完整多模态判别器

    处理流程:
        1. 输入投影: dim -> dis_hidden_dim
        2. N 层 DiscriminatorBlock 提取判别特征 (多层 Transformer encoder)
        3. 最终归一化 + 序列池化 (mean pooling, 支持 padding mask)
        4. 两个输出头:
           - logits_head:  真假判别 logits   [B, dis_output_dim]
                           (经 sigmoid 即为真假概率; GAN 训练用 BCEWithLogitsLoss)
           - feature_head: 对比学习特征向量  [B, dis_feature_dim]

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.dis_hidden_dim

        # 输入投影: 主模型维度 -> 判别器隐藏维度
        self.input_proj = nn.Linear(cfg.dim, cfg.dis_hidden_dim)
        self.input_drop = nn.Dropout(cfg.dis_dropout)

        # N 层判别块 (Transformer encoder)
        self.blocks = nn.ModuleList(
            [DiscriminatorBlock(cfg) for _ in range(cfg.dis_n_layers)]
        )

        # 最终归一化
        self.final_norm = nn.LayerNorm(cfg.dis_hidden_dim, eps=1e-6)

        # ── 两个输出头 ──
        # 头1: 真假判别 logits (不加激活, 配合 BCEWithLogitsLoss)
        self.logits_head = nn.Linear(cfg.dis_hidden_dim, cfg.dis_output_dim)
        # 头2: 对比学习特征向量
        self.feature_head = nn.Linear(cfg.dis_hidden_dim, cfg.dis_feature_dim)

    # ------------------------------------------------------------------
    def _pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """序列维度池化

        有 mask 时做掩码加权均值池化 (忽略填充位置),
        无 mask 时做简单均值池化。

        Args:
            x:    [B, T, hidden_dim]
            mask: [B, T] 可选, 1 有效 / 0 填充
        Returns:
            [B, hidden_dim]
        """
        if mask is None:
            return x.mean(dim=1)

        # mask: [B, T] -> [B, T, 1]
        m = mask.unsqueeze(-1).to(x.dtype)
        # 有效 token 数, 加 eps 防止全填充时除零
        denom = m.sum(dim=1).clamp(min=1e-6)
        return (x * m).sum(dim=1) / denom

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> DiscriminatorOutput:
        """前向计算

        Args:
            x:    [B, T, dim] 多模态特征序列
            mask: [B, T] 可选 padding mask (1 有效, 0 填充)
        Returns:
            DiscriminatorOutput(logits, features):
              - logits:   [B, dis_output_dim]   真假判别 logits
                          (sigmoid(logits) 即真假概率)
              - features: [B, dis_feature_dim]  对比学习特征
        """
        # 1. 输入投影
        h = self.input_drop(self.input_proj(x))

        # 2. N 层判别块
        for block in self.blocks:
            h = block(h, mask=mask)

        # 3. 最终归一化 + 池化
        h = self.final_norm(h)
        pooled = self._pool(h, mask)  # [B, hidden_dim]

        # 4. 两头输出
        logits = self.logits_head(pooled)     # [B, dis_output_dim]
        features = self.feature_head(pooled)  # [B, dis_feature_dim]

        return DiscriminatorOutput(logits=logits, features=features)


# ============================================================
# 对齐损失 (对比学习)
# ============================================================
def compute_alignment_loss(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """真实/生成特征对齐损失 (对比学习)

    用于将生成器 (或被评估的编码) 的特征分布拉向真实数据分布,
    使判别器特征空间中 real 与 fake 对齐, 提升生成质量与评估稳定性。

    损失由三部分组成:
      1. 余弦对齐损失: 最大化配对样本的余弦相似度 -> 最小化 (1 - cos)
      2. 一阶矩匹配:   最小化 real/fake 均值向量之差的 L2 范数 (feature matching)
      3. 二阶矩匹配:   最小化 real/fake 标准差之差的 L2 范数 (分布展度对齐)

    三项均为非负标量, 最终返回其加权和。

    Args:
        real_features: [B, D] 真实样本特征
        fake_features: [B, D] 生成样本特征
        temperature:   温度系数, 用于调节余弦项的锐度
    Returns:
        标量损失张量 (可反向传播)
    """
    assert real_features.ndim == 2 and fake_features.ndim == 2, (
        "特征张量必须为 [B, D] 形状"
    )
    assert fake_features.shape == real_features.shape, (
        "real/fake 特征形状必须一致"
    )

    eps = 1e-8

    # ── 1. 余弦对齐损失 (per-sample 配对) ──
    # L2 归一化后内积 = 余弦相似度
    real_norm = real_features / (
        real_features.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    )
    fake_norm = fake_features / (
        fake_features.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    )
    # 逐样本余弦相似度 [B]
    cos_sim = (real_norm * fake_norm).sum(dim=-1)
    # 最小化 (1 - cos)
    cosine_loss = (1.0 - cos_sim).mean()

    # ── 2. 一阶矩匹配 (均值特征匹配) ──
    real_mean = real_features.mean(dim=0)   # [D]
    fake_mean = fake_features.mean(dim=0)   # [D]
    mean_loss = (real_mean - fake_mean).pow(2).mean()

    # ── 3. 二阶矩匹配 (标准差匹配, 对齐分布展度) ──
    real_std = real_features.std(dim=0)     # [D]
    fake_std = fake_features.std(dim=0)     # [D]
    std_loss = (real_std - fake_std).pow(2).mean()

    # 加权组合: 余弦对齐为主, 矩匹配为辅
    loss = temperature * cosine_loss + mean_loss + 0.5 * std_loss
    return loss


__all__ = [
    "DiscriminatorOutput",
    "MultiHeadSelfAttention",
    "FeedForward",
    "DiscriminatorBlock",
    "Discriminator",
    "compute_alignment_loss",
]


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 小型配置以便快速自检
    cfg = MultimodalConfig(
        dim=256,
        dis_hidden_dim=128,
        dis_n_layers=2,
        dis_n_heads=4,
        dis_dropout=0.0,
        dis_output_dim=1,
        dis_feature_dim=64,
    )

    print("=" * 60)
    print("Discriminator 自检")
    print("=" * 60)

    dis = Discriminator(cfg)

    # 1) 前向: [B, T, dim] -> logits [B,1] + features [B, dis_feature_dim]
    B, T = 3, 10
    feats = torch.randn(B, T, cfg.dim)
    out = dis(feats)
    print(f"输入: {tuple(feats.shape)}")
    print(f"  logits  : {tuple(out.logits.shape)}")
    print(f"  features: {tuple(out.features.shape)}")
    assert out.logits.shape == (B, 1), f"logits 形状错误: {out.logits.shape}"
    assert out.features.shape == (B, cfg.dis_feature_dim), (
        f"features 形状错误: {out.features.shape}"
    )

    # 2) 真假概率 (sigmoid)
    prob = torch.sigmoid(out.logits)
    print(f"  真假概率: {tuple(prob.shape)}, 范围 [{prob.min():.4f}, {prob.max():.4f}]")

    # 3) 带 padding mask 的前向
    mask = torch.ones(B, T)
    mask[0, 6:] = 0  # 第 0 个样本后 4 个位置填充
    out_m = dis(feats, mask=mask)
    print(f"带 mask: logits {tuple(out_m.logits.shape)}")
    assert out_m.logits.shape == (B, 1)

    # 4) 梯度流
    feats_g = torch.randn(B, T, cfg.dim)
    out_g = dis(feats_g)
    out_g.logits.sum().backward()
    ok_grad = dis.logits_head.weight.grad is not None
    print(f"梯度流: {'OK' if ok_grad else 'FAIL'}")
    assert ok_grad

    # 5) 对齐损失
    rf = torch.randn(B, cfg.dis_feature_dim)
    ff = torch.randn(B, cfg.dis_feature_dim)
    loss = compute_alignment_loss(rf, ff, temperature=0.1)
    print(f"对齐损失: {float(loss):.6f}")
    assert loss.ndim == 0, "对齐损失应为标量"

    print("\n[PASS] 判别器 (Discriminator) 测试全部通过.")
