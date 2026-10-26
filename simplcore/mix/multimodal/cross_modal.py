# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
跨模态融合层 (CrossModalFusion)
================================
将视觉 token 序列与文本 token 序列融合为统一的表征序列,
供下游语言模型或多模态主干消费。

支持三种融合模式:
  1. ``"prepend"``:        视觉 token 前插到文本 token 之前, 形成 [V; T]
                           随后经过若干自注意力融合块让两模态交互。
  2. ``"interleave"``:     视觉与文本 token 交错排列 (逐 token 交替),
                           长度不等时较短序列耗尽后追加较长序列的剩余 token。
  3. ``"cross_attention"``: 文本作为 query 关注视觉 key/value,
                           文本序列长度保持不变, 视觉信息被注入文本表征。

此外提供对比学习损失 (InfoNCE) 用于跨模态对齐训练:
  - 将文本 / 视觉特征分别投影到共享对比空间并 L2 归一化;
  - 以温度缩放的余弦相似度矩阵计算对称 InfoNCE 损失 (CLIP 风格)。

底层张量库为 ``simplcore.tensor`` (双后端: 自研 NumPy 引擎 / PyTorch),
完全脱离对 PyTorch 的硬依赖, 与 SimplCore Mix 其余模块保持一致。
"""
from __future__ import annotations

from typing import Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from .config import MultimodalConfig


# ============================================================
# 多头注意力 (通用, 支持自注意力与交叉注意力)
# ============================================================
class MultiHeadAttention(nn.Module):
    """标准多头注意力, 可同时承担自注意力与交叉注意力。

    - 自注意力:       forward(x, x, x)
    - 交叉注意力:     forward(query, key, value)

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

        # Q/K/V 独立投影 (双后端兼容, 与 mix 模块风格一致)
        self.wq = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向计算。

        Args:
            query:            [B, Tq, hidden_dim]
            key:              [B, Tk, hidden_dim]
            value:            [B, Tk, hidden_dim]
            key_padding_mask: [B, Tk] 可选, 1 有效 / 0 填充 (将被屏蔽)
        Returns:
            [B, Tq, hidden_dim]
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        # 投影并重排为多头: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = self.wq(query).reshape(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(key).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(value).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: [B, n_heads, Tq, Tk]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用 key 侧 padding mask (屏蔽填充位置)
        # 自研张量引擎未重载 __eq__, 此处用 < 0.5 等价表达 "mask==0"
        if key_padding_mask is not None:
            # mask: [B, Tk] -> [B, 1, 1, Tk]
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(key_mask < 0.5, -1e4)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和: [B, n_heads, Tq, head_dim] -> 合并头 [B, Tq, inner_dim]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, Tq, self.inner_dim)
        return self.wo(out)


# ============================================================
# 前馈网络 (GELU)
# ============================================================
class FeedForward(nn.Module):
    """两层前馈网络: hidden -> mlp_dim -> hidden, 使用 GELU 激活。

    Args:
        hidden_dim: 输入 / 输出维度
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
# 自注意力融合块 (Pre-Norm) — 用于 prepend / interleave 模式
# ============================================================
class FusionBlock(nn.Module):
    """Pre-Norm 自注意力 + Pre-Norm FFN 融合块。

    对拼接后的混合序列做全局自注意力, 使视觉与文本 token 互相交互。

    结构::
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        hidden = cfg.dim
        mlp_dim = hidden * 4

        self.norm1 = nn.RMSNorm(hidden)
        self.attn = MultiHeadAttention(
            hidden_dim=hidden,
            n_heads=cfg.fusion_n_heads,
            dropout=cfg.fusion_dropout,
        )
        self.norm2 = nn.RMSNorm(hidden)
        self.ffn = FeedForward(hidden, mlp_dim, cfg.fusion_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ============================================================
# 交叉注意力融合块 (Pre-Norm) — 用于 cross_attention 模式
# ============================================================
class CrossAttentionBlock(nn.Module):
    """Pre-Norm 交叉注意力 + Pre-Norm FFN 融合块。

    文本作为 query, 视觉作为 key/value, 文本长度保持不变。

    结构::
        text = text + cross_attn(norm_q(text), norm_kv(vision))
        text = text + ffn(norm_ffn(text))

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        hidden = cfg.dim
        mlp_dim = hidden * 4

        # query 侧归一化 (文本)
        self.norm_q = nn.RMSNorm(hidden)
        # key/value 侧归一化 (视觉)
        self.norm_kv = nn.RMSNorm(hidden)
        self.cross_attn = MultiHeadAttention(
            hidden_dim=hidden,
            n_heads=cfg.fusion_n_heads,
            dropout=cfg.fusion_dropout,
        )
        self.norm_ffn = nn.RMSNorm(hidden)
        self.ffn = FeedForward(hidden, mlp_dim, cfg.fusion_dropout)

    def forward(
        self,
        text: torch.Tensor,
        vision: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(text)
        kv = self.norm_kv(vision)
        h = self.cross_attn(q, kv, kv, key_padding_mask=vision_mask)
        text = text + h

        h = self.norm_ffn(text)
        h = self.ffn(h)
        text = text + h
        return text


# ============================================================
# 跨模态融合层
# ============================================================
class CrossModalFusion(nn.Module):
    """跨模态融合层 — 视觉 token 与文本 token 融合。

    支持三种融合模式 (``fusion_mode`` 配置或 forward 时显式指定):

      * ``"prepend"``: 视觉 token 前插到文本之前 [V; T],
        再经过 ``fusion_n_layers`` 层自注意力融合块做跨模态交互。
      * ``"interleave"``: 视觉与文本 token 逐位交错,
        较短序列耗尽后追加较长序列剩余 token, 同样过自注意力融合块。
      * ``"cross_attention"``: 文本 query 关注视觉 key/value,
        文本长度保持不变, 视觉信息被注入文本表征。

    对比学习对齐 (CLIP 风格 InfoNCE):
      将文本 / 视觉特征分别投影到共享对比空间 (维度 = ``dis_feature_dim``),
      L2 归一化后以温度 ``fusion_contrastive_temperature`` 缩放余弦相似度矩阵,
      计算对称 InfoNCE 损失 (行方向 text->vision, 列方向 vision->text)。

    Args:
        cfg: MultimodalConfig 配置对象
    """

    #: 支持的融合模式
    SUPPORTED_MODES: Tuple[str, ...] = (
        "prepend",
        "interleave",
        "cross_attention",
    )

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.default_mode = cfg.fusion_mode
        self.temperature = cfg.fusion_contrastive_temperature

        # ── 模态类型嵌入: 区分文本 / 视觉 token ──
        # 0 = text, 1 = vision
        self.modality_embed = nn.Parameter(torch.randn(2, cfg.dim) * 0.02)

        # ── 自注意力融合块 (prepend / interleave 模式) ──
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(cfg) for _ in range(cfg.fusion_n_layers)]
        )

        # ── 交叉注意力融合块 (cross_attention 模式) ──
        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(cfg) for _ in range(cfg.fusion_n_layers)]
        )

        # ── 融合后归一化 ──
        self.fusion_norm = nn.RMSNorm(cfg.dim)

        # ── 对比学习投影头: dim -> dis_feature_dim ──
        feat_dim = cfg.dis_feature_dim
        self.text_proj = nn.Linear(cfg.dim, feat_dim)
        self.vision_proj = nn.Linear(cfg.dim, feat_dim)

        # 对比投影后的归一化 (稳定的对比空间)
        self.text_proj_norm = nn.RMSNorm(feat_dim)
        self.vision_proj_norm = nn.RMSNorm(feat_dim)

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------
    def _resolve_mode(self, mode: Optional[str]) -> str:
        """确定本次前向使用的融合模式。"""
        mode = mode if mode is not None else self.default_mode
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"不支持的融合模式: {mode!r}, 可选: {self.SUPPORTED_MODES}"
            )
        return mode

    def _add_modality_embed(self, features: torch.Tensor, is_vision: bool) -> torch.Tensor:
        """为 token 序列叠加模态类型嵌入。

        Args:
            features:   [B, T, dim]
            is_vision:  True -> 视觉嵌入, False -> 文本嵌入
        Returns:
            [B, T, dim]
        """
        idx = 1 if is_vision else 0
        # modality_embed[idx]: [dim] -> [1, 1, dim] 广播
        return features + self.modality_embed[idx].unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _pool_to_2d(features: torch.Tensor) -> torch.Tensor:
        """将 [B, T, D] 序列均值池化为 [B, D]; 若已是 [B, D] 则原样返回。"""
        if features.ndim == 3:
            return features.mean(dim=1)
        return features

    # --------------------------------------------------------
    # 三种融合模式
    # --------------------------------------------------------
    def _fuse_prepend(
        self,
        text: torch.Tensor,
        vision: torch.Tensor,
    ) -> torch.Tensor:
        """前插融合: [V; T] + 自注意力融合块。

        Args:
            text:    [B, Tt, dim]
            vision:  [B, Tv, dim]
        Returns:
            [B, Tv + Tt, dim]
        """
        text = self._add_modality_embed(text, False)
        vision = self._add_modality_embed(vision, True)
        fused = torch.cat([vision, text], dim=1)  # [B, Tv+Tt, dim]
        for block in self.fusion_blocks:
            fused = block(fused)
        return self.fusion_norm(fused)

    def _fuse_interleave(
        self,
        text: torch.Tensor,
        vision: torch.Tensor,
    ) -> torch.Tensor:
        """交错融合: 视觉与文本 token 逐位交替排列。

        顺序为 [v0, t0, v1, t1, ...]; 较短序列耗尽后追加较长序列剩余 token。
        交错后经过自注意力融合块做跨模态交互。

        Args:
            text:    [B, Tt, dim]
            vision:  [B, Tv, dim]
        Returns:
            [B, Tt + Tv, dim]
        """
        text = self._add_modality_embed(text, False)
        vision = self._add_modality_embed(vision, True)

        B, Tt, D = text.shape
        Tv = vision.shape[1]
        min_len = min(Tt, Tv)

        # 交错部分: [v_i, t_i] 逐对堆叠
        # stack: [B, min_len, 2, D] -> reshape [B, 2*min_len, D]
        stacked = torch.stack(
            [vision[:, :min_len], text[:, :min_len]], dim=2
        )  # [B, min_len, 2, D]
        interleaved = stacked.reshape(B, 2 * min_len, D)

        # 追加较长序列的剩余 token
        parts = [interleaved]
        if Tt > min_len:
            parts.append(text[:, min_len:, :])
        if Tv > min_len:
            parts.append(vision[:, min_len:, :])
        fused = torch.cat(parts, dim=1)

        for block in self.fusion_blocks:
            fused = block(fused)
        return self.fusion_norm(fused)

    def _fuse_cross_attention(
        self,
        text: torch.Tensor,
        vision: torch.Tensor,
    ) -> torch.Tensor:
        """交叉注意力融合: 文本 query 关注视觉 key/value。

        文本序列长度保持不变, 视觉信息被注入文本表征。

        Args:
            text:    [B, Tt, dim]
            vision:  [B, Tv, dim]
        Returns:
            [B, Tt, dim]
        """
        text = self._add_modality_embed(text, False)
        vision = self._add_modality_embed(vision, True)

        out = text
        for block in self.cross_attn_blocks:
            out = block(out, vision)
        return self.fusion_norm(out)

    # --------------------------------------------------------
    # 前向接口
    # --------------------------------------------------------
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: Optional[torch.Tensor],
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """融合视觉 token 与文本 token。

        Args:
            text_features:   [B, Tt, dim] 文本特征序列
            vision_features: [B, Tv, dim] 视觉特征序列;
                             为 None 时按纯文本处理 (直接归一化返回)。
            mode:            融合模式, None 时使用配置默认值 ``fusion_mode``。
                             可选: "prepend" | "interleave" | "cross_attention"
        Returns:
            fused_features: 融合后的特征序列
              - prepend / interleave: [B, Tv + Tt, dim] / [B, Tt + Tv, dim]
              - cross_attention:      [B, Tt, dim]
        """
        # 纯文本: 无视觉信息可融合, 直接归一化返回
        if vision_features is None:
            return self.fusion_norm(text_features)

        mode = self._resolve_mode(mode)

        if mode == "prepend":
            return self._fuse_prepend(text_features, vision_features)
        elif mode == "interleave":
            return self._fuse_interleave(text_features, vision_features)
        else:  # cross_attention
            return self._fuse_cross_attention(text_features, vision_features)

    # --------------------------------------------------------
    # 对比学习损失 (InfoNCE)
    # --------------------------------------------------------
    def contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """跨模态对比学习损失 (CLIP 风格对称 InfoNCE)。

        流程:
          1. 将 3D 序列特征均值池化为 [B, dim] 句向量;
          2. 分别经投影头映射到共享对比空间 [B, feat_dim] 并归一化;
          3. 计算温度缩放的余弦相似度矩阵 [B, B];
          4. 对角线为正样本对, 其余为负样本, 计算对称 InfoNCE。

        Args:
            text_features:   [B, Tt, dim] 或 [B, dim] 文本特征
            vision_features: [B, Tv, dim] 或 [B, dim] 视觉特征
        Returns:
            标量损失张量 (可反向传播)
        """
        # 1. 池化为句向量
        text_vec = self._pool_to_2d(text_features)      # [B, dim]
        vision_vec = self._pool_to_2d(vision_features)  # [B, dim]

        # 2. 投影到对比空间 + 归一化
        text_emb = self.text_proj_norm(self.text_proj(text_vec))      # [B, feat_dim]
        vision_emb = self.vision_proj_norm(self.vision_proj(vision_vec))  # [B, feat_dim]

        text_emb = torch.normalize(text_emb, dim=-1)
        vision_emb = torch.normalize(vision_emb, dim=-1)

        # 3. 温度缩放的余弦相似度矩阵 [B, B]
        #    logits[i, j] = <text_i, vision_j> / temperature
        logits = torch.matmul(text_emb, vision_emb.transpose(-2, -1)) / self.temperature

        # 4. 对称 InfoNCE: 行方向 (text->vision) + 列方向 (vision->text)
        #    正样本位于对角线, 标签 = arange(B)
        B = logits.shape[0]
        labels = torch.arange(B)

        loss_t2v = torch.cross_entropy(logits, labels)
        loss_v2t = torch.cross_entropy(logits.transpose(-2, -1), labels)

        return (loss_t2v + loss_v2t) * 0.5


__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "FusionBlock",
    "CrossAttentionBlock",
    "CrossModalFusion",
]
