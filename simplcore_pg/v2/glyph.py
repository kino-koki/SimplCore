# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GlyphStream 无词表连续符号接口 (输入/输出两端彻底取代 token 词表)。

链路::

    文本 ──> GlyphRenderer (字符→字形位图, 复用 perception.py)
         ──> GlyphEncoder  (patch ViT "看见"字形, 复用 perception.py)
         ──> 字符向量 (patch 均值) ──> 投影到 H
         ──> EntropyDrivenPooler (熵驱动动态合并, "学出来的分词")
         ──> 连续语义单元流 (B, L', H) ──> TriMemory 推理核

    推理核输出 ──> PrototypeDecoder (可学习字形原型库, 余弦匹配)
              ──> 字符索引 ──> charset 映射 ──> 文本

**没有任何离散词表**：没有 nn.Embedding(V,H)、没有 lm_head(V,H)、没有
分词器文件。原型库是可学习参数 (P×H)，不是查表字典；字符合并边界由
边界预测网络从数据中学出，不是人工词条。
"""
from __future__ import annotations

import string
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm, init_weights
from ..perception import GlyphRenderer, GlyphEncoder

# 默认字符集：ASCII 可见字符 + 常用汉字 (索引 0 保留为 UNK/控制槽)。
_DEFAULT_CN = list("的一是在不了有和人这中大来上国个到说时为来分生对于就那得着下")
DEFAULT_CHARSET = ["<unk>"] + sorted(set(list(string.printable.strip()) + _DEFAULT_CN))


# ----------------------------------------------------------------------
# 输入：文本 -> 连续字符向量 -> 熵驱动合并
# ----------------------------------------------------------------------
class GlyphStreamEncoder(nn.Module):
    """文本 -> 连续语义单元流。

    Args:
        config: V2Config (取 glyph_* / hidden_size / pool_* 字段)。
        charset: 字符集 (仅用于边界启发式与调试；编码本身不需要字符身份)。
    """

    def __init__(self, config, charset: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.config = config
        self.charset = list(charset) if charset is not None else DEFAULT_CHARSET
        self.renderer = GlyphRenderer(glyph_size=config.glyph_size, channels=1)
        self.glyph_encoder = GlyphEncoder(config)      # 复用 perception.GlyphEncoder
        self.char_proj = nn.Linear(config.glyph_hidden, config.hidden_size, bias=False)
        self.char_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    def encode_chars(self, text: str) -> torch.Tensor:
        """单条文本 -> (Lc, H) 每字符连续向量。"""
        if len(text) == 0:
            return torch.zeros(0, self.config.hidden_size)
        glyphs = self.renderer.render(text)                    # (Lc, 1, gh, gw)
        feats = self.glyph_encoder.encode(glyphs)              # (Lc, P, glyph_hidden)
        char_vec = feats.mean(dim=1)                           # (Lc, glyph_hidden)
        return self.char_norm(self.char_proj(char_vec))        # (Lc, H)

    def forward(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """批编码：texts -> (x (B, Lmax, H), mask (B, Lmax) bool)。"""
        vecs = [self.encode_chars(t) for t in texts]
        B = len(vecs)
        Lmax = max(1, max(v.shape[0] for v in vecs))
        H = self.config.hidden_size
        x = torch.zeros(B, Lmax, H)
        mask = torch.zeros(B, Lmax, dtype=torch.bool)
        for b, v in enumerate(vecs):
            if v.shape[0] > 0:
                x[b, : v.shape[0]] = v
                mask[b, : v.shape[0]] = True
        return x, mask

    # ------------------------------------------------------------------
    def heuristic_boundaries(self, text: str) -> List[int]:
        """边界启发式 (池化辅助损失的教学信号)：

        在以下位置之后切分：空白/标点之后、CJK 与 ASCII 类型切换处。
        返回与字符等长的 0/1 列表 (1 = 该字符之后是边界)。
        """
        def _cls(ch: str) -> int:
            if ch.isspace():
                return 0
            if ord(ch) > 0x2E7F:      # CJK 及扩展区
                return 2
            if ch.isalnum():
                return 1
            return 3                  # ASCII 标点/符号
        bounds = []
        n = len(text)
        for i, ch in enumerate(text):
            if i == n - 1:
                bounds.append(1)
                continue
            cur, nxt = _cls(ch), _cls(text[i + 1])
            bounds.append(1 if (cur == 0 or cur == 3 or cur != nxt) else 0)
        return bounds


class EntropyDrivenPooler(nn.Module):
    """熵驱动动态合并：学习式"分词"边界的替代物。

    边界预测网络给出每位置的边界概率；超过阈值 (或达到
    ``max_chars_per_unit``) 即切段；段内用**注意力加权合并**
    (可学习打分) 得到语义单元向量。

    辅助损失：边界概率对启发式边界的 BCE —— 让边界预测器先对齐
    语言直觉，端到端阶段再由主损失继续塑形。
    """

    def __init__(self, config) -> None:
        super().__init__()
        H = config.hidden_size
        self.config = config
        self.boundary_net = nn.Sequential(
            nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1),
        )
        self.merge_score = nn.Linear(H, 1, bias=False)
        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    def segment(self, boundary_prob: torch.Tensor) -> List[Tuple[int, int]]:
        """由边界概率切分单条序列，返回 [(start, end), ...] 半开区间。"""
        thr = self.config.pool_boundary_threshold
        cap = self.config.max_chars_per_unit
        segs, start = [], 0
        L = boundary_prob.shape[0]
        for i in range(L):
            if boundary_prob[i].item() >= thr or (i - start + 1) >= cap or i == L - 1:
                segs.append((start, i + 1))
                start = i + 1
        return segs

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        boundary_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]], Optional[torch.Tensor]]:
        """合并字符向量为语义单元。

        Args:
            x: (B, L, H) 字符向量。
            mask: (B, L) bool 有效位置。
            boundary_targets: 可选 (B, L) float 启发式边界 (辅助损失)。

        Returns:
            (pooled, segments, aux_loss)：
            pooled (B, L', H) (右对齐padding为零)；segments 每条样本的
            切分区间；aux_loss 边界 BCE (无 boundary_targets 时为 None)。
        """
        B, L, H = x.shape
        logits = self.boundary_net(x).squeeze(-1)              # (B, L)
        prob = torch.sigmoid(logits)

        pooled_rows: List[torch.Tensor] = []
        all_segs: List[List[Tuple[int, int]]] = []
        for b in range(B):
            valid_len = int(mask[b].sum().item())
            xb = x[b, :valid_len]
            pb = prob[b, :valid_len]
            segs = self.segment(pb) if valid_len > 0 else []
            units = []
            for (s, e) in segs:
                members = xb[s:e]                              # (m, H)
                w = torch.softmax(self.merge_score(members).squeeze(-1), dim=0)
                units.append((members * w.unsqueeze(-1)).sum(dim=0))
            if units:
                pooled_rows.append(torch.stack(units, dim=0))
            else:
                pooled_rows.append(torch.zeros(0, H, device=x.device, dtype=x.dtype))
            all_segs.append(segs)

        Lp = max(1, max(r.shape[0] for r in pooled_rows))
        pooled = torch.zeros(B, Lp, H, device=x.device, dtype=x.dtype)
        for b, r in enumerate(pooled_rows):
            if r.shape[0] > 0:
                pooled[b, : r.shape[0]] = r

        aux = None
        if boundary_targets is not None:
            aux = F.binary_cross_entropy_with_logits(
                logits[mask], boundary_targets[mask].float()
            )
        return pooled, all_segs, aux


# ----------------------------------------------------------------------
# 输出：连续向量 -> 原型库匹配 -> 字符
# ----------------------------------------------------------------------
class PrototypeDecoder(nn.Module):
    """可学习字形原型解码器 (取代 lm_head)。

    logits = τ · cos(head(x), prototypes)，τ 为可学习温度。
    原型是参数 (P×H)：与词表嵌入的本质区别在于它**连续可插值**、
    规模与字符覆盖无关 (新字符可分配新原型而不重训词表)。
    """

    def __init__(self, hidden_size: int, num_prototypes: int,
                 charset: Optional[Sequence[str]] = None,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.charset = list(charset) if charset is not None else DEFAULT_CHARSET
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.head = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, hidden_size))
        nn.init.normal_(self.prototypes, std=0.02)
        self.log_temperature = nn.Parameter(torch.zeros(1))    # τ = exp(·)
        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., H) -> (..., P) 原型匹配 logits。"""
        h = self.head(self.norm(x))
        h = F.normalize(h, p=2, dim=-1, eps=1e-6)
        p = F.normalize(self.prototypes, p=2, dim=-1, eps=1e-6)
        return torch.exp(self.log_temperature) * (h @ p.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    # ------------------------------------------------------------------
    def decode(self, x: torch.Tensor) -> List[str]:
        """x: (..., H) -> 最近原型对应的字符列表 (索引越界映射到 <unk>)。"""
        idx = self.logits(x).argmax(dim=-1).reshape(-1).tolist()
        n = len(self.charset)
        return [self.charset[i] if 0 <= i < n else "<unk>" for i in idx]

    def char_to_index(self, ch: str) -> int:
        """字符 -> 原型索引 (未收录字符映射到 0 号 <unk> 槽)。"""
        try:
            return self.charset.index(ch)
        except ValueError:
            return 0
