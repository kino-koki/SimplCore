# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""原生感知前端 (token-free perception front-end)。

本模块是"模型不看 Token，而是直接看见原始内容"的核心：
  * ``GlyphRenderer``   把字符渲染成字形位图 (像素)，完全绕过 subword 分词；
  * ``GlyphEncoder``    用视觉编码器"看见"字形，输出连续向量；
  * ``ImageEncoder``    把图像像素编码为连续特征；
  * ``VideoEncoder``    逐帧共享图像编码器 + 时序聚合，得到视频连续特征；
  * ``PerceptualResampler`` 用无注意力感知重采样 (LocalInteraction + 跨模态加性
                         汇聚) 把变长视觉特征压缩为固定 K 个连续"感知 token"，作为
                         推理核的统一输入。

所有路径产出连续潜变量 (B, *, D)，不经过任何离散词表。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .layers import RMSNorm, FFN, init_weights
from .multimodal import SimplCoreViTEncoder
from .localmix import LocalInteraction


# ----------------------------------------------------------------------
# 字形渲染器：字符 -> 字形位图
# ----------------------------------------------------------------------
class GlyphRenderer:
    """把字符渲染为字形位图 (像素)，跳过文本 Token 化。

    默认使用**确定性合成字形**：每个字符由其 Unicode 码点唯一决定一张
    平滑灰度图 (含码点位签名区域)，保证同一字符始终得到同一张图、不同
    字符得到不同图，且不依赖任何字体文件 —— 因而在任意环境 (含 CI/CPU)
    都能复现。若提供真实字体 (``try_load_font`` / ``use_real_font``)，
    则渲染真实字形位图，使模型感知到真实字形形状。

    输出张量取值范围 [0, 1]，形状 ``(N, C, gh, gw)``，N 为字符数。
    """

    def __init__(
        self,
        glyph_size: int = 32,
        channels: int = 1,
        use_real_font: bool = False,
        font_path: Optional[str] = None,
    ) -> None:
        self.glyph_size = glyph_size
        self.channels = channels
        self.font = None
        if use_real_font:
            self.font = self._load_font(font_path)

    # ----- 字体加载 (可选) -----
    @staticmethod
    def _load_font(font_path: Optional[str]):
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        except Exception:
            return None
        candidates = []
        if font_path:
            candidates.append(font_path)
        # 常见 CJK 字体路径 (Linux/macOS/Windows)
        import os
        candidates += [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                try:
                    return ImageFont.truetype(c, 24)
                except Exception:
                    continue
        return None

    def try_load_font(self, font_path: Optional[str] = None):
        """尝试加载真实字体；成功则后续渲染使用真实字形。"""
        self.font = self._load_font(font_path)
        return self.font is not None

    # ----- 公共渲染接口 -----
    def render(self, text) -> torch.Tensor:
        """渲染文本为字形位图。

        Args:
            text: 字符串或字符串列表。每个字符生成一张字形位图。

        Returns:
            (N, C, gh, gw) 浮点张量，N = 所有字符总数。
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        glyphs = []
        for t in texts:
            for ch in t:
                glyphs.append(self._render_char(ch))
        if len(glyphs) == 0:
            return torch.zeros(1, self.channels, self.glyph_size, self.glyph_size)
        return torch.stack(glyphs, dim=0)

    # ----- 单字符渲染 -----
    def _render_char(self, ch: str) -> torch.Tensor:
        if self.font is not None:
            try:
                return self._render_font(ch)
            except Exception:
                pass
        return self._render_synthetic(ch)

    def _render_synthetic(self, ch: str) -> torch.Tensor:
        gh = gw = self.glyph_size
        cp = ord(ch)
        # 1) 平滑光栅场：相位/频率由码点决定，使不同字符得到不同但平滑的图
        phase = (cp % 1000) / 1000.0 * math.pi * 2.0
        freq = 0.20 + (cp % 7) / 7.0 * 0.45
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, gh), torch.linspace(0.0, 1.0, gw), indexing="ij"
        )
        pat = 0.5 + 0.5 * torch.sin(xx * freq * math.pi * 4.0 + phase) \
            * torch.cos(yy * freq * math.pi * 4.0 - phase)
        g = pat * 0.8
        # 2) 码点位签名区域 (保证不同字符必然得到不同图)
        br = 6
        base = 2
        for b in range(36):
            if (cp >> b) & 1:
                r = b // br
                c = b % br
                g[base + r, base + c] = 1.0
        g = g.clamp(0.0, 1.0)
        img = g.unsqueeze(0)
        if self.channels == 3:
            img = img.repeat(3, 1, 1)
        return img

    def _render_font(self, ch: str) -> torch.Tensor:
        from PIL import Image, ImageDraw
        import numpy as np
        gh = gw = self.glyph_size
        img = Image.new("L", (gw, gh), 0)
        draw = ImageDraw.Draw(img)
        draw.text((gw // 2, gh // 2), ch, fill=255, font=self.font, anchor="mm")
        arr = torch.from_numpy(np.asarray(img)).to(torch.float32) / 255.0
        arr = arr.unsqueeze(0)
        if self.channels == 3:
            arr = arr.repeat(3, 1, 1)
        return arr


# ----------------------------------------------------------------------
# 各模态编码器 (均复用 SimplCoreViTEncoder，输出 patch 级连续特征)
# ----------------------------------------------------------------------
class GlyphEncoder(nn.Module):
    """字形编码器：字形位图 -> 连续字形特征。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = SimplCoreViTEncoder(
            image_size=config.glyph_size, patch_size=config.glyph_patch_size,
            hidden_size=config.glyph_hidden, num_layers=config.glyph_encoder_layers,
            num_heads=config.glyph_encoder_heads, output_dim=config.glyph_hidden,
            in_channels=config.glyph_in_channels,
        )

    def encode(self, glyphs: torch.Tensor) -> torch.Tensor:
        """glyphs: (N, C, gh, gw) -> (N, P, D)。"""
        return self.encoder.encode_image(glyphs)

    def forward(self, glyphs: torch.Tensor) -> torch.Tensor:
        return self.encode(glyphs)


class ImageEncoder(nn.Module):
    """图像编码器：图像像素 -> 连续图像特征。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = SimplCoreViTEncoder(
            image_size=config.vision_image_size, patch_size=config.vision_patch_size,
            hidden_size=config.vision_hidden, num_layers=config.vision_encoder_layers,
            num_heads=config.vision_encoder_heads, output_dim=config.vision_hidden,
            in_channels=config.vision_in_channels,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> (B, P, D)。"""
        return self.encoder.encode_image(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class VideoEncoder(nn.Module):
    """视频编码器：逐帧共享图像编码器 + 无注意力时序混合 (LocalInteraction)。

    输出 (B, P, vision_hidden)，P 为每帧 patch 数。
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_frames = config.video_frames
        self.encoder = SimplCoreViTEncoder(
            image_size=config.vision_image_size, patch_size=config.vision_patch_size,
            hidden_size=config.vision_hidden, num_layers=config.vision_encoder_layers,
            num_heads=config.vision_encoder_heads, output_dim=config.vision_hidden,
            in_channels=config.vision_in_channels,
        )
        D = config.vision_hidden
        self.temporal_layers = nn.ModuleList(
            [_TemporalLayer(D, config.video_temporal_heads, config.rms_norm_eps)
             for _ in range(config.video_temporal_layers)]
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, 3, H, W) -> (B, P, D)。"""
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)
        lat = self.encoder.encode_image(flat)            # (B*T, P, D)
        P, D = lat.shape[1], lat.shape[2]
        lat = lat.reshape(B, T, P, D).permute(0, 2, 1, 3).reshape(B * P, T, D)
        for layer in self.temporal_layers:
            lat = layer(lat)                              # 无注意力时序混合 (沿 T)
        lat = lat.reshape(B, P, T, D).mean(dim=2)        # 沿帧均值池化 -> (B, P, D)
        return lat

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encode(video)


class _TemporalLayer(nn.Module):
    """沿时序维度 (T) 的无注意力混合层。

    用 ``LocalInteraction``（因果深度卷积 + 门控线性递归）替代自注意力做帧间
    时序聚合：frames 因果混合（帧 t 仅见过去帧），复杂度 O(T·H)，无 Q/K/V/RoPE/SDPA。
    """

    def __init__(self, hidden: int, heads: int, eps: float) -> None:
        super().__init__()
        # heads 不再用于注意力；仅决定卷积核大小 (随头数微增感受野)
        kernel = max(3, min(heads * 2 + 1, 9))
        self.mix = LocalInteraction(hidden, kernel_size=kernel, dilation=1,
                                    use_recurrence=True, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*P, T, D) —— 沿 T 做无注意力因果混合
        return self.mix(x)


# ----------------------------------------------------------------------
# 感知重采样 (Perceiver)：变长特征 -> 固定 K 个连续感知 token
# ----------------------------------------------------------------------
class PerceptualResampler(nn.Module):
    """无注意力感知重采样：把变长视觉特征压缩为固定 K 个感知 token。

    所有模态共用一个重采样器 (在 H 维空间工作)；各模态特征在进入前已由
    对应投影层对齐到 H。内部 ``_PerceiverLayer`` 用跨模态加性汇聚 + 查询序列
    LocalInteraction 替代原始 Perceiver 的 cross/self-attention。
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_size))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList(
            [_PerceiverLayer(hidden_size, num_heads, eps) for _ in range(num_layers)]
        )
        self.apply(lambda m: init_weights(m, std=0.02))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, P, H) -> (B, K, H)。"""
        B = features.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, K, H)
        for layer in self.layers:
            q = layer(q, features)
        return q


class _PerceiverLayer(nn.Module):
    """单层感知重采样 (无注意力)：跨模态门控汇聚 + 查询序列 LocalInteraction + FFN。

    替代原始 Perceiver 的 cross-attention + self-attention：
      * 跨模态：queries 由全局特征上下文 (features 沿 P 均值，经线性投影) 做加性
        门控更新，**无 Q·K^T 点积 / 无 softmax 归一化**；
      * 查询内：``LocalInteraction`` (因果卷积 + 门控线性递归) 替代 self-attention。
    整体复杂度 O((P+K)·H)，无 Q/K/V/RoPE/SDPA。
    """

    def __init__(self, hidden: int, heads: int, eps: float) -> None:
        super().__init__()
        # cross-mix 投影 (Q 来自 queries，K 来自 features 的全局均值)
        self.c_q = nn.Linear(hidden, hidden, bias=False)
        self.c_k = nn.Linear(hidden, hidden, bias=False)
        self.c_o = nn.Linear(hidden, hidden, bias=False)
        self.c_norm_q = RMSNorm(hidden, eps=eps)
        self.c_norm_f = RMSNorm(hidden, eps=eps)
        # 查询内混合 (无注意力)
        kernel = max(3, min(heads * 2 + 1, 9))
        self.self_mix = LocalInteraction(hidden, kernel_size=kernel, dilation=1,
                                         use_recurrence=True, eps=eps)
        # ffn
        self.ffn = FFN(hidden, intermediate_ratio=4.0)
        self.ffn_norm = RMSNorm(hidden, eps=eps)

    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        B, K, H = queries.shape

        # ---- 跨模态加性汇聚 (无 QK 点积) ----
        fctx = self.c_k(self.c_norm_f(features)).mean(dim=1, keepdim=True)   # (B,1,H)
        q_proj = self.c_q(self.c_norm_q(queries))                            # (B,K,H)
        ca = self.c_o(q_proj + fctx)                                         # (B,K,H)
        queries = queries + ca

        # ---- 查询内混合 (无注意力) ----
        queries = self.self_mix(queries)

        # ---- ffn ----
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries
