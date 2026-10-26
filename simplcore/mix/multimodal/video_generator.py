# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
视频生成器模块 (扩散模型)
========================

基于 3D UNet 去噪网络 + 时序/空间注意力的潜空间视频扩散生成器。

核心组件:
  1. SinusoidalPositionEmbeddings - 时间步正弦编码
  2. ResidualBlock               - 残差块 (含时间步条件注入)
  3. TemporalAttention            - 时序自注意力 (帧间建模)
  4. SpatialAttention             - 空间自注意力 (帧内建模)
  5. CrossAttention               - 文本条件交叉注意力
  6. Downsample / Upsample        - 空间下采样 / 上采样
  7. UNet3D                       - 3D UNet 去噪网络
     - 编码器-中间层-解码器结构
     - 跳跃连接 (U-Net)
     - 时间步条件注入 (通过 ResidualBlock 的 time MLP)
     - 文本条件注入 (通过 CrossAttention)
  8. NoiseScheduler               - 噪声调度器 (linear / cosine)
     - add_noise(x, t) -> noisy_x
     - step(x, t, model_output) -> denoised_x (DDIM eta=0)
  9. VideoGenerator               - 主类
     - forward(noise, timesteps, text_emb) -> denoised_latent
     - generate(text_emb, num_frames) -> video_latent
     - sample(text_emb) -> generated_frames

潜空间布局约定:
  视频 latent 形状为 ``[B, latent_dim, F, H, W]``
  其中 F 为帧数, H/W 为潜空间空间分辨率 (默认 32x32), latent_dim=4。
  空间卷积在 [B*F, C, H, W] 上执行; 注意力按需重塑维度。

底层张量库为 ``simplcore.tensor`` (双后端: 自研 NumPy 引擎 / PyTorch),
完全脱离对 PyTorch 的硬依赖, 与 SimplCore Mix 其余模块保持一致。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.mix.core import (
    SCModule,
    SCLinear,
    SCDropout,
    SCParameter,
    SCModuleList,
    SCSequential,
    SCRMSNorm,
)
from .config import MultimodalConfig


# ============================================================
# 后端桥接: 卷积 / 归一化 / 上采样（青小团自研实现，零第三方框架依赖）
# ------------------------------------------------------------
# 早期版本的 torch 后端桥接已被彻底移除；本模块完全基于自研张量算子，
# 提供可训练的前向实现，满足视频生成的采样循环。
# ============================================================
class Conv2d(nn.Module):
    """2D 卷积（自研实现，可训练）。

    使用 Tensor.conv2d 算子，Kaiming 风格初始化。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_ch * kernel_size * kernel_size
        std = 1.0 / math.sqrt(fan_in)
        self.weight = SCParameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * std)
        self.bias = SCParameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.conv2d(self.weight, self.bias,
                        stride=self.stride, padding=self.padding)


class GroupNorm(nn.Module):
    """分组归一化（自研实现，可训练）。"""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        # 防止通道数不足时退化为单组
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = SCParameter(torch.ones(num_channels))
        self.bias = SCParameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        N, C, H, W = x.shape
        G = self.num_groups
        xg = x.reshape(N, G, C // G, H, W)
        mean = xg.mean(dim=(2, 3, 4), keepdim=True)
        var = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        xn = (xg - mean) / (var + self.eps).sqrt()
        xn = xn.reshape(N, C, H, W)
        return xn * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)


def interpolate_nearest(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """最近邻上采样（空间维度 x factor），通过 repeat_interleave 实现。"""
    r = int(factor)
    x = x.repeat_interleave(r, dim=2)
    x = x.repeat_interleave(r, dim=3)
    return x


def _num_groups(channels: int, preferred: int = 8) -> int:
    """选择不超过 channels 且能整除 channels 的分组数。"""
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


# ============================================================
# 时间步正弦位置编码
# ============================================================
class SinusoidalPositionEmbeddings(SCModule):
    """时间步正弦/余弦位置编码。

    将整数/浮点时间步 t 映射为 ``[B, dim]`` 的连续向量,
    供后续 time MLP 进一步变换后注入 ResidualBlock。

    Args:
        dim: 输出维度 (实际 sin/cos 各占 dim//2)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.half = dim // 2

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            time: [B] 时间步 (float 或 int)
        Returns:
            [B, dim]
        """
        device = time.data if hasattr(time, "data") else None
        half = self.half
        max_period = 10000.0
        # 频率向量 [half]
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half).to(torch.float32) / half
        )
        # time: [B] -> [B, 1]; args: [B, half]
        t = time.to(torch.float32) if hasattr(time, "to") else time
        args = t.reshape(-1, 1) * freqs.reshape(1, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # 处理奇数维度
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(emb.shape[0], self.dim - emb.shape[-1])
            emb = torch.cat([emb, pad], dim=-1)
        return emb


# ============================================================
# 多头注意力 (支持自注意力与交叉注意力)
# ============================================================
class Attention(SCModule):
    """多头注意力, 兼容自注意力 (context=None) 与交叉注意力 (context=text_emb)。

    Args:
        dim:     查询维度
        n_heads: 注意力头数
        head_dim: 每头维度 (None 则 dim//n_heads)
        dim_kv:  键/值维度 (None 则等于 dim)
        dropout: 注意力权重 dropout
    """

    def __init__(self, dim: int, n_heads: int = 8,
                 head_dim: Optional[int] = None,
                 dim_kv: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else max(1, dim // n_heads)
        self.inner = self.n_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        kv_dim = dim_kv if dim_kv is not None else dim
        self.to_q = SCLinear(dim, self.inner, bias=False)
        self.to_k = SCLinear(kv_dim, self.inner, bias=False)
        self.to_v = SCLinear(kv_dim, self.inner, bias=False)
        self.to_out = SCSequential(
            SCLinear(self.inner, dim), SCDropout(dropout)
        )

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向计算。

        Args:
            x:       [BN, Tq, dim] 查询序列
            context: [BN, Tk, kv_dim] 键值序列 (None 则自注意力)
        Returns:
            [BN, Tq, dim]
        """
        bn, tq, _ = x.shape
        h = self.n_heads
        q = self.to_q(x).reshape(bn, tq, h, self.head_dim).transpose(1, 2)
        ctx = x if context is None else context
        tk = ctx.shape[1]
        k = self.to_k(ctx).reshape(bn, tk, h, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).reshape(bn, tk, h, self.head_dim).transpose(1, 2)
        # [bn, h, tq, tk]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [bn, h, tq, head_dim]
        out = out.transpose(1, 2).reshape(bn, tq, self.inner)
        return self.to_out(out)


# ============================================================
# 空间自注意力 (帧内建模)
# ============================================================
class SpatialAttention(SCModule):
    """帧内空间自注意力。

    输入 [B, C, F, H, W] -> 每帧在 [B*F, H*W, C] 上做自注意力 -> 残差相加。
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = SCRMSNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        # [B,C,F,H,W] -> [B,F,H,W,C] -> [B*F, H*W, C]
        tok = x.permute(0, 2, 3, 4, 1).reshape(B * F, H * W, C)
        tok = self.norm(tok)
        tok = self.attn(tok)
        # 还原
        tok = tok.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        return x + tok


# ============================================================
# 时序自注意力 (帧间建模)
# ============================================================
class TemporalAttention(SCModule):
    """帧间时序自注意力。

    输入 [B, C, F, H, W] -> 每个空间位置在 [B*H*W, F, C] 上做自注意力 -> 残差相加。
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = SCRMSNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        # [B,C,F,H,W] -> [B,H,W,F,C] -> [B*H*W, F, C]
        tok = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, F, C)
        tok = self.norm(tok)
        tok = self.attn(tok)
        # 还原 [B,H,W,F,C] -> [B,C,F,H,W]
        tok = tok.reshape(B, H, W, F, C).permute(0, 4, 3, 1, 2)
        return x + tok


# ============================================================
# 文本条件交叉注意力
# ============================================================
class CrossAttention(SCModule):
    """文本条件交叉注意力。

    查询来自视频 latent 的所有时空 token, 键/值来自文本嵌入。

    Args:
        dim:      视频 latent 通道维度
        text_dim: 文本嵌入维度
        n_heads:  注意力头数
        dropout:  dropout 概率
    """

    def __init__(self, dim: int, text_dim: int,
                 n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = SCRMSNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads,
                              dim_kv=text_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        # [B,C,F,H,W] -> [B, F*H*W, C]
        q = x.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        q = self.norm(q)
        out = self.attn(q, context=text_emb)  # [B, F*H*W, C]
        out = out.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        return x + out


# ============================================================
# 残差块 (含时间步条件)
# ============================================================
class ResidualBlock(SCModule):
    """扩散模型残差块: GroupNorm-SiLU-Conv -> 时间注入 -> GroupNorm-SiLU-Conv -> 残差。

    Args:
        in_ch:    输入通道
        out_ch:    输出通道
        time_dim:  时间嵌入维度 (来自全局 time MLP)
        dropout:   dropout 概率
        num_groups: GroupNorm 分组数
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.0, num_groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        ng1 = _num_groups(in_ch, num_groups)
        ng2 = _num_groups(out_ch, num_groups)
        self.norm1 = GroupNorm(ng1, in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, 3, padding=1)
        # 时间嵌入投影到 out_ch
        self.time_mlp = SCSequential(
            SCLinear(time_dim, out_ch), nn.SiLU(), SCLinear(out_ch, out_ch)
        )
        self.norm2 = GroupNorm(ng2, out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = SCDropout(dropout)
        if in_ch != out_ch:
            self.residual_conv = Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            x:     [B, C, F, H, W]
            t_emb: [B, time_dim]
        Returns:
            [B, out_ch, F, H, W]
        """
        B, C, F, H, W = x.shape
        # 5D -> 4D: [B*F, C, H, W]
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

        h = self.conv1(torch.silu(self.norm1(x2d)))  # [B*F, out_ch, H, W]

        # 时间嵌入: [B, out_ch] -> [B*F, out_ch, 1, 1]
        t = self.time_mlp(t_emb)  # [B, out_ch]
        och = t.shape[1]
        t = t.reshape(B, 1, och, 1, 1)
        t = t.expand(B, F, och, 1, 1).reshape(B * F, och, 1, 1)
        h = h + t

        h = self.conv2(torch.silu(self.norm2(h)))
        h = self.dropout(h)

        res = self.residual_conv(x2d)
        out = h + res  # [B*F, out_ch, H, W]

        # 4D -> 5D
        oH, oW = out.shape[2], out.shape[3]
        out = out.reshape(B, F, self.out_ch, oH, oW).permute(0, 2, 1, 3, 4)
        return out


# ============================================================
# 下采样 / 上采样
# ============================================================
class Downsample(SCModule):
    """步幅 2 卷积下采样 (保持通道数)。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        out = self.conv(x2d)
        oc, oH, oW = out.shape[1], out.shape[2], out.shape[3]
        return out.reshape(B, F, oc, oH, oW).permute(0, 2, 1, 3, 4)


class Upsample(SCModule):
    """最近邻上采样 x2 + 卷积。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        x2d = interpolate_nearest(x2d, 2)
        out = self.conv(x2d)
        oc, oH, oW = out.shape[1], out.shape[2], out.shape[3]
        return out.reshape(B, F, oc, oH, oW).permute(0, 2, 1, 3, 4)


# ============================================================
# UNet 基本块: 残差 + (可选) 空间/时序/交叉注意力
# ============================================================
class UNetBlock(SCModule):
    """一个残差块 + 可选的注意力组合。

    Args:
        in_ch, out_ch: 通道
        time_dim:       时间嵌入维度
        has_attn:       是否启用注意力
        n_heads:        注意力头数
        text_dim:       文本嵌入维度 (交叉注意力用)
        dropout:        dropout
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 has_attn: bool, n_heads: int, text_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.has_attn = has_attn
        self.res = ResidualBlock(in_ch, out_ch, time_dim, dropout=dropout)
        if has_attn:
            self.spatial = SpatialAttention(out_ch, n_heads, dropout)
            self.temporal = TemporalAttention(out_ch, n_heads, dropout)
            self.cross = CrossAttention(out_ch, text_dim, n_heads, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        if self.has_attn:
            x = self.spatial(x)
            x = self.temporal(x)
            x = self.cross(x, text_emb)
        return x


# ============================================================
# 3D UNet 去噪网络
# ============================================================
class UNet3D(SCModule):
    """3D UNet 去噪网络 (编码器-中间层-解码器 + 跳跃连接)。

    处理 5D 视频 latent [B, latent_dim, F, H, W]。
    空间卷积在 [B*F, C, H, W] 上执行; 帧间/帧内建模通过注意力模块完成。

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.video_unet_dim
        dim_mults: Tuple[int, ...] = tuple(cfg.video_unet_dim_mults)
        n_layers = cfg.video_unet_n_layers
        latent_dim = cfg.video_latent_dim
        n_heads = cfg.video_temporal_heads
        text_dim = cfg.video_text_dim
        dropout = cfg.video_dropout

        self.latent_dim = latent_dim
        self.dim = dim

        # ── 时间步嵌入: 正弦编码 -> MLP ──
        self.time_embed = SinusoidalPositionEmbeddings(dim)
        self.time_mlp = SCSequential(
            SCLinear(dim, dim * 4), nn.SiLU(), SCLinear(dim * 4, dim)
        )

        # 各阶段通道数
        channels = [dim * m for m in dim_mults]
        # 注意力门控: 仅在空间分辨率 <= attn_res 时启用注意力 (控制显存)
        attn_res = 16
        self.attn_res = attn_res

        # ── 初始卷积 ──
        self.init_conv = Conv2d(latent_dim, channels[0], 3, padding=1)

        # ── 编码器 (下采样阶段) ──
        # down_blocks[i] = (ModuleList[UNetBlock], Downsample or None)
        self.down_blocks = SCModuleList()
        self.down_samples = SCModuleList()
        cur_res = cfg.video_latent_size[0]
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            blocks = SCModuleList()
            for j in range(n_layers):
                ic = in_ch if j == 0 else out_ch
                has_attn = cur_res <= attn_res
                blocks.append(UNetBlock(ic, out_ch, dim, has_attn,
                                        n_heads, text_dim, dropout))
            self.down_blocks.append(blocks)
            in_ch = out_ch
            if i < len(channels) - 1:
                self.down_samples.append(Downsample(out_ch))
                cur_res = cur_res // 2
            else:
                self.down_samples.append(nn.Identity())

        # ── 中间层 ──
        mid_ch = channels[-1]
        mid_attn = cur_res <= attn_res
        self.mid_block1 = UNetBlock(mid_ch, mid_ch, dim, mid_attn,
                                    n_heads, text_dim, dropout)
        self.mid_block2 = UNetBlock(mid_ch, mid_ch, dim, mid_attn,
                                    n_heads, text_dim, dropout)

        # ── 解码器 (上采样阶段) ──
        # up_blocks[i] = (ModuleList[UNetBlock], Upsample or None)
        # 顺序与 down_blocks 相反; 每个上采样块输入 = 解码特征 ++ 对应跳跃特征
        self.up_blocks = SCModuleList()
        self.up_samples = SCModuleList()
        rev_channels = list(reversed(channels))
        rev_res = []
        r = cur_res
        # 计算每个上采样阶段对应的空间分辨率 (与编码器镜像)
        for _ in range(len(channels)):
            rev_res.append(r)
            r = r * 2
        # rev_res 对应 reversed(channels) 的分辨率; 去掉超出原始分辨率的部分
        in_ch_up = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            blocks = SCModuleList()
            res_here = rev_res[i]
            has_attn = res_here <= attn_res
            for j in range(n_layers):
                ic = in_ch_up * 2  # 拼接跳跃连接
                blocks.append(UNetBlock(ic, out_ch, dim, has_attn,
                                        n_heads, text_dim, dropout))
                in_ch_up = out_ch
            self.up_blocks.append(blocks)
            if i < len(rev_channels) - 1:
                self.up_samples.append(Upsample(out_ch))
            else:
                self.up_samples.append(nn.Identity())

        # ── 输出层 ──
        self.out_norm = GroupNorm(_num_groups(channels[0], 8), channels[0])
        self.out_conv = Conv2d(channels[0], latent_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        """前向去噪 (预测噪声 eps)。

        Args:
            x:         [B, latent_dim, F, H, W] 噪声 latent
            timesteps: [B] 时间步
            text_emb:  [B, L, text_dim] 文本条件嵌入
        Returns:
            [B, latent_dim, F, H, W] 预测噪声
        """
        # 时间嵌入
        t = self.time_embed(timesteps)
        t_emb = self.time_mlp(t)  # [B, dim]

        # 初始卷积
        B, _, F, H, W = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * F, self.latent_dim, H, W)
        x2d = self.init_conv(x2d)
        oc, H2, W2 = x2d.shape[1], x2d.shape[2], x2d.shape[3]
        h = x2d.reshape(B, F, oc, H2, W2).permute(0, 2, 1, 3, 4)  # [B, C0, F, H, W]

        # ── 编码器: 收集跳跃连接 ──
        skips: List[torch.Tensor] = []
        for blocks, sample in zip(self.down_blocks, self.down_samples):
            for blk in blocks:
                h = blk(h, t_emb, text_emb)
            skips.append(h)
            if not isinstance(sample, nn.Identity):
                h = sample(h)

        # ── 中间层 ──
        h = self.mid_block1(h, t_emb, text_emb)
        h = self.mid_block2(h, t_emb, text_emb)

        # ── 解码器: 消费跳跃连接 ──
        for blocks, sample in zip(self.up_blocks, self.up_samples):
            # 弹出对应跳跃 (编码器最后压入的先消费)
            skip = skips.pop()
            # 拼接: 在通道维拼接 [B, C_skip + C_h, F, H, W]
            # 注意: 空间尺寸需一致 (上采样已对齐)
            if h.shape != skip.shape:
                # 若尺寸不一致 (边缘情况), 对齐到 h 的尺寸
                skip = _align_to(skip, h.shape)
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, t_emb, text_emb)
            if not isinstance(sample, nn.Identity):
                h = sample(h)

        # ── 输出 ──
        B, C, Ff, Hh, Ww = h.shape
        h2d = h.permute(0, 2, 1, 3, 4).reshape(B * Ff, C, Hh, Ww)
        h2d = self.out_norm(h2d)
        h2d = torch.silu(h2d)
        h2d = self.out_conv(h2d)
        oc2, Hh2, Ww2 = h2d.shape[1], h2d.shape[2], h2d.shape[3]
        out = h2d.reshape(B, Ff, oc2, Hh2, Ww2).permute(0, 2, 1, 3, 4)
        return out


def _align_to(x: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """裁剪/对齐 x 到目标形状 (空间维度)。"""
    if x.shape == target_shape:
        return x
    # 仅对齐空间维度 (最后两个)
    H = target_shape[-2]
    W = target_shape[-1]
    return x[..., :H, :W]


# ============================================================
# 噪声调度器 (linear / cosine)
# ============================================================
class NoiseScheduler(SCModule):
    """扩散模型噪声调度器。

    支持 linear 与 cosine (Nichol & Dhariwal, 2021) 两种 beta 调度。
    step() 采用 DDIM (eta=0) 确定性更新, 适合 50 步快速采样。

    Args:
        num_train_steps: 训练步数 T (默认 1000)
        schedule:        "linear" 或 "cosine"
        beta_start:      linear 调度起始 beta
        beta_end:        linear 调度终止 beta
        s:                cosine 调度偏移
    """

    def __init__(self, num_train_steps: int = 1000,
                 schedule: str = "cosine",
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 s: float = 0.008):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.schedule = schedule.lower()

        # ── 计算 betas (纯 python/math, 双后端一致) ──
        if self.schedule == "linear":
            betas = [
                beta_start + (beta_end - beta_start) * i / (num_train_steps - 1)
                for i in range(num_train_steps)
            ]
        elif self.schedule == "cosine":
            steps = num_train_steps + 1
            x = [i / (steps - 1) for i in range(steps)]
            acp_raw = [math.cos((xi + s) / (1.0 + s) * math.pi / 2.0) ** 2
                       for xi in x]
            acp0 = acp_raw[0]
            acp_norm = [a / acp0 for a in acp_raw]
            betas = [1.0 - acp_norm[i + 1] / acp_norm[i]
                     for i in range(num_train_steps)]
            betas = [min(max(b, 0.0), 0.999) for b in betas]
        else:
            raise ValueError(f"未知噪声调度: {schedule} (可选 linear / cosine)")

        # ── 计算 alphas / alpha_cumprod / alpha_cumprod_prev ──
        alphas = [1.0 - b for b in betas]
        acp = [0.0] * num_train_steps
        acp[0] = alphas[0]
        for i in range(1, num_train_steps):
            acp[i] = acp[i - 1] * alphas[i]
        # prev: 在前面补 1.0, 使 t=0 时 prev=1.0 (恢复 x0)
        acp_prev = [1.0] + acp[:-1]

        self._betas = betas
        self._alphas = alphas
        self._acp = acp
        self._acp_prev = acp_prev

        # 同时以 buffer 形式保存张量 (便于序列化 / 设备迁移)
        self.register_buffer("betas_t", torch.tensor(betas))
        self.register_buffer("alphas_cumprod", torch.tensor(acp))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(acp_prev))

    # ── 辅助: 提取 python int 时间步 ──
    @staticmethod
    def _to_int(t: Union[int, float, torch.Tensor]) -> int:
        if isinstance(t, (int, float)):
            return int(t)
        # tensor
        if hasattr(t, "flatten"):
            t = t.flatten()
        if hasattr(t, "item"):
            try:
                return int(t.item())
            except Exception:
                pass
        if hasattr(t, "data"):
            return int(t.data.flat[0])
        return int(t)

    def add_noise(self, x: torch.Tensor, t: Union[int, torch.Tensor],
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向加噪: x_t = sqrt(acp_t) * x_0 + sqrt(1-acp_t) * noise。

        Args:
            x:     [B, latent_dim, F, H, W] 原始 latent x_0
            t:     时间步 (int 或 tensor, 批次内统一)
            noise: 可选噪声, None 则随机采样
        Returns:
            noisy_x [B, latent_dim, F, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x)
        t_int = self._to_int(t)
        t_int = max(0, min(t_int, self.num_train_steps - 1))
        acp_t = self._acp[t_int]
        s_acp = math.sqrt(acp_t)
        s_1m_acp = math.sqrt(max(1.0 - acp_t, 0.0))
        return s_acp * x + s_1m_acp * noise

    def step(self, x: torch.Tensor, t: Union[int, torch.Tensor],
             model_output: torch.Tensor) -> torch.Tensor:
        """DDIM (eta=0) 确定性反向更新一步。

        pred_x0 = (x_t - sqrt(1-acp_t) * eps) / sqrt(acp_t)
        x_{t-1} = sqrt(acp_prev) * pred_x0 + sqrt(1-acp_prev) * eps

        Args:
            x:             [B, latent_dim, F, H, W] 当前 x_t
            t:             时间步 (int 或 tensor, 批次内统一)
            model_output:  [B, latent_dim, F, H, W] 模型预测的噪声 eps
        Returns:
            denoised_x [B, latent_dim, F, H, W] (x_{t-1})
        """
        t_int = self._to_int(t)
        t_int = max(0, min(t_int, self.num_train_steps - 1))
        acp_t = self._acp[t_int]
        acp_prev = self._acp_prev[t_int]
        eps = 1e-8
        s_acp = math.sqrt(acp_t) + eps
        s_1m_acp = math.sqrt(max(1.0 - acp_t, 0.0))
        s_acp_prev = math.sqrt(acp_prev)
        s_1m_acp_prev = math.sqrt(max(1.0 - acp_prev, 0.0))

        pred_x0 = (x - s_1m_acp * model_output) / s_acp
        return s_acp_prev * pred_x0 + s_1m_acp_prev * model_output


# ============================================================
# 视频生成器主类
# ============================================================
class VideoGenerator(SCModule):
    """基于扩散模型的视频生成器。

    组合 UNet3D 去噪网络与 NoiseScheduler, 提供训练前向与采样接口。

    Args:
        cfg: MultimodalConfig 配置对象

    接口:
        forward(noise, timesteps, text_emb) -> denoised_latent
            训练前向: 给定噪声 latent 与时间步, 预测噪声。
        generate(text_emb, num_frames) -> video_latent
            完整采样: 从纯噪声出发, 反向去噪生成视频 latent。
        sample(text_emb) -> generated_frames
            默认帧数的完整采样, 返回 [B, F, latent_dim, H, W]。
    """

    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.video_latent_dim
        self.latent_size: Tuple[int, int] = tuple(cfg.video_latent_size)
        self.max_frames = cfg.video_max_frames
        self.num_train_steps = cfg.video_diffusion_steps
        self.num_sample_steps = cfg.video_diffusion_sampling_steps

        # 去噪网络
        self.unet = UNet3D(cfg)
        # 噪声调度器 (默认 cosine)
        self.scheduler = NoiseScheduler(
            num_train_steps=self.num_train_steps,
            schedule="cosine",
        )

    # ------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------
    def forward(self, noise: torch.Tensor, timesteps: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        """前向: 预测噪声 (训练目标)。

        Args:
            noise:     [B, latent_dim, F, H, W] 噪声 latent (x_t)
            timesteps: [B] 时间步
            text_emb:  [B, L, text_dim] 文本条件嵌入
        Returns:
            [B, latent_dim, F, H, W] 预测噪声 (denoised_latent 方向)
        """
        return self.unet(noise, timesteps, text_emb)

    # ------------------------------------------------------------
    # 完整采样循环
    # ------------------------------------------------------------
    def _ddim_timesteps(self) -> List[int]:
        """生成 DDIM 采样的时间步子序列 (降序)。"""
        T = self.num_train_steps
        S = self.num_sample_steps
        if S <= 0:
            S = 1
        if S >= T:
            ts = list(range(T - 1, -1, -1))
        else:
            ts = [int(round((T - 1) * i / (S - 1))) for i in range(S)]
        ts = sorted(set(ts), reverse=True)
        return ts

    def _sample_loop(self, text_emb: torch.Tensor,
                     num_frames: int) -> torch.Tensor:
        """从纯噪声出发执行完整 DDIM 反向采样。

        Args:
            text_emb:   [B, L, text_dim]
            num_frames: 生成帧数
        Returns:
            video_latent [B, latent_dim, F, H, W]
        """
        B = text_emb.shape[0]
        H, W = self.latent_size
        x = torch.randn(B, self.latent_dim, num_frames, H, W)
        timesteps = self._ddim_timesteps()

        with torch.no_grad():
            for t in timesteps:
                # 批次内统一时间步
                t_batch = torch.full((B,), float(t))
                eps = self.unet(x, t_batch, text_emb)
                x = self.scheduler.step(x, t, eps)
        return x

    def generate(self, text_emb: torch.Tensor,
                 num_frames: Optional[int] = None) -> torch.Tensor:
        """生成视频 latent。

        Args:
            text_emb:   [B, L, text_dim] 文本条件嵌入
            num_frames: 帧数 (None 则使用配置最大帧数)
        Returns:
            video_latent [B, latent_dim, F, H, W]
        """
        if num_frames is None:
            num_frames = self.max_frames
        if num_frames <= 0:
            num_frames = self.max_frames
        if num_frames > self.max_frames:
            num_frames = self.max_frames
        self.eval()
        return self._sample_loop(text_emb, num_frames)

    def sample(self, text_emb: torch.Tensor) -> torch.Tensor:
        """完整采样, 返回生成的视频帧 (latent 表示)。

        Args:
            text_emb: [B, L, text_dim] 文本条件嵌入
        Returns:
            generated_frames [B, F, latent_dim, H, W]
            (帧维度提前, 便于按帧取用)
        """
        latent = self.generate(text_emb, num_frames=self.max_frames)
        # [B, latent_dim, F, H, W] -> [B, F, latent_dim, H, W]
        return latent.permute(0, 2, 1, 3, 4)


__all__ = [
    "SinusoidalPositionEmbeddings",
    "Attention",
    "SpatialAttention",
    "TemporalAttention",
    "CrossAttention",
    "ResidualBlock",
    "Downsample",
    "Upsample",
    "UNetBlock",
    "UNet3D",
    "NoiseScheduler",
    "VideoGenerator",
    "Conv2d",
    "GroupNorm",
]


# ============================================================
# 自检 / 冒烟测试 (仅直接运行本文件时执行)
# ============================================================
if __name__ == "__main__":
    cfg = MultimodalConfig()
    # 缩小规模以便快速冒烟测试
    cfg.video_unet_dim = 64
    cfg.video_unet_dim_mults = (1, 2, 4, 4)
    cfg.video_unet_n_layers = 1
    cfg.video_latent_size = (8, 8)
    cfg.video_max_frames = 4
    cfg.video_diffusion_steps = 1000
    cfg.video_diffusion_sampling_steps = 8
    cfg.video_temporal_heads = 4
    cfg.video_text_dim = 64
    cfg.video_dropout = 0.0

    gen = VideoGenerator(cfg)
    print(f"[VideoGenerator] 后端: 自研引擎 (native)")

    # 训练前向
    B, F = 1, cfg.video_max_frames
    H, W = cfg.video_latent_size
    noise = torch.randn(B, cfg.video_latent_dim, F, H, W)
    t = torch.full((B,), 500.0)
    text = torch.randn(B, 4, cfg.video_text_dim)
    eps = gen.forward(noise, t, text)
    print(f"[forward] 预测噪声形状: {tuple(eps.shape)}")

    # 采样
    frames = gen.sample(text)
    print(f"[sample]  生成帧形状: {tuple(frames.shape)}")
    print("[OK] 视频生成器自检通过。")
