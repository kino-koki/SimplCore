# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""多模态入口（0.1-α）—— 把连续模态变成**软 token 拼接**进残差流.

无词表 / 无位置表架构对多模态是天然友好的：模态特征既不需要词表
（连续值直接进嵌入空间），也不需要位置插值（序列拼接即位置）。本模块
把「文本 token id + 模态特征」拼成一条序列，送进**同一个** SimplCore
block 栈——不引入注意力、不引入位置编码、不改动 ``SimplCoreModel``
本体（组合复用 ``tok_emb`` / ``blocks`` / ``ln_f`` / ``head``）。

结构
----
* :class:`ImageAdapter`：图像 (B,C,H,W) → patchify Conv2d → (B,P,dim) 软 token；
* :class:`AudioAdapter`：log-mel (B,n_mels,T) → Conv1d 帧下采样 → (B,T',dim)；
* :class:`SimplCoreMultimodal`：``[模态软 token ‖ 分隔 ‖ 文本 token]`` 拼接
  前向；``generate`` 走基座 O(1) 流式采样（模态前缀先 prefill）。

诚实边界
--------
* 模态 adapter **未经任何真实数据训练验证**——本模块先把接口、形状、
  梯度通路与"无模态时逐位等于纯文本路径"立成守护测试；
* 流式推理暂不支持模态前缀（``stream_step`` 只接受 token id；
  「soft token 进流式」需要基座增加 embeds 入口，列为后续工作）；
* 零初始化纪律在此**不适用**：模态 adapter 是输入端（不是残差流注入），
  零初始化会让模态 token 无信息；用标准初始化 + LayerNorm。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .model import SimplCoreModel

__all__ = ["ImageAdapter", "AudioAdapter", "SimplCoreMultimodal"]


class ImageAdapter(nn.Module):
    """图像 → 软 token：patchify 卷积（每 patch 一个 token，无重叠）。"""

    def __init__(self, dim: int, patch_size: int = 16, in_channels: int = 3) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = int(patch_size)
        self.patchify = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                                  stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, C, H, W) → (B, P, dim)，P = (H/patch)·(W/patch)。"""
        if pixels.dim() != 4:
            raise ValueError(f"expect (B,C,H,W), got {tuple(pixels.shape)}")
        t = self.patchify(pixels)                       # (B, dim, H/p, W/p)
        B, D, H, W = t.shape
        t = t.reshape(B, D, H * W).transpose(1, 2)      # (B, P, dim)
        return self.norm(t)


class AudioAdapter(nn.Module):
    """log-mel 谱 → 软 token：两层 Conv1d，第二层 stride 下采样帧率。"""

    def __init__(self, dim: int, n_mels: int = 80, downsample: int = 4) -> None:
        super().__init__()
        if downsample <= 0:
            raise ValueError(f"downsample must be positive, got {downsample}")
        self.downsample = int(downsample)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, dim, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=self.downsample,
                      stride=self.downsample),
            nn.GroupNorm(1, dim),
        )

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """mels: (B, n_mels, T) → (B, T', dim)，T' = T // downsample。"""
        if mels.dim() != 3:
            raise ValueError(f"expect (B,n_mels,T), got {tuple(mels.shape)}")
        t = self.conv(mels)                             # (B, dim, T')
        return t.transpose(1, 2)                        # (B, T', dim)


class SimplCoreMultimodal(nn.Module):
    """文本 + 模态（图像/音频）统一前向：软 token 拼接 → SimplCore block 栈.

    无模态输入时与 ``SimplCoreModel.forward`` **逐位一致**（同一 embedding /
    block 栈 / head，测试锁定）。序列顺序：模态软 token 在前（因果前缀），
    文本在后；分隔向量是可学参数。
    """

    def __init__(self, model: SimplCoreModel, patch_size: int = 16,
                 n_mels: int = 80, audio_downsample: int = 4) -> None:
        super().__init__()
        self.base = model
        d = int(model.tok_emb.embedding_dim)
        self.image = ImageAdapter(d, patch_size=patch_size,
                                  in_channels=3)
        self.audio = AudioAdapter(d, n_mels=n_mels,
                                  downsample=audio_downsample)
        # 模态 → 文本 的分隔向量（可学；normal 0.02 —— 输入端标准初始化）
        self.sep = nn.Parameter(torch.randn(d) * 0.02)

    # -- 序列构造 -------------------------------------------------------------
    def embed_sequence(self, idx: torch.Tensor,
                       image: Optional[torch.Tensor] = None,
                       audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """文本 id (B,L) + 模态特征 → (B, L_total, dim) 软 token 序列。"""
        text = self.base.tok_emb(idx)                   # (B, L, dim)
        pre: List[torch.Tensor] = []
        if image is not None:
            pre.append(self.image(image))
        if audio is not None:
            pre.append(self.audio(audio))
        if not pre:
            return text
        sep = self.sep.to(text.dtype).view(1, 1, -1).expand(text.shape[0], 1, -1)
        return torch.cat(pre + [sep, text], dim=1)

    # -- 前向（复用基座 block 栈）----------------------------------------------
    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None):
        """与 ``SimplCoreModel.forward`` 同约定（targets 同位对齐，**不自动
        shift**——见恒等陷阱文档）；模态 token 占据序列前缀，``targets``
        的长度必须等于 ``idx``（模态前缀不参与 loss，用 ``ignore_index``
        的思想在调用方构造，或直接传完整长度标签）。"""
        x = self.embed_sequence(idx, image=image, audio=audio)
        for blk in self.base.blocks:
            x = blk(x, None)
        logits = self.base.head(self.base.ln_f(x))
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100)
            return logits, loss
        return logits

    # -- 生成（模态前缀 prefill + O(1) 流式采样）--------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 image: Optional[torch.Tensor] = None,
                 audio: Optional[torch.Tensor] = None,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """模态条件下的自回归采样：模态软 token 以 ``inputs_embeds`` 逐帧
        prefill 进 O(1) 状态（真 O(1)，无近似），文本走基座流式递推。"""
        base = self.base
        B = idx.shape[0]
        pre: List[torch.Tensor] = []
        if image is not None:
            pre.append(self.image(image))
        if audio is not None:
            pre.append(self.audio(audio))
        was_training = base.training
        base.eval()
        state = base.init_stream(B, device=idx.device,
                                 dtype=base.tok_emb.weight.dtype)
        if pre:
            soft = torch.cat(pre + [self.sep.to(pre[0].dtype)
                                    .view(1, 1, -1).expand(B, 1, -1)], dim=1)
            for i in range(soft.shape[1]):            # 模态前缀：O(P) 一步一帧
                logits, state = base.stream_step(
                    None, state, inputs_embeds=soft[:, i:i + 1])
        logits = None
        for t in range(idx.shape[1]):
            logits, state = base.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = base._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = base.stream_step(nxt, state)
        if was_training:
            base.train()
        return out