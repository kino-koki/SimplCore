# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""原生 token-free 多模态模型 (SimplCoreNativeMM)。

设计目标：模型**不看 Token**，而是直接"看见"原始内容——

  * 文本  -> 渲染为字形位图 (GlyphRenderer) -> 字形编码器 -> 每字符一个连续 token；
  * 图像  -> 像素 -> 图像 ViT -> 感知重采样 -> K 个连续感知 token；
  * 视频  -> 逐帧像素 -> 视频编码器 -> 感知重采样 -> K 个连续感知 token；

三者统一投影到共享隐空间 (B, L, H)，由 MoE 推理核 (SimplCoreReasoningCore)
以**因果自回归**方式直接推理。输出同样不经过离散词表：文本由
``ContinuousTextHead`` 预测连续查询向量，再用"最近字形匹配"从字形原型库
解码出字符 (token-free 解码)；图像由 ``ContinuousImageHead`` 解码像素。

整个链路：原始内容 --(像素/字形感知)--> 连续感知 token --(MoE 推理)--> 连续输出
--(最近字形匹配 / 像素解码)--> 字符 / 图像。全程无 subword 分词、无 token id。
"""
from __future__ import annotations

import string
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import SimplCoreConfig
from .layers import RMSNorm, init_weights
from .perception import (
    GlyphRenderer, GlyphEncoder, ImageEncoder, VideoEncoder, PerceptualResampler,
)
from .exceptions import ModelError


# 默认字符集 (用于字形原型库 / 最近字形解码)。覆盖 ASCII 可见字符 + 常用汉字。
_DEFAULT_CN = list("的一是在不了有和人这中大来上国个到说时为来分生对于就那得着下")
DEFAULT_CHARSET = list(string.printable.strip()) + _DEFAULT_CN


class NativeInput:
    """原生多模态输入容器 (承载**原始**内容，而非 token)。

    Attributes:
        text: 提示文本 (str 或 List[str]，批内每样本一个字符串)。
        images: 可选图像张量 (B, 3, H, W) 或 None。
        videos: 可选视频张量 (B, T, 3, H, W) 或 None。
        target_text: 训练目标文本 (同 text 格式)。训练时提供。
    """

    def __init__(
        self,
        text: Optional[str | List[str]] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        target_text: Optional[str | List[str]] = None,
    ) -> None:
        self.text = text
        self.images = images
        self.videos = videos
        self.target_text = target_text

    @property
    def batch_size(self) -> int:
        if self.images is not None:
            return self.images.shape[0]
        if self.videos is not None:
            return self.videos.shape[0]
        if isinstance(self.text, list):
            return len(self.text)
        return 1

    def texts_as_list(self) -> List[str]:
        if self.text is None:
            return [""] * self.batch_size
        if isinstance(self.text, str):
            return [self.text] * self.batch_size
        return list(self.text)


# ----------------------------------------------------------------------
# MoE 推理核：消费连续嵌入，无 token embedding / lm_head
# ----------------------------------------------------------------------
class SimplCoreReasoningCore(nn.Module):
    """MoE 推理核 (连续输入版本)。

    与 ``SimplCoreMoEModel`` 的区别：不接收 ``input_ids``，直接接收连续嵌入
    序列 (B, L, H)；不含 token 嵌入与离散 LM Head；聚合各层 MoE 辅助损失。

    支持两种模式：
      * ``streaming=False``（默认）：全量因果自回归，KV 缓存增量前向；
      * ``streaming=True``：长上下文流式核心——序列切成固定 ``chunk_size`` 逐块
        流过各层，跨块仅传递固定大小的 ``StreamState`` (RFP 频域 + DMC 记忆 +
        窗口 KV)，三者均与序列长度 n 无关，从而 1M 上下文翻倍仅增常数显存。
    """

    def __init__(self, config: SimplCoreConfig, streaming: bool = False) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.streaming = streaming
        H = config.hidden_size
        # 统一使用无注意力流式块 (SimplCoreStreamingBlock)：LocalInteraction→RFP(分块谱)→DMC→MoE。
        # 无论 streaming 与否，增量生成均经固定大小 StreamState，彻底无 KV Cache / 无自注意力。
        from .longctx import SimplCoreStreamingBlock
        self.blocks = nn.ModuleList(
            [SimplCoreStreamingBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(H, eps=config.rms_norm_eps)
        self.apply(lambda m: init_weights(m, std=0.02))
        self._last_aux: Optional[torch.Tensor] = None
        self.gradient_checkpointing: bool = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.streaming:
            return self._forward_streaming(x, training=training)
        return self._forward_full(x, attention_mask, training=training, causal=causal)

    # ---- 全量模式 (同样走固定大小 StreamState，无 KV Cache) ----
    def _forward_full(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        training: bool,
        causal: bool,
    ) -> torch.Tensor:
        from .longctx import StreamState
        state = self._init_stream_state(x.shape[0], x.device, x.dtype)
        x, state, _ = self._run_stream_blocks(x, state, training=training)
        self._last_aux = state.aux_accum
        return self.final_norm(x)

    # ---- 流式模式：固定状态长上下文 ----
    def _forward_streaming(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        from .longctx import StreamState
        B = x.shape[0]
        state = self._init_stream_state(B, x.device, x.dtype)
        chunk_outs: List[torch.Tensor] = []
        for i in range(0, x.shape[1], self.config.chunk_size):
            seg = x[:, i:i + self.config.chunk_size, :]
            seg, state, aux = self._run_stream_blocks(seg, state, training=training)
            chunk_outs.append(seg)
        self._last_aux = state.aux_accum
        return self.final_norm(torch.cat(chunk_outs, dim=1))

    def _init_stream_state(self, batch: int, device, dtype) -> "StreamState":
        from .longctx import SimplCoreStreamingBlock, StreamState
        rfp_states: List[torch.Tensor] = []
        dmc_rt: List[object] = []
        for block in self.blocks:
            if block.rfp is not None:
                rfp_states.append(block.rfp.init_state(batch, device, dtype))
            if block.dmc is not None:
                dmc_rt.append(block.dmc._make_runtime(batch, device))
        return StreamState(rfp_spec=rfp_states, dmc_runtime=dmc_rt, chunk_idx=0)

    def _run_stream_blocks(self, x: torch.Tensor, state: "StreamState", training: bool):
        total_aux: Optional[torch.Tensor] = None
        for block in self.blocks:
            x, state, aux = block(x, state, training=training)
            if training and aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        state.aux_accum = total_aux if state.aux_accum is None else state.aux_accum + total_aux
        return x, state, total_aux

    @torch.no_grad()
    def stream_init(self, batch: int, device, dtype) -> "StreamState":
        """流式自回归：初始化跨块固定状态。"""
        return self._init_stream_state(batch, device, dtype)

    @torch.no_grad()
    def stream_step(self, x_new: torch.Tensor, state: "StreamState"
                    ) -> Tuple[torch.Tensor, "StreamState"]:
        """流式自回归：处理一个新 token 的连续嵌入 (B, 1, H)，更新并返回状态。"""
        x, state, _ = self._run_stream_blocks(x_new, state, training=False)
        return self.final_norm(x), state

    @torch.no_grad()
    def forward_incremental(
        self, x_new: torch.Tensor, past_key_values=None
    ) -> Tuple[torch.Tensor, object]:
        """增量前向（无注意力 / 无 KV Cache，基于固定大小 StreamState）。

        首调用 ``past_key_values=None`` 时初始化跨块固定状态；之后复用返回的
        ``state``（实为 ``StreamState``）作为下一轮的 ``past``。这是注意力自由
        架构的标准增量机制：状态大小恒定，不随序列增长线性膨胀。
        """
        from .longctx import StreamState
        if self.streaming:
            raise ModelError(
                "streaming 模式请使用 stream_init/stream_step，而非 forward_incremental。",
                details={"streaming": True},
            )
        state = past_key_values if isinstance(past_key_values, StreamState) else \
            self.stream_init(x_new.shape[0], x_new.device, x_new.dtype)
        x, state = self.stream_step(x_new, state)
        return x, state

    @property
    def last_aux_loss(self) -> Optional[torch.Tensor]:
        return self._last_aux


# ----------------------------------------------------------------------
# 连续输出头
# ----------------------------------------------------------------------
class ContinuousTextHead(nn.Module):
    """连续文本头：把隐状态映射到"字形查询"向量 (同字形编码器空间)。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class ContinuousImageHead(nn.Module):
    """连续图像头：把隐状态解码为像素 (默认 64x64 RGB)。"""

    def __init__(self, hidden_size: int, image_size: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.out = nn.Linear(hidden_size, 3 * image_size * image_size, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, H = h.shape
        px = self.out(h)
        return self.act(px).reshape(B, L, 3, self.image_size, self.image_size)


# ----------------------------------------------------------------------
# 原生多模态模型
# ----------------------------------------------------------------------
class SimplCoreNativeMM(nn.Module):
    """原生 token-free 多模态模型。"""

    # 控制 token 索引 (在 control_embeddings 中)
    BOS, EOS, SEP, PAD = 0, 1, 2, 3

    def __init__(self, config: SimplCoreConfig, charset: Optional[List[str]] = None,
                 streaming: bool = False) -> None:
        super().__init__()
        if not config.use_native_multimodal:
            raise ModelError(
                "SimplCoreNativeMM 需要 config.use_native_multimodal=True。",
                details={"use_native_multimodal": config.use_native_multimodal},
            )
        config.validate()
        self.config = config
        self.streaming = streaming
        H = config.hidden_size

        # ---- 感知前端 ----
        self.renderer = GlyphRenderer(
            glyph_size=config.glyph_size, channels=config.glyph_in_channels,
        )
        self.glyph_encoder = GlyphEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.video_encoder = VideoEncoder(config)

        # 各模态特征 -> 推理核维度 H 的投影
        self.glyph_proj = nn.Linear(config.glyph_hidden, H, bias=False)
        self.image_proj = nn.Linear(config.vision_hidden, H, bias=False)
        self.video_proj = nn.Linear(config.vision_hidden, H, bias=False)

        # 感知重采样 (图像/视频用；文本每字符已是紧凑 token，不压缩)
        self.resampler = PerceptualResampler(
            hidden_size=H, num_queries=config.perceptual_tokens,
            num_layers=config.perceiver_layers, num_heads=config.perceiver_heads,
            eps=config.rms_norm_eps,
        )

        # 模态类型嵌入 (text / image / video)
        self.modality_embed = nn.Parameter(torch.zeros(3, H))
        nn.init.normal_(self.modality_embed, std=0.02)
        # 控制 token 嵌入 (bos/eos/sep/pad)
        self.control_embed = nn.Embedding(4, H)

        # ---- 推理核 ----
        self.core = SimplCoreReasoningCore(config, streaming=streaming)

        # ---- 输出头 ----
        self.text_head = ContinuousTextHead(H)
        self.image_head = ContinuousImageHead(H, image_size=64)

        # ---- 字形原型库 (最近字形解码用) ----
        self.charset = charset if charset is not None else DEFAULT_CHARSET
        self._prototype_bank: Optional[torch.Tensor] = None  # (Nc, H)
        self._prototype_norm: Optional[torch.Tensor] = None

        self.apply(lambda m: init_weights(m, std=0.02))
        # 避免覆盖已专门初始化的嵌入
        nn.init.normal_(self.control_embed.weight, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

    # ------------------------------------------------------------------
    # 字形原型库构建
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_prototype_bank(self, device, dtype) -> torch.Tensor:
        if self._prototype_bank is not None and self._prototype_bank.device == device:
            return self._prototype_bank
        glyphs = self.renderer.render(self.charset).to(device)        # (Nc, C, gh, gw)
        feats = self.glyph_encoder.encode(glyphs)                      # (Nc, Pg, Dg)
        feats = feats.mean(dim=1)                                     # (Nc, Dg)
        char_proj = self.glyph_proj(feats)                            # (Nc, H)
        char_proj = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(char_proj)
        # 控制 token 原型 (bos/eos/sep/pad) 一并纳入，使模型可输出 [EOS] 终止
        ctrl = self.control_embed.weight                              # (4, H)
        ctrl = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(ctrl)
        bank = torch.cat([char_proj, ctrl], dim=0)                    # (Nc+4, H)
        self._prototype_bank = bank.to(device)
        self._prototype_chars = list(self.charset) + ["[BOS]", "[EOS]", "[SEP]", "[PAD]"]
        self._prototype_norm = bank / (bank.norm(dim=-1, keepdim=True) + 1e-8)
        return self._prototype_bank

    # ------------------------------------------------------------------
    # 文本 -> 每字符连续 token
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _text_to_tokens(self, texts: List[str], device) -> List[torch.Tensor]:
        """每个样本返回 (Nc, H) 的每字符连续 token (Nc=字符数，可为 0)。"""
        out = []
        for t in texts:
            if len(t) == 0:
                out.append(None)
                continue
            glyphs = self.renderer.render(t).to(device)               # (Nc, C, gh, gw)
            feats = self.glyph_encoder.encode(glyphs).mean(dim=1)     # (Nc, Dg)
            toks = self.glyph_proj(feats)                             # (Nc, H)
            out.append(toks)
        return out

    # ------------------------------------------------------------------
    # 组装单样本的连续序列 (B=1)
    # ------------------------------------------------------------------
    def _perceive_inputs(
        self, inp: NativeInput, include_target: bool = False
    ) -> Tuple[torch.Tensor, int, List[int], List[int]]:
        """把原始输入感知为单样本连续序列 (1, L, H)。

        Args:
            include_target: 训练时设为 True，把目标文本字符也作为 (teacher-forced)
                输入 token 追加到序列末尾。

        Returns:
            (seq (1,L,H), L, prompt_text_positions, target_text_positions)。
        """
        device = next(self.parameters()).device
        H = self.config.hidden_size
        parts: List[torch.Tensor] = []

        # 控制 BOS
        parts.append(self.control_embed(torch.tensor([self.BOS], device=device)).unsqueeze(0))  # (1,1,H)

        # 图像
        if inp.images is not None:
            img_feats = self.image_encoder.encode(inp.images.to(device))      # (B,P,D)
            img_feats = self.image_proj(img_feats)
            img_res = self.resampler(img_feats)                               # (B,K,H)
            img_res = img_res + self.modality_embed[1].view(1, 1, H)
            parts.append(img_res[:1])

        # 视频
        if inp.videos is not None:
            vid_feats = self.video_encoder.encode(inp.videos.to(device))      # (B,P,D)
            vid_feats = self.video_proj(vid_feats)
            vid_res = self.resampler(vid_feats)                               # (B,K,H)
            vid_res = vid_res + self.modality_embed[2].view(1, 1, H)
            parts.append(vid_res[:1])

        # 提示文本 (每字符一个 token)
        prompt_positions: List[int] = []
        texts = inp.texts_as_list()
        prompt_tok_raw = self._text_to_tokens(texts, device)[0]             # (Nc, H)
        if prompt_tok_raw is not None and prompt_tok_raw.shape[0] > 0:
            n_prompt = prompt_tok_raw.shape[0]
            prompt_tok = prompt_tok_raw.unsqueeze(0) + self.modality_embed[0].view(1, 1, H)
            parts.append(prompt_tok)                                         # (1, Nc, H)
            base = sum(p.shape[1] for p in parts[:-1])
            prompt_positions = list(range(base, base + n_prompt))

        # 目标文本 (仅训练时，作为 teacher-forced 输入)
        target_positions: List[int] = []
        if include_target and inp.target_text is not None:
            tgt_texts = inp.target_text if isinstance(inp.target_text, str) else inp.target_text[0]
            tgt_tok_raw = self._text_to_tokens([tgt_texts], device)[0]       # (Mt, H)
            if tgt_tok_raw is not None and tgt_tok_raw.shape[0] > 0:
                n_tgt = tgt_tok_raw.shape[0]
                tgt_tok = tgt_tok_raw.unsqueeze(0) + self.modality_embed[0].view(1, 1, H)
                parts.append(tgt_tok)                                        # (1, Mt, H)
                base = sum(p.shape[1] for p in parts[:-1])
                target_positions = list(range(base, base + n_tgt))

        seq = torch.cat(parts, dim=1)                                        # (1, L, H)
        return seq, seq.shape[1], prompt_positions, target_positions

    # ------------------------------------------------------------------
    # 前向 (训练 / 推理)
    # ------------------------------------------------------------------
    def forward(
        self,
        inp: NativeInput,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        include_target = training and inp.target_text is not None
        seq, L, ppos, tpos = self._perceive_inputs(inp, include_target=include_target)  # (1, L, H)
        # 因果注意力 (自回归)；无 padding
        hidden = self.core(seq, attention_mask=None, training=training, causal=True)

        result: Dict[str, torch.Tensor] = {"hidden": hidden, "aux_loss": self.core.last_aux_loss
                                            if training else torch.zeros((), device=seq.device)}
        # 文本查询
        queries = self.text_head(hidden)                                     # (1, L, H)
        result["text_queries"] = queries

        # 训练时若有 target_text，计算最近字形回归损失
        if training and inp.target_text is not None:
            loss = self._compute_text_loss(hidden, inp, ppos, tpos)
            result["loss"] = loss
        return result

    def _compute_text_loss(
        self, hidden, inp, prompt_text_positions, target_text_positions
    ) -> torch.Tensor:
        """teacher-forcing 文本回归损失：每个预测位置预测下一个字符的字形嵌入。"""
        device = hidden.device

        # 目标字符序列 (按 batch 内第一个样本；单样本训练)
        target = inp.target_text if isinstance(inp.target_text, str) else inp.target_text[0]
        # 目标字符的字形嵌入 (作为回归目标，detach)
        tgt_glyphs = self.renderer.render(target).to(device)                 # (Mt, C, gh, gw)
        tgt_feats = self.glyph_encoder.encode(tgt_glyphs).mean(dim=1)        # (Mt, Dg)
        tgt_proj = self.glyph_proj(tgt_feats)                                # (Mt, H)
        tgt_proj = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(tgt_proj).detach()

        # 预测位置 = 提示文本最后一位置 + 所有目标位置；目标 = 下一字符 (末尾为 EOS)
        last_prompt_pos = prompt_text_positions[-1] if prompt_text_positions else (hidden.shape[1] - 1)
        pred_positions = [last_prompt_pos] + list(target_text_positions)
        # 构建目标嵌入序列：tgt[0..Mt-1] + EOS 向量
        eos_vec = self.control_embed(torch.tensor([self.EOS], device=device)).squeeze(0)  # (H,)
        targets = torch.cat([tgt_proj, eos_vec.unsqueeze(0)], dim=0)         # (Mt+1, H)

        queries = self.text_head(hidden[0])                                  # (L, H)
        pred_q = queries[pred_positions]                                     # (P, H)
        # 归一化后余弦距离损失
        qn = pred_q / (pred_q.norm(dim=-1, keepdim=True) + 1e-8)
        tn = targets / (targets.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (qn * tn).sum(dim=-1)                                          # (P,)
        loss = (1.0 - sim).mean()
        return loss

    # ------------------------------------------------------------------
    # 最近字形解码
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_text(self, queries: torch.Tensor) -> List[str]:
        """把连续查询向量解码为最近字形字符。

        Args:
            queries: (B, H) 或 (B, L, H)。
        Returns:
            List[str]，每个查询对应的字符。
        """
        if queries.dim() == 3:
            queries = queries.reshape(-1, queries.shape[-1])
        device = queries.device
        bank = self._build_prototype_bank(device, queries.dtype)            # (Nc, H)
        bank_n = self._prototype_norm                                        # (Nc, H)
        qn = queries / (queries.norm(dim=-1, keepdim=True) + 1e-8)
        sim = qn @ bank_n.T                                                  # (B, Nc+4)
        idx = sim.argmax(dim=-1)
        chars = getattr(self, "_prototype_chars", self.charset)
        return [chars[i] for i in idx.tolist()]

    # ------------------------------------------------------------------
    # token-free 自回归生成 (带 KV 缓存)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inp: NativeInput,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        eos_on: bool = True,
    ) -> str:
        """原生 token-free 生成：直接对连续感知 token 自回归，最近字形解码输出字符。

        不经过任何 subword 分词 / token id 字典 (解码用的原型库是连续字形嵌入)。
        流式模式 (``self.streaming=True``) 使用固定状态的 ``stream_init/stream_step``，
        常驻显存极小，支持百万级上下文。
        """
        device = next(self.parameters()).device
        seq, L, ppos, tpos = self._perceive_inputs(inp)                     # (1, L, H)

        if self.streaming:
            state = self.core.stream_init(1, device, seq.dtype)
            # 逐块灌入提示序列，建立固定状态 (整块过流式核心，保留块内 RFP/DMC 上下文)
            hidden_last = None
            for i in range(0, seq.shape[1], self.config.chunk_size):
                seg = seq[:, i:i + self.config.chunk_size, :]
                hidden_last, state = self.core.stream_step(seg, state)
            if hidden_last is None:
                hidden_last = self.core.final_norm(seq)
            # 用最后一块的末位隐状态预测首字符
            cur_query = self.text_head(hidden_last[:, -1:, :])               # (1,1,H)
            pred_chars = self.decode_text(cur_query)[0]
            generated: List[str] = []
            for _ in range(max_new_tokens):
                if eos_on and pred_chars == self.control_char(self.EOS):
                    break
                generated.append(pred_chars)
                next_emb = self._char_embedding(pred_chars, device)         # (1,1,H)
                hidden_last, state = self.core.stream_step(next_emb, state)
                cur_query = self.text_head(hidden_last)
                pred_chars = self.decode_text(cur_query)[0]
            return "".join(generated)

        # ---- 全量模式 (同样基于固定大小 StreamState，无 KV Cache) ----
        # 整段提示一次性过流式核心，建立固定状态 (短序列 n<=chunk 时等价单块处理)；
        # 之后逐 token 用同一状态自回归，提示上下文经 StreamState 持续保留。
        state = self.core.stream_init(1, device, seq.dtype)
        hidden_last, state = self.core.stream_step(seq, state)
        cur_query = self.text_head(hidden_last[:, -1:, :])                 # (1,1,H)
        pred_chars = self.decode_text(cur_query)[0]

        generated = []
        for _ in range(max_new_tokens):
            if eos_on and pred_chars == self.control_char(self.EOS):
                break
            generated.append(pred_chars)
            next_emb = self._char_embedding(pred_chars, device)             # (1,1,H)
            hidden_last, state = self.core.stream_step(next_emb, state)
            cur_query = self.text_head(hidden_last)                        # (1,1,H)
            pred_chars = self.decode_text(cur_query)[0]
        return "".join(generated)

    def control_char(self, idx: int) -> str:
        """EOS 等控制字符在解码时映射为特殊标记字符串，便于终止判定。"""
        if idx == self.EOS:
            return "[EOS]"
        return ""

    @torch.no_grad()
    def _char_embedding(self, ch: str, device) -> torch.Tensor:
        """取某字符的字形原型嵌入 (在 H 空间) 作为下一 token 输入。"""
        bank = self._build_prototype_bank(device, torch.float32)
        if ch in self.charset:
            i = self.charset.index(ch)
            return bank[i].reshape(1, 1, -1).to(device)
        # 未知字符：用平均原型
        return bank.mean(dim=0, keepdim=True).reshape(1, 1, -1).to(device)

    # ------------------------------------------------------------------
    def num_parameters(self, include_non_embedding: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"H={self.config.hidden_size}, layers={self.config.num_layers}, "
                f"experts={self.config.num_experts}, charset={len(self.charset)}")
