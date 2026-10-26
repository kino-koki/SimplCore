# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 生产级配置 (SimplCoreConfig).

集中描述 SimplCore 架构的全部超参数，提供：
  * 强类型字段与构造期校验 (``validate``)；
  * 参数量估算 (``estimate_parameters``)；
  * 预设工厂 (nano / small / 150m / 7b / 14b / 72b) 与 ``from_preset``；
  * 兼容旧命名的 property 访问器。

本模块不依赖 PyTorch，可独立用于配置校验与规划。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SimplCoreConfig:
    """SimplCore 架构统一配置。

    Attributes 见子节分组；所有字段在 :meth:`__post_init__` 中经 :meth:`validate`
    校验，非法取值立即抛出 :class:`ValueError`。
    """

    # ---- 词表与隐层 ----
    vocab_size: int = 32_000
    hidden_size: int = 768
    num_layers: int = 12

    # ---- LocalInteraction (原生局部交互层，替代自注意力) ----
    # use_attention 已废弃（架构正式放弃 Transformer）；统一由 use_localmix 控制
    # 是否启用 SimplCore 原生局部交互层（因果深度卷积 + 可选线性递归）。
    use_localmix: bool = True
    use_attention: bool = False   # 残留字段，固定 False，便于旧配置解析不报错
    local_kernel_size: int = 7
    local_dilation: int = 1
    local_use_conv: bool = True
    local_use_recurrence: bool = True

    # ---- RFP (共振场处理, Resonance Field Processing) ----
    num_waves: int = 8
    rfp_num_steps: int = 3
    rfp_damping: float = 0.10
    rfp_dt: float = 0.10
    rfp_sigma: float = 0.50
    # 是否冻结 FFT 演化（True=仅投影可学，FFT 演化走 no_grad；用于 FFT 反向
    # 泄漏严重的环境）。默认 False：经 checkpoint 包裹真正可训练。
    rfp_freeze_evolution: bool = False
    rfp_use_checkpoint: bool = True

    # ---- DMC (动态记忆通道, Dynamic Memory Channels) ----
    num_channels: int = 64
    channel_size: int = 128
    dmc_num_heads: int = 4

    # ---- TAR (拓扑感知路由, Topology-Aware Routing) ----
    num_paths: int = 4
    min_depth: int = 2
    max_depth: int = 8
    tar_topk: int = 2
    tar_dropout: float = 0.0

    # ---- FFN ----
    ffn_intermediate_ratio: float = 4.0

    # ---- MoE (Mixture-of-Experts) ----
    # num_experts == 1 表示稠密模型 (不使用 MoE)；>1 启用专家路由。
    num_experts: int = 1
    experts_per_tok: int = 2
    moe_intermediate_size: int = 0          # 0 = 由 ffn_intermediate_ratio 推导
    moe_aux_loss_coeff: float = 0.01
    moe_z_loss_coeff: float = 1e-4
    moe_capacity_factor: float = 1.25
    moe_use_shared_expert: bool = False
    shared_expert_intermediate_size: int = 0

    # ---- SimplCore 专属子层开关 (MoE 大模型默认关，以对齐标准 LLM 参数账) ----
    use_rfp: bool = True
    use_dmc: bool = True

    # ---- 序列与多模态 ----
    max_sequence_length: int = 2048
    # 长上下文流式核心参数 (与序列长度 n 无关的固定大小)
    chunk_size: int = 512          # RFP 分块频域演化块大小；激活峰值 O(chunk) 常数
    num_modalities: int = 3
    modality_dims: Tuple[int, ...] = (768, 768, 768)

    # ---- 训练与推演 ----
    tie_word_embeddings: bool = True
    dropout: float = 0.10
    use_rollout: bool = False
    rollout_max_steps: int = 8

    # ---- 原生多模态 (token-free) ----
    # 开启后模型跳过"文本/图像/视频 -> Token"的中间编码：
    #   * 文本 -> 渲染为字形位图 (glyph)，由视觉编码器"看见"字形；
    #   * 图像/视频 -> 像素/帧直接进视觉编码器；
    #   * 三者统一投影为连续"感知 token"进入 MoE 推理核，不再有离散词表。
    use_native_multimodal: bool = False
    # 字形编码器 (GlyphEncoder)：把单个字符的字形位图编码为连续向量。
    glyph_size: int = 32                 # 字形位图边长 (GlyphEncoder 期望 sqrt 可被 patch 整除)
    glyph_in_channels: int = 1           # 1=灰度, 3=RGB
    glyph_patch_size: int = 8
    glyph_hidden: int = 256
    glyph_encoder_layers: int = 4
    glyph_encoder_heads: int = 8
    # 图像编码器 (ImageEncoder)：patch 视觉 Transformer。
    vision_image_size: int = 224
    vision_in_channels: int = 3
    vision_patch_size: int = 16
    vision_hidden: int = 768
    vision_encoder_layers: int = 6
    vision_encoder_heads: int = 8
    # 视频编码器 (VideoEncoder)：逐帧共享 ImageEncoder + 时序聚合。
    video_frames: int = 8                # 每段视频采样帧数
    video_temporal_layers: int = 2
    video_temporal_heads: int = 8
    # 感知重采样 (PerceptualResampler)：cross-attention perceiver，
    # 将变长视觉特征压缩为固定 K 个连续"感知 token"。
    perceptual_tokens: int = 64
    perceiver_layers: int = 2
    perceiver_heads: int = 8
    # 输出解码：文本用"最近字形匹配"从连续向量解码出字符 (无 token 字典)，
    # 图像用像素解码器。char_set_size 仅决定字形原型库规模 (不计入参数)。
    char_set_size: int = 6000

    # ---- 归一化与特殊 token ----
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    eos_token_id: int = 1

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """构造后立即校验配置合法性。"""
        self.validate()

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """校验所有字段取值，非法时抛出 ``ValueError``。

        Raises:
            ValueError: 任一字段取值非法。
            TypeError: 字段类型错误。
        """
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_waves": self.num_waves,
            "num_channels": self.num_channels,
            "channel_size": self.channel_size,
            "num_paths": self.num_paths,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "max_sequence_length": self.max_sequence_length,
            "chunk_size": self.chunk_size,
            "num_modalities": self.num_modalities,
            "rfp_num_steps": self.rfp_num_steps,
            "dmc_num_heads": self.dmc_num_heads,
            "tar_topk": self.tar_topk,
            "num_experts": self.num_experts,
            "experts_per_tok": self.experts_per_tok,
            # ---- 原生多模态字段 ----
            "glyph_size": self.glyph_size,
            "glyph_in_channels": self.glyph_in_channels,
            "glyph_patch_size": self.glyph_patch_size,
            "glyph_hidden": self.glyph_hidden,
            "glyph_encoder_layers": self.glyph_encoder_layers,
            "glyph_encoder_heads": self.glyph_encoder_heads,
            "vision_image_size": self.vision_image_size,
            "vision_in_channels": self.vision_in_channels,
            "vision_patch_size": self.vision_patch_size,
            "vision_hidden": self.vision_hidden,
            "vision_encoder_layers": self.vision_encoder_layers,
            "vision_encoder_heads": self.vision_encoder_heads,
            "video_frames": self.video_frames,
            "video_temporal_layers": self.video_temporal_layers,
            "video_temporal_heads": self.video_temporal_heads,
            "perceptual_tokens": self.perceptual_tokens,
            "perceiver_layers": self.perceiver_layers,
            "perceiver_heads": self.perceiver_heads,
            "char_set_size": self.char_set_size,
        }
        for name, value in positive_int_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} 必须为整数，得到 {type(value).__name__}。")
            if value <= 0:
                raise ValueError(f"{name} 必须为正整数，得到 {value}。")

        if self.min_depth > self.max_depth:
            raise ValueError(
                f"min_depth({self.min_depth}) 不能大于 max_depth({self.max_depth})。"
            )
        if self.tar_topk > self.num_paths:
            raise ValueError(
                f"tar_topk({self.tar_topk}) 不能大于 num_paths({self.num_paths})。"
            )
        if self.num_experts > 1 and not (1 <= self.experts_per_tok <= self.num_experts):
            raise ValueError(
                f"MoE 启用时 experts_per_tok({self.experts_per_tok}) 必须在 [1, {self.num_experts}]。"
            )
        if self.num_experts > 1 and self.moe_intermediate_size < 1 and self.ffn_intermediate_ratio <= 0:
            raise ValueError("MoE 启用时 moe_intermediate_size 或 ffn_intermediate_ratio 必须为正。")
        if self.hidden_size % self.dmc_num_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 dmc_num_heads"
                f"({self.dmc_num_heads}) 整除。"
            )
        if self.ffn_intermediate_ratio <= 0:
            raise ValueError(f"ffn_intermediate_ratio 必须为正数，得到 {self.ffn_intermediate_ratio}。")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1)，得到 {self.dropout}。")
        if not (0.0 <= self.tar_dropout < 1.0):
            raise ValueError(f"tar_dropout 必须在 [0, 1)，得到 {self.tar_dropout}。")
        if self.rfp_dt <= 0:
            raise ValueError(f"rfp_dt 必须为正数，得到 {self.rfp_dt}。")
        if self.rfp_sigma <= 0:
            raise ValueError(f"rfp_sigma 必须为正数，得到 {self.rfp_sigma}。")
        if self.rfp_damping < 0:
            raise ValueError(f"rfp_damping 不能为负，得到 {self.rfp_damping}。")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps 必须为正数，得到 {self.rms_norm_eps}。")
        if self.use_rollout and self.rollout_max_steps <= 0:
            raise ValueError("启用 use_rollout 时 rollout_max_steps 必须为正。")
        if self.use_native_multimodal:
            if self.glyph_size % self.glyph_patch_size != 0:
                raise ValueError(
                    f"glyph_size({self.glyph_size}) 必须能被 glyph_patch_size"
                    f"({self.glyph_patch_size}) 整除。"
                )
            if self.vision_image_size % self.vision_patch_size != 0:
                raise ValueError(
                    f"vision_image_size({self.vision_image_size}) 必须能被 vision_patch_size"
                    f"({self.vision_patch_size}) 整除。"
                )
            if self.glyph_hidden % self.glyph_encoder_heads != 0:
                raise ValueError(
                    f"glyph_hidden({self.glyph_hidden}) 必须能被 glyph_encoder_heads"
                    f"({self.glyph_encoder_heads}) 整除。"
                )
            if self.vision_hidden % self.vision_encoder_heads != 0:
                raise ValueError(
                    f"vision_hidden({self.vision_hidden}) 必须能被 vision_encoder_heads"
                    f"({self.vision_encoder_heads}) 整除。"
                )
            if self.hidden_size % self.perceiver_heads != 0:
                raise ValueError(
                    f"hidden_size({self.hidden_size}) 必须能被 perceiver_heads"
                    f"({self.perceiver_heads}) 整除。"
                )
            if self.hidden_size % self.video_temporal_heads != 0:
                raise ValueError(
                    f"hidden_size({self.hidden_size}) 必须能被 video_temporal_heads"
                    f"({self.video_temporal_heads}) 整除。"
                )
        if len(self.modality_dims) != self.num_modalities:
            raise ValueError(
                f"modality_dims 长度({len(self.modality_dims)}) 必须等于 "
                f"num_modalities({self.num_modalities})。"
            )
        for i, d in enumerate(self.modality_dims):
            if not isinstance(d, int) or isinstance(d, bool) or d <= 0:
                raise ValueError(f"modality_dims[{i}] 必须为正整型，得到 {d!r}。")
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError(f"pad_token_id({self.pad_token_id}) 必须在 [0, {self.vocab_size})。")
        if not (0 <= self.eos_token_id < self.vocab_size):
            raise ValueError(f"eos_token_id({self.eos_token_id}) 必须在 [0, {self.vocab_size})。")

    # ------------------------------------------------------------------
    # 参数量估算
    # ------------------------------------------------------------------
    def _vit_param_count(self, image_size: int, patch: int, in_ch: int, hidden: int, layers: int) -> int:
        """估算一个 ViT 编码器的参数量 (patch embed + pos + 块 + 输出投影 + norm)。"""
        n_patch = (image_size // patch) ** 2
        embed = in_ch * hidden * patch * patch + hidden
        posemb = (n_patch + 1) * hidden
        block = layers * (4 * hidden * hidden + hidden + 3 * 4 * hidden * hidden)
        proj_out = hidden * hidden
        return embed + posemb + block + proj_out + hidden

    def estimate_parameters(self) -> Dict[str, int]:
        """基于超参数精确估算模型各组件参数量。

        稠密模型与 MoE 模型采用不同口径：
          * ``num_experts == 1``：沿用 RFP/DMC/TAR/FFN 估算；
          * ``num_experts  > 1``：RFP/DMC/TAR 关闭，改为 专家FFN + 路由 估算。

        Returns:
            dict: 各组件参数量与 total。
        """
        H, V, N = self.hidden_size, self.vocab_size, self.num_layers
        W, C, S, P = self.num_waves, self.num_channels, self.channel_size, self.num_paths
        R = self.ffn_intermediate_ratio
        hd = H // self.dmc_num_heads          # 记忆头维度
        inter = int(round(R * H))             # FFN 中间维度

        embeddings = V * H
        lm_head = 0 if self.tie_word_embeddings else V * H

        # LocalInteraction（替代自注意力）：depthwise conv(kernel) 无 bias +
        # 可选线性递归 (Linear H->H) + 融合 Linear(H*2 -> H)。无 Q/K/V/RoPE。
        # 为保守估算，用 H 近似通道数 C。
        lm_conv = self.local_kernel_size * H
        lm_recur = H * H + H if self.local_use_recurrence else 0
        lm_fuse = (2 * H if self.local_use_recurrence else H) * H
        localmix_per = lm_conv + lm_recur + lm_fuse + H

        parts: Dict[str, int] = {
            "embeddings": int(embeddings),
            "lm_head": int(lm_head),
            "localmix": int(localmix_per * N) if self.use_localmix else 0,
        }

        if self.num_experts > 1:
            # ---- MoE 口径 ----
            e_inter = self.moe_intermediate_size or inter
            expert_per = 3 * H * e_inter                       # SwiGLU 专家 FFN
            router_per = H * self.num_experts                 # 路由线性
            parts["experts"] = int(expert_per * N * self.num_experts)
            parts["router"] = int(router_per * N)
            if self.moe_use_shared_expert:
                s_inter = self.shared_expert_intermediate_size or e_inter
                parts["shared_expert"] = int(3 * H * s_inter * N)
            # 生产级默认启用 RFP/DMC：开启时计入其参数，确保参数账与真实模型一致
            rfp_per = (H * W + W) + (W * H + H) + 3 * W + 2 + H
            dmc_per = 2 * (H * H) + 2 * (hd * hd) + C * S * hd + H
            parts["rfp"] = int(rfp_per * N) if self.use_rfp else 0
            parts["dmc"] = int(dmc_per * N) if self.use_dmc else 0
            parts["tar"] = parts["ffn"] = 0
        else:
            # ---- 稠密口径 ----
            rfp_per = (H * W + W) + (W * H + H) + 3 * W + 2 + H
            dmc_per = 2 * (H * H) + 2 * (hd * hd) + C * S * hd + H
            tar_per = (H * P + P) + P * (H * H) + 1 + H
            ffn_per = 3 * H * inter
            parts["rfp"] = int(rfp_per * N) if self.use_rfp else 0
            parts["dmc"] = int(dmc_per * N) if self.use_dmc else 0
            parts["tar"] = int(tar_per * N)
            parts["ffn"] = int(ffn_per * N)
            parts["experts"] = parts["router"] = parts["shared_expert"] = 0

        if self.use_native_multimodal:
            # 原生模式下没有全词表嵌入与离散 LM Head：
            #   * 用 4 个控制 token 嵌入 (bos/eos/sep/pad) 替代 V×H 嵌入；
            #   * 输出头为连续"最近字形匹配" + 像素解码，无 token 分类头。
            parts["embeddings"] = int(4 * H)
            parts["lm_head"] = 0
            glyph_enc = self._vit_param_count(
                self.glyph_size, self.glyph_patch_size, self.glyph_in_channels,
                self.glyph_hidden, self.glyph_encoder_layers)
            image_enc = self._vit_param_count(
                self.vision_image_size, self.vision_patch_size, self.vision_in_channels,
                self.vision_hidden, self.vision_encoder_layers)
            video_temp = self.video_temporal_layers * (8 * H * H + 2 * H)
            # 三种模态各一个 D_in -> H 投影，将编码器特征对齐到推理核维度
            proj = (self.glyph_hidden * H) + (self.vision_hidden * H) * 2
            perceiver = (self.perceptual_tokens * H
                         + self.perceiver_layers * (20 * H * H + 2 * H))
            parts["perception"] = int(glyph_enc + image_enc + video_temp + proj + perceiver)
            parts["control"] = int(4 * H)
            parts["text_head"] = int(H * H)
            parts["image_head"] = int(6 * H * H)

        parts["total"] = sum(parts.values())
        return parts

    # ------------------------------------------------------------------
    # 预设工厂
    # ------------------------------------------------------------------
    @classmethod
    def get_nano_config(cls) -> "SimplCoreConfig":
        """超小配置，用于单元测试 / 冒烟测试。"""
        return cls(
            vocab_size=1000, hidden_size=64, num_layers=2, num_waves=2,
            num_channels=4, channel_size=8, num_paths=2, min_depth=1, max_depth=2,
            ffn_intermediate_ratio=2.0, max_sequence_length=64, num_modalities=3,
            modality_dims=(64, 64, 64), dropout=0.0, rfp_num_steps=2,
            dmc_num_heads=2, tar_topk=1, use_rollout=False, rollout_max_steps=2,
        )

    get_tiny_config = get_nano_config

    @classmethod
    def get_small_config(cls) -> "SimplCoreConfig":
        """小型配置，用于感知管道 / 快速实验。"""
        return cls(
            vocab_size=32_000, hidden_size=256, num_layers=6, num_waves=8,
            num_channels=32, channel_size=64, num_paths=4, min_depth=2, max_depth=6,
            ffn_intermediate_ratio=4.0, max_sequence_length=1024, num_modalities=3,
            modality_dims=(256, 256, 256), dropout=0.1, rfp_num_steps=3,
            dmc_num_heads=4, tar_topk=2, use_rollout=False, rollout_max_steps=4,
        )

    @classmethod
    def get_150m_config(cls) -> "SimplCoreConfig":
        """约 150M 训练配置。"""
        return cls(
            vocab_size=32_000, hidden_size=768, num_layers=12, num_waves=8,
            num_channels=32, channel_size=64, num_paths=4, min_depth=2, max_depth=6,
            ffn_intermediate_ratio=4.0, max_sequence_length=1024, num_modalities=3,
            modality_dims=(768, 768, 768), dropout=0.1, rfp_num_steps=3,
            dmc_num_heads=8, tar_topk=2, use_rollout=False, rollout_max_steps=4,
        )

    @classmethod
    def get_7b_config(cls) -> "SimplCoreConfig":
        """7B 目标堆叠配置。"""
        return cls(
            vocab_size=32_000, hidden_size=4096, num_layers=32, num_waves=16,
            num_channels=64, channel_size=128, num_paths=8, min_depth=2, max_depth=8,
            ffn_intermediate_ratio=4.0, max_sequence_length=4096, num_modalities=3,
            modality_dims=(4096, 4096, 4096), dropout=0.0, rfp_num_steps=3,
            dmc_num_heads=16, tar_topk=2, use_rollout=True, rollout_max_steps=8,
        )

    @classmethod
    def get_14b_config(cls) -> "SimplCoreConfig":
        return cls(
            vocab_size=32_000, hidden_size=5120, num_layers=40, num_waves=16,
            num_channels=64, channel_size=128, num_paths=8, min_depth=2, max_depth=8,
            ffn_intermediate_ratio=4.0, max_sequence_length=4096, num_modalities=3,
            modality_dims=(5120, 5120, 5120), dropout=0.0, rfp_num_steps=3,
            dmc_num_heads=16, tar_topk=2, use_rollout=True, rollout_max_steps=8,
        )

    @classmethod
    def get_72b_config(cls) -> "SimplCoreConfig":
        return cls(
            vocab_size=128_000, hidden_size=8192, num_layers=80, num_waves=16,
            num_channels=64, channel_size=128, num_paths=8, min_depth=2, max_depth=8,
            ffn_intermediate_ratio=4.0, max_sequence_length=4096, num_modalities=3,
            modality_dims=(8192, 8192, 8192), tie_word_embeddings=False, dropout=0.0,
            rfp_num_steps=3, rfp_damping=0.1, dmc_num_heads=8, tar_topk=2,
            use_rollout=True, rollout_max_steps=8,
        )

    @classmethod
    def get_56b_moe_config(cls) -> "SimplCoreConfig":
        """56B 中文 MoE 目标配置 (7 专家 × ~7.8B，top-2 激活)。

        参数账 (精确)：
          * H=4096, layers=14, heads=32, vocab=128000 (中文优化词表)；
          * 每专家 FFN 中间维度 45344 → 单专家单层 0.557B，
            跨 14 层累计 **每专家角色 ≈ 7.80B**；
          * 7 专家合计 54.6B + 共享注意力 0.94B + 嵌入 0.52B ≈ **56.07B**；
          * 每 token 激活 2/7 专家 (≈15.6B 专家参数)，推理算力省 ~3.5×。

        **生产级模型默认启用 SimplCore 自研架构**：``use_rfp`` / ``use_dmc``
        均为真，每个 MoE block 在注意力之后、MoE 之前插入共振场处理
        (谱全局卷积, O(n log n), 可训练) 与动态记忆通道 (内容寻址外部记忆,
        可训练)。参数量估算见 :meth:`estimate_parameters` (MoE 口径会计入
        RFP/DMC)。纯中文语料策略见 ``SimplCore_56B_MoE_Plan.md``。
        """
        return cls(
            vocab_size=128_000, hidden_size=4096, num_layers=14,
            num_waves=8, num_channels=64, channel_size=128, num_paths=8,
            min_depth=2, max_depth=8, ffn_intermediate_ratio=4.0,
            max_sequence_length=4096, num_modalities=3,
            modality_dims=(4096, 4096, 4096), tie_word_embeddings=True,
            dropout=0.0, rfp_num_steps=3, dmc_num_heads=32,
            tar_topk=2, use_rollout=False, rollout_max_steps=8,
            # ---- MoE ----
            num_experts=7, experts_per_tok=2, moe_intermediate_size=45344,
            moe_aux_loss_coeff=0.01, moe_z_loss_coeff=1e-4,
            moe_capacity_factor=1.25, moe_use_shared_expert=False,
            shared_expert_intermediate_size=0,
            # ---- 生产级默认启用 SimplCore 自研子层 ----
            use_rfp=True, use_dmc=True,
        )

    @classmethod
    def get_native_moe_config(cls) -> "SimplCoreConfig":
        """原生 token-free 多模态目标配置 (56B MoE 推理核 + 原生感知前端)。

        与 ``get_56b_moe_config`` 共享 56B MoE 推理核 (H=4096, 14 层, 7 专家)，
        额外挂载原生感知前端：字形编码器 (灰度字形 -> 连续向量)、图像/视频
        ViT 编码器、感知重采样 (固定 K 个连续感知 token)。推理核直接消费连续
        感知 token，不经过任何 subword 词表。
        """
        return cls(
            vocab_size=128_000, hidden_size=4096, num_layers=14,
            num_waves=8, num_channels=64, channel_size=128, num_paths=8,
            min_depth=2, max_depth=8, ffn_intermediate_ratio=4.0,
            max_sequence_length=4096, num_modalities=3,
            modality_dims=(4096, 4096, 4096), tie_word_embeddings=True,
            dropout=0.0, rfp_num_steps=3, dmc_num_heads=32,
            tar_topk=2, use_rollout=False, rollout_max_steps=8,
            # ---- MoE 推理核 (56B) ----
            num_experts=7, experts_per_tok=2, moe_intermediate_size=45344,
            moe_aux_loss_coeff=0.01, moe_z_loss_coeff=1e-4,
            moe_capacity_factor=1.25, moe_use_shared_expert=False,
            shared_expert_intermediate_size=0,
            use_rfp=True, use_dmc=True,
            # ---- 原生多模态 ----
            use_native_multimodal=True,
            glyph_size=32, glyph_in_channels=1, glyph_patch_size=8,
            glyph_hidden=512, glyph_encoder_layers=6, glyph_encoder_heads=8,
            vision_image_size=336, vision_in_channels=3, vision_patch_size=14,
            vision_hidden=1024, vision_encoder_layers=24, vision_encoder_heads=16,
            video_frames=8, video_temporal_layers=4, video_temporal_heads=16,
            perceptual_tokens=256, perceiver_layers=3, perceiver_heads=16,
            char_set_size=6000,
        )

    @classmethod
    def get_longctx_config(cls, ctx_len: int = 1_048_576) -> "SimplCoreConfig":
        """长上下文流式配置（无注意力）：从 1M token 起，序列翻倍显存近似常数增长。

        关键设置 (全部为定值，不随 ctx_len 变化)：
          * ``chunk_size=512``：RFP 分块频域演化，激活峰值 O(512) 常数；
          * 局部依赖由 ``LocalInteraction``（因果深度卷积 + 可选线性递归，O(n)，
            无 KV 缓存）承担，不随 ctx_len 增长；
          * ``use_rfp=True / use_dmc=True``：自研子层提供全局依赖与固定记忆；
          * ``max_sequence_length`` 仅用于容量规划，不决定激活显存。

        显存增长来源：固定大小的 RFP 频域状态 (B, 257, W) + DMC 记忆
        (C·S·hd) + 分块激活 (B, chunk, H) —— 均不含 ctx_len 项，因此上下文
        从 1M→2M→4M 翻倍时，激活显存仅增加由 ``chunk`` 决定的常数量
        (目标 ~常数 GB/翻倍)。
        """
        cfg = cls(
            vocab_size=128_000, hidden_size=4096, num_layers=14,
            num_waves=8, num_channels=64, channel_size=128, num_paths=8,
            min_depth=2, max_depth=8, ffn_intermediate_ratio=4.0,
            max_sequence_length=max(ctx_len, 8192), num_modalities=3,
            modality_dims=(4096, 4096, 4096), tie_word_embeddings=True,
            dropout=0.0, rfp_num_steps=3, dmc_num_heads=32,
            tar_topk=2, use_rollout=False, rollout_max_steps=8,
            num_experts=7, experts_per_tok=2, moe_intermediate_size=45344,
            moe_aux_loss_coeff=0.01, moe_z_loss_coeff=1e-4,
            moe_capacity_factor=1.25, moe_use_shared_expert=False,
            shared_expert_intermediate_size=0,
            use_localmix=True, use_rfp=True, use_dmc=True,
            chunk_size=512,
        )
        return cfg

    @classmethod
    def get_attention_free_config(cls, hidden_size: int = 128, num_layers: int = 2,
                                  vocab_size: int = 128) -> "SimplCoreConfig":
        """无注意力（纯 SimplCore）最小可训练配置：用于论文级消融与冒烟。

        组成：LocalInteraction（局部）+ RFP（全局谱卷积）+ DMC（固定记忆）+ MoE。
        完全不含自注意力 / Q-K-V / RoPE / SDPA。
        """
        return cls(
            vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
            num_waves=4, rfp_num_steps=3, num_channels=8, channel_size=16,
            dmc_num_heads=2, ffn_intermediate_ratio=2.0,
            max_sequence_length=2048, dropout=0.0,
            local_kernel_size=7, local_dilation=1, local_use_recurrence=True,
            use_localmix=True, use_rfp=True, use_dmc=True,
            num_experts=2, experts_per_tok=2, moe_intermediate_size=64,
            chunk_size=64,
        )

    @classmethod
    def from_preset(cls, name: str) -> "SimplCoreConfig":
        """按名称获取预设配置。

        Raises:
            ValueError: 未知预设名。
        """
        presets = {
            "tiny": cls.get_tiny_config,
            "nano": cls.get_nano_config,
            "small": cls.get_small_config,
            "150m": cls.get_150m_config,
            "7b": cls.get_7b_config,
            "14b": cls.get_14b_config,
            "72b": cls.get_72b_config,
            "56b-moe": cls.get_56b_moe_config,
            "native-moe": cls.get_native_moe_config,
            "attention-free": cls.get_attention_free_config,
        }
        key = (name or "").strip().lower()
        if key not in presets:
            raise ValueError(f"未知预设 '{name}'，可选: {sorted(presets.keys())}。")
        return presets[key]()

    # ------------------------------------------------------------------
    # 兼容访问器
    # ------------------------------------------------------------------
    @property
    def max_seq_len(self) -> int:
        return self.max_sequence_length

    @property
    def layer_norm_eps(self) -> float:
        return self.rms_norm_eps

    @property
    def ffn_ratio(self) -> int:
        return int(round(self.ffn_intermediate_ratio))

    @property
    def num_heads(self) -> int:
        return self.dmc_num_heads

    @property
    def num_rollout_steps(self) -> int:
        return self.rollout_max_steps if self.use_rollout else 0

    def as_dict(self) -> Dict[str, object]:
        """返回可 JSON 序列化的配置快照。"""
        return {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SimplCoreConfig":
        """从 :meth:`as_dict` 的快照重建配置（仅取已知字段，兼容旧快照）。

        Args:
            data: 配置快照（可含多余字段，如 checkpoint 元数据）。
        Returns:
            SimplCoreConfig: 重建后的配置。
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs: Dict[str, object] = {}
        for k, v in data.items():
            if k not in known:
                continue
            # as_dict 把 tuple 序列化为 list，这里还原为 tuple 以保持类型一致
            if k == "modality_dims" and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)
