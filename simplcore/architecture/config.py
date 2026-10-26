# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 配置 (SimplCoreConfig).

集中描述 SimplCore 架构的所有超参数:
  - 词表与隐层: vocab_size, hidden_size, num_layers
  - SGC (共振场处理): num_waves, sgc_num_steps, sgc_damping, sgc_dt, sgc_sigma
  - DMC (动态记忆通道): num_channels, channel_size, dmc_num_heads
  - CCR (条件计算路由): num_paths, min_depth, max_depth, ccr_topk, ccr_dropout
  - FFN: ffn_intermediate_ratio
  - 序列/多模态: max_sequence_length, num_modalities, modality_dims
  - 训练/推演: dropout, use_rollout, rollout_max_steps
  - 归一化: rms_norm_eps
  - 特殊 token: pad_token_id, eos_token_id

SimplCoreConfig 同时提供:
  - validate(): 校验所有字段取值合法, 不合法抛出 ValueError.
  - estimate_parameters(): 基于超参数估算各组件参数量 (dict).
  - get_nano_config() / get_small_config() / get_72b_config() / from_preset():
    预设配置工厂, 便于训练/感知管道直接调用.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SimplCoreConfig:
    """SimplCore 架构统一配置.

    Attributes:
        vocab_size: 词表大小.
        hidden_size: 隐层维度 H.
        num_layers: SimplCoreBlock 堆叠层数 N.
        num_waves: 谱通道数 W (用于 SGC / 旧 SGC).
        num_channels: DMC 记忆通道数 C.
        channel_size: DMC 每个通道的 slot 数 S.
        num_paths: CCR 候选路径数 P (条件计算路由).
        min_depth: CCR 最小计算深度.
        max_depth: CCR 最大计算深度.
        ffn_intermediate_ratio: FFN 中间层放缩系数.
        max_sequence_length: 最大序列长度.
        num_modalities: 模态数量.
        modality_dims: 各模态输入维度.
        tie_word_embeddings: 是否绑定输入嵌入与 LM Head 权重.
        dropout: dropout 概率.
        sgc_num_steps: SGC/旧RFP 谱演化步数.
        sgc_damping: SGC/旧RFP 衰减系数初值.
        sgc_dt: SGC/旧RFP 离散化时间步长.
        sgc_sigma: SGC/旧RFP 带通高斯窗带宽.
        dmc_num_heads: DMC 读取头数.
        ccr_topk: CCR top-k 路由的 k.
        ccr_dropout: CCR 路由 logits dropout.
        use_rollout: 是否启用潜空间推演 (RolloutEngine).
        rollout_max_steps: 推演最大步数.
        rms_norm_eps: 归一化层数值稳定常数.
        pad_token_id: padding token id.
        eos_token_id: eos token id.
    """

    # ---- 词表与隐层 ----
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12

    # ---- SGC (共振场处理) ----
    num_waves: int = 8            # 谱通道数 W (SGC / 旧 SGC)
    sgc_num_steps: int = 3
    sgc_damping: float = 0.1
    sgc_dt: float = 0.1
    sgc_sigma: float = 0.5

    # ---- DMC (动态记忆通道) ----
    num_channels: int = 64        # DMC 记忆通道数
    channel_size: int = 128       # DMC 每个 channel 的 slot 数
    dmc_num_heads: int = 4

    # ---- CCR (条件计算路由) ----
    num_paths: int = 4            # CCR 路径数
    min_depth: int = 2            # CCR 最小深度
    max_depth: int = 8            # CCR 最大深度
    ccr_topk: int = 2
    ccr_dropout: float = 0.0

    # ---- FFN ----
    ffn_intermediate_ratio: float = 4.0

    # ---- 序列与多模态 ----
    max_sequence_length: int = 2048
    num_modalities: int = 3
    modality_dims: Tuple[int, ...] = (768, 768, 768)

    # ---- 训练与推演 ----
    tie_word_embeddings: bool = True
    dropout: float = 0.1
    use_rollout: bool = False
    rollout_max_steps: int = 8
    rollout_steps: int = 0
    rollout_adaptive: bool = False
    rollout_detach: bool = False
    use_attn_res: bool = False
    use_vision: bool = False
    image_size: int = 224
    patch_size: int = 16

    # ---- 归一化与特殊 token ----
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    eos_token_id: int = 1

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """构造后立即校验配置合法性."""
        if self.rollout_steps > 0:
            self.rollout_max_steps = self.rollout_steps
        self.validate()

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """校验所有字段取值, 不合法时抛出 ValueError.

        Raises:
            ValueError: 任一字段取值非法.
        """
        # ---- 正整数约束 ----
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
            "num_modalities": self.num_modalities,
            "sgc_num_steps": self.sgc_num_steps,
            "dmc_num_heads": self.dmc_num_heads,
            "ccr_topk": self.ccr_topk,
        }
        for name, value in positive_int_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} 必须为整数, 得到 {type(value).__name__}.")
            if value <= 0:
                raise ValueError(f"{name} 必须为正整数, 得到 {value}.")

        # ---- 深度关系 ----
        if self.min_depth > self.max_depth:
            raise ValueError(
                f"min_depth({self.min_depth}) 不能大于 max_depth({self.max_depth})."
            )

        # ---- CCR topk ----
        if self.ccr_topk > self.num_paths:
            raise ValueError(
                f"ccr_topk({self.ccr_topk}) 不能大于 num_paths({self.num_paths})."
            )

        # ---- DMC 多头整除 ----
        if self.hidden_size % self.dmc_num_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 dmc_num_heads"
                f"({self.dmc_num_heads}) 整除."
            )

        # ---- 比例与步长 ----
        if self.ffn_intermediate_ratio <= 0:
            raise ValueError(
                f"ffn_intermediate_ratio 必须为正数, 得到 {self.ffn_intermediate_ratio}."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout 必须在 [0, 1), 得到 {self.dropout}.")
        if not (0.0 <= self.ccr_dropout < 1.0):
            raise ValueError(f"ccr_dropout 必须在 [0, 1), 得到 {self.ccr_dropout}.")

        # ---- SGC 物理参数 ----
        if self.sgc_dt <= 0:
            raise ValueError(f"sgc_dt 必须为正数, 得到 {self.sgc_dt}.")
        if self.sgc_sigma <= 0:
            raise ValueError(f"sgc_sigma 必须为正数, 得到 {self.sgc_sigma}.")
        if self.sgc_damping < 0:
            raise ValueError(f"sgc_damping 不能为负, 得到 {self.sgc_damping}.")

        # ---- 归一化 ----
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps 必须为正数, 得到 {self.rms_norm_eps}.")

        # ---- 推演 ----
        if self.use_rollout and self.rollout_max_steps <= 0:
            raise ValueError(
                f"启用 use_rollout 时 rollout_max_steps 必须为正, "
                f"得到 {self.rollout_max_steps}."
            )
        if self.rollout_steps < 0:
            raise ValueError(f"rollout_steps 不能为负, 得到 {self.rollout_steps}.")
        if self.image_size <= 0:
            raise ValueError(f"image_size 必须为正数, 得到 {self.image_size}.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size 必须为正数, 得到 {self.patch_size}.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size({self.image_size}) 必须能被 patch_size({self.patch_size}) 整除."
            )

        # ---- 多模态维度 ----
        if len(self.modality_dims) != self.num_modalities:
            raise ValueError(
                f"modality_dims 长度({len(self.modality_dims)}) 必须等于 "
                f"num_modalities({self.num_modalities})."
            )
        for i, d in enumerate(self.modality_dims):
            if d <= 0:
                raise ValueError(f"modality_dims[{i}] 必须为正, 得到 {d}.")

        # ---- 特殊 token ----
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError(
                f"pad_token_id({self.pad_token_id}) 必须在 [0, {self.vocab_size})."
            )
        if not (0 <= self.eos_token_id < self.vocab_size):
            raise ValueError(
                f"eos_token_id({self.eos_token_id}) 必须在 [0, {self.vocab_size})."
            )

    # ------------------------------------------------------------------
    # 参数量估算
    # ------------------------------------------------------------------
    def estimate_parameters(self) -> Dict[str, int]:
        """基于超参数估算模型各组件参数量.

        估算口径 (与 simplcore_block.py 内部实现一致):
          - embeddings: token 嵌入 (vocab_size * hidden_size).
          - lm_head: 若不绑定权重则为 vocab_size * hidden_size, 否则 0.
          - rfp/dmc/tar/ffn: 单层参数量 * num_layers.
          - total: 上述之和.

        Returns:
            dict: {
              "embeddings", "lm_head", "rfp", "dmc", "tar", "ffn", "total"
            } 各值为整数参数量.
        """
        H = self.hidden_size
        V = self.vocab_size
        N = self.num_layers
        W = self.num_waves
        C = self.num_channels
        S = self.channel_size
        P = self.num_paths
        R = self.ffn_intermediate_ratio

        # ---- 嵌入与输出头 ----
        embeddings = V * H
        lm_head = 0 if self.tie_word_embeddings else V * H

        # ---- 单层 SGC ----
        # in_proj: H*W + W; out_proj: W*H + H
        # omega/alpha/gamma: 3*W; beta: 1; out_scale: 1
        # LayerNorm: 2*H
        sgc_per_layer = (H * W + W) + (W * H + H) + 3 * W + 1 + 1 + 2 * H

        # ---- 单层 DMC ----
        # query/key/output_proj: 各 H*H (无偏置); write_value_proj: H*H + H
        # write_gate: H*C + C; channel_weights: C; LayerNorm: 2*H
        dmc_per_layer = 3 * (H * H) + (H * H + H) + (H * C + C) + C + 2 * H

        # ---- 单层 CCR ----
        # router: H*P + P; transform: H*H + H; 2 LayerNorm: 4*H; temperature: 1
        ccr_per_layer = (H * P + P) + (H * H + H) + 4 * H + 1

        # ---- 单层 FFN ----
        # Linear(H, R*H) + Linear(R*H, H); ffn_norm LayerNorm: 2*H
        ffn_inter = int(round(R * H))
        ffn_per_layer = (H * ffn_inter + ffn_inter) + (ffn_inter * H + H) + 2 * H

        rfp = sgc_per_layer * N
        dmc = dmc_per_layer * N
        tar = ccr_per_layer * N
        ffn = ffn_per_layer * N

        total = embeddings + lm_head + rfp + dmc + tar + ffn

        return {
            "embeddings": int(embeddings),
            "lm_head": int(lm_head),
            "rfp": int(rfp),
            "dmc": int(dmc),
            "tar": int(tar),
            "ffn": int(ffn),
            "total": int(total),
        }

    # ------------------------------------------------------------------
    # 预设工厂
    # ------------------------------------------------------------------
    @classmethod
    def get_nano_config(cls) -> "SimplCoreConfig":
        """超小配置, 用于单元测试 / 冒烟测试."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_waves=2,
            num_channels=4,
            channel_size=8,
            num_paths=2,
            min_depth=1,
            max_depth=2,
            ffn_intermediate_ratio=2.0,
            max_sequence_length=64,
            num_modalities=3,
            modality_dims=(64, 64, 64),
            tie_word_embeddings=True,
            dropout=0.0,
            sgc_num_steps=2,
            sgc_damping=0.1,
            dmc_num_heads=2,
            ccr_topk=1,
            use_rollout=False,
            rollout_max_steps=2,
        )

    @classmethod
    def get_tiny_config(cls) -> "SimplCoreConfig":
        """tiny 兼容入口；等价于 nano 预设。"""
        return cls.get_nano_config()

    @classmethod
    def get_small_config(cls) -> "SimplCoreConfig":
        """小型配置, 用于感知管道 / 快速实验."""
        return cls(
            vocab_size=32000,
            hidden_size=256,
            num_layers=6,
            num_waves=8,
            num_channels=32,
            channel_size=64,
            num_paths=4,
            min_depth=2,
            max_depth=6,
            ffn_intermediate_ratio=4.0,
            max_sequence_length=1024,
            num_modalities=3,
            modality_dims=(256, 256, 256),
            tie_word_embeddings=True,
            dropout=0.1,
            sgc_num_steps=3,
            sgc_damping=0.1,
            dmc_num_heads=4,
            ccr_topk=2,
            use_rollout=False,
            rollout_max_steps=4,
        )

    @classmethod
    def get_150m_config(cls) -> "SimplCoreConfig":
        """约 150M 训练配置，适合 AMD 680M 这类受限显存设备做基础训练。"""
        return cls(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            num_waves=8,
            num_channels=32,
            channel_size=64,
            num_paths=4,
            min_depth=2,
            max_depth=6,
            ffn_intermediate_ratio=4.0,
            max_sequence_length=1024,
            num_modalities=3,
            modality_dims=(768, 768, 768),
            tie_word_embeddings=True,
            dropout=0.1,
            sgc_num_steps=3,
            sgc_damping=0.1,
            dmc_num_heads=8,
            ccr_topk=2,
            use_rollout=False,
            rollout_max_steps=4,
        )

    @classmethod
    def get_7b_config(cls) -> "SimplCoreConfig":
        """7B 目标堆叠配置，用于参数映射和目标模型规划。"""
        return cls(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_waves=16,
            num_channels=64,
            channel_size=128,
            num_paths=8,
            min_depth=2,
            max_depth=8,
            ffn_intermediate_ratio=4.0,
            max_sequence_length=4096,
            num_modalities=3,
            modality_dims=(4096, 4096, 4096),
            tie_word_embeddings=True,
            dropout=0.0,
            sgc_num_steps=3,
            sgc_damping=0.1,
            dmc_num_heads=16,
            ccr_topk=2,
            use_rollout=True,
            rollout_max_steps=8,
        )

    @classmethod
    def get_14b_config(cls) -> "SimplCoreConfig":
        """14B 目标堆叠配置，用于后续扩展规划。"""
        return cls(
            vocab_size=32000,
            hidden_size=5120,
            num_layers=40,
            num_waves=16,
            num_channels=64,
            channel_size=128,
            num_paths=8,
            min_depth=2,
            max_depth=8,
            ffn_intermediate_ratio=4.0,
            max_sequence_length=4096,
            num_modalities=3,
            modality_dims=(5120, 5120, 5120),
            tie_word_embeddings=True,
            dropout=0.0,
            sgc_num_steps=3,
            sgc_damping=0.1,
            dmc_num_heads=16,
            ccr_topk=2,
            use_rollout=True,
            rollout_max_steps=8,
        )

    @classmethod
    def get_72b_config(cls) -> "SimplCoreConfig":
        """72B 量级配置 (仅用于参数量估算与规划, 实际训练需分布式)."""
        return cls(
            vocab_size=128000,
            hidden_size=8192,
            num_layers=80,
            num_waves=16,
            num_channels=64,
            channel_size=128,
            num_paths=8,
            min_depth=2,
            max_depth=8,
            ffn_intermediate_ratio=4.0,
            max_sequence_length=4096,
            num_modalities=3,
            modality_dims=(8192, 8192, 8192),
            tie_word_embeddings=False,
            dropout=0.0,
            sgc_num_steps=3,
            sgc_damping=0.1,
            dmc_num_heads=8,
            ccr_topk=2,
            use_rollout=True,
            rollout_max_steps=8,
        )

    @classmethod
    def from_preset(cls, name: str) -> "SimplCoreConfig":
        """按名称获取预设配置.

        Args:
            name: 预设名 ("nano" / "small" / "72b").

        Returns:
            SimplCoreConfig 实例.

        Raises:
            ValueError: 未知预设名.
        """
        presets = {
            "tiny": cls.get_tiny_config,
            "nano": cls.get_nano_config,
            "small": cls.get_small_config,
            "150m": cls.get_150m_config,
            "7b": cls.get_7b_config,
            "14b": cls.get_14b_config,
            "72b": cls.get_72b_config,
        }
        key = name.strip().lower()
        if key not in presets:
            raise ValueError(
                f"未知预设 '{name}', 可选: {sorted(presets.keys())}."
            )
        return presets[key]()

    @property
    def max_seq_len(self) -> int:
        """兼容旧代码中的 max_seq_len 命名。"""
        return self.max_sequence_length

    @property
    def layer_norm_eps(self) -> float:
        """兼容旧代码中的 layer_norm_eps 命名。"""
        return self.rms_norm_eps

    @property
    def ffn_ratio(self) -> int:
        """兼容旧代码中的 ffn_ratio 命名。"""
        return int(round(self.ffn_intermediate_ratio))

    @property
    def num_heads(self) -> int:
        """兼容旧代码中的 num_heads 命名。"""
        return self.dmc_num_heads

    @property
    def num_rollout_steps(self) -> int:
        """兼容旧集成模型；未启用 rollout 时返回 0。"""
        return self.rollout_max_steps if self.use_rollout else 0


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SimplCoreConfig 测试")
    print("=" * 60)

    # ---- 默认配置 ----
    cfg = SimplCoreConfig()
    print(f"[默认] hidden={cfg.hidden_size}, layers={cfg.num_layers}, "
          f"vocab={cfg.vocab_size}")
    print("  [PASS] 默认配置构造通过 validate.")

    # ---- 参数量估算 ----
    est = cfg.estimate_parameters()
    print("\n[参数量估算] (默认配置)")
    for k, v in est.items():
        print(f"  {k:>12s}: {v:>12,}")
    assert est["total"] == sum(est[k] for k in
                               ("embeddings", "lm_head", "rfp", "dmc", "tar", "ffn"))
    print("  [PASS] total 等于各分量之和.")

    # ---- nano 预设 ----
    nano = SimplCoreConfig.get_nano_config()
    print(f"\n[nano] hidden={nano.hidden_size}, layers={nano.num_layers}, "
          f"heads={nano.dmc_num_heads}")
    assert nano.hidden_size % nano.dmc_num_heads == 0
    est_nano = nano.estimate_parameters()
    print(f"  nano 总参数量: {est_nano['total']:,}")
    print("  [PASS] nano 预设通过.")

    # ---- from_preset ----
    small = SimplCoreConfig.from_preset("small")
    assert small.hidden_size == 256
    print(f"\n[small via from_preset] hidden={small.hidden_size}")
    print("  [PASS] from_preset 正常.")

    # ---- 校验失败用例 ----
    print("\n[校验失败用例]")
    bad_cases = [
        {"vocab_size": 0},
        {"min_depth": 10, "max_depth": 4},
        {"ccr_topk": 10, "num_paths": 4},
        {"hidden_size": 64, "dmc_num_heads": 5},
        {"dropout": 1.5},
        {"modality_dims": (768, 768), "num_modalities": 3},
    ]
    for i, patch in enumerate(bad_cases):
        base = dict(vocab_size=1000, hidden_size=64, num_layers=2,
                    num_waves=2, num_channels=4, channel_size=8,
                    num_paths=2, min_depth=1, max_depth=2,
                    dmc_num_heads=2, ccr_topk=1, num_modalities=3,
                    modality_dims=(64, 64, 64))
        base.update(patch)
        try:
            SimplCoreConfig(**base)
            print(f"  case {i}: 未抛出异常 (异常!)")
            raise AssertionError(f"case {i} 应当抛出 ValueError")
        except ValueError as e:
            print(f"  case {i}: ValueError -> {e}")
    print("  [PASS] 所有非法配置均被拦截.")

    print("\n" + "=" * 60)
    print("SimplCoreConfig 全部测试 PASSED.")
    print("=" * 60)
