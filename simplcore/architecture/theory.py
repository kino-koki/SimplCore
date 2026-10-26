# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 架构理论基础 (Architecture Theory)
=============================================

本模块提供 SimplCore 架构的理论分析工具:
  1. 各组件参数量精确计算 (SGC, DMC, CCR, FFN, Embedding)
  2. 给定配置的参数总量估算
  3. 复杂度对比表 (SGC vs 标准注意力)

参数计算公式基于各组件的实际实现:
  - SGC: omega + alpha + gamma + beta + out_scale + in_proj + out_proj + norm
  - DMC: query_proj + key_proj + write_value_proj + output_proj + write_gate + channel_weights + norm
  - CCR: router + transform + norm + out_norm + temperature
  - FFN: 两层 Linear (dim → dim*ratio → dim)
  - Embedding: vocab_size * dim
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

# 尝试从 .config 导入 SimplCoreConfig, 如果模块损坏则使用回退定义
try:
    from .config import SimplCoreConfig
except Exception:
    from dataclasses import dataclass, field

    @dataclass
    class SimplCoreConfig:
        """SimplCore 配置 (回退定义, 当 config.py 不可用时使用).

        Attributes:
            vocab_size: 词表大小.
            hidden_size: 隐藏层维度.
            num_layers: SimplCoreBlock 层数.
            num_waves: SGC 波场通道数.
            num_channels: DMC 记忆通道数.
            channel_size: DMC 每通道 slot 数.
            num_heads: DMC 注意力头数.
            num_paths: CCR 候选路径数.
            ffn_ratio: FFN 隐藏层扩展比例.
            max_seq_len: 最大序列长度.
            dropout: dropout 概率.
            use_vision: 是否启用视觉编码器.
            image_size: 图像尺寸.
            patch_size: 图像分块大小.
            num_rollout_steps: 推演步数.
        """
        vocab_size: int = 32000
        hidden_size: int = 768
        num_layers: int = 12
        num_waves: int = 8
        num_channels: int = 64
        channel_size: int = 128
        num_heads: int = 8
        num_paths: int = 4
        ffn_ratio: int = 4
        max_seq_len: int = 2048
        dropout: float = 0.1
        use_vision: bool = False
        image_size: int = 224
        patch_size: int = 16
        num_rollout_steps: int = 5

        @classmethod
        def from_preset(cls, name: str) -> "SimplCoreConfig":
            """从预设名称创建配置."""
            presets = {
                "tiny": cls(vocab_size=1000, hidden_size=64, num_layers=2),
                "150m": cls(vocab_size=32000, hidden_size=768, num_layers=12),
                "7b": cls(vocab_size=32000, hidden_size=4096, num_layers=32),
                "14b": cls(vocab_size=32000, hidden_size=5120, num_layers=40),
            }
            key = name.lower()
            if key not in presets:
                raise ValueError(f"未知预设: {name}. 可用: {list(presets.keys())}")
            return presets[key]


# ======================================================================
# 架构理论分析类
# ======================================================================

class ArchitectureTheory:
    """SimplCore 架构理论分析工具.

    提供各组件的参数量精确计算和复杂度分析。

    Usage:
        >>> theory = ArchitectureTheory()
        >>> theory.count_parameters_rfp(dim=768, num_waves=8)
        >>> theory.estimate_total(SimplCoreConfig.from_preset('150m'))
        >>> theory.print_parameter_breakdown(SimplCoreConfig.from_preset('150m'))
        >>> theory.complexity_table([512, 1024, 4096, 16384])
    """

    def __init__(self) -> None:
        """初始化理论分析工具."""
        pass

    # ==================================================================
    # 各组件参数量计算
    # ==================================================================

    def count_parameters_rfp(self, dim: int, num_waves: int) -> int:
        """计算谱混合层 (SGC 旧实现) 的参数量.

        SGC 组件参数:
          - omega_raw:  num_waves (共振频率)
          - alpha_raw:  num_waves (阻尼系数)
          - gamma:       num_waves (源项系数)
          - beta:        1         (全局阻尼)
          - out_scale:   1         (输出缩放)
          - in_proj:     dim * num_waves + num_waves  (输入投影)
          - out_proj:    num_waves * dim + dim         (输出投影)
          - norm:        2 * dim    (LayerNorm)

        Args:
            dim: 特征维度 d.
            num_waves: 波场通道数 W.

        Returns:
            SGC 层的总参数量.
        """
        omega = num_waves
        alpha = num_waves
        gamma = num_waves
        beta = 1
        out_scale = 1
        in_proj = dim * num_waves + num_waves   # Linear(dim, num_waves, bias=True)
        out_proj = num_waves * dim + dim         # Linear(num_waves, dim, bias=True)
        norm = 2 * dim                            # LayerNorm(dim)

        total = omega + alpha + gamma + beta + out_scale + in_proj + out_proj + norm
        return total

    def count_parameters_dmc(
        self,
        dim: int,
        num_channels: int,
        channel_size: int,
        num_heads: int,
    ) -> int:
        """计算动态记忆通道 (DMC) 的参数量.

        DMC 组件参数:
          - query_proj:       dim * dim           (无偏置)
          - key_proj:         dim * dim           (无偏置)
          - write_value_proj: dim * dim + dim     (有偏置)
          - output_proj:      dim * dim + dim     (有偏置)
          - write_gate:       dim * num_channels + num_channels  (有偏置)
          - channel_weights:  num_channels
          - norm:             2 * dim             (LayerNorm)

        Args:
            dim: 特征维度 d.
            num_channels: 记忆通道数 C.
            channel_size: 每通道 slot 数 S (不影响参数量, 记忆库不存储为参数).
            num_heads: 注意力头数 (不影响参数量, 仅影响 head_dim).

        Returns:
            DMC 层的总参数量.
        """
        query_proj = dim * dim                            # Linear(dim, dim, bias=False)
        key_proj = dim * dim                              # Linear(dim, dim, bias=False)
        write_value_proj = dim * dim + dim                # Linear(dim, dim, bias=True)
        output_proj = dim * dim + dim                     # Linear(dim, dim, bias=True)
        write_gate = dim * num_channels + num_channels    # Linear(dim, num_channels, bias=True)
        channel_weights = num_channels                     # Parameter
        norm = 2 * dim                                     # LayerNorm(dim)

        total = (
            query_proj + key_proj + write_value_proj + output_proj
            + write_gate + channel_weights + norm
        )
        return total

    def count_parameters_tar(self, dim: int, num_paths: int) -> int:
        """计算条件计算路由器 (CCR) 的参数量.

        CCR 组件参数:
          - router:      dim * num_paths + num_paths  (路由投影)
          - transform:   dim * dim + dim              (路径变换)
          - norm:        2 * dim                       (LayerNorm)
          - out_norm:    2 * dim                       (LayerNorm)
          - temperature: 1                              (温度参数)

        Args:
            dim: 特征维度 d.
            num_paths: 候选路径数 P.

        Returns:
            CCR 层的总参数量.
        """
        router = dim * num_paths + num_paths    # Linear(dim, num_paths, bias=True)
        transform = dim * dim + dim             # Linear(dim, dim, bias=True)
        norm = 2 * dim                          # LayerNorm(dim)
        out_norm = 2 * dim                      # LayerNorm(dim)
        temperature = 1                          # Parameter

        total = router + transform + norm + out_norm + temperature
        return total

    def count_parameters_ffn(self, dim: int, ratio: int) -> int:
        """计算前馈网络 (FFN) 的参数量.

        FFN 结构: Linear(dim, dim*ratio) → GELU → Linear(dim*ratio, dim)

        Args:
            dim: 特征维度 d.
            ratio: 隐藏层扩展比例 (如 4 表示 dim → 4*dim → dim).

        Returns:
            FFN 的总参数量.
        """
        hidden = dim * ratio
        fc1 = dim * hidden + hidden     # Linear(dim, hidden, bias=True)
        fc2 = hidden * dim + dim        # Linear(hidden, dim, bias=True)

        total = fc1 + fc2
        return total

    def count_parameters_embedding(self, vocab_size: int, dim: int) -> int:
        """计算词嵌入层的参数量.

        Args:
            vocab_size: 词表大小 V.
            dim: 嵌入维度 d.

        Returns:
            嵌入层的总参数量.
        """
        return vocab_size * dim

    def count_parameters_block(
        self,
        dim: int,
        num_waves: int,
        num_channels: int,
        channel_size: int,
        num_heads: int,
        num_paths: int,
        ffn_ratio: int,
    ) -> int:
        """计算单个 SimplCoreBlock 的参数量 (SGC + DMC + CCR + FFN + LayerNorm).

        Args:
            各组件参数, 见对应方法.

        Returns:
            单个 Block 的总参数量.
        """
        rfp = self.count_parameters_rfp(dim, num_waves)
        dmc = self.count_parameters_dmc(dim, num_channels, channel_size, num_heads)
        tar = self.count_parameters_tar(dim, num_paths)
        ffn = self.count_parameters_ffn(dim, ffn_ratio)
        # Block 级别的 LayerNorm (如果有)
        block_norm = 2 * dim

        return rfp + dmc + tar + ffn + block_norm

    def count_parameters_output_head(self, dim: int, vocab_size: int) -> int:
        """计算输出头 (LM Head) 的参数量.

        Args:
            dim: 特征维度 d.
            vocab_size: 词表大小 V.

        Returns:
            输出头参数量.
        """
        # Linear(dim, vocab_size, bias=False) 或带 bias
        return dim * vocab_size + vocab_size

    # ==================================================================
    # 总量估算
    # ==================================================================

    def estimate_total(self, config: SimplCoreConfig) -> Dict[str, int]:
        """估算给定配置下的模型总参数量.

        Args:
            config: SimplCoreConfig 配置实例.

        Returns:
            dict: 各组件及总量的参数数.
        """
        dim = config.hidden_size
        num_layers = config.num_layers

        embedding = self.count_parameters_embedding(config.vocab_size, dim)
        block_params = self.count_parameters_block(
            dim=dim,
            num_waves=config.num_waves,
            num_channels=config.num_channels,
            channel_size=config.channel_size,
            num_heads=config.num_heads,
            num_paths=config.num_paths,
            ffn_ratio=config.ffn_ratio,
        )
        output_head = self.count_parameters_output_head(dim, config.vocab_size)

        total_blocks = block_params * num_layers
        # 最终 LayerNorm
        final_norm = 2 * dim
        # 位置嵌入 (可学习)
        pos_embed = config.max_seq_len * dim

        total = embedding + total_blocks + output_head + final_norm + pos_embed

        return {
            "embedding": embedding,
            "blocks_total": total_blocks,
            "blocks_per_layer": block_params,
            "output_head": output_head,
            "final_norm": final_norm,
            "pos_embed": pos_embed,
            "total": total,
        }

    # ==================================================================
    # 格式化输出
    # ==================================================================

    def print_parameter_breakdown(self, config: SimplCoreConfig) -> None:
        """打印格式化的参数量分解.

        Args:
            config: SimplCoreConfig 配置实例.
        """
        dim = config.hidden_size
        params = self.estimate_total(config)

        print("=" * 65)
        print("SimplCore 参数量分解")
        print("=" * 65)
        print(f"  配置: hidden_size={dim}, num_layers={config.num_layers}, "
              f"vocab_size={config.vocab_size}")
        print(f"{'─' * 65}")

        # 各组件
        rfp = self.count_parameters_rfp(dim, config.num_waves)
        dmc = self.count_parameters_dmc(
            dim, config.num_channels, config.channel_size, config.num_heads
        )
        tar = self.count_parameters_tar(dim, config.num_paths)
        ffn = self.count_parameters_ffn(dim, config.ffn_ratio)

        print(f"  [单层组件]")
        print(f"    SGC (谱混合):       {rfp:>12,}  ({rfp / params['total'] * 100:.2f}%)")
        print(f"    DMC (动态记忆):      {dmc:>12,}  ({dmc / params['total'] * 100:.2f}%)")
        print(f"    CCR (条件计算路由):  {tar:>12,}  ({tar / params['total'] * 100:.2f}%)")
        print(f"    FFN (前馈网络):      {ffn:>12,}  ({ffn / params['total'] * 100:.2f}%)")
        print(f"    单层合计:            {params['blocks_per_layer']:>12,}")
        print(f"{'─' * 65}")

        print(f"  [全局组件]")
        print(f"    Embedding:           {params['embedding']:>12,}  "
              f"({params['embedding'] / params['total'] * 100:.2f}%)")
        print(f"    Blocks ({config.num_layers} 层):     {params['blocks_total']:>12,}  "
              f"({params['blocks_total'] / params['total'] * 100:.2f}%)")
        print(f"    Output Head:         {params['output_head']:>12,}  "
              f"({params['output_head'] / params['total'] * 100:.2f}%)")
        print(f"    Position Embed:      {params['pos_embed']:>12,}  "
              f"({params['pos_embed'] / params['total'] * 100:.2f}%)")
        print(f"    Final Norm:          {params['final_norm']:>12,}")
        print(f"{'─' * 65}")

        total = params["total"]
        if total >= 1e9:
            total_str = f"{total / 1e9:.2f} B"
        elif total >= 1e6:
            total_str = f"{total / 1e6:.2f} M"
        elif total >= 1e3:
            total_str = f"{total / 1e3:.2f} K"
        else:
            total_str = str(total)

        print(f"  总参数量:              {total:>12,}  ({total_str})")
        print(f"{'=' * 65}")

    # ==================================================================
    # 复杂度对比表
    # ==================================================================

    def complexity_table(self, seq_lengths: List[int]) -> None:
        """打印不同序列长度下的复杂度对比表.

        对比 SGC (O(n log n)) 与标准注意力 (O(n^2)) 的理论 FLOPs。

        Args:
            seq_lengths: 待对比的序列长度列表.
        """
        print("=" * 75)
        print("SimplCore 复杂度对比表: SGC (O(n log n)) vs 标准注意力 (O(n^2))")
        print("=" * 75)
        print(f"  {'序列长度':>10}  {'SGC FLOPs':>15}  {'Attn FLOPs':>15}  "
              f"{'加速比':>10}  {'SGC Big-O':>12}  {'Attn Big-O':>12}")
        print(f"{'─' * 75}")

        for n in seq_lengths:
            # SGC: 2*n*d*w + T*(5*n*log2(n) + n*w) ≈ O(n log n)
            # 简化: 使用 n * log2(n) 作为 SGC 复杂度
            sgc_flops = n * math.log2(max(n, 2))
            # 标准注意力: n^2 * d
            attn_flops = n * n

            speedup = attn_flops / max(sgc_flops, 1.0)

            print(f"  {n:>10,}  {sgc_flops:>15,.0f}  {attn_flops:>15,.0f}  "
                  f"{speedup:>9.1f}x  {'O(n log n)':>12}  {'O(n^2)':>12}")

        print(f"{'─' * 75}")
        print("  注: FLOPs 为归一化值 (d=1, w=1, T=1), 实际值需乘以相应系数。")
        print("  SGC 的 FFT 拉普拉斯算子计算为 O(n log n), 标准注意力 QK^T 为 O(n^2)。")
        print("=" * 75)

    # ==================================================================
    # 理论性质验证
    # ==================================================================

    def verify_parameter_growth(
        self,
        config: SimplCoreConfig,
        max_layers: int = 100,
    ) -> Dict[str, Any]:
        """验证参数量随层数的线性增长性质.

        SimplCore 的参数量应与 num_layers 成线性关系。

        Args:
            config: 基准配置.
            max_layers: 最大测试层数.

        Returns:
            dict: 验证结果.
        """
        results = []
        base_config = config
        for n_layers in [1, 2, 3, 4, 5, 6, 7, 8]:
            test_config = self._copy_config_with_layers(config, n_layers)
            params = self.estimate_total(test_config)
            results.append({
                "num_layers": n_layers,
                "total_params": params["total"],
            })

        # 验证线性增长
        is_linear = True
        if len(results) >= 2:
            base_increment = results[1]["total_params"] - results[0]["total_params"]
            for i in range(1, len(results)):
                increment = results[i]["total_params"] - results[i - 1]["total_params"]
                if abs(increment - base_increment) > base_increment * 0.01:
                    is_linear = False
                    break

        return {
            "is_linear_growth": is_linear,
            "params_per_layer": results[1]["total_params"] - results[0]["total_params"]
            if len(results) >= 2
            else 0,
            "data": results,
        }

    def _copy_config_with_layers(
        self,
        config: SimplCoreConfig,
        num_layers: int,
    ) -> SimplCoreConfig:
        """复制配置并修改层数."""
        if hasattr(config, "__dataclass_fields__"):
            # dataclass 配置
            import dataclasses
            return dataclasses.replace(config, num_layers=num_layers)
        else:
            # 普通对象, 手动复制
            new_config = type(config)()
            for attr in dir(config):
                if not attr.startswith("_") and not callable(getattr(config, attr)):
                    setattr(new_config, attr, getattr(config, attr))
            new_config.num_layers = num_layers
            return new_config


# ======================================================================
# 预设配置参数量表
# ======================================================================

def print_preset_comparison() -> None:
    """打印所有预设配置的参数量对比."""
    theory = ArchitectureTheory()
    presets = ["tiny", "150m", "7b", "14b"]

    print("=" * 65)
    print("SimplCore 预设配置参数量对比")
    print("=" * 65)
    print(f"  {'预设':>8}  {'hidden_size':>12}  {'num_layers':>10}  "
          f"{'总参数量':>15}  {'人类可读':>12}")
    print(f"{'─' * 65}")

    for name in presets:
        try:
            config = SimplCoreConfig.from_preset(name)
            params = theory.estimate_total(config)
            total = params["total"]

            if total >= 1e9:
                readable = f"{total / 1e9:.2f} B"
            elif total >= 1e6:
                readable = f"{total / 1e6:.2f} M"
            elif total >= 1e3:
                readable = f"{total / 1e3:.2f} K"
            else:
                readable = str(total)

            print(f"  {name:>8}  {config.hidden_size:>12}  {config.num_layers:>10}  "
                  f"{total:>15,}  {readable:>12}")
        except Exception as e:
            print(f"  {name:>8}  (配置不可用: {e})")

    print(f"{'=' * 65}")


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("ArchitectureTheory 自检")
    print("=" * 65)

    theory = ArchitectureTheory()

    # 测试 SGC 参数量
    sgc_params = theory.count_parameters_rfp(dim=768, num_waves=8)
    print(f"SGC (dim=768, waves=8): {sgc_params:,} params")
    assert sgc_params > 0
    # 手动验证: 3*8 + 1 + 1 + (768*8+8) + (8*768+768) + 2*768
    # = 24 + 2 + 6152 + 6912 + 1536 = 14626
    expected_rfp = 3 * 8 + 1 + 1 + (768 * 8 + 8) + (8 * 768 + 768) + 2 * 768
    assert sgc_params == expected_rfp, f"SGC: {sgc_params} != {expected_rfp}"
    print(f"[PASS] SGC 参数量正确: {sgc_params}")

    # 测试 DMC 参数量
    dmc_params = theory.count_parameters_dmc(
        dim=768, num_channels=64, channel_size=128, num_heads=8
    )
    print(f"DMC (dim=768, ch=64, size=128): {dmc_params:,} params")
    assert dmc_params > 0
    # 手动验证: 768*768 + 768*768 + (768*768+768) + (768*768+768) + (768*64+64) + 64 + 2*768
    expected_dmc = (
        768 * 768 + 768 * 768 + (768 * 768 + 768) + (768 * 768 + 768)
        + (768 * 64 + 64) + 64 + 2 * 768
    )
    assert dmc_params == expected_dmc, f"DMC: {dmc_params} != {expected_dmc}"
    print(f"[PASS] DMC 参数量正确: {dmc_params}")

    # 测试 CCR 参数量
    ccr_params = theory.count_parameters_tar(dim=768, num_paths=4)
    print(f"CCR (dim=768, paths=4): {ccr_params:,} params")
    assert ccr_params > 0
    expected_tar = (768 * 4 + 4) + (768 * 768 + 768) + 2 * 768 + 2 * 768 + 1
    assert ccr_params == expected_tar, f"CCR: {ccr_params} != {expected_tar}"
    print(f"[PASS] CCR 参数量正确: {ccr_params}")

    # 测试 FFN 参数量
    ffn_params = theory.count_parameters_ffn(dim=768, ratio=4)
    print(f"FFN (dim=768, ratio=4): {ffn_params:,} params")
    assert ffn_params > 0
    expected_ffn = (768 * 3072 + 3072) + (3072 * 768 + 768)
    assert ffn_params == expected_ffn, f"FFN: {ffn_params} != {expected_ffn}"
    print(f"[PASS] FFN 参数量正确: {ffn_params}")

    # 测试 Embedding 参数量
    emb_params = theory.count_parameters_embedding(vocab_size=32000, dim=768)
    print(f"Embedding (vocab=32000, dim=768): {emb_params:,} params")
    assert emb_params == 32000 * 768
    print(f"[PASS] Embedding 参数量正确: {emb_params}")

    # 测试总量估算
    config = SimplCoreConfig.from_preset("150m")
    total = theory.estimate_total(config)
    print(f"\n150m 配置参数量估算:")
    for k, v in total.items():
        print(f"  {k}: {v:,}")
    assert total["total"] > 0
    print(f"[PASS] 总量估算正常: {total['total']:,}")

    # 测试参数量分解打印
    print()
    theory.print_parameter_breakdown(config)

    # 测试复杂度表
    print()
    theory.complexity_table([128, 512, 1024, 4096, 16384, 65536])

    # 测试线性增长验证
    print()
    growth = theory.verify_parameter_growth(config, max_layers=64)
    print(f"参数线性增长: {growth['is_linear_growth']}")
    print(f"每层参数量: {growth['params_per_layer']:,}")
    assert growth["is_linear_growth"], "参数量应随层数线性增长"
    print("[PASS] 参数线性增长验证通过")

    # 测试预设对比
    print()
    print_preset_comparison()

    print("\n" + "=" * 65)
    print("ArchitectureTheory 所有测试 PASSED.")
    print("=" * 65)
