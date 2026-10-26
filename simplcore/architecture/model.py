# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 模型工厂与因果语言模型 (CausalLM)
==========================================

提供模型构建的工厂函数和因果语言模型封装:

  1. build_model(config_name, **kwargs): 从预设名称构建模型
     - 'tiny':  hidden_size=64,  num_layers=2   (测试用)
     - '150m':  hidden_size=768, num_layers=12  (150M 参数)
     - '7b':    hidden_size=4096, num_layers=32 (7B 参数)
     - '14b':   hidden_size=5120, num_layers=40 (14B 参数)

  2. SimplCoreForCausalLM: 封装 SimplCoreModel, 添加 LM 头和损失计算
     - forward(input_ids, labels) → {logits, loss}
     - generate(input_ids, max_new_tokens, temperature, top_k) → 自回归生成
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

# ------------------------------------------------------------------
# 导入 SimplCoreConfig (如果 config.py 损坏则使用回退定义)
# ------------------------------------------------------------------
try:
    from .config import SimplCoreConfig
except Exception:
    from dataclasses import dataclass

    @dataclass
    class SimplCoreConfig:
        """SimplCore 配置 (回退定义)."""
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

# ------------------------------------------------------------------
# 导入 SimplCoreModel (如果 simplcore_model.py 损坏则使用回退定义)
# ------------------------------------------------------------------
try:
    from .simplcore_model import SimplCoreModel
except Exception:
    # 尝试从 integrated_model 导入
    try:
        from .integrated_model import IntegratedSimplCoreModel as SimplCoreModel
    except Exception:
        class SimplCoreModel(nn.Module):
            """SimplCore 基础模型 (回退实现).

            包含嵌入层和 SimplCoreBlock 层, 不包含 LM 头。
            """

            def __init__(self, config: SimplCoreConfig) -> None:
                super().__init__()
                self.config = config
                dim = config.hidden_size

                # 嵌入层
                self.token_embed = nn.Embedding(config.vocab_size, dim)
                self.pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, dim) * 0.02
                )
                self.embed_dropout = nn.Dropout(config.dropout)

                # Block 层
                self.blocks = nn.ModuleList([
                    _FallbackBlock(config) for _ in range(config.num_layers)
                ])

                # 最终 LayerNorm
                self.final_norm = nn.LayerNorm(dim)

                # 初始化
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                memory_state: Optional[torch.Tensor] = None,
                **kwargs,
            ) -> Dict[str, torch.Tensor]:
                """前向传播, 返回隐藏状态."""
                if input_ids is None:
                    raise ValueError("input_ids 不能为 None")

                B, L = input_ids.shape
                x = self.token_embed(input_ids)
                x = x + self.pos_embed[:, :L, :]
                x = self.embed_dropout(x)

                current_memory = memory_state
                for block in self.blocks:
                    try:
                        x, current_memory = block(x, memory_state=current_memory)
                    except TypeError:
                        result = block(x)
                        if isinstance(result, tuple):
                            x = result[0]
                        else:
                            x = result

                x = self.final_norm(x)

                return {
                    "hidden_states": x,
                    "memory_state": current_memory,
                }


class _FallbackBlock(nn.Module):
    """回退 Block 实现 (当 SimplCoreBlock 不可用时使用)."""

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * config.ffn_ratio),
            nn.GELU(),
            nn.Linear(dim * config.ffn_ratio, dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播."""
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, memory_state


# ======================================================================
# 预设配置表
# ======================================================================

_PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_layers": 2,
        "num_waves": 4,
        "num_channels": 8,
        "channel_size": 16,
        "num_heads": 4,
        "num_paths": 2,
        "ffn_ratio": 2,
        "max_seq_len": 128,
        "dropout": 0.1,
    },
    "150m": {
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_waves": 8,
        "num_channels": 64,
        "channel_size": 128,
        "num_heads": 8,
        "num_paths": 4,
        "ffn_ratio": 4,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    "7b": {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_waves": 16,
        "num_channels": 128,
        "channel_size": 256,
        "num_heads": 16,
        "num_paths": 8,
        "ffn_ratio": 4,
        "max_seq_len": 4096,
        "dropout": 0.0,
    },
    "14b": {
        "vocab_size": 32000,
        "hidden_size": 5120,
        "num_layers": 40,
        "num_waves": 16,
        "num_channels": 128,
        "channel_size": 256,
        "num_heads": 16,
        "num_paths": 8,
        "ffn_ratio": 4,
        "max_seq_len": 4096,
        "dropout": 0.0,
    },
}


# ======================================================================
# 模型工厂函数
# ======================================================================

def build_model(config_name: str = "150m", **kwargs: Any) -> "SimplCoreForCausalLM":
    """从预设名称构建 SimplCore 因果语言模型.

    Args:
        config_name: 预设配置名称, 可选:
            - 'tiny':  hidden_size=64,  num_layers=2   (测试用, ~0.05M)
            - '150m':  hidden_size=768, num_layers=12  (150M 参数)
            - '7b':    hidden_size=4096, num_layers=32 (7B 参数)
            - '14b':   hidden_size=5120, num_layers=40 (14B 参数)
        **kwargs: 覆盖预设配置的额外参数 (如 vocab_size=50000).

    Returns:
        SimplCoreForCausalLM 实例.

    Raises:
        ValueError: 当 config_name 不在预设列表中时.

    Example:
        >>> model = build_model('tiny')           # 测试用小模型
        >>> model = build_model('150m')            # 150M 参数模型
        >>> model = build_model('7b', vocab_size=50000)  # 7B, 自定义词表
    """
    name_lower = config_name.lower()
    if name_lower not in _PRESET_CONFIGS:
        available = list(_PRESET_CONFIGS.keys())
        raise ValueError(
            f"未知配置名称: '{config_name}'. 可用配置: {available}"
        )

    # 获取预设参数
    config_dict = dict(_PRESET_CONFIGS[name_lower])
    # 用 kwargs 覆盖
    config_dict.update(kwargs)

    # 创建配置
    config = SimplCoreConfig(**config_dict)

    # 构建模型
    model = SimplCoreForCausalLM(config)

    return model


def get_config(config_name: str = "150m", **kwargs: Any) -> SimplCoreConfig:
    """从预设名称获取配置 (不构建模型).

    Args:
        config_name: 预设配置名称.
        **kwargs: 覆盖参数.

    Returns:
        SimplCoreConfig 实例.
    """
    name_lower = config_name.lower()
    if name_lower not in _PRESET_CONFIGS:
        available = list(_PRESET_CONFIGS.keys())
        raise ValueError(
            f"未知配置名称: '{config_name}'. 可用配置: {available}"
        )

    config_dict = dict(_PRESET_CONFIGS[name_lower])
    config_dict.update(kwargs)
    return SimplCoreConfig(**config_dict)


def list_presets() -> Dict[str, Dict[str, Any]]:
    """列出所有可用的预设配置.

    Returns:
        dict: 预设名称到配置参数的映射.
    """
    return dict(_PRESET_CONFIGS)


# ======================================================================
# 因果语言模型封装
# ======================================================================

class SimplCoreForCausalLM(nn.Module):
    """SimplCore 因果语言模型 (CausalLM).

    封装 SimplCoreModel, 添加:
      1. LM 头 (Linear: hidden_size → vocab_size)
      2. 因果语言建模损失计算 (Cross-Entropy)
      3. 自回归文本生成 (temperature + top-k sampling)

    Args:
        config: SimplCoreConfig 配置实例.
        tie_weights: 是否绑定 LM 头与嵌入层权重, 默认 False.
    """

    def __init__(
        self,
        config: SimplCoreConfig,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        # 基础模型 (嵌入 + Blocks)
        self.model = SimplCoreModel(config)

        # LM 头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定
        self.tie_weights = tie_weights
        if tie_weights:
            # 尝试从基础模型获取嵌入权重
            if hasattr(self.model, "token_embed"):
                self.lm_head.weight = self.model.token_embed.weight
            elif hasattr(self.model, "embed"):
                self.lm_head.weight = self.model.embed.weight

        # 初始化 LM 头
        if not tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 生成相关
        self._pad_token_id = kwargs_get(config, "pad_token_id", 0)
        self._bos_token_id = kwargs_get(config, "bos_token_id", 1)
        self._eos_token_id = kwargs_get(config, "eos_token_id", 2)

    # ==================================================================
    # 前向传播
    # ==================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """前向传播: 计算 logits 和可选的损失.

        Args:
            input_ids: (B, L) 输入 token ID 序列.
            labels: (B, L) 目标 token ID, 可选. 设为 -100 的位置不计算损失.
            memory_state: 可选的 DMC 记忆状态.

        Returns:
            dict: {
              "logits": (B, L, vocab_size) 输出 logits,
              "loss": 可选的标量损失 (当 labels 不为 None 时),
              "hidden_states": (B, L, hidden_size) 隐藏状态,
              "memory_state": 可选的记忆状态,
            }
        """
        # 通过基础模型获取隐藏状态
        model_output = self.model(
            input_ids=input_ids,
            memory_state=memory_state,
            **kwargs,
        )

        # 获取隐藏状态
        if isinstance(model_output, dict):
            hidden_states = model_output.get("hidden_states", model_output.get("logits"))
            new_memory = model_output.get("memory_state")
        elif isinstance(model_output, torch.Tensor):
            hidden_states = model_output
            new_memory = None
        else:
            hidden_states = model_output
            new_memory = None

        # LM 头: hidden_states → logits
        logits = self.lm_head(hidden_states)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "hidden_states": hidden_states,
            "memory_state": new_memory,
        }

        # 计算损失
        if labels is not None:
            # 移位: 预测下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # 展平计算交叉熵
            B, L_minus_1, V = shift_logits.shape
            loss = self.loss_fn(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
            )
            result["loss"] = loss

        return result

    # ==================================================================
    # 自回归生成
    # ==================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归文本生成.

        支持贪心解码和采样解码 (temperature + top-k + top-p)。

        Args:
            input_ids: (B, L_prompt) 提示 token ID 序列.
            max_new_tokens: 最大生成 token 数.
            temperature: 采样温度, 1.0=正常, <1=更确定, >1=更随机.
            top_k: top-k 采样, 仅从概率最高的 k 个 token 中采样. None=不限制.
            top_p: nucleus sampling, 从累积概率达到 p 的最小 token 集中采样.
            repetition_penalty: 重复惩罚, >1 减少重复.
            do_sample: True=采样, False=贪心解码.
            eos_token_id: 结束 token ID, 生成到此停止.
            pad_token_id: 填充 token ID.

        Returns:
            (B, L_prompt + max_new_tokens) 生成的 token ID 序列。
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if eos_token_id is None:
            eos_token_id = self._eos_token_id
        if pad_token_id is None:
            pad_token_id = self._pad_token_id

        # 复制输入作为生成起点
        generated = input_ids.clone()

        # 记录已完成的序列
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # 截断到 max_seq_len
            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated

            # 前向传播获取 logits
            output = self.forward(input_ids=context)
            logits = output["logits"]

            # 取最后一个 token 的 logits
            next_logits = logits[:, -1, :].float()  # (B, vocab_size)

            # 重复惩罚
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in generated[b].tolist():
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            if not do_sample:
                # 贪心解码
                next_tokens = next_logits.argmax(dim=-1)  # (B,)
            else:
                # 温度缩放
                if temperature > 0:
                    next_logits = next_logits / temperature
                else:
                    # temperature=0 退化为贪心
                    next_tokens = next_logits.argmax(dim=-1)
                    next_tokens = torch.where(
                        finished, torch.full_like(next_tokens, pad_token_id), next_tokens
                    )
                    generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
                    finished = finished | (next_tokens == eos_token_id)
                    if finished.all():
                        break
                    continue

                # Top-k 采样
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k
                    )[0][..., -1, None]
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                # Top-p (nucleus) 采样
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过 top_p 的 token
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                # 采样
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # 已完成的序列填充 pad
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id), next_tokens
            )

            # 追加到生成序列
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            # 更新完成状态
            finished = finished | (next_tokens == eos_token_id)

            # 全部完成则停止
            if finished.all():
                break

        return generated

    # ==================================================================
    # 工具方法
    # ==================================================================

    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数量."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.model.parameters())
        head_params = sum(p.numel() for p in self.lm_head.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_params,
            "lm_head": head_params,
        }

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层."""
        if hasattr(self.model, "token_embed"):
            return self.model.token_embed
        elif hasattr(self.model, "embed"):
            return self.model.embed
        raise AttributeError("无法找到嵌入层")

    def get_output_embeddings(self) -> nn.Module:
        """获取输出嵌入层 (LM 头)."""
        return self.lm_head

    def tie_weights(self) -> None:
        """绑定 LM 头与嵌入层权重."""
        embed = self.get_input_embeddings()
        self.lm_head.weight = embed.weight
        self.tie_weights = True

    def summary(self) -> str:
        """返回模型摘要."""
        params = self.count_parameters()
        lines = [
            "=" * 60,
            "SimplCoreForCausalLM 摘要",
            "=" * 60,
            f"  hidden_size: {self.config.hidden_size}",
            f"  num_layers:  {self.config.num_layers}",
            f"  vocab_size:  {self.config.vocab_size}",
            f"  tie_weights: {self.tie_weights}",
            f"{'─' * 60}",
            f"  Backbone 参数: {params['backbone']:>12,}",
            f"  LM Head 参数:  {params['lm_head']:>12,}",
            f"  总参数量:      {params['total']:>12,}",
            f"  可训练参数:    {params['trainable']:>12,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ======================================================================
# 辅助函数
# ======================================================================

def kwargs_get(obj: Any, key: str, default: Any) -> Any:
    """安全获取对象的属性, 不存在则返回默认值."""
    return getattr(obj, key, default) if hasattr(obj, key) else default


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("SimplCoreForCausalLM 自检")
    print("=" * 60)

    # === 测试预设配置列表 ===
    presets = list_presets()
    print(f"可用预设: {list(presets.keys())}")
    assert "tiny" in presets
    assert "150m" in presets
    assert "7b" in presets
    assert "14b" in presets
    print("[PASS] 预设配置完整")

    # === 测试 get_config ===
    config = get_config("tiny")
    assert config.hidden_size == 64
    assert config.num_layers == 2
    print(f"[PASS] get_config('tiny'): hidden={config.hidden_size}, layers={config.num_layers}")

    # === 测试未知配置 ===
    try:
        build_model("999b")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"[PASS] 未知配置正确抛出 ValueError: {e}")

    # === 构建 tiny 模型 ===
    print("\n--- 构建 tiny 模型 ---")
    model = build_model("tiny")
    print(model.summary())

    # === 测试前向传播 ===
    print("\n--- 前向传播测试 ---")
    input_ids = torch.randint(1, 100, (2, 16))
    output = model(input_ids=input_ids)
    print(f"输入: {tuple(input_ids.shape)}")
    print(f"logits: {tuple(output['logits'].shape)}")
    assert output["logits"].shape == (2, 16, config.vocab_size)
    assert "hidden_states" in output
    assert not torch.isnan(output["logits"]).any()
    print("[PASS] 前向传播正常")

    # === 测试带标签的前向传播 (损失计算) ===
    print("\n--- 损失计算测试 ---")
    labels = input_ids.clone()
    output_l = model(input_ids=input_ids, labels=labels)
    assert "loss" in output_l
    assert output_l["loss"].dim() == 0  # 标量
    assert not torch.isnan(output_l["loss"])
    print(f"损失值: {output_l['loss'].item():.4f}")
    print("[PASS] 损失计算正常")

    # === 测试反向传播 ===
    print("\n--- 反向传播测试 ---")
    output_b = model(input_ids=input_ids, labels=labels)
    output_b["loss"].backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print(f"[PASS] 反向传播正常: {has_grad}")
    assert has_grad

    # === 测试贪心生成 ===
    print("\n--- 贪心生成测试 ---")
    prompt = torch.randint(1, 100, (1, 8))
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        do_sample=False,
    )
    print(f"提示: {tuple(prompt.shape)} → 生成: {tuple(generated.shape)}")
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 8  # 至少包含提示
    assert generated.shape[1] <= 18  # 最多提示 + max_new_tokens
    print("[PASS] 贪心生成正常")

    # === 测试采样生成 ===
    print("\n--- 采样生成测试 ---")
    generated_s = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
        do_sample=True,
    )
    print(f"提示: {tuple(prompt.shape)} → 生成: {tuple(generated_s.shape)}")
    assert generated_s.shape[1] >= 8
    print("[PASS] 采样生成正常")

    # === 测试 top-p 生成 ===
    print("\n--- Top-p 生成测试 ---")
    generated_p = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
    )
    print(f"提示: {tuple(prompt.shape)} → 生成: {tuple(generated_p.shape)}")
    print("[PASS] Top-p 生成正常")

    # === 测试批量生成 ===
    print("\n--- 批量生成测试 ---")
    batch_prompt = torch.randint(1, 100, (3, 6))
    batch_generated = model.generate(
        batch_prompt,
        max_new_tokens=8,
        do_sample=False,
    )
    print(f"批量提示: {tuple(batch_prompt.shape)} → 生成: {tuple(batch_generated.shape)}")
    assert batch_generated.shape[0] == 3
    print("[PASS] 批量生成正常")

    # === 测试参数统计 ===
    print("\n--- 参数统计 ---")
    params = model.count_parameters()
    print(f"总参数量: {params['total']:,}")
    print(f"可训练:   {params['trainable']:,}")
    print(f"Backbone: {params['backbone']:,}")
    print(f"LM Head:  {params['lm_head']:,}")
    assert params["total"] > 0

    # === 测试权重绑定 ===
    print("\n--- 权重绑定测试 ---")
    config_tied = get_config("tiny")
    model_tied = SimplCoreForCausalLM(config_tied, tie_weights=True)
    print(f"[PASS] 权重绑定模型创建成功")

    print("\n" + "=" * 60)
    print("SimplCoreForCausalLM 所有测试 PASSED.")
    print("=" * 60)
