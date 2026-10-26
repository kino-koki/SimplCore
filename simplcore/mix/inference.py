# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Mix 生产级推理引擎
============================

功能特性:
  1. KV Cache 优化的自回归生成 — 首步处理完整 prompt, 后续步只处理最后一个 token
  2. 多种采样策略:
     - Greedy decoding (贪心)
     - Temperature sampling (温度采样)
     - Top-K sampling
     - Top-P (nucleus) sampling
     - Beam search (束搜索)
  3. 重复惩罚 (Repetition Penalty)
  4. 上下文窗口管理: 超过 max_seq_len 时滑动窗口 (重置缓存 + 重新编码)
  5. 流式生成: yield 逐 token 输出 (字节级增量解码, 正确处理多字节 UTF-8)
  6. 批量生成: 支持 batch_size > 1 (左填充 + KV Cache 批量复用)

双后端兼容:
  - PyTorch 后端: 使用 torch.no_grad() 和 model.eval()
  - NumPy 后端:   使用 simplcore.tensor.no_grad()

使用示例
--------
    from simplcore.mix.model import SimplCoreMix
    from simplcore.mix.config import mix_1b
    from simplcore.mix.inference import InferenceEngine, GenerationConfig
    from simplcore.mix.tokenizer import BPETokenizer

    cfg = mix_1b()
    model = SimplCoreMix(cfg)
    tokenizer = BPETokenizer(cfg.vocab_size)
    tokenizer.load("tokenizer.json")

    engine = InferenceEngine(model, tokenizer, cfg)

    # 贪心生成
    text = engine.generate("你好", GenerationConfig(strategy="greedy"))

    # 温度采样
    text = engine.generate("你好", GenerationConfig(
        strategy="sample", temperature=0.8, top_k=50, top_p=0.9
    ))

    # 流式生成
    for chunk in engine.generate_stream("你好", GenerationConfig(strategy="sample")):
        print(chunk, end="")

    # 批量生成
    results = engine.batch_generate(["你好", "世界"], GenerationConfig(batch_size=2))

    # 束搜索
    text = engine.generate("你好", GenerationConfig(
        strategy="beam", beam_size=4, length_penalty=1.0
    ))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import simplcore.tensor as torch
from simplcore.tensor import backend as _backend  # noqa: F401 (kept for API symmetry)

from .model import SimplCoreMix
from .config import SimplCoreMixConfig


# ============================================================
# 后端兼容辅助函数
# ============================================================

def _tolist(t) -> list:
    """将张量转为 Python 列表 (双后端兼容)。

    PyTorch 后端: tensor.tolist()
    NumPy 后端:   tensor.data.tolist()  (Tensor 类无 tolist 方法)
    """
    if hasattr(t, "tolist") and callable(getattr(t, "tolist", None)):
        return t.tolist()
    return t.data.tolist()


def _to_item(t):
    """获取张量的标量值 (双后端兼容)。"""
    return t.item()


# ============================================================
# KV Cache 管理
# ============================================================

class KVCache:
    """KV 缓存管理器。

    封装模型 ``forward`` 所需的 ``kv_cache`` 字典, 提供:

    - **reset()**: 清空缓存
    - **is_empty()**: 判断缓存是否为空
    - **truncate(keep_len)**: 截断保留最后 ``keep_len`` 个位置 (滑动窗口)
    - **reorder(indices)**: 按 ``indices`` 重排 batch 维度 (束搜索)

    跨步复用机制:
        首步 forward 时传入空 dict, 模型会将每层的 K/V 写入;
        后续步 forward 传入同一 dict, 模型自动拼接新旧 K/V。
        由此实现 O(1) 的增量解码。

    Attributes:
        max_seq_len: 最大序列长度 (用于滑动窗口判断)
        data:        底层 KV 缓存字典 {layer_key: {k: tensor, v: tensor}}
        length:      当前缓存的序列长度
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.length: int = 0

    def reset(self) -> None:
        """清空缓存。"""
        self.data = {}
        self.length = 0

    def is_empty(self) -> bool:
        """缓存是否为空。"""
        return len(self.data) == 0

    def truncate(self, keep_len: int) -> None:
        """截断缓存, 仅保留最后 ``keep_len`` 个位置。

        注意: 此操作不更新 RoPE 位置编码。对于使用 RoPE 的模型,
        截断后缓存中 K/V 的 RoPE 位置与实际位置不匹配,
        可能影响生成质量。如需正确滑动窗口, 请使用 reset + 重新编码。
        """
        if keep_len >= self.length or keep_len <= 0:
            return
        for key in list(self.data.keys()):
            for sub_key in list(self.data[key].keys()):
                tensor = self.data[key][sub_key]
                self.data[key][sub_key] = tensor[:, -keep_len:]
        self.length = keep_len

    def reorder(self, indices) -> None:
        """按 ``indices`` 重排 batch 维度 (束搜索时重排缓存)。

        Args:
            indices: shape [beam_size] 的索引张量或列表
        """
        if isinstance(indices, (list, tuple)):
            indices = torch.tensor(indices, dtype=torch.int64)
        for key in self.data:
            for sub_key in self.data[key]:
                self.data[key][sub_key] = self.data[key][sub_key][indices]

    # ── dict-like 接口 (供模型 forward 中 ``cache_key in kv_cache`` 使用) ──
    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> dict:
        return self.data[key]

    def __setitem__(self, key: str, value: dict) -> None:
        self.data[key] = value
        # 更新缓存长度
        if isinstance(value, dict):
            for v in value.values():
                if hasattr(v, 'shape') and len(v.shape) >= 2:
                    self.length = max(self.length, v.shape[1])
                    break


# ============================================================
# 生成配置
# ============================================================

@dataclass
class GenerationConfig:
    """生成配置。

    采样策略 (``strategy``):
        - ``"auto"``   (默认): 自动选择 (见 resolve_strategy)
        - ``"greedy"``: 贪心解码, 每步选概率最高的 token
        - ``"sample"``: 采样解码, 依次应用 temperature / top_k / top_p 过滤
        - ``"beam"``:  束搜索, 维护 beam_size 个候选序列

    属性:
        max_tokens:          最大生成 token 数
        temperature:         采样温度 (1.0=不缩放, <1 更确定, >1 更随机, 0=贪心)
        top_k:               Top-K 采样 (0=禁用)
        top_p:               Top-P (nucleus) 采样 (1.0=禁用)
        beam_size:           束搜索束宽 (1=不使用束搜索)
        repetition_penalty:  重复惩罚 (1.0=禁用, >1 惩罚重复)
        strategy:            采样策略
        max_seq_len:         上下文窗口长度 (None=使用模型配置)
        n_loops:             推理循环深度 (None=使用模型配置)
        eos_token_id:        终止 token ID (None=使用 tokenizer 的 eos)
        pad_token_id:        填充 token ID (None=使用 tokenizer 的 pad)
        batch_size:          批量生成时的批大小
        seed:                随机种子 (None=不设置)
        early_stopping:      束搜索是否在所有 beam 完成 EOS 后提前终止
        length_penalty:      长度惩罚 (1.0=按长度归一化, 0.0=不归一化)
        do_sample:           是否启用采样 (strategy="auto" 时影响策略选择)
    """
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    beam_size: int = 1
    repetition_penalty: float = 1.0
    strategy: str = "auto"
    max_seq_len: Optional[int] = None
    n_loops: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    batch_size: int = 1
    seed: Optional[int] = None
    early_stopping: bool = True
    length_penalty: float = 1.0
    do_sample: bool = False

    def resolve_strategy(self) -> str:
        """解析实际使用的采样策略。

        优先级:
            1. strategy != "auto" -> 直接返回
            2. beam_size > 1      -> "beam"
            3. temperature == 0   -> "greedy"
            4. do_sample 或 temperature != 1.0 或 top_k > 0 或 top_p < 1.0 -> "sample"
            5. 默认               -> "greedy"
        """
        if self.strategy != "auto":
            return self.strategy
        if self.beam_size > 1:
            return "beam"
        if self.temperature == 0:
            return "greedy"
        if self.do_sample or self.temperature != 1.0 or self.top_k > 0 or self.top_p < 1.0:
            return "sample"
        return "greedy"


# ============================================================
# 推理引擎
# ============================================================

class InferenceEngine:
    """生产级推理引擎。

    支持 KV Cache 优化、多种采样策略、重复惩罚、滑动窗口、
    流式生成和批量生成。兼容自研 NumPy 后端和 PyTorch 后端。

    Args:
        model:     SimplCoreMix 模型实例
        tokenizer: 分词器 (需支持 encode / decode / eos_token_id / pad_token_id)
        config:    SimplCoreMixConfig 模型配置 (None=使用 model.cfg)
    """

    def __init__(
        self,
        model: SimplCoreMix,
        tokenizer,
        config: Optional[SimplCoreMixConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # 模型配置
        if config is not None:
            self.cfg = config
        elif hasattr(model, "cfg"):
            self.cfg = model.cfg
        else:
            from .config import mix_tiny_text
            self.cfg = mix_tiny_text()

        # 自研引擎后端（青小团张量引擎，恒为 CPU / float32，无第三方框架依赖）
        self.is_torch = False
        self._real_torch = None
        self.device = "cpu"

        # 推理模式
        self.model.eval()

    # ========================================================
    # 配置解析辅助
    # ========================================================

    def _resolve_eos_id(self, config: GenerationConfig) -> int:
        if config.eos_token_id is not None:
            return config.eos_token_id
        if hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id
        return 1

    def _resolve_pad_id(self, config: GenerationConfig) -> int:
        if config.pad_token_id is not None:
            return config.pad_token_id
        if hasattr(self.tokenizer, "pad_token_id"):
            return self.tokenizer.pad_token_id
        return 0

    def _resolve_max_seq_len(self, config: GenerationConfig) -> int:
        return config.max_seq_len or self.cfg.max_seq_len

    def _resolve_n_loops(self, config: GenerationConfig) -> Optional[int]:
        return config.n_loops if config.n_loops is not None else self.cfg.max_loop_iters

    @staticmethod
    def _resolve_config(config: Optional[GenerationConfig]) -> GenerationConfig:
        return config if config is not None else GenerationConfig()

    def _no_grad(self):
        """获取 no_grad 上下文管理器 (双后端兼容)。

        - PyTorch 后端: torch.no_grad()
        - NumPy 后端:   simplcore.tensor.no_grad()
        """
        return torch.no_grad()

    # ========================================================
    # 编码 / 解码辅助
    # ========================================================

    def _encode(self, prompt: str, max_seq_len: Optional[int] = None) -> torch.Tensor:
        """编码文本为 token ID 张量 ``[1, T]``。

        若 prompt 为空, 使用 EOS token。
        若 prompt 超过 max_seq_len, 截断保留最后 max_seq_len 个 token。
        """
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [self._resolve_eos_id(GenerationConfig())]
        if max_seq_len is not None and len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:]
        return torch.tensor([tokens], dtype=torch.int64)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """解码 token ID 列表为文本。"""
        return self.tokenizer.decode(token_ids)

    def _get_token_bytes(self, token_id: int) -> bytes:
        """获取单个 token 的 UTF-8 字节表示 (用于流式解码)。

        支持 BPE 和 Char 两种 tokenizer。
        """
        # BPE tokenizer: 有 _bytes_for_token 方法
        if hasattr(self.tokenizer, "_bytes_for_token"):
            return self.tokenizer._bytes_for_token(token_id)
        # BPE tokenizer: 有 vocab 和 byte_offset 属性
        if hasattr(self.tokenizer, "vocab") and hasattr(self.tokenizer, "byte_offset"):
            byte_offset = self.tokenizer.byte_offset
            if byte_offset <= token_id < byte_offset + 256:
                return bytes([token_id - byte_offset])
            return self.tokenizer.vocab.get(token_id, b"")
        # Char tokenizer 或其他: decode 单个 token
        decoded = self.tokenizer.decode([token_id])
        return decoded.encode("utf-8") if decoded else b""

    # ========================================================
    # 公开 API
    # ========================================================

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """单次生成文本。

        Args:
            prompt: 输入提示文本
            config: 生成配置 (None=默认 GenerationConfig)
        Returns:
            生成的文本字符串 (不含 prompt)
        """
        config = self._resolve_config(config)
        strategy = config.resolve_strategy()

        if config.seed is not None:
            torch.manual_seed(config.seed)

        max_seq_len = self._resolve_max_seq_len(config)
        input_ids = self._encode(prompt, max_seq_len)

        with self._no_grad():
            kv_cache = KVCache(max_seq_len)
            if strategy == "beam" and config.beam_size > 1:
                output_ids = self._beam_search(input_ids, config, kv_cache)
            elif strategy == "sample":
                output_ids = self._sample(input_ids, config, kv_cache)
            else:
                output_ids = self._greedy_decode(input_ids, config, kv_cache)

        prompt_len = input_ids.shape[1]
        generated_ids = _tolist(output_ids[0])[prompt_len:]
        return self._decode_tokens(generated_ids)

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """流式生成文本, 逐 token yield。

        使用字节级增量解码, 正确处理多字节 UTF-8 字符:
        累积 token 字节, 尝试 UTF-8 解码, 仅 yield 已完成的字符。

        注意: 流式生成不支持束搜索 (strategy="beam" 时回退为非流式)。

        Args:
            prompt: 输入提示文本
            config: 生成配置 (None=默认)
        Yields:
            文本片段 (完整的 UTF-8 字符序列)
        """
        config = self._resolve_config(config)
        strategy = config.resolve_strategy()

        # 流式不支持束搜索, 回退到非流式
        if strategy == "beam":
            yield self.generate(prompt, config)
            return

        if config.seed is not None:
            torch.manual_seed(config.seed)

        max_seq_len = self._resolve_max_seq_len(config)
        n_loops = self._resolve_n_loops(config)
        eos_id = self._resolve_eos_id(config)
        input_ids = self._encode(prompt, max_seq_len)

        kv_cache = KVCache(max_seq_len)
        working_ids = input_ids.clone()
        cur_pos = 0
        byte_buffer = bytearray()

        with self._no_grad():
            for _ in range(config.max_tokens):
                if kv_cache.is_empty():
                    cur_tokens = working_ids
                    start_pos = 0
                else:
                    cur_tokens = working_ids[:, -1:]
                    start_pos = cur_pos

                # 滑动窗口
                if start_pos >= max_seq_len:
                    keep = max_seq_len - 1
                    working_ids = working_ids[:, -keep:]
                    kv_cache.reset()
                    cur_tokens = working_ids
                    start_pos = 0

                logits = self.model(
                    cur_tokens,
                    n_loops=n_loops,
                    kv_cache=kv_cache.data,
                    start_pos=start_pos,
                )
                cur_pos = start_pos + cur_tokens.shape[1]
                next_logits = logits[:, -1, :]

                # 重复惩罚
                if config.repetition_penalty != 1.0:
                    next_logits = self._apply_repetition_penalty(
                        next_logits, working_ids, config.repetition_penalty
                    )

                # 采样或贪心
                if strategy == "sample":
                    next_token = self._sample_token(next_logits, config)
                else:
                    next_token = next_logits.argmax(dim=-1).unsqueeze(-1)

                token_id = _to_item(next_token[0])

                if token_id == eos_id:
                    break

                working_ids = torch.cat([working_ids, next_token], dim=1)

                # 增量解码 (字节级, 正确处理多字节 UTF-8)
                token_bytes = self._get_token_bytes(token_id)
                byte_buffer.extend(token_bytes)
                try:
                    text = byte_buffer.decode("utf-8")
                    if text:
                        yield text
                    byte_buffer = bytearray()
                except UnicodeDecodeError as e:
                    valid_len = e.start
                    if valid_len > 0:
                        text = byte_buffer[:valid_len].decode("utf-8")
                        if text:
                            yield text
                        byte_buffer = byte_buffer[valid_len:]

        # 输出剩余未完成字节
        if byte_buffer:
            yield byte_buffer.decode("utf-8", errors="replace")

    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """批量生成文本。

        将 prompt 按 ``batch_size`` 分组, 组内使用左填充并行生成。
        支持 KV Cache 跨步复用和每个序列独立的 EOS 终止。

        注意: 左填充的 pad token 会参与注意力计算 (模型不支持
        attention mask), 对结果有轻微影响。按长度排序可最小化填充。

        Args:
            prompts: 输入提示文本列表
            config:  生成配置 (batch_size 控制并行度)
        Returns:
            生成文本列表 (顺序与 prompts 一致)
        """
        config = self._resolve_config(config)
        batch_size = max(1, config.batch_size)

        if batch_size <= 1 or len(prompts) <= 1:
            return [self.generate(p, config) for p in prompts]

        results: List[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if len(batch) == 1:
                results.append(self.generate(batch[0], config))
            else:
                results.extend(self._batch_decode(batch, config))
        return results

    # ========================================================
    # 解码策略: 贪心
    # ========================================================

    def _greedy_decode(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """贪心解码 (KV Cache 优化)。

        首步处理完整 prompt, 后续步只处理最后一个 token。
        每步选取概率最高的 token (argmax)。

        Args:
            input_ids: [1, T] prompt token IDs
            config:    生成配置
            kv_cache:  KV 缓存管理器
        Returns:
            [1, T + generated] 包含 prompt 和生成 token 的张量
        """
        max_seq_len = self._resolve_max_seq_len(config)
        n_loops = self._resolve_n_loops(config)
        eos_id = self._resolve_eos_id(config)

        working_ids = input_ids.clone()
        cur_pos = 0

        for _ in range(config.max_tokens):
            if kv_cache.is_empty():
                cur_tokens = working_ids
                start_pos = 0
            else:
                cur_tokens = working_ids[:, -1:]
                start_pos = cur_pos

            # 滑动窗口: 超过 max_seq_len 时重置缓存 + 重新编码
            if start_pos >= max_seq_len:
                keep = max_seq_len - 1
                working_ids = working_ids[:, -keep:]
                kv_cache.reset()
                cur_tokens = working_ids
                start_pos = 0

            logits = self.model(
                cur_tokens,
                n_loops=n_loops,
                kv_cache=kv_cache.data,
                start_pos=start_pos,
            )
            cur_pos = start_pos + cur_tokens.shape[1]
            next_logits = logits[:, -1, :]

            # 重复惩罚
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, working_ids, config.repetition_penalty
                )

            # 贪心: argmax
            next_token = next_logits.argmax(dim=-1).unsqueeze(-1)

            working_ids = torch.cat([working_ids, next_token], dim=1)

            if _to_item(next_token[0]) == eos_id:
                break

        return working_ids

    # ========================================================
    # 解码策略: 采样
    # ========================================================

    def _sample(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """采样解码 (温度 / Top-K / Top-P)。

        依次应用: 重复惩罚 → 温度缩放 → Top-K 过滤 → Top-P 过滤 → 多项采样。
        temperature=0 时退化为贪心。

        Args:
            input_ids: [1, T] prompt token IDs
            config:    生成配置
            kv_cache:  KV 缓存管理器
        Returns:
            [1, T + generated] 包含 prompt 和生成 token 的张量
        """
        max_seq_len = self._resolve_max_seq_len(config)
        n_loops = self._resolve_n_loops(config)
        eos_id = self._resolve_eos_id(config)

        working_ids = input_ids.clone()
        cur_pos = 0

        for _ in range(config.max_tokens):
            if kv_cache.is_empty():
                cur_tokens = working_ids
                start_pos = 0
            else:
                cur_tokens = working_ids[:, -1:]
                start_pos = cur_pos

            # 滑动窗口
            if start_pos >= max_seq_len:
                keep = max_seq_len - 1
                working_ids = working_ids[:, -keep:]
                kv_cache.reset()
                cur_tokens = working_ids
                start_pos = 0

            logits = self.model(
                cur_tokens,
                n_loops=n_loops,
                kv_cache=kv_cache.data,
                start_pos=start_pos,
            )
            cur_pos = start_pos + cur_tokens.shape[1]
            next_logits = logits[:, -1, :]

            # 重复惩罚
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, working_ids, config.repetition_penalty
                )

            # 采样
            next_token = self._sample_token(next_logits, config)

            working_ids = torch.cat([working_ids, next_token], dim=1)

            if _to_item(next_token[0]) == eos_id:
                break

        return working_ids

    # ========================================================
    # 解码策略: 束搜索
    # ========================================================

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """束搜索解码。

        维护 ``beam_size`` 个候选序列, 每步从 ``beam_size * vocab_size``
        候选中选择累积对数概率最高的 ``beam_size`` 个。支持 KV Cache
        跨步复用和 beam 重排。

        长度惩罚: ``score = log_prob_sum / (length ** length_penalty)``
        - length_penalty=1.0: 按长度归一化 (避免偏好短序列)
        - length_penalty=0.0: 不归一化 (偏好长序列)

        Args:
            input_ids: [1, T] prompt token IDs
            config:    生成配置
            kv_cache:  KV 缓存管理器 (用于 batch 维度 = beam_size)
        Returns:
            [1, T + generated] 最佳 beam 的 token 序列
        """
        beam_size = min(config.beam_size, self.cfg.vocab_size)
        max_seq_len = self._resolve_max_seq_len(config)
        n_loops = self._resolve_n_loops(config)
        eos_id = self._resolve_eos_id(config)
        vocab_size = self.cfg.vocab_size
        prompt_len = input_ids.shape[1]

        # 扩展为 beam_size 个相同序列
        beam_input = torch.cat([input_ids] * beam_size, dim=0)  # [beam_size, T]

        # 初始前向: 处理完整 prompt
        logits = self.model(
            beam_input, n_loops=n_loops, kv_cache=kv_cache.data, start_pos=0
        )
        cur_pos = prompt_len
        next_logits = logits[:, -1, :]  # [beam_size, vocab_size]

        # 重复惩罚
        if config.repetition_penalty != 1.0:
            next_logits = self._apply_repetition_penalty(
                next_logits, beam_input, config.repetition_penalty
            )

        log_probs = torch.log_softmax(next_logits, dim=-1)

        # 初始 beam scores: 只有 beam 0 活跃
        beam_scores = torch.full((beam_size,), float("-inf"))
        beam_scores[0] = 0.0

        # 从 beam 0 的 log_probs 中选择 top beam_size 个 token
        top_scores, top_indices = torch.topk(log_probs[0], beam_size, dim=-1)
        beam_scores = top_scores
        beam_tokens = top_indices.unsqueeze(-1)  # [beam_size, 1]
        beam_sequences = torch.cat(
            [beam_input, beam_tokens], dim=1
        )  # [beam_size, T+1]

        # 已完成的 beam 列表
        finished_beams: List[Dict[str, Any]] = []
        for b in range(beam_size):
            if _to_item(top_indices[b]) == eos_id:
                gen_len = beam_sequences.shape[1] - prompt_len
                penalty = gen_len ** config.length_penalty if gen_len > 0 else 1.0
                finished_beams.append({
                    "score": _to_item(beam_scores[b]) / max(penalty, 1e-6),
                    "tokens": beam_sequences[b].clone(),
                })

        # 束搜索主循环
        for step in range(1, config.max_tokens):
            if len(finished_beams) >= beam_size and config.early_stopping:
                break

            # 滑动窗口
            if cur_pos >= max_seq_len:
                keep = max_seq_len - 1
                beam_sequences = beam_sequences[:, -keep:]
                kv_cache.reset()
                logits = self.model(
                    beam_sequences,
                    n_loops=n_loops,
                    kv_cache=kv_cache.data,
                    start_pos=0,
                )
                cur_pos = beam_sequences.shape[1]
                next_logits = logits[:, -1, :]
            else:
                cur_tokens = beam_sequences[:, -1:]  # [beam_size, 1]
                logits = self.model(
                    cur_tokens,
                    n_loops=n_loops,
                    kv_cache=kv_cache.data,
                    start_pos=cur_pos,
                )
                cur_pos += 1
                next_logits = logits[:, -1, :]  # [beam_size, vocab_size]

            # 重复惩罚
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, beam_sequences, config.repetition_penalty
                )

            log_probs = torch.log_softmax(next_logits, dim=-1)

            # 累积分数: [beam_size, 1] + [beam_size, vocab_size]
            cumulative_scores = beam_scores.unsqueeze(-1) + log_probs

            # 展平并选择 top beam_size
            flat_scores = cumulative_scores.view(-1)
            top_scores, top_indices = torch.topk(flat_scores, beam_size)

            # 解码 beam_idx 和 token_idx
            beam_indices, token_indices = self._tensor_divmod(
                top_indices, vocab_size
            )

            # 重排 KV cache (按 beam 来源重排 batch 维度)
            kv_cache.reorder(beam_indices)

            # 更新 beam 序列和分数
            beam_sequences = torch.cat(
                [beam_sequences[beam_indices], token_indices.unsqueeze(-1)],
                dim=1,
            )
            beam_scores = top_scores

            # 检查完成的 beam
            for b in range(beam_size):
                if _to_item(token_indices[b]) == eos_id:
                    gen_len = beam_sequences.shape[1] - prompt_len
                    penalty = gen_len ** config.length_penalty if gen_len > 0 else 1.0
                    finished_beams.append({
                        "score": _to_item(beam_scores[b]) / max(penalty, 1e-6),
                        "tokens": beam_sequences[b].clone(),
                    })

        # 选择最佳 beam
        if finished_beams:
            best = max(finished_beams, key=lambda x: x["score"])
            return best["tokens"].unsqueeze(0)

        # 无 beam 完成, 返回当前分数最高的 beam
        best_idx = _to_item(beam_scores.argmax())
        return beam_sequences[best_idx].unsqueeze(0)

    # ========================================================
    # 批量解码
    # ========================================================

    def _batch_decode(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[str]:
        """批量解码 (左填充 + KV Cache)。

        将不同长度的 prompt 左填充到相同长度后批量处理。
        每个序列可独立终止 (EOS), 已终止的序列用 pad_token 继续
        参与前向以保持 batch 维度一致。

        Args:
            prompts: 输入提示文本列表
            config:  生成配置
        Returns:
            生成文本列表
        """
        max_seq_len = self._resolve_max_seq_len(config)
        n_loops = self._resolve_n_loops(config)
        eos_id = self._resolve_eos_id(config)
        pad_id = self._resolve_pad_id(config)
        strategy = config.resolve_strategy()

        # 编码并左填充
        all_tokens: List[List[int]] = []
        for p in prompts:
            tokens = self.tokenizer.encode(p)
            if len(tokens) == 0:
                tokens = [eos_id]
            if len(tokens) > max_seq_len:
                tokens = tokens[-max_seq_len:]
            all_tokens.append(tokens)

        max_len = max(len(t) for t in all_tokens)
        padded: List[List[int]] = []
        for tokens in all_tokens:
            pad_len = max_len - len(tokens)
            padded.append([pad_id] * pad_len + tokens)

        input_ids = torch.tensor(padded, dtype=torch.int64)

        B = input_ids.shape[0]
        kv_cache = KVCache(max_seq_len)
        working_ids = input_ids.clone()
        cur_pos = 0
        finished = [False] * B
        output_tokens: List[List[int]] = [[] for _ in range(B)]

        with self._no_grad():
            for _ in range(config.max_tokens):
                if kv_cache.is_empty():
                    cur_tokens = working_ids
                    start_pos = 0
                else:
                    cur_tokens = working_ids[:, -1:]
                    start_pos = cur_pos

                # 滑动窗口
                if start_pos >= max_seq_len:
                    keep = max_seq_len - 1
                    working_ids = working_ids[:, -keep:]
                    kv_cache.reset()
                    cur_tokens = working_ids
                    start_pos = 0

                logits = self.model(
                    cur_tokens,
                    n_loops=n_loops,
                    kv_cache=kv_cache.data,
                    start_pos=start_pos,
                )
                cur_pos = start_pos + cur_tokens.shape[1]
                next_logits = logits[:, -1, :]  # [B, vocab_size]

                # 重复惩罚
                if config.repetition_penalty != 1.0:
                    next_logits = self._apply_repetition_penalty(
                        next_logits, working_ids, config.repetition_penalty
                    )

                # 采样或贪心
                if strategy == "sample":
                    next_tokens = self._sample_token(next_logits, config)
                else:
                    next_tokens = next_logits.argmax(dim=-1).unsqueeze(-1)

                # 处理已完成 / 未完成的序列
                for b in range(B):
                    if finished[b]:
                        next_tokens[b, 0] = pad_id
                    else:
                        tid = _to_item(next_tokens[b, 0])
                        if tid == eos_id:
                            finished[b] = True
                        else:
                            output_tokens[b].append(tid)

                working_ids = torch.cat([working_ids, next_tokens], dim=1)

                if all(finished):
                    break

        return [self._decode_tokens(tokens) for tokens in output_tokens]

    # ========================================================
    # 采样辅助
    # ========================================================

    def _sample_token(
        self, logits: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """从 logits 采样单个 token。

        依次应用: 温度缩放 → Top-K 过滤 → Top-P 过滤 → 多项采样。
        temperature=0 时退化为贪心 (argmax)。

        Args:
            logits: [B, vocab_size] logits
            config: 生成配置
        Returns:
            [B, 1] 采样的 token ID
        """
        # temperature=0 退化为贪心
        if config.temperature <= 0:
            return logits.argmax(dim=-1).unsqueeze(-1)

        # 温度缩放
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-K 过滤
        if config.top_k > 0:
            logits = self._top_k_filter(logits, config.top_k)

        # Top-P 过滤
        if config.top_p < 1.0:
            logits = self._top_p_filter(logits, config.top_p)

        # 转概率并采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-K 过滤: 仅保留概率最高的 k 个 token, 其余设为 -inf。

        Args:
            logits: [B, vocab_size]
            k:      保留的 token 数
        Returns:
            [B, vocab_size] 过滤后的 logits
        """
        k = min(k, logits.shape[-1])
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1:]  # [B, 1] — top-k 中的最小值
        return logits.masked_fill(logits < min_values, float("-inf"))

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-P (nucleus) 过滤: 保留累积概率达到 top_p 的最小 token 集合。

        算法:
            1. 按 logits 降序排序
            2. 计算累积概率
            3. 移除累积概率超过 top_p 的 token (右移一位以保留首个 token)
            4. 将排序后的掩码散布回原始顺序

        Args:
            logits: [B, vocab_size]
            top_p:  累积概率阈值
        Returns:
            [B, vocab_size] 过滤后的 logits
        """
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 右移一位: 始终保留概率最高的 token (避免全部移除)
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # 将排序后的掩码散布回原始顺序
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚。

        对已出现过的 token, 根据 logit 正负进行惩罚:
        - logit > 0: ``logit / penalty`` (降低概率)
        - logit < 0: ``logit * penalty`` (进一步降低)

        这保证惩罚后 logit 始终变小 (概率降低), 无论正负。

        Args:
            logits:    [B, vocab_size]
            input_ids: [B, T] 已生成的 token IDs
            penalty:   惩罚系数 (>1 惩罚重复, <1 鼓励重复, 1.0 无效果)
        Returns:
            [B, vocab_size] 惩罚后的 logits
        """
        gathered = logits.gather(1, input_ids)  # [B, T]
        gathered = torch.where(
            gathered < 0,
            gathered * penalty,
            gathered / penalty,
        )
        return logits.scatter(1, input_ids, gathered)

    # ========================================================
    # 张量运算辅助 (双后端兼容)
    # ========================================================

    def _tensor_divmod(self, t: torch.Tensor, divisor: int):
        """张量整除和取模 (双后端兼容)。

        用于束搜索中将 flat index 分解为 (beam_idx, token_idx):
        - beam_idx  = flat_index // vocab_size
        - token_idx = flat_index %  vocab_size

        PyTorch 后端原生支持 ``//`` 和 ``%``。
        NumPy 后端 Tensor 类未实现 ``__floordiv__`` / ``__mod__``,
        需通过 ``.data`` 操作 numpy 数组。

        Args:
            t:        1-D 整数张量
            divisor:  除数 (vocab_size)
        Returns:
            (beam_indices, token_indices) 两个张量
        """
        if self.is_torch:
            return t // divisor, t % divisor
        else:
            import numpy as np
            data = t.data
            return torch.tensor(data // divisor), torch.tensor(data % divisor)


# ============================================================
# 便捷工厂函数
# ============================================================

def create_engine_from_checkpoint(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    config: Optional[SimplCoreMixConfig] = None,
    config_name: str = "1b",
    device: Optional[str] = None,
) -> InferenceEngine:
    """从 checkpoint 文件创建推理引擎。

    Args:
        model_path:    模型权重路径
        tokenizer_path: tokenizer 路径 (None=使用字符 tokenizer)
        config:        模型配置 (None=从 checkpoint 恢复或按 config_name 选择)
        config_name:   预设配置名 ("1b"|"7b"|"33b"|"tiny"|"150m"|"3b")
        device:        设备 (None=自动)
    Returns:
        InferenceEngine 实例
    """
    from pathlib import Path
    from .config import mix_1b, mix_7b, mix_33b, mix_tiny_text
    from .tokenizer import BPETokenizer, CharTokenizer

    config_map = {
        "1b": mix_1b, "7b": mix_7b, "33b": mix_33b,
        "tiny": mix_tiny_text, "150m": None, "3b": None,
    }

    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # 恢复配置
    if config is not None:
        cfg = config
    elif isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        if isinstance(cfg_dict, SimplCoreMixConfig):
            cfg = cfg_dict
        elif isinstance(cfg_dict, dict):
            cfg = SimplCoreMixConfig(**cfg_dict)
        else:
            cfg = mix_1b()
    else:
        factory = config_map.get(config_name, mix_1b)
        cfg = factory() if factory else mix_1b()

    # 创建模型
    model = SimplCoreMix(cfg)

    # 加载权重
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    # 加载 tokenizer
    if tokenizer_path and Path(tokenizer_path).exists():
        try:
            tokenizer = BPETokenizer(cfg.vocab_size)
            tokenizer.load(tokenizer_path)
        except Exception:
            tokenizer = CharTokenizer()
            tokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.char_to_id = {"<pad>": 0, "<eos>": 1}
        tokenizer.id_to_char = {0: "<pad>", 1: "<eos>"}
        for i in range(256):
            ch = chr(i)
            tokenizer.char_to_id[ch] = i + 2
            tokenizer.id_to_char[i + 2] = ch

    return InferenceEngine(model, tokenizer, cfg)
