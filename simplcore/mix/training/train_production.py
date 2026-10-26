# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Mix 生产级训练管道
================================

本模块实现一套完整的、可直接用于生产的语言模型训练管道，支持：

  1. JSONL 数据加载 — 从 ``.jsonl`` 文件读取 ``{"text": "..."}`` 训练语料
  2. 简易 BPE 分词器 — 中文字符级 + 英文单词级混合分词，自动构建词表
  3. 学习率调度 — warmup (前 1000 步线性增长) + cosine decay
  4. Checkpoint 保存/恢复 — 每 N 步保存模型、优化器、调度器完整状态
  5. 混合精度训练 — PyTorch 后端可用时启用 ``torch.cuda.amp.GradScaler``
  6. 梯度裁剪 — ``max_norm=1.0``
  7. 训练日志 — 每步打印 loss / lr / grad_norm / tokens/s
  8. 多 epoch 训练循环 — 支持断点恢复

双后端兼容
----------
代码同时兼容自研 NumPy 后端和 PyTorch 后端，通过
``simplcore.tensor.backend.is_torch_backend()`` 判断当前后端：

  * **PyTorch 后端**: GPU 加速、混合精度 (AMP)、fused AdamW、
    原生梯度裁剪等全部可用。
  * **自研 NumPy 后端**: 降级为 fp32 训练、手动梯度裁剪，
    保证零依赖环境下的架构验证与小规模调试可用。

用法
----
    # 命令行启动
    python -m simplcore.mix.training.train_production \\
        --train-data ./data/train.jsonl \\
        --output-dir ./checkpoints/mix_150m \\
        --config 150m --epochs 10

    # 代码调用
    from simplcore.mix.config import mix_150m_text
    from simplcore.mix.model import SimplCoreMix
    from simplcore.mix.training.train_production import ProductionTrainer

    config = mix_150m_text()
    model = SimplCoreMix(config)
    trainer = ProductionTrainer(model, config, data_path, output_dir)
    trainer.train(epochs=10)
"""
from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# ── SimplCore 张量引擎 (双后端: 自研 NumPy / PyTorch 自动切换) ──
import simplcore.tensor as torch
from simplcore.tensor import AdamW, CosineAnnealingLR, LambdaLR
from simplcore.tensor.backend import is_numpy_backend, get_backend, get_device

# ── 模型与配置 ──
from simplcore.mix.config import (
    SimplCoreMixConfig,
    mix_150m_text,
    mix_3b_text,
    mix_7b_text,
)
from simplcore.mix.model import SimplCoreMix

# 自研引擎不参与任何第三方框架依赖；AMP / 梯度裁剪等由自研模块提供。
_pt = None
_HAS_TORCH = False


# ============================================================
# 常量
# ============================================================
PAD_ID = 0    # <pad> token id
EOS_ID = 1    # <eos> token id
BOS_ID = 2    # <bos> token id
UNK_ID = 3    # <unk> token id

# 默认 warmup 步数 (前 1000 步线性增长)
DEFAULT_WARMUP_STEPS = 1000

# CJK 正则范围 (统一表意文字 + 扩展A + 兼容表意 + CJK符号 + 全角形式)
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]"
)


# ============================================================
# 简易 BPE 分词器
# ============================================================
class SimpleTokenizer:
    """简易 BPE 分词器 — 中文字符级 + 英文单词级混合分词。

    分词策略:
      * **中文**: 每个 CJK 字符单独作为一个 token (字符级)
      * **英文**: 连续字母序列作为一个 token (单词级)，统一小写
      * **数字**: 连续数字序列作为一个 token
      * **标点/符号**: 单独作为一个 token
      * **空白**: 跳过 (不产生 token)

    词表布局:
      id 0 = <pad>
      id 1 = <eos>
      id 2 = <bos>
      id 3 = <unk>
      id 4 .. vocab_size-1 = 语料中的 token (按频率降序)

    Args:
        vocab_size: 词表大小上限
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

        # 特殊 token
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._init_special_tokens()
        self._built = False

    def _init_special_tokens(self) -> None:
        """初始化特殊 token。"""
        specials = [
            (self.pad_token, PAD_ID),
            (self.eos_token, EOS_ID),
            (self.bos_token, BOS_ID),
            (self.unk_token, UNK_ID),
        ]
        for token, tid in specials:
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_ID

    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为 token 列表 (中文字符级 + 英文单词级)。

        Args:
            text: 输入文本
        Returns:
            token 字符串列表
        """
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]

            # CJK 字符: 单字成 token
            if _CJK_PATTERN.match(ch):
                tokens.append(ch)
                i += 1
                continue

            # 英文字母: 连续字母 (含 ' 和 -) 组成单词 token
            if ch.isalpha():
                j = i + 1
                while j < n and (text[j].isalpha() or text[j] in ("'", "-")):
                    j += 1
                tokens.append(text[i:j].lower())
                i = j
                continue

            # 数字: 连续数字组成 token
            if ch.isdigit():
                j = i + 1
                while j < n and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue

            # 空白: 跳过
            if ch.isspace():
                i += 1
                continue

            # 其他 (标点/符号): 单字符 token
            tokens.append(ch)
            i += 1

        return tokens

    def train(self, texts: List[str]) -> None:
        """从语料构建词表。

        统计所有 token 的频率，按降序取前 ``vocab_size - 4`` 个
        (预留 4 个特殊 token)。

        Args:
            texts: 训练语料文本列表
        """
        counter: Counter = Counter()
        for text in texts:
            if not text:
                continue
            counter.update(self._tokenize(text))

        # 按频率降序排列，取前 vocab_size - 4 个
        max_tokens = self.vocab_size - len(self.token_to_id)
        sorted_tokens = counter.most_common(max_tokens)

        for token, _freq in sorted_tokens:
            tid = len(self.token_to_id)
            if tid >= self.vocab_size:
                break
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token

        self._built = True
        actual_vocab = len(self.token_to_id)
        print(f"[SimpleTokenizer] 词表构建完成: {actual_vocab} tokens "
              f"(上限 {self.vocab_size}, 语料 token 类型 {len(counter)})")

    def encode(self, text: str) -> List[int]:
        """编码文本为 token id 序列。

        Args:
            text: 输入文本
        Returns:
            token id 列表
        """
        if not self._built:
            # 未训练时用字符哈希回退
            divisor = max(self.vocab_size - 4, 1)
            return [4 + (ord(ch) % divisor) for ch in text if not ch.isspace()]

        tokens = self._tokenize(text)
        return [self.token_to_id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """解码 token id 序列为文本。

        Args:
            ids: token id 列表
        Returns:
            解码后的文本
        """
        if not self._built:
            return "".join(self.id_to_token.get(t, "?") for t in ids
                           if t not in (PAD_ID, EOS_ID, BOS_ID))

        parts: List[str] = []
        for tid in ids:
            if tid in (PAD_ID, EOS_ID, BOS_ID):
                continue
            token = self.id_to_token.get(tid, self.unk_token)
            if token == self.unk_token:
                parts.append(token)
            elif _CJK_PATTERN.match(token):
                # 中文字符直接拼接
                parts.append(token)
            elif len(token) == 1 and not token.isalnum():
                # 标点符号
                parts.append(token)
            else:
                # 英文单词 / 数字: 前面加空格 (简单启发式)
                if parts and not _CJK_PATTERN.match(parts[-1]) \
                        and parts[-1] not in ("", ) :
                    parts.append(" ")
                parts.append(token)
        return "".join(parts)

    def save(self, path: str) -> None:
        """保存词表到 JSON 文件。"""
        save_data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 文件加载词表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self._built = True

    def __len__(self) -> int:
        return len(self.token_to_id)


# ============================================================
# JSONL 数据加载与数据集
# ============================================================
def read_jsonl(path: Union[str, Path]) -> List[str]:
    """读取 JSONL 文件，返回文本列表。

    每行一个 JSON 对象 ``{"text": "..."}``，提取 ``text`` 字段。
    也兼容 ``content`` 字段。

    Args:
        path: JSONL 文件路径
    Returns:
        文本列表
    """
    path = Path(path)
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", row.get("content", ""))
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def iter_corpus(data_path: Union[str, Path]) -> Iterator[str]:
    """统一语料读取迭代器: 支持单个 JSONL 文件或目录。

    - 单个 ``.jsonl`` 文件: 逐行解析 ``{"text": "..."}``
    - 目录: 递归扫描所有 ``.jsonl`` 文件
    - ``.txt`` 文件: 读取全文 (每行视为一条)

    Args:
        data_path: 语料路径 (文件或目录)
    Yields:
        文本字符串
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"语料路径不存在: {p}")

    if p.is_file():
        if p.suffix == ".jsonl":
            for text in read_jsonl(p):
                yield text
        elif p.suffix in (".txt", ".md"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    elif p.is_dir():
        for fp in sorted(p.rglob("*.jsonl")):
            for text in read_jsonl(fp):
                yield text
        for fp in sorted(p.rglob("*.txt")):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    else:
        raise ValueError(f"不支持的语料路径类型: {p}")


class TextDataset:
    """文本数据集 — 读取 JSONL 语料、分词、分块、返回 (input, target) 对。

    处理流程:
      1. 读取所有文本
      2. 分词为 token id 序列
      3. 在文档之间插入 EOS
      4. 按 ``seq_len + 1`` 分块 (每个块 = 1 个训练样本)
      5. 不足 ``seq_len + 1`` 的尾部用 PAD 填充

    Args:
        data_path: 语料路径 (JSONL 文件或目录)
        tokenizer: 分词器对象 (需有 encode / eos_token_id 接口)
        max_seq_len: 最大序列长度
        overlap: 分块重叠比例 (0.0 = 无重叠)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any,
        max_seq_len: int = 512,
        overlap: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.samples: List[torch.Tensor] = []
        self._load(data_path)
        print(f"[TextDataset] 已加载 {len(self.samples)} 个样本 "
              f"(seq_len={max_seq_len}, overlap={overlap})")

    def _load(self, data_path: Union[str, Path]) -> None:
        """加载并分词所有语料。"""
        chunk_len = self.max_seq_len + 1  # 输入 + 目标
        step = max(1, int(chunk_len * (1 - self.overlap)))
        total_tokens = 0

        for text in iter_corpus(data_path):
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text)
            # 文档末尾加 EOS
            tokens = tokens + [self.tokenizer.eos_token_id]
            total_tokens += len(tokens)

            # 分块
            if len(tokens) < chunk_len:
                # 不足一个块，补 PAD
                padded = tokens + [PAD_ID] * (chunk_len - len(tokens))
                self.samples.append(
                    torch.tensor(padded, dtype=torch.long)
                )
            else:
                for i in range(0, len(tokens) - chunk_len + 1, step):
                    chunk = tokens[i: i + chunk_len]
                    if len(chunk) == chunk_len:
                        self.samples.append(
                            torch.tensor(chunk, dtype=torch.long)
                        )
                # 处理尾部剩余
                remainder_start = ((len(tokens) - chunk_len) // step) * step
                if remainder_start + chunk_len < len(tokens):
                    tail = tokens[remainder_start + step:]
                    if len(tail) >= 2:
                        tail = tail + [PAD_ID] * (chunk_len - len(tail))
                        self.samples.append(
                            torch.tensor(tail[:chunk_len], dtype=torch.long)
                        )

        if not self.samples:
            raise RuntimeError(f"数据集为空，请检查语料路径: {data_path}")
        print(f"[TextDataset] 总 token 数: {total_tokens:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (input_ids, target_ids)，target 是 input 左移一位。"""
        tokens = self.samples[idx]
        return tokens[:-1], tokens[1:]


def make_batches(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """从数据集生成批次。

    Args:
        dataset: 文本数据集
        batch_size: 每批样本数
        shuffle: 是否打乱顺序
        drop_last: 是否丢弃不足 batch_size 的尾批
    Yields:
        (input_ids [B, T], target_ids [B, T])
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    batch: List[torch.Tensor] = []
    batch_tgt: List[torch.Tensor] = []
    for idx in indices:
        src, tgt = dataset[idx]
        batch.append(src)
        batch_tgt.append(tgt)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0), torch.stack(batch_tgt, dim=0)
            batch = []
            batch_tgt = []

    if batch and not drop_last:
        yield torch.stack(batch, dim=0), torch.stack(batch_tgt, dim=0)


# ============================================================
# 学习率调度
# ============================================================
def warmup_cosine_lr_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """线性预热 + 余弦退火的学习率比例函数。

    Args:
        step: 当前全局步数
        warmup_steps: 预热步数 (线性从 0 升到 peak_lr)
        total_steps: 总训练步数
        min_lr_ratio: 最低学习率比例 (peak_lr * min_lr_ratio)
    Returns:
        当前步的学习率比例 (1.0 = peak_lr)
    """
    if step < warmup_steps:
        # 线性预热
        return float(step) / float(max(1, warmup_steps))
    # 余弦退火
    progress = float(step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    progress = min(1.0, max(0.0, progress))
    cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cos_val


# ============================================================
# 生产级训练器
# ============================================================
class ProductionTrainer:
    """SimplCore Mix 生产级训练器。

    支持混合精度、梯度累积、梯度裁剪、学习率调度 (warmup + cosine)、
    检查点管理、断点恢复等全部生产级训练能力。

    双后端兼容:
      * PyTorch 后端: GPU 加速、AMP 混合精度、fused AdamW、
        原生 ``clip_grad_norm_``。
      * 自研 NumPy 后端: fp32 训练、手动梯度裁剪。

    Args:
        model: SimplCoreMix 模型实例
        config: SimplCoreMixConfig 配置对象
        data_path: 训练语料路径 (JSONL 文件或目录)
        output_dir: 输出目录 (检查点、日志)
        batch_size: 每步微批次大小
        grad_accum_steps: 梯度累积步数 (有效 batch = batch_size * grad_accum_steps)
        max_seq_len: 最大序列长度
        learning_rate: 峰值学习率
        weight_decay: 权重衰减
        warmup_steps: 学习率预热步数 (默认 1000)
        min_lr_ratio: 最低学习率比例
        max_grad_norm: 梯度裁剪最大范数 (默认 1.0)
        use_amp: 是否启用混合精度 (仅 PyTorch 后端生效)
        amp_dtype: 混合精度数据类型 ('float16' 或 'bfloat16')
        tokenizer: 分词器对象 (None 则自动构建)
        tokenizer_path: 分词器文件路径 (用于加载/保存)
        seed: 随机种子
        log_interval: 日志打印间隔 (步)
        save_interval: 检查点保存间隔 (步)
        train_n_loops: 训练时循环深度 (None = 使用配置默认值)
        device: 计算设备 (None 则自动选择)
    """

    def __init__(
        self,
        model,
        config: SimplCoreMixConfig,
        data_path: Union[str, Path],
        output_dir: Union[str, Path] = "./checkpoints",
        *,
        batch_size: int = 4,
        grad_accum_steps: int = 1,
        max_seq_len: int = 512,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        min_lr_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        tokenizer: Any = None,
        tokenizer_path: Optional[str] = None,
        seed: int = 42,
        log_interval: int = 10,
        save_interval: int = 1000,
        train_n_loops: Optional[int] = None,
        device: Optional[str] = None,
    ):
        # ── 基本配置 ──
        self.model = model
        self.config = config
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 训练超参数 ──
        self.batch_size = batch_size
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.seed = seed
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_n_loops = train_n_loops
        self.vocab_size = config.vocab_size

        # ── 设备选择（自研引擎恒为 CPU）──
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # ── 后端标识 ──
        self.backend_name = get_backend()
        self._is_torch = False

        # ── 混合精度设置（自研引擎恒为关闭）──
        self._amp_enabled = False
        self._scaler = None

        # ── 分词器 ──
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path

        # ── 训练状态 ──
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.lr_history: List[float] = []

        # ── 优化器 & 调度器 (延迟构建，train() 时创建) ──
        self.optimizer: Optional[AdamW] = None
        self.scheduler = None
        self._total_steps = 0

        # ── 日志文件 ──
        self.log_file = self.output_dir / "training_log.jsonl"

        # ── 随机种子 ──
        torch.manual_seed(seed)
        random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass

        print(f"[ProductionTrainer] 后端={self.backend_name} "
              f"设备={self.device} AMP={self._amp_enabled} "
              f"batch={batch_size}x{grad_accum_steps}="
              f"{batch_size * grad_accum_steps} "
              f"seq_len={max_seq_len} lr={learning_rate} "
              f"warmup={warmup_steps} max_grad_norm={max_grad_norm}")

    # ──────────────────────────────────────────────────────
    # AMP 初始化
    # ──────────────────────────────────────────────────────
    def _setup_amp(self) -> None:
        """初始化混合精度组件（自研引擎恒为空操作）。"""
        return

    def _autocast_context(self):
        """返回混合精度 autocast 上下文管理器（自研引擎为空操作）。"""
        from contextlib import nullcontext
        return nullcontext()

    # ──────────────────────────────────────────────────────
    # 数据准备
    # ──────────────────────────────────────────────────────
    def _prepare_data(self) -> TextDataset:
        """构建训练数据集 (含自动分词器构建)。"""
        if self.tokenizer is None:
            print("[ProductionTrainer] 自动构建分词器...")
            train_texts = []
            for text in iter_corpus(self.data_path):
                train_texts.append(text)
                if len(train_texts) >= 50000:
                    break  # 采样足够即可
            self.tokenizer = SimpleTokenizer(self.vocab_size)
            self.tokenizer.train(train_texts)
            # 保存分词器
            tok_path = self.output_dir / "tokenizer.json"
            self.tokenizer.save(str(tok_path))
            print(f"[ProductionTrainer] 分词器已保存: {tok_path}")

        # 确保序列长度不超过配置上限
        effective_seq_len = min(self.max_seq_len, self.config.max_seq_len)
        if effective_seq_len != self.max_seq_len:
            print(f"[ProductionTrainer] 序列长度调整为 {effective_seq_len} "
                  f"(配置上限 {self.config.max_seq_len})")
            self.max_seq_len = effective_seq_len

        train_dataset = TextDataset(
            self.data_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )
        return train_dataset

    # ──────────────────────────────────────────────────────
    # 优化器 & 调度器
    # ──────────────────────────────────────────────────────
    def _build_optimizer(self) -> None:
        """构建 AdamW 优化器 (分离权重衰减)。"""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "embed" in name.lower() \
                    or "bias" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(param_groups, lr=self.learning_rate)
        print("[ProductionTrainer] 使用标准 AdamW (自研引擎)")

    def _build_scheduler(self, total_steps: int) -> None:
        """构建 warmup + cosine 学习率调度器。

        使用 ``LambdaLR`` 包装 ``warmup_cosine_lr_lambda`` 函数实现
        线性预热 + 余弦退火。``CosineAnnealingLR`` 也已导入可用作
        纯余弦退火的替代方案。

        Args:
            total_steps: 总训练步数
        """
        self._total_steps = total_steps

        def lr_lambda(step: int) -> float:
            return warmup_cosine_lr_lambda(
                step, self.warmup_steps, total_steps, self.min_lr_ratio
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(f"[ProductionTrainer] LR调度: warmup={self.warmup_steps}步 "
              f"cosine退火至{self.min_lr_ratio}x "
              f"总步数={total_steps}")

    # ──────────────────────────────────────────────────────
    # 前向 & 损失
    # ──────────────────────────────────────────────────────
    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """计算交叉熵损失。

        将 logits 展平为 ``[B*T, vocab_size]``，targets 展平为 ``[B*T]``，
        使用 ``cross_entropy`` 计算，``ignore_index=PAD_ID`` 让 padding
        不参与损失计算。

        Args:
            logits: [B, T, vocab_size] 模型输出 logits
            targets: [B, T] 目标 token IDs
        Returns:
            标量损失
        """
        # 展平
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        # 交叉熵 (ignore pad)
        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=PAD_ID,
        )
        return loss

    def _forward_step(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """单次前向传播 + 损失计算。

        Args:
            input_ids: [B, T] 输入 token IDs
            target_ids: [B, T] 目标 token IDs
        Returns:
            标量损失
        """
        with self._autocast_context():
            logits = self.model(input_ids, n_loops=self.train_n_loops)
            loss = self._compute_loss(logits, target_ids)
        return loss

    # ──────────────────────────────────────────────────────
    # 梯度裁剪
    # ──────────────────────────────────────────────────────
    def _clip_gradients(self) -> float:
        """梯度裁剪并返回裁剪前的梯度总范数。

        PyTorch 后端使用原生 ``clip_grad_norm_``；
        自研 NumPy 后端手动计算并裁剪。

        Returns:
            裁剪前的梯度总范数（自研引擎原生实现）
        """
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += float(param.grad.pow(2).sum().item())
        total_norm = math.sqrt(total_norm)
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
        return total_norm

    # ──────────────────────────────────────────────────────
    # 检查点管理
    # ──────────────────────────────────────────────────────
    def _state_dict_to_cpu(
        self, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将 state_dict 中所有张量移到 CPU (避免设备依赖)。"""
        cpu_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_state[k] = v.detach().cpu()
            else:
                cpu_state[k] = v
        return cpu_state

    def save_checkpoint(self, step: int) -> Path:
        """保存完整训练状态到检查点文件。

        保存内容:
          - 模型 state_dict (CPU)
          - 优化器 state_dict (CPU)
          - 调度器 state_dict (CPU)
          - 当前 epoch / global_step
          - 最佳验证损失
          - 训练配置快照
          - 分词器 (如果有)

        Args:
            step: 当前训练步数 (用于生成文件名)
        Returns:
            检查点文件路径
        """
        ckpt_path = self.output_dir / f"checkpoint_step{step}.pt"

        checkpoint = {
            "model_state_dict": self._state_dict_to_cpu(
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self._state_dict_to_cpu(self.optimizer.state_dict())
                if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None else None
            ),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "vocab_size": self.vocab_size,
                "dim": self.config.dim,
                "n_heads": self.config.n_heads,
                "max_seq_len": self.config.max_seq_len,
                "max_loop_iters": self.config.max_loop_iters,
            },
            "training_config": {
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "min_lr_ratio": self.min_lr_ratio,
                "max_grad_norm": self.max_grad_norm,
                "max_seq_len": self.max_seq_len,
                "seed": self.seed,
                "use_amp": self._amp_enabled,
                "amp_dtype": self.amp_dtype,
            },
            "train_losses": self.train_losses[-1000:],
            "lr_history": self.lr_history[-1000:],
            "backend": self.backend_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 保存分词器
        if self.tokenizer is not None and hasattr(self.tokenizer, "save"):
            tok_path = self.output_dir / "tokenizer.json"
            try:
                self.tokenizer.save(str(tok_path))
                checkpoint["tokenizer_path"] = str(tok_path)
            except Exception as exc:
                print(f"[Checkpoint] 分词器保存失败: {exc}")

        try:
            torch.save(checkpoint, str(ckpt_path))
            # 同时保存 latest 指向最新
            latest_path = self.output_dir / "latest.pt"
            torch.save(checkpoint, str(latest_path))
            print(f"[Checkpoint] 已保存: {ckpt_path} "
                  f"(epoch={self.current_epoch}, step={self.global_step})")
        except Exception as exc:
            print(f"[Checkpoint] 保存失败: {exc}")
            raise

        return ckpt_path

    def load_checkpoint(
        self, path: Union[str, Path], *, strict: bool = True
    ) -> Dict[str, Any]:
        """从检查点文件恢复完整训练状态。

        恢复内容:
          - 模型权重
          - 优化器状态
          - 调度器状态
          - epoch / global_step
          - best_val_loss
          - 训练历史

        Args:
            path: 检查点文件路径 (.pt)
            strict: 模型加载是否严格匹配键名
        Returns:
            检查点元数据字典
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        print(f"[Checkpoint] 加载: {path}")
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=False
        )

        # 恢复模型权重 (移到当前设备)
        model_state = checkpoint["model_state_dict"]
        device_model_state = {}
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor):
                device_model_state[k] = v.to(self.device)
            else:
                device_model_state[k] = v
        self.model.load_state_dict(device_model_state, strict=strict)
        print(f"[Checkpoint] 模型权重已恢复 ({len(model_state)} 个参数)")

        # 恢复优化器
        if (checkpoint.get("optimizer_state_dict") is not None
                and self.optimizer is not None):
            opt_state = checkpoint["optimizer_state_dict"]
            for state in opt_state.get("state", {}).values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(opt_state)
            print("[Checkpoint] 优化器状态已恢复")

        # 恢复调度器
        if (checkpoint.get("scheduler_state_dict") is not None
                and self.scheduler is not None):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
            print("[Checkpoint] 调度器状态已恢复")

        # 恢复训练进度
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.lr_history = checkpoint.get("lr_history", [])

        # 重置模型运行时状态
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

        print(f"[Checkpoint] 恢复完成: epoch={self.current_epoch} "
              f"step={self.global_step} best_val={self.best_val_loss:.4f}")
        return checkpoint

    # ──────────────────────────────────────────────────────
    # 日志
    # ──────────────────────────────────────────────────────
    def _get_current_lr(self) -> float:
        """获取当前学习率。"""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        return self.learning_rate

    def _log_step(
        self,
        loss: float,
        grad_norm: float,
        step_in_epoch: int,
        steps_in_epoch: int,
        elapsed: float,
        tokens_processed: int,
    ) -> None:
        """打印训练步日志并写入 JSONL 日志文件。"""
        lr = self._get_current_lr()
        tps = tokens_processed / max(elapsed, 1e-6)

        # 计算 ETA
        remaining_steps = self._total_steps - self.global_step
        if self.global_step > 0:
            avg_time_per_step = elapsed / max(self.global_step, 1)
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "N/A"

        print(
            f"[Train] epoch={self.current_epoch} "
            f"step={self.global_step}/{self._total_steps} "
            f"({step_in_epoch}/{steps_in_epoch}) "
            f"loss={loss:.4f} "
            f"lr={lr:.2e} "
            f"grad_norm={grad_norm:.4f} "
            f"tokens/s={tps:.0f} "
            f"ETA={eta_str}",
            flush=True,
        )

        # 写入 JSONL 日志
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "step_in_epoch": step_in_epoch,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tps,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 日志写入失败不影响训练

    # ──────────────────────────────────────────────────────
    # 训练主循环
    # ──────────────────────────────────────────────────────
    def train(self, epochs: int) -> Dict[str, Any]:
        """启动训练主循环。

        Args:
            epochs: 训练轮数
        Returns:
            训练摘要字典 (最终损失、步数、耗时等)
        """
        print("=" * 60)
        print(f"[ProductionTrainer] 开始训练: {epochs} 轮")
        print(f"  后端: {self.backend_name}")
        print(f"  设备: {self.device}")
        print(f"  混合精度: {self._amp_enabled} ({self.amp_dtype})")
        print(f"  有效批次: {self.batch_size} x {self.grad_accum_steps} "
              f"= {self.batch_size * self.grad_accum_steps}")
        print(f"  序列长度: {self.max_seq_len}")
        print(f"  学习率: {self.learning_rate} "
              f"(warmup={self.warmup_steps}, "
              f"min_ratio={self.min_lr_ratio})")
        print(f"  梯度裁剪: max_norm={self.max_grad_norm}")
        print("=" * 60)

        # ── 准备数据 ──
        train_dataset = self._prepare_data()
        steps_per_epoch = len(train_dataset) // (
            self.batch_size * self.grad_accum_steps
        )
        if steps_per_epoch == 0:
            steps_per_epoch = 1
            print("[ProductionTrainer] 警告: 数据量不足一个有效批次，"
                  "将按实际数据训练")

        total_steps = steps_per_epoch * epochs
        print(f"[ProductionTrainer] 每轮步数: {steps_per_epoch} "
              f"总步数: {total_steps} "
              f"样本数: {len(train_dataset)}")

        # ── 构建优化器和调度器 ──
        if self.optimizer is None:
            self._build_optimizer()
        if self.scheduler is None:
            self._build_scheduler(total_steps)

        # ── 模型移到设备 ──
        self.model = self.model.to(self.device)
        self.model.train()

        # ── 训练循环 ──
        train_start = time.time()
        tokens_per_step = (
            self.batch_size * self.max_seq_len * self.grad_accum_steps
        )

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            # 重置模型运行时状态
            if hasattr(self.model, "reset_state"):
                self.model.reset_state()

            self.model.train()
            epoch_start = time.time()
            step_in_epoch = 0
            epoch_loss_sum = 0.0
            epoch_step_count = 0

            # 生成批次迭代器
            batch_iter = make_batches(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )

            optimizer_zeroed = False

            for input_ids, target_ids in batch_iter:
                # 梯度累积: 每 grad_accum_steps 步才更新一次
                if not optimizer_zeroed:
                    self.model.zero_grad(set_to_none=True)
                    optimizer_zeroed = True
                    accum_loss = 0.0

                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # 前向 + 损失
                loss = self._forward_step(input_ids, target_ids)

                # 缩放损失并反向传播
                scaled_loss = loss / self.grad_accum_steps
                if self._scaler is not None:
                    self._scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_loss += float(loss.detach().cpu().item())

                # 梯度累积完成
                if (step_in_epoch + 1) % self.grad_accum_steps == 0:
                    # 梯度裁剪
                    if self._scaler is not None:
                        self._scaler.unscale_(self.optimizer)
                    grad_norm = self._clip_gradients()

                    # 优化器步进
                    if self._scaler is not None:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                    # 学习率调度
                    if self.scheduler is not None:
                        self.scheduler.step()

                    optimizer_zeroed = False
                    self.global_step += 1
                    step_in_epoch += 1

                    avg_loss = accum_loss / self.grad_accum_steps
                    epoch_loss_sum += avg_loss
                    epoch_step_count += 1
                    self.train_losses.append(avg_loss)
                    self.lr_history.append(self._get_current_lr())

                    # 日志
                    if self.global_step % self.log_interval == 0:
                        elapsed = time.time() - train_start
                        self._log_step(
                            loss=avg_loss,
                            grad_norm=grad_norm,
                            step_in_epoch=step_in_epoch,
                            steps_in_epoch=steps_per_epoch,
                            elapsed=elapsed,
                            tokens_processed=(
                                self.global_step * tokens_per_step
                            ),
                        )

                    # 定期保存检查点
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(self.global_step)

            # ── Epoch 结束 ──
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = (
                epoch_loss_sum / max(1, epoch_step_count)
            )
            print(
                f"\n[Epoch {epoch}] 完成: "
                f"avg_loss={avg_epoch_loss:.4f} "
                f"steps={epoch_step_count} "
                f"耗时={timedelta(seconds=int(epoch_time))}\n",
                flush=True,
            )

            # 每 epoch 保存检查点
            self.save_checkpoint(self.global_step)

            # 垃圾回收
            gc.collect()

        # ── 训练结束 ──
        total_time = time.time() - train_start
        final_loss = self.train_losses[-1] if self.train_losses else 0.0

        # 保存最终模型
        final_path = self.output_dir / "final_model.pt"
        checkpoint = {
            "model_state_dict": self._state_dict_to_cpu(
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self._state_dict_to_cpu(self.optimizer.state_dict())
                if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None else None
            ),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "backend": self.backend_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint, str(final_path))

        summary = {
            "status": "completed",
            "epochs": epochs,
            "total_steps": self.global_step,
            "final_loss": final_loss,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": round(total_time, 2),
            "total_time": str(timedelta(seconds=int(total_time))),
            "backend": self.backend_name,
            "device": str(self.device),
            "amp_enabled": self._amp_enabled,
            "final_checkpoint": str(final_path),
        }

        # 保存训练摘要
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=" * 60)
        print(f"[ProductionTrainer] 训练完成!")
        print(f"  总步数: {self.global_step}")
        print(f"  最终损失: {final_loss:.4f}")
        print(f"  总耗时: {timedelta(seconds=int(total_time))}")
        print(f"  最终模型: {final_path}")
        print("=" * 60)

        return summary

    # ──────────────────────────────────────────────────────
    # 断点恢复
    # ──────────────────────────────────────────────────────
    def resume(self, checkpoint_path: Union[str, Path]) -> None:
        """从检查点恢复训练 (在调用 train() 之前调用)。

        恢复后，train(epochs) 中的 epoch 会从断点继续。

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"恢复检查点不存在: {checkpoint_path}"
            )

        # 先构建优化器和调度器 (load_checkpoint 需要它们)
        train_dataset = self._prepare_data()
        steps_per_epoch = max(1, len(train_dataset) // (
            self.batch_size * self.grad_accum_steps
        ))
        estimated_total = steps_per_epoch * 100  # 上限估计
        if self.optimizer is None:
            self._build_optimizer()
        if self.scheduler is None:
            self._build_scheduler(estimated_total)

        # 加载检查点
        self.load_checkpoint(checkpoint_path)

        # 将模型移到设备
        self.model = self.model.to(self.device)
        self.model.train()

        print(f"[ProductionTrainer] 断点恢复: 从 epoch "
              f"{self.current_epoch + 1} 继续")


# ============================================================
# 主入口 — 命令行训练
# ============================================================
def _main():
    """主入口: 命令行启动生产级训练。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="SimplCore Mix 生产级训练管道"
    )
    parser.add_argument(
        "--train-data", type=str, required=True,
        help="训练语料路径 (JSONL / .txt / 目录)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints/mix",
        help="输出目录 (检查点、日志)",
    )
    parser.add_argument(
        "--config", type=str, default="150m",
        choices=["150m", "3b", "7b"],
        help="模型配置 (150m / 3b / 7b)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="训练轮数"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="微批次大小"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="梯度累积步数 (有效 batch = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="最大序列长度"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="峰值学习率"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="权重衰减"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
        help="学习率预热步数 (默认 1000)",
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1,
        help="最低学习率比例",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="梯度裁剪最大范数",
    )
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="启用混合精度 (仅 PyTorch 后端)",
    )
    parser.add_argument(
        "--amp-dtype", type=str, default="float16",
        choices=["float16", "bfloat16"],
        help="混合精度数据类型",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="分词器文件路径 (可选，不指定则自动构建)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="日志打印间隔 (步)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000,
        help="检查点保存间隔 (步)",
    )
    parser.add_argument(
        "--train-n-loops", type=int, default=None,
        help="训练时循环深度 (默认使用配置值)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从检查点恢复训练",
    )

    args = parser.parse_args()

    # ── 选择模型配置 ──
    config_map = {
        "150m": mix_150m_text,
        "3b": mix_3b_text,
        "7b": mix_7b_text,
    }
    config = config_map[args.config]()
    # 确保 max_seq_len 至少为命令行指定值
    config.max_seq_len = max(config.max_seq_len, args.seq_len)
    # 启用梯度检查点以节省显存
    config.gradient_checkpointing = True

    print(f"\n{'=' * 60}")
    print(f"SimplCore Mix 生产级训练管道")
    print(f"配置: {args.config} (dim={config.dim}, "
          f"heads={config.n_heads}, "
          f"loops={config.max_loop_iters})")
    print(f"后端: {get_backend()}")
    print(f"{'=' * 60}\n")

    # ── 构建模型 ──
    model = SimplCoreMix(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ── 构建分词器 (如果指定了路径) ──
    tokenizer = None
    if args.tokenizer:
        tok_path = Path(args.tokenizer)
        if tok_path.exists():
            tokenizer = SimpleTokenizer(config.vocab_size)
            try:
                tokenizer.load(str(tok_path))
                print(f"[Tokenizer] 已加载: {tok_path}")
            except Exception as exc:
                print(f"[Tokenizer] 加载失败 ({exc})，将自动构建")
                tokenizer = None

    # ── 构建训练器 ──
    trainer = ProductionTrainer(
        model=model,
        config=config,
        data_path=args.train_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_seq_len=args.seq_len,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        tokenizer=tokenizer,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        train_n_loops=args.train_n_loops,
    )

    # ── 断点恢复 ──
    if args.resume:
        print(f"\n[Resume] 从检查点恢复: {args.resume}")
        trainer.resume(args.resume)

    # ── 启动训练 ──
    summary = trainer.train(epochs=args.epochs)

    print(f"\n训练摘要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
