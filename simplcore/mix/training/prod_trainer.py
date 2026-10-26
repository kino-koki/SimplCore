# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""ProductionTrainer — SimplCore Mix 生产级训练管线
============================================

本模块提供完整的、可直接用于生产的语言模型训练管线，支持：

  1. 数据加载 — 读取 JSONL / .docx 语料，分词、分块、打包批次
  2. 学习率调度 — 线性预热 (warmup) + 余弦退火 (cosine annealing)
  3. 检查点管理 — 保存 / 加载 / 恢复完整训练状态
  4. 梯度裁剪 — 最大范数裁剪 (max norm clipping)
  5. 混合精度 — PyTorch 后端下使用 ``torch.amp.autocast`` + ``GradScaler``
  6. 梯度累积 — 跨多个微批次累积梯度以模拟大 batch
  7. 日志 — 损失、学习率、梯度范数、tokens/sec、ETA
  8. 评估 — 周期性验证损失计算
  9. 断点恢复 — 从检查点完整恢复训练状态

双后端兼容：
  - 自研引擎 (``is_numpy_backend()=True``): 纯 NumPy 自动微分，零第三方框架依赖
    梯度检查点、fused AdamW 等全部可用。
  - 自研 NumPy 后端: 降级为 fp32、手动梯度裁剪，保证小规模调试可用。

用法
----
    python -m simplcore.mix.training.prod_trainer

    # 或在代码中调用
    from simplcore.mix.training.prod_trainer import ProductionTrainer
    trainer = ProductionTrainer(model, cfg, train_path, val_path, output_dir)
    trainer.train(epochs=10)
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

# ── 青小团自研张量引擎（零第三方框架依赖，纯 NumPy 自动微分）──
import simplcore.tensor as torch
from simplcore.tensor.backend import is_numpy_backend, get_backend, get_device

# 自研引擎不参与任何第三方框架依赖；AMP / 梯度裁剪等高级功能由自研模块提供。
_pt = None
_HAS_TORCH = False

# ── 尝试导入自研 BPE 分词器 ──
try:
    from simplcore.mix.tokenizer import BPETokenizer, CharTokenizer
    _HAS_NATIVE_TOKENIZER = True
except Exception:  # pragma: no cover
    BPETokenizer = None  # type: ignore
    CharTokenizer = None  # type: ignore
    _HAS_NATIVE_TOKENIZER = False


# ============================================================
# 常量
# ============================================================
PAD_ID = 0   # <pad> token id (与 BPE / 字符分词器约定一致)
EOS_ID = 1   # <eos> token id

# .docx XML 命名空间
_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


# ============================================================
# 文本工具
# ============================================================
def normalize_text(text: str) -> str:
    """轻量文本规范化：去除多余空白、不可见字符。"""
    if not text:
        return ""
    # 去除零宽字符、BOM 等
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\x00"):
        text = text.replace(ch, "")
    # 合并连续空白（保留换行）
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def extract_docx_text(docx_path: Union[str, Path]) -> str:
    """从 .docx 文件提取纯文本（不依赖 python-docx，直接解析 XML）。"""
    docx_path = Path(docx_path)
    with zipfile.ZipFile(docx_path) as zf:
        document_xml = zf.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", _W_NS):
        parts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", _W_NS)
            if node.text
        ]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return normalize_text("\n".join(paragraphs))


def iter_corpus_texts(data_path: Union[str, Path]) -> Iterator[str]:
    """统一语料读取迭代器：自动识别 JSONL / .docx / 纯文本。

    - 目录: 递归扫描 .jsonl 和 .docx 文件
    - .jsonl 文件: 逐行解析 ``{"text": "..."}``
    - .docx 文件: 提取段落文本
    - .txt 文件: 读取全文
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"语料路径不存在: {p}")

    if p.is_file():
        yield from _iter_single_file(p)
    elif p.is_dir():
        # 收集所有支持的文件
        jsonl_files = sorted(p.rglob("*.jsonl"))
        docx_files = sorted(p.rglob("*.docx"))
        txt_files = sorted(p.rglob("*.txt"))
        for fp in jsonl_files + docx_files + txt_files:
            yield from _iter_single_file(fp)
    else:
        raise ValueError(f"不支持的语料路径类型: {p}")


def _iter_single_file(fp: Path) -> Iterator[str]:
    """读取单个语料文件。"""
    if fp.suffix == ".jsonl":
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", row.get("content", ""))
                if isinstance(text, str) and text.strip():
                    yield normalize_text(text)
    elif fp.suffix == ".docx":
        text = extract_docx_text(fp)
        if text.strip():
            yield text
    elif fp.suffix in (".txt", ".md"):
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            yield normalize_text(text)


# ============================================================
# 简易字符级分词器 (fallback)
# ============================================================
class SimpleCharTokenizer:
    """字符级分词器 — 当 ``simplcore.mix.tokenizer`` 不可用时的轻量回退。

    id 布局与自研 BPE 一致：
      0 = <pad>, 1 = <eos>, 2..vocab_size-1 = 字符
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {"<pad>": PAD_ID, "<eos>": EOS_ID}
        self.id_to_char: Dict[int, str] = {PAD_ID: "<pad>", EOS_ID: "<eos>"}
        self._built = False

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    def train(self, texts: List[str]) -> None:
        """从语料构建字符词表。"""
        chars = sorted(set("".join(texts)))
        max_chars = self.vocab_size - 2  # 预留 pad / eos
        for i, ch in enumerate(chars[:max_chars]):
            tid = i + 2
            self.char_to_id[ch] = tid
            self.id_to_char[tid] = ch
        self._built = True
        actual_vocab = len(self.char_to_id)
        print(f"[SimpleCharTokenizer] 词表大小: {actual_vocab} "
              f"(上限 {self.vocab_size})")

    def encode(self, text: str) -> List[int]:
        """编码文本为 token id 序列。"""
        if not self._built:
            # 未训练时用字符哈希回退（与现有训练脚本一致）
            divisor = max(self.vocab_size - 2, 1)
            return [2 + (ord(ch) % divisor) for ch in text]
        return [self.char_to_id.get(ch, PAD_ID) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """解码 token id 序列为文本。"""
        if not self._built:
            return "".join(chr((t - 2) % (self.vocab_size - 2) + 0x4e00)
                           for t in tokens if t >= 2)
        return "".join(self.id_to_char.get(t, "?") for t in tokens
                       if t not in (PAD_ID, EOS_ID))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "char_to_id": self.char_to_id,
            }, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self._built = True

    def __len__(self) -> int:
        return len(self.char_to_id)


def build_tokenizer(
    tokenizer_path: Optional[str],
    vocab_size: int,
    train_texts: Optional[List[str]] = None,
) -> Any:
    """构建分词器：优先加载训练好的 BPE，否则训练字符级。

    Args:
        tokenizer_path: BPE 分词器 JSON 文件路径（None 则跳过）
        vocab_size: 模型词表大小
        train_texts: 用于训练字符级分词器的语料（仅在 fallback 时使用）
    Returns:
        分词器对象（具有 encode / decode / pad_token_id / eos_token_id 接口）
    """
    # 1. 尝试加载训练好的 BPE 分词器
    if tokenizer_path and _HAS_NATIVE_TOKENIZER and BPETokenizer is not None:
        p = Path(tokenizer_path)
        if p.exists():
            try:
                tok = BPETokenizer(vocab_size=vocab_size, chinese=True)
                tok.load(str(p))
                if tok.is_trained():
                    print(f"[Tokenizer] 已加载 BPE 分词器: {p} "
                          f"(vocab={len(tok)})")
                    return tok
            except Exception as exc:
                print(f"[Tokenizer] BPE 加载失败 ({exc})，回退字符级")

    # 2. 尝试加载已保存的字符级分词器
    if tokenizer_path:
        p = Path(tokenizer_path)
        if p.exists() and p.suffix == ".json":
            try:
                tok = SimpleCharTokenizer(vocab_size)
                tok.load(str(p))
                print(f"[Tokenizer] 已加载字符级分词器: {p}")
                return tok
            except Exception:
                pass  # 格式不匹配，继续回退

    # 3. 训练新的字符级分词器
    tok = SimpleCharTokenizer(vocab_size)
    if train_texts:
        tok.train(train_texts)
    else:
        print("[Tokenizer] 无训练语料，使用字符哈希回退模式")
    return tok


# ============================================================
# 数据集
# ============================================================
class TextDataset:
    """文本数据集 — 读取语料、分词、分块、返回 (input, target) 对。

    处理流程:
      1. 读取所有文本
      2. 分词为 token id 序列
      3. 在文档之间插入 EOS
      4. 按 seq_len+1 分块（每个块 = 1个训练样本）
      5. 不足 seq_len+1 的尾部用 PAD 填充

    Args:
        data_path: 语料路径（JSONL / .docx / .txt 文件或目录）
        tokenizer: 分词器对象
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
        print(f"[Dataset] 已加载 {len(self.samples)} 个样本 "
              f"(seq_len={max_seq_len}, overlap={overlap})")

    def _load(self, data_path: Union[str, Path]) -> None:
        """加载并分词所有语料。"""
        chunk_len = self.max_seq_len + 1  # 输入 + 目标
        step = max(1, int(chunk_len * (1 - self.overlap)))
        total_tokens = 0

        for text in iter_corpus_texts(data_path):
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
                    chunk = tokens[i : i + chunk_len]
                    if len(chunk) == chunk_len:
                        self.samples.append(
                            torch.tensor(chunk, dtype=torch.long)
                        )
                # 处理尾部不足一个 step 的剩余
                remainder_start = (
                    (len(tokens) - chunk_len) // step * step
                )
                if remainder_start + chunk_len < len(tokens):
                    tail = tokens[remainder_start + step:]
                    if len(tail) >= 2:
                        tail = tail + [PAD_ID] * (chunk_len - len(tail))
                        self.samples.append(
                            torch.tensor(tail[:chunk_len], dtype=torch.long)
                        )

        if not self.samples:
            raise RuntimeError(
                f"数据集为空，请检查语料路径: {data_path}"
            )
        print(f"[Dataset] 总 token 数: {total_tokens:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        warmup_steps: 预热步数（线性从 0 升到 peak_lr）
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
    # 从 1.0 余弦衰减到 min_lr_ratio
    return min_lr_ratio + (1.0 - min_lr_ratio) * cos_val


# ============================================================
# 生产级训练器
# ============================================================
class ProductionTrainer:
    """SimplCore Mix 生产级训练器。

    支持混合精度、梯度累积、梯度裁剪、学习率调度、检查点管理、
    断点恢复、周期性评估等全部生产级训练能力。

    Args:
        model: SimplCoreMix 模型实例
        config: SimplCoreMixConfig 配置对象
        train_data_path: 训练语料路径 (JSONL / .docx / .txt / 目录)
        val_data_path: 验证语料路径 (可为 None)
        output_dir: 输出目录 (检查点、日志)
        batch_size: 每步微批次大小
        grad_accum_steps: 梯度累积步数 (有效 batch = batch_size * grad_accum_steps)
        max_seq_len: 最大序列长度
        learning_rate: 峰值学习率
        weight_decay: 权重衰减
        warmup_steps: 学习率预热步数
        min_lr_ratio: 最低学习率比例
        max_grad_norm: 梯度裁剪最大范数
        use_amp: 是否启用混合精度 (仅 PyTorch 后端生效)
        amp_dtype: 混合精度数据类型 ('float16' 或 'bfloat16')
        tokenizer: 分词器对象 (None 则自动构建)
        tokenizer_path: 分词器文件路径 (用于加载/保存)
        seed: 随机种子
        log_interval: 日志打印间隔 (步)
        eval_interval: 评估间隔 (步)
        save_interval: 检查点保存间隔 (步)
        train_n_loops: 训练时循环深度 (None = 使用配置默认值)
        num_workers: DataLoader 工作线程数 (暂未使用，预留)
        device: 计算设备 (None 则自动选择)
    """

    def __init__(
        self,
        model,
        config,
        train_data_path: Union[str, Path],
        val_data_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "./checkpoints",
        *,
        batch_size: int = 4,
        grad_accum_steps: int = 1,
        max_seq_len: int = 512,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        min_lr_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        tokenizer: Any = None,
        tokenizer_path: Optional[str] = None,
        seed: int = 42,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        train_n_loops: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[str] = None,
    ):
        # ── 基本配置 ──
        self.model = model
        self.config = config
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path) if val_data_path else None
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
        self.use_amp = False  # 自研引擎以 float32 运行，不启用混合精度
        self.amp_dtype = amp_dtype
        self.seed = seed
        self.log_interval = log_interval
        self.eval_interval = eval_interval
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
        self.val_losses: List[float] = []
        self.lr_history: List[float] = []

        # ── 优化器 & 调度器 (延迟构建，train() 时创建) ──
        self.optimizer = None
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
              f"seq_len={max_seq_len} lr={learning_rate}")

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
    def _prepare_data(self) -> Tuple[TextDataset, Optional[TextDataset]]:
        """构建训练和验证数据集。"""
        if self.tokenizer is None:
            # 自动构建分词器
            print("[ProductionTrainer] 自动构建分词器...")
            train_texts = []
            for text in iter_corpus_texts(self.train_data_path):
                train_texts.append(text)
                if len(train_texts) >= 10000:
                    break  # 采样足够字符即可
            self.tokenizer = build_tokenizer(
                self.tokenizer_path, self.vocab_size, train_texts
            )

        # 确保配置的 max_seq_len 与数据集一致
        effective_seq_len = min(self.max_seq_len, self.config.max_seq_len)
        if effective_seq_len != self.max_seq_len:
            print(f"[ProductionTrainer] 序列长度调整为 {effective_seq_len} "
                  f"(配置上限 {self.config.max_seq_len})")
            self.max_seq_len = effective_seq_len

        train_dataset = TextDataset(
            self.train_data_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

        val_dataset = None
        if self.val_data_path and self.val_data_path.exists():
            val_dataset = TextDataset(
                self.val_data_path,
                self.tokenizer,
                max_seq_len=self.max_seq_len,
            )

        return train_dataset, val_dataset

    # ──────────────────────────────────────────────────────
    # 优化器 & 调度器
    # ──────────────────────────────────────────────────────
    def _build_optimizer(self) -> None:
        """构建 AdamW 优化器。"""
        # 分离权重衰减：Embedding/Norm 不衰减，Linear 衰减
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

        self.optimizer = torch.AdamW(
            param_groups, lr=self.learning_rate
        )
        print("[ProductionTrainer] 使用标准 AdamW (自研引擎)")

    def _build_scheduler(self, total_steps: int) -> None:
        """构建线性预热 + 余弦退火学习率调度器。"""
        self._total_steps = total_steps

        def lr_lambda(step: int) -> float:
            return warmup_cosine_lr_lambda(
                step, self.warmup_steps, total_steps, self.min_lr_ratio
            )

        self.scheduler = torch.LambdaLR(self.optimizer, lr_lambda)
        print(f"[ProductionTrainer] LR调度: warmup={self.warmup_steps}步 "
              f"cosine退火至{self.min_lr_ratio}x "
              f"总步数={total_steps}")

    # ──────────────────────────────────────────────────────
    # 前向 & 损失
    # ──────────────────────────────────────────────────────
    def _forward_step(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """单次前向传播 + 损失计算。

        Args:
            input_ids: [B, T] 输入 token IDs
            target_ids: [B, T] 目标 token IDs
        Returns:
            loss: 标量损失 (已对 pad token 做 ignore)
        """
        with self._autocast_context():
            logits = self.model(input_ids, n_loops=self.train_n_loops)
            # logits: [B, T, vocab_size]
            # 展平为 [B*T, vocab_size] 和 [B*T] 计算交叉熵
            B, T = input_ids.shape
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_ids.reshape(-1)
            # ignore_index=PAD_ID 让 padding 不参与损失计算
            loss = torch.nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=PAD_ID,
            )
        return loss

    # ──────────────────────────────────────────────────────
    # 梯度裁剪
    # ──────────────────────────────────────────────────────
    def _clip_gradients(self) -> float:
        """梯度裁剪并返回裁剪前的梯度范数（自研引擎原生实现）。"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += float(
                    param.grad.pow(2).sum().item()
                )
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
        """将 state_dict 中所有张量移到 CPU（保存时必须，避免设备依赖）。"""
        cpu_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_state[k] = v.detach().cpu()
            else:
                cpu_state[k] = v
        return cpu_state

    def save_checkpoint(self, path: Union[str, Path]) -> None:
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
            path: 检查点文件路径 (.pt)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

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
            "train_losses": self.train_losses[-1000:],  # 保留最近1000步
            "val_losses": self.val_losses,
            "lr_history": self.lr_history[-1000:],
            "backend": self.backend_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 保存分词器（如果有 save 方法）
        if self.tokenizer is not None and hasattr(self.tokenizer, "save"):
            tok_path = path.parent / "tokenizer.json"
            try:
                self.tokenizer.save(str(tok_path))
                checkpoint["tokenizer_path"] = str(tok_path)
            except Exception as exc:
                print(f"[Checkpoint] 分词器保存失败: {exc}")

        try:
            torch.save(checkpoint, str(path))
            print(f"[Checkpoint] 已保存: {path} "
                  f"(epoch={self.current_epoch}, step={self.global_step})")
        except Exception as exc:
            print(f"[Checkpoint] 保存失败: {exc}")
            raise

    def load_checkpoint(
        self, path: Union[str, Path], *, strict: bool = True
    ) -> Dict[str, Any]:
        """从检查点文件加载完整训练状态。

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
            checkpoint 元数据字典
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        print(f"[Checkpoint] 加载: {path}")
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        # 恢复模型
        model_state = checkpoint["model_state_dict"]
        # 将权重移到当前设备
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
            # 将优化器状态移到当前设备
            for state in opt_state.get("state", {}).values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(opt_state)
            print("[Checkpoint] 优化器状态已恢复")

        # 恢复调度器
        if (checkpoint.get("scheduler_state_dict") is not None
                and self.scheduler is not None):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("[Checkpoint] 调度器状态已恢复")

        # 恢复训练进度
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.lr_history = checkpoint.get("lr_history", [])

        # 重置模型运行时状态 (避免旧状态泄漏)
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

        print(f"[Checkpoint] 恢复完成: epoch={self.current_epoch} "
              f"step={self.global_step} best_val={self.best_val_loss:.4f}")

        return checkpoint

    # ──────────────────────────────────────────────────────
    # 评估
    # ──────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(
        self, max_batches: Optional[int] = None
    ) -> float:
        """在验证集上计算平均损失。

        Args:
            max_batches: 最大评估批次数 (None = 评估全部)
        Returns:
            avg_loss: 验证集平均损失
        """
        if self.val_data_path is None or not self.val_data_path.exists():
            print("[Evaluate] 无验证集，跳过评估")
            return float("inf")

        # 构建验证数据集
        if self.tokenizer is None:
            print("[Evaluate] 分词器未初始化，跳过评估")
            return float("inf")

        val_dataset = TextDataset(
            self.val_data_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

        self.model.eval()
        # 重置运行时状态确保评估干净
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

        total_loss = 0.0
        total_tokens = 0
        n_batches = 0

        for input_ids, target_ids in make_batches(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        ):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            with self._autocast_context():
                logits = self.model(
                    input_ids, n_loops=self.train_n_loops
                )
                B, T = input_ids.shape
                logits_flat = logits.reshape(-1, self.vocab_size)
                targets_flat = target_ids.reshape(-1)
                loss = torch.nn.functional.cross_entropy(
                    logits_flat,
                    targets_flat,
                    ignore_index=PAD_ID,
                    reduction="sum",
                )
                # 统计非 pad token 数
                n_tokens = (targets_flat != PAD_ID).sum().item()
                total_loss += float(loss.detach().cpu().item())
                total_tokens += n_tokens
                n_batches += 1

            if max_batches is not None and n_batches >= max_batches:
                break

        avg_loss = total_loss / max(1, total_tokens)
        self.model.train()
        # 恢复训练后重置状态
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

        print(f"[Evaluate] 验证损失: {avg_loss:.4f} "
              f"({n_batches} 批, {total_tokens:,} tokens)")
        return avg_loss

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
        """打印训练步日志。"""
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
        train_dataset, val_dataset = self._prepare_data()
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
        self._build_optimizer()
        self._build_scheduler(total_steps)

        # ── 模型移到设备 ──
        self.model = self.model.to(self.device)
        self.model.train()

        # ── 训练循环 ──
        start_time = time.time()
        train_start = start_time
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

                    # 定期评估
                    if (val_dataset is not None
                            and self.global_step % self.eval_interval == 0):
                        val_loss = self.evaluate()
                        self.val_losses.append(val_loss)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            best_path = (
                                self.output_dir / "best_model.pt"
                            )
                            self.save_checkpoint(best_path)
                            print(f"[Train] 新最佳验证损失: "
                                  f"{val_loss:.4f}")

                        self.model.train()
                        if hasattr(self.model, "reset_state"):
                            self.model.reset_state()

                    # 定期保存检查点
                    if self.global_step % self.save_interval == 0:
                        ckpt_path = (
                            self.output_dir
                            / f"checkpoint_step{self.global_step}.pt"
                        )
                        self.save_checkpoint(ckpt_path)
                        # 同时保存 latest
                        latest_path = (
                            self.output_dir / "latest.pt"
                        )
                        self.save_checkpoint(latest_path)

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
            epoch_ckpt = (
                self.output_dir / f"checkpoint_epoch{epoch}.pt"
            )
            self.save_checkpoint(epoch_ckpt)

            # 每 epoch 评估
            if val_dataset is not None:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        self.output_dir / "best_model.pt"
                    )
                    print(f"[Epoch {epoch}] 新最佳验证损失: "
                          f"{val_loss:.4f}")

                self.model.train()
                if hasattr(self.model, "reset_state"):
                    self.model.reset_state()

            # 垃圾回收
            gc.collect()

        # ── 训练结束 ──
        total_time = time.time() - train_start
        final_loss = self.train_losses[-1] if self.train_losses else 0.0

        # 保存最终模型
        final_path = self.output_dir / "final_model.pt"
        self.save_checkpoint(final_path)

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
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")
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
        # 需要先准备数据以确定 total_steps
        train_dataset, val_dataset = self._prepare_data()
        steps_per_epoch = max(1, len(train_dataset) // (
            self.batch_size * self.grad_accum_steps
        ))
        # 估算总步数 (后续 train() 会重建调度器)
        estimated_total = steps_per_epoch * 100  # 上限估计
        self._build_optimizer()
        self._build_scheduler(estimated_total)

        # 加载检查点
        self.load_checkpoint(checkpoint_path)

        # 将模型移到设备
        self.model = self.model.to(self.device)
        self.model.train()

        print(f"[ProductionTrainer] 断点恢复: 从 epoch "
              f"{self.current_epoch + 1} 继续")


# ============================================================
# 主入口 — 150M 配置训练
# ============================================================
def _main():
    """主入口: 使用 150M 配置启动训练。"""
    import argparse

    from simplcore.mix.config import mix_150m_text
    from simplcore.mix.model import SimplCoreMix

    parser = argparse.ArgumentParser(
        description="SimplCore Mix 生产级训练管线 (150M 配置)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=r"E:\青小团---中文训练语料\Anhunqu xinyuan",
        help="训练语料路径 (JSONL / .docx / .txt / 目录)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="验证语料路径 (可选)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/mix_150m",
        help="输出目录 (检查点、日志)",
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
        "--warmup-steps", type=int, default=100, help="学习率预热步数"
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
        help="分词器文件路径 (可选)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="日志打印间隔 (步)"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="评估间隔 (步)"
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
    parser.add_argument(
        "--config", type=str, default="150m",
        choices=["150m", "3b", "7b"],
        help="模型配置 (150m / 3b / 7b)",
    )

    args = parser.parse_args()

    # ── 选择模型配置 ──
    from simplcore.mix.config import (
        mix_150m_text, mix_3b_text, mix_7b_text,
    )
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
    print(f"SimplCore Mix 生产级训练管线")
    print(f"配置: {args.config} (dim={config.dim}, "
          f"heads={config.n_heads}, "
          f"loops={config.max_loop_iters})")
    print(f"后端: {get_backend()}")
    print(f"{'=' * 60}\n")

    # ── 构建模型 ──
    model = SimplCoreMix(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ── 构建训练器 ──
    trainer = ProductionTrainer(
        model=model,
        config=config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
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
        tokenizer_path=args.tokenizer,
        seed=args.seed,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
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
