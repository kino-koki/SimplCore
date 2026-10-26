# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 数据加载器
支持：JSONL、原始文本、内存映射、流式读取
不依赖第三方datasets库
"""
import json
import os
import random
import mmap
from pathlib import Path
from typing import List, Optional, Iterator, Callable, Union
import simplcore.tensor as torch
from simplcore.tensor.utils.data import Dataset, IterableDataset


class SCDataLoader(torch.utils.data.DataLoader):
    """SimplCore Mix 数据加载器。

    继承当前隐藏生态后端的数据管线能力，对外使用 SimplCore 命名。
    """

    backend_name = "simplcore_native"


class TextDataset(Dataset):
    """文本数据集（加载整个文件到内存）"""

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        max_seq_len: int = 2048,
        overlap: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.samples: List[torch.Tensor] = []

        if isinstance(data_path, str):
            data_path = [data_path]

        for path in data_path:
            self._load_file(path)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {len(data_path)} file(s)")

    def _load_file(self, path: str):
        path = Path(path)
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data.get("text", data.get("content", ""))
                    self._tokenize_and_chunk(text)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item if isinstance(item, str) else item.get("text", "")
                        self._tokenize_and_chunk(text)
                elif isinstance(data, dict):
                    text = data.get("text", "")
                    self._tokenize_and_chunk(text)
        else:
            # 原始文本文件
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                self._tokenize_and_chunk(text)

    def _tokenize_and_chunk(self, text: str):
        tokens = self.tokenizer.encode(text)
        step = int(self.max_seq_len * (1 - self.overlap))
        for i in range(0, len(tokens) - self.max_seq_len, step):
            chunk = tokens[i : i + self.max_seq_len + 1]
            if len(chunk) == self.max_seq_len + 1:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        tokens = self.samples[idx]
        return tokens[:-1], tokens[1:]


class MemoryMappedDataset(Dataset):
    """内存映射数据集（适合超大文件，不加载整个文件到RAM）"""

    def __init__(
        self,
        bin_path: str,
        max_seq_len: int = 2048,
        dtype: str = "uint16",
    ):
        self.max_seq_len = max_seq_len
        self.dtype = torch.uint16 if dtype == "uint16" else torch.int32

        self.file = open(bin_path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.num_tokens = len(self.mm) // 2 if dtype == "uint16" else len(self.mm) // 4
        self.num_samples = max(0, self.num_tokens - max_seq_len)

        print(f"[MemmapDataset] {bin_path}: {self.num_tokens:,} tokens, {self.num_samples:,} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        idx = min(idx, self.num_samples - 1)
        byte_offset = idx * (2 if self.dtype == torch.uint16 else 4)
        num_bytes = (self.max_seq_len + 1) * (2 if self.dtype == torch.uint16 else 4)

        data = self.mm[byte_offset : byte_offset + num_bytes]
        tokens = torch.frombuffer(bytearray(data), dtype=self.dtype).long()

        if len(tokens) < self.max_seq_len + 1:
            tokens = torch.cat([tokens, torch.zeros(self.max_seq_len + 1 - len(tokens), dtype=torch.long)])

        return tokens[:-1], tokens[1:]

    def __del__(self):
        if hasattr(self, "mm"):
            self.mm.close()
        if hasattr(self, "file"):
            self.file.close()


class StreamingTextDataset(IterableDataset):
    """流式文本数据集（无限迭代，适合在线学习/大规模数据）"""

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        max_seq_len: int = 2048,
        shuffle_buffer: int = 10000,
    ):
        self.data_path = [data_path] if isinstance(data_path, str) else data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer

    def _read_texts(self) -> Iterator[str]:
        """无限循环读取文本"""
        while True:
            random.shuffle(self.data_path)
            for path in self.data_path:
                path = Path(path)
                if path.suffix == ".jsonl":
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            yield data.get("text", data.get("content", ""))
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        yield f.read()

    def _token_stream(self) -> Iterator[int]:
        """生成token流"""
        for text in self._read_texts():
            yield from self.tokenizer.encode(text)

    def __iter__(self) -> Iterator[tuple]:
        buffer = []
        token_iter = self._token_stream()

        for token in token_iter:
            buffer.append(token)
            if len(buffer) >= self.max_seq_len + 1:
                sample = torch.tensor(buffer[: self.max_seq_len + 1], dtype=torch.long)
                yield sample[:-1], sample[1:]
                buffer = buffer[self.max_seq_len :]

                # Shuffle buffer
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)


class PretokenizedDataset(Dataset):
    """预分词数据集（token已预先计算好，直接加载）"""

    def __init__(self, token_file: str, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.tokens = torch.load(token_file, weights_only=True)
        if self.tokens.dim() == 1:
            self.num_samples = max(0, len(self.tokens) - max_seq_len)
        else:
            self.num_samples = len(self.tokens)
        print(f"[PretokenizedDataset] Loaded {self.num_samples:,} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        if self.tokens.dim() == 1:
            tokens = self.tokens[idx : idx + self.max_seq_len + 1]
            return tokens[:-1], tokens[1:]
        else:
            tokens = self.tokens[idx]
            return tokens[:-1], tokens[1:]


def build_data_loader(
    data_path: Union[str, List[str]],
    tokenizer,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    num_workers: int = 2,
    shuffle: bool = True,
    dataset_type: str = "auto",
) -> SCDataLoader:
    """构建数据加载器（自动推断数据集类型）"""

    path = Path(data_path[0] if isinstance(data_path, list) else data_path)

    if dataset_type == "auto":
        if path.suffix in (".pt", ".pth"):
            dataset_type = "pretokenized"
        elif path.suffix == ".bin":
            dataset_type = "memmap"
        else:
            dataset_type = "text"

    if dataset_type == "memmap":
        dataset = MemoryMappedDataset(str(path), max_seq_len)
    elif dataset_type == "pretokenized":
        dataset = PretokenizedDataset(str(path), max_seq_len)
    elif dataset_type == "streaming":
        dataset = StreamingTextDataset(data_path, tokenizer, max_seq_len)
        return SCDataLoader(dataset, batch_size=batch_size, num_workers=0)
    else:
        dataset = TextDataset(data_path, tokenizer, max_seq_len)

    return SCDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def pretokenize_corpus(
    input_paths: List[str],
    tokenizer,
    output_path: str,
    dtype: str = "uint16",
) -> None:
    """将语料预分词并保存为二进制文件（供MemoryMappedDataset使用）"""
    print(f"[Pretokenize] Processing {len(input_paths)} files...")

    all_tokens = []
    for path in input_paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data.get("text", data.get("content", ""))
                    all_tokens.extend(tokenizer.encode(text))
        else:
            with open(path, "r", encoding="utf-8") as f:
                all_tokens.extend(tokenizer.encode(f.read()))

    tokens = torch.tensor(all_tokens, dtype=torch.long)

    if dtype == "uint16":
        if tokens.max() >= 65536:
            print("[Warning] Vocab size > 65535, falling back to int32")
            dtype = "int32"

    if dtype == "uint16":
        tokens = tokens.to(torch.uint16)
        element_size = 2
    else:
        tokens = tokens.to(torch.int32)
        element_size = 4

    with open(output_path, "wb") as f:
        f.write(tokens.numpy().tobytes())

    print(f"[Pretokenize] Saved {len(all_tokens):,} tokens to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
