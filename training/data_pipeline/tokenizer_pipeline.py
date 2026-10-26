# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# tokenizer_pipeline.py - 大规模分布式分词管线
# 支持多进程并行分词、序列打包、多格式输出（arrow/parquet/jsonl/mmap）、token统计

import os
import json
import math
import logging
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("qingxiaotuan.tokenizer_pipeline")


@dataclass
class TokenizationStats:
    """青小团分词统计"""
    total_documents: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    packed_sequences: int = 0
    token_lengths: Dict[str, float] = field(default_factory=dict)
    per_dataset_tokens: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_tokens_per_doc(self) -> float:
        if self.total_documents > 0:
            return self.total_tokens / self.total_documents
        return 0.0

    @property
    def avg_tokens_per_seq(self) -> float:
        if self.total_sequences > 0:
            return self.total_tokens / self.total_sequences
        return 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 分词管线 - 统计报告",
            "=" * 60,
            f"总文档数:            {self.total_documents:,}",
            f"总Token数:           {self.total_tokens:,}",
            f"总序列数:            {self.total_sequences:,}",
            f"打包序列数:          {self.packed_sequences:,}",
            f"每文档平均Token:     {self.avg_tokens_per_doc:.2f}",
            f"每序列平均Token:     {self.avg_tokens_per_seq:.2f}",
        ]
        if self.token_lengths:
            lines.extend([
                "-" * 60,
                "Token长度分布:",
                f"  最小:   {self.token_lengths.get('min', 0):,}",
                f"  最大:   {self.token_lengths.get('max', 0):,}",
                f"  平均:   {self.token_lengths.get('mean', 0):.2f}",
                f"  P50:    {self.token_lengths.get('p50', 0):,}",
                f"  P90:    {self.token_lengths.get('p90', 0):,}",
                f"  P99:    {self.token_lengths.get('p99', 0):,}",
            ])
        lines.append("=" * 60)
        return "\n".join(lines)


class SequencePacker:
    """青小团序列打包器 - 将短序列打包成最大长度序列以减少padding浪费"""

    def __init__(self, max_length: int = 524288, max_sequences: int = 8):
        self.max_length = max_length
        self.max_sequences = max_sequences

    def pack(
        self, token_ids: List[int], attention_mask: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        if len(token_ids) <= self.max_length:
            seq = {
                "input_ids": token_ids,
                "attention_mask": attention_mask or [1] * len(token_ids),
                "labels": token_ids.copy(),
            }
            return [seq]

        packed = []
        current_tokens: List[int] = []
        current_mask: List[int] = []
        current_labels: List[int] = []
        seq_count = 0

        sep_id = token_ids[-1] if len(token_ids) > 0 else 0
        start = 0

        while start < len(token_ids) and seq_count < self.max_sequences:
            remaining = self.max_length - len(current_tokens)
            if remaining <= 0:
                break

            chunk = token_ids[start:start + remaining]
            chunk_mask = [1] * len(chunk)
            chunk_labels = chunk.copy()

            if current_tokens:
                current_tokens.append(sep_id)
                current_mask.append(1)
                current_labels.append(-100)

            current_tokens.extend(chunk)
            current_mask.extend(chunk_mask)
            current_labels.extend(chunk_labels)
            seq_count += 1
            start += len(chunk)

        if current_tokens:
            pad_len = self.max_length - len(current_tokens)
            current_tokens.extend([0] * pad_len)
            current_mask.extend([0] * pad_len)
            current_labels.extend([-100] * pad_len)

            packed.append({
                "input_ids": current_tokens,
                "attention_mask": current_mask,
                "labels": current_labels,
            })

        return packed

    def pack_batch(
        self, batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for item in batch:
            token_ids = item.get("input_ids", [])
            attention_mask = item.get("attention_mask", None)
            packed_seqs = self.pack(token_ids, attention_mask)
            results.extend(packed_seqs)
        return results


class MemoryMappedWriter:
    """青小团内存映射文件写入器 - 用于零拷贝大文件写入"""

    def __init__(self, filepath: str, dtype: np.dtype = np.int32):
        self.filepath = filepath
        self.dtype = dtype
        self._mmap = None
        self._current_offset = 0

    def open(self, reserve_size: int = 10_737_418_240):
        self._file = open(self.filepath, "wb+")
        self._file.truncate(reserve_size)
        self._mmap = self._file.buffer
        self._current_offset = 0

    def append(self, data: np.ndarray):
        if self._mmap is None:
            self.open()
        bytes_data = data.astype(self.dtype).tobytes()
        self._mmap.write(bytes_data)
        self._current_offset += len(data)

    def close(self):
        if self._mmap:
            self._mmap.flush()
            self._file.truncate(self._current_offset * self.dtype.itemsize)
            self._file.close()
            self._mmap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class TokenizerPipeline:
    """青小团分词管线 - 批量文本分词、并行处理、多格式输出"""

    def __init__(
        self,
        tokenizer_path: str,
        tokenizer_type: str = "huggingface",
        max_length: int = 524288,
        truncation: bool = True,
        padding: bool = False,
        add_special_tokens: bool = True,
        sequence_packing: bool = True,
        packing_max_sequences: int = 8,
        num_workers: int = 32,
        output_dir: str = "./output/tokenized",
        compression: str = "zstd",
        shuffle_before_save: bool = True,
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.add_special_tokens = add_special_tokens
        self.sequence_packing = sequence_packing
        self.packing_max_sequences = packing_max_sequences
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.shuffle_before_save = shuffle_before_save

        self._tokenizer = None
        self.packer = SequencePacker(
            max_length=max_length,
            max_sequences=packing_max_sequences,
        )
        self.stats = TokenizationStats()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"青小团分词管线初始化: type={tokenizer_type}, max_len={max_length}, "
            f"workers={num_workers}, packing={sequence_packing}"
        )

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        if self.tokenizer_type == "huggingface":
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    use_fast=True,
                    trust_remote_code=True,
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info(f"青小团加载 HuggingFace tokenizer: {self.tokenizer_path}")
            except Exception as e:
                logger.error(f"青小团加载 tokenizer 失败: {e}")
                raise

        elif self.tokenizer_type == "sentencepiece":
            try:
                import sentencepiece as spm
                self._tokenizer = spm.SentencePieceProcessor()
                self._tokenizer.load(self.tokenizer_path)
                logger.info(f"青小团加载 SentencePiece tokenizer: {self.tokenizer_path}")
            except Exception as e:
                logger.error(f"青小团加载 SentencePiece 失败: {e}")
                raise

        elif self.tokenizer_type == "bpe":
            try:
                from tokenizers import Tokenizer as HFTokenizer
                self._tokenizer = HFTokenizer.from_file(self.tokenizer_path)
                logger.info(f"青小团加载 BPE tokenizer: {self.tokenizer_path}")
            except Exception as e:
                logger.error(f"青小团加载 BPE tokenizer 失败: {e}")
                raise

        else:
            raise ValueError(f"青小团不支持的 tokenizer 类型: {self.tokenizer_type}")

        return self._tokenizer

    def tokenize_text(self, text: str) -> Dict[str, Any]:
        tokenizer = self._load_tokenizer()

        if self.tokenizer_type == "huggingface":
            encoded = tokenizer(
                text,
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                add_special_tokens=self.add_special_tokens,
                return_tensors=None,
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded.get("attention_mask", [1] * len(encoded["input_ids"])),
            }

        elif self.tokenizer_type == "sentencepiece":
            tokens = tokenizer.encode(text)
            if self.truncation and len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            return {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens),
            }

        elif self.tokenizer_type == "bpe":
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            if self.truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }

        return {"input_ids": [], "attention_mask": []}

    def tokenize_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        logger.info(f"青小团开始批量分词: {len(texts)} 文档, workers={self.num_workers}")

        if self.num_workers <= 1:
            results = []
            for text in tqdm(texts, desc="青小团分词", ncols=80):
                results.append(self.tokenize_text(text))
            return results

        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_map = {}
            for i, text in enumerate(texts):
                future = executor.submit(self.tokenize_text, text)
                future_map[future] = i

            for future in tqdm(
                as_completed(future_map), total=len(future_map),
                desc="青小团分词", ncols=80
            ):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"青小团分词文档 {idx} 失败: {e}")
                    results[idx] = {"input_ids": [], "attention_mask": []}

        return results

    def tokenize_stream(
        self, text_stream: Iterator[str], batch_size: int = 1000
    ) -> Iterator[Dict[str, Any]]:
        batch = []
        for text in text_stream:
            batch.append(text)
            if len(batch) >= batch_size:
                for result in self.tokenize_batch(batch):
                    yield result
                batch = []

        if batch:
            for result in self.tokenize_batch(batch):
                yield result

    def apply_sequence_packing(
        self, tokenized_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.sequence_packing:
            return tokenized_batch

        packed = self.packer.pack_batch(tokenized_batch)
        self.stats.packed_sequences += len(packed)
        return packed

    def process_and_save(
        self,
        documents: List[Dict[str, Any]],
        output_format: str = "arrow",
        dataset_name: str = "default",
    ) -> TokenizationStats:
        logger.info(
            f"青小团开始处理 {len(documents)} 文档, 输出格式={output_format}"
        )

        texts = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            if isinstance(text, str) and text.strip():
                texts.append(text)

        tokenized = self.tokenize_batch(texts)
        self.stats.total_documents = len(texts)

        for t in tokenized:
            self.stats.total_tokens += len(t.get("input_ids", []))

        if self.sequence_packing:
            tokenized = self.apply_sequence_packing(tokenized)

        self.stats.total_sequences = len(tokenized)
        self._compute_token_stats(tokenized)

        self._save_output(tokenized, output_format, dataset_name)
        logger.info(f"\n{self.stats.summary()}")

        return self.stats

    def _compute_token_stats(self, tokenized: List[Dict[str, Any]]):
        lengths = [len(t.get("input_ids", [])) for t in tokenized if t.get("input_ids")]
        if lengths:
            sorted_lens = sorted(lengths)
            n = len(sorted_lens)
            self.stats.token_lengths = {
                "min": sorted_lens[0],
                "max": sorted_lens[-1],
                "mean": sum(lengths) / n,
                "p50": sorted_lens[int(n * 0.50)],
                "p90": sorted_lens[int(n * 0.90)],
                "p99": sorted_lens[int(n * 0.99)],
            }

    def _save_output(
        self,
        tokenized: List[Dict[str, Any]],
        output_format: str,
        dataset_name: str,
    ):
        if self.shuffle_before_save:
            np.random.shuffle(tokenized)

        if output_format == "arrow":
            self._save_as_arrow(tokenized, dataset_name)
        elif output_format == "parquet":
            self._save_as_parquet(tokenized, dataset_name)
        elif output_format == "jsonl":
            self._save_as_jsonl(tokenized, dataset_name)
        elif output_format == "mmap":
            self._save_as_mmap(tokenized, dataset_name)
        else:
            raise ValueError(f"青小团不支持的输出格式: {output_format}")

    def _save_as_arrow(self, tokenized: List[Dict[str, Any]], dataset_name: str):
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError:
            logger.error("青小团: 需要安装 pyarrow: pip install pyarrow")
            raise

        output_file = self.output_dir / f"{dataset_name}.arrow"
        logger.info(f"青小团保存 Arrow 格式: {output_file}")

        input_ids_list = [t.get("input_ids", []) for t in tokenized]
        attention_mask_list = [t.get("attention_mask", []) for t in tokenized]
        labels_list = [t.get("labels", t.get("input_ids", [])) for t in tokenized]

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(input_ids_list, type=pa.list_(pa.int32())),
                pa.array(attention_mask_list, type=pa.list_(pa.int32())),
                pa.array(labels_list, type=pa.list_(pa.int32())),
            ],
            names=["input_ids", "attention_mask", "labels"],
        )

        table = pa.Table.from_batches([batch])

        if self.compression == "zstd":
            with pa.OSFile(str(output_file), "wb") as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
        else:
            with pa.OSFile(str(output_file), "wb") as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)

        logger.info(f"青小团 Arrow 保存完成: {output_file}")

    def _save_as_parquet(self, tokenized: List[Dict[str, Any]], dataset_name: str):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("青小团: 需要安装 pyarrow: pip install pyarrow")
            raise

        output_file = self.output_dir / f"{dataset_name}.parquet"
        logger.info(f"青小团保存 Parquet 格式: {output_file}")

        input_ids_list = [t.get("input_ids", []) for t in tokenized]
        attention_mask_list = [t.get("attention_mask", []) for t in tokenized]
        labels_list = [t.get("labels", t.get("input_ids", [])) for t in tokenized]

        table = pa.table({
            "input_ids": pa.array(input_ids_list, type=pa.list_(pa.int32())),
            "attention_mask": pa.array(attention_mask_list, type=pa.list_(pa.int32())),
            "labels": pa.array(labels_list, type=pa.list_(pa.int32())),
        })

        compression = self.compression if self.compression != "none" else None
        pq.write_table(table, str(output_file), compression=compression)
        logger.info(f"青小团 Parquet 保存完成: {output_file}")

    def _save_as_jsonl(self, tokenized: List[Dict[str, Any]], dataset_name: str):
        output_file = self.output_dir / f"{dataset_name}.jsonl"
        logger.info(f"青小团保存 JSONL 格式: {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for item in tokenized:
                record = {
                    "input_ids": item.get("input_ids", []),
                    "attention_mask": item.get("attention_mask", []),
                    "labels": item.get("labels", item.get("input_ids", [])),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"青小团 JSONL 保存完成: {output_file}")

    def _save_as_mmap(self, tokenized: List[Dict[str, Any]], dataset_name: str):
        output_prefix = self.output_dir / dataset_name
        logger.info(f"青小团保存 MMap 格式: {output_prefix}")

        ids_path = str(output_prefix) + "_input_ids.bin"
        mask_path = str(output_prefix) + "_attention_mask.bin"
        labels_path = str(output_prefix) + "_labels.bin"
        meta_path = str(output_prefix) + "_meta.json"

        all_ids = np.concatenate([t.get("input_ids", []) for t in tokenized]).astype(np.int32)
        all_masks = np.concatenate([t.get("attention_mask", []) for t in tokenized]).astype(np.int32)
        all_labels = np.concatenate([t.get("labels", t.get("input_ids", [])) for t in tokenized]).astype(np.int32)

        seq_lengths = [len(t.get("input_ids", [])) for t in tokenized]

        with MemoryMappedWriter(ids_path) as writer:
            writer.append(all_ids)
        with MemoryMappedWriter(mask_path) as writer:
            writer.append(all_masks)
        with MemoryMappedWriter(labels_path) as writer:
            writer.append(all_labels)

        meta = {
            "dataset": dataset_name,
            "num_sequences": len(tokenized),
            "total_tokens": int(all_ids.shape[0]),
            "sequence_lengths": seq_lengths,
            "max_length": self.max_length,
            "dtype": "int32",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info(f"青小团 MMap 保存完成: {ids_path}, {meta_path}")

    def process_multiformat(
        self,
        documents: List[Dict[str, Any]],
        formats: List[str] = None,
        dataset_name: str = "default",
    ) -> TokenizationStats:
        if formats is None:
            formats = ["arrow", "parquet", "jsonl", "mmap"]

        texts = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            if isinstance(text, str) and text.strip():
                texts.append(text)

        tokenized = self.tokenize_batch(texts)
        self.stats.total_documents = len(texts)

        for t in tokenized:
            self.stats.total_tokens += len(t.get("input_ids", []))

        if self.sequence_packing:
            tokenized = self.apply_sequence_packing(tokenized)

        self.stats.total_sequences = len(tokenized)
        self._compute_token_stats(tokenized)

        for fmt in formats:
            self._save_output(tokenized, fmt, dataset_name)

        logger.info(f"\n{self.stats.summary()}")
        return self.stats

    def get_tokenizer_info(self) -> Dict[str, Any]:
        tokenizer = self._load_tokenizer()
        info = {
            "type": self.tokenizer_type,
            "path": self.tokenizer_path,
            "max_length": self.max_length,
        }
        if hasattr(tokenizer, "vocab_size"):
            info["vocab_size"] = tokenizer.vocab_size
        if hasattr(tokenizer, "model_max_length"):
            info["model_max_length"] = tokenizer.model_max_length
        return info

    def report_summary(self) -> str:
        return self.stats.summary()
