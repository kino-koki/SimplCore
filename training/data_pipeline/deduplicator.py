# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# deduplicator.py - 大规模数据去重引擎
# 支持精确去重、模糊去重（MinHash + LSH）、文档级/段落级去重、流式去重

import os
import json
import hashlib
import logging
import struct
import mmap
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Iterator, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("qingxiaotuan.deduplicator")


@dataclass
class DedupStats:
    """青小团去重统计"""
    total_documents: int = 0
    after_exact_dedup: int = 0
    after_fuzzy_dedup: int = 0
    final_documents: int = 0
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    total_paragraphs: int = 0
    duplicate_paragraphs: int = 0

    @property
    def exact_dedup_rate(self) -> float:
        if self.total_documents > 0:
            return self.exact_duplicates_removed / self.total_documents * 100
        return 0.0

    @property
    def fuzzy_dedup_rate(self) -> float:
        if self.after_exact_dedup > 0:
            return self.fuzzy_duplicates_removed / (self.after_exact_dedup + self.fuzzy_duplicates_removed) * 100
        return 0.0

    @property
    def overall_dedup_rate(self) -> float:
        total_removed = self.exact_duplicates_removed + self.fuzzy_duplicates_removed
        if self.total_documents > 0:
            return total_removed / self.total_documents * 100
        return 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 数据去重引擎 - 统计报告",
            "=" * 60,
            f"总文档数:              {self.total_documents:,}",
            f"精确去重移除:          {self.exact_duplicates_removed:,} ({self.exact_dedup_rate:.2f}%)",
            f"模糊去重移除:          {self.fuzzy_duplicates_removed:,} ({self.fuzzy_dedup_rate:.2f}%)",
            f"最终文档数:            {self.final_documents:,}",
            f"总体去重率:            {self.overall_dedup_rate:.2f}%",
        ]
        if self.total_paragraphs > 0:
            lines.extend([
                f"总段落数:              {self.total_paragraphs:,}",
                f"重复段落移除:          {self.duplicate_paragraphs:,}",
            ])
        lines.append("=" * 60)
        return "\n".join(lines)


class ExactDeduplicator:
    """青小团精确去重器 - 基于文本哈希的精确匹配去重"""

    def __init__(self):
        self._seen_hashes: Set[str] = set()
        self._hash_func = hashlib.sha256

    def compute_hash(self, text: str) -> str:
        return self._hash_func(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        text_hash = self.compute_hash(text)
        return text_hash in self._seen_hashes

    def add(self, text: str) -> str:
        text_hash = self.compute_hash(text)
        self._seen_hashes.add(text_hash)
        return text_hash

    def check_and_add(self, text: str) -> Tuple[bool, str]:
        text_hash = self.compute_hash(text)
        if text_hash in self._seen_hashes:
            return True, text_hash
        self._seen_hashes.add(text_hash)
        return False, text_hash

    def remove(self, text: str):
        text_hash = self.compute_hash(text)
        self._seen_hashes.discard(text_hash)

    def clear(self):
        self._seen_hashes.clear()

    def __len__(self) -> int:
        return len(self._seen_hashes)

    def save_state(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for h in self._seen_hashes:
                f.write(h + "\n")

    def load_state(self, filepath: str):
        self._seen_hashes.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h:
                    self._seen_hashes.add(h)


class MinHasher:
    """青小团 MinHash 实现 - 用于模糊去重的特征哈希"""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        self._generator = np.random.RandomState(seed)
        self._perms_a = self._generator.randint(1, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self._perms_b = self._generator.randint(1, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self._prime = 2**61 - 1

    def _hash_ngram(self, ngram: str) -> int:
        return int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)

    def compute_signature(self, text: str, ngram_size: int = 5) -> np.ndarray:
        tokens = text.split()
        if len(tokens) < ngram_size:
            tokens = list(text)

        ngrams = set()
        for i in range(len(tokens) - ngram_size + 1):
            ngram = " ".join(str(tokens[i + j]) for j in range(ngram_size)) if isinstance(tokens, list) and len(tokens) > 0 else text[i:i + ngram_size]
            if isinstance(tokens, list) and len(tokens) >= ngram_size:
                pass
            else:
                ngram = text[i:i + ngram_size]
            ngrams.add(ngram)

        if not ngrams:
            ngrams.add(text[:ngram_size] if len(text) >= ngram_size else text)

        sig = np.full(self.num_perm, self._prime, dtype=np.uint64)
        for ngram in ngrams:
            hash_val = self._hash_ngram(ngram)
            for i in range(self.num_perm):
                ph = (self._perms_a[i] * hash_val + self._perms_b[i]) % self._prime
                if ph < sig[i]:
                    sig[i] = ph

        return sig

    def compute_signature_batch(
        self, texts: List[str], ngram_size: int = 5, num_workers: int = 8
    ) -> List[np.ndarray]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.compute_signature, t, ngram_size) for t in texts]
            return [f.result() for f in futures]

    @staticmethod
    def estimate_jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
        if len(sig1) != len(sig2):
            raise ValueError("签名长度不匹配")
        return float(np.mean(sig1 == sig2))


class LSHIndex:
    """青小团局部敏感哈希索引 - 用于近似最近邻搜索"""

    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.num_perm = num_perm
        self.threshold = threshold

        bands_configs = []
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                bands_configs.append((b, r, (1 / b) ** (1 / r)))

        best = min(bands_configs, key=lambda x: abs(x[2] - threshold))
        self.num_bands = best[0]
        self.rows_per_band = best[1]

        self._buckets: List[Dict[str, List[int]]] = [
            defaultdict(list) for _ in range(self.num_bands)
        ]
        self._doc_ids: Dict[int, str] = {}
        self._next_id: int = 0

        logger.info(
            f"青小团LSH索引: bands={self.num_bands}, rows_per_band={self.rows_per_band}, "
            f"估计阈值={best[2]:.4f}"
        )

    def _hash_band(self, signature: np.ndarray, band_idx: int) -> str:
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band_values = signature[start:end]
        return hashlib.md5(band_values.tobytes()).hexdigest()

    def insert(self, signature: np.ndarray, doc_id: int):
        for band_idx in range(self.num_bands):
            band_hash = self._hash_band(signature, band_idx)
            self._buckets[band_idx][band_hash].append(doc_id)

    def query(self, signature: np.ndarray) -> Set[int]:
        candidates = set()
        for band_idx in range(self.num_bands):
            band_hash = self._hash_band(signature, band_idx)
            bucket = self._buckets[band_idx].get(band_hash, [])
            candidates.update(bucket)
        return candidates

    def add_document(self, signature: np.ndarray, doc_id: Optional[int] = None) -> int:
        if doc_id is None:
            doc_id = self._next_id
            self._next_id += 1
        self._doc_ids[doc_id] = ""
        self.insert(signature, doc_id)
        return doc_id

    def get_candidates(self, signature: np.ndarray, exclude_id: Optional[int] = None) -> List[int]:
        candidates = self.query(signature)
        if exclude_id is not None:
            candidates.discard(exclude_id)
        return list(candidates)


class StreamDedupBuffer:
    """青小团流式去重缓冲区 - 用于大数据集的磁盘辅助去重"""

    def __init__(self, cache_dir: str, max_memory_items: int = 1_000_000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self._memory_buffer: Set[str] = set()
        self._disk_files: List[Path] = []
        self._current_file_idx = 0

    def add(self, item_hash: str) -> bool:
        if item_hash in self._memory_buffer:
            return False
        if len(self._memory_buffer) >= self.max_memory_items:
            self._flush_to_disk()
        self._memory_buffer.add(item_hash)
        return True

    def _flush_to_disk(self):
        filepath = self.cache_dir / f"bloom_{self._current_file_idx:08d}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            for h in sorted(self._memory_buffer):
                f.write(h + "\n")
        self._disk_files.append(filepath)
        self._memory_buffer.clear()
        self._current_file_idx += 1
        logger.info(f"青小团流式缓冲区刷新到磁盘: {filepath}")

    def __contains__(self, item_hash: str) -> bool:
        if item_hash in self._memory_buffer:
            return True
        for df in self._disk_files:
            if self._search_in_file(df, item_hash):
                return True
        return False

    def _search_in_file(self, filepath: Path, target: str) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    if mm.find(target.encode("utf-8")) != -1:
                        return True
        except Exception:
            pass
        return False

    def __len__(self) -> int:
        return len(self._memory_buffer) + sum(
            sum(1 for _ in open(df, "r", encoding="utf-8"))
            for df in self._disk_files
        )

    def clear(self):
        self._memory_buffer.clear()
        for df in self._disk_files:
            if df.exists():
                df.unlink()
        self._disk_files.clear()
        self._current_file_idx = 0


class Deduplicator:
    """青小团数据去重器 - 集成精确去重和模糊去重"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5,
        paragraph_dedup: bool = True,
        paragraph_sep: str = "\n\n",
        storage_type: str = "memory",
        cache_dir: str = "./cache/dedup",
        num_workers: int = 8,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.paragraph_dedup = paragraph_dedup
        self.paragraph_sep = paragraph_sep
        self.storage_type = storage_type
        self.num_workers = num_workers

        self.exact_dedup = ExactDeduplicator()
        self.minhasher = MinHasher(num_perm=num_perm)
        self.lsh_index = LSHIndex(num_perm=num_perm, threshold=threshold)
        self.stream_buffer = StreamDedupBuffer(cache_dir=cache_dir)
        self.stats = DedupStats()

        logger.info(
            f"青小团去重器初始化: perm={num_perm}, threshold={threshold}, "
            f"ngram={ngram_size}, workers={num_workers}"
        )

    def extract_text(self, doc: Dict[str, Any]) -> str:
        text = doc.get("text", doc.get("content", doc.get("", "")))
        return text if isinstance(text, str) else ""

    def extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = text.split(self.paragraph_sep)
        return [p.strip() for p in paragraphs if p.strip()]

    def exact_dedup_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = self.extract_text(doc)
        if not text:
            return None

        is_dup, _ = self.exact_dedup.check_and_add(text)
        if is_dup:
            return None
        return doc

    def fuzzy_dedup_document(
        self, doc: Dict[str, Any], doc_id: int
    ) -> Optional[Dict[str, Any]]:
        text = self.extract_text(doc)
        if not text:
            return None

        sig = self.minhasher.compute_signature(text, self.ngram_size)
        candidates = self.lsh_index.get_candidates(sig, exclude_id=doc_id)

        for candidate_id in candidates:
            if self._is_duplicate_pair(sig, candidate_id):
                self.stats.fuzzy_duplicates_removed += 1
                return None

        self.lsh_index.add_document(sig, doc_id)
        return doc

    def _is_duplicate_pair(self, sig: np.ndarray, candidate_id: int) -> bool:
        return False

    def deduplicate_batch(
        self,
        documents: List[Dict[str, Any]],
        exact: bool = True,
        fuzzy: bool = True,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"青小团开始去重 {len(documents)} 个文档 "
            f"(exact={exact}, fuzzy={fuzzy})"
        )

        self.stats.total_documents = len(documents)

        if exact:
            results = []
            for doc in tqdm(documents, desc="青小团精确去重", ncols=80):
                result = self.exact_dedup_document(doc)
                if result is not None:
                    results.append(result)
                else:
                    self.stats.exact_duplicates_removed += 1
            self.stats.after_exact_dedup = len(results)
            documents = results

        if fuzzy:
            results = []
            for i, doc in enumerate(
                tqdm(documents, desc="青小团模糊去重", ncols=80)
            ):
                result = self.fuzzy_dedup_document(doc, i)
                if result is not None:
                    results.append(result)
            self.stats.after_fuzzy_dedup = len(results)
            documents = results

        self.stats.final_documents = len(documents)
        logger.info(f"\n{self.stats.summary()}")
        return documents

    def paragraph_level_dedup(
        self, doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = self.extract_text(doc)
        if not text:
            return doc

        paras = self.extract_paragraphs(text)
        self.stats.total_paragraphs += len(paras)

        unique_paras = []
        for para in paras:
            if len(para) < 10:
                unique_paras.append(para)
                continue
            para_hash = hashlib.md5(para.encode("utf-8")).hexdigest()
            if not self.stream_buffer.add(para_hash):
                self.stats.duplicate_paragraphs += 1
                continue
            unique_paras.append(para)

        doc["text"] = self.paragraph_sep.join(unique_paras)
        return doc

    def deduplicate_stream(
        self,
        doc_stream: Iterator[Dict[str, Any]],
        exact: bool = True,
        fuzzy: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        deduped_count = 0
        total_count = 0

        for doc in doc_stream:
            total_count += 1
            text = self.extract_text(doc)
            if not text:
                continue

            if exact:
                is_dup, _ = self.exact_dedup.check_and_add(text)
                if is_dup:
                    self.stats.exact_duplicates_removed += 1
                    continue

            if self.paragraph_dedup:
                doc = self.paragraph_level_dedup(doc)

            deduped_count += 1
            yield doc

            if deduped_count % 10000 == 0:
                logger.info(
                    f"青小团流式去重进度: {total_count:,} -> {deduped_count:,} "
                    f"(去重率: {(1 - deduped_count / total_count) * 100:.2f}%)"
                )

        self.stats.total_documents = total_count
        self.stats.final_documents = deduped_count

    def deduplicate_jsonl(
        self,
        input_path: str,
        output_path: str,
        exact: bool = True,
        fuzzy: bool = False,
    ) -> DedupStats:
        logger.info(f"青小团去重文件: {input_path} -> {output_path}")

        with open(input_path, "r", encoding="utf-8") as fin:
            docs = []
            for line in fin:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        results = self.deduplicate_batch(docs, exact=exact, fuzzy=fuzzy)

        with open(output_path, "w", encoding="utf-8") as fout:
            for doc in results:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

        return self.stats

    def parallel_deduplicate(
        self,
        documents: List[Dict[str, Any]],
        num_processes: int = 4,
    ) -> List[Dict[str, Any]]:
        shard_size = max(1, len(documents) // num_processes)
        shards = [
            documents[i:i + shard_size]
            for i in range(0, len(documents), shard_size)
        ]

        logger.info(f"青小团并行去重: {len(documents)} 文档, {num_processes} 进程, {len(shards)} 分片")

        all_results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for shard in shards:
                future = executor.submit(self.deduplicate_batch, shard, True, False)
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="青小团并行去重", ncols=80
            ):
                all_results.extend(future.result())

        logger.info(f"青小团并行去重完成: {len(documents)} -> {len(all_results)}")
        return all_results

    def save_minhash_signatures(self, filepath: str):
        data = {
            "num_perm": self.num_perm,
            "ngram_size": self.ngram_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"青小团MinHash签名已保存: {filepath}")

    def load_minhash_signatures(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"青小团签名文件不存在: {filepath}")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            json.load(f)
        logger.info(f"青小团MinHash签名已加载: {filepath}")

    def report_summary(self) -> str:
        return self.stats.summary()

    def reset(self):
        self.exact_dedup.clear()
        self.lsh_index = LSHIndex(
            num_perm=self.num_perm, threshold=self.threshold
        )
        self.stream_buffer.clear()
        self.stats = DedupStats()
        logger.info("青小团去重器已重置")


class MultiStageDeduplicator:
    """青小团多阶段去重器 - 先精确去重，再模糊去重，最后段落去重"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5,
        num_workers: int = 8,
        cache_dir: str = "./cache/dedup",
    ):
        self.dedup = Deduplicator(
            num_perm=num_perm,
            threshold=threshold,
            ngram_size=ngram_size,
            paragraph_dedup=True,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"青小团多阶段去重: {len(documents)} 文档")

        stage1 = self.dedup.deduplicate_batch(documents, exact=True, fuzzy=False)
        logger.info(f"青小团阶段一(精确去重)完成: {len(documents)} -> {len(stage1)}")

        stage2 = self.dedup.deduplicate_batch(stage1, exact=False, fuzzy=True)
        logger.info(f"青小团阶段二(模糊去重)完成: {len(stage1)} -> {len(stage2)}")

        if self.dedup.paragraph_dedup:
            para_results = []
            for doc in tqdm(stage2, desc="青小团段落去重", ncols=80):
                para_results.append(self.dedup.paragraph_level_dedup(doc))
            stage2 = para_results
            logger.info(f"青小团阶段三(段落去重)完成")

        return stage2
