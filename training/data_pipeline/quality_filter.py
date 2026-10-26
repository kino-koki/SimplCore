# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# quality_filter.py - 基于启发式的数据质量过滤引擎
# 支持语言检测、困惑度过滤、重复n-gram、标点符号、词长度、特殊字符、URL、停用词等多维过滤

import os
import re
import json
import math
import logging
import threading
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("qingxiaotuan.quality_filter")


@dataclass
class FilterStats:
    """青小团质量过滤统计"""
    total_documents: int = 0
    passed: int = 0
    filtered_out: int = 0
    filter_reasons: Counter = field(default_factory=Counter)
    per_filter_stats: Dict[str, Dict] = field(default_factory=dict)
    total_input_bytes: int = 0
    total_output_bytes: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_documents > 0:
            return self.passed / self.total_documents * 100
        return 0.0

    @property
    def filter_rate(self) -> float:
        if self.total_documents > 0:
            return self.filtered_out / self.total_documents * 100
        return 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 质量过滤引擎 - 统计报告",
            "=" * 60,
            f"总文档数:          {self.total_documents:,}",
            f"通过:              {self.passed:,} ({self.pass_rate:.2f}%)",
            f"过滤掉:            {self.filtered_out:,} ({self.filter_rate:.2f}%)",
            "-" * 60,
            "过滤原因分布:",
        ]
        for reason, count in self.filter_reasons.most_common():
            pct = count / self.total_documents * 100 if self.total_documents > 0 else 0
            lines.append(f"  {reason}: {count:,} ({pct:.2f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class FilterConfig:
    """青小团过滤配置"""

    # 语言检测
    lang_detect_enabled: bool = True
    target_languages: List[str] = field(default_factory=lambda: ["zh", "en"])

    # 困惑度
    perplexity_enabled: bool = True
    perplexity_max_threshold: float = 1000.0

    # n-gram 重复
    ngram_repeat_enabled: bool = True
    ngram_n: int = 3
    ngram_max_ratio: float = 0.30

    # 标点符号
    punctuation_enabled: bool = True
    punctuation_min_ratio: float = 0.05
    punctuation_max_ratio: float = 0.60

    # 平均词长度
    word_length_enabled: bool = True
    word_length_min_avg: float = 3.0
    word_length_max_avg: float = 50.0

    # 特殊字符
    special_chars_enabled: bool = True
    special_chars_max_ratio: float = 0.20
    special_chars: str = "�\x00\x01\x02\x1a"

    # URL
    url_enabled: bool = True
    url_max_ratio: float = 0.40

    # 停用词
    stopword_enabled: bool = True
    stopword_min_ratio: float = 0.02

    # 文本长度
    min_text_length: int = 50
    max_text_length: int = 1_000_000

    def to_dict(self) -> Dict:
        return {k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
                for k, v in self.__dataclass_fields__.items()}


class TextQualityHeuristics:
    """青小团文本质量启发式评估器"""

    URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"(?:/[-\w$.+!*\'(),;:@&=?/~#%]*)?",
        re.IGNORECASE
    )

    PUNCTUATION_SET = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，、；：？！…—·「」『』【】（）《》")
    CHINESE_PUNCTUATION_SET = set("。，、；：？！…—·「」『』【】（）《》""''")
    SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\U00010000-\U0010ffff]")

    ZH_EN_STOPWORDS = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "can", "could", "shall", "should", "may", "might",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "as", "into", "through", "during", "before", "after",
        "and", "but", "or", "nor", "not", "so", "yet", "if",
        "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them",
    }

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

    def compute_ngram_repeat_ratio(self, text: str) -> float:
        """计算重复 n-gram 的比例"""
        if not text or len(text) < self.config.ngram_n + 1:
            return 0.0
        n = self.config.ngram_n
        tokens = text.split()
        if len(tokens) < n + 1:
            tokens = list(text)
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        unique_ngrams = set(ngrams)
        repeated_count = len(ngrams) - len(unique_ngrams)
        return repeated_count / len(ngrams)

    def compute_punctuation_ratio(self, text: str) -> float:
        """计算标点符号比例"""
        if not text:
            return 0.0
        punct_count = sum(1 for ch in text if ch in self.PUNCTUATION_SET or ch in self.CHINESE_PUNCTUATION_SET)
        return punct_count / len(text) if len(text) > 0 else 0.0

    def compute_average_word_length(self, text: str) -> float:
        """计算平均词长度"""
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        avg_len = sum(len(w) for w in words) / len(words)
        return avg_len

    def compute_special_char_ratio(self, text: str) -> float:
        """计算特殊字符比例"""
        if not text:
            return 0.0
        special_count = sum(1 for ch in text if ch in self.config.special_chars)
        matches = self.SPECIAL_CHAR_PATTERN.findall(text)
        special_count += len(matches)
        return special_count / len(text)

    def compute_url_ratio(self, text: str) -> float:
        """计算 URL 比例"""
        if not text:
            return 0.0
        urls = self.URL_PATTERN.findall(text)
        url_chars = sum(len(url) for url in urls)
        return url_chars / len(text)

    def compute_stopword_ratio(self, text: str) -> float:
        """计算停用词比例"""
        if not text:
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0
        stopword_count = sum(1 for w in words if w in self.ZH_EN_STOPWORDS)
        return stopword_count / len(words)

    def compute_all_metrics(self, text: str) -> Dict[str, float]:
        """计算所有质量指标"""
        return {
            "length": len(text),
            "ngram_repeat_ratio": self.compute_ngram_repeat_ratio(text),
            "punctuation_ratio": self.compute_punctuation_ratio(text),
            "avg_word_length": self.compute_average_word_length(text),
            "special_char_ratio": self.compute_special_char_ratio(text),
            "url_ratio": self.compute_url_ratio(text),
            "stopword_ratio": self.compute_stopword_ratio(text),
        }

    def check_text_length(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查文本长度"""
        length = len(text)
        if length < self.config.min_text_length:
            return False, f"text_too_short: {length} < {self.config.min_text_length}"
        if length > self.config.max_text_length:
            return False, f"text_too_long: {length} > {self.config.max_text_length}"
        return True, None

    def check_ngram_repeat(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查 n-gram 重复比例"""
        if not self.config.ngram_repeat_enabled:
            return True, None
        ratio = self.compute_ngram_repeat_ratio(text)
        if ratio > self.config.ngram_max_ratio:
            return False, f"high_ngram_repeat: {ratio:.4f} > {self.config.ngram_max_ratio}"
        return True, None

    def check_punctuation(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查标点符号比例"""
        if not self.config.punctuation_enabled:
            return True, None
        ratio = self.compute_punctuation_ratio(text)
        if ratio < self.config.punctuation_min_ratio:
            return False, f"low_punctuation: {ratio:.4f} < {self.config.punctuation_min_ratio}"
        if ratio > self.config.punctuation_max_ratio:
            return False, f"high_punctuation: {ratio:.4f} > {self.config.punctuation_max_ratio}"
        return True, None

    def check_word_length(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查平均词长度"""
        if not self.config.word_length_enabled:
            return True, None
        avg_len = self.compute_average_word_length(text)
        if avg_len < self.config.word_length_min_avg:
            return False, f"short_avg_word_length: {avg_len:.2f} < {self.config.word_length_min_avg}"
        if avg_len > self.config.word_length_max_avg:
            return False, f"long_avg_word_length: {avg_len:.2f} > {self.config.word_length_max_avg}"
        return True, None

    def check_special_chars(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查特殊字符比例"""
        if not self.config.special_chars_enabled:
            return True, None
        ratio = self.compute_special_char_ratio(text)
        if ratio > self.config.special_chars_max_ratio:
            return False, f"too_many_special_chars: {ratio:.4f} > {self.config.special_chars_max_ratio}"
        return True, None

    def check_url_ratio(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查 URL 比例"""
        if not self.config.url_enabled:
            return True, None
        ratio = self.compute_url_ratio(text)
        if ratio > self.config.url_max_ratio:
            return False, f"too_many_urls: {ratio:.4f} > {self.config.url_max_ratio}"
        return True, None

    def check_stopword_ratio(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查停用词比例"""
        if not self.config.stopword_enabled:
            return True, None
        ratio = self.compute_stopword_ratio(text)
        if ratio < self.config.stopword_min_ratio:
            return False, f"low_stopword_ratio: {ratio:.4f} < {self.config.stopword_min_ratio}"
        return True, None


class LanguageDetector:
    """青小团语言检测器 - 包装 langdetect 和 fasttext"""

    def __init__(self, fasttext_model_path: Optional[str] = None):
        self.fasttext_model_path = fasttext_model_path
        self._fasttext_model = None

    def detect_langdetect(self, text: str) -> Optional[str]:
        """使用 langdetect 检测语言"""
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42
            if len(text) < 20:
                return None
            return detect(text)
        except ImportError:
            logger.warning("青小团: langdetect 未安装，跳过语言检测")
            return None
        except Exception:
            return None

    def detect_fasttext(self, text: str) -> Optional[str]:
        """使用 fasttext 检测语言"""
        try:
            import fasttext
            if self._fasttext_model is None and self.fasttext_model_path:
                if os.path.exists(self.fasttext_model_path):
                    self._fasttext_model = fasttext.load_model(self.fasttext_model_path)
            if self._fasttext_model is None:
                return None
            cleaned = text.replace("\n", " ").strip()[:1000]
            if not cleaned:
                return None
            predictions = self._fasttext_model.predict(cleaned, k=1)
            lang = predictions[0][0].replace("__label__", "")
            return lang
        except ImportError:
            logger.debug("青小团: fasttext 未安装")
            return None
        except Exception:
            return None

    def detect(self, text: str) -> Optional[str]:
        """综合检测语言"""
        lang = self.detect_fasttext(text)
        if lang:
            return lang
        lang = self.detect_langdetect(text)
        return lang

    def is_target_language(self, text: str, target_langs: List[str]) -> bool:
        """检查文本是否为目标语言"""
        lang = self.detect(text)
        if lang is None:
            return True
        lang = lang[:2].lower()
        for target in target_langs:
            if lang == target[:2].lower():
                return True
        return False


class PerplexityFilter:
    """青小团困惑度过滤器 - 使用 n-gram 语言模型估算"""

    def __init__(self, max_threshold: float = 1000.0):
        self.max_threshold = max_threshold
        self._char_ngram_counts = Counter()
        self._total_ngrams = 0
        self._trained = False

    def train_fast(self, corpus: List[str], n: int = 4):
        """快速训练字符级 n-gram 模型用于困惑度估算"""
        for doc in corpus:
            doc_clean = doc.strip().lower()[:5000]
            for i in range(len(doc_clean) - n + 1):
                ngram = doc_clean[i:i + n]
                self._char_ngram_counts[ngram] += 1
                self._total_ngrams += 1
        self._trained = True

    def estimate_perplexity(self, text: str, n: int = 4) -> float:
        """估算文本困惑度"""
        if not self._trained or self._total_ngrams == 0:
            return 500.0
        text_clean = text.strip().lower()[:10000]
        if len(text_clean) < n:
            return 500.0

        log_prob_sum = 0.0
        ngram_count = 0

        for i in range(len(text_clean) - n + 1):
            ngram = text_clean[i:i + n]
            count = self._char_ngram_counts.get(ngram, 0)
            prob = (count + 1) / (self._total_ngrams + len(self._char_ngram_counts))
            log_prob_sum += math.log2(prob)
            ngram_count += 1

        if ngram_count == 0:
            return 500.0

        avg_log_prob = log_prob_sum / ngram_count
        perplexity = math.pow(2, -avg_log_prob)
        return perplexity

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """检查文本困惑度是否在阈值内"""
        if not self._trained:
            return True, None
        ppl = self.estimate_perplexity(text)
        if ppl > self.max_threshold:
            return False, f"high_perplexity: {ppl:.2f} > {self.max_threshold}"
        return True, None


class QualityFilter:
    """青小团质量过滤器 - 集成所有启发式过滤规则"""

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        fasttext_model_path: Optional[str] = None,
    ):
        self.config = config or FilterConfig()
        self.heuristics = TextQualityHeuristics(self.config)
        self.lang_detector = LanguageDetector(fasttext_model_path)
        self.perplexity_filter = PerplexityFilter(
            max_threshold=self.config.perplexity_max_threshold
        )
        self.stats = FilterStats()
        self._filters: List[Tuple[str, Callable[[str], Tuple[bool, Optional[str]]]]] = []
        self._build_filter_chain()
        logger.info("青小团质量过滤器初始化完成")

    def _build_filter_chain(self):
        """构建过滤链"""
        self._filters = [
            ("text_length", lambda t: self.heuristics.check_text_length(t)),
            ("ngram_repeat", lambda t: self.heuristics.check_ngram_repeat(t)),
            ("punctuation", lambda t: self.heuristics.check_punctuation(t)),
            ("avg_word_length", lambda t: self.heuristics.check_word_length(t)),
            ("special_chars", lambda t: self.heuristics.check_special_chars(t)),
            ("url_ratio", lambda t: self.heuristics.check_url_ratio(t)),
            ("stopword_ratio", lambda t: self.heuristics.check_stopword_ratio(t)),
        ]

    def train_perplexity_model(self, corpus: List[str]):
        """训练困惑度模型"""
        logger.info(f"青小团开始训练困惑度模型, 使用 {len(corpus)} 个文档")
        self.perplexity_filter.train_fast(corpus)

    def check_document(self, doc: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict]:
        """检查单个文档的质量"""
        text = doc.get("text", doc.get("content", doc.get("", "")))
        if not isinstance(text, str):
            return False, "empty_or_invalid_text", {}

        metrics = self.heuristics.compute_all_metrics(text)
        reasons = []

        if self.config.lang_detect_enabled and self.config.target_languages:
            if not self.lang_detector.is_target_language(text, self.config.target_languages):
                reasons.append("unwanted_language")

        for filter_name, filter_fn in self._filters:
            passed, reason = filter_fn(text)
            if not passed:
                reasons.append(reason or filter_name)

        if self.config.perplexity_enabled:
            passed, reason = self.perplexity_filter.check(text)
            if not passed:
                reasons.append(reason or "high_perplexity")

        if reasons:
            return False, "; ".join(reasons), metrics

        return True, None, metrics

    def filter_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """过滤单个文档，通过则返回文档，不通过则返回 None"""
        passed, reason, metrics = self.check_document(doc)
        self.stats.total_documents += 1

        if passed:
            self.stats.passed += 1
            doc["quality_metrics"] = metrics
            doc["quality_score"] = self._compute_quality_score(metrics)
            return doc
        else:
            self.stats.filtered_out += 1
            self.stats.filter_reasons[reason or "unknown"] += 1
            return None

    def _compute_quality_score(self, metrics: Dict[str, float]) -> float:
        """计算综合质量分数 (0-1)"""
        score = 1.0

        ngram_penalty = max(0, metrics.get("ngram_repeat_ratio", 0) - 0.1) * 2
        score -= ngram_penalty

        punct_ratio = metrics.get("punctuation_ratio", 0)
        if punct_ratio < 0.05:
            score -= 0.2
        elif punct_ratio > 0.5:
            score -= 0.1

        special_penalty = metrics.get("special_char_ratio", 0) * 3
        score -= special_penalty

        url_penalty = metrics.get("url_ratio", 0) * 1.5
        score -= url_penalty

        return max(0.0, min(1.0, score))

    def filter_batch(
        self,
        documents: List[Dict[str, Any]],
        num_workers: int = 8,
    ) -> List[Optional[Dict[str, Any]]]:
        """批量过滤文档"""
        logger.info(f"青小团开始批量过滤 {len(documents)} 个文档, workers={num_workers}")

        self.stats = FilterStats()

        if num_workers <= 1:
            results = []
            for doc in tqdm(documents, desc="青小团质量过滤", ncols=80):
                results.append(self.filter_document(doc))
            return results

        results = [None] * len(documents)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {}
            for i, doc in enumerate(documents):
                future = executor.submit(self.filter_document, doc)
                future_map[future] = i

            for future in tqdm(
                as_completed(future_map), total=len(future_map),
                desc="青小团质量过滤", ncols=80
            ):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"青小团过滤文档 {idx} 异常: {e}")
                    results[idx] = None

        return results

    def filter_stream(
        self,
        doc_stream: Iterator[Dict[str, Any]],
        num_workers: int = 8,
    ) -> Iterator[Dict[str, Any]]:
        """流式过滤文档生成器"""
        batch = []
        batch_size = 10000
        processed = 0

        for doc in doc_stream:
            batch.append(doc)
            if len(batch) >= batch_size:
                results = self.filter_batch(batch, num_workers)
                for r in results:
                    if r is not None:
                        yield r
                processed += len(batch)
                logger.info(f"青小团流式过滤进度: {processed:,} 文档已处理")
                batch = []

        if batch:
            results = self.filter_batch(batch, num_workers)
            for r in results:
                if r is not None:
                    yield r

    def filter_from_jsonl(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        num_workers: int = 8,
    ) -> FilterStats:
        """从 JSONL 文件读取并过滤"""
        logger.info(f"青小团开始过滤文件: {input_path} -> {output_path}")

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            docs = []
            for line in fin:
                line = line.strip()
                if line:
                    try:
                        doc = json.loads(line)
                        docs.append(doc)
                    except json.JSONDecodeError:
                        continue

            results = self.filter_batch(docs, num_workers)

            for r in results:
                if r is not None:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"\n{self.stats.summary()}")
        return self.stats

    def report_summary(self) -> str:
        """生成过滤报告"""
        return self.stats.summary()

    def get_filter_config(self) -> Dict:
        """获取当前过滤配置"""
        return self.config.to_dict()
