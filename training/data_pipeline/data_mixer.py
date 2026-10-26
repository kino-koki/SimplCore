# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# data_mixer.py - 大规模数据混合引擎
# 支持带权重的多数据源混合、无限生成器模式、动态权重调整、混合比例日志

import os
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple, Any, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("qingxiaotuan.data_mixer")


@dataclass
class MixingStats:
    """青小团数据混合统计"""
    total_samples: int = 0
    per_category_samples: Dict[str, int] = field(default_factory=dict)
    per_source_samples: Dict[str, int] = field(default_factory=dict)
    actual_ratios: Dict[str, float] = field(default_factory=dict)
    target_ratios: Dict[str, float] = field(default_factory=dict)
    adjustment_count: int = 0
    start_time: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 数据混合引擎 - 统计报告",
            "=" * 60,
            f"总样本数:            {self.total_samples:,}",
            f"权重调整次数:        {self.adjustment_count}",
            "-" * 60,
            "各类别比例:",
        ]
        for cat in sorted(set(list(self.target_ratios.keys()) + list(self.actual_ratios.keys()))):
            target = self.target_ratios.get(cat, 0)
            actual = self.actual_ratios.get(cat, 0)
            samples = self.per_category_samples.get(cat, 0)
            lines.append(f"  {cat}: target={target:.2%}, actual={actual:.2%}, samples={samples:,}")

        if self.per_source_samples:
            lines.extend(["-" * 60, "各数据源样本数:"])
            for src, cnt in sorted(self.per_source_samples.items(), key=lambda x: -x[1]):
                pct = cnt / self.total_samples * 100 if self.total_samples > 0 else 0
                lines.append(f"  {src}: {cnt:,} ({pct:.2f}%)")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DataSourceConfig:
    """青小团数据源配置"""
    name: str
    weight: float = 1.0
    category: str = "general"
    max_samples: Optional[int] = None
    data_path: Optional[str] = None
    data_iterator: Optional[Iterator[Dict[str, Any]]] = None


class WeightedSampler:
    """青小团加权采样器 - 按权重从多个数据源采样"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample(self, weights: List[float]) -> int:
        total = sum(weights)
        if total <= 0:
            return self.rng.randint(0, len(weights) - 1)
        r = self.rng.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def sample_batch(self, weights: List[float], n: int) -> List[int]:
        indices = self.np_rng.choice(
            len(weights), size=n, p=[w / sum(weights) for w in weights]
        )
        return indices.tolist()


class DynamicWeightAdjuster:
    """青小团动态权重调整器 - 根据实际采样比例调整权重"""

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.80,
        adjustment_rate: float = 0.1,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_rate = adjustment_rate
        self._sample_counts: Dict[str, int] = defaultdict(int)

    def record_sample(self, category: str):
        self._sample_counts[category] += 1

    def adjust_weights(
        self, current_weights: Dict[str, float], total_samples: int
    ) -> Dict[str, float]:
        if total_samples == 0:
            return current_weights

        adjusted = current_weights.copy()
        actual_ratios = {
            cat: count / total_samples
            for cat, count in self._sample_counts.items()
        }

        for cat in adjusted:
            target = current_weights[cat]
            actual = actual_ratios.get(cat, 0)

            if actual > target:
                overage = (actual - target) / target if target > 0 else 1.0
                reduction = min(overage * self.adjustment_rate, 0.1)
                adjusted[cat] = max(self.min_weight, adjusted[cat] * (1 - reduction))
            elif actual < target:
                deficit = (target - actual) / target if target > 0 else 1.0
                increase = min(deficit * self.adjustment_rate, 0.1)
                adjusted[cat] = min(self.max_weight, adjusted[cat] * (1 + increase))

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted


class SourceLoader:
    """青小团数据源加载器 - 从不同格式加载数据"""

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        documents = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        documents.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return documents

    @staticmethod
    def load_jsonl_stream(filepath: str) -> Iterator[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def load_arrow(filepath: str) -> List[Dict[str, Any]]:
        try:
            import pyarrow.ipc as ipc
            with open(filepath, "rb") as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
                columns = table.column_names
                documents = []
                for i in range(table.num_rows):
                    doc = {}
                    for col in columns:
                        doc[col] = table.column(col)[i].as_py()
                    documents.append(doc)
                return documents
        except ImportError:
            logger.error("青小团: 需要安装 pyarrow: pip install pyarrow")
            raise

    @staticmethod
    def load_parquet(filepath: str) -> List[Dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(filepath)
            columns = table.column_names
            documents = []
            for i in range(table.num_rows):
                doc = {}
                for col in columns:
                    doc[col] = table.column(col)[i].as_py()
                documents.append(doc)
            return documents
        except ImportError:
            logger.error("青小团: 需要安装 pyarrow: pip install pyarrow")
            raise


class DataMixer:
    """青小团数据混合器 - 多数据源加权混合"""

    def __init__(
        self,
        strategy: str = "weighted",
        epoch_less: bool = True,
        shuffle_buffer_size: int = 1_000_000,
        refresh_weights_interval: int = 10000,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.epoch_less = epoch_less
        self.shuffle_buffer_size = shuffle_buffer_size
        self.refresh_weights_interval = refresh_weights_interval
        self.seed = seed

        self.sampler = WeightedSampler(seed=seed)
        self.weight_adjuster = DynamicWeightAdjuster()
        self.sources: Dict[str, DataSourceConfig] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.stats = MixingStats()
        self._current_weights: Dict[str, float] = {}
        self._category_weights: Dict[str, float] = {}
        self._source_data: Dict[str, List[Dict[str, Any]]] = {}
        self._source_iterators: Dict[str, Iterator[Dict[str, Any]]] = {}
        self._step_count: int = 0

        logger.info(
            f"青小团数据混合器初始化: strategy={strategy}, "
            f"epoch_less={epoch_less}, buffer={shuffle_buffer_size:,}"
        )

    def add_source(
        self,
        name: str,
        data: Optional[List[Dict[str, Any]]] = None,
        data_path: Optional[str] = None,
        weight: float = 1.0,
        category: str = "general",
        max_samples: Optional[int] = None,
    ):
        if name in self.sources:
            logger.warning(f"青小团数据源 {name} 已存在，将被覆盖")

        config = DataSourceConfig(
            name=name, weight=weight, category=category,
            max_samples=max_samples, data_path=data_path,
        )

        if data is not None:
            self._source_data[name] = data
        elif data_path is not None:
            if data_path.endswith(".jsonl"):
                self._source_data[name] = SourceLoader.load_jsonl(data_path)
            elif data_path.endswith(".arrow"):
                self._source_data[name] = SourceLoader.load_arrow(data_path)
            elif data_path.endswith(".parquet"):
                self._source_data[name] = SourceLoader.load_parquet(data_path)
            else:
                raise ValueError(f"青小团不支持的数据格式: {data_path}")

        self.sources[name] = config
        self.categories[category].append(name)

        self._current_weights[name] = weight
        self._category_weights[category] = self._category_weights.get(category, 0) + weight

        logger.info(
            f"青小团添加数据源: {name} (category={category}, weight={weight}, "
            f"samples={len(self._source_data.get(name, []))})"
        )

        self._source_iterators[name] = self._create_iterator(name)

    def add_source_from_iterator(
        self,
        name: str,
        iterator: Iterator[Dict[str, Any]],
        weight: float = 1.0,
        category: str = "general",
    ):
        config = DataSourceConfig(
            name=name, weight=weight, category=category,
            data_iterator=iterator,
        )
        self.sources[name] = config
        self.categories[category].append(name)
        self._current_weights[name] = weight
        self._category_weights[category] = self._category_weights.get(category, 0) + weight
        self._source_iterators[name] = iterator

        logger.info(f"青小团添加流式数据源: {name} (category={category}, weight={weight})")

    def _create_iterator(self, name: str) -> Iterator[Dict[str, Any]]:
        data = self._source_data.get(name, [])
        config = self.sources[name]

        def generator():
            sample_count = 0
            while True:
                if config.max_samples and sample_count >= config.max_samples:
                    break

                if self.epoch_less:
                    indices = list(range(len(data)))
                    self.sampler.rng.shuffle(indices)
                    for idx in indices:
                        if config.max_samples and sample_count >= config.max_samples:
                            break
                        yield data[idx]
                        sample_count += 1

                    if not self.epoch_less:
                        break
                else:
                    if sample_count >= len(data):
                        break
                    yield data[sample_count]
                    sample_count += 1

        return generator()

    def _get_category_weight(self, category: str) -> float:
        return self._category_weights.get(category, 0)

    def _select_source(self) -> Tuple[str, str]:
        if self.strategy == "weighted":
            if self.weight_adjuster and self._step_count > 0 and \
               self._step_count % self.refresh_weights_interval == 0:
                self._adjust_weights()

            cat_names = list(self._category_weights.keys())
            cat_weights = list(self._category_weights.values())
            cat_idx = self.sampler.sample(cat_weights)
            selected_cat = cat_names[cat_idx]

            sources_in_cat = self.categories[selected_cat]
            src_weights = [self._current_weights[s] for s in sources_in_cat]
            src_idx = self.sampler.sample(src_weights)
            selected_src = sources_in_cat[src_idx]

            return selected_cat, selected_src

        elif self.strategy == "proportional":
            all_names = list(self._current_weights.keys())
            all_weights = list(self._current_weights.values())
            idx = self.sampler.sample(all_weights)
            selected = all_names[idx]
            return self.sources[selected].category, selected

        else:
            all_names = list(self._current_weights.keys())
            selected = self.sampler.rng.choice(all_names)
            return self.sources[selected].category, selected

    def _adjust_weights(self):
        old_weights = self._category_weights.copy()
        self._category_weights = self.weight_adjuster.adjust_weights(
            self._category_weights, self._step_count
        )
        self.stats.adjustment_count += 1
        logger.debug(
            f"青小团权重调整: {old_weights} -> {self._category_weights}"
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.generate()

    def __next__(self) -> Dict[str, Any]:
        return next(self.generate())

    def generate(self) -> Iterator[Dict[str, Any]]:
        while True:
            category, source_name = self._select_source()
            iterator = self._source_iterators.get(source_name)

            if iterator is None:
                logger.warning(f"青小团数据源 {source_name} 无迭代器")
                continue

            try:
                doc = next(iterator)
            except StopIteration:
                logger.info(f"青小团数据源 {source_name} 已耗尽，重置")
                iterator = self._create_iterator(source_name)
                self._source_iterators[source_name] = iterator
                try:
                    doc = next(iterator)
                except StopIteration:
                    continue

            doc["_source"] = source_name
            doc["_category"] = category

            self.stats.total_samples += 1
            self.stats.per_category_samples[category] = \
                self.stats.per_category_samples.get(category, 0) + 1
            self.stats.per_source_samples[source_name] = \
                self.stats.per_source_samples.get(source_name, 0) + 1
            self.weight_adjuster.record_sample(category)
            self._step_count += 1

            yield doc

    def generate_buffered(self, batch_size: int = 10000) -> Iterator[List[Dict[str, Any]]]:
        """带混洗缓冲的批量生成"""
        buffer = []

        for doc in self.generate():
            buffer.append(doc)

            if len(buffer) >= batch_size:
                self.sampler.rng.shuffle(buffer)
                yield buffer[:batch_size]
                buffer = buffer[batch_size:]

        if buffer:
            self.sampler.rng.shuffle(buffer)
            yield buffer

    def generate_with_shuffle(self) -> Iterator[Dict[str, Any]]:
        """使用大缓冲区混洗的生成器"""
        buffer = []

        for doc in self.generate():
            buffer.append(doc)

            if len(buffer) >= self.shuffle_buffer_size:
                self.sampler.rng.shuffle(buffer)
                yield from buffer
                buffer = []

        if buffer:
            self.sampler.rng.shuffle(buffer)
            yield from buffer

    def set_category_weights(self, weights: Dict[str, float]):
        total = sum(weights.values())
        self._category_weights = {k: v / total for k, v in weights.items()}
        logger.info(f"青小团设置类别权重: {self._category_weights}")

    def set_source_weights(self, weights: Dict[str, float]):
        total = sum(weights.values())
        for name, w in weights.items():
            if name in self._current_weights:
                self._current_weights[name] = w / total
        logger.info(f"青小团设置数据源权重: {self._current_weights}")

    def get_mixing_report(self) -> MixingStats:
        total = self.stats.total_samples or 1
        self.stats.actual_ratios = {
            cat: count / total
            for cat, count in self.stats.per_category_samples.items()
        }
        self.stats.target_ratios = dict(self._category_weights)
        return self.stats

    def report_summary(self) -> str:
        self.get_mixing_report()
        return self.stats.summary()

    def save_state(self, filepath: str):
        state = {
            "step_count": self._step_count,
            "sample_counts": dict(self.weight_adjuster._sample_counts),
            "category_weights": self._category_weights,
            "source_weights": self._current_weights,
            "stats": {
                "total_samples": self.stats.total_samples,
                "per_category": dict(self.stats.per_category_samples),
                "per_source": dict(self.stats.per_source_samples),
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"青小团数据混合状态已保存: {filepath}")

    def load_state(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"青小团状态文件不存在: {filepath}")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._step_count = state.get("step_count", 0)
        self.weight_adjuster._sample_counts = defaultdict(
            int, state.get("sample_counts", {})
        )
        self._category_weights = state.get("category_weights", {})
        self._current_weights = state.get("source_weights", {})
        logger.info(f"青小团数据混合状态已加载: {filepath}")


class CategoryBuilder:
    """青小团类别构建器 - 简化多类别数据源的设置"""

    def __init__(self, mixer: DataMixer):
        self.mixer = mixer

    def add_general_corpus(
        self,
        sources: Dict[str, Any],
        total_weight: float = 0.70,
    ):
        per_source_weight = total_weight / max(len(sources), 1)
        for name, data_or_path in sources.items():
            if isinstance(data_or_path, list):
                self.mixer.add_source(
                    name, data=data_or_path,
                    weight=per_source_weight, category="general"
                )
            else:
                self.mixer.add_source(
                    name, data_path=data_or_path,
                    weight=per_source_weight, category="general"
                )

    def add_code_corpus(
        self,
        sources: Dict[str, Any],
        total_weight: float = 0.20,
    ):
        per_source_weight = total_weight / max(len(sources), 1)
        for name, data_or_path in sources.items():
            if isinstance(data_or_path, list):
                self.mixer.add_source(
                    name, data=data_or_path,
                    weight=per_source_weight, category="code"
                )
            else:
                self.mixer.add_source(
                    name, data_path=data_or_path,
                    weight=per_source_weight, category="code"
                )

    def add_math_corpus(
        self,
        sources: Dict[str, Any],
        total_weight: float = 0.10,
    ):
        per_source_weight = total_weight / max(len(sources), 1)
        for name, data_or_path in sources.items():
            if isinstance(data_or_path, list):
                self.mixer.add_source(
                    name, data=data_or_path,
                    weight=per_source_weight, category="math"
                )
            else:
                self.mixer.add_source(
                    name, data_path=data_or_path,
                    weight=per_source_weight, category="math"
                )
