"""
青小团 合成数据生成引擎 - 基类模块
青奈科技 (QingNai Technology)

提供通用数据生成基类 BaseGenerator，支持模板替换、
随机种子控制、批量/流式生成、进度报告和数据验证。
"""

import abc
import hashlib
import json
import logging
import os
import random
import string
import time
from collections.abc import Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import yaml

logger = logging.getLogger("qingxiaotuan.synthetic")


@dataclass
class GenerationContext:
    seed: int = 42
    batch_size: int = 1000
    total_count: int = 10000
    output_dir: str = "./output"
    verbose: bool = True
    generation_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRecord:
    id: str
    source: str
    data_type: str
    content: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ProgressReporter:
    def __init__(self, total: int, desc: str = "生成中", bar_length: int = 40):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.start_time = time.time()
        self.lock = Lock()
        self._last_reported = 0

    def update(self, current: int):
        with self.lock:
            if current - self._last_reported < max(1, self.total // 100):
                return
            self._last_reported = current
            elapsed = time.time() - self.start_time
            ratio = current / self.total if self.total > 0 else 0
            filled = int(self.bar_length * ratio)
            bar = "█" * filled + "░" * (self.bar_length - filled)
            speed = current / elapsed if elapsed > 0 else 0
            eta = (self.total - current) / speed if speed > 0 else 0
            print(
                f"\r{self.desc}: |{bar}| {current}/{self.total} "
                f"({ratio*100:.1f}%) {speed:.0f}条/秒 ETA:{eta:.0f}s",
                end="", flush=True
            )

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n✓ {self.desc}完成，耗时{elapsed:.1f}秒，总计{self.total}条")


class TemplateEngine:
    def __init__(self, templates: dict[str, str]):
        self.templates = templates
        self._cache: dict[str, str] = {}

    def render(self, template_name: str, **kwargs) -> str:
        template = self.templates.get(template_name)
        if template is None:
            raise ValueError(f"模板 '{template_name}' 不存在")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"模板 '{template_name}' 缺少参数: {e}")
            return template

    def add_template(self, name: str, template: str):
        self.templates[name] = template

    def get_keys(self, template_name: str) -> list[str]:
        template = self.templates.get(template_name, "")
        keys = []
        in_brace = False
        current = []
        for ch in template:
            if ch == '{' and not in_brace:
                in_brace = True
                current = []
            elif ch == '}' and in_brace:
                in_brace = False
                keys.append(''.join(current))
                current = []
            elif in_brace:
                current.append(ch)
        return keys


class DataValidator:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.errors: list[str] = []

    def validate(self, record: DataRecord) -> bool:
        self.errors.clear()
        content = record.content
        if not content:
            self.errors.append("内容为空")
            return False
        if not isinstance(content, dict):
            self.errors.append("内容类型错误，期望dict")
            return False
        min_len = self.config.get("min_length", 10)
        max_len = self.config.get("max_length", 8192)
        text = json.dumps(content, ensure_ascii=False)
        if len(text) < min_len:
            self.errors.append(f"内容过短 ({len(text)} < {min_len})")
            return False
        if len(text) > max_len:
            self.errors.append(f"内容过长 ({len(text)} > {max_len})")
            return False
        if self.config.get("check_language", False):
            if not self._check_language_consistency(text):
                self.errors.append("语言一致性检查失败")
                return False
        return True

    def _check_language_consistency(self, text: str) -> bool:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total = chinese_chars + ascii_chars
        if total == 0:
            return True
        ratio = chinese_chars / total
        return not (0.1 < ratio < 0.9 and chinese_chars > 50 and ascii_chars > 50)

    def get_error_messages(self) -> list[str]:
        return self.errors[:]


class BaseGenerator(abc.ABC):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        self.config = config
        self.context = context or GenerationContext()
        self.seed = self.context.seed
        self._rng = random.Random(self.seed)
        self._lock = Lock()
        self._generated_count = 0
        self.template_engine = TemplateEngine(self._init_templates())
        self.validator = DataValidator(config.get("validation", {}))
        self._init_logging()

    def _init_logging(self):
        log_dir = self.config.get("global", {}).get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(
            os.path.join(log_dir, f"generator_{self.context.generation_id}.log"),
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)

    def _init_templates(self) -> dict[str, str]:
        return {}

    def reset_seed(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def _make_id(self, prefix: str = "data") -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        rand = self._rng.randint(0, 999999)
        raw = f"{prefix}_{timestamp}_{rand}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @abc.abstractmethod
    def generate_one(self, **kwargs) -> DataRecord:
        raise NotImplementedError

    def generate_batch(self, count: int, **kwargs) -> list[DataRecord]:
        records = []
        reporter = ProgressReporter(count, desc=self.__class__.__name__)
        for i in range(count):
            record = self.generate_one(**kwargs)
            records.append(record)
            if self.context.verbose:
                reporter.update(i + 1)
        if self.context.verbose:
            reporter.finish()
        self._generated_count += len(records)
        return records

    def generate_stream(self, count: int, **kwargs) -> Generator[DataRecord, None, None]:
        for i in range(count):
            record = self.generate_one(**kwargs)
            self._generated_count += 1
            yield record
            if self.context.verbose and (i + 1) % 100 == 0:
                logger.info(f"流式生成进度: {i+1}/{count}")

    def save_records(self, records: list[DataRecord], filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_json() + "\n")
        logger.info(f"已保存 {len(records)} 条记录到 {filepath}")

    def save_records_parquet(self, records: list[DataRecord], filepath: str):
        try:
            import pandas as pd
            data = [r.to_dict() for r in records]
            df = pd.DataFrame(data)
            df.to_parquet(filepath, compression="gzip")
            logger.info(f"已保存 {len(records)} 条记录到 {filepath} (parquet)")
        except ImportError:
            logger.warning("pandas未安装，回退到JSONL格式")
            self.save_records(records, filepath.replace(".parquet", ".jsonl"))

    def _pick_weighted(self, options: dict[str, float]) -> str:
        items = list(options.items())
        keys = [k for k, _ in items]
        weights = [v for _, v in items]
        return self._rng.choices(keys, weights=weights, k=1)[0]

    def _random_text(self, length: int = 50, lang: str = "zh-CN") -> str:
        if lang == "zh-CN":
            chars = string.ascii_letters + string.digits + "，。、；：？！""''【】（）"
            base_chars = "的一是不了人我在有他这中大来上国个到说会子要也出就得可自们看时没如手心样日面前开"
            pool = base_chars * 3 + chars
        else:
            pool = string.ascii_letters + string.digits + ".,;:!?\"'- "
        return ''.join(self._rng.choices(pool, k=length))

    def validate_record(self, record: DataRecord) -> bool:
        return self.validator.validate(record)

    def get_stats(self) -> dict[str, Any]:
        return {
            "generator_type": self.__class__.__name__,
            "total_generated": self._generated_count,
            "seed": self.seed,
            "template_count": len(self.template_engine.templates),
        }

    @classmethod
    def from_config(cls, config_path: str) -> "BaseGenerator":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} seed={self.seed} generated={self._generated_count}>"
