"""
青小团 合成数据生成引擎 - 合成数据工厂（编排所有生成器）
青奈科技 (QingNai Technology)

根据配置生成指定数量和类型的混合数据，
支持分布式生成（多进程）、自动负载平衡、
数据汇总统计、中断恢复和多格式输出。
"""

import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from base_generator import BaseGenerator, DataRecord, GenerationContext, ProgressReporter
from code_generator import CodeGenerator
from conversation_generator import ConversationGenerator
from data_augmentation import DataAugmentation
from data_validator import DataValidator, ValidationReport
from instruction_generator import InstructionGenerator
from knowledge_generator import KnowledgeGenerator
from reasoning_generator import ReasoningGenerator

logger = logging.getLogger("qingxiaotuan.factory")

_GENERATOR_REGISTRY = {
    "instruction": InstructionGenerator,
    "reasoning": ReasoningGenerator,
    "code": CodeGenerator,
    "conversation": ConversationGenerator,
    "knowledge": KnowledgeGenerator,
}


class DatasetShard:
    def __init__(self, path: str, records: list[DataRecord], shard_id: int):
        self.path = path
        self.records = records
        self.shard_id = shard_id
        self.size = len(records)

    def save_jsonl(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(record.to_json() + "\n")

    def save_parquet(self):
        try:
            import pandas as pd
            data = [r.to_dict() for r in self.records]
            df = pd.DataFrame(data)
            df.to_parquet(self.path.replace(".jsonl", ".parquet"), compression="gzip")
        except ImportError:
            logger.warning("pandas未安装，跳过parquet格式")


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, data_type: str, count: int, generation_id: str):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{data_type}.json")
        checkpoint = {
            "data_type": data_type,
            "count": count,
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)

    def load_checkpoint(self, data_type: str) -> Optional[dict]:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{data_type}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return None
        return None

    def clear_checkpoint(self, data_type: str):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{data_type}.json")
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_"):
                os.remove(os.path.join(self.checkpoint_dir, f))


class DatasetStatistics:
    def __init__(self):
        self.total_records = 0
        self.type_counts: dict[str, int] = {}
        self.type_details: dict[str, dict] = {}
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.duration_seconds: float = 0.0
        self.output_files: list[str] = []
        self.augmentation_stats: dict[str, int] = {}
        self.validation_report: Optional[dict] = None

    def add_type_count(self, data_type: str, count: int):
        self.type_counts[data_type] = self.type_counts.get(data_type, 0) + count
        self.total_records += count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "type_counts": self.type_counts,
            "type_details": self.type_details,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 2),
            "output_files": self.output_files,
            "augmentation_stats": self.augmentation_stats,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "  青小团 合成数据生成统计报告",
            "=" * 55,
            f"  开始时间：{self.start_time}",
            f"  结束时间：{self.end_time}",
            f"  总耗时：{self.duration_seconds:.1f}秒",
            f"  生成速度：{self.total_records/self.duration_seconds:.0f}条/秒" if self.duration_seconds > 0 else "",
            "-" * 55,
            "  数据生成统计：",
        ]
        for dtype, count in sorted(self.type_counts.items(), key=lambda x: -x[1]):
            ratio = count / self.total_records * 100 if self.total_records > 0 else 0
            lines.append(f"    {dtype}: {count:>8,} 条 ({ratio:>5.1f}%)")
        lines.append(f"  {'合计':>15}: {self.total_records:>8,} 条")
        if self.augmentation_stats:
            lines.append("-" * 55)
            lines.append("  数据增强统计：")
            for tech, count in self.augmentation_stats.items():
                lines.append(f"    {tech}: {count}")
        lines.append("-" * 55)
        lines.append("  输出文件：")
        for f in self.output_files:
            lines.append(f"    • {f}")
        lines.append("=" * 55)
        return "\n".join(lines)


class SyntheticDatasetFactory:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.global_cfg = config.get("global", {})
        self.output_cfg = config.get("output", {})
        self.data_types_cfg = config.get("data_types", {})
        self.aug_cfg = config.get("augmentation", {})
        self.val_cfg = config.get("validation", {})
        self.dist_cfg = config.get("distributed", {})
        self.generation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_mgr = CheckpointManager(
            self.global_cfg.get("tmp_dir", "./tmp")
        )
        self.generators: dict[str, BaseGenerator] = {}
        self.all_records: dict[str, list[DataRecord]] = {}
        self.stats = DatasetStatistics()
        self._init_logging()

    def _init_logging(self):
        log_dir = self.global_cfg.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(
            os.path.join(log_dir, f"factory_{self.generation_id}.log"),
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))
        logger.addHandler(console)

    def _create_context(self) -> GenerationContext:
        return GenerationContext(
            seed=self.global_cfg.get("random_seed", 42),
            batch_size=self.global_cfg.get("batch_size", 1000),
            total_count=0,
            output_dir=self.global_cfg.get("output_dir", "./output"),
            verbose=self.global_cfg.get("verbose", True),
            generation_id=self.generation_id,
        )

    def _init_generators(self):
        context = self._create_context()
        for data_type, gen_class in _GENERATOR_REGISTRY.items():
            type_cfg = self.data_types_cfg.get(data_type, {})
            if type_cfg.get("enabled", False):
                self.generators[data_type] = gen_class(self.config, context)
                logger.info(f"初始化生成器: {data_type}")

    def _generate_type_data(self, data_type: str,
                              count: int, use_checkpoint: bool = True) -> list[DataRecord]:
        generator = self.generators.get(data_type)
        if generator is None:
            logger.error(f"未知数据类型: {data_type}")
            return []
        resume_count = 0
        if use_checkpoint:
            cp = self.checkpoint_mgr.load_checkpoint(data_type)
            if cp:
                resume_count = cp.get("count", 0)
                logger.info(f"从检查点恢复 {data_type}: 已生成 {resume_count}/{count}")
        remaining = count - resume_count
        if remaining <= 0:
            logger.info(f"{data_type} 已全部生成 ({count})")
            return []
        logger.info(f"开始生成 {data_type} 数据: {remaining} 条")
        records = []
        batch_size = self.global_cfg.get("batch_size", 1000)
        checkpoint_interval = self.dist_cfg.get("checkpoint_interval", 5000)
        for i in range(0, remaining, batch_size):
            batch_count = min(batch_size, remaining - i)
            batch = generator.generate_batch(batch_count)
            records.extend(batch)
            total_generated = resume_count + len(records)
            if use_checkpoint and total_generated % checkpoint_interval < batch_size:
                self.checkpoint_mgr.save_checkpoint(data_type, total_generated, self.generation_id)
            logger.info(f"{data_type} 进度: {total_generated}/{count}")
        if use_checkpoint:
            self.checkpoint_mgr.save_checkpoint(data_type, count, self.generation_id)
        return records

    def _generate_parallel(self, type_targets: dict[str, int]) -> dict[str, list[DataRecord]]:
        num_workers = self.global_cfg.get("num_workers", 4)
        if num_workers <= 1:
            result = {}
            for data_type, count in type_targets.items():
                result[data_type] = self._generate_type_data(data_type, count)
            return result
        logger.info(f"启动多进程生成 (workers={num_workers})")
        with mp.Pool(processes=num_workers) as pool:
            async_results = {}
            for data_type, count in type_targets.items():
                if data_type in self.generators:
                    async_results[data_type] = pool.apply_async(
                        self._generate_type_data,
                        (data_type, count, False)
                    )
            result = {}
            for data_type, async_res in async_results.items():
                try:
                    result[data_type] = async_res.get(timeout=36000)
                except Exception as e:
                    logger.error(f"多进程生成 {data_type} 失败: {e}")
                    result[data_type] = []
            pool.close()
            pool.join()
        return result

    def generate(self, targets: Optional[dict[str, int]] = None,
                 parallel: bool = False) -> "SyntheticDatasetFactory":
        self.stats.start_time = datetime.now().isoformat()
        start_wall = time.time()
        self._init_generators()
        if not self.generators:
            logger.error("没有启用任何数据生成器")
            return self
        if targets is None:
            targets = {}
            for data_type, type_cfg in self.data_types_cfg.items():
                if type_cfg.get("enabled", False):
                    targets[data_type] = type_cfg.get("total_count", 10000)
        logger.info(f"合成数据生成任务开始")
        logger.info(f"目标: {json.dumps(targets, ensure_ascii=False)}")
        self.all_records = self._generate_parallel(targets) if parallel else {}
        if not parallel:
            for data_type, count in targets.items():
                self.all_records[data_type] = self._generate_type_data(data_type, count)
        for dtype, records in self.all_records.items():
            self.stats.add_type_count(dtype, len(records))
        self.stats.duration_seconds = time.time() - start_wall
        self.stats.end_time = datetime.now().isoformat()
        logger.info(f"数据生成完成，总计 {self.stats.total_records} 条")
        return self

    def augment(self) -> "SyntheticDatasetFactory":
        if not self.aug_cfg.get("enabled", False):
            logger.info("数据增强未启用")
            return self
        logger.info("开始数据增强...")
        augmenter = DataAugmentation(self.config)
        augmented_count = 0
        for data_type in list(self.all_records.keys()):
            records = self.all_records[data_type]
            augmented = augmenter.augment_dataset(records)
            if len(augmented) > len(records):
                self.all_records[data_type] = augmented
                aug_diff = len(augmented) - len(records)
                augmented_count += aug_diff
                self.stats.add_type_count(f"{data_type}_augmented", aug_diff)
        self.stats.augmentation_stats = {"total_augmented": augmented_count}
        logger.info(f"数据增强完成，新增 {augmented_count} 条")
        return self

    def validate(self) -> "SyntheticDatasetFactory":
        if not self.val_cfg.get("enabled", False):
            logger.info("数据验证未启用")
            return self
        logger.info("开始数据验证...")
        validator = DataValidator(self.config)
        all_records = []
        for records in self.all_records.values():
            all_records.extend(records)
        report = validator.validate_batch(all_records)
        validator.save_report(report)
        self.stats.validation_report = report.to_dict()
        logger.info(f"验证完成: 有效 {report.valid_count}/{report.total_records}")
        if report.invalid_count > 0:
            logger.warning(f"发现 {report.invalid_count} 条无效数据")
        return self

    def save(self, output_dir: Optional[str] = None) -> "SyntheticDatasetFactory":
        output_dir = output_dir or self.global_cfg.get("output_dir", "./output")
        formats = self.output_cfg.get("formats", ["jsonl"])
        shard_by_topic = self.output_cfg.get("shard_by_topic", True)
        max_file_size = self.output_cfg.get("max_file_size_mb", 100) * 1024 * 1024
        os.makedirs(output_dir, exist_ok=True)
        self.stats.output_files = []
        if shard_by_topic:
            for data_type, records in self.all_records.items():
                shard_id = 0
                for i in range(0, len(records), 5000):
                    shard = records[i:i + 5000]
                    if "jsonl" in formats:
                        filename = f"{data_type}_shard_{shard_id}.jsonl"
                        filepath = os.path.join(output_dir, filename)
                        shard_obj = DatasetShard(filepath, shard, shard_id)
                        shard_obj.save_jsonl()
                        self.stats.output_files.append(filepath)
                    if "parquet" in formats:
                        filename = f"{data_type}_shard_{shard_id}.parquet"
                        filepath = os.path.join(output_dir, filename)
                        try:
                            import pandas as pd
                            data = [r.to_dict() for r in shard]
                            df = pd.DataFrame(data)
                            df.to_parquet(filepath, compression="gzip")
                            self.stats.output_files.append(filepath)
                        except ImportError:
                            pass
                    shard_id += 1
        else:
            all_records = []
            for records in self.all_records.values():
                all_records.extend(records)
            if "jsonl" in formats:
                filepath = os.path.join(output_dir, "all_data.jsonl")
                with open(filepath, "w", encoding="utf-8") as f:
                    for record in all_records:
                        f.write(record.to_json() + "\n")
                self.stats.output_files.append(filepath)
            if "parquet" in formats:
                filepath = os.path.join(output_dir, "all_data.parquet")
                try:
                    import pandas as pd
                    data = [r.to_dict() for r in all_records]
                    df = pd.DataFrame(data)
                    df.to_parquet(filepath, compression="gzip")
                    self.stats.output_files.append(filepath)
                except ImportError:
                    pass
        meta_path = os.path.join(output_dir, "generation_metadata.json")
        metadata = {
            "generation_id": self.generation_id,
            "created_at": datetime.now().isoformat(),
            "statistics": self.stats.to_dict(),
            "config": {
                "data_types": {
                    k: {"enabled": v.get("enabled", False),
                        "total_count": v.get("total_count", 0)}
                    for k, v in self.data_types_cfg.items()
                }
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        self.stats.output_files.append(meta_path)
        logger.info(f"数据已保存到 {output_dir}")
        return self

    def get_stats(self) -> DatasetStatistics:
        return self.stats

    def clear_checkpoints(self):
        self.checkpoint_mgr.clear_all()
        logger.info("已清除所有检查点")

    @classmethod
    def from_config_file(cls, config_path: str) -> "SyntheticDatasetFactory":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run_pipeline(self, targets: Optional[dict[str, int]] = None,
                     parallel: bool = False) -> DatasetStatistics:
        logger.info("=" * 55)
        logger.info("  青小团 合成数据生成流水线启动")
        logger.info("=" * 55)
        self.generate(targets, parallel)
        self.augment()
        self.validate()
        self.save()
        stats = self.get_stats()
        logger.info("\n" + str(stats))
        return stats
