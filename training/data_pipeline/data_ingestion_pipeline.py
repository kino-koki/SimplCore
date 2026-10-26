# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# data_ingestion_pipeline.py - 数据预处理完整流水线编排引擎
# 整合下载、过滤、去重、分词、混合、缓存全流程，支持分布式协调和断点恢复

import os
import sys
import json
import time
import yaml
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import numpy as np
from tqdm import tqdm

from downloader import DataDownloader, DownloadStats
from quality_filter import QualityFilter, FilterConfig, FilterStats
from deduplicator import Deduplicator, DedupStats, MultiStageDeduplicator
from tokenizer_pipeline import TokenizerPipeline, TokenizationStats
from data_mixer import DataMixer, MixingStats, CategoryBuilder
from cache_manager import CacheManager, CacheStats

logger = logging.getLogger("qingxiaotuan.pipeline")


class PipelineStage(Enum):
    """青小团流水线阶段枚举"""
    INIT = "init"
    DOWNLOAD = "download"
    FILTER = "filter"
    DEDUP = "dedup"
    TOKENIZE = "tokenize"
    MIX = "mix"
    CACHE = "cache"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineState:
    """青小团流水线状态"""
    current_stage: PipelineStage = PipelineStage.INIT
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    stages_completed: List[str] = field(default_factory=list)
    stage_times: Dict[str, float] = field(default_factory=dict)
    stage_errors: Dict[str, str] = field(default_factory=dict)
    is_running: bool = False
    is_paused: bool = False

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 数据流水线 - 状态报告",
            "=" * 60,
            f"当前阶段:        {self.current_stage.value}",
            f"运行状态:        {'运行中' if self.is_running else '已停止'}",
            f"已用时间:        {self.elapsed_seconds:.2f} 秒",
            "-" * 60,
            "阶段完成情况:",
        ]
        for stage in self.stages_completed:
            stage_time = self.stage_times.get(stage, 0)
            err = self.stage_errors.get(stage)
            status = f"完成 ({stage_time:.2f}s)" if not err else f"失败: {err}"
            lines.append(f"  {stage}: {status}")

        if not self.stages_completed:
            lines.append("  (尚无阶段完成)")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """青小团流水线配置"""
    project_name: str = "qingxiaotuan_70b_moe"
    seed: int = 42
    num_workers: int = 32
    output_dir: str = "./output"
    temp_dir: str = "./tmp"
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay_sec: float = 5.0

    download_config: Dict = field(default_factory=dict)
    filter_config: Dict = field(default_factory=dict)
    dedup_config: Dict = field(default_factory=dict)
    tokenizer_config: Dict = field(default_factory=dict)
    mixing_config: Dict = field(default_factory=dict)
    cache_config: Dict = field(default_factory=dict)
    output_config: Dict = field(default_factory=dict)
    distributed_config: Dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()
        global_cfg = data.get("global", {})

        config.project_name = global_cfg.get("project_name", config.project_name)
        config.seed = global_cfg.get("seed", config.seed)
        config.num_workers = global_cfg.get("num_workers", config.num_workers)
        config.output_dir = global_cfg.get("output_dir", config.output_dir)
        config.temp_dir = global_cfg.get("temp_dir", config.temp_dir)
        config.log_level = global_cfg.get("log_level", config.log_level)
        config.max_retries = global_cfg.get("max_retries", config.max_retries)
        config.retry_delay_sec = global_cfg.get("retry_delay_sec", config.retry_delay_sec)

        config.download_config = data.get("datasets", {})
        config.filter_config = data.get("quality_filter", {})
        config.dedup_config = data.get("deduplication", {})
        config.tokenizer_config = data.get("tokenization", {})
        config.mixing_config = data.get("mixing", {})
        config.cache_config = data.get("cache", {})
        config.output_config = data.get("output", {})
        config.distributed_config = data.get("distributed", {})

        return config


class DataIngestionPipeline:
    """青小团数据预处理完整流水线
    编排下载、质量过滤、去重、分词、混合、缓存全流程
    支持错误重试、断点恢复、分布式协调
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[str] = None,
    ):
        if config_path:
            self.config = PipelineConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfig()

        self.state = PipelineState()
        self.state_file = Path(self.config.output_dir) / ".pipeline_state.json"

        self._setup_logging()

        self.downloader: Optional[DataDownloader] = None
        self.filter: Optional[QualityFilter] = None
        self.dedup: Optional[Deduplicator] = None
        self.multi_dedup: Optional[MultiStageDeduplicator] = None
        self.tokenizer_pipeline: Optional[TokenizerPipeline] = None
        self.mixer: Optional[DataMixer] = None
        self.cache_manager: Optional[CacheManager] = None

        self._download_stats: Optional[DownloadStats] = None
        self._filter_stats: Optional[FilterStats] = None
        self._dedup_stats: Optional[DedupStats] = None
        self._tokenize_stats: Optional[TokenizationStats] = None
        self._mix_stats: Optional[MixingStats] = None
        self._cache_stats: Optional[CacheStats] = None

        self._load_state()
        self._init_components()

        logger.info(f"青小团数据流水线初始化完成: project={self.config.project_name}")

    def _setup_logging(self):
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "pipeline.log"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(str(log_file), encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logger.info(f"青小团日志初始化: {log_file}")

    def _init_components(self):
        output_dir = self.config.output_dir
        temp_dir = self.config.temp_dir
        num_workers = self.config.num_workers

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        self.downloader = DataDownloader(
            output_dir=os.path.join(output_dir, "raw"),
            temp_dir=os.path.join(temp_dir, "downloads"),
            num_workers=max(1, num_workers // 4),
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay_sec,
        )

        filter_cfg = self.config.filter_config
        self.filter = QualityFilter(
            config=FilterConfig(
                lang_detect_enabled=filter_cfg.get("lang_detect", True),
                target_languages=filter_cfg.get("target_languages", ["zh", "en"]),
                ngram_repeat_enabled=filter_cfg.get("ngram_repeat", {}).get("enabled", True),
                ngram_max_ratio=filter_cfg.get("ngram_repeat", {}).get("max_ratio", 0.30),
                url_max_ratio=filter_cfg.get("url_ratio", {}).get("max_ratio", 0.40),
                min_text_length=filter_cfg.get("min_text_length", 50),
                max_text_length=filter_cfg.get("max_text_length", 1000000),
            ),
        )

        dedup_cfg = self.config.dedup_config
        minhash_cfg = dedup_cfg.get("minhash", {})
        self.dedup = Deduplicator(
            num_perm=minhash_cfg.get("num_perm", 128),
            threshold=minhash_cfg.get("threshold", 0.80),
            ngram_size=minhash_cfg.get("ngram_size", 5),
            paragraph_dedup=dedup_cfg.get("paragraph_level", True),
            cache_dir=os.path.join(temp_dir, "dedup"),
            num_workers=num_workers,
        )
        self.multi_dedup = MultiStageDeduplicator(
            num_perm=minhash_cfg.get("num_perm", 128),
            threshold=minhash_cfg.get("threshold", 0.80),
            num_workers=num_workers,
        )

        tok_cfg = self.config.tokenizer_config
        self.tokenizer_pipeline = TokenizerPipeline(
            tokenizer_path=tok_cfg.get("tokenizer_path", "./models/tokenizer"),
            tokenizer_type=tok_cfg.get("tokenizer_type", "huggingface"),
            max_length=tok_cfg.get("max_length", 524288),
            truncation=tok_cfg.get("truncation", True),
            padding=tok_cfg.get("padding", False),
            sequence_packing=tok_cfg.get("sequence_packing", True),
            packing_max_sequences=tok_cfg.get("packing_max_sequences", 8),
            num_workers=tok_cfg.get("num_workers", num_workers),
            output_dir=os.path.join(output_dir, "tokenized"),
            compression=tok_cfg.get("compression", "zstd"),
        )

        mix_cfg = self.config.mixing_config
        self.mixer = DataMixer(
            strategy=mix_cfg.get("strategy", "weighted"),
            epoch_less=mix_cfg.get("epoch_less", True),
            shuffle_buffer_size=mix_cfg.get("shuffle_buffer_size", 1_000_000),
            refresh_weights_interval=mix_cfg.get("refresh_weights_interval", 10000),
            seed=self.config.seed,
        )

        cache_cfg = self.config.cache_config
        self.cache_manager = CacheManager(
            l1_max_size_gb=cache_cfg.get("l1_cache", {}).get("max_size_gb", 64.0),
            l2_cache_dir=cache_cfg.get("l2_cache", {}).get("cache_dir", "./cache/disk"),
            l2_max_size_gb=cache_cfg.get("l2_cache", {}).get("max_size_gb", 1024.0),
            l3_backend=cache_cfg.get("l3_cache", {}).get("backend", "redis"),
            mmap_enabled=cache_cfg.get("mmap_enabled", True),
            serialization=cache_cfg.get("serialization", "arrow"),
        )

    def _save_state(self):
        state_data = {
            "current_stage": self.state.current_stage.value,
            "start_time": self.state.start_time,
            "stages_completed": self.state.stages_completed,
            "stage_times": self.state.stage_times,
            "stage_errors": self.state.stage_errors,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                self.state.current_stage = PipelineStage(state_data.get("current_stage", "init"))
                self.state.start_time = state_data.get("start_time")
                self.state.stages_completed = state_data.get("stages_completed", [])
                self.state.stage_times = state_data.get("stage_times", {})
                self.state.stage_errors = state_data.get("stage_errors", {})
                logger.info(f"青小团加载流水线状态: 当前阶段={self.state.current_stage.value}, "
                          f"已完成={self.state.stages_completed}")
            except Exception as e:
                logger.warning(f"青小团加载状态失败: {e}")

    def _stage_runner(self, stage_name: str, func: Callable, *args, **kwargs) -> Any:
        if stage_name in self.state.stages_completed:
            logger.info(f"青小团阶段 {stage_name} 已完成，跳过")
            return None

        self.state.current_stage = PipelineStage(stage_name.upper())
        self.state.is_running = True
        self._save_state()

        logger.info(f"{'='*60}")
        logger.info(f"青小团开始阶段: {stage_name}")
        logger.info(f"{'='*60}")

        stage_start = time.time()
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - stage_start
                self.state.stages_completed.append(stage_name)
                self.state.stage_times[stage_name] = elapsed
                logger.info(f"青小团阶段 {stage_name} 完成, 耗时 {elapsed:.2f} 秒")
                self._save_state()
                return result

            except Exception as e:
                last_error = e
                elapsed = time.time() - stage_start
                logger.error(f"青小团阶段 {stage_name} 尝试 {attempt}/{self.config.max_retries} 失败: {e}")
                traceback.print_exc()

                if attempt < self.config.max_retries:
                    wait = self.config.retry_delay_sec * attempt
                    logger.info(f"青小团 {wait} 秒后重试...")
                    time.sleep(wait)
                else:
                    self.state.stage_errors[stage_name] = str(e)
                    self.state.current_stage = PipelineStage.FAILED
                    self.state.is_running = False
                    self._save_state()
                    raise RuntimeError(f"青小团阶段 {stage_name} 在 {self.config.max_retries} 次尝试后失败") from e

    def run_pipeline(
        self,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """运行完整流水线"""
        self.state.start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"青小团数据预处理流水线启动")
        logger.info(f"{'='*60}")

        all_stages = ["download", "filter", "dedup", "tokenize", "mix", "cache"]
        if stages:
            all_stages = [s for s in all_stages if s in stages]

        results = {}

        if "download" in all_stages:
            results["download"] = self._stage_runner("download", self._run_download)
        if "filter" in all_stages:
            results["filter"] = self._stage_runner("filter", self._run_filter)
        if "dedup" in all_stages:
            results["dedup"] = self._stage_runner("dedup", self._run_dedup)
        if "tokenize" in all_stages:
            results["tokenize"] = self._stage_runner("tokenize", self._run_tokenize)
        if "mix" in all_stages:
            results["mix"] = self._stage_runner("mix", self._run_mix)
        if "cache" in all_stages:
            results["cache"] = self._stage_runner("cache", self._run_cache)

        self.state.current_stage = PipelineStage.COMPLETE
        self.state.end_time = time.time()
        self.state.is_running = False
        self._save_state()

        total_time = self.state.end_time - self.state.start_time
        logger.info(f"{'='*60}")
        logger.info(f"青小团流水线完成! 总耗时: {total_time:.2f} 秒")
        logger.info(f"{'='*60}")
        logger.info(f"\n{self.state.summary()}")

        if self._download_stats:
            logger.info(f"\n{self._download_stats.summary()}")
        if self._filter_stats:
            logger.info(f"\n{self._filter_stats.summary()}")
        if self._dedup_stats:
            logger.info(f"\n{self._dedup_stats.summary()}")
        if self._tokenize_stats:
            logger.info(f"\n{self._tokenize_stats.summary()}")
        if self._mix_stats:
            logger.info(f"\n{self._mix_stats.summary()}")
        if self._cache_stats:
            logger.info(f"\n{self._cache_stats.summary()}")

        return results

    def _run_download(self) -> DownloadStats:
        ds_config = self.config.download_config
        datasets_to_download = []

        for category_key, category_val in ds_config.items():
            if isinstance(category_val, dict) and "sources" in category_val:
                for src in category_val["sources"]:
                    if isinstance(src, dict):
                        datasets_to_download.append({
                            "name": src.get("name", "unknown"),
                            "source": "huggingface",
                            "split": src.get("split", "train"),
                            "max_samples": src.get("max_samples"),
                        })

        logger.info(f"青小团需要下载 {len(datasets_to_download)} 个数据集")
        self.downloader.download_multiple(datasets_to_download)
        self._download_stats = self.downloader.stats
        return self._download_stats

    def _run_filter(self) -> FilterStats:
        raw_dir = Path(self.config.output_dir) / "raw"
        filtered_dir = Path(self.config.output_dir) / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)

        total_filtered = 0
        total_passed = 0

        for ds_dir in raw_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for jsonl_file in ds_dir.glob("*.jsonl"):
                output_file = filtered_dir / f"{ds_dir.name}_filtered.jsonl"
                logger.info(f"青小团过滤: {jsonl_file} -> {output_file}")

                stats = self.filter.filter_from_jsonl(
                    str(jsonl_file),
                    str(output_file),
                    num_workers=self.config.num_workers,
                )
                total_filtered += stats.filtered_out
                total_passed += stats.passed

        logger.info(f"青小团过滤完成: 通过={total_passed}, 过滤={total_filtered}")
        self._filter_stats = self.filter.stats
        return self._filter_stats

    def _run_dedup(self) -> DedupStats:
        filtered_dir = Path(self.config.output_dir) / "filtered"
        deduped_dir = Path(self.config.output_dir) / "deduped"
        deduped_dir.mkdir(parents=True, exist_ok=True)

        all_docs = []
        for jsonl_file in filtered_dir.glob("*_filtered.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_docs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        logger.info(f"青小团加载 {len(all_docs)} 个文档进行去重")

        deduped = self.multi_dedup.process(all_docs)

        output_file = deduped_dir / "deduped.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in deduped:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"青小团去重完成: {len(all_docs)} -> {len(deduped)}")
        self._dedup_stats = self.multi_dedup.dedup.stats
        return self._dedup_stats

    def _run_tokenize(self) -> TokenizationStats:
        deduped_dir = Path(self.config.output_dir) / "deduped"
        jsonl_file = deduped_dir / "deduped.jsonl"

        if not jsonl_file.exists():
            logger.warning(f"青小团去重文件不存在: {jsonl_file}，跳过分词")
            return TokenizationStats()

        docs = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        output_formats = self.config.tokenizer_config.get(
            "output_formats", ["arrow", "parquet", "jsonl", "mmap"]
        )

        self._tokenize_stats = self.tokenizer_pipeline.process_multiformat(
            docs,
            formats=output_formats,
            dataset_name="qingxiaotuan_final",
        )

        return self._tokenize_stats

    def _run_mix(self) -> MixingStats:
        categories = self.config.mixing_config.get("categories", {})
        cat_weights = {}
        for cat_name, cat_cfg in categories.items():
            cat_weights[cat_name] = cat_cfg.get("weight", 0)

        self.mixer.set_category_weights(cat_weights)

        deduped_dir = Path(self.config.output_dir) / "deduped"
        jsonl_file = deduped_dir / "deduped.jsonl"

        if jsonl_file.exists():
            docs = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            docs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            general_size = int(len(docs) * 0.7)
            code_size = int(len(docs) * 0.2)
            math_size = len(docs) - general_size - code_size

            self.mixer.add_source(
                "general_corpus", data=docs[:general_size],
                weight=0.70, category="general",
            )
            self.mixer.add_source(
                "code_corpus", data=docs[general_size:general_size + code_size],
                weight=0.20, category="code",
            )
            self.mixer.add_source(
                "math_corpus", data=docs[general_size + code_size:],
                weight=0.10, category="math",
            )

            sampled = []
            for i, doc in enumerate(self.mixer.generate_with_shuffle()):
                if i >= 100000:
                    break
                sampled.append(doc)

            mix_output = Path(self.config.output_dir) / "mixed" / "mixed_output.jsonl"
            mix_output.parent.mkdir(parents=True, exist_ok=True)
            with open(mix_output, "w", encoding="utf-8") as f:
                for doc in sampled:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            logger.info(f"青小团混合完成: {len(sampled)} 样本")

        self._mix_stats = self.mixer.get_mixing_report()
        return self._mix_stats

    def _run_cache(self) -> CacheStats:
        cache_cfg = self.config.cache_config
        if not cache_cfg.get("enabled", True):
            logger.info("青小团缓存已禁用")
            return CacheStats()

        tokenized_dir = Path(self.config.output_dir) / "tokenized"
        if tokenized_dir.exists():
            for arrow_file in tokenized_dir.glob("*.arrow"):
                logger.info(f"青小团缓存 tokenized 数据: {arrow_file}")
                cache_key = f"tokenized:{arrow_file.stem}"
                self.cache_manager.put(cache_key, str(arrow_file), level=2)

        if cache_cfg.get("warmup_on_start", False):
            self.cache_manager.warmup(list(self.cache_manager.l2._metadata.keys())[:100])

        self._cache_stats = self.cache_manager.get_stats()
        return self._cache_stats

    def run_distributed(
        self,
        num_nodes: int = 1,
        node_rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 29500,
    ):
        """分布式运行流水线"""
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=num_nodes,
            rank=node_rank,
        )

        try:
            if node_rank == 0:
                logger.info("青小团主节点开始运行流水线")
                self.run_pipeline()
            else:
                logger.info(f"青小团工作节点 {node_rank} 等待主节点")
                dist.barrier()

        finally:
            dist.destroy_process_group()

    def resume(self) -> Dict[str, Any]:
        """从上次中断点恢复流水线"""
        logger.info(f"青小团从阶段 {self.state.current_stage.value} 恢复流水线")

        completed = set(self.state.stages_completed)
        remaining = []

        all_stages = ["download", "filter", "dedup", "tokenize", "mix", "cache"]
        current_idx = 0

        for i, stage in enumerate(all_stages):
            if stage in completed:
                continue
            remaining = all_stages[i:]
            break
        else:
            logger.info("青小团所有阶段已完成")
            return {}

        return self.run_pipeline(stages=remaining)

    def get_report(self) -> Dict[str, Any]:
        """获取完整报告"""
        report = {
            "project": self.config.project_name,
            "state": {
                "current_stage": self.state.current_stage.value,
                "is_running": self.state.is_running,
                "stages_completed": self.state.stages_completed,
                "total_time": self.state.elapsed_seconds,
            },
            "stats": {},
        }

        if self._download_stats:
            report["stats"]["download"] = {
                "total_datasets": self._download_stats.total_datasets,
                "completed": self._download_stats.completed_datasets,
                "failed": self._download_stats.failed_datasets,
                "total_samples": self._download_stats.total_samples_downloaded,
                "throughput_mbps": self._download_stats.throughput_mbps,
            }

        if self._filter_stats:
            report["stats"]["filter"] = {
                "total": self._filter_stats.total_documents,
                "passed": self._filter_stats.passed,
                "filtered": self._filter_stats.filtered_out,
                "pass_rate": self._filter_stats.pass_rate,
            }

        if self._dedup_stats:
            report["stats"]["dedup"] = {
                "total": self._dedup_stats.total_documents,
                "final": self._dedup_stats.final_documents,
                "exact_removed": self._dedup_stats.exact_duplicates_removed,
                "fuzzy_removed": self._dedup_stats.fuzzy_duplicates_removed,
                "dedup_rate": self._dedup_stats.overall_dedup_rate,
            }

        if self._tokenize_stats:
            report["stats"]["tokenize"] = {
                "documents": self._tokenize_stats.total_documents,
                "tokens": self._tokenize_stats.total_tokens,
                "sequences": self._tokenize_stats.total_sequences,
                "avg_tokens_per_doc": self._tokenize_stats.avg_tokens_per_doc,
            }

        if self._mix_stats:
            report["stats"]["mix"] = {
                "total_samples": self._mix_stats.total_samples,
                "adjustments": self._mix_stats.adjustment_count,
                "per_category": dict(self._mix_stats.per_category_samples),
            }

        if self._cache_stats:
            report["stats"]["cache"] = {
                "hits": self._cache_stats.hits,
                "misses": self._cache_stats.misses,
                "hit_rate": self._cache_stats.hit_rate,
                "l1_size": self._cache_stats.l1_size,
                "l2_size": self._cache_stats.l2_size,
            }

        return report

    def print_report(self):
        report = self.get_report()
        print("\n" + "=" * 60)
        print(f"青小团 ({self.config.project_name}) 数据管线报告")
        print("=" * 60)
        print(f"状态: {'运行中' if report['state']['is_running'] else '空闲'}")
        print(f"当前阶段: {report['state']['current_stage']}")
        print(f"总耗时: {report['state']['total_time']:.2f} 秒")

        for stage_name, stats in report["stats"].items():
            print(f"\n--- {stage_name.upper()} ---")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                elif isinstance(v, dict):
                    print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
                else:
                    print(f"  {k}: {v}")
        print("=" * 60)

    def cleanup(self):
        """清理临时文件和缓存"""
        if self.downloader:
            self.downloader.clean_temp_files()
        if self.cache_manager:
            self.cache_manager.clear()
        if self.dedup:
            self.dedup.reset()

        import shutil
        temp_dir = Path(self.config.temp_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("青小团流水线清理完成")


def run_pipeline_from_config(config_path: str) -> DataIngestionPipeline:
    """从配置文件运行流水线"""
    config = PipelineConfig.from_yaml(config_path)
    pipeline = DataIngestionPipeline(config=config)
    pipeline.run_pipeline()
    return pipeline


def main():
    """青小团流水线入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="青小团数据预处理流水线")
    parser.add_argument("--config", type=str, default="pipeline_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--stages", type=str, nargs="+",
                        choices=["download", "filter", "dedup", "tokenize", "mix", "cache"],
                        help="指定要运行的阶段")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断点恢复")
    parser.add_argument("--report", action="store_true",
                        help="仅打印报告")
    parser.add_argument("--cleanup", action="store_true",
                        help="清理临时文件")
    parser.add_argument("--distributed", action="store_true",
                        help="分布式模式")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)

    args = parser.parse_args()

    pipeline = DataIngestionPipeline(config_path=args.config)

    if args.report:
        pipeline.print_report()
        return

    if args.cleanup:
        pipeline.cleanup()
        return

    if args.resume:
        pipeline.resume()
    elif args.distributed:
        pipeline.run_distributed(
            num_nodes=args.num_nodes,
            node_rank=args.node_rank,
        )
    else:
        pipeline.run_pipeline(stages=args.stages)

    pipeline.print_report()


if __name__ == "__main__":
    main()
