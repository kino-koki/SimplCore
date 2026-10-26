# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# downloader.py - 大规模数据下载引擎
# 支持从 HuggingFace、URL、S3 下载数据集，断点续传，多线程并行下载，进度报告

import os
import sys
import json
import zstandard
import tarfile
import hashlib
import logging
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Callable, Iterator, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, Dataset, DatasetDict

logger = logging.getLogger("qingxiaotuan.downloader")


@dataclass
class DownloadStats:
    """青小团数据下载统计"""
    total_datasets: int = 0
    completed_datasets: int = 0
    failed_datasets: int = 0
    total_bytes_downloaded: int = 0
    total_samples_downloaded: int = 0
    skipped_datasets: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    per_dataset_stats: Dict[str, Dict] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def throughput_mbps(self) -> float:
        if self.elapsed_seconds > 0:
            return (self.total_bytes_downloaded / 1_048_576) / self.elapsed_seconds
        return 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 数据下载引擎 - 统计报告",
            "=" * 60,
            f"总数据集:          {self.total_datasets}",
            f"已完成:            {self.completed_datasets}",
            f"失败:              {self.failed_datasets}",
            f"跳过:              {self.skipped_datasets}",
            f"总下载字节:        {self._format_bytes(self.total_bytes_downloaded)}",
            f"总下载样本:        {self.total_samples_downloaded:,}",
            f"总耗时:            {self.elapsed_seconds:.2f} 秒",
            f"吞吐量:            {self.throughput_mbps:.2f} MB/s",
            "-" * 60,
        ]
        for ds_name, ds_stats in self.per_dataset_stats.items():
            lines.append(
                f"  {ds_name}: {ds_stats.get('samples', 0):,} 样本, "
                f"{self._format_bytes(ds_stats.get('bytes', 0))}, "
                f"状态: {ds_stats.get('status', 'unknown')}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"


class DownloadProgressTracker:
    """青小团下载进度追踪器 - 支持多线程安全的进度报告"""

    def __init__(self, total: int, desc: str = "下载进度"):
        self._lock = threading.Lock()
        self._total = total
        self._current = 0
        self._desc = desc
        self._pbar: Optional[tqdm] = None

    def __enter__(self):
        self._pbar = tqdm(
            total=self._total,
            desc=self._desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        )
        return self

    def __exit__(self, *args):
        if self._pbar:
            self._pbar.close()

    def update(self, n: int = 1):
        with self._lock:
            self._current += n
            if self._pbar:
                self._pbar.update(n)

    @property
    def progress(self) -> float:
        if self._total > 0:
            return self._current / self._total * 100
        return 0.0


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """计算文件哈希值，用于断点续传完整性校验"""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def decompress_zst(filepath: str, output_path: str) -> str:
    """解压缩 .zst 文件"""
    dctx = zstandard.ZstdDecompressor(max_window_size=2_147_483_648)
    with open(filepath, "rb") as src, open(output_path, "wb") as dst:
        dctx.copy_stream(src, dst)
    return output_path


def extract_tar_gz(filepath: str, output_dir: str) -> List[str]:
    """解压缩 tar.gz 文件"""
    extracted = []
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=output_dir)
        for member in tar.getmembers():
            extracted.append(os.path.join(output_dir, member.name))
    return extracted


def stream_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
    """流式读取 JSONL 文件，青小团大数据集不存内存"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class DataDownloader:
    """青小团大规模数据下载器
    支持数据源: HuggingFace, HTTP/HTTPS URL, S3
    支持格式: tar.gz, zst, jsonl
    支持断点续传、多线程下载、进度报告
    """

    SUPPORTED_DATASETS = {
        "c4": ("allenai/c4", "huggingface"),
        "the_pile": ("EleutherAI/the_pile", "huggingface"),
        "refined_web": ("tiiuae/falcon-refinedweb", "huggingface"),
        "starcoder": ("bigcode/starcoderdata", "huggingface"),
        "wudao": ("BAAI/WuDaoCorpus2.0", "huggingface"),
        "mc4": ("allenai/mc4", "huggingface"),
        "pile_cc": ("EleutherAI/pile-cc", "huggingface"),
        "bookcorpus": ("bookcorpus", "huggingface"),
        "wikipedia": ("wikipedia", "huggingface"),
        "github_code": ("codeparrot/github-code", "huggingface"),
        "the_stack": ("bigcode/the-stack", "huggingface"),
        "openwebmath": ("openwebmath", "huggingface"),
        "math_dataset": ("math_dataset", "huggingface"),
        "proof_pile": ("EleutherAI/proof_pile", "huggingface"),
        "numina_math": ("AI-MO/numina_math", "huggingface"),
    }

    def __init__(
        self,
        output_dir: str = "./data/raw",
        temp_dir: str = "./tmp/downloads",
        num_workers: int = 8,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        resume_enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.resume_enabled = resume_enabled
        self.stats = DownloadStats()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Qingxiaotuan-DataPipeline/1.0 (Qingnai Technology)"
        })

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._resume_dir = self.temp_dir / ".resume"
        self._resume_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"青小团数据下载器初始化: output={output_dir}, workers={num_workers}")

    def _get_resume_file(self, dataset_name: str) -> Path:
        resume_name = f"{dataset_name}.resume.json"
        return self._resume_dir / resume_name

    def _save_resume_state(self, dataset_name: str, state: Dict):
        if not self.resume_enabled:
            return
        resume_file = self._get_resume_file(dataset_name)
        with open(resume_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_resume_state(self, dataset_name: str) -> Optional[Dict]:
        if not self.resume_enabled:
            return None
        resume_file = self._get_resume_file(dataset_name)
        if resume_file.exists():
            with open(resume_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _clear_resume_state(self, dataset_name: str):
        resume_file = self._get_resume_file(dataset_name)
        if resume_file.exists():
            resume_file.unlink()

    def download_from_huggingface(
        self,
        dataset_name: str,
        hf_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        output_format: str = "jsonl",
    ) -> Dict[str, Any]:
        """从 HuggingFace Datasets 下载数据集"""
        ds_output_dir = self.output_dir / dataset_name
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = ds_output_dir / f"{split}.{output_format}"

        resume_state = self._load_resume_state(dataset_name)
        if resume_state and resume_state.get("completed"):
            logger.info(f"青小团检测到 {dataset_name}[{split}] 已下载完成，跳过")
            self.stats.skipped_datasets += 1
            return {"status": "skipped", "path": str(output_file), "samples": resume_state.get("samples", 0)}

        logger.info(f"青小团开始下载数据集: {hf_path}[{split}] -> {output_file}")
        try:
            ds = load_dataset(hf_path, split=split, streaming=True)
            sample_count = 0
            byte_count = 0

            with open(output_file, "w", encoding="utf-8") as f:
                for i, example in enumerate(ds):
                    if max_samples and i >= max_samples:
                        break
                    line = json.dumps(example, ensure_ascii=False)
                    f.write(line + "\n")
                    sample_count += 1
                    byte_count += len(line.encode("utf-8"))
                    if sample_count % 10000 == 0:
                        logger.debug(f"  已下载 {dataset_name}: {sample_count:,} 样本")

            self._save_resume_state(dataset_name, {
                "completed": True,
                "samples": sample_count,
                "bytes": byte_count,
                "timestamp": datetime.now().isoformat(),
            })
            self.stats.completed_datasets += 1
            self.stats.total_samples_downloaded += sample_count
            self.stats.total_bytes_downloaded += byte_count
            self.stats.per_dataset_stats[dataset_name] = {
                "samples": sample_count, "bytes": byte_count, "status": "completed"
            }

            logger.info(f"青小团数据集 {dataset_name} 下载完成: {sample_count:,} 样本, {self._format_bytes(byte_count)}")
            return {"status": "completed", "path": str(output_file), "samples": sample_count}

        except Exception as e:
            logger.error(f"青小团数据集 {dataset_name} 下载失败: {e}")
            self.stats.failed_datasets += 1
            self.stats.per_dataset_stats[dataset_name] = {"status": "failed", "error": str(e)}
            return {"status": "failed", "error": str(e)}

    def download_from_url(
        self,
        url: str,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """从 HTTP/HTTPS URL 下载文件，支持断点续传"""
        if output_path is None:
            output_path = str(self.temp_dir)
        if filename is None:
            filename = os.path.basename(urlparse(url).path) or "download"

        dest_path = Path(output_path) / filename
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        logger.info(f"青小团开始URL下载: {url} -> {dest_path}")

        existing_size = 0
        if temp_path.exists() and self.resume_enabled:
            existing_size = temp_path.stat().st_size
            logger.info(f"  检测到已下载部分: {self._format_bytes(existing_size)}，继续断点续传")

        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

        try:
            with self._session.get(url, headers=headers, stream=True, timeout=(30, 3600)) as resp:
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0)) + existing_size

                with DownloadProgressTracker(total=total_size, desc=f"下载 {filename}") as tracker:
                    with open(temp_path, "ab" if existing_size > 0 else "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192 * 1024):
                            if chunk:
                                f.write(chunk)
                                tracker.update(len(chunk))

                temp_path.rename(dest_path)

                file_size = dest_path.stat().st_size
                logger.info(f"青小团URL下载完成: {filename}, {self._format_bytes(file_size)}")
                return {"status": "completed", "path": str(dest_path), "bytes": file_size}

        except requests.exceptions.RequestException as e:
            logger.error(f"青小团URL下载失败 [{url}]: {e}")
            return {"status": "failed", "error": str(e)}

    def download_from_s3(
        self,
        s3_uri: str,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """从 S3 下载数据 (兼容 MinIO, COS, OSS)"""
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            logger.error("青小团S3下载需要安装 boto3: pip install boto3")
            return {"status": "failed", "error": "boto3 not installed"}

        parsed = urlparse(s3_uri)
        bucket = parsed.hostname
        key = parsed.path.lstrip("/")

        if output_path is None:
            output_path = str(self.temp_dir)
        if filename is None:
            filename = os.path.basename(key) or "s3_download"

        dest_path = Path(output_path) / filename
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        config = Config(connect_timeout=30, read_timeout=3600, max_pool_connections=50)
        client = boto3.client("s3", config=config)

        existing_size = 0
        if temp_path.exists() and self.resume_enabled:
            existing_size = temp_path.stat().st_size

        logger.info(f"青小团开始S3下载: s3://{bucket}/{key} -> {dest_path}")

        try:
            if existing_size > 0:
                resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={existing_size}-")
            else:
                resp = client.get_object(Bucket=bucket, Key=key)

            total_size = (resp["ContentLength"] or 0) + existing_size
            body = resp["Body"]

            with DownloadProgressTracker(total=total_size, desc=f"S3 {filename}") as tracker:
                with open(temp_path, "ab" if existing_size > 0 else "wb") as f:
                    for chunk in iter(lambda: body.read(8192 * 1024), b""):
                        f.write(chunk)
                        tracker.update(len(chunk))

            temp_path.rename(dest_path)
            file_size = dest_path.stat().st_size
            logger.info(f"青小团S3下载完成: {filename}, {self._format_bytes(file_size)}")
            return {"status": "completed", "path": str(dest_path), "bytes": file_size}

        except Exception as e:
            logger.error(f"青小团S3下载失败 [s3://{bucket}/{key}]: {e}")
            return {"status": "failed", "error": str(e)}

    def download_dataset_by_name(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """通过数据集名称自动下载（青小团预注册的数据集）"""
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"青小团不支持的数据集: {dataset_name}. 支持的: {list(self.SUPPORTED_DATASETS.keys())}")

        hf_path, source_type = self.SUPPORTED_DATASETS[dataset_name]

        if source_type == "huggingface":
            return self.download_from_huggingface(
                dataset_name=dataset_name,
                hf_path=hf_path,
                split=split,
                max_samples=max_samples,
            )
        raise ValueError(f"不支持的数据源类型: {source_type}")

    def download_multiple(
        self,
        datasets_config: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """批量并行下载多个数据集"""
        self.stats.start_time = datetime.now()
        self.stats.total_datasets = len(datasets_config)

        logger.info(f"青小团开始批量下载 {len(datasets_config)} 个数据集, 工作线程={self.num_workers}")
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_map = {}

            for ds_config in datasets_config:
                ds_name = ds_config.get("name", "unknown")
                source = ds_config.get("source", ds_config.get("url", ""))

                if source == "huggingface" or ds_name in self.SUPPORTED_DATASETS:
                    future = executor.submit(
                        self.download_dataset_by_name,
                        dataset_name=ds_name,
                        split=ds_config.get("split", "train"),
                        max_samples=ds_config.get("max_samples", None),
                    )
                elif source.startswith("s3://"):
                    future = executor.submit(
                        self.download_from_s3,
                        s3_uri=source,
                        output_path=ds_config.get("output_dir"),
                        filename=ds_config.get("filename"),
                    )
                elif source.startswith("http://") or source.startswith("https://"):
                    future = executor.submit(
                        self.download_from_url,
                        url=source,
                        output_path=ds_config.get("output_dir"),
                        filename=ds_config.get("filename"),
                    )
                else:
                    future = executor.submit(
                        self.download_from_huggingface,
                        dataset_name=ds_name,
                        hf_path=source,
                        split=ds_config.get("split", "train"),
                        max_samples=ds_config.get("max_samples", None),
                    )

                future_map[future] = ds_name

            with tqdm(total=len(future_map), desc="青小团批量下载", ncols=80) as pbar:
                for future in as_completed(future_map):
                    ds_name = future_map[future]
                    try:
                        result = future.result()
                        results[ds_name] = result
                    except Exception as e:
                        logger.error(f"青小团数据集 {ds_name} 下载异常: {e}")
                        results[ds_name] = {"status": "failed", "error": str(e)}
                        self.stats.failed_datasets += 1
                    pbar.update(1)

        self.stats.end_time = datetime.now()
        logger.info(f"\n{self.stats.summary()}")
        return results

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息（配置数量、大小等）"""
        if dataset_name not in self.SUPPORTED_DATASETS:
            return {"error": f"未知数据集: {dataset_name}"}

        hf_path, _ = self.SUPPORTED_DATASETS[dataset_name]
        try:
            configs = get_dataset_config_names(hf_path)
            return {"name": dataset_name, "hf_path": hf_path, "configs": configs, "num_configs": len(configs)}
        except Exception as e:
            return {"name": dataset_name, "hf_path": hf_path, "error": str(e)}

    def auto_detect_and_download(
        self,
        dataset_list: List[str],
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """自动检测并下载数据集列表"""
        if max_workers:
            self.num_workers = max_workers

        configs = []
        for name in dataset_list:
            configs.append({"name": name, "source": "huggingface", "split": "train"})

        return self.download_multiple(configs)

    def verify_download(self, dataset_name: str) -> bool:
        """验证下载完整性"""
        ds_dir = self.output_dir / dataset_name
        if not ds_dir.exists():
            return False
        jsonl_files = list(ds_dir.glob("*.jsonl"))
        if not jsonl_files:
            return False
        for jf in jsonl_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)
            except (json.JSONDecodeError, Exception):
                return False
        return True

    def clean_temp_files(self):
        """清理临时文件"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("青小团临时文件已清理")

    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"


class MultiProcessDownloader:
    """青小团多进程下载器 - 用于超大规模数据集的并行下载"""

    def __init__(self, num_processes: int = 4):
        self.num_processes = num_processes
        self._downloaders = [
            DataDownloader(num_workers=2) for _ in range(num_processes)
        ]

    def download_sharded(
        self,
        dataset_name: str,
        hf_path: str,
        num_shards: int = 10,
        split: str = "train",
    ) -> List[Dict[str, Any]]:
        """分片并行下载大型数据集"""
        logger.info(f"青小团多进程分片下载 {dataset_name}, 分片数={num_shards}")

        shard_size = max(1, num_shards // self.num_processes)
        shard_groups = [
            list(range(i, min(i + shard_size, num_shards)))
            for i in range(0, num_shards, shard_size)
        ]

        results = []
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for group_idx, shard_ids in enumerate(shard_groups):
                downloader = self._downloaders[group_idx % self.num_processes]
                for shard_id in shard_ids:
                    future = executor.submit(
                        downloader.download_from_huggingface,
                        dataset_name=f"{dataset_name}_shard_{shard_id:04d}",
                        hf_path=hf_path,
                        split=split,
                    )
                    futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"多进程 {dataset_name}"):
                results.append(future.result())

        return results

    def cleanup(self):
        for d in self._downloaders:
            d.clean_temp_files()
