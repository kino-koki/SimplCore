# 青奈科技 (Qingnai Technology) - 青小团 (Qingxiaotuan) 大模型数据管线
# cache_manager.py - 多级缓存管理系统
# 支持 L1(内存) + L2(磁盘) + L3(分布式) 三级缓存，LRU+LFU混合淘汰，mmap零拷贝

import os
import io
import json
import time
import pickle
import shutil
import struct
import logging
import hashlib
import threading
import mmap as mmap_lib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

logger = logging.getLogger("qingxiaotuan.cache_manager")


@dataclass
class CacheEntry:
    """青小团缓存条目"""
    key: str
    value: Any
    size_bytes: int = 0
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    serialized: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_access


@dataclass
class CacheStats:
    """青小团缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    l1_size: int = 0
    l2_size: int = 0
    l3_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0

    @property
    def usage_ratio(self) -> float:
        return self.total_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "青小团 缓存管理引擎 - 统计报告",
            "=" * 60,
            f"命中:                {self.hits:,} ({self.hit_rate:.2f}%)",
            f"未命中:              {self.misses:,}",
            f"淘汰数:              {self.evictions:,}",
            f"总条目数:            {self.total_entries:,}",
            f"总大小:              {self._format_bytes(self.total_size_bytes)}",
            f"最大容量:            {self._format_bytes(self.max_size_bytes)}",
            f"使用率:              {self.usage_ratio:.2%}",
            f"L1(内存):            {self._format_bytes(self.l1_size)}",
            f"L2(磁盘):            {self._format_bytes(self.l2_size)}",
            f"L3(分布式):          {self._format_bytes(self.l3_size)}",
        ]
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"


class LRUEvictionPolicy:
    """青小团 LRU 淘汰策略"""

    def __init__(self):
        self._order = OrderedDict()

    def access(self, key: str):
        if key in self._order:
            self._order.move_to_end(key)

    def add(self, key: str):
        self._order[key] = None

    def remove(self, key: str):
        self._order.pop(key, None)

    def evict(self, count: int = 1) -> List[str]:
        evicted = []
        for _ in range(count):
            if self._order:
                key, _ = self._order.popitem(last=False)
                evicted.append(key)
        return evicted

    def clear(self):
        self._order.clear()


class LFUEvictionPolicy:
    """青小团 LFU 淘汰策略"""

    def __init__(self):
        self._freq: Counter = Counter()

    def access(self, key: str):
        self._freq[key] += 1

    def add(self, key: str):
        if key not in self._freq:
            self._freq[key] = 1

    def remove(self, key: str):
        self._freq.pop(key, None)

    def evict(self, count: int = 1) -> List[str]:
        evicted = []
        for key, _ in self._freq.most_common()[:-count - 1:-1]:
            if len(evicted) >= count:
                break
            evicted.append(key)
            self._freq.pop(key, None)
        return evicted

    def clear(self):
        self._freq.clear()


class LRULFUMixedPolicy:
    """青小团 LRU+LFU 混合淘汰策略"""

    def __init__(self, lru_weight: float = 0.5):
        self.lru = LRUEvictionPolicy()
        self.lfu = LFUEvictionPolicy()
        self.lru_weight = lru_weight

    def access(self, key: str):
        self.lru.access(key)
        self.lfu.access(key)

    def add(self, key: str):
        self.lru.add(key)
        self.lfu.add(key)

    def remove(self, key: str):
        self.lru.remove(key)
        self.lfu.remove(key)

    def evict(self, count: int = 1) -> List[str]:
        lru_evict = int(count * self.lru_weight)
        lfu_evict = count - lru_evict

        if lru_evict > 0:
            self.lru.evict(lru_evict)
        if lfu_evict > 0:
            self.lfu.evict(lfu_evict)

        return []

    def clear(self):
        self.lru.clear()
        self.lfu.clear()


class L1MemoryCache:
    """青小团 L1 内存缓存 - 使用 LRU+LFU 混合淘汰"""

    def __init__(self, max_size_bytes: int = 64 * 1024 * 1024 * 1024):
        self.max_size = max_size_bytes
        self._cache: Dict[str, CacheEntry] = {}
        self._policy = LRULFUMixedPolicy()
        self._lock = threading.RLock()
        self._current_size = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.last_access = time.time()
                entry.access_count += 1
                self._policy.access(key)
                return entry.value
            return None

    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> bool:
        with self._lock:
            if size_bytes is None:
                size_bytes = self._estimate_size(value)

            while self._current_size + size_bytes > self.max_size and self._cache:
                self._evict_one()

            entry = CacheEntry(
                key=key, value=value, size_bytes=size_bytes,
                created_at=time.time(), last_access=time.time(),
                access_count=1,
            )
            self._cache[key] = entry
            self._current_size += size_bytes
            self._policy.add(key)
            return True

    def _evict_one(self):
        keys = self._policy.evict(1)
        for key in keys:
            entry = self._cache.pop(key, None)
            if entry:
                self._current_size -= entry.size_bytes
                self._policy.remove(key)

    def remove(self, key: str):
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._current_size -= entry.size_bytes
                self._policy.remove(key)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._policy.clear()
            self._current_size = 0

    def contains(self, key: str) -> bool:
        return key in self._cache

    @property
    def size(self) -> int:
        return self._current_size

    @property
    def count(self) -> int:
        return len(self._cache)

    def _estimate_size(self, value: Any) -> int:
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024


class L2DiskCache:
    """青小团 L2 磁盘缓存 - 支持 mmap 文件映射"""

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int = 1024 * 1024 * 1024 * 1024,
        mmap_enabled: bool = True,
        serialization: str = "arrow",
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_bytes
        self.mmap_enabled = mmap_enabled
        self.serialization = serialization
        self._lock = threading.RLock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mmap_dir = self.cache_dir / "mmap"
        self._mmap_dir.mkdir(parents=True, exist_ok=True)

        self._meta_file = self.cache_dir / "metadata.json"
        self._metadata: Dict[str, Dict] = {}
        self._load_metadata()
        self._policy = LRUEvictionPolicy()

        for key in self._metadata:
            self._policy.add(key)

        logger.info(f"青小团L2磁盘缓存初始化: dir={cache_dir}, mmap={mmap_enabled}")

    def _load_metadata(self):
        if self._meta_file.exists():
            try:
                with open(self._meta_file, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.warning(f"青小团加载缓存元数据失败: {e}")
                self._metadata = {}

    def _save_metadata(self):
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)

    def _get_cache_path(self, key: str) -> Path:
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / key_hash[:4] / f"{key_hash}.cache"

    def _get_mmap_path(self, key: str) -> Path:
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._mmap_dir / f"{key_hash}.mmap"

    def put(self, key: str, value: Any) -> bool:
        with self._lock:
            cache_path = self._get_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if self.serialization == "pickle":
                    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(cache_path, "wb") as f:
                        f.write(data)
                elif self.serialization == "arrow":
                    self._save_as_arrow(value, cache_path)
                elif self.serialization == "numpy":
                    self._save_as_numpy(value, cache_path)
                else:
                    data = pickle.dumps(value)
                    with open(cache_path, "wb") as f:
                        f.write(data)

                file_size = cache_path.stat().st_size
                self._metadata[key] = {
                    "size": file_size,
                    "created": time.time(),
                    "last_access": time.time(),
                    "path": str(cache_path),
                }
                self._policy.add(key)
                self._save_metadata()

                return True

            except Exception as e:
                logger.error(f"青小团L2缓存写入失败 [{key}]: {e}")
                return False

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            meta = self._metadata.get(key)
            if meta is None:
                return None

            cache_path = Path(meta["path"])
            if not cache_path.exists():
                self._metadata.pop(key, None)
                self._save_metadata()
                return None

            try:
                if self.mmap_enabled and self.serialization == "numpy":
                    value = self._load_with_mmap(cache_path)
                elif self.serialization == "pickle":
                    with open(cache_path, "rb") as f:
                        value = pickle.load(f)
                elif self.serialization == "arrow":
                    value = self._load_arrow(cache_path)
                elif self.serialization == "numpy":
                    value = np.load(cache_path)
                else:
                    with open(cache_path, "rb") as f:
                        value = pickle.load(f)

                self._metadata[key]["last_access"] = time.time()
                self._policy.access(key)
                self._save_metadata()

                return value

            except Exception as e:
                logger.error(f"青小团L2缓存读取失败 [{key}]: {e}")
                return None

    def _save_as_arrow(self, value: Any, path: Path):
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc

            if isinstance(value, dict):
                table = pa.table({k: pa.array([v]) if not isinstance(v, (list, np.ndarray)) else pa.array([v]) for k, v in value.items()})
            elif isinstance(value, list):
                table = pa.table({"data": pa.array(value)})
            elif isinstance(value, np.ndarray):
                table = pa.table({"data": pa.array(value.tolist())})
            else:
                table = pa.table({"value": pa.array([value])})

            with pa.OSFile(str(path), "wb") as sink:
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
        except ImportError:
            with open(path, "wb") as f:
                pickle.dump(value, f)

    def _load_arrow(self, path: Path) -> Any:
        try:
            import pyarrow.ipc as ipc
            with open(path, "rb") as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
                return {col: table.column(col)[0].as_py() for col in table.column_names}
        except Exception:
            return None

    def _save_as_numpy(self, value: Any, path: Path):
        if isinstance(value, np.ndarray):
            np.save(path, value)
        else:
            np.save(path, np.array(value))

    def _load_with_mmap(self, path: Path) -> np.ndarray:
        return np.load(path, mmap_mode="r")

    def remove(self, key: str):
        with self._lock:
            meta = self._metadata.pop(key, None)
            if meta:
                path = Path(meta["path"])
                if path.exists():
                    path.unlink()
                self._policy.remove(key)
                self._save_metadata()

    def clear(self):
        with self._lock:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._mmap_dir.mkdir(parents=True, exist_ok=True)
            self._metadata.clear()
            self._policy.clear()
            self._save_metadata()

    def contains(self, key: str) -> bool:
        return key in self._metadata

    @property
    def size(self) -> int:
        return sum(m.get("size", 0) for m in self._metadata.values())

    @property
    def count(self) -> int:
        return len(self._metadata)


class L3DistributedCache:
    """青小团 L3 分布式缓存 - 支持 Redis / Ray 后端"""

    def __init__(
        self,
        backend: str = "redis",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        cluster_mode: bool = False,
        prefix: str = "qingxiaotuan_cache:",
    ):
        self.backend = backend
        self.prefix = prefix
        self._client = None
        self._lock = threading.RLock()

        if backend == "redis":
            self._init_redis(redis_host, redis_port, redis_password, cluster_mode)
        elif backend == "ray":
            self._init_ray()

        logger.info(f"青小团L3分布式缓存初始化: backend={backend}")

    def _init_redis(self, host: str, port: int, password: Optional[str], cluster: bool):
        try:
            if cluster:
                from redis.cluster import RedisCluster
                self._client = RedisCluster(
                    host=host, port=port, password=password,
                    decode_responses=False,
                )
            else:
                import redis
                self._client = redis.Redis(
                    host=host, port=port, password=password,
                    decode_responses=False,
                    socket_timeout=5,
                )
                self._client.ping()
            logger.info(f"青小团Redis缓存连接成功: {host}:{port}")
        except ImportError:
            logger.warning("青小团: redis 未安装，L3缓存不可用")
        except Exception as e:
            logger.warning(f"青小团Redis连接失败: {e}")

    def _init_ray(self):
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._client = ray
            logger.info("青小团Ray缓存初始化成功")
        except ImportError:
            logger.warning("青小团: ray 未安装，L3缓存不可用")

    def put(self, key: str, value: Any, ttl: int = 3600) -> bool:
        if self._client is None:
            return False

        cache_key = self.prefix + key
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            if self.backend == "redis":
                self._client.setex(cache_key, ttl, data)
            elif self.backend == "ray":
                ray_ref = ray.put(value)
                self._client.put(cache_key, ray_ref)

            return True
        except Exception as e:
            logger.debug(f"青小团L3缓存写入失败: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        if self._client is None:
            return None

        cache_key = self.prefix + key
        try:
            if self.backend == "redis":
                data = self._client.get(cache_key)
                if data is None:
                    return None
                return pickle.loads(data)
            elif self.backend == "ray":
                ray_ref = self._client.get(cache_key)
                if ray_ref is None:
                    return None
                return ray.get(ray_ref)
            return None
        except Exception as e:
            logger.debug(f"青小团L3缓存读取失败: {e}")
            return None

    def remove(self, key: str):
        if self._client is None:
            return
        cache_key = self.prefix + key
        try:
            self._client.delete(cache_key)
        except Exception:
            pass

    def clear(self):
        if self._client is None:
            return
        try:
            if self.backend == "redis":
                for key in self._client.scan_iter(match=self.prefix + "*"):
                    self._client.delete(key)
        except Exception:
            pass

    def contains(self, key: str) -> bool:
        if self._client is None:
            return False
        cache_key = self.prefix + key
        try:
            return self._client.exists(cache_key) > 0
        except Exception:
            return False


class CacheManager:
    """青小团多级缓存管理器 - 统一管理 L1(内存) + L2(磁盘) + L3(分布式) 缓存"""

    def __init__(
        self,
        l1_max_size_gb: float = 64.0,
        l2_cache_dir: str = "./cache/disk",
        l2_max_size_gb: float = 1024.0,
        l3_backend: str = "redis",
        mmap_enabled: bool = True,
        serialization: str = "arrow",
        cleanup_interval_hours: int = 24,
    ):
        self.l1 = L1MemoryCache(max_size_bytes=int(l1_max_size_gb * 1024**3))
        self.l2 = L2DiskCache(
            cache_dir=l2_cache_dir,
            max_size_bytes=int(l2_max_size_gb * 1024**3),
            mmap_enabled=mmap_enabled,
            serialization=serialization,
        )
        self.l3 = L3DistributedCache(backend=l3_backend)

        self.stats = CacheStats(max_size_bytes=int(l1_max_size_gb * 1024**3))
        self._cleanup_interval = cleanup_interval_hours * 3600
        self._last_cleanup = time.time()

        logger.info(
            f"青小团缓存管理器初始化: L1={l1_max_size_gb}GB, "
            f"L2={l2_max_size_gb}GB, L3={l3_backend}, mmap={mmap_enabled}"
        )

    def get(self, key: str) -> Optional[Any]:
        value = self.l1.get(key)
        if value is not None:
            self.stats.hits += 1
            logger.debug(f"青小团缓存 L1 命中: {key}")
            return value

        value = self.l2.get(key)
        if value is not None:
            self.stats.hits += 1
            self.l1.put(key, value)
            logger.debug(f"青小团缓存 L2 命中: {key}")
            return value

        value = self.l3.get(key)
        if value is not None:
            self.stats.hits += 1
            self.l2.put(key, value)
            self.l1.put(key, value)
            logger.debug(f"青小团缓存 L3 命中: {key}")
            return value

        self.stats.misses += 1
        return None

    def put(self, key: str, value: Any, level: int = 1):
        if level <= 1:
            self.l1.put(key, value)
        if level <= 2:
            self.l2.put(key, value)
        if level <= 3:
            self.l3.put(key, value)

        self.stats.total_entries = self.l1.count + self.l2.count
        self.stats.l1_size = self.l1.size
        self.stats.l2_size = self.l2.size
        self._check_cleanup()

    def remove(self, key: str):
        self.l1.remove(key)
        self.l2.remove(key)
        self.l3.remove(key)

    def clear(self):
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.stats = CacheStats(max_size_bytes=self.stats.max_size_bytes)
        logger.info("青小团所有缓存已清空")

    def warmup(self, keys: List[str]):
        loaded = 0
        for key in tqdm(keys, desc="青小团缓存预热", ncols=80):
            value = self.l2.get(key)
            if value is not None:
                self.l1.put(key, value)
                loaded += 1
            else:
                value = self.l3.get(key)
                if value is not None:
                    self.l2.put(key, value)
                    self.l1.put(key, value)
                    loaded += 1
        logger.info(f"青小团缓存预热完成: {loaded}/{len(keys)} 已加载")

    def _check_cleanup(self):
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup()
            self._last_cleanup = now

    def cleanup(self):
        self.stats.l1_size = self.l1.size
        self.stats.l2_size = self.l2.size
        self.stats.total_size_bytes = self.l1.size + self.l2.size
        logger.info(f"青小团缓存清理完成: L1={self._format_bytes(self.l1.size)}, L2={self._format_bytes(self.l2.size)}")

    def contains(self, key: str, check_l3: bool = False) -> bool:
        if self.l1.contains(key):
            return True
        if self.l2.contains(key):
            return True
        if check_l3 and self.l3.contains(key):
            return True
        return False

    def get_stats(self) -> CacheStats:
        self.stats.l1_size = self.l1.size
        self.stats.l2_size = self.l2.size
        self.stats.total_size_bytes = self.l1.size + self.l2.size
        self.stats.total_entries = self.l1.count + self.l2.count
        return self.stats

    def report_summary(self) -> str:
        return self.get_stats().summary()

    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"
