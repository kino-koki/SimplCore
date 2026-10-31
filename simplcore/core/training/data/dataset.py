# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""manifest 读取与内存映射数据集。"""
from __future__ import annotations


from .pack import compute_sha256
from .types import DataPipelineError
from .constants import _LOG, MANIFEST_NAME

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset



# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def read_manifest(path: Any) -> dict:
    """读取 corpus manifest。

    Args:
        path: ``index.json`` 路径, 或包含它的目录。
    """
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"corpus manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def shard_paths(manifest: dict, out_dir: Any, split: str = "train") -> List[Path]:
    """从 manifest 取出某个 split 的全部分片绝对路径 (按写入顺序)。"""
    out_dir = Path(out_dir)
    entry = (manifest.get("splits") or {}).get(split)
    if not entry:
        raise KeyError(f"split {split!r} not in manifest (have: {list((manifest.get('splits') or {}).keys())})")
    return [out_dir / f["path"] for f in entry["files"]]


def _load_memmap(path: Path, dtype: Any, expected_tokens: Optional[int] = None) -> np.ndarray:
    """把一个 ``.bin`` 映射成只读 numpy memmap, 并可选校验 token 数。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token shard missing: {path}")
    dt = np.dtype(dtype)
    size = path.stat().st_size
    if size % dt.itemsize:
        raise DataPipelineError(
            f"{path} size {size} is not a multiple of itemsize {dt.itemsize}; the file is truncated"
        )
    n = size // dt.itemsize
    if expected_tokens is not None and n != int(expected_tokens):
        raise DataPipelineError(f"{path} holds {n} tokens, manifest says {expected_tokens}")
    if n == 0:
        # numpy 无法 mmap 0 字节文件 ("cannot mmap an empty file"), 直接给空数组。
        return np.empty(0, dtype=dt)
    return np.memmap(str(path), dtype=dt, mode="r", shape=(n,))


class ShardedTokenDataset(Dataset):
    """按 ``seq_len`` 切窗的定长语言建模数据集 (mmap 分片, 常驻内存约等于 0)。

    每个样本是 ``(seq_len,)`` 的 ``torch.long`` 窗口 —— 对 causal LM 来说
    ``input_ids`` 与 ``labels`` 是同一个窗口错位对齐, 因此默认只返回输入窗口,
    由 collate / 模型侧做 ``labels = input_ids`` 的 shift。

    数据布局::

        train-00000.bin  train-00001.bin ...   # uint32 little-endian token 流

    窗口 ``i`` 就是 ``tokens[i * seq_len : (i + 1) * seq_len]``, 与
    ``pack_documents(drop_last=True)`` 的写入约定一致: 不足一窗的尾部根本不会
    写进文件, 因此**不存在**需要在这一层丢弃的孤儿 token。

    Args:
        root: 语料目录 (含 ``index.json`` 与 ``*.bin``), 或直接用 ``index.json`` 路径。
        split: ``"train"`` / ``"val"`` / ``"test"``。
        seq_len: 窗口长度; ``None`` 时用 manifest 里的值。
        dtype: ``None`` 时用 manifest 里的 ``dtype`` (默认 uint32)。
        verify_hashes: 加载时校验每个分片的 SHA-256 (默认 True; 换来"读到坏盘或
            半截下载文件"能被立刻发现, 代价是启动时多读一遍)。
        return_labels: 是否额外返回 ``labels`` (同一窗口; 便于直接在训练循环里 unpack)。
        token_dtype: 返回张量的 dtype, 默认 ``torch.long``。
        transform: 可选的一元变换, 作用在返回的 tensor 上。

    Attributes:
        manifest: 原始 manifest dict。
        vocab_size: 词表大小。
        eos_id: EOS id。
        num_tokens: 该 split 的 token 总数。
        num_windows: 窗口数。
        shards: 每个分片的 ``{path, n_tokens, offset, bytes}``。
    """

    def __init__(
        self,
        root: Any,
        split: str = "train",
        *,
        seq_len: Optional[int] = None,
        dtype: Optional[str] = None,
        verify_hashes: bool = True,
        return_labels: bool = False,
        token_dtype: torch.dtype = torch.long,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        root_path = Path(root)
        self.root = root_path if root_path.is_dir() else root_path.parent
        self.manifest = read_manifest(root_path)
        self.split = str(split)
        if self.split not in (self.manifest.get("splits") or {}):
            raise KeyError(f"split {self.split!r} not in manifest {self.root / MANIFEST_NAME}")

        entry = self.manifest["splits"][self.split]
        self.seq_len = int(seq_len if seq_len is not None else self.manifest["seq_len"])
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        self.dtype = np.dtype(dtype or self.manifest.get("dtype", "uint32"))
        if self.dtype.kind != "u":
            raise ValueError(f"manifest dtype must be unsigned, got {self.dtype}")
        self.vocab_size = int(self.manifest.get("vocab_size", 0) or 0)
        self.eos_id = int(self.manifest.get("eos_id", -1))
        self.return_labels = bool(return_labels)
        self.token_dtype = token_dtype
        self.transform = transform

        files = list(entry["files"])
        if not files:
            raise DataPipelineError(f"split {self.split!r} has no shards in the manifest")

        self.shards: List[Dict[str, Any]] = []
        offsets: List[int] = []
        running = 0
        for f in files:
            path = self.root / f["path"]
            if verify_hashes and f.get("sha256"):
                actual = compute_sha256(path)
                if actual != f["sha256"]:
                    raise DataPipelineError(
                        f"sha256 mismatch for {path}: manifest={f['sha256'][:16]}... actual={actual[:16]}..."
                    )
            offsets.append(running)
            self.shards.append({
                "path": str(path),
                "n_tokens": int(f.get("n_tokens", 0)),
                "offset": int(running),
                "bytes": int(path.stat().st_size),
            })
            running += int(f.get("n_tokens", 0))
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._arrays: List[np.ndarray] = [
            _load_memmap(Path(s["path"]), self.dtype, expected_tokens=s["n_tokens"]) for s in self.shards
        ]
        self.num_tokens = int(running)
        # 窗口数始终按"实际 token 数 // seq_len"计算: 文件里最后一个 token 永远
        # 可被某个窗口用到 (写入时已保证 token 数正好是整数个 seq_len 窗口)。
        # manifest 里的 n_windows 只是同一算式在 seq_len 下的快照, 仅作交叉校验。
        self.num_windows = self.num_tokens // self.seq_len
        declared = int(entry.get("n_windows", self.num_windows))
        if declared != self.num_windows:
            _LOG.warning(
                "%s: manifest declares %d windows of %d, the %d tokens on disk give %d; using the disk value",
                self.split, declared, self.seq_len, self.num_tokens, self.num_windows,
            )
        if self.num_windows <= 0:
            raise DataPipelineError(
                f"split {self.split!r} has {self.num_tokens} tokens, fewer than one window of {self.seq_len}"
            )

    # -- 读取 ---------------------------------------------------------------
    def _window(self, index: int) -> np.ndarray:
        """取第 ``index`` 个窗口的 numpy 视图 (跨分片时做一次拷贝)。"""
        start = index * self.seq_len
        shard_idx = int(np.searchsorted(self._offsets, start, side="right") - 1)
        shard_idx = max(shard_idx, 0)
        arr = self._arrays[shard_idx]
        local = start - int(self._offsets[shard_idx])
        if local + self.seq_len <= arr.size:
            return arr[local:local + self.seq_len]
        head = arr[local:]
        need = self.seq_len - head.size
        if shard_idx + 1 >= len(self._arrays):
            raise DataPipelineError(f"window {index} runs past the last shard of split {self.split!r}")
        return np.concatenate((head, self._arrays[shard_idx + 1][:need]))

    def __len__(self) -> int:
        """窗口数 (``num_tokens // seq_len``)。"""
        return int(self.num_windows)

    def __getitem__(self, index: int):
        """返回 ``(seq_len,)`` 的窗口; ``return_labels=True`` 时返回 ``(x, y)``。"""
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"index {index} out of range for split {self.split!r} ({n} windows)")
        # mmap 是只读视图, torch.from_numpy 会为"不可写数组"发警告, 并让张量
        # 直接引用磁盘页; 这里拷一份 (每次一个窗口, 开销可忽略) 换取干净的语义。
        window = np.asarray(self._window(index)).copy()
        x = torch.from_numpy(window).to(self.token_dtype)
        if self.transform is not None:
            x = self.transform(x)
        if self.return_labels:
            return x, x.clone()
        return x

    # -- 内省 ---------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ShardedTokenDataset(split={self.split!r}, seq_len={self.seq_len}, "
            f"windows={self.num_windows}, tokens={self.num_tokens}, "
            f"dtype={self.dtype.name}, shards={len(self.shards)})"
        )

    def token_stream(self) -> Iterator[int]:
        """顺序产出该 split 的全部 token (调试/统计用; 大语料上很慢)。"""
        for arr in self._arrays:
            yield from arr.tolist()

    def num_bytes(self) -> int:
        """该 split 在磁盘上的字节数。"""
        return int(sum(s["bytes"] for s in self.shards))

    def close(self) -> None:
        """释放 mmap 句柄 (Windows 上删除/覆盖文件前必须先调用)。"""
        self._arrays = []

    def __getstate__(self) -> dict:
        """支持 ``DataLoader(num_workers>0)``: 不把 mmap 对象跨进程序列化。"""
        state = self.__dict__.copy()
        state["_arrays"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        """反序列化后按需重新打开 mmap (每个 worker 自己持有句柄)。"""
        self.__dict__.update(state)
        self._arrays = [
            _load_memmap(Path(s["path"]), self.dtype, expected_tokens=s["n_tokens"]) for s in self.shards
        ]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def load_shard(path: Any, start: int = 0, count: Optional[int] = None, dtype: str = "uint32") -> np.ndarray:
    """读一个 token 分片的一段 (默认全部), 返回 numpy 数组。

    这是"``.bin`` 能否解析"最直接的抓手: 它不依赖 manifest,
    只按字节数推断 token 数并检查是否整除了 itemsize。
    """
    arr = _load_memmap(Path(path), np.dtype(dtype))
    if count is None:
        return np.asarray(arr[start:])
    return np.asarray(arr[start:start + int(count)])


def load_split(
    root: Any,
    split: str = "train",
    *,
    seq_len: Optional[int] = None,
    verify_hashes: bool = True,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """把某个 split 读成 ``(n_windows, seq_len)`` 数组 (调试用, 会占内存)。"""
    ds = ShardedTokenDataset(root, split, seq_len=seq_len, verify_hashes=verify_hashes)
    n = len(ds) if max_windows is None else min(len(ds), int(max_windows))
    out = np.empty((n, ds.seq_len), dtype=ds.dtype)
    for i in range(n):
        out[i] = ds[i]
    return out

