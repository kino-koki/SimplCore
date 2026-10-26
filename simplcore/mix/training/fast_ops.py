# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""极致训练优化原语 — 质量完全不变，只减墙钟时间。

本模块所有优化**不改变训练数学**，只消除工程开销：
  1. ``encode_text_fast``: 向量化字符哈希分词，输出 id 与旧 ``encode_text`` 完全一致。
     旧实现按字符逐个 ``ord(ch) % ...`` 是纯 Python 循环；新实现用
     utf-32-le 一次性取全码点 + 张量取模，省掉逐字符开销。
  2. ``TokenizedRowCache``: 把语料一次性 tokenize 成行缓存（int32 memmap），
     之后每个 epoch 直接读行，跳过慢速 parquet ``to_pylist`` + 逐字符编码。
  3. ``prefetch``: 后台线程预取下一批 CPU 张量，让 GPU 计算与数据准备重叠。
  4. ``CacheSweepPolicy``: 不再每步 ``empty_cache()``（CUDA 内存池杀手）；
     只在显存压力时或设定间隔清扫。
  5. ``make_adamw``: CUDA 上启用 fused AdamW（V100 优化器提速）。

正确性保证：所有编码路径与旧实现逐 id 一致（见 tests/test_fast_tokenizer.py）。
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import simplcore.tensor as torch


# ============================================================
# 1. 向量化字符哈希分词（id 与旧 encode_text 完全一致）
# ============================================================

def encode_text_fast(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """向量化字符哈希编码，输出与旧 ``encode_text`` 逐 id 一致。

    旧实现: ``[2 + (ord(ch) % max(vocab_size - 2, 1)) for ch in text]``
    新实现: 用 utf-32-le 一次性得到全部真实码点（与 ord() 一致，含 astral
    字符），再张量取模。省掉每个字符的 Python 循环。
    """
    divisor = max(vocab_size - 2, 1)
    if not text:
        return torch.zeros(seq_len + 1, dtype=torch.long)
    # utf-32-le: 每码点 4 字节，无代理对，与 ord() 完全一致
    cp = torch.frombuffer(text.encode("utf-32-le"), dtype=torch.int32)
    n = min(cp.numel(), seq_len + 1)
    ids = (2 + (cp[:n] % divisor)).to(torch.long)
    if n < seq_len + 1:
        ids = torch.cat([ids, torch.zeros(seq_len + 1 - n, dtype=torch.long)])
    return ids


def batch_stream_fast(
    texts: Iterator[str], *, vocab_size: int, batch_size: int, seq_len: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """与旧 ``batch_stream`` 完全相同的 yield 语义（含"不产出尾批"），
    仅编码用向量化实现。保证批次组成逐批一致 → 训练质量不变。"""
    batch: List[torch.Tensor] = []
    for text in texts:
        batch.append(encode_text_fast(text, vocab_size, seq_len))
        if len(batch) == batch_size:
            tokens = torch.stack(batch, dim=0)
            yield tokens[:, :-1], tokens[:, 1:]
            batch = []
    # 注意：与旧实现一致，不足 batch_size 的尾批不产出


# ============================================================
# 2. TokenizedRowCache — 语料一次性 tokenize，epoch 复用
# ============================================================

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # 无 numpy 环境（如本调试机）降级为内存缓存
    _np = None
    _HAS_NUMPY = False


def _release_memmap(arr) -> None:
    """释放 numpy memmap 文件句柄 (Windows 上必须显式关闭, 否则目录无法删除)。

    numpy 2.x 的 np.load(mmap_mode='r') 返回的 memmap 没有公开 close() 方法,
    需通过底层 _mmap 对象关闭; 兜底方案是删除引用让 GC 回收。
    """
    if arr is None:
        return
    m = getattr(arr, "_mmap", None)
    if m is not None:
        try:
            m.close()
        except Exception:
            pass
    else:
        try:
            arr.close()
        except (AttributeError, Exception):
            pass


class TokenizedRowCache:
    """把文本行 tokenize 成 [N, seq_len+1] 行缓存，epoch>=2 零 CPU 重编码。

    - 首次遍历时建缓存（逐行写入 int32 分片文件，内存有界）。
    - 之后每次遍历直接读缓存行，跳过 parquet to_pylist + 逐字符编码。
    - 缓存键 = (corpus_key, seq_len, vocab_size)，corpus_key 由调用方给定
      （应含 source 路径 + rank + world，保证 DDP 各 rank 独立缓存）。
    - 无 numpy 时降级为内存行列表（单次运行内仍免重复编码）。

    行内容与 ``encode_text_fast`` 完全一致（int32 存储，读时转 long）。
    """

    def __init__(
        self,
        cache_dir: Path,
        corpus_key: str,
        seq_len: int,
        vocab_size: int,
        rows_per_file: int = 200_000,
    ):
        self.cache_dir = Path(cache_dir)
        self.corpus_key = corpus_key
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rows_per_file = rows_per_file
        self._mem_rows: Optional[List[torch.Tensor]] = None  # 无 numpy 时用
        self._files: List[Path] = []
        self._n_rows = 0

    @property
    def exists(self) -> bool:
        return bool(self._files) or (self._n_rows > 0 and self._mem_rows is not None)

    def _cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"{self.corpus_key}_seq{self.seq_len}_{idx:05d}.npy"

    def build(self, text_iter: Iterator[str]) -> None:
        """从文本流建缓存（首次遍历调用）。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_NUMPY:
            cur_file_idx = 0
            cur_arr = _np.zeros((self.rows_per_file, self.seq_len + 1), dtype=_np.int32)
            row_in_file = 0
            for text in text_iter:
                row = encode_text_fast(text, self.vocab_size, self.seq_len).to(torch.int32).numpy()
                cur_arr[row_in_file] = row
                row_in_file += 1
                self._n_rows += 1
                if row_in_file >= self.rows_per_file:
                    _np.save(self._cache_path(cur_file_idx), cur_arr[:row_in_file])
                    self._files.append(self._cache_path(cur_file_idx))
                    cur_file_idx += 1
                    cur_arr = _np.zeros((self.rows_per_file, self.seq_len + 1), dtype=_np.int32)
                    row_in_file = 0
            if row_in_file > 0:
                _np.save(self._cache_path(cur_file_idx), cur_arr[:row_in_file])
                self._files.append(self._cache_path(cur_file_idx))
        else:
            self._mem_rows = []
            for text in text_iter:
                self._mem_rows.append(
                    encode_text_fast(text, self.vocab_size, self.seq_len).to(torch.int32)
                )
            self._n_rows = len(self._mem_rows)

    def load_known(self) -> bool:
        """尝试直接加载既有缓存；成功返回 True。"""
        if _HAS_NUMPY:
            self._files = sorted(
                self.cache_dir.glob(f"{self.corpus_key}_seq{self.seq_len}_*.npy")
            )
            if self._files:
                # 统计行数（读各分片头部 shape，mmap 懒加载，开销小）
                total = 0
                for fp in self._files:
                    try:
                        hdr = _np.load(fp, mmap_mode="r")
                        try:
                            total += int(hdr.shape[0])
                        finally:
                            _release_memmap(hdr)  # 释放 mmap 文件句柄 (Windows 必需)
                    except Exception:
                        return False
                self._n_rows = total
                return True
            return False
        # 无 numpy：无法跨进程持久化，仅本次运行内存缓存
        return False

    def iterate(self) -> Iterator[torch.Tensor]:
        """产出 [seq_len+1] 行（与 encode_text_fast 一致）。"""
        if _HAS_NUMPY and self._files:
            for fp in self._files:
                arr = _np.load(fp, mmap_mode="r")  # memmap，内存有界
                try:
                    for i in range(arr.shape[0]):
                        # .to(torch.long) 已拷贝，mmap 关闭不影响产出行
                        yield torch.from_numpy(arr[i]).to(torch.long)
                finally:
                    _release_memmap(arr)  # 释放 mmap 文件句柄 (Windows 必需)
        elif self._mem_rows is not None:
            for row in self._mem_rows:
                yield row.to(torch.long)
        else:
            raise RuntimeError("TokenizedRowCache 未构建也未加载")


# ============================================================
# 3. 后台预取（数据准备与 GPU 计算重叠）
# ============================================================

def prefetch(
    iterable: Iterable,
    max_inflight: int = 2,
    *,
    pin_memory: bool = False,
) -> Iterator:
    """后台线程预取迭代器，产出顺序与输入一致。

    pin_memory=True 时（CUDA）预取线程把批次钉在页锁定内存，主线程用
    ``.to(device, non_blocking=True)`` 重叠 H2D 拷贝。
    """
    q: "queue.Queue" = queue.Queue(maxsize=max_inflight)
    sentinel = object()
    finished = threading.Event()

    def _worker():
        try:
            for item in iterable:
                if pin_memory and isinstance(item, torch.Tensor):
                    try:
                        item = item.pin_memory()
                    except Exception:
                        pass
                q.put(item)
        except Exception as exc:  # 预取失败不静默：放入异常由主循环抛出
            q.put(exc)
        finally:
            q.put(sentinel)
            finished.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            thread.join()
            return
        if isinstance(item, Exception):
            raise item
        yield item


# ============================================================
# 4. 自适应显存清扫策略（不再逐步 empty_cache）
# ============================================================

class CacheSweepPolicy:
    """显存清扫策略。

    关键洞察：逐步 ``torch.cuda.empty_cache()`` 是训练循环的头号性能杀手——
    它强制 cudaFree 释放已缓存块，下一步又要 cudaMalloc 重新分配，来回开销
    巨大。正确做法：让 CUDA caching allocator 自己复用内存，只在必要时清扫。

    sweep_every:
        0   从不逐步清扫（默认；显存足够时最快）
        N>0 每 N 步清扫一次（DirectML 小显存逃生舱）
    on_checkpoint:
        保存 checkpoint 前清扫一次（导出前后释放峰值）
    """

    def __init__(self, backend: str, sweep_every: int = 0):
        self.backend = backend
        self.sweep_every = max(0, sweep_every)

    def maybe_sweep(self, step: int) -> None:
        if self.sweep_every <= 0:
            return
        if step % self.sweep_every == 0:
            self.sweep()

    def sweep(self) -> None:
        if self.backend == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        elif self.backend == "directml":
            try:
                # torch_directml 已替换为自研引擎（CPU模式）
                torch_directml.empty_cache()
            except Exception:
                pass


# torch_directml 兼容壳（DirectML 后端不可用时回退到 CPU 语义）
class _DirectMLCompat:
    def device(self, idx=0): return 'cpu'
    def device_count(self): return 0
    def empty_cache(self): pass


torch_directml = _DirectMLCompat()


# ============================================================
# 5. fused AdamW（CUDA 优化器提速，V100 关键）
# ============================================================

def make_adamw(
    params,
    lr: float,
    *,
    backend: str,
    fused: bool = True,
    **kwargs,
) -> torch.optim.AdamW:
    """构造 SCAdamW；CUDA 上优先 fused（torch>=2.0 原生支持）。

    fused 版把多步优化状态更新融合为单个 CUDA kernel，对 V100 是
    优化器侧 ~2x 提速。数学等价（仅在浮点舍入上有微小差异），质量不变。
    """
    from simplcore.mix.optim import SCAdamW

    if backend == "cuda" and fused:
        try:
            return SCAdamW(params, lr=lr, fused=True, **kwargs)
        except (TypeError, ValueError, RuntimeError):
            pass  # 不支持的 dtype/形状 → 回退非 fused
    return SCAdamW(params, lr=lr, **kwargs)
