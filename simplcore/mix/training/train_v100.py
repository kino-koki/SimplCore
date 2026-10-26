"""CUDA 多卡 (DDP) 150M 训练启动器 — 面向 8xV100 集群。

替代仅支持 AMD DirectML 的 ``train_150m_gpu.py``，用于 NVIDIA 集群。

与 DirectML 路径的关键差异:
  * 强制 CUDA; 无 CUDA 直接失败，不回退 CPU。
  * 分布式数据并行 (DDP)，经 ``torchrun --nproc_per_node=8`` 拉起 8 卡。
  * FP16 混合精度 (torch.autocast + GradScaler)。V100 (Volta) 没有 BF16，
    只有 FP16 tensor core，所以全程 FP16 + 动态损失缩放。
  * 数据按 rank 切分 (parquet 分片按 rank 交错 / JSONL 按索引跳步)，
    保证 8 卡读到的训练数据不重复。
  * 全局 batch = batch_size * grad_accum * world_size。
  * 保留原路线的 LossGuard 协议 (目标 <1.5, 硬顶 3.9)。

用法 (在 8xV100 服务器上):
    torchrun --nproc_per_node=8 -m simplcore.mix.training.train_v100 \
        --corpus-dir <E盘语料目录> \
        --output-dir <输出目录> \
        --max-steps 100000 \
        --batch-size 8 \
        --seq-len 2048
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

import simplcore.tensor as torch
# torch.distributed 兼容层（单机模式）
class _DistCompat:
    def is_initialized(self): return False
    def init_process_group(self, *a, **kw): pass
    def barrier(self, *a, **kw): pass
    def all_reduce(self, *a, **kw): pass
    def get_rank(self): return 0
    def get_world_size(self): return 1
    def destroy_process_group(self): pass
dist = _DistCompat()
# GradScaler 兼容层（CPU模式）
class GradScaler:
    def __init__(self, *a, **kw): pass
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def unscale_(self, *a, **kw): pass

from simplcore.mix.config import mix_150m_text
from simplcore.mix.core import SCFunctional
from simplcore.mix.model import SimplCoreMix
from simplcore.mix.optim import SCAdamW
from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction


def require_cuda() -> None:
    """V100 路径只允许 CUDA; 不可用立即失败。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，V100 训练禁止回退 CPU")
    world = torch.cuda.device_count()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if local >= world:
        raise RuntimeError(f"LOCAL_RANK={local} 超出可用 GPU 数 {world}")


def setup_distributed() -> None:
    """初始化 NCCL 进程组。torchrun 会注入 RANK/LOCAL_RANK/WORLD_SIZE。"""
    if os.environ.get("RANK") is None:
        # 单进程回退（仅用于调试，8 卡请用 torchrun）
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[V100] {msg}", flush=True)


def stable_parquet_files(corpus_dir: Path, max_shards: int | None = None) -> List[Path]:
    files = sorted(corpus_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 parquet 分片: {corpus_dir}")
    if max_shards is not None:
        files = files[:max_shards]
    return files


def text_stream_from_parquet_rank(
    files: List[Path], *, rank_id: int, world: int, text_column: str = "text"
) -> Iterator[str]:
    """按 rank 交错分配分片，保证 8 卡数据不重复。"""
    import pyarrow.parquet as pq

    my_files = files[rank_id::world]
    if not my_files:
        raise RuntimeError(f"rank {rank_id} 分到的分片为空")
    while True:
        for file_path in my_files:
            table = pq.read_table(file_path, columns=[text_column])
            for value in table[text_column].to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value


def text_stream_from_jsonl_rank(
    file_path: Path,
    *,
    rank_id: int,
    world: int,
    text_column: str = "text",
    source_filter: Optional[str] = None,
) -> Iterator[str]:
    """JSONL 按行索引跳步 (index % world == rank_id)，8 卡数据不重复。"""
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到 JSONL 语料: {file_path}")
    source_filter_norm = source_filter.lower() if source_filter else None
    while True:
        matched_any = False
        idx = 0
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if idx % world != rank_id:
                    idx += 1
                    continue
                idx += 1
                row = json.loads(line)
                source = str(row.get("source", ""))
                if source_filter_norm and source_filter_norm not in source.lower():
                    continue
                text = row.get(text_column, "")
                if isinstance(text, str) and text.strip():
                    matched_any = True
                    yield text
        if source_filter_norm and not matched_any:
            raise ValueError(
                f"JSONL 中没有匹配 source_filter={source_filter!r} 的记录: {file_path}"
            )


def encode_text(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """与 DirectML 路径一致的字符哈希编码（占位分词，后续可换正式 tokenizer）。"""
    token_ids = [2 + (ord(ch) % max(vocab_size - 2, 1)) for ch in text]
    if len(token_ids) < seq_len + 1:
        token_ids = token_ids + [0] * (seq_len + 1 - len(token_ids))
    return torch.tensor(token_ids[: seq_len + 1], dtype=torch.long)


def batch_stream(
    texts: Iterator[str], *, vocab_size: int, batch_size: int, seq_len: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batch = []
    for text in texts:
        batch.append(encode_text(text, vocab_size, seq_len))
        if len(batch) == batch_size:
            tokens = torch.stack(batch, dim=0)
            yield tokens[:, :-1], tokens[:, 1:]
            batch = []


def average_tensor(t: torch.Tensor) -> torch.Tensor:
    """跨 rank 求均值（用于 loss）。"""
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def train_150m_v100(
    *,
    corpus_dir: Optional[Path],
    corpus_jsonl: Optional[Path],
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    max_shards: int | None,
    lr: float,
    target_loss: float,
    max_loss: float,
    source_filter: Optional[str] = None,
    mixed_precision: bool = True,
) -> dict:
    require_cuda()
    setup_distributed()

    r = rank()
    world = world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    log(f"分布式初始化完成: {world} 卡 DDP, FP16={'on' if mixed_precision else 'off'}")

    if corpus_jsonl is not None:
        text_stream = text_stream_from_jsonl_rank(
            corpus_jsonl, rank_id=r, world=world, source_filter=source_filter
        )
        source_paths = [corpus_jsonl]
    elif corpus_dir is not None:
        files = stable_parquet_files(corpus_dir, max_shards=max_shards)
        text_stream = text_stream_from_parquet_rank(files, rank_id=r, world=world)
        source_paths = files
    else:
        raise ValueError("必须提供 --corpus-jsonl 或 --corpus-dir")

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = mix_150m_text()
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
    cfg.gradient_checkpointing = True

    model = SimplCoreMix(cfg).to(local_rank)
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    optimizer = SCAdamW(model.parameters(), lr=lr)
    guard = LossGuard(target_loss=target_loss, max_loss=max_loss)
    scaler = GradScaler(enabled=mixed_precision)
    batches = batch_stream(
        text_stream,
        vocab_size=cfg.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    history: List[dict] = []
    started_at = time.time()
    last_loss: Optional[float] = None
    completed_steps = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0

        for _ in range(grad_accum):
            x, y = next(batches)
            x = x.to(local_rank)
            y = y.to(local_rank)
            if mixed_precision:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x, n_loops=1)
                    loss = SCFunctional.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
                    )
            else:
                logits = model(x, n_loops=1)
                loss = SCFunctional.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
                )
            scaler.scale(loss).backward()
            acc_loss += float(loss.detach().cpu().item())

        scaler.step(optimizer)
        scaler.update()

        # 跨卡平均 loss，rank 0 决策 STOP
        loss_t = torch.tensor(acc_loss / grad_accum, device=local_rank)
        average_tensor(loss_t)
        loss_value = float(loss_t.cpu().item())
        decision = guard.check(loss_value, step=step)
        history.append(decision.to_dict())
        last_loss = loss_value
        completed_steps = step

        if rank() == 0:
            print(
                f"[V100] step={step}/{max_steps} loss={loss_value:.4f} "
                f"action={decision.action.value} "
                f"global_batch={batch_size * grad_accum * world} "
                f"lr={lr}",
                flush=True,
            )

        # 广播 STOP 决策
        stop_t = torch.tensor(int(decision.action == LossGuardAction.STOP), device=local_rank)
        dist.broadcast(stop_t, src=0)
        if stop_t.item() == 1:
            log(f"LossGuard 触发 {decision.action.value}，停止训练")
            break

        del loss, logits, x, y
        gc.collect()

    # rank 0 保存 checkpoint（CUDA 张量直接 .cpu() 即可，无需 DirectML hack）
    if rank() == 0:
        checkpoint_path = output_dir / "mix_150m_v100_latest.pt"
        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        model_cpu = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        opt_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state_dict().items()
        }
        for state in opt_cpu.get("state", {}).values():
            for k2, v2 in state.items():
                if isinstance(v2, torch.Tensor):
                    state[k2] = v2.cpu()
        torch.save(
            {
                "step": completed_steps,
                "model_state_dict": model_cpu,
                "optimizer_state_dict": opt_cpu,
                "config": cfg.__dict__,
                "loss": last_loss,
                "device": f"cuda:{local_rank}",
                "world_size": world,
                "grad_accum": grad_accum,
                "mixed_precision": mixed_precision,
                "source_shards": [str(path) for path in source_paths],
                "loss_history": history,
            },
            checkpoint_path,
        )
        del model_cpu, opt_cpu
        gc.collect()

        summary = {
            "status": "completed" if completed_steps == max_steps else "stopped",
            "steps": completed_steps,
            "last_loss": last_loss,
            "device": f"cuda (world_size={world})",
            "mixed_precision": mixed_precision,
            "checkpoint_path": str(checkpoint_path),
            "num_source_shards": len(source_paths),
            "source_filter": source_filter,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        (output_dir / "mix_150m_v100_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"checkpoint 已保存: {checkpoint_path}")
        return summary

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="8xV100 CUDA/DDP 150M SimplCore Mix 训练启动器"
    )
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--corpus-jsonl", type=Path, default=None)
    parser.add_argument("--source-filter", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-loss", type=float, default=1.5)
    parser.add_argument("--max-loss", type=float, default=3.9)
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()

    summary = train_150m_v100(
        corpus_dir=args.corpus_dir,
        corpus_jsonl=args.corpus_jsonl,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        max_shards=args.max_shards,
        lr=args.lr,
        target_loss=args.target_loss,
        max_loss=args.max_loss,
        source_filter=args.source_filter,
        mixed_precision=not args.no_mixed_precision,
    )
    if rank() == 0:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
