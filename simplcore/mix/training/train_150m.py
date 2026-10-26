"""统一 150M 训练入口 — AMD 680M (DirectML) 与 NVIDIA V100 (CUDA) 兼容。

修复命门 6（设备双轨不统一）：
  此前 `train_150m_gpu.py`（DirectML）和 `train_v100.py`（CUDA）各写各的，
  设备探测、显存清理、checkpoint 保存逻辑全不同。
  本文件是统一入口，经 `training.device_utils` 自动探测设备：
    - CUDA + WORLD_SIZE>1 → 8 卡 DDP（V100 集群）
    - CUDA 单卡 → 单卡（调试）
    - DirectML → 单卡（AMD 680M 核显，不支持分布式）
    - 无 GPU → 报错（除非 SIMPLCORE_ALLOW_CPU=1）

命门 7（多样性）支持：
  `--seed` 指定训练种子；不同 150M 变体用不同 seed，保证堆叠多样性。

用法：
   # V100 集群（8 卡）
   torchrun --nproc_per_node=8 -m simplcore.mix.training.train_150m \
       --corpus-dir <目录> --output-dir <out> --seed 1001 --max-steps 50000

   # AMD 680M 核显（单卡）
   python -m simplcore.mix.training.train_150m \
       --corpus-jsonl <语料> --output-dir <out> --seed 1002 --max-steps 2000 \
       --batch-size 2 --seq-len 256

   # 仅堆叠用的快速变体训练（小步数 + 小 batch）
   python -m simplcore.mix.training.train_150m \
       --corpus-jsonl <语料> --output-dir <out> --seed 1003 --max-steps 500 \
       --batch-size 2 --seq-len 128
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

from simplcore.mix.config import mix_150m_text
from simplcore.mix.core import SCFunctional
from simplcore.mix.model import SimplCoreMix
from simplcore.mix.optim import SCAdamW
from simplcore.mix.training.device_utils import (
    describe_device,
    empty_device_cache,
    resolve_train_device,
    state_dict_to_cpu,
)
from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[TRAIN150M] {msg}", flush=True)


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
    """字符哈希编码（占位 tokenizer，见堆叠命门 4；正式 BPE 后续替换）。"""
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


def train_150m(
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
    source_filter: Optional[str],
    seed: int,
) -> dict:
    device, backend, is_ddp = resolve_train_device()
    if backend == "directml":
        is_ddp = False  # torch_directml 不支持分布式
    torch.manual_seed(seed)

    r = rank() if is_ddp else 0
    world = world_size() if is_ddp else 1
    mixed_precision = backend == "cuda"  # V100 FP16 AMP; DirectML 走 fp32

    log(f"设备={describe_device(device, backend)} 后端={backend} "
        f"DDP={is_ddp}(world={world}) seed={seed}")

    # 数据源（DDP 时按 rank 切分）
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

    model = SimplCoreMix(cfg).to(device)
    model.train()
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    optimizer = SCAdamW(model.parameters(), lr=lr)
    # 预热期 (默认 500 步) 内不触发 STOP: 新模型初始 loss ~ ln(vocab) 必然
    # 高于 max_loss, 直接 STOP 会让训练第一步就死。
    guard = LossGuard(
        target_loss=target_loss,
        max_loss=max_loss,
        warmup_steps=max(0, min(500, max_steps // 2)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    batches = batch_stream(
        text_stream, vocab_size=cfg.vocab_size, batch_size=batch_size, seq_len=seq_len
    )

    history = []
    started_at = time.time()
    last_loss: Optional[float] = None
    completed_steps = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0

        for _ in range(grad_accum):
            x, y = next(batches)
            x = x.to(device)
            y = y.to(device)
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

        loss_value = acc_loss / grad_accum
        # DDP: 跨卡平均 loss
        if is_ddp:
            loss_t = torch.tensor(loss_value, device=device)
            torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.AVG)
            loss_value = float(loss_t.cpu().item())

        decision = guard.check(loss_value, step=step)
        history.append(decision.to_dict())
        last_loss = loss_value
        completed_steps = step

        if rank() == 0:
            print(
                f"[TRAIN150M] step={step}/{max_steps} loss={loss_value:.4f} "
                f"action={decision.action.value} global_batch={batch_size * grad_accum * world}",
                flush=True,
            )

        if is_ddp:
            stop_t = torch.tensor(
                int(decision.action == LossGuardAction.STOP), device=device
            )
            torch.distributed.broadcast(stop_t, src=0)
            if stop_t.item() == 1:
                break
        elif decision.action == LossGuardAction.STOP:
            log(f"LossGuard 触发 {decision.action.value}，停止")
            break

        del loss, logits, x, y
        gc.collect()
        empty_device_cache(backend)

    # rank 0 保存 checkpoint（统一 CPU 化）
    if rank() == 0:
        raw_model = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
        ckpt_path = output_dir / f"mix_150m_seed{seed}.pt"
        torch.save(
            {
                "step": completed_steps,
                "seed": seed,
                "model_state_dict": state_dict_to_cpu(raw_model.state_dict()),
                "optimizer_state_dict": state_dict_to_cpu(optimizer.state_dict()),
                "config": cfg.__dict__,
                "loss": last_loss,
                "backend": backend,
                "device": str(device),
                "source_shards": [str(p) for p in source_paths],
                "loss_history": history,
            },
            ckpt_path,
        )
        summary = {
            "status": "completed" if completed_steps == max_steps else "stopped",
            "steps": completed_steps,
            "last_loss": last_loss,
            "backend": backend,
            "seed": seed,
            "checkpoint_path": str(ckpt_path),
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        (output_dir / f"mix_150m_seed{seed}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"checkpoint 已保存: {ckpt_path}")
        return summary
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统一 150M 训练入口（AMD 680M / V100 兼容）"
    )
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--corpus-jsonl", type=Path, default=None)
    parser.add_argument("--source-filter", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-loss", type=float, default=1.5)
    parser.add_argument("--max-loss", type=float, default=3.9)
    parser.add_argument("--seed", type=int, default=42, help="150M 变体训练种子（多样性指纹）")
    args = parser.parse_args()

    summary = train_150m(
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
        seed=args.seed,
    )
    if rank() == 0:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
