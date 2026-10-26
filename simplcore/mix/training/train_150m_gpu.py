from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Iterator, List, Optional

import simplcore.tensor as torch

# torch_directml 已替换为自研引擎（CPU模式）：DirectML 不可用时回退到 CPU 语义
class _DirectMLCompat:
    def device(self, idx=0): return 'cpu'
    def device_count(self): return 0
    def empty_cache(self): pass

try:
    torch_directml = _DirectMLCompat()
except ImportError:
    torch_directml = None

from simplcore.mix.config import mix_150m_text
from simplcore.mix.core import SCFunctional
from simplcore.mix.model import SimplCoreMix
from simplcore.mix.optim import SCAdamW
from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction


def require_directml_device() -> torch.device:
    """强制使用 AMD/DirectML GPU；不可用时直接失败，不回退 CPU。"""
    try:
        torch_directml = _DirectMLCompat()
    except ImportError as exc:
        raise RuntimeError("未安装 torch_directml，禁止回退 CPU 训练") from exc

    device = torch_directml.device()
    probe = torch.ones(1, device=device)
    if str(probe.device) == "cpu":
        raise RuntimeError("DirectML 设备探测落到 CPU，已停止训练")
    return device


def stable_parquet_files(corpus_dir: Path, max_shards: int | None = None) -> List[Path]:
    """锁定当前已完成分片，避免读到正在下载的新文件。"""
    files = sorted(corpus_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 parquet 分片: {corpus_dir}")
    if max_shards is not None:
        files = files[:max_shards]
    return files


def text_stream_from_parquet(files: List[Path], text_column: str = "text") -> Iterator[str]:
    import pyarrow.parquet as pq

    while True:
        for file_path in files:
            table = pq.read_table(file_path, columns=[text_column])
            for value in table[text_column].to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value


def text_stream_from_jsonl(
    file_path: Path,
    *,
    text_column: str = "text",
    source_filter: Optional[str] = None,
) -> Iterator[str]:
    """从用户授权 JSONL 语料循环读取文本。

    JSONL 每行至少包含 text 字段，可选包含 source 字段。source_filter
    用于优先训练某一批作品，例如 Anhunqu xinyuan。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到 JSONL 语料: {file_path}")
    source_filter_norm = source_filter.lower() if source_filter else None
    while True:
        matched_any = False
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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
    """轻量字符哈希编码，用于先启动 150M 训练链路。"""
    token_ids = [2 + (ord(ch) % max(vocab_size - 2, 1)) for ch in text]
    if len(token_ids) < seq_len + 1:
        token_ids = token_ids + [0] * (seq_len + 1 - len(token_ids))
    return torch.tensor(token_ids[: seq_len + 1], dtype=torch.long)


def batch_stream(texts: Iterator[str], *, vocab_size: int, batch_size: int, seq_len: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batch = []
    for text in texts:
        batch.append(encode_text(text, vocab_size, seq_len))
        if len(batch) == batch_size:
            tokens = torch.stack(batch, dim=0)
            yield tokens[:, :-1], tokens[:, 1:]
            batch = []


def train_150m_gpu(
    *,
    corpus_dir: Optional[Path],
    corpus_jsonl: Optional[Path],
    output_dir: Path,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    max_shards: int | None,
    lr: float,
    target_loss: float,
    max_loss: float,
    source_filter: Optional[str] = None,
) -> dict:
    device = require_directml_device()
    if corpus_jsonl is not None:
        text_stream = text_stream_from_jsonl(corpus_jsonl, source_filter=source_filter)
        source_paths = [corpus_jsonl]
    elif corpus_dir is not None:
        files = stable_parquet_files(corpus_dir, max_shards=max_shards)
        text_stream = text_stream_from_parquet(files)
        source_paths = files
    else:
        raise ValueError("必须提供 --corpus-jsonl 或 --corpus-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = mix_150m_text()
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)
    cfg.gradient_checkpointing = True

    model = SimplCoreMix(cfg).to(device)
    model.train()
    optimizer = SCAdamW(model.parameters(), lr=lr)
    guard = LossGuard(target_loss=target_loss, max_loss=max_loss)
    batches = batch_stream(
        text_stream,
        vocab_size=cfg.vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    history = []
    started_at = time.time()
    last_loss = None
    completed_steps = 0

    for step in range(1, max_steps + 1):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, n_loops=1)
        loss = SCFunctional.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        decision = guard.check(loss_value, step=step)
        history.append(decision.to_dict())
        last_loss = loss_value
        completed_steps = step
        print(f"[150M-GPU] step={step} loss={loss_value:.4f} action={decision.action.value}")

        # ── 显存回收 (DirectML 集成GPU显存有限) ──
        del loss, logits, x, y
        gc.collect()
        try:
            torch_directml.empty_cache()
        except Exception:
            pass

        if decision.action == LossGuardAction.STOP:
            break

    # ── 保存 checkpoint (DirectML 张量必须先移到 CPU) ──
    checkpoint_path = output_dir / "mix_150m_gpu_latest.pt"
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    opt_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in optimizer.state_dict().items()
    }
    # opt state 里的张量也要移到 CPU
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
            "device": str(device),
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
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "num_source_shards": len(source_paths),
        "source_filter": source_filter,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (output_dir / "mix_150m_gpu_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-only 150M SimplCore Mix training launcher")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--corpus-jsonl", type=Path, default=None)
    parser.add_argument("--source-filter", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-loss", type=float, default=1.5)
    parser.add_argument("--max-loss", type=float, default=3.9)
    args = parser.parse_args()

    summary = train_150m_gpu(
        corpus_dir=args.corpus_dir,
        corpus_jsonl=args.corpus_jsonl,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_shards=args.max_shards,
        lr=args.lr,
        target_loss=args.target_loss,
        max_loss=args.max_loss,
        source_filter=args.source_filter,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
