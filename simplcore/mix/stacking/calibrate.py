# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""堆叠后校准执行器（Calibrator）。

修复命门 3 / 10：
- 命门 3: `build_calibration_plan` 只有计划文本，本文件提供**可执行**的校准训练。
- 命门 10: 堆叠后的模型初始 loss 会异常偏高（重复模式 + 随机 router），
  直接套用 LossGuard 的 max_loss=3.9 会在第一步就触发 STOP。
  本校准器提供 `guard_warmup_steps` 容忍窗口，前 N 步只记录不裁决。

设计：
  - 低学习率短程继续训练，让堆叠注入的权重"消化"（校准而非预训练）。
  - 双设备兼容（AMD 680M / V100），复用 `training.device_utils`。
  - 保留可追踪的 loss 历史 + 检查点。

用法：
    from simplcore.mix.stacking.calibrate import run_calibration
    summary = run_calibration(
        model=stacked_model,
        text_stream=text_iter,          # 迭代产生中文文本
        output_dir=Path("artifacts/calib_7b"),
        steps=2000, lr=5e-5, guard_warmup_steps=100,
    )
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Iterator, Optional

import simplcore.tensor as torch

from simplcore.mix.core import SCFunctional
from simplcore.mix.optim import SCAdamW
from simplcore.mix.training.device_utils import (
    empty_device_cache,
    resolve_train_device,
    state_dict_to_cpu,
)
from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction


def encode_text_chars(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """字符哈希编码（占位 tokenizer，见命门 4；正式 BPE 后续替换）。"""
    token_ids = [2 + (ord(ch) % max(vocab_size - 2, 1)) for ch in text]
    if len(token_ids) < seq_len + 1:
        token_ids = token_ids + [0] * (seq_len + 1 - len(token_ids))
    return torch.tensor(token_ids[: seq_len + 1], dtype=torch.long)


def _batch_stream(
    texts: Iterator[str], *, vocab_size: int, batch_size: int, seq_len: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batch = []
    for text in texts:
        batch.append(encode_text_chars(text, vocab_size, seq_len))
        if len(batch) == batch_size:
            tokens = torch.stack(batch, dim=0)
            yield tokens[:, :-1], tokens[:, 1:]
            batch = []


def run_calibration(
    model: torch.nn.Module,
    text_stream: Iterator[str],
    *,
    output_dir: Path,
    steps: int = 2000,
    batch_size: int = 4,
    seq_len: int = 512,
    lr: float = 5e-5,
    target_loss: float = 1.5,
    max_loss: float = 3.9,
    guard_warmup_steps: int = 100,
    stall_patience: int = 20,
    improvement_threshold: float = 0.01,
    checkpoint_every: int = 500,
    mixed_precision: bool = True,
) -> dict:
    """在堆叠后的模型上做低 lr 短程校准训练。

    校准的 LossGuard 与预训练不同（修复命门 10）：
      堆叠后的初始 loss 会异常偏高（重复模式 + 随机 router），
      用预训练的 max_loss=3.9 会在第一步就误杀。因此校准阶段判定
      改为"是否持续改善"而非"是否低于绝对阈值"：
        - 前 guard_warmup_steps 步：只记录，不裁决（放行堆叠初始值）
        - 之后：loss 持续改善 → improving；
                连续 stall_patience 步无改善 → stalled 停止；
                loss 爆炸（> max_loss*3）→ diverged 停止。
    """
    device, backend, is_ddp = resolve_train_device()
    if backend == "directml":
        mixed_precision = False  # DirectML 无 AMP 路径，保持 fp32 稳定

    model.to(device)
    model.train()
    optimizer = SCAdamW(model.parameters(), lr=lr)

    vocab_size = model.cfg.vocab_size
    batches = _batch_stream(
        text_stream, vocab_size=vocab_size, batch_size=batch_size, seq_len=seq_len
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started_at = time.time()
    last_loss: Optional[float] = None
    completed_steps = 0
    baseline_loss: Optional[float] = None  # warmup 结束时 loss 基线
    stall_count = 0

    for step in range(1, steps + 1):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if mixed_precision and backend == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, n_loops=1)
                loss = SCFunctional.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1)
                )
        else:
            logits = model(x, n_loops=1)
            loss = SCFunctional.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        last_loss = loss_value
        completed_steps = step

        # ── 校准判定（堆叠场景专用，命门 10 修复）──
        if step <= guard_warmup_steps:
            action_str = "warmup"
            decision_stop = False
            baseline_loss = loss_value
        else:
            if loss_value > max_loss * 3:
                action_str = "diverged"
                decision_stop = True
            elif baseline_loss is None or loss_value < baseline_loss * (1 - improvement_threshold):
                # 相比基线明显改善 → 重置 stall 计数，更新基线
                if baseline_loss is None or loss_value < baseline_loss:
                    baseline_loss = loss_value
                stall_count = 0
                action_str = "improving"
                decision_stop = False
            else:
                stall_count += 1
                if stall_count >= stall_patience:
                    action_str = "stalled"
                    decision_stop = True
                else:
                    action_str = f"flat({stall_count}/{stall_patience})"
                    decision_stop = False
        history.append({"step": step, "loss": loss_value, "action": action_str})
        print(
            f"[CALIB] step={step}/{steps} loss={loss_value:.4f} "
            f"action={action_str} lr={lr} backend={backend}",
            flush=True,
        )

        if step % checkpoint_every == 0:
            ckpt = output_dir / f"calib_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": state_dict_to_cpu(model.state_dict()),
                    "config": model.cfg.__dict__,
                    "loss": loss_value,
                    "backend": backend,
                },
                ckpt,
            )
            print(f"[CALIB] checkpoint 已保存: {ckpt}", flush=True)

        if decision_stop:
            print(f"[CALIB] 校准终止 ({action_str}) at step {step}", flush=True)
            break

        del loss, logits, x, y
        gc.collect()
        empty_device_cache(backend)

    ckpt_path = output_dir / "calib_final.pt"
    torch.save(
        {
            "step": completed_steps,
            "model_state_dict": state_dict_to_cpu(model.state_dict()),
            "config": model.cfg.__dict__,
            "loss": last_loss,
            "backend": backend,
        },
        ckpt_path,
    )

    summary = {
        "status": "completed" if completed_steps == steps else "stopped",
        "steps": completed_steps,
        "last_loss": last_loss,
        "backend": backend,
        "checkpoint_path": str(ckpt_path),
        "guard_warmup_steps": guard_warmup_steps,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (output_dir / "calib_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
