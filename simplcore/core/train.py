# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore training loop.

Designed for 1 x AMD Instinct MI250 (128 GB HBM2e, bf16, sequences of 4096)
while remaining fully runnable on CPU for small-scale convergence proof.

Features that matter for a *real* training run, not a demo:

* bf16 autocast on CUDA (no GradScaler needed) with fp32 parameter/optimizer state;
* TF32 matmuls on Ampere+ (CDNA2 also honours the same matmul precision knobs);
* AdamW with decoupled weight decay applied only to matrices (norms, biases, the
  oscillator ``feed`` term and ``out_scale`` are excluded) -- this matters because
  weight decay on the oscillator parameters destroys long-range memory;
* warmup + cosine decay, gradient clipping, gradient accumulation;
* multi-GPU data parallelism via ``torchrun`` + ``DistributedDataParallel``
  (see "Distributed training" below);
* optional ``torch.compile`` and activation checkpointing, both fenced so a failure
  degrades to eager instead of crashing the run;
* resumable checkpoints (model + optimizer + scheduler + RNG + step);
* JSONL metrics plus a human-readable log, so curves can be replotted later;
* deterministic seeding and a token-budget accounting that is honest about how many
  tokens were actually consumed.

Distributed training
--------------------
The 1B/3B presets do not fit single-GPU training budgets (AdamW state alone is
~8x params). The path for them is ``torchrun`` data parallelism::

    torchrun --nproc_per_node=8 -m simplcore.core.train \\
        --preset tuan-3b --global-batch-tokens 524288 ...

Launch mechanics follow the standard PyTorch contract: when ``WORLD_SIZE > 1``
(env set by ``torchrun``) the script initialises a process group, pins each
rank to ``cuda:LOCAL_RANK``, shards the data with ``DistributedSampler``,
synchronises gradients with DDP, reduces the loss/token accounting so logged
numbers are *global*, and only rank 0 writes logs and checkpoints. Checkpoints
store the model without the DDP wrapper, so a multi-GPU run can be resumed
single-GPU and vice versa. Without ``torchrun`` everything behaves exactly as
the single-process script it always was.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import SimplCoreConfig, get_preset
from .model import SimplCoreModel
from .training_integration import build_enhanced_model
from .seqlen_curriculum import build_curriculum, SeqLenCurriculum


# ---------------------------------------------------------------------------
# distributed setup (torchrun contract)
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple:
    """按 ``torchrun`` 约定初始化进程组。

    由 ``torchrun`` 拉起时环境里有 ``WORLD_SIZE/RANK/LOCAL_RANK``：初始化
    进程组（CUDA 用 nccl，否则 gloo）并把当前进程钉到 ``cuda:LOCAL_RANK``，
    返回 ``(rank, local_rank, world_size)``。单进程直接运行时返回
    ``(0, 0, 1)``，不做任何分布式初始化——行为与历史版本完全一致。
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class LMWindows(Dataset):
    """Thin adapter exposing ``(inputs, targets)`` windows for the training loop.

    The corpus schema, sharding, SHA-256 verification and cross-shard windowing all
    live in :mod:`simplcore.core.data` -- this class deliberately does not
    re-implement them. It asks for windows one token longer than the training
    sequence length so that shifting by one yields exactly ``seq_len`` targets.
    """

    def __init__(self, root: str, split: str, seq_len: int,
                 verify_hashes: bool = True) -> None:
        from .data import ShardedTokenDataset
        self.seq_len = int(seq_len)
        self.inner = ShardedTokenDataset(root, split, seq_len=self.seq_len + 1,
                                        verify_hashes=verify_hashes)
        self.vocab_size = int(getattr(self.inner, "vocab_size", 0) or 0)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        w = self.inner[index]
        if isinstance(w, (tuple, list)):
            w = w[0]
        w = w.long()
        return w[:-1], w[1:]


def resolve_data_dir(explicit: Optional[str]) -> Optional[str]:
    from .data import MANIFEST_NAME
    if explicit:
        return explicit
    for cand in ("data/processed/tuan_zh", "data/processed/tuan_zh_smoke"):
        if os.path.exists(os.path.join(cand, MANIFEST_NAME)):
            return cand
    return None


# ---------------------------------------------------------------------------
# optimiser parameter groups
# ---------------------------------------------------------------------------
def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Decay matrices only.

    The oscillator parameters (``raw_rho``, ``raw_theta``), the instantaneous
    ``feed`` term, ``out_scale``, all norms and the embeddings are left undecayed:
    weight decay on a decay-rate parameter is a direct attack on long-range memory,
    which is the whole point of this architecture.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not any(k in name for k in ("tok_emb", "head")):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if total <= warmup:
        return lr
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def apply_rocm_tuning(force: bool = False) -> bool:
    """MI250 / ROCm 训练调优。

    PyTorch 在 ROCm 上仍把设备暴露为 ``"cuda"``，``torch.cuda.*`` 通用；分布式
    后端 ``"nccl"`` 在 ROCm 上自动映射到 RCCL。返回是否处于 ROCm 环境。

    这些旋钮必须在**首次显存分配之前**设置（``PYTORCH_HIP_ALLOC_CONF`` 是惰性
    读取的），所以本函数在 ``main`` 里确定 device 之后、建模型之前调用。
    """
    is_hip = getattr(torch.version, "hip", None) is not None
    if not (is_hip or force):
        return False
    # 展开式显存段：128GB HBM 上大幅减少碎片，避免长训练中途 OOM。
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    # 固定输入形状（micro_batch × seq_len 全程不变）下，让 MIOpen 选最快卷积算法。
    torch.backends.cudnn.benchmark = True
    print("[train] ROCm/MI250 tuning applied: expandable_segments, cudnn.benchmark=True")
    return True


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model, optim, step: int, tokens: int, cfg: SimplCoreConfig,
                    best_val: float, elapsed: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "tokens": tokens,
        "best_val": best_val,
        "elapsed": elapsed,
        "config": cfg.to_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optim=None, restore_rng: bool = True) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if restore_rng:
        try:
            torch.set_rng_state(ckpt["torch_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            random.setstate(ckpt["python_rng"])
        except Exception:
            pass
    return ckpt


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: Optional[int] = None,
             chunk_size: int = 0) -> dict:
    """Token-weighted validation loss / perplexity.

    ``chunk_size`` > 0 时走分块前向（与训练同一套跨块状态，激活 O(chunk)），
    用于超长验证窗口；0 保持整段前向的旧行为。
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
            else torch.autocast("cpu", enabled=False)
        with ctx:
            if chunk_size > 0 and hasattr(model, "forward_chunked"):
                _, loss = model.forward_chunked(
                    x, y, chunk_size=chunk_size, detach_state=False)
            else:
                _, loss = model(x, y)
        n = y.numel()
        total_nll += float(loss) * n
        total_tokens += n
    model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": total_tokens}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train an SGC language model.")
    ap.add_argument("--config", type=str, default=None,
                    help="path to a JSON config, or a preset name (e.g. tuan-150m)")
    ap.add_argument("--preset", type=str, default=None, help="preset name")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="runs/sgc")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--seq-len-curriculum", type=str, default=None,
                    help="序列长度课程（opt-in，零回归）：逗号分隔 'seq_len:截止比例'，"
                         "末段比例可省（默认 1.0）。例：4096:0.1,16384:0.25,65536:0.45,"
                         "262144:0.65,1048576:0.85,4000000。空 = 固定 --seq-len（旧行为）。"
                         "预设 context.curriculum 也可提供，CLI 优先。注意：SGC 的 O(1) "
                         "状态只覆盖推理；本训练路径仍是整段并行 FFT，核与激活为 O(L)，"
                         "超长阶段会被 --max-kernel-gib 拒绝（请改用 trainer.py 分块路径）。")
    ap.add_argument("--max-kernel-gib", type=float, default=4.0,
                    help="整段 FFT 训练窗口的 (L,D) fp32 核预算（GiB）。0=不检查。"
                         "这是整段路径的真实显存结构约束，与 O(1) 流式状态无关。"
                         "启用 --train-chunk 后预算按 chunk 长度（而非总长）计。")
    ap.add_argument("--train-chunk", type=int, default=0,
                    help="分块训练窗口（截断 BPTT）：>0 时主路径改走 "
                         "SimplCoreModel.forward_chunked，激活 O(chunk)、跨块状态 detach。"
                         "与 --enhance / --compile / --cuda-graph 不兼容（会显式报错）。")
    ap.add_argument("--gate2", action="store_true",
                    help="课程首次进入 1M 阶段时跑一次 needle 召回测量门（GATE-2）；"
                         "只记录，不在默认情况下阻塞训练。")
    ap.add_argument("--gate2-lengths", default="262144,1048576",
                    help="GATE-2 评估长度（逗号分隔）；受当前课程 seq_len 截断。")
    ap.add_argument("--gate2-depths", default="0.25,0.5,0.75",
                    help="needle 相对深度（逗号分隔）。")
    ap.add_argument("--gate2-trials", type=int, default=1,
                    help="每个深度的 trial 数；1M 长度下 1 条已需要 ~256 个 chunk。")
    ap.add_argument("--gate2-chunk", type=int, default=4096,
                    help="GATE-2 评分的分块窗口（与 --train-chunk 同源）。")
    ap.add_argument("--gate2-strict", action="store_true",
                    help="below_chance 时硬停（默认只记录到 gate2.json 并继续）。")
    ap.add_argument("--global-batch-tokens", type=int, default=None,
                    help="target tokens per optimiser step; sets accumulation automatically")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="soft stop after this many minutes (writes a checkpoint)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ignore data, train on random tokens, for stack validation")
    ap.add_argument("--skip-hash-check", action="store_true",
                    help="skip SHA-256 verification of the corpus shards (faster startup)")
    ap.add_argument("--override", type=str, nargs="*", default=[],
                    help="config overrides, e.g. dim=512 num_layers=8")
    # ---- 早停 / 最佳检查点（与 sft.py 对齐；默认全关保持旧行为）--------
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="连续 N 次 eval 无 val 改善即停；0=关闭")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="val 下降 < 该值视为'无改善'，防锯齿误停")
    ap.add_argument("--restore-best", action="store_true",
                    help="结束回滚到 best.pt（记忆场景下末态过拟合）")
    # ---- 编译 / 图 / MI250 ---------------------------------------------
    ap.add_argument("--compile-mode", type=str, default="max-autotune",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile 模式；max-autotune 在 ROCm 上选 CK 内核（大幅提速）")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="实验性：用 HIP/CUDA Graph 捕获单步 fwd+bwd（固定形状才有收益；"
                         "DDP / grad-checkpoint 下自动禁用，失败回退 eager）")
    ap.add_argument("--mi250", action="store_true",
                    help="强制套用 MI250/ROCm 调优（即使在 CUDA 机器上也能测这条路径）")
    # ---- 训练增强集成（opt-in，默认关闭；不开任何增强时逐位不变）--------
    ap.add_argument("--enhance", type=str, default="",
                    help="逗号分隔的增强列表：ponder,envlearn,accel,smm,dream,reasoning。"
                         "默认空 = 原训练逐位不变。详见 simplcore.core.training_integration")
    ap.add_argument("--enhance-prompts", type=str, default=None,
                    help="reasoning 增强用的提示（文件：每行一个；或单条字符串）")
    ap.add_argument("--unity-experts", type=int, default=4,
                    help="--enhance 含 unity 时：初始小专家数（可经 spawn_expert 增长）")
    ap.add_argument("--unity-topk", type=int, default=2,
                    help="--enhance 含 unity 时：每 token 激活的专家数（top-k 稀疏度）")
    ap.add_argument("--unity-domains", type=str, default=None,
                    help="--enhance 含 unity 时：逗号分隔的若干域语料目录（每个 = data/"
                         "processed/<域> 形态，含 manifest）。第 i 个专家只在第 i 个域上"
                         "隔离训练（强制「训练数据必须不同」），随后才进入联合 MoE 微调。")
    ap.add_argument("--unity-warmup-steps", type=int, default=200,
                    help="--unity-domains 每个域的隔离 warm-up 步数")
    # ---- 生物计算主义（--enhance bio）旋钮 --------------------------------
    ap.add_argument("--bio-energy-weight", type=float, default=0.01,
                    help="--enhance 含 bio 时：代谢效率损失权重（超额能耗受罚强度）")
    ap.add_argument("--bio-metabolic-budget", type=float, default=1.5,
                    help="--enhance 含 bio 时：每 token 归一化代谢能量预算（超过即受罚）")
    ap.add_argument("--bio-field-dt", type=float, default=0.5,
                    help="--enhance 含 bio 时：连续场基础演化步长（macro 课程会动态调制）")
    ap.add_argument("--bio-field-scale", type=float, default=1.0,
                    help="--enhance 含 bio 时：连续场→token 调制门增益基础缩放")
    ap.add_argument("--bio-adaptive-curriculum", action="store_true",
                    help="自适应序列长度课程：阶段推进由连续场就绪度（meso→macro 反馈）"
                         "门控——模型未消化当前长度时推迟拉长上下文。需同时开 --enhance bio")
    ap.add_argument("--bio-readiness-gate", type=float, default=0.5,
                    help="自适应课程的就绪度阈值（0~1）：低于该值则推迟阶段推进")
    ap.add_argument("--bio-readiness-grace", type=int, default=200,
                    help="自适应课程的宽限步数：越过阶段边界超过该步数仍未达阈值则强制推进")
    return ap.parse_args(argv)


def _load_enhance_prompts(path: Optional[str]) -> Optional[List[str]]:
    """解析 --enhance-prompts：文件则按行读，否则作为单条提示。"""
    if not path:
        return None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        return lines or None
    return [path]


def build_config(args) -> SimplCoreConfig:
    if args.config and os.path.exists(args.config):
        cfg = SimplCoreConfig.from_json(args.config)
    elif args.preset:
        cfg = get_preset(args.preset)
    elif args.config:
        cfg = get_preset(args.config)
    else:
        cfg = get_preset("tuan-150m")
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config field {k!r}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            setattr(cfg, k, v.lower() in ("1", "true", "yes"))
        elif isinstance(cur, int):
            setattr(cfg, k, int(v))
        elif isinstance(cur, float):
            setattr(cfg, k, float(v))
        elif cur is None:
            setattr(cfg, k, int(v) if v.isdigit() else v)
        else:
            setattr(cfg, k, v)
    return cfg


_KERNEL_DTYPE_BYTES = 4  # fp32 kernel bank


def train_window_bytes(seq_len: int, dim: int) -> int:
    """整段并行 FFT 训练窗口必须物化的 ``(L, D)`` 核字节数（fp32）。

    这是 ``O(L·D)`` 项，与推理路径的 ``O(1)`` 流式状态无关。主训练路径
    （``SimplCoreModel.forward``）每个 block 都会 ``build_kernel(L)``，所以
    "架构支持 4M" 与 "这条训练路径能跑 4M" 是两件事：前者由核的闭式定义
    保证，后者由本函数给出硬预算。

    SGC-R（``mixer="sgc-r"``）口径不变：分组核按组**即建即弃**（峰值仍是
    1×(L, D) 核），路由的 G 倍开销落在 ``u_g`` 激活内存（属激活预算，
    不属本核预算）。启用 --train-chunk 后按 chunk 长度计，同理不变。
    """
    return int(seq_len) * int(dim) * _KERNEL_DTYPE_BYTES


def assert_train_window_fits(seq_len: int, dim: int, max_kernel_gib: float,
                             *, where: str = "start") -> None:
    """在启动/课程换挡前拒绝"必然 OOM"的训练窗口。

    旧行为是把它留给 CUDA 在第一个长窗口上 OOM，或者在 CPU 上静默用满
    内存 —— 两种失败都不告诉用户"这不是架构不支持，而是主路径还没分块"。
    显式拒绝比事后排查便宜。
    """
    if max_kernel_gib is None or max_kernel_gib <= 0:
        return
    need = train_window_bytes(seq_len, dim)
    budget = int(max_kernel_gib * (1024 ** 3))
    if need <= budget:
        return
    raise SystemExit(
        f"[train] {where}: seq_len={seq_len:,}, dim={dim} 的整段 FFT 核需要 "
        f"{need / (1024 ** 3):.2f} GiB（fp32 (L,D)），超过 "
        f"--max-kernel-gib={max_kernel_gib}。\n"
        f"  这不是数值或架构限制：SGC 的核在任意长度有定义、流式状态是 O(1)；\n"
        f"  但本训练路径仍是整段并行 FFT，核与激活都是 O(L)。\n"
        f"  可行动作：\n"
        f"    1) 降低 seq_len / 缩短课程阶梯；\n"
        f"    2) 走分块 BPTT（与后端契约同源）：python -m simplcore.core.trainer；\n"
        f"    3) 显式提高 --max-kernel-gib（自担 OOM 风险）。")


def main(argv=None) -> int:
    args = parse_args(argv)
    rank, local_rank, world_size = setup_distributed()
    is_master = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # device / dtype（torchrun 下各 rank 钉在自己的 GPU 上）
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if args.dtype == "auto":
        amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    elif args.dtype == "fp32":
        amp_dtype = None
    else:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if device.type == "cuda":
        # TF32 是 NVIDIA Ampere+ 概念；ROCm(CDNA2) 无 TF32，设为 no-op 不影响，
        # 仅 NVIDIA 卡真正生效。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # ROCm / MI250 调优（自动探测 HIP，或 --mi250 强制测这条路径）。
    # 必须在首次显存分配前调用。
    apply_rocm_tuning(force=args.mi250)

    os.makedirs(args.out_dir, exist_ok=True)
    # ---- 配置 + 数据（先建 cfg，便于课程决定初始 seq_len）--------------------
    cfg = build_config(args)
    if args.dry_run:
        data_dir = None
        train_ds = val_ds = None
        vocab_size = cfg.vocab_size
    else:
        from .data import MANIFEST_NAME, read_manifest
        data_dir = resolve_data_dir(args.data_dir)
        if data_dir is None:
            raise SystemExit("no packed dataset found; pass --data-dir or use --dry-run")
        manifest = read_manifest(os.path.join(data_dir, MANIFEST_NAME))
        vocab_size = int(manifest["vocab_size"])
        splits = manifest.get("splits") or {}
        if "train" not in splits:
            raise SystemExit(f"{data_dir} has no 'train' split; found {sorted(splits)}")

    # ---- 序列长度课程（opt-in）：决定初始 seq_len ----------------------------
    # CLI --seq-len-curriculum 优先，其次预设 context.curriculum；都不给则固定
    # --seq-len（旧行为，零回归）。SGC 无 max_len、O(1) 状态，拉长序列不改动权重。
    curriculum = build_curriculum(args, cfg, args.max_steps)
    cur_seq_len = curriculum.seq_len_at(0) if curriculum else args.seq_len
    if curriculum and is_master:
        print(f"[train] seq-len curriculum ENABLED: stages="
              f"{curriculum.to_dict()['stages']} (max={curriculum.max_seq_len:,})")
    # 整段路径用课程的**最大长度**预检；启用 --train-chunk 后 kernel 只与
    # chunk 长度有关，预算按 chunk 计（这才是分块训练真正的显存结构）。
    _budget_len = (args.train_chunk if args.train_chunk > 0
                   else (curriculum.max_seq_len if curriculum else cur_seq_len))
    assert_train_window_fits(
        _budget_len, cfg.dim, args.max_kernel_gib,
        where="启动预检（chunk 模式）" if args.train_chunk else "启动预检")
    cfg.vocab_size = vocab_size
    cfg.grad_checkpoint = args.grad_checkpoint

    model = SimplCoreModel(cfg).to(device)
    n_params = model.num_params()
    amp_dtype = amp_dtype if device.type == "cuda" else None

    # ---- 训练增强集成（opt-in，默认关闭；不开任何增强时逐位不变）--------
    # 把 accel/nonlinear/envlearn/smm/dream/reasoning 六个控制器接入训练闭环。
    # 不开 --enhance 时 em.model 就是原基座，训练行为与历史版本完全一致。
    enhance_list = [s.strip().lower() for s in (args.enhance or "").split(",") if s.strip()]
    if args.train_chunk:
        # 分块训练只改"怎么前向"，不改"训练哪个模块"。增强 wrapper 没有
        # forward_chunked 契约，静默退回整段会让 --train-chunk 变成假开关，
        # 所以在这里显式拒绝，而不是让它"看起来生效"。
        if args.train_chunk < 0:
            raise SystemExit("--train-chunk 必须 >= 0")
        if enhance_list:
            raise SystemExit(
                "--train-chunk 目前只支持无增强基座；"
                f"检测到 --enhance {','.join(enhance_list)}。"
                "请先去掉增强，或改用整段路径。")
        if args.compile:
            raise SystemExit("--train-chunk 与 --compile 不兼容（自定义前向不会被编译）")
        if args.cuda_graph:
            raise SystemExit("--train-chunk 与 --cuda-graph 不兼容")
        if is_master:
            print(f"[train] chunked training ENABLED: chunk={args.train_chunk:,} "
                  f"(truncated BPTT, activation O(chunk), "
                  f"{'DDP no_sync across chunks' if world_size > 1 else 'single process'})")
    em = build_enhanced_model(
        model, cfg, enhance_list,
        eval_prompts=_load_enhance_prompts(args.enhance_prompts),
        unity_experts=args.unity_experts, unity_topk=args.unity_topk,
        bio_energy_weight=args.bio_energy_weight,
        bio_metabolic_budget=args.bio_metabolic_budget,
        bio_field_dt=args.bio_field_dt, bio_field_scale=args.bio_field_scale)
    if enhance_list:
        print(f"[train] enhancements enabled: {enhance_list}")
        # 显式报告"同位置替代"关系：旧实现 last-wins 静默覆盖，用户以为叠加。
        if em.superseded:
            pairs = ", ".join(f"{k} 被 {v} 替代" for k, v in em.superseded.items())
            print(f"[train] WARNING 增强互斥（同位置 wrapper）：{pairs}")
        if em.consumed:
            print(f"[train] enhancements consumed: {em.consumed}")
        if "envlearn" in em.extras:
            print(f"[train] envlearn budget: {em.extras['envlearn'].budget_dict()}")
        if "unity" in em.extras:
            print(f"[train] unity budget: {em.extras['unity'].budget_dict()}")
        if "bio" in em.extras:
            print(f"[train] bio (biological computationalism): "
                  f"{em.extras['bio'].budget_dict()}")

    # ---- Unity 按域隔离 warm-up：强制「每个专家训练数据必须不同」----------
    if "unity" in em.extras and args.unity_domains:
        domain_dirs = [d.strip() for d in args.unity_domains.split(",") if d.strip()]
        domain_loaders = []
        for d in domain_dirs:
            if not os.path.isdir(d):
                print(f"[train] unity-domain '{d}' 不存在，跳过")
                continue
            ds = LMWindows(d, "train", cur_seq_len, verify_hashes=False)
            domain_loaders.append(DataLoader(ds, batch_size=args.micro_batch,
                                             shuffle=True, num_workers=0,
                                             drop_last=True))
        if domain_loaders:
            from .training_integration import warmup_unity_domains
            print(f"[train] unity per-domain warm-up on {len(domain_loaders)} 域"
                  f"（每域 {args.unity_warmup_steps} 步，数据互不相同）")
            warmup_unity_domains(
                em.extras["unity"].unity, domain_loaders,
                steps=args.unity_warmup_steps, device=device, amp_dtype=amp_dtype)
            # warm-up 后 train_expert 只留「域专家 + 路由」可学；联合 MoE 阶段需
            # 把全部路由 + 专家重新设为可学（基座仍冻结），否则非域专家不训。
            _unity = em.extras["unity"].unity
            for p in _unity.router.parameters():
                p.requires_grad_(True)
            for ex in _unity.experts:
                for p in ex.parameters():
                    p.requires_grad_(True)
            # 注意：下方 model=em.model 之后会用 build_param_groups 重建优化器，
            # 自动仅收纳 requires_grad=True 的路由 + 专家（基座被跳过）。
    model = em.model                 # 真正参与前向/反向、被 DDP 包裹、存盘的模块

    # 增强在基座之后才构造（unity 的路由/专家、bio 的连续场/耦合器都是新模块），
    # 它们默认在 CPU 上；基座早已 .to(device)。不补这一步，GPU 上首次前向就会
    # device mismatch（CPU 训练掩盖了它）。
    model = model.to(device)

    groups = build_param_groups(model, args.weight_decay)
    optim = torch.optim.AdamW(groups, lr=args.lr, betas=tuple(args.betas), eps=1e-8)

    if args.compile:
        try:
            # max-autotune 在 ROCm 上选 Composable Kernel(CK) 内核，matmul/conv
            # 显著快于朴素 HIP 内核；首个 step 编译开销几十秒，短跑看不出来。
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[train] torch.compile enabled (mode={args.compile_mode})")
        except Exception as exc:                       # pragma: no cover
            print(f"[train] torch.compile unavailable, running eager: {exc}")

    # ---- DDP 包装（在 resume 之前也可行：DDP.load_state_dict 透传到 module，
    #      而 checkpoint 存的是裸模型 state_dict，无 "module." 前缀）----
    ddp_wrapped = False
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )
        ddp_wrapped = True
        if is_master:
            print(f"[train] DDP enabled: world_size={world_size} backend={dist.get_backend()}")

    def raw_model():
        """DDP 包装下的裸模型引用（存 checkpoint / 评测用）。"""
        return model.module if ddp_wrapped else model

    # accumulation
    if args.grad_accum is not None:
        accum = args.grad_accum
    elif args.global_batch_tokens:
        accum = max(1, round(args.global_batch_tokens / (args.micro_batch * cur_seq_len)))
    else:
        accum = 1

    start_step, tokens_consumed, best_val, elapsed0, patience = 0, 0, float("inf"), 0.0, 0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optim)
        start_step = int(ck.get("step", 0))
        # checkpoint 里的 tokens 是全局口径；单 rank 分摊 = 全局 / world_size。
        # 单进程 <-> 多卡互相 resume 时记账保持诚实（token 预算不因 rank 数漂移）。
        tokens_consumed = int(ck.get("tokens", 0)) // world_size
        best_val = float(ck.get("best_val", float("inf")))
        elapsed0 = float(ck.get("elapsed", 0.0))
        if is_master:
            print(f"[train] resumed from {args.resume} at step {start_step}")

    header = {
        "config": cfg.to_dict(),
        "n_params": n_params,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "micro_batch": args.micro_batch,
        "grad_accum": accum,
        "seq_len_initial": cur_seq_len,
        "seq_len_max": curriculum.max_seq_len if curriculum else args.seq_len,
        "seq_len_curriculum": curriculum.to_dict() if curriculum else None,
        "world_size": world_size,
        "rank": rank,
        "tokens_per_step": args.micro_batch * cur_seq_len * accum * world_size,
        "data_dir": data_dir,
        "argv": sys.argv,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    if is_master:
        print("[train] " + json.dumps({k: v for k, v in header.items() if k != "config"}))
        print(f"[train] params={n_params:,}  config={cfg.to_dict()}")
        with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as fh:
            json.dump(header, fh, indent=2, ensure_ascii=False)
        metrics_fh = open(metrics_path, "a", encoding="utf-8")
    else:
        metrics_fh = None

    # data loaders（多卡时用 DistributedSampler 分片，保证各 rank 看到不同窗口）。
    # 抽成 _make_loaders：课程切换到更长的 seq_len 时复用同一函数重建数据窗口与
    # 加载器（模型权重零改动，仅数据形状变化）。
    train_sampler = None
    train_loader = val_loader = None
    batch_iter = None

    def _make_loaders(seq_len: int):
        nonlocal train_loader, val_loader, train_sampler, batch_iter, \
            train_ds, val_ds, cur_seq_len
        cur_seq_len = seq_len
        if args.dry_run:
            train_ds = val_ds = None
            train_loader = val_loader = None
            train_sampler = None
            batch_iter = None
            return
        train_ds = LMWindows(data_dir, "train", seq_len,
                             verify_hashes=not args.skip_hash_check)
        val_ds = LMWindows(data_dir, "val", seq_len,
                           verify_hashes=not args.skip_hash_check) if "val" in splits else None
        if val_ds is None:
            print("[train] WARNING: no 'val' split in the manifest -- "
                  "you will have no overfitting early-warning signal")
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True)
        else:
            train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=args.num_workers, drop_last=True,
                                  pin_memory=(device.type == "cuda"),
                                  persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False,
                                num_workers=max(0, args.num_workers // 2),
                                drop_last=False,
                                pin_memory=(device.type == "cuda")) if val_ds else None
        batch_iter = iter(train_loader) if train_loader is not None else None

    _make_loaders(cur_seq_len)

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True
        print(f"[train] signal {signum} received; will checkpoint and stop")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except Exception:
            pass

    def _all_reduce_sum(v: float) -> float:
        """多卡时对每个 rank 的标量求和，得到全局口径（单进程原样返回）。"""
        if world_size <= 1:
            return v
        t = torch.tensor([v], dtype=torch.float64, device=device)
        dist.all_reduce(t)
        return float(t.item())

    # ---- HIP/CUDA Graph：捕获单 micro-batch 的 fwd+bwd，消除每步 kernel 启动开销
    #      （AMD 上启动开销显著；固定形状 micro_batch×seq_len 才有收益）。
    #      仅在单卡、未开 grad-checkpoint、未开 compile 时尝试；任何失败都回退 eager。
    graph_runner = None
    if args.cuda_graph and world_size == 1 and not args.grad_checkpoint and not args.compile \
            and not args.train_chunk \
            and not curriculum:
        try:
            static_x = torch.zeros(args.micro_batch, cur_seq_len, dtype=torch.long,
                                   device=device)
            static_y = torch.zeros(args.micro_batch, cur_seq_len, dtype=torch.long,
                                   device=device)
            gph = torch.cuda.CUDAGraph()
            # 预热：让 HIP/Aten 缓存与 autograd 节点就位后再捕获，避免捕获到
            # 未 warmed 的缓慢路径。
            with (torch.autocast("cuda", dtype=amp_dtype)
                  if amp_dtype is not None else torch.autocast("cpu", enabled=False)):
                _, _warm = raw_model()(static_x, static_y)
                _warm.backward()
            raw_model().zero_grad(set_to_none=True)
            with torch.cuda.graph(gph):
                with (torch.autocast("cuda", dtype=amp_dtype)
                      if amp_dtype is not None else torch.autocast("cpu", enabled=False)):
                    _, static_loss = raw_model()(static_x, static_y)
                    # 与 eager 路径一致：梯度按 accum 归一（同一步内所有 micro-batch
                    # 共享 pre-step 参数，optim.step 在 accum 循环之后统一做）。
                    (static_loss / accum).backward()
            def _replay(x, y):
                static_x.copy_(x, non_blocking=True)
                static_y.copy_(y, non_blocking=True)
                gph.replay()
                return static_loss
            graph_runner = _replay
            if is_master:
                print("[train] CUDA/HIP graph captured (fwd+bwd per micro-batch)")
        except Exception as exc:                       # pragma: no cover
            print(f"[train] CUDA/HIP graph capture failed, running eager: {exc}")
            graph_runner = None

    model.train()
    t_start = time.time()
    t_log = time.perf_counter()
    step = start_step
    micro = 0
    epoch = 0
    # 预置 lr：--max-steps 0（只做启动预检/launch dry-run）时循环不执行，
    # 末尾的 DONE 汇总仍需要一个已定义的 final_lr。
    lr = cosine_with_warmup(start_step, args.max_steps, args.warmup,
                            args.lr, args.lr_min)
    optim.zero_grad(set_to_none=True)
    gate2_at = 0                    # 最近一次 GATE-2 的 seq_len（0=未跑）
    pending = 0                    # 本日志窗口内本 rank 消费的 token 数
    pending_loss_w = 0.0           # Σ(本 micro-batch 平均 loss × token 数)，用于全局加权
    batch_iter: Iterator = iter(train_loader) if train_loader is not None else None

    # 单步 GPU 计时事件（仅 cuda 设备）。移除每步 synchronize 的 stall：
    # 事件只记录不阻塞，真正同步延迟到 log 时做一次。
    use_step_evt = device.type == "cuda"
    step_evt_s = torch.cuda.Event(enable_timing=True) if use_step_evt else None
    step_evt_e = torch.cuda.Event(enable_timing=True) if use_step_evt else None

    while step < args.max_steps and not stop["flag"]:
        lr = cosine_with_warmup(step, args.max_steps, args.warmup, args.lr, args.lr_min)
        for g in optim.param_groups:
            g["lr"] = lr

        # ---- 连续场就绪度（meso 尺度探针，每步一次）-----------------------
        # 这是 meso→macro 的反馈信号：回答「模型现在能承受更长的上下文了吗」。
        # readiness() 内部维护场能量 EMA，故每步只调用一次（同时用于课程门控
        # 与 metrics 记录，避免 EMA 被重复更新）。
        bio_readiness = None
        if "bio" in em.extras:
            bio_readiness = em.extras["bio"].readiness()

        # ---- 序列长度课程：到达新阶段则拉长 seq_len 并重建数据窗口 ----------
        # 模型权重零改动（SGC 长度无关、O(1) 状态），仅数据形状与累计步数变化。
        if curriculum:
            # 自适应模式：meso→macro 反馈。连续场就绪度不足 → 推迟拉长
            # （模型还没消化当前长度时硬拉到 4M 只会把表征练废）。
            # 没开 --enhance bio 时无就绪度信号（None）→ 退化为按调度推进，
            # 不能因为缺信号就把课程永远卡在第 0 阶段。
            if curriculum.adaptive:
                if curriculum.promote(step, readiness=bio_readiness,
                                      gate_threshold=args.bio_readiness_gate,
                                      grace_steps=args.bio_readiness_grace):
                    if is_master:
                        rs = ("n/a" if bio_readiness is None
                              else f"{bio_readiness:.3f}")
                        print(f"[train] curriculum promote -> stage "
                              f"{curriculum.achieved_stage} (readiness={rs})")
            target_seq = curriculum.seq_len_at(step)
            if target_seq != cur_seq_len:
                old = cur_seq_len
                _budget = args.train_chunk if args.train_chunk > 0 else target_seq
                assert_train_window_fits(_budget, cfg.dim, args.max_kernel_gib,
                                         where=f"课程换挡 step={step}")
                _make_loaders(target_seq)          # 内部更新 cur_seq_len 并重建迭代器
                if args.global_batch_tokens:
                    accum = max(1, round(args.global_batch_tokens
                                        / (args.micro_batch * cur_seq_len)))
                if is_master:
                    print(f"[train] curriculum step {step}: seq_len {old} -> "
                          f"{cur_seq_len:,} (accum={accum})")

        # ---- GATE-2：课程首次进入 1M 阶段时测一次 needle 召回 -----------------
        # 这是"测量门"而非质量门：轴 D 承认精确检索必输，GATE-2 只回答"到底
        # 输多少"。用 chunked 评分，避免 4M/1M 整段前向把核与激活一次性物化。
        if (args.gate2 and gate2_at == 0
                and cur_seq_len >= 1_048_576):
            gate2_at = cur_seq_len
            from .gate2 import evaluate_gate2
            lengths = [int(x) for x in args.gate2_lengths.split(",") if x.strip()]
            lengths = [L for L in lengths if L <= max(cur_seq_len, 1_048_576)]
            if not lengths:
                lengths = [min(cur_seq_len, 1_048_576)]
            depths = [float(x) for x in args.gate2_depths.split(",") if x.strip()]
            rep = None
            if is_master:
                rep = evaluate_gate2(
                    raw_model(), lengths=lengths, depths=depths,
                    trials_per_depth=args.gate2_trials,
                    chunk_size=args.gate2_chunk,
                    out_path=os.path.join(args.out_dir, "gate2.json"))
                print(f"[train] GATE-2 {rep['status']} "
                      f"recall={rep['overall']['recall']:.3f} "
                      f"chance={rep['overall']['chance']:.3f} "
                      f"worst_depth={rep['overall']['worst_depth_recall']:.3f} "
                      f"(lengths={lengths})")
            model.train()
            stop = torch.tensor([0], device=device)
            if (is_master and rep is not None
                    and rep["status"] == "below_chance" and args.gate2_strict):
                stop[0] = 1
            if world_size > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                dist.barrier()
            if bool(stop.item()):
                raise SystemExit(
                    "[train] GATE-2 below_chance：见 runs/.../gate2.json")

        # ---- 生物计算主义：注入 macro 尺度信号（课程进度 → 连续场演化步长）----
        # 这是「尺度不可分」支柱的 macro→meso 耦合入口：训练编排（高尺度）连续地
        # 约束连续动力（中尺度）。无课程时退化为 step/max_steps（仍提供宏观尺度）。
        if "bio" in em.extras:
            if curriculum and len(curriculum.stages) > 1:
                frac = curriculum.stage_index_at(step) / (len(curriculum.stages) - 1)
            else:
                frac = step / max(1, args.max_steps)
            em.extras["bio"].curriculum_frac = frac

        # ---- fetch a global batch (accumulated) ----------------------------
        if step_evt_s is not None:
            step_evt_s.record()          # 仅记录，不阻塞 CPU（避免每步 stall）
        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        step_loss_w = 0.0
        for _ in range(accum):
            if train_loader is None:
                x = torch.randint(0, cfg.vocab_size, (args.micro_batch, cur_seq_len),
                                  device=device)
                y = torch.randint(0, cfg.vocab_size, (args.micro_batch, cur_seq_len),
                                  device=device)
            else:
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        # 每个 epoch 换洗牌种子，否则多卡永远吃到同一批窗口
                        train_sampler.set_epoch(epoch)
                    batch_iter = iter(train_loader)
                    x, y = next(batch_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            if graph_runner is not None:
                # HIP/CUDA Graph 回放：把本 micro-batch 拷入静态张量后 replay，
                # 跳过每步的 kernel 启动开销。
                static_loss = graph_runner(x, y)
                step_loss_w += float(static_loss.detach()) * y.numel()
            else:
                ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
                    else torch.autocast("cpu", enabled=False)
                if args.train_chunk:
                    # 截断 BPTT：每块前向后**立即** backward，只保留当前块的
                    # 计算图 —— 这是激活真正降到 O(chunk) 的关键。把多个块的
                    # logits 拼起来再 backward 会让显存退回 O(L)。
                    #
                    # MI250 是 2×64GB GCD：用 2 进程 DDP 才能吃满 128GB 与
                    # 整卡算力。DDP 的 reducer 支持"多次 forward+backward，
                    # 最后一次同步"（标准梯度累积语义）：块间与 micro-batch
                    # 间都用 no_sync，只有 accum 周期的最后一块正常同步。
                    rm = raw_model()
                    n_total = max(int((y != -100).sum().item()), 1)
                    st = rm.init_stream(x.shape[0], device=x.device,
                                        dtype=rm.tok_emb.weight.dtype)
                    positions = list(range(0, x.shape[1], args.train_chunk))
                    chunk_loss_sum = 0.0
                    for ci, pos in enumerate(positions):
                        end = min(pos + args.train_chunk, x.shape[1])
                        xc, yc = x[:, pos:end], y[:, pos:end]
                        n_part = int((yc != -100).sum().item())
                        sync_now = (ci == len(positions) - 1
                                    and (micro + 1) % accum == 0)
                        sync_ctx = (model.no_sync()
                                    if ddp_wrapped and not sync_now
                                    else nullcontext())
                        with sync_ctx:
                            with ctx:
                                # 经 DDP.forward 进入单块带状态前向，reducer
                                # 的 prepare/同步才能与 backward 正确配对。
                                _, loss_mean = model(
                                    xc, yc, stream_state=st)
                            if n_part > 0:
                                (loss_mean * (n_part / n_total)
                                 / accum).backward()
                        chunk_loss_sum += float(loss_mean.detach()) * n_part
                        st = rm._detach_stream_state(st)
                    step_loss_w += (chunk_loss_sum / n_total) * y.numel()
                else:
                    with ctx:
                        _, loss = model(x, y)
                        step_loss_w += float(loss.detach()) * y.numel()
                        loss = loss / accum
                    loss.backward()
            pending += y.numel()
            tokens_consumed += y.numel()
            micro += 1

        if args.grad_clip and args.grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            gnorm = torch.tensor(0.0)
        optim.step()
        step += 1

        if step_evt_e is not None:
            step_evt_e.record()          # 单步 GPU 计时；真正同步延迟到 log 时做一次
        dt = time.perf_counter() - t0
        pending_loss_w += step_loss_w

        if step % args.log_every == 0 or step == 1:
            # Throughput must be measured over the *whole* logging interval.
            # Using only the last step's ``dt`` with a token count accumulated
            # across several steps inflated ``tokens_per_s`` by a factor of
            # ``log_every`` -- a number that looked 25x better than reality.
            now = time.perf_counter()
            span = max(now - t_log, 1e-9)
            # 全局口径：各 rank 的 token 与加权 loss 求和（DDP 下各 rank
            # 数据分片不同，单看 rank0 的 loss 会是有偏估计）
            g_tokens = _all_reduce_sum(float(pending))
            g_loss_w = _all_reduce_sum(pending_loss_w)
            tps = g_tokens / span
            avg = g_loss_w / max(g_tokens, 1)
            # 单步计时：CUDA/HIP 下用事件在 log 时做一次同步读 GPU 时间，
            # 避免每步 synchronize 卡住 CPU（让数据加载与下一 step 的 CPU 侧
            # 工作能与 GPU 重叠，长训练实测吞吐更高）。
            if step_evt_s is not None:
                torch.cuda.synchronize()
                dt = step_evt_s.elapsed_time(step_evt_e) / 1000.0
            rec = {"step": step, "loss": round(avg, 4), "lr": lr,
                   "grad_norm": float(gnorm), "step_s": round(dt, 3),
                   "tokens_per_s": round(tps, 1),
                   "tokens": tokens_consumed * world_size,
                   "tokens_since_log": int(g_tokens),
                   "elapsed_s": round(time.time() - t_start + elapsed0, 1)}
            # 生物计算主义代谢账本（macro 尺度 frac 也一并记录，便于观察跨尺度耦合）
            if "bio" in em.extras:
                m = em.extras["bio"].meter
                rec["bio_energy"] = m.last_energy
                rec["bio_field_energy"] = m.field_energy
                rec["bio_waste"] = m.waste
                rec["bio_curriculum_frac"] = round(
                    float(em.extras["bio"].curriculum_frac), 3)
                if bio_readiness is not None:
                    rec["bio_readiness"] = round(float(bio_readiness), 4)
                if curriculum and curriculum.adaptive:
                    rec["bio_curriculum_stage"] = curriculum.achieved_stage
            if is_master:
                if device.type == "cuda":
                    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                print("[train] " + json.dumps(rec))
                metrics_fh.write(json.dumps(rec) + "\n")
                metrics_fh.flush()
            pending, pending_loss_w = 0, 0.0
            t_log = now

        # ---- evaluation -----------------------------------------------------
        # 各 rank 对同一 val 集求值（val_loader 无 sampler，全 rank 看到同一份），
        # 权重又经 DDP 同步，因此 ev["loss"] 各 rank 完全一致 → 早停判定对称，
        # 不会出现"一侧停一侧不停"的不可比。
        if val_loader is not None and (step % args.eval_every == 0 or step == args.max_steps):
            ev = evaluate(raw_model(), val_loader, device, amp_dtype, args.eval_batches,
                          chunk_size=args.train_chunk)
            enh_extra = {}
            for h in em.eval_hooks:
                try:
                    out = h(em, val_loader, device, amp_dtype)
                    if out:
                        enh_extra.update(out)
                except Exception as exc:                       # pragma: no cover
                    print(f"[eval] enhance hook failed: {exc}")
            if is_master:
                rec = {"step": step, "val_loss": round(ev["loss"], 4),
                       "val_ppl": round(ev["ppl"], 4), "val_tokens": ev["tokens"],
                       "elapsed_s": round(time.time() - t_start + elapsed0, 1)}
                if enh_extra:
                    rec["enhance"] = enh_extra
                print("[eval] " + json.dumps(rec))
                metrics_fh.write(json.dumps(rec) + "\n")
                metrics_fh.flush()
            # 早停判定（每个 rank 都算，保证对称停止）；只有主 rank 落盘 best。
            improved = ev["loss"] < best_val - args.early_stop_min_delta
            if improved:
                best_val = ev["loss"]
                patience = 0
                if is_master:
                    save_checkpoint(os.path.join(args.out_dir, "best.pt"),
                                    raw_model(), optim, step,
                                    tokens_consumed * world_size, cfg, best_val,
                                    time.time() - t_start + elapsed0)
            else:
                patience += 1
            if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
                if is_master:
                    print(f"[train] early-stop at step {step} "
                          f"(patience={patience}, best_val={best_val:.4f})")
                stop["flag"] = True

        if is_master and step % args.save_every == 0:
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), raw_model(), optim,
                            step, tokens_consumed * world_size, cfg, best_val,
                            time.time() - t_start + elapsed0)

        if args.time_budget_min > 0 and (time.time() - t_start) / 60.0 >= args.time_budget_min:
            if is_master:
                print(f"[train] time budget {args.time_budget_min} min reached at step {step}")
            stop["flag"] = True

    if is_master:
        # ---- 训练后增强（dream 自进化巩固等，opt-in）--------------------
        for h in em.post_hooks:
            try:
                res = h(em, val_loader)
                print("[train] post-hook result: "
                      + json.dumps(res, default=str, ensure_ascii=False))
            except Exception as exc:                       # pragma: no cover
                print(f"[train] post-hook failed: {exc}")
        # 回滚到最佳 val 检查点：记忆场景下训练末态（train 很低）是过拟合的，
        # best.pt 才是泛化最好的权重。载入后再写 last.pt，使最终产物即最佳点。
        if args.restore_best and best_val < float("inf"):
            best_path = os.path.join(args.out_dir, "best.pt")
            if os.path.exists(best_path):
                load_checkpoint(best_path, raw_model(), restore_rng=False)
                print(f"[train] restore_best: loaded best.pt (val={best_val:.4f})")
        save_checkpoint(os.path.join(args.out_dir, "last.pt"), raw_model(), optim,
                        step, tokens_consumed * world_size, cfg, best_val,
                        time.time() - t_start + elapsed0)
    summary = {"steps": step, "tokens": tokens_consumed * world_size,
               "best_val_loss": best_val,
               "wall_s": round(time.time() - t_start + elapsed0, 1),
               "params": n_params, "final_lr": lr, "world_size": world_size}
    if is_master:
        print("[train] DONE " + json.dumps(summary))
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        metrics_fh.close()
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
