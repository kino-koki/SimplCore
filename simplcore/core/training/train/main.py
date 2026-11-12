# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""训练主循环：DDP 分片、梯度累积、JSONL 指标、断点续训。"""
from __future__ import annotations


from .cli import (_load_enhance_prompts, assert_train_window_fits,
                  build_config, parse_args,
                  train_window_bytes)
from .checkpoint import load_checkpoint, save_checkpoint
from .data_windows import LMWindows, resolve_data_dir, unpack_lm_batch
from .distributed import setup_distributed
from .evaluate import evaluate
from .optim import apply_rocm_tuning, build_param_groups, cosine_with_warmup

import json
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ...model.model import SimplCoreModel
from ...model.config import SimplCoreConfig
from ..seqlen_curriculum import build_curriculum, SeqLenCurriculum
from ..training_integration import build_enhanced_model


def _ncm_chunk_contribution(aux: dict, total_counts: dict,
                           config: SimplCoreConfig) -> torch.Tensor:
    """Scale chunk auxiliary loss sums to the per-microbatch source means."""
    weighted = aux["ncm_loss"].new_zeros(())
    for source, weight_name in (
        ("future_latent", "ncm_future_weight"),
        ("concept_labels", "ncm_label_weight"),
        ("token_groups", "ncm_group_weight"),
    ):
        count = int(total_counts.get(source, 0))
        if count:
            loss_sum = aux.get(f"{source}_loss_sum")
            if loss_sum is not None:
                weighted = weighted + (
                    getattr(config, weight_name) * loss_sum / count)
    return weighted



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
        from ..data import MANIFEST_NAME, read_manifest
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
    if cfg.use_ncm and args.cuda_graph:
        raise SystemExit(
            "--cuda-graph currently does not support NCM auxiliary objectives")

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
            # 与主 train_loader 同口径的传输参数（此前硬编码 num_workers=0：
            # 多域隔离 warm-up 时每个域都在主进程串行解码 —— MI250 上这是
            # 实测过的输入瓶颈，GPU 利用率被 IO 拖低）。
            domain_loaders.append(DataLoader(
                ds, batch_size=args.micro_batch, shuffle=True,
                num_workers=args.num_workers, drop_last=True,
                pin_memory=(device.type == "cuda"),
                persistent_workers=args.num_workers > 0))
        if domain_loaders:
            from ..training.training_integration import warmup_unity_domains
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
    # R-edge（0.1-α）：--optimizer adafactor——factored 二阶矩，状态 O(m+n)
    # 而非 O(mn)：2GB 级机器上 1 亿参数可训（AdamW 16B/参数直接爆）。
    # 端侧实测：108.5M 参数 + Adafactor ≈ 0.87 GB，300 步真实训练
    # train −63.8% / val −43.8%（docs/EDGE_TRAINING_REPORT.md）。
    if getattr(args, "optimizer", "adamw") == "adafactor":
        from .optim import Adafactor
        optim = Adafactor(groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(groups, lr=args.lr, betas=tuple(args.betas),
                                  eps=1e-8)

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
                                  persistent_workers=args.num_workers > 0,
                                  prefetch_factor=(2 if args.num_workers > 0 else None))
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False,
                                num_workers=max(0, args.num_workers // 2),
                                drop_last=False,
                                pin_memory=(device.type == "cuda"),
                                persistent_workers=args.num_workers // 2 > 0
                                ) if val_ds else None
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
    pending_ncm_aux_w = 0.0
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
            from ...eval.gate2 import evaluate_gate2
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
        step_ncm_aux_w = 0.0
        for _ in range(accum):
            if train_loader is None:
                x = torch.randint(0, cfg.vocab_size, (args.micro_batch, cur_seq_len),
                                  device=device)
                y = torch.randint(0, cfg.vocab_size, (args.micro_batch, cur_seq_len),
                                  device=device)
                ncm_labels = ncm_groups = None
            else:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        # 每个 epoch 换洗牌种子，否则多卡永远吃到同一批窗口
                        train_sampler.set_epoch(epoch)
                    batch_iter = iter(train_loader)
                    batch = next(batch_iter)
                x, y, ncm_labels, ncm_groups = unpack_lm_batch(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if ncm_labels is not None:
                    ncm_labels = ncm_labels.to(device, non_blocking=True)
                if ncm_groups is not None:
                    ncm_groups = ncm_groups.to(device, non_blocking=True)
            if cfg.use_ncm:
                if cfg.ncm_label_classes > 0 and ncm_labels is None:
                    raise ValueError(
                        "NCM label classes are configured but the training "
                        "batch has no ncm_label_targets")
                if cfg.ncm_group_classes > 0 and ncm_groups is None:
                    raise ValueError(
                        "NCM group classes are configured but the training "
                        "batch has no ncm_group_targets")
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
                    ncm_total_counts = {
                        "future_latent": x.shape[0] * x.shape[1],
                        "concept_labels": (
                            int((ncm_labels != -100).sum().item())
                            if ncm_labels is not None else 0),
                        "token_groups": (
                            int((ncm_groups != -100).sum().item())
                            if ncm_groups is not None else 0),
                    }
                    st = rm.init_stream(x.shape[0], device=x.device,
                                        dtype=rm.tok_emb.weight.dtype)
                    positions = list(range(0, x.shape[1], args.train_chunk))
                    chunk_loss_sum = 0.0
                    chunk_ncm_aux_sum = 0.0
                    for ci, pos in enumerate(positions):
                        end = min(pos + args.train_chunk, x.shape[1])
                        xc, yc = x[:, pos:end], y[:, pos:end]
                        labels_c = (ncm_labels[:, pos:end]
                                    if ncm_labels is not None else None)
                        groups_c = (ncm_groups[:, pos:end]
                                    if ncm_groups is not None else None)
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
                                if cfg.use_ncm:
                                    _, loss_mean, aux = model(
                                        xc, yc, stream_state=st,
                                        ncm_label_targets=labels_c,
                                        ncm_group_targets=groups_c,
                                        return_ncm_aux=True)
                                    ncm_contribution = _ncm_chunk_contribution(
                                        aux, ncm_total_counts, cfg)
                                else:
                                    _, loss_mean = model(
                                        xc, yc, stream_state=st)
                                    ncm_contribution = loss_mean.new_zeros(())
                                train_loss = (
                                    loss_mean * (n_part / n_total)
                                    + ncm_contribution)
                            if n_part > 0 or cfg.use_ncm:
                                (train_loss / accum).backward()
                        chunk_loss_sum += float(loss_mean.detach()) * n_part
                        chunk_ncm_aux_sum += float(ncm_contribution.detach())
                        st = rm._detach_stream_state(st)
                    step_loss_w += (chunk_loss_sum / n_total) * y.numel()
                    step_ncm_aux_w += chunk_ncm_aux_sum * y.numel()
                else:
                    with ctx:
                        if cfg.use_ncm:
                            _, total_loss, aux = model(
                                x, y, ncm_label_targets=ncm_labels,
                                ncm_group_targets=ncm_groups,
                                return_ncm_aux=True)
                            aux_loss = aux["ncm_loss"]
                            ce_loss = total_loss - aux_loss
                            step_ncm_aux_w += float(aux_loss.detach()) * y.numel()
                        else:
                            _, ce_loss = model(x, y)
                            total_loss = ce_loss
                        step_loss_w += float(ce_loss.detach()) * y.numel()
                        total_loss = total_loss / accum
                    total_loss.backward()
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
        pending_ncm_aux_w += step_ncm_aux_w

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
            g_ncm_aux_w = _all_reduce_sum(pending_ncm_aux_w)
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
            if cfg.use_ncm:
                rec["ncm_aux_loss"] = round(
                    g_ncm_aux_w / max(g_tokens, 1), 6)
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
            pending, pending_loss_w, pending_ncm_aux_w = 0, 0.0, 0.0
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
                if ev.get("ncm_aux"):
                    rec["ncm_aux"] = ev["ncm_aux"]
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
