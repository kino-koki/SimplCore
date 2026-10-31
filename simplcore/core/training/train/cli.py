# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""命令行参数与配置构建 + 训练窗口显存预算账本。"""
from __future__ import annotations



import argparse
import os
from typing import Iterator, Optional


from ...model.config import SimplCoreConfig, get_preset



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
    ap.add_argument("--optimizer", choices=("adamw", "adafactor"),
                    default="adamw",
                    help="adafactor = factored 二阶矩（2GB 级机器上 1 亿参数可训）")
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

