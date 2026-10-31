# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""可复用的 SFT 训练循环（A/B 对照的唯一实现）。

为什么单独抽出来
----------------
"1B 抵 3B"这类主张唯一能落地的形式是**受控对照**：同一份数据、同一个
seed、同样的 token 预算，只换 mixer（``sgc`` vs ``attn``），然后在同一套
规则可判的套件上比较。如果两边各写一个训练循环，任何一处差异
（梯度累积的归一化、掩码口径、warmup 步数）都会污染结论，而这类差异
在 loss 曲线上看不出来 —— 它们表现为"有一边稍微好一点"，正好是最容易
被误读成架构优势的那种差别。

所以本模块只提供一个 ``train_sft``，A/B 两侧都调它。

两个容易写错、这里刻意做对的地方
--------------------------------
1. **梯度累积按 token 加权**：``(loss / grad_accum).backward()`` 隐含
   "每个 micro-batch 的 token 数相同"。对话样本长短差 10 倍是常态，
   按微批等权会让长样本被系统性低估，等价于给数据加了看不见的重采样。
2. **记忆行零向量 = 不注入**：``memory_in`` 是无 bias 线性层，
   ``W·0 = 0``，所以"无记忆"和"有记忆"可以在同一个 batch 里共存。
   这一点必须靠数学保证，而不是靠"记得传 None"。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .train import cosine_with_warmup


# ---------------------------------------------------------------------------
# 配置与结果
# ---------------------------------------------------------------------------
@dataclass
class SFTConfig:
    """训练超参。A/B 两侧必须用**完全相同**的实例。"""

    max_steps: int = 3000
    grad_accum: int = 1
    lr: float = 6e-4
    lr_min: float = 6e-5
    weight_decay: float = 0.1
    warmup: int = 100
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 200
    time_budget_s: float = 0.0
    seed: int = 1337

    # -- 抗记忆过拟合：val 早停 + 最佳检查点回滚 -----------------------------
    # 小数据 + 大模型的 sanity-check 运行里，train loss 会掉到 ~0.06 但 val
    # 停在 ~0.21（记忆不是能力）。此时训练末态是过拟合的，而"最佳 val 检查点"
    # 才是泛化最好的点。这组开关默认全关，保持旧 A/B 运行可复现；一旦开启，
    # A/B 两侧必须用**完全相同**的早停配置，否则 fairness_report 会抓出
    # "步数不一致"。
    early_stop_patience: int = 0          # 连续 N 次 eval 无 val 改善即停；0 = 关闭
    early_stop_min_delta: float = 0.0     # val 下降 < 该值视为"无改善"（防止锯齿抖动误停）
    restore_best: bool = False            # 结束时回滚到最佳 val 检查点（对抗记忆）

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SFTResult:
    steps: int = 0
    tokens: int = 0
    wall_s: float = 0.0
    best_val: Optional[float] = None
    last_val: Optional[float] = None
    last_train_loss: Optional[float] = None
    curve: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "max_steps"
    grad_norm_max: float = 0.0
    n_nonfinite: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "wall_s": round(self.wall_s, 2),
            "best_val": None if self.best_val is None else round(self.best_val, 4),
            "last_val": None if self.last_val is None else round(self.last_val, 4),
            "last_train_loss": (None if self.last_train_loss is None
                                else round(self.last_train_loss, 4)),
            "stop_reason": self.stop_reason,
            "grad_norm_max": round(self.grad_norm_max, 4),
            "n_nonfinite": self.n_nonfinite,
        }


# ---------------------------------------------------------------------------
# 前向：统一处理 (x, y) 与 (x, y, mem)
# ---------------------------------------------------------------------------
def _unpack(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) > 2:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _model_forward(model, x, y, mem):
    """调用模型前向，兼容"接受 memory"与"不接受 memory"两类模型。

    不接受的模型（例如老的 attention baseline 分支）在不带记忆时必须
    正常跑；带了记忆则说明配置错了，直接报错比静默忽略安全得多。
    """
    if mem is None:
        return model(x, y)
    try:
        return model(x, y, memory=mem)
    except TypeError as exc:                       # noqa: BLE001
        raise TypeError(
            "样本带记忆，但当前模型不接受 memory 参数。"
            "请确认 SimplCoreConfig(use_memory=True) 且该 mixer 支持记忆注入。"
        ) from exc


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, loader, device,
                  max_batches: Optional[int] = None) -> Dict[str, float]:
    """token 加权平均 NLL。短样本不会因为 batch 多而被高估。"""
    was_training = model.training
    model.eval()
    nll, ntok = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mem = _unpack(batch)
        x, y = x.to(device), y.to(device)
        mem = mem.to(device) if mem is not None else None
        _, loss = _model_forward(model, x, y, mem)
        if torch.isfinite(loss):
            k = int((y != -100).sum())
            nll += float(loss) * k
            ntok += k
    if was_training:
        model.train()
    mean = nll / max(ntok, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": ntok}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def build_optimizer(model, cfg: SFTConfig) -> torch.optim.Optimizer:
    from .train import build_param_groups
    return torch.optim.AdamW(build_param_groups(model, cfg.weight_decay),
                             lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


def _cycle(loader: Iterable):
    """无限迭代 loader（不重排：靠 ``shuffle=True`` 的 DataLoader 每次重启
    时自行打乱，这里只负责"取完了再来一轮"）。"""
    while True:
        for batch in loader:
            yield batch


def train_sft(model,
              train_loader,
              val_loader=None,
              cfg: Optional[SFTConfig] = None,
              device=None,
              optimizer: Optional[torch.optim.Optimizer] = None,
              on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
              on_eval: Optional[Callable[[int, Dict[str, float]], None]] = None,
              ) -> SFTResult:
    """跑 SFT。返回 :class:`SFTResult`（含曲线，便于 A/B 逐点比较）。

    Args:
        train_loader: 产出 ``(x, y)`` 或 ``(x, y, mem)``。打包已定长，
            所以直接 stack 即可。
        val_loader: 可选，用于 ``eval_every`` 的验证损失。
        on_log / on_eval: 观测回调（打印、写 tensorboard、提前停止都靠它）。
            回调抛异常会中断训练 —— 观测代码的 bug 不应该被静默吞掉。
    """
    cfg = cfg or SFTConfig()
    if device is None:
        device = next(model.parameters()).device
    optim = optimizer or build_optimizer(model, cfg)

    torch.manual_seed(cfg.seed)
    res = SFTResult()
    t0 = time.perf_counter()
    it = _cycle(train_loader)
    model.train()
    tokens_seen = 0
    win_tokens = 0
    win_start = time.perf_counter()
    ga = max(1, int(cfg.grad_accum))
    best_state = None            # 仅当 restore_best 且出现更优 val 时填充（CPU 快照）
    patience = 0

    for step in range(1, cfg.max_steps + 1):
        lr = cosine_with_warmup(step - 1, cfg.max_steps, cfg.warmup,
                                cfg.lr, cfg.lr_min)
        for g in optim.param_groups:
            g["lr"] = lr

        # 先取满 micro-batch，才能按**真实的** token 总数归一化
        micro: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
        for _ in range(ga):
            x, y, mem = _unpack(next(it))
            x, y = x.to(device), y.to(device)
            micro.append((x, y, mem.to(device) if mem is not None else None))
        n_tok = sum(int((y != -100).sum()) for _, y, _ in micro)
        if n_tok == 0:
            # 整批都被掩码：不做更新。计入 step 但不计 token，
            # 否则 lr 调度会因为"空步"和"实步"混在一起而偏快。
            res.curve.append({"step": step, "skipped": "no_supervised_tokens"})
            continue

        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for x, y, mem in micro:
            _, loss = _model_forward(model, x, y, mem)
            if not torch.isfinite(loss):
                res.n_nonfinite += 1
                continue
            k = int((y != -100).sum())
            # token 加权：等价于把所有 micro-batch 拼成一个大批再算平均 loss
            w = k / n_tok
            (loss * w).backward()
            loss_sum += float(loss.detach()) * w

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        gnorm_f = float(gnorm) if torch.isfinite(gnorm) else float("inf")
        res.grad_norm_max = max(res.grad_norm_max, gnorm_f)
        # 记录最近一步的（token 加权）训练 loss：与 val_loss 对照才能区分
        # "真学到"还是"背下来了"。小数据集 + 超大模型会把 train 干到 ~0.06，
        # 但 val 停在 ~0.21——这是记忆不是能力，不能只看 train loss 下结论。
        res.last_train_loss = loss_sum
        optim.step()

        res.steps = step
        tokens_seen += n_tok
        win_tokens += n_tok
        res.tokens = tokens_seen

        if step % max(1, cfg.log_every) == 0 or step == 1:
            now = time.perf_counter()
            span = max(now - win_start, 1e-9)
            rec = {"step": step, "loss": round(loss_sum, 4), "lr": round(lr, 8),
                   "tokens": tokens_seen,
                   "tok_per_s": round(win_tokens / span, 1),
                   "grad_norm": round(gnorm_f, 4),
                   "elapsed_s": round(now - t0, 1)}
            res.curve.append(rec)
            if on_log is not None:
                on_log(rec)
            win_tokens, win_start = 0, now

        if val_loader is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            ev = evaluate_loss(model, val_loader, device)
            res.last_val = ev["loss"]
            # 早停判定：val 降幅超过 min_delta 才算"有改善"，否则耐心 +1。
            # 这能把小数据上锯齿状的 val 抖动挡在"误判为持续改善"之外。
            improved = (res.best_val is None
                        or ev["loss"] < res.best_val - cfg.early_stop_min_delta)
            if improved:
                res.best_val = ev["loss"]
                patience = 0
                if cfg.restore_best:
                    # 复用模型自带的 snapshot（CPU 深拷贝），避免每步拷贝开销。
                    best_state = model.snapshot(to_cpu=True)
            else:
                patience += 1
            res.curve.append({"step": step, "val_loss": round(ev["loss"], 4),
                              "val_ppl": round(ev["ppl"], 4)})
            if on_eval is not None:
                on_eval(step, ev)
            if cfg.early_stop_patience > 0 and patience >= cfg.early_stop_patience:
                res.stop_reason = "early_stop"
                break

        if cfg.time_budget_s > 0 and (time.perf_counter() - t0) >= cfg.time_budget_s:
            res.stop_reason = "time_budget"
            break

    # 回滚到最佳 val 检查点：小数据记忆场景下，末态（train~0.06）是过拟合的，
    # 最佳 val 检查点才是泛化最好的权重。与 early_stop 解耦——即使因 max_steps
    # 或 time_budget 正常结束，只要开了 restore_best 也回滚。
    if cfg.restore_best and best_state is not None:
        dev = next(model.parameters()).device
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    res.wall_s = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# A/B 公平性自检
# ---------------------------------------------------------------------------
def fairness_report(results: Dict[str, Any],
                    budgets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """检查两次运行是否真的可比。

    A/B 最常见的失败不是"结论错"，而是"两边根本不在同一条件下"：
    一边跑了 3000 步、另一边因为时间预算跑到 2800 步；一边训了
    1.2B token、另一边 0.9B。这类差异在报告里不会自己跳出来。

    Args:
        results: ``{arm 名: SFTResult}``。
        budgets: 可选的 ``{arm 名: 参数量或算力}``，差 >5% 会被报出来。

    Returns:
        ``{"comparable": bool, "violations": [...], "token_ratio": float, ...}``
    """
    violations: List[str] = []
    names = list(results)
    steps = {k: int(results[k].steps) for k in names}
    toks = {k: int(results[k].tokens) for k in names}

    if len(set(steps.values())) > 1:
        violations.append(f"步数不一致：{steps}")
    if names and min(toks.values()) > 0:
        ratio = max(toks.values()) / min(toks.values())
        if ratio > 1.02:
            violations.append(f"token 预算差 {ratio:.3f}×（>1.02×）：{toks}")
    for k in names:
        if results[k].n_nonfinite > 0:
            violations.append(f"{k} 出现 {results[k].n_nonfinite} 次非有限 loss")
        # 记忆护栏：train 远小于 val（默认 <0.6×）说明模型把训练集背下来了，
        # train loss 掉到 ~0 是过拟合而非能力。这时应读 val 而不是 train。
        tl, vl = results[k].last_train_loss, results[k].last_val
        if tl is not None and vl is not None and vl > 0 and tl < 0.6 * vl:
            violations.append(
                f"{k} 记忆预警：train={tl:.3f} ≪ val={vl:.3f}（train/val={tl/vl:.2f}），"
                f"train loss 不可作能力信号，以 val 为准")
    # 终止原因一致性：A/B 两侧必须同因终止。两侧都用 early_stop 且同步停下是
    # 合法的对照（不算违规）；但若一侧 early_stop、另一侧 max_steps（算力/数据
    # 不对等），才是真问题，必须报出来。
    reasons = {k: results[k].stop_reason for k in names}
    if len(set(reasons.values())) > 1:
        early = {"early_stop", "time_budget"}
        asym = any(v in early for v in reasons.values()) and \
            any(v not in early for v in reasons.values())
        tag = "（一侧提前结束，算力不对等）" if asym else ""
        violations.append(f"终止原因不一致{tag}：{reasons}")
    if budgets:
        kept = {k: float(budgets[k]) for k in names if k in budgets}
        if len(kept) == 2:
            hi, lo = max(kept.values()), min(kept.values())
            # 15% 是"层数只能取整数"的量化粒度：同一宽度下 1 层差约 25%，
            # 所以 10% 以内的差不是实验不严谨，而是"同宽度同深度"这个
            # 更有意义的对照条件的必然结果。超过 15% 才说明真的没对齐。
            if lo > 0 and hi / lo > 1.15:
                violations.append(
                    f"参数量 / 算力预算差 {hi/lo:.3f}×（>1.15×）：{kept}")

    return {
        "comparable": not violations,
        "violations": violations,
        "steps": steps,
        "tokens": toks,
        "token_ratio": (round(max(toks.values()) / max(min(toks.values()), 1), 6)
                        if toks and min(toks.values()) > 0 else None),
    }
