# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Phase 1 / Phase 2 训练驱动 —— 分块流式训练的课程循环。

**驱动与内核分离**：本模块只做编排（课程推进、分块 BPTT、优化器步进、
no-regression 回滚、账本），每一个张量运算都经由 :class:`BackendAdapter`——
今天跑在 ``TorchReferenceAdapter``（参考数学 oracle）上，明天把适配器换成
自研 CUDA 后端的包装，**这行代码不用改**。这就是 Phase 1 bring-up 的
驱动就绪形态：后端一过 conformance，训练循环已经在等它。

每个 batch 的训练语义
----------------------
* 序列切成 chunk；每 chunk 前向（``AdapterModel.forward_chunk``）→
  目标段 CE → **立即 backward**（截断 BPTT：chunk 间状态 detach，
  梯度窗口 = chunk，显存 O(chunk) 与总长 L 无关——战役账本的分块路径）；
* 无有效目标的 chunk 只推状态不进梯度（no_grad）；
* 优化器步进走 ``adamw_frozen_mask``（冻结位 p/m/v 逐位不动）+
  ``weight_decay_exempt``（ρ/θ 豁免 decay）；master 权重 fp32；
* no-regression gate：``maybe_commit`` 只保留比历史最优更好的 checkpoint，
  退化超过容差 → 从快照逐位回滚（快照经 ``stream_state_serialize``，
  byte-exact 语义与 conformance C-rollback 同源）；
* 课程：``ContextCurriculum`` 按 token 预算推进 1M→2M→4M 阶梯
  （测试/演示按比例缩小，口径不变）。

诚实边界
--------
* 驱动是**参考编排**，不是性能实现：吞吐、融合调度、数据管线并行是
  后端/阶段 bring-up 的事；本模块的价值是语义正确 + 驱动就绪。
* 截断 BPTT 的梯度窗口 = chunk：跨 chunk 的长程信用分配由流式状态
  前向承载（前向无损），梯度按窗口截断——这是分块流式训练的标准取舍，
  与 campaign 账本口径一致。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..model.adapter_model import AdapterModel
from ..backend.conformance_runner import BackendAdapter

__all__ = [
    "CurriculumStage",
    "ContextCurriculum",
    "TrainLedger",
    "StreamingTrainer",
    "main",
]


# ---------------------------------------------------------------------------
# 课程
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurriculumStage:
    """课程一档：上下文长度 + 该档 token 预算。"""

    seq_len: int
    max_tokens: int

    def as_dict(self) -> Dict[str, int]:
        return {"seq_len": self.seq_len, "max_tokens": self.max_tokens}


class ContextCurriculum:
    """按累计 token 数推进的上下文课程（1M→2M→4M 阶梯的通用化）。"""

    def __init__(self, stages: Sequence[CurriculumStage]) -> None:
        if not stages:
            raise ValueError("课程至少要有一档")
        self.stages: List[CurriculumStage] = list(stages)
        # 每档的起点（累计 token）
        self.starts: List[int] = []
        acc = 0
        for st in self.stages:
            self.starts.append(acc)
            acc += st.max_tokens

    @property
    def total_tokens(self) -> int:
        return self.starts[-1] + self.stages[-1].max_tokens

    def stage_for(self, tokens_seen: int) -> Tuple[int, CurriculumStage]:
        """累计 token → (档号, 档)。超出总预算时停在最后一档。"""
        for i in range(len(self.stages) - 1, -1, -1):
            if tokens_seen >= self.starts[i]:
                return i, self.stages[i]
        return 0, self.stages[0]

    def exhausted(self, tokens_seen: int) -> bool:
        return tokens_seen >= self.total_tokens

    def as_dict(self) -> Dict[str, object]:
        return {"stages": [s.as_dict() for s in self.stages],
                "total_tokens": self.total_tokens}


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
@dataclass
class TrainLedger:
    """训练账本：只记事实（步数/token/回滚/课程推进），不记自评。"""

    steps: int = 0
    tokens_seen: int = 0
    rollbacks: int = 0
    commits: int = 0
    stage_index: int = 0
    losses: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"steps": self.steps, "tokens_seen": self.tokens_seen,
                "rollbacks": self.rollbacks, "commits": self.commits,
                "stage_index": self.stage_index,
                "loss_first5": self.losses[:5], "loss_last5": self.losses[-5:]}


# ---------------------------------------------------------------------------
# 训练驱动
# ---------------------------------------------------------------------------
class StreamingTrainer:
    """分块流式训练驱动（课程 + BPTT + 冻结掩码优化器 + 回滚闸门）。"""

    def __init__(self, model: AdapterModel, adapter: BackendAdapter,
                 curriculum: ContextCurriculum, *, lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.1,
                 adamw_eps: float = 1e-8, frozen_names: Optional[Set[str]] = None,
                 rollback_tol: float = 0.02) -> None:
        self.model = model
        self.adapter = adapter
        self.curriculum = curriculum
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.adamw_eps = adamw_eps
        self.frozen_names = frozenset(frozen_names or ())
        self.rollback_tol = rollback_tol
        self.ledger = TrainLedger()
        self.best_eval: Optional[float] = None
        self._best_snap: Optional[Dict[str, bytes]] = None
        # 优化器状态（master fp32；m/v 全 fp32，16 B/参数口径）
        self.opt_m: Dict[str, torch.Tensor] = {}
        self.opt_v: Dict[str, torch.Tensor] = {}
        names = model.param_names()
        self.exempt: Dict[str, bool] = {
            k: bool(adapter.weight_decay_exempt([k])[0]) for k in names}
        for k, p in model.named_parameters():
            self.opt_m[k] = torch.zeros_like(p)
            self.opt_v[k] = torch.zeros_like(p)

    # -- 课程 ------------------------------------------------------------------
    def stage(self) -> Tuple[int, CurriculumStage]:
        return self.curriculum.stage_for(self.ledger.tokens_seen)

    # -- 单序列分块训练（截断 BPTT：chunk 即梯度窗口） ---------------------------
    def _train_sequence(self, idx: torch.Tensor, targets: torch.Tensor,
                        chunk: int) -> float:
        model = self.model
        states = model.init_stream_states(idx.shape[0])
        length = 0
        total, n_valid = 0.0, 0
        for s in range(0, idx.shape[1], chunk):
            uc = idx[:, s:s + chunk]
            tc = targets[:, s:s + chunk]
            if bool((tc != -100).any()):
                _, loss = model.forward_chunk(uc, states, length, tc)
                loss.backward()                    # 梯度窗口 = chunk
                total += float(loss.detach())
                n_valid += 1
            else:
                # 无目标段：只推状态，不进梯度（省显存与时间）
                with torch.no_grad():
                    model.forward_chunk(uc, states, length, None)
            length += uc.shape[1]
            # chunk 间状态 detach：梯度不跨窗
            for st in states:
                for k in st:
                    st[k] = st[k].detach()
        return total / max(n_valid, 1)

    # -- 优化器（经适配器的冻结掩码 AdamW） --------------------------------------
    def _optimizer_step(self) -> None:
        self.ledger.steps += 1
        step = self.ledger.steps
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                g = p.grad
                if g is None:
                    continue
                frozen = torch.tensor(name in self.frozen_names) if name in self.frozen_names \
                    else torch.zeros((), dtype=torch.bool)
                wd = 0.0 if self.exempt[name] else self.weight_decay
                p_new, m, v = self.adapter.adamw_frozen_mask(
                    p, g, self.opt_m[name], self.opt_v[name], frozen,
                    self.lr, self.beta1, self.beta2, self.adamw_eps, wd, step)
                p.copy_(p_new)
                self.opt_m[name], self.opt_v[name] = m, v
                p.grad = None

    def train_step(self, batch_idx: torch.Tensor, batch_targets: torch.Tensor
                   ) -> Dict[str, float]:
        """一个 batch：分块前向/回传 → 优化器步进 → 账本/课程推进。"""
        stage_i, stage = self.stage()
        if batch_idx.shape[1] != stage.seq_len:
            raise ValueError(
                f"当前课程档 seq_len={stage.seq_len}，收到 {batch_idx.shape[1]}")
        loss = self._train_sequence(batch_idx, batch_targets, chunk=min(
            stage.seq_len, max(stage.seq_len // 4, 1)))
        self._optimizer_step()
        n_tokens = int(batch_idx.numel())
        self.ledger.tokens_seen += n_tokens
        self.ledger.losses.append(loss)
        new_i, _ = self.stage()
        self.ledger.stage_index = new_i
        return {"loss": loss, "tokens": n_tokens, "stage": stage_i}

    # -- 评估 / 快照 / 回滚 -------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, idx: torch.Tensor, targets: torch.Tensor, chunk: int) -> float:
        """流式评估（无梯度）：全序列 CE。"""
        _, loss = self.model.forward_streaming(idx, chunk, targets)
        return float(loss)

    def snapshot(self) -> Dict[str, bytes]:
        """参数 + 优化器状态的 byte-exact 快照（经适配器序列化）。"""
        blob: Dict[str, bytes] = {}
        for name, p in self.model.named_parameters():
            blob["p:" + name] = self.adapter.stream_state_serialize({"t": p.detach()})
        for name in self.opt_m:
            blob["m:" + name] = self.adapter.stream_state_serialize({"t": self.opt_m[name]})
            blob["v:" + name] = self.adapter.stream_state_serialize({"t": self.opt_v[name]})
        return blob

    def restore(self, snap: Dict[str, bytes]) -> None:
        """从快照逐位恢复（C-rollback 同源语义）。"""
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for key, blob in snap.items():
                kind, name = key.split(":", 1)
                t = self.adapter.stream_state_deserialize(blob)["t"]
                if kind == "p":
                    params[name].copy_(t)
                elif kind == "m":
                    self.opt_m[name].copy_(t)
                else:
                    self.opt_v[name].copy_(t)

    def maybe_commit(self, eval_idx: torch.Tensor, eval_targets: torch.Tensor,
                     chunk: int) -> Dict[str, object]:
        """no-regression gate：更好→提交为新最优；退化超容差→逐位回滚。"""
        ev = self.evaluate(eval_idx, eval_targets, chunk)
        out: Dict[str, object] = {"eval_loss": ev, "rollback": False,
                                  "committed": False, "best": self.best_eval}
        if self.best_eval is None or ev < self.best_eval:
            self.best_eval = ev
            self._best_snap = self.snapshot()
            self.ledger.commits += 1
            out["committed"] = True
        elif ev > self.best_eval * (1.0 + self.rollback_tol):
            self.restore(self._best_snap)
            self.ledger.rollbacks += 1
            out["rollback"] = True
        return out

    # -- checkpoint（断点续训；与 snapshot 同源的 byte-exact 语义） ---------------
    CKPT_FORMAT = "simplcore-ckpt-v1"

    def save_checkpoint(self, path, *, data_state: Optional[Dict[str, object]] = None,
                        extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """持久化训练状态：参数 + m/v + 账本 + 最优快照 + RNG。

        * 状态全部经 ``snapshot``（适配器序列化）——与 C-rollback 同源，
          byte-exact 语义由 conformance 守护，这里不另造一套序列化；
        * ``data_state`` 携带数据源游标（如 CorpusBatches 的 cursor/epochs），
          续训时数据从同一位置继续——否则"断点续训"只是参数续训；
        * ``best_eval`` 与 ``_best_snap`` 一并保存：恢复后 no-regression gate
          的历史基线不丢，不会出现"重启后把退化当最优"的口径断裂。
        """
        meta: Dict[str, object] = {
            "adapter": self.adapter.name,
            "lr": self.lr, "betas": [self.beta1, self.beta2],
            "weight_decay": self.weight_decay, "adamw_eps": self.adamw_eps,
            "frozen_names": sorted(self.frozen_names),
            "rollback_tol": self.rollback_tol,
            "curriculum": self.curriculum.as_dict(),
            "ledger": self.ledger.as_dict(),
            "losses_full": list(self.ledger.losses),
            "best_eval": self.best_eval,
            "data_state": dict(data_state or {}),
        }
        if extra:
            meta.update(extra)
        torch.save({"format": self.CKPT_FORMAT, "meta": meta,
                    "state": self.snapshot(),
                    "best_state": dict(self._best_snap or {}),
                    "rng": torch.get_rng_state()}, str(path))
        return meta

    @classmethod
    def load_checkpoint(cls, path, model: AdapterModel, adapter: BackendAdapter,
                        curriculum: Optional[ContextCurriculum] = None,
                        *, strict_curriculum: bool = True
                        ) -> Tuple["StreamingTrainer", Dict[str, object]]:
        """从 checkpoint 恢复：返回 (trainer, meta)。课程/超参不一致即拒绝。

        ``strict_curriculum``：默认要求课程逐档一致（训练预算被换掉而参数
        看起来还在续训，是最隐蔽的账目造假）；显式传 False 才允许换课程。
        """
        import torch as _torch
        blob = _torch.load(str(path), weights_only=False)
        if blob.get("format") != cls.CKPT_FORMAT:
            raise ValueError(f"checkpoint 格式不符: {blob.get('format')!r} "
                             f"(期望 {cls.CKPT_FORMAT!r})")
        meta = blob["meta"]
        if meta["adapter"] != adapter.name:
            raise ValueError(f"adapter 不一致: ckpt={meta['adapter']!r} "
                             f"current={adapter.name!r}")
        cur = ContextCurriculum([CurriculumStage(**s) for s in
                                 meta["curriculum"]["stages"]])
        if curriculum is not None:
            if strict_curriculum and curriculum.as_dict() != cur.as_dict():
                raise ValueError("课程与 checkpoint 不一致（strict_curriculum=True）")
            cur = curriculum
        tr = cls(model, adapter, cur, lr=meta["lr"],
                 betas=tuple(meta["betas"]), weight_decay=meta["weight_decay"],
                 adamw_eps=meta["adamw_eps"],
                 frozen_names=set(meta["frozen_names"]),
                 rollback_tol=meta["rollback_tol"])
        tr.restore(blob["state"])
        tr.best_eval = meta["best_eval"]
        tr._best_snap = dict(blob["best_state"]) or None
        led = meta["ledger"]
        tr.ledger.steps = int(led["steps"])
        tr.ledger.tokens_seen = int(led["tokens_seen"])
        tr.ledger.rollbacks = int(led["rollbacks"])
        tr.ledger.commits = int(led["commits"])
        tr.ledger.stage_index = int(led["stage_index"])
        tr.ledger.losses = list(meta["losses_full"])
        _torch.set_rng_state(blob["rng"])
        return tr, meta


# ---------------------------------------------------------------------------
# CLI 演示（tiny 课程实跑，loss 必须可观测下降）
# ---------------------------------------------------------------------------
def _tiny_config():
    from ..model.config import SimplCoreConfig
    return SimplCoreConfig(vocab_size=32, dim=32, num_layers=2, num_bases=8,
                           local_kernel=4, ffn_ratio=2.0, norm="rmsnorm",
                           decay_max=256.0, tie_embeddings=False)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SimplCore 分块流式训练驱动（tiny 演示）")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args(argv)

    from ..backend.conformance_runner import TorchReferenceAdapter

    torch.manual_seed(7)
    cfg = _tiny_config()
    from ..model.model import SimplCoreModel
    native = SimplCoreModel(cfg)
    model = AdapterModel(cfg, TorchReferenceAdapter())
    model.load_native(native.state_dict())

    # 课程演示：3 档 × 8/16/32 长度（真实战役是 1M/2M/4M，口径相同）
    curriculum = ContextCurriculum([CurriculumStage(16, 16 * 4 * 10),
                                    CurriculumStage(32, 32 * 4 * 10),
                                    CurriculumStage(64, 64 * 4 * (args.steps // 3 + 1))])
    trainer = StreamingTrainer(model, model.adapter, curriculum, lr=args.lr)

    def make_batch(seq_len: int, batch: int = 4):
        # 周期模式数据：模型必须学会数数 → loss 可观测下降
        base = torch.arange(seq_len + 1) % cfg.vocab_size
        x = base[:-1].unsqueeze(0).repeat(batch, 1)
        y = base[1:].unsqueeze(0).repeat(batch, 1)
        return x, y

    print(f"=== StreamingTrainer tiny 演示（adapter={model.adapter.name}）===")
    first = last = None
    for _ in range(args.steps):
        si, st = trainer.stage()
        x, y = make_batch(st.seq_len)
        r = trainer.train_step(x, y)
        if first is None:
            first = r["loss"]
        last = r["loss"]
        if trainer.ledger.steps % 10 == 0:
            e = trainer.evaluate(*make_batch(st.seq_len), chunk=args.chunk)
            print(f" step {trainer.ledger.steps:3d} | stage {si} (L={st.seq_len}) | "
                  f"train {r['loss']:.4f} | eval {e:.4f} | tokens {trainer.ledger.tokens_seen}")
    # no-regression gate 演示
    gate = trainer.maybe_commit(*make_batch(curriculum.stages[-1].seq_len),
                                chunk=args.chunk)
    print(f" gate: eval={gate['eval_loss']:.4f} committed={gate['committed']} "
          f"rollback={gate['rollback']}")
    ok = last is not None and last < first
    print(f"== loss {first:.4f} → {last:.4f}（{'下降 ✓' if ok else '未下降 ✗'}）；"
          f"steps={trainer.ledger.steps} rollbacks={trainer.ledger.rollbacks} ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
