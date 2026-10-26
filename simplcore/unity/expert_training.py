# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity · 专家模型训练框架 (Expert Training Framework).

在 AMD Radeon 680M (集成显卡 / 12CU / ~2GB 共享内存 / 无独立显存 / 无 ROCm)
这类弱设备上, 直接训练 12B 模型在物理上不可行. 本项目采用的务实且更聪明的路线是:

    1. 先用 **真实中文语料** (simplcore/data/corpus) 分领域训练若干个
       100M~200M 的 **专家小模型 (Expert)** —— 每个专家专精一个领域
       (古文 / 现代文 / 科普 / 哲学 ...);
    2. 再用 **SimplUnity** 把这些专家 "Unity" 到一起 (专家集成 / 路由融合),
       得到一个覆盖多领域、且可在线自进化的整体系统.

本模块提供:
    * ExpertTrainer    —— 单个专家的训练器 (next-unit 语言建模, 真实反向传播);
    * build_expert_config —— 针对 680M 显存预算的 100M/200M 级配置;
    * train_domain_expert —— 一键训练并保存某个领域的专家 checkpoint.

全部基于自研 simplcore.tensor (NumPy 后端, 零 PyTorch 依赖), 与既有
LatticeModel / UnitMap 无缝衔接.
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
import simplcore.tensor.nn.functional as F
import simplcore.tensor.optim as optim
from simplcore.tensor.optim import CosineAnnealingLR

from simplcore.architecture.lattice_model import LatticeLM, LatticeConfig, parameter_count
from simplcore.data.corpus import RealCorpus, UnitMap


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class ExpertTrainConfig:
    """单个专家的训练超参."""
    domain: str = "default"
    seq_len: int = 128
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 50
    total_steps: int = 2000
    log_interval: int = 50
    save_interval: int = 500
    grad_clip: float = 1.0
    seed: int = 1234
    output_dir: str = "output/experts"
    # ── 性能/显存优化 ──
    use_checkpoint: bool = False     # 梯度检查点（省激活内存, GPU 大专家推荐开）
    mixed_precision: bool = False    # fp16 混合精度（权重 fp16, 主副本 fp32）
    amp_dtype: Any = None            # 显式 dtype（如 torch.float16）


# 680M 预算下的专家规模预设 (纯字符级 vocab ~1200, 估算参数量)
# hidden_size / num_layers 已标定到 ~100M 与 ~200M 区间.
EXPERT_PRESETS: Dict[str, Dict[str, int]] = {
    "100m": dict(hidden_size=768, num_layers=12, num_bands=8, kernel_size=31,
                  num_tables=64, slots_per_table=128, ffn_ratio=4),
    "200m": dict(hidden_size=1024, num_layers=10, num_bands=8, kernel_size=31,
                  num_tables=64, slots_per_table=128, ffn_ratio=4),
}


def build_expert_config(vocab_size: int, preset: str = "100m",
                        seq_len: int = 1024) -> LatticeConfig:
    """构造一个目标规模 (~100M/200M) 的专家模型配置.

    Args:
        vocab_size: 紧凑 Unit 表大小 (通常 ~1200 量级).
        preset: "100m" 或 "200m".
        seq_len: 训练序列长度 (680M 上建议 <=1024, 配合梯度累积).
    """
    p = EXPERT_PRESETS[preset]
    return LatticeConfig(
        vocab_size=vocab_size,
        hidden_size=p["hidden_size"],
        num_layers=p["num_layers"],
        num_bands=p["num_bands"],
        kernel_size=p["kernel_size"],
        num_tables=p["num_tables"],
        slots_per_table=p["slots_per_table"],
        ffn_ratio=p["ffn_ratio"],
        max_seq_len=seq_len,
        dropout=0.0,
    )


# ----------------------------------------------------------------------
# 数据: 把领域文本打包成训练流
# ----------------------------------------------------------------------
class DomainDataset:
    """把一个领域的真实文本打包成固定长度序列流 (next-unit 预测)."""

    def __init__(self, texts: List[str], unit_map: UnitMap, seq_len: int) -> None:
        self.unit_map = unit_map
        self.seq_len = seq_len
        # 把所有段落编码后拼接成一个长序列 (用 <pad> 分隔以便模型学习边界)
        ids: List[int] = []
        pad = unit_map.PAD
        for t in texts:
            enc = unit_map.encode(t)
            ids.extend(enc)
            ids.append(pad)  # 段间分隔
        # 过滤过短
        self.ids = [i for i in ids if i >= 0]
        self.n = len(self.ids)
        # 总样本数 (滑窗, 步长=1 会太多; 这里用不重叠+1 的滑窗以提速)
        self.n_samples = max(0, self.n - seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def sample_batch(self, batch_size: int, rng) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采 batch 个 (x, y) 对, y = x 右移一位 (next-unit 预测)."""
        xs, ys = [], []
        L = self.seq_len
        for _ in range(batch_size):
            start = rng.randint(0, max(1, self.n - L - 1))
            x = self.ids[start: start + L]
            y = self.ids[start + 1: start + L + 1]
            # 不足补 pad
            if len(x) < L:
                x = x + [self.unit_map.PAD] * (L - len(x))
            if len(y) < L:
                y = y + [self.unit_map.PAD] * (L - len(y))
            xs.append(x)
            ys.append(y)
        xt = torch.tensor(xs, dtype=torch.long)
        yt = torch.tensor(ys, dtype=torch.long)
        return xt, yt


# ----------------------------------------------------------------------
# 训练器
# ----------------------------------------------------------------------
class ExpertTrainer:
    """单个专家模型训练器."""

    def __init__(self, cfg: ExpertTrainConfig, model: LatticeLM,
                 dataset: DomainDataset) -> None:
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.step = 0
        self.loss_history: List[float] = []
        self._pending_rng_state = b""
        os.makedirs(cfg.output_dir, exist_ok=True)

        # ── 混合精度: 维护 fp32 主副本, 前向 cast 到 fp16 ──
        self.mixed_precision = cfg.mixed_precision
        self._master_params: List[np.ndarray] = []
        if self.mixed_precision:
            for p in model.parameters():
                if p.data.dtype != np.float16:
                    p.data = p.data.astype(np.float16)
                self._master_params.append(p.data.astype(np.float32))
            print(f"[Expert:{cfg.domain}] 混合精度已启用 (fp16 权重 + "
                  f"fp32 主副本)")

        self.optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.total_steps, eta_min=cfg.lr * 0.1
        )
        # 梯度裁剪
        from simplcore.tensor.nn.utils import clip_grad_norm_
        self.clip = clip_grad_norm_

    def _lr_warmup(self) -> None:
        if self.step < self.cfg.warmup_steps:
            scale = (self.step + 1) / max(1, self.cfg.warmup_steps)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr * scale

    def train_step(self, rng) -> float:
        import gc
        self.model.train()
        xt, yt = self.dataset.sample_batch(self.cfg.batch_size, rng)
        self.optimizer.zero_grad()
        out = self.model(xt)
        logits = out["logits"]                      # (B, L, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yt.reshape(-1)
        )
        loss.backward()
        if self.mixed_precision:
            # 混合精度: 把 fp16 梯度 cast 到 fp32 主副本, 优化器更新主副本,
            # 再 cast 回 fp16 权重。
            self._master_tensors = []
            for p, mp in zip(self.model.parameters(), self._master_params):
                if p.grad is None:
                    continue
                t = torch.tensor(mp.copy(), requires_grad=True)
                t.grad = p.grad.astype(np.float32)
                self._master_tensors.append(t)
            # 用独立优化器更新主副本（与模型参数解耦）
            self._master_optim = optim.AdamW(
                self._master_tensors, lr=self.optimizer.param_groups[0]["lr"],
                betas=(0.9, 0.95), eps=1e-8, weight_decay=self.cfg.weight_decay,
            )
            self._master_optim.zero_grad()
            for t in self._master_tensors:
                t.grad = t.grad  # 已设置
            self.clip(self._master_tensors, self.cfg.grad_clip)
            self._master_optim.step()
            # cast 回 fp16 权重
            params = list(self.model.parameters())
            for p, t in zip(params, self._master_tensors):
                p.data = t.data.astype(np.float16)
        else:
            self.clip(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        # 显式释放本步计算图, 避免跨步累积导致内存增长
        loss_val = float(loss.data.ravel()[0])
        del out, logits, loss, xt, yt
        gc.collect()
        return loss_val

    def run(self, rng_seed: Optional[int] = None,
            resume: bool = False) -> Dict[str, float]:
        import random
        rng = random.Random(rng_seed if rng_seed is not None else self.cfg.seed)
        t0 = time.time()

        if resume and getattr(self, "step", 0) > 0:
            # 续训：注入 RNG + 跳过已训步数
            self._inject_rng(rng)
            start = self.step
            print(f"\n[Expert:{self.cfg.domain}] 续训 (从第 {start} 步继续, "
                  f"目标 {self.cfg.total_steps} 步)")
        else:
            start = 0
            print(f"\n[Expert:{self.cfg.domain}] 开始训练 "
                  f"(目标 {self.cfg.total_steps} 步, seq={self.cfg.seq_len}, "
                  f"batch={self.cfg.batch_size})")

        for s in range(start, self.cfg.total_steps):
            loss = self.train_step(rng)
            self.loss_history.append(loss)
            if (s + 1) % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  step {s+1:5d} | loss {loss:.4f} | lr {lr:.2e} | "
                      f"{(s+1)/ (time.time()-t0):.2f} step/s")
            if (s + 1) % self.cfg.save_interval == 0:
                # 存续训包（含 RNG/优化器/调度器）
                self.save_resume_bundle(self.cfg.output_dir, rng)
        # 末态也存续训包（便于最终 resume 或导出）
        self.save_resume_bundle(self.cfg.output_dir, rng)
        elapsed = time.time() - t0
        print(f"[Expert:{self.cfg.domain}] 完成. 末损 {self.loss_history[-1]:.4f} "
              f"耗时 {elapsed:.1f}s")
        return {
            "domain": self.cfg.domain,
            "final_loss": self.loss_history[-1],
            "first_loss": self.loss_history[0],
            "steps": self.cfg.total_steps,
            "elapsed_s": elapsed,
        }

    # ── 容错续训：保存完整训练态（参数 + 优化器 + 调度器 + RNG + step + memory）─
    def save(self, out_dir: str, tag: str = "final", rng_state: bytes = b"") -> str:
        """保存完整训练态，支持断点续训（崩了不重来）。

        存储内容：
            - 模型参数 state_dict
            - 优化器状态 (AdamW 动量 m/v, 已改用稳定键, 续训不丢)
            - 调度器状态 (last_epoch)
            - 当前 step / loss_history
            - HashMemory 记忆状态 (若有)
            - RNG 随机种子状态 (用于精确恢复采样)
        """
        from simplcore.tensor.io import save
        path = os.path.join(out_dir, f"expert_{self.cfg.domain}_{tag}.pt")

        # 模型参数
        states = {n: p.data for n, p in self.model.named_parameters()}

        # HashMemory 记忆状态（若模型暴露）
        memory_state = None
        if hasattr(self.model, "get_memory_state"):
            with torch.no_grad():
                memory_state = self.model.get_memory_state()

        meta = {
            "domain": self.cfg.domain,
            "step": self.step,
            "vocab_size": self.model.cfg.vocab_size,
            "config": vars(self.model.cfg),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_last_epoch": self.scheduler.last_epoch,
            "loss_history": self.loss_history,
            "rng_state": rng_state,
            "memory_state": memory_state,
            "backend": torch.get_backend(),
            "torch_version": "simplcore-native",
        }
        save({"state_dict": states, "meta": meta}, path)
        return path

    def save_resume_bundle(self, out_dir: str, rng) -> str:
        """保存带 RNG 状态的续训包（每 save_interval 调用）。"""
        rng_state = b""
        try:
            rng_state = repr(rng.getstate()).encode("utf-8")
        except Exception:
            pass
        return self.save(out_dir, tag=f"step{self.step}", rng_state=rng_state)

    @classmethod
    def resume(cls, cfg: "ExpertTrainConfig", model: LatticeLM,
               dataset: "DomainDataset", path: str) -> "ExpertTrainer":
        """从 checkpoint 恢复训练器（含优化器/调度器/RNG/step）。"""
        trainer = cls(cfg, model, dataset)
        from simplcore.tensor.io import load
        ckpt = load(path)
        sd = ckpt["state_dict"]
        meta = ckpt.get("meta", {})

        # 1) 恢复参数
        for n, p in model.named_parameters():
            if n in sd:
                p.data = np.array(sd[n], dtype=p.data.dtype)

        # 2) 恢复优化器
        if "optim_state" in meta:
            trainer.optimizer.load_state_dict(meta["optim_state"])
        # 3) 恢复调度器
        if "scheduler_last_epoch" in meta:
            trainer.scheduler.last_epoch = meta["scheduler_last_epoch"]
        # 4) 恢复 step / loss_history
        trainer.step = int(meta.get("step", 0))
        trainer.loss_history = list(meta.get("loss_history", []))
        # 5) 恢复 memory
        if meta.get("memory_state") is not None and hasattr(model, "set_memory_state"):
            model.set_memory_state(meta["memory_state"])
        # 6) RNG 状态稍后由 run() 注入
        trainer._pending_rng_state = meta.get("rng_state", b"")
        return trainer

    def _inject_rng(self, rng):
        state = getattr(self, "_pending_rng_state", b"")
        if state:
            try:
                import ast
                rng.setstate(ast.literal_eval(state.decode("utf-8")))
            except Exception:
                pass


# ----------------------------------------------------------------------
# 一键训练某领域专家
# ----------------------------------------------------------------------
def train_domain_expert(
    domain: str,
    corpus: RealCorpus,
    preset: str = "100m",
    seq_len: int = 128,
    total_steps: int = 2000,
    batch_size: int = 8,
    output_dir: str = "output/experts",
    seed: int = 1234,
    verbose: bool = True,
) -> Dict[str, float]:
    """训练并保存一个领域的专家模型, 返回训练统计."""
    if not corpus._loaded:
        corpus.load(verbose=False)
    texts = corpus.get_domain_texts(domain)
    if not texts:
        raise ValueError(f"领域 '{domain}' 无语料")
    unit_map = corpus.make_unit_map()
    cfg = build_expert_config(len(unit_map), preset=preset, seq_len=seq_len)
    if verbose:
        print(f"专家配置 ({domain}, {preset}): 参数量约 "
              f"{parameter_count(cfg)/1e6:.1f}M, vocab={len(unit_map)}")
    model = LatticeLM(cfg)
    ds = DomainDataset(texts, unit_map, seq_len=seq_len)
    ecfg = ExpertTrainConfig(
        domain=domain, seq_len=seq_len, batch_size=batch_size,
        total_steps=total_steps, output_dir=output_dir, seed=seed,
    )
    trainer = ExpertTrainer(ecfg, model, ds)
    return trainer.run(rng_seed=seed)


if __name__ == "__main__":
    # 快速自测: 训练 classical 领域的微型专家 (极小配置, 几秒完成)
    c = RealCorpus().load(verbose=False)
    um = c.make_unit_map()
    tiny_cfg = LatticeConfig(
        vocab_size=len(um), hidden_size=64, num_layers=2,
        num_bands=4, kernel_size=15, num_tables=8, slots_per_table=16,
        ffn_ratio=2, max_seq_len=128,
    )
    model = LatticeLM(tiny_cfg)
    ds = DomainDataset(c.get_domain_texts("classical"), um, seq_len=64)
    ecfg = ExpertTrainConfig(domain="classical", seq_len=64, batch_size=8,
                             total_steps=300, output_dir="output/experts")
    tr = ExpertTrainer(ecfg, model, ds)
    res = tr.run(rng_seed=1)
    print("自测结果:", res)
