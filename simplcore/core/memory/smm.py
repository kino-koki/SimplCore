# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Simpl Model Memory（SMM）—— SimplCore 的**最大自主学习整合架构**。

三层记忆 × 一个巩固闭环
========================
SMM 把 SimplCore 已有的记忆/自进化组件整合为单一门面，形成人脑式的
"工作记忆 → 情景记忆 → 语义记忆"分层，以及连接情景与语义的
**巩固（consolidation）闭环**：

    ┌──────────────────────────────────────────────────────────────┐
    │  在线（每一步推理）                                            │
    │                                                              │
    │  工作记忆 = O(1) 流式状态（mixer IIR 递推，常驻、与长度无关）  │
    │      │  会话结束 / 经验产生                                   │
    │      ▼                                                       │
    │  情景记忆 = CMT 压缩记忆库（检索 → 聚合 → 注入残差流）         │
    │      │  sleep()（离线巩固，Dreaming 循环）                     │
    │      ▼                                                       │
    │  语义记忆 = 权重（蒸馏写回 + no-regression gate + 回滚）       │
    └──────────────────────────────────────────────────────────────┘

整合的组件（全部已存在，SMM 是它们的**接线板**）
------------------------------------------------
* ``memory.CompressedMemoryBank`` —— 情景记忆库（定容 LRU）；
* ``memory.TextKeyEncoder``       —— 文本 → 定长检索键（signed hashing）；
* ``dream.ExperienceBuffer``      —— 优先级经验回放池；
* ``dream.ReflectionLoop``          —— 巩固闭环（HARVEST→REFLECT→SYNTHESIZE→
  CONSOLIDATE→VERIFY→COMMIT/ROLLBACK）；
* ``model.SimplCoreModel``        —— 基座（工作记忆 + 记忆注入投影）；
* 可选 ``envlearn.LearnedAdapter`` —— 可学适配器（权重预算受
  ``WeightBudget`` 硬约束）。

在线循环（SMM 补齐的接线）
--------------------------
1. ``experience()``：一次交互落两处——经验池（供反思）与情景记忆库（供检索）；
2. ``recall()``：把查询编码成键 → 检索 top-k → top-aggregation 成 (B, D) 记忆向量；
3. ``answer()``：带记忆注入的 O(1) 流式生成（``init_stream(memory=...)``）；
   ``think_max_iters>0`` 时每步 logits 前先做 latent 迭代思考
   （:class:`LatentRefiner`，与记忆注入在同一 O(1) 状态上叠加）；
4. ``sleep()``：攒够 ``consolidate_every`` 条经验后跑 Dreaming 轮——
   洞见写回情景记忆、（mode 允许时）蒸馏进语义记忆，退化自动回滚。

硬性约束（继承自 dream.py，SMM 不放松）
----------------------------------------
1. 写语义记忆（权重）必须过 no-regression gate，退化即回滚（权重+记忆库）；
2. ``mode="weights"`` 而未提供 verifier 时，巩固被拒绝——绝不无验证写权重；
3. 有预算：经验池/记忆库定容，每轮洞见/样本/写入数有硬上限；
4. 可观测：``stats()`` 给出三层记忆与闭环的全部计数。

诚实边界
--------
SMM 是**整合架构与接线层**，不是"已训好的自学习模型"：

* 键/值编码默认用 signed-hashing（词面级），不是语义嵌入；要语义检索请注入
  ``key_encoder``/``value_encoder``（接口一致即可，如模型隐状态）；
* 思考层（``think_max_iters>0``）的 cell/halting 参数**不在** Dreaming 巩固
  的优化器范围内——latent 思考本身如何"越想越好"超出本整合层的职责；
* 巩固质量取决于 reflector/synthesizer/verifier，与 dream.py 的诚实边界一致；
* 未在真实任务上实测"越用越好"之前，不得宣称自学习有效。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .dream import (
    ReflectionConfig,
    ReflectionReport,
    ReflectionLoop,
    ExperienceBuffer,
    Trajectory,
)
from .memory import CompressedMemoryBank, TextKeyEncoder

__all__ = [
    "MemoryTier",
    "SMMConfig",
    "SMMLedger",
    "SimplModelMemory",
]


# ---------------------------------------------------------------------------
# 分层与配置
# ---------------------------------------------------------------------------
class MemoryTier:
    """三层记忆的名字（int 常量，供 stats 键使用）。"""

    WORKING = "working"        # O(1) 流式状态（mixer 递推态）
    EPISODIC = "episodic"      # CMT 压缩记忆库
    SEMANTIC = "semantic"      # 蒸馏进权重的知识（含适配器）


@dataclass
class SMMConfig:
    """Simpl Model Memory 的整合配置。

    Attributes:
        bank_capacity: 情景记忆库容量（定容 LRU 驱逐）。
        top_k: 检索聚合条数。
        retrieval_temperature: top-aggregation softmax 温度。
        min_recall_score: 低于该检索分数放弃注入（检索置信度阈值）。
        dream: Dreaming 循环配置（mode 在这里整体接管：memory/weights/both/none）。
        consolidate_every: 每积累多少条经验跑一轮 sleep()。
        auto_sleep: ``experience()`` 积攒够后是否自动 sleep。
        key_dim: 检索键维度（0 = 跟随模型 dim）。
        key_ngram: 键编码器 n-gram 宽度。
        consolidate_lr: 语义巩固的小步学习率（透传 default_consolidator）。
        think_max_iters: 每步生成前 latent 思考迭代上限（0 = 关闭，
            与直通逐位一致；>0 时 ``answer()`` 走 :class:`LatentRefiner`
            的流式路径——记忆注入 + latent 精化在同一 O(1) 状态上叠加）。
    """

    bank_capacity: int = 4096
    top_k: int = 4
    retrieval_temperature: float = 0.07
    min_recall_score: float = 0.0
    dream: ReflectionConfig = field(default_factory=ReflectionConfig)
    consolidate_every: int = 32
    auto_sleep: bool = False
    key_dim: int = 0
    key_ngram: int = 3
    consolidate_lr: float = 1e-5
    think_max_iters: int = 0


@dataclass
class SMMLedger:
    """SMM 全生命周期计数（``stats()`` 的数据源）。"""

    experiences: int = 0            # 累计经验条数
    recalls: int = 0                # 累计检索次数
    recall_hits: int = 0            # 检索分数达阈值的次数
    injected: int = 0               # 真正注入模型的次数
    sleep_rounds: int = 0           # Dreaming 轮数
    weights_commits: int = 0        # 语义巩固成功提交次数
    rollbacks: int = 0              # 回滚次数
    memory_writes: int = 0          # 情景记忆写入条数

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiences": self.experiences,
            "recalls": self.recalls,
            "recall_hits": self.recall_hits,
            "injected": self.injected,
            "sleep_rounds": self.sleep_rounds,
            "weights_commits": self.weights_commits,
            "rollbacks": self.rollbacks,
            "memory_writes": self.memory_writes,
        }


# ---------------------------------------------------------------------------
# 整合架构
# ---------------------------------------------------------------------------
class SimplModelMemory:
    """SimplCore 的最大自主学习整合架构（三层记忆 + 巩固闭环门面）。

    Args:
        model: ``SimplCoreModel``（需要 ``use_memory=True`` 才能注入检索结果；
            未开启时 ``answer(inject=True)`` 会显式报错，绝不静默丢弃记忆）。
        codec: 编解码器（experience 记键、巩固训练样本编码都需要）。
        config: :class:`SMMConfig`。
        dream_loop: 可选外部 :class:`ReflectionLoop`（注入自定义
            reflector/synthesizer/verifier）；缺省按 ``config.dream`` 自建。
        key_encoder: 可选自定义键编码器（签名同 :class:`TextKeyEncoder`）。
        value_encoder: 可选 ``(text) -> (D,) 张量`` 的值编码器；缺省用键
            编码器的向量（词面级）。要语义级注入请传模型隐状态编码器。
    """

    def __init__(self, model, codec, config: Optional[SMMConfig] = None,
                 dream_loop: Optional[ReflectionLoop] = None,
                 key_encoder: Optional[TextKeyEncoder] = None,
                 value_encoder=None) -> None:
        self.model = model
        self.codec = codec
        self.config = config or SMMConfig()
        cfg = self.config
        dim = int(getattr(model.config, "dim", 0))
        if dim <= 0:
            raise ValueError("model.config.dim 必须为正")
        key_dim = cfg.key_dim if cfg.key_dim > 0 else dim

        # 情景记忆层
        self.bank = CompressedMemoryBank(dim=dim, capacity=cfg.bank_capacity)
        self.key_encoder = key_encoder or TextKeyEncoder(dim=key_dim,
                                                         ngram=cfg.key_ngram)
        self.value_encoder = value_encoder or self.key_encoder

        # 经验层 + 巩固闭环
        self.buffer = ExperienceBuffer()
        self.dream = dream_loop or ReflectionLoop(
            model=model,
            buffer=self.buffer,
            config=cfg.dream,
            memory_bank=self.bank,
            codec=codec,
        )

        # 思考层（可选）：think_max_iters>0 时 answer() 走 latent 思考路径。
        # 懒导入避免模块级循环；LatentRefiner 的零初始化纪律保证
        # 思考层未训练时不改变基座行为（cell 是恒等映射）。
        self.thinker = None
        if cfg.think_max_iters > 0:
            from ..model.nonlinear import LatentRefiner
            self.thinker = LatentRefiner(
                model, max_iters=int(cfg.think_max_iters)).eval()

        self.ledger = SMMLedger()
        self._since_sleep: int = 0

    # ------------------------------------------------------------------
    # 在线：经验落两处
    # ------------------------------------------------------------------
    def experience(self, prompt: str, response: str, confidence: float = 1.0,
                   grounded: bool = True, score: float = 0.0,
                   evidence: str = "", auto_sleep: Optional[bool] = None) -> Trajectory:
        """记录一次交互：经验池（供反思）+ 情景记忆库（供检索）。

        Args:
            prompt / response: 交互文本。
            confidence: 模型自评置信（经验池优先级用）。
            grounded: 是否有外部证据支撑（对应 L4 拒答门）。
            score: 事后评分（>0 覆盖默认优先级）。
            evidence: 外部证据原文（grounded=False 时尤其重要——
                没有证据原文的洞见不会被合成进训练信号）。
            auto_sleep: 覆盖 ``config.auto_sleep``（攒够即巩固）。

        Returns:
            落池的 :class:`Trajectory`。
        """
        traj = Trajectory(prompt=prompt, response=response,
                          confidence=float(confidence), grounded=bool(grounded),
                          score=float(score),
                          meta={"evidence": evidence or ""})
        self.buffer.add(traj)
        self.ledger.experiences += 1
        self._since_sleep += 1

        # 情景记忆：键 = 上下文（prompt），值 = 应答（+证据）文本的编码。
        # 写入策略：低置信或未 grounded 的经验**优先**写入——它们是"下次要查"
        # 的教训；高置信且 grounded 的也写（正例）。
        key = self.key_encoder(prompt)
        val_text = response if not evidence else f"{response}\n{evidence}"
        value = self.value_encoder(val_text)
        if value.shape[-1] != self.bank.dim:
            raise ValueError(
                f"value_encoder 输出维度 {value.shape[-1]} != 记忆库维度 {self.bank.dim}")
        self.bank.add(key.reshape(1, -1), value.reshape(1, -1),
                      meta=[{"prompt": prompt, "grounded": traj.grounded}])
        self.ledger.memory_writes += 1

        if auto_sleep if auto_sleep is not None else self.config.auto_sleep:
            if self._since_sleep >= self.config.consolidate_every:
                self.sleep()
        return traj

    # ------------------------------------------------------------------
    # 在线：检索
    # ------------------------------------------------------------------
    def recall(self, query: str) -> tuple:
        """检索与 ``query`` 最相关的记忆，聚合成一个 (D,) 注入向量。

        Returns:
            ``(memory (D,) or None, score float, meta list)``。
            库空 / 分数低于 ``min_recall_score`` 时 memory 为 ``None``。
        """
        self.ledger.recalls += 1
        q = self.key_encoder(query).reshape(1, -1)
        aggregated, scores = self.bank.aggregate(
            q, top_k=self.config.top_k,
            temperature=self.config.retrieval_temperature)
        top = float(scores.reshape(-1)[0]) if scores.numel() else 0.0
        meta = [m for m in self.bank.metas()]
        if (scores.numel() == 0 or aggregated.numel() == 0
                or top < self.config.min_recall_score):
            return None, top, []
        self.ledger.recall_hits += 1
        return aggregated.reshape(-1), top, meta[: self.config.top_k]

    # ------------------------------------------------------------------
    # 在线：带记忆的生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def answer(self, idx: torch.Tensor, max_new_tokens: int = 32,
               inject: bool = True, query: str = "",
               temperature: float = 1.0, top_k: Optional[int] = None,
               top_p: Optional[float] = None) -> Dict[str, Any]:
        """带情景记忆注入的 O(1) 流式生成。

        Args:
            idx: (1, L) 提示 token id。
            inject: 是否检索并注入记忆。
            query: 检索查询文本（缺省不检索，``inject=True`` 时必填）。
        Returns:
            ``{"tokens": (1, L+new), "memory": tensor|None, "recall_score": float}``
        """
        if not inject:
            gen = self.thinker.generate if self.thinker is not None \
                else self.model.generate
            out = gen(idx, max_new_tokens=max_new_tokens,
                      temperature=temperature, top_k=top_k, top_p=top_p)
            return {"tokens": out, "memory": None, "recall_score": 0.0}

        if not query:
            raise ValueError("inject=True 需要 query（检索查询文本）")
        memory, score, _meta = self.recall(query)
        used = None
        if memory is not None:
            # 注入经 init_stream → model._project_memory（use_memory=False 时
            # 显式报错，绝不静默丢弃 —— 与 model 层的约定一致）
            used = memory
            self.ledger.injected += 1
        state = self.model.init_stream(
            idx.shape[0], device=idx.device,
            dtype=self.model.tok_emb.weight.dtype,
            memory=used.reshape(1, -1) if used is not None else None)
        # 思考层开启时走 latent 思考流式路径（thinker._consume 与
        # model.stream_step 同构，且兼容 state.memory 注入）；
        # 关闭时直接用基座模型（逐位一致的默认路径）。
        streamer = self.thinker if self.thinker is not None else self.model
        logits = None
        for t in range(idx.shape[1]):
            logits, state = streamer.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self.model._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            logits, state = streamer.stream_step(nxt, state)
        return {"tokens": out, "memory": used, "recall_score": score}

    # ------------------------------------------------------------------
    # 离线：巩固（睡眠）
    # ------------------------------------------------------------------
    def sleep(self) -> ReflectionReport:
        """跑一轮 Dreaming 巩固（HARVEST→…→COMMIT/ROLLBACK）。

        权重巩固（mode=weights/both）由 ReflectionLoop 内部按 no-regression
        gate 判定；SMM 在报告层追账（提交/回滚计数）。
        """
        rep = self.dream.step()
        self.ledger.sleep_rounds += 1
        if rep.rolled_back:
            self.ledger.rollbacks += 1
        elif rep.wrote_weights:
            self.ledger.weights_commits += 1
        self._since_sleep = 0
        return rep

    def sleep_all(self, rounds: Optional[int] = None) -> List[ReflectionReport]:
        """连续巩固若干轮（缺省清空经验池或取 ``max_rounds``）。"""
        out: List[ReflectionReport] = []
        cap = int(rounds) if rounds is not None else self.config.dream.max_rounds
        for _ in range(max(cap, 0)):
            if len(self.buffer) == 0:
                break
            rep = self.sleep()
            out.append(rep)
        return out

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """情景记忆 + 键编码器 + 经验池统计的完整快照（不含模型权重）。"""
        return {
            "bank": self.bank.state_dict(),
            "key_encoder": self.key_encoder.state_dict(),
            "ledger": self.ledger.as_dict(),
            "buffer_stats": self.buffer.stats(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从 :meth:`state_dict` 恢复情景记忆与键编码器。"""
        self.bank.load_state_dict(state["bank"])
        self.key_encoder = TextKeyEncoder.from_state(state["key_encoder"])
        self.value_encoder = self.key_encoder

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """三层记忆 + 闭环的全量账目。"""
        d = int(getattr(self.model.config, "dim", 0))
        return {
            MemoryTier.WORKING: {
                "stream_bytes_1": self.model.stream_bytes(1),
                "note": "O(1) 常驻递推态，与上下文长度无关",
            },
            MemoryTier.EPISODIC: {
                **self.bank.stats(),
                "buffer": self.buffer.stats(),
                "key_encoder_fingerprint": self.key_encoder.fingerprint(),
            },
            MemoryTier.SEMANTIC: {
                "dream_rounds": len(self.dream.history),
                "mode": self.config.dream.mode,
                "commits": self.ledger.weights_commits,
                "rollbacks": self.ledger.rollbacks,
            },
            "ledger": self.ledger.as_dict(),
            "think": (self.thinker.ledger.as_dict()
                      if self.thinker is not None else {"enabled": False}),
            "dim": d,
        }

    # -- 守卫 -----------------------------------------------------------------
    def attention_like_modules(self):
        """SMM 全部组件中的注意力模块（应为空；SMM 本身非 nn.Module，只扫模型）。"""
        from ..arch.purity import attention_like_modules as _attn
        hits = list(_attn(self.model))
        if self.thinker is not None:
            hits += list(_attn(self.thinker))
        return hits


def main(argv: Optional[list] = None) -> int:
    """``python -m simplcore.core.smm`` —— SMM 全链路机制演示（玩具模型）。"""
    import argparse

    from ..model.config import SimplCoreConfig
    from ..model.model import SimplCoreModel
    from ..model.tokenization import CharTokenizer

    ap = argparse.ArgumentParser(description="SimplCore SMM 演示（玩具模型）")
    ap.add_argument("--interactions", type=int, default=6)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=64, num_layers=2, num_bases=8, decay_max=128,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc",
        use_memory=True)
    model = SimplCoreModel(cfg).eval()
    smm = SimplModelMemory(model, codec, SMMConfig(consolidate_every=4,
                                                   think_max_iters=2))

    # 1) 经验写入（情景记忆 + 经验池）
    smm.experience("用户的名字是小林。", "好的，小林。", confidence=0.4,
                   grounded=True, evidence="用户说：我叫小林。")
    smm.experience("小林喜欢什么？", "小林喜欢蓝色。", confidence=0.6,
                   grounded=True, evidence="小林说：我最喜欢蓝色。")

    # 2) 检索
    mem, score, _ = smm.recall("用户的名字")
    print(f"recall score = {score:.3f}, memory is {'not None' if mem is not None else 'None'}")

    # 3) 带记忆生成（走 latent 思考流式路径）
    ids = torch.tensor([codec.encode("用户")])
    out = smm.answer(ids, max_new_tokens=8, query="用户的名字")
    print(f"generated: {codec.decode(out['tokens'][0, ids.shape[1]:].tolist())!r}")
    print(f"think ledger: {smm.thinker.ledger.as_dict()}")

    # 4) 巩固（memory 模式：只写情景记忆，零遗忘风险）
    rep = smm.sleep()
    print(f"dream round={rep.round_index} stage={rep.stage.value} "
          f"wrote_memory={rep.wrote_memory} rolled_back={rep.rolled_back}")

    # 5) 账目
    print(f"stats: {smm.stats()['ledger']}")
    print(f"attention modules: {len(smm.attention_like_modules())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
