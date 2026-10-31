# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""ReflectionLoop 主循环 + ReflectionConfig（四模式巩固、可恢复、硬预算）。"""

from .buffer import ExperienceBuffer
from .evaluation import EvalScores, GateVerdict, regression_gate
from .hooks import (_collect_evidence, default_consolidator, default_reflector,
                    default_synthesizer, default_verifier)
from .types import ReflectionReport, ReflectionStage, Insight, Trajectory, TrainExample

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ...arch.grounding import EvidenceIndex, split_claims
from ..memory import TextKeyEncoder



# ---------------------------------------------------------------------------
# 循环配置
# ---------------------------------------------------------------------------
@dataclass
class ReflectionConfig:
    """Dreaming 循环配置。"""
    mode: str = "memory"                  # memory | weights | both | none
    batch_size: int = 32                  # 每轮采样轨迹数
    max_rounds: int = 100                 # 轮数上限
    max_insights: int = 64                # 每轮洞见上限
    max_examples: int = 256               # 每轮合成样本上限
    sampling: str = "priority"            # priority | recent | uniform
    regression_tolerance: float = 0.0     # 允许的退化幅度（负向为容差）
    require_regression_check: bool = True # 是否必须跑 no-regression gate
    # -- gate 的统计口径（多指标 + 配对 bootstrap） -------------------------
    ci_level: float = 0.95                # 置信水平
    bootstrap_samples: int = 1000         # bootstrap 重采样次数
    seed: int = 0
    # -- 反思/合成的预算 ----------------------------------------------------
    max_claims_per_trajectory: int = 16   # 每条轨迹最多核验的句子数
    min_repair_coverage: float = 0.30     # 低于此覆盖率不给 repair（不给证据原文）
    max_memory_writes: int = 256          # 每轮最多写入记忆库的条目数
    # -- 权重巩固（default_consolidator）的超参 -------------------------------
    consolidate_lr: float = 1e-5          # 巩固是小步微调，远小于常规 SFT
    consolidate_epochs: int = 1
    consolidate_batch_size: int = 4
    # -- 泛化判定（held-out，「越用越好」的唯一合法判据） ----------------------
    #: verifier 只证明「不退步」（no-regression gate，写回权重的必要条件）。
    #: 「越用越好」的判据必须是**与经验池/巩固信号不重叠**的 held-out 评测：
    #: 提供时（签名同 verifier：``(model) -> EvalScores|float|dict``），VERIFY
    #: 通过后在旧/新权重上各跑一次，记账 ``ReflectionReport.heldout_delta``；
    #: 不提供时报告明确标注"泛化主张无判据"，不得宣称「越用越好」。
    heldout_eval: Optional[Callable] = None


# ---------------------------------------------------------------------------
# 循环本体
# ---------------------------------------------------------------------------
class ReflectionLoop:
    """架构层的自进化循环。

    Args:
        model: 被巩固的模型（实现 ``snapshot``/``restore`` 者可获得权重回滚能力）。
        buffer: 经验回放池。
        config: 循环配置。
        reflector: ``(model, trajectories) -> List[Insight]``。
        synthesizer: ``(model, insights) -> List[TrainExample]``。
        consolidator: ``(model, examples) -> Dict``，可返回 ``{"snapshot": ...}``。
        verifier: ``(model) -> EvalScores | Dict[str, float|Sequence[float]] | float``，
            越高越好。**传入逐样本分数**才能得到置信下界判定。
            注意它只证明「不退步」；「越用越好」请另配 ``config.heldout_eval``。
        memory_bank: CMT 压缩记忆库；``mode="memory"`` 时洞见写入这里。
    """

    def __init__(
        self,
        model,
        buffer: Optional[ExperienceBuffer] = None,
        config: Optional[ReflectionConfig] = None,
        reflector: Optional[Callable] = None,
        synthesizer: Optional[Callable] = None,
        consolidator: Optional[Callable] = None,
        verifier: Optional[Callable] = None,
        memory_bank=None,
        codec=None,
    ) -> None:
        self.model = model
        self.buffer = buffer if buffer is not None else ExperienceBuffer()
        self.config = config or ReflectionConfig()
        # 默认钩子需要读 config（预算/阈值），自定义钩子保持原签名不变
        self.reflector = reflector or (
            lambda m, trajs: default_reflector(
                m, trajs,
                min_repair_coverage=self.config.min_repair_coverage,
                max_claims=self.config.max_claims_per_trajectory,
            )
        )
        self.synthesizer = synthesizer or default_synthesizer
        self.consolidator = consolidator
        #: 编解码器（``default_consolidator`` 需要它把文本编码成 token id）
        self.codec = codec
        #: 调用方没给 verifier 时，权重写回必须被拒绝（见 step 的 VERIFY 段）
        self.verifier_is_default = verifier is None
        self.verifier = verifier or default_verifier
        self.memory_bank = memory_bank
        self.history: List[ReflectionReport] = []
        self._round = 0
        self._encoder = None                       # 延迟构造的 TextKeyEncoder
        self._harvested: List[Trajectory] = []     # 本轮采集到的轨迹（记忆写入用）

    # -- 基线 ---------------------------------------------------------------
    def evaluate(self, rep: Optional[ReflectionReport] = None) -> Optional[EvalScores]:
        """跑一次 held-out 评测，返回逐样本分数容器。

        评测器异常/返回不可用结构时返回 ``None`` 并记录原因 —— 绝不用
        "常量 0.0" 冒充有效评测（那会让 gate 恒通过）。
        """
        try:
            return EvalScores.from_any(self.verifier(self.model))
        except Exception as exc:                      # noqa: BLE001
            self._note(f"评测失败：{exc}", rep)
            return None

    def baseline(self) -> Optional[float]:
        """当前模型的基线分数（单值视图，保留给旧调用方）。

        多指标情形下取 ``score`` 或字典序第一个指标的均值；做退化判定请用
        :meth:`evaluate` + :func:`regression_gate`，那才有多指标与置信下界。
        """
        if not self.config.require_regression_check:
            return None
        scores = self.evaluate()
        return None if scores is None else scores.primary()

    # -- 快照 / 回滚 --------------------------------------------------------
    @staticmethod
    def _examples_have_ids(examples: Sequence[TrainExample]) -> bool:
        """样本是否自带已编码的 ``meta["input_ids"]``（自带就不需要 codec）。"""
        for ex in examples:
            meta = getattr(ex, "meta", None)
            if isinstance(meta, dict) and meta.get("input_ids") is not None:
                return True
        return False

    def snapshot(self) -> Optional[Any]:
        """模型权重快照（若模型支持）。"""
        if self.model is None or not hasattr(self.model, "snapshot"):
            return None
        try:
            return self.model.snapshot()
        except Exception:                             # noqa: BLE001
            return None

    def restore(self, snap: Optional[Any]) -> bool:
        """回滚模型权重。"""
        if snap is None or self.model is None:
            return False
        if not hasattr(self.model, "restore"):
            return False
        try:
            self.model.restore(snap)
            return True
        except Exception:                             # noqa: BLE001
            return False

    def memory_snapshot(self) -> Optional[Any]:
        """记忆库快照。

        回滚必须**同时**回滚权重与记忆：只回滚权重会把"这一轮学错的东西"
        留在记忆库里，下一轮继续被检索注入 —— 于是"回滚"看起来生效了，
        错误却还在。
        """
        bank = self.memory_bank
        if bank is None:
            return None
        try:
            return bank.state_dict()
        except Exception:                             # noqa: BLE001
            return None

    def memory_restore(self, snap: Optional[Any]) -> bool:
        """回滚记忆库。"""
        bank = self.memory_bank
        if bank is None or snap is None:
            return False
        try:
            bank.load_state_dict(snap)
            return True
        except Exception:                             # noqa: BLE001
            return False

    # -- 主循环 -------------------------------------------------------------
    def step(self) -> ReflectionReport:
        """跑一轮 Dreaming。"""
        t0 = time.perf_counter()
        cfg = self.config
        self._round += 1
        rep = ReflectionReport(round_index=self._round, stage=ReflectionStage.IDLE)

        # 1) HARVEST
        rep.stage = ReflectionStage.HARVEST
        trajs = self.buffer.sample(cfg.batch_size, strategy=cfg.sampling)
        self._harvested = list(trajs)
        rep.harvested = len(trajs)
        if not trajs:
            rep.stage = ReflectionStage.IDLE
            rep.seconds = time.perf_counter() - t0
            rep.notes.append("经验池为空，本轮无输入")
            self.history.append(rep)
            return rep

        # 2) REFLECT
        rep.stage = ReflectionStage.REFLECT
        try:
            insights = list(self.reflector(self.model, trajs) or [])
        except Exception as exc:                      # noqa: BLE001
            rep.stage = ReflectionStage.FAILED
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"反思阶段异常：{exc}")
            self.history.append(rep)
            return rep
        insights = insights[:cfg.max_insights]
        rep.insights = len(insights)

        # 3) SYNTHESIZE
        rep.stage = ReflectionStage.SYNTHESIZE
        try:
            examples = list(self.synthesizer(self.model, insights) or [])
        except Exception as exc:                      # noqa: BLE001
            rep.stage = ReflectionStage.FAILED
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"合成阶段异常：{exc}")
            self.history.append(rep)
            return rep
        examples = examples[:cfg.max_examples]
        rep.synthesized = len(examples)

        # 记忆巩固与权重巩固是两件事：前者不需要训练样本
        # （"把这次查到的证据存进库"和"这次答错了要不要改权重"无关）。
        # 旧实现把两者绑在同一个 if 上，于是"全部答对"的一轮什么都不写，
        # 记忆库永远是空的 —— 这正是"记忆写了但从不被召回"的另一半原因。
        memory_work = cfg.mode in ("memory", "both")
        do_memory = memory_work and self.memory_bank is not None
        weight_work = (cfg.mode in ("weights", "both") and bool(examples))

        if cfg.mode == "none" or not (do_memory or weight_work or examples):
            if not examples:
                if insights:
                    rep.notes.append(
                        f"{len(insights)} 条洞见均无外部证据原文（repair=None），"
                        f"不合成训练信号 —— 此时正确做法是补检索，而不是用自评训练自己"
                    )
                else:
                    rep.notes.append("无洞见（响应全部可归因或轨迹无证据）")
            rep.stage = ReflectionStage.IDLE
            rep.seconds = time.perf_counter() - t0
            self.history.append(rep)
            return rep

        # 4) CONSOLIDATE
        rep.stage = ReflectionStage.CONSOLIDATE
        # gate 的 base 必须**在巩固前**评测（旧权重）：若在巩固后对同一权重
        # 连测两次，配对 delta 恒为噪声——那不是 gate，是装饰。
        gate_base: Optional[EvalScores] = None
        if weight_work and cfg.require_regression_check and not self.verifier_is_default:
            gate_base = self.evaluate(rep)
            if gate_base is None:
                # 基线评测失败 → 无法证明未退化 → 直接拒绝写权（保守）
                rep.rolled_back = True
                rep.stage = ReflectionStage.ROLLBACK
                rep.notes.append("巩固前基线评测失败，无法证明未退化，已拒绝写回权重")
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep
        # held-out 泛化基线（「越用越好」判据的旧权重侧）
        heldout_base: Optional[EvalScores] = None
        if cfg.heldout_eval is not None and (weight_work or do_memory):
            try:
                heldout_base = EvalScores.from_any(cfg.heldout_eval(self.model))
            except Exception as exc:                      # noqa: BLE001
                rep.notes.append(f"held-out 基线评测失败：{exc}")
        snap = self.snapshot()
        mem_snap = self.memory_snapshot()
        wrote_weights = False
        try:
            if weight_work:
                # 没给 consolidator 时用**内置**的 default_consolidator ——
                # 此前这里直接跳过，于是报告写着 "consolidated=True"，
                # 而权重一个字节都没变：一份自称成功、实际没做事的报告。
                fn = self.consolidator
                if fn is None:
                    if self.codec is None and not self._examples_have_ids(examples):
                        raise ValueError(
                            "mode=weights 需要 consolidator，或提供 codec "
                            "（或让样本自带 meta['input_ids']）—— 没有它就无法"
                            "把训练信号写进权重"
                        )
                    fn = lambda m, ex: default_consolidator(       # noqa: E731
                        m, ex, codec=self.codec,
                        lr=self.config.consolidate_lr,
                        epochs=self.config.consolidate_epochs,
                        batch_size=self.config.consolidate_batch_size,
                    )
                stats = fn(self.model, examples) or {}
                rep.notes.append(
                    f"权重巩固：{stats.get('steps', '?')} 步，"
                    f"样本 {stats.get('examples', len(examples))}，"
                    f"loss {stats.get('loss_before')} → {stats.get('loss_after')}"
                )
                wrote_weights = True
            if do_memory:
                wrote = self._write_memory(insights, examples)
                rep.wrote_memory = int(wrote)
                rep.notes.append(f"记忆库写入 {wrote} 条（事实 + 修正）")
            elif memory_work:
                rep.notes.append("mode=memory 但未提供 memory_bank，记忆巩固被跳过")
            # ``consolidated`` 说的是**阶段**完成；"到底写了没有"看
            # ``wrote_weights`` / ``wrote_memory``。
            rep.consolidated = True
            rep.wrote_weights = wrote_weights
        except Exception as exc:                      # noqa: BLE001
            rep.stage = ReflectionStage.FAILED
            self.restore(snap)
            self.memory_restore(mem_snap)
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"巩固阶段异常，已回滚：{exc}")
            self.history.append(rep)
            return rep

        # 5) VERIFY（no-regression gate：多指标 + 配对 bootstrap 置信下界）
        if cfg.require_regression_check and weight_work:
            rep.stage = ReflectionStage.VERIFY
            if self.verifier_is_default:
                # 没有独立评测器时让 gate 恒通过 = 没有 gate。直接拒绝写回权重。
                rep.rolled_back = True
                rep.stage = ReflectionStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.notes.append(
                    "未提供独立 held-out 评测器（verifier 为默认常量），"
                    "no-regression gate 无法判定退化，已拒绝写回权重"
                )
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep

            new = self.evaluate(rep)
            base = gate_base          # 巩固前评测的旧权重基线（见 CONSOLIDATE 段）
            rep.baseline_score = None if base is None else base.primary()
            rep.new_score = None if new is None else new.primary()

            if base is None or new is None:
                # 评测不可用 → 不能声称"没退化"，按退化处理（保守）
                rep.rolled_back = True
                rep.stage = ReflectionStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.notes.append("评测不可用，无法证明未退化，已回滚")
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep

            verdict = regression_gate(
                base, new,
                tolerance=cfg.regression_tolerance,
                ci_level=cfg.ci_level,
                n_boot=cfg.bootstrap_samples,
                seed=cfg.seed,
            )
            rep.metrics_base = dict(verdict.metrics_base)
            rep.metrics_new = dict(verdict.metrics_new)
            rep.ci_lower = dict(verdict.ci_lower)
            rep.failed_metrics = list(verdict.failed_metrics)
            delta = verdict.score_delta
            rep.score_delta = delta
            rep.notes.extend(verdict.notes)

            if not verdict.passed:
                rep.stage = ReflectionStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.rolled_back = True
                rep.notes.append(
                    f"退化指标 {', '.join(verdict.failed_metrics) or '(判定失败)'} "
                    f"超过容差 {abs(cfg.regression_tolerance):.4f}，已回滚权重与记忆"
                )
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep

        # 5.5) 泛化记账（held-out）——「不退步」≠「越用越好」
        if (weight_work or do_memory) and not rep.rolled_back:
            if cfg.heldout_eval is not None:
                try:
                    heldout_new = EvalScores.from_any(cfg.heldout_eval(self.model))
                    if heldout_base is not None and heldout_new is not None:
                        rep.heldout_base = heldout_base.primary()
                        rep.heldout_new = heldout_new.primary()
                        rep.heldout_delta = rep.heldout_new - rep.heldout_base
                        rep.notes.append(
                            f"held-out 泛化 delta={rep.heldout_delta:+.6f} —— "
                            + ("正向才支持「越用越好」口径"
                               if rep.heldout_delta > 0 else
                               "未提升，「越用越好」在本轮不成立"))
                    else:
                        rep.notes.append("held-out 评测不可用，泛化主张本轮无判据")
                except Exception as exc:                  # noqa: BLE001
                    rep.notes.append(f"held-out 终评失败：{exc}")
            else:
                rep.notes.append(
                    "未提供 config.heldout_eval：「不退步」已由 gate 保证，"
                    "「越用越好」无判据，不得宣称")

        # 6) COMMIT
        rep.stage = ReflectionStage.COMMIT
        rep.seconds = time.perf_counter() - t0
        self.history.append(rep)
        return rep

    def run(self, max_rounds: Optional[int] = None) -> List[ReflectionReport]:
        """连续跑若干轮，直到触达轮数上限。"""
        limit = max_rounds if max_rounds is not None else self.config.max_rounds
        out: List[ReflectionReport] = []
        for _ in range(int(limit)):
            rep = self.step()
            out.append(rep)
            if rep.stage is ReflectionStage.FAILED:
                break
        return out

    # -- 记忆写入 -----------------------------------------------------------
    def _encoder_for(self, dim: int) -> "TextKeyEncoder":
        if self._encoder is None or getattr(self._encoder, "dim", None) != int(dim):
            self._encoder = TextKeyEncoder(int(dim))
        return self._encoder

    def _write_memory(self, insights: Sequence[Insight],
                      examples: Sequence[TrainExample]) -> int:
        """把本轮学到的东西写进 CMT 压缩记忆库（mode=memory 的巩固方式）。

        写两类条目，键空间一致（都是"将来会出现的查询"），值一律是**证据原文**：

        * ``knowledge`` —— 本轮被**证据支撑**的分句。这是记忆库存在的主要理由：
          只存"改过错的地方"会让库变成错题本，而不是知识库；线上新查询检索不到
          任何东西，``aggregate`` 返回零向量，注入等于没注入。
        * ``repair``    —— 被判定不可归因、但**找得到原文**的分句（``repair`` 非空）。
          它记的是"这种问法以前答错过，正确说法是这句原文"。

        键与值的分工
        ------------
        * **键**编码 ``source.prompt`` —— "将来什么样的查询应当召回这条记忆"，
          与线上 ``aggregate(query)`` 处同一空间，检索才对得上；
        * **值**编码证据原文。

        旧实现用 ``float(abs(hash(text)) % 1e8) / 1e8`` 填满整条向量：所有条目的键
        彼此共线（余弦恒为 1.0），检索退化成"永远返回库里任意一条"；且内置
        ``hash()`` 带进程随机盐，重启后同一文本的键就变了。这类缺陷不会让任何
        形状测试失败，只表现为"记忆写了但从不被召回"。
        """
        bank = self.memory_bank
        if bank is None:
            return 0
        dim = getattr(bank, "dim", None)
        if not dim:
            return 0
        enc = self._encoder_for(int(dim))
        cfg = self.config

        # (key_text, value_text, meta) 去重表；同一 (键, 值) 不重复写入
        entries: List[Tuple[str, str, Dict[str, Any]]] = []
        seen: set = set()

        def push(key_text: str, value_text: str, meta: Dict[str, Any]) -> None:
            key_text = (key_text or "").strip()
            value_text = (value_text or "").strip()
            if not key_text or not value_text:
                return
            sig = (key_text, value_text)
            if sig in seen:
                return
            seen.add(sig)
            entries.append((key_text, value_text, meta))

        # ① 事实记录：本轮轨迹里被证据支撑的分句
        for traj in self._harvested:
            if not (traj.grounded or traj.confidence >= 0.5):
                continue
            ctx, src = _collect_evidence(traj)
            if not ctx:
                continue
            index = EvidenceIndex.from_text(ctx, source=src or "trajectory.evidence")
            if not index.spans:
                continue
            spans: List[str] = []
            for b in index.bind_all(
                split_claims(traj.response, max_claims=cfg.max_claims_per_trajectory)
            ):
                if b.supported and b.span is not None and b.span.text not in spans:
                    spans.append(b.span.text)
            if spans:
                push(traj.prompt, "。".join(spans),
                     {"kind": "knowledge", "prompt": traj.prompt,
                      "evidence": "。".join(spans),
                      "evidence_source": src, "weight": 1.0,
                      "n_spans": len(spans), "from_evidence": True})

        # ② 修正记录：有证据原文的洞见
        for ins in insights:
            traj = ins.source
            key_text = ((traj.prompt if traj is not None else "") or ins.text or "")
            content = (ins.repair or ins.evidence or "")
            push(key_text, content, {
                "text": ins.text,
                "weight": float(ins.weight),
                "kind": ins.kind,
                "prompt": key_text.strip(),
                "evidence": content.strip(),
                "evidence_source": ins.meta.get("evidence_source", ""),
                "from_evidence": True,
            })

        if not entries:
            return 0
        entries = entries[: max(int(cfg.max_memory_writes), 1)]

        keys, values, metas = [], [], []
        for key_text, value_text, meta in entries:
            key = enc(key_text)
            if float(key.norm()) <= 0.0:
                # 归一化后为零向量（纯标点/空白提示）：写进去只会污染检索
                continue
            keys.append(key)
            values.append(enc(value_text))
            metas.append(meta)
        if keys:
            bank.add(torch.stack(keys), torch.stack(values), metas)
        return len(keys)

    def _note(self, msg: str, rep: Optional[ReflectionReport] = None) -> None:
        """记一条备注。``rep`` 未给时落到最近一条报告（step 早期阶段用）。"""
        target = rep if rep is not None else (self.history[-1] if self.history else None)
        if target is not None:
            target.notes.append(msg)

# 0.1-α 命名降调前的旧公共名（pickle / 外部脚本兼容）
DreamingLoop = ReflectionLoop
DreamConfig = ReflectionConfig
