# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""默认反思/合成/巩固/校验钩子（全部可被注入替换）。"""
from __future__ import annotations


from .types import ReflectionReport, Insight, Trajectory, TrainExample

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ...arch.grounding import EvidenceIndex, split_claims



# ---------------------------------------------------------------------------
# 默认钩子
# ---------------------------------------------------------------------------
#: 轨迹 meta 里可能承载外部证据的键（按优先级）。
_EVIDENCE_META_KEYS = ("evidence", "context", "contexts", "retrieved",
                       "hits", "documents", "sources")


def _collect_evidence(traj: Trajectory) -> Tuple[str, str]:
    """从轨迹的 ``meta`` 里取出外部证据原文，返回 ``(拼接文本, 来源标识)``。

    支持三种常见投放方式（与 ``abstain.GroundedAnswerer`` 的入参一致）：

    * ``meta["context"] = "整段上下文"``；
    * ``meta["hits"] = [{"text": ..., "source": ...}, ...]``（检索命中）；
    * ``meta["evidence"] = ["片段1", "片段2"]``。

    **从 response 里反推证据是不允许的** —— 那会让"自我反思"退化成
    "把模型自己的话当成事实"，正是抗幻觉要防的那件事。
    """
    for key in _EVIDENCE_META_KEYS:
        raw = traj.meta.get(key)
        if raw is None:
            continue
        pieces: List[str] = []
        srcs: List[str] = []
        if isinstance(raw, str):
            if raw.strip():
                pieces.append(raw)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    if item.strip():
                        pieces.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or item.get("span")
                    if txt:
                        pieces.append(str(txt))
                    src = item.get("source") or item.get("doc_id") or item.get("id")
                    if src:
                        srcs.append(str(src))
                else:
                    txt = getattr(item, "text", None)
                    if txt:
                        pieces.append(str(txt))
                    src = getattr(item, "source", None)
                    if src:
                        srcs.append(str(src))
        if pieces:
            return "。".join(pieces), (",".join(srcs) if srcs else str(key))
    return "", ""


def default_reflector(
    model,
    trajectories: Sequence[Trajectory],
    min_repair_coverage: float = 0.30,
    max_claims: int = 16,
) -> List[Insight]:
    """默认反思器：**基于外部证据的逐句归因**，不调用模型自评。

    旧实现是把"低置信度 / 无证据"的轨迹打上标记就结束 —— 标记不是知识，
    所以合成器永远拿不到 ``repair``，``synthesized`` 恒为 0，整条自进化链
    是空转的（``docs/THEORETICAL_EDGE.md`` 里如实记着这一点）。

    本实现换了一条不需要外部批判者模型的路：**用证据绑定做归因**
    （:mod:`simplcore.core.grounding`），逐句判定响应里哪些话在证据里找得到。
    ``repair`` 一律取**证据原文**，绝不取模型自己的改写 —— 这样"自进化"
    产出的每一条训练目标都能追回一句人类可以核对的原文。

    三种洞见
    --------
    * ``no_evidence``        —— 轨迹没有任何外部证据可用。**不产 repair**
      （无依据可修），只作标记，用于统计"该补检索"的比例；
    * ``partially_supported``—— 句子与某条证据高度重叠但**未过判据**
      （典型：数字被篡改、否定被翻转、跨文档拼接）。``repair`` = 那条证据原文，
      这是最有价值的信号：错在细节，而不是全无依据；
    * ``unsupported``        —— 证据里找不到对应片段。**不产 repair**
      （没有原文可给），只作标记；
    * ``corrected``          —— 轨迹自带 ``meta['corrected_response']``
      （来自外部批判-修订链路），原样作为目标。

    Args:
        min_repair_coverage: 低于此覆盖率视为"证据里根本没这句话"，
            不给 repair（避免把不相干的证据硬塞成训练目标）。
        max_claims: 每条轨迹最多核验的句子数（预算硬约束）。
    """
    out: List[Insight] = []
    seen: set = set()
    for traj in trajectories:
        # ① 外部修订优先：它已是完整响应，不像 claim 那样需要回填
        corrected = traj.meta.get("corrected_response")
        if corrected:
            out.append(Insight(
                text=f"外部修订：{traj.prompt[:60]}",
                source=traj,
                weight=1.0 + max(0.0, 1.0 - float(traj.confidence)),
                kind="corrected",
                repair=str(corrected),
                evidence=str(corrected),
                meta={"evidence_source": str(traj.meta.get("corrected_source", "external"))},
            ))
            continue

        ctx, src = _collect_evidence(traj)
        if not ctx:
            if (not traj.grounded) or traj.confidence < 0.5:
                out.append(Insight(
                    text=f"未获证据支撑或置信度不足：{traj.prompt[:60]}",
                    source=traj,
                    weight=(1.0 - float(traj.confidence)) + (0.0 if traj.grounded else 1.0),
                    kind="no_evidence",
                    meta={"evidence_source": ""},
                ))
            continue

        index = EvidenceIndex.from_text(ctx, source=src or "trajectory.evidence")
        if not index.spans:
            continue
        bindings = index.bind_all(split_claims(traj.response, max_claims=max_claims))
        for b in bindings:
            if b.supported:
                continue
            if b.span is not None and b.coverage >= float(min_repair_coverage):
                kind = "partially_supported"
                repair: Optional[str] = b.span.text
            else:
                kind = "unsupported"
                repair = None
            dedup_key = (kind, repair or b.claim.text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # 严重度：越不像证据越该反思；数字篡改与否定翻转额外加权
            sev = max(1.0 - float(b.coverage), 0.0)
            if not b.numbers_ok:
                sev += 0.5
            if not b.polarity_ok:
                sev += 0.25
            out.append(Insight(
                text=f"[{kind}] 句子未获证据支撑：{b.claim.text[:60]}",
                source=traj,
                weight=max(sev, 0.05),
                kind=kind,
                repair=repair,
                evidence=(b.span.text if b.span is not None else None),
                meta={
                    "claim": b.claim.text,
                    "coverage": float(b.coverage),
                    "ngram": float(b.ngram),
                    "numbers_ok": bool(b.numbers_ok),
                    "polarity_ok": bool(b.polarity_ok),
                    "span_id": (b.span.span_id if b.span is not None else None),
                    "evidence_source": (b.span.source if b.span is not None else src),
                },
            ))
    return out


def _replace_claim(response: str, claim: str, repair: str) -> str:
    """把响应里那句不可归因的话替换成证据原文，得到"修订后的响应"。

    直接拿一条证据片段当训练目标会丢掉整段回答的语境（目标变成一句孤立事实），
    自蒸馏时学到的就不是"怎么答"而是"怎么抄"。因此这里保留原响应结构，
    只替换出错的那一句 —— 这是**最小编辑**的修订目标。
    """
    claim = (claim or "").strip()
    if claim and response and claim in response:
        return response.replace(claim, repair, 1)
    if not (response or "").strip():
        return repair
    # 找不到原句（响应被归一化/截断过）：不硬改写，退回证据原文并在 meta 标注
    return repair


def default_synthesizer(model, insights: Sequence[Insight]) -> List[TrainExample]:
    """默认合成器：**只接受带 ``repair``（证据原文）的洞见**。

    这是整条自进化链的诚实性闸门：
    ``repair is None`` 的洞见（``no_evidence`` / ``unsupported``）**一律丢弃** ——
    没有原文可给的时候，"修复它"只能靠模型自己编，那正是要防的东西。

    因此 ``synthesized == 0`` 是一个**有意义的结果**：它意味着本轮所有发现
    都没有可引用的原文，此时正确行为是补检索，而不是拿模型自评去训练自己。
    """
    out: List[TrainExample] = []
    seen: set = set()
    for ins in insights:
        traj = ins.source
        if traj is None or not ins.repair:
            continue
        repair = str(ins.repair).strip()
        if not repair:
            continue
        if ins.kind == "corrected":
            target = repair
        else:
            target = _replace_claim(traj.response, str(ins.meta.get("claim") or ""), repair)
        target = (target or "").strip()
        if not target:
            continue
        key = (traj.prompt, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(TrainExample(
            prompt=traj.prompt,
            target=target,
            weight=ins.weight,
            meta={
                "from_insight": ins.text,
                "kind": ins.kind,
                "claim": ins.meta.get("claim"),
                "evidence": ins.evidence,
                "evidence_source": ins.meta.get("evidence_source"),
                # 供下游审计：目标里的每个事实都应能在 evidence 里查到
                "from_evidence": ins.evidence is not None,
            },
        ))
    return out


def _encode_ids(codec, text: str) -> List[int]:
    """把文本编码成 token id 列表（兼容返回原生 list / HF Encoding 的编解码器）。"""
    if codec is None or not text:
        return []
    out = codec.encode(text)
    if out is None:
        return []
    if isinstance(out, int):
        return [out]
    try:
        return [int(i) for i in out]
    except TypeError:
        return []


def default_consolidator(
    model,
    examples: Sequence[TrainExample],
    *,
    codec=None,
    seq_len: int = 512,
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    max_steps: int = 50,
    device=None,
    seed: int = 0,
) -> Dict[str, Any]:
    """默认巩固器：**把训练信号真正写进模型权重**。

    这是支柱 C（"去真实环境写代码 → 验证 → 写回权重"）里唯一缺的那一环。
    :class:`ReflectionLoop` 负责决定**该不该写**（no-regression gate）与
    **怎么回滚**（``snapshot`` / ``restore``），但"怎么把样本变成梯度步"
    必须有人做 —— 此前只有调用方自备 ``consolidator``，没有内置实现，
    于是"写回权重"这条链路在默认配置下是断的。

    关键约定
    --------
    * **标签错位一格**。``SimplCoreModel`` 的 loss 是 ``CE(logits[t], targets[t])``
      （模型内部**不做** shift），因此标签必须自己错位：
      ``labels[t] = ids[t+1]``，且只在**目标段**计损失（提示段一律 ``-100``）。
    * **不对一维参数做 weight decay**。振荡基参数（``raw_rho`` / ``raw_theta``）、
      norm、bias、``feed``、``out_scale`` 都是一维或标量；给它们加衰减会直接
      掐死长程记忆（与 ``train.py`` 的同一条纪律）。
    * **样本权重生效**。``TrainExample.weight`` 按逐样本加权，而不是被平均掉。

    Args:
        model: 目标模型（``SimplCoreModel`` 或同接口的 ``nn.Module``）。
        examples: 训练样本；``meta["input_ids"]`` 里若已带编码结果则优先使用。
        codec: 编解码器，需有 ``encode(text) -> list[int]``；
            不传时样本必须自带 ``meta["input_ids"]``。
        seq_len: 截断长度。
        lr: 学习率（**巩固是小步微调**，默认远小于常规 SFT）。
        epochs / batch_size / max_steps: 训练规模上限。
        weight_decay: 权重衰减（只作用于二维及以上参数）。
        grad_clip: 梯度裁剪阈值。
        device: 训练设备；``None`` 表示跟随模型。
        seed: 随机种子（决定 batch 顺序）。

    Returns:
        统计字典：``steps`` / ``loss_before`` / ``loss_after`` / ``examples`` /
        ``tokens`` / ``lr`` / ``grad_norm``。
    """
    import torch.nn.functional as F

    if model is None:
        raise ValueError("default_consolidator 需要一个模型")
    ex_list = list(examples or [])
    if not ex_list:
        return {"steps": 0, "examples": 0, "tokens": 0,
                "loss_before": None, "loss_after": None,
                "lr": lr, "grad_norm": None}

    # ---- 1. 构造 (ids, labels, weight) -----------------------------------
    samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for ex in ex_list:
        ids = ex.meta.get("input_ids") if isinstance(ex.meta, dict) else None
        if ids is None:
            if codec is None:
                raise ValueError(
                    "default_consolidator 需要 codec 来编码样本，或让样本在 "
                    "meta['input_ids'] 里带上已编码的 id"
                )
            p_ids = _encode_ids(codec, ex.prompt)
            t_ids = _encode_ids(codec, ex.target)
            ids = p_ids + t_ids
            prompt_len = len(p_ids)
        else:
            ids = [int(i) for i in ids]
            prompt_len = int((ex.meta or {}).get("prompt_len", 0))
        ids = ids[:seq_len]
        if len(ids) < 2:
            continue
        # labels[t] = ids[t+1]；只从"提示段最后一位"开始计目标 token 的损失
        labels = [-100] * len(ids)
        for t in range(max(prompt_len - 1, 0), len(ids) - 1):
            labels[t] = ids[t + 1]
        if all(v == -100 for v in labels):
            continue                       # 没有目标 token，这条样本学不到东西
        samples.append((torch.tensor(ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long),
                        float(ex.weight)))

    if not samples:
        return {"steps": 0, "examples": 0, "tokens": 0,
                "loss_before": None, "loss_after": None,
                "lr": lr, "grad_norm": None}

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev)

    # ---- 2. 优化器：只对二维及以上参数做 weight decay ----------------------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(weight_decay)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(lr),
    )

    gen = torch.Generator().manual_seed(int(seed))
    order = list(range(len(samples)))
    was_training = bool(model.training)
    model.train()

    def _loss_on(batch_idx: List[int]) -> torch.Tensor:
        width = max(len(samples[i][0]) for i in batch_idx)
        ids = torch.zeros(len(batch_idx), width, dtype=torch.long)
        labels = torch.full((len(batch_idx), width), -100, dtype=torch.long)
        w = torch.zeros(len(batch_idx), dtype=torch.float32)
        for r, i in enumerate(batch_idx):
            x, y, wi = samples[i]
            ids[r, :x.numel()] = x
            labels[r, :y.numel()] = y
            w[r] = wi
        ids, labels, w = ids.to(dev), labels.to(dev), w.to(dev)
        logits = model(ids)
        V = logits.size(-1)
        ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                             ignore_index=-100, reduction="none").view(labels.shape)
        mask = (labels != -100).to(ce.dtype)
        per_sample = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-6), int(mask.sum().item())

    # ---- 3. 训练循环 ------------------------------------------------------
    steps = 0
    grad_norm = None
    loss_before = None
    try:
        with torch.no_grad():
            loss_before, _ = _loss_on(order[:max(1, min(len(order), batch_size))])
        for _ep in range(max(1, int(epochs))):
            perm = torch.randperm(len(order), generator=gen).tolist()
            for s in range(0, len(perm), max(1, int(batch_size))):
                if steps >= max_steps:
                    break
                batch = perm[s:s + max(1, int(batch_size))]
                opt.zero_grad(set_to_none=True)
                loss, _ = _loss_on(batch)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip) if grad_clip else float("inf"))
                grad_norm = float(gn)
                opt.step()
                steps += 1
            if steps >= max_steps:
                break
    finally:
        if not was_training:
            model.eval()

    with torch.no_grad():
        loss_after, tokens = _loss_on(order[:max(1, min(len(order), batch_size))])

    return {
        "steps": steps,
        "examples": len(samples),
        "tokens": tokens,
        "loss_before": None if loss_before is None else float(loss_before),
        "loss_after": None if loss_after is None else float(loss_after),
        "lr": float(lr),
        "grad_norm": grad_norm,
    }


def default_verifier(model) -> float:
    """默认验证器：**恒定返回 0.0**，因此**不能**用于判定退化。

    它存在只是为了让 ``ReflectionLoop`` 的签名完整。``ReflectionLoop`` 会记住
    "调用方没给 verifier"，并在 ``mode in ("weights", "both")`` 时**拒绝写回
    权重**（见 :meth:`ReflectionLoop.step` 的 VERIFY 段）—— 没有独立 held-out    评测集时让 gate 恒通过，等于把 no-regression gate 变成装饰品。

    生产实现应返回逐样本分数::

        lambda model: {"grounding_f1": [...每样本 f1...],
                       "abstain_auroc": [...每样本 auroc...]}
    """
    return 0.0

