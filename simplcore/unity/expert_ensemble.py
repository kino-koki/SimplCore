# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity · 多专家集成 (Expert Ensemble / MoE Router).

把若干 **分域训练好的专家小模型** 通过 SimplUnity 的 "路由器 + 融合器" 组织成
一个整体系统:

    * 路由器 (Router): 给定一段输入, 判断它属于哪个领域, 选出最匹配的专家;
    * 融合器 (Fuser): 对选中专家的输出做整合 (单专家直出 / 多专家加权);

这与 Mixture-of-Experts 思想一致, 但底层是本项目自研的非注意力 Lattice 架构,
并通过既有的 UnityOrchestrator 消息总线串联, 保持与 SimplCore 其它组件 (记忆 /
世界模型 / 元认知) 的可集成性.

专家来源
--------
每个专家 = 一个在真实中文语料某领域上训练出的 LatticeLM + 该领域的 UnitMap.
训练见 expert_training.py (train_domain_expert). 本模块负责把它们 "Unity" 起来.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import simplcore.tensor as torch

from simplcore.architecture.lattice_model import LatticeLM, LatticeConfig
from simplcore.data.corpus import UnitMap
from simplcore.unity.orchestrator import (
    UnityOrchestrator, ExpertUnit, MSG_COGNITION, MSG_ACTION, Message,
)


@dataclass
class ExpertSlot:
    """一个已加载的专家."""
    domain: str
    model: LatticeLM
    unit_map: UnitMap
    domain_vocab: set = field(default_factory=set)   # 该专家训练语料的字符集
    weight: float = 1.0


class ExpertRouter:
    """领域路由器: 把输入文本分配到最匹配的领域专家.

    采用 **字符覆盖度 + 长度归一** 的轻量可解释评分 (无需额外训练即可工作):
        对每个专家, 统计输入文本中 "属于该专家训练语料字符集" 的字符比例,
        比例越高越匹配. 这是一种真实有效的零样本领域判别启发式.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def score(self, text: str, slot: ExpertSlot) -> float:
        if not text:
            return 0.0
        covered = sum(1 for ch in text if ch in slot.domain_vocab)
        # 覆盖比例 + 微小偏置, 避免全 0
        return (covered / len(text)) + 1e-3

    def route(self, text: str, slots: List[ExpertSlot]) -> Tuple[int, List[float]]:
        """返回 (最佳专家下标, 全专家得分列表)."""
        scores = [self.score(text, s) for s in slots]
        best = max(range(len(scores)), key=lambda i: scores[i])
        return best, scores


class ExpertEnsemble:
    """多专家集成: 路由 + 生成.

    用法:
        ens = ExpertEnsemble()
        ens.add_expert("classical", model, unit_map, domain_vocab)
        ens.add_expert("science", ...)
        out = ens.generate("道可道", max_len=20)   # 自动路由到 classical 专家续写
    """

    def __init__(self, router: Optional[ExpertRouter] = None) -> None:
        self.slots: List[ExpertSlot] = []
        self.router = router or ExpertRouter()
        self.domain_index: Dict[str, int] = {}

    def add_expert(self, domain: str, model: LatticeLM, unit_map: UnitMap,
                   domain_vocab: Optional[set] = None, weight: float = 1.0) -> None:
        idx = len(self.slots)
        self.slots.append(ExpertSlot(
            domain=domain, model=model, unit_map=unit_map,
            domain_vocab=domain_vocab or set(), weight=weight,
        ))
        self.domain_index[domain] = idx

    def __len__(self) -> int:
        return len(self.slots)

    def route(self, text: str) -> Tuple[int, List[float]]:
        return self.router.route(text, self.slots)

    @torch.no_grad()
    def generate(self, prompt: str, max_len: int = 32,
                 temperature: float = 1.0) -> Dict:
        """路由到最佳专家, 由其续写 (贪心/采样 next-unit)."""
        if not self.slots:
            raise RuntimeError("集成中没有任何专家")
        best, scores = self.route(prompt)
        slot = self.slots[best]
        model, um = slot.model, slot.unit_map
        model.eval()

        ids = um.encode(prompt)
        if not ids:
            ids = [um.UNK]
        ctx = ids[-model.cfg.max_seq_len:]   # 截断到模型上下文窗
        generated: List[int] = list(ctx)
        for _ in range(max_len):
            x = torch.tensor([generated[-model.cfg.max_seq_len:]],
                             dtype=torch.long)
            out = model(x)
            logits = out["logits"][0, -1, :]          # (V,)
            probs = self._softmax(logits.data / max(temperature, 1e-3))
            # 采样 (带温度)
            import numpy as np
            nxt = int(np.random.choice(len(probs), p=probs))
            if nxt == um.PAD:
                break
            generated.append(nxt)
        text_out = um.decode(generated)
        return {
            "prompt": prompt,
            "generated": text_out,
            "routed_domain": slot.domain,
            "scores": scores,
            "best_index": best,
        }

    @staticmethod
    def _softmax(x) -> "np.ndarray":
        import numpy as np
        z = x - x.max()
        e = np.exp(z)
        return e / e.sum()


# ----------------------------------------------------------------------
# 与 UnityOrchestrator 集成: 把集成作为一个 ExpertUnit 挂到总线上
# ----------------------------------------------------------------------
class EnsembleUnit(ExpertUnit):
    """把 ExpertEnsemble 包装成 SimplUnity 的一个能力单元.

    当收到 cognition 消息 (含待处理文本) 时, 调用集成进行路由+续写,
    并把结果作为 action 消息发回总线.
    """

    def __init__(self, ensemble: ExpertEnsemble, name: str = "expert_ensemble") -> None:
        super().__init__(name)
        self.ensemble = ensemble

    def handle(self, msg: "Message") -> Optional["Message"]:
        if msg.kind != MSG_COGNITION:
            return None
        text = msg.payload
        if not isinstance(text, str):
            return None
        try:
            result = self.ensemble.generate(text, max_len=24)
        except Exception as e:  # 容错: 集成失败不阻断总线
            return Message(MSG_ACTION, self.name, "action",
                           {"error": str(e), "prompt": text})
        return Message(MSG_ACTION, self.name, "action", result)


def assemble_unity(experts: List[Tuple[str, LatticeLM, UnitMap, set]],
                   router: Optional[ExpertRouter] = None) -> UnityOrchestrator:
    """组装一个最小可用的 SimplUnity 实例: 感知 -> 集成专家 -> 行动.

    Args:
        experts: [(domain, model, unit_map, domain_vocab), ...]
        router:  可选自定义路由器.
    Returns:
        UnityOrchestrator 实例 (已注册感知单元与集成专家单元).
    """
    ensemble = ExpertEnsemble(router=router)
    for domain, model, um, vocab in experts:
        ensemble.add_expert(domain, model, um, domain_vocab=vocab)

    uni = UnityOrchestrator()
    uni.register(EnsembleUnit(ensemble))
    uni._ensemble = ensemble  # 便于外部访问
    return uni


# ----------------------------------------------------------------------
# 从磁盘 checkpoint 装载专家
# ----------------------------------------------------------------------
def load_expert_from_checkpoint(domain: str, ckpt_path: str,
                                unit_map: UnitMap) -> ExpertSlot:
    """从 train_domain_expert 保存的 checkpoint 装载一个专家.

    注意: 需要预先知道该专家的 LatticeConfig (由 domain 推断或随 ckpt 保存).
    """
    from simplcore.tensor.io import load
    ckpt = load(ckpt_path)
    cfg_dict = ckpt["meta"]["config"]
    cfg = LatticeConfig(**{k: v for k, v in cfg_dict.items()
                           if k in LatticeConfig.__init__.__code__.co_varnames})
    model = LatticeLM(cfg)
    sd = ckpt["state_dict"]
    for n, p in model.named_parameters():
        if n in sd:
            p.data = sd[n]
    return ExpertSlot(domain=domain, model=model, unit_map=unit_map,
                      domain_vocab=set(), weight=1.0)


if __name__ == "__main__":
    # 轻量自测: 不训练, 直接构造 2 个空专家验证路由与组装逻辑
    um = UnitMap(list("学而时习之道德经仁义礼智信"))
    slot_a = ExpertSlot("classical", None, um,
                        domain_vocab=set("道德经仁义礼智信"))
    slot_b = ExpertSlot("science", None, um,
                        domain_vocab=set("太阳地球行星引力光"))
    ens = ExpertEnsemble()
    ens.add_expert("classical", None, um, domain_vocab=slot_a.domain_vocab)
    ens.add_expert("science", None, um, domain_vocab=slot_b.domain_vocab)
    best, scores = ens.route("道德经讲什么")
    print(f"路由测试: 输入='道德经讲什么' -> 命中 {ens.slots[best].domain}, 得分 {scores}")
    uni = assemble_unity([
        ("classical", None, um, slot_a.domain_vocab),
        ("science", None, um, slot_b.domain_vocab),
    ])
    print("SimplUnity 组装完成, 已注册单元:", list(uni.units.keys()))
