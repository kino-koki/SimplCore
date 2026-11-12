# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""旧平铺路径兼容别名（0.1-α 结构化拆分）。

core/ 曾是 60+ 文件的平铺包；拆分为 arch/model/memory/dialogue/
training/eval/backend/deploy 八个子包后，历史脚本与老 checkpoint 里的
``simplcore.core.<模块>`` 路径通过 sys.modules 别名继续可用。
旧路径按原语义解析到新子包模块（同名子包 model/memory/eval 另有
__init__ 级 re-export 兜底）。此机制是**兼容垫片**，新代码应使用新路径。
"""
from __future__ import annotations

import importlib
import sys

_OLD_TO_NEW = {
    "ab": "simplcore.core.training.ab",
    "abstain": "simplcore.core.arch.abstain",
    "accel": "simplcore.core.arch.accel",
    "adapter_model": "simplcore.core.model.adapter_model",
    "backend_bridge": "simplcore.core.backend.backend_bridge",
    "backend_contract": "simplcore.core.backend.backend_contract",
    "bio_computation": "simplcore.core.bio_enhance",
    "bench": "simplcore.core.eval.bench",
    "campaign": "simplcore.core.deploy.campaign",
    "chat": "simplcore.core.dialogue.chat",
    "code_exec": "simplcore.core.memory.code_exec",
    "code_passk_eval": "simplcore.core.eval.code_passk_eval",
    "config": "simplcore.core.model.config",
    "conformance_runner": "simplcore.core.backend.conformance_runner",
    "data": "simplcore.core.training.data",
    "deploy_budget": "simplcore.core.deploy.deploy_budget",
    "dialogue_data": "simplcore.core.dialogue.dialogue_data",
    "dialogue_eval": "simplcore.core.dialogue.dialogue_eval",
    "dialogue_runtime": "simplcore.core.dialogue.dialogue_runtime",
    "dream": "simplcore.core.memory.dream",
    "envlearn": "simplcore.core.training.envlearn",
    "equal_budget": "simplcore.core.eval.equal_budget",
    "eval": "simplcore.core.eval.eval",
    "export": "simplcore.core.model.export",
    "extrapolation": "simplcore.core.eval.extrapolation",
    "gate2": "simplcore.core.training.gate2",
    "gates": "simplcore.core.arch.gates",
    "grounding": "simplcore.core.arch.grounding",
    "hf_bridge": "simplcore.core.model.hf_bridge",
    "longctx_eval": "simplcore.core.eval.longctx_eval",
    "mathproof": "simplcore.core.eval.mathproof",
    "memory": "simplcore.core.memory.memory",
    "memprobe": "simplcore.core.eval.memprobe",
    "model": "simplcore.core.model.model",
    "nonlinear": "simplcore.core.model.nonlinear",
    "numerical_stability": "simplcore.core.arch.numerical_stability",
    "paper_audit": "simplcore.core.eval.paper_audit",
    "phase1_train": "simplcore.core.training.phase1_train",
    "probes": "simplcore.core.eval.probes",
    "purity": "simplcore.core.arch.purity",
    "reasoning": "simplcore.core.dialogue.reasoning",
    "registry": "simplcore.core.arch.registry",
    "retrieval_bound": "simplcore.core.eval.retrieval_bound",
    "scaling": "simplcore.core.eval.scaling",
    "selective": "simplcore.core.arch.selective",
    "seqlen_curriculum": "simplcore.core.training.seqlen_curriculum",
    "sft": "simplcore.core.training.sft",
    "sgc": "simplcore.core.arch.sgc",
    "smm": "simplcore.core.memory.smm",
    "sprint30": "simplcore.core.deploy.sprint30",
    "srf": "simplcore.core.arch.srf",
    "ssm": "simplcore.core.arch.ssm",
    "theory": "simplcore.core.eval.claims_audit",
    "throughput_eval": "simplcore.core.eval.throughput_eval",
    "tokenization": "simplcore.core.model.tokenization",
    "tokenizer": "simplcore.core.model.tokenizer",
    "train": "simplcore.core.training.train",
    "trainer": "simplcore.core.training.trainer",
    "training_integration": "simplcore.core.training.training_integration",
}

_OLD_TO_LAB = {
    "jitrl": "simplcore.lab.jitrl",
    "conditions_ledger": "simplcore.lab.conditions_ledger",
    "pilot": "simplcore.lab.pilot",
}


def install() -> None:
    """注册旧路径别名（幂等，由 simplcore.core.__init__ 调用）。"""
    core_pkg = sys.modules.get('simplcore.core')
    for old, new in {**_OLD_TO_NEW, **_OLD_TO_LAB}.items():
        mod = importlib.import_module(new)
        sys.modules.setdefault('simplcore.core.' + old, mod)
        if core_pkg is not None:
            if not hasattr(core_pkg, old):
                setattr(core_pkg, old, mod)
