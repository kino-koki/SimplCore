# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""旧平铺路径兼容别名（0.1-α 结构化拆分）。

core/ 曾是 60+ 文件的平铺包；拆分为 arch/model/memory/dialogue/
training/eval/backend/deploy/theory 子包后，历史脚本与老 checkpoint 里的
``simplcore.core.<模块>`` 路径通过 sys.modules 别名继续可用。
旧路径按原语义解析到新子包模块（同名子包 model/memory/eval 另有
__init__ 级 re-export 兜底）。此机制是**兼容垫片**，新代码应使用新路径。

关于 lab 旧名（jitrl / conditions_ledger / pilot）
------------------------------------------------
pyproject.toml 规定"core 产品线不得反向 import lab"。因此 lab 旧路径
**不做立即 import**：通过 meta_path finder 惰性解析，只有真正有人使用
``simplcore.core.jitrl`` 等旧路径时才加载 simplcore.lab.* 模块；
``import simplcore.core`` 本身不再拉起 lab。
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from importlib.util import LazyLoader

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
    "gate2": "simplcore.core.eval.gate2",
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

_CORE_PREFIX = "simplcore.core."


class _LazyLabAliasFinder:
    """把 ``simplcore.core.<lab旧名>`` 解析为懒加载的 lab 模块。

    仅在有人显式 import 旧路径时才加载 simplcore.lab.*（见模块 docstring）；
    core 包自身导入不会触发本 finder。
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_CORE_PREFIX):
            return None
        old = fullname[len(_CORE_PREFIX):]
        new = _OLD_TO_LAB.get(old)
        if new is None:
            return None
        spec = importlib.util.find_spec(new)
        if spec is None or spec.origin is None:
            return None
        # 以别名 fullname 重建 loader：LazyLoader 包装的 loader 与 spec.name
        # 必须一致，否则 ImportError("loader ... cannot handle ...")。
        loader = importlib.machinery.SourceFileLoader(fullname, spec.origin)
        spec.loader = LazyLoader(loader)
        spec.name = fullname
        return spec


def install() -> None:
    """注册旧路径别名（幂等，由 simplcore.core.__init__ 调用）。

    core 内部旧名立即解析（import 的是 core 自身子包，无边界问题）；
    lab 旧名注册 meta_path finder，惰性加载。
    """
    core_pkg = sys.modules.get('simplcore.core')
    for old, new in _OLD_TO_NEW.items():
        mod = importlib.import_module(new)
        sys.modules.setdefault('simplcore.core.' + old, mod)
        if core_pkg is not None:
            if not hasattr(core_pkg, old):
                setattr(core_pkg, old, mod)
    if not any(isinstance(f, _LazyLabAliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LazyLabAliasFinder())


def lab_getattr(name: str):
    """供 simplcore.core 的 PEP 562 ``__getattr__`` 使用。

    属性形式访问 ``simplcore.core.jitrl`` 等旧名时按需 import 对应 lab
    模块；非 lab 旧名一律 AttributeError（交回正常属性查找语义）。
    """
    new = _OLD_TO_LAB.get(name)
    if new is None:
        raise AttributeError(name)
    return importlib.import_module(new)
