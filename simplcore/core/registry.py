# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Mixer 注册表 —— SimplCore 非注意力 mixer 的**单一事实源**.

为什么需要它
------------
在注册表出现之前，新增一种 mixer 需要同步修改四处平行实现：

1. ``config.validate()`` 里的白名单元组；
2. ``config.analytic_params()`` 里的解析参数公式分支；
3. ``model.SimplCoreBlock.__init__`` 里的 if/elif 构造分支；
4. ``purity`` 的判定口径（靠人肉保持一致）。

四处只要漏改一处，就会出现"配置能过 validate、模型造不出来"或
"参数预算与实际模型不符"这类静默漂移。注册表把「 mixer 叫什么、
怎么构造、参数怎么算」收敛到一个 ``MixerSpec`` 里：

* ``build(cfg)`` —— 构造 mixer 实例（惰性 import，保持 config 无 torch 依赖）；
* ``family`` —— 解析参数公式的家族分支（``"sgc"`` 振荡器项 / ``"srf"`` SSM 项）；
* ``respects_content_gate`` / ``respects_local_conv`` —— 配置开关是否生效
  （``sgc-nogate`` 不吃 content_gate、``sgc-nolocal`` 不吃 local_kernel，
  这两个语义以前散落在构造分支和参数公式两处，现在只写一遍）。

新增 mixer 的完整步骤（只有两处，且都有测试兜底）::

    1. 实现 mixer 类并提供 ``init_state`` / ``step``（O(1) 流式）；
    2. 在本文件 ``_register_builtin()`` 里登记一个 ``MixerSpec``。

``gate_ab`` 是横切开关（把 mixer 覆盖为 ``GatedSpectralConv2`` 多头谱基），
不属于任何一种 mixer，因此不在注册表内，由 ``model.SimplCoreBlock``
在查表前处理——``analytic_params`` 对它有对应的 +2·D²+2·D 项。

纯度约束：注册表里的类一律不得是注意力模块。``purity.assert_registry_pure()``
会在 CI 里遍历注册表做类名断言；新增条目若是注意力实现，注册当场被测出。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Tuple

__all__ = [
    "MixerSpec",
    "get_mixer_spec",
    "build_mixer",
    "mixer_kinds",
    "iter_mixer_classes",
    "register_mixer",
]


@dataclass(frozen=True)
class MixerSpec:
    """一种 mixer 的全部静态事实。

    Attributes:
        name: ``SimplCoreConfig.mixer`` 的合法取值。
        build: ``(cfg) -> nn.Module`` 构造函数（内部惰性 import mixer 类）。
        family: 解析参数家族——``"sgc"``（振荡器基字典项）或
            ``"srf"``（selective SSM 项）。新增家族需同步扩展
            ``config.analytic_params`` 的对应分支。
        respects_content_gate: ``cfg.content_gate`` 对该 mixer 是否生效。
        respects_local_conv: ``cfg.local_kernel``（深度局部卷积）是否生效。
        docstring: 一句话说明（``simplcore-presets`` 报表用）。
    """

    name: str
    build: Callable
    family: str
    respects_content_gate: bool = True
    respects_local_conv: bool = True
    docstring: str = ""


_REGISTRY: Dict[str, MixerSpec] = {}


def register_mixer(spec: MixerSpec) -> None:
    """登记一种 mixer；重名登记直接抛错（防止无意覆盖既有语义）."""
    if spec.name in _REGISTRY:
        raise ValueError(f"mixer {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


# ----------------------------------------------------------------------------
# 内置 mixer 的构造函数（惰性 import：config.py 引用本模块时不得连带 torch）
# ----------------------------------------------------------------------------
def _build_sgc(cfg):
    from .sgc import SpectralGlobalConv
    return SpectralGlobalConv(
        dim=cfg.dim,
        num_bases=cfg.num_bases,
        decay_max=cfg.decay_max,
        local_kernel=cfg.local_kernel,
        content_gate=cfg.content_gate,
        out_gate=cfg.out_gate,
        evidence_gate=cfg.evidence_gate,
        uncertainty_gate=cfg.uncertainty_gate,
        norm=cfg.norm,
        fft_dtype=cfg.resolve_fft_dtype(),
        eps=cfg.norm_eps,
        use_mixer_stream=cfg.use_mixer_stream,
        path_gate=getattr(cfg, "sgc_path_gate", False),
        inst_nonlin=getattr(cfg, "sgc_inst_nonlin", False),
        init_mode=getattr(cfg, "sgc_init_mode", "logfreq"),
        rho_param=getattr(cfg, "sgc_rho_param", "sigmoid"),
    )


def _build_sgc_nogate(cfg):
    from .sgc import SpectralGlobalConv
    return SpectralGlobalConv(
        dim=cfg.dim,
        num_bases=cfg.num_bases,
        decay_max=cfg.decay_max,
        local_kernel=cfg.local_kernel,
        content_gate=False,          # sgc-nogate：内容门控被本变体移除
        out_gate=cfg.out_gate,
        evidence_gate=cfg.evidence_gate,
        uncertainty_gate=cfg.uncertainty_gate,
        norm=cfg.norm,
        fft_dtype=cfg.resolve_fft_dtype(),
        eps=cfg.norm_eps,
        use_mixer_stream=cfg.use_mixer_stream,
        path_gate=getattr(cfg, "sgc_path_gate", False),
        inst_nonlin=getattr(cfg, "sgc_inst_nonlin", False),
        init_mode=getattr(cfg, "sgc_init_mode", "logfreq"),
        rho_param=getattr(cfg, "sgc_rho_param", "sigmoid"),
    )


def _build_sgc_nolocal(cfg):
    from .sgc import SpectralGlobalConv
    return SpectralGlobalConv(
        dim=cfg.dim,
        num_bases=cfg.num_bases,
        decay_max=cfg.decay_max,
        local_kernel=1,              # sgc-nolocal：深度局部卷积被本变体移除
        content_gate=cfg.content_gate,
        out_gate=cfg.out_gate,
        evidence_gate=cfg.evidence_gate,
        uncertainty_gate=cfg.uncertainty_gate,
        norm=cfg.norm,
        fft_dtype=cfg.resolve_fft_dtype(),
        eps=cfg.norm_eps,
        use_mixer_stream=cfg.use_mixer_stream,
        path_gate=getattr(cfg, "sgc_path_gate", False),
        inst_nonlin=getattr(cfg, "sgc_inst_nonlin", False),
        init_mode=getattr(cfg, "sgc_init_mode", "logfreq"),
        rho_param=getattr(cfg, "sgc_rho_param", "sigmoid"),
    )


def _build_sgc_routed(cfg):
    from .sgc import RoutedSpectralGlobalConv
    return RoutedSpectralGlobalConv(
        dim=cfg.dim,
        num_bases=cfg.num_bases,
        decay_max=cfg.decay_max,
        local_kernel=cfg.local_kernel,
        content_gate=cfg.content_gate,
        out_gate=cfg.out_gate,
        evidence_gate=cfg.evidence_gate,
        uncertainty_gate=cfg.uncertainty_gate,
        norm=cfg.norm,
        fft_dtype=cfg.resolve_fft_dtype(),
        eps=cfg.norm_eps,
        path_gate=getattr(cfg, "sgc_path_gate", False),
        inst_nonlin=getattr(cfg, "sgc_inst_nonlin", False),
        init_mode=getattr(cfg, "sgc_init_mode", "logfreq"),
        rho_param=getattr(cfg, "sgc_rho_param", "sigmoid"),
        route_groups=getattr(cfg, "sgc_route_groups", 4),
        route_topk=getattr(cfg, "sgc_route_topk", 2),
        route_aux=getattr(cfg, "sgc_route_aux", 1.0),
    )


def _build_srf(cfg):
    from .srf import SpectralResonanceFlow
    return SpectralResonanceFlow(
        dim=cfg.dim,
        num_bases=cfg.num_bases,
        num_heads=cfg.srf_heads,
        chunk_size=cfg.srf_chunk,
        decay_max=cfg.decay_max,
        local_kernel=cfg.local_kernel,
        content_gate=cfg.content_gate,
        out_gate=cfg.out_gate,
        evidence_gate=cfg.evidence_gate,
        uncertainty_gate=cfg.uncertainty_gate,
        norm=cfg.norm,
        eps=cfg.norm_eps,
        use_mixer_stream=cfg.use_mixer_stream,
    )


def _register_builtin() -> None:
    for spec in (
        MixerSpec(
            name="sgc",
            build=_build_sgc,
            family="sgc",
            docstring="内容门控因果谱全局卷积（默认 mixer）",
        ),
        MixerSpec(
            name="sgc-nogate",
            build=_build_sgc_nogate,
            family="sgc",
            respects_content_gate=False,
            docstring="SGC 去内容门控消融变体",
        ),
        MixerSpec(
            name="sgc-nolocal",
            build=_build_sgc_nolocal,
            family="sgc",
            respects_local_conv=False,
            docstring="SGC 去深度局部卷积消融变体",
        ),
        MixerSpec(
            name="sgc-r",
            build=_build_sgc_routed,
            family="sgc",
            docstring="SGC-R 谱专家路由变体（MiTA 式 shared+routed，MoE top-k，非注意力）",
        ),
        MixerSpec(
            name="srf",
            build=_build_srf,
            family="srf",
            docstring="谱振流网络（SGC + selective SSM，输入相关 Δ）",
        ),
    ):
        register_mixer(spec)


_register_builtin()


#: 各 mixer 的实现类路径（惰性 import；purity 纯度检查用）
_MIXER_CLASS_PATHS = {
    "sgc": ("sgc", "SpectralGlobalConv"),
    "sgc-nogate": ("sgc", "SpectralGlobalConv"),
    "sgc-nolocal": ("sgc", "SpectralGlobalConv"),
    "sgc-r": ("sgc", "RoutedSpectralGlobalConv"),
    "srf": ("srf", "SpectralResonanceFlow"),
}


# ----------------------------------------------------------------------------
# 查询接口
# ----------------------------------------------------------------------------
def get_mixer_spec(name: str) -> MixerSpec:
    """按名取 spec；未登记的 mixer 名在此报错（validate 的白名单即来源于此）."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown mixer {name!r}; SimplCore ships non-attention mixers only "
            f"({', '.join(sorted(_REGISTRY))})"
        ) from None


def build_mixer(cfg):
    """按 ``cfg.mixer`` 查表构造 mixer 实例."""
    return get_mixer_spec(cfg.mixer).build(cfg)


def mixer_kinds() -> Tuple[str, ...]:
    """所有已登记 mixer 名（``config.validate()`` 的白名单）."""
    return tuple(sorted(_REGISTRY))


def iter_mixer_classes() -> Iterator[Tuple[str, type]]:
    """遍历 ``(mixer 名, mixer 类)``，供 ``purity.assert_registry_pure`` 使用.

    类路径单独成表（惰性 import），只为纯度检查服务；新增 mixer 时
    ``purity`` 测试会强制这里的键与注册表一致。
    """
    import importlib
    for name, (mod_name, cls_name) in _MIXER_CLASS_PATHS.items():
        mod = importlib.import_module(f".{mod_name}", __package__)
        yield name, getattr(mod, cls_name)
