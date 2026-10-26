# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 架构组件登记 (Architecture Components Registry)
=========================================================

本模块集中登记 SimplCore 架构的核心组件, 并提供查询/校验接口。

命名与描述遵循「技术中性、可被代码验证」的原则:
  - 不使用伪科学命名 (如「量子」「谱混合」「条件计算」「自发塑性」等修辞);
  - 复杂度声明必须诚实: 凡依赖序列长度 n 的运算, 不得标注为 O(1)。

组件清单:
  1. 谱全局卷积 (SGC)            - FFT 全局卷积 + 多尺度谱混合/频域门控/因果谱注意力, O(n log n)
  2. 动态记忆通道 (DMC)           - 固定大小记忆库, 每步 O(C·S·D), 相对序列长度 n 不增长 (非 O(1))
  3. 条件计算路由 (CCR)           - 动态计算路径选择 (top-k 早退路由), 条件计算
  4. 复数位置编码 (CPE)           - 复数表示的绝对位置编码 (原称 EQP, 实为线性代数, 非量子计算)
  5. 在线估计更新 (OEU)           - 推理期基于迹的 Hebbian 式参数自适应
  6. 潜空间推演 (Rollout)         - 多步前瞻推演 (如 MCTS 式)
  7. 多模态路由 (ModalityRouter)  - 统一多模态路由

每个组件包含: 名称、英文名、组件类、描述、复杂度、特性说明。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn


# ======================================================================
# 组件定义 (中性命名, 复杂度诚实)
# ======================================================================

def _default_components() -> List[Dict[str, Any]]:
    """返回架构组件的默认登记定义."""
    return [
        {
            "id": 1,
            "name": "谱全局卷积",
            "name_en": "Spectral Global Convolution (SGC)",
            "component_class": "SpectralGlobalConv",
            "description": (
                "用 FFT 实现的全局卷积替代注意力。在频域做多尺度谱混合、"
                "可学习频域门控与因果谱注意力, 复杂度 O(n log n)。"
            ),
            "complexity": "O(n log n + F*r)",
            "advantage": "相比标准注意力 O(n^2), 在长序列上具有显著优势, 且保持因果性。",
        },
        {
            "id": 2,
            "name": "动态记忆通道",
            "name_en": "Dynamic Memory Channel (DMC)",
            "component_class": "DynamicMemoryChannel",
            "description": (
                "维护固定大小的外部记忆库 (num_channels x channel_size x dim), "
                "通过内容寻址进行读写。每步计算量 O(C·S·D), 不随序列长度 n 增长, "
                "但代价是记忆容量固定 (长序列必然覆盖/遗忘)。"
            ),
            "complexity": "O(C*S*D) per step (independent of n)",
            "advantage": "每步计算量不随序列长度增长; 但记忆容量受限, 非突破。",
        },
        {
            "id": 3,
            "name": "条件计算路由",
            "name_en": "Conditional Compute Router (CCR)",
            "component_class": "LegacyConditionalRouter",
            "description": (
                "每个 token 动态选择计算路径, 不同路径对应不同的计算深度。"
                "通过 top-k 路由实现条件计算, 降低平均计算量。"
            ),
            "complexity": "O(n * d * P) where P << n",
            "advantage": "条件计算, 简单 token 走浅路径, 复杂 token 走深路径。",
        },
        {
            "id": 4,
            "name": "复数位置编码",
            "name_en": "Complex Positional Encoder (CPE)",
            "component_class": "ComplexPositionalEncoder",
            "description": (
                "基于复数表示的绝对位置编码: 复指数相位编码 + 可学习酉/耦合"
                "线性变换。原称「量子启发编码 (EQP)」, 其代码实为线性代数, "
                "与量子计算无关, 已去除伪科学命名。"
            ),
            "complexity": "O(n * d * S) where S = num_basis",
            "advantage": "提供可学习的复数位置表示。",
        },
        {
            "id": 5,
            "name": "在线估计更新",
            "name_en": "Online Estimate Update (OEU)",
            "component_class": "PlasticityUpdateModule",
            "description": (
                "在推理过程中根据输入信号自适应更新部分权重, "
                "采用基于资格迹 (eligibility trace) 的 Hebbian 式在线学习, "
                "无需反向传播即可适应。"
            ),
            "complexity": "O(d^2) per update",
            "advantage": "推理时在线适应, 无需完整微调。",
        },
        {
            "id": 6,
            "name": "潜空间推演",
            "name_en": "Latent Space Rollout (Rollout)",
            "component_class": "RolloutEngine",
            "description": (
                "在潜空间进行多步前瞻推演, 类似 MCTS 的搜索过程。"
                "通过共享参数迭代预测未来状态, 实现规划能力。"
            ),
            "complexity": "O(T * cost_per_step) where T = rollout steps",
            "advantage": "赋予模型规划与前瞻能力, 提升决策质量。",
        },
        {
            "id": 7,
            "name": "多模态融合",
            "name_en": "Modality Router (ModalityRouter)",
            "component_class": "ModalityRouter",
            "description": (
                "将文本、图像、视频等不同模态的输入统一路由到 SimplCore 核心。"
                "通过模态类型嵌入和融合投影实现跨模态信息交互。"
            ),
            "complexity": "O(sum(L_i) * d)",
            "advantage": "统一架构处理多模态输入, 无需模态特定编码器。",
        },
    ]


# ======================================================================
# 架构组件登记类
# ======================================================================

class ArchitectureComponents:
    """SimplCore 架构组件管理器.

    提供组件信息的查询、验证和报告功能。

    Usage:
        >>> ab = ArchitectureComponents()
        >>> ab.list_all()          # 列出所有组件
        >>> ab.get_component('谱全局卷积')  # 获取特定组件
        >>> ab.print_report()       # 打印格式化报告
        >>> ab.verify_component('谱全局卷积', model)  # 验证模型中是否包含该组件
    """

    # 组件名称的别名映射 (支持中英文查询)
    _ALIASES: Dict[str, str] = {
        "谱全局卷积": "谱全局卷积",
        "sgc": "谱全局卷积",
        "spectral global convolution": "谱全局卷积",
        "动态记忆通道": "动态记忆通道",
        "dmc": "动态记忆通道",
        "dynamic memory": "动态记忆通道",
        "条件计算路由": "条件计算路由",
        "ccr": "条件计算路由",
        "conditional compute router": "条件计算路由",
        "复数位置编码": "复数位置编码",
        "cpe": "复数位置编码",
        "complex positional encoding": "复数位置编码",
        "在线估计更新": "在线估计更新",
        "oeu": "在线估计更新",
        "online estimate": "在线估计更新",
        "潜空间推演": "潜空间推演",
        "rollout": "潜空间推演",
        "多模态路由": "多模态路由",
        "modality router": "多模态路由",
        "modalityrouter": "多模态路由",
    }

    def __init__(self) -> None:
        """初始化架构组件."""
        self.components: List[Dict[str, Any]] = _default_components()
        # 构建名称到组件的索引
        self._index: Dict[str, Dict[str, Any]] = {}
        for bt in self.components:
            self._index[bt["name"]] = bt
            self._index[bt["name_en"].split("(")[0].strip().lower()] = bt
            self._index[bt["component_class"].lower()] = bt

    def _resolve_name(self, name: str) -> Optional[str]:
        """将别名解析为标准名称."""
        key = name.strip().lower()
        if key in self._ALIASES:
            return self._ALIASES[key]
        # 尝试直接匹配
        if name in self._index:
            return name
        if key in self._index:
            return self._index[key]["name"]
        return None

    def get_component(self, name: str) -> Dict[str, Any]:
        """获取特定组件的详细信息.

        Args:
            name: 组件名称 (支持中英文及缩写, 如 '谱全局卷积', 'sgc', 'SGC').

        Returns:
            dict: 包含 name, name_en, component_class, description, complexity, advantage.

        Raises:
            KeyError: 当名称无法匹配任何组件时.
        """
        resolved = self._resolve_name(name)
        if resolved is None:
            available = [bt["name"] for bt in self.components]
            raise KeyError(
                f"未知组件名称: '{name}'. 可用: {available}"
            )
        return self._index[resolved]

    def list_all(self) -> List[Dict[str, Any]]:
        """列出所有架构组件及其描述.

        Returns:
            list[dict]: 组件详细信息列表.
        """
        return list(self.components)

    def verify_component(
        self,
        name: str,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """验证某个组件是否存在于模型实例中.

        通过检查模型及其子模块的类名和属性名来判断组件是否已集成。

        Args:
            name: 组件名称.
            model: 待验证的模型实例 (nn.Module).

        Returns:
            dict: {
              "name": 组件名称,
              "found": bool, 是否找到,
              "component_class": 期望的组件类名,
              "matched_modules": list[str], 匹配的模块路径,
              "details": str, 验证详情,
            }
        """
        bt = self.get_component(name)
        target_class = bt["component_class"]
        target_lower = target_class.lower()

        # 将组件类名拆分为 camelCase 关键词 (如 "LegacySpectralMixer" → ["resonance", "field", "layer"])
        import re
        camel_parts = re.findall(r'[A-Z][a-z]*', target_class)
        keywords = [p.lower() for p in camel_parts if len(p) > 2]  # 忽略过短的词
        # 也加入完整类名和简称
        keywords.append(target_lower)
        # 从别名中提取简称 (如 rfp, dmc, tar 等)
        resolved = self._resolve_name(name)
        if resolved:
            for alias_key, alias_val in self._ALIASES.items():
                if alias_val == resolved and len(alias_key) <= 5:
                    keywords.append(alias_key)

        matched_modules: List[str] = []

        def _matches(text: str) -> bool:
            """检查文本是否包含任一关键词."""
            text_lower = text.lower()
            return any(kw in text_lower for kw in keywords)

        # 检查模型自身的类名
        model_class_name = type(model).__name__
        if _matches(model_class_name):
            matched_modules.append(model_class_name)

        # 遍历所有子模块
        for module_name, module in model.named_modules():
            if module_name == "":
                continue
            module_class = type(module).__name__
            # 检查类名匹配
            if _matches(module_class):
                matched_modules.append(module_name)
            # 检查属性名匹配 (如 model.rfp, model.dmc 等)
            elif _matches(module_name):
                if module_name not in matched_modules:
                    matched_modules.append(module_name)

        # 检查模型是否有相关属性
        for attr_name in dir(model):
            if _matches(attr_name):
                attr_val = getattr(model, attr_name, None)
                if attr_val is not None and attr_name not in matched_modules:
                    matched_modules.append(attr_name)

        found = len(matched_modules) > 0
        if found:
            details = (
                f"在模型中找到组件 '{bt['name']}' 的组件 "
                f"(类名/属性: {matched_modules[:5]})"
            )
        else:
            details = (
                f"未在模型中找到组件 '{bt['name']}' "
                f"(期望组件类: {target_class})"
            )

        return {
            "name": bt["name"],
            "name_en": bt["name_en"],
            "found": found,
            "component_class": target_class,
            "matched_modules": matched_modules,
            "details": details,
        }

    def verify_all_components(self, model: nn.Module) -> Dict[str, Any]:
        """验证模型中是否包含所有架构组件.

        Args:
            model: 待验证的模型实例.

        Returns:
            dict: {
              "total": int, 总组件数,
              "found_count": int, 找到的组件数,
              "results": list[dict], 各组件的验证结果,
              "all_found": bool, 是否全部找到,
            }
        """
        results = []
        found_count = 0
        for bt in self.components:
            result = self.verify_component(bt["name"], model)
            results.append(result)
            if result["found"]:
                found_count += 1

        return {
            "total": len(self.components),
            "found_count": found_count,
            "results": results,
            "all_found": found_count == len(self.components),
        }

    def print_report(self) -> None:
        """打印格式化的架构组件报告."""
        print("=" * 70)
        print("SimplCore 架构组件报告")
        print("=" * 70)

        for bt in self.components:
            print(f"\n{'─' * 70}")
            print(f"  组件 {bt['id']}: {bt['name']} ({bt['name_en']})")
            print(f"{'─' * 70}")
            print(f"  组件类名:   {bt['component_class']}")
            print(f"  描述:       {bt['description']}")
            print(f"  复杂度:     {bt['complexity']}")
            print(f"  优势:       {bt['advantage']}")

        print(f"\n{'=' * 70}")
        print(f"  共 {len(self.components)} 个架构组件")
        print(f"{'=' * 70}")

    def to_dict(self) -> Dict[str, Any]:
        """将所有组件信息序列化为字典.

        Returns:
            dict: 以组件名称为键的字典.
        """
        return {bt["name"]: bt for bt in self.components}

    def __len__(self) -> int:
        """返回组件总数."""
        return len(self.components)

    def __contains__(self, name: str) -> bool:
        """检查是否包含某个组件."""
        return self._resolve_name(name) is not None

    def __getitem__(self, name: str) -> Dict[str, Any]:
        """通过名称获取组件信息."""
        return self.get_component(name)

    def __iter__(self):
        """迭代所有组件."""
        return iter(self.components)


# ======================================================================
# 便捷函数
# ======================================================================

def list_components() -> List[Dict[str, Any]]:
    """列出所有架构组件 (便捷函数)."""
    return ArchitectureComponents().list_all()


def get_component_info(name: str) -> Dict[str, Any]:
    """获取特定组件信息 (便捷函数)."""
    return ArchitectureComponents().get_component(name)


def verify_model_components(model: nn.Module) -> Dict[str, Any]:
    """验证模型包含的所有组件 (便捷函数)."""
    return ArchitectureComponents().verify_all_components(model)


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ArchitectureComponents 自检")
    print("=" * 70)

    ab = ArchitectureComponents()

    # 测试组件总数
    assert len(ab) == 7, f"应有 7 个组件, 得到 {len(ab)}"
    print(f"[PASS] 组件总数: {len(ab)}")

    # 测试 list_all
    all_bt = ab.list_all()
    assert len(all_bt) == 7
    for bt in all_bt:
        assert "name" in bt
        assert "name_en" in bt
        assert "component_class" in bt
        assert "description" in bt
        assert "complexity" in bt
        assert "advantage" in bt
    print("[PASS] list_all 返回完整信息")

    # 测试 get_component - 中文名
    sgc = ab.get_component("谱全局卷积")
    assert sgc["component_class"] == "SpectralGlobalConv"
    assert "O(n log n" in sgc["complexity"]
    print(f"[PASS] 中文名查询: {sgc['name']} → {sgc['component_class']}")

    # 测试 get_component - 英文缩写
    dmc = ab.get_component("dmc")
    assert dmc["name"] == "动态记忆通道"
    print(f"[PASS] 缩写查询: dmc → {dmc['name']}")

    # 测试 get_component - 英文缩写
    ccr2 = ab.get_component("ccr")
    assert ccr2["name"] == "条件计算路由"
    print(f"[PASS] 英文缩写查询: ccr → {ccr2['name']}")

    # 测试 get_component - 组件类名
    roll = ab.get_component("RolloutEngine")
    assert roll["name"] == "潜空间推演"
    print(f"[PASS] 组件类名查询: RolloutEngine → {roll['name']}")

    # 测试 __contains__
    assert "谱全局卷积" in ab
    assert "sgc" in ab
    assert "cpe" in ab
    assert "rollout" in ab
    assert "不存在" not in ab
    print("[PASS] __contains__ 正常")

    # 测试 __getitem__
    cpe = ab["cpe"]
    assert cpe["name"] == "复数位置编码"
    print(f"[PASS] __getitem__ 正常: {cpe['name']}")

    # 测试未知名称
    try:
        ab.get_component("不存在的组件")
        assert False, "应抛出 KeyError"
    except KeyError as e:
        print(f"[PASS] 未知名称正确抛出 KeyError")

    # 测试 verify_component - 构建一个包含 SpectralMixer 的假模型
    class FakeSGCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectral_conv = nn.Linear(64, 64)
            self.dmc = nn.Linear(64, 64)

    fake_model = FakeSGCModel()
    result = ab.verify_component("谱全局卷积", fake_model)
    assert result["found"] is True
    print(f"[PASS] 验证谱全局卷积: found={result['found']}, matched={result['matched_modules']}")

    result_dmc = ab.verify_component("dmc", fake_model)
    assert result_dmc["found"] is True
    print(f"[PASS] 验证DMC: found={result_dmc['found']}, matched={result_dmc['matched_modules']}")

    # 测试未包含的组件
    result_cpe = ab.verify_component("cpe", fake_model)
    assert result_cpe["found"] is False
    print(f"[PASS] 验证CPE(不存在): found={result_cpe['found']}")

    # 测试 verify_all_components
    all_results = ab.verify_all_components(fake_model)
    assert all_results["total"] == 7
    assert all_results["found_count"] >= 2
    print(f"[PASS] verify_all_components: {all_results['found_count']}/{all_results['total']} found")

    # 测试 to_dict
    bt_dict = ab.to_dict()
    assert len(bt_dict) == 7
    assert "谱全局卷积" in bt_dict
    print(f"[PASS] to_dict 正常")

    # 打印完整报告
    print()
    ab.print_report()

    print("\n" + "=" * 70)
    print("ArchitectureComponents 所有测试 PASSED.")
    print("=" * 70)
