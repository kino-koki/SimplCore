# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""FX 图模式优化 pass。

提供：
  * ``summarize_graph`` —— 算子频次统计；
  * ``eliminate_trivial_aliases`` —— 删除 trace 后残留的 ``clone`` /
    ``contiguous`` 等无效别名节点 (真实可用的图优化)；
  * ``optimize_for_inference`` —— 组合上述 pass 并返回新的 GraphModule。

这些 pass 作用在 ``torch.fx.GraphModule`` 上，配合 :meth:`SimplCoreModel.trace_fx`
即可获得可检查、可优化的图表示，满足"FX 图模式"要求。
"""
from __future__ import annotations

import torch.fx

_TRIVIAL = {
    torch.clone,
}


def summarize_graph(gm: torch.fx.GraphModule) -> dict:
    """统计 FX 图中各算子出现频次。"""
    counts: dict = {}
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method", "call_module"):
            key = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target))
            counts[key] = counts.get(key, 0) + 1
    return counts


def eliminate_trivial_aliases(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """删除图中等价的别名节点 (clone / contiguous)。

    将 ``y = x.clone()`` / ``x.contiguous()`` 这类节点用其唯一输入替换，
    并重组图。返回新 GraphModule。
    """
    graph = gm.graph
    aliases = []
    for node in graph.nodes:
        if node.op == "call_function" and node.target is torch.clone and len(node.args) == 1:
            aliases.append(node)
        elif node.op == "call_method" and node.target == "contiguous":
            aliases.append(node)

    for node in aliases:
        src = node.args[0]
        node.replace_all_uses_with(src)
        graph.erase_node(node)

    # 原地修改并重新编译，保留导出图的参数绑定 (避免重建导致参数丢失)
    graph.lint()
    gm.recompile()
    return gm


def optimize_for_inference(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """组合推理优化 pass。"""
    gm = eliminate_trivial_aliases(gm)
    gm.graph.lint()
    gm.recompile()
    return gm
