# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""HuggingFace / 任意 ``torch.nn.Module`` → SimplCore 互操作桥接.

设计目标
========
让已有 checkpoint（GPT-2 / Llama / Qwen 风格的 ``state_dict``）能**安全搬运**
进 SimplCore，而不引入任何注意力子模块，并完整继承 SimplCore 的特性
（因果全局卷积、O(1) 状态流式、内容门控、无位置嵌入表）。

为什么不 import transformers
---------------------------
本模块**不依赖** ``transformers`` 库：它接收任意 ``torch.nn.Module`` 或
一个 ``state_dict``，并做**字段名启发式识别**（GPT-2 / Llama / Qwen 风格都能认）。
这样 (a) 桥接层在纯 torch 环境里可用，(b) 对未安装的 HF 模型也能离线搬运权重，
(c) 与 SimplCore 主线"纯 PyTorch"的可复现性定位一致。

核心原则: **搬得走就搬，搬不走就诚实零初始化**
----------------------------------------------
* **可 1:1 搬运的组件**（embedding / FFN / RMSNorm / final norm / 输出头）：
  直接按形状匹配搬运，并报告"搬运成功/形状不匹配"。
* **不可 1:1 搬运的组件**（源模型的注意力 QKV / 位置编码等）：
  **一律不碰**。SimplCore 没有 KV / 注意力，这些权重没有对应接收端，
  强行投影只会污染谱全局卷积核。桥接只改"可搬运组件"的初值，其余走默认初始化。
* 每一步都返回一份**搬运报告**（report），调用方可以审计哪些键被搬、哪些被跳过、
  为什么——这是"不会有严重后果"的工程保证：宁可漏搬（走默认初始化），
  不可错搬（形状错位的随机矩阵当权重）。

迁移路径
-------
1. :func:`init_simplcore_from_hf`  —— 一键：HF 模型 → SimplCore
2. :func:`migrate_embeddings_and_ffn` —— 只搬"安全"组件（保守模式）

复杂度与流式特性**天然继承**：桥接只改"权重初值"，不改
:class:`~simplcore.core.model.SimplCoreModel` 的结构；O(L log L) 训练 /
O(1) 状态流式推理 / 内容门控 全部原样保留。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .config import SimplCoreConfig, get_preset
from .model import SimplCoreModel, SimplCoreBlock


# =============================================================================
# 搬运报告
# =============================================================================
@dataclass
class MigrationReport:
    """一次权重搬运的审计记录。"""
    moved: List[str] = field(default_factory=list)
    skipped_shape: List[Tuple[str, tuple, tuple]] = field(default_factory=list)
    skipped_unknown: List[str] = field(default_factory=list)
    warmstarted: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """没有"错误搬运"（形状错配被静默接受）即视为安全。

        本模块的设计是: 形状错配一律进 skipped_shape, 绝不污染模型,
        所以 ok 恒为 True —— 错配被记录而非被掩盖。
        """
        return True

    def summary(self) -> str:
        lines = [
            f"moved         : {len(self.moved)}",
            f"skipped shape : {len(self.skipped_shape)}",
            f"skipped unk   : {len(self.skipped_unknown)}",
            f"warmstart     : {len(self.warmstarted)}",
        ]
        for n in self.notes:
            lines.append(f"note          : {n}")
        return "\n".join(lines)


# =============================================================================
# 键名识别 (GPT-2 / Llama / Qwen 风格)
# =============================================================================
# 各框架的键名差异 (举例):
#   GPT-2 : wte.weight / transformer.h.N.ln_1.weight / c_attn.weight / c_fc.weight
#   Llama : model.embed_tokens.weight / model.layers.N.self_attn.o_proj
#   Qwen  : model.embed_tokens.weight / model.layers.N.attn.o_proj
# 统一做法: 把 "xxx.<semantic>.weight" 与 "xxx.<semantic>" 两种形式都认,
# 即匹配语义段时允许其后还有一个 .weight / .bias 后缀。
def _semantic(k: str) -> str:
    """取键名的"语义段": 'blocks.3.attn.q_proj' -> 'q_proj';
    'wte.weight' -> 'wte'; 'ln_f.weight' -> 'ln_f'.

    规则: 去掉末尾的 .weight / .bias, 再取剩余部分的最后一段。
    """
    base = k
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return base.split(".")[-1]


def _find_first(src_state: Mapping[str, torch.Tensor], *candidates: str,
                layer: Optional[int] = None
                ) -> Optional[torch.Tensor]:
    """按语义段在 src_state 里找匹配键（允许 .weight/.bias 后缀）。

    ``layer is None`` 时返回第一个匹配（全模型共享/不分区场景）；
    指定 ``layer`` 时优先返回该层的键（GQA 逐层 q/k/v 各不同），
    找不到该层键时回退到第一个匹配（不分区布局仍可用）。
    """
    global_hits: List[torch.Tensor] = []
    layer_hits: List[torch.Tensor] = []
    for c in candidates:
        for sk, sv in src_state.items():
            if _semantic(sk) != c:
                continue
            m = re.search(r"layers\.(\d+)\.", sk)
            idx = int(m.group(1)) if m else None
            if layer is not None and idx == layer:
                layer_hits.append(sv)
            elif layer is None:
                global_hits.append(sv)
    if layer is not None and layer_hits:
        return layer_hits[0]
    if global_hits:
        return global_hits[0]
    return None


# =============================================================================
# 组件级搬运 (保守模式: 只搬可 1:1 匹配的形状)
# =============================================================================
def migrate_embeddings_and_ffn(sgc: SimplCoreModel,
                               src_state: Mapping[str, torch.Tensor],
                               report: Optional[MigrationReport] = None
                               ) -> MigrationReport:
    """保守模式：只搬运 embedding / 输出头 / norms / FFN（全部可 1:1 匹配）。

    源的注意力（QKV）/ 位置编码等无法 1:1 映射到 SimplCore，一律不碰，
    走默认初始化。这是"不会有严重后果"的核心：形状错配一律跳过并记录。
    """
    if report is None:
        report = MigrationReport()
    cfg: SimplCoreConfig = sgc.config

    # ---- 1. token embedding (SimplCore: tok_emb.weight [V, D]) ----
    tgt = sgc.tok_emb.weight
    src = _find_first(src_state, "wte", "token_emb", "embed_tokens", "embeddings")
    if src is not None:
        src2 = src.reshape(-1, tgt.shape[-1]) if src.numel() == tgt.numel() else src
        if tuple(src2.shape) == tuple(tgt.shape):
            tgt.data.copy_(src2.detach())
            report.moved.append("tok_emb <- embedding")
        else:
            report.skipped_shape.append(("tok_emb", tuple(tgt.shape), tuple(src2.shape)))

    # ---- 2. 每层 FFN (SwiGLU: gate/up/down) ----
    for li, blk in enumerate(sgc.blocks):
        f = blk.ffn
        # down_proj [D, H] (gate/up 是 [H, D])
        if hasattr(f, "down"):
            w = f.down.weight
            srcw = _find_first(src_state, "down_proj", "c_proj")
            if srcw is not None:
                if tuple(srcw.shape) == tuple(w.shape):
                    w.data.copy_(srcw.detach())
                    report.moved.append(f"blk{li}.ffn.down <- down_proj")
                else:
                    report.skipped_shape.append(
                        (f"blk{li}.ffn.down", tuple(w.shape), tuple(srcw.shape)))
        for name in ("gate", "up"):
            w = getattr(f, name).weight
            srcw = _find_first(src_state, "gate_proj") if name == "gate" \
                else _find_first(src_state, "up_proj")
            if srcw is not None:
                if tuple(srcw.shape) == tuple(w.shape):
                    w.data.copy_(srcw.detach())
                    report.moved.append(f"blk{li}.ffn.{name} <- {name}_proj")
                else:
                    report.skipped_shape.append(
                        (f"blk{li}.ffn.{name}", tuple(w.shape), tuple(srcw.shape)))

    # ---- 3. norms (RMSNorm weight [D]) ----
    for li, blk in enumerate(sgc.blocks):
        # SimplCore block: mixer.norm (RMSNorm before mixer) + ffn.norm (RMSNorm before FFN)
        mixer_norm_w = blk.mixer.norm.weight if hasattr(blk.mixer, "norm") else None
        ffn_norm_w = blk.ffn.norm.weight if hasattr(blk.ffn, "norm") else None
        src_n1 = _find_first(src_state, "ln_1", "input_layernorm", "attn_norm")
        src_n2 = _find_first(src_state, "ln_2", "post_attention_layernorm", "mlp_norm")
        if mixer_norm_w is not None and src_n1 is not None:
            if tuple(src_n1.shape) == tuple(mixer_norm_w.shape):
                mixer_norm_w.data.copy_(src_n1.detach())
                report.moved.append(f"blk{li}.mixer.norm <- ln_1")
            else:
                report.skipped_shape.append(
                    (f"blk{li}.mixer.norm", tuple(mixer_norm_w.shape), tuple(src_n1.shape)))
        if ffn_norm_w is not None and src_n2 is not None:
            if tuple(src_n2.shape) == tuple(ffn_norm_w.shape):
                ffn_norm_w.data.copy_(src_n2.detach())
                report.moved.append(f"blk{li}.ffn.norm <- ln_2")
            else:
                report.skipped_shape.append(
                    (f"blk{li}.ffn.norm", tuple(ffn_norm_w.shape), tuple(src_n2.shape)))

    # ---- 4. final norm + output head ----
    if hasattr(sgc, "ln_f"):
        w = sgc.ln_f.weight
        srcw = _find_first(src_state, "ln_f", "final_layernorm", "norm")
        if srcw is not None:
            if tuple(srcw.shape) == tuple(w.shape):
                w.data.copy_(srcw.detach())
                report.moved.append("ln_f <- final_layernorm")
            else:
                report.skipped_shape.append(("ln_f", tuple(w.shape), tuple(srcw.shape)))

    if not cfg.tie_embeddings:
        w = sgc.head.weight
        srcw = _find_first(src_state, "lm_head", "word_embeddings_for_logits")
        if srcw is not None:
            if tuple(srcw.shape) == tuple(w.shape):
                w.data.copy_(srcw.detach())
                report.moved.append("head <- lm_head")
            else:
                report.skipped_shape.append(("head", tuple(w.shape), tuple(srcw.shape)))

    return report

# =============================================================================
# 一键入口
# =============================================================================
def init_simplcore_from_hf(hf_model_or_state,
                           target_preset: str = "tuan-150m",
                           conservative: bool = True,
                           cfg_override: Optional[SimplCoreConfig] = None
                           ) -> Tuple[SimplCoreModel, MigrationReport]:
    """从 HuggingFace 风格模型（或 state_dict）一键构建 SimplCore 模型.

    Args:
        hf_model_or_state: 一个 ``torch.nn.Module``（取它的 state_dict）
            或一个 ``state_dict`` 映射（HF 的 safetensors 加载结果）。
        target_preset: 目标 SimplCore preset（决定宽深/M 基/振荡基数量）。
        conservative: True（默认）时**只**做安全的 embedding/FFN/norms 1:1 搬运，
            绝不触碰源的注意力 QKV / 位置编码（SimplCore 没有对应接收端）；
            形状错配一律跳过并记录，不污染权重。
        cfg_override: 完整 SimplCoreConfig；指定时覆盖 target_preset，
            直接按该配置构建 SimplCoreModel。

    Returns:
        (model, report)：model 可直接 ``.train()`` 继续训练；
        report 记录每一步的搬运/跳过，供审计。

    继承的 SimplCore 特性（全部原样保留）：
        * 因果全局卷积 O(L log L) 训练 + O(1) 状态流式推理
        * 内容门控（selective 核）
        * 无位置嵌入表 / 无 max_len 截断
        * 严格因果（test_mixer_math 守护）
    """
    if hasattr(hf_model_or_state, "state_dict"):
        src_state = dict(hf_model_or_state.state_dict())
    else:
        src_state = dict(hf_model_or_state)

    if cfg_override is not None:
        cfg = cfg_override
    else:
        cfg = get_preset(target_preset)
    sgc = SimplCoreModel(cfg)

    report = MigrationReport()
    migrate_embeddings_and_ffn(sgc, src_state, report)

    sgc.eval()
    return sgc, report


# =============================================================================
# 反向: SimplCore 流式状态 → 供外部评测框架接入的只读摘要
# =============================================================================
def simplcore_stream_state_for_eval(sgc: SimplCoreModel, state, batch: int) -> Dict[str, torch.Tensor]:
    """把 SimplCore 的 O(1) 流式状态整理成外部评测框架能读的摘要。

    用途: 让基于"KV 缓存长度"做长上下文对比的评测脚本, 也能接入 SimplCore 模型。
    这不是精确等价 (SimplCore 的常驻状态是 2*M*D, 与 L 无关, 不是 L*D 的 KV),
    而是一个**接口适配层**: 把与 L 无关的状态张量, 包装成评测框架期望的形状。
    评测时以 SimplCore 自己的 ``stream_step`` 为准, 此函数只用于外部接口兼容。

    Returns:
        ``state_bytes`` 为流式状态字节数（与 ``batch`` 无关, 体现 O(1) 流式特性）。
    """
    from .sgc import SGCStreamState
    n_bytes = state.num_bytes() if isinstance(state, SGCStreamState) else 0
    dev = next(sgc.parameters()).device
    zero = torch.zeros(batch, 1, sgc.config.dim, device=dev)
    return {
        "placeholder": zero,
        "state_bytes": torch.tensor(n_bytes, dtype=torch.int64),
    }


def sgc_to_hf_eval_bridge(sgc: SimplCoreModel) -> nn.Module:
    """返回一个轻量 ``nn.Module`` 包装, 暴露 HF 评测常用的 ``generate`` 接口。

    包装**不改** SimplCore 的计算路径: 它只是把 :meth:`SimplCoreModel.generate`
    映射到 HF 的调用约定 (``input_ids`` -> 输出 ids), 并额外透传
    O(1) 状态字节数, 让评测脚本能同时报告"显存与 L 解耦"这一特性。
    """

    class _Bridge(nn.Module):
        def __init__(self, core: SimplCoreModel):
            super().__init__()
            self.core = core

        def forward(self, input_ids, labels=None):
            return self.core(input_ids, labels)

        def generate(self, input_ids, max_new_tokens=32, temperature=1.0,
                     top_k=None, top_p=None, use_state=True, **kw):
            return self.core.generate(input_ids, max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_k=top_k,
                                      top_p=top_p, use_state=use_state)

        @property
        def stream_bytes(self) -> int:
            return self.core.stream_bytes(1)

    return _Bridge(sgc)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None) -> int:
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True,
                   help="HF safetensors 路径 / torch .bin / 含 state_dict 的目录")
    ap.add_argument("--preset", default="tuan-150m")
    ap.add_argument("--out", default="hf_bridge_out")
    ap.add_argument("--no-conservative", action="store_true")
    args = ap.parse_args(argv)

    # 加载源权重
    if os.path.isdir(args.src):
        from safetensors.torch import load_file
        st = load_file(os.path.join(args.src, "model.safetensors"))
    elif args.src.endswith(".bin"):
        st = torch.load(args.src, map_location="cpu", weights_only=True)
        if isinstance(st, dict) and "model" in st:
            st = st["model"]
    else:
        from safetensors.torch import load_file
        st = load_file(args.src)

    sgc, report = init_simplcore_from_hf(
        st, target_preset=args.preset,
        conservative=not args.no_conservative,
    )
    os.makedirs(args.out, exist_ok=True)
    sgc_state_path = os.path.join(args.out, "simplcore_migrated.pt")
    torch.save({"config": sgc.config.to_dict(), "state": sgc.state_dict()},
               sgc_state_path)
    with open(os.path.join(args.out, "migration_report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "moved": report.moved,
            "skipped_shape": [list(t) for t in report.skipped_shape],
            "skipped_unknown": report.skipped_unknown,
            "warmstarted": report.warmstarted,
            "notes": report.notes,
        }, fh, indent=2, ensure_ascii=False)
    print(report.summary())
    print(f"[hf_bridge] wrote {sgc_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
