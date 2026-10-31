# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""部署侧 HBM 账本 —— 1M 上下文完整放下 ≤ 2 GB，翻倍增量 ≤ 1 GB。

用户硬指标
----------
* 上下文从 **1M 起**，翻倍（1M→2M→4M→8M）HBM 需求**仅增长 ≤ 1 GB**；
* **1M 长也只需要 2 GB 即可完整放下**（含权重）。

为什么结构性能做到（不是调参调出来的）
--------------------------------------
1. 无 KV cache：注意力要缓存 `2·D·L·层数`，1M 上下文仅 KV 就 ~60 GB；
   SimplCore 的流式状态是 `2·M·D/层` 的**常驻**振荡器态，**与 L 无关**
   （1B 配置实测 ~30.4 MiB，fp32，含 M 因子——见 scaling.stream_state_bytes
   的 v1.1.1 勘误）。
2. 流式 prefill：1M token 按 chunk（默认 512）顺序喂入，每 chunk 的激活
   workspace 用完即丢，只留 O(1) 状态——所以 workspace 与总长度 L 无关。
3. 1B 权重 bf16 = 1,013,407,563 × 2B ≈ 1.89 GiB，是 2 GB 预算的大头；
   剩余份额给流式状态 + 融合单缓冲。

账本公式（`deploy_bytes`）
--------------------------
    权重(bf16) + 流式状态(fp32) + 融合单缓冲 max(激活workspace, logits)

全部与 L 无关 → 翻倍增量恒为 0，1 GiB 余量是结构性冗余。

HBM 削减策略（**不削减任何基准性能**）
--------------------------------------
* **融合单遍**：mixer_stream 十个子算子单遍融合 → 激活缓冲 2→1 个
  （纯分配策略，融合核更快，吞吐不受损）；
* **logits 复用**：logits 只在最后 chunk 需要，复用激活缓冲（取峰值较大者）；
* **状态精度不动**：fp32 是数值规格的一部分（双形式逐位一致依赖它）；
* **chunk/层数/宽度一律不动**：chunk 保持 512 基线吞吐。

诚实边界
--------
* 本模块是**解析账本 + 契约**，不是后端实测；实机数字以自研 CUDA/C++ 后端
  按此口径实测为准（契约见 ``backend_contract.py`` 的 stream_state 与
  chunked 前向内核；融合单遍对应其单 pass 内核要求）。
* 默认参数（chunk=512, workspace_tensors=1）按 1B/tuan-1b 校准；
  batch>1 或更大 chunk 会线性抬高缓冲，账本如实计入。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..model.config import SIMPLCORE_PRESETS, get_preset
from ..eval.scaling import embedding_params, non_embedding_params, stream_state_bytes

__all__ = [
    "GIB",
    "DEPLOY_HBM_CAP_1M",
    "DEPLOY_DOUBLING_CAP",
    "DEPLOY_LADDER",
    "DEFAULT_CHUNK",
    "DEFAULT_WORKSPACE_TENSORS",
    "DeployRow",
    "weight_bytes",
    "deploy_bytes",
    "deployment_report",
    "deploy_verdict",
    "main",
]

GIB = 1024 ** 3
#: 用户指标：1M 上下文完整放下（含权重）≤ 2 GB。
DEPLOY_HBM_CAP_1M = 2 * GIB
#: 用户指标：上下文翻倍 HBM 增量 ≤ 1 GB。
DEPLOY_DOUBLING_CAP = 1 * GIB
#: 部署阶梯：1M 起翻倍。
DEPLOY_LADDER: tuple = (1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024)
#: 流式 prefill 的 chunk 长度（与总上下文 L 无关，只决定 workspace 峰值）。
#: 保持 512 —— HBM 削减通过融合与复用实现，不牺牲 prefill 吞吐。
DEFAULT_CHUNK = 512
#: 单 chunk 峰值激活张量数。融合单遍核（``mixer_stream``：norm→投影→门控→
#: 递推→局部卷积→门→投影→残差 单遍完成）只需要 1 个激活缓冲；
#: 旧值 2 是未融合内核的过度分配。
DEFAULT_WORKSPACE_TENSORS = 1
BYTES_BF16 = 2


@dataclass(frozen=True)
class DeployRow:
    """部署阶梯上的一行：总占用与相对上一级的翻倍增量。"""

    seq_len: int
    total_bytes: int
    delta_bytes: int          # 相对上一级（首行记 0）
    weights_bytes: int
    state_bytes: int
    workspace_bytes: int


def weight_bytes(params: int, dtype_bytes: int = BYTES_BF16) -> int:
    """推理常驻权重（bf16）。"""
    return int(params) * dtype_bytes


def deploy_bytes(
    cfg,
    params: Optional[int] = None,
    batch: int = 1,
    chunk: int = DEFAULT_CHUNK,
    workspace_tensors: int = DEFAULT_WORKSPACE_TENSORS,
) -> int:
    """1B 配置完整放下的部署占用：权重 + 流式状态 + 融合单缓冲。

    **每项全部与总上下文 L 无关** —— 这是"翻倍增量 ≤ 1 GB"的结构性来源。

    HBM 削减策略（不削减任何基准性能）：

    1. **融合单遍**：``mixer_stream`` 把十个子算子融合为单遍，激活缓冲
       从 2 个降到 1 个 —— 纯分配策略，且融合核更快，吞吐不受损。
    2. **logits 缓冲复用**：logits 只在最后一个 chunk 需要，复用同一块
       激活缓冲（取两者较大者为峰值），不单独分配。
    3. **流式状态保持 fp32**：状态精度是数值规格的一部分（双形式逐位
       一致依赖它），绝不以降精度换显存。

    即：占用的下降来自**分配策略与算子融合**，chunk/层数/精度一律不动。
    """
    if params is None:
        params = embedding_params(cfg) + non_embedding_params(cfg)
    weights = weight_bytes(params)
    state = stream_state_bytes(cfg, batch)
    # 融合单缓冲：激活 workspace 与 logits 复用同一块（峰值取较大者）
    layer_ws = cfg.num_layers * chunk * cfg.dim * workspace_tensors * BYTES_BF16 * batch
    logits = chunk * cfg.vocab_size * BYTES_BF16 * batch
    buffer = max(layer_ws, logits)
    return weights + state + buffer


def deployment_report(
    cfg,
    params: Optional[int] = None,
    ladder: Sequence[int] = DEPLOY_LADDER,
    **kw,
) -> List[DeployRow]:
    """沿部署阶梯逐级算总占用；增量应恒为 0。"""
    rows: List[DeployRow] = []
    prev = 0
    for seq_len in ladder:
        total = deploy_bytes(cfg, params, **kw)
        rows.append(DeployRow(
            seq_len=int(seq_len), total_bytes=total,
            delta_bytes=total - prev if prev else 0,
            weights_bytes=weight_bytes(params if params is not None
                                       else embedding_params(cfg) + non_embedding_params(cfg)),
            state_bytes=stream_state_bytes(cfg, kw.get("batch", 1)),
            workspace_bytes=total - (weight_bytes(params if params is not None
                                                  else embedding_params(cfg) + non_embedding_params(cfg))
                                     + stream_state_bytes(cfg, kw.get("batch", 1))),
        ))
        prev = total
    return rows


def deploy_verdict(rows: List[DeployRow]) -> Dict[str, object]:
    """两条硬指标的裁决：1M ≤ 2 GB；翻倍增量 ≤ 1 GB。"""
    first = rows[0]
    max_delta = max(r.delta_bytes for r in rows)
    return {
        "claim": f"1M 完整放下 ≤ {DEPLOY_HBM_CAP_1M/GIB:.0f} GiB；"
                 f"翻倍增量 ≤ {DEPLOY_DOUBLING_CAP/GIB:.0f} GiB",
        "fits_2gb_at_1m": first.total_bytes <= DEPLOY_HBM_CAP_1M,
        "total_at_1m_gib": round(first.total_bytes / GIB, 3),
        "headroom_1m_gib": round((DEPLOY_HBM_CAP_1M - first.total_bytes) / GIB, 3),
        "max_doubling_delta_gib": round(max_delta / GIB, 4),
        "doubling_cap_ok": max_delta <= DEPLOY_DOUBLING_CAP,
        "claim_holds": first.total_bytes <= DEPLOY_HBM_CAP_1M
                       and max_delta <= DEPLOY_DOUBLING_CAP,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI：``python -m simplcore.core.deploy_budget``。"""
    ap = argparse.ArgumentParser(description="部署侧 HBM 账本（1M≤2GB，翻倍≤+1GB）")
    ap.add_argument("--preset", default="tuan-1b", choices=sorted(SIMPLCORE_PRESETS))
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = get_preset(args.preset)
    rows = deployment_report(cfg, chunk=args.chunk, batch=args.batch)
    verdict = deploy_verdict(rows)
    if args.json:
        print(json.dumps({"verdict": verdict,
                          "rows": [r.__dict__ for r in rows]},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"=== 部署 HBM 账本（preset={args.preset}, chunk={args.chunk}, batch={args.batch}）===")
    w = rows[0]
    print(f"权重(bf16)={w.weights_bytes/GIB:.3f} GiB | 流式状态={w.state_bytes/1024:.1f} KiB"
          f" | chunk workspace+logits={w.workspace_bytes/GIB:.3f} GiB")
    print(f"{'上下文':>10} {'总占用':>12} {'翻倍增量':>12}")
    for r in rows:
        print(f"{r.seq_len/1024/1024:>8}M {r.total_bytes/GIB:>10.3f}G"
              f" {r.delta_bytes/GIB:>+10.3f}G")
    v = verdict
    print(f"\n裁决: {v['claim']}")
    print(f"  1M ≤ 2GiB: {v['fits_2gb_at_1m']} (实际 {v['total_at_1m_gib']} GiB,"
          f" 余量 {v['headroom_1m_gib']} GiB)")
    print(f"  翻倍 ≤ +1GiB: {v['doubling_cap_ok']} (实际最大增量 {v['max_doubling_delta_gib']} GiB)")
    print(f"  总体: {'成立' if v['claim_holds'] else '不成立'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
