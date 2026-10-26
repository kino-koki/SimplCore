# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""在 AMD Radeon 680M 上分域训练多个 100M~200M 专家小模型.

执行流程
--------
对语料中的每个领域 (classical / modern / science / philosophy ...):
  1. 用真实中文语料 (simplcore/data/corpus) 构造训练流;
  2. 以 ~100M/200M 预设配置构建 LatticeLM 专家;
  3. 用 simplcore.tensor 自研引擎做 next-unit 语言建模训练;
  4. 周期性保存 checkpoint 到 output/experts/.

之后用 run_unity_demo.py 把这些专家 Unity (路由+融合) 起来.

680M 显存/算力预算说明
-----------------------
Radeon 680M: 12CU RDNA2, ~2GB 共享内存, 无独立显存, 无 ROCm; 走 CPU/内存训练.
  * 100M 专家 (fp32 ~0.4GB, 优化器状态 ~1.2GB) 可单卡常驻;
  * 多个专家 **串行训练** (不是同时驻留), 故显存不随专家数线性增长;
  * 训练用小 batch + 梯度累积, 长序列按需分块.

用法
----
    python -m simplcore.unity.train_experts --preset 100m --steps 20000
    python -m simplcore.unity.train_experts --preset 200m --domains classical modern
"""

from __future__ import annotations

import argparse
import os
import sys

# 确保项目根在路径上 (无论从何处调用)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simplcore.data.corpus import RealCorpus
from simplcore.unity.expert_training import (
    train_domain_expert, build_expert_config,
)
from simplcore.architecture.lattice_model import parameter_count, LatticeConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="SimplUnity 分域专家训练 (680M)")
    ap.add_argument("--preset", choices=["100m", "200m"], default="100m")
    ap.add_argument("--steps", type=int, default=20000,
                    help="每个专家的训练步数 (680M 上 100M 模型约需数小时/专家)")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--domains", nargs="*", default=None,
                    help="指定领域 (默认训练全部)")
    ap.add_argument("--output_dir", default="output/experts")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    corpus = RealCorpus().load(verbose=True)
    domains = args.domains or corpus.list_domains()
    print(f"\n将训练 {len(domains)} 个专家: {domains} (预设 {args.preset})")

    # 打印估算规模
    um_size = len(corpus.make_unit_map())
    cfg = build_expert_config(um_size, preset=args.preset, seq_len=args.seq_len)
    print(f"单专家估算参数量: {parameter_count(cfg)/1e6:.1f}M")

    for dom in domains:
        print(f"\n{'='*60}\n  训练专家: {dom}\n{'='*60}")
        try:
            res = train_domain_expert(
                domain=dom, corpus=corpus, preset=args.preset,
                seq_len=args.seq_len, total_steps=args.steps,
                batch_size=args.batch_size, output_dir=args.output_dir,
                seed=args.seed,
            )
            print(f"  ✓ {dom} 完成: 末损 {res['final_loss']:.4f}, "
                  f"耗时 {res['elapsed_s']:.0f}s")
        except Exception as e:
            print(f"  ✗ {dom} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n全部专家训练结束. checkpoints 位于:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
