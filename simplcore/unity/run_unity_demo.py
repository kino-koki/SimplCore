# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 端到端演示: 训练微型专家 -> Unity 集成 -> 路由续写.

这个脚本把整个 "分域训练专家 + SimplUnity 集成" 流程串起来, 既能在 680M 上
跑真实规模 (--preset 100m/200m --steps 大), 也能在本演示用极小配置快速验证
训练有效性与路由正确性 (默认 --sandbox_demo).

流程
----
  1. 加载真实中文语料 (classical / modern / science / philosophy);
  2. 对每个领域训练一个 LatticeLM 专家 (next-unit 语言建模);
  3. 把专家装入 ExpertEnsemble, 接入 UnityOrchestrator 消息总线;
  4. 对若干提示做领域路由 + 续写, 验证 "Unity" 生效.

说明: 本沙箱有 CPU 时间上限, 默认 sandbox_demo 仅训练极少步数以验证流程;
     在 680M 设备运行请使用 --preset 100m --steps 20000 等真实配置.
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import simplcore.tensor as torch
from simplcore.data.corpus import RealCorpus, UnitMap
from simplcore.architecture.lattice_model import LatticeLM
from simplcore.unity.expert_training import (
    ExpertTrainer, ExpertTrainConfig, DomainDataset, build_expert_config,
)
from simplcore.unity.expert_ensemble import (
    ExpertEnsemble, ExpertRouter, assemble_unity,
)


def _tiny_config(vocab_size: int):
    """沙箱演示用极小配置 (可在时间上限内完成训练)."""
    from simplcore.architecture.lattice_model import LatticeConfig
    return LatticeConfig(
        vocab_size=vocab_size, hidden_size=48, num_layers=2, num_bands=3,
        kernel_size=11, num_tables=6, slots_per_table=12, ffn_ratio=2,
        max_seq_len=96,
    )


def train_mini_experts(corpus: RealCorpus, domains, steps: int, seq_len: int,
                       output_dir: str, seed: int):
    """训练 (或加载已存在的) 微型专家, 返回 [(domain, model, unit_map, vocab)]."""
    um = corpus.make_unit_map()
    experts = []
    for dom in domains:
        texts = corpus.get_domain_texts(dom)
        if not texts:
            continue
        cfg = _tiny_config(len(um))
        model = LatticeLM(cfg)
        ds = DomainDataset(texts, um, seq_len=seq_len)
        ecfg = ExpertTrainConfig(
            domain=dom, seq_len=seq_len, batch_size=4,
            total_steps=steps, save_interval=10**9,
            output_dir=output_dir, seed=seed,
        )
        trainer = ExpertTrainer(ecfg, model, ds)
        trainer.run(rng_seed=seed)
        # 领域字符集 (用于路由)
        vocab = set()
        for t in texts:
            vocab.update(t)
        experts.append((dom, model, um, vocab))
        print(f"  ✓ 专家 '{dom}' 训练完成 (末损 {trainer.loss_history[-1]:.3f})")
    return experts


def main() -> None:
    ap = argparse.ArgumentParser(description="SimplUnity 端到端演示")
    ap.add_argument("--sandbox_demo", action="store_true", default=True,
                    help="极小配置快速验证 (默认开启)")
    ap.add_argument("--steps", type=int, default=30,
                    help="每个专家训练步数 (沙箱建议 <=40 以规避 CPU 时间上限)")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--output_dir", default="output/experts")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print("=" * 64)
    print("  SimplUnity · 分域专家训练 + 多专家集成 演示")
    print("=" * 64)

    corpus = RealCorpus().load(verbose=True)
    domains = args.domains or ["classical", "science", "philosophy"]
    domains = [d for d in domains if d in corpus.list_domains()]
    if not domains:
        domains = corpus.list_domains()[:2]

    print(f"\n[1/3] 训练 {len(domains)} 个领域的微型专家 (真实中文语料) ...")
    os.makedirs(args.output_dir, exist_ok=True)
    experts = train_mini_experts(
        corpus, domains, steps=args.steps, seq_len=args.seq_len,
        output_dir=args.output_dir, seed=args.seed,
    )

    print(f"\n[2/3] 把 {len(experts)} 个专家 Unity 起来 (路由+融合) ...")
    router = ExpertRouter()
    ensemble = ExpertEnsemble(router=router)
    for dom, model, um, vocab in experts:
        ensemble.add_expert(dom, model, um, domain_vocab=vocab)
    uni = assemble_unity([(d, m, u, v) for d, m, u, v in experts],
                         router=router)
    print("  SimplUnity 总线已注册单元:", list(uni.units.keys()))

    print(f"\n[3/3] 路由 + 续写验证 ...")
    prompts = ["道可道", "太阳系由太阳和", "北冥有鱼", "春盼望着"]
    for p in prompts:
        best, scores = ensemble.route(p)
        routed = ensemble.slots[best].domain
        # 续写 (沙箱模型极小, 续写仅作示意)
        gen = ensemble.generate(p, max_len=16)
        print(f"  提示: {p!r:18} -> 路由【{routed}】 续写: "
              f"{gen['generated'][:len(p)+12]!r}")

    print("\n演示完成. 在 680M 设备上请用:")
    print("  python -m simplcore.unity.train_experts --preset 100m --steps 20000")
    print("  python -m simplcore.unity.run_unity_demo --steps 0  # 仅加载已训专家做集成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
