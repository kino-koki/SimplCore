# -*- coding: utf-8 -*-
"""
青小团 · 12B Unity 专家训练入口 (Train 12B Unity Experts)
=========================================================
统一入口：串行训练 12 个 ~0.966B 分领域专家，每个训完存盘（HuggingFace 格式），
再拼成 11.6B 缝合体。支持：
    - 逐专家训练（一次一个进显存，适配 5090 32GB）
    - 容错续训（崩了从最后一个 step 包继续，不重来 12万步）
    - fp16 混合精度 + 梯度检查点（省显存）
    - 可选 CPU 占用限制（≤5%，仅建议用于下载/预处理，真训练慎用）
    - 训练完自动导出到 HuggingFace（防海马云实例关机丢失）

用法：
    # 在 5090 云电脑上（推荐）：
    python train_12b_experts.py --backend cuda --steps 120000 --export-hf \
        --hf-repo your-name/qingxiaotuan-12b --hf-token hf_xxx

    # 在 680M 本地（仅能训小专家，演示用，--preset 100m）：
    python train_12b_experts.py --backend auto --preset 100m --steps 2000

    # 续训（崩了接着跑）：
    python train_12b_experts.py --backend cuda --steps 120000 --resume
"""
from __future__ import annotations

import os
import sys
import argparse
import glob

# 确保项目根可导入
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import simplcore.tensor as torch
from simplcore.architecture.lattice_model import LatticeLM, LatticeConfig
from simplcore.unity.expert_training import (
    ExpertTrainConfig, ExpertTrainer, DomainDataset
)
from simplcore.data.corpus import RealCorpus
from simplcore.unity.unity_12b_plan import (
    Unity12BConfig, EXPERT_DOMAINS, train_all_experts
)


def build_expert_model(cfg: Unity12BConfig, V: int, preset: str,
                       mixed_precision: bool, use_checkpoint: bool) -> LatticeLM:
    if preset == "100m":
        mcfg = LatticeConfig(vocab_size=V, hidden_size=768, num_layers=12,
                             num_bands=8, kernel_size=31, num_tables=64,
                             slots_per_table=128, ffn_ratio=4,
                             max_seq_len=1024)
    elif preset == "200m":
        mcfg = LatticeConfig(vocab_size=V, hidden_size=1024, num_layers=10,
                             num_bands=8, kernel_size=31, num_tables=64,
                             slots_per_table=128, ffn_ratio=4,
                             max_seq_len=1024)
    else:  # 0.966B 专家（12B 缝合体用）
        mcfg = LatticeConfig(
            vocab_size=V, hidden_size=cfg.expert_hidden,
            num_layers=cfg.expert_layers, num_bands=cfg.num_bands,
            kernel_size=cfg.kernel_size, num_tables=cfg.num_tables,
            slots_per_table=cfg.slots_per_table, ffn_ratio=cfg.ffn_ratio,
            max_seq_len=cfg.max_seq_len,
        )
    dtype = torch.float16 if mixed_precision else None
    return LatticeLM(mcfg, use_checkpoint=use_checkpoint, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "numpy", "cuda", "opencl"])
    ap.add_argument("--preset", default="966m",
                    choices=["100m", "200m", "966m"])
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--mixed-precision", action="store_true", default=True)
    ap.add_argument("--no-mixed-precision", dest="mixed_precision",
                    action="store_false")
    ap.add_argument("--checkpoint", action="store_true", default=True,
                    help="梯度检查点（省显存）")
    ap.add_argument("--no-checkpoint", dest="checkpoint",
                    action="store_false")
    ap.add_argument("--resume", action="store_true",
                    help="从最后一个 step 包续训")
    ap.add_argument("--max-cpu-percent", type=float, default=None,
                    help="CPU 占用上限（如 5.0）。仅建议下载/预处理用")
    ap.add_argument("--corpus-root", default=r"D:/新建文件夹/qingxiaotuan_corpus")
    ap.add_argument("--export-hf", action="store_true",
                    help="训完导出到 HuggingFace")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_TOKEN", ""))
    ap.add_argument("--domain", default=None,
                    help="只训单个领域（调试用）")
    args = ap.parse_args()

    torch.set_backend(args.backend)
    print(f"[入口] 后端: {torch.get_backend()}, 设备: {torch.get_device()}")
    if torch.is_gpu_backend():
        print(f"[入口] GPU: {torch.gpu_name()}, 显存: {torch.gpu_total_gb():.1f}GB")

    cfg = Unity12BConfig()
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    # CPU 限制（可选）
    limiter = None
    if args.max_cpu_percent:
        sys.path.insert(0, os.path.join(args.corpus_root, "scripts"))
        from cpu_throttle import CPULimiter
        limiter = CPULimiter(max_cpu_percent=args.max_cpu_percent,
                             bind_cores=1)
        limiter.apply()
        print(f"[入口] CPU 限制已启用: {limiter.report()}")

    # 语料：优先 D 盘大语料，回退项目内演示小语料
    corpus_d = None
    try:
        corpus_d = RealCorpus(args.corpus_root).load(verbose=False)
    except Exception:
        corpus_d = None
    corpus_default = RealCorpus().load(verbose=False)  # 项目内小语料

    def get_texts(domain_tag, fallback_dom=None):
        txts = []
        for tag in [domain_tag, fallback_dom]:
            if not tag:
                continue
            if corpus_d is not None:
                txts = corpus_d.get_domain_texts(tag)
            if not txts:
                txts = corpus_default.get_domain_texts(tag)
            if txts:
                break
        return txts

    # unit_map 用默认（含演示语料字符），D 盘语料字符是其超集，编码兼容
    unit_map = corpus_default.make_unit_map()
    V = len(unit_map)
    out_dir = os.path.join(args.corpus_root, "models", "experts_12b")
    os.makedirs(out_dir, exist_ok=True)

    domains = [d for d in EXPERT_DOMAINS if (args.domain is None or d["id"] == args.domain)]

    for d in domains:
        dom = d["id"]
        print(f"\n{'='*64}\n  训练专家 [{dom}] ({d['desc']})\n{'='*64}")
        model = build_expert_model(cfg, V, args.preset,
                                    args.mixed_precision, args.checkpoint)
        texts = get_texts(d["corpus"], fallback_dom=dom) or \
            get_texts("own_text", fallback_dom=dom)
        if not texts:
            print(f"  [跳过] 领域 {dom} 无语料（需先下载 {d['corpus']}）")
            del model
            continue
        ds = DomainDataset(texts, unit_map, seq_len=cfg.max_seq_len)
        ecfg = ExpertTrainConfig(
            domain=dom, seq_len=cfg.max_seq_len, batch_size=args.batch_size,
            total_steps=args.steps, save_interval=args.save_interval,
            use_checkpoint=args.checkpoint, mixed_precision=args.mixed_precision,
            output_dir=out_dir,
        )
        # 续训检测
        ckpts = sorted(glob.glob(os.path.join(out_dir, f"expert_{dom}_step*.pt")))
        if args.resume and ckpts:
            print(f"  [续训] 从 {os.path.basename(ckpts[-1])} 恢复...")
            trainer = ExpertTrainer.resume(ecfg, model, ds, ckpts[-1])
            trainer.run(resume=True)
        else:
            trainer = ExpertTrainer(ecfg, model, ds)
            trainer.run()
        trainer.save(out_dir, tag="final")
        print(f"  [完成] {dom} 已存: {os.path.join(out_dir, f'expert_{dom}_final.pt')}")
        del model, trainer, ds
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 导出 HF
    if args.export_hf:
        if not args.hf_repo or not args.hf_token:
            print("⚠️ 导出 HF 需 --hf-repo 和 --hf-token，跳过上传（仅本地导出）。")
            args.hf_repo = args.hf_repo or "local/qingxiaotuan-12b"
            args.hf_token = args.hf_token or ""
        sys.path.insert(0, os.path.join(args.corpus_root, "scripts"))
        from export_hf import export_batch
        res = export_batch(out_dir, args.hf_repo, args.hf_token,
                           args.corpus_root, push=bool(args.hf_token))
        print(f"\n[入口] 导出 {len(res)} 个专家到 HF: {args.hf_repo}")

    print("\n[入口] 全部专家训练完成。")


if __name__ == "__main__":
    main()
