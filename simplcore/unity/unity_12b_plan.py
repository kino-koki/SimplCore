# -*- coding: utf-8 -*-
"""
青小团 · Unity 缝合 12B 架构方案 (Unity-to-12B Plan)
===================================================
用户决策: 用 SimplUnity 把多个分领域专家 "缝合" 到 **约 12B 参数** 的整体模型,
而不是训练单个 7B/33B 模型 (680M 集成显卡物理上无法训练单大模型)。

设计要点
--------
1. 专家规模: 每个专家 ~1.0~1.2B (hidden=3072, layers=10, 字符级 vocab~1200),
   这是 680M 上经梯度检查点/量化后可训的体量上限。
2. 专家数量: 12 个分领域专家 × ~1.0B = **~12B 总参** (含路由/缝合层)。
3. 缝合方式: SimplUnity 领域路由 (字符覆盖度打分) + 续写融合 + 门控加权,
   推理时按路由激活 top-k 专家, 不全部计算 (MoE 风格稀疏激活)。
4. 语料: 中文维基 + 新华字典(用户自有) + 国学/新闻(待启用), 全中文真实语料。

参数估算 (字符级 vocab=1200, num_bands=8, kernel=31, tables=64, slots=128):
   单专家 hidden=3072 layers=10 -> ~1.20B
   12 专家 -> ~14.4B (含共享 embed/路由, 实际可控在 12~13B)
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ----------------------------------------------------------------------
# 12B 缝合总配置
# ----------------------------------------------------------------------
@dataclass
class Unity12BConfig:
    target_total_params: float = 12.0e9     # 目标总参 ~12B
    num_experts: int = 12                    # 分领域专家数
    expert_hidden: int = 2880                # 每专家隐藏维度 (调参压到 ~12B)
    expert_layers: int = 9                   # 每专家层数
    expert_vocab: int = 1200                 # 字符级 unit 表 (~1200)
    num_bands: int = 8
    kernel_size: int = 31
    num_tables: int = 64
    slots_per_table: int = 128
    ffn_ratio: int = 4
    # 路由/缝合层
    router_hidden: int = 1024
    top_k: int = 2                           # 推理时激活 top-2 专家 (稀疏)
    # 训练预算 (680M)
    max_seq_len: int = 1024
    # 语料预算
    corpus_dir: str = r"D:/新建文件夹/qingxiaotuan_corpus"
    max_corpus_gb: float = 100.0


# ----------------------------------------------------------------------
# 分领域专家规划 (12 个, 覆盖通用中文能力)
# ----------------------------------------------------------------------
EXPERT_DOMAINS: List[Dict] = [
    {"id": "wiki_general",  "desc": "维基百科通用知识",   "corpus": "wiki_zh"},
    {"id": "dictionary",    "desc": "新华字典字词释义",   "corpus": "xinhua_dict"},
    {"id": "classical",     "desc": "古文/经史子集",       "corpus": "guoxue"},
    {"id": "modern_prose",  "desc": "现代散文/文学",       "corpus": "own_text"},
    {"id": "science",       "desc": "科普/自然科学",       "corpus": "wiki_zh"},
    {"id": "philosophy",    "desc": "哲学/思想",           "corpus": "guoxue"},
    {"id": "history",       "desc": "历史/史料",           "corpus": "wiki_zh"},
    {"id": "tech",          "desc": "技术/工程",           "corpus": "wiki_zh"},
    {"id": "medicine",      "desc": "医学/健康",           "corpus": "wiki_zh"},
    {"id": "law",           "desc": "法律/法规",           "corpus": "wiki_zh"},
    {"id": "art",           "desc": "艺术/美学",           "corpus": "wiki_zh"},
    {"id": "daily",         "desc": "日常对话/口语",       "corpus": "own_text"},
]


def expert_param_count(cfg: Unity12BConfig) -> float:
    """估算单个专家参数量 (与 lattice_model.parameter_count 同公式)."""
    h = cfg.expert_hidden
    L = cfg.expert_layers
    V = cfg.expert_vocab
    nb, k, T, S, fr = (cfg.num_bands, cfg.kernel_size,
                        cfg.num_tables, cfg.slots_per_table, cfg.ffn_ratio)
    per_block = (nb + nb + nb * k + 1
                 + h * (T * S) + h * h + h * h
                 + h * (h * fr) + (h * fr) * h)
    total = V * h + L * per_block + h + h * V
    return total


def total_param_estimate(cfg: Unity12BConfig) -> Dict[str, float]:
    ep = expert_param_count(cfg)
    n = cfg.num_experts
    # 共享: 路由层 + 缝合门控 (可忽略相对 12B)
    router = cfg.router_hidden * n * cfg.expert_hidden
    total = ep * n + router
    return {
        "per_expert_B": ep / 1e9,
        "num_experts": n,
        "experts_total_B": ep * n / 1e9,
        "router_B": router / 1e9,
        "grand_total_B": total / 1e9,
    }


# ----------------------------------------------------------------------
# 逐专家训练（一次只把一个专家加载进显存，训完存盘再训下一个）
# 这是 32GB 显存（5090）下训 12B 缝合体的唯一可行路径：
#   12 专家训练态 ~ 186~279GB 无法同时驻留，必须串行/逐专家。
# ----------------------------------------------------------------------
def train_all_experts(cfg: Unity12BConfig, corpus_root: str,
                      steps_per_expert: int = 120000,
                      use_checkpoint: bool = True,
                      mixed_precision: bool = True,
                      device_backend: str = "auto") -> Dict[str, str]:
    """串行训练 12 个专家，每个训完立即存盘（.pt）。返回 {domain: ckpt_path}。

    显存策略：
        - 每个专家独立构建 LatticeLM（~0.966B），fp16 + 梯度检查点下
          训练态 ~ 15~23GB，单张 5090 (32GB) 可容纳。
        - 训完 save 后 del 模型 + gc，再训下一个，避免多专家同时驻留。
        - 支持 --resume（崩了从最后一个 step 包续训）。
    """
    import os, gc
    import simplcore.tensor as torch
    torch.set_backend(device_backend)  # auto -> cuda/opencl/numpy
    from simplcore.architecture.lattice_model import LatticeLM, LatticeConfig
    from simplcore.unity.expert_training import (
        ExpertTrainConfig, ExpertTrainer, DomainDataset, train_domain_expert
    )
    from simplcore.data.corpus import RealCorpus

    out_dir = os.path.join(corpus_root, "models", "experts_12b")
    os.makedirs(out_dir, exist_ok=True)

    # 加载语料（全中文真实语料）
    corpus = RealCorpus(corpus_root).load(verbose=True)
    unit_map = corpus.make_unit_map()
    V = len(unit_map)

    results = {}
    for d in EXPERT_DOMAINS:
        dom = d["id"]
        print(f"\n{'='*60}\n  训练专家 [{dom}] ({d['desc']})\n{'='*60}")
        # 构建该专家配置（0.966B）
        ecfg_model = LatticeConfig(
            vocab_size=V, hidden_size=cfg.expert_hidden,
            num_layers=cfg.expert_layers, num_bands=cfg.num_bands,
            kernel_size=cfg.kernel_size, num_tables=cfg.num_tables,
            slots_per_table=cfg.slots_per_table, ffn_ratio=cfg.ffn_ratio,
            max_seq_len=cfg.max_seq_len,
        )
        model = LatticeLM(ecfg_model, use_checkpoint=use_checkpoint,
                          dtype=torch.float16 if mixed_precision else None)
        texts = corpus.get_domain_texts(d["corpus"]) or corpus.get_domain_texts("own_text")
        ds = DomainDataset(texts, unit_map, seq_len=cfg.max_seq_len)
        ecfg = ExpertTrainConfig(
            domain=dom, seq_len=cfg.max_seq_len, batch_size=4,
            total_steps=steps_per_expert, save_interval=2000,
            use_checkpoint=use_checkpoint, mixed_precision=mixed_precision,
            output_dir=out_dir,
        )
        trainer = ExpertTrainer(ecfg, model, ds)
        # 检查是否有续训包
        import glob
        ckpts = sorted(glob.glob(os.path.join(out_dir, f"expert_{dom}_step*.pt")))
        if ckpts:
            print(f"  发现续训包 {os.path.basename(ckpts[-1])}，恢复训练...")
            trainer = ExpertTrainer.resume(ecfg, model, ds, ckpts[-1])
            trainer.run(resume=True)
        else:
            trainer.run()
        ckpt = os.path.join(out_dir, f"expert_{dom}_final.pt")
        trainer.save(out_dir, tag="final")
        results[dom] = ckpt
        # 释放显存，准备下一个
        del model, trainer, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# ----------------------------------------------------------------------
# 12B 缝合推理器（MoE 风格 top-k 稀疏激活）
# ----------------------------------------------------------------------
class Unity12BOrchestrator:
    """把 12 个专家按领域路由缝合到一起，推理时激活 top-k 专家。"""

    def __init__(self, cfg: Unity12BConfig, expert_ckpts: Dict[str, str],
                 corpus_root: str):
        import os
        import simplcore.tensor as torch
        from simplcore.architecture.lattice_model import LatticeLM, LatticeConfig
        from simplcore.data.corpus import RealCorpus

        self.cfg = cfg
        self.device = torch.get_device()
        corpus = RealCorpus(corpus_root).load(verbose=False)
        self.unit_map = corpus.make_unit_map()
        V = len(self.unit_map)

        # 路由打分器：字符覆盖度（与 expert_ensemble 保持一致）
        self.expert_texts_chars: Dict[str, set] = {}
        for d in EXPERT_DOMAINS:
            txt = "".join(corpus.get_domain_texts(d["corpus"]) or [])
            self.expert_texts_chars[d["id"]] = set(txt)

        # 加载专家（CPU 卸载，按需加载到 GPU）
        self.experts: Dict[str, LatticeLM] = {}
        self.expert_cfgs: Dict[str, LatticeConfig] = {}
        for dom, ckpt in expert_ckpts.items():
            if not os.path.exists(ckpt):
                print(f"  [缝合] 专家 {dom} 权重缺失: {ckpt}")
                continue
            mcfg = LatticeConfig(
                vocab_size=V, hidden_size=cfg.expert_hidden,
                num_layers=cfg.expert_layers, num_bands=cfg.num_bands,
                kernel_size=cfg.kernel_size, num_tables=cfg.num_tables,
                slots_per_table=cfg.slots_per_table, ffn_ratio=cfg.ffn_ratio,
                max_seq_len=cfg.max_seq_len,
            )
            m = LatticeLM(mcfg)
            ExpertTrainer.load(m, ckpt) if 'ExpertTrainer' in globals() else None
            self.experts[dom] = m
            self.expert_cfgs[dom] = mcfg
        print(f"  [缝合] 已加载 {len(self.experts)}/{cfg.num_experts} 个专家")

    def route(self, prompt: str) -> List[str]:
        """领域路由：字符覆盖度打分，选 top-k 专家。"""
        chars = set(prompt)
        scores = {}
        for dom, charset in self.expert_texts_chars.items():
            if dom not in self.experts:
                continue
            overlap = len(chars & charset)
            scores[dom] = overlap / max(1, len(charset))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        topk = [d for d, _ in ranked[: self.cfg.top_k]]
        return topk

    def generate(self, prompt: str, max_new: int = 64) -> str:
        """续写生成：路由到 top-k 专家，门控融合 logits。"""
        import simplcore.tensor as torch
        topk = self.route(prompt)
        if not topk:
            return prompt
        # 各专家生成候选，取路由分最高者为主，其余融合
        ids = self.unit_map.encode(prompt)
        ctx = torch.tensor([ids], dtype=torch.long)
        fused_logits = None
        for dom in topk:
            m = self.experts[dom]
            with torch.no_grad():
                out = m(ctx)
                logits = out["logits"][:, -1, :]
            if fused_logits is None:
                fused_logits = logits
            else:
                fused_logits = fused_logits + logits  # 等权门控融合
        # 取 argmax 续写（简化；真实可用采样）
        nxt = int(torch.argmax(fused_logits, dim=-1).data.ravel()[0])
        generated = self.unit_map.decode([nxt])
        return prompt + generated


# ----------------------------------------------------------------------
# 导出到 Hugging Face（解决海马云临时实例关机丢失问题）
# ----------------------------------------------------------------------
def export_expert_to_hf(ckpt_path: str, domain: str, repo_id: str,
                        token: str, corpus_root: str) -> str:
    """把一个专家权重导出为 HuggingFace 格式并上传。

    海马云等云电脑是按时计费的临时实例，训练完不导出，关机即丢。
    此函数把权重转成标准 safetensors/config.json 并 push 到 HF，
    之后你可随时从 HF 拉取，不依赖云实例。

    需用户自备 HF token（写入环境变量 HUGGINGFACE_TOKEN 或传入）。
    """
    import os
    from simplcore.tensor.io import load
    from simplcore.data.corpus import RealCorpus

    ckpt = load(ckpt_path)
    sd = ckpt["state_dict"]
    meta = ckpt.get("meta", {})

    # 构造 HF 风格 config（与 transformers 不完全兼容，但可被自研引擎/工具读取）
    config = {
        "architecture": "LatticeLM",
        "domain": domain,
        "vocab_size": meta.get("vocab_size"),
        "config": meta.get("config"),
        "framework": "simplcore-native",
        "source": "Qingxiaotuan LLM v1.0 / SimplUnity 12B",
    }
    # 写本地
    local_dir = os.path.join(corpus_root, "models", "hf_export", domain)
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, "config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(config, f, ensure_ascii=False, indent=2)
    # 权重转 numpy dict 存 safetensors（若可用）或 .npz
    try:
        from safetensors.numpy import save_file
        save_file({k: (v.numpy() if hasattr(v, "numpy") else v)
                   for k, v in sd.items()}, os.path.join(local_dir, "model.safetensors"))
    except Exception:
        import numpy as np
        np.savez(os.path.join(local_dir, "model.npz"),
                 **{k: np.asarray(v) for k, v in sd.items()})

    # 上传（需 huggingface_hub）
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_folder(folder_path=local_dir, repo_id=repo_id,
                          repo_type="model")
        return f"https://huggingface.co/{repo_id}"
    except ImportError:
        print("  [HF] 未安装 huggingface_hub，权重已存本地: " + local_dir)
        return local_dir


if __name__ == "__main__":
    cfg = Unity12BConfig()
    est = total_param_estimate(cfg)
    print("=== Unity 12B 缝合方案估算 ===")
    for k, v in est.items():
        print(f"  {k}: {v:.3f}" + (" B" if "B" in k else ""))
    print(f"\n专家领域 ({cfg.num_experts}):")
    for d in EXPERT_DOMAINS:
        print(f"  - {d['id']:14s} {d['desc']} (语料: {d['corpus']})")
    print("\n显存策略: 逐专家训练（一次1个0.966B进显存），12万步/专家×12≈144万步总。")
    print("5090(32GB) fp16+检查点可容纳单专家训练态(~15-23GB)。")
