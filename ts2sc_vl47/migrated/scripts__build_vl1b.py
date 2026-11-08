#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 TS2SC-VL-1B：SimplCore 迁移语言侧 + MiniCPM-V 4.7 视觉侧。

目标：一个可加载、参数量精确、分片可推送的 1B 级多模态模型包。

语言侧（Qwen3-0.6B → SimplCore 迁移）
  保留：embed_tokens / norm / 每层 ln + mlp.*（FFN 逐位直搬，知识在这里）
  丢弃：self_attn.*（176.17M，被 SGC mixer 替代）
  注入：QMIG_R2/ck 的 28 个训练好的 mixer state_dict
  tie_word_embeddings=True，lm_head 与 embed_tokens 同一份权重

视觉侧（MiniCPM-V-4.7-1B）
  V47/vision/ 的 13 个分片（vision_tower 522.30M + merger 25.97M）

关键兼容性（不是凑出来的）
  MiniCPM-V 4.7 text_config.hidden_size = 1024
  Qwen3-0.6B / SimplCore 语言侧      D = 1024
  merger.mlp.0.linear_2 输出维     = 1024
  → 视觉 token 的投影输出维度与语言侧残差流**逐位对齐**，
    无需新增随机初始化的投影层（不引入未训练参数）。

产物目录：/data/workspace/VL1B/
"""
from __future__ import annotations
import json, math, os, struct, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

WS = Path("/data/workspace")
OUT = WS / "VL1B"; OUT.mkdir(exist_ok=True)
VSRC = WS / "V47" / "vision"
MQ = WS / "mig" / "model.safetensors"
CK = WS / "QMIG_R2" / "ck"
N_L, D = 28, 1024
MAX_TENSOR = 88 * 1024 * 1024
MAX_SHARD = 92 * 1024 * 1024

torch.set_num_threads(2)


def log(*a):
    print(*a, flush=True)


DTYPE_MAP = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
    torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
    torch.uint8: "U8", torch.int8: "I8", torch.bool: "BOOL",
}


def write_safetensors(path, tensors):
    """tensors: name -> torch.Tensor（cpu contiguous）"""
    hdr, off = {}, 0
    for nk, t in tensors.items():
        t = t.contiguous()
        nb = t.numel() * t.element_size()
        hdr[nk] = {"dtype": DTYPE_MAP[t.dtype], "shape": list(t.shape),
                   "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for t in tensors.values():
            f.write(t.contiguous().numpy().tobytes()
                    if t.dtype != torch.bfloat16
                    else t.view(torch.uint16).numpy().tobytes())
    return off


def main():
    t0 = time.perf_counter()
    log("=" * 78)
    log("组装 TS2SC-VL-1B  (SimplCore 语言侧 + MiniCPM-V 4.7 视觉侧)")
    log("=" * 78)

    # ── 1) 语言侧：保留张量 ───────────────────────────────────
    sf = safe_open(str(MQ), framework="pt", device="cpu")
    all_keys = list(sf.keys())
    keep = [k for k in all_keys
            if k in ("model.embed_tokens.weight", "model.norm.weight")
            or (k.startswith("model.layers.") and
                ("input_layernorm" in k or "post_attention_layernorm" in k
                 or ".mlp." in k))]
    drop = [k for k in all_keys if k not in keep and k != "lm_head.weight"]
    log(f"\n[1] 语言侧原始张量 {len(all_keys)}  → 保留 {len(keep)}，"
        f"丢弃 {len(drop)}（self_attn + 重复 lm_head）")

    lang = {}
    lang_params = 0
    for k in keep:
        t = sf.get_tensor(k)
        lang[k] = t
        lang_params += t.numel()
    log(f"    保留参数 {lang_params:,} = {lang_params/1e6:.3f} M")
    log(f"    embed_tokens {tuple(lang['model.embed_tokens.weight'].shape)} "
        f"{DTYPE_MAP[lang['model.embed_tokens.weight'].dtype]}")

    # ── 2) 注入 28 个训练好的 mixer ───────────────────────────
    log("\n[2] 注入 SimplCore mixer (QMIG_R2/ck)")
    mixer_params = 0
    mixer_keys = set()
    for i in range(N_L):
        st = torch.load(CK / f"mixer_{i}.pt", map_location="cpu")
        sd = st["best_mixer"] if "best_mixer" in st else st["mixer"]
        for k, v in sd.items():
            nk = f"model.layers.{i}.mixer.{k}"
            lang[nk] = v.detach().cpu()
            mixer_params += v.numel()
            mixer_keys.add(k)
    log(f"    28 层 × {len(mixer_keys)} 个参数张量 = "
        f"{mixer_params:,} = {mixer_params/1e6:.3f} M")
    log(f"    单层 {mixer_params//N_L:,} 参数")
    lang_total = lang_params + mixer_params
    log(f"    ★ 语言侧合计 {lang_total:,} = {lang_total/1e6:.3f} M "
        f"= {lang_total/1e9:.4f} B")

    # ── 3) 大张量切片 + 打包语言侧 ────────────────────────────
    log("\n[3] 语言侧分片打包")
    slices, pack = {}, {}
    for nk, t in lang.items():
        nb = t.numel() * t.element_size()
        if nb <= MAX_TENSOR:
            pack[nk] = t
            continue
        k = math.ceil(nb / MAX_TENSOR)
        step = math.ceil(t.shape[0] / k)
        n = 0
        for i, st_ in enumerate(range(0, t.shape[0], step)):
            en = min(st_ + step, t.shape[0])
            pack[f"{nk}__part{i}of{k}"] = t[st_:en].contiguous()
            n += 1
        slices[nk] = {"orig_shape": list(t.shape), "axis": 0, "n": n,
                      "parts": [f"{nk}__part{i}of{n}" for i in range(n)]}
        log(f"    切片 {nk}: {tuple(t.shape)} → {n} 块")

    shards, cur, cur_bytes = [], {}, 0
    for nk, t in pack.items():
        nb = t.numel() * t.element_size()
        if cur and cur_bytes + nb > MAX_SHARD:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[nk] = t; cur_bytes += nb
    if cur:
        shards.append(cur)

    n_lang_shards = len(shards)
    weight_map = {}
    for i, sh in enumerate(shards, 1):
        name = f"lang-{i:05d}-of-{n_lang_shards:05d}.safetensors"
        nb = write_safetensors(OUT / name, sh)
        for nk in sh:
            weight_map[nk] = name
        log(f"    {name}  {nb/1e6:>7.1f} MB  {len(sh):>4} 张量")

    # ── 4) 视觉侧：硬链接已有分片，合并索引 ────────────────────
    log("\n[4] 合并视觉侧")
    vidx = json.loads((VSRC / "model.safetensors.index.json").read_text())
    vmeta = json.loads((VSRC / "vision_meta.json").read_text())
    for name in sorted(set(vidx["weight_map"].values())):
        src, dst = VSRC / name, OUT / name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
    for k, name in vidx["weight_map"].items():
        weight_map[k] = name
    v_params = vmeta["n_params"]
    log(f"    视觉侧 {len(set(vidx['weight_map'].values()))} 分片 / "
        f"{len(vidx['weight_map'])} 张量 / {v_params:,} = {v_params/1e6:.3f} M")

    # ── 5) 统一索引 / 配置 / 参数账本 ─────────────────────────
    total = lang_total + v_params
    log("\n" + "=" * 78)
    log(f"  ★ 总参数 {total:,} = {total/1e6:.3f} M = {total/1e9:.4f} B")
    log("=" * 78)

    index = {
        "metadata": {
            "total_size": int(sum(
                (OUT / n).stat().st_size for n in set(weight_map.values()))),
            "total_params": int(total),
            "format": "pt",
            "architecture": "SimplCoreVLForConditionalGeneration",
            "tie_word_embeddings": True,
            "note": "lm_head.weight is tied to model.embed_tokens.weight; "
                    "reconstruct sharded tensors listed in slices.json along axis 0.",
        },
        "weight_map": weight_map,
    }
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "slices.json").write_text(
        json.dumps({"language": slices, "vision": vmeta.get("slices", {})},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    config = {
        "architectures": ["SimplCoreVLForConditionalGeneration"],
        "model_type": "simplcore_vl",
        "version": "1.0",
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "total_params": int(total),
        "text_config": {
            "architecture": "SimplCore SGC + Qwen3 FFN (TS2SC direct migration)",
            "hidden_size": D,
            "num_hidden_layers": N_L,
            "vocab_size": 151936,
            "intermediate_size": 3072,
            "rms_norm_eps": 1e-6,
            "mixer": {
                "type": "SpectralGlobalConv",
                "num_bases": 32,
                "decay_max": 4096,
                "local_kernel": 4,
                "rho_param": "logspace",
                "init_mode": "logfreq",
                "content_gate": True,
                "state_bytes_per_layer": 274432,
                "state_scaling": "O(1) w.r.t. sequence length",
            },
            "source": "Qwen/Qwen3-0.6B (FFN/emb/norm verbatim, attention replaced)",
        },
        "vision_config": json.loads((WS / "V47" / "config.json").read_text())[
            "vision_config"],
        "vision_merger": {
            "source": "MiniCPM-V-4.7-1B model.merger.mlp.0",
            "out_dim": 1024,
            "note": "output dim equals text hidden_size — no random projection added",
        },
        "projection_alignment": {
            "vision_output_dim": 1024,
            "text_hidden_size": 1024,
            "aligned": True,
            "extra_untrained_params": 0,
        },
        "known_limitations": [
            "MiniCPM-V 4.7 vocab 248094 vs Qwen3 151936: image/slice special "
            "token ids (248056 etc.) exceed the text vocab; remap or extend "
            "embedding before multimodal inference.",
            "MiniCPM-V 4.7 uses mrope/canvas while Qwen3 uses standard RoPE; "
            "position scheme is not unified.",
            "No cross-modal alignment training has been performed: the package "
            "is a weight-level merge, NOT a usable VQA model.",
            "Text side is a base model (not instruct) and the migrated mixer "
            "still has ~0.7 block cosine — generation quality is not validated.",
        ],
    }
    (OUT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    accounting = {
        "total_params": int(total),
        "total_B": round(total / 1e9, 4),
        "language": {
            "verbatim_qwen": int(lang_params),
            "verbatim_detail": {
                "embed_tokens": int(151936 * 1024),
                "28x_mlp_and_layernorms": int(lang_params - 151936 * 1024 - 1024),
                "final_norm": 1024,
            },
            "mixer_28_layers": int(mixer_params),
            "mixer_per_layer": int(mixer_params // N_L),
            "subtotal": int(lang_total),
            "dropped_attention_params": 176_160_768,
            "lm_head": "tied to embed_tokens (NOT double counted)",
        },
        "vision": {
            "vision_tower": 522_299_312,
            "merger_mlp": 25_967_104,
            "subtotal": int(v_params),
            "source": "MiniCPM-V-4.7-1B model.vision_tower.* + model.merger.*",
        },
        "shards": {
            "language": n_lang_shards,
            "vision": len(set(vidx["weight_map"].values())),
            "total": n_lang_shards + len(set(vidx["weight_map"].values())),
            "index_tensors": len(weight_map),
        },
    }
    (OUT / "param_accounting.json").write_text(
        json.dumps(accounting, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  分片: 语言 {n_lang_shards} + 视觉 "
        f"{len(set(vidx['weight_map'].values()))} = "
        f"{n_lang_shards + len(set(vidx['weight_map'].values()))}")
    log(f"  ✓ 索引/配置/账本已写入 {OUT}")
    log(f"  耗时 {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
