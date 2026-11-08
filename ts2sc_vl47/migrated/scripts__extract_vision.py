#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从稀疏分片中精确提取视觉编码器权重，重打包为标准 HF safetensors 分片。

输入：V47/vpart_20..31（覆盖绝对偏移 [20*P, 3,109,054,480)）
      V47/header.json（原 model.safetensors 的完整张量索引）
输出：V47/vision/model-00001-of-000NN.safetensors + model.safetensors.index.json
      V47/vision/vision_meta.json（参数量、张量清单、命名映射）

★ 不合并 3.11GB 原文件（沙箱单文件上限 200MB），跨分片按字节区间直接读取。
"""
from __future__ import annotations
import json, math, os, struct
from pathlib import Path

import numpy as np

# 手写 safetensors writer：避免依赖安装不稳定的 safetensors 包，
# 且 BF16 是原生合法 dtype，PyTorch 加载端可直接得到 torch.bfloat16。
def write_safetensors(path, tensors):
    """tensors: name -> np.uint16 array（位模式即 bfloat16）"""
    hdr, off = {}, 0
    for nk, a in tensors.items():
        n = a.nbytes
        hdr[nk] = {"dtype": "BF16", "shape": list(a.shape),
                   "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in tensors.values():
            f.write(a.tobytes())
    return off

ROOT = Path("/data/workspace/V47")
OUT = ROOT / "vision"; OUT.mkdir(exist_ok=True)
PART = 99_614_720
HEADER_LEN = 101_936          # 8 + 101928
TOTAL = 3_109_054_480
MAX_SHARD = 90 * 1024 * 1024  # 每片 < 95MB，避开单文件 200MB 上限

h = json.loads((ROOT / "header.json").read_text())
meta = h.get("__metadata__", {})
tensors = {k: v for k, v in h.items() if k != "__metadata__"}

# 视觉张量：vision_tower 全族（含 embeddings / encoder / vit_merger）
# 视觉侧 = vision_tower（ViT + vit_merger）+ model.merger（官方视觉→语言投影 MLP）
vis = {k: v for k, v in tensors.items()
       if "vision_tower" in k or k.startswith("model.merger.")}
print(f"视觉侧张量 {len(vis)} 个（vision_tower + merger）", flush=True)
bytes_total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in vis.values())
print(f"总字节 {bytes_total:,} = {bytes_total/1e6:.1f} MB "
      f"≈ {bytes_total/2/1e6:.1f} M 参数 (bf16)", flush=True)

# 检查所有视觉张量都落在已下载分片范围内
lo = min(v["data_offsets"][0] for v in vis.values()) + HEADER_LEN
hi = max(v["data_offsets"][1] for v in vis.values()) + HEADER_LEN
print(f"绝对字节区间 [{lo:,}, {hi:,})  → vpart_{lo//PART:02d}..vpart_{(hi-1)//PART:02d}")
assert lo // PART >= 20 and (hi - 1) // PART <= 31, "视觉张量越出已下载范围"

parts = {}
for i in range(20, 32):
    p = ROOT / f"vpart_{i:02d}"
    assert p.exists(), p
    parts[i] = p


def read_abs(start: int, end: int) -> bytes:
    """跨分片读取绝对字节区间 [start, end)。"""
    buf = bytearray()
    pos = start
    while pos < end:
        idx = pos // PART
        off = pos % PART
        assert idx in parts, f"需要 vpart_{idx:02d}（未下载）"
        n = min(end - pos, PART - off)
        with open(parts[idx], "rb") as f:
            f.seek(off)
            chunk = f.read(n)
        assert len(chunk) == n, f"vpart_{idx:02d} 读取不足 {len(chunk)}/{n}"
        buf.extend(chunk)
        pos += n
    return bytes(buf)


# 逐张量解码（bf16 → 以 uint16 原样保存，避免精度往返；safetensors 支持 BF16 dtype）
# safetensors.numpy 不支持 bfloat16 原生，改用 uint16 + dtype 标注在 meta 里，
# 加载端用 torch.frombuffer(...).view(torch.bfloat16) 还原。
arr_map = {}
for k, v in sorted(vis.items(), key=lambda kv: kv[1]["data_offsets"][0]):
    s, e = v["data_offsets"]
    raw = read_abs(HEADER_LEN + s, HEADER_LEN + e)
    dtype = v["dtype"]
    assert dtype == "BF16", f"{k} dtype={dtype} 未处理"
    a = np.frombuffer(raw, dtype=np.uint16).reshape(v["shape"])
    # 去掉 "model." 前缀，命名更干净；原 key 记在映射表里
    nk = k[len("model."):] if k.startswith("model.") else k
    arr_map[nk] = (a, k, v["shape"])

print(f"解码完成 {len(arr_map)} 张量", flush=True)

# ★ 大张量切片：单个张量 >90MB 无法装进 GitHub 100MB 单文件限制，
#   按 dim0 切成若干块，命名 <name>__part{i}of{k}，加载端 cat 还原。
#   语义完全等价（行切分），仅 key 名变化，slices 元数据显式记录。
MAX_TENSOR = 88 * 1024 * 1024
slices = {}
pack = {}
for nk, (a, orig, shape) in arr_map.items():
    if a.nbytes <= MAX_TENSOR:
        pack[nk] = (a, orig, shape)
        continue
    k = math.ceil(a.nbytes / MAX_TENSOR)
    step = math.ceil(a.shape[0] / k)
    n = 0
    for i, st in enumerate(range(0, a.shape[0], step)):
        en = min(st + step, a.shape[0])
        pn = f"{nk}__part{i}of{k}"
        pack[pn] = (a[st:en].copy(), orig, list(a[st:en].shape))
        n += 1
    slices[nk] = {"orig_shape": list(a.shape), "axis": 0,
                  "parts": [f"{nk}__part{i}of{n}" for i in range(n)],
                  "n": n}
    print(f"  切片 {nk}: {a.shape} → {n} 块 (每块 ≤{MAX_TENSOR/1e6:.0f}MB)",
          flush=True)

# 打包分片
shards, cur, cur_bytes, sizes = [], {}, 0, []
for nk, (a, orig, shape) in pack.items():
    nb = a.nbytes
    if cur and cur_bytes + nb > MAX_SHARD:
        shards.append(cur); sizes.append(cur_bytes); cur, cur_bytes = {}, 0
    cur[nk] = a
    cur_bytes += nb
if cur:
    shards.append(cur); sizes.append(cur_bytes)

n = len(shards)
weight_map = {}
for i, sh in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{n:05d}.safetensors"
    write_safetensors(OUT / name, sh)
    for nk in sh:
        weight_map[nk] = name
    print(f"  {name}  {sizes[i-1]/1e6:>8.1f} MB  {len(sh):>4} 张量", flush=True)

index = {"metadata": {"total_size": int(bytes_total),
                      "format": "pt",
                      "dtype": "BF16",
                      "source": "MiniCPM-V-4.7-1B / model.vision_tower.*"},
         "weight_map": weight_map}
(OUT / "model.safetensors.index.json").write_text(
    json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

# 参数量与命名映射
params = sum(int(np.prod(shape)) for _, _, shape in arr_map.values())
by_group = {}
for nk, (a, orig, shape) in arr_map.items():
    g = ".".join(nk.split(".")[:3])
    by_group[g] = by_group.get(g, [0, 0])
    by_group[g][0] += 1
    by_group[g][1] += int(np.prod(shape))

vmeta = {
    "source_repo": "kino-koki/MiniCPM-V-4.7-1B",
    "source_file": "model.safetensors",
    "source_shards": [f"vpart_{i:02d}" for i in range(20, 32)],
    "n_tensors": len(arr_map),
    "dtype": "BF16 (native safetensors dtype)",
    "n_params": params,
    "n_params_M": round(params / 1e6, 3),
    "bytes": int(bytes_total),
    "n_output_shards": n,
    "rename": {"prefix_stripped": "model.", "note": "orig key = 'model.' + new key"},
    "groups": {g: {"tensors": c, "params": p} for g, (c, p) in by_group.items()},
    "name_map": {nk: orig for nk, (_, orig, _) in pack.items()},
    "slices": slices,
    "load_note": ("BF16 is a native safetensors dtype; load with "
                  "safetensors.torch.load_file -> torch.bfloat16. "
                  "For keys in 'slices', cat the __part* along axis 0 "
                  "to reconstruct the original tensor."),
}
(OUT / "vision_meta.json").write_text(
    json.dumps(vmeta, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"\n✓ 视觉编码器参数 {params:,} = {params/1e6:.2f} M", flush=True)
print("  分组:")
for g, (c, p) in sorted(by_group.items(), key=lambda x: -x[1][1])[:8]:
    print(f"    {g:<52} {p/1e6:>8.2f} M  ({c} 张量)")
print(f"✓ 输出 {OUT}", flush=True)
