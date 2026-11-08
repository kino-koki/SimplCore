#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 VL1B 的 27 个大分片重切为 ≤24MB 分片，适配 GitHub REST API 上限。

背景：本沙箱 git push 返回 403，只能走 api.github.com；REST API 实测
      25.3MB 文件（base64 33.8MB）OK，50.3MB（base64 67MB）422。
      故统一切到 24MB。

设计：不把 2.2GB 张量全读进内存（cgroup 2GB），pack 里只存
      (源 key, axis0 切片范围, dtype, shape)，写出时按需从 mmap 读。
重切后每个分片仍是独立合法 safetensors，参数量完全不变。
"""
from __future__ import annotations
import json, math, mmap, re, struct
from pathlib import Path

import numpy as np

ROOT = Path("/data/workspace/VL1B")
MAX_TENSOR = 21 * 1024 * 1024
MAX_SHARD = 24 * 1024 * 1024
DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
      "F64": np.float64, "I64": np.int64, "I32": np.int32,
      "U8": np.uint8, "I8": np.int8, "BOOL": np.bool_}

idx = json.loads((ROOT / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
old_slices = json.loads((ROOT / "slices.json").read_text())

def logical(k):
    return re.sub(r"__part\d+of\d+$", "", k)

cache = {}
def load(k):
    name = wm[k]
    if name not in cache:
        f = open(ROOT / name, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack("<Q", mm[:8])[0]
        cache[name] = (json.loads(mm[8:8 + n]), mm, 8 + n, f)
    h, mm, base, _ = cache[name]
    e = h[k]
    s, t = e["data_offsets"]
    dt = DT[e["dtype"]]
    a = np.frombuffer(mm, dtype=dt, count=(t - s) // np.dtype(dt).itemsize,
                      offset=base + s)
    return a.reshape(e["shape"]), e["dtype"]

# pack: new_key -> (src_key, st, en, dtype, shape, nbytes)
pack, groups = {}, {}
for k in wm:
    a, dt = load(k)
    lg = logical(k)
    if a.ndim == 0 or a.nbytes <= MAX_TENSOR:
        pack[k] = (k, None, None, dt, list(a.shape), a.nbytes)
        groups.setdefault(lg, []).append(k)
        del a
        continue
    n = math.ceil(a.nbytes / MAX_TENSOR)
    step = math.ceil(a.shape[0] / n) if n > 1 else a.shape[0]
    for j, st in enumerate(range(0, a.shape[0], step)):
        en = min(st + step, a.shape[0])
        sub = a[st:en]
        nk = k if n == 1 else f"{k}__c{j}"
        pack[nk] = (k, st, en, dt, list(sub.shape), sub.nbytes)
        groups.setdefault(lg, []).append(nk)
    del a

# 重建 slices：逻辑名 → 全部块（顺序即拼接顺序）
slices_new = {"language": {}, "vision": {}}
for lg, keys in groups.items():
    if len(keys) <= 1:
        continue
    if lg in old_slices.get("language", {}):
        orig_shape = old_slices["language"][lg]["orig_shape"]
    elif lg in old_slices.get("vision", {}):
        orig_shape = old_slices["vision"][lg]["orig_shape"]
    else:
        shapes = [pack[k][4] for k in keys]
        orig_shape = [sum(s[0] for s in shapes)] + list(shapes[0][1:])
    grp = "language" if lg.startswith("model.") else "vision"
    slices_new[grp][lg] = dict(orig_shape=list(orig_shape), axis=0,
                               n=len(keys), parts=keys)

# 打包
shards, cur, cur_bytes = [], {}, 0
for nk, desc in pack.items():
    nb = desc[5]
    if cur and cur_bytes + nb > MAX_SHARD:
        shards.append(cur); cur, cur_bytes = {}, 0
    cur[nk] = desc; cur_bytes += nb
if cur:
    shards.append(cur)

def write_st(path, entries):
    hdr, off = {}, 0
    for nk, d in entries.items():
        hdr[nk] = {"dtype": d[3], "shape": d[4],
                   "data_offsets": [off, off + d[5]]}
        off += d[5]
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for nk, d in entries.items():
            src, st, en = d[0], d[1], d[2]
            a, _ = load(src)
            part = a if st is None else a[st:en]
            f.write(np.ascontiguousarray(part).tobytes())

n = len(shards)
new_wm = {}
for i, sh in enumerate(shards, 1):
    name = f"shard-{i:05d}-of-{n:05d}.safetensors"
    write_st(ROOT / name, sh)
    for nk in sh:
        new_wm[nk] = name
    print(f"  {name}  {sum(d[5] for d in sh.values())/1e6:>7.2f} MB  "
          f"{len(sh):>3} 张量", flush=True)

# ★ 不在这里删旧分片：mmap 仍活着（np.frombuffer 持有导出指针），
#   close() 会 BufferError；且若先 unlink 再访问映射会 SIGBUS。
#   删除统一放到全部写盘完成之后的最后一步（Linux unlink 已映射文件是安全的）。
total_size = sum((ROOT / f"shard-{i:05d}-of-{n:05d}.safetensors").stat().st_size
                 for i in range(1, n + 1))
idx["metadata"]["total_size"] = total_size
idx["weight_map"] = new_wm
(ROOT / "model.safetensors.index.json").write_text(
    json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
(ROOT / "slices.json").write_text(
    json.dumps(slices_new, ensure_ascii=False, indent=1), encoding="utf-8")
acc = json.loads((ROOT / "param_accounting.json").read_text())
acc["shards"] = {"total": n, "index_tensors": len(new_wm),
                 "max_shard_bytes": max(
                     (ROOT / f"shard-{i:05d}-of-{n:05d}.safetensors").stat().st_size
                     for i in range(1, n + 1)),
                 "note": "repacked to <=24MB for GitHub REST API upload"}
(ROOT / "param_accounting.json").write_text(
    json.dumps(acc, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"\n✓ 重切完成 {n} 分片 / {len(new_wm)} 张量 / {total_size/1e9:.3f} GB")
print(f"  参数量不变 {idx['metadata']['total_params']:,}")
print(f"  最大单分片 {acc['shards']['max_shard_bytes']/1e6:.2f} MB")

# ── 收尾：删除旧分片（此时不再有任何 mmap 访问）─────────────────
new_names = set(new_wm.values())
removed = 0
for name in set(wm.values()):
    if name in new_names:
        continue
    p = ROOT / name
    if p.exists():
        p.unlink(); removed += 1
print(f"✓ 清理旧分片 {removed} 个；当前分片 {len(new_names)} 个")
