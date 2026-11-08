#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-VL-1B 参考加载器 —— 零 torch 依赖（numpy + struct + mmap）。

用法：
    from load_vl1b import VL1B
    m = VL1B("/path/to/TS2SC-VL-1B")
    print(m.n_params)                             # 1057320252
    w = m.get("model.layers.0.mixer.coef")
    emb = m.get("model.embed_tokens.weight")     # 自动拼回 4 块切片

说明：
  * BF16 在 numpy 下以 uint16 位模式读出；bf16_to_f32() 可无损转 fp32
  * 被 axis0 切分的张量按 slices.json 自动拼回
  * lm_head.weight 与 embed_tokens tie，不在 index 里
"""
from __future__ import annotations
import json, mmap, struct
from pathlib import Path

import numpy as np

DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
      "F64": np.float64, "I64": np.int64, "I32": np.int32,
      "U8": np.uint8, "I8": np.int8, "BOOL": np.bool_}


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    """uint16 位模式（bfloat16）→ fp32，无损。"""
    b = (u.astype(np.uint32) << 16).view(np.float32)
    return b


class VL1B:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.wm = self.index["weight_map"]
        self.slices = json.loads((self.root / "slices.json").read_text())
        self._parts = {}
        for group in ("language", "vision"):
            for orig, info in self.slices.get(group, {}).items():
                for i, p in enumerate(info["parts"]):
                    self._parts[p] = (orig, info["axis"], info["n"], i)
        self._cache = {}
        self.n_params = self.index["metadata"]["total_params"]

    def _shard(self, name):
        if name in self._cache:
            return self._cache[name]
        p = self.root / name
        f = open(p, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack("<Q", mm[:8])[0]
        h = json.loads(mm[8:8 + n])
        self._cache[name] = (h, mm, 8 + n, f)
        return self._cache[name]

    def _raw(self, key):
        name = self.wm[key]
        h, mm, base, _ = self._shard(name)
        e = h[key]
        s, t = e["data_offsets"]
        dt = DT[e["dtype"]]
        a = np.frombuffer(mm, dtype=dt, count=(t - s) // np.dtype(dt).itemsize,
                          offset=base + s)
        return a.reshape(e["shape"]), e["dtype"]

    def get(self, key, as_f32=False):
        """取张量；若 key 是被切分的大张量，自动按 axis0 拼回。"""
        if key in self.wm:
            a, dt = self._raw(key)
            return bf16_to_f32(a) if (as_f32 and dt == "BF16") else a

        # 被切分：找所有 part
        info = None
        for group in ("language", "vision"):
            if key in self.slices.get(group, {}):
                info = self.slices[group][key]
                break
        if info is None:
            if key == "lm_head.weight":
                # tie：指向 embedding
                return self.get("model.embed_tokens.weight", as_f32)
            raise KeyError(f"未知张量 {key!r}")
        arrs = [np.array(self._raw(p)[0]) for p in info["parts"]]
        cat = np.concatenate(arrs, axis=info["axis"])
        assert list(cat.shape) == info["orig_shape"], \
            f"{key} 拼接形状 {cat.shape} != {info['orig_shape']}"
        return bf16_to_f32(cat) if (as_f32 and cat.dtype == np.uint16) else cat

    def keys(self):
        ks = set(self.wm)
        for group in ("language", "vision"):
            ks |= set(self.slices.get(group, {}))
        # 去掉 part key，暴露原始名
        for p in self._parts:
            ks.discard(p)
        return sorted(ks)

    def summary(self):
        return dict(n_params=self.n_params,
                    n_shards=len(set(self.wm.values())),
                    n_index_tensors=len(self.wm),
                    n_reconstructed=len(
                        {v[0] for v in self._parts.values()}),
                    total_size=self.index["metadata"]["total_size"])


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent)
    m = VL1B(root)
    print("summary:", m.summary())
    for k in ["model.embed_tokens.weight",
              "model.layers.0.mixer.coef",
              "model.layers.27.mlp.down_proj.weight",
              "vision_tower.embeddings.patch_embedding.weight",
              "vision_tower.vit_merger.linear_1.weight",
              "merger.mlp.0.linear_2.weight",
              "lm_head.weight"]:
        a = m.get(k)
        print(f"  {k:<58} {str(a.shape):<22} {str(a.dtype):<10} "
              f"n={a.size:,}")
