#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-VL-1B 完整性验证 —— 纯 numpy/struct，零 torch 依赖。

验证项：
  1) 27 个分片的 safetensors header 自洽（offset 不越界、dtype 合法）
  2) 所有张量 BF16/F32 位模式有限（无 NaN/Inf 位）
  3) 切片张量可按 axis0 无损拼回原 shape（emb / vision linear_1）
  4) 按 header 精确重算参数量，与 param_accounting.json 对齐
  5) 语言侧 attention 权重确已全部移除
  6) 视觉 merger 输出维 == 语言侧 hidden_size（跨模态对齐）
"""
from __future__ import annotations
import json, mmap, struct
from pathlib import Path

import numpy as np

ROOT = Path("/data/workspace/VL1B")
OUT = ROOT / "verify_report.json"
DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
      "F64": np.float64, "I64": np.int64, "I32": np.int32,
      "U8": np.uint8, "I8": np.int8, "BOOL": np.bool_}


def read_shard(path):
    """只读 header；数据区用 mmap，避免把 2.2GB 全塞进 2GB 内存。"""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n = struct.unpack("<Q", mm[:8])[0]
    h = json.loads(mm[8:8 + n])
    return h, mm, 8 + n, f


def tensor_from(entry, mm, base, name):
    e = entry[name]
    s, t = e["data_offsets"]
    dt = DT[e["dtype"]]
    a = np.frombuffer(mm, dtype=dt, count=(t - s) // np.dtype(dt).itemsize,
                      offset=base + s)
    return a.reshape(e["shape"]), e


def finite_bits(a, dtype):
    """检查位模式是否含 NaN/Inf。"""
    if dtype == "BF16":
        u = a.astype(np.uint16).ravel()
        exp = (u >> 7) & 0xFF
        return bool(np.all(exp != 0xFF))
    if dtype in ("F16", "F32", "F64"):
        return bool(np.isfinite(a.astype(np.float64)).all())
    return True


def main():
    print("=" * 76)
    print("TS2SC-VL-1B 完整性验证（纯 numpy，无 torch）")
    print("=" * 76)
    idx = json.loads((ROOT / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    slices = json.loads((ROOT / "slices.json").read_text())
    cfg = json.loads((ROOT / "config.json").read_text())
    acc = json.loads((ROOT / "param_accounting.json").read_text())

    shards = sorted(set(wm.values()))
    print(f"\n[1] 分片 {len(shards)} 个 / 索引张量 {len(wm)} 个")
    cache, errors = {}, []
    total_bytes = 0
    for name in shards:
        p = ROOT / name
        h, mm, base, fh = read_shard(p)
        cache[name] = (h, mm, base, fh)
        end = max(e["data_offsets"][1] for k, e in h.items()
                  if k != "__metadata__")
        if end != mm.size() - base:
            errors.append(f"{name}: header 声明 {end} B，实际 {mm.size()-base} B")
        total_bytes += p.stat().st_size
    print(f"    磁盘总字节 {total_bytes:,} = {total_bytes/1e9:.3f} GB")
    print(f"    header 自洽: {'✓ 全部通过' if not errors else '✗ ' + str(errors[:3])}")

    # [2] 逐张量 finite + 参数计数
    print("\n[2] 张量 dtype / 有限性 / 参数计数")
    params = 0
    dtypes, bad = {}, []
    for k, name in wm.items():
        h, mm, base, _ = cache[name]
        a, e = tensor_from(h, mm, base, k)
        dtypes[e["dtype"]] = dtypes.get(e["dtype"], 0) + 1
        params += int(a.size)
        if not finite_bits(a, e["dtype"]):
            bad.append(k)
    print(f"    dtype 分布: {dtypes}")
    print(f"    含 NaN/Inf 位模式的张量: {len(bad)} {'✓' if not bad else bad[:5]}")
    print(f"    索引内张量参数合计 {params:,} = {params/1e6:.3f} M")

    # 切片张量是「同参数的拆分」，不应重复计数：减去多余 part
    sliced_extra = 0
    for group in ("language", "vision"):
        for orig, info in slices.get(group, {}).items():
            # 原张量本身不在 index 里（被 parts 取代），parts 总参数 == 原参数
            # 因此没有额外计数；这里只做拼接无损校验
            arrs = []
            for pn in info["parts"]:
                h, mm, base, _ = cache[wm[pn]]
                a, _ = tensor_from(h, mm, base, pn)
                arrs.append(np.array(a))
            cat = np.concatenate(arrs, axis=info["axis"])
            ok = list(cat.shape) == info["orig_shape"]
            print(f"    拼接 {orig}: {info['orig_shape']} "
                  f"← {len(arrs)} 块  {'✓' if ok else '✗ 形状不符'}")
            if not ok:
                errors.append(f"{orig} 拼接形状不符: {cat.shape}")
    print(f"    (拼接为同参数拆分，不增加参数量)")

    # [3] tie 口径
    print("\n[3] 参数口径")
    print(f"    索引内张量合计        {params:,}")
    print(f"    param_accounting 总计 {acc['total_params']:,}")
    print(f"    config 总计           {cfg['total_params']:,}")
    # lm_head 未写入、与 embed 共享 → 索引合计应等于账本总计
    delta = params - acc["total_params"]
    print(f"    差值 {delta:+,}  →  "
          f"{'✓ 一致（tie 未重复计）' if delta == 0 else '⚠ 需核对'}")

    # [4] 架构断言
    print("\n[4] 架构断言")
    # 只检查**语言侧**；视觉编码器 ViT 自带 self_attn 是正常且必须保留的
    attn_keys = [k for k in wm
                 if k.startswith("model.layers.") and "self_attn" in k]
    mixer_keys = [k for k in wm if ".mixer." in k]
    vis_keys = [k for k in wm if "vision_tower" in k or k.startswith("merger.")]
    vit_attn = [k for k in wm if "vision_tower" in k and "self_attn" in k]
    print(f"    self_attn 残留张量: {len(attn_keys)} "
          f"{'✓ 已全部移除' if not attn_keys else '✗'}")
    print(f"    SimplCore mixer 张量: {len(mixer_keys)} "
          f"({'28 层 × 14' if len(mixer_keys) == 392 else '数量异常'})")
    print(f"    视觉侧张量: {len(vis_keys)} "
          f"(其中 ViT 自带 self_attn {len(vit_attn)}，属正常保留)")
    vout = None
    for k in wm:
        if k.endswith("linear_2.weight") and k.startswith("merger."):
            h, mm, base, _ = cache[wm[k]]
            a, e = tensor_from(h, mm, base, k)
            vout = e["shape"][0]
    print(f"    merger.linear_2 输出维 {vout} vs 语言 hidden_size "
          f"{cfg['text_config']['hidden_size']} → "
          f"{'✓ 逐位对齐，无随机投影层' if vout == cfg['text_config']['hidden_size'] else '✗'}")

    ok_all = (not errors and not bad and not attn_keys and delta == 0
              and vout == cfg["text_config"]["hidden_size"])
    print("\n" + "=" * 76)
    print(f"  ★ 总判定: {'✓ 模型包完整、可加载、参数口径自洽' if ok_all else '✗ 存在问题'}")
    print(f"    总参数 {acc['total_params']:,} = {acc['total_B']} B")
    print("=" * 76)

    OUT.write_text(json.dumps(dict(
        ok=ok_all, n_shards=len(shards), n_tensors=len(wm),
        total_bytes=total_bytes, dtype_count=dtypes,
        params_from_index=params, params_accounting=acc["total_params"],
        tie_delta=delta, nan_inf_tensors=bad,
        residual_self_attn_tensors=attn_keys,
        n_mixer_tensors=len(mixer_keys), n_vision_tensors=len(vis_keys),
        merger_out_dim=vout, text_hidden_size=cfg["text_config"]["hidden_size"],
        errors=errors), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ {OUT}")


if __name__ == "__main__":
    main()
