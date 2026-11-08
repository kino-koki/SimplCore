#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只下载含视觉编码器权重的分片（vpart_20..31），不下载全 3.11GB。

safetensors header 显示视觉张量 data_offsets 从 2,064,353,920 起，
绝对文件偏移 = 101936 + data_offset → 2,064,455,856 ≈ vpart_20 中部，
到 3,109,054,480（文件尾）= vpart_31。故只需 12 片 ≈ 1.2GB。

GitHub raw 对大文件返回 403，改用 contents API + Accept: raw。
"""
import json, os, sys, time, urllib.request
from pathlib import Path

GH = os.environ["GH_TOKEN"]
REPO = "kino-koki/MiniCPM-V-4.7-1B"
OUT = Path("/data/workspace/V47"); OUT.mkdir(exist_ok=True)
API = "https://api.github.com"
H = {"Authorization": f"Bearer {GH}", "Accept": "application/vnd.github.raw",
     "User-Agent": "ts2sc/1"}
PART = 99_614_720
NEED = list(range(20, 32))          # vpart_20 .. vpart_31

# 从 tree 拿每个分片的 size（末片更小）
tree = json.loads(urllib.request.urlopen(urllib.request.Request(
    f"{API}/repos/{REPO}/git/trees/main?recursive=1",
    headers={"Authorization": f"Bearer {GH}", "User-Agent": "ts2sc/1"}),
    timeout=30).read())
sizes = {e["path"]: e.get("size", PART) for e in tree["tree"]}


def fetch(name):
    p = OUT / name
    want = sizes.get(name, PART)
    if p.exists() and p.stat().st_size == want:
        print(f"  {name} ✓ 已存在 {want:,}", flush=True)
        return True
    url = f"{API}/repos/{REPO}/contents/{name}"
    for attempt in range(4):
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(url, headers=H)
            with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            got = p.stat().st_size
            if got == want:
                print(f"  {name} ✓ {got:,} B  [{time.perf_counter()-t0:.0f}s]",
                      flush=True)
                return True
            print(f"  {name} ⚠ 尺寸不符 {got:,} != {want:,}，重试", flush=True)
            p.unlink()
        except Exception as e:
            print(f"  {name} ✗ {type(e).__name__}: {e} (第{attempt+1}次)", flush=True)
            if p.exists():
                p.unlink()
            time.sleep(2 + attempt * 3)
    return False


def main():
    print(f"下载 {len(NEED)} 个分片 → {OUT}", flush=True)
    ok = 0
    for i in NEED:
        if fetch(f"vpart_{i:02d}"):
            ok += 1
        else:
            print(f"  !! vpart_{i:02d} 失败", flush=True)
    total = sum((OUT / f"vpart_{i:02d}").stat().st_size
                for i in NEED if (OUT / f"vpart_{i:02d}").exists())
    print(f"\n完成 {ok}/{len(NEED)}  共 {total/1e9:.3f} GB", flush=True)
    return 0 if ok == len(NEED) else 1


if __name__ == "__main__":
    sys.exit(main())
