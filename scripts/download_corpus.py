# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Download a real Simplified-Chinese pretraining corpus.

Default source is FineWeb-Edu-Chinese (``opencsg/chinese-fineweb-edu``), a large
filtered educational web corpus. `huggingface.co` is blocked from some networks;
set ``HF_ENDPOINT`` to a mirror (``https://hf-mirror.com`` works from mainland
China).

The script downloads the *smallest* shards first so that a usable corpus exists
after a few minutes rather than after the whole 300 GB dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="opencsg/chinese-fineweb-edu")
    ap.add_argument("--out", default="data/raw/fineweb_zh")
    ap.add_argument("--max-gb", type=float, default=200.0,
                    help="soft cap on downloaded parquet bytes")
    ap.add_argument("--max-files", type=int, default=0, help="0 = no explicit file cap")
    ap.add_argument("--endpoint", default=None,
                    help="HF endpoint mirror, e.g. https://hf-mirror.com")
    args = ap.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(args.dataset, files_metadata=True)
    rows = sorted(
        ((s.size or 0, s.rfilename) for s in (info.siblings or [])
         if s.rfilename.endswith(".parquet")),
        key=lambda r: r[0],
    )
    if not rows:
        print(f"no parquet files found in {args.dataset}", file=sys.stderr)
        return 1

    total = sum(r[0] for r in rows)
    print(f"{args.dataset}: {len(rows)} parquet shards, {total/1e9:.1f} GB total")
    os.makedirs(args.out, exist_ok=True)

    budget = args.max_gb * 1e9
    picked, got = 0, 0
    manifest = []
    for size, name in rows:
        if got >= budget:
            break
        if args.max_files and picked >= args.max_files:
            break
        print(f"  [{picked+1}] {size/1e6:8.1f} MB  {name}", flush=True)
        try:
            path = hf_hub_download(args.dataset, name, repo_type="dataset",
                                   local_dir=args.out)
        except Exception as exc:                       # network flakiness is expected
            print(f"      FAILED ({type(exc).__name__}); continuing", file=sys.stderr)
            continue
        real = os.path.getsize(path)
        got += real
        picked += 1
        manifest.append({"file": name, "bytes": real, "path": path})

    with open(os.path.join(args.out, "download_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "endpoint": os.environ["HF_ENDPOINT"],
                   "files": manifest, "total_bytes": got}, fh, indent=2)

    print(f"\nDONE: {picked} shards, {got/1e9:.2f} GB -> {args.out}")
    if got == 0:
        print("nothing downloaded; check network / --endpoint", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
