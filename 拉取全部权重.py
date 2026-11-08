#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch the complete weight set for
  - Qwen3-1.7B-Instruct        (language tower, 3.789 GiB, qa_*/qb_* shards)
  - MiniCPM-V-4.7-1B          (source repo, 2.914 GiB, vpart_* shards)
  - TS2SC-VL-1B               (already-repacked 111 safetensors shards, 2.136 GiB;
                               vision tower = shard-00064 .. shard-00110)

Run on YOUR machine (no 500 MiB sandbox quota):

    export GH_TOKEN='<PAT with Contents: read on kino-koki private repos>'
    python 拉取全部权重.py --out ./weights
    python 拉取全部权重.py --out ./weights --only vision     # vision tower only
    python 拉取全部权重.py --out ./weights --only language   # Qwen3 language only
    python 拉取全部权重.py --out ./weights --workers 8 --verify

Notes
-----
* Uses  https://api.github.com/repos/<repo>/contents/<path>?ref=main
  with  Accept: application/vnd.github.raw
  This returns the REAL LFS bytes (not the 133-byte pointer). Measured ~8.4 MB/s
  per stream from this sandbox; 8 workers should be substantially faster.
* Resumable: a file whose size already equals the Git-tree size is skipped.
* Every downloaded file is sha256-checked into  <out>/manifest.json.
* The token is read ONLY from the GH_TOKEN environment variable. It is never
  printed, never written to any output file. Revoke it when finished.
* Reassembly of qa_*/qb_* and vpart_* into full safetensors is documented in
  RESTORE_分片重组说明.md / SPLIT_RESTORE_原始分片重组.md (shipped in 批次01).
"""
import argparse, hashlib, json, os, ssl, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OWNER = "kino-koki"
REPOS = ["Qwen3-1.7B-Instruct", "MiniCPM-V-4.7-1B", "TS2SC-VL-1B"]
API = "https://api.github.com"
TIMEOUT = 120


def token():
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not t:
        sys.exit("ERROR: set GH_TOKEN first (PAT needs Contents: read on the three private repos).")
    return t.strip()


def ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def api_json(path, tok):
    req = urllib.request.Request(
        API + path,
        headers={"User-Agent": "weight-fetch", "Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, context=ctx(), timeout=TIMEOUT) as r:
        return json.loads(r.read())


def file_list(tok):
    out = []
    for repo in REPOS:
        t = api_json("/repos/%s/%s/git/trees/main?recursive=1" % (OWNER, repo), tok)
        if t.get("truncated"):
            print("  WARNING: tree truncated for", repo)
        for x in t.get("tree", []):
            if x["type"] == "blob" and x.get("size", 0) > 0:
                out.append({"repo": repo, "path": x["path"], "bytes": x["size"]})
    return out


def classify(f):
    p, r = f["path"], f["repo"]
    if r == "TS2SC-VL-1B" and p.startswith("shard-"):
        n = int(p.split("-")[1])
        return "vision" if n >= 64 else "language"
    if r == "MiniCPM-V-4.7-1B" and p.startswith("vpart_"):
        n = int(p.split("_")[1])
        return "vision" if n >= 20 else "language"
    if r == "Qwen3-1.7B-Instruct" and (p.startswith("qa_") or p.startswith("qb_")):
        return "language"
    return "small"


def download(f, tok, out, overwrite=False):
    dst = os.path.join(out, f["repo"], f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not overwrite and os.path.isfile(dst) and os.path.getsize(dst) == f["bytes"]:
        return {"repo": f["repo"], "path": f["path"], "bytes": f["bytes"],
                "status": "skipped-existing"}
    url = "%s/repos/%s/%s/contents/%s?ref=main" % (API, OWNER, f["repo"], f["path"])
    req = urllib.request.Request(url, headers={
        "User-Agent": "weight-fetch",
        "Accept": "application/vnd.github.raw",
        "Authorization": "Bearer " + tok})
    h = hashlib.sha256(); n = 0; t0 = time.time()
    tmp = dst + ".part"
    with urllib.request.urlopen(req, context=ctx(), timeout=TIMEOUT) as r, open(tmp, "wb") as w:
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            h.update(c); w.write(c); n += len(c)
    os.replace(tmp, dst)
    if n != f["bytes"]:
        return {"repo": f["repo"], "path": f["path"], "bytes": n,
                "status": "SIZE-MISMATCH expected %d" % f["bytes"]}
    return {"repo": f["repo"], "path": f["path"], "bytes": n,
            "sha256": h.hexdigest(), "seconds": round(time.time() - t0, 2),
            "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./weights")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=["all", "vision", "language", "small"], default="all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true", help="re-hash local files, no download")
    a = ap.parse_args()
    tok = token()
    os.makedirs(a.out, exist_ok=True)

    files = file_list(tok)
    sel = [f for f in files if a.only == "all" or classify(f) == a.only]
    total = sum(f["bytes"] for f in sel)
    print("repos: %s | selected %d files (%.3f GiB) | out=%s | workers=%d" %
          (", ".join(REPOS), len(sel), total / 2**30, a.out, a.workers))

    if a.verify:
        res = []
        for f in sel:
            p = os.path.join(a.out, f["repo"], f["path"])
            if not os.path.isfile(p):
                res.append({**f, "status": "MISSING"}); continue
            h = hashlib.sha256()
            with open(p, "rb") as fp:
                for c in iter(lambda: fp.read(1 << 20), b""):
                    h.update(c)
            res.append({**f, "sha256": h.hexdigest(),
                        "status": "ok" if os.path.getsize(p) == f["bytes"] else "SIZE-MISMATCH"})
    else:
        res = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(download, f, tok, a.out, a.overwrite): f for f in sel}
            done = 0
            for fu in as_completed(futs):
                r = fu.result(); res.append(r); done += 1
                print("  [%3d/%3d] %-9s %10d B  %s" %
                      (done, len(sel), r["status"], r["bytes"],
                       r["repo"] + "/" + os.path.basename(r["path"])))

    ok = sum(1 for r in res if r["status"] in ("ok", "skipped-existing"))
    mp = os.path.join(a.out, "manifest.json")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "only": a.only,
               "n_files": len(res), "n_ok": ok,
               "total_bytes": sum(f["bytes"] for f in sel), "files": res},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nDONE %d/%d ok | manifest: %s" % (ok, len(res), mp))
    bad = [r for r in res if r["status"] not in ("ok", "skipped-existing")]
    if bad:
        print("NOT OK (%d):" % len(bad), [b["path"] for b in bad][:10])
        sys.exit(2)


if __name__ == "__main__":
    main()
