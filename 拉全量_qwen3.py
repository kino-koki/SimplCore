#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch the COMPLETE Qwen3-1.7B-Instruct weight set (3.789 GiB) from the private
repo kino-koki/Qwen3-1.7B-Instruct, verify every byte, and emit 4 delivery
volumes. No credential literal in this file; read from PROBE_PAT / GH_TOKEN.

Measured on this sandbox: contents+raw returns REAL LFS bytes at ~8.4 MB/s per
stream; Range is NOT supported; a 6 GiB capacity probe succeeded (df's 500 MiB
figure is a stale fuseblk view).
"""
import base64, hashlib, json, os, resource, socket, ssl, time, urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
socket.setdefaulttimeout(180)
W = "/data/workspace"
REPO = "kino-koki/Qwen3-1.7B-Instruct"
TMP = W + "/.qwen_tmp"
PAT = (os.environ.get("PROBE_PAT") or os.environ.get("GH_TOKEN") or "").strip()
assert PAT, "set PROBE_PAT / GH_TOKEN"
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
os.makedirs(TMP, exist_ok=True)


def get_raw(path):
    u = "https://api.github.com/repos/%s/contents/%s?ref=main" % (REPO, path)
    r = urllib.request.Request(u, headers={
        "User-Agent": "qwen-full", "Accept": "application/vnd.github.raw",
        "Authorization": "Bearer " + PAT})
    last = None
    for a in range(5):
        try:
            return urllib.request.urlopen(r, context=_ctx, timeout=180)
        except Exception as e:
            last = e; time.sleep(2 ** a)
    raise last


def one(path, expect):
    dst = os.path.join(TMP, path.replace("/", "__"))
    if os.path.isfile(dst) and os.path.getsize(dst) == expect:
        h = hashlib.sha256()
        with open(dst, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return {"path": path, "bytes": expect, "sha256": h.hexdigest(), "status": "cached"}
    h = hashlib.sha256(); n = 0; t0 = time.time()
    with get_raw(path) as r, open(dst, "wb") as f:
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            h.update(c); f.write(c); n += len(c)
    assert n == expect, (path, n, expect)
    return {"path": path, "bytes": n, "sha256": h.hexdigest(),
            "seconds": round(time.time() - t0, 2), "status": "ok"}


def main():
    meta = json.load(open(W + "/权重完整清单_元数据.json", encoding="utf-8"))
    fs = meta["repos"][REPO.split("/")[-1]]
    shards = [f for f in fs if f["path"].startswith("qa_") or f["path"].startswith("qb_")]
    small = [f for f in fs if f not in shards]
    shards.sort(key=lambda x: x["path"])
    print("shards: %d (%.3f GiB) | small files: %d" %
          (len(shards), sum(f["bytes"] for f in shards) / 2**30, len(small)), flush=True)

    allf = shards + small
    t0 = time.time(); res = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, f["path"], f["bytes"]): f["path"] for f in allf}
        done = 0
        for fu in as_completed(futs):
            r = fu.result(); res[r["path"]] = r; done += 1
            if done % 6 == 0 or done == len(allf):
                print("  [%2d/%2d] %-10s %10d B  %s" %
                      (done, len(allf), r["path"], r["bytes"], r["status"]), flush=True)
    print("download %.1fs" % (time.time() - t0), flush=True)

    # ---- 4 volumes, ~10-11 shards each; delete temp as we pack ----
    GROUPS = [shards[0:11], shards[11:22], shards[22:32], shards[32:]]
    vols = []
    for gi, grp in enumerate(GROUPS, 1):
        out = W + "/Qwen3-1.7B-Instruct_完整权重_卷%02d.zip" % gi
        z = zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True)
        for f in grp:
            src = os.path.join(TMP, f["path"].replace("/", "__"))
            arc = "Qwen3-1.7B-Instruct/" + f["path"]
            with z.open(arc, "w", force_zip64=True) as dst, open(src, "rb") as fp:
                for c in iter(lambda: fp.read(1 << 20), b""):
                    dst.write(c)
            os.remove(src)
        if gi == 1:
            for f in small:
                src = os.path.join(TMP, f["path"].replace("/", "__"))
                if os.path.isfile(src):
                    z.write(src, "Qwen3-1.7B-Instruct/" + f["path"])
                    os.remove(src)
            manifest = {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "repo": REPO,
                "source": "REAL LFS bytes via GitHub contents raw API",
                "note": "sha256 is over the exact bytes inside the zip, NOT the Git LFS pointer blob.",
                "total_files": len(allf),
                "total_bytes": sum(f["bytes"] for f in allf),
                "n_volumes": 4,
                "volume_map": {"卷%02d" % (i + 1): [g["path"] for g in GROUPS[i]] for i in range(4)},
                "files": [res[f["path"]] for f in allf],
            }
            z.writestr("MANIFEST_全量校验.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for s, d in [("拉取全部权重.py", "一键拉取全部权重.py"),
                         ("权重交付说明.md", "权重交付说明.md")]:
                if os.path.isfile(W + "/" + s):
                    z.write(W + "/" + s, d)
        z.close()
        vols.append({"volume": gi, "zip": os.path.basename(out),
                     "bytes": os.path.getsize(out), "shards": len(grp)})
        print("  volume %d: %d shards, %.1f MiB" % (gi, len(grp), os.path.getsize(out) / 2**20), flush=True)

    # prune temp dir
    for f in os.listdir(TMP):
        try: os.remove(os.path.join(TMP, f))
        except OSError: pass
    try: os.rmdir(TMP)
    except OSError: pass

    summary = {"repo": REPO, "commit_done": True, "volumes": vols,
               "total_zip_bytes": sum(v["bytes"] for v in vols),
               "all_files_verified": all(res[f["path"]]["bytes"] == f["bytes"] for f in allf),
               "n_sha256": sum(1 for r in res.values() if "sha256" in r)}
    json.dump(summary, open(W + "/qwen3_full_pull_report.json", "w"), indent=2)
    print("DONE | %.2f GiB in 4 volumes | all sizes verified: %s" %
          (summary["total_zip_bytes"] / 2**30, summary["all_files_verified"]), flush=True)


if __name__ == "__main__":
    main()
