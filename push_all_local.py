#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push every local SimplCore file that is absent from the remote repository.

Reads sync_diff.json (produced by the diff scan) and uses the Git Data API:
  create blobs -> create tree (base_tree = current HEAD tree) -> create commit
  -> update refs/heads/main.

The token is read ONLY from the PROBE_PAT / GH_TOKEN environment variable.
No credential literal appears in this file, and no artifact is written with it.
"""
import base64, json, os, ssl, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OWNER, REPO = "kino-koki", "SimplCore"
API = "https://api.github.com"
TOK = (os.environ.get("PROBE_PAT") or os.environ.get("GH_TOKEN") or "").strip()
if not TOK:
    sys.exit("ERROR: set PROBE_PAT / GH_TOKEN in the environment first.")
W = "/data/workspace"

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def req(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"User-Agent": "simplcore-sync",
                 "Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + TOK,
                 "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r, context=_ctx, timeout=90) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except Exception as e:
            code = getattr(e, "code", None)
            if attempt == 4 or code not in (403, 429, 500, 502, 503, 504):
                raise
            time.sleep(2 ** attempt + 1)


def make_blob(item):
    p = os.path.join(W, item["path"])
    b = open(p, "rb").read()
    assert len(b) == item["bytes"], item["path"]
    out = req("POST", "/repos/%s/%s/git/blobs" % (OWNER, REPO),
              {"content": base64.b64encode(b).decode(), "encoding": "base64"})
    assert out["sha"] == item["blob"], (item["path"], out["sha"], item["blob"])
    return item["path"], out["sha"]


def main():
    d = json.load(open(W + "/sync_diff.json", encoding="utf-8"))
    head, base_tree, items = d["head"], d["base_tree"], d["diff"]
    print("pushing %d files (%.1f MiB) on top of %s" %
          (len(items), sum(x["bytes"] for x in items) / 2**20, head[:10]))

    shas = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(make_blob, it) for it in items]
        done = 0
        for fu in as_completed(futs):
            path, sha = fu.result()
            shas[path] = sha
            done += 1
            if done % 50 == 0:
                print("  blobs %3d/%3d  %.1fs" % (done, len(items), time.time() - t0))
    print("blobs done:", len(shas), "in %.1fs" % (time.time() - t0))

    entries = [{"path": p, "mode": "100644", "type": "blob", "sha": s}
               for p, s in sorted(shas.items())]
    # extra static documentation (also local-only)
    extras = {
        "PUSH_ALL_LOCAL.md":
            "# Local-to-cloud sync\n\n"
            "This commit contains every local SimplCore file whose content was "
            "absent from the remote tree at the time of the sync. "
            "Files whose git blob sha1 already existed remotely (under any path) "
            "were skipped, so upstream snapshot copies are not duplicated. "
            "Large (>50 MiB) weight bundles, `__pycache__`, and any file containing "
            "a credential prefix were excluded. See `sync_report.json` for the "
            "per-file verification result.\n",
        "sync_diff.json": json.dumps(
            {k: v for k, v in d.items() if k != "diff"}, indent=2) +
            "\n",
    }
    for p, txt in extras.items():
        b = txt.encode()
        out = req("POST", "/repos/%s/%s/git/blobs" % (OWNER, REPO),
                  {"content": base64.b64encode(b).decode(), "encoding": "base64"})
        entries.append({"path": p, "mode": "100644", "type": "blob", "sha": out["sha"]})

    tree = req("POST", "/repos/%s/%s/git/trees" % (OWNER, REPO),
               {"base_tree": base_tree, "tree": entries})
    commit = req("POST", "/repos/%s/%s/git/commits" % (OWNER, REPO), {
        "message": "Sync all local-only SimplCore code, reports and evidence (%d files)" % len(items),
        "tree": tree["sha"], "parents": [head]})
    req("PATCH", "/repos/%s/%s/git/refs/heads/main" % (OWNER, REPO),
        {"sha": commit["sha"], "force": False})
    print("new commit:", commit["sha"])

    # verification: re-read remote tree, confirm every path has the expected sha
    t = req("GET", "/repos/%s/%s/git/trees/%s?recursive=1" % (OWNER, REPO, commit["sha"]))
    remote = {x["path"]: x["sha"] for x in t.get("tree", []) if x["type"] == "blob"}
    bad = [p for p, s in shas.items() if remote.get(p) != s]
    report = {
        "new_head": commit["sha"], "parent": head,
        "n_pushed": len(shas), "n_remote_blobs": len(remote),
        "n_mismatch": len(bad), "mismatches": bad[:20],
        "verification": "PASS" if not bad else "FAIL",
        "excluded": "upstream snapshots, >50 MiB bundles, __pycache__, credential-bearing",
        "commit_url": "https://github.com/%s/%s/commit/%s" % (OWNER, REPO, commit["sha"]),
    }
    print("verification:", report["verification"], "| remote blobs:", len(remote),
          "| mismatches:", len(bad))
    json.dump(report, open(W + "/sync_report.json", "w"), indent=2)
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
