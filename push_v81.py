"""
push_v81.py -- two incremental commits on the CURRENT remote HEAD.

Commit 1: v8.1 deliverables (five reviewer fixes, migration timing /
          acceptance document, TS2SC artefacts, rebuilt PDFs).
Commit 2: the remote-tree verification report produced after commit 1.

Credential is read ONLY from GH_TOKEN (environment). It is never printed,
never written into any pushed file, and never stored in the workspace.
Run: GH_TOKEN=... python3 push_v81.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error

W = "/data/workspace"
R = "kino-koki/SimplCore"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set.")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v81-push",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, method="GET", payload=None):
    raw = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        "https://api.github.com" + path, data=raw, method=method,
        headers={**H, **({"Content-Type": "application/json"} if raw else {})})
    try:
        with urllib.request.urlopen(req, timeout=180) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        sys.exit("API %s %s -> HTTP %d: %s" % (method, path, e.code, body))


ER = W + "/tools/eval_baselines/results"
VL = W + "/ts2sc_vl47"

# ------------------------------------------------ commit 1 file list -----
FILES = [
    # rebuilt submission PDFs (v8.1)
    ("simplcore.pdf",                       W + "/simplcore.pdf"),
    ("paper_main.pdf",                      W + "/paper_main.pdf"),
    ("paper_appendix.pdf",                  W + "/paper_appendix.pdf"),
    # generator + the exact patch that produced v8.1
    ("build_simplcore_pdf_v8.py",           W + "/build_simplcore_pdf_v8.py"),
    ("patch_v8_five_fixes.py",              W + "/patch_v8_five_fixes.py"),
    # machine-readable long-form tables (fix 5)
    ("appendix_D_fusion_long.csv",          W + "/appendix_D_fusion_long.csv"),
    ("appendix_D_fusion_wide.csv",          W + "/appendix_D_fusion_wide.csv"),
    # migration timing / acceptance / 5-fix report
    ("MIGRATION_TIMING_ACCEPTANCE.md",      W + "/MIGRATION_TIMING_ACCEPTANCE.md"),
    ("ts2sc_680m.ps1",                     W + "/ts2sc_680m.ps1"),
    ("TS2SC_VL47_PREFLIGHT.md",             W + "/TS2SC_VL47_PREFLIGHT.md"),
    ("TS2SC_VL47_CONTINUE.md",              W + "/TS2SC_VL47_CONTINUE.md"),
    ("PAPER_TS2SC_ADDENDUM.md",             W + "/PAPER_TS2SC_ADDENDUM.md"),
    # logs (v8.1 sections restored)
    ("REVISION_LOG.md",                     W + "/REVISION_LOG.md"),
    ("CHANGELOG.md",                        W + "/CHANGELOG.md"),
    # raw numeric evidence
    ("ts2sc_vl47/se2_ci.json",              VL + "/se2_ci.json"),
    ("ts2sc_vl47/five_fix_pdf_check.json",  VL + "/five_fix_pdf_check.json"),
    ("ts2sc_vl47/migration_timing_evidence.json", VL + "/migration_timing_evidence.json"),
    ("ts2sc_vl47/rollback_check.json",      VL + "/rollback_check.json"),
    ("tools/eval_baselines/results/desk_check_v7.json", ER + "/desk_check_v7.json"),
    ("tools/eval_baselines/results/split_check.json",   ER + "/split_check.json"),
    # this push tool, kept as an auditable artefact
    ("push_v81.py",                         W + "/push_v81.py"),
    # bundled deliverable
    ("SimplCore_五处修正与迁移验收包.zip",  W + "/SimplCore_五处修正与迁移验收包.zip"),
]

missing = [p for _, p in FILES if not os.path.isfile(p)]
if missing:
    sys.exit("Missing local files:\n  " + "\n  ".join(missing))

# hard guard: credential must never appear in a pushed byte stream.
# The prefix is assembled at runtime so this source file does not itself
# contain a contiguous credential-prefix literal.
PFX = ("gh" + "p_").encode()
for path, local in FILES:
    raw = open(local, "rb").read()
    if PFX in raw or TOK.encode() in raw:
        sys.exit("Credential-like bytes found in %s -- aborting." % path)
assert not any("token" in os.path.basename(p).lower() for p, _ in FILES)


def blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def push_commit(files, message):
    head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
    base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
    print("base =", head[:12], "| uploading", len(files), "files")
    tree, manifest = [], []
    for path, local in files:
        raw = open(local, "rb").read()
        want = blob_sha(raw)
        b = api("/repos/%s/git/blobs" % R, "POST",
                {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
        if b["sha"] != want:
            sys.exit("blob sha mismatch for %s" % path)
        tree.append({"path": path, "mode": "100644", "type": "blob",
                     "sha": b["sha"]})
        manifest.append({"path": path, "bytes": len(raw),
                         "blob_sha1": b["sha"],
                         "sha256": hashlib.sha256(raw).hexdigest()})
    t = api("/repos/%s/git/trees" % R, "POST",
            {"base_tree": base, "tree": tree})
    c = api("/repos/%s/git/commits" % R, "POST",
            {"message": message, "tree": t["sha"], "parents": [head]})
    api("/repos/%s/git/refs/heads/main" % R, "PATCH",
        {"sha": c["sha"], "force": False})
    return c["sha"], head, manifest


sha1, parent1, man1 = push_commit(
    FILES,
    "v8.1: five reviewer fixes, migration timing/acceptance, TS2SC artefacts\n\n"
    "Theorem A(a) conditioned bound; Result 2(ii) premises made explicit;\n"
    "SE2 0.10% gap demoted to a mean coincidence with 95% t-CI;\n"
    "notation ledger (B/b/state bytes); long-form CSV and unified T*.\n"
    "PDF: 17 pages, main text 8 pages. desk-check 46/46, split 11/11.")

# ---- post-push verification: read the remote tree back and compare ------
def read_tree(commit_sha):
    t = api("/repos/%s/git/commits/%s" % (R, commit_sha))["tree"]["sha"]
    out = api("/repos/%s/git/trees/%s?recursive=1" % (R, t))
    return {e["path"]: e["sha"] for e in out["tree"] if e["type"] == "blob"}

remote = read_tree(sha1)
ver = []
for m in man1:
    ver.append({"path": m["path"],
                "local_blob": m["blob_sha1"],
                "remote_blob": remote.get(m["path"]),
                "match": remote.get(m["path"]) == m["blob_sha1"]})
ok1 = all(v["match"] for v in ver)
print("commit 1 =", sha1[:12], "| remote-tree verify:",
      "%d/%d" % (sum(v["match"] for v in ver), len(ver)),
      "PASS" if ok1 else "FAIL")

report = {"repo": R, "commit_1": sha1, "parent": parent1,
          "commit_1_message": "v8.1: five reviewer fixes + migration acceptance",
          "n_files": len(FILES),
          "total_bytes": sum(m["bytes"] for m in man1),
          "remote_tree_verify": {"n": len(ver),
                                 "n_pass": sum(v["match"] for v in ver),
                                 "all_pass": ok1},
          "manifest": man1,
          "verification": ver,
          "note": "Credential read only from GH_TOKEN; no token bytes in any file."}
rp = W + "/push_report_v81.json"
json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

sha2, parent2, man2 = push_commit(
    [("push_report_v81.json", rp)],
    "v8.1: remote-tree verification report (blob sha1/sha256 readback)")
report["commit_2"] = sha2
report["commit_2_message"] = "remote-tree verification report"
json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("commit 2 =", sha2[:12])
print("DONE | files:", len(FILES) + 1, "| bytes:", report["total_bytes"])
