#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers: paired statistics + GitHub Contents/Git-Data API push.

The credential is read from the environment (GH_TOKEN) or passed in-process;
it is NEVER written to a delivered file and NEVER printed.
"""
from __future__ import annotations

import base64, hashlib, json, math, os, urllib.request, urllib.error

OWNER, REPO = "kino-koki", "SimplCore"
API = "https://api.github.com"


# ---------------------------------------------------------------------------
def aggregate(rows, arms=None):
    """Mean/sd per arm + paired difference and 95% normal CI against a ref."""
    by = {}
    for r in rows:
        if "final_val_bpc" not in r:
            continue
        by.setdefault(r["arm"], []).append(r)
    out = {}
    for a, rs in by.items():
        v = [r["final_val_bpc"] for r in rs]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        lg = [r["bpc_long"] for r in rs if r.get("bpc_long") is not None]
        sh = [r["bpc_short"] for r in rs if r.get("bpc_short") is not None]
        out[a] = dict(n=n, mean=round(m, 5), sd=round(sd, 5),
                      sem=round(sd / math.sqrt(n), 5) if n else 0.0,
                      values=[round(x, 5) for x in v],
                      bpc_long_mean=round(sum(lg) / len(lg), 5) if lg else None,
                      bpc_short_mean=round(sum(sh) / len(sh), 5) if sh else None,
                      n_params=rs[0]["n_params"],
                      gradcheck_worst=max(r["gradcheck"]["worst_rel_err"] for r in rs))
    if arms and len(arms) == 2 and arms[0] in out and arms[1] in out:
        a, b = arms
        ra = sorted(by[a], key=lambda r: r["seed"])
        rb = sorted(by[b], key=lambda r: r["seed"])
        if len(ra) == len(rb):
            d = [x["final_val_bpc"] - y["final_val_bpc"] for x, y in zip(ra, rb)]
            n = len(d); m = sum(d) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
            out["paired"] = dict(a=a, b=b, n=n, mean_diff=round(m, 5),
                                 sd=round(sd, 5),
                                 ci95=[round(m - 1.96 * sd / math.sqrt(n), 5),
                                       round(m + 1.96 * sd / math.sqrt(n), 5)],
                                 all_same_sign=bool(all(x > 0 for x in d) or
                                                    all(x < 0 for x in d)))
    return out


# ---------------------------------------------------------------------------
def _req(method, path, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "simplcore-push",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=90) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def git_blob_sha1(content: bytes):
    h = hashlib.sha1()
    h.update(b"blob " + str(len(content)).encode() + b"\0")
    h.update(content)
    return h.hexdigest()


def push(files, message, token):
    """files: list of (local_abs_path, repo_relative_path)."""
    repo = _req("GET", f"/repos/{OWNER}/{REPO}", token)
    default = repo["default_branch"]
    ref = _req("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{default}", token)
    head = ref["object"]["sha"]
    commit = _req("GET", f"/repos/{OWNER}/{REPO}/git/commits/{head}", token)
    base_tree = commit["tree"]["sha"]
    entries, local_shas = [], {}
    for local, rel in files:
        content = open(local, "rb").read()
        blob = _req("POST", f"/repos/{OWNER}/{REPO}/git/blobs", token,
                    {"content": base64.b64encode(content).decode(),
                     "encoding": "base64"})
        entries.append({"path": rel, "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})
        local_shas[rel] = (git_blob_sha1(content), len(content))
    tree = _req("POST", f"/repos/{OWNER}/{REPO}/git/trees", token,
                {"base_tree": base_tree, "tree": entries})
    newc = _req("POST", f"/repos/{OWNER}/{REPO}/git/commits", token,
                {"message": message, "tree": tree["sha"], "parents": [head]})
    _req("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{default}", token,
         {"sha": newc["sha"], "force": False})
    # read back the committed tree
    back = _req("GET", f"/repos/{OWNER}/{REPO}/git/trees/{newc['sha']}?recursive=1", token)
    remote = {e["path"]: e["sha"] for e in back["tree"] if e["type"] == "blob"}
    mism = [p for p, (s, _) in local_shas.items() if remote.get(p) != s]
    return dict(commit=newc["sha"], parent=head, branch=default,
                n_files=len(files), mismatches=mism,
                verified=len(files) - len(mism),
                files=[dict(path=p, blob_sha1=s, bytes=n)
                       for p, (s, n) in local_shas.items()])
