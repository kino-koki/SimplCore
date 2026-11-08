"""把「理论形式化 + 相关工作核实」同步到 kino-koki/SimplCore@main。

单 commit：
  MODIFY  SimplCore_Paper.md                       (0.2-β 主稿，795 行)
  MODIFY  docs/PAPER_CONSISTENCY_CHECK.json        (44/44 PASS)
  MODIFY  docs/PAPER_REWRITE_NOTE.md               (追加第二轮说明)
  ADD     docs/EXPRESSIVITY_BOUNDARIES.md
  ADD     docs/RELATED_WORK.md
  ADD     docs/BOUNDARY_UPDATE_NOTE.md
  ADD     tools/boundary_probe/  (README / 脚本 / 两份原始结果)

走 GitHub Git Data API。token 只从 GH_TOK 环境变量读取，不写入任何产物。

  GH_TOK=... python push_boundary.py --dry-run
  GH_TOK=... python push_boundary.py --apply
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO = "kino-koki/SimplCore"
BRANCH = "main"
API = "https://api.github.com"
TOKEN = os.environ.get("GH_TOK", "")
ROOT = "/data/workspace/train"

if not TOKEN:
    sys.exit("GH_TOK 未设置")


def req(path, method="GET", payload=None):
    r = urllib.request.Request(
        API + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "simplcore-boundary-push",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(r, timeout=90) as f:
        return json.loads(f.read().decode("utf-8"))


def exists(path):
    try:
        req(f"/repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={BRANCH}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


MODIFY = {
    "SimplCore_Paper.md": f"{ROOT}/PAPER_NEW.md",
    "docs/PAPER_CONSISTENCY_CHECK.json": f"{ROOT}/results/paper_consistency.json",
    "docs/PAPER_REWRITE_NOTE.md": f"{ROOT}/PAPER_REWRITE_NOTE.md",
}
ADD = {
    "docs/EXPRESSIVITY_BOUNDARIES.md": f"{ROOT}/EXPRESSIVITY_BOUNDARIES.md",
    "docs/RELATED_WORK.md": f"{ROOT}/RELATED_WORK.md",
    "docs/BOUNDARY_UPDATE_NOTE.md": f"{ROOT}/BOUNDARY_UPDATE_NOTE.md",
    "tools/boundary_probe/README.md": f"{ROOT}/README_boundary.md",
    "tools/boundary_probe/b1_additive_separation.py": f"{ROOT}/b1_additive_separation.py",
    "tools/boundary_probe/results/b1_corollary.json": f"{ROOT}/results/b1_corollary.json",
    "tools/boundary_probe/results/b1_run.txt": f"{ROOT}/results/b1_run.txt",
}

COMMIT = """paper(theory+related): formalize B1/B4, verify 15 related-work citations, keep "not ready to submit"

Second-pass continuation of the 0.2-beta rewrite. NO training tasks restarted
(per the user's freeze). Only the two hard gates that can be moved with theory,
literature verification and a seconds-long NumPy check are addressed.

THEORY (docs/EXPRESSIVITY_BOUNDARIES.md, paper 4.1/4.4):
- Proposition B1 is now a STRICT PROOF, not an observational claim. A static
  kernel output y_L = A + K(0)E(q) is additively separable: the history
  summary A is independent of q. Taking two queries pointing at different
  target positions and subtracting gives
      K(0)W(E(q1)-E(q2)) = E(v_j1)-E(v_j2),
  whose LHS is independent of value assignment while the RHS varies freely
  -> contradiction. Premises: linearly independent keys, freely assignable
  values, linear readout.
  Scope is EXPLICITLY limited to a single mixer + linear readout; extension
  to deep stacks with position-wise nonlinear FFNs is marked "not proven here".
- Corollary B1.1 (interpretive framework, NOT a theorem): breaking the
  separation requires a (q,k_p) pairwise-matching term. This unifies why
  longer memory / rho tuning / orthogonal grids / per-channel coef / gating
  all fail, and why CAS, FD-CAO, linear attention, attention and hybrid
  necessarily converge to a content-addressed state.
- Lemma B4 is now a strict proof: for the LTV 2nd-order recurrence
  K(s+1,s)=a_{s+1}c_s and K(s+2,s+1)=a_{s+2}c_{s+1}; Toeplitz would require
  equality, which state feedback generally violates -> non-Toeplitz -> no
  single frequency-domain kernel -> FFT duality fails.
- B2/B3 remain empirical observations + finite-sample estimators, NOT
  theorems. 0.2484 > 0.2476 already shows the oracle is an estimator.

MOST IMPORTANT HONESTY CORRECTION: we do NOT claim first discovery.
Zoology/MQAR (arXiv:2312.04927, ICLR 2024), Repeat After Me (2402.01032),
The Illusion of State (2404.08819, ICML 2024) already established that
static / constant-state models are weak at associative recall, copying and
state tracking; S5 (2208.04933) noted convolution requires time invariance
and Mamba (2312.00752) noted selectivity breaks the convolutional view.
Novelty is narrowed to three defensible items: formal tightening onto SGC's
checkable algebraic properties, BINDING the two boundaries into one
architectural constraint (duality <-> context-dependent kernel mutual
exclusion), and a reproducible evidence chain. Phrases like "first proof
that static convolution cannot do content addressing" are explicitly listed
as indefensible and forbidden.

RELATED WORK (docs/RELATED_WORK.md, paper 8):
- 15 entries verified one by one for title + arXiv ID + year/venue, arranged
  in a unified classification table (state / input-dependent kernel / FFT
  path / content addressing). Entries without a verifiable ID (CKConv, an
  ICML 2026 same-name extension) are deliberately NOT listed.
- New self-check enforces "zero arXiv IDs outside the verified whitelist";
  currently 15/15 used, 0 stray.
- STILL MISSING: equal-parameter / equal-compute quantitative comparison
  against Mamba-2 / GLA / DeltaNet / Hyena / TS. Added as a new P0 row in
  Appendix C. The table is qualitative classification, NOT an experiment.

NUMERICAL COROLLARY (tools/boundary_probe/, seconds-long, NOT training):
Deliberately the MOST favorable setting (orthogonal one-hot keys, shared unit
value embeddings, query == key). Held-out L2 over 12 (V,d) combos:
  arbitrary FIXED linear map on concatenated u (more permissive than K(tau))
      -> always >0.5;
  naive product interaction with position averaged out -> always >0.5;
  sum_p (q.k_p) v_p  (key/value separation + pairwise matching) -> ~0.
"Product interaction alone also fails" is an unexpected but useful result:
it corroborates B1.1 (pairwise (q,k) matching is required, not arbitrary
nonlinearity or gating).

PAPER EDITS (12 exact replacements, each checked for unique match):
contribution table C2/C4 upgraded + C8 added; 4.1 rewritten; 4.3 oracle
caveat corrected; 4.4 lemma inserted with S5/Mamba attribution; 5.0 [R]
row; 7.1(6) and 7.3(12) marked partially done; section 8 fully rewritten;
section 9 conclusion point 1; appendices A/B paths+commands; appendix C
two gates marked partially done + new P0 quantitative-comparison row.

NOT DONE: no training restarted (multi-seed, kdrand, mod/osc arms, >=10M-token
main experiment all still frozen), no new or fabricated experimental numbers,
no source/config/checkpoint/guard-test changes, no equal-compute benchmark,
B2/B3 and the deep-stack extension of B1 not passed off as theorems,
AI-usage declaration still a placeholder.
"NOT READY TO SUBMIT" conclusion is UNCHANGED.

Consistency self-check upgraded 26/26 -> 44/44 PASS.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        sys.exit("指定 --dry-run 或 --apply")

    ok = True
    print("== 本地文件 ==")
    files = {}
    for path, local in {**MODIFY, **ADD}.items():
        if not os.path.exists(local):
            print(f"  [FAIL] 本地缺失 {local}"); ok = False; continue
        txt = open(local, encoding="utf-8").read()
        files[path] = txt
        print(f"  [OK] {path} ({len(txt)} chars)")

    print("\n== 远端路径状态 ==")
    for path in MODIFY:
        e = exists(path)
        print(f"  [{'OK' if e else 'FAIL'}] MODIFY {path} {'存在' if e else '不存在'}")
        ok &= e
    for path in ADD:
        e = exists(path)
        print(f"  [{'CONFLICT' if e else 'free'}] ADD    {path}")
        ok &= not e

    print("\n== 自检 ==")
    chk = json.load(open(f"{ROOT}/results/paper_consistency.json"))
    print(f"  一致性: {chk['passed']}/{chk['total']} PASS, {chk['failed']} FAIL")
    ok &= (chk["failed"] == 0)
    print(f"  arXiv 白名单: 使用 {len(chk['ids_used'])} / 核实 {chk['whitelist_size']}, "
          f"stray={chk['stray_ids']}")
    ok &= (len(chk["stray_ids"]) == 0)
    b1 = json.load(open(f"{ROOT}/results/b1_corollary.json"))
    print(f"  B1 数值: static={b1['static_stays_positive']} "
          f"naive={b1['naive_bilinear_also_fails']} "
          f"matched={b1['matched_bilinear_collapses']}")
    ok &= all([b1["static_stays_positive"], b1["naive_bilinear_also_fails"],
               b1["matched_bilinear_collapses"], b1["analytic_toy"]["contradiction"]])
    paper = files.get("SimplCore_Paper.md", "")
    print(f"  主稿首行: {paper.splitlines()[0] if paper else '?'}")
    print(f"  主稿行数: {len(paper.splitlines())}")
    print(f"  仍标不宜投稿: {'不宜直接投稿' in paper}")
    ok &= ("不宜直接投稿" in paper and "不主张首次发现" in paper)

    if not ok:
        sys.exit("\n校验未过，终止（远端未改动）")
    if a.dry_run:
        print(f"\nDRY-RUN OK：{len(MODIFY)} 修改 + {len(ADD)} 新增 = {len(files)} 文件，远端未改动")
        return

    print("\n== 构造单个 commit ==")
    base = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    base_tree = req(f"/repos/{REPO}/git/commits/{base}")["tree"]["sha"]
    print(f"  base {base[:12]}  tree {base_tree[:12]}")
    entries = [{"path": p, "mode": "100644", "type": "blob", "content": c}
               for p, c in files.items()]
    tree = req(f"/repos/{REPO}/git/trees", "POST",
               {"base_tree": base_tree, "tree": entries})
    print(f"  new tree {tree['sha'][:12]} ({len(tree.get('tree', []))} entries)")
    cm = req(f"/repos/{REPO}/git/commits", "POST",
             {"message": COMMIT, "tree": tree["sha"], "parents": [base]})
    print(f"  new commit {cm['sha'][:12]}")
    cur = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    if cur != base:
        sys.exit(f"main 已推进到 {cur[:12]}，终止 PATCH（未覆盖）")
    res = req(f"/repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH",
              {"sha": cm["sha"], "force": False})
    print(f"\nPUSH OK -> {BRANCH} @ {res['object']['sha'][:12]}")
    print(f"https://github.com/{REPO}/commit/{cm['sha']}")


if __name__ == "__main__":
    main()
