"""
push_governance.py -- push the v7.1 governance round to kino-koki/SimplCore@main.

    export GH_TOKEN=<token with contents:write>
    python3 push_governance.py

Incremental commit on the CURRENT remote HEAD; never rewrites history.
The credential is read ONLY from the environment and is never printed or
written to disk.
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request

W = "/data/workspace"
R = "kino-koki/SimplCore"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set. Export a token with contents:write and rerun.")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v71-governance",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, method="GET", payload=None):
    raw = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("https://api.github.com" + path, data=raw,
                                 method=method,
                                 headers={**H, **({"Content-Type": "application/json"}
                                                  if raw else {})})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


T = W + "/theory_work"
TR = W + "/train/superposition_probe/results"
ER = W + "/tools/eval_baselines/results"

FILES = [
    ("paper_main.pdf", W + "/paper_main.pdf"),
    ("paper_appendix.pdf", W + "/paper_appendix.pdf"),
    ("simplcore.pdf", W + "/simplcore.pdf"),
    ("GOVERNANCE_SUMMARY.md", W + "/GOVERNANCE_SUMMARY.md"),
    ("cfp_lock.md", W + "/cfp_lock.md"),
    ("claim_registry.md", W + "/claim_registry.md"),
    ("proof_audit.md", W + "/proof_audit.md"),
    ("p0_scan.md", W + "/p0_scan.md"),
    ("number_freeze.md", W + "/number_freeze.md"),
    ("review_sim.md", W + "/review_sim.md"),
    ("rebuttal_prep.md", W + "/rebuttal_prep.md"),
    ("submit_checklist.md", W + "/submit_checklist.md"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
    ("MANUSCRIPT_INDEX.md", W + "/MANUSCRIPT_INDEX.md"),
    ("theory_work/governance.py", T + "/governance.py"),
    ("verify_governance.py", W + "/verify_governance.py"),
    ("split_submission.py", W + "/split_submission.py"),
    ("patch_governance.py", W + "/patch_governance.py"),
    ("build_simplcore_pdf_v7.py", W + "/build_simplcore_pdf_v7.py"),
    ("tools/eval_baselines/desk_check_v7.py",
     W + "/tools/eval_baselines/desk_check_v7.py"),
    ("tools/eval_baselines/results/governance_gate.json", ER + "/governance_gate.json"),
    ("tools/eval_baselines/results/desk_check_v7.json", ER + "/desk_check_v7.json"),
    ("tools/eval_baselines/results/split_check.json", ER + "/split_check.json"),
    ("theory_work/results/governance_s4.json", T + "/results/governance_s4.json"),
    ("theory_work/results/governance_namespace_before.json",
     T + "/results/governance_namespace_before.json"),
    ("theory_work/results/governance_namespace_after.json",
     T + "/results/governance_namespace_after.json"),
    ("theory_work/results/governance_number_freeze.json",
     T + "/results/governance_number_freeze.json"),
    ("train/superposition_probe/results/fusion_guard.json", TR + "/fusion_guard.json"),
]

missing = [p for _, p in FILES if not os.path.isfile(p)]
if missing:
    sys.exit("Missing local files:\n  " + "\n  ".join(missing))

head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
print("base", head[:12], "| uploading", len(FILES), "files")

tree = []
for path, local in FILES:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-62s %8d B sha256=%s" %
          (path, len(raw), hashlib.sha256(raw).hexdigest()[:12]))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v7.1: submission-governance workflow, executable S4 check, two-PDF split

Executed the ICML 2027 submission skill pack, Phases 0-5. No new theorem,
no new experiment, no new number. Training remains frozen.

FOUR REAL P0 DEFECTS FIXED (all present in the previous 16-page build):
  * Theorem A's premises used P1-P5 while supporting propositions also used
    P1-P8; the same section had both 'P5: two accuracy notions' and
    'P5 gives a matching state-order bound'. Premises renamed TAP1-TAP5,
    with an explicit disjoint-namespace rule in Section 3.1.
  * 'Theorem B' meant both Result 2 and the state-order statement folded
    into A(e). Removed everywhere (0 occurrences).
  * Section 3.1(c) and Table 1 cited P1 for the Toeplitz equivalence;
    the argument actually uses P6. Corrected.
  * Full proofs duplicated in the main text (3.1.1/3.1.2/3.2.1/3.4) and
    again in Appendix A.1-A.4 were diverging. Removed from the main text
    (5.9k chars); the main text keeps premise + statement + short proof
    intuition + evidence anchor, Appendix holds the single complete copy.
  * Table 28 caption now labels 0.2969 as a single-seed, different
    configuration; value injected from the raw carrier JSON.

ARCHITECTURE: governance rules became computable objects.
  theory_work/governance.py implements B = sum_t log2|pi^-1(v_t)| as
  budget_B(), enumerates 3 carriers x m=1..4 and asserts the lower bound
  B >= log2|A| and the tightness witness |A| <= 2^B use the SAME quantity.
  Result: S4 PASS, tight witness key_only_tag. A counterfactual re-run
  under the v6 definition (alphabet expansion NOT charged to B) produces
  4 contradictions |A|=2^m > 2^B=1, showing why the formula change was
  necessary. Also: claim-namespace validation (before/after JSON) and a
  16-entry single-source-of-truth number freeze.
  verify_governance.py chains 5 scripts -> governance_gate.json 12/12 PASS.

DELIVERABLES (skill pack 0.5 output contract, 10 items):
  01 cfp_lock.md  02 claim_registry.md  03 proof_audit.md  04 p0_scan.md
  05 number_freeze.md  06 paper_main.pdf  07 paper_appendix.pdf
  08 review_sim.md  09 rebuttal_prep.md  010 submit_checklist.md
  Only the two PDFs may enter the submission package; work files stay local.
  paper_main.pdf = 8 pages (7 main + Broader Impact/References),
  paper_appendix.pdf = 9 pages (cover + Appendices A-J, 34 raw tables).
  split_submission.py locates the split by CONTENT, not a hard-coded page.

CFP LOCK (official, queried 2026-11-07): conference dates 2027-07-31..08-06
and Riocentro, Rio de Janeiro are CONFIRMED. The 2027 Call for Papers,
Author Instructions, submission/abstract deadlines and style file are NOT
published. A third-party tracker's Jan 22 2027 deadline conflicts with the
already-confirmed venue and is NOT adopted; deadline stays UNCONFIRMED.

HONEST STATUS:
  * 60% internal quality gate = NOT EVALUABLE (3 independent blind
    reviewers not available). NOT converted into an acceptance probability.
  * Phase 7 SUSPENDED: 2027 .sty, LLM-use disclosure, reciprocal-reviewer
    nomination, per-reference verification, two-PDF repacking.
  * References were NOT re-verified entry by entry this round; no
    'verified' self-assertion is made.
  * No >=10M-token LM, equal-compute strong baselines, real MQAR
    multi-layer, or TS2SC end-to-end migration.
  * No LM parity / SOTA / throughput / general self-improvement claim.

Checks: desk_check 45/45, fusion_guard 27/27, split 11/11,
governance_gate 12/12, namespace_unique true, Theorem B = 0,
P0 internal = 0, self-identifying paths = 0, meta-address = 0."""

c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"]})
print("\nPUSHED", c["sha"])
print("https://github.com/%s/commit/%s" % (R, c["sha"]))
