"""
push_rounds.py -- one incremental commit on the CURRENT remote HEAD.

Pushes two previously-unpushed rounds at once:
  * v7.1 submission-governance package (was blocked by a missing credential)
  * v7.2 competitiveness roadmap + G1 first real fill

Credential is read ONLY from GH_TOKEN (environment); it is never printed,
never written into the commit, and never included in any pushed file.

Run: GH_TOKEN=$(cat .gh_token) python3 push_rounds.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

W = "/data/workspace"
R = "kino-koki/SimplCore"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set.")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v72-push",
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
        sys.exit("API %s %s -> HTTP %d: %s" %
                 (method, path, e.code, e.read().decode()[:400]))


T = W + "/theory_work"
TR = W + "/train/superposition_probe/results"
ER = W + "/tools/eval_baselines/results"

# ---------------- v7.1: governance package (pending since last round) -----
V71 = [
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
    ("tools/eval_baselines/results/governance_gate.json",
     ER + "/governance_gate.json"),
    ("tools/eval_baselines/results/desk_check_v7.json",
     ER + "/desk_check_v7.json"),
    ("tools/eval_baselines/results/split_check.json", ER + "/split_check.json"),
    ("theory_work/results/governance_s4.json", T + "/results/governance_s4.json"),
    ("theory_work/results/governance_namespace_before.json",
     T + "/results/governance_namespace_before.json"),
    ("theory_work/results/governance_namespace_after.json",
     T + "/results/governance_namespace_after.json"),
    ("theory_work/results/governance_number_freeze.json",
     T + "/results/governance_number_freeze.json"),
    ("train/superposition_probe/results/fusion_guard.json",
     TR + "/fusion_guard.json"),
]

# ---------------- v7.2: competitiveness roadmap + G1 fill -----------------
V72 = [
    ("创世级架构路线图.md", W + "/创世级架构路线图.md"),
    ("competitiveness_gate.py", W + "/competitiveness_gate.py"),
    ("gate_results.json", W + "/gate_results.json"),
    ("bit_allocator_probe.py", W + "/bit_allocator_probe.py"),
    ("bit_allocator_results.json", W + "/bit_allocator_results.json"),
    ("bit_allocator_run.log", W + "/bit_allocator_run.log"),
    ("SimplCore_竞争力路线与G1填充包.zip",
     W + "/SimplCore_竞争力路线与G1填充包.zip"),
    ("train/superposition_probe/results/convergence_curve.json",
     TR + "/convergence_curve.json"),
    ("train/superposition_probe/results/convergence_matrix.json",
     TR + "/convergence_matrix.json"),
    ("train/superposition_probe/results/fusion_evolution.json",
     TR + "/fusion_evolution.json"),
    ("SMM_SGC_FUSION_THEORY.md", W + "/SMM_SGC_FUSION_THEORY.md"),
    ("SMM_SGC_SELF_EVOLUTION.md", W + "/SMM_SGC_SELF_EVOLUTION.md"),
]

FILES = V71 + V72

# hard guard: never let the credential file become a tree entry
assert not any(".gh_token" in p for p, _ in FILES)
assert not any("token" in os.path.basename(p).lower() and p.endswith(".md")
               for p, _ in FILES)

missing = [p for _, p in FILES if not os.path.isfile(p)]
if missing:
    sys.exit("Missing local files:\n  " + "\n  ".join(missing))

# local content hash (git blob sha1) for post-push verification
def blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
print("base =", head[:12], "| uploading", len(FILES), "files",
      "(v7.1: %d, v7.2: %d)" % (len(V71), len(V72)))

tree = []
manifest = []
for path, local in FILES:
    raw = open(local, "rb").read()
    want = blob_sha(raw)
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    if b["sha"] != want:
        sys.exit("blob sha mismatch for %s" % path)
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    manifest.append(dict(path=path, bytes=len(raw), git_blob=b["sha"],
                         sha256=hashlib.sha256(raw).hexdigest()))
    print("  + %-64s %8d B  %s" % (path, len(raw), b["sha"][:10]))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})

msg = """v7.1 governance + v7.2 competitiveness: push the two pending rounds

Both rounds were completed locally but never pushed: v7.1 lacked a
credential; v7.2 was delivered as files only. This is ONE incremental
commit on the previous HEAD; history is not rewritten. No training,
no new theorem, no new large-scale experiment.

--- v7.1: ICML 2027 submission-governance workflow (Phase 0-5) ---
Four real P0 defects fixed: Theorem A premises P1-P5 collided with
supporting propositions P1-P8 (renamed TAP1-TAP5, disjoint namespace);
'Theorem B' had two meanings (removed, 0 occurrences); Section 3.1(c)
and Table 1 cited P1 for an argument that uses P6; full proofs duplicated
in main text and Appendix were diverging (removed 5.9k chars from main).
Architecture: theory_work/governance.py makes B = sum_t log2|pi^-1(v_t)|
computable, asserts the lower bound B >= log2|A| and the tightness witness
|A| <= 2^B use the SAME quantity (S4 PASS); a v6-definition counterfactual
produces 4 contradictions. verify_governance.py -> 12/12 PASS.
Deliverables: 10 skill-pack artifacts; paper_main.pdf 8 pages,
paper_appendix.pdf 9 pages, split by CONTENT. Desk-check 45/45,
fusion_guard 27/27, split 11/11.
Official CFP status (queried 2026-11-07): 2027-07-31..08-06 and Riocentro,
Rio de Janeiro CONFIRMED; Call for Papers, Author Instructions, deadlines
and style file NOT published. Third-party deadline trackers not adopted.

--- v7.2: competitiveness roadmap + G1 first real fill ---
Defines 'genesis-level' as five falsifiable criteria C1-C5 (orthogonal
freedom, real-compute capability, transferability, scale, hardware).
External landscape: 2026 hybrids keep 10-25% softmax attention; Zoology
attributes ~82% of the recall gap to associative recall; the Impossibility
Triangle proves Efficiency+Compactness+Recall cannot all hold, with a
recall ceiling O(poly(d)/log V) over 52 architectures; KATA, Log-Linear
Attention, Gated DeltaNet, RWKV-7 and Mamba-3 each optimise a DIFFERENT
axis. The open axis identified here: where information is placed in the
sequence, and its input-side bit budget B.
Seven gates G1-G7 with machine-checkable verdicts and pre-written
falsification conditions: gate_results.json = 6 NOT MET / 1 PARTIAL
(G6). NOT MET is never reported as PARTIAL.
G1 real run (bit_allocator_probe.py, 5 seeds x 4 arms x 2 hops): an
input-side learned allocator g_t = sigmoid(w . phi_t) modulates ONLY the
input signal, leaving the system operator unchanged. Result:
protocol_valid = FALSE, so G1 = NOT MET (protocol invalid), not PASS.
  none flat, no gate:            H=1 1.0000  H=2 0.9719
  tok gate (current token):      H=1 1.0000  H=2 0.9422
  ctx gate (current+predecessor):H=1 1.0000  H=2 0.9547
  oracle codebook + role gate:   H=1 0.2371  H=2 0.1484
In the flat alphabet a VALUE id is itself a valid KEY id, so the chain
closes by itself and no |A|>=2 ambiguity fibre exists; the required bit
is never demanded. The oracle arm is also invalid (no codebook-swap M).
A v1 gate-indexing bug (using the current position's gate on the
predecessor signal) was found and fixed with a trace. Learned gates have
I(role;g) ~ 0 yet reach 1.0000, a counterexample clue to bit
internalisation. F3 not satisfied, F1 not applicable, F2 not evaluable,
F4 not triggered.
SOLID POSITIVE RESULT: modulating only the input signal preserves every
SimplCore invariant -- Toeplitz deviation 0, FFT<->Toeplitz rel err
2.9e-16..4.3e-16, future-perturbation changed bits 0, state independent
of L, gradcheck all pass (guards_all_pass=True). This is the architecture
licence to continue. ctx gradcheck initially reported 0.0217; traced to
central-difference round-off when analytic gradients are ~1e-8 (e=1e-4
gives 2e-4); threshold NOT relaxed, magnitudes recorded.
G1-v2 validity gate written down BEFORE the next run: construct an
|A|>=2 fibre without baking the answer into the alphabet, then require
none < 0.5 AND oracle >= 0.99 before measuring I(role;g). Otherwise G1
is falsified and the roadmap moves to G2 (B as an external schedulable
resource).

HONEST STATUS (unchanged):
  * 60% quality gate = NOT EVALUABLE (no 3 independent blind reviewers).
    NOT converted into an acceptance probability.
  * Phase 7 SUSPENDED: 2027 .sty, LLM-use disclosure, reciprocal
    reviewer nomination, per-reference verification, PDF repacking.
  * G2-G7 NOT MET. No >=10M-token LM, equal-compute strong baselines,
    real MQAR, fused kernel, real-sequence self-evolution, or TS2SC
    end-to-end migration.
  * No LM parity / SOTA / throughput / general self-improvement claim.
  * The credential is environment-only and is NOT part of this commit."""

c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"]})

# ---- post-push verification: read the tree back and compare blob shas ----
back = api("/repos/%s/git/trees/%s?recursive=1" % (R, nt["sha"]))
remote = {e["path"]: e["sha"] for e in back.get("tree", []) if e["type"] == "blob"}
mism = [m["path"] for m in manifest if remote.get(m["path"]) != m["git_blob"]]

report = dict(
    pushed_at=datetime.now().isoformat(timespec="seconds"),
    repository=R, base=head, commit=c["sha"],
    commit_url="https://github.com/%s/commit/%s" % (R, c["sha"]),
    n_files=len(manifest), n_v71=len(V71), n_v72=len(V72),
    total_bytes=sum(m["bytes"] for m in manifest),
    remote_tree_verified=not mism, mismatches=mism,
    credential_in_commit=False,
    files=manifest,
)
with open(W + "/push_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\nPUSHED", c["sha"])
print("https://github.com/%s/commit/%s" % (R, c["sha"]))
print("remote tree verified:", not mism, "| files:", len(manifest),
      "| bytes:", report["total_bytes"])
