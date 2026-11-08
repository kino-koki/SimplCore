"""
push_v72.py -- incremental push of the v7.2 round to kino-koki/SimplCore@main.

    export GH_TOKEN=<token with contents:write>
    python3 push_v72.py

Reads the credential ONLY from the environment; never prints or persists it.
Builds a tree on the CURRENT remote HEAD (no history rewrite).
"""
import base64, hashlib, json, os, sys, urllib.request

W = "/data/workspace"
R = "kino-koki/SimplCore"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set. Export a token with contents:write and rerun.")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v72",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, method="GET", payload=None):
    raw = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("https://api.github.com" + path, data=raw,
                                 method=method,
                                 headers={**H, **({"Content-Type": "application/json"}
                                                  if raw else {})})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


TR = W + "/train/superposition_probe/results"
ER = W + "/tools/eval_baselines/results"
TH = W + "/theory_work/results"
TS = W + "/ts2sc_results"

F = [
 ("paper_main.pdf", W + "/paper_main.pdf"),
 ("paper_appendix.pdf", W + "/paper_appendix.pdf"),
 ("simplcore.pdf", W + "/simplcore.pdf"),
 ("V72_REPORT.md", W + "/V72_REPORT.md"),
 ("tighten_experiments.py", W + "/tighten_experiments.py"),
 ("ts2sc_numpy_migrate.py", W + "/ts2sc_numpy_migrate.py"),
 ("patch_v72.py", W + "/patch_v72.py"),
 ("patch_v72b.py", W + "/patch_v72b.py"),
 ("build_simplcore_pdf_v7.py", W + "/build_simplcore_pdf_v7.py"),
 ("build_simplcore_pdf_v7_pre72.py", W + "/build_simplcore_pdf_v7_pre72.py"),
 ("theory_work/governance.py", W + "/theory_work/governance.py"),
 ("tools/eval_baselines/desk_check_v7.py", W + "/tools/eval_baselines/desk_check_v7.py"),
 ("verify_governance.py", W + "/verify_governance.py"),
 ("split_submission.py", W + "/split_submission.py"),
 ("cfp_lock.md", W + "/cfp_lock.md"),
 ("claim_registry.md", W + "/claim_registry.md"),
 ("proof_audit.md", W + "/proof_audit.md"),
 ("p0_scan.md", W + "/p0_scan.md"),
 ("number_freeze.md", W + "/number_freeze.md"),
 ("review_sim.md", W + "/review_sim.md"),
 ("rebuttal_prep.md", W + "/rebuttal_prep.md"),
 ("submit_checklist.md", W + "/submit_checklist.md"),
 ("GOVERNANCE_SUMMARY.md", W + "/GOVERNANCE_SUMMARY.md"),
 ("CHANGELOG.md", W + "/CHANGELOG.md"),
 ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
 ("MANUSCRIPT_INDEX.md", W + "/MANUSCRIPT_INDEX.md"),
 ("tools/eval_baselines/results/tight_freeze.json", ER + "/tight_freeze.json"),
 ("tools/eval_baselines/results/tight_audit.json", ER + "/tight_audit.json"),
 ("tools/eval_baselines/results/desk_check_v7.json", ER + "/desk_check_v7.json"),
 ("tools/eval_baselines/results/governance_gate.json", ER + "/governance_gate.json"),
 ("tools/eval_baselines/results/split_check.json", ER + "/split_check.json"),
 ("theory_work/results/governance_namespace_after.json", TH + "/governance_namespace_after.json"),
 ("theory_work/results/governance_s4.json", TH + "/governance_s4.json"),
 ("train/superposition_probe/results/fusion_guard.json", TR + "/fusion_guard.json"),
 ("ts2sc_results/ts2sc_migration.json", TS + "/ts2sc_migration.json"),
]

missing = [p for _, p in F if not os.path.isfile(p)]
if missing:
    sys.exit("Missing:\n  " + "\n  ".join(missing))

head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
print("base", head[:12], "| uploading", len(F), "files")

tree = []
for path, local in F:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-62s %8d B sha256=%s" %
          (path, len(raw), hashlib.sha256(raw).hexdigest()[:12]))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v7.2: tighten all experiments, NumPy TS2SC migration, N1-N7 revision

Executed in the user-specified order: (1) collect every experiment to its
tightest form, (2) TS2SC migration, (3) revise the paper per the detailed
v7 revision order. Training remains frozen; no new theorem, no LM training.

STEP 1 -- tighten_experiments.py (pure NumPy, CPU).
  Exhaustive fibre enumeration m=1..4 x B=0..m+1 (18 rows, all satisfy
  acc <= 2^B/|A|); SGC conservation FFT/IIR 9.05e-16, Toeplitz true,
  causal past deviation 0.0, state independent of L; paired bootstrap
  H=2 two-vs-one layer +0.8191, 95% CI [0.8000, 0.8375], 5/5 positive;
  exactly 2 carriers counted in B; 21-entry single-source freeze table.
  Audit 14/14 PASS. Crucially, the 23 occurrences of 0.0312 in the paper
  are mostly REAL 16/512 accuracies, not the chance label: only the
  explicit "chance 0.0312" annotation was changed to 0.0313.

STEP 2 -- ts2sc_numpy_migrate.py (NO torch / safetensors / transformers /
  GPU). Parses the safetensors header directly, decodes BF16 by bit shift.
  On the REAL 2,000,000 B Qwen3 teacher prefix: 310 tensors in header,
  256 real embed rows read; Stage A COPY bitwise equal; Stage B DERIVE
  bitwise equal and deterministic; Stage C SGC dual on real teacher rows
  (Toeplitz true, causal deviation 0.0, O(1) state). FFT/IIR error
  1.87e-08 from the BF16->float32 view and 1.14e-15 after explicit float64
  promotion; both recorded, neither hidden.
  MiniCPM-V-4.7-1B (the user's "MiniCMP-V 4.7") is a PRIVATE VL repo:
  anonymous 404, egress credential 401, GH_TOKEN empty, ZERO weight bytes
  in sandbox. Only a Stage-0 provenance PLAN is delivered, explicitly
  marked "no COPY/DERIVE executed". End-to-end logit-KL / ppl / generation
  NOT run. The NumPy engine removes the torch dependency; the remaining
  blockers are the weight bytes and the credential.

STEP 3 -- N1-N7 of the revision order, string/table/wording only.
  N1: Theorem A premises TAP1-TAP5 -> A1-A5 (per the order, not the
      previous round's TAP naming); P1-P8 reserved for propositions;
      bare P5(i)/P5(ii) cleared; dependency map Proposition prefixes;
      dangling "positivity premise of P2" removed; Theorem A(b)/A(c)
      prefixes. A1-A5 23 hits, TAP 0, Theorem B 0, ambiguity 0.
  N2: two bit carriers + one support restriction everywhere; Table 7
      third row removed (data retained for Table 28); caption fixed.
  N3: title "An Input-Side Information Bound, Tight on a Constructible
      Family, ..."; abstract/conclusion explicitly say the bound is loose
      on the content-chain task (|A|<=5, log2|A(v)|<=2.3219); log2|A|
      unified to log2|A(v)|.
  N4: [4] corrected to the REAL S5 paper (Smith, Warrington, Linderman,
      arXiv:2208.04933, 2022) -- the old entry misattributed Gu et al.'s
      S4D title to the S5 arXiv id. ADDITIONALLY found and fixed [11]:
      CKConv's real arXiv id is 2102.02611 (old: 2202.06037), ICLR 2022.
      Verified online: S5, S4D=2206.11893, DeltaNet=NeurIPS 2024,
      GLA=ICML 2024, Mamba-2=ICML 2024, CKConv. [10] Hyena venue did NOT
      get an independent authoritative page this round; left as-is and
      listed for pre-deadline recheck, NOT claimed as verified.
  N5: removed the contradictory "unnormalised product" sentence; one
      T* = C_key^T C_val; R^(d x d) -> R^(D x D).
  N6: d split into D (interface/width/rank) and c (branching factor);
      o*D <= H*D; c^(H+1).
  N7: chance annotation 0.0313; removed 0.9609 (not a Table 2 value);
      de-duplicated the identical explicit/gapped Table 12 row without
      changing any value; neutral generative-AI disclosure restored in
      Broader Impact with a 2027-CFP caveat; removed the "7/7 gates pass"
      count from the main text.

PDF: 17 pages total, main body 8 pages (ICML 2026 limit), page 9 is
Broader Impact + References. paper_main.pdf 9 pages, paper_appendix.pdf
9 pages. The one extra page comes ONLY from the sentences the revision
order explicitly requires; no padding.

CHECKS: desk_check 46/46 (new anti-hallucination [4] assertion),
verify_governance 12/12, split 11/11, namespace unique, fusion guard
27/27, tighten audit 14/14, P0=0, veto=0, self-identifying paths 0.

HONEST STATUS:
  * internal quality gate still NOT EVALUABLE (3 independent blind
    reviewers unavailable). NOT converted to an acceptance probability.
  * No >=10M-token LM, equal-compute strong baselines, real MQAR
    multi-layer, TS2SC end-to-end migration, 2027 .sty / reciprocal
    reviewer / two-PDF repacking.
  * No LM parity / SOTA / throughput / general self-improvement claim."""

c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"]})
print("\nPUSHED", c["sha"])
print("https://github.com/%s/commit/%s" % (R, c["sha"]))
