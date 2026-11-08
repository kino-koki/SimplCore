"""Push v7 to kino-koki/SimplCore@main. Credential from the environment only."""
import base64
import hashlib
import json
import os
import urllib.request

W = "/data/workspace"
R = "kino-koki/SimplCore"
T = os.environ["GH_TOKEN"]
H = {"Authorization": "Bearer " + T,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v7",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, method="GET", payload=None):
    raw = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request("https://api.github.com" + path, data=raw,
                                 method=method,
                                 headers={**H, **({"Content-Type": "application/json"}
                                                  if raw else {})})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
print("base", head[:12])

files = [
    ("simplcore.pdf", W + "/simplcore.pdf"),
    ("build_simplcore_pdf_v7.py", W + "/build_simplcore_pdf_v7.py"),
    ("expand_v7.py", W + "/expand_v7.py"),
    ("expand2_v7.py", W + "/expand2_v7.py"),
    ("expand3_v7.py", W + "/expand3_v7.py"),
    ("expand4_v7.py", W + "/expand4_v7.py"),
    ("expand5_v7.py", W + "/expand5_v7.py"),
    ("fix_v7.py", W + "/fix_v7.py"),
    ("update_logs_v7.py", W + "/update_logs_v7.py"),
    ("V7_REPORT.md", W + "/V7_REPORT.md"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
    ("MANUSCRIPT_INDEX.md", W + "/MANUSCRIPT_INDEX.md"),
    ("tools/eval_baselines/desk_check_v7.py",
     W + "/tools/eval_baselines/desk_check_v7.py"),
    ("tools/eval_baselines/results/desk_check_v7.json",
     W + "/tools/eval_baselines/results/desk_check_v7.json"),
    ("tools/eval_baselines/fibre_probe.py",
     W + "/tools/eval_baselines/fibre_probe.py"),
    ("tools/eval_baselines/results/fibre.json",
     W + "/tools/eval_baselines/results/fibre.json"),
]

tree = []
for path, local in files:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-64s %8d B" % (path, len(raw)))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v7: full rewrite resolving all seven desk-check defect classes

This is a REWRITE, not a patch. build_simplcore_pdf_v7.py is now the sole
current generator; build_simplcore_pdf_v5.py is retained as the v6.2 historical
generator and no longer produces the submission.

SEVEN DEFECT CLASSES, ALL CLOSED
1. B REDEFINED. Fix a flat reference alphabet Sigma_0 and a role-erasing
   projection pi. B = sum_t b_t, b_t = log2 |pi^(-1)(v_t)|, the total bits
   by which the ACTUAL alphabet exceeds the flat reference. The V -> 2V
   partition's folded bit is now COUNTED in B, so tightness holds formally.
   A(c) is DEMOTED to a within-class-M design statement ("IF one chooses to
   preserve duality, the bits must be supplied on the input side"), with the
   explicit caveat that this negates only the single-FFT path inside M.
2. A(a) IS PER-FIBRE. P5 now defines (i) per-fibre conditional accuracy
   acc(F|v) under the uniform measure on a fixed fibre and (ii) marginal
   accuracy E_v[acc(F|v)]. A(a) states acc(F|v) <= 2^B/|A(v)|. The old
   unconditional "accuracy <= 1/|A| <= 1/2" is gone; Table 2, 6 and 12
   label every number with its statistic, seeds, n and source.
3. ONE VERBATIM SENTENCE, used three times: "The alphabet partition folds
   the mandatory bit into the symbol id: it is not bit-free, it is
   annotation-free." No conflicting "free of run-time annotation" wording.
4. EXACTLY FOUR CORE RESULTS: Result 1 = Theorem A (six parts a-f, with the
   former Theorem B's serial-depth / state-order statements folded into
   (e)); Result 2 = Theorem B (order-2 state + exactly H layers, two tight
   bounds jointly attained); Result 3 = Construction C; Result 4 = Empirical
   E. Everything else is P1-P8 / L1-L3. No Theorem 1/5/6, no T1-T6, no NT*,
   no A.31+, no FOUR/SIX/eight counting conflict.
5. P2 USES AN EXPLICIT THREE-CASE CONSTRUCTION (singular L_j; invertible L_j
   with another nonzero block; invertible L_j with all other blocks zero)
   and states "never infers a target-coordinate perturbation from dim ker L
   > 0 alone". The bare kernel-dimension argument is gone from the whole PDF.
6. ONE LAYER INTERFACE H6: q_(h+1) = T_h z_h / ||T_h z_h||_2, with T*
   = C_key^T C_val as the analytic optimum. "Maps VALUE half to KEY half"
   is the action of T*, not a second interface. P1's oscillatory kernel and
   the strictly-positive real-pole retrieval family are explicitly separated,
   with the reason (P2's positivity premise) stated.
7. ONE NUMERIC CONVENTION. chance = 1/32 = 0.0313 for this task; 0.25 is
   ONLY the historical four-pair task's chance and is never placed beside
   0.0313. Section 4.13's "near chance" wording is deleted. Table 10 is a
   full numeric-provenance audit of every drifted value (0.2969/0.2711,
   0.9316/0.9414/0.9328, the four FFT/IIR errors, chance mixing). Table 7
   drops its duplicate accuracy column; the ablation paragraph no longer
   re-quotes superseded single-seed values.

PRESENTATION / COMPLIANCE
- First-page Table 1 is now a pure "Known result / Precise increment" table;
  all reviewer-directed meta sentences are deleted (0 occurrences).
- References: 14 entries, all with venue + arXiv ID; Zoology restored to
  ICLR 2024; the self-certifying "verified 2026-11-07" sentence deleted.
- Broader Impact precedes References.
- OLD APPENDIX LEDGER DELETED (NT01-NT31, A.32/A.33, 130/130, 23/23,
  51-item ledger, Statement Classification). 25 pages -> 10 pages total,
  main paper 7 pages (within the 8-page limit). Appendices reduced to
  A (complete proofs), B (6-row withdrawal register), C (reproducibility).
- No self-identifying paths, internal IDs, HTML entities or TeX fragments.
- 2027 CFP/deadline, LLM-use disclosure and reciprocal-reviewer obligations
  are flagged in Appendix C as needing recheck against the official template.

P1 ITEMS
- P5 state-order bound promoted from witness to a PROVED CONDITIONAL theorem
  (Appendix A.5): o >= H+1 when each hop offers d mutually orthogonal
  candidate successors; the branching-factor-2 degeneration to a
  log2(H+1) bit requirement is stated explicitly, not hidden.
- Tight witness bridged to the ORIGINAL content-chain task with a new
  reachability enumeration (fibre_probe.py, 5 seeds x 256 samples):
  flat/H=4 gives max |A| = 5 and log2|A| <= 2.3219; codebook gives |A| = 1
  for every H; the partition supplies 8 bits and an explicit key-tag 4 bits.
  The paper states plainly that this budget is SUFFICIENT BUT NOT TIGHT on
  the original task, and that tightness belongs to the binary-chain witness.
- Also added: precise relation to linear attention (conditional containment
  only, no new-paradigm claim), bit/depth/order frontier, full 5x5 grid,
  carrier conservation, depth invariance, full 5-seed statistics, ablation
  ledger, per-statement premise/claim/boundary/non-vacuity table, explicit
  falsification conditions, and a complete reproduction protocol.

VERIFICATION: desk_check_v7.py runs 45 reproducible assertions over the built
PDF, 45/45 PASS. It is an AUTOMATED desk-check, NOT a human reviewer
simulation and NOT a mathematical proof. Section numbering has no duplicates;
only Theorem A and Theorem B are named; all spot-checked numbers match the
raw result JSON exactly. W12 remains withdrawn. No LM training, perplexity,
generation, throughput comparison, converged strong baseline, learnable
codebook, real MQAR run or TS2SC end-to-end migration was performed; all are
listed as open in Sections 6.1 and 7.

Main paper is 7 pages, not 8: filling the eighth would require duplicating
appendix proofs, which would be padding. No acceptance probability is claimed.
No private weight bytes or credentials are part of this commit.
"""
c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"], "force": False})
print("\nPUSHED:", c["sha"])
print("https://github.com/kino-koki/SimplCore/commit/" + c["sha"])
print("PDF sha256:", hashlib.sha256(open(W + "/simplcore.pdf", "rb").read()).hexdigest())
