"""Push v6.2 to kino-koki/SimplCore@main. Credential comes from the environment."""
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
     "User-Agent": "simplcore-v62",
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

EB = W + "/tools/eval_baselines/"
files = [
    ("simplcore.pdf", W + "/simplcore.pdf"),
    ("build_simplcore_pdf_v5.py", W + "/build_simplcore_pdf_v5.py"),
    ("patch_v62.py", W + "/patch_v62.py"),
    ("patch_v62b.py", W + "/patch_v62b.py"),
    ("V62_REPORT.md", W + "/V62_REPORT.md"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
    ("tools/eval_baselines/ag.py", EB + "ag.py"),
    ("tools/eval_baselines/fair_probe.py", EB + "fair_probe.py"),
    ("tools/eval_baselines/results/fair_baselines.json", EB + "results/fair_baselines.json"),
    ("tools/eval_baselines/results/baseline_budget.json", EB + "results/baseline_budget.json"),
    ("tools/eval_baselines/results/degradation.json", EB + "results/degradation.json"),
    ("tools/eval_baselines/results/fair_adjudication.json", EB + "results/fair_adjudication.json"),
]

tree = []
for path, local in files:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-62s %9d B" % (path, len(raw)))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v6.2: same-run training evidence, degradation curves, and W12 withdrawal

This round supplies the two P1 evidence blocks requested by review, and
reports a REPRODUCTION FAILURE rather than polishing it away.

PART A - SAME-RUN BASELINES (Sec 4.13, main paper, 5 seeds)
  Seven mechanisms trained under ONE self-contained autodiff, Adam, 400 steps,
  batch 64, lr 0.02, width d=24, identical visible ids and external query.
  Parameter counts are recorded, NOT forced equal. Baselines are minimal
  mechanism references in tools/eval_baselines/ (linear attention, gated
  linear attention, one-step delta-rule surrogate, selective diagonal SSM,
  2-layer self+cross softmax attention). They live OUTSIDE the SimplCore
  architecture, which still contains no N x N score and no KV cache; they are
  NOT reproductions of GLA / DeltaNet / Mamba-2 / Transformer.

  TRAINED LAYER BOUND (main evidence): from RANDOM INITIALISATION,
    H=1: 1 layer 0.9996 +/- 0.0008, 2 layers 0.3805 +/- 0.0360
    H=2: 1 layer 0.1055 +/- 0.0180, 2 layers 0.9512 +/- 0.0094
  paired difference tsc_h2 - tsc_h1 at H=2 = +0.8457,
  95% CI [+0.8281, +0.8598], 5/5 seeds positive. The exactly-H-layer bound
  of T2 / Theorem B is therefore NOT an artifact of the zero-training
  analytic construction.

  W12 (FORMAL WITHDRAWAL): the earlier single-hop parity statement
  (Fisher p=0.4998, 1.0000 vs 0.9987) used a STORED attention record. In the
  same-run audit the attention references did not converge: across
  400/1200/3000 steps and lr 0.02/0.10 they stay in [0.0332, 0.3125], near
  the four-pair chance level 0.25. The stored number did NOT reproduce. We
  WITHDRAW the parity statement entirely and make NO claim of equivalence,
  superiority or inferiority with respect to attention. The toy references
  are unoptimised; their low numbers are a LIMITATION OF THIS AUDIT, not
  evidence about published models. All rows are released so the audit can be
  falsified. Abstract, Sec 5, Sec 7, Sec 8, Broader Impact, Sec 4.9 and
  Appendix F are all updated consistently (0 occurrences of the old wording).

PART B - DEGRADATION CURVES (5 seeds, analytic-gradient trainer, 600 steps)
  role-tag noise p = 0/.05/.10/.20/.40 -> 1.0000 / 0.8777 / 0.7797 /
  0.6305 / 0.3625 (monotone). Duplicate keys 0.9328, dup+noise 0.7301.
  Non-orthogonal codes with Gram off-diagonal up to 0.476 still 1.0000,
  whereas RANK COLLAPSE to 16 gives 0.5652 and to 4 gives 0.0781
  (chance 0.0312). Observation (NOT a new theorem): the failure mode is
  RANK LOSS, not loss of orthogonality; a full-rank merely non-orthogonal
  code is learnable. Reported with measured Gram diagnostics. A learnable
  codebook was NOT implemented and is stated as an open item.

PAPER: simplcore.pdf, 25 pages, MAIN PAPER exactly 8 pages (Sec 4.8's long
audit table compressed to make room; no padding). Section numbering
3.1-3.10 / 4.1-4.13 / 5-8 has NO duplicates. Main paper has 0 occurrences
of internal IDs (W9/NT31/17/17/EXACTLY), self-identifying paths, HTML
entities or TeX fragments. Broader Impact before References. Appendix F gains
W10/W11/W12; the stale W9 row claiming 'R3 proves the single remaining bit
NECESSARY and MINIMAL' is corrected to Theorem A(b)'s total-bit bound.

VERIFIERS: same-run protocol 7/7, Theorem A 18/18, Theorem B witnesses
130/130. Every number in the paper and the report is injected from the
result JSON, not typed by hand.

STILL NOT DONE, NOT CLAIMED: >=10M-token equal-compute LM, val perplexity,
generation readability, measured throughput comparison, full reproductions
of published strong baselines, and a learnable codebook. Synthetic content
chains only; no LM claim, no architecture-superiority claim.

No private weight bytes and no credential are part of this commit.
"""
c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"], "force": False})
print("\nPUSHED:", c["sha"])
print("https://github.com/kino-koki/SimplCore/commit/" + c["sha"])
print("PDF sha256:", hashlib.sha256(open(W + "/simplcore.pdf", "rb").read()).hexdigest()[:20])
