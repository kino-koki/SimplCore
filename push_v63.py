"""Push v6.3 to kino-koki/SimplCore@main. Credential from environment only."""
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
     "User-Agent": "simplcore-v63",
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
    ("patch_v63.py", W + "/patch_v63.py"),
    ("patch_v63b.py", W + "/patch_v63b.py"),
    ("patch_v63c.py", W + "/patch_v63c.py"),
    ("patch_v63d.py", W + "/patch_v63d.py"),
    ("V63_REPORT.md", W + "/V63_REPORT.md"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
    ("tools/eval_baselines/fair_probe2.py", EB + "fair_probe2.py"),
    ("tools/eval_baselines/results/corrected_baselines.json",
     EB + "results/corrected_baselines.json"),
    ("tools/eval_baselines/results/analytic_equivalence.json",
     EB + "results/analytic_equivalence.json"),
    ("tools/eval_baselines/results/analytic_h2.json",
     EB + "results/analytic_h2.json"),
    ("tools/eval_baselines/results/corrected_adjudication.json",
     EB + "results/corrected_adjudication.json"),
]

tree = []
for path, local in files:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-64s %9d B" % (path, len(raw)))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v6.3: audit our own fairness claim, withdraw W13, give analytic equivalence witnesses

WHAT WE AUDITED
v6.2 stated that every compared arm received "exactly the same one-hot
visible ids". True for the ids, FALSE for the computation. Two asymmetries:

  1. the predecessor view Xp was consumed ONLY by the TSC branch; all
     baselines used U = X @ E alone;
  2. shared width d=24 < V=64, with a LEARNED random embedding E, gives
     every linear embedding a null space, while TSC retained a hard-coded
     (predecessor -> current) pairing bias.

v6.3 removes both: all eight arms receive (U, Up, q, present-mask) with ONE
fixed full-width orthonormal Hadamard embedding (d=V=64, nothing learned).

W13 -- FORMAL WITHDRAWAL
Removing the asymmetries REVERSES v6.2's headline training result:
  v6.2: tsc_h2 0.9512, paired +0.8457 CI [+0.8281,+0.8598]
  v6.3: tsc_h2 0.3625 +/- 0.0809 vs attn_x2 0.9543 +/- 0.0315,
        paired -0.5918, 95% CI [-0.6500,-0.5012], 5/5 seeds negative.
We WITHDRAW the claim that the exactly-H-layer bound was corroborated by
end-to-end training from random initialisation. Theorem B ITSELF IS NOT
WITHDRAWN: it is a constructive existence statement with an analytic proof
under stated premises (Sec 3.6, Appendix A). Only the training corroboration
is withdrawn. Abstract, Sec 5, Sec 8 and Appendix F are updated.

NEW POSITIVE EVIDENCE -- EXPLICIT ZERO-TRAINING WITNESSES
On the SAME corrected data, with NO learned parameters:
  H=1: static-LTI layer 1.0000, linear attention 1.0000,
       single-layer cross-attention 1.0000 (5 seeds each)
  H=2: two-layer static-LTI stack 1.0000, two-layer attention stack 1.0000
Construction: Wk=Wv=Wq=I, Wout=C[V_K:,:]^T,
  hop(q) = sum_s (q . Up_s) U_s K(tau) [static LTI]
         = sharpened softmax(q . Up_s / sqrt(d)) . U_s [attention],
  M = C[V_K:,:]^T C[:V_K,:] maps the value code to the next key code,
  q2 = normalise(hop(q) @ M), answer = argmax hop(q2) @ CVAL^T.
Reading: the two function classes COINCIDE on this task. The training gap is
OPTIMISATION / INDUCTIVE BIAS, not expressivity. This is an observation with
an explicit witness, NOT a new theorem, and NOT a superiority claim in either
direction. It upgrades T6's rho=1 single-pole isomorphism from an open
attitude to an explicit witness, and pre-empts the review question
"isn't this linear attention with extra steps".

W12 RETAINED, REASON SHARPENED
The stored 0.9987 attention record still has NO verifiable same-run
reproduction, so the Fisher-parity statement remains withdrawn. But the
earlier explanation was imprecise: v6.2's low reference numbers came from an
UNFAIR IMPLEMENTATION, not from the published mechanisms. No claim of
equivalence, superiority, inferiority or expressivity advantage with respect
to attention is made anywhere.

PAPER: simplcore.pdf, 25 pages, MAIN PAPER exactly 8 pages. Sec 4.13 fully
rewritten and shortened (no padding); heading clipped in v6.3 draft is
fixed; ALL-CAPS normalised to standard prose; table header disambiguated to
"witness"; section numbering 3.1-3.10 / 4.1-4.13 / 5-8 has NO duplicates;
main paper has 0 occurrences of internal IDs, self-identifying paths, HTML
entities or TeX fragments. Broader Impact before References.

VERIFIERS: corrected protocol 7/7, Theorem A 18/18, Theorem B witnesses
130/130, H=1 analytic 3/3 exact, H=2 analytic 2/2 exact. Every number in the
paper and report is injected from the result JSON.

PURITY: baseline references live ONLY in tools/eval_baselines/ (independent
evaluation directory). The SimplCore architecture code is UNCHANGED; active
model code still has no N x N score and no KV cache. v6.2 probes and results
are retained as the evidence chain, not overwritten.

STILL NOT DONE, NOT CLAIMED: >=10M-token equal-compute LM, val perplexity,
generation readability, measured throughput comparison, full reproductions
of published strong baselines, learnable codebook. Synthetic content chains
only; no LM claim, no architecture-superiority claim, no acceptance-probability
claim. ssm_pair stays near chance (0.033-0.041) and is NOT tuned; recorded
as-is.

No private weight bytes and no credential are part of this commit.
"""
c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"], "force": False})
print("\nPUSHED:", c["sha"])
print("https://github.com/kino-koki/SimplCore/commit/" + c["sha"])
print("PDF sha256:", hashlib.sha256(open(W + "/simplcore.pdf", "rb").read()).hexdigest()[:20])
