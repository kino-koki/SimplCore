"""Push the v6.1 deliverables to kino-koki/SimplCore@main via the Git Data API.

No credential is stored in this file: GH_TOKEN is read from the environment.
No private weights, no token bytes, no generated secrets are included.
"""
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
     "User-Agent": "simplcore-v61",
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

SP = W + "/train/superposition_probe/"
TH = W + "/theory_work/"
files = [
    ("simplcore.pdf", W + "/simplcore.pdf"),
    ("build_simplcore_pdf_v5.py", W + "/build_simplcore_pdf_v5.py"),
    ("patch_v61.py", W + "/patch_v61.py"),
    ("V60_INFO_BOUND_REPORT.md", W + "/V60_INFO_BOUND_REPORT.md"),
    ("V61_REPORT.md", W + "/V61_REPORT.md"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
    ("repro/arxiv_reference_audit.json", W + "/repro/arxiv_reference_audit.json"),
    ("theory_work/info_bound.py", TH + "info_bound.py"),
    ("theory_work/nt31.py", TH + "nt31.py"),
    ("theory_work/results/info_bound_theorem.json", TH + "results/info_bound_theorem.json"),
    ("theory_work/results/info_bound_check.json", TH + "results/info_bound_check.json"),
    ("tools/native_probe/info_bound_probe.py", SP + "info_bound_probe.py"),
    ("tools/native_probe/theoremB_probe.py", SP + "theoremB_probe.py"),
    ("tools/native_probe/results/info_bound_scan.json", SP + "results/info_bound_scan.json"),
    ("tools/native_probe/results/info_bound_depth.json", SP + "results/info_bound_depth.json"),
    ("tools/native_probe/results/info_bound_carriers.json", SP + "results/info_bound_carriers.json"),
    ("tools/native_probe/results/info_bound_adjudication.json", SP + "results/info_bound_adjudication.json"),
    ("tools/native_probe/results/theoremB_results.json", SP + "results/theoremB_results.json"),
]

tree = []
for path, local in files:
    raw = open(local, "rb").read()
    blob = api("/repos/%s/git/blobs" % R, "POST",
               {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    print("  + %-62s %9d B" % (path, len(raw)))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v6.1: Theorem B (bit / serial-depth / state-order conservation), finish the interrupted v6 round

This commit lands the previous round's Theorem A work (it had been written to
disk but never pushed: remote main was still 7158641 / v5.2) and adds a second
core result.

THEOREM B (Sec 4.12, premises + claim + full proof + table + witnesses + scope
all in the MAIN PAPER)
  (a) B < log2|A| makes exact H-hop retrieval impossible for any depth, state
      order or kernel (Theorem A(b)).
  (b) For B >= log2|A| and order-2 state, the computation graph contains a
      dependency chain of exactly H serial content matches. This number is
      invariant to the number of static-LTI layers, kernel parameters and the
      input-side bit budget: hop h+1's query is the value selected by hop h.
  (c) The H matches can live in an H-layer stack (one role bit per token) or in
      ONE LTI layer plus an H-step read-out chain (hop-index annotation); both
      attain 1.0000 at H serial matches and O(H d^2) state.
  (d) Compressing serial depth to 1 requires a state enumerating 2^H endpoints:
      the (H+1)-order witness has d^(H+1) elements, exponential in H. It is a
      lower-bound witness, explicitly NOT a proposed layer.
  (e) The (B, serial depth, state order) frontier is a discrete STEP: input bits
      RELOCATE the H matches, they do not remove them.
  Witnesses 130/130, five seeds, H=1..4, four regimes, exhaustive bit scan.

FIXED IN THIS ROUND (real defects, not polish)
  - Duplicate section numbers 3.7 and 4.8; renumbered to 3.7/3.8/3.9/3.10 and
    4.8/4.9/4.10/4.11/4.12. No duplicates remain.
  - Appendix A.31 still asserted 'exactly one bit, necessary and sufficient'
    unconditionally, contradicting Theorem A(b). Now: one bit/token is a
    SUFFICIENT carrier; the necessary quantity is total B >= log2|A|. Lemma R3
    is retained as the |A|=2 worst-case special case (W11).
  - Theorem A(b) explicit family: replaced the vague orthogonal-code-row
    argument with a clean binary-chain construction (H independent unlabelled
    binary choices -> 2^H distinct endpoints on bit-identical visible inputs).
  - References: 15 entries now carry author names verified against the arXiv
    abs pages (record in repro/arxiv_reference_audit.json). Citing other
    authors does not break double-blind.
  - PDF metadata title synced with the manuscript.

PAPER: simplcore.pdf, 25 pages, MAIN PAPER 8 pages (within the 8-page limit),
Broader Impact before References, no self-identifying paths, no internal IDs
(W9/NT31/17/17), no entity or TeX-fragment residue in the main paper.
Theorem A verifier 18/18; all other verifiers re-run clean.

STILL NOT DONE, NOT CLAIMED: a small, fair, >=5-seed paired-CI same-condition
comparison (MQAR / GLA / DeltaNet / Mamba-2 / 2-layer Transformer); >=10M-token
equal-compute LM, val perplexity, generation readability, measured throughput
comparison. Per review advice these were deliberately NOT added as unfair
large-scale runs; Sec 5 / Sec 7 state so. Synthetic chains only; no LM claim.

No training was run in this round. No private weight bytes, no credential and
no generated secret are part of this commit.
"""
c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"], "force": False})
print("\nPUSHED:", c["sha"])
print("https://github.com/kino-koki/SimplCore/commit/" + c["sha"])
print("PDF sha256:", hashlib.sha256(open(W + "/simplcore.pdf", "rb").read()).hexdigest()[:20])
