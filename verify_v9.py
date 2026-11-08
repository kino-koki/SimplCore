#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 acceptance: one check per issue raised in the reader's review."""
import json, fitz

def load(p):
    d = fitz.open(p)
    t = "\n".join(x.get_text() for x in d)
    n = len(d)
    d.close()
    return " ".join(t.split()), t, n

F, _, NF = load("simplcore.pdf")
M, _, NM = load("paper_main.pdf")
A, _, NA = load("paper_appendix.pdf")
checks = []
def ck(name, cond, detail=""):
    checks.append(dict(name=name, pass_=bool(cond), detail=detail))

# 1. positioning / non-tautological claim
ck("title no longer overclaims", "Joint Bounds for Static-LTI Layers" in F and "A Tight Joint Bound" not in F)
ck("abstract: counting condition + LOOSE", "pigeonhole" in M and "LOOSE on the content-chain task" in M)
ck("intro: increment is placement, not inequality", "the increment is the proof of where those bits must be placed" in M)
ck("conclusion: no tight numerical claim on task", "not a tight numerical bound on that task" in F)
ck("supplement cover title unified", "Restated Premises, Complete Proofs and Raw Measurements" in A)

# 2. order reconciliation
ck("design rule no longer says order-(H+1) state", "order-(H+1) state" not in F)
ck("one central reconciliation section", "Reconciliation of the two order bounds" in F)
ck("Result 2 flagged as tensor ORDER", "TENSOR ORDER" in F)
ck("A.5 unified on D, no o*d", "o*d" not in F and "oD" in F)
ck("A.5 kernel condition in full", "dim ker Phi >= (H+1-o)D >= D > 0" in F)
ck("A.5 D>=1 / independence stated as premise", "D >= 1" in F and "part of the premise" in F)
ck("A(e) broken sentence removed", "Earlier drafts headline result" not in F)

# 3. exactly H layers is conditional
ck("depth bound called interface-dependent", "not an architecture-independent law" in F)
ck("H_L >= H sufficiency stated", "every H_L >= H sufficient" in F)

# 4. experiment honesty
ck("abstract: inter-layer map FIXED", "inter-layer map FIXED" in M and "0/5 seeds converged" in M)
ck("Table 14 reports learned-random number", "0.2492" in F and "NOT evidence that the inter-layer map is learned" in F)
ck("SE2 CI width quantified", "about 73% of the mean" in F)
ck("SE2 agreement has no discriminative power", "almost no discriminative power" in F)
ck("SE2 premise-fixed direction kept", "premise was fixed to i.i.d. arrival BEFORE" in F)

# 5. editing
ck("Table 0 rebuilt as clean two columns", "bits per insight per round (CT7; distinct from B)" in F and "successor rank per hop, tensor order (P5)" in F)
ck("Tables 18-19 now point to Appendix D", "Tables 18-19 in Appendix D" in F)
ck("no stale A.7-A.11 xref in 4.13", "Appendix A.7-A.11. These are" not in F)
ck("W10/W11 split into two rows", "W10/W11" not in F and "'Exactly one bit per token is NECESSARY'" in F)
ck("sparse withdrawal IDs explained", "Gaps in the sequence are retired entries" in F)
ck("no CFP / author to-do", all(k not in F for k in ["2027", "call for papers", "reciprocal", "LLM-use disclosure", "rechecked against"]))
ck("no reviewer-directed meta", "invite the community to falsify" not in F and "A reviewer" not in F)

# 6. supplement genuinely self-contained
ck("A.0 restates A1-A5", "A1-A5 (premises, restated)" in A)
ck("A.0 restates H6", "H6 (unique layer interface, restated)" in A)
ck("A.0 restates four core results", "Four core results (restated)" in A)
ck("A.0 gives complete CT7 (a)(b)(c)", "CT7 (zero-order self-evolution, complete statement)" in A and "instantaneous bandwidth" in A)
ck("cover claims only what is true", "Appendix A.0 restates every" in A)
ck("Appendix G compressed to checkpoints", "Fusion Evolution Curves at Checkpoints" in F and "40-round checkpoints" in F)
ck("all appendices A-J present", all(("Appendix " + c) in A for c in "ABCDEFGHIJ"))

# 7. compliance / layout
ck("no self-identifying paths", not any(k in F for k in ["/data/workspace", ".py", ".json", "results/", "github.com", "kino-koki"]))
ck("no entities / TeX fragments", not any(k in F for k in ["&lt;", "&gt;", "_{", "^{"]))
ck("no credential literal", not any(k in F for k in ["gh" + "p_", "github" + "_pat_"]))
ck("main paper within 8 text pages", NM <= 9)
ck("merged PDF pages", NF == 18, "18")
ck("appendix PDF pages", NA == 10, "10")

# geometry + table rule no longer crosses last row
d = fitz.open("simplcore.pdf")
oob = tiny = 0
for p in d:
    for b in p.get_text("dict")["blocks"]:
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                x0, y0, x1, y1 = s["bbox"]
                if x0 < 18 or y0 < 12 or x1 > 594 or y1 > 780: oob += 1
                if s["size"] < 5.9: tiny += 1
p2 = d[1]
rules = sorted(round(it[1][1], 1) for dr in p2.get_drawings()
               for it in dr["items"]
               if it[0] == "l" and abs(it[1][1]-it[2][1]) < 0.6 and it[1][0] < 310)
last_row_y1 = max(s["bbox"][3] for b in p2.get_text("dict")["blocks"]
                  if "lines" in b for l in b["lines"] for s in l["spans"]
                  if s["text"].strip() in ("M", "number of poles")
                  and 470 < s["bbox"][1] < 492)
ck("geometry: 0 out-of-page spans", oob == 0, str(oob))
ck("geometry: 0 sub-6pt spans", tiny == 0, str(tiny))
ck("Table 0 bottom rule below last row", rules and rules[-1] > last_row_y1,
   "rule %.1f vs text %.1f" % (rules[-1] if rules else -1, last_row_y1))

n_pass = sum(c["pass_"] for c in checks)
out = {"version": "v9", "n_checks": len(checks), "n_pass": n_pass,
       "all_pass": n_pass == len(checks), "checks": checks,
       "pages": {"merged": NF, "main": NM, "appendix": NA}}
json.dump(out, open("v9_review_check.json", "w"), indent=2)
for c in checks:
    print("  %-46s %s" % (c["name"], "PASS" if c["pass_"] else "FAIL"))
print("\n%d/%d PASS" % (n_pass, len(checks)))
