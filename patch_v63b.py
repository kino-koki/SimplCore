"""v6.3b: fix one syntax error and two edits skipped by a too-loose idempotence check."""
import subprocess
import sys

G = "/data/workspace/build_simplcore_pdf_v5.py"
s = open(G, encoding="utf-8").read()

# --- 1. syntax: % must follow the string, not be a separate argument
old = '''      "analytic proof under stated premises (Sec 3.6, Appendix A).",
      % (_cb(2, "tsc_h2")[0], _cb(2, "attn_x2")[0], _ci["mean"],
         _ci["ci95"][0], _ci["ci95"][1]),
      size=SMALL, lead=SLEAD, after=2.0)'''
new = '''      "analytic proof under stated premises (Sec 3.6, Appendix A)." %
      (_cb(2, "tsc_h2")[0], _cb(2, "attn_x2")[0], _ci["mean"],
       _ci["ci95"][0], _ci["ci95"][1]),
      size=SMALL, lead=SLEAD, after=2.0)'''
assert s.count(old) == 1, ("syntax", s.count(old))
s = s.replace(old, new)
print("fixed % operator position")

# --- 2. Sec 5 second bullet
old = '''       "Section 4.13 supplies same-condition TOY mechanism references, not "
       "full reproductions of GLA / DeltaNet / Mamba-2 / Transformer; they "
       "are a reproduction audit, not a performance ranking. Earlier O1-O4 "
       "lack >=5-seed paired CIs; the Section 4.13 results do have them.",
       size=SMALL, lead=SLEAD)'''
new = '''       "Section 4.13 supplies same-condition TOY mechanism references, not "
       "full reproductions of GLA / DeltaNet / Mamba-2 / Transformer; they "
       "are a FAIRNESS / reproduction audit, not a performance ranking. "
       "Earlier O1-O4 lack >=5-seed paired CIs; the Section 4.13 results "
       "do have them.",
       size=SMALL, lead=SLEAD)'''
if "FAIRNESS / reproduction audit" not in s:
    assert s.count(old) == 1, ("sec5", s.count(old))
    s = s.replace(old, new)
    print("fixed Sec 5 second bullet")
else:
    print("Sec 5 already fixed")

# --- 3. Appendix F: sharpen W12 reason and add W13 row
old = '''        ["W12: single-hop parity with attention (Fisher p=0.4998)", "WITHDRAWN: the 0.9987 baseline came from a stored record and did not reproduce in the same-run audit of Sec 4.13. No equivalence or superiority claim is made."],
    ], [130, 108], size=7.6, lead=9.4)'''
new = '''        ["W12: single-hop parity with attention (Fisher p=0.4998)", "WITHDRAWN: the 0.9987 baseline came from a stored record with no verifiable same-run reproduction. No equivalence or superiority claim is made."],
        ["W13: v6.2 trained layer bound (0.9512, paired +0.8457)", "WITHDRAWN: that comparison gave the predecessor view Xp ONLY to the TSC branch and used d=24 < V=64. After removing BOTH asymmetries, tsc_h2 = 0.3625 vs attn_x2 = 0.9543 (paired -0.5918). Theorem B is a constructive existence statement and is NOT withdrawn; only the end-to-end-training corroboration is. Explicit zero-training witnesses for H=1 and H=2 are all 1.0000 for BOTH mechanisms, so the gap is optimisation / inductive bias, not expressivity."],
    ], [130, 108], size=7.6, lead=9.4)'''
if "W13: v6.2 trained layer bound" not in s:
    assert s.count(old) == 1, ("appendixF", s.count(old))
    s = s.replace(old, new)
    print("fixed Appendix F (W12 sharpened + W13 added)")
else:
    print("Appendix F already fixed")

open(G, "w", encoding="utf-8").write(s)
r = subprocess.run([sys.executable, G], cwd="/data/workspace",
                   capture_output=True, text=True)
print((r.stdout or "")[-800:])
if r.returncode:
    print((r.stderr or "")[-2500:])
    sys.exit(1)
print("REBUILD OK")
