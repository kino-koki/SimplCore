"""v6.2b: fix two stale sentences that the idempotence guard skipped."""
import subprocess
import sys

G = "/data/workspace/build_simplcore_pdf_v5.py"
s = open(G, encoding="utf-8").read()

old1 = '''    P("Every entry below is a 5-seed mean with a 10,000-resample paired "
      "bootstrap CI over 512 held-out samples per seed. We report "
      "'no significant difference' as what the statistic supports; p-values "
      "are NOT equivalence proofs. The baseline for the earlier parity run "
      "came from a historical record file, not a same-run re-execution, and "
      "we keep that limitation visible.", size=SMALL, lead=SLEAD, after=2.0)'''
new1 = '''    P("Every entry below is a 5-seed mean with a 10,000-resample paired "
      "bootstrap CI over 512 held-out samples per seed. We report "
      "'no significant difference' only for comparisons WITHIN this table; "
      "p-values are NOT equivalence proofs. The earlier attention-parity "
      "claim that relied on a stored record is WITHDRAWN (W12, Sec 4.13).",
      size=SMALL, lead=SLEAD, after=2.0)'''
assert s.count(old1) == 1, ("old1", s.count(old1))
s = s.replace(old1, new1)
print("fixed stale sentence 1 (Sec 4.9)")

old2 = '''    P("<b>What we deliberately did NOT do.</b> We did not add a >=10M-token "
       "equal-compute comparison, same-run Mamba-2 / GLA / DeltaNet / Hyena / "
       "Transformer baselines, or a throughput comparison. An unfair "
       "large-scale baseline would be worse than none for a theory paper. "
       "Small, clean, seeded comparisons plus an explicit 'no LM claim' are "
       "the honest scope here.", size=SMALL, lead=SLEAD)'''
new2 = '''    P("<b>What we deliberately did NOT do.</b> We did not add a >=10M-token "
       "equal-compute comparison or a throughput comparison. Section 4.13 "
       "adds small same-run TOY mechanism references for a reproduction "
       "audit; they are NOT reproductions of Mamba-2 / GLA / DeltaNet / "
       "Hyena / Transformer and are NOT a performance ranking. An unfair "
       "large-scale baseline would be worse than none for a theory paper. "
       "Small, clean, seeded comparisons plus an explicit 'no LM claim' are "
       "the honest scope here.", size=SMALL, lead=SLEAD)'''
assert s.count(old2) == 1, ("old2", s.count(old2))
s = s.replace(old2, new2)
print("fixed stale sentence 2 (Sec 7)")

open(G, "w", encoding="utf-8").write(s)
r = subprocess.run([sys.executable, G], cwd="/data/workspace",
                   capture_output=True, text=True)
print((r.stdout or "")[-600:])
if r.returncode:
    print((r.stderr or "")[-2000:])
    sys.exit(1)
print("REBUILD OK")
