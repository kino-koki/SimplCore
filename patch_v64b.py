"""patch_v64b.py -- fix table numbering collision and unify the W13 trail.

The v7 generator already uses Tables 13 and 14. The two new tables of
Section 4.12 must therefore be 15 and 16, and the pre-v6.4 training figures
in Tables 2-3 must be labelled as an audit trail rather than as current
evidence.
"""
import io

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()


def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, "%s: %d matches" % (label, n)
    s = s.replace(old, new, 1)
    print("  ok", label)


rep('P(d, "<b>Table 13.</b> 1500 Adam steps',
    'P(d, "<b>Table 15.</b> 1500 Adam steps', "table 15")

rep('P(d, "<b>Table 14.</b> Trained layer-by-hop grid',
    'P(d, "<b>Table 16.</b> Trained layer-by-hop grid', "table 16")

rep('''      "reused. No language-modeling or attention-comparison claim is made." %''',
    '''      "reused. The earlier same-run training figures in Tables 2-3 are kept "
      "as the W13 audit trail only; Tables 15-16 supersede them as the "
      "training evidence for Result 4. No language-modeling or "
      "attention-comparison claim is made." %''', "W13 trail")

rep('''      "0.25 is never placed beside 0.0313. Table 2 is the single source for "
      "every number quoted anywhere in this paper.",''',
    '''      "0.25 is never placed beside 0.0313. Table 2 is the single source for "
      "the pre-v6.4 numbers; Section 4.12 injects its own values from its "
      "convergence curves.",''', "single-source caveat")

io.open(P, "w", encoding="utf-8").write(s)
print("patched", P)
