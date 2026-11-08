"""patch_v64c.py -- make the table numbering contiguous (13, 14, 15).

The generator's own Table 13 (statement audit) is the highest pre-existing
number; the two new tables must be 14 and 15, not 15 and 16.
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


rep('P(d, "<b>Table 15.</b> 1500 Adam steps',
    'P(d, "<b>Table 14.</b> 1500 Adam steps', "table 14")
rep('P(d, "<b>Table 16.</b> Trained layer-by-hop grid',
    'P(d, "<b>Table 15.</b> Trained layer-by-hop grid', "table 15")
rep('"as the W13 audit trail only; Tables 15-16 supersede them as the "',
    '"as the W13 audit trail only; Tables 14-15 supersede them as the "',
    "cross-reference")

io.open(P, "w", encoding="utf-8").write(s)
print("patched", P)
