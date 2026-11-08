"""patch_v72b.py -- second pass: clear the four residual N1/N6 strings."""
import io, re
P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
before = len(s)
log = []

def rep(old, new, label):
    global s
    n = s.count(old)
    s = s.replace(old, new, n)
    log.append((label, n))

# N1: one remaining bare statistic label, split across a string boundary.
rep("marginal statistic of P5(ii)", "marginal statistic of A5(ii)",
    "N1 marginal statistic of A5(ii)")
rep("P5(i)", "A5(i)", "N1 any bare P5(i)")
rep("P5(ii)", "A5(ii)", "N1 any bare P5(ii)")

# N6: Appendix A.5 has "d mutually" and "orthogonal" in separate literals.
rep("there are d mutually ", "there are c mutually ",
    "N6 A.5 branching factor")
rep("width d=", "width D=", "N6 any remaining width d")
rep("R^(d x d)", "R^(D x D)", "N6 any remaining R^(d x d)")
rep("d mutually orthogonal", "c mutually orthogonal",
    "N6 any remaining d mutually orthogonal")

# Assert the generator source is now clean (case-sensitive).
for pat in ("P5(i)", "P5(ii)", "width d=", "R^(d x d)",
            "d mutually orthogonal", "TAP1", "Theorem B"):
    assert pat not in s, "residual in generator source: " + pat

io.open(P, "w", encoding="utf-8").write(s)
print("patch_v72b: %d -> %d bytes" % (before, len(s)))
for k, n in log:
    print("  %-46s x%d" % (k, n))
