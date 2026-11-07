"""expand5_v7.py — final main-paper section: a complete premise / claim /
boundary / non-vacuity table and explicit falsification conditions.
Both are hard content requested by the review, not padding."""
G = "/data/workspace/build_simplcore_pdf_v7.py"
s = open(G, encoding="utf-8").read()


def ins(anchor, block, label):
    global s
    assert s.count(anchor) == 1, (label, s.count(anchor))
    s = s.replace(anchor, block + anchor)
    print("  inserted", label)


ins('    Hh(d, "4  Experiments: One Table, One Numeric Convention",\n',
    r'''
    Hh(d, "3.7  Premise, claim, boundary and non-vacuity, per statement",
       size=9.5, before=4.0, after=2.0)
    P(d, "Every statement in this paper is listed below with its exact "
      "premise, its conclusion, the boundary outside which it is NOT "
      "claimed, and one concrete instance satisfying the premise. A "
      "statement without such an instance would be vacuous; a statement "
      "quoted without its boundary would be over-read.",
      size=SMALL, lead=SLEAD, after=2.0)
    d.table([
        ["stmt", "premise", "conclusion", "boundary", "non-vacuity"],
        ["A(a)", "deterministic F, fibre v", "acc(F|v) <= 2^B/|A(v)|",
         "conditional, not marginal", "n=4, m=2, B=0, |A|=4"],
        ["A(b)", "determinism only", "B >= log2|A|", "exactness on all instances",
         "binary chain, |A|=2^m"],
        ["A(c)", "class M, P6", "bits cannot be made inside", "IF duality kept",
         "static 2-layer stack"],
        ["A(d)", "B = sum log2|pi^-1|", "3 carriers, 1 bit/token",
         "position prior is support", "V -> 2V partition"],
        ["A(e)", "A(b) + Result 2(ii)", "right-angle frontier", "no stop/select",
         "Table 6, m=1..4"],
        ["A(f)", "P4 last-layer read-out", "H_L>H overshoots",
         "retained intermediates suffice", "H=2, H_L=3"],
        ["B(i)", "P2 linear summary, n>=2", "min state order = 2",
         "single mixer", "Construction C"],
        ["B(ii)", "H6, P4, distinct keys", "min layers = H",
         "overshoot is read-out", "Table 5 diagonal"],
        ["P5", "d orthogonal successors/hop", "o >= H+1",
         "branching 2 degenerates", "H=2, d=64"],
    ], [26, 58, 52, 56, 46], size=6.4, lead=7.6, after=2.0)
    P(d, "<b>Table 13.</b> Statement audit. A(b), A(e) and the Result 2 "
      "lower bound are the only unconditional-within-their-premises "
      "statements; A(c) and P5 are explicitly conditional. No statement is "
      "promoted by a numerical check.", size=SMALL, lead=SLEAD, after=2.5)

''', "3.7 statement audit")

ins('    Hh(d, "5  What This Does NOT Establish", size=10.2, before=5.0,\n',
    r'''
    Hh(d, "4.11  Explicit falsification conditions", size=9.5, before=4.0,
       after=2.0)
    P(d, "Each result has a stated failure mode that can be checked without "
      "re-running any training. A(b) is falsified by any exact H-hop "
      "construction with B < log2|A| on a fibre where |A| > 2^B; the "
      "identifiable place is the injection step in Appendix A.1. A(c) is "
      "falsified by a duality-preserving layer whose coefficients depend on "
      "x_t yet whose impulse response is Toeplitz, contradicting P6's "
      "induction on the lag. Result 2(i) is falsified by an order-1 state "
      "achieving exact one-hop retrieval under P2; Result 2(ii) by a "
      "single static-LTI layer exact on H >= 2 hops, or by a two-layer stack "
      "exact on H=1 under P4. P5 is falsified by an exact H-hop construction "
      "with o <= H under its d-orthogonal-successor premise. Construction C "
      "is falsified by a non-Toeplitz impulse response or an L-dependent "
      "state, both checked in Table 4. The W12 audit is falsified by "
      "reproducing the stored 0.9987 attention baseline under the Section "
      "4.6 protocol. Each condition names the exact measurement that would "
      "have to come out differently.", size=SMALL, lead=SLEAD, after=2.5)

''', "4.11 falsification")

open(G, "w", encoding="utf-8").write(s)
print("EXPANSION 5 WRITTEN")
