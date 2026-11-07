"""expand3_v7.py — final hard-content pass to reach exactly 8 main pages."""
import json

G = "/data/workspace/build_simplcore_pdf_v7.py"
R = "/data/workspace/train/superposition_probe/results/"
s = open(G, encoding="utf-8").read()
S5 = json.load(open(R + "stress_five_seed.json"))
IBD = json.load(open(R + "info_bound_depth.json"))


def ins(anchor, block, label):
    global s
    assert s.count(anchor) == 1, (label, s.count(anchor))
    s = s.replace(anchor, block + anchor)
    print("  inserted", label)


# -------------------------------------------------- 2.2 formal task / fibre
ins('    Hh(d, "3  Four Core Results", size=10.2, before=5.0, after=2.5)\n',
    r'''
    Hh(d, "2.2  Task distribution and the fibre A(v), formally",
       size=9.4, before=4.0, after=2.0)
    P(d, "An instance G is a directed binding graph on n pairs. A role "
      "assignment rho in {0,1}^(2n) decides, for each pair, which token is "
      "the KEY and which the VALUE; pair order is arbitrary and an arbitrary "
      "number of zero FILLER tokens may appear between pairs. The answer is "
      "k_H(G), the endpoint of the H-hop chain starting at the externally "
      "supplied query q = k_0. The instance distribution is uniform over "
      "role assignments, pair permutations, filler counts and distinct key "
      "choices from [V_K], conditioned on distinct keys unless a duplicate-"
      "key condition is stated. For a fixed flat projection v we define the "
      "FIBRE R(v) = { G : pi(x(G)) = v } and the compatible-answer set "
      "A(v) = { k_H(G) : G in R(v) }. The per-fibre statistic P5(i) is the "
      "uniform measure on R(v); the marginal statistic P5(ii) is the "
      "push-forward of the instance distribution under pi. These two measures "
      "are different objects, and Theorem A(a) constrains only the first. "
      "Section 4.3 computes |A(v)| for the concrete task by reachability "
      "enumeration rather than assuming a value.", size=SMALL, lead=SLEAD,
      after=2.5)

''', "2.2 formal fibre")

# -------------------------------------------------- A(d,e,f) main-text proof
ins('    Hh(d, "3.2  Result 2 - Theorem B (two tight structural bounds)",\n',
    r'''
    Hh(d, "3.1.2  Algebra of (d), (e) and (f)", size=9.3, before=4.0,
       after=2.0)
    P(d, "For (d), the Section 2 definition "
      "B = sum_t log2 |pi^(-1)(v_t)| counts an alphabet partition with "
      "|pi^(-1)| = 2 as exactly one bit per non-empty token, and counts an "
      "explicit tag as the same quantity. Tagging only the KEY token of each "
      "of n pairs gives B = n, attaining B >= n from (b). A positional prior "
      "is NOT a per-token channel: it restricts the SUPPORT of the "
      "distribution, thereby reducing |A(v)|, and under random filler "
      "spacing that restriction no longer holds, so the effective supplied "
      "information falls to zero. This is why the position carrier is "
      "analysed through A(v) and never added to B. For (e), (b) makes "
      "B < log2|A| infeasible for EVERY deterministic model and hence for "
      "every H_L; for B >= log2|A| Result 2(ii) gives minimum depth exactly "
      "H. The frontier is a right angle. For (f), under P4 the decoded "
      "value at depth H_L is k_(H_L); with distinct keys and H_L > H this "
      "differs from k_H. Retaining and selecting the H-th intermediate "
      "output gives the correct answer for any H_L >= H, so the lower bound "
      "is unchanged and 'exactly H' is not an architectural law. QED.",
      size=SMALL, lead=SLEAD, after=2.5)

''', "3.1.2 A(d/e/f)")

# -------------------------------------------------- non-vacuity itemised
ins('    Hh(d, "3.4  Supporting statements (not theorems)",\n',
    r'''
    Hh(d, "3.3.4  Non-vacuity witnesses, itemised", size=9.3, before=4.0,
       after=2.0)
    P(d, "A theorem whose premises admit no instance is empty. Five "
      "independent witnesses are supplied. (W1) n=4, m=2, V_K=32, H=2, "
      "flat alphabet, arbitrary filler spacing: B=0 gives |A|=4 and "
      "acc(F|v) <= 1/4, while B=2 gives exactness. (W2) The exhaustive "
      "binary-chain enumeration gives 2^(B-m) for B < m and 1 at B=m for "
      "m=1..4, with bit-identical visible inputs across all 2^m role "
      "assignments. (W3) Depth invariance: a static-LTI stack of depth "
      "1..8 produces bit-identical outputs across compatible role "
      "assignments (maximum deviation %.1e), confirming that depth cannot "
      "manufacture the bits. (W4) Three carriers each supply one bit per "
      "ambiguous token (Table 7). (W5) The 5 x 5 layer x hop grid (Table 5) "
      "attains the exactly-H-layer bound. Each witness is a finite, "
      "reproducible object; none is used as a premise of a proof." %
      max(q["max_X_dev"] for q in IBD["premise_P2"]),
      size=SMALL, lead=SLEAD, after=2.0)
    _dv = [["m", "B", "layers", "|A|", "acc", "dev"]]
    for _r in IBD["rows"][:8]:
        _dv.append([str(_r["m"]), str(_r["B"]), str(_r["n_layers"]),
                    str(_r["n_answer_classes"]), "%.4f" % _r["acc"],
                    "%.1e" % _r["max_output_dev_across_d"]])
    d.table(_dv, [20, 20, 30, 26, 34, 38], size=6.8, lead=8.0,
            after=2.0)
    P(d, "<b>Table 11.</b> Depth-invariance witness. Accuracy is the "
      "per-fibre uniform statistic and does not change with layer count; "
      "the output deviation across compatible role assignments is zero. "
      "This is the empirical form of A(e), not its proof.",
      size=SMALL, lead=SLEAD, after=2.5)

''', "3.3.4 non-vacuity")

# -------------------------------------------------- full 5-seed statistics
ins('    Hh(d, "5  What This Does NOT Establish", size=10.2, before=5.0,\n',
    r'''
    Hh(d, "4.9  Full five-seed paired statistics", size=9.5, before=4.0,
       after=2.0)
    P(d, "Table 12 reports every five-seed configuration used in the "
      "adversarial and codebook experiments, with its mean, standard "
      "deviation and 10,000-resample paired CI. Entries are marginal "
      "accuracies under P5(ii). The paired difference between an explicit "
      "role bit and the parity format prior on the gapped layout is "
      "%+.4f with CI [%+.4f, %+.4f] over five seeds, all positive." %
      (next(r["mean"] for r in S5 if r.get("paired")),
       next(r["ci95"][0] for r in S5 if r.get("paired")),
       next(r["ci95"][1] for r in S5 if r.get("paired"))),
      size=SMALL, lead=SLEAD, after=2.0)
    _fs = [["mask / layout", "mean", "std", "CI95"]]
    for _r in S5:
        if "paired" in _r:
            continue
        _fs.append(["%s / %s" % (_r["mask"], _r["layout"]),
                    "%.4f" % _r["mean"], "%.4f" % _r["std"],
                    "[%.4f, %.4f]" % tuple(_r["ci95"])])
    d.table(_fs, [66, 32, 32, 66], size=6.8, lead=8.0, after=2.0)
    P(d, "<b>Table 12.</b> Five-seed statistics. The per-fibre uniform "
      "value 0.0625 of Table 6 and the marginal value %.4f of this table "
      "are different statistics under P5; neither is a correction of the "
      "other." % s5mean("none", "alternating"),
      size=SMALL, lead=SLEAD, after=2.5)

''', "4.9 five-seed")

open(G, "w", encoding="utf-8").write(s)
print("EXPANSION 3 WRITTEN")
