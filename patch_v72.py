"""
patch_v72.py -- execute the v7 detailed revision order (N1-N7) on the
SOLE current generator.

Strictly string / numbering / table / wording. NO new theorem, NO new
experiment, NO new number, NO new page beyond the explicitly ordered
title/abstract/conclusion sentences.
"""
import io, re, sys

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
orig = len(s)
log = []

def rep(old, new, label, required=True):
    global s
    n = s.count(old)
    if n == 0:
        log.append(("MISS", label, 0))
        if required:
            raise AssertionError("required replacement missing: " + label)
        return
    s = s.replace(old, new, n)
    log.append(("OK", label, n))

# =============================================================== N1 (FIRST)
# A1-A5 for Theorem A premises; P1-P8 reserved for propositions.
for i in range(1, 6):
    rep("TAP%d" % i, "A%d" % i, "N1 TAP%d -> A%d" % (i, i))
rep("Premises (A1-A5)", "Premises (A1-A5)", "N1 premises header preserved", required=False)
rep("A1-A5 name only the premises of Theorem A; P1-P8 name supporting propositions",
    "A1-A5 name only the premises of Theorem A; P1-P8 name supporting propositions",
    "N1 namespace sentence preserved", required=False)
rep('The per-fibre uniform statistic of P5(i)',
    'The per-fibre uniform statistic of A5(i)', "N1 bare P5(i) -> A5(i)", required=False)
rep('statistic of P5(i)', 'statistic of A5(i)', "N1 P5(i) generic", required=False)
rep('under P5(ii)', 'under A5(ii)', "N1 P5(ii) generic", required=False)
rep('Entries are marginal accuracies under P5(ii)',
    'Entries are marginal accuracies under A5(ii)', "N1 Table 12 P5(ii)", required=False)
rep('different statistics under P5;', 'different statistics under A5;',
    "N1 Table 12 caption", required=False)
rep('positivity premise of P2 and by the matching argument',
    "positivity premise of Result 2's matching argument",
    "N1 dangling positivity premise of P2")
rep('Under P2 with a linear ',
    'Under Proposition P2 with a linear ',
    "N1 Proposition P2 (3.2)")
rep('P2 and P6 feed Result 2(i); P1 and P6 feed A(c); P5 feeds A(e);',
    'Proposition P2 and P6 feed Result 2(i); Proposition P1 and P6 feed '
    'Theorem A(c); Proposition P5 feeds A(e);', "N1 dependency map")
rep('A(c) states the consequence WITHIN class M',
    'Theorem A(c) states the consequence WITHIN class M', "N1 Table 1 third row")
rep('P4 records the rho=1 single-pole isomorphism',
    'Proposition P4 records the rho=1 single-pole isomorphism',
    "N1 Table 1 / 3.3 / 5 Proposition P4")
rep("P5's state-order bound is conditional on d mutually orthogonal",
    "Proposition P5's state-order bound is conditional on c mutually orthogonal",
    "N1+N6 section 7")
rep('A.5  P5: conditional state-order lower bound',
    'A.5  Proposition P5: conditional state-order lower bound',
    "N1 Appendix A.5 title")
rep("it is why A(b)'s pigeonhole argument applies at all",
    "it is why Theorem A(b)'s pigeonhole argument applies at all",
    "N1 section 2.1")
rep('A(c) is falsified by a duality-preserving layer',
    'Theorem A(c) is falsified by a duality-preserving layer',
    "N1 section 4.11", required=False)
rep('A(c) states the consequence', 'Theorem A(c) states the consequence',
    "N1 any remaining bare A(c)", required=False)

# =============================================================== N2 (carriers)
rep('We therefore characterize where the bits may live: an "\n'
    '      "explicit tag, an alphabet partition, and a symbol id carrying a role "\n'
    '      "each supply one bit per ambiguous token. The alphabet partition folds "\n'
    '      "the mandatory bit into the symbol id: it is not bit-free, it is "\n'
    '      "annotation-free.',
    'We therefore characterize where the bits may live: an explicit "\n'
    '      "KEY/VALUE tag and an alphabet partition V -> 2V each supply one "\n'
    '      "bit per ambiguous token, counted in B. A positional format prior "\n'
    '      "is not a per-token channel; it restricts the support of the "\n'
    '      "instance distribution and reduces |A(v)| instead. The alphabet "\n'
    '      "partition folds the mandatory bit into the symbol id: it is not "\n'
    '      "bit-free, it is annotation-free.', "N2 abstract")
rep('Strongest evidence: two further tight structural ',
    'Strongest evidence: two tight structural ', "N3 abstract strongest evidence")
rep('three equivalent carriers, and the bit-depth frontier.',
    'two bit carriers plus one support restriction, and the bit-depth frontier.',
    "N2 section 1.1 Result 1")
rep('<b>(d) Tightness and carriers.</b> The bound is attained by three "\n'
    '      "information-equivalent carriers, each supplying one bit per ambiguous "\n'
    '      "token: an explicit KEY/VALUE tag; an alphabet partition V -> 2V, "\n'
    '      "which is exactly log2(2V_K / V_K) = 1 bit folded into the id and is "\n'
    '      "COUNTED in B by the definition in Section 2; and an absolute-position "\n'
    '      "format prior, which encodes the same bit in a support assumption.',
    '<b>(d) Tightness and carriers.</b> The bound is attained by two bit "\n'
    '      "carriers, each supplying one bit per ambiguous token and each "\n'
    '      "COUNTED in B by the definition in Section 2: an explicit KEY/VALUE "\n'
    '      "tag, and an alphabet partition V -> 2V, which is exactly "\n'
    '      "log2(2V_K / V_K) = 1 bit folded into the id. A third mechanism, the "\n'
    '      "absolute-position format prior, is NOT a per-token channel: it "\n'
    '      "restricts the SUPPORT of the instance distribution, thereby reducing "\n'
    '      "|A(v)| rather than increasing B.', "N2 3.1(d) opening")
rep("the third carrier's assumption fails ",
    "that support restriction no longer holds ",
    "N2 3.1(d) closing a")
rep("and its effective information falls to zero.",
    "and its effective information falls to zero (measured 0.3438, five seeds).",
    "N2 3.1(d) closing b")
rep('(W4) Three carriers each supply one bit per "\n'
    '      "ambiguous token (Table 7).',
    '(W4) Two bit carriers each supply one bit per ambiguous token "\n'
    '      "(Table 7); the positional format prior is a support restriction, "\n'
    '      "not a carrier.', "N2 W4")
# Table 7: filter the third carrier out of the displayed table only.
if 'for _r in IBC:' in s:
    rep('for _r in IBC:',
        'for _r in [r for r in IBC if r["carrier"] != "position_format_prior"]:',
        "N2 Table 7 row filter")
else:
    rep('    d.table(_car, [70, 20, 34, 42, 40], size=6.8, lead=8.0,',
        '    _car = [row for row in _car if "position" not in str(row[0]).lower()]\n'
        '    d.table(_car, [70, 20, 34, 42, 40], size=6.8, lead=8.0,',
        "N2 Table 7 row filter (fallback)")
rep('<b>Table 7.</b> Three carriers of the same one bit per ambiguous "\n'
    '      "token. The alphabet partition folds the mandatory bit into the "\n'
    '      "symbol id: it is not bit-free, it is annotation-free. The position "\n'
    '      "prior is a support restriction, not a per-token channel; it is "\n'
    '      "absent from this table and analysed through A(v) in Section 4.3.',
    '<b>Table 7.</b> Two carriers of the same one bit per ambiguous token; "\n'
    '      "both are counted in B by Section 2. The alphabet partition folds "\n'
    '      "the mandatory bit into the symbol id: it is not bit-free, it is "\n'
    '      "annotation-free. The positional format prior is a support "\n'
    '      "restriction on |A(v)|, not a per-token channel; it is analysed in "\n'
    '      "Section 3.1(d) and measured in Section 4.9.', "N2 Table 7 caption")

# =============================================================== N3 (tight scope)
rep('for line in ["Where the Bits Must Live: A Tight Input-Side",\n'
    '                     "Information Bound for Fixed-State Spectral Models"]:',
    'for line in ["Where the Bits Must Live: An Input-Side",\n'
    '                     "Information Bound, Tight on a Constructible Family,",\n'
    '                     "for Fixed-State Spectral Models"]:', "N3 title block")
rep('self.c.setTitle("Where the Bits Must Live: A Tight Input-Side Information Bound for Fixed-State Spectral Models")',
    'self.c.setTitle("Where the Bits Must Live: An Input-Side Information Bound, Tight on a Constructible Family, for Fixed-State Spectral Models")',
    "N3 PDF metadata title")
rep('      "answers compatible with the flat projection v (Theorem A). The "\n'
    '      "difficulty is that the bound is distribution-free and depth-free: "',
    '      "answers compatible with the flat projection v (Theorem A). The bound "\n'
    '      "is attained on an explicit binary-chain family with |A| = 2^m; on "\n'
    '      "the content-chain task used in our experiments, reachability "\n'
    '      "enumeration gives |A| <= 5 (log2|A(v)| <= 2.3219), so the bound is "\n'
    '      "loose there. The difficulty is that the bound is distribution-free "\n'
    '      "and depth-free: "', "N3 abstract inserted sentence")
rep('the contribution here is a tight, "\n'
    '      "distribution-free quantification of the input-side deficit, and a "\n'
    '      "construction showing that supplying exactly that deficit restores "',
    'the contribution here is a "\n'
    '      "distribution-free quantification of the input-side deficit, attained "\n'
    '      "on an explicit constructible family, and a construction showing "\n'
    '      "that supplying exactly that deficit restores "', "N3 intro para 2")
rep('<b>Result 1 (Theorem A).</b> A tight input-side information bound "\n'
    '      "B >= log2 |A(v)|,',
    '<b>Result 1 (Theorem A).</b> An input-side information bound "\n'
    '      "B >= log2 |A(v)|, tight on the binary-chain family,',
    "N3 section 1.1 Result 1")
rep('"not contain. We proved a tight lower bound on that information "\n'
    '      "(B >= log2 |A|, Theorem A), stated precisely where the bits may live "',
    '"not contain. We proved a distribution-free lower bound on that "\n'
    '      "information (B >= log2 |A(v)|, Theorem A), attained on an explicit "\n'
    '      "binary-chain family, stated precisely where the bits may live "',
    "N3 section 8 opening")
# section 8 added sentence: append to its opening paragraph.
rep('"frontier is a right angle: extra layers cannot substitute for missing "\n'
    '      "bits. Two tight structural bounds (order-2 state for one hop; exactly "\n',
    '"frontier is a right angle: extra layers cannot substitute for missing "\n'
    '      "bits. On the content-chain task studied in our experiments the "\n'
    '      "bound is loose (Section 4.3); what transfers is the "\n'
    '      "distribution-free inequality A(b), which applies to any flat "\n'
    '      "projection, including standard associative-recall protocols. Two "\n'
    '      "tight structural bounds (order-2 state for one hop; exactly "\n',
    "N3 section 8 added sentence")
# uniform log2|A(v)| (the already-correct form contains no bare "log2 |A|")
rep('log2 |A|', 'log2 |A(v)|', "N3/N7 unify log2|A(v)|", required=False)

# =============================================================== N4 (references)
rep('"[4] Gu, A. et al. On the Parameterization and Initialization of "\n'
    '        "Diagonal State Space Models (S5). NeurIPS 2022. arXiv:2208.04933.",',
    '"[4] Smith, J. T. H., Warrington, A., and Linderman, S. W. "\n'
    '        "Simplified State Space Layers for Sequence Modeling. "\n'
    '        "arXiv:2208.04933, 2022.",',
    "N4 reference [4] S5 un-misattribution")
rep('"[11] Romero, D. W. et al. CKConv: Continuous Kernel Convolution For "\n'
    '        "Sequential Data. ICLR 2022. arXiv:2202.06037.",',
    '"[11] Romero, D. W. et al. CKConv: Continuous Kernel Convolution For "\n'
    '        "Sequential Data. ICLR 2022. arXiv:2102.02611.",',
    "N4 reference [11] CKConv arXiv corrected")

# =============================================================== N5 (T* interface)
rep('"Every occurrence of the interface in Sections 2-4 uses this exact "\n'
    '      "expression. The phrase \'maps the VALUE half to the KEY half\' describes "\n'
    '      "the action of T*, not a second interface; the unnormalised product "\n'
    '      "(z_h C_val^T) C_key is the special case T_h = C_val^T C_key with the "\n'
    '      "same normalisation applied.',
    '"Every occurrence of the interface in Sections 2-4 uses this exact "\n'
    '      "expression. The phrase \'maps the VALUE half to the KEY half\' "\n'
    '      "describes the action of T* = C_key^T C_val, not a second "\n'
    '      "interface. The read-out is last-layer only (A4).',
    "N5 remove contradictory unnormalised-product sentence")

# =============================================================== N6 (symbol d)
rep('T_h in R^(d x d)', 'T_h in R^(D x D)', "N6 interface dimension")
rep('width d=%d', 'width D=%d', "N6 Table 3 caption")
rep('rank at most d', 'rank at most D', "N6 A.5 rank")
rep('d mutually orthogonal', 'c mutually orthogonal', "N6 branching factor", required=False)
rep('d^(H+1)', 'c^(H+1)', "N6 candidate set size", required=False)
rep('o*d <= H*d', 'o*D <= H*D', "N6 kernel dimension inequality", required=False)
rep('o <= H*d', 'o <= H*D', "N6 secondary", required=False)

# =============================================================== N7 (small fixes)
# #3 chance annotation only: real measured 0.0312 values are untouched.
rep('4 gives %.4f (chance %.4f). The failure mode is rank loss,',
    '4 gives %.4f (chance 0.0313). The failure mode is rank loss,',
    "N7 #3 chance annotation")
rep('dmean("rank_16"), dmean("rank_04"), CHANCE),',
    'dmean("rank_16"), dmean("rank_04")),',
    "N7 #3 remove formatting argument")
# #4 drop 0.9609 (a value not in Table 2) and keep the qualitative statement.
rep('"alternating layout gives %.4f while a gapped layout gives %.4f because "\n'
    '      "zero fillers suppress the spurious peak (L1).',
    '"alternating layout gives %.4f while a gapped layout is higher because "\n'
    '      "zero fillers suppress the spurious peak (L1).',
    "N7 #4 remove 0.9609")
rep('(s5mean("none", "alternating"), cbe("flat", "none", "gapped")))',
    '(s5mean("none", "alternating")))', "N7 #4 remove formatting argument")
# #5 de-duplicate the identical explicit/gapped row in Table 12 without
# altering any value (the source JSON itself contains the duplicate).
rep('    for _r in S5:\n        if "paired" in _r:\n            continue\n        _fs.append(',
    '    _seen = set()\n'
    '    for _r in S5:\n'
    '        if "paired" in _r:\n            continue\n'
    '        _key = (_r["mask"], _r["layout"], round(_r["mean"], 6))\n'
    '        if _key in _seen:\n            continue\n'
    '        _seen.add(_key)\n        _fs.append(',
    "N7 #5 Table 12 duplicate row")
rep('"<b>Table 12.</b> Five-seed statistics. The per-fibre uniform "',
    '"<b>Table 12.</b> Five-seed statistics (duplicate identical rows in "\n'
    '      "the source JSON are shown once). The per-fibre uniform "',
    "N7 #5 caption note")
# #6 Broader Impact disclosure (neutral, CFP-conditional).
rep('    Hh(d, "Broader Impact", size=11.0, before=0.0, after=4.0, span=False)\n'
    '    P(d, "This is a theoretical and synthetic-experimental study of "',
    '    Hh(d, "Broader Impact", size=11.0, before=0.0, after=4.0, span=False)\n'
    '    P(d, "Generative AI tools were used for implementation assistance "\n'
    '      "and language editing; all research questions, proofs and "\n'
    '      "conclusions were specified and verified by the human author. "\n'
    '      "This disclosure is subject to the ICML 2027 author instructions, "\n'
    '      "which are not yet published.", size=SMALL, lead=SLEAD, after=2.0)\n'
    '    P(d, "This is a theoretical and synthetic-experimental study of "',
    "N7 #6 AI disclosure")
# #7 remove the check-count as a contribution.
rep('"formation, witness agreement and protocol consistency "\n'
    '      "(%d/%d gates pass); it is NOT a mathematical proof, and all proofs in "\n'
    '      "this paper are analytic." %\n'
    '      (next(iter(ADJ["checks"].values())) is not None and ADJ["n_pass"],\n'
    '       ADJ["n_checks"]),',
    '"formation, witness agreement and protocol consistency. It is NOT a "\n'
    '      "mathematical proof, and all proofs in this paper are analytic.",',
    "N7 #7 remove gate count")

io.open(P, "w", encoding="utf-8").write(s)
print("patched: %d -> %d bytes" % (orig, len(s)))
for st, label, n in log:
    print("  %-4s %-58s x%d" % (st, label, n))
miss = [x for x in log if x[0] == "MISS"]
print("MISS count:", len(miss))
