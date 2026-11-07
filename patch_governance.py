"""
patch_governance.py -- execute the skill pack's Phase 1/2/4 verdicts on the
SOLE current generator.

Changes, all driven by the audit evidence in theory_work/results/governance_*.json:

  * TAP1-TAP5 is now a namespace RESERVED for Theorem A's premises.
    P1-P8 stays reserved for supporting propositions. The two no longer
    overlap (the previous build used "P5: two accuracy notions" and
    "P5 gives a state-order bound" inside the same section).
  * The ambiguous headline label "Theorem B" is removed. Result 2 has one
    name. The state-order statement folded into A(e) no longer borrows a
    headline label.
  * "by P1 that destroys the Toeplitz structure" corrected to P6 (the
    Toeplitz iff time-invariance proposition), which is what the argument
    actually uses.
  * Table 1's "P1 proves Toeplitz iff ..." corrected to P6.
  * Full proofs duplicated in the main text (3.1.1, 3.1.2, 3.2.1, and the
    P2 explicit construction) are removed; the main text keeps a short
    proof intuition + evidence anchor, and Appendix A already holds the
    complete proofs. This is the skill pack's "proof audit before pages"
    rule and removes the exact failure mode it names: two full copies of a
    proof drifting apart.
  * Table 28 caption now labels the 0.2969 value as a DIFFERENT
    configuration (single-seed position-prior carrier), so it cannot be
    read as a second frozen value of the 5-seed parity-gapped number.
"""
import io

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
orig = len(s)


def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, "%s: %d matches (expected 1)" % (label, n)
    s = s.replace(old, new, 1)
    print("  ok  " + label)


def block(start, end, new, label):
    """Replace the whole region from `start` up to (not including) `end`."""
    global s
    i = s.find(start)
    j = s.find(end, i + 1)
    assert i >= 0 and j > i, "%s: markers not found (%d,%d)" % (label, i, j)
    s = s[:i] + new + s[j:]
    print("  ok  " + label + " (%d -> %d chars)" % (j - i, len(new)))


# ---------------------------------------------------------------- 1. namespace
rep('"A(v) = { k_H(G) : G in R(v) }. The per-fibre statistic P5(i) is the "',
    '"A(v) = { k_H(G) : G in R(v) }. The per-fibre statistic TAP5(i) is the "',
    "2.2 per-fibre statistic")

rep('"uniform measure on R(v); the marginal statistic P5(ii) is the "',
    '"uniform measure on R(v); the marginal statistic TAP5(ii) is the "',
    "2.2 marginal statistic")

# Whole premises paragraph: rename and add the namespace rule.
block(
    '''    P(d, "<b>Premises.</b> P1 (task): an instance has n binding pairs, a "''',
    '''    P(d, "<b>Two accuracy notions (P5).</b>''',
    '''    P(d, "<b>Premises (TAP1-TAP5).</b> TAP1 (task): an instance has n "
      "binding pairs, a role assignment deciding which token of each pair "
      "is the KEY, arbitrary pair order, arbitrary zero fillers, and answer "
      "k_H(G), the endpoint of the H-hop chain from an external query "
      "q = k_0. TAP2 (visible input): the model sees (v, s) with v the flat "
      "projection and s in {0,1}^B the refinement counted above. TAP3 "
      "(class M): H_L static-LTI layers with time-invariant, content-"
      "independent coefficients, fixed-dimensional state, fixed linear "
      "interfaces, deterministic last-layer read-out. TAP4: only the last "
      "layer is decoded. TAP5: two accuracy notions, defined below. Parts "
      "(a) and (b) need NONE of TAP3. <b>Namespace.</b> TAP1-TAP5 name only "
      "the premises of Theorem A; P1-P8 name supporting propositions "
      "(Section 3.4). The two namespaces are disjoint by construction.",
      size=SMALL, lead=SLEAD, after=2.0)
''',
    "3.1 premises -> TAP namespace")

rep('"<b>Two accuracy notions (P5).</b> (i)',
    '"<b>Two accuracy notions (TAP5).</b> (i)',
    "3.1 accuracy notions")

rep('"<b>(c) A within-class-M location statement.</b> Within P3, "',
    '"<b>(c) A within-class-M location statement.</b> Within TAP3, "',
    "3.1(c) within class M")

rep('input-dependent and hence time-varying; by P1 that destroys the ',
    'input-dependent and hence time-varying; by P6 (Toeplitz iff '
    'time-invariance) that destroys the ',
    "3.1(c) wrong citation P1 -> P6")

rep('right angle, not a trade-off curve. P5 gives a ',
    'right angle, not a trade-off curve. Supporting proposition P5 gives a ',
    "3.1(e) disambiguate P5")

rep('''      "listed this as a separate Theorem B; it is part (e) here.",''',
    '''      "headline result; the state-order statement is folded into (e) here.",''',
    "3.1(e) remove Theorem B")

rep('H_L > H follows from P4. Retaining intermediate',
    'H_L > H follows from TAP4. Retaining intermediate',
    "3.1(f) TAP4")

rep('proofs, the binary-chain witness, and the P5 state-order argument are ',
    'proofs, the binary-chain witness, and the supporting P5 state-order '
    'argument are ',
    "3.1 proof route")

rep('3.2  Result 2 - Theorem B (two tight structural bounds)',
    '3.2  Result 2 (two tight structural bounds)',
    "3.2 title")

rep('Under H6, P4 and distinct keys, one ',
    'Under H6, TAP4 and distinct keys, one ',
    "3.2(ii) TAP4")

rep('diagonal = exact (Theorem B)',
    'diagonal = exact (Result 2)',
    "Figure 1 legend")

rep('"<b>Result 2 (Theorem B).</b> Two structural bounds proved jointly "',
    '"<b>Result 2.</b> Two structural bounds proved jointly "',
    "1.1 Result 2 label")

rep('"A.4  Result 2 (Theorem B): induction and tightness"',
    '"A.4  Result 2: induction and tightness"',
    "A.4 title")

rep('"P1 proves Toeplitz iff time-invariance for the full second-order family with c_t != 0"',
    '"P6 proves Toeplitz iff time-invariance for the full second-order family with c_t != 0"',
    "Table 1 wrong proposition")

rep('"this proves (b). Neither step uses P3. The marginal accuracy "',
    '"this proves (b). Neither step uses TAP3. The marginal accuracy "',
    "A.1 TAP3")

rep('"angle. A(f): under P4 the decoded value at depth H_L is k_(H_L); for "',
    '"angle. A(f): under TAP4 the decoded value at depth H_L is k_(H_L); for "',
    "A.3 TAP4")


# ---------------------------------------------------------------- 2. proof shift
INTUITION_31 = '''    P(d, "<b>Proof intuition.</b> A deterministic decoder sees at most 2^B "
      "distinct refinements of a flat projection v, so a fibre with "
      "|A(v)| > 2^B answers forces two distinct answers onto one output "
      "(pigeonhole). No property of the kernel, depth or state order enters "
      "this step, which is why (a) and (b) are distribution-free and "
      "depth-free. For (c), a static-LTI stack with fixed interfaces is a "
      "function of v alone once the refinement is removed, so the missing "
      "bits cannot be manufactured inside the recurrence; P6 says making "
      "the coefficients input-dependent destroys Toeplitz. For (d), the "
      "same budget B = sum_t log2|pi^(-1)(v_t)| is charged by an explicit "
      "tag, by the alphabet partition, and by the KEY-only tag, so the "
      "lower bound and its tightness witness use ONE quantity (Appendix H "
      "records the executable bit-accounting check). A position prior is "
      "instead a restriction on |A(v)|, not a contribution to B. Complete "
      "proofs are in Appendix A.1-A.3; the exhaustive binary-chain witness "
      "and the per-fibre statistic are in Tables 6 and 26.",
      size=SMALL, lead=SLEAD, after=2.5)

'''
INTUITION_32 = '''    P(d, "<b>Proof intuition.</b> One static-LTI layer resolves at most "
      "one content match, so H hops need at least H layers; the H-th "
      "intermediate output through the unique H6 interface supplies the "
      "next query, giving the matching upper bound by induction. An "
      "order-1 vector state is ruled out by P2, while Construction C "
      "attains order 2, so the minimum is exactly 2. The full induction "
      "is in Appendix A.4 and the attaining 5 x 5 grid is Table 5. "
      "Overshoot H_L > H is A(f), a read-out convention, not a failure "
      "of the lower bound.", size=SMALL, lead=SLEAD, after=2.5)

'''
INTUITION_34 = '''    P(d, "<b>P2 proof intuition.</b> With a fixed linear historical "
      "summary A = sum_p L_p v_p + c, one can always change a value "
      "assignment while keeping (A, q) fixed, either through a singular "
      "block L_j, through a compensating pair of blocks, or by redirecting "
      "the query when A depends on a single value. The complete three-case "
      "construction is in Appendix A.3. It never infers a target-"
      "coordinate perturbation from dim ker L > 0 alone; this is the "
      "only form of the argument used anywhere in the paper.",
      size=SMALL, lead=SLEAD, after=2.0)

'''
block('    Hh(d, "3.1.1  Algebra of (a), (b) and (c)"',
      '    Hh(d, "3.2  Result 2 (two tight structural bounds)"',
      INTUITION_31,
      "remove duplicated 3.1.1/3.1.2 full proofs")

block('    Hh(d, "3.2.1  Induction and the attaining construction"',
      '    Hh(d, "3.3  Result 3',
      INTUITION_32,
      "remove duplicated 3.2.1 full proof")

block('    P(d, "<b>P2, explicit construction (replaces the earlier kernel-"',
      '    Hh(d, "3.5  Dependency map and falsification"',
      INTUITION_34,
      "remove duplicated P2 explicit construction from main text")

rep('"impossibility, explicit construction below). <b>P3</b>',
    '"impossibility; complete construction in Appendix A.3). <b>P3</b>',
    "3.4 pointer to appendix")


# ---------------------------------------------------------------- 3. number scope
old_cap = '''    P(d, "<b>Table 28.</b> All %d bit carriers. The position format prior "
      "is a support restriction, not a per-token channel, and is analysed "
      "through A(v) rather than through B; its effective bit count under "
      "arbitrary filler is zero." % len(IBC), size=SMALL, lead=SLEAD,
      after=2.5, span=False)'''
new_cap = '''    P(d, "<b>Table 28.</b> All %d bit carriers. The position-format-prior "
      "accuracy %.4f is a SINGLE-SEED measurement on the alternating layout "
      "under that carrier's support assumption; it is a DIFFERENT "
      "configuration from the frozen five-seed parity-gapped value in "
      "Table 10 and is never quoted as a conclusion elsewhere. The position "
      "format prior is a support restriction, not a per-token channel, and "
      "is analysed through A(v) rather than through B; its effective bit "
      "count under arbitrary filler is zero." %
      (len(IBC), next(r["acc"] for r in IBC
                      if r["carrier"] == "position_format_prior")),
      size=SMALL, lead=SLEAD, after=2.5, span=False)'''
rep(old_cap, new_cap, "Table 28 caption with injected single-seed value")

io.open(P, "w", encoding="utf-8").write(s)
print("patched: %d -> %d bytes" % (orig, len(s)))
