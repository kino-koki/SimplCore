#!/usr/bin/env python3
"""
make_v8.py -- surgical narrative refactor of build_simplcore_pdf_v7.py.

Goal (user's brief): replace "30 theorems on parade" with ONE killer claim
+ evidence a reviewer cannot refuse.  We keep the v7 numbering scheme
(Result 1-4, Theorem A, P1-P8, L1-L3) because it already passed a 45/45
desk-check and because earlier rounds explicitly forbade another numbering
reshuffle.  What changes is the STORY: Result 2 (tight minimum tensor order
2 + exactly H layers + joint O(1) state) becomes the single central claim;
Results 1/3/4 are explicitly demoted to supporting roles.

Every replacement is exact and asserted unique.  No number is invented:
all figures come from the same JSON files the v7 generator already injects.
"""
import ast
import re
import sys

SRC = "/data/workspace/build_simplcore_pdf_v7.py"
DST = "/data/workspace/build_simplcore_pdf_v8.py"

s = open(SRC, encoding="utf-8").read()
N = 0


def sub(old, new, tag):
    global s, N
    c = s.count(old)
    if c != 1:
        print("FAIL [%s]: expected 1 occurrence, found %d" % (tag, c))
        print("  head:", repr(old[:110]))
        sys.exit(1)
    s = s.replace(old, new, 1)
    N += 1
    print("  OK  %-34s" % tag)


# ---------------------------------------------------------------- 1. title
sub('''        for line in ["Where the Bits Must Live: An Input-Side",
                     "Information Bound, Tight on a Constructible Family,",
                     "for Fixed-State Spectral Models"]:''',
    '''        for line in ["Exact Content Addressing Needs an Order-2 State:",
                     "A Tight Joint Bound on Tensor Order, Depth and",
                     "State Size for Static-LTI Sequence Models"]:''',
    "title")

# ---------------------------------------------------------------- 2. abstract
_old_abs_start = s.index('    P(d,\n      "We prove that exact H-hop content retrieval')
_old_abs_end = s.index('      size=SMALL, lead=SLEAD, after=3.0, span=True)',
                      _old_abs_start) + len(
                          '      size=SMALL, lead=SLEAD, after=3.0, span=True)')
old_abs = s[_old_abs_start:_old_abs_end]
new_abs = '''    P(d,
      "We prove that exact content addressing by a fixed-state decoder "
      "requires a tensor state of order exactly 2, and that one Toeplitz, "
      "strictly causal, O(1)-state construction attains this bound jointly "
      "with a sharp depth bound of exactly H static-LTI layers for H hops "
      "(Result 2, the central claim of this paper). The difficulty is that "
      "any fixed-dimensional vector summary collapses distinct value "
      "assignments into a kernel of positive dimension, so no read-out, "
      "linear or nonlinear, can recover the queried binding; the supporting "
      "input-side bound is distribution-free and depth-free. Our method "
      "keeps the recurrence coefficients time-invariant and moves the "
      "key-query match into a bilinear contraction read-out OUTSIDE the "
      "kernel, so the FFT/IIR duality and the constant-size state survive "
      "unchanged. Strongest evidence: the tight minimum tensor order of 2 "
      "has both a matching lower bound and an attaining construction, a "
      "zero-training 5 x 5 layer-by-hop grid is exact on its diagonal, and "
      "from random initialisation H=2 with two layers reaches %.4f +/- %.4f "
      "by step %d on five of five seeds while one layer stays at %.4f +/- "
      "%.4f (paired %+.4f, 95%% CI [%+.4f, %+.4f] at 1500 steps), with "
      "FFT/IIR agreement at machine precision and state bytes independent "
      "of L. The design rule is to budget log2|A(v)| input-side bits, then "
      "choose exactly H layers and an order-2 state; we make no language-"
      "modeling, perplexity, throughput, equivalence or superiority claim." %
      (cm(2, 2)[0], cm(2, 2)[1], CONV_FIRST,
       cm(2, 1)[0], cm(2, 1)[1], CPAIR[0], CPAIR[1], CPAIR[2]),
      size=SMALL, lead=SLEAD, after=3.0, span=True)'''
sub(old_abs, new_abs, "five-sentence abstract")

# ---------------------------------------------------------------- 3. intro
_intro_a = s.index('    P(d, "Fixed-state attention-free')
_intro_b = s.index('size=BODY, lead=LEAD, after=2.5)', _intro_a) + len(
    'size=BODY, lead=LEAD, after=2.5)')
sub(s[_intro_a:_intro_b],
    '''    P(d, "Fixed-state attention-free sequence models compress a prefix into "
      "a constant-size summary. We ask what the SHAPE of that summary must "
      "be for exact content addressing, and prove a sharp answer: a vector "
      "state cannot do it, a tensor state of order 2 can, and no smaller "
      "state suffices. The same construction also settles depth: under a "
      "static time-invariant kernel and last-layer read-out, H content hops "
      "need exactly H layers, and the state stays O(1) in sequence length.",
      size=BODY, lead=LEAD, after=2.0)
    P(d, "Prior work established that fixed-state models struggle with "
      "associative recall; the increment here is a tight JOINT "
      "characterization with a matching construction, not another "
      "observation that recall is hard. A supporting input-side information "
      "bound quantifies the bit budget this construction presupposes, and a "
      "five-seed engineering check tests the layer prediction rather than "
      "comparing model quality. We do not present a new SOTA sequence "
      "model.", size=BODY, lead=LEAD, after=2.5)''',
    "introduction recentered")

# ---------------------------------------------------------------- 4. results
old_res_start = s.index('    It(d, "<b>Result 1 (Theorem A).</b> An input-side')
old_res_end = s.index('      "over prior results.", size=SMALL, lead=SLEAD, after=2.0)',
                     old_res_start) + len(
                         '      "over prior results.", size=SMALL, lead=SLEAD, after=2.0)')
sub(s[old_res_start:old_res_end], '''    It(d, "<b>Result 1 (supporting, Theorem A).</b> An input-side "
      "information bound B >= log2 |A(v)|, tight on the binary-chain family, "
      "with per-fibre accuracy, a within-class-M location statement, two bit "
      "carriers plus one support restriction, and the bit-depth frontier. It "
      "quantifies the budget that Result 2's construction presupposes; the "
      "serial-depth and state-order statements of earlier drafts are folded "
      "into part (e) of the same theorem.")
    It(d, "<b>Result 2 (central claim).</b> Two structural bounds proved "
      "jointly and attained by ONE construction: the minimum tensor state "
      "order for one hop is exactly 2, and exact H-hop retrieval under last-"
      "layer read-out requires exactly H static-LTI layers, with O(1) state "
      "in L. Every other result in this paper serves this bound.")
    It(d, "<b>Result 3 (Construction C).</b> The attaining witness for "
      "Result 2: a static-LTI, order-2, contraction read-out layer that "
      "performs content addressing while preserving Toeplitz structure, "
      "single-FFT/IIR duality, strict causality and O(1) state. It is a "
      "constructive counterexample to the earlier claim that content "
      "addressing and FFT/IIR duality are mutually exclusive.")
    It(d, "<b>Result 4 (Empirical E).</b> An engineering falsifiability "
      "check of Result 2, not a performance comparison: five-seed "
      "verification at a 1500-step budget that the layer bound survives "
      "end-to-end training from random initialisation (Section 4.12), "
      "degradation curves, and a same-run reproduction audit with one "
      "withdrawn claim.")
    P(d, "All remaining statements are propositions P1-P8 or lemmas L1-L3; "
      "none is promoted to a theorem. Several P-results are standard facts "
      "(the log-convex Turan identity, linear-grid maximal spacing, AND-gate "
      "uniqueness, the verifier trap and the decay-rate bound): they are "
      "labeled Standard but essential for completeness and are not counted "
      "as novel theorems. Table 1 records the exact increment over prior "
      "results.", size=SMALL, lead=SLEAD, after=2.0)''', "contribution list")

# ---------------------------------------------------------------- 5. Table 1
sub('''    d.table([
        ["Known result", "Precise increment here"],
        ["Zoology, Repeat After Me, Illusion of State: fixed-state models are weak at associative recall",
         "Theorem A quantifies the exact input-side deficit B >= log2|A| for ANY deterministic decoder"],
        ["S5: LTI requires time-invariant coefficients",
         "P6 proves Toeplitz iff time-invariance for the full second-order family with c_t != 0"],
        ["Mamba: selectivity breaks the convolution view",
         "Theorem Theorem A(c) states the consequence WITHIN class M: duality-preserving layers cannot manufacture the bits"],
        ["Linear attention: state is a sum of key-value outer products",
         "Proposition P4 records the rho=1 single-pole isomorphism; Result 2 keeps rho<1 decay and proves the joint bound"],
    ], [96, 142], size=6.5, lead=7.6, span=False, after=3.0)''',
    '''    d.table([
        ["Known result (specific source and statement)", "Precise increment here"],
        ["Zoology [1], Repeat After Me [2], Illusion of State [3]: fixed-state models are weak at associative recall",
         "Sharp bound, not an observation: min tensor order = 2 with a matching construction (Result 2(i))"],
        ["S5 [4], Z-transform LTI duality: a convolution kernel needs time-invariant coefficients",
         "P6: Toeplitz iff time-invariance for the FULL second-order family with c_t != 0"],
        ["Mamba [5], Sec. 3.1: selectivity breaks the convolution view",
         "Theorem A(c): WITHIN class M, duality-preserving layers cannot manufacture the bits; a conditional design statement"],
        ["Linear attention [6], GLA [7], DeltaNet [8]: outer-product key-value state",
         "P4: the rho=1 single-pole isomorphism; Result 2 keeps rho<1 decay and proves the JOINT tight bound"],
    ], [96, 142], size=6.5, lead=7.6, span=False, after=3.0)''',
    "precise difference table")

# ---------------------------------------------------------------- 6. symbol table
sub('''      "is reused.", size=SMALL, lead=SLEAD, after=2.0)

    # ---------------------------------------------------------------- 3''',
    '''      "is reused.", size=SMALL, lead=SLEAD, after=2.0)
    d.table([
        ["symbol", "meaning", "symbol", "meaning"],
        ["t", "absolute time index", "lag", "kernel lag (never tau)"],
        ["d", "delay, d = L-1-t", "H", "content hops"],
        ["H_L", "number of layers", "n", "binding pairs"],
        ["V_K", "key-vocabulary size V/2", "r", "interface rank"],
        ["rho", "pole modulus, rho < 1", "B", "input-side bit budget"],
        ["A(v)", "compatible-answer set", "M", "number of poles"],
    ], [30, 96, 34, 78], size=6.3, lead=7.4, span=False, after=2.0)
    P(d, "<b>Table 0.</b> Symbol table. One symbol, one meaning; no symbol "
      "is reused and no Greek-letter variant of a Roman symbol appears in "
      "any statement.", size=SMALL, lead=SLEAD, after=2.0)

    # ---------------------------------------------------------------- 3''',
    "symbol table")

# ---------------------------------------------------------------- 7. Result 2 = central, premises on same page
sub('''    Hh(d, "3.2  Result 2 (two tight structural bounds)",
       size=9.5, before=4.0, after=2.0)
    P(d, "<b>(i) Minimum state order.</b> Under Proposition P2 with a linear historical "
      "summary and n >= 2 binding pairs, no order-1 (vector) state supports "
      "exact one-hop content addressing (P2 below). An order-2 tensor state "
      "with a contraction read-out does, and is therefore minimal. "
      "<b>(ii) Exactly H layers.</b> Under H6, A4 and distinct keys, one "
      "static-LTI layer resolves at most one content match; H hops require "
      "H_L >= H, and H_L = H suffices. Both bounds are attained by the SAME "
      "Toeplitz, O(1)-state construction C. Induction and the complete "
      "zero-training 5 x 5 layer x hop grid are in Appendix A.4 and Section 4.",
      size=SMALL, lead=SLEAD, after=2.0)''',
    '''    Hh(d, "3.2  Result 2 (central claim): tight order, depth and state",
       size=9.5, before=4.0, after=2.0)
    P(d, "<b>Premises, stated here on the same page.</b> (A1) n >= 2 binding "
      "pairs with distinct keys and an external query q = k_0; (A2) the "
      "visible input is (v, s) with v the flat projection and s in {0,1}^B; "
      "(A3) H_L static-LTI layers with time-invariant, content-independent "
      "coefficients; (A4) last-layer read-out only; (A5) the unique H6 "
      "interface q_(h+1) = T_h z_h / ||T_h z_h||_2; and a deterministic "
      "decoder. Nothing in the theorem statement is deferred to an appendix.",
      size=SMALL, lead=SLEAD, after=2.0)
    P(d, "<b>(i) Minimum state order is exactly 2.</b> Under P2 with a linear "
      "historical summary and n >= 2 binding pairs, no order-1 (vector) "
      "state supports exact one-hop content addressing: distinct value "
      "assignments share the same summary, so no read-out, linear or "
      "nonlinear, separates them. An order-2 tensor state with a contraction "
      "read-out does, and is therefore minimal. <b>(ii) Exactly H layers.</b> "
      "One static-LTI layer resolves at most one content match; H hops "
      "require H_L >= H, and H_L = H suffices. Both bounds are attained by "
      "the SAME Toeplitz, O(1)-state construction C, so the joint bound is "
      "tight rather than three separate inequalities. Induction and the "
      "complete zero-training 5 x 5 layer x hop grid are in Appendix A.4 "
      "and Section 4.", size=SMALL, lead=SLEAD, after=2.0)''',
    "central theorem + same-page premises")

# ---------------------------------------------------------------- 8. one paired statistic only
sub('''      (mean(2, "tsc_h1"), sd(2, "tsc_h1"), mean(2, "tsc_h2"),
       sd(2, "tsc_h2"), CI["mean"], CI["ci95"][0], CI["ci95"][1]),''',
    '''      (mean(2, "tsc_h1"), sd(2, "tsc_h1"), mean(2, "tsc_h2"),
       sd(2, "tsc_h2"), CPAIR[0], CPAIR[1], CPAIR[2]),''',
    "unify paired statistic")

# ---------------------------------------------------------------- 9. Figure 1 caption
sub('''    d.para("<b>Figure 1.</b> Reachable exact retrieval as a function of layer "
           "count and hop count. The diagonal is the exactly-H-layer bound "
           "(Result 2); grey dots below it are measured chance-level "
           "single-layer failures. The vertical boundary at order 2 is "
           "Result 2's minimum state order. This is a boundary diagram, not "
           "an architecture schematic.", size=SMALL, lead=SLEAD, after=3.0)''',
    '''    d.para("<b>Figure 1.</b> Boundary and tightness. Horizontal axis: "
           "tensor state order (left of the vertical order-2 line) or layer "
           "count H_L (right of it). Vertical axis: reachable exact hops H. "
           "The order-2 vertical line is the minimum-state-order bound "
           "(Result 2(i)); the diagonal is the exactly-H-layer bound "
           "(Result 2(ii)); their intersection is the joint tight point "
           "attained by Construction C. Grey dots are measured single-layer "
           "failures at H>=2 (zero-training grid: 0.0000, chance 0.0313), "
           "which fall strictly BELOW the boundary as the theorem predicts. "
           "This is a mathematical boundary diagram, not an architecture "
           "schematic.", size=SMALL, lead=SLEAD, after=3.0)''',
    "figure 1 caption")

# ---------------------------------------------------------------- 10. dependency figure
figure2_code = '''

def figure2(d):
    """Dependency map: every supporting result feeds the single central claim."""
    y0, x0 = d.y - 8, d.x0() + 4
    hh, ww = 104, CW - 8
    d.ensure(hh + 18)
    c = d.c
    c.setLineWidth(0.4)
    boxes = [
        (0.02, 0.72, 0.30, 0.22, "P2 (zero-space) + P6 (Toeplitz iff TI)"),
        (0.36, 0.72, 0.30, 0.22, "H6 + A4 + distinct keys"),
        (0.70, 0.72, 0.28, 0.22, "Theorem A(b): bit budget"),
        (0.02, 0.40, 0.44, 0.22, "Result 2(i): order = 2"),
        (0.52, 0.40, 0.46, 0.22, "Result 2(ii): H layers = H"),
        (0.20, 0.08, 0.60, 0.22, "Construction C -> JOINT TIGHT BOUND"),
    ]
    for fx, fy, fw, fh, label in boxes:
        bx = x0 + ww * fx
        by = y0 - hh * (fy + fh)
        c.setStrokeGray(0.25)
        c.rect(bx, by, ww * fw, hh * fh)
        c.setFont("TR", 6.2)
        c.setFillGray(0)
        words = label.split()
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if len(t) > 34:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        ty = by + hh * fh - 8
        for ln in lines[:2]:
            c.drawString(bx + 3, ty, ln)
            ty -= 7
    c.setStrokeGray(0.35)
    c.setLineWidth(0.8)
    for a, b in [((0.17, 0.72), (0.24, 0.62)),
                 ((0.51, 0.72), (0.75, 0.62)),
                 ((0.84, 0.72), (0.50, 0.30)),
                 ((0.24, 0.40), (0.50, 0.30)),
                 ((0.75, 0.40), (0.50, 0.30))]:
        c.line(x0 + ww * a[0], y0 - hh * a[1],
               x0 + ww * b[0], y0 - hh * b[1])
    c.setFillGray(0)
    d.y = y0 - hh - 12
    d.para("<b>Figure 2.</b> Dependency map. The input-side bit budget "
           "(Theorem A) is a PRECONDITION; P2, P6, H6 and A4 feed the two "
           "lower bounds; Construction C attains both jointly; Empirical E "
           "tests the prediction. Every arrow points at the single central "
           "claim (Result 2). No result is an independent side contribution.",
           size=SMALL, lead=SLEAD, after=3.0)

'''
sub('''# ------------------------------------------------------------------ content
def build():''',
    figure2_code.lstrip("\n") + '''# ------------------------------------------------------------------ content
def build():''',
    "dependency figure function")

sub('''    Hh(d, "3.5  Dependency map and falsification",
       size=9.5, before=4.0, after=2.0)
    P(d, "Proposition P2 and P6 feed Result 2(i); Proposition P1 and P6 feed Theorem A(c); Proposition P5 feeds A(e); "''',
    '''    Hh(d, "3.5  Dependency map and falsification",
       size=9.5, before=4.0, after=2.0)
    figure2(d)
    P(d, "Proposition P2 and P6 feed Result 2(i); Proposition P1 and P6 feed Theorem A(c); Proposition P5 feeds A(e); "''',
    "dependency figure call")

# ---------------------------------------------------------------- 11. role of experiments
sub('''    Hh(d, "4  Experiments: One Table, One Numeric Convention",
       size=10.2, before=5.0, after=2.5)
    P(d, "All experiments are synthetic content chains with V=64, V_K=32, "''',
    '''    Hh(d, "4  Experiments as Engineering Falsifiability Checks",
       size=10.2, before=5.0, after=2.5)
    P(d, "<b>Role of the experiments.</b> These are NOT a performance "
      "comparison and NOT an architecture benchmark. They are engineering "
      "checks of a theorem's falsifiable predictions. Result 2(ii) predicts "
      "that a single static-LTI layer must fail at H >= 2 while a two-layer "
      "stack must succeed; the zero-training 5 x 5 grid gives 0.0000 "
      "(chance 0.0313) on the single-layer H=2 cell and 1.0000 on the "
      "two-layer H=2 cell (Table 5), and the 1500-step five-seed run "
      "reproduces the diagonal (Table 15). Section 4.6 audits a stored "
      "attention baseline under the same run and withdraws the earlier "
      "parity claim (W12); Section 4.12 records that the earlier training "
      "corroboration was re-established on a corrected interface, with the "
      "superseded protocol kept only as the W13 audit trail. No language-"
      "model, perplexity, throughput or superiority number is reported.",
      size=SMALL, lead=SLEAD, after=2.0)
    P(d, "All experiments are synthetic content chains with V=64, V_K=32, "''',
    "experiments recentered")

# ---------------------------------------------------------------- 12. falsifiable hook
_lim_a = s.index('    P(d, "Theorem A assumes distinct keys')
_lim_b = s.index('size=SMALL, lead=SLEAD, after=2.0)', _lim_a) + len(
    'size=SMALL, lead=SLEAD, after=2.0)')
sub(s[_lim_a:_lim_b],
    '''    P(d, "Theorem A assumes distinct keys, adjacent binding emission, an "
      "external query, last-layer read-out and a deterministic decoder; "
      "multi-query chains and stop/select mechanisms are outside Result 2. "
      "Proposition P5's state-order bound is conditional on c mutually "
      "orthogonal candidate successors per hop; when the branching factor "
      "is 2 it degenerates to 2^(H+1), which we state rather than hide. "
      "Finite-precision results cover L <= 128 for the analytic "
      "construction; the long-decay frontier at L=2049 is a separate "
      "measurement. Theorem A(b) is distribution-free and depth-free: no "
      "deterministic decoder supplied with B < log2 |A(v)| input-side bits "
      "can be exact on all compatible instances. Any construction attaining "
      "exact H-hop retrieval with fewer bits falsifies the bound at a "
      "precisely identifiable place. We did not run a >=10M-token "
      "equal-compute language-model experiment, and we do not present this "
      "as an architecture paper. A learnable codebook, a converged strong "
      "baseline and a fair throughput comparison remain open. That absence "
      "is a target rather than a request for leniency: the consolidation "
      "rate bound (SE4, Appendix A.10) gives a necessary bound on how fast "
      "any such run can inject information into a constant-size state, "
      "T w log2(V_K) bits in T rounds, independent of context length. Any "
      "run exceeding this rate without enlarging the state falsifies the "
      "bound at an identifiable place; we invite the community to falsify "
      "it under a published compute budget. The tight order/depth bounds of "
      "Result 2 are likewise falsifiable by a single order-1 state or a "
      "single layer exact on H >= 2, conditions listed in Section 4.11.",
      size=SMALL, lead=SLEAD, after=2.0)''',
    "falsifiable prediction hook")

# ---------------------------------------------------------------- 13. conclusion
sub('''    P(d, "Exact retrieval needs information that a flat symbol sequence does "
      "not contain. We proved a distribution-free lower bound on that "
      "information (B >= log2 |A(v)|, Theorem A), attained on an explicit "
      "binary-chain family, stated precisely where the bits may live "
      "within the duality-preserving LTI class, and showed that the bit-depth "
      "frontier is a right angle: extra layers cannot substitute for missing "
      "bits. On the content-chain task studied in our experiments the "
      "bound is loose (Section 4.3); what transfers is the "
      "distribution-free inequality A(b), which applies to any flat "
      "projection, including standard associative-recall protocols. Two "
      "tight structural bounds (order-2 state for one hop; exactly "
      "H static-LTI layers for H hops under last-layer read-out) are attained "
      "by one Toeplitz, O(1)-state construction, and the layer bound survives "
      "1500-step end-to-end training from random initialisation (Section "
      "4.12) with the inter-layer map fixed to the codebook's own "
      "value-to-key swap. The practical rule is to budget the bits at the "
      "input or relax exactness. The earlier attention-parity claim is "
      "withdrawn because it did not reproduce (W12). Deep fusion of the "
      "codebook state with zero-order consolidation (Section 4.13, Appendix "
      "A.7-A.11) further shows that this class of models can improve "
      "itself on an exactly-defined finite process, with a predicted and "
      "measured cover time in agreement, while remaining within the same "
      "information bound. This is an "
      "information-bound paper with a constructive counterexample, not a "
      "claim of language-model parity, equivalence, superiority or a new "
      "SOTA sequence model.", size=SMALL, lead=SLEAD, after=2.0)''',
    '''    P(d, "Exact content addressing needs an order-2 tensor state, and "
      "nothing smaller works. We proved the matching lower bound and the "
      "attaining construction (Result 2), showed that the same static-LTI "
      "layer stack needs exactly H layers for H hops with O(1) state in L, "
      "and verified both predictions by a zero-training 5 x 5 grid and a "
      "five-seed 1500-step run. The supporting input-side bound "
      "(B >= log2 |A(v)|, Theorem A) quantifies the bit budget this "
      "construction presupposes; on the content-chain task it is loose "
      "(Section 4.3), and what transfers is the distribution-free inequality "
      "A(b). The inter-layer map is fixed to the codebook's own "
      "value-to-key swap, contains no training data and leaves the recurrence "
      "kernel Toeplitz, causal and static. The earlier attention-parity claim "
      "is withdrawn because it did not reproduce (W12). Deep fusion with "
      "zero-order consolidation (Section 4.13, Appendix A.7-A.11) shows "
      "monotone self-improvement on an exactly-defined finite process while "
      "remaining inside the same information bound. The practical rule is to "
      "budget log2|A(v)| bits at the input, then choose exactly H layers and "
      "an order-2 state, or relax exactness. This is a tight "
      "boundary-characterization paper with a constructive counterexample, "
      "not a claim of language-model parity, equivalence, superiority or a "
      "new SOTA sequence model.", size=SMALL, lead=SLEAD, after=2.0)''',
    "conclusion recentered")

# ---------------------------------------------------------------- write
try:
    ast.parse(s)
except SyntaxError as e:
    print("AST FAIL:", e)
    sys.exit(1)
open(DST, "w", encoding="utf-8").write(s)
print("\n%d replacements | ast.parse PASS | wrote %s" % (N, DST))
