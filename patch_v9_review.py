#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 review patch: fix the issues listed by the reader, WITHOUT adding results,
changing numbers, or relaxing any check.

Scope (one-to-one with the reader's list):
  1. Honest positioning: the input-side inequality is a counting necessary
     condition, tight only on a constructed family and loose on the
     content-chain task; title no longer overclaims.
  2. One central reconciliation of "order 2" (tensor order, Result 2(i)) vs
     "o >= H+1" (conditional linear-summary rank bound, P5); design rule fixed.
  3. A.5: unify d/D, state the kernel-dimension condition in full.
  4. "Exactly H layers" explicitly conditional on H6 + A4; not a law.
  5. Abstract / Table 14: say plainly that the inter-layer map is FIXED;
     learned random interface stays 0.2492 +/- 0.0325, 0/5 converged.
  6. SE2: state the CI width as a fraction of the mean; the 0.10% gap has
     almost no discriminative power.
  7. Table 0 rebuilt as a clean two-column table; cross-references fixed;
     Appendix B sparse IDs explained and W10/W11 split; Appendix C / Broader
     Impact / Limitations stripped of author to-do and reviewer-directed meta.
  8. Appendix A.0 restates A1-A5, H6, all four core results and the complete
     CT7 statement, so the supplement is genuinely self-contained.
  9. Appendix G trajectories printed at checkpoints (full series in artifact)
     to keep the supplement readable when sent alone.
"""
import sys

SRC = "/data/workspace/build_simplcore_pdf_v8.py"
DST = "/data/workspace/build_simplcore_pdf_v9.py"
SPLIT = "/data/workspace/split_submission.py"

s = open(SRC, encoding="utf-8").read()
REPL = []


def rep(label, old, new):
    REPL.append((label, old, new))


# ---------------------------------------------------------------- 0. vars
rep("vars-LR",
    'OPTG = J(E + "opt_guard.json")\n',
    'OPTG = J(E + "opt_guard.json")\n'
    '_lr = next(x for x in CONV["summary"]\n'
    '           if x["arm"] == "legacy_learnable_T1_random")\n'
    'LR_M, LR_S, LR_C = _lr["mean"], _lr["sd"], _lr["converged_seeds"]\n')

# ---------------------------------------------------------------- 1. title
rep("title-meta",
    'self.c.setTitle("Exact Content Addressing Needs an Order-2 State: A Tight Joint Bound on Tensor Order, Depth and State Size for Static-LTI Sequence Models")',
    'self.c.setTitle("Exact Content Addressing Needs an Order-2 State: Joint Bounds for Static-LTI Layers under a Fixed Read-Out Interface")')
rep("title-lines",
    'for line in ["Exact Content Addressing Needs an Order-2 State:",\n'
    '                     "A Tight Joint Bound on Tensor Order, Depth and",\n'
    '                     "State Size for Static-LTI Sequence Models"]:',
    'for line in ["Exact Content Addressing Needs an Order-2 State:",\n'
    '                     "Joint Bounds for Static-LTI Layers",\n'
    '                     "under a Fixed Read-Out Interface"]:')

# ---------------------------------------------------------------- 2. abstract
rep("abstract",
    '''    P(d,
      "We prove that exact content addressing by a fixed-state decoder "
      "requires a tensor state of order exactly 2, and that one Toeplitz, "
      "strictly causal, O(1)-state construction attains this bound jointly "
      "with a sharp depth bound of H static-LTI layers for H hops "
       "under H6, last-layer read-out, distinct keys and one content "
       "match per layer (Result 2, the central claim of this paper). "
       "The difficulty is that "
      "any fixed-dimensional vector summary collapses distinct value "
      "assignments into a kernel of positive dimension, so no read-out, "
      "linear or nonlinear, can recover the queried binding; the supporting "
      "input-side bound is distribution-free and depth-free. Our method "
      "keeps the recurrence coefficients time-invariant and moves the "
      "key-query match into a bilinear contraction read-out outside the "
      "kernel, so the FFT/IIR duality and the constant-size state survive "
      "unchanged; the alphabet partition folds the mandatory bit into the "
      "symbol id, so it is not bit-free, it is annotation-free. Strongest "
      "evidence: the tight minimum tensor order of 2 "
      "has both a matching lower bound and an attaining construction, a "
      "zero-training 5 x 5 layer-by-hop grid is exact on its diagonal, and "
      "from random initialisation H=2 with two layers reaches %.4f +/- %.4f "
      "by step %d on five of five seeds while one layer stays at %.4f +/- "
      "%.4f (paired %+.4f, 95%% CI [%+.4f, %+.4f] at 1500 steps), with "
      "FFT/IIR agreement at machine precision and state bytes independent "
      "of L. The design rule is to budget log2|A(v)| input-side bits, then "
      "choose H layers under Result 2(ii)'s stated premises and an order-2 "
       "state; we make no language-"
      "modeling, perplexity, throughput, equivalence or superiority claim." %
      (cm(2, 2)[0], cm(2, 2)[1], CONV_FIRST,
       cm(2, 1)[0], cm(2, 1)[1], CPAIR[0], CPAIR[1], CPAIR[2]),
      size=SMALL, lead=SLEAD, after=3.0, span=True)''',
    '''    P(d,
      "We prove that, under a fixed linear historical summary, exact one-hop "
      "content addressing requires a tensor state of order exactly 2 "
      "(Result 2(i)), and give the attaining construction. Under the H6 "
      "layer interface, last-layer read-out, distinct keys and one content "
      "match per layer, the same construction also needs exactly H "
      "static-LTI layers for H hops; this depth statement is a property of "
      "the stated interface, not an architecture-independent law. The "
      "supporting inequality B >= log2|A(v)| is a counting (pigeonhole) "
      "necessary condition: tight on a constructed binary-chain family but "
      "LOOSE on the content-chain task studied here, so what transfers is "
      "the placement of the bits, not the inequality itself. The difficulty "
      "is that any fixed-dimensional vector summary collapses distinct "
      "value assignments into a kernel of positive dimension, so no "
      "read-out, linear or nonlinear, can recover the queried binding; our "
      "method keeps the recurrence coefficients time-invariant and moves "
      "the key-query match into a bilinear contraction read-out outside the "
      "kernel, preserving the FFT/IIR duality and the O(1) state. Strongest "
      "evidence: the order-2 bound has both a matching proof and an "
      "attaining construction, a zero-training 5 x 5 layer-by-hop grid is "
      "exact on its diagonal, and with the inter-layer map FIXED to the "
      "codebook value-to-key swap (only the external-query projection and "
      "read-out learned from random initialisation) H=2 with two layers "
      "reaches %.4f +/- %.4f by step %d on 5/5 seeds while one layer stays "
      "at %.4f +/- %.4f (paired %+.4f, 95%% CI [%+.4f, %+.4f] at 1500 "
      "steps); a fully learned inter-layer interface from random init "
      "reaches only %.4f +/- %.4f with %d/5 seeds converged (Table 14). "
      "The design rule is to budget log2|A(v)| input-side bits, then choose "
      "H layers under Result 2(ii)'s premises and an order-2 tensor state "
      "per layer; the separate P5 bound o >= H+1 is a conditional rank "
      "bound and does not replace the tensor-order result (Section 3.3.2). "
      "We make no language-modeling, perplexity, throughput, equivalence "
      "or superiority claim." %
      (cm(2, 2)[0], cm(2, 2)[1], CONV_FIRST,
       cm(2, 1)[0], cm(2, 1)[1], CPAIR[0], CPAIR[1], CPAIR[2],
       LR_M, LR_S, LR_C),
      size=SMALL, lead=SLEAD, after=3.0, span=True)''')

# ---------------------------------------------------------------- 3. intro
rep("intro-honest",
    '''      "five-seed engineering check tests the layer prediction rather than "
      "comparing model quality. We do not present a new SOTA sequence "
      "model.", size=BODY, lead=LEAD, after=2.5)''',
    '''      "five-seed engineering check tests the layer prediction rather than "
      "comparing model quality. The input-side inequality itself is a "
      "pigeonhole counting necessary condition; the increment is the proof "
      "of where those bits must be placed and the tight joint order/depth "
      "characterization, not the inequality alone. We do not present a new "
      "SOTA sequence model.", size=BODY, lead=LEAD, after=2.5)''')

rep("result2-order-note",
    '''       "factor P5's state-order statement reduces accordingly. "
       "Every other result in this paper serves this bound.")''',
    '''       "factor P5's state-order statement reduces accordingly. "
       "The two 'order' bounds are different quantities: Result 2(i) fixes "
       "the TENSOR ORDER of the state at 2 for one hop under P2, whereas "
       "P5's o >= H+1 is a conditional rank bound on a linear summary under "
       "c orthogonal successors per hop; they are reconciled in Section "
       "3.3.2 and do not conflict. Every other result in this paper serves "
       "this bound.")''')

# ---------------------------------------------------------------- 4. Table 0
rep("table0",
    '''    d.table([
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
      "any statement.", size=SMALL, lead=SLEAD, after=2.0)''',
    '''    d.table([
        ["symbol", "meaning"],
        ["t", "absolute time index"],
        ["d", "delay, d = L-1-t"],
        ["lag", "kernel lag (never tau)"],
        ["H", "content hops"],
        ["H_L", "number of layers"],
        ["n", "binding pairs"],
        ["V_K", "key-vocabulary size V/2"],
        ["r", "interface rank"],
        ["rho", "pole modulus, rho < 1"],
        ["B", "total input-side bit budget (bits)"],
        ["b", "bits per insight per round (CT7; distinct from B)"],
        ["D", "channel width"],
        ["c, o", "successor rank per hop, tensor order (P5)"],
        ["A(v)", "compatible-answer set"],
        ["M", "number of poles"],
    ], [40, 198], size=6.6, lead=8.0, span=False, after=2.0)
    P(d, "<b>Table 0.</b> Symbol table: one symbol, one meaning. B (total "
      "input bits), b (per-insight bits) and state bytes are three distinct "
      "quantities and are never compared numerically (Appendix A.0).",
      size=SMALL, lead=SLEAD, after=2.0)''')

# ---------------------------------------------------------------- 5. A(e) broken sentence
rep("Ae-broken",
    '''      "matching state-order bound: when each hop offers c mutually orthogonal "
      "candidate successors, exact H-hop retrieval requires an order-o "
      "tensor state with o >= H+1 (proved in Appendix A.5). Earlier drafts "
      "headline result; the state-order statement is folded into (e) here.",''',
    '''      "separate, CONDITIONAL rank bound: when each hop offers c mutually "
      "orthogonal candidate successors, a linear summary must have order "
      "o >= H+1 (Appendix A.5). It is not a second value for the tensor "
      "order of the state, which Result 2(i) fixes at 2 for one hop; "
      "Section 3.3.2 reconciles the two quantities.",''')

# ---------------------------------------------------------------- 6. Result 2(ii) conditional
rep("R2ii-conditional",
    '''      "read-out does, and is therefore minimal. <b>(ii) Exactly H layers.</b> "
      "One static-LTI layer resolves at most one content match; H hops "
      "require H_L >= H, and H_L = H suffices. Both bounds are attained by "''',
    '''      "read-out does, and is therefore minimal. <b>(ii) Exactly H layers "
      "under the stated interface.</b> "
      "One static-LTI layer resolves at most one content match; H hops "
      "require H_L >= H, and H_L = H suffices. Retaining and selecting "
      "intermediate outputs makes every H_L >= H sufficient (A(f)), so "
      "'exactly H' is a property of the H6 interface and A4 last-layer "
      "read-out, not an architecture-independent law. Both bounds are attained by "''')

# ---------------------------------------------------------------- 7. design rule + reconciliation
rep("design-rule",
    '''    P(d, "The three quantities form a step function rather than a smooth "
      "trade-off. Table 6 gives the measured frontier. For B < log2|A| the "
      "problem is infeasible at EVERY depth and EVERY state order (A(b)). At "
      "B >= log2|A| the minimum depth is H (Result 2(ii), under its "
      "premises) and the "
      "conditional minimum state order is H+1 (P5, when each hop offers d "
      "orthogonal successors). Pairwise scoring is never required inside "
      "the recurrence: the q.k match occurs in the contraction read-out "
      "outside the static kernel. Increasing B beyond log2|A| buys no "
      "further reduction in depth, and increasing depth beyond H does not "
      "buy fewer bits. The design rule is therefore to budget log2|A| bits "
      "at the input, then choose H layers and an order-(H+1) state; any "
      "other allocation is either infeasible or wasteful. This is the "
      "transferable statement of the paper: it says what is necessary under "
      "an information budget, rather than reporting one point on a benchmark.",
      size=SMALL, lead=SLEAD, after=2.0)''',
    '''    P(d, "The three quantities form a step function rather than a smooth "
      "trade-off. Table 6 gives the measured frontier. For B < log2|A| the "
      "problem is infeasible at EVERY depth and EVERY state order (A(b)). At "
      "B >= log2|A| the minimum depth is H (Result 2(ii), under its "
      "premises), and each layer uses an order-2 tensor state (Result 2(i)). "
      "P5's o >= H+1 is a SEPARATE conditional bound on the rank of a "
      "linear summary when each hop offers c mutually orthogonal candidate "
      "successors; it does not set the tensor order of the state. Pairwise "
      "scoring is never required inside the recurrence: the q.k match "
      "occurs in the contraction read-out outside the static kernel. "
      "Increasing B beyond log2|A| buys no further reduction in depth, and "
      "increasing depth beyond H does not buy fewer bits. The design rule "
      "is therefore to budget log2|A| bits at the input, then choose H "
      "layers under Result 2(ii)'s premises and an order-2 tensor state "
      "per layer; any other allocation is either infeasible or wasteful. "
      "The transferable statement is a necessary-condition frontier under "
      "an information budget, not a benchmark point.",
      size=SMALL, lead=SLEAD, after=2.0)
    P(d, "<b>Reconciliation of the two order bounds.</b> This is the single "
      "place where they are compared; the abstract, Section 1.1 and Section "
      "8 use the same convention. Result 2(i) concerns the TENSOR ORDER of "
      "the recurrent state under P2 (vector = order 1, matrix = order 2) "
      "for ONE hop, and is tight with an attaining construction. P5 "
      "concerns the minimum rank/order o of a LINEAR SUMMARY that must "
      "simultaneously resolve H+1 independent matching directions when "
      "each hop offers c orthogonal successors; it is a conditional "
      "counting bound, degenerates to a log2(H+1) bit requirement for "
      "branching factor 2, and makes no claim about the tensor order of "
      "the state. Construction C uses an order-2 tensor state at EACH of "
      "the H layers, so both statements hold without contradiction; "
      "neither is a lower bound on the other.", size=SMALL, lead=SLEAD,
      after=2.0)''')

# ---------------------------------------------------------------- 8. A.5 rewrite
rep("A5-rank",
    '''    P(d, "Premise. The interface is H6 with rank at most D; the codebook rows "
      "are mutually orthogonal; at each of H hops there are c mutually "
      "orthogonal candidate successors; answers are determined by the joint "
      "choice of the start key and the H successors, so the candidate set "
      "has size c^(H+1). Claim. An order-o tensor state whose per-layer "
      "linear summary has image dimension at most o*d cannot be exact unless "
      "o >= H+1. Proof. The H+1 independent matching directions must be "
      "simultaneously recoverable from the summary for every one of the "
      "c^(H+1) joint choices. If o <= H, the summary is a linear map from a "
      "space of dimension (H+1)d into one of dimension o*D <= H*D; its "
      "kernel has dimension at least d > 0. By the P2 construction (with the "
      "blocks taken to be the coefficient matrices of the H+1 directions) "
      "there exist two distinct joint choices producing the same summary and "
      "the same q, hence the same read-out, yet requiring different answers. "
      "Exactness is impossible; therefore o >= H+1. This is a CONDITIONAL "
      "bound: when each hop offers only 2 candidates the same counting gives "
      "2^(H+1) classes and the necessary order is weaker, degenerating to a "
      "log2(H+1) bit requirement. We state that degeneration explicitly "
      "rather than presenting c^(H+1) as unconditional. QED.",
      size=SMALL, lead=SLEAD, after=3.0, span=False)''',
    '''    P(d, "PREMISE. Let D be the channel width, the SAME D as in the state "
      "size 2MD + (K-1)D of Section 2.1, with D >= 1. The H6 interface has "
      "rank at most D; codebook rows are mutually orthogonal; at each of H "
      "hops there are c mutually orthogonal candidate successors; answers "
      "are determined by the joint choice of the start key and the H "
      "successors, giving c^(H+1) classes; the H+1 matching directions are "
      "linearly independent and each lives in a D-dimensional subspace, so "
      "their joint span has dimension exactly (H+1)D. CLAIM. An order-o "
      "tensor state whose per-layer linear summary has image dimension at "
      "most oD cannot be exact unless o >= H+1. PROOF. The summary is a "
      "linear map Phi : R^((H+1)D) -> R^(oD). If o <= H then "
      "dim ker Phi >= (H+1-o)D >= D > 0, because D >= 1. Applying the "
      "explicit three-case P2 construction (Appendix A.3) with the "
      "coefficient blocks given by the H+1 matching directions yields two "
      "distinct joint choices with the same summary and the same q, hence "
      "the same read-out, but different required answers; exactness is "
      "impossible, so o >= H+1. SCOPE. This is a CONDITIONAL rank bound on "
      "a linear summary, not a statement about the tensor order of the "
      "recurrent state (Result 2(i) fixes that at 2 for one hop). When each "
      "hop offers only 2 candidates the counting gives 2^(H+1) classes and "
      "the necessary order degenerates to a log2(H+1) bit requirement; we "
      "state that degeneration rather than presenting c^(H+1) as "
      "unconditional. The condition D >= 1 and the independence of the "
      "H+1 directions are part of the premise, not derived facts. QED.",
      size=SMALL, lead=SLEAD, after=3.0, span=False)''')

# ---------------------------------------------------------------- 9. strip meta / to-do
rep("compute-todo",
    '''      "and generation quality are NOT measured. The 2027 call for papers "
      "and submission deadline had not been published when this version was "
      "prepared; page count, the author-side LLM-use disclosure and "
      "reciprocal-reviewer obligations must be rechecked against the "
      "official template when it appears.", size=SMALL, lead=SLEAD,''',
    '''      "and generation quality are NOT measured.", size=SMALL, lead=SLEAD,''')

rep("impact-todo",
    '''      "conclusions were specified and verified by the human author. "
      "This disclosure is subject to the ICML 2027 author instructions, "
      "which are not yet published.", size=SMALL, lead=SLEAD, after=2.0)''',
    '''      "conclusions were specified and verified by the human author.",
      size=SMALL, lead=SLEAD, after=2.0)''')

rep("appC-todo",
    '''      "Seeds are listed per table. The environment, "
      "dependency versions and a one-command reproduction script are in the "
      "artifact. The 2027 call for papers and submission deadline were not "
      "published at the time of writing; page count, the author-side LLM-use "
      "disclosure and reciprocal-reviewer obligations must be rechecked "
      "against the official template when it appears.",
      size=SMALL, lead=SLEAD, after=3.0, span=False)''',
    '''      "Seeds are listed per table. The environment, "
      "dependency versions and a one-command reproduction script are in the "
      "artifact.", size=SMALL, lead=SLEAD, after=3.0, span=False)''')

rep("limitations-meta",
    '''      "bound at an identifiable place; we invite the community to falsify "
      "it under a published compute budget. The tight order/depth bounds of "''',
    '''      "bound at an identifiable place; it is stated so that any such run "
      "can be checked against a published compute budget. The tight "
      "order/depth bounds of "''')

# ---------------------------------------------------------------- 10. cross refs
rep("xref-413",
    '''      "trajectories and the gate x corruption grid are Tables 18-19 in "
      "Appendix A.7-A.11. These are theorems and measurements on an "''',
    '''      "trajectories and the gate x corruption grid are Tables 18-19 in "
      "Appendix D (per-seed endpoints in Appendix J). These are theorems "
      "and measurements on an "''')

rep("table14-learned",
    '''      "annotation." % (CONV["protocol"]["lr"], CONV["protocol"]["batch"]),
      size=SMALL, lead=SLEAD, after=2.0)''',
    '''      "annotation. The random learnable interface reaches %.4f +/- %.4f "
      "with %d/5 seeds converged at this budget, so the convergence "
      "reported here is NOT evidence that the inter-layer map is learned." %
      (CONV["protocol"]["lr"], CONV["protocol"]["batch"],
       LR_M, LR_S, LR_C),
      size=SMALL, lead=SLEAD, after=2.0)''')

# ---------------------------------------------------------------- 12. SE2 CI width
rep("se2-ci",
    '''       "percent, 95%% t-CI [%.2f, %.2f] rounds, n = 5). The mean agrees "
       "with the prediction, but the seed-to-seed spread is large: the "
       "small relative gap is a coincidence of the MEAN and is NOT a "
       "precision claim. The premise was fixed to i.i.d. arrival BEFORE "''',
    '''       "percent, 95%% t-CI [%.2f, %.2f] rounds, n = 5). The CI spans "
       "%.0f rounds, about %.0f%% of the mean, so the numerical agreement "
       "has almost no discriminative power: the small relative gap is a "
       "coincidence of the MEAN and is NOT a precision claim. SE2's "
       "evidential content is the premise-fixed prediction direction, not "
       "the 0.10%% gap. The premise was fixed to i.i.d. arrival BEFORE "''')

rep("se2-tuple",
    '''        SE2_CI["ci95"][0], SE2_CI["ci95"][1]),
       size=SMALL, lead=SLEAD, after=2.5)''',
    '''        SE2_CI["ci95"][0], SE2_CI["ci95"][1],
        SE2_CI["ci95"][1] - SE2_CI["ci95"][0],
        100 * (SE2_CI["ci95"][1] - SE2_CI["ci95"][0]) /
        AGREE["SE2_measured_mean_first_cover"]),
       size=SMALL, lead=SLEAD, after=2.5)''')

# ---------------------------------------------------------------- 13. Appendix B
rep("B-explain",
    '''    Hh(d, "Appendix B  Withdrawn Statements", size=11.5, before=6.0,
       after=3.0, span=False)
    d.table([''',
    '''    Hh(d, "Appendix B  Withdrawn Statements", size=11.5, before=6.0,
       after=3.0, span=False)
    P(d, "Identifiers are stable labels assigned when each withdrawal was "
      "first logged. Gaps in the sequence are retired entries with no "
      "replacement and are intentionally not renumbered, so cross-version "
      "references remain valid; no entry is missing from this register. "
      "It is an audit aid, not a contribution.", size=SMALL, lead=SLEAD,
      after=2.5, span=False)
    d.table([''')

rep("B-W10W11",
    '        ["W10/W11", "\'Exactly one bit per token is necessary and sufficient\': the necessary quantity is total B >= log2|A|; one bit per token is a sufficient carrier."],',
    '        ["W10", "\'Exactly one bit per token is NECESSARY\': withdrawn; the necessary quantity is total B >= log2|A(v)|, not one bit at every token."],\n'
    '        ["W11", "\'Exactly one bit per token is SUFFICIENT on every task\': withdrawn; one bit per ambiguous token is a sufficient carrier on the binary-chain family only."],')

# ---------------------------------------------------------------- 14. A.0 self-contained
rep("A0-restated",
    '''    Hh(d, "A.0  Notation and unit ledger", size=9.8, before=4.0,
        after=2.0, span=False)
    d.table([''',
    '''    Hh(d, "A.0  Restated premises, interface, claim statements and notation",
       size=9.8, before=4.0, after=2.0, span=False)
    P(d, "This section makes the supplement SELF-CONTAINED: every premise, "
      "the layer interface, and the statements being proved are restated "
      "here, so no proof in Appendices A.1-A.11 requires the main text.",
      size=SMALL, lead=SLEAD, after=2.0, span=False)
    P(d, "<b>A1-A5 (premises, restated).</b> A1 (task): an instance has n "
      "binding pairs with distinct keys, a role assignment deciding which "
      "token of each pair is the KEY, arbitrary pair order, arbitrary zero "
      "fillers, and answer k_H(G), the endpoint of the H-hop chain from an "
      "external query q = k_0. A2 (visible input): the model sees (v, s) "
      "with v the flat projection and s in {0,1}^B the refinement. A3 "
      "(class M): H_L static-LTI layers with time-invariant, content-"
      "independent coefficients, fixed-dimensional state, fixed linear "
      "interfaces and deterministic last-layer read-out. A4: only the last "
      "layer is decoded. A5: two accuracy notions - per-fibre conditional "
      "acc(F|v) under the uniform measure on the fibre, and marginal "
      "E_v[acc(F|v)]; Theorem A bounds the first. Parts (a) and (b) of "
      "Theorem A use none of A3.", size=SMALL, lead=SLEAD, after=2.0,
      span=False)
    P(d, "<b>H6 (unique layer interface, restated).</b> Layer h computes "
      "m_h[t] = <q_h, e(x_(t-1))> and z_h = sum_t K_h(L-1-t) m_h[t] "
      "e(x_t); for h < H_L the interface is q_(h+1) = T_h z_h / "
      "||T_h z_h||_2 with T_h in R^(D x D). The analytic optimum is "
      "T* = C_val^T C_key, mapping a VALUE code to the next-hop KEY code; "
      "the read-out is last-layer only (A4). The recurrent state has "
      "2MD + (K-1)D scalars, independent of L; strict causality follows by "
      "induction on the two-tap recurrence. B = sum_t log2|pi^(-1)(v_t)| "
      "is the total input-side bit budget, pi erasing all role information "
      "to a flat reference alphabet of size V_K = V/2. An alphabet "
      "partition folds the mandatory bit into the symbol id: it is not "
      "bit-free, it is annotation-free.", size=SMALL, lead=SLEAD,
      after=2.0, span=False)
    P(d, "<b>Four core results (restated).</b> Result 1 (Theorem A, "
      "supporting): (a) acc(F|v) <= 2^B/|A(v)| on each fibre; (b) exactness "
      "requires B >= log2|A(v)| for every v; (c) within class M, IF duality "
      "is preserved the bits must enter on the input side (a design "
      "consequence, not a physical force); (d) the bound is tight on a "
      "binary-chain family with two carriers counted in B; (e) the "
      "(B, minimum H_L) frontier is a right angle; (f) overshoot H_L > H "
      "is a read-out convention. The inequality itself is a pigeonhole "
      "counting necessary condition and is LOOSE on the content-chain task. "
      "Result 2 (central): under A1-A5, H6 and A4, the minimum tensor state "
      "order for one hop is exactly 2 and H hops need exactly H static-LTI "
      "layers, both attained by Construction C; the depth claim is "
      "conditional on the interface and is not an architectural law. "
      "Result 3 (Construction C): a static-LTI order-2 tensor state with a "
      "bilinear contraction read-out that performs content addressing "
      "while preserving Toeplitz, FFT/IIR duality, causality and O(1) "
      "state. Result 4 (Empirical E): a five-seed 1500-step engineering "
      "falsifiability check of the layer bound, not a performance "
      "comparison. P1-P8 and L1-L3 are propositions and lemmas; no theorem "
      "is introduced in this supplement.", size=SMALL, lead=SLEAD,
      after=2.0, span=False)
    P(d, "<b>CT7 (zero-order self-evolution, complete statement).</b> "
      "(a) In the constant-state architecture the online retrieval path "
      "answer() is executed without gradients and the hard top-k selection "
      "has an almost-everywhere-vanishing Jacobian; a long-context "
      "experience written back into weights must therefore use a zero-order "
      "consolidation operator. (b) T consolidation rounds inject at most "
      "T w b bits into the state (w insights per round, b bits per "
      "insight), independent of context length L and bank capacity; the "
      "instantaneous bandwidth is at most k b bits. (c) For the finite "
      "consolidation process defined in A.7-A.11, monotone improvement is "
      "proved by SE1-SE5, resolving the earlier statement that improvement "
      "was not proved, for that process only. CT7 is a supporting result, "
      "not one of the four core results, and makes no general "
      "self-improvement claim.", size=SMALL, lead=SLEAD, after=2.0,
      span=False)
    d.table([''')

# ---------------------------------------------------------------- 15. Appendix G compress
rep("G-title",
    '    Hh(d, "Appendix G  All Fusion Evolution Curves", size=11.5,',
    '    Hh(d, "Appendix G  Fusion Evolution Curves at Checkpoints", size=11.5,')
rep("G-intro",
    '''    P(d, "Section 4.13 reports final accuracies. Here the full trajectory "
      "of each arm is printed: accuracy is sampled every 10 rounds and "
      "averaged over the five seeds at each sampled round. Protocol: 32 "''',
    '''    P(d, "Section 4.13 reports final accuracies. Here each trajectory is "
      "printed at 40-round checkpoints (plus the final round), averaged "
      "over the five seeds; the complete 10-round series and every "
      "individual per-seed curve are in the anonymised artifact, and "
      "Appendix J lists the per-seed endpoints. Protocol: 32 "''')
rep("G-rounds",
    '''    _rounds = sorted({int(c["round"]) for r in FEV["rows"]
                      for c in r["curve"]})''',
    '''    _all_rounds = sorted({int(c["round"]) for r in FEV["rows"]
                          for c in r["curve"]})
    _rounds = [x for x in _all_rounds
               if x % 40 == 0 or x == _all_rounds[-1]]''')

# ---------------------------------------------------------------- 16. conclusion
# ---- conclusion: extract the exact old block from source (whitespace-safe) ----
_ca = s.find('    P(d, "Exact content addressing needs an order-2 tensor state, and "')
_cb = s.find('    d.end_main()', _ca)
assert _ca >= 0 and _cb > _ca, "conclusion block not found"
_concl_new = '''    P(d, "Exact content addressing needs an order-2 tensor state under a "
      "fixed linear historical summary, and nothing smaller works. We "
      "proved the matching lower bound and the attaining construction "
      "(Result 2), showed that the same static-LTI layer stack needs H "
      "layers for H hops under H6, A4, distinct keys and one content match "
      "per layer - a conditional interface statement, not an architectural "
      "law - and verified both predictions by a zero-training 5 x 5 grid "
      "and a five-seed 1500-step run. The supporting input-side bound "
      "(B >= log2|A(v)|, Theorem A) is a counting necessary condition that "
      "quantifies the bit budget the construction presupposes; on the "
      "content-chain task it is LOOSE (Section 4.3), so what transfers is "
      "the distribution-free inequality A(b) and the proof of where the "
      "bits must be placed, not a tight numerical bound on that task. The "
      "inter-layer map is FIXED to the codebook's own value-to-key swap: "
      "only the external-query projection and read-out are learned, and a "
      "fully learned inter-layer interface from random init does not "
      "converge (Table 14). The earlier attention-parity claim is withdrawn "
      "because it did not reproduce (W12). Deep fusion with zero-order "
      "consolidation (Section 4.13, Appendices A.9-A.10 and D) shows "
      "monotone self-improvement on an exactly-defined finite process while "
      "remaining inside the same information bound. The practical rule is "
      "to budget log2|A(v)| bits at the input, then choose H layers under "
      "Result 2(ii)'s premises and an order-2 tensor state per layer, or "
      "relax exactness; P5's o >= H+1 is a separate conditional rank bound "
      "(Section 3.3.2). This is a boundary-characterization paper with a "
      "constructive counterexample, not a claim of language-model parity, "
      "equivalence, superiority or a new SOTA sequence model.",
      size=SMALL, lead=SLEAD, after=2.0)

'''
REPL.insert(0, ("conclusion", s[_ca:_cb], _concl_new))

# ---------------------------------------------------------------- apply
for label, old, new in REPL:
    n = s.count(old)
    if n != 1:
        print("FAIL %-24s count=%d" % (label, n))
        sys.exit(2)
    s = s.replace(old, new, 1)
    print("  ok %-24s" % label)

open(DST, "w", encoding="utf-8").write(s)
print("\nwrote", DST, len(s), "chars")

# ---------------------------------------------------------------- split cover
t = open(SPLIT, encoding="utf-8").read()
SR = []
SR.append((
    '''    ("Anonymous submission. This document contains the appendices to the", 9.5, 176),
    ("main paper. It is self-contained: each proof restates the final", 9.5, 192),
    ("premises and the identifier of the claim it establishes, and proofs", 9.5, 208),
    ("cross-reference those identifiers rather than referring to 'the", 9.5, 224),
    ("proof above'. Numerical witnesses certify non-vacuity and", 9.5, 240),
    ("agreement; they are never used as premises.", 9.5, 256),''',
    '''    ("Anonymous submission. This document contains the appendices to the", 9.5, 176),
    ("main paper. It is self-contained: Appendix A.0 restates every", 9.5, 192),
    ("premise (A1-A5), the unique layer interface H6, all four core", 9.5, 208),
    ("results and the complete CT7 statement. Each proof then restates", 9.5, 224),
    ("the premises it uses and the identifier of the claim it proves;", 9.5, 240),
    ("numerical witnesses certify non-vacuity and agreement, and are", 9.5, 256),
    ("never used as premises.", 9.5, 272),'''))
SR.append((
    '    ("", 10, 280),\n    ("Contents", 11, 300),\n    ("Appendix A   Complete analytic proofs (A.1-A.11)", 9.5, 322),',
    '    ("", 10, 292),\n    ("Contents", 11, 312),\n    ("Appendix A   Restated premises (A.0) and complete proofs (A.1-A.11)", 9.5, 334),'))
SR.append((
    '''    ("Appendix B   Withdrawn statements", 9.5, 340),
    ("Appendix C   Reproducibility", 9.5, 358),
    ("Appendix D   All raw fusion measurements", 9.5, 376),
    ("Appendix E   Premise, boundary and non-vacuity ledger", 9.5, 394),
    ("Appendix F   All 1500-step convergence curves", 9.5, 412),
    ("Appendix G   All fusion evolution curves", 9.5, 430),
    ("Appendix H   Exhaustive information-bound enumeration", 9.5, 448),
    ("Appendix I   Complete numeric provenance and witness detail", 9.5, 466),
    ("Appendix J   Per-seed raw curves at key steps", 9.5, 484),''',
    '''    ("Appendix B   Withdrawn statements", 9.5, 352),
    ("Appendix C   Reproducibility", 9.5, 370),
    ("Appendix D   All raw fusion measurements", 9.5, 388),
    ("Appendix E   Premise, boundary and non-vacuity ledger", 9.5, 406),
    ("Appendix F   All 1500-step convergence curves", 9.5, 424),
    ("Appendix G   Fusion evolution curves at checkpoints", 9.5, 442),
    ("Appendix H   Exhaustive information-bound enumeration", 9.5, 460),
    ("Appendix I   Complete numeric provenance and witness detail", 9.5, 478),
    ("Appendix J   Per-seed raw curves at key steps", 9.5, 496),'''))
SR.append((
    '''    ("", 10, 510),
    ("The main paper states four core results. SE1-SE5 in Appendix A.7-A.11", 9.5, 532),
    ("are supporting statements about an exactly-defined finite", 9.5, 548),
    ("consolidation process; P1-P8 and L1-L3 are propositions and lemmas.", 9.5, 564),
    ("No theorem is introduced in the supplementary material.", 9.5, 580),''',
    '''    ("", 10, 522),
    ("The main paper states four core results, all restated in A.0.", 9.5, 544),
    ("SE1-SE5 (A.7-A.11) are supporting statements about an exactly", 9.5, 560),
    ("defined finite consolidation process; P1-P8 and L1-L3 are", 9.5, 576),
    ("propositions and lemmas. No theorem is introduced here.", 9.5, 592),'''))
SR.append((
    '''           note=("The 2027 Call for Papers and author instructions are not "
                 "yet published. A single combined PDF plus supplementary "
                 "material is the 2026 convention; this two-PDF split is "
                 "the skill pack's internal deliverable and must be "
                 "repacked to whatever the 2027 submission system requires."))''',
    '''           note=("This two-PDF split is an internal deliverable and must "
                 "be repacked to whatever the submission system requires."))'''))
for i, (o, nw) in enumerate(SR):
    c = t.count(o)
    if c != 1:
        print("SPLIT FAIL block %d count=%d" % (i, c))
        sys.exit(3)
    t = t.replace(o, nw, 1)
open(SPLIT, "w", encoding="utf-8").write(t)
print("patched", SPLIT)
