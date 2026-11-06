"""
Formal statements of the 15 candidate propositions.

This module is importable from the architecture.  It carries NO training code,
NO model weights and NO experimental data.  Each proposition records:
  - its exact hypothesis (the assumptions under which it holds),
  - its claim,
  - its verdict (THEOREM / FALSIFIED / CONJECTURE),
  - the existing architecture module it constrains,
  - the machine-check identifier in verify.py.

Verdict discipline: a theorem is a universal statement derived only from
definitions.  Measured accuracies are finite-sample observations and are never
promoted to theorems here.

NUMBERING STATUS (important):
  The T7..T16 ids below are HISTORICAL architecture-internal ids from the
  previous round.  They have been CONSOLIDATED by smm_sgc.CORE_THEOREMS into
  a 7-theorem core set (CT1..CT7).  In particular:
    * T16 is WITHDRAWN (replaced by CT7, which matches the actual upstream
      multi-metric paired-bootstrap gate instead of a scalar monotone score);
    * T7,T8,T9,T12,T14 are DEMOTED to lemmas inside CT5 / appendix;
    * T11 strengthens CT3, T10 strengthens CT4, T15 is a CT6 corollary.
  See docs/SMM_SGC_SELF_EVOLUTION.md for the consolidation table.
  This module is kept for the complete audit trail, NOT as the submission
  theorem list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Proposition:
    pid: str
    theorem: Optional[str]          # T-number if upgraded, else None
    title: str
    hypothesis: str
    claim: str
    proof: str
    verdict: str                   # THEOREM | FALSIFIED | CONJECTURE
    module: str                    # architecture module affected
    check: str                     # verify.py function name
    note: str = ""


PROPOSITIONS: List[Proposition] = [
    Proposition(
        pid="P01", theorem="T7",
        title="Tensor-channel Toeplitz property and FFT/IIR duality",
        hypothesis="rho_{p,n} is independent of t and of the code index r; "
                   "u_t[r,n] = a_t[r] b_t[n]; recurrence s_t = rho s_{t-1} + u_t.",
        claim="The channel-wise impulse response is h_{p,n}(t,s) = rho_{p,n}^{t-s}, "
              "hence Toeplitz; a single FFT convolution and the streaming recurrence "
              "compute the same operator over the reals.",
        proof="Unroll the first-order recurrence: s_t = sum_{tau<=t} rho^{t-tau} u_tau. "
              "The coefficient of u_s depends on t-s only. Toeplitz kernels are exactly "
              "those diagonalizable by one DFT, giving the FFT path.",
        verdict="THEOREM", module="core/arch/sgc.py + theory/TSC", check="t7",
        note="Real-field equality only; fp32 rounding order differs (Thm1 remark). "
             "Verified: impulse dev 2.22e-16, FFT/IIR rel 4.53e-16."),
    Proposition(
        pid="P02", theorem="T8",
        title="Closed form of the contraction read-out",
        hypothesis="z_n = sum_{p,r} w_{p,n} q_r s_{p,r,n,L}; s is the TSC state of P01.",
        claim="z_n = sum_tau (q . k_tau) K_n(L-1-tau) v_{tau,n} with "
              "K_n(d) = sum_p w_{p,n} rho_{p,n}^d. Content addressing is an explicit "
              "kernel, not an emergent behaviour.",
        proof="Substitute the unrolled state and exchange the summation order. "
              "The inner sum over r is exactly q . a_tau = q . k_tau.",
        verdict="THEOREM", module="theory/TSC", check="t8",
        note="Verified max abs diff 1.33e-15. Makes the (q,k) match term explicit."),
    Proposition(
        pid="P03", theorem="T9",
        title="Zero crosstalk of a deterministic orthogonal code",
        hypothesis="Code is a Sylvester Hadamard matrix of order V = 2^k; "
                   "query and key rows are code rows.",
        claim="Code Code^T = V I, so q . k_tau = V delta_{q,k_tau}; off-diagonal "
              "interference is exactly zero.",
        proof="Inductive Sylvester construction H_{2m} = [[H,H],[H,-H]] preserves "
              "H H^T = m I with m doubled at each step.",
        verdict="THEOREM", module="core/arch/sgc.py (init_mode grid) + theory/TSC",
        check="t9",
        note="V=64 verified: Gram off-diagonal 0.0, diagonal 64.0. "
             "For non-power-of-two V a random orthogonal projection gives the "
             "same property with probability 1, but that is an existence result, "
             "not a deterministic construction."),
    Proposition(
        pid="P04", theorem="T10",
        title="Quantitative kernel lower bound and the n=1 exception",
        hypothesis="History summary L: R^{nd} -> R^d is linear; n key-value pairs; "
                   "values independently assignable.",
        claim="dim ker L >= nd - d. For n=1 the kernel can be zero, so the "
              "'at least two distinguishable targets' premise of Thm5 is necessary "
              "and cannot be dropped.",
        proof="Rank-nullity: rank L <= d, hence nullity >= nd - d. For n=1 a random "
              "square L has full rank almost surely, giving ker = 0.",
        verdict="THEOREM", module="core/arch/sgc.py (vector-signal path)", check="t10",
        note="n=1..6 verified. This quantifies, rather than merely asserts, the "
             "arbitrary-read-out impossibility of Thm5."),
    Proposition(
        pid="P05", theorem="T11",
        title="P-order generalization of the Toeplitz equivalence",
        hypothesis="y_t = sum_{j=1..P} a_{j,t} y_{t-j} + c_t u_t with c_t != 0.",
        claim="The impulse response is Toeplitz iff every coefficient a_{j,t} and "
              "c_t is constant in t. This generalizes Thm4 from P=2 to any order.",
        proof="By induction on the delay tau: tau=0 forces c constant; tau=k forces "
              "a_{k} constant once a_1..a_{k-1} are known. Sufficiency is the LTI "
              "transfer function. P>=1 arbitrary.",
        verdict="THEOREM", module="core/arch/sgc.py (general second-order IIR dictionary)",
        check="t11",
        note="P=1..5 verified: constant coefficients give deviation 0.0; "
             "time-varying give 0.09..0.35."),
    Proposition(
        pid="P06", theorem="T12",
        title="BIBO stability iff rho<1; rho=1 is critical",
        hypothesis="Scalar/per-channel decay rho>0; bounded input |u_t|<=U.",
        claim="sum_tau |rho^tau| < infinity iff rho<1, with sum = 1/(1-rho). "
              "For rho=1 the state amplitude grows as Theta(L) and any "
              "finite-precision register eventually overflows.",
        proof="Geometric series. At rho=1 the partial sums are exactly L*u, so no "
              "finite bound exists.",
        verdict="THEOREM", module="core/arch/sgc.py + deploy/", check="t12",
        note="rho=1 gives perfect finite-L accuracy in fp32 but is not BIBO stable; "
             "this is why the architecture selects rho=0.999."),
    Proposition(
        pid="P07", theorem="T13",
        title="Quantization resolvable radius",
        hypothesis="Quantizer Q with uniform step s satisfying |Q(x)-x| <= s/2; "
                   "bounded input |u|<=U. No peak-scale model is assumed.",
        claim="A history contribution rho^tau u is indistinguishable from zero once "
              "rho^tau U < s/2. The resolvable radius is "
              "tau*(rho,r) = ln(r)/ln(rho) with r = s/(2U) in (0,1), and "
              "tau* diverges as rho -> 1.",
        proof="Direct inequality plus monotonicity of ln(rho). tau* ~ 1/|ln rho|.",
        verdict="THEOREM", module="deploy/ (low-precision state registers)", check="t13",
        note="CONDITIONAL theorem: uniform-step quantizer. tau* is a conservative "
             "sufficient radius, NOT an exact drop-point predictor. The physics probe "
             "uses per-step dynamic scaling, so measured drop points (int8 L=2049, "
             "fp16 L=4097, fp32 L=16385) are consistent in direction but are NOT "
             "derived from this theorem."),
    Proposition(
        pid="P08", theorem="T14",
        title="Dynamic scale dominates a fixed global scale",
        hypothesis="Peak-calibrated quantization; effective resolvable levels "
                   "are inversely proportional to the step size.",
        claim="For a state whose peak grows with L, a scale calibrated once at the "
              "largest L yields effective resolution no greater than a scale "
              "recomputed from the current peak, for every L.",
        proof="Let peak(L) be nondecreasing. Fixed scale = peak(L_max)/B, dynamic "
              "scale = peak(L)/B. Since peak(L) <= peak(L_max), "
              "1/fixed_scale <= 1/dynamic_scale.",
        verdict="THEOREM", module="deploy/ (quantization policy)", check="t14",
        note="Verified over L=65..16385. This is the analytic counterpart of the "
             "measured result that fixed int8 scales degrade earlier (L=1025 -> 0.19) "
             "than dynamic scaling. It does NOT claim dynamic scaling is optimal."),
    Proposition(
        pid="P09", theorem="T15",
        title="One-hop / two-hop tensor-degree separation",
        hypothesis="A single-layer contraction read-out z = sum_tau g(q,k_tau) v_tau.",
        claim="The output is degree 1 in the history values and carries one factor "
              "of q. A two-hop target sum_{tau1,tau2} g(q,k_tau1) g(k_tau1,k_tau2) "
              "v_tau2 is degree 2 and cannot be represented by that form. Escaping "
              "requires raising the tensor order of the state (a NECESSARY condition).",
        proof="Compare polynomial degree in the history. A linear form cannot equal "
              "a nonzero quadratic form on an open set; least-squares residual is "
              "strictly positive.",
        verdict="THEOREM", module="theory/TSC + core/arch/", check="t15",
        note="Verified: one-hop fit RMSE 19.43, with second-order term 1.45e-15. "
             "The observed two-hop accuracy 0.0322 is a finite-sample observation, "
             "not a theorem. No sufficient multi-hop construction is given."),
    Proposition(
        pid="P10", theorem="T16",
        title="SMM consolidation monotonicity and bounded tiers",
        hypothesis="SimplModelMemory's no-regression gate: a consolidation commit "
                   "is accepted iff the verifier score v_new >= v_old; otherwise "
                   "weights and the episodic bank are rolled back. Bank capacity "
                   "is fixed at C.",
        claim="(a) The verifier score sequence is non-decreasing over the full "
              "lifecycle. (b) The episodic tier has at most C entries. "
              "Together with Thm3 (O(1) working state), the resident state is "
              "bounded independently of sequence length.",
        proof="(a) Induction: each step either commits with v_new >= v_old or "
              "rolls back with v unchanged. (b) LRU eviction enforces |bank| <= C.",
        verdict="WITHDRAWN", module="core/memory/smm.py + core/memory/dream/", check="t16",
        note="WITHDRAWN and REPLACED BY CT7. Reason: this was a trivial induction "
             "on a scalar score AND it did not match the source. Upstream "
             "regression_gate uses multi-metric paired-bootstrap CI lower bounds "
             "and its own docstring disclaims 'improves with use'. Keeping it as a "
             "theorem would be exactly the theorem inflation the reviewer warned "
             "about. The mathematical content survives as CT7(a)+(b) with correct "
             "hypotheses and a rate bound."),
    Proposition(
        pid="P11", theorem=None,
        title="Increasing the basis count M lifts content addressing",
        hypothesis="Vector signal, fixed linear read-out, M basis functions.",
        claim="FALSIFIED. The history summary A is a fixed-size linear map whose "
              "rank is at most d regardless of M; its kernel persists.",
        proof="rank(L) <= d independent of M (P04). Increasing M only changes the "
              "coefficients inside A, not its codomain dimension.",
        verdict="FALSIFIED", module="core/arch/sgc.py (grid M)", check="f11",
        note="M=256 verified: rank 8, kernel 16 for n=3,d=8."),
    Proposition(
        pid="P12", theorem=None,
        title="rho=1 is usable for an unbounded stream",
        hypothesis="rho=1, bounded nonzero input.",
        claim="FALSIFIED. Amplitude grows linearly in L; no BIBO stability; any "
              "finite-precision register overflows eventually.",
        proof="Partial sums equal L*u exactly (P06).",
        verdict="FALSIFIED", module="core/arch/sgc.py", check="f12",
        note="rho=1 was measured to give perfect finite-L fp32 accuracy; that "
             "finite-window result does not extend to unbounded streams."),
    Proposition(
        pid="P13", theorem=None,
        title="An arbitrary nonlinear read-out rescues a vector-signal summary",
        hypothesis="Fixed linear history summary A, vector signal.",
        claim="FALSIFIED. Two distinct value assignments can produce identical A "
              "and identical q, so any f(A,q) gives the same output while the "
              "required answers differ.",
        proof="Kernel vector of L gives an explicit collision (P04, Thm5).",
        verdict="FALSIFIED", module="core/arch/sgc.py (vector-signal path)", check="f13",
        note="Verified: tanh read-out outputs identical to 1e-12 on the collision pair."),
    Proposition(
        pid="P14", theorem=None,
        title="RSI-style self-improvement as a theorem-level mechanism",
        hypothesis="Not available. RSI occurs in this repository only as an "
                   "external survey citation (docs/TUAN_1B_TRAINING_PLAN.md:172, "
                   "'L2 policy autonomy'); no RSI operator is implemented.",
        claim="CONJECTURE (retained). Cannot be upgraded without a formal "
              "definition of the self-modification operator. Upgrade path: once "
              "defined, T13/T14 impose a resolvable-radius lower bound on any "
              "weight-update loop carried in a finite-precision state, and T16 "
              "gives the gate condition under which the loop is non-regressing.",
        proof="n/a",
        verdict="CONJECTURE", module="(not implemented)", check="(none)",
        note="Honest boundary: no theorem can be stated about an unimplemented "
             "mechanism. We do not invent a definition for RSI."),
    Proposition(
        pid="P15", theorem=None,
        title="Scaling: deep multi-hop, V>>64 codes, >=10M-token LM parity",
        hypothesis="Unstated; requires experiments that have not been run.",
        claim="CONJECTURE (retained). T15 supplies a necessary condition "
              "(higher tensor order) but no sufficient construction. T9 covers "
              "V=2^k exactly; non-power-of-two V needs a construction and "
              "measurement. Language-modeling parity is an empirical claim.",
        proof="n/a",
        verdict="CONJECTURE", module="core/arch/ + training/", check="(none)",
        note="No theorem can establish empirical SOTA or general-language parity. "
             "This is the single most important honesty boundary of this work."),
]


THEOREMS = {p.theorem: p for p in PROPOSITIONS
           if p.theorem and p.verdict == "THEOREM"}
BY_MODULE = {}
for _p in PROPOSITIONS:
    BY_MODULE.setdefault(_p.module, []).append(_p.pid)


def summary() -> dict:
    return {
        "total": len(PROPOSITIONS),
        "theorems": sum(p.verdict == "THEOREM" for p in PROPOSITIONS),
        "falsified": sum(p.verdict == "FALSIFIED" for p in PROPOSITIONS),
        "conjectures": sum(p.verdict == "CONJECTURE" for p in PROPOSITIONS),
        "withdrawn": sum(p.verdict == "WITHDRAWN" for p in PROPOSITIONS),
        "theorem_ids": sorted(THEOREMS),
        "note": "historical ids; consolidated into CT1..CT7 by smm_sgc.py",
    }


__all__ = ["Proposition", "PROPOSITIONS", "THEOREMS", "BY_MODULE", "summary"]
