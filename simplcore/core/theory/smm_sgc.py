"""
SMM x SGC : a long-context self-evolution path and its hard theorems.

This module is the THEORY of the coupling between

  * SGC  : constant-size, differentiable, LOSSY spectral state  s_L = F(x_{1:L})
  * SMM  : exact episodic copies read through a HARD top-k selection
           (simplcore/core/memory/memory.py : top_aggregate / self_match_scores)
           plus a no-regression consolidation loop
           (simplcore/core/memory/dream/ : regression_gate, ReflectionLoop.VERIFY)

Source facts that the hypotheses below are built on (verified against the
upstream source, NOT invented):

  F1  memory.py:73  top_scores, top_idx = scores.topk(k)   -- hard index choice
  F2  memory.py:74  weights = softmax(top_scores / tau)    -- continuous weights
  F3  memory.py:48  scores = normalize(q) @ normalize(k).T -- content-dependent
  F4  smm.py:288    @torch.no_grad() on answer()            -- online path has NO graph
  F5  hooks.py:431  default_verifier returns CONSTANT 0.0
  F6  loop.py:335   if verifier is default -> REFUSE weight writeback
  F7  evaluation.py:146 regression_gate uses PAIRED BOOTSTRAP CI LOWER BOUNDS,
      multiple metrics, "any metric fails -> reject"
  F8  evaluation.py gate docstring: proves NO-REGRESSION (necessary condition),
      explicitly NOT "gets better with use"

NO TRAINING. NO GRADIENT DESCENT. NO MODEL WEIGHTS. NO EXPERIMENTAL DATA.
Verified by verify_smm_sgc.py (pure numpy, fixed seed).

Theorems defined here are numbered CT1..CT7 for the CORE set; see statements.py
for the consolidation table.  The two genuinely new, non-trivial ones are CT6
(Information-Gradient Duality) and CT7 (zero-order consolidation guarantees).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Formal objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledArchitecture:
    """The SMM x SGC object the theorems quantify over.

    d      : value / state channel dimension
    D      : dimension of the constant-size online spectral state (independent of L)
    C      : episodic bank capacity (constant, upstream default 4096)
    k      : top_k, number of exactly retrievable slots per read (upstream default 4)
    w      : max insights committed per consolidation round (upstream hard budget)
    b      : bits per insight in the finite-precision register
    L      : context length (UNBOUNDED in the theorems)
    tau    : consolidation tolerance; gate accepts iff CI lower bound >= -|tau|
    alpha  : significance level of the one-sided paired bootstrap CI
    m      : number of metrics in the verifier
    """
    d: int = 2048
    D: int = 2048
    C: int = 4096
    k: int = 4
    w: int = 8
    b: int = 32
    L: int = 1_000_000
    tau: float = 0.0
    alpha: float = 0.05
    m: int = 3

    def nonempty_instance_note(self) -> str:
        return ("Upstream defaults C=4096, k=4, tau=0.0, ci_level=0.95 (alpha=.05); "
                "w and b are the hard insight/bit budgets of the consolidation loop. "
                "L=1e6 > C, so the pigeonhole premise of CT6(i) is satisfied. "
                "A non-default held-out verifier must be supplied for CT7(a) to be "
                "active (F6); with the default constant verifier the gate refuses "
                "writeback and CT7(a) is vacuously satisfied, not contradicted.")


# ---------------------------------------------------------------------------
# CT6 : Information-Gradient Duality   (the hard, non-trivial theorem)
# ---------------------------------------------------------------------------
CT6 = {
    "id": "CT6",
    "name": "Information-Gradient Duality for constant-state coupled architectures",
    "hypothesis": (
        "An architecture has (a) an online state s_L = F(x_{1:L}) in R^D with D "
        "independent of L and F continuous (differentiable a.e.); (b) an external "
        "episodic bank of C EXACT copies, read by a hard top-k choice "
        "S_k(q) = argtop_k_j <phi(q), k_j> followed by continuous weights "
        "alpha = softmax(top_scores / temp); (c) no N x N score matrix over the "
        "sequence and no resident KV cache; (d) the task requires exact retrieval "
        "of an ARBITRARY one of L history items, with |X| > 1 and L unbounded."
    ),
    "claim": (
        "No single channel of this architecture is simultaneously (I) differentiable, "
        "(II) covers all L history items, and (III) losslessly retrieves any one of "
        "them. Exactly one of the three must fail, and the three possible repairs "
        "are mutually exclusive."
    ),
    "proof": """
Let A = X^L be the set of histories, |A| = |X|^L.

PART (i) -- differentiable + covering => not lossless.
The online state is a map F : A -> R^D with D fixed. Exact retrieval of an
arbitrary item requires F to be injective on A (two histories differing only in
the requested item must map to different states, otherwise no read-out can
separate them). But R^D realized in B bytes has at most 2^(8B) distinct
configurations. For L > 8B / log2|X|, |A| > 2^(8B) and injectivity is
impossible by the pigeonhole principle. Hence (I)+(II) forces NOT (III).
This is Thm 6(a) specialized; we use it as one horn, not as the whole theorem.

PART (ii) -- hard top-k is exact but NOT covering and NOT differentiable in the
selection.
Write the retrieved vector

    m(q) = sum_{j in S_k(q)} alpha_j(q) v_j .

On the open set O = {scores with all pairwise ties broken}, S_k(q) is locally
CONSTANT: for a small enough perturbation of any parameter theta, the ordering
of the scores is unchanged, hence top_idx is unchanged. Therefore

    d 1[j in S_k(q)] / d theta  = 0   for a.e. theta in O            (1)

and the Jacobian decomposes as

    dm/dtheta = sum_{j in S_k} (dalpha_j/dtheta) v_j
              + sum_{j in S_k} alpha_j (dv_j/dtheta)
              + sum_{j in S_k} alpha_j v_j * d1[j in S_k]/dtheta .
                                          \--------- = 0 a.e. by (1)

Consequently supp(dm/dtheta) is contained in S_k(q), a set of size k = O(1).
All C - k unselected episodic items contribute EXACTLY ZERO to any
gradient-based update, no matter how long or how precisely they were stored.
So hard retrieval is exact on its k slots (III holds there) but fails (I) for
selection and fails (II) because |S_k| = k < L.

PART (iii) -- THE NON-TRIVIAL STEP: softening the selection to restore
differentiability re-triggers the impossibility theorem.
The natural repair for (ii) is to replace hard top-k by a differentiable soft
aggregate over ALL C copies,

    m_soft(q) = sum_{j=1..C} alpha_j(q) v_j ,   alpha(q) continuous.

Now m_soft is differentiable and covers all C copies. But for a FIXED query q,
the map V = (v_1..v_C) in R^{C d} -> m_soft in R^d is LINEAR with rank at most d.
By rank-nullity, dim ker >= C d - d >= (C-1) d > 0 for C >= 2. Choose a nonzero
dV in this kernel supported on the coordinate of the target item, so that
m_soft is unchanged while the required answer changes (values are independently
assignable and at least two distinguishable targets exist; cf. Thm 5 / T10).
Two distinct value assignments then produce identical m_soft and identical q,
hence identical output for ANY read-out f(m_soft, q), linear or nonlinear,
while the correct answers differ. Contradiction. Thus (I)+(II) again forces
NOT (III): exact content addressing is lost.

CONCLUSION. The three horns are exclusive: (i) kills losslessness by counting,
(ii) kills coverage and differentiability of selection, (iii) kills exactness
by the kernel argument. Softening is not a repair -- it maps the architecture
straight back into Thm 5. Therefore no constant-state channel has all three
properties at once, and no repair within (a)-(c) can provide them. QED
""",
    "boundary": (
        "The theorem does NOT say the architecture cannot retrieve at all: hard "
        "top-k retrieves k items exactly (this is the L2 pointer sidecar). It says "
        "no SINGLE channel is differentiable + covering + lossless. It does not "
        "forbid a NON-differentiable zero-order path (that is CT7), nor does it "
        "assert anything about empirical LM quality."
    ),
    "source_facts": ["F1", "F2", "F3", "F4"],
}


# ---------------------------------------------------------------------------
# CT7 : zero-order consolidation -- statistical guarantee + rate bound
# ---------------------------------------------------------------------------
CT7 = {
    "id": "CT7",
    "name": "Zero-order self-evolution: gate validity and information rate",
    "hypothesis": (
        "CT6 holds. The consolidation loop writes to weights only through a "
        "no-regression gate that compares paired-bootstrap CI LOWER BOUNDS "
        "l_alpha (level 1-alpha, one-sided) against -|tau|, for m metrics, and "
        "rejects if ANY metric fails. An independent held-out verifier is supplied "
        "(otherwise the loop refuses writeback, F6). Per round at most w insights "
        "are committed, each representable in b bits; each read touches at most "
        "k = top_k items."
    ),
    "claim": (
        "(a) STATISTICAL GUARANTEE: under asymptotic coverage of the paired "
        "bootstrap CI, P(false commit | true regression) <= alpha for one metric, "
        "and <= m*alpha for m metrics (or <= alpha with Bonferroni allocation "
        "alpha/m). (b) RATE BOUND: the information about the L-token history that "
        "can enter the weights over T rounds is <= T*w*b bits, INDEPENDENT of L "
        "and of the bank capacity C; the instantaneous bandwidth is <= k*b bits "
        "per round. (c) The verifier score sequence is non-decreasing ONLY in the "
        "sense of 'no accepted regression'; improvement is NOT proved."
    ),
    "proof": """
WHY THE PATH MUST BE ZERO-ORDER (corollary of CT6, not a design choice).
By F4 the online answer() path is @torch.no_grad(): no graph exists there.
By CT6(ii) the selection Jacobian vanishes a.e., so no gradient can carry an
unselected long-context item into an update. By CT6(iii) making the retrieval
differentiable destroys exactness. Hence the ONLY channel that can write
long-context experience into the weights without breaking the architecture's
purity is one that uses NO retrieval gradient: a scalar/CI acceptance test on
the verifier output. The upstream gate (F7) is exactly such a zero-order test.

PART (a) -- false-commit probability.
Let Delta_true be the true paired delta of one metric, and define regression as
Delta_true < -|tau|. The gate accepts that metric iff l_alpha >= -|tau|.
A one-sided CI lower bound with asymptotic level 1-alpha satisfies the coverage
property

    P( l_alpha <= Delta_true ) >= 1 - alpha ,   hence
    P( l_alpha >  Delta_true ) <=  alpha .                        (2)

A false commit on this metric requires l_alpha >= -|tau| > Delta_true, i.e.
l_alpha > Delta_true, which is exactly the coverage-failure event in (2).
Therefore P(false commit on metric i) <= alpha. With m metrics the gate accepts
only if ALL pass, so a false commit of the ROUND requires at least one metric to
fail coverage; by the union bound

    P(false commit of round) <= sum_i P(coverage failure_i) <= m*alpha .

With Bonferroni allocation alpha_i = alpha/m the bound is <= alpha.
This is an ANALYTIC consequence of the coverage property; no Monte-Carlo run
is part of the proof. Conditions are explicit: paired i.i.d. samples, bootstrap
consistency, and a NON-DEFAULT verifier (F6) -- with the default constant
verifier the loop refuses writeback and the gate is inactive, which is a
vacuous but not contradictory instance.

PART (b) -- information rate.
Let H be the L-token history, R_t the set of items read at round t, I_t the
insights committed, and Delta theta_t the weight increment. The consolidation
operator U_t maps (I_t, verifier output) -> Delta theta_t. By the data
processing inequality,

    I(H ; Delta theta_t) <= I(H ; I_t, verifier_t) <= H(I_t, verifier_t) .

I_t contains at most w insights of b bits each, and the verifier output is a
finite sample plus CI scalars, itself bounded by O(w*b). Hence

    I(H ; Delta theta_t) <= c * w * b  bits (c a small constant),
    I(H ; Delta theta_1..T) <= T * c * w * b  bits .

Crucially L and C appear NOWHERE: by CT6(ii) only the k selected items can
influence a read, and the unselected L-k items (including everything that makes
the context 'long') contribute zero to the update. The instantaneous bandwidth
is therefore <= k*b bits per round, with k = top_k = 4 in the upstream default.
Long context does NOT buy a faster path into the weights; it buys retrieval at
inference time, not learning rate. QED

PART (c) -- what is NOT proved.
The gate proves NO-REGRESSION (a necessary condition for writeback), exactly as
the upstream docstring F8 states. Non-decreasing accepted scores do NOT imply
monotone real capability: the verifier may be misspecified, held-out overlap
may exist, and generalization improvement is tracked SEPARATELY by
ReflectionConfig.heldout_eval. We therefore withdraw the earlier T16 claim
('score sequence monotone => self-improvement monotone') as an over-reading.
""",
    "boundary": (
        "(a) is a CONDITIONAL theorem (bootstrap coverage + i.i.d. + independent "
        "verifier). (b) is an upper bound, not a claim that the loop reaches it. "
        "Neither (a) nor (b) asserts that self-evolution improves any real task. "
        "Numerical checks in verify_smm_sgc.py are consistency illustrations, "
        "NOT proofs."
    ),
    "source_facts": ["F4", "F5", "F6", "F7", "F8"],
}


# ---------------------------------------------------------------------------
# CT5 support: deterministic orthogonal code for ARBITRARY vocabulary size
# ---------------------------------------------------------------------------
DETERMINISTIC_CODE = {
    "id": "CT5-L2",
    "name": "Deterministic zero-crosstalk code for arbitrary V (upgrades P15 part 1)",
    "hypothesis": "V >= 1 arbitrary (not necessarily a power of two).",
    "claim": (
        "Let R = 2^{ceil(log2 V)} and Code_V = the first V rows of the Sylvester "
        "Hadamard matrix H_R. Then Code_V Code_V^T = R * I_V EXACTLY (zero "
        "crosstalk, delta response), and R < 2V. Hence a deterministic orthogonal "
        "code exists for EVERY vocabulary size with code dimension less than 2V."
    ),
    "proof": """
By the Sylvester recursion H_{2m} = [[H_m,H_m],[H_m,-H_m]] and induction,
H_R H_R^T = R I_R. Taking the leading V x V principal submatrix of that
identity gives Code_V Code_V^T = R I_V: off-diagonal entries are exactly 0 and
the diagonal is R. The bound R < 2V follows from
R = 2^{ceil(log2 V)} < 2^{log2 V + 1} = 2V for V > 1 (V = 1 gives R = 1).
The query/key inner product is therefore <Code[q], Code[k]> = R * delta_{q,k},
so the TSC contraction z_n = sum_tau (q.k_tau) K_n(.) v_tau retains EXACT zero
interference for arbitrary V. No learned projection, no randomness, no
assumption on V. QED
""",
    "boundary": (
        "This proves existence and gives an explicit construction. It does NOT "
        "prove that a V = 151936 code is cheap in a deployed kernel (R = 262144, "
        "1.725x V), nor that it scales to downstream LM training; those remain "
        "empirical questions."
    ),
    "instances": [
        {"V": 64, "R": 64}, {"V": 65, "R": 128}, {"V": 1000, "R": 1024},
        {"V": 50000, "R": 65536}, {"V": 151936, "R": 262144},
    ],
}


# ---------------------------------------------------------------------------
# Consolidated CORE set: 7 theorems, everything else demoted
# ---------------------------------------------------------------------------
CORE_THEOREMS: List[Dict] = [
    dict(id="CT1", name="Exact real-field FFT/IIR duality",
         from_="Thm1", status="kept",
         note="Z-transform identity; unconditional fp32 bitwise claim withdrawn."),
    dict(id="CT2", name="Strict causality + O(1) resident state",
         from_="Thm2+Thm3 merged", status="kept",
         note="Structural; byte counts are observations, L-independence is the theorem."),
    dict(id="CT3", name="Toeplitz iff time-invariant coefficients (arbitrary order P)",
         from_="Thm4 + T11", status="strengthened",
         note="Induction on delay tau; c_t != 0 guaranteed by theta clamp."),
    dict(id="CT4", name="Content addressing impossible for any read-out (quantitative)",
         from_="Thm5 + T10", status="strengthened",
         note="dim ker L >= nd-d; n=1 exception makes the >=2-target premise necessary."),
    dict(id="CT5", name="TSC constructive counterexample: addressing AND duality coexist",
         from_="Sec4 + T7 + T8 + T9 + CT5-L2", status="kept+extended",
         note="THE structural result the reviewer singled out. Now covers arbitrary V."),
    dict(id="CT6", name="Information-Gradient Duality", from_="NEW", status="NEW",
         note="The hard theorem. Softening top-k re-triggers CT4; no single channel "
              "is differentiable + covering + lossless."),
    dict(id="CT7", name="Zero-order consolidation: gate validity + rate bound",
         from_="NEW (replaces T16)", status="NEW",
         note="False-commit <= alpha; <= T*w*b bits, independent of L and C."),
]

DEMOTED_TO_LEMMA: List[Dict] = [
    dict(id="L1", from_="T7", reason="channel-wise Toeplitz of an LTI recursion is a "
         "standard fact; used inside CT5's proof, not a standalone theorem."),
    dict(id="L2", from_="T9", reason="Sylvester orthogonality is standard; its "
         "non-trivial arbitrary-V extension is CT5-L2."),
    dict(id="L3", from_="T12", reason="geometric-series BIBO is a textbook result."),
    dict(id="L4", from_="T14", reason="peak-scale monotonicity gives a one-line "
         "partial order; useful for deploy but too ordinary to occupy a theorem slot."),
    dict(id="L5", from_="T8", reason="contraction closed form is a proof step of CT5, "
         "not an independent theorem."),
    dict(id="L6", from_="T13", reason="quantization resolvable radius is a CONDITIONAL "
         "theorem; kept in appendix, not core."),
    dict(id="L7", from_="T15", reason="one-hop/two-hop degree separation gives a "
         "NECESSARY condition; kept in appendix as a corollary of CT6."),
]

WITHDRAWN: List[Dict] = [
    dict(id="T16", reason="REPLACED BY CT7. T16 was a trivial induction on a scalar "
         "score and did not match the source: regression_gate uses multi-metric "
         "paired-bootstrap CI lower bounds (F7) and explicitly disclaims "
         "'improves with use' (F8). Keeping it would be theorem inflation."),
]

FALSIFIED: List[Dict] = [
    dict(id="P11", claim="increasing basis count M lifts content addressing",
         reason="rank(L) <= d independent of M (CT4/T10)."),
    dict(id="P12", claim="rho=1 usable for unbounded stream",
         reason="s_L = L*U, no BIBO stability; finite-window fp32 success does not extend."),
    dict(id="P13", claim="arbitrary nonlinear read-out rescues vector-signal summary",
         reason="explicit kernel collision gives identical f(A,q) for different answers."),
]

CONJECTURE_DISPOSITION: List[Dict] = [
    dict(id="P14", name="RSI-style self-improvement", verdict="PARTIALLY RESOLVED",
         resolved="The self-evolution OPERATOR now has a precise definition: the "
                  "zero-order consolidation loop U with no-regression gate. CT6 "
                  "proves it must be zero-order; CT7(a) gives its statistical "
                  "validity; CT7(b) gives its information-rate ceiling. These are "
                  "genuine theorems about RSI's substrate.",
         remains="That this operator reaches superhuman / SOTA capability is an "
                 "EMPIRICAL claim. No theorem can establish it. Retained as a "
                 "conjecture with an explicit falsification experiment."),
    dict(id="P15a", name="V >> 64 deterministic code", verdict="PROVED (CT5-L2)",
         resolved="Explicit Sylvester-prefix construction for arbitrary V, R < 2V."),
    dict(id="P15b", name="deep multi-hop retrieval", verdict="PARTIALLY RESOLVED",
         resolved="CT6/L7 gives a NECESSARY condition (raise tensor order).",
         remains="No sufficient construction proved; stacking/recurrence untested."),
    dict(id="P15c", name=">=10M-token LM parity / superiority", verdict="CONJECTURE",
         remains="Empirical claim; requires the equal-compute experiment that has "
                 "NOT been run. Explicitly NOT a theorem."),
]


def summary() -> Dict:
    v = [c["verdict"] for c in CONJECTURE_DISPOSITION]
    return {
        "core_theorems": len(CORE_THEOREMS),
        "demoted_to_lemma": len(DEMOTED_TO_LEMMA),
        "withdrawn": len(WITHDRAWN),
        "falsified": len(FALSIFIED),
        "new_hard_theorems": ["CT6", "CT7", "CT5-L2"],
        "conjecture_items": len(CONJECTURE_DISPOSITION),
        "conjectures_fully_proved": v.count("PROVED (CT5-L2)"),
        "conjectures_partially_resolved": v.count("PARTIALLY RESOLVED"),
        "conjectures_still_open": v.count("CONJECTURE")
                                 + v.count("PARTIALLY RESOLVED"),
        "note": "P15a fully proved; P14/P15b partially resolved with an explicit "
                "remaining empirical part; P15c is an empirical claim and cannot "
                "be a theorem.",
    }


__all__ = [
    "CoupledArchitecture", "CT6", "CT7", "DETERMINISTIC_CODE",
    "CORE_THEOREMS", "DEMOTED_TO_LEMMA", "WITHDRAWN", "FALSIFIED",
    "CONJECTURE_DISPOSITION", "summary",
]
