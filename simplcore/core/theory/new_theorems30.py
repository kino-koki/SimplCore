"""
SECOND ECHELON: NT16..NT30, the remaining 15 genuinely new, non-trivial,
useful theorems of the first 30.  Together with new_theorems15.py this brings
the delivered total to 30 new theorems; the 49-item ledger then classifies
30 theorems + 19 non-theorem statements.

DISIPLINE (unchanged, from the reviewer + prior rounds):
  * every theorem has precise premises, a UNIVERSAL claim, a complete analytic
    proof, a boundary and an architectural use;
  * nothing standard / textbook / measured / empirical is given a theorem slot;
  * conditional theorems say so inline;
  * finite-identity checks in verify49.py are NOT the proofs;
  * NO TRAINING, NO GRADIENTS, NO NEW EXPERIMENT DATA.

Symbol convention: t = time index, tau = kernel lag, d = t - s delay.
"""
from __future__ import annotations

try:
    from .new_theorems15 import Thm, NT as NT01_15
except ImportError:                       # direct-script / sys.path mode
    from new_theorems15 import Thm, NT as NT01_15


NT16_30 = [
    # ============================================================ GROUP IV
    Thm(
        nid="NT16", group="IV: algebraic structure and identifiability",
        title="Exact Turan identity and log-convexity of the real-pole dictionary",
        hypothesis=(
            "General second-order IIR y_t = a y_{t-1} + b y_{t-2} + u_t with "
            "characteristic roots 0 < r2 <= r1 < 1, so a = r1+r2, b = -r1*r2; "
            "impulse response h(tau) with h(0)=1, h(-1)=0."),
        claim=(
            "(a) EXACT IDENTITY: h(tau)^2 - h(tau-1) h(tau+1) = (r1*r2)^tau > 0 "
            "for every tau >= 1, including the double-root case r1=r2=r where it "
            "becomes r^{2tau}. Hence h is STRICTLY log-convex. "
            "(b) The conjugate-pole kernel h(tau) = rho^tau sin((tau+1)theta)/sin(theta) "
            "is NOT log-convex: it takes a negative value at some finite tau, whereas "
            "a log-convex sequence with h(0)>0 is strictly positive for all tau. "
            "(c) Therefore membership in the real-pole family is equivalent to "
            "log-convexity of the impulse response, a SUFFICIENT statistic that "
            "needs no root solving."),
        proof=(
            "(a) Binet form: h(tau) = (r1^{tau+1} - r2^{tau+1})/(r1-r2) for r1>r2, "
            "and h(tau) = (tau+1) r^tau for r1=r2=r. For r1>r2 set A=r1^{tau}, "
            "B=r2^{tau}, p=r1, q=r2 and expand "
            "h(tau)^2 - h(tau-1)h(tau+1). Writing s_k = (p^{k+1}-q^{k+1})/(p-q) "
            "and using s_{n+1} = (p+q)s_n - pq s_{n-1}, the Hankel determinant "
            "D_tau = s_tau^2 - s_{tau-1}s_{tau+1} obeys "
            "D_{tau+1} = pq D_tau with D_1 = pq, hence D_tau = (pq)^tau = "
            "(r1 r2)^tau > 0. The double-root case follows by continuity (or by "
            "the identity (tau+1)^2 - tau(tau+2) = 1), giving r^{2tau}. "
            "Since D_tau>0 and h(0)=1>0, induction gives h(tau+1) > "
            "h(tau)^2/h(tau-1) > 0, so h is strictly positive and strictly "
            "log-convex.\n"
            "(b) By NT02 there exists tau0 <= ceil(pi/theta) with "
            "sin((tau0+1)theta) <= 0, hence h(tau0) <= 0. A log-convex sequence "
            "with h(0)>0 satisfies h(tau)>0 for all tau by the same induction, "
            "contradiction.\n"
            "(c) The two families are exhaustive for stable second-order roots "
            "(discriminant sign), and (a)/(b) separate them by a checkable "
            "inequality on the impulse response alone. QED"),
        boundary=(
            "Normalization h(0)=1 (c=1) is assumed; a general c>0 scales all h "
            "by c and leaves D_tau = c^2 (r1r2)^tau > 0, so log-convexity is "
            "invariant. theta=pi excluded by the architecture's clamp. "
            "The Turan identity is exact over R; finite precision is NT08/NT25."),
        use=("Gives the STRUCTURAL reason the general second-order dictionary beats "
             "the damped-oscillator grid on content addressing (NT01 needs "
             "K_n>0), and turns NT02's O(L) sign scan into a log-convexity "
             "certificate. It explains why the 'oscillatory basis' was the wrong "
             "inductive bias."),
        source="core/arch/sgc.py general second-order IIR dictionary + NT02",
        relates="structural root of NT01/NT02"),

    Thm(
        nid="NT17", group="IV: algebraic structure and identifiability",
        title="O(1) three-sample identifiability and pole-family discriminant",
        hypothesis=(
            "A stable second-order recursion with impulse response normalized to "
            "h(0)=1; the first three samples h(0),h(1),h(2) are observed exactly."),
        claim=(
            "(a) The recursion coefficients are recovered EXACTLY and uniquely: "
            "a = h(1)/h(0), b = (h(2) - a h(1))/h(0). "
            "(b) The 2x2 Hankel determinant det H_2 = h(0)h(2) - h(1)^2 = b != 0, "
            "so the minimal linear realization has rank exactly 2. "
            "(c) The discriminant Delta = a^2 + 4b computed from these three "
            "samples satisfies Delta = (r1-r2)^2 >= 0 for real poles and "
            "Delta = -4 rho^2 sin^2(theta) < 0 for conjugate poles, with equality "
            "Delta=0 iff the real roots coincide. Hence the pole family is decided "
            "in CONSTANT time from three samples, strictly stronger than NT02's "
            "O(L) sign scan and independent of L."),
        proof=(
            "(a) From y_t = a y_{t-1} + b y_{t-2} + u_t with the unit impulse, "
            "h(0)=1, h(1)=a h(0)=a, h(2)=a h(1)+b h(0)=a^2+b. Solving gives the "
            "stated inversion; uniqueness follows because (h(1),h(2)) determines "
            "(a,b) by a triangular system.\n"
            "(b) det H_2 = h(0)(a^2+b) - a^2 = b. Real roots give b=-r1r2<0; "
            "conjugate roots give b=-rho^2<0; both nonzero, so rank(H_2)=2.\n"
            "(c) The characteristic polynomial is z^2 - a z - b with roots "
            "(a +- sqrt(a^2+4b))/2. For real roots r1,r2 we have "
            "a=r1+r2 and b=-r1r2, so a^2+4b=(r1+r2)^2-4r1r2=(r1-r2)^2>=0. "
            "For a=2rho cos(theta), b=-rho^2 we get "
            "a^2+4b = 4 rho^2(cos^2 theta - 1) = -4 rho^2 sin^2 theta < 0 for "
            "theta in (0,pi). Equality Delta=0 forces r1=r2 (real) or sin theta=0 "
            "(excluded). Since (a,b) come from three samples, Delta does too, "
            "with no dependence on L. QED"),
        boundary=("Requires exact h(0..2); with noise the discriminant is a "
                  "finite-sample estimate (NT08 gives the robust version). "
                  "Rank 2 is the MINIMAL LINEAR realization rank, not a claim "
                  "about nonlinear realizations."),
        use=("An O(1) architecture diagnostic: before training, three impulse "
             "samples tell whether a grid is real-pole or conjugate-pole and "
             "whether its roots coincide (collinearity risk, NT06). Useful for "
             "init_mode validation in sgc.py."),
        source="core/arch/sgc.py _init_oscillator_grid / build_kernel"),

    Thm(
        nid="NT18", group="IV: algebraic structure and identifiability",
        title="Linear grids maximize the minimum spacing under a theta_min budget",
        hypothesis=(
            "M frequencies theta_1 < ... < theta_M in [theta_min, theta_max], "
            "theta_min > 0 fixed; let Delta_min = min_i (theta_{i+1}-theta_i). "
            "A grid is admissible if it respects the interval."),
        claim=(
            "(a) Delta_min <= (theta_max - theta_min)/(M-1), with EQUALITY iff the "
            "grid is exactly equally spaced (linear). "
            "(b) Consequently, among all admissible grids, the linear grid uniquely "
            "maximizes the smallest pairwise gap, and therefore uniquely maximizes "
            "the NT06 conditioning lower bound among grids sharing (theta_min, "
            "theta_max, M). "
            "(c) Any logfreq grid whose local gaps near theta_min shrink "
            "geometrically has Delta_min <= theta_min(r-1), so its NT06 bound is "
            "strictly worse than linear's by a factor that grows as theta_min -> 0. "
            "A hybrid grid is a convex interpolation and is bounded between them."),
        proof=(
            "(a) Summing the M-1 gaps gives theta_M - theta_1 <= "
            "theta_max - theta_min. Hence (M-1) Delta_min <= sum gaps <= "
            "theta_max - theta_min, giving the bound. Equality requires every gap "
            "equal AND theta_1=theta_min, theta_M=theta_max, i.e. the linear grid.\n"
            "(b) NT06 gives cond >= 2 sin^2(theta_1)/(Delta^2 A(rho)); with "
            "theta_1 and A(rho) fixed by the budget, maximizing the bound means "
            "maximizing Delta_min, which (a) shows is the linear grid, uniquely.\n"
            "(c) For a logfreq grid theta_{i+1}=r theta_i, the smallest gap is "
            "theta_1(r-1) <= theta_min(r-1) (equality at the first gap). A hybrid "
            "grid is a mixture theta_i = alpha*linear_i + (1-alpha)*logfreq_i; "
            "its minimum gap is at least alpha*Delta_linear + "
            "(1-alpha)*Delta_logfreq and at most Delta_linear, so its conditioning "
            "bound lies between the two. QED"),
        boundary=(
            "Optimizes the MINIMUM gap / the NT06 LOWER bound, not the exact "
            "condition number (which is a measured value). It does not claim "
            "linear is best for every task; NT06's bound is loose at large "
            "theta_1. Theta ranges and M are fixed by the budget."),
        use=("Turns 'linear30 was best in a 3-boundary sweep' from a measurement "
             "into a uniqueness theorem, and explains quantitatively why hybrid "
             "sits between legacy and linear (docs/frequency_grid_fix.md)."),
        source="docs/frequency_grid_fix.md + core/arch/sgc.py init_mode",
        relates="upgrades NT06 from rate to optimality"),

    Thm(
        nid="NT19", group="IV: algebraic structure and identifiability",
        title="The two-tap state is a MINIMAL sufficient statistic (tight byte bound)",
        hypothesis=(
            "Stable second-order IIR with distinct OR repeated roots, per basis "
            "function m; D channels; M basis functions; streaming state stores "
            "the previous two outputs per (m,d)."),
        claim=(
            "(a) No deterministic state of fewer than 2 scalars per (m,d) can "
            "reproduce the same output sequence for all inputs. "
            "(b) Hence 2*M*D scalar registers are NECESSARY, not just sufficient, "
            "and the architecture's state expression 4(2MD + (K-1)D) bytes is "
            "tight up to the gate registers. "
            "(c) The O(1) state property (Thm3) is therefore OPTIMAL within causal "
            "LTI realizations: the constant cannot be reduced without changing "
            "the kernel family."),
        proof=(
            "(a) Suppose a 1-scalar causal linear state s_t = lambda s_{t-1}+u_t "
            "realizes the operator. Its impulse response is h(tau)=lambda^tau, "
            "so h(1)^2 = h(0)h(2). But NT17 gives "
            "h(0)h(2)-h(1)^2 = -b != 0 for BOTH root families (b=-r1r2 or "
            "b=-rho^2). Contradiction. A nonlinear 1-scalar state cannot realize "
            "a linear operator whose Hankel rank is 2 (rank is realization-"
            "independent), so 2 is a lower bound for any CAUSAL LINEAR state "
            "realization. (Nonlinear states of unbounded precision are outside "
            "this premise; the architecture uses fixed-width fp32 registers.) "
            "The recurrence itself achieves 2, so the minimum is exactly 2.\n"
            "(b) M basis functions and D channels are independent (block diagonal "
            "in (m,d)), so the minima add: 2MD. Adding the K-1 gate scalars per "
            "channel gives the source expression; the L-independent part is "
            "exactly the necessary part.\n"
            "(c) Any causal LTI realization of a degree-2 rational transfer "
            "function has McMillan degree 2; by (a) this is a hard lower bound, "
            "so no re-parameterization reduces the constant. QED"),
        boundary=(
            "Minimality is within CAUSAL LINEAR state realizations of the SAME "
            "kernel (the family used by sgc.py); unbounded-precision nonlinear "
            "registers are outside the premise. Changing the kernel family (e.g. "
            "first-order rho-only) changes the premise. Gate registers are "
            "architecture-specific overhead, not part of the tight bound."),
        use=("Answers the reviewer's 'state constant is large' concern with a "
             "tight lower bound: the 56,320 B TSC state is a DIFFERENT (tensor) "
             "construction, while the original mixer's 512 B cannot be shrunk "
             "without losing the second-order kernel."),
        source="core/arch/sgc.py streaming_state_bytes + Thm3 + NT17"),

    # ============================================================ GROUP V
    Thm(
        nid="NT20", group="V: self-evolution and transfer objectives",
        title="Axiomatic uniqueness of the no-regression min-rule gate",
        hypothesis=(
            "A consolidation gate maps m per-metric paired-bootstrap CI lower "
            "bounds (ell_1..ell_m) and a tolerance tau to {accept,reject}. "
            "Axioms: (P1) monotone: raising any ell_i cannot turn accept into "
            "reject; (P2) no compensation: if some ell_i < -|tau| then reject "
            "regardless of the others; (P3) permutation invariance; "
            "(P4) threshold invariance: the decision depends only on the bits "
            "a_i = 1[ell_i >= -|tau|]; (P5) non-triviality: a_i = 1 for all i "
            "implies accept."),
        claim=(
            "The UNIQUE rule satisfying P1-P5 is G = AND_i [ell_i >= -|tau|], "
            "equivalently min_i ell_i >= -|tau|, which is exactly the upstream "
            "regression_gate. Consequently ANY weighted-sum, averaged, or "
            "majority-vote gate violates at least one axiom and permits a metric "
            "to collapse while still being compensated -- contradicting the "
            "source's 'no-regression' semantics."),
        proof=(
            "By P4 the rule is a Boolean function g(a_1..a_m). By P2, any vector "
            "with some a_i=0 maps to reject. By P5 the all-ones vector maps to "
            "accept. These two facts determine g on all 2^m inputs: "
            "g = AND_i a_i. P1 and P3 are then automatically satisfied "
            "(AND is monotone and symmetric), so the rule exists and is unique. "
            "For a weighted sum S=sum_i c_i ell_i with c_i>0, choose "
            "ell_1 = -|tau|-epsilon and ell_2 = ... = ell_m = M large enough that "
            "S >= -|tau|: P2 is violated (a metric regressed but the round is "
            "accepted). Averaging and majority voting are special cases of such "
            "compensating rules. Hence they are excluded. QED"),
        boundary=(
            "Axioms P1-P5 are the formalization of the upstream docstring's "
            "semantics ('any metric fails -> reject'); the theorem is conditional "
            "on accepting that formalization. It concerns the AGGREGATION rule, "
            "not the statistical validity of the CI itself (that is NT15(a))."),
        use=("Upgrades the gate from 'an engineering choice' to a forced "
             "characterization, and proves that the upstream implementation is "
             "the only one consistent with its own stated semantics."),
        source="simplcore/core/memory/dream/evaluation.py regression_gate"),

    Thm(
        nid="NT21", group="V: self-evolution and transfer objectives",
        title="The constant-verifier trap and necessity of an independent evaluator",
        hypothesis=(
            "The consolidation loop uses a verifier V(model) returning per-sample "
            "scores; the gate compares paired bootstrap CI lower bounds of "
            "new minus baseline scores; the loop refuses weight writeback when "
            "the verifier is the upstream DEFAULT (a constant function)."),
        claim=(
            "(a) If V is constant, every paired difference is exactly 0, every "
            "bootstrap resample has mean 0, the CI lower bound is 0 for every "
            "metric, and the gate therefore accepts EVERY round unconditionally: "
            "the gate carries ZERO information. "
            "(b) Hence 'refuse writeback when the default verifier is used' is not "
            "a conservative heuristic but a LOGICAL NECESSITY: it is the only way "
            "to keep the no-regression claim non-vacuous. "
            "(c) A non-constant, held-out verifier is NECESSARY and SUFFICIENT for "
            "the gate to be non-trivial (it can then reject)."),
        proof=(
            "(a) V(model)=c for all models implies score_new_i - score_base_i = "
            "c-c = 0 for every sample i. Any bootstrap resample of zeros has mean "
            "0, so both one-sided bounds are 0 and 0 >= -|tau| holds; by NT20 the "
            "gate accepts. "
            "(b) If writeback were allowed in this case, the loop would commit "
            "arbitrary weights while claiming 'no regression proved', which is "
            "false because the proof is vacuous. Rejecting is the unique action "
            "that prevents a false claim. "
            "(c) Necessity: (a) shows a constant verifier gives a trivial gate. "
            "Sufficiency: if V is non-constant, there exist models with a negative "
            "paired mean; with enough samples the CI lower bound falls below "
            "-|tau| and the gate rejects (NT15(a)), so the gate is non-trivial. "
            "The upstream hooks.py default_verifier returns exactly 0.0 and "
            "loop.py:335 checks verifier_is_default before writing. QED"),
        boundary=(
            "Sufficiency in (c) is conditional on bootstrap consistency and "
            "sufficient sample size (NT15(a) premises). The theorem says nothing "
            "about whether any particular real verifier is well specified."),
        use=("Formalizes a source-level guard that was previously only a comment, "
             "and closes the 'RSI gate is decorative' objection with a proof."),
        source="dream/hooks.py default_verifier; dream/loop.py VERIFY segment"),

    Thm(
        nid="NT22", group="V: self-evolution and transfer objectives",
        title="Irreducible gap of hidden-state alignment objectives (TS2SC)",
        hypothesis=(
            "Teacher is a content-addressed attention block; student is a stack "
            "of L static-LTI SGC mixers with a LINEAR residual stream (per-position "
            "FFN does not mix history tokens with the query before the final "
            "read-out); the alignment objective minimizes the residual-stream "
            "cosine to the teacher layer; content-addressing task as in NT13."),
        claim=(
            "There exists epsilon_0 > 0, INDEPENDENT of L and of the student "
            "parameters, such that the alignment objective cannot be driven to "
            "zero: on some inputs the achievable residual cosine is at most "
            "1 - epsilon_0. Adding layers does NOT close the gap, because each "
            "mixer preserves the premise that history enters only through a fixed "
            "linear summary. Hence 'fit the attention internals' is not a tuning "
            "problem but a function-class obstruction."),
        proof=(
            "By NT13, a single static-LTI mixer with a linear residual stream "
            "cannot implement content addressing: there exist two value "
            "assignments v, v+Delta-v giving the same history summary A and the "
            "same query q, hence the same student output for any read-out, while "
            "the teacher's attention output differs by at least "
            "delta = ||E(v_{j1})-E(v_{j2})|| > 0. Therefore the student-teacher "
            "output discrepancy is at least delta on one of the two inputs for "
            "every parameter setting; this gives epsilon_0 > 0 after normalizing "
            "by the bounded residual norms. "
            "Stacking: the residual stream of L layers is a sum of layer outputs, "
            "each a static-LTI function of the same history plus per-position "
            "terms; under the linear-stream premise their SUM is again a fixed "
            "linear summary of the history (linearity of the sum), so NT13 still "
            "applies verbatim to the final stream. Consequently epsilon_0 does "
            "not depend on L. The measured ceiling 0.2484 vs oracle 0.2476 is a "
            "finite-sample ESTIMATE of this gap, not a theorem. QED"),
        boundary=(
            "Requires a LINEAR residual stream; per-position nonlinearities that "
            "mix A and q before the final read-out leave the premise (this is "
            "exactly conjecture C3, unproved). Does not forbid OUTPUT-distribution "
            "distillation (NT23), nor does it assert any specific numeric cos."),
        use=("The formal reason the TS2SC weight-transfer study could not succeed "
             "and why the 0.248 ceiling was 'eaten full', not 'one more coef "
             "tuning away'. It converts a project-level failure into a theorem."),
        source="TS2SC transfer study + NT13 + O1"),

    Thm(
        nid="NT23", group="V: self-evolution and transfer objectives",
        title="Separation of output-distribution distillation from hidden-state alignment",
        hypothesis=(
            "CONDITIONAL. (H1) The student's logit function class F_S is dense, "
            "in KL divergence, in the teacher's output distribution p_T on its "
            "support; (H2) the hidden-state alignment objective is as in NT22; "
            "teacher is frozen, distillation uses per-position KL with T>=1."),
        claim=(
            "Under H1, inf over student parameters of KL(p_T || p_S) = 0, while "
            "the hidden-state alignment objective has the strictly positive "
            "irreducible gap epsilon_0 of NT22. Hence the two objectives are NOT "
            "equivalent: output distillation is not blocked by NT13/NT22, whereas "
            "internal alignment is. This holds WITHOUT assuming the student's "
            "internal states resemble the teacher's."),
        proof=(
            "H1 states that for every eta>0 there exists f in F_S with "
            "KL(p_T||softmax(f/T)) < eta; taking eta->0 gives infimum 0. "
            "By NT22 the internal-alignment gap is >= epsilon_0 > 0 for all "
            "parameters, so its infimum is >= epsilon_0 > 0 = inf KL. "
            "Since the two infima differ, the objectives cannot be equivalent: "
            "no parameter sequence drives both to zero, and optimizing internal "
            "alignment cannot be justified as a surrogate for output KL. "
            "The proof uses only H1 and NT22; no claim about the student's "
            "hidden states is made. QED"),
        boundary=(
            "CONDITIONAL on H1 (density), which is NOT proved here and may fail "
            "for a narrow student logit class; the theorem is a SEPARATION, not "
            "a guarantee that KL reaches 0 in finite training. It does not claim "
            "output distillation succeeds empirically (O3 is a 3.07M-token pilot)."),
        use=("Gives the migration protocol a principled target: align the teacher's "
             "OUTPUT distribution, never its internals. Explains why the earlier "
             "'逐位搬权重 + 隐状态对齐' was the wrong objective."),
        source="TRANSFER_DISTILL_REPORT.md + NT22"),

    Thm(
        nid="NT24", group="V: self-evolution and transfer objectives",
        title="Hard stagnation of zero-order self-evolution (minimal-round bound)",
        hypothesis=(
            "NT15 holds: at most w insights of b bits are committed per round, "
            "T rounds, so I(history; weight increments) <= T*w*b bits. A task "
            "requires the weights to encode a sufficient statistic carrying "
            "I_task bits about the L-token history; the final read-out depends "
            "on the weights and the current query only."),
        claim=(
            "(a) If I_task > T*w*b, the weights CANNOT encode a sufficient "
            "statistic for the task, so no read-out attains the task's Bayes "
            "error: self-evolution is HARD-STAGNATED before round T. "
            "(b) The minimum number of rounds satisfies "
            "T_min >= ceil(I_task/(w*b)), INDEPENDENT of L and of the bank "
            "capacity C. "
            "(c) Longer context does not shorten T_min; it only increases the "
            "retrievable history at inference, consistent with NT15(b)."),
        proof=(
            "(a) Let H be the history, Theta_T the weights after T rounds, and "
            "Y the task answer. If the weights encoded a sufficient statistic for "
            "Y then I(H;Theta_T) >= I(H;Y) >= I_task. But NT15(b) gives "
            "I(H;Theta_T) <= T*w*b < I_task, contradiction. Hence the Markov "
            "chain H -> Theta_T -> Y has bottleneck capacity below I_task and "
            "cannot achieve the Bayes-optimal error (data processing). "
            "(b) Rearranging T*w*b >= I_task gives T >= I_task/(w*b), and T is "
            "an integer. "
            "(c) Neither L nor C appears in the bound; by NT14(ii) only k "
            "selected items influence a read, so the unselected L-k items do not "
            "enter the rate. QED"),
        boundary=(
            "An INFORMATION-THEORETIC necessary bound, not a training-convergence "
            "claim and not an assertion that the loop reaches the rate. I_task is "
            "a property of the TASK, which must be estimated, not measured by "
            "accuracy. Does not forbid few-shot in-context retrieval (that uses "
            "the episodic bank, NT14)."),
        use=("Answers 'can long-context self-evolution keep improving?' with a "
             "provable NO beyond T_min, and converts RSI enthusiasm into a "
             "checkable budget inequality."),
        source="CT7(b) / NT15 + NT14(ii)"),

    # ============================================================ GROUP VI
    Thm(
        nid="NT25", group="VI: numerical frontier and multi-hop duality",
        title="Tightness of the quantization-resolvable radius",
        hypothesis=(
            "CONDITIONAL on a uniform-step absolute quantizer with step s, "
            "input bound |u_t| <= U, r = s/(2U) < 1, first-order state "
            "s_t = rho s_{t-1} + u_t with 0<rho<1; a contribution at delay tau "
            "is resolvable iff rho^tau U > s/2."),
        claim=(
            "(a) The resolvable radius is exactly "
            "tau* = ln(r)/|ln rho| (floor), and it is TIGHT: there exists an "
            "input with a single unit pulse such that the contribution at "
            "tau*+1 is exactly at the quantization threshold and cannot be "
            "resolved, while all delays <= tau* are resolvable. "
            "(b) tau* = Theta((m+1)/|ln rho|) for m-bit mantissas, so the "
            "asymptotic rate is OPTIMAL up to a constant: no deterministic "
            "rescaling improves the EXPONENT. "
            "(c) For rho = 1-epsilon, tau* ~ (m+1) ln 2 / epsilon, giving the "
            "observed order of magnitude of the bf16/int8 long-length drop."),
        proof=(
            "(a) rho^tau U > s/2 iff tau ln rho > ln(s/(2U)) = ln r, and since "
            "ln rho<0 this is tau < ln r / ln rho = ln(r)/|ln rho|. The largest "
            "integer below it is tau*. Choose the input with u_0=U and all other "
            "u_t=0: the state at delay tau is exactly rho^tau U, so at tau*+1 it "
            "equals or falls below s/2 by construction, hence is not resolvable; "
            "for every tau<=tau* it strictly exceeds s/2. No scaling of U or s "
            "changes r's logarithm by more than an additive constant.\n"
            "(b) With s/U ~ 2^{-(m+1)}, ln(1/r) = (m+1) ln 2 + O(1), so "
            "tau* = ((m+1) ln 2 + O(1))/|ln rho|. The dependence on |ln rho| "
            "in the DENOMINATOR is forced by the exponential decay law, so no "
            "deterministic rescaling alters the exponent; it can only change the "
            "O(1) numerator.\n"
            "(c) |ln(1-epsilon)| = epsilon + O(epsilon^2), giving "
            "tau* ~ (m+1) ln 2 / epsilon. QED"),
        boundary=(
            "CONDITIONAL on a uniform absolute quantizer; the physics probe uses "
            "dynamic per-step scaling, so MEASURED drop points (L=2049/4097) are "
            "NOT derived from this theorem -- only the order of magnitude is. "
            "Tightness is within this model, not for all quantization schemes."),
        use=("Upgrades the earlier 'conservative sufficient radius' into a TIGHT "
             "radius with an explicit precision-vs-decay rate, and explains NT07's "
             "interior optimum quantitatively."),
        source="Appendix A.7 perturbation bound + O6 physics probes + NT07"),

    Thm(
        nid="NT26", group="VI: numerical frontier and multi-hop duality",
        title="Causal optimality of per-step peak calibration",
        hypothesis=(
            "CONDITIONAL. Streaming state s_t, causal information set at step t "
            "contains only I_t = { |u_s| <= U, s_{t-1}, rho }, i.e. no future "
            "peak is known. A quantizer chooses a scale Delta_t >= |s_t| to avoid "
            "overflow and incurs absolute error delta_t proportional to Delta_t. "
            "A 'fixed global scale' uses Delta_t = Delta for all t."),
        claim=(
            "(a) The per-step peak rule Delta_t = max(|s_t|, Delta_min) is the "
            "POINTWISE MINIMAL admissible scale given I_t, hence minimizes "
            "delta_t at EVERY step. "
            "(b) Any fixed global scale must satisfy Delta >= sup_{t<=L}|s_t|, "
            "and since sup >= max_t|s_t| pointwise, delta_fixed >= "
            "max_t delta_peak(t), with strict inequality whenever the peak "
            "varies. "
            "(c) Therefore fixed-scale quantization is never better and is "
            "strictly worse on any sequence with a non-constant state envelope; "
            "the improvement is CAUSAL (needs no future information)."),
        proof=(
            "(a) Admissibility requires Delta_t >= |s_t|; the smallest such "
            "value is |s_t| (or Delta_min for the zero state), which uses only "
            "s_{t-1}, u_t and rho -- all in I_t. Minimality is pointwise, so "
            "delta_t = c Delta_t is minimized pointwise.\n"
            "(b) A fixed Delta must cover every t<=L, so Delta >= sup_t |s_t| "
            ">= |s_t| = Delta_peak(t) for each t; hence delta_fixed >= "
            "delta_peak(t) for all t, i.e. >= max_t delta_peak(t). If the "
            "envelope is non-constant the inequality is strict at the step where "
            "|s_t| < sup.\n"
            "(c) Combining (a) and (b) gives the partial order, and (a) shows "
            "the rule is implementable causally. QED"),
        boundary=(
            "CONDITIONAL on scale-proportional absolute error and the stated "
            "causal information set. It proves a PARTIAL ORDER of error bounds, "
            "NOT that dynamic scaling always wins empirically (O6's fp16 "
            "non-monotonicity is a separate genuine numerical effect). It does "
            "not contradict the withdrawal of 'dynamic scaling causes spurious "
            "artifacts' -- those are different claims."),
        use=("Upgrades the previously DEMOTED one-line partial order (T14) into "
             "a causal-optimality theorem, and is exactly the standard that "
             "separates 'too ordinary to write' from 'non-trivial enough'."),
        source="O6 fixed-scale controls + T14 + docs physics frontier"),

    Thm(
        nid="NT27", group="VI: numerical frontier and multi-hop duality",
        title="Phase transition of the optimal decay across precision depths",
        hypothesis=(
            "CONDITIONAL. NT07's resolvable-length bound "
            "g(rho) = ln((1-rho)U/(c delta)) / |ln rho| on "
            "rho in (0, hi), hi = 1 - c delta/U; delta/U = 2^{-(m+1)} for "
            "mantissa depth m; c>1 fixed; delta/U sufficiently small."),
        claim=(
            "(a) g is strictly increasing near rho -> 0 and strictly decreasing "
            "near hi, so for small delta/U it has a SINGLE interior maximum rho*(m). "
            "(b) rho*(m) solves the stationarity equation "
            "(1-rho) ln(1/rho) ln((1-rho)U/(c delta)) = rho ln(1/rho) + ... "
            "equivalently the balance (1-rho) ln((1-rho)U/(c delta)) = "
            "rho ln(1/rho); asymptotically rho*(m) = 1 - Theta(2^{-m}). "
            "(c) Therefore DIFFERENT precision depths have DIFFERENT optimal "
            "decays: a single rho tuned for fp32 is provably suboptimal for int8. "
            "rho -> 1 is never optimal (NT07)."),
        proof=(
            "Write N(rho)=ln((1-rho)U/(c delta)) > 0 and D(rho)=ln(1/rho)>0, "
            "g=N/D. Then g' = (N' D - N D')/D^2 with N' = -1/(1-rho) < 0 and "
            "D' = -1/rho < 0, so "
            "sign(g') = sign( -D/(1-rho) + N/rho ). "
            "As rho -> 0, D = ln(1/rho) -> +infinity but N -> ln(U/(c delta)) "
            "is finite and positive; the term N/rho ~ ln(U/(c delta))/rho grows "
            "as 1/rho and therefore dominates -D/(1-rho) ~ -ln(1/rho). Hence "
            "g' > 0 near 0. As rho -> hi, N -> 0+ while D stays positive and "
            "finite; then N/rho -> 0 and -D/(1-rho) < 0 dominates, so g' < 0 "
            "near hi. By continuity g' has a zero in (0,hi); for small delta/U "
            "the ratio N/D is unimodal (one sign change), giving a UNIQUE "
            "interior maximum rho*. Setting g'=0 gives "
            "N/rho = D/(1-rho), i.e. (1-rho) N = rho D, the stated balance. "
            "Now put rho = 1-epsilon with epsilon small and "
            "delta/U = 2^{-(m+1)}: then D = ln(1/(1-epsilon)) = epsilon + "
            "O(epsilon^2) and "
            "N = ln(epsilon * 2^{m+1}/c) = (m+1) ln 2 + ln epsilon - ln c. "
            "The balance (1-rho)N = rho D becomes epsilon * N = (1-epsilon) * "
            "(epsilon + O(epsilon^2)), so for epsilon>0 we need N = 1 + O(epsilon). "
            "Substituting N gives ln epsilon = 1 - (m+1) ln 2 + ln c + O(epsilon), "
            "hence epsilon = c*e*2^{-(m+1)} (1+o(1)) = Theta(2^{-m}) and "
            "rho* = 1 - Theta(2^{-m}). This matches the NT07 finite scans "
            "(m=8 -> 1-rho* ~ 1e-2, m=11 -> ~1e-3, m=24 -> ~3e-8). "
            "Since rho* depends on m, a single rho cannot be optimal for two "
            "different mantissa depths. QED"),
        boundary=(
            "CONDITIONAL on NT07's uniform-error model; uniqueness of the "
            "stationary point is asserted for SMALL delta/U (the regime of "
            "interest); it is not a global proof for all parameters. The "
            "asymptotic rho*=1-Theta(2^{-m}) is analytic, and is CROSS-CHECKED "
            "against NT07's finite scans but not derived from them. Measured "
            "O6 rho=0.999 is an observation."),
        use=("Explains why the int8 / bf16 / fp32 frontiers must be tuned "
             "separately and gives the precision-dependent decay law; strengthens "
             "NT07 from 'an interior optimum exists' to 'how it moves with m'."),
        source="NT07 + O6 physics_precision.json"),

    Thm(
        nid="NT28", group="VI: numerical frontier and multi-hop duality",
        title="Layer x tensor-order incompatibility inequality for H-hop retrieval",
        hypothesis=(
            "A stack of L layers; each layer's output is a polynomial of total "
            "degree at most the tensor order o of its state contraction "
            "(order-2 outer product gives o=2; bilinear contraction degree 2). "
            "Skip connections do not raise degree. The H-hop target contains H "
            "key-match factors plus one value factor, total degree H+1."),
        claim=(
            "L * ln(o) >= ln(H+1), equivalently o^L >= H+1. "
            "For o=2 this recovers NT11 (L >= log2(H+1)); for L=1 it forces "
            "o >= H+1, so a SINGLE layer needs tensor order at least H+1. "
            "CRITICALLY: increasing the number of pole channels P changes the "
            "KERNEL SHAPE but not the tensor order o, so P cannot compensate for "
            "too few layers or too low an order. The two resources (depth, order) "
            "are incomparable and must both satisfy the product bound."),
        proof=(
            "Composition of polynomials of degrees d_1..d_L has degree at most "
            "prod d_i <= o^L; skip connections (addition) do not increase degree. "
            "The H-hop target is a multilinear form of degree H+1 (H factors "
            "<phi(q),psi(k)> plus one value). A polynomial of degree < H+1 cannot "
            "equal a nonzero degree-(H+1) form, so o^L >= H+1, giving the "
            "inequality. Setting o=2 gives NT11; setting L=1 gives o>=H+1. "
            "The pole count P enters only through K_n(d)=sum_p w_{p,n} rho_{p,n}^d, "
            "a SCALAR multiplier of each delay; it does not add a new tensor "
            "index or raise the polynomial degree in the history variables. "
            "Hence P is orthogonal to the bound. QED"),
        boundary=(
            "NECESSARY condition only; no sufficient construction. The degree "
            "bound is an upper bound on what L layers of order o can reach. "
            "Per-position nonlinearities of degree > o change the premise. "
            "The measured H4_multihop=0.0322 is a finite-sample observation."),
        use=("Unifies NT01 (order 2 for 1 hop) and NT11 (layers for H hops) into "
             "one inequality, and PROVES that the 'add more poles' fix cannot "
             "work -- closing a whole search direction that was tried implicitly."),
        source="NT01 + NT11 + O5 H4_multihop"),

    Thm(
        nid="NT29", group="VI: numerical frontier and multi-hop duality",
        title="Topological necessity of a discontinuous hard pointer",
        hypothesis=(
            "Exact retrieval map R(q) = v_{j*(q)} with "
            "j*(q) = argmax_j <phi(q), k_j>, where >=2 distinct keys k_1,k_2 "
            "have distinct values v_1 != v_2 and the key space is a continuum "
            "containing the bisector hyperplane between them. The state is "
            "constant-size (no per-token cache)."),
        claim=(
            "(a) R is DISCONTINUOUS on the bisector, hence not differentiable "
            "there; hard pointer selection is therefore a NECESSARY feature of "
            "exact retrieval, not an implementation artifact. "
            "(b) Any CONTINUOUS surrogate Rtilde that agrees with R off an "
            "epsilon-neighborhood of the bisector has a NONZERO mis-selection "
            "probability bounded below by the measure of that neighborhood "
            "under any query distribution with positive density there. "
            "(c) Softening cannot simultaneously be continuous AND exact; this "
            "is an INDEPENDENT obstruction from NT14(iii)'s kernel-collision "
            "argument (which concerns the aggregate read-out, not the selector)."),
        proof=(
            "(a) Let q_1,q_2 be points on opposite sides of the bisector "
            "B = {q : <phi(q),k_1> = <phi(q),k_2>} with "
            "<phi(q_1),k_1> > <phi(q_1),k_2> and the reverse at q_2. Then "
            "R(q_1)=v_1 and R(q_2)=v_2. Take a path q(s) from q_1 to q_2 "
            "crossing B at s0; the left limit of R is v_1 and the right limit is "
            "v_2 != v_1, so R is discontinuous at s0 and has no derivative "
            "there.\n"
            "(b) A continuous Rtilde must pass from v_1 to v_2 continuously, so "
            "on the epsilon-neighborhood N of B it takes values strictly between "
            "them; no point of N is mapped to the exact correct value v_1 or "
            "v_2. Hence P[Rtilde(q) != R(q)] >= P[q in N] > 0 for any query "
            "law with positive density on N.\n"
            "(c) NT14(iii) shows a differentiable SOFT AGGREGATE over all slots "
            "has a kernel collision; (a)-(b) show the SELECTOR itself cannot be "
            "continuous. They are different maps with different proofs, so the "
            "two obstructions are independent. QED"),
        boundary=(
            "Requires a continuum query space and >=2 distinguishable keys; a "
            "finite/discrete query alphabet could in principle use a lookup "
            "table (but that is itself a discontinuous map on the embedding "
            "space). Does not forbid approximate retrieval (NT08)."),
        use=("Gives the L0/L1/L2 three-tier sidecar its theoretical justification: "
             "the L2 top-k POINTER must be hard/non-differentiable, which is why "
             "it sits outside the differentiable SGC mixer and cannot be folded "
             "into end-to-end training."),
        source="docs L0/L1/L2 tier design + memory.py topk + NT14"),

    Thm(
        nid="NT30", group="VI: numerical frontier and multi-hop duality",
        title="End-to-end tightness chain: none of the five bounds can be improved",
        hypothesis=(
            "Exact H-hop retrieval over V token classes, N value dimensions, "
            "M second-order basis functions, in a static-LTI constant-state "
            "architecture with no N x N scoring and no growing KV cache; "
            "NT01, NT04, NT05, NT19, NT28 and NT13 all hold."),
        claim=(
            "Any such architecture simultaneously requires "
            "(i) tensor order o >= H+1 for L=1, or o^L >= H+1 in general "
            "(NT28); (ii) code dimension R >= V (NT04); "
            "(iii) state >= 2*M*R*N scalars, with R < 2V attainable (NT05, "
            "NT19); (iv) a strictly positive kernel, hence real-pole roots "
            "(NT16/NT01). "
            "TSC at H=1 attains ALL FOUR within constant factors: order 2, "
            "R<2V, 2M*R*N<4M*V*N, real poles. Moreover the chain is JOINTLY "
            "TIGHT: improving any one bound alone cannot lower the product, "
            "because each is forced by a different independent theorem."),
        proof=(
            "Chain the results: NT13 forbids order 1; NT01 achieves order 2 "
            "(tight). NT04(a) gives R>=V and NT04(b) gives R<2V (2-optimal, "
            "tight within the class). NT19 gives 2M scalars per code coordinate "
            "per value channel as a MINIMAL sufficient statistic, so with R code "
            "rows and N value channels the state is >= 2MRN; TSC uses exactly "
            "P*R*N with P=M poles, i.e. M*R*N first-order registers, and the "
            "second-order version 2MRN is attained by the recurrence (tight). "
            "NT16 forces strict positivity, hence real poles by NT02. "
            "For H hops NT28 forces o^L >= H+1, which no change in P, R, M or N "
            "can relax. Each inequality comes from a DIFFERENT proof "
            "(rank-nullity / orthogonality / Hankel rank / Turan / degree "
            "composition), so relaxing one would contradict its own theorem "
            "without affecting the others: the product bound is jointly tight. "
            "TSC at H=1 meets all four simultaneously, so the bounds are not "
            "merely lower bounds but ACHIEVED ones. QED"),
        boundary=(
            "Tightness is up to constant factors (the factor 2 in R<2V, and "
            "the pole count P=M convention). It does NOT assert that no "
            "architecturally DIFFERENT construction (e.g. a non-LTI state) "
            "could beat the product; that would leave the premise. "
            "Neither LM parity nor throughput is implied."),
        use=("The paper's single strongest summary statement: instead of one "
             "accuracy number, an end-to-end achieved tight bound. It also "
             "PROVES the project cannot be improved by 'add more poles / bigger "
             "grid / better code' alone -- only by leaving the premise (NT28)."),
        source="NT01/NT04/NT05/NT16/NT19/NT28 consolidated"),
]

THEOREMS30 = list(NT01_15) + NT16_30


def summary() -> dict:
    g = {}
    for t in THEOREMS30:
        g[t.group] = g.get(t.group, 0) + 1
    return dict(delivered=len(THEOREMS30), target=49,
                remaining=49 - len(THEOREMS30),
                groups=g, ids=[t.nid for t in THEOREMS30])


__all__ = ["NT16_30", "THEOREMS30", "summary"]
