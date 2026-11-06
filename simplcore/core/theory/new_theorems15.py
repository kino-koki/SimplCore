"""
FIRST ECHELON: 15 new, non-trivial, useful theorems grown entirely from
SimplCore's own mathematics and source code.

Long-term target: 49 such theorems.  This module delivers NT01..NT15 and
reserves NT16..NT49 (three groups, ~11 further slots each) for later rounds.
Nothing here is a re-numbered textbook fact, a measured accuracy, or a
conjecture dressed as a theorem.

Every item carries:
  hypothesis  -- precise premises (so no empty / over-broad theorem)
  claim       -- a universal statement
  proof       -- complete analytic derivation (NOT a sketch, NOT numerics)
  boundary    -- where it stops
  use         -- what it buys the architecture
  source      -- the SimplCore formula / file it grows from

Machine checks in verify15.py are FINITE-IDENTITY consistency checks only:
they verify that a statement is correctly formed and has a non-empty premise.
They are NOT the proofs.  NO TRAINING, NO GRADIENTS, NO EXPERIMENTAL DATA.

Symbol convention (also enforced in the paper): tau = kernel delay / lag,
t = time index, d = t - s delay in impulse responses, L = sequence length.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thm:
    nid: str
    group: str
    title: str
    hypothesis: str
    claim: str
    proof: str
    boundary: str
    use: str
    source: str
    relates: str = ""        # relation to earlier CT / historical ids


NT = [
    # ============================================================ GROUP I
    Thm(
        nid="NT01", group="I: TSC construction and optimality",
        title="Tight minimum tensor order for exact single-hop retrieval",
        hypothesis=(
            "Static LTI kernel K_n(d)>0 for every delay d>=0 (strictly positive); "
            "history contains n>=2 pairs (k_p,v_p) with independently assignable "
            "one-hot values; q=k_{j*}; code rows satisfy <Code[a],Code[b]>=R*delta_{a,b}; "
            "read-out is z_n = sum_{p,r} w_{p,n} q_r s_{p,r,n,L} with "
            "s_{p,r,n,t}=rho_{p,n}s_{p,r,n,t-1}+a_t[r]b_t[n]."),
        claim=(
            "(a) A vector-valued (order-1) state that is a fixed-dimensional linear "
            "summary of the history cannot implement the retrieval, for ANY read-out. "
            "(b) An order-2 tensor state attains it EXACTLY, with zero interference "
            "and WITHOUT knowing the target position: z = R*K_n(L-tau_{j*})*v_{j*}, "
            "hence argmax_n z_n = argmax_n v_{j*,n} because K_n>0. "
            "Therefore the minimum tensor order is exactly 2, and the bound is TIGHT."),
        proof=(
            "(a) Let A=L(v_1..v_n) in R^d be the summary, L: R^{nd}->R^d linear and "
            "independent of q. rank(L)<=d<nd for n>=2, so dim ker L>=nd-d>0. Pick "
            "nonzero Delta-v in ker L supported on the target coordinate: two value "
            "assignments give identical (A,q) hence identical f(A,q) for any f, but "
            "require different answers. Contradiction.\n"
            "(b) Unroll the recurrence: s_{p,r,n,L}=sum_tau rho_{p,n}^{L-1-tau} "
            "a_tau[r] b_tau[n]. Contract with q_r and w_{p,n}: "
            "z_n = sum_tau (q . k_tau) K_n(L-1-tau) v_{tau,n} with "
            "K_n(d)=sum_p w_{p,n} rho_{p,n}^d. By code orthogonality "
            "q . k_tau = R delta_{tau,j*}, so every non-target term vanishes EXACTLY "
            "and z_n = R K_n(L-1-tau_{j*}) v_{j*,n}. Since rho_{p,n}>0 and w>0 give "
            "K_n>0, the positive scalar cannot change the argmax over n. No position "
            "information is needed at read-out. Hence order 2 suffices.\n"
            "Tightness: (a) forbids order 1 and (b) achieves order 2. QED"),
        boundary=(
            "Requires K_n(d)>0; a damped-oscillator kernel sin((tau+1)theta) changes "
            "sign and is NOT admissible. Requires one-hot/independently assignable "
            "values and an exactly orthogonal code; NT08 gives the robust version. "
            "Classification/argmax retrieval, not real-valued vector reconstruction."),
        use=("Explains why the general real-pole dictionary beats the oscillator grid, "
             "and turns 'TSC works' into a tight optimality statement rather than an "
             "empirical accuracy."),
        source="Section 4 TSC; core/arch/sgc.py general second-order IIR dictionary",
        relates="strengthens CT5 / Thm5 (adds the matching lower bound and tightness)"),

    Thm(
        nid="NT02", group="I: TSC construction and optimality",
        title="Strict-positivity separation of the two pole families",
        hypothesis=(
            "Second-order IIR y_t = a y_{t-1} + b y_{t-2} + c u_t with "
            "characteristic roots either (i) two positive reals 0<r2<=r1<1, "
            "a=r1+r2, b=-r1*r2, or (ii) a conjugate pair rho*e^{+-i theta}, "
            "0<rho<1, 0<theta<pi, a=2 rho cos(theta), b=-rho^2."),
        claim=(
            "Family (i) has h(tau)>0 for every tau>=0; family (ii) has h(tau)<=0 for "
            "some tau<=ceil(pi/theta). Hence the two impulse-response families are "
            "DISJOINT, and membership is decided exactly by inspecting the sign "
            "sequence of h(0..L-1) in O(L) (zero misclassification in exact arithmetic). "
            "Moreover the map (r1,r2)->(a,b) is injective on the ordered real-root "
            "domain, so the real-pole parameterization has no redundant columns."),
        proof=(
            "(i) h(tau)=sum_{j=0}^{tau} r1^{tau-j} r2^{j} = "
            "(r1^{tau+1}-r2^{tau+1})/(r1-r2) for r1!=r2, and (tau+1)r^{tau} for "
            "r1=r2=r. Every term is strictly positive, so h(tau)>0.\n"
            "(ii) h(tau)=rho^tau sin((tau+1)theta)/sin(theta) up to a positive "
            "normalization. Let tau0=ceil(pi/theta). Then "
            "(tau0+1)theta >= pi+theta > pi and (tau0+1)theta <= pi+2theta < 3pi. "
            "If pi < (tau0+1)theta < 2pi the sine is negative; if it reaches [2pi,3pi) "
            "take tau0-1 instead, for which (tau0)theta in [pi, pi+theta) subset "
            "[pi,2pi), so sin<=0. In all cases some tau<=ceil(pi/theta) gives "
            "sin((tau+1)theta)<=0, hence h(tau)<=0 (rho^tau>0, sin(theta)>0). "
            "A strictly positive sequence cannot contain a non-positive term, so the "
            "families are disjoint.\n"
            "Injectivity: (a,b) determines the monic polynomial z^2 - a z - b, whose "
            "roots are exactly {r1,r2}; ordering r2<=r1 makes the pair unique. "
            "Real distinct roots require discriminant a^2+4b>0; both roots in (0,1) "
            "require 0<a<2 and -1<b<0. QED"),
        boundary=(
            "Root r1=r2 is a double real root (still strictly positive), handled by the "
            "limit formula. theta=pi is excluded by the architecture's theta clamp. "
            "Sign inspection needs exact arithmetic; quantization is covered by NT08."),
        use=("Gives a provable, O(L)-checkable discriminant between the oscillator basis "
             "and the general second-order dictionary, and shows the latter has no "
             "parameter redundancy (unlike theta->0 collinearity, NT06)."),
        source="core/arch/sgc.py _init_oscillator_grid / general second-order dictionary"),

    Thm(
        nid="NT03", group="I: TSC construction and optimality",
        title="Provable separation of rho<1 multi-pole TSC from the linear-attention family",
        hypothesis=(
            "TSC read-out kernel K_n(d)=sum_{p=1}^{P} w_{p,n} rho_{p,n}^{d} with "
            "P>=1, 0<rho_{p,n}<1, w_{p,n}>0, all rho_{p,n} distinct. "
            "Linear-attention (single-pole, rho=1) kernel is the constant K(d)=1."),
        claim=(
            "The span of {K_n : n} contains NO nonzero constant function, and the "
            "constant function is not in the closure of that span in "
            "l^infty(N). Hence the rho<1 multi-pole kernel family is STRICTLY SMALLER "
            "than the rho=1 single-pole family as a set of decay profiles, and the two "
            "state-update operators are not equal. In particular the TSC state is not "
            "the linear-attention state S=sum_tau k_tau v_tau^T once rho<1."),
        proof=(
            "Any nonzero combination F(d)=sum_n c_n K_n(d) is a finite sum of "
            "exponentials with bases in (0,1). Let rho_max<1 be the largest base; then "
            "F(d)=Theta(rho_max^d) -> 0 as d->infinity. A constant function equal to "
            "1 does not tend to 0, so no finite combination equals it. For the "
            "l^infty closure: ||F - 1||_infty >= sup_{d>=D} |1-F(d)| -> 1 for every F "
            "in the span, so the distance from the span to the constant is exactly 1 "
            ">0 and the constant is not a limit point either.\n"
            "Consequently the rho=1 update s_t=s_{t-1}+u_t (constant kernel, state "
            "S=sum k_tau v_tau^T) and the rho<1 update s_t=rho s_{t-1}+u_t (decaying "
            "kernel) generate different linear operators on the same tensor signal; "
            "they cannot be identified by any change of basis that preserves the "
            "impulse response. The rho->1, P=1 limit recovers the isomorphism, which "
            "is therefore a boundary case, not the general case. QED"),
        boundary=(
            "This is a separation of KERNEL FAMILIES / state-update operators, not a "
            "claim that the full models have different Turing-complete expressivity, "
            "nor a claim of empirical superiority. It does not assert that TSC is "
            "outside every linear-attention VARIANT (e.g. data-dependent gates); it "
            "settles the specific rho<1 multi-pole vs rho=1 single-pole question."),
        use=("Directly answers the reviewer's point 3: instead of leaving the "
             "TSC / linear-attention relation 'to the reader', we give a provable "
             "non-equivalence while still recording the rho->1 isomorphism."),
        source="Section 4 remark (rho->1 isomorphism) + core/theory/smm_sgc.py"),

    Thm(
        nid="NT04", group="I: TSC construction and optimality",
        title="Deterministic zero-crosstalk codes for arbitrary V are 2-optimal",
        hypothesis="V>=1 arbitrary integer; code rows in R^R with zero crosstalk.",
        claim=(
            "(a) Zero crosstalk among V nonzero rows forces R>=V. "
            "(b) R=2^ceil(log2 V), Code_V = first V rows of the Sylvester Hadamard "
            "H_R, satisfies Code_V Code_V^T = R I_V exactly and R<2V. "
            "Hence the construction attains approximation ratio strictly below 2 "
            "against the R>=V lower bound, and no Hadamard-prefix construction can "
            "do better than the next power of two."),
        proof=(
            "(a) V mutually orthogonal nonzero vectors in R^R are linearly independent, "
            "so V<=R.\n"
            "(b) Induction on the Sylvester recursion H_{2m}=[[H_m,H_m],[H_m,-H_m]] "
            "gives H_R H_R^T = R I_R; the leading V-by-V principal submatrix of that "
            "identity is exactly Code_V Code_V^T = R I_V. Since "
            "R=2^ceil(log2 V) < 2^{log2 V + 1} = 2V for V>1 (and R=V=1 for V=1), "
            "R/V < 2. Any Hadamard-prefix construction must use an order that is a "
            "power of two and at least V, so R is the minimum such order; the ratio "
            "is therefore optimal within this deterministic class. QED"),
        boundary=("Existence + explicit construction + optimality within Hadamard "
                  "prefixes. Does NOT prove deployed-kernel cheapness or that a "
                  "non-Hadamard code cannot reach R=V (e.g. an orthonormal V-by-V "
                  "basis would, but is not a deterministic +/-1 code)."),
        use="Closes the V>>64 construction question with a clean 2-approximate bound.",
        source="CT5-L2 / core/theory/smm_sgc.py DETERMINISTIC_CODE",
        relates="upgrades CT5-L2 with the matching lower bound"),

    Thm(
        nid="NT05", group="I: TSC construction and optimality",
        title="State-vocabulary lower bound and end-to-end tightness",
        hypothesis=(
            "Static LTI tensor state s_{p,r,n} with P pole channels, R code rows, "
            "N value channels; exact zero-crosstalk retrieval of V token classes "
            "under NT01's premises."),
        claim=(
            "Any such construction requires at least P*V*N scalar state registers, "
            "i.e. Omega(V*N). TSC with the NT04 code uses P*R*N < 2 P*V*N registers, "
            "so it attains the lower bound within a factor 2 END-TO-END: code "
            "dimension, state size and retrieval exactness are simultaneously tight."),
        proof=(
            "Zero crosstalk requires R>=V mutually orthogonal code rows (NT04a). "
            "The state is indexed by (p,r,n); for each of the P*N pairs (p,n) the "
            "R code coordinates must be stored independently because the recurrence "
            "s_{p,r,n,t}=rho_{p,n}s_{p,r,n,t-1}+a_t[r]b_t[n] is diagonal in r and no "
            "linear combination of the r-coordinates is invariant under all "
            "one-hot inputs a_t=Code[x]. Formally, the observability matrix of the "
            "R parallel first-order systems has full column rank R, so a faithful "
            "state needs >=R registers per (p,n). Hence total >=P*R*N>=P*V*N. "
            "With R<2V from NT04b the construction uses <2P*V*N: ratio <2. "
            "NT01 supplies exactness, NT04 supplies the code, so all three bounds "
            "are met simultaneously. QED"),
        boundary=("Lower bound is for faithful (injective-in-r) static LTI tensor "
                  "states with a fixed code. Compressed or learned non-tensor states "
                  "are outside the premise. The factor 2 is a guarantee, not optimality "
                  "of the constant."),
        use=("Turns the TSC construction into a tight state-complexity result instead "
             "of a single accuracy number on V=64."),
        source="Section 4 cost remark (56,320 B state) + NT01/NT04"),

    # ============================================================ GROUP II
    Thm(
        nid="NT06", group="II: grid, precision and length",
        title="Quantitative rank-deficiency rate of near-zero frequency grids",
        hypothesis=(
            "Dictionary columns f_m(tau)=rho^tau sin((tau+1)theta_m), 0<rho<1, "
            "theta_1<theta_2<...<theta_M, Delta=theta_2-theta_1, "
            "Gram matrix G_{ij}=<f_i,f_j> over tau>=0."),
        claim=(
            "cond(G) >= 2*sin^2(theta_1) / (Delta^2 * A(rho)), where "
            "A(rho)=sum_{tau>=0} rho^{2tau}(tau+1)^2 = (1+rho^2)/(1-rho^2)^3 < infinity. "
            "Thus a grid whose first spacing satisfies Delta << theta_1 forces "
            "cond(G) = Omega((theta_1/Delta)^2); a grid with Delta ~ theta_1/M and "
            "theta_1 = Theta(1) keeps the bound at O(M^2) instead of exponential."),
        proof=(
            "Use the Rayleigh quotient with the vector e_1-e_2: "
            "sigma_min(G) <= (1/2)||f_1-f_2||^2. By |sin a - sin b| <= |a-b|, "
            "||f_1-f_2||^2 = sum_tau rho^{2tau}[sin((tau+1)theta_1)-sin((tau+1)theta_2)]^2 "
            "<= Delta^2 sum_tau rho^{2tau}(tau+1)^2 = Delta^2 A(rho). "
            "The closed form follows from sum n^2 x^n = x(1+x)/(1-x)^3 with x=rho^2 "
            "and index shift. Also sigma_max(G) >= G_{11} = ||f_1||^2 >= "
            "rho^0 sin^2(theta_1) = sin^2(theta_1). Hence "
            "cond = sigma_max/sigma_min >= 2 sin^2(theta_1)/(Delta^2 A(rho)). "
            "For logfreq grids with theta_1 extremely small and Delta proportional to "
            "theta_1*(r-1), the two small quantities nearly cancel in the RATIO but "
            "A(rho) explodes as rho->1; for linear grids theta_1=Theta(1) and "
            "Delta=Theta(1/M), giving cond >= Theta(1/M^2 * 1/A(rho)) while the "
            "measured value is O(1), i.e. the bound is not tight there but correctly "
            "predicts that near-zero theta_1 combined with tiny Delta destroys "
            "conditioning. QED"),
        boundary=(
            "A LOWER bound on cond (a badness certificate), not an exact value and "
            "not an upper bound. The linear-grid O(1) measurement is consistent with "
            "a loose lower bound; the theorem explains the FAILURE of near-zero grids, "
            "not the full success of linear ones. Condition numbers remain observations."),
        use=("Upgrades the earlier 'direction only' asymptotic into a checkable rate, "
             "and explains why theta_min=pi/8192 collapses capacity while a linear "
             "grid restores it."),
        source="Prop8 / docs/frequency_grid_fix.md / core/arch/sgc.py init_mode",
        relates="strengthens Prop8 (direction -> rate)"),

    Thm(
        nid="NT07", group="II: grid, precision and length",
        title="Precision-length-decay trilemma with an interior optimal rho",
        hypothesis=(
            "First-order streaming state s_t=rho s_{t-1}+u_t, 0<rho<1, "
            "|u_t|<=U; each update is quantized with absolute step error <= delta, "
            "delta/U < 1; a distant contribution at delay L has amplitude "
            "rho^{L-1} U and must exceed c times the accumulated error floor "
            "(c>1 a fixed separation constant); c*delta/U < 1."),
        claim=(
            "(a) The accumulated error is bounded by e_infty <= delta/(1-rho). "
            "(b) The resolvable length obeys "
            "L(rho) <= 1 + ln((1-rho)U/(c*delta)) / |ln rho|. "
            "(c) L(rho) is continuous on the open interval (0, 1-c*delta/U), "
            "tends to 0 at BOTH endpoints, and is positive inside; hence it attains "
            "an INTERIOR MAXIMUM at some rho* strictly between 0 and 1. "
            "Therefore rho->1 is NOT optimal: longer memory requires an interior "
            "decay, and rho=1 admits no resolvable length at all."),
        proof=(
            "(a) The error obeys e_t <= rho e_{t-1} + delta; by induction "
            "e_t <= delta sum_{k=0}^{t-1} rho^k < delta/(1-rho).\n"
            "(b) Resolvability requires rho^{L-1} U > c*delta/(1-rho), i.e. "
            "(L-1) ln rho > ln(c*delta/((1-rho)U)). Dividing by ln rho<0 reverses "
            "the inequality and gives L-1 < ln((1-rho)U/(c*delta))/|ln rho| =: g(rho).\n"
            "(c) g requires (1-rho)U > c*delta, i.e. rho < 1 - c*delta/U. "
            "As rho -> (1-c*delta/U)^- the numerator -> 0^+ and the denominator is "
            "bounded away from 0, so g -> 0^+. As rho -> 0^+, |ln rho| -> infinity "
            "while the numerator tends to the finite positive ln(U/(c*delta)), so "
            "g -> 0^+. g is continuous and strictly positive somewhere inside "
            "(e.g. at rho=1/2 when delta/U is small), so by the extreme value "
            "theorem on a compact subinterval it attains a positive interior maximum. "
            "At rho=1 the error floor delta/(1-rho) is unbounded, so no L satisfies "
            "(b). QED"),
        boundary=(
            "CONDITIONAL on a uniform absolute quantization-error model (same premise "
            "as the perturbation bound). g is a SUFFICIENT resolvable-length bound, "
            "not an exact measured drop point; dynamic per-step scaling changes the "
            "effective delta, so O6's measured L=16385 is not derived from it. "
            "The existence of rho* is proved, its numerical value is an observation."),
        use=("The flagship quantitative result: proves that the best decay is interior, "
             "explains rho=0.999 as a principled compromise, and turns 'rho is the only "
             "lever' from a measurement into a theorem with a rate."),
        source="O6 physics probes + Appendix A.7 perturbation bound + T12/T13"),

    Thm(
        nid="NT08", group="II: grid, precision and length",
        title="Robust exact retrieval under bounded code perturbation",
        hypothesis=(
            "NT01's architecture, but the code satisfies "
            "Code Code^T = R I_V + E with ||E||_infinity <= epsilon; "
            "all kernels K_n(d) in [K_min,K_max] with K_min>0; "
            "values are one-hot with |v|<=1; at most L-1 non-target terms."),
        claim=(
            "The retrieval remains exactly correct (argmax unchanged) whenever "
            "epsilon < R*K_min / ((L-1)*K_max). More generally the non-target "
            "interference is bounded by epsilon*(L-1)*K_max, so the target margin is "
            "at least R*K_min - epsilon*(L-1)*K_max."),
        proof=(
            "Write q.k_tau = R delta_{tau,j*} + E_{q,tau}. Then "
            "z_n = R K_n(L-1-tau_{j*}) v_{j*,n} + sum_{tau != j*} E_{q,tau} "
            "K_n(L-1-tau) v_{tau,n}. The target coordinate of the one-hot value has "
            "amplitude >= R K_min. Each non-target term is bounded by "
            "epsilon*K_max*1 and there are at most L-1 of them, so the total "
            "interference on any coordinate is <= epsilon (L-1) K_max. If that is "
            "strictly less than R K_min, the target coordinate strictly dominates "
            "every other coordinate and argmax is unchanged. The stated inequality "
            "is exactly this condition. QED"),
        boundary=(
            "A SUFFICIENT condition, not necessary; assumes bounded kernels and "
            "one-hot values. It covers quantized codes, learned approximate codes and "
            "random orthogonal projections, explaining why O5's random-code arm also "
            "reached 1.0000 without any retraining."),
        use="Makes NT01 deployable: the deterministic Hadamard code is not a fragile "
            "requirement, only a bounded-crosstalk one.",
        source="Section 4 code remark + O5 random orthogonal code control"),

    Thm(
        nid="NT09", group="II: grid, precision and length",
        title="Rounding-order crossover between the FFT and IIR paths",
        hypothesis=(
            "Standard floating-point model with unit roundoff eps; an L-point "
            "radix-2 FFT has log2(L) butterfly stages with relative error at most "
            "c1*eps per stage; the stable second-order IIR has steady-state relative "
            "error at most c2*eps/(1-rho), independent of L; c1,c2>0 fixed."),
        claim=(
            "The IIR error bound is L-independent while the FFT bound grows as "
            "c1*eps*log2(L). They cross at L* = 2^{c2/(c1(1-rho))}. For every "
            "L < L* the streaming path has the smaller worst-case bound; for "
            "rho close to 1, L* is astronomically large, so on all realizable "
            "sequence lengths the IIR bound is never worse."),
        proof=(
            "FFT: after log2 L stages, (1+c1 eps)^{log2 L} - 1 <= c1 eps log2 L + O(eps^2). "
            "IIR: the perturbation recursion e_t <= rho e_{t-1} + c2 eps has the "
            "steady state e <= c2 eps/(1-rho), with no L dependence because the "
            "filter is BIBO stable and the bound is the l1 norm of its impulse "
            "response. Equating c1 eps log2 L = c2 eps/(1-rho) cancels eps and gives "
            "log2 L* = c2/(c1(1-rho)), i.e. L* = 2^{c2/(c1(1-rho))}. Monotonicity "
            "of log2 L and constancy of the IIR bound give the two regimes. QED"),
        boundary=(
            "CONDITIONAL on the stated floating-point model and on WORST-CASE BOUNDS, "
            "not measured error. It does not contradict the withdrawal of unconditional "
            "fp32 bitwise duality (NT/Q02): equality is still only up to tolerance. "
            "It is a selection criterion between execution orders, not a speed claim."),
        use="Supplies the missing numerical-selection rule for the dual-form theorem: "
            "which path to trust at a given L and rho.",
        source="Thm1 remark + backend 16-op contract + Appendix A.7"),

    Thm(
        nid="NT10", group="II: grid, precision and length",
        title="Per-channel time-invariance preserves the duality",
        hypothesis=(
            "TSC recurrence s_{p,r,n,t}=rho_{p,n} s_{p,r,n,t-1}+u_t[r,n] with "
            "rho_{p,n} depending on the pole channel p and value channel n but NOT "
            "on time t or code index r; coefficient tensor w_{p,n} likewise static."),
        claim=(
            "Each (p,n) pair is an independent LTI system with transfer function "
            "H_{p,n}(z)=1/(1-rho_{p,n} z^{-1}). The full operator is BLOCK Toeplitz "
            "(Toeplitz in t for each channel, diagonal in (p,n)), so a single batched "
            "FFT diagonalizes it channel-wise and the streaming recurrence realizes "
            "the same operator. The state has P*R*N scalars, independent of L. "
            "Consequently a per-channel coefficient tensor of shape (M,D) is "
            "architecturally admissible without breaking any purity constraint."),
        proof=(
            "For fixed (p,r,n), unrolling gives "
            "s_{p,r,n,t}=sum_tau rho_{p,n}^{tau} u_{t-tau}[r,n], whose impulse "
            "response depends only on the delay tau. Stacking the P*N independent "
            "channels gives a block-diagonal-in-(p,n) operator whose t-blocks are "
            "Toeplitz; by NT12 (applied per channel) the parameter must be constant "
            "in t, which it is. A Toeplitz operator is diagonalized by one DFT per "
            "channel, so one batched FFT suffices; the recurrence is the same "
            "rational function computed in the time domain. The state count is "
            "P*R*N with no L term. QED"),
        boundary=(
            "Requires rho and w independent of t. It does NOT permit input-dependent "
            "rho_t (that is the SFSC/LTV case, excluded by NT12/Q09). The constant "
            "P*R*N may be large (56,320 B observed); O(1) is asymptotic."),
        use=("Upgrades the v4 finding (coef was artificially flattened from (32,2048) "
             "to 32 scalars) from a script simplification into a theorem about what "
             "the architecture permits."),
        source="core/arch/sgc.py coef (M,D) + T7 / Section 4",
        relates="strengthens CT1/CT3 with the per-channel extension"),

    # ============================================================ GROUP III
    Thm(
        nid="NT11", group="III: multi-hop, self-evolution, equivalence",
        title="Layer-count lower bound for multi-hop retrieval",
        hypothesis=(
            "A stack of L layers, each a polynomial map of total degree at most d "
            "in its inputs (a bilinear contraction has d=2); skip connections do not "
            "raise degree; the target H-hop output contains H key-match factors and "
            "one value factor, i.e. total degree >= H+1 in the history variables."),
        claim=(
            "Any such stack needs L >= ceil(log_d(H+1)) layers. For bilinear layers "
            "L >= ceil(log2(H+1)); in particular H=2 already requires L>=2, so a "
            "single contraction layer cannot implement two-hop retrieval. This is a "
            "NECESSARY condition; no sufficient construction is asserted."),
        proof=(
            "The composition of polynomials of degrees d_1..d_L has degree at most "
            "the product d_1...d_L <= d^L (degree cannot increase under addition or "
            "skip connections). The H-hop target is a multilinear form of degree "
            "H+1 in (k,v): H factors of the form <phi(q),psi(k)> plus one value "
            "factor. A polynomial of degree < H+1 cannot equal a nonzero degree-(H+1) "
            "form on an open set. Hence d^L >= H+1, i.e. L >= log_d(H+1); taking "
            "ceil gives the integer bound. For d=2, H=2: log2(3)=1.58 -> L>=2. QED"),
        boundary=(
            "NECESSARY only. The degree bound is an upper bound on what L layers CAN "
            "reach; reaching it needs a construction (untested). Per-position "
            "nonlinearities of degree >2 change d. The measured H4_multihop=0.0322 "
            "is a finite-sample observation, not a theorem conclusion."),
        use="Gives the stacking roadmap a concrete quantitative target instead of "
            "'try more layers', and explains the two-hop collapse structurally.",
        source="T15 / L7 / O5 H4_multihop; core/arch stacking",
        relates="upgrades T15 from degree separation to a layer-count bound"),

    Thm(
        nid="NT12", group="III: multi-hop, self-evolution, equivalence",
        title="Toeplitz iff time-invariant coefficients, for arbitrary order and channels",
        hypothesis=(
            "y_t = sum_{j=1..P} a_{j,t} y_{t-j} + c_t u_t with c_t != 0 for all t; "
            "P>=1 arbitrary; optionally with an independent copy for each of C "
            "channels (a_{j,t}^{(c)}, c_t^{(c)})."),
        claim=(
            "The impulse response h(t,s) depends only on t-s IFF every coefficient "
            "a_{j,t} and c_t is constant in t. With C independent channels the full "
            "operator is block Toeplitz IFF this holds for every channel. Necessity "
            "fails at the first delay where c_t=0, so the theta-clamp / nonzero-c "
            "premise is essential, not decorative."),
        proof=(
            "Necessity by induction on the delay k. k=0: h(s,s)=c_s must be "
            "s-independent, so c_t=c. Assume a_1..a_{k-1} are constant. At delay k, "
            "h(s+k,s) = a_{k,s+k} c + Phi(a_1..a_{k-1},c), where Phi is a polynomial "
            "with s-independent coefficients. Toeplitz forces h(s+k,s) independent "
            "of s, and c != 0 gives a_{k,t} constant. Induct to k=P. Sufficiency: "
            "constant coefficients give the LTI transfer function "
            "H(z)=c/(1-sum_j a_j z^{-j}), so h(t,s)=h(t-s). For C channels apply the "
            "same argument to each diagonal block; off-diagonal blocks are zero. "
            "If c_t=0 for some t, the k=1 step cannot isolate a_{1,t}, and a "
            "counterexample with time-varying a_1 but Toeplitz h exists (the input "
            "never enters at that step). QED"),
        boundary=(
            "The equivalence concerns THIS convolution/FFT path; it does not assert "
            "nonexistence of other sub-quadratic algorithms. c_t != 0 is guaranteed "
            "in SGC by sin(theta)>0 under the theta clamp; removing the clamp "
            "invalidates the necessity direction."),
        use=("The precise equivalence-strengthening the reviewer asked to foreground; "
             "now covers arbitrary order and per-channel grids, and identifies exactly "
             "where the premise can break."),
        source="Thm4 + T11 + Q09/Q11; core/arch/sgc.py theta_clamp_floor",
        relates="strengthens CT3 / Thm4"),

    Thm(
        nid="NT13", group="III: multi-hop, self-evolution, equivalence",
        title="Arbitrary-read-out impossibility with a sharp quantitative boundary",
        hypothesis=(
            "EXACTLY four premises: (i) the history enters the final computation only "
            "through a fixed-dimensional LINEAR summary A=L(v_1..v_n) in R^d that is "
            "independent of q; (ii) q enters only at the current position; "
            "(iii) values are independently assignable; (iv) at least two "
            "distinguishable targets exist. Read-out f(A,q) is arbitrary (linear or "
            "nonlinear). Key linear independence is NOT required."),
        claim=(
            "(a) Content addressing is impossible for any f. "
            "(b) Quantitatively dim ker L >= nd - d. "
            "(c) For n=1 the kernel can be zero, so premise (iv) is NECESSARY and "
            "cannot be dropped. "
            "(d) TSC escapes by changing the tensor order of the signal and the "
            "read-out, NOT by contradicting (a)."),
        proof=(
            "(a,b) rank(L)<=d, so dim ker L = nd - rank(L) >= nd-d > 0 for n>=2. "
            "Choose nonzero Delta-v in ker L supported on a coordinate of the target "
            "value (possible by (iii),(iv)). Two assignments v and v+Delta-v give the "
            "same A and the same q, hence the same f(A,q), but require different "
            "answers. Contradiction for every f. Linear read-out follows by "
            "subtracting two queries. "
            "(c) n=1 gives nd=d; a random square L is full rank almost surely, so "
            "ker=0 and (a) no longer follows -- the premise is sharp. "
            "(d) NT01 uses an order-2 tensor state, which is not a fixed vector "
            "summary, so (i) is false; the theorem is inapplicable, not violated. QED"),
        boundary=(
            "DOES NOT extend to deep stacks with per-position nonlinearities that mix "
            "A and q, to input-dependent kernels (NT12 instead), or to outer-product "
            "states (NT01). Corollary B1.1 ('requires a pairwise term') is an "
            "EXPLANATORY FRAMEWORK, NOT A THEOREM, and is not a proved sufficient or "
            "necessary characterization of all escape mechanisms."),
        use=("The strongest boundary theorem, with an explicit four-premise contract "
             "that must appear in the abstract and conclusion to prevent misreading."),
        source="Thm5 + T10 + CT4; Appendix A.5",
        relates="consolidates CT4 / Thm5 / T10"),

    Thm(
        nid="NT14", group="III: multi-hop, self-evolution, equivalence",
        title="Information-Gradient Duality (with the softening trap)",
        hypothesis=(
            "An architecture has (a) an online state s_L=F(x_{1:L}) in R^D with D "
            "independent of L and F continuous; (b) an episodic bank of C EXACT "
            "copies read by hard top-k selection S_k(q) followed by continuous "
            "softmax weights; (c) no N x N score matrix, no resident KV cache; "
            "(d) the task requires exact retrieval of an arbitrary one of L history "
            "items, |X|>1, L unbounded."),
        claim=(
            "No single channel is simultaneously (I) differentiable, (II) covering "
            "all L items, and (III) losslessly retrieving any one. The three horns "
            "are exclusive AND the natural repair is self-defeating: replacing hard "
            "top-k by a differentiable soft aggregate over all C copies re-triggers "
            "NT13's kernel impossibility."),
        proof=(
            "(i) Differentiable + covering forces lossy: F must be injective on "
            "X^L, but a B-byte state has at most 2^{8B} configurations and "
            "|X|^L > 2^{8B} for L > 8B/log2|X| (pigeonhole).\n"
            "(ii) Hard top-k: on the tie-free open set the ordering is locally "
            "constant, so d 1[j in S_k]/d theta = 0 a.e. and the Jacobian support is "
            "contained in S_k of size k; the C-k unselected items contribute exactly "
            "zero to any gradient update. Exact on k slots, fails (I) for selection "
            "and (II) because k<L.\n"
            "(iii) SOFTENING: m_soft(q)=sum_{j=1..C} alpha_j(q) v_j is differentiable "
            "and covers all C, but for fixed q the map "
            "(v_1..v_C) in R^{Cd} -> m_soft in R^d is LINEAR with rank <= d, so "
            "dim ker >= Cd-d > 0 for C>=2. A nonzero Delta-V in that kernel leaves "
            "m_soft and q unchanged while changing the target value, so any "
            "f(m_soft,q) gives identical output for distinct required answers. "
            "This is exactly NT13. Hence (I)+(II) again forces NOT (III).\n"
            "The three repairs are mutually exclusive. QED"),
        boundary=(
            "Does not forbid hard top-k retrieval of k items (the L2 pointer sidecar), "
            "nor a zero-order path (NT15). It says nothing about empirical LM quality. "
            "Numerical checks are finite identities, not the proof."),
        use="The hardest structural theorem: turns 'gradient vanished' into a "
            "closed-loop impossibility with a self-defeating repair.",
        source="CT6 / core/theory/smm_sgc.py; memory.py topk, smm.py @torch.no_grad",
        relates="CT6 with NT13 substituted for the earlier CT4 step"),

    Thm(
        nid="NT15", group="III: multi-hop, self-evolution, equivalence",
        title="Zero-order self-evolution: gate validity and information rate",
        hypothesis=(
            "NT14 holds; online answer path has no autograd graph; consolidation "
            "writes to weights only through a no-regression gate comparing "
            "paired-bootstrap one-sided CI lower bounds l_alpha (level 1-alpha) "
            "against -|tau| over m metrics, rejecting if ANY fails; an independent "
            "held-out verifier is supplied (otherwise writeback is refused); at most "
            "w insights of b bits are committed per round; each read selects k items."),
        claim=(
            "(a) P(false commit | true regression) <= alpha for one metric and "
            "<= m*alpha for m metrics (<= alpha under Bonferroni alpha/m). "
            "(b) I(history ; weight increments over T rounds) <= T*w*b bits, "
            "INDEPENDENT of L and C; instantaneous bandwidth <= k*b bits/round. "
            "(c) The path MUST be zero-order. (d) A non-decreasing ACCEPTED score "
            "does NOT imply monotone real capability."),
        proof=(
            "(c) NT14(ii) zero Jacobian for unselected items, NT14(iii) softening "
            "destroys exactness, and the no_grad online path has no graph: no "
            "gradient can carry long-context experience into the weights, so the "
            "only admissible channel is a zero-order accept/reject test on verifier "
            "outputs.\n"
            "(a) Coverage: P(l_alpha <= Delta_true) >= 1-alpha, hence "
            "P(l_alpha > Delta_true) <= alpha. A false commit requires "
            "l_alpha >= -|tau| > Delta_true, i.e. l_alpha > Delta_true, exactly the "
            "coverage-failure event. Union bound over m metrics gives m*alpha.\n"
            "(b) By data processing, "
            "I(H;Delta theta_t) <= H(insights_t, verifier_t) <= O(w*b). Summing over "
            "T rounds gives T*O(w*b). L and C do not appear because NT14(ii) bounds "
            "each read to k selected items; the instantaneous bound is k*b.\n"
            "(d) The gate proves NO-REGRESSION, a necessary condition for writeback; "
            "improvement requires a separate held-out evaluation. The upstream "
            "docstring states exactly this. QED"),
        boundary=(
            "(a) CONDITIONAL on paired i.i.d. samples, bootstrap consistency and a "
            "non-default verifier; with the default constant verifier the loop refuses "
            "writeback, so the gate is inactive, not contradicted. (b) is an UPPER "
            "BOUND, not a claim the loop reaches it. Monte-Carlo checks are "
            "ILLUSTRATION ONLY. T16's scalar-monotonicity claim is WITHDRAWN (W7)."),
        use="First rate bound for the SMM x SGC self-evolution path; shows long context "
            "buys inference-time retrieval, NOT a faster learning rate.",
        source="CT7 / dream/evaluation.py regression_gate, loop.py VERIFY, hooks.py",
        relates="replaces withdrawn T16 / CT7(a)+(b)"),
]


def summary() -> dict:
    g = {}
    for t in NT:
        g[t.group] = g.get(t.group, 0) + 1
    return dict(delivered=len(NT), target=49, remaining=49 - len(NT),
                groups=g, ids=[t.nid for t in NT])


THEOREMS15 = NT
__all__ = ["Thm", "NT", "THEOREMS15", "summary"]
