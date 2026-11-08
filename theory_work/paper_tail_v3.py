# ------------------------------------------------------------------ content
# v3: 15 new theorems (first echelon of the 49-theorem roadmap), de-duplicated
# into 3 groups, with all five reviewer corrections applied.
import sys as _sys
_sys.path.insert(0, "/data/workspace")
try:
    from simplcore.core.theory.new_theorems15 import NT as THEOREMS
except ImportError:
    from theory_work.new_theorems15 import NT as THEOREMS


def _short_proof(t):
    """One-sentence proof route for the main paper; full proof in Appendix A."""
    first = t.proof.strip().split("\n")[0]
    return first[:300]


def build():
    d = Doc(ROOT + "/simplcore.pdf")
    P = d.para
    Hh = d.heading
    Fm = d.formula
    It = d.item

    # ---------------------------------------------------------------- abstract
    Hh("Abstract", size=10.0, before=0.0, after=2.0)
    P("Constant-state attention-free models replace content-dependent pairwise scoring "
      "with a fixed-size state. LTI convolution/recurrence duality, strict causality, "
      "O(1) state, and the recall weakness of fixed-state models are already "
      "established in the SSM / linear-attention / Zoology literature; we do not "
      "claim otherwise. This paper contributes <b>15 precise theorems grown from the "
      "SimplCore operator itself</b>, organized in three de-duplicated groups, and "
      "foregrounds the genuinely new ones: a tight minimum tensor order of 2 for exact "
      "retrieval; a strict-positivity separation between real-pole and conjugate-pole "
      "dictionaries; a provable separation of the rho&lt;1 multi-pole kernel from the "
      "linear-attention constant kernel; 2-optimal deterministic codes for arbitrary "
      "vocabulary size with a matching Omega(V*N) state bound; a quantitative "
      "condition-number rate for near-zero frequency grids; a precision-length-decay "
      "trilemma with an <i>interior</i> optimal decay; a rounding-order crossover "
      "between the dual execution paths; a layer-count lower bound for multi-hop "
      "retrieval; and an information-gradient duality whose most natural repair "
      "re-triggers our own impossibility theorem. We strengthen the Toeplitz "
      "equivalence to arbitrary order and per-channel grids, and give the "
      "arbitrary-read-out impossibility theorem with a sharp quantitative boundary. "
      "<b>Theorem NT13 has exactly four premises</b> -- fixed-dimensional linear "
      "history summary, query only at the current position, independently assignable "
      "values, and at least two distinguishable targets -- and Tensor Spectral "
      "Convolution escapes it by changing the <i>tensor order of the signal</i>, not "
      "by contradicting it. <b>Corollary B1.1 is an explanatory framework, not a "
      "theorem</b>; it is not a proved necessary-and-sufficient characterization. "
      "We further <b>withdraw</b> the earlier mutual-exclusivity claim between content "
      "addressing and FFT duality as a core contribution. On one synthetic single-hop "
      "task we did <b>not detect a significant difference</b> from a stored windowed "
      "attention record (1.0000 vs 0.9987, Fisher exact "
      f"p = {FP:.4f}); this is <b>not an equivalence proof</b>, the baseline is not a "
      "same-run global attention model, and no throughput comparison exists. All "
      "numbers are finite-sample observations; none is promoted to a theorem. "
      ">=10M-token equal-compute language modeling, multi-hop stacking and "
      "same-run baselines remain unperformed.",
      size=SMALL, lead=SLEAD, after=5.0)

    # ---------------------------------------------------------------- 1 intro
    Hh("1  Introduction")
    P("A constant inference state must summarize an unbounded history. The literature "
      "has established the LTI duality, the time-invariance requirement for a "
      "convolution view, and the recall/copying weakness of fixed-state models "
      "(arXiv:2111.00396, 2208.04933, 2312.04927, 2402.01032, 2404.08819). What has "
      "been less precise is <i>which algebraic property of a specific architecture</i> "
      "produces each boundary, and <i>which single premise</i> must be removed to "
      "escape it. We answer both for Spectral Global Convolution (SGC) and its tensor "
      "extension, with 15 de-duplicated theorems.")
    P("<b>Evidence discipline.</b> A theorem is a universal statement derived only "
      "from definitions. A finite-sample observation is a measured value tied to a "
      "configuration, seed count, precision and task. A conjecture is explicitly "
      "unverified. No measured number is upgraded to a theorem; no theorem is inferred "
      "from a measured number. Conditional theorems state their extra assumption inline.")
    P("<b>Theorem organization (de-duplication).</b> We deliberately avoid theorem "
      "inflation: standard facts (geometric-series BIBO, Sylvester orthogonality, "
      "rank-nullity, scalar-score induction) are used as <i>proof steps or lemmas</i>, "
      "not given theorem slots. The 15 theorems fall into three groups. "
      "<b>Group I (NT01-NT05)</b> is the TSC construction and its tight optimality. "
      "<b>Group II (NT06-NT10)</b> is the quantitative grid / precision / length "
      "frontier. <b>Group III (NT11-NT15)</b> is multi-hop, self-evolution and the "
      "equivalence characterizations. Eleven statements are new precise propositions; "
      "NT10, NT12, NT13 and NT15 strengthen results established in earlier versions "
      "of this project. Withdrawn and falsified statements are registered in "
      "Appendices C and F rather than recycled as theorems.")
    P("<b>Positioning.</b> This is a <i>boundary-characterization paper with a "
      "constructive scope correction and auditable negative results</i>. It is not a "
      "new SOTA sequence model, does not claim general language-model parity or "
      "superiority over attention, and does not claim a formal upper bound on all "
      "constant-state models.")

    # ---------------------------------------------------------------- 2 prelim
    Hh("2  Preliminaries and Notation")
    P("Let x_{1:L} be a causal token sequence. We use <b>t</b> for the time index, "
      "<b>tau</b> for a kernel lag and <b>d = t - s</b> for a delay in an impulse "
      "response; these are kept distinct throughout. SGC uses the damped-oscillator "
      "kernel K(tau) = rho^tau sin((tau+1)theta) with 0 &lt; rho &lt; 1 and theta in "
      "(0, pi), or the general second-order real-pole kernel "
      "h(tau) = sum_{j=0}^{tau} r1^{tau-j} r2^{j}. The architecture exposes an "
      "FFT whole-sequence path and a streaming two-tap IIR path. A content-addressing "
      "task supplies n key-value pairs and a query q equal to one key; the answer is "
      "the associated value. A is the part of the final state that depends on the "
      "history but not on q.")
    P("The state register is finite and its size is independent of L (an asymptotic "
      "statement with a possibly large constant). Theorem statements are analytic; "
      "numerical values appear only as finite-sample observations with provenance in "
      "Appendix D. Complete proofs of all 15 theorems are in Appendix A.")

    # ---------------------------------------------------------------- 3 theorems
    Hh("3  Fifteen Theorems")
    P("Each theorem below states its hypotheses, its universal claim and the proof "
      "route; the full derivation is in Appendix A. Machine-checkable finite "
      "identities in the supplementary artifact verify that each statement is "
      "correctly formed and has a non-empty premise -- they are <b>not</b> the proofs.",
      size=SMALL, lead=SLEAD, after=4.0)

    group_label = {}
    for t in THEOREMS:
        if t.group not in group_label:
            group_label[t.group] = True
            Hh("3.%d  %s" % (len(group_label), t.group), size=9.6,
               before=6.0, after=2.5)
        d.sub("%s  %s." % (t.nid, t.title))
        P("<b>Premises.</b> " + t.hypothesis, size=SMALL, lead=SLEAD, after=1.5)
        P("<b>Claim.</b> " + t.claim, size=SMALL, lead=SLEAD, after=1.5)
        P("<i>Proof route.</i> " + _short_proof(t) + " Full proof: Appendix A.%d." %
          int(t.nid[2:]), size=SMALL, lead=SLEAD, after=1.5)
        P("<i>Boundary / use.</i> " + t.boundary + " " + t.use,
          size=SMALL, lead=SLEAD, after=4.0)

    # ---------------------------------------------------------------- 4 TSC
    Hh("4  Constructive Scope Correction: TSC")
    P("SGC constrains its <i>kernel</i> to be a static LTI operator; it does not "
      "constrain the <i>tensor order of its input signal</i>. Replacing the vector "
      "signal by an outer product and the read-out by a contraction yields Tensor "
      "Spectral Convolution:")
    Fm(r"u_t[r,n]=a_t[r]\,b_t[n],\quad s_{p,r,n,t}=\rho_{p,n}s_{p,r,n,t-1}+u_t[r,n]")
    Fm(r"z_n=\sum_{p,r}w_{p,n}\,q_r\,s_{p,r,n,L}"
       r"=\sum_{\tau}(q\cdot k_{\tau})K_n(L{-}1{-}\tau)v_{\tau,n}")
    P("NT01 proves that order 2 is both necessary and sufficient for exact, "
      "position-free retrieval under a strictly positive kernel; NT02 shows that "
      "strict positivity holds exactly for the real-pole dictionary and fails for "
      "the conjugate-pole dictionary; NT10 proves that per-channel rho_{p,n} keeps "
      "the operator block Toeplitz, so a coefficient tensor of shape (M,D) is "
      "architecturally admissible. NT04/NT05 give a deterministic code for every "
      "vocabulary size with R &lt; 2V and a matching Omega(V*N) state bound; NT08 "
      "makes retrieval robust to bounded code perturbation.")
    P("<b>Relation to linear attention (a provable separation, not a judgement left "
      "open).</b> At rho -&gt; 1 with one pole the state sum_tau k_tau v_tau^T is "
      "isomorphic to a linear-attention state; we record this openly. NT03 proves "
      "that for rho &lt; 1 with multiple poles the kernel family contains no nonzero "
      "constant function and has l^infty distance exactly 1 from it, so the state "
      "update operator is <b>not</b> the linear-attention operator. We therefore "
      "position TSC as <i>a construction that realizes content addressing under a "
      "static LTI kernel with a tensor state and a contraction read-out</i>, not as "
      "a claim of having left the linear-attention paradigm entirely. The "
      "construction has no pairwise scoring matrix, no softmax normalization and no "
      "growing KV cache.")
    P("<b>Cost.</b> The state holds P*R*N scalars (56,320 B observed versus 512 B for "
      "the original mixer); O(1) is asymptotic. A coarse analytic crossover suggests "
      "no compute advantage below L ~ 880; this has not been measured against an "
      "attention implementation.")

    # ---------------------------------------------------------------- 5 obs
    Hh("5  Finite-Sample Observations (Not Theorems)")
    P("Each item is a finite-sample estimate under a fixed configuration; none is a "
      "theorem, none claims universality. Precision, seed count and provenance are "
      "in Appendix D.", size=SMALL, lead=SLEAD, after=4.0)
    d.sub("O1 - saturated transfer ceiling.")
    P("In a 1.7B weight-transfer fitting study the 28-layer residual-stream cosine "
      "reached 0.2484 against a 0.2476 static-kernel oracle; a same-input single-layer "
      "control reached 0.5430. All alignment metrics improved while generation stayed "
      "non-sentential. The oracle is a finite-layer estimator, not a proved bound.")
    d.sub("O2 - phase-resonance content addressing.")
    P("A fixed-frequency-slot variant raised single-hop accuracy from 0.4160 to "
      "0.8672; an equal-parameter low-rank attention baseline reached 1.0000 "
      "(5,217 vs 5,152 parameters).")
    d.sub("O3 - native pilot training.")
    P("A 470K-parameter native model on 3.07M tokens reached validation perplexity "
      "10.817 against a same-budget Transformer's 11.281, and 10.217 with "
      "output-distribution distillation. Generation was non-sentential on all arms.")
    d.sub("O4 - non-replication of the periodic-axis gain.")
    P("An external white-box run reported 0.9844 vs 0.8535; an in-repository probe "
      "with seven grids produced 0.1094-0.1354, all below the copy-last baseline "
      "0.1575, with a paired CI crossing zero. The advantage is not established.")
    d.sub("O5 - TSC synthetic single-hop result (no significant difference detected).")
    P(f"On a single-hop synthetic binding task (V = 64, L = 9, held-out 512 per "
      f"seed), W1 reached {acc('W1_code64_lr'):.4f} over {nseed('W1_code64_lr')} seeds "
      f"at {par('W1_code64_lr')} parameters, against {SCALE:.4f} for a <b>stored</b> "
      f"windowed softmax-attention record at {FUS['UP_softmax'][0]['params']} "
      f"parameters (Fisher exact p = {FP:.4f}). <b>No significant difference was "
      f"detected in this configuration; p &gt; 0.05 is not an equivalence proof</b>, "
      f"the baseline is not a same-run global attention model, and no throughput "
      f"comparison exists. W4_half reached 1.0000 with {par('W4_half')} parameters "
      f"({100*W4R:.1f}% of the reference) over {nseed('W4_half')} seeds; M5 reached "
      f"1.0000 with {par('M5_d8n30')} parameters "
      f"({10240/par('M5_d8n30'):.2f}x efficiency). These are parameter-efficiency "
      f"observations on ONE synthetic task only. An analytic evaluation stayed at "
      f"1.0000 up to L = 1025 ({len(LENS)} lengths tested); a value-permutation "
      f"control stayed at 1.0000 and removing the predecessor-key encoding collapsed "
      f"accuracy to {acc('T7_noK'):.4f}. A random orthogonal code also reached 1.0000 "
      f"(NT08 explains why). A two-hop variant collapsed to "
      f"{acc('H4_multihop'):.4f} (chance); NT11 gives the layer lower bound that "
      f"explains it.")
    d.sub("O6 - physical Pareto frontier.")
    P(f"fp64/fp32 stayed at 1.0000 through L = 4097; bf16 and int8 first dropped at "
      f"L = 2049 ({PP['phase']['bf16']['L2049']:.4f}); fp16 dropped at L = 4097 "
      f"({PP['phase']['fp16']['L4097']:.4f}). The decay coefficient is the effective "
      f"lever: fp32 with rho = 0.999 (BIBO-stable) stayed at 1.0000 to L = 16,385. "
      f"NT07 proves that the optimal rho is <i>interior</i>, so rho -&gt; 1 is not "
      f"the right target; NT09 gives the rounding-order crossover. Measured "
      f"reference-implementation cost stayed near "
      f"{PT['lengths']['L4097']['us_per_token']:.2f} us/token with a constant "
      f"{PT['state_bytes_fp32']} B state; <b>only the scaling is informative, not "
      f"the absolute rate</b>. Conservation checks passed "
      f"{CHK['passed']}/{CHK['total']}.")

    # ---------------------------------------------------------------- 6 limits
    Hh("6  Limitations and Missing Evidence")
    It("<b>Three decisive experiments are missing.</b> (i) Same-run global/windowed "
       "attention re-run plus measured throughput; (ii) >=10M-token equal-compute "
       "language modeling against Mamba-2 / GLA / DeltaNet / Hyena / Transformer; "
       "(iii) TSC multi-layer stacking for multi-hop retrieval. Without these the "
       "TSC positive result is a concept validation only. O3 is a 3.07M-token pilot.")
    It("Not every observation has a paired CI over >=5 seeds: O5 and O6 do; the "
       "transfer, phase-resonance, pilot and non-replication observations do not.")
    It("All positive TSC results are on one synthetic single-hop task with V = 64 and "
       "one layer. Length extrapolation beyond L = 129 permits repeated keys, so it "
       "is an extrapolation test, not an in-distribution test.")
    It("O5 compares against a stored windowed-attention record; a full global "
       "attention baseline and a measured throughput comparison are absent. "
       "No significant difference detected is not equivalence, and parity on one "
       "synthetic task is not general-capability parity.")
    It("<b>NT13 has exactly four premises</b> and is not extended to deep stacks with "
       "per-position nonlinearities; <b>Corollary B1.1 is an explanatory framework, "
       "not a theorem</b>, and is not a proved sufficient/necessary characterization.")
    It("int8 remains a hard long-length boundary; rho = 1 is only critically stable "
       "and diverges on unbounded streams. NT11 supplies a necessary layer bound, "
       "not a multi-hop construction.")
    It("NT03 separates kernel families / state-update operators; it does not claim "
       "full-model expressivity separation from every gated linear-attention variant.")

    # ---------------------------------------------------------------- 7 related
    Hh("7  Related Work")
    P("Structured state-space and long-convolution models (arXiv:2111.00396, "
      "2208.04933, 2302.10866) established the LTI convolution/recurrence duality and "
      "the time-invariance requirement. Selective and linear-attention state models "
      "(arXiv:2312.00752, 2405.21060, 2312.06635, 2406.06484, 2006.16236, "
      "2305.13048, 2407.04620) obtain content addressing through input-dependent "
      "updates at the cost of the LTI convolution view. Zoology and related work "
      "(arXiv:2312.04927, 2402.01032, 2404.08819) already established that fixed-state "
      "models are weaker on recall and copying.")
    P("<b>What is new here, precisely.</b> We do <b>not</b> claim the six structural "
      "facts above as new. The additions are: the tight order-2 / Omega(V*N) / "
      "2-optimal-code construction (NT01, NT04, NT05); the strict-positivity "
      "separation of pole families and the provable separation from the linear-"
      "attention constant kernel (NT02, NT03); quantitative rates for grid "
      "conditioning and for the precision-length-decay trilemma (NT06, NT07, NT09); "
      "the robustness bound (NT08); the per-channel block-Toeplitz and arbitrary-order "
      "equivalence (NT10, NT12); the arbitrary-read-out impossibility with a sharp "
      "quantitative boundary (NT13); the information-gradient duality and the "
      "zero-order rate bound (NT14, NT15); and the layer lower bound (NT11). "
      "The comparison remains qualitative and bibliographic: equal-parameter and "
      "equal-compute quantitative comparisons have not been performed.")

    # ---------------------------------------------------------------- 8 concl
    Hh("8  Conclusion")
    P("Fifteen de-duplicated theorems characterize what a constant-state spectral "
      "model can express. The genuinely new precise propositions are the tight "
      "order-2 construction with a matching state bound, the pole-family and "
      "linear-attention separations, the quantitative grid / precision / length "
      "rates, the interior-optimal decay, the layer lower bound, and the "
      "information-gradient duality. <b>The arbitrary-read-out impossibility "
      "(NT13) holds only under its four stated premises</b>; TSC evades it by "
      "changing the tensor order of the signal, not by disproving it. "
      "<b>Corollary B1.1 is not a theorem</b> and must not be read as a proved "
      "necessary-and-sufficient condition. <b>Withdrawing the mutual-exclusivity "
      "claim between content addressing and FFT duality is a core contribution</b>, "
      "as is the withdrawal register in Appendix F. All numbers are finite-sample "
      "observations; 'no significant difference detected' is not equivalence. "
      ">=10M-token equal-compute language modeling, same-run attention baselines, "
      "multi-hop stacking and throughput comparisons remain open work, not "
      "conclusions of this paper.")

    d.end_main()
    return d


def back(d):
    P = d.para
    Hh = d.heading
    It = d.item

    # ---------------------------------------------------------------- refs
    Hh("References", size=11.5, before=0.0, after=5.0)
    P("Entries are keyed by arXiv identifier with verified title/venue.",
      size=SMALL, lead=SLEAD, after=5.0)
    refs = [
        ("arXiv:2111.00396", "Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022."),
        ("arXiv:2208.04933", "Simplified State Space Layers for Sequence Modeling. 2022."),
        ("arXiv:2302.10866", "Hyena Hierarchy: Towards Larger Convolutional Language Models. 2023."),
        ("arXiv:2312.00752", "Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM 2024."),
        ("arXiv:2405.21060", "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024."),
        ("arXiv:2312.06635", "Gated Linear Attention Transformers with Hardware-Efficient Training. 2023."),
        ("arXiv:2406.06484", "Parallelizing Linear Transformers with the Delta Rule over Sequence Length. 2024."),
        ("arXiv:2006.16236", "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020."),
        ("arXiv:2305.13048", "RWKV: Reinventing RNNs for the Transformer Era. EMNLP 2023."),
        ("arXiv:2407.04620", "Learning to (Learn at Test Time): RNNs with Expressive Hidden States. 2024."),
        ("arXiv:2307.02486", "LongNet: Scaling Transformers to 1,000,000,000 Tokens. 2023."),
        ("arXiv:2403.19887", "Jamba: A Hybrid Transformer-Mamba Language Model. 2024."),
        ("arXiv:2312.04927", "Zoology: Measuring and Improving Recall in Efficient Language Models. ICLR 2024."),
        ("arXiv:2402.01032", "Repeat After Me: Transformers are Better than State Space Models at Copying. 2024."),
        ("arXiv:2404.08819", "The Illusion of State in State-Space Models. ICML 2024."),
    ]
    for i, (k, v) in enumerate(refs, 1):
        P("[%d] %s - %s" % (i, k, v), size=SMALL, lead=SLEAD, after=1.5)

    # ---------------------------------------------------------------- impact
    d.newpage()
    Hh("Broader Impact", size=11.5, before=0.0, after=5.0)
    P("This work is theoretical and empirical on synthetic tasks. It involves no human "
      "subjects, personal data, deployed systems or dual-use capabilities. Its positive "
      "potential is a clearer account of when constant-state models suffice, which can "
      "reduce long-context inference cost and discourage over-engineering "
      "attention-free architectures for tasks that structurally require content "
      "addressing. Its main risk is interpretive: a reader may take 'no significant "
      "difference detected' on one synthetic binding task as general language-model "
      "parity, or read 15 theorems as 15 independent new discoveries. We have therefore "
      "stated every premise, de-duplicated the theorem set into three groups, listed "
      "the missing large-scale experiments, and declined every superiority claim. A "
      "second risk is dismissing an entire architecture family from a negative result; "
      "NT01 and Section 4 argue specifically against that reading. Generative AI tools "
      "were used for implementation assistance and language editing; all research "
      "questions, conjectures and conclusions were specified by the human author, who "
      "takes full responsibility.")

    # ---------------------------------------------------------------- appendix A
    d.newpage()
    Hh("Appendix A  Complete Proofs of NT01-NT15", size=11.5, before=0.0, after=5.0)
    P("All 15 proofs are analytic. The finite-identity checks in the supplementary "
      "artifact verify statement formation and non-empty premises only; they are not "
      "part of any proof. Symbol convention: t = time index, tau = lag, d = t - s.",
      size=SMALL, lead=SLEAD, after=5.0)
    for t in THEOREMS:
        d.sub("A.%d  %s - %s" % (int(t.nid[2:]), t.nid, t.title))
        P("<b>Premises.</b> " + t.hypothesis, size=SMALL, lead=SLEAD, after=1.5)
        P("<b>Claim.</b> " + t.claim, size=SMALL, lead=SLEAD, after=1.5)
        for chunk in t.proof.split("\n"):
            if chunk.strip():
                P(chunk.strip(), size=SMALL, lead=SLEAD, after=1.5)
        P("<b>Boundary.</b> " + t.boundary, size=SMALL, lead=SLEAD, after=1.5)
        P("<b>Architectural use.</b> " + t.use + " <i>Source:</i> " + t.source,
          size=SMALL, lead=SLEAD, after=4.0)

    # ---------------------------------------------------------------- appendix B/C
    d.newpage()
    Hh("Appendix B  Conjectures and Open Questions", size=11.5, before=0.0, after=5.0)
    P("Explicitly unverified. None is a premise of any theorem or a claimed result. "
      "NT04 closes C2; NT11 gives C1/C3 a necessary condition but no construction.",
      size=SMALL, lead=SLEAD, after=5.0)
    cj = [
        ("C1", "Stacking TSC layers or inter-layer recurrence restores multi-hop "
               "retrieval. NT11 gives the layer lower bound; no stacking experiment exists."),
        ("C2", "CLOSED by NT04: a deterministic orthogonal code exists for every V "
               "with R < 2V."),
        ("C3", "NT13 extends to deep stacks with per-position nonlinearities. "
               "Cross-terms between A and q leave the fixed-summary premise; unproved."),
        ("C4", "Freezing rho = 0.999 from the start of training remains stable on "
               ">=10M-token language modeling. Verified only on a synthetic task."),
        ("C5", "Layered quantization or residual accumulation pushes the int8 "
               "long-length boundary further. NT07 bounds the trade-off; untested schemes remain."),
        ("C6", "On equal compute the periodic and content-addressing axes are "
               "complementary. The periodic-axis gain was NOT replicated (O4)."),
        ("C7", "CLOSED IN PART by NT03: the rho<1 multi-pole kernel is provably not "
               "the linear-attention constant kernel. Full-model separation from all "
               "gated variants remains open."),
        ("C8", "The decay spectrum gives an inductive bias on oscillatory/damped "
               "physical signals that attention lacks. No measured comparison exists."),
    ]
    for k, v in cj:
        It("<b>%s.</b> %s" % (k, v), size=SMALL, lead=SLEAD, after=2.5)

    Hh("Appendix C  Falsified Statements", size=11.5, before=8.0, after=5.0)
    fal = [
        ("F1", "Increasing rho_max extends the effective memory window. Rejected by measurement."),
        ("F2", "State-feedback rho_t/theta_t modulation (SFSC) preserves FFT duality. "
               "Rejected: LTV is non-Toeplitz (NT12); FFT-proxy relative error 0.9720."),
        ("F3", "Phase-resonance slots exceed equal-parameter attention. Rejected (O2)."),
        ("F4", "Content addressing and FFT/IIR duality are mutually exclusive. "
               "WITHDRAWN via NT01/NT10; this withdrawal is a core contribution."),
        ("F5", "An oscillatory basis itself provides the inductive advantage. "
               "Rejected: a random orthogonal basis reached 1.0000; NT02/NT08 explain why."),
        ("F6", "Dynamic quantization scaling causes spurious non-monotonicity. "
               "Rejected by same-sample prefix controls."),
        ("F7", "Shannon capacity limits long-length retrieval here. Rejected: 176x headroom; "
               "NT07 identifies the decay/precision trade-off instead."),
        ("F8", "A monotone verifier score implies monotone self-improvement (former T16). "
               "WITHDRAWN (W7); trivial scalar induction and inconsistent with the "
               "multi-metric paired-bootstrap gate. Replaced by NT15."),
    ]
    for k, v in fal:
        It("<b>%s.</b> %s" % (k, v), size=SMALL, lead=SLEAD, after=2.5)

    # ---------------------------------------------------------------- appendix D/E/F
    d.newpage()
    Hh("Appendix D  Observation Provenance", size=11.5, before=0.0, after=5.0)
    P("All values were computed from raw result files accompanying the anonymized "
      "supplementary artifact; none was transcribed from a secondary summary.",
      size=SMALL, lead=SLEAD, after=5.0)
    d.table([
        ["Obs.", "Config / seeds", "Key values", "File"],
        ["O1", "1.7B transfer, 1 run", "0.2484 / 0.2476 / 0.5430", "transfer study"],
        ["O2", "single-hop, 3 seeds", "0.8672 vs 1.0000", "phase-resonance probe"],
        ["O3", "470K, 3.07M tokens", "10.817 / 11.281 / 10.217", "native pilot"],
        ["O4", "7 grids, 1 run", "0.1094-0.1354 vs 0.1575", "periodic-axis probe"],
        ["O5", "V=64, 5 seeds, 512/seed", "1.0000 / 0.9987 / p=0.4998", "TSC probe"],
        ["O6", "5 seeds, 4 precisions", "L=16385, 43/43 checks", "physics probes"],
    ], [34, 96, 88, 76], size=7.6, lead=9.4)
    P("Measured configuration details. O5: W1 has %d parameters, W4_half %d, M5 %d; "
      "the stored reference record has %d. <b>Fisher p = %.4f means no significant "
      "difference was detected, NOT equivalence.</b> The paired bootstrap on the "
      "general second-order dictionary gave mean difference %.4f with 95%% CI "
      "[%.4f, %.4f] over 3 seeds (sample-internal variance only). O6: frozen-rho "
      "retraining held-out means %s, between-seed SDs %s; L = 16,385 accuracy mean "
      "%.4f, min %.4f; per-token cost at L = 4097 was %.2f us with a %d B state. "
      "Conservation guards reported FFT/IIR relative errors of order 1e-16." % (
          par("W1_code64_lr"), par("W4_half"), par("M5_d8n30"),
          FUS["UP_softmax"][0]["params"], FP, BOOT["mean_diff"],
          BOOT["ci95"][0], BOOT["ci95"][1],
          str([round(x, 4) for x in G5["heldout_mean"]]),
          str([round(x, 4) for x in G5["heldout_std"]]),
          G5["L16385_mean"], G5["L16385_min"],
          PT["lengths"]["L4097"]["us_per_token"], PT["state_bytes_fp32"]),
      size=SMALL, lead=SLEAD)

    Hh("Appendix E  Reproducibility and Tool Use", size=11.5, before=8.0, after=5.0)
    P("Every theorem is derived analytically in Appendix A and requires no execution. "
      "Every observation comes from a deterministic script with recorded seeds, "
      "precision and sample counts. The supplementary artifact contains raw result "
      "files, gradient finite-difference checks, conservation assertions, the 15 "
      "finite-identity checks (15/15 PASS) and the theorem ledger. Gradient checks "
      "passed 8/8, 10/10, 28/28 and 44/44; the physical consistency checker passed "
      "%d/%d. No training run is used to support any theorem." % (CHK["passed"], CHK["total"]),
      size=SMALL, lead=SLEAD)
    P("Generative AI tools were used for implementation assistance, experiment "
      "scripting and language editing. They did not originate the research questions, "
      "conjectures or conclusions; no experimental data, citation or result was "
      "generated by such a tool. The human author verified every numerical claim "
      "against raw files and accepts full responsibility.", size=SMALL, lead=SLEAD)

    Hh("Appendix F  Withdrawn Claims Register", size=11.5, before=8.0, after=5.0)
    P("Earlier working-version statements, formally withdrawn so the revision is "
      "auditable. Withdrawals are contributions, not hidden errata.",
      size=SMALL, lead=SLEAD, after=5.0)
    d.table([
        ["Withdrawn statement", "Reason"],
        ["Unconditional fp32 bitwise duality", "Different rounding orders; real-field only (NT09)"],
        ["Content addressing <-> duality exclusive", "WITHDRAWN, core contribution: NT01/NT10 counterexample"],
        ["Periodic-axis superiority over attention", "Not replicated in-repository (O4)"],
        ["Equivalent 4-7B parameter claim", "No supporting measurement"],
        ["NT13 holds for any read-out unrestricted", "Exactly four premises required; stated in abstract"],
        ["Dynamic scaling causes spurious effects", "Same-sample prefix controls (O6)"],
        ["W7: monotone score => self-improvement (T16)", "Trivial induction + mismatched source gate; replaced by NT15"],
    ], [130, 108], size=7.6, lead=9.4)

    return d


if __name__ == "__main__":
    d = build()
    back(d)
    total, mainp = d.finish()
    print("PDF written:", ROOT + "/simplcore.pdf")
    print("total pages:", total, "| main-paper ends on page:", mainp)
    if mainp > 8:
        print("WARNING: main paper exceeds the 8-page limit")
    else:
        print("main paper within the 8-page limit")
