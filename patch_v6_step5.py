p = "build_simplcore_pdf_v5.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, label, n=1):
    global s
    assert s.count(old) == n, f"{label}: count={s.count(old)}"
    s = s.replace(old, new)


anchor4 = '''    # ------------------------------------------------------------ 4 CONSTRUCTIVE'''
sec39 = '''    Hh("3.9  Non-vacuity and explicit falsification conditions",
       size=9.4, before=5.0, after=2.0)
    P("A theorem with an empty premise set is worthless. Table 5 gives, for "
       "each core result, one concrete instance satisfying ALL premises and "
       "one observation that would FALSIFY it. These are the conditions we "
       "invite the community to attack.", size=SMALL, lead=SLEAD, after=2.0)
    nv = [["result", "satisfying instance", "falsified if"]]
    for a, b, c in [
        ("T1 order=2", "V=64, n=4, outer-product state",
         "an order-1 state retrieves 2 targets"),
        ("T2 H layers", "V=64, H=2, H_L=2, rho=(.9,.999)",
         "H_L=1 solves H=2, or H_L=3 does"),
        ("T3 O(1) state", "L in {9,65,1025,4097}, 512 B/layer",
         "state bytes grow with L"),
        ("T4 Toeplitz iff", "a,b,c constant; c != 0",
         "time-varying a_t with a Toeplitz h"),
        ("T5 no read-out", "n=2, linear summary A, any f(A,q)",
         "two value assignments differ but A same"),
        ("T6 info bound", "|S| = 2^(8B), unbounded L",
         "injective x -> s_L for all L"),
        ("R3 one bit", "flat alphabet, perm [1,0,2], L=6",
         "ANY interface is correct on all perms"),
        ("R2 sufficiency", "codebook halves, zero training",
         "any pair perm/filler gives < 1"),
    ]:
        nv.append([a, b, c])
    d.table(nv, [66, 92, 86], size=6.6, lead=8.0, after=2.5)
    P("<b>How the checks are scoped.</b> Our verifiers certify statement "
      "formation, non-vacuity, and agreement with the recorded witnesses; "
      "they are NOT the proofs. Every proof above is analytic, and every "
      "measured number is labeled as a finite-sample estimate with its "
      "seed count, precision, task and sample size. A '%d/%d PASS' string "
      "never appears in the abstract or the conclusion." %
      (OB["n_pass"], OB["n_checks"]), size=SMALL, lead=SLEAD, after=3.0)

'''
rep(anchor4, sec39 + anchor4, "sec3.9")

anchor5 = '''    # ------------------------------------------------------------ 5 NOT ESTABLISH'''
sec410 = '''    Hh("4.10  Reproducing the final TSC-H layer",
       size=9.4, before=5.0, after=2.0)
    P("<b>Reference implementation (pure NumPy, no GPU).</b> Given the code "
      "C as a fast Walsh-Hadamard transform, the whole layer is: "
      "(1) encode KEY id k as C[k] and VALUE id k as C[V_K+k], filler as 0; "
      "(2) A_t = code(x_(t-1)), b_t = code(x_t); "
      "(3) K_h(d) = sum_p w_hp rho_hp^d with P=2 poles; "
      "(4) for h = 1..H_L compute m_h[t] = <q_h,A_t>, "
      "z_h = sum_t K_h(L-1-t) m_h[t] b_t, and "
      "q_(h+1) = normalize(T_h z_h); "
      "(5) logits_j = tau <z_(H_L), C[V_K+j]>; "
      "(6) train T_h = U_h W_h^T/sqrt(r) and tau by Adam on cross-entropy "
      "with the analytic gradient of Section 4.6. No attention, no softmax "
      "over tokens, no KV cache, no learned codebook.",
       size=SMALL, lead=SLEAD, after=2.0)
    P("<b>Verification checklist actually executed.</b> Analytic gradient "
      "versus central finite differences: %d checks, worst relative error "
      "%.2e (threshold %.0e). Conservation on the FINAL learned T: Toeplitz "
      "deviation 0.0, FFT/IIR %.2e, last-time %.2e, strict causality, state "
      "constant in L. Adjudication %d/%d checks pass; NT31 %d/%d, R1 %d/%d, "
      "R2 %d/%d, R3/R4 %d/%d. All six theorem families are re-run after "
      "every architecture change, not just after the paper edit." %
      (GC["checked"], GC["worst_rel_error"], GC["threshold"],
       GD["fft_iir_rel_error"], GD["last_time_dev"],
       ADJ["n_pass"], ADJ["n_checks"],
       N31["n_pass"], N31["n_checks"], RL1C["n_pass"], RL1C["n_checks"],
       R2C["n_pass"], R2C["n_checks"], OB["n_pass"], OB["n_checks"]),
       size=SMALL, lead=SLEAD, after=2.0)
    P("<b>Compute statement.</b> All experiments in this paper are CPU-only "
      "NumPy on synthetic content chains (V<=512 for the ablations, "
      "V<=2^18 for the FWHT timing only, L<=4097). No language-model "
      "training, no perplexity, no generation sample and no equal-compute "
      "comparison was run. Listing this here is deliberate: the theoretical "
      "claims do not borrow authority from experiments we did not perform.",
       size=SMALL, lead=SLEAD, after=3.0)

'''
rep(anchor5, sec410 + anchor5, "sec4.10")

open(p, "w", encoding="utf-8").write(s)
print("STEP 5 OK")
