p = "build_simplcore_pdf_v5.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, label, n=1):
    global s
    assert s.count(old) == n, f"{label}: count={s.count(old)}"
    s = s.replace(old, new)


# ---- helper: bootstrap CI from the raw 5-seed arms ----
rep('''def rk(r):
    return next(x["mean_acc"] for x in RNK if x["rank"] == r)''',
'''def rk(r):
    return next(x["mean_acc"] for x in RNK if x["rank"] == r)


def arm_seeds(arm, enc=None, lay=None):
    return [r["acc"] for r in FA if r["arm"] == arm
            and (enc is None or r["encoding"] == enc)
            and (lay is None or r["layout"] == lay)]


def ci95(vals, seed=24680):
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(a, len(a), replace=True).mean()
                   for _ in range(10000)])
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))''', "ci helper")

# ---- §3.8 dependency map (inserted before 4 CONSTRUCTIVE) ----
anchor4 = '''    # ------------------------------------------------------------ 4 CONSTRUCTIVE'''
sec38 = '''    Hh("3.8  Theorem dependency map and symbol discipline",
       size=9.4, before=5.0, after=2.0)
    P("A reviewer should not have to reconstruct the logic. Table 3 states "
       "which lemma each core theorem consumes and which conclusion it "
       "supports. The single narrative is: order 2 (T1) + H layers (T2) + "
       "exactly one bit (R3/R2) jointly close the search space; T3-T6 supply "
       "the conservation and impossibility boundaries that make the closure "
       "meaningful.", size=SMALL, lead=SLEAD, after=2.0)
    dep = [["core", "consumes", "supports"]]
    for a, b, c in [
        ("T1 order=2", "Z-transform, P1-P2", "necessity of tensor state"),
        ("T2 H layers", "R2 induction, R3 necessity", "main tight bound"),
        ("T3 causality+O(1)", "T1, P4", "state bound, no L growth"),
        ("T4 Toeplitz iff", "delay induction", "why time variation breaks FFT"),
        ("T5 no read-out", "zero-space lemma", "why order 1 cannot work"),
        ("T6 info bound", "pigeonhole, P8", "correct 'O(1) must lose' form"),
    ]:
        dep.append([a, b, c])
    d.table(dep, [68, 88, 88], size=6.8, lead=8.1, after=2.0)
    P("<b>Symbol discipline.</b> t is absolute time, d = L-1-t is a delay, "
      "tau is a kernel lag, H is hops, H_L is layers, n is pairs, V_K=V/2 is "
      "the key-vocabulary size, r is the layer-interface rank, rho<1 is a "
      "pole modulus. No symbol is reused. Every theorem in Sections 3.1-3.6 "
      "carries its FULL premises on the same page; Appendix A repeats them "
      "with the complete proofs. Figure 1 places all three bounds and the "
      "observed two-hop failure point on one boundary-and-tightness diagram.",
       size=SMALL, lead=SLEAD, after=3.0)

'''
rep(anchor4, sec38 + anchor4, "sec3.8")

# ---- §4.9 5-seed statistics (inserted before 5 NOT ESTABLISH) ----
anchor5 = '''    # ------------------------------------------------------------ 5 NOT ESTABLISH'''
sec49 = '''    Hh("4.9  Five-seed paired statistics (Table 4)",
       size=9.4, before=5.0, after=2.0)
    P("Every entry below is a 5-seed mean with a 10,000-resample paired "
      "bootstrap CI over 512 held-out samples per seed. We report "
      "'no significant difference' as what the statistic supports; p-values "
      "are NOT equivalence proofs. The baseline for the earlier parity run "
      "came from a historical record file, not a same-run re-execution, and "
      "we keep that limitation visible.", size=SMALL, lead=SLEAD, after=2.0)
    st = [["configuration", "mean", "std", "CI95"]]
    for lab, vals in [
        ("1 bit / alternating", arm_seeds("learned_T", "codebook", "alternating")),
        ("1 bit / gapped", arm_seeds("learned_T", "codebook", "gapped")),
        ("1 bit / dense", arm_seeds("learned_T", "codebook", "dense")),
        ("1 bit / padded", arm_seeds("learned_T", "codebook", "padded")),
        ("0 bit / alternating", arm_seeds("learned_T", "flat", "alternating")),
        ("0 bit / gapped", arm_seeds("learned_T", "flat", "gapped")),
        ("0 bit / dense", arm_seeds("learned_T", "flat", "dense")),
        ("0 bit / padded", arm_seeds("learned_T", "flat", "padded")),
        ("2 bits / +gate", arm_seeds("two_bits_plus_gate")),
        ("analytic T* (0 train)", arm_seeds("fixed_Tstar")),
        ("frozen random T", arm_seeds("frozen_random_T")),
    ]:
        lo, hi = ci95(vals)
        st.append([lab, "%.4f" % float(np.mean(vals)),
                   "%.4f" % float(np.std(vals, ddof=1)),
                   "[%.4f, %.4f]" % (lo, hi)])
    d.table(st, [92, 40, 38, 74], size=6.6, lead=8.0, after=2.5)
    P("<b>Reading.</b> The four 1-bit rows have zero variance and a "
      "degenerate CI at 1.0000. The 2-bit row is far below every 1-bit row. "
      "The 0-bit rows rise from %.4f (tight) to %.4f (dense), which is why "
      "Lemma R3 must be stated in the WORST-CASE sense. We do not convert "
      "any of these numbers into a language-model claim." %
      (float(np.mean(arm_seeds("learned_T", "flat", "alternating"))),
       float(np.mean(arm_seeds("learned_T", "flat", "dense")))),
       size=SMALL, lead=SLEAD, after=3.0)

'''
rep(anchor5, sec49 + anchor5, "sec4.9")

# ---- §5 add the v6 boundary explicitly ----
rep('''    It("<b>Not a SOTA model.</b> Positioning: boundary-characterization paper + "''',
'''    It("<b>Exactly one bit is a worst-case optimum, not a uniform one.</b> "
       "Zero-bit models reach %.4f on dense and %.4f on padded layouts, "
       "because zero fillers implicitly suppress the spurious peak. Lemma "
       "R3 proves they cannot be correct on ALL legal inputs; it does NOT "
       "claim they fail everywhere. The 2-bit result (%.4f) is strictly "
       "worse than the 1-bit result, so 'more role information is safer' is "
       "also false." % (fc_layout("flat", "dense"),
                        fc_layout("flat", "padded"),
                        fc("two_bits_plus_gate")), size=SMALL, lead=SLEAD)
    It("<b>Not a SOTA model.</b> Positioning: boundary-characterization paper + "''', "sec5 boundary")

open(p, "w", encoding="utf-8").write(s)
print("STEP 4 OK")
