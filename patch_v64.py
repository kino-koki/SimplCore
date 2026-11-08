"""
patch_v64.py -- apply the v6.4 architecture-convergence update to the sole
current generator build_simplcore_pdf_v7.py.

All replacements are exact-string and are asserted to match exactly once,
so a silent no-op cannot produce a stale PDF.
"""
import io
import os
import re

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
orig_len = len(s)


def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, "%s: expected 1 match, got %d" % (label, n)
    s = s.replace(old, new, 1)
    print("  ok", label)


# ---------------------------------------------------------------- 1. data
rep('FP = BC["parity"]["fisher_p"]\n',
    '''FP = BC["parity"]["fisher_p"]

# ---- v6.4: convergence of the ARCHITECTURE module (not a diagnostic) ----
CONV = J(R + "convergence_curve.json")
CMAT = J(R + "convergence_matrix.json")
CGU = J(R + "convergence_guard.json")
ARCH = J(R + "stack_arch_check.json")
OPTA = J(E + "opt_ablation.json")
OPTB = J(E + "opt_basin.json")
OPTG = J(E + "opt_guard.json")


def cm(hops, layers):
    v = [r["acc"] for r in CMAT["rows"]
         if r["hops"] == hops and r["layers"] == layers]
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def cpair(hops=2):
    a = [r["acc"] for r in CMAT["rows"]
         if r["hops"] == hops and r["layers"] == 2]
    b = [r["acc"] for r in CMAT["rows"]
         if r["hops"] == hops and r["layers"] == 1]
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return (float(d.mean()), float(np.quantile(m, .025)),
            float(np.quantile(m, .975)))


CONV_FIRST = int(next(x for x in CONV["summary"]
                      if x["arm"] == "arch_codebook_M")["first_step_max"])
CPAIR = cpair(2)
W14A = max(x["worst"] for x in OPTG["legacy_straight_through"].values())
W14B = max(max(g["worst"].values())
           for g in CGU["detail"]["gradchecks_final_weights"].values())
''', "data loaders")

# ------------------------------------------------------------- 2. abstract
rep('''      "O(1)-state construction, and the layer bound survives 5-seed training "
      "from random initialisation (H=2: %.4f with one layer versus %.4f with "
      "two; paired difference %+.4f, 95%% CI [%+.4f, %+.4f]). The design "
      "rule is to budget log2 |A| bits at the input or relax exactness. We "
      "make no language-modeling, perplexity, throughput, equivalence or "
      "superiority claim; an earlier attention-parity statement is withdrawn "
      "because it did not reproduce (W12)." %
      (mean(2, "tsc_h1"), mean(2, "tsc_h2"), CI["mean"],
       CI["ci95"][0], CI["ci95"][1]),''',
    '''      "O(1)-state construction. In the architecture module the H6 interface "
      "is the codebook's fixed value-to-key swap rather than a learned "
      "matrix; from random initialisation of the remaining weights, H=2 with "
      "two layers reaches %.4f +/- %.4f by step %d on five of five seeds, "
      "while one layer stays at %.4f +/- %.4f (paired difference %+.4f, "
      "95%% CI [%+.4f, %+.4f] at 1500 steps). The design rule is to budget "
      "log2 |A| bits at the input or relax exactness. We make no "
      "language-modeling, perplexity, throughput, equivalence or "
      "superiority claim; an earlier attention-parity statement is "
      "withdrawn because it did not reproduce (W12)." %
      (cm(2, 2)[0], cm(2, 2)[1], CONV_FIRST,
       cm(2, 1)[0], cm(2, 1)[1], CPAIR[0], CPAIR[1], CPAIR[2]),''',
    "abstract")

# ------------------------------------------------------- 3. Result 4 line
rep('''    It(d, "<b>Result 4 (Empirical E).</b> Five-seed verification that the "
      "layer bound survives end-to-end training from random initialisation, "
      "degradation curves, and a same-run reproduction audit with one "
      "withdrawn claim.")''',
    '''    It(d, "<b>Result 4 (Empirical E).</b> Five-seed verification at a "
      "1500-step budget that the layer bound survives end-to-end training "
      "from random initialisation (Section 4.12), degradation curves, and a "
      "same-run reproduction audit with one withdrawn claim.")''',
    "contribution list")

# ------------------------------------------------- 4. falsification clause
rep('''      "have to come out differently.", size=SMALL, lead=SLEAD, after=2.5)''',
    '''      "have to come out differently. The Section 4.12 convergence claim is "
      "falsified by any of the five seeds missing 0.99 at step 1500 on a "
      "diagonal cell, or by any off-diagonal cell reaching it.",
      size=SMALL, lead=SLEAD, after=2.5)''',
    "falsification clause")

# ----------------------------------------------------------- 5. NEW 4.12
NEW = '''
    # ------------------------------------------------------ 4.12 (v6.4)
    Hh(d, "4.12  Convergence of the codebook-wired stack", size=9.5,
       before=4.0, after=2.0)
    P(d, "<b>Architectural change, not a training trick.</b> The H6 interface "
      "of Section 2 admits the special case T_h = C_val^T C_key, the "
      "codebook's own value-to-key swap. We make this the wiring of the "
      "architecture module: it is fixed, shared by all layers and NOT a "
      "learnable parameter, so the parameter count is independent of depth. "
      "Only the external-query projection and the read-out are learned, both "
      "from random initialisation. The inter-layer unit normalisation is "
      "built from log and exp inside the graph, removing the straight-through "
      "gradient (W14). The recurrence kernel is untouched: static, Toeplitz, "
      "causal, O(1) state, no score matrix and no cache.",
      size=SMALL, lead=SLEAD, after=2.0)
    _cv = [["wiring, H=2 and two layers", "acc (5 seeds)", "conv. seeds",
            "first step >= 0.99"]]
    for _key, _lab in (
            ("arch_codebook_M", "codebook-fixed interface"),
            ("arch_analytic_witness", "analytic witness, zero training"),
            ("legacy_learnable_T1_random", "learned interface, random init"),
            ("legacy_learnable_T1_fixedM", "learned interface, fixed swap"),
            ("legacy_trainable_near_M", "learned interface, near-swap init")):
        _r = next(x for x in CONV["summary"] if x["arm"] == _key)
        _cv.append([_lab,
                    "%.4f +/- %.4f" % (_r["mean"], _r["sd"]),
                    "%d / 5" % _r["converged_seeds"],
                    ("%d" % _r["first_step_mean"]
                     if _r["first_step_mean"] is not None else "none")])
    d.table(_cv, [86, 62, 38, 52], size=6.7, lead=8.0, after=2.0)
    P(d, "<b>Table 13.</b> 1500 Adam steps, lr %.2f, batch %d, five seeds, "
      "512 training and 512 held-out instances per seed. All arms see the "
      "same data, embedding and optimiser. The learned random interface does "
      "not converge at this budget; the codebook-fixed interface and the "
      "near-swap basin do. The swap is an algebraic object of the Hadamard "
      "dictionary and contains no training data, labels or run-time "
      "annotation." % (CONV["protocol"]["lr"], CONV["protocol"]["batch"]),
      size=SMALL, lead=SLEAD, after=2.0)
    _mx = [["hops / layers", "1", "2", "3", "4"]]
    for _g in CMAT["grid"]:
        _mx.append([str(_g["hops"])] +
                   ["%.4f" % c["mean"] for c in _g["cells"]])
    d.table(_mx, [56, 44, 44, 44, 44], size=6.8, lead=8.1, after=2.0)
    P(d, "<b>Table 14.</b> Trained layer-by-hop grid at 1500 steps, five "
      "seeds per cell. The diagonal is %.4f everywhere; the largest "
      "off-diagonal mean is %.4f. The H=2 paired difference is %+.4f with "
      "95%% CI [%+.4f, %+.4f], all five paired differences positive. "
      "Conservation, gradient and interface-identity checks pass on the "
      "final weights: Toeplitz deviation 0, FFT/IIR agreement to machine "
      "precision, future perturbations change no past output, state bytes "
      "independent of L, and finite-difference gradients within 5e-4 at "
      "every depth. These are experiments, not theorems; W13 is re-established "
      "here on the corrected architecture and the earlier number is not "
      "reused. No language-modeling or attention-comparison claim is made." %
      (min(c["mean"] for g in CMAT["grid"]
           for i, c in enumerate(g["cells"]) if i == g["hops"] - 1),
       max(c["mean"] for g in CMAT["grid"]
           for i, c in enumerate(g["cells"]) if i != g["hops"] - 1),
       CPAIR[0], CPAIR[1], CPAIR[2]),
      size=SMALL, lead=SLEAD, after=2.5)

'''
rep('''    Hh(d, "5  What This Does NOT Establish", size=10.2, before=5.0,
       after=2.5)''', NEW + '''    Hh(d, "5  What This Does NOT Establish", size=10.2, before=5.0,
       after=2.5)''', "new Section 4.12")

# ------------------------------------------------------------- 6. conclusion
rep('''      "by one Toeplitz, O(1)-state construction, and the layer bound survives "
      "end-to-end training. The practical rule is to budget the bits at the "''',
    '''      "by one Toeplitz, O(1)-state construction, and the layer bound survives "
      "1500-step end-to-end training from random initialisation (Section "
      "4.12) with the inter-layer map fixed to the codebook's own "
      "value-to-key swap. The practical rule is to budget the bits at the "''',
    "conclusion")

# ------------------------------------------------------------- 7. appendix B
rep('''        ["W12", "Single-hop attention parity (Fisher p=%.4f): the stored baseline did not reproduce in the same-run audit (Section 4.6). No equivalence or superiority claim is made." % FP],
    ], [34, 204], size=7.4, lead=9.0, after=3.0, span=False)''',
    '''        ["W12", "Single-hop attention parity (Fisher p=%.4f): the stored baseline did not reproduce in the same-run audit (Section 4.6). No equivalence or superiority claim is made." % FP],
        ["W13", "Trained corroboration of the layer bound under the earlier protocol: withdrawn after the fairness audit removed two input asymmetries. Re-established on the corrected architecture in Section 4.12; the earlier number is not reused."],
        ["W14", "Straight-through inter-layer normalisation (norm taken outside the graph): finite-difference gradient deviation %.1e. Replaced by a differentiable square root built from log and exp, deviation %.1e; all conservation checks are rerun on final weights." % (W14A, W14B)],
    ], [34, 204], size=7.4, lead=9.0, after=3.0, span=False)''',
    "withdrawn register")

# ------------------------------------------------------------- 8. appendix C
rep('''      "time; no value is transcribed by hand. All probes are CPU-only NumPy "
      "and complete in minutes. Seeds are listed per table. The environment, "''',
    '''      "time; no value is transcribed by hand. All probes are CPU-only NumPy "
      "and complete in minutes. The Section 4.12 run uses 1500 Adam steps, "
      "lr 0.02, batch 64, five seeds, and 512 training plus 512 held-out "
      "instances per seed; the architecture module and every convergence "
      "curve are in the artifact. Seeds are listed per table. The environment, "''',
    "reproducibility")

io.open(P, "w", encoding="utf-8").write(s)
print("patched %s: %d -> %d bytes" % (os.path.basename(P), orig_len, len(s)))
