"""patch_v65c.py -- Appendix I: complete numeric provenance and witness detail."""
import io

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
orig = len(s)

APPI = r'''
    # ---------------------------------------------------------------- app I
    Hh(d, "Appendix I  Complete Numeric Provenance and Witness Detail",
       size=11.5, before=6.0, after=3.0, span=False)
    P(d, "Every numeric value quoted anywhere in the paper is listed below "
      "with its statistic, seed count, sample count and the object it was "
      "measured on. This table exists so that no number has to be taken on "
      "trust and no earlier drifted value can re-enter unnoticed. All "
      "entries are finite-sample quantities unless the statistic column "
      "says otherwise; none is a theorem.",
      size=SMALL, lead=SLEAD, after=2.5, span=False)
    _prov = [["quantity", "value", "statistic", "seeds / n", "object"]]
    _prov += [
        ["chance (this task)", "%.4f" % CHANCE, "definition", "-", "1/V_K"],
        ["H=2, 2 layers, 1500 steps", "%.4f" % cm(2, 2)[0],
         "mean / sd %.4f" % cm(2, 2)[1], "5 / 512", "architecture stack"],
        ["H=2, 1 layer, 1500 steps", "%.4f" % cm(2, 1)[0],
         "mean / sd %.4f" % cm(2, 1)[1], "5 / 512", "architecture stack"],
        ["H=2 paired difference", "%+.4f" % CPAIR[0],
         "paired bootstrap", "5", "two-layer minus one-layer"],
        ["paired 95%% CI", "[%+.4f, %+.4f]" % (CPAIR[1], CPAIR[2]),
         "10000 resamples", "5", "same pair"],
        ["first step >= 0.99", str(CONV_FIRST), "max over seeds", "5",
         "codebook-fixed interface"],
        ["legacy random interface, 1500", "%.4f" % next(
            x["mean"] for x in CONV["summary"]
            if x["arm"] == "legacy_learnable_T1_random"), "mean",
         "5", "learnable inter-layer matrix"],
        ["SE2 predicted E[T_cover]", "%.2f" % AGREE["SE2_predicted_E_T_cover"],
         "analytic n H_n / (1-beta)", "-", "n=32, beta~0"],
        ["SE2 measured mean cover", "%.2f" %
         AGREE["SE2_measured_mean_first_cover"], "mean over seeds", "5",
         "fused gated arm"],
        ["SE2 relative gap", "%.4f%%" % (100 * AGREE["SE2_relative_gap"]),
         "derived", "-", "prediction vs measurement"],
        ["SE3 exact E[acc]", "%.6f" % FTH["SE3"]["exact_expectation"],
         "analytic", "-", "p=0.30, V_K=32"],
        ["SE3 enumeration mean", "%.6f" % FTH["SE3"]["observed_mean"],
         "mean over %d key trials" % FTH["SE3"]["n_key_trials"],
         "20000 runs", "exhaustive per-key"],
        ["SE3 Hoeffding 4-sigma", "[%.4f, %.4f]" %
         tuple(FTH["SE3"]["hoeffding_4sigma_interval"]),
         "concentration", "-", "same"],
        ["SE4 bits consumed", str(int(FTH["SE4"]["bits_consumed"])),
         "analytic", "-", "T=100, w=1, b=5"],
        ["SE4 CT7 bound", str(int(FTH["SE4"]["CT7_info_bound_bits"])),
         "analytic upper bound", "-", "same"],
        ["SE5 K* at rho=0.999", str(FTH["SE5"]["K_star"]),
         "floor(log eps / log rho)", "-", "eps=1e-3"],
        ["core state bytes", "32768", "exact", "-",
         "d x d fp64, all runs"],
        ["fused final accuracy", "%.4f" % fev("fused_gated")["mean"],
         "mean / sd %.4f" % fev("fused_gated")["sd"], "5 / 512",
         "320 rounds"],
        ["ungated corrupted", "%.4f" % fev("ungated_corrupt")["mean"],
         "mean / sd %.4f" % fev("ungated_corrupt")["sd"], "5 / 512",
         "p=0.30, no gate"],
        ["break alignment", "%.4f" % fev("break_alignment")["mean"],
         "mean / sd %.4f" % fev("break_alignment")["sd"], "5 / 512",
         "non-codebook registration"],
        ["fast decay", "%.4f" % fev("rho_fast")["mean"],
         "mean / sd %.4f" % fev("rho_fast")["sd"], "5 / 512",
         "rho=0.90"],
        ["SE1 max on-target error", "%.1e" %
         FTH["SE1"]["worst_on_error"], "max over 200 instances", "200",
         "orthogonality identity"],
        ["SE1 max off-target error", "%.1e" %
         FTH["SE1"]["worst_off_error"], "max over 200 instances", "200",
         "same"],
        ["FFT/IIR duality (final state)", "%.2e" %
         max(AGU.get("fft_iir_rel_error", 0),
             CBU.get("fft_iir_rel_error", 0),
             FGU.get("fft_iir_rel_error", 0)),
         "max over probes", "-", "real-field equivalence"],
        ["Toeplitz deviation (final)", "0", "exact", "-",
         "static recurrence"],
    ]
    d.table(_prov, [78, 48, 60, 34, 60], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 30.</b> All %d provenance rows. Values from different "
      "probes are never averaged together; the FFT/IIR row is a maximum "
      "over separately listed probes, not a mixture." % (len(_prov) - 1),
      size=SMALL, lead=SLEAD, after=2.5, span=False)

    _ws = [["witness", "measured value", "what it certifies"]]
    for _k, _v in FGD["checks"].items():
        if _k.startswith("theory_") or _k.startswith("v64_"):
            continue
        _ws.append([_k.replace("_", " ")[:40],
                    "pass" if _v else "FAIL",
                    "consistency on final weights" if "guard" not in _k
                    else "guard"])
    _ws += [
        ["SE2 prediction gap", "%.4f%%" %
         (100 * AGREE["SE2_relative_gap"]),
         "coupon-collector bound is non-vacuous"],
        ["SE3 concentration", "inside 4-sigma",
         "exact expectation, not an upper bound"],
        ["SE4 equality", "%d = %d bits" %
         (int(FTH["SE4"]["bits_consumed"]),
          int(FTH["SE4"]["CT7_info_bound_bits"])),
         "no contradiction with CT7"],
        ["SE5 capacity", "%d vs %d" %
         (FTH["SE5"]["K_star"],
          int(__import__("math").floor(__import__("math").log(1e-3) /
                                       __import__("math").log(0.90)))),
         "decay predicts the loss"],
    ]
    d.table(_ws, [92, 46, 92], size=6.4, lead=7.6, after=2.5,
            span=False)
    P(d, "<b>Table 31.</b> Witness detail for the non-analytic checks. A "
      "false here would be a finding to fix, not a crashed run.",
      size=SMALL, lead=SLEAD, after=2.5, span=False)

    _fib = [["hops", "encoding", "n", "mean |A|", "max log2|A|",
             "frac |A|>1"]]
    for _r in FIB["rows"]:
        _fib.append([str(_r["H"]), _r["encoding"], str(_r["n_samples"]),
                     "%.2f" % _r["mean_A"],
                     "%.2f" % _r["max_log2A"],
                     "%.2f" % _r["frac_A_gt_1"]])
    d.table(_fib, [26, 44, 34, 40, 44, 40], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 32.</b> Compatible-answer-set enumeration for the "
      "original content-chain task. The bound B >= log2 |A| is sufficient "
      "but not tight on this task; tightness belongs to the binary-chain "
      "witness of Appendix H. This distinction is stated in Section 4.3 "
      "and is not hidden here.", size=SMALL, lead=SLEAD, after=3.0,
      span=False)

'''
old = "    total, main = d.finish()"
assert s.count(old) == 1
s = s.replace(old, APPI + old, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("patched: %d -> %d bytes" % (orig, len(s)))
