"""
patch_v65b.py -- v6.5 part 2. Appendices F, G, H: all raw curves and the
exhaustive information-bound enumeration. Every row is a real measurement.
"""
import io

P = "/data/workspace/build_simplcore_pdf_v7.py"
s = io.open(P, encoding="utf-8").read()
orig = len(s)


def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, "%s: %d matches" % (label, n)
    s = s.replace(old, new, 1)
    print("  ok", label)


APPFGH = r'''
    # ---------------------------------------------------------------- app F
    Hh(d, "Appendix F  All 1500-Step Convergence Curves", size=11.5,
       before=6.0, after=3.0, span=False)
    P(d, "Section 4.12 reports only the diagonal and the first-crossing "
      "step. Here every sampled point of every arm is printed for all five "
      "seeds, followed by every individual cell of the layer-by-hop matrix. "
      "Protocol: 1500 Adam steps, lr 0.02, batch 64, 512 training and 512 "
      "held-out instances per seed, width d = 64, fixed Hadamard embedding, "
      "evaluation every 150 steps. The architecture module's inter-layer "
      "map is the fixed codebook value-to-key swap; only the query "
      "projection and read-out are learned, from random initialisation.",
      size=SMALL, lead=SLEAD, after=2.5, span=False)
    _steps = sorted({int(c["step"]) for r in CONV["rows"]
                     for c in r["curve"]})
    _f1 = [["arm", "step", "mean acc", "sd", "n seeds"]]
    for _a in ("arch_codebook_M", "arch_analytic_witness",
               "legacy_learnable_T1_random", "legacy_learnable_T1_fixedM",
               "legacy_trainable_near_M"):
        _rs = [r for r in CONV["rows"] if r["arm"] == _a]
        for _st in _steps:
            _v = [next(c["acc"] for c in r["curve"] if int(c["step"]) == _st)
                  for r in _rs
                  if any(int(c["step"]) == _st for c in r["curve"])]
            if not _v:
                continue
            _f1.append([_a.replace("_", " ")[:34], str(_st),
                        "%.4f" % (sum(_v) / len(_v)),
                        "%.4f" % (float(np.std(_v))), str(len(_v))])
    d.table(_f1, [96, 30, 36, 32, 32], size=6.4, lead=7.6, after=2.5,
            span=False)
    P(d, "<b>Table 22.</b> All %d arm-step cells. The random learnable "
      "interface stays near 0.25 for the entire budget; the codebook-fixed "
      "interface and the near-swap basin cross 0.99 by step %d on every "
      "seed. The analytic witness is exact at step 0 by construction." %
      (len(_f1) - 1, CONV_FIRST), size=SMALL, lead=SLEAD, after=2.5,
      span=False)

    _f2 = [["hops", "layers", "seed", "acc", "params", "steps"]]
    for _r in CMAT["rows"]:
        _f2.append([str(_r["hops"]), str(_r["layers"]), str(_r["seed"]),
                    "%.4f" % _r["acc"], str(_r["params"]),
                    str(_r["steps"])])
    d.table(_f2, [30, 34, 26, 40, 40, 34], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 23.</b> All %d layer-by-hop runs. Parameter count is "
      "%.0f in every row because the inter-layer map is shared and is not "
      "a parameter; depth adds computation, not weights. The diagonal is "
      "the only set of cells above 0.99." %
      (len(CMAT["rows"]),
       next(r["params"] for r in CMAT["rows"])),
      size=SMALL, lead=SLEAD, after=2.5, span=False)

    # ---------------------------------------------------------------- app G
    Hh(d, "Appendix G  All Fusion Evolution Curves", size=11.5,
       before=6.0, after=3.0, span=False)
    P(d, "Section 4.13 reports final accuracies. Here the full trajectory "
      "of each arm is printed: accuracy is sampled every 10 rounds and "
      "averaged over the five seeds at each sampled round. Protocol: 32 "
      "bindings, 320 i.i.d. arrival rounds, rho = 0.999 except the "
      "fast-decay arm, epsilon = 1e-3, capacity 64, one-sided paired-CI "
      "gate. The core state is 32768 bytes at every sampled point of every "
      "run.", size=SMALL, lead=SLEAD, after=2.5, span=False)
    _rounds = sorted({int(c["round"]) for r in FEV["rows"]
                      for c in r["curve"]})
    _g1 = [["arm", "round", "mean acc", "sd", "mean slots"]]
    for _a in FARMS:
        _rs = [r for r in FEV["rows"] if r["arm"] == _a]
        for _rd in _rounds:
            _v, _sl = [], []
            for _r in _rs:
                _c = next((c for c in _r["curve"]
                           if int(c["round"]) == _rd), None)
                if _c is not None:
                    _v.append(_c["acc"]); _sl.append(_c["slots"])
            if not _v:
                continue
            _g1.append([_a.replace("_", " ")[:34], str(_rd),
                        "%.4f" % (sum(_v) / len(_v)),
                        "%.4f" % float(np.std(_v)),
                        "%.1f" % (sum(_sl) / len(_sl))])
    d.table(_g1, [92, 32, 36, 32, 40], size=6.4, lead=7.6, after=2.5,
            span=False)
    P(d, "<b>Table 24.</b> All %d arm-round cells. The fused arm is "
      "monotone on all five seeds; the ungated corrupted arm rises then "
      "fluctuates as later writes overwrite earlier correct registrations; "
      "the fast-decay arm loses slots as SE5's finite-precision capacity "
      "is exhausted." % (len(_g1) - 1), size=SMALL, lead=SLEAD,
      after=2.5, span=False)

    _g2 = [["comparison", "paired diff", "CI95 low", "CI95 high", "all pos"]]
    for _k, _v in FEV["paired"].items():
        _g2.append([_k.replace("fused_minus_", "").replace("_", " "),
                    "%+.4f" % _v["mean"],
                    "%+.4f" % _v["ci95"][0],
                    "%+.4f" % _v["ci95"][1],
                    "yes" if _v["all_positive"] else "no"])
    d.table(_g2, [76, 36, 36, 36, 28], size=6.4, lead=7.6, after=2.5,
            span=False)
    P(d, "<b>Table 25.</b> All %d paired bootstrap comparisons, 10000 "
      "resamples, five seeds. The ungated clean arm ties exactly: with no "
      "corruption the gate is irrelevant, which is the measured form of "
      "SE3's scope, not a failure of the comparison." %
      len(FEV["paired"]), size=SMALL, lead=SLEAD, after=2.5, span=False)

    # ---------------------------------------------------------------- app H
    Hh(d, "Appendix H  Exhaustive Information-Bound Enumeration",
       size=11.5, before=6.0, after=3.0, span=False)
    P(d, "Theorem A is checked against the complete enumeration of the "
      "binary-chain fibre space, not against a sample. Every (m, B) pair "
      "below is enumerated over all 2^m role assignments with bit-identical "
      "visible inputs. The depth scan then verifies that stacking 1-8 "
      "static-LTI layers produces bit-identical outputs across compatible "
      "role assignments. These are finite exhaustive objects; they certify "
      "non-vacuity, they are not the proof.",
      size=SMALL, lead=SLEAD, after=2.5, span=False)
    _h1 = [["m", "B", "answer classes", "fibres", "acc", "predicted"]]
    for _r in IBS:
        _h1.append([str(_r["m"]), str(_r["B"]),
                    str(_r["n_answer_classes"]), str(_r["n_fibres"]),
                    "%.4f" % _r["acc"], "%.4f" % _r["predicted_upper"]])
    d.table(_h1, [22, 22, 46, 32, 40, 40], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 26.</b> All %d (m, B) enumeration cells. Accuracy "
      "equals the predicted upper bound 2^B / |A| in every row and reaches "
      "1 exactly at B = m." % len(IBS), size=SMALL, lead=SLEAD,
      after=2.5, span=False)

    _h2 = [["m", "B", "layers", "|A|", "acc", "output dev"]]
    for _r in IBD["rows"]:
        _h2.append([str(_r["m"]), str(_r["B"]), str(_r["n_layers"]),
                    str(_r["n_answer_classes"]),
                    "%.4f" % _r["acc"],
                    "%.1e" % _r["max_output_dev_across_d"]])
    d.table(_h2, [22, 22, 30, 30, 40, 44], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 27.</b> All %d depth-scan cells. Accuracy and output "
      "deviation are invariant to layer count, which is the empirical form "
      "of Theorem A(e): depth cannot manufacture the missing bits." %
      len(IBD["rows"]), size=SMALL, lead=SLEAD, after=2.5, span=False)

    _h3 = [["carrier", "bits / token", "acc", "Toeplitz", "run-time tag"]]
    for _r in IBC:
        _h3.append([_r["carrier"].replace("_", " ")[:34],
                    str(_r["B_bits_per_ambiguous_token"]),
                    "%.4f" % _r["acc"],
                    "yes" if _r["toeplitz"] else "no",
                    "yes" if _r["external_run_time_annotation"] else "no"])
    d.table(_h3, [96, 36, 36, 32, 40], size=6.4, lead=7.6,
            after=2.5, span=False)
    P(d, "<b>Table 28.</b> All %d bit carriers. The position format prior "
      "is a support restriction, not a per-token channel, and is analysed "
      "through A(v) rather than through B; its effective bit count under "
      "arbitrary filler is zero." % len(IBC), size=SMALL, lead=SLEAD,
      after=2.5, span=False)

    _h4 = [["hops", "layers", "acc", "per-layer accuracies"]]
    for _r in GRID:
        _h4.append([str(_r["hops"]), str(_r["n_layers"]),
                    "%.4f" % _r["acc"],
                    " ".join("%.2f" % x for x in _r["per_layer_acc"])])
    d.table(_h4, [26, 30, 38, 110], size=6.4, lead=7.6, after=2.5,
            span=False)
    P(d, "<b>Table 29.</b> All %d zero-training layer-by-hop cells with "
      "the full per-layer accuracy vector. The last-layer read-out is why "
      "H_L > H overshoots fail: the correct intermediate value is computed "
      "at layer H but is not the layer that is read out." % len(GRID),
      size=SMALL, lead=SLEAD, after=3.0, span=False)

'''
rep('''    total, main = d.finish()''', APPFGH + '''    total, main = d.finish()''',
    "Appendix F-G-H")

io.open(P, "w", encoding="utf-8").write(s)
print("patched: %d -> %d bytes" % (orig, len(s)))
