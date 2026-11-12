# -*- coding: utf-8 -*-
"""Pre-registered acceptance test for the upgraded architecture.

Criteria are fixed BEFORE looking at the final numbers:

  C1  gradcheck of every operator and of the full model passes (<2e-4 rel).
  C2  every one of the 54 parameter groups actually changes logits
      (non-uniform perturbation; a uniform shift is invisible to LayerNorm
      and to softmax and is NOT a valid membership test).
  C3  static-kernel mixture, gated write, fusion gate, local window and
      global Toeplitz path are all in the forward graph.
  C4  held-out character bpc is better than a smoothed trigram baseline
      trained on the SAME split, and better than the STEP-6 single-mixer
      reference (3.4583 bpc).
  C5  at least 4 of 6 prompts produce non-degenerate text:
      rep3 < 0.5, no same-char run >= 12, >= 8 distinct words, has letters.
  C6  the six generations are not identical to each other.

"SPEAKS" means: an end-to-end trained SimplCore model loads, generates
coherent, varied, non-degenerate text, and all of its architectural
components are demonstrably inside the forward graph. It does NOT mean a
1B-scale assistant, factual QA, or parity with a pretrained Transformer.
"""
from __future__ import annotations

import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
RES = os.path.join(RT, "results")


def trigram_baseline(train, val, V, lam=(0.60, 0.28, 0.12)):
    uni = np.ones(V, dtype=np.float64) * 0.1
    bi = np.ones((V, V), dtype=np.float64) * 0.01
    tri = np.ones((V, V, V), dtype=np.float64) * 0.001
    for t in range(len(train)):
        uni[train[t]] += 1
        if t >= 1:
            bi[train[t - 1], train[t]] += 1
        if t >= 2:
            tri[train[t - 2], train[t - 1], train[t]] += 1
    uni /= uni.sum(); bi /= bi.sum(axis=1, keepdims=True)
    tri /= tri.sum(axis=2, keepdims=True)
    nll = []
    for t in range(2, len(val)):
        a, b, c = int(val[t - 2]), int(val[t - 1]), int(val[t])
        p = lam[0] * tri[a, b, c] + lam[1] * bi[b, c] + lam[2] * uni[c]
        nll.append(-math.log(max(p, 1e-12)) / math.log(2.0))
    return float(np.mean(nll))


def main():
    from train import build_corpus
    train, val, V, stoi, _ = build_corpus()
    base = trigram_baseline(train, val, V)
    print("smoothed trigram baseline val_bpc = %.4f" % base, flush=True)

    v = json.load(open(os.path.join(RES, "step7_verify.json")))
    ops = v["op_gradchecks"]
    gm = v["graph_membership"]
    C1 = bool(ops["all_ops_pass"])
    C2 = bool(gm["all_nonzero"]) and gm["n_nonzero"] == gm["n_groups"]
    C3 = all(x["all_nonzero"] for x in gm["critical"].values())

    prog = json.load(open(os.path.join(RES, "step7_progress.json")))
    bpc = prog["end_val_bpc"]
    C4 = bool(bpc < base and bpc < 3.4583)

    gen_path = os.path.join(RES, "step7_generation.json")
    C5 = C6 = False
    samples = []
    if os.path.exists(gen_path):
        g = json.load(open(gen_path))
        samples = g["samples"]
        good = 0
        for s in samples:
            m = s["metrics"]
            if (m["rep3"] < 0.5 and m["longest_same_char_run"] < 12
                    and m["distinct_words"] >= 8 and m["has_letters"]):
                good += 1
        C5 = good >= 4
        C6 = len({s["text"] for s in samples}) == len(samples) and len(samples) >= 2

    out = dict(
        criteria=dict(C1_gradcheck=C1, C2_all_params_in_graph=C2,
                      C3_critical_paths_in_graph=C3,
                      C4_better_than_trigram_and_step6=C4,
                      C5_nondegenerate_generations=C5,
                      C6_varied_generations=C6),
        trigram_baseline_bpc=base, model_val_bpc=bpc,
        step6_reference_bpc=3.4583, ckpt_step=prog["step"],
        n_param_groups=gm["n_groups"], n_nonzero=gm["n_nonzero"],
        n_samples=len(samples),
        good_samples=sum(
            1 for s in samples
            if s["metrics"]["rep3"] < 0.5
            and s["metrics"]["longest_same_char_run"] < 12
            and s["metrics"]["distinct_words"] >= 8
            and s["metrics"]["has_letters"]),
        state_bytes_per_layer=gm["state_bytes_per_layer_per_sample"],
        window_W=gm["window_W"], kv_cache=gm["kv_cache"],
        full_NxN_score=gm["full_NxN_score"],
        SPEAKS=bool(C1 and C2 and C3 and C4 and C5 and C6),
        caveat="character-level 202k-parameter model on 1.1 MB real text; "
               "not a 1B assistant, not factual QA, not Transformer parity")
    json.dump(out, open(os.path.join(RES, "step7_acceptance.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("SAVED", os.path.join(RES, "step7_acceptance.json"))


if __name__ == "__main__":
    main()
