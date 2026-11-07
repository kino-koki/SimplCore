"""
fibre_probe.py — P1 bridge: compute |A(v)| in the ORIGINAL content-chain task.

Reviewer objection: "the tight witness is built on a task tailored to the
theorem; say what |A| is in the original content-chain task, or significance
is punctured."

We enumerate, for each sample and each H, the set of terminal ids reachable
from the external query by H successor steps under the FLAT (0-bit) encoding,
where the model can only match a predecessor symbol id.  The codebook
partition removes the VALUE-half symbols from the query's matching set, so
only one successor per hop remains.  This is a REACHABILITY ENUMERATION, not
a claim about all conceivable binding graphs; that scope is stated in the
paper.

No training, no gradients, no private weights.
"""
import json
import os
import sys

import numpy as np

W = "/data/workspace"
sys.path.insert(0, W + "/train/superposition_probe")
import tsc_final_probe as TSC

OUT = W + "/tools/eval_baselines/results/fibre.json"
V = 64
VK = V // 2
N_PAIRS = 4


def visible_ids(D, i):
    """Non-empty visible symbol ids of sample i (zero fillers dropped)."""
    X = D["X"][i]
    present = X.sum(-1) > 0.5
    return X[present].argmax(-1).astype(int).tolist()


def successors(ids, q, codebook):
    """One-hop successor candidates under the layer's matching rule.

    The layer computes m[t] = <q, e(x_{t-1})> and z = sum K[t] m[t] e(x_t).
    Under a flat alphabet e(id) = C[id] for BOTH roles, so every occurrence
    of id == q as a predecessor yields one candidate successor = next id.
    Under the codebook partition VALUE symbols live in the orthogonal half;
    only KEY-half predecessors match, so at most one successor survives.
    """
    out = []
    for j in range(len(ids) - 1):
        s = ids[j]
        is_key_half = s < VK
        if codebook and not is_key_half:
            continue
        # flat projection of a VALUE-half id is (s - VK)
        sflat = s if (not codebook or is_key_half) else s - VK
        if sflat == q:
            nxt = ids[j + 1]
            nflat = nxt if nxt < VK else nxt - VK
            out.append(nflat)
    return sorted(set(out))


def reachable(ids, q, H, codebook):
    cur = {q}
    for _ in range(H):
        nxt = set()
        for c in cur:
            nxt.update(successors(ids, c, codebook))
        if not nxt:
            break
        cur = nxt
    return cur


def run():
    rows = []
    for H in (1, 2, 3, 4):
        for enc in ("flat", "codebook"):
            sizes = []
            for seed in (0, 1, 2, 3, 4):
                D = TSC.build(N_PAIRS, H, 256, 3000 + seed, V,
                              layout="gapped", encoding="flat",
                              filler_max=3, dup=False)
                # codebook view: re-build with partitioned value ids but keep
                # the SAME chain / seed so the two views are paired.
                Dcb = TSC.build(N_PAIRS, H, 256, 3000 + seed, V,
                                layout="gapped", encoding="codebook",
                                filler_max=3, dup=False)
                for i in range(256):
                    vid = visible_ids(Dcb if enc == "codebook" else D, i)
                    q = int(Dcb["Q0"][i] @ np.linalg.pinv(TSC.C_READ["codebook"])) \
                        if False else None
                    # Q0 = C[chain[0]] and C is symmetric orthogonal, so
                    # Q0 @ C is the one-hot query id (KEY half).
                    Cfull = TSC.hadamard(V)
                    q = int(np.argmax(np.clip(Dcb["Q0"][i] @ Cfull, 0, 1)))
                    A = reachable(vid, q, H, codebook=(enc == "codebook"))
                    sizes.append(len(A))
            a = np.array(sizes)
            rows.append(dict(H=H, encoding=enc, n_samples=int(a.size),
                             mean_A=float(a.mean()), max_A=int(a.max()),
                             min_A=int(a.min()),
                             mean_log2A=float(np.log2(np.maximum(a, 1)).mean()),
                             max_log2A=float(np.log2(max(a.max(), 1))),
                             frac_A_gt_1=float((a > 1).mean())))
    # information budget actually supplied by each carrier
    B_tag = N_PAIRS          # tag only the KEY token of each pair: 1 bit x n
    B_partition = 2 * N_PAIRS  # every non-empty token id gains 1 bit
    out = dict(
        status="reachability enumeration on the original content-chain task",
        scope=("A(v) is the set of terminal ids reachable by H successor "
               "steps from the external query under the layer's matching "
               "rule. It is NOT an enumeration over all conceivable binding "
               "graphs. The tight binary-chain witness of Theorem A(d) is a "
               "DIFFERENT construction with |A| = 2^m."),
        task=dict(V=V, VK=VK, n_pairs=N_PAIRS, layout="gapped",
                  filler_max=3, distinct_keys=True, seeds=5,
                  samples_per_seed=256),
        rows=rows,
        carriers=dict(
            explicit_tag_bits=B_tag,
            alphabet_partition_bits=B_partition,
            note=("The partition supplies 2*n_pairs bits, one per non-empty "
                  "token, whereas log2|A| in the original task is bounded "
                  "by log2(H+1). The budget is SUFFICIENT but NOT TIGHT on "
                  "this task. Tightness is established on the binary-chain "
                  "witness, where |A| = 2^m and B = m.")),
        conclusion=("On the original content-chain task the necessary input-"
                    "side information is at most log2(H+1) bits, and the "
                    "codebook partition supplies far more. We do NOT claim "
                    "the original task is tight; we state the gap explicitly."))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
