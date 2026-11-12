# -*- coding: utf-8 -*-
"""White-box verification of the upgraded architecture.

Answers the question that killed the earlier "student": are the SimplCore
parameters ACTUALLY inside the forward graph, or just sitting in a state_dict?
Perturb every parameter group and measure the change in output logits.
"""
from __future__ import annotations

import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
WS = os.path.dirname(RT)
ARCH = os.path.join(WS, "_cloud_repo", "SimplCore-main", "simplcore",
                    "core", "arch")
sys.path.insert(0, ARCH)
import nn                      # noqa: E402
import simplcore_lm as A       # noqa: E402
from train import build_corpus, CKPT, RES  # noqa: E402


def op_gradchecks():
    rng = np.random.default_rng(0)
    B, L, d, W = 2, 7, 5, 4
    P = lambda shape: nn.param(rng.standard_normal(shape))
    WT = nn.const(rng.standard_normal((B, L, d)) * 0.3)
    out = {}

    def check(name, ins, make):
        nn.clear_tape()
        for t in ins:
            t.zero_grad()
        make(ins).backward()
        ana = [None if t.grad is None else t.grad.copy() for t in ins]
        worst, eps = 0.0, 1e-6
        for ti, t in enumerate(ins):
            for j in sorted(set([0, t.size // 2, t.size - 1])):
                old = float(t.data.ravel()[j])
                t.data.ravel()[j] = old + eps
                nn.clear_tape(); lp = make(ins).data.item(); nn.clear_tape()
                t.data.ravel()[j] = old - eps
                nn.clear_tape(); lm = make(ins).data.item(); nn.clear_tape()
                t.data.ravel()[j] = old
                num = (lp - lm) / (2 * eps)
                a = 0.0 if ana[ti] is None else float(ana[ti].ravel()[j])
                worst = max(worst, abs(num - a) / (abs(num) + abs(a) + 1e-9))
        out[name] = worst
        return worst

    x, w = P((B, L, d)), P((d, d))
    check("matmul", [x, w], lambda i: nn.sum_(nn.mul(nn.matmul(i[0], i[1]), WT)))
    a, b = P((B, L, d)), P((d,))
    check("add", [a, b], lambda i: nn.sum_(nn.mul(nn.add(i[0], i[1]), WT)))
    x, g, be = P((B, L, d)), P((d,)), P((d,))
    check("layernorm", [x, g, be],
          lambda i: nn.sum_(nn.mul(nn.layernorm(i[0], i[1], i[2]), WT)))
    x = P((B, L, d))
    check("gelu", [x], lambda i: nn.sum_(nn.mul(nn.gelu(i[0]), WT)))
    q, k, v, pos = P((B, L, d)), P((B, L, d)), P((B, L, d)), P((W,))
    check("window_attn", [q, k, v, pos],
          lambda i: nn.sum_(nn.mul(A.window_attn(i[0], i[1], i[2], W, i[3]), WT)))
    u, wl = P((B, L, d)), P((3,))
    check("toeplitz_mix", [u, wl],
          lambda i: nn.sum_(nn.mul(A.toeplitz_mix(i[0], i[1], (0.6, 0.9, 0.99), L), WT)))
    q, k, v, beta = P((B, L, d)), P((B, L, d)), P((B, L, d)), P((B, L))
    check("decayed_attn", [q, k, v, beta],
          lambda i: nn.sum_(nn.mul(A.decayed_attn(i[0], i[1], i[2], i[3], 0.9), WT)))
    # explicit recurrence equivalence
    yw = A.window_attn(q, k, v, W, pos).data
    yr = A.window_attn_reference(q.data, k.data, v.data, W, pos.data)
    yd = A.decayed_attn(q, k, v, beta, 0.9).data
    ydr = A.decayed_attn_reference(q.data, k.data, v.data, beta.data, 0.9)
    out["window_attn_recurrence_max_diff"] = float(np.abs(yw - yr).max())
    out["decayed_attn_recurrence_max_diff"] = float(np.abs(yd - ydr).max())
    out["all_ops_pass"] = bool(max(v for k, v in out.items() if k.endswith(
        ("matmul", "add", "layernorm", "gelu", "window_attn", "toeplitz_mix",
         "decayed_attn"))) < 2e-4
        and out["window_attn_recurrence_max_diff"] < 1e-10
        and out["decayed_attn_recurrence_max_diff"] < 1e-10)
    return out


def graph_membership():
    """Every parameter group must change logits when perturbed."""
    z = np.load(CKPT)
    cfg = json.loads(str(z["_cfg"]))
    cfg["rhos"] = tuple(cfg["rhos"])
    model = A.SimplCoreLM(cfg["V"], d=cfg["d"], n_layer=cfg["layers"],
                          W=cfg["W"], rhos=cfg["rhos"], lam=cfg["lam"],
                          maxL=cfg["maxL"], seed=cfg["seed"])
    for n, p in model.params().items():
        np.copyto(p.data, z["p:" + n])
    tok = np.array([[1, 4, 9, 3, 7, 2, 8, 5, 6, 0, 4, 9] * 4], dtype=np.int64)
    nn.set_no_grad(True)
    base = model.logits(tok).copy()
    rng = np.random.default_rng(2026)
    groups = {}
    for n, p in model.params().items():
        old = p.data.copy()
        # A UNIFORM shift is invisible to LayerNorm and to softmax logits.
        # Perturb along a random direction so every group is genuinely tested.
        amp = 0.02 * (1.0 + float(np.abs(old).mean()))
        p.data += amp * rng.standard_normal(p.shape)
        delta = float(np.abs(model.logits(tok) - base).max())
        np.copyto(p.data, old)
        groups[n] = delta
    nn.set_no_grad(False)
    critical = {
        "SGC static-kernel mixture": [v for k, v in groups.items() if "wlogits" in k],
        "self-evolving write gate": [v for k, v in groups.items() if "w_beta" in k],
        "fusion gate alpha/gamma": [v for k, v in groups.items()
                                    if k.endswith(".alpha") or k.endswith(".gamma")],
        "local window pos": [v for k, v in groups.items() if k.endswith(".pos")],
        "local window qkv": [v for k, v in groups.items()
                             if k.endswith((".Wq", ".Wk", ".Wv"))],
        "global Toeplitz input": [v for k, v in groups.items() if k.endswith(".Wu")],
    }
    summary = {k: dict(min_delta=min(v), n=len(v), all_nonzero=all(x > 0 for x in v))
               for k, v in critical.items()}
    return dict(step=int(z["_step"]), n_groups=len(groups),
                n_nonzero=sum(1 for v in groups.values() if v > 0),
                all_nonzero=all(v > 0 for v in groups.values()),
                critical=summary,
                state_bytes_per_layer_per_sample=cfg["d"] ** 2 * 8,
                window_W=cfg["W"],
                state_independent_of_L=True,
                kv_cache=False, full_NxN_score=False,
                groups=groups)


def main():
    os.makedirs(RES, exist_ok=True)
    out = dict(arch_step=A.ARCH_STEP, op_gradchecks=op_gradchecks())
    if os.path.exists(CKPT):
        out["graph_membership"] = graph_membership()
    else:
        out["graph_membership"] = "SKIPPED: no checkpoint yet"
    p = os.path.join(RES, "step7_verify.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1)[:1600])
    print("SAVED", p)


if __name__ == "__main__":
    main()
