# -*- coding: utf-8 -*-
"""Autoregressive generation from a trained SimplCore LM checkpoint.

No teacher weights, no pretrained embedding, no renamed state_dict: the only
source of logits is SimplCoreLM.forward() with parameters loaded from
step7_ckpt.npz, which was produced by end-to-end training on real text.
"""
from __future__ import annotations

import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
WS = os.path.dirname(RT)
ARCH = os.path.join(WS, "_cloud_repo", "SimplCore-main", "simplcore",
                    "core", "arch")
sys.path.insert(0, ARCH)
import nn                      # noqa: E402
import simplcore_lm as arch    # noqa: E402
from train import build_corpus, CKPT, RES  # noqa: E402

PROMPTS = [
    "First Citizen:\n",
    "HAMLET:\n",
    "Q: What is your name?\nA:",
    "Q: Shall we proceed?\nA:",
    "Q: Who goes there?\nA:",
    "To be, or not to be,",
]


def load_model(cfg):
    model = arch.SimplCoreLM(cfg["V"], d=cfg["d"], n_layer=cfg["layers"],
                            W=cfg["W"], rhos=tuple(cfg["rhos"]),
                            lam=cfg["lam"], maxL=cfg["maxL"], seed=cfg["seed"])
    z = np.load(CKPT)
    for n, p in model.params().items():
        np.copyto(p.data, z["p:" + n])
    model.step = int(z["_step"])
    return model


def sample(logits, temp, top_p, rng):
    z = logits / temp
    z = z - z.max()
    e = np.exp(np.clip(z, -60, 60))
    p = e / e.sum()
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    keep = order[:int(np.searchsorted(cum, top_p)) + 1]
    pk = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=pk))


def generate(model, stoi, itos, prompt, n, temp, top_p, rng):
    nn.set_no_grad(True)
    ids = [stoi.get(c, 0) for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-model.maxL:]], dtype=np.int64)
        logits = model.logits(ctx)[0, -1, :]
        ids.append(sample(logits, temp, top_p, rng))
    nn.set_no_grad(False)
    return "".join(itos[i] for i in ids[len(prompt):])


def metrics(text):
    grams = [text[i:i + 3] for i in range(max(0, len(text) - 2))]
    rep3 = 1.0 - len(set(grams)) / max(1, len(grams))
    longest = 1
    run = 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        longest = max(longest, run)
    words = [w for w in text.split() if len(w) > 1]
    return dict(chars=len(text), distinct_chars=len(set(text)),
                rep3=round(rep3, 4), longest_same_char_run=longest,
                words=len(words), distinct_words=len(set(words)),
                has_letters=bool(any(c.isalpha() for c in text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(RES, "step7_generation.json"))
    a = ap.parse_args()
    z = np.load(CKPT)
    cfg = json.loads(str(z["_cfg"]))
    cfg["rhos"] = tuple(cfg["rhos"]) if isinstance(cfg["rhos"], list) else cfg["rhos"]
    model = load_model(cfg)
    _, _, V, stoi, _ = build_corpus()
    itos = {i: c for c, i in stoi.items()}
    rng = np.random.default_rng(a.seed)
    out = dict(arch_step=arch.ARCH_STEP, ckpt_step=model.step,
               params=sum(p.size for p in model.params().values()),
               temp=a.temp, top_p=a.top_p, seed=a.seed, samples=[])
    for pr in PROMPTS:
        txt = generate(model, stoi, itos, pr, a.n, a.temp, a.top_p, rng)
        m = metrics(txt)
        out["samples"].append(dict(prompt=pr, text=txt, metrics=m))
        print("PROMPT", repr(pr))
        print(txt[:400].replace("\n", "\\n"))
        print("  ", m)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
