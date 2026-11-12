# -*- coding: utf-8 -*-
"""Standalone SimplCore LM inference. Pure NumPy, no torch, no network.

    python chat.py                      # 6 demo prompts
    python chat.py "Q: Who goes there?\nA:"   # your own prompt
    python chat.py "HAMLET:\n" --n 300 --temp 0.8
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn
import simplcore_lm as A

D = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(D, "config.json"), encoding="utf8"))
vocab = json.load(open(os.path.join(D, "vocab.json"), encoding="utf8"))
stoi = {c: int(i) for c, i in vocab["stoi"].items()}
itos = {int(i): c for c, i in vocab["stoi"].items()}

m = A.SimplCoreLM(cfg["V"], d=cfg["d"], n_layer=cfg["layers"], W=cfg["W"],
                  rhos=tuple(cfg["rhos"]), lam=cfg["lam"],
                  maxL=cfg["maxL"], seed=cfg["seed"])
z = np.load(os.path.join(D, "model.npz"))
for n, p in m.params().items():
    np.copyto(p.data, z["p:" + n])


def sample(logits, temp, top_p, rng):
    z = logits / temp
    z = z - z.max()
    e = np.exp(np.clip(z, -60, 60))
    p = e / e.sum()
    o = np.argsort(p)[::-1]
    keep = o[:int(np.searchsorted(np.cumsum(p[o]), top_p)) + 1]
    pk = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=pk))


def chat(prompt, n=220, temp=0.85, top_p=0.92, seed=7):
    rng = np.random.default_rng(seed)
    nn.set_no_grad(True)
    ids = [stoi.get(c, 0) for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-m.maxL:]], dtype=np.int64)
        logits = m.logits(ctx)[0, -1, :]
        ids.append(sample(logits, temp, top_p, rng))
    nn.set_no_grad(False)
    return "".join(itos[i] for i in ids[len(prompt):])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    prompts = ([a.prompt] if a.prompt else
               ["First Citizen:\n", "HAMLET:\n",
                "Q: What is your name?\nA:", "Q: Shall we proceed?\nA:",
                "Q: Who goes there?\nA:", "To be, or not to be,"])
    for pr in prompts:
        print("PROMPT:", repr(pr))
        print(chat(pr, a.n, a.temp, a.top_p, a.seed))
        print("-" * 60)
