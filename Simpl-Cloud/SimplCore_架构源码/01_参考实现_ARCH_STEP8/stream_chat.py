#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone O(1)-state SimplCore LM chat. Pure NumPy, no torch, no network.

    python stream_chat.py
    python stream_chat.py "Q: Who goes there?\\nA:" --n 200
    python stream_chat.py "HAMLET:\\n" --temp 0.8 --seed 7

Unlike ``chat.py`` (which re-runs the whole prefix for every token), this uses
``stream_lm.StreamSimplCoreLM``: a fixed-size ring buffer, K IIR states and one
outer-product state per layer.  ``verify_final.py`` checks it against the full
unroll on the trained weights to < 1e-9.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stream_lm as S  # noqa: E402

cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
vocab = json.load(open(os.path.join(HERE, "vocab.json"), encoding="utf-8"))
stoi = {c: int(i) for c, i in vocab["stoi"].items()}
itos = {int(i): c for c, i in vocab["stoi"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    m, _ = S.load_trained(HERE)
    sm = S.StreamSimplCoreLM(m)
    prompts = ([a.prompt] if a.prompt else
               ["First Citizen:\n", "HAMLET:\n",
                "Q: What is your name?\nA:", "Q: Shall we proceed?\nA:",
                "Q: Who goes there?\nA:", "To be, or not to be,"])
    for pr in prompts:
        ids = sm.generate([stoi.get(c, 0) for c in pr], n_new=a.n,
                          temp=a.temp, top_p=a.top_p, seed=a.seed)
        print("PROMPT:", repr(pr))
        print("".join(itos[i] for i in ids[len(pr):]))
        print("-" * 60)


if __name__ == "__main__":
    main()
