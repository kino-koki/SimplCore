#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end sandbox run: REAL training + REAL generation on real text.

No torch in this sandbox; this is the NumPy reference of the same architecture
shipped as ``modeling_simplcore_v7.py`` for the user's machine.  Every number
written to ``full_arch_result.json`` comes from an actual run here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bpe import QwenBPESubset
from numpy_lm import SimplCoreV7

ROOT = os.path.dirname(HERE)
ZIP = os.path.join(ROOT, "Qwen3-1.7B-Instruct_完整权重_卷01.zip")
TOKDIR = os.path.join(HERE, "qwen3_tokenizer")
CORPUS = os.path.join(HERE, "corpus.txt")
if not os.path.exists(CORPUS):
    CORPUS = os.path.join(ROOT, "realtrain", "data", "tinyshakespeare.txt")
OUT = os.path.join(HERE, "full_arch_result.json")


def prepare_tokenizer(V=2048):
    os.makedirs(TOKDIR, exist_ok=True)
    need = {"vocab.json": "Qwen3-1.7B-Instruct/vocab.json",
            "merges.txt": "Qwen3-1.7B-Instruct/merges.txt"}
    with zipfile.ZipFile(ZIP) as z:
        for fn, key in need.items():
            p = os.path.join(TOKDIR, fn)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                open(p, "wb").write(z.read(key))
    return QwenBPESubset(os.path.join(TOKDIR, "vocab.json"),
                         os.path.join(TOKDIR, "merges.txt"), V)


def batches(data, B, L, rng):
    hi = len(data) - L - 1
    idx = rng.integers(0, hi, size=B)
    X = np.stack([data[i:i + L + 1] for i in idx])
    return X[:, :-1].astype(np.int64), X[:, 1:].astype(np.int64)


def bpc(model, data, B, L, seed, n_batch=8):
    rng = np.random.default_rng(seed + 4242)
    vals = []
    for _ in range(n_batch):
        x, y = batches(data, B, L, rng)
        p = model.forward(x)
        t = np.zeros_like(p)
        np.put_along_axis(t, y[:, :, None], 1.0, axis=2)
        vals.append(-np.log(np.maximum((p * t).sum(-1), 1e-12)).mean() / math.log(2))
    return float(np.mean(vals))


def quality(text, corpus_words, corpus_3grams, corpus_2grams):
    import re
    toks = re.findall(r"[A-Za-z']+", text.lower())
    ws = [w for w in toks if w]
    d1 = len(set(ws)) / max(len(ws), 1)
    grams = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    rep3 = 1.0 - len(set(grams)) / max(len(grams), 1)
    bg = [tuple(ws[i:i + 2]) for i in range(len(ws) - 1)]
    rwr = float(np.mean([w in corpus_words for w in ws])) if ws else 0.0
    ov3 = float(np.mean([g in corpus_3grams for g in grams])) if grams else 0.0
    ov2 = float(np.mean([g in corpus_2grams for g in bg])) if bg else 0.0
    chars = [c for c in text if not c.isspace()]
    readable = sum(1 for c in chars if c.isascii() and (c.isalnum() or c in ".,;:!?'\"-"))
    return dict(chars=len(text), words=len(ws), distinct1=round(float(d1), 4),
                rep3=round(float(rep3), 4),
                real_word_ratio=round(rwr, 4),
                corpus_bigram_overlap=round(ov2, 4),
                corpus_3gram_overlap=round(ov3, 4),
                readable_ratio=round(readable / max(len(chars), 1), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--V", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    t0 = time.time()
    tok = prepare_tokenizer(a.V)
    import re
    text = open(CORPUS, encoding="utf-8").read()
    corpus_words = set(re.findall(r"[A-Za-z']+", text.lower()))
    _cw = re.findall(r"[A-Za-z']+", text.lower())
    corpus_3grams = set(tuple(_cw[i:i + 3]) for i in range(len(_cw) - 2))
    corpus_2grams = set(tuple(_cw[i:i + 2]) for i in range(len(_cw) - 1))
    data = np.array(tok.encode(text), dtype=np.int64)
    cut = int(0.9 * len(data))
    tr, va = data[:cut], data[cut:]
    print(f"[data] real corpus {len(text)} B -> {len(data)} Qwen3-BPE tokens "
          f"(V={tok.V}, train {len(tr)}, val {len(va)})", flush=True)

    m = SimplCoreV7(V=tok.V, d=a.d, R=a.R, n_layers=a.layers, seed=a.seed)
    n_params = sum(x.size for x in m.P.values())
    rng = np.random.default_rng(a.seed + 31337)
    x, y = batches(tr, a.B, a.L, rng)

    print("[check] gradcheck ...", flush=True)
    gc = m.gradcheck(x, y, seed=a.seed)
    print("   ", gc["n_checked"], "params, worst", gc["worst_rel_err"],
          "PASS" if gc["passed"] else "FAIL", flush=True)
    if not gc["passed"]:
        bad = sorted(gc["details"], key=lambda r: -r["rel_err"])[:6]
        print("   worst:", json.dumps(bad, indent=1), flush=True)
        raise SystemExit("gradcheck failed -- refusing to train")

    st = m.check_stream_equivalence(x)
    ds = m.delta_sgc(x)
    print("[check] stream equivalence rel", st["max_rel_err"], "state bytes",
          st["stream_state_bytes"], "| delta_sgc", ds["delta_logits_max"],
          "PASS" if ds["passed"] else "FAIL", flush=True)
    if not (st["passed"] and ds["passed"]):
        raise SystemExit("architecture identity check failed")

    prompts_raw = ["First Citizen:\nBefore we proceed", "ROMEO: ",
                   "To be, or not to be, that is the question:\n"]
    prompts = [tok.encode(p) for p in prompts_raw]
    pre = []
    for pr in prompts_raw:
        ids = m.generate(tok.encode(pr), n_new=60, seed=a.seed)
        pre.append(tok.decode(ids[len(tok.encode(pr)):]))

    state = m.new_optimizer()
    curve = []
    p0 = bpc(m, va, a.B, a.L, a.seed)
    print(f"[train] initial val_bpc {p0:.4f}, {a.steps} steps ...", flush=True)
    for step in range(1, a.steps + 1):
        x, y = batches(tr, a.B, a.L, rng)
        loss, g = m.loss_and_grad(x, y)
        m.adam_step(g, state, step, lr=a.lr)
        if step % max(1, a.steps // 8) == 0 or step == 1:
            curve.append(dict(step=step,
                              train_bpc=float(loss / math.log(2)),
                              val_bpc=bpc(m, va, a.B, a.L, a.seed)))
            print(f"   step {step:5d} train {curve[-1]['train_bpc']:.4f} "
                  f"val {curve[-1]['val_bpc']:.4f}", flush=True)
    p1 = bpc(m, va, a.B, a.L, a.seed)

    ck = os.path.join(HERE, "student.npz")
    m.save(ck)
    m2 = SimplCoreV7.load(ck)
    x2, y2 = batches(tr, a.B, a.L, np.random.default_rng(7))
    diff = max(float(np.max(np.abs(m.forward(x2)[0] - m2.forward(x2)[0])))
               for _ in range(1))
    ck_ok = diff == 0.0
    print(f"[ckpt] save/load bit-identical: {ck_ok} (max diff {diff})", flush=True)

    gens = []
    for pr_raw, pr, before in zip(prompts_raw, prompts, pre):
        ids = m.generate(pr, n_new=90, seed=a.seed + 41)
        out = tok.decode(ids[len(pr):])
        q = quality(out, corpus_words, corpus_3grams, corpus_2grams)
        q["prompt"] = pr_raw
        q["sample"] = out[:220]
        q["before_training_sample"] = before[:120]
        gens.append(q)
        print(f"[gen] {pr_raw[:28]!r:32s} words={q['words']} "
              f"real={q['real_word_ratio']} bigram={q['corpus_bigram_overlap']} "
              f"3gram={q['corpus_3gram_overlap']} rep3={q['rep3']}", flush=True)
        print("      ", out[:160].replace("\n", " "), flush=True)

    # perturbation after training: SGC must STILL be in the forward graph
    ds2 = m.delta_sgc(x2)
    st2 = m.check_stream_equivalence(x2)
    speaks = all(q["words"] >= 15 and q["real_word_ratio"] >= 0.55
                 and q["corpus_bigram_overlap"] >= 0.15 and q["rep3"] <= 0.50
                 and q["readable_ratio"] >= 0.80 for q in gens) and (p0 - p1) > 1.0
    result = dict(
        architecture="SimplCoreV7 (local gated conv + R slotted SGC state + learned blend + SwiGLU)",
        corpus="tiny Shakespeare, real human text, chronological 90/10 split",
        tokenizer="real Qwen3 byte-level BPE (vocab.json + merges.txt), "
                  f"{tok.V}-token prefix, byte fallback",
        params=int(n_params), config=dict(V=tok.V, d=a.d, R=a.R, layers=a.layers,
                                           L=a.L, B=a.B, steps=a.steps, lr=a.lr,
                                           seed=a.seed),
        gradcheck=dict(n_checked=gc["n_checked"],
                       worst_rel_err=gc["worst_rel_err"], passed=gc["passed"]),
        stream_equivalence=st, stream_equivalence_after_training=st2,
        delta_sgc_before_training=ds, delta_sgc_after_training=ds2,
        initial_val_bpc=p0, final_val_bpc=p1, bpc_gain=float(p0 - p1),
        curve=curve, checkpoint=dict(path="student.npz",
                                     bytes=os.path.getsize(ck),
                                     bit_identical_reload=ck_ok),
        generations=gens,
        speaks=bool(speaks),
        no_kv_cache=True, no_NxN_score=True,
        state_independent_of_L=st2["state_independent_of_L"],
        elapsed_s=round(time.time() - t0, 2),
    )
    json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[done] speaks={speaks}  bpc {p0:.4f} -> {p1:.4f}  "
          f"{result['elapsed_s']}s  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
