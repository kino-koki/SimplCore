# -*- coding: utf-8 -*-
"""End-to-end REAL training of the upgraded SimplCore LM (ARCH_STEP 7).

Corpus: tiny Shakespeare, character level, chronological 90/10 split,
plus a Q/A-formatted stream extracted from the script's own dialogue so the
model learns to answer a prompt. No synthetic key/value task, no injected
oracle, no teacher weights, no renamed checkpoint.

Usage:
  python train.py --steps 600 --resume results/step7_ckpt.npz
  python train.py --steps 20 --smoke      # speed / gradcheck only
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)                     # .../realtrain
WS = os.path.dirname(RT)                       # workspace root
ARCH = os.path.join(WS, "_cloud_repo", "SimplCore-main", "simplcore",
                    "core", "arch")
sys.path.insert(0, ARCH)
import nn                      # noqa: E402
import simplcore_lm as arch    # noqa: E402

CORPUS = os.path.join(RT, "data", "tinyshakespeare.txt")
RES = os.path.join(RT, "results")
os.makedirs(RES, exist_ok=True)
CKPT = os.path.join(RES, "step7_ckpt.npz")


# ---------------------------------------------------------------------------
def build_corpus():
    raw = open(CORPUS, encoding="utf-8").read()
    lines = [l.strip() for l in raw.splitlines()]
    qa = []
    for i, l in enumerate(lines[:-1]):
        if "?" in l and 8 < len(l) < 120 and lines[i + 1] and len(lines[i + 1]) < 160:
            qa.append("Q: %s\nA: %s\n\n" % (l, lines[i + 1]))
    qa_text = "".join(qa)
    cut = int(0.9 * len(raw))
    # Q/A templates are TRAIN-ONLY; validation is the untouched script tail.
    train_text = raw[:cut] + "\n" + qa_text + "\n" + qa_text
    val_text = raw[cut:]
    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    tr = np.array([stoi[c] for c in train_text], dtype=np.int64)
    va = np.array([stoi[c] for c in val_text], dtype=np.int64)
    return tr, va, len(chars), stoi, len(qa)


def batch(data, B, L, rng):
    hi = len(data) - L - 1
    idx = rng.integers(0, hi, size=B)
    X = np.stack([data[i:i + L + 1] for i in idx])
    return X[:, :-1], X[:, 1:]


def eval_bpc(model, val, B, L, seed, n_batch=12):
    rng = np.random.default_rng(seed + 991)
    vals = []
    nn.set_no_grad(True)
    for _ in range(n_batch):
        tok, tgt = batch(val, B, L, rng)
        logits = model.logits(tok)
        z = logits - logits.max(-1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(-1, keepdims=True)
        pt = np.take_along_axis(p, tgt[..., None], -1)[..., 0]
        vals.append(float((-np.log(np.maximum(pt, 1e-12)) / math.log(2.0)).mean()))
    nn.set_no_grad(False)
    return float(np.mean(vals)), float(np.std(vals))


def gradcheck(model, tok, tgt, seed=0, n_params=14):
    rng = np.random.default_rng(seed + 7)
    nn.set_no_grad(False)
    loss = model.loss(tok, tgt)
    loss.backward()
    ps = model.params()
    names = list(ps)
    rng.shuffle(names)
    out = []
    for name in names[:n_params]:
        p = ps[name]
        j = int(rng.integers(p.size))
        ana = 0.0 if p.grad is None else float(p.grad.ravel()[j])
        old = float(p.data.ravel()[j])
        eps = 1e-5
        p.data.ravel()[j] = old + eps
        lp = model.loss(tok, tgt).data.item()
        nn.clear_tape()
        p.data.ravel()[j] = old - eps
        lm = model.loss(tok, tgt).data.item()
        nn.clear_tape()
        p.data.ravel()[j] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
        out.append(dict(param=name, analytic=ana, numeric=num, rel_err=rel))
    worst = max(o["rel_err"] for o in out)
    return dict(n=len(out), worst_rel_err=worst, passed=bool(worst < 2e-4),
                details=out)


def save(model, st, step, rng, cfg):
    d = {}
    for n, p in model.params().items():
        d["p:" + n] = p.data
        d["m:" + n] = st[n][0]
        d["v:" + n] = st[n][1]
    d["_step"] = np.array(step, dtype=np.int64)
    d["_cfg"] = json.dumps(cfg)
    np.savez_compressed(CKPT, **d)


def load(model, cfg):
    if not os.path.exists(CKPT):
        st = {n: (np.zeros_like(p.data), np.zeros_like(p.data))
              for n, p in model.params().items()}
        return st, 0, np.random.default_rng(cfg["seed"] + 31337)
    z = np.load(CKPT)
    ps = model.params()
    for n, p in ps.items():
        np.copyto(p.data, z["p:" + n])
    st = {n: (z["m:" + n].copy(), z["v:" + n].copy()) for n in ps}
    step = int(z["_step"])
    rng = np.random.default_rng(cfg["seed"] + 31337 + step)
    return st, step, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--total", type=int, default=6000,
                    help="total schedule length for cosine decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log-every", type=int, default=200)
    a = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    train, val, V, stoi, n_qa = build_corpus()
    cfg = dict(V=V, d=a.d, layers=a.layers, W=12,
               rhos=(0.6, 0.9, 0.99), lam=0.9, seed=a.seed,
               maxL=512, n_qa=n_qa)
    model = arch.SimplCoreLM(V, d=a.d, n_layer=a.layers, seed=a.seed)
    print("corpus: train=%d val=%d V=%d qa_pairs=%d params=%d" %
          (len(train), len(val), V, n_qa,
           sum(p.size for p in model.params().values())))

    rng = np.random.default_rng(a.seed + 31337)
    tok, tgt = batch(train, a.batch, a.seq, rng)
    gc = gradcheck(model, tok, tgt, seed=a.seed)
    print("GRADCHECK", gc["n"], "worst_rel_err=%.3e" % gc["worst_rel_err"],
          "PASS" if gc["passed"] else "FAIL")
    if not gc["passed"]:
        json.dump(gc, open(os.path.join(RES, "step7_gradcheck_FAIL.json"), "w"), indent=1)
        sys.exit(2)

    if a.smoke:
        t0 = time.time()
        st = {n: (np.zeros_like(p.data), np.zeros_like(p.data))
              for n, p in model.params().items()}
        for i in range(1, 21):
            tk, tg = batch(train, a.batch, a.seq, rng)
            ls = model.loss(tk, tg)
            ls.backward()
            adam(model, st, i, a.lr, a.total)
            for p in model.params().values():
                p.zero_grad()
        dt = time.time() - t0
        b, s = eval_bpc(model, val, a.batch, a.seq, a.seed, n_batch=3)
        print("SMOKE 20 steps %.2fs (%.3fs/step) val_bpc=%.4f" %
              (dt, dt / 20, b))
        json.dump(dict(gradcheck=gc, sec_per_step=dt / 20, val_bpc=b),
                  open(os.path.join(RES, "step7_smoke.json"), "w"), indent=1)
        return

    st, step0, rng = (load(model, cfg) if (a.resume and not a.fresh)
                      else ({n: (np.zeros_like(p.data), np.zeros_like(p.data))
                             for n, p in model.params().items()},
                            0, np.random.default_rng(a.seed + 31337)))
    print("resume from step", step0)
    b0, _ = eval_bpc(model, val, a.batch, a.seq, a.seed)
    print("start val_bpc=%.4f" % b0)
    hist = []
    t0 = time.time()
    for i in range(1, a.steps + 1):
        step = step0 + i
        tk, tg = batch(train, a.batch, a.seq, rng)
        ls = model.loss(tk, tg)
        ls.backward()
        adam(model, st, step, a.lr, a.total)
        for p in model.params().values():
            p.zero_grad()
        if i % max(1, a.log_every) == 0 or i == a.steps:
            b, s = eval_bpc(model, val, a.batch, a.seq, a.seed, n_batch=6)
            hist.append(dict(step=step, train_loss=float(ls.data),
                             val_bpc=b, val_std=s, sec=round(time.time() - t0, 1)))
            print("step %5d  train %.4f  val_bpc %.4f  %.1fs" %
                  (step, ls.data, b, time.time() - t0), flush=True)
            save(model, st, step, rng, cfg)
            json.dump(dict(arch_step=arch.ARCH_STEP, step=step,
                           history=hist, end_val_bpc=hist[-1]["val_bpc"]),
                      open(os.path.join(RES, "step7_progress.json"), "w"), indent=1)
    save(model, st, step0 + a.steps, rng, cfg)
    out = dict(arch_step=arch.ARCH_STEP, gradcheck=gc, start_step=step0,
               end_step=step0 + a.steps, steps=a.steps, batch=a.batch,
               seq=a.seq, lr=a.lr, total_schedule=a.total,
               params=sum(p.size for p in model.params().values()),
               start_val_bpc=b0, end_val_bpc=hist[-1]["val_bpc"],
               history=hist, wall_sec=round(time.time() - t0, 2))
    json.dump(out, open(os.path.join(RES, "step7_train_last.json"), "w"), indent=1)
    print("SAVED", CKPT, "end_val_bpc=%.4f" % out["end_val_bpc"])


def adam(model, st, step, lr0, total):
    lr = lr0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total)))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for n, p in model.params().items():
        m, v = st[n]
        g = np.zeros_like(p.data) if p.grad is None else p.grad
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        p.data -= lr * mh / (np.sqrt(vh) + eps)


if __name__ == "__main__":
    main()
