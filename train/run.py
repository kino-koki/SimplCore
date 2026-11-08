"""真实训练 / 迁移蒸馏 runner（纯 NumPy，分块续跑）。

三臂等预算对照：
  ts  : Transformer from scratch（同时充当 teacher）
  sgc : SimplCore 原生 from scratch
  kd  : SimplCore + teacher 输出分布蒸馏（主实验，teacher 冻结、黑盒）

用法:
  python run.py --arm ts  --total 3000 --chunk 600 --tag ts
  python run.py --arm sgc --total 3000 --chunk 600 --tag sgc
  python run.py --arm kd  --total 3000 --chunk 400 --tag kd --teacher ts
  python run.py --final        # 三臂评估 + 生成 + 判决
"""
import argparse
import json
import os
import pickle
import time
import numpy as np

import core as C
import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
B, L = 16, 64
EVAL_BATCHES = 8
T_KD = 5.0
SEED = 0


def state_path(tag):
    return os.path.join(HERE, f"ckpt_{tag}.pkl")


def log_path(tag):
    return os.path.join(HERE, f"log_{tag}.json")


def load_state(tag):
    with open(state_path(tag), "rb") as f:
        return pickle.load(f)


def save_state(tag, obj):
    with open(state_path(tag), "wb") as f:
        pickle.dump(obj, f, protocol=4)


def build():
    ap = os.path.join(HERE, "arr.npy")
    cp = os.path.join(HERE, "chars.json")
    if os.path.exists(ap) and os.path.exists(cp):
        arr = np.load(ap)
        chars = json.load(open(cp))
        c2i = {ch: i for i, ch in enumerate(chars)}
    else:
        text, c2i, chars = D.build_vocab()
        arr = D.encode(text, c2i)
        np.save(ap, arr)
        json.dump(chars, open(cp, "w"), ensure_ascii=False)
    tr, va = D.make_splits(arr)
    return tr, va, c2i, chars


def eval_ppl(P, va, arm, psi_cache=None):
    bt = D.Batcher(va, L, seed=12345)
    tot, n = 0.0, 0
    for _ in range(EVAL_BATCHES):
        x, y = bt.batch(B)
        lg, _ = C.forward(x, P)
        ls = C.log_softmax(lg, axis=-1)
        loss = -ls.reshape(-1, C.V)[np.arange(B * L), y.ravel()].mean()
        tot += float(loss); n += 1
    return float(np.exp(tot / n)), tot / n


def eval_kd_kl(P_s, P_t, va):
    """student 与 teacher 在 val 上的每位置 KL（teacher→student，T=1 口径）。"""
    bt = D.Batcher(va, L, seed=999)
    kls = []
    for _ in range(EVAL_BATCHES):
        x, y = bt.batch(B)
        ls_t, _ = C.forward(x, P_t)
        ls_s, _ = C.forward(x, P_s)
        pt = C.softmax(ls_t, axis=-1)
        kls.append(float((pt * (np.log(pt + 1e-9) - C.log_softmax(ls_s, axis=-1))).sum(-1).mean()))
    return float(np.mean(kls))


def generate(P, prompt, n=90, arm="sgc", c2i=None, chars=None,
             temp=0.85, topk=40, seed=17):
    """固定 seed 的 top-k 温度采样。greedy 会塌缩到高频字符（三臂都会），
    无法检验『是否成句』，故用采样；另返回 greedy 重复率作辅助指标。"""
    rng = np.random.default_rng(seed)
    ids = D.encode(prompt, c2i).astype(np.int32)
    out = list(ids)
    for _ in range(n):
        ctx = np.array(out[-L:], dtype=np.int32)[None, :]
        lg, _ = C.forward(ctx, P)
        z = lg[0, -1].astype(np.float64)
        k = min(topk, z.size)
        cand = np.argpartition(z, -k)[-k:]
        p = np.exp((z[cand] - z[cand].max()) / temp)
        p /= p.sum()
        out.append(int(rng.choice(cand, p=p)))
    g = out[len(ids):]
    gen = D.decode(g, chars)
    tri = [tuple(g[i:i + 3]) for i in range(len(g) - 2)]
    rep = 1 - len(set(tri)) / max(len(tri), 1)
    return gen, float(rep)


def verify_stream(P, c2i, chars):
    """验证 O(1) 流式 IIR 与整段 Toeplitz 前向一致（fp32）。"""
    bt = D.Batcher(np.arange(1000, dtype=np.int32) % C.V, L, seed=7)
    x, _ = bt.batch(1)
    lg_full, _ = C.forward(x, P)
    st = C.init_stream_state(P, 1)
    rows = []
    for t in range(x.shape[1]):
        lg_t = C.stream_step(x[:, t], P, st)
        rows.append(lg_t[0])
    lg_stream = np.stack(rows)[None, ...]
    d = np.abs(lg_full.astype(np.float64) - lg_stream.astype(np.float64))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rel=float(d.max() / max(np.abs(lg_full).max(), 1e-9)),
                state_bytes=int(sum(s.nbytes for s in st)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ts", "sgc", "kd"])
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=600)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--teacher", type=str, default="ts")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    tr, va, c2i, chars = build()

    if a.final:
        return do_final(va, c2i, chars)

    tag = a.tag or a.arm
    if os.path.exists(state_path(tag)):
        S = load_state(tag)
    else:
        P = C.init_params("sgc" if a.arm in ("sgc", "kd") else "ts", seed=a.seed)
        S = dict(P=P, opt=C.AdamW(P, lr=a.lr), step=0, hist=[], arm=a.arm,
                 total=a.total, lr=a.lr, seed=a.seed)
    P, opt = S["P"], S["opt"]
    P_t = load_state(a.teacher)["P"] if a.arm == "kd" else None
    bt = D.Batcher(tr, L, seed=1000 + a.seed)

    prompt = D.decode(va[:14], chars)
    t0 = time.time()
    while S["step"] < a.total and S["step"] < a.total:
        # 本 chunk
        end = min(S["step"] + a.chunk, a.total)
        losses = []
        for i in range(S["step"], end):
            x, y = bt.batch(B)
            alpha = 0.7 - 0.4 * (i / a.total)      # KD 权重 0.7 → 0.3
            if a.arm == "kd":
                lg_t, _ = C.forward(x, P_t)         # teacher 冻结、无反向
                lg_s, ca = C.forward(x, P)
                l_kd, dz_kd = C.kd_loss(lg_s, lg_t, T_KD)
                l_ce, dz_ce = C.ce_loss(lg_s, y)
                dz = (alpha * dz_kd + (1 - alpha) * dz_ce).astype(C.F32)
                loss = alpha * l_kd + (1 - alpha) * l_ce
                g = C.backward(dz, P, ca)
            else:
                lg, ca = C.forward(x, P)
                loss, dz = C.ce_loss(lg, y)
                g = C.backward(dz, P, ca)
            opt.step(P, g)
            losses.append(float(loss))
            if (i + 1) % 100 == 0:
                S["hist"].append(dict(step=i + 1, loss=float(np.mean(losses[-100:]))))
        S["step"] = end
        ppl, ce = eval_ppl(P, va, a.arm)
        gen, rep = generate(P, prompt, arm=a.arm, c2i=c2i, chars=chars)
        rec = dict(step=end, val_ppl=ppl, val_ce=ce, gen=gen[:120], rep3=rep,
                   train_loss=float(np.mean(losses[-100:])))
        if a.arm == "kd":
            rec["val_kl_T1"] = eval_kd_kl(P, P_t, va)
        S["hist"].append(rec)
        save_state(tag, S)
        with open(log_path(tag), "w") as f:
            json.dump(dict(arm=a.arm, total=a.total, lr=a.lr, seed=a.seed,
                           param_count=C.param_count(P), hist=S["hist"]), f,
                      indent=1, ensure_ascii=False)
        print(f"[{tag}] step {end}/{a.total}  ppl={ppl:.2f}  rep3={rep:.3f}  "
              f"time={time.time()-t0:.0f}s", flush=True)
        if "val_kl_T1" in rec:
            print(f"        val_KL(T=1)={rec['val_kl_T1']:.4f}", flush=True)
        print(f"        gen: {gen[:80]!r}", flush=True)
        break            # 每次调用只跑一个 chunk，外层 bash 循环续跑


def do_final(va, c2i, chars):
    out = {}
    P_t = load_state("ts")["P"]
    prompt = D.decode(va[:14], chars)
    for tag in ["ts", "sgc", "kd"]:
        S = load_state(tag); P = S["P"]
        ppl, ce = eval_ppl(P, va, tag)
        gen, rep = generate(P, prompt, arm=tag, c2i=c2i, chars=chars)
        kl = eval_kd_kl(P, P_t, va)
        row = dict(param_count=C.param_count(P), steps=S["step"],
                   val_ppl=ppl, val_ce=ce, kl_vs_teacher_T1=kl,
                   gen=gen[:200], rep3=rep)
        if tag == "sgc":
            row["stream_check"] = verify_stream(P, c2i, chars)
        out[tag] = row
        print(f"== {tag} == ppl={ppl:.3f} kl_T1={kl:.4f} rep3={rep:.3f}")
        print("   gen:", repr(gen[:120]))
    # 判决（判据在协议里先写死；预算不足如实标注）
    pts, psg, pkd = out["ts"]["val_ppl"], out["sgc"]["val_ppl"], out["kd"]["val_ppl"]
    ratio_kd = pkd / pts
    ratio_sg = psg / pts
    kd_scratch_gain = (psg - pkd) / psg
    if ratio_kd <= 1.10 and out["kd"]["rep3"] < out["sgc"]["rep3"]:
        verdict = "SUPPORTED"
    elif ratio_kd > 1.25 or kd_scratch_gain <= 0.0:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = dict(
        result=verdict,
        ratio_kd_vs_ts=float(ratio_kd),
        ratio_sgc_scratch_vs_ts=float(ratio_sg),
        kd_gain_over_sgc_scratch=float(kd_scratch_gain),
        token_budget_per_arm=int(3000 * B * L),
        chinchilla_20x_tokens=int(20 * out["ts"]["param_count"]),
        note="预算为前哨级（3.07M token/臂 = 3000 步 × 16 × 64；"
             "Chinchilla 20× 按 TS 臂 524,928 参数计为 10.50M）；"
             "结论按此预算给出，不外推到规模化训练。")
    with open(os.path.join(HERE, "verdict.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nVERDICT:", json.dumps(out["verdict"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
