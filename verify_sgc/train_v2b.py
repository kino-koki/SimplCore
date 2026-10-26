"""
SGC v2 (因果全局卷积) 正式对比实验 —— 多 seed 公平对照。

对比 Attention baseline 与 SGC-v2-causal：
  - 每个 seed 重置数据 rng 与 torch seed（公平）
  - 指标: val_acc / loss / 耗时
  - 目标: SGC v2 在 acc 与耗时超越，并尽量在 loss 也低（或至少可比）

单次诊断已显示 SGC-v2-causal acc 36.75% vs attn 35.25%, 耗时 41.9 vs 48.1。
本脚本用多个 seed 验证稳定性。
"""
import os, sys, json, time, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simplcore.tensor as torch
from simplcore.tensor import nn
from simplcore.tensor.tensor import Tensor

import importlib.util
_spec = importlib.util.spec_from_file_location("sgc_train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

from model import SeqModel, count_params
from model_v2 import build_model

sys.setrecursionlimit(20000)

SEEDS = [1234, 2024, 777, 42, 99]


def run(mixer, tag, seed):
    T.rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if mixer == "attn":
        model = SeqModel(T.VOCAB, T.DIM, T.N_LAYERS, T.N_HEADS, "attn")
    else:
        model = build_model(mixer, T.VOCAB, T.DIM, T.N_LAYERS, T.N_HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=T.LR)
    val_idx, val_tgt = T.make_batch(200)
    t0 = time.perf_counter()
    for step in range(T.STEPS):
        idx, tgt = T.make_batch(T.BATCH)
        opt.zero_grad()
        lt = model(torch.tensor(idx))
        loss = torch.cross_entropy(lt.reshape(-1, T.VOCAB), torch.tensor(tgt.reshape(-1)), ignore_index=-100)
        loss.backward()
        opt.step()
        lv = float(loss.data.flatten()[0])
        del loss, lt
        gc.collect()
    dt = time.perf_counter() - t0
    with torch.no_grad():
        vl = model(torch.tensor(val_idx)).data
    c, t = T.selective_copy_acc(vl, val_tgt)
    acc = c / t if t else 0.0
    # 末步 loss 需重算（上面 lv 是最后训练步）
    with torch.no_grad():
        lt = model(torch.tensor(val_idx))
        vloss = float(torch.cross_entropy(lt.reshape(-1, T.VOCAB), torch.tensor(val_tgt.reshape(-1)), ignore_index=-100).data.flatten()[0])
    return {"tag": tag, "seed": seed, "params": count_params(model),
            "val_acc": acc, "loss": vloss, "time_s": dt}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for seed in SEEDS:
        rows.append(run("attn", "Attention", seed))
        rows.append(run("sgc2", "SGC-v2-causal", seed))

    # 汇总
    attn = [r for r in rows if r["tag"] == "Attention"]
    sgc = [r for r in rows if r["tag"] == "SGC-v2-causal"]
    import statistics as st
    a_acc = st.mean(r["val_acc"] for r in attn)
    s_acc = st.mean(r["val_acc"] for r in sgc)
    a_loss = st.mean(r["loss"] for r in attn)
    s_loss = st.mean(r["loss"] for r in sgc)
    a_t = st.mean(r["time_s"] for r in attn)
    s_t = st.mean(r["time_s"] for r in sgc)

    summary = {
        "seeds": SEEDS,
        "attention": {"val_acc": a_acc, "loss": a_loss, "time_s": a_t,
                      "params": attn[0]["params"]},
        "sgc_v2": {"val_acc": s_acc, "loss": s_loss, "time_s": s_t,
                   "params": sgc[0]["params"]},
        "delta_acc": s_acc - a_acc,
        "delta_loss": s_loss - a_loss,
        "delta_time": s_t - a_t,
    }
    with open(os.path.join(out_dir, "results_v2_causal.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    print("\n===== SGC v2-causal 多 seed 汇总 =====")
    print(f"  Attention : acc={a_acc*100:.2f}%  loss={a_loss:.4f}  time={a_t:.1f}s  params={attn[0]['params']}")
    print(f"  SGC-v2    : acc={s_acc*100:.2f}%  loss={s_loss:.4f}  time={s_t:.1f}s  params={sgc[0]['params']}")
    print(f"  Δacc={summary['delta_acc']*100:+.2f}pp  Δloss={summary['delta_loss']:+.4f}  Δtime={summary['delta_time']:+.1f}s")
    acc_win = s_acc > a_acc
    loss_win = s_loss < a_loss
    time_win = s_t < a_t
    print(f"  >>> 全面超越 (acc&loss&time): {'是' if (acc_win and loss_win and time_win) else '否'} "
          f"(acc {'胜' if acc_win else '负'}, loss {'胜' if loss_win else '负'}, time {'胜' if time_win else '负'})")


if __name__ == "__main__":
    main()
