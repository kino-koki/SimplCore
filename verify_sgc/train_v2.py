"""
SGC v2 对比实验（纯自研 simplcore.tensor 后端）

对比三种 mixer 在 selective-copy 任务上的表现（同等规模、同等配置）:
  - attn  : Attention baseline (causal multi-head)
  - sgc   : SGC v1 (原三件套 + 8-band 瓶颈)
  - sgc2  : SGC v2 (无瓶颈谱卷积 + 轻量因果局部修正)

指标: val_acc (COPY 位置精确匹配) / cross_entropy loss / 训练耗时。
目标: SGC v2 在 loss、val_acc、耗时三项全面超越 Attention baseline。

同时额外跑一个 longer-run (STEPS*3) 验证 SGC v2 的收敛优势是否随步数扩大。
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

from model_v2 import build_model, count_params

sys.setrecursionlimit(20000)


def run(mixer, tag, steps=None, lr=None):
    steps = steps or T.STEPS
    lr = lr or T.LR
    print(f"\n===== {tag} ({mixer}) steps={steps} lr={lr} =====")
    torch.manual_seed(T.SEED)
    model = build_model(mixer, T.VOCAB, T.DIM, T.N_LAYERS, T.N_HEADS)
    nparams = count_params(model)
    print(f"params: {nparams:,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    val_idx, val_tgt = T.make_batch(200)
    history = []

    t0 = time.perf_counter()
    for step in range(steps):
        idx, tgt = T.make_batch(T.BATCH)
        opt.zero_grad()
        logits_t = model(torch.tensor(idx))
        loss = torch.cross_entropy(logits_t.reshape(-1, T.VOCAB),
                               torch.tensor(tgt.reshape(-1)),
                               ignore_index=-100)
        loss.backward()
        opt.step()
        loss_val = float(loss.data.flatten()[0])
        del loss, logits_t
        gc.collect()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                vlogits = model(torch.tensor(val_idx)).data
            c, t = T.selective_copy_acc(vlogits, val_tgt)
            acc = c / t if t else 0.0
            print(f"  step {step:4d}  loss={loss_val:.4f}  val_acc={acc*100:.2f}%")
            history.append({"step": step, "loss": loss_val, "val_acc": acc})

    dt = time.perf_counter() - t0
    with torch.no_grad():
        vlogits = model(torch.tensor(val_idx)).data
    c, t = T.selective_copy_acc(vlogits, val_tgt)
    final_acc = c / t if t else 0.0
    print(f"  >>> {tag} final val_acc={final_acc*100:.2f}%  time={dt:.1f}s")
    return {"tag": tag, "mixer": mixer, "params": nparams,
            "final_val_acc": final_acc, "final_loss": history[-1]["loss"],
            "time_s": dt, "history": history}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    # 同等规模、同等步数对比
    results.append(run("attn", "Attention-Baseline"))
    results.append(run("sgc", "SGC-v1"))
    results.append(run("sgc2", "SGC-v2"))
    # 更长训练，验证收敛优势
    results.append(run("sgc2", "SGC-v2-Long", steps=T.STEPS * 3))

    with open(os.path.join(out_dir, "results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== V2 COMPARISON SUMMARY =====")
    for r in results:
        print(f"  {r['tag']:20s} params={r['params']:>9,}  val_acc={r['final_val_acc']*100:5.2f}%  "
              f"loss={r['final_loss']:.4f}  time={r['time_s']:.0f}s")

    # 结论：SGC-v2 是否全面超越 attn
    attn = results[0]
    sgc2 = results[2]
    acc_better = sgc2["final_val_acc"] > attn["final_val_acc"]
    loss_better = sgc2["final_loss"] < attn["final_loss"]
    time_better = sgc2["time_s"] < attn["time_s"]
    all_better = acc_better and loss_better and time_better
    print(f"\n  SGC-v2 vs Attention: acc {sgc2['final_val_acc']*100:.2f}% vs {attn['final_val_acc']*100:.2f}% "
          f"({'超越' if acc_better else '未超越'}) | "
          f"loss {sgc2['final_loss']:.4f} vs {attn['final_loss']:.4f} ({'更低' if loss_better else '更高'}) | "
          f"time {sgc2['time_s']:.0f}s vs {attn['time_s']:.0f}s ({'更快' if time_better else '更慢'})")
    print(f"  >>> 全面超越: {'是' if all_better else '否'}")


if __name__ == "__main__":
    main()
