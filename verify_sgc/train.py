"""
SGC vs Attention 下游有效性验证 —— 训练脚本（纯自研 simplcore.tensor 后端）

任务：Selective Copy（长程依赖探针）
  - 序列长度 L=64，词表 = {0:PAD, 1:START, 2..9:记号token, 10:COPY}
  - 在序列前段随机选 n_copy=8 个位置放入记号 token；
  - 在第 L-n_copy 位起连续放入 n_copy 个 COPY 占位；
  - 模型需在这些 COPY 位置上，按原顺序复现那 8 个记号。
  - 这是一个强制"跨长距离全局混合"的任务，正好检验 SGC 的谱全局卷积能力。

两个模型唯一区别：block 内混合层 = AttentionMixer / SGCMixer。
指标：在 COPY 位置上的 token 精确匹配准确率 + cross_entropy loss。
"""
import os, sys, json, time, random, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simplcore.tensor as torch
from simplcore.tensor import nn
from simplcore.tensor.tensor import Tensor

from model import SeqModel, count_params

# ---------------- 配置 ----------------
L = 48
N_COPY = 6
V_COPY = 10          # COPY token id
V_START = 1
V_SIGNAL = 2         # 记号 token 起始 id (2..9 共 8 个)
N_SIGNAL = 8
VOCAB = V_COPY + 1   # 0=PAD,1=START,2..9 signal,10=COPY  => 11
DIM = 64
N_LAYERS = 2
N_HEADS = 4
BATCH = 32
STEPS = 400
LR = 1e-3
SEED = 1234

rng = np.random.default_rng(SEED)


def make_batch(bsz):
    """返回 (idx_np, tgt_np)，仅 COPY 位置参与 loss。"""
    idx = np.zeros((bsz, L), dtype=np.int64)
    tgt = np.full((bsz, L), -100, dtype=np.int64)  # -100 = ignore
    for b in range(bsz):
        idx[b, 0] = V_START
        # 随机选 n_copy 个信号位置（分布在前 L/2 段，制造长距）
        pos = rng.choice(L // 2, size=N_COPY, replace=False)
        sig = rng.integers(V_SIGNAL, V_SIGNAL + N_SIGNAL, size=N_COPY)
        for p, s in zip(pos, sig):
            idx[b, p] = s
        # COPY 区：末尾 n_copy 位
        copy_pos = np.arange(L - N_COPY, L)
        idx[b, copy_pos] = V_COPY
        tgt[b, copy_pos] = sig  # 目标：按序复现
    return idx, tgt


def selective_copy_acc(logits, tgt):
    """logits:(B,L,V) tgt:(B,L) 仅统计 tgt>=0 位置"""
    B, L, V = logits.shape
    correct = 0
    total = 0
    for b in range(B):
        for l in range(L):
            if tgt[b, l] >= 0:
                total += 1
                if int(np.argmax(logits[b, l])) == tgt[b, l]:
                    correct += 1
    return correct, total


def run(mixer, tag):
    print(f"\n===== {tag} ({mixer}) =====")
    torch.manual_seed(SEED)
    model = SeqModel(vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS,
                     n_heads=N_HEADS, mixer=mixer)
    nparams = count_params(model)
    print(f"params: {nparams:,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    val_idx, val_tgt = make_batch(200)
    history = []

    t0 = time.perf_counter()
    for step in range(STEPS):
        idx, tgt = make_batch(BATCH)
        opt.zero_grad()
        logits_t = model(torch.tensor(idx))
        logits = logits_t  # (B,L,V) Tensor
        # cross_entropy: 对 tgt>=0 位置
        loss = torch.cross_entropy(logits.reshape(-1, VOCAB),
                               torch.tensor(tgt.reshape(-1)),
                               ignore_index=-100)
        loss.backward()
        opt.step()
        loss_val = float(loss.data.flatten()[0])
        # 释放计算图，避免自研后端图累积导致内存爆炸/segfault
        del loss, logits_t
        gc.collect()

        if step % 100 == 0 or step == STEPS - 1:
            # 验证集准确率
            with torch.no_grad():
                vlogits = model(torch.tensor(val_idx)).data
            c, t = selective_copy_acc(vlogits, val_tgt)
            acc = c / t if t else 0.0
            print(f"  step {step:4d}  loss={loss_val:.4f}  val_acc={acc*100:.2f}%")
            history.append({"step": step, "loss": loss_val, "val_acc": acc})

    dt = time.perf_counter() - t0
    # 末步验证
    with torch.no_grad():
        vlogits = model(torch.tensor(val_idx)).data
    c, t = selective_copy_acc(vlogits, val_tgt)
    final_acc = c / t if t else 0.0
    print(f"  >>> {tag} final val_acc={final_acc*100:.2f}%  time={dt:.1f}s")
    return {"tag": tag, "mixer": mixer, "params": nparams,
            "final_val_acc": final_acc, "time_s": dt, "history": history}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    results.append(run("attn", "Attention-Baseline"))
    results.append(run("sgc", "SGC-Model"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"  {r['tag']:20s} params={r['params']:>9,}  final_val_acc={r['final_val_acc']*100:5.2f}%  time={r['time_s']:.0f}s")
    # 结论
    a = results[0]["final_val_acc"]; s = results[1]["final_val_acc"]
    better = "SGC" if s > a else ("Attention" if a > s else "tie")
    print(f"  => 相对表现: {better}  (Δ={(s-a)*100:+.2f}pp)")


if __name__ == "__main__":
    main()
