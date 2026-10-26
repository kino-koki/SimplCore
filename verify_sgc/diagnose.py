"""
SGC 落后 Attention 的根因诊断（纯自研 simplcore.tensor 后端）

假设清单：
H1. FFT 全局卷积是"无向"的（双向），而 selective-copy 是因果任务；
    但本脚本里 Attention 也用因果掩码，SGC 当前的 CSA 也是因果的。
    真正的差异：SGC 把序列整体做 FFT，把"位置信息"混进了频域相位，
    对"精确位置 copy"这类需要强位置绑定的任务，频域表示不如注意力直接。
H2. SGC 经过 in_proj(num_bands=8) 把 D=64 压到 8 个谱通道，信息瓶颈严重
    （64->8->64），而 Attention 保持 D=64 全维度交互。这是容量差距主因。
H3. 谱门控 + 谱注意力在短序列(L=48)上 F=25 个频率，低秩 rank=16 几乎退化为
    全连接，没有发挥 O(n log n) 优势，反而引入额外非线性噪声。
H4. 训练步数(400)是否不足？Attention 与 SGC 同_steps，若 SGC 学习更慢则需更多步。

本脚本量化 H2（通道瓶颈）与 H4（收敛速度），并对比 "纯谱卷积(无门控/无谱注意力)" 的
朴素 SGC 版本，确认额外模块是否帮倒忙。
"""
import os, sys, json, time, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simplcore.tensor as torch
from simplcore.tensor import nn
from simplcore.tensor.tensor import Tensor

from model import SeqModel, count_params

# 复用 verify_sgc/train.py 的配置与数据（显式指定模块路径，避免误引入根目录 train.py）
import importlib.util
_spec = importlib.util.spec_from_file_location("sgc_train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

sys.setrecursionlimit(20000)


def run_sgc_variant(tag, mixer_factory):
    """用自定义 mixer 工厂替换 SGCMixer，做对照。"""
    print(f"\n===== {tag} =====")
    torch.manual_seed(T.SEED)
    model = SeqModel(vocab_size=T.VOCAB, dim=T.DIM, n_layers=T.N_LAYERS,
                     n_heads=T.N_HEADS, mixer="attn")  # 先建 attn 骨架
    # 替换所有 block 的 mixer
    for blk in model.blocks:
        blk.mixer = mixer_factory(T.DIM, T.N_HEADS)
    nparams = count_params(model)
    print(f"params: {nparams:,}")
    opt = torch.optim.Adam(model.parameters(), lr=T.LR)
    val_idx, val_tgt = T.make_batch(200)
    history = []
    t0 = time.perf_counter()
    for step in range(T.STEPS):
        idx, tgt = T.make_batch(T.BATCH)
        opt.zero_grad()
        logits_t = model(torch.tensor(idx))
        loss = torch.cross_entropy(logits_t.reshape(-1, T.VOCAB),
                                   torch.tensor(tgt.reshape(-1)), ignore_index=-100)
        loss.backward()
        opt.step()
        loss_val = float(loss.data.flatten()[0])
        del loss, logits_t
        gc.collect()
        if step % 100 == 0 or step == T.STEPS - 1:
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
    return {"tag": tag, "params": nparams, "final_val_acc": final_acc,
            "time_s": dt, "history": history}


# ---- 变体 A：宽谱通道（提升容量，num_bands=64==D）----
class SGCMixerWide(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        from simplcore.architecture.spectral_global_conv import SpectralGlobalConv
        self.sgc = SpectralGlobalConv(dim=dim, num_bands=dim, num_scales=3, rank=16)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.sgc(x))


# ---- 变体 B：朴素谱卷积（去掉门控+谱注意力，仅 MSSM 频域窗）----
class NaiveSpectralConv(nn.Module):
    """直接用可微 FFT 做"全局卷积"：频域逐点可学习乘子（每个频率一个 D 维复增益）。
    这是谱卷积的最简强基线，无因果掩码、无门控、无谱注意力。"""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # 可学习频域复增益: (F, D) 实部+虚部
        self.fre = nn.Parameter(torch.zeros(1, 25, dim))  # 占位, F 在 forward 定
        self.fim = nn.Parameter(torch.zeros(1, 25, dim))
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        from simplcore.architecture.fft_ops import rfft1d, irfft1d
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        B, L, D = u.shape
        cre, cim = rfft1d(u, dim=1)
        Fdim = cre.shape[1]
        # 动态创建频域增益（缓存到 buffer 风格 Parameter，长度随 L）
        if self.fre.shape[1] != Fdim:
            dev = cre.data
            self.fre = nn.Parameter(torch.ones(1, Fdim, D) * 0.0)
            self.fim = nn.Parameter(torch.zeros(1, Fdim, D))
        # 复数乘: (cre+ i cim)*(gre + i gim)
        gre = self.fre[:, :Fdim, :]
        gim = self.fim[:, :Fdim, :]
        o_re = cre * gre - cim * gim
        o_im = cre * gim + cim * gre
        y = irfft1d(o_re, o_im, n=L, dim=1)
        out = self.out_proj(y)
        return residual + self.out_scale * out


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    # baseline
    results.append(T.run("attn", "Attention-Baseline"))
    results.append(T.run("sgc", "SGC-v1-orig"))
    results.append(run_sgc_variant("SGC-WideBands", lambda d, h: SGCMixerWide(d, h)))
    results.append(run_sgc_variant("Naive-SpectralConv", lambda d, h: NaiveSpectralConv(d, h)))
    with open(os.path.join(out_dir, "diagnose.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n===== DIAGNOSE SUMMARY =====")
    for r in results:
        print(f"  {r['tag']:20s} params={r['params']:>9,}  val_acc={r['final_val_acc']*100:5.2f}%  time={r['time_s']:.0f}s")


if __name__ == "__main__":
    main()
