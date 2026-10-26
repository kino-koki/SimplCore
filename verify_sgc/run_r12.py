import os, sys, time, gc, json, statistics as st
sys.path.insert(0, '..')
import numpy as np
import simplcore.tensor as torch
from simplcore.tensor import nn
import importlib.util
spec = importlib.util.spec_from_file_location('sgc_train', 'train.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from model import SeqModel, count_params
from simplcore.architecture.spectral_global_conv import SpectralGlobalConvV2

SEEDS = [1234, 2024, 777, 42, 99]
STEPS = 1200
RANK = 12

def make_sgc(d, h):
    m = SpectralGlobalConvV2(d, 3, max_len=64, rank=RANK)
    class W(nn.Module):
        def __init__(self, mm):
            super().__init__(); self.sgc = mm
        def forward(self, x):
            return self.sgc(x)
    return W(m)

def run_sgc(seed):
    T.rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = SeqModel(T.VOCAB, T.DIM, T.N_LAYERS, T.N_HEADS, 'attn')
    for blk in model.blocks:
        blk.mixer = make_sgc(T.DIM, T.N_HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=T.LR)
    vi, vt = T.make_batch(200)
    t0 = time.perf_counter()
    for s in range(STEPS):
        ii, tt = T.make_batch(T.BATCH); opt.zero_grad()
        ll = model(torch.tensor(ii))
        lo = torch.cross_entropy(ll.reshape(-1, T.VOCAB), torch.tensor(tt.reshape(-1)), ignore_index=-100)
        lo.backward(); opt.step(); del lo, ll; gc.collect()
    dt = time.perf_counter() - t0
    with torch.no_grad():
        vl = model(torch.tensor(vi)).data
        c, t = T.selective_copy_acc(vl, vt)
        vl2 = float(torch.cross_entropy(model(torch.tensor(vi)).reshape(-1, T.VOCAB),
                                        torch.tensor(vt.reshape(-1)), ignore_index=-100).data.flatten()[0])
    return {'acc': c/t, 'loss': vl2, 'time': dt, 'params': count_params(model)}

def run_attn(seed):
    T.rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = SeqModel(T.VOCAB, T.DIM, T.N_LAYERS, T.N_HEADS, 'attn')
    opt = torch.optim.Adam(model.parameters(), lr=T.LR)
    vi, vt = T.make_batch(200)
    t0 = time.perf_counter()
    for s in range(STEPS):
        ii, tt = T.make_batch(T.BATCH); opt.zero_grad()
        ll = model(torch.tensor(ii))
        lo = torch.cross_entropy(ll.reshape(-1, T.VOCAB), torch.tensor(tt.reshape(-1)), ignore_index=-100)
        lo.backward(); opt.step(); del lo, ll; gc.collect()
    dt = time.perf_counter() - t0
    with torch.no_grad():
        vl = model(torch.tensor(vi)).data
        c, t = T.selective_copy_acc(vl, vt)
        vl2 = float(torch.cross_entropy(model(torch.tensor(vi)).reshape(-1, T.VOCAB),
                                        torch.tensor(vt.reshape(-1)), ignore_index=-100).data.flatten()[0])
    return {'acc': c/t, 'loss': vl2, 'time': dt, 'params': count_params(model)}

attn = [run_attn(s) for s in SEEDS]
sgc = [run_sgc(s) for s in SEEDS]
aa = st.mean(r['acc'] for r in attn); sa = st.mean(r['acc'] for r in sgc)
al = st.mean(r['loss'] for r in attn); sl = st.mean(r['loss'] for r in sgc)
at = st.mean(r['time'] for r in attn); stt = st.mean(r['time'] for r in sgc)
res = {'attn': attn, 'sgc': sgc, 'seeds': SEEDS, 'steps': STEPS, 'rank': RANK}
json.dump(res, open('results/v2_1200_r12.json', 'w'), indent=2)
print(f'Attention(1200,r8): acc={aa*100:.2f}% loss={al:.4f} time={at:.1f}s params={attn[0]["params"]:,}')
print(f'SGC-v2(1200,r{RANK})  : acc={sa*100:.2f}% loss={sl:.4f} time={stt:.1f}s params={sgc[0]["params"]:,}')
print(f'Dacc={(sa-aa)*100:+.2f}pp  Dloss={sl-al:+.4f}  Dtime={stt-at:+.1f}s')
print('overall win:', (sa > aa and sl < al and stt < at))
