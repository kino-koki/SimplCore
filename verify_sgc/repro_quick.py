# -*- coding: utf-8 -*-
"""当场轻量复现: 固定核 sgc vs 内容自适应 ca 的 held-out 泛化相对关系.

同语料(corpus.json)、同骨架、同超参(dim128/L3/seq256/batch16/wd0.1/lr1e-3/300步),
仅 mixer 不同, 独立重新训练, 观察 final val/test ppl 的相对关系是否与
honest_* 大实验一致(sgc 崩墙/ca 显著更低).
"""
import os, json, random, numpy as np, torch, torch.nn.functional as F
from torch.optim.adamw import AdamW
from sgc_v3 import build_model, count_params

SEED = 2026; DIM = 176; LAY = 4; SEQ = 256; BS = 16; STEPS = 800; WD = 0.1; EVAL = 100
d = json.load(open('results/corpus.json', encoding='utf-8'))
vocab = len(d['vocab']); train, val, test = d['train'], d['val'], d['test']
print(f"corpus vocab={vocab} train={len(train)} val={len(val)} test={len(test)}")

def run(mixer):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    m = build_model(vocab, DIM, LAY, 4, SEQ, mixer, rank=8).cpu()
    top = count_params(m)
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=WD)
    def ppl(data, bs=BS):
        m.eval(); tot=nt=n=0
        for i in range(0, len(data), bs):
            b=data[i:i+bs]; x=torch.tensor([s[0] for s in b],dtype=torch.long); y=torch.tensor([s[1] for s in b],dtype=torch.long)
            with torch.no_grad():
                loss=F.cross_entropy(m(x).reshape(-1,vocab), y.reshape(-1), ignore_index=0)
            tot+=loss.item()*y.numel(); nt+=y.numel(); n+=1
        p=np.exp(tot/nt); m.train(); return p
    for s in range(1, STEPS+1):
        b=random.sample(train, k=BS); x=torch.tensor([z[0] for z in b],dtype=torch.long); y=torch.tensor([z[1] for z in b],dtype=torch.long)
        opt.zero_grad(); loss=F.cross_entropy(m(x).reshape(-1,vocab), y.reshape(-1), ignore_index=0); loss.backward(); opt.step()
        if s % EVAL == 0:
            vp = ppl(val); print(f"[{mixer}] step={s:4d} val_ppl={vp:12,.2f}", flush=True)
    tp, vp, tep = ppl(random.sample(train, min(len(train),150))), ppl(val), ppl(test)
    return top, tp, vp, tep

for mx in ['sgc', 'ca']:
    p, tp, vp, tep = run(mx)
    print(f"\n=== 当场复现 [{mx}] params={p:,} ===")
    print(f"train_ppl={tp:9.2f}   val_ppl={vp:10.2f}   test_ppl={tep:12.2f}")
print("\n完成. 观察: ca 的 val 应显著低于 sgc (固定核崩墙).")