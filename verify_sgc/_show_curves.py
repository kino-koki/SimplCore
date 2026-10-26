# -*- coding: utf-8 -*-
import json
for name in ['honest_sgc.json', 'honest_ca.json']:
    d = json.load(open('results/' + name, encoding='utf-8'))
    vp = d['curves']['val']
    tp = d['curves']['train']
    print(f"{name}:")
    print("   step   val_ppl")
    for s, v in vp:
        print(f"   {s:4d}   {v:16,.2f}")
    print(f"   train_last={tp[-1][1]:.2f}\n")