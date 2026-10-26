# -*- coding: utf-8 -*-
import os
import sys, io, os, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
import build_corpus as bc
docs = bc.collect_documents()
print("documents collected:", len(docs))
tot = sum(len(t) for _, t in docs)
print("total raw chars in selected docs:", tot)
by_pat = {"root_txt":0, "docs_md":0, "java_txt":0, "simplcore_txt":0}
for fp, t in docs:
    rp = os.path.relpath(fp, bc.PROJECT_ROOT)
    if rp.startswith("docs" + os.sep): by_pat["docs_md"] += 1
    elif rp.startswith("java-high-performance"): by_pat["java_txt"] += 1
    elif rp.startswith("simplcore"): by_pat["simplcore_txt"] += 1
    else: by_pat["root_txt"] += 1
print("doc counts by source pattern:", by_pat)
print("\nper-document char counts (sorted desc):")
for fp, t in sorted(docs, key=lambda z: -len(z[1]))[:40]:
    print(f"  {len(t):8d}  {os.path.relpath(fp, bc.PROJECT_ROOT)}")

for seq_len, step in [(256, None), (512, None)]:
    vocab, train, val, test = bc.make_dataset(seq_len, step=step)
    tr_tok = sum(len(x) for x, y in train); va_tok = sum(len(x) for x, y in val); te_tok = sum(len(x) for x, y in test)
    print(f"\n--- seq_len={seq_len} step={step or max(1,seq_len//4)} ---")
    print(f"  vocab={len(vocab)}  windows: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  tokens: train={tr_tok} val={va_tok} test={te_tok}")
    # overlap check: are any val windows substrings of train windows' documents?
    import collections
    trset = set(tuple(x) for x, y in train)
    ov = 0
    for x, y in val:
        for s in range(0, seq_len, 64):
            if tuple(x[s:s+192]) in trset: ov += 1; break
    print(f"  val windows whose {seq_len}-window prefix appears in train set as a full window: {ov}/{len(val)}")
