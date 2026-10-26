# -*- coding: utf-8 -*-
import os
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
import build_corpus as bc, random
docs = bc.collect_documents()
seq_len, step = 256, 64
tmp_vocab, tmp_stoi = bc.build_tokenizer([t for _, t in docs])
dw = {fp: len(bc.doc_to_windows(bc.encode(t, tmp_stoi), seq_len, step)) for fp, t in docs}
order = docs[:]; rng = random.Random(1234); rng.shuffle(order)
total = sum(dw[fp] for fp, _ in order)
tv, tt = int(total*0.1), int(total*0.1)
av = at = 0; tr=[]; va=[]; te=[]
for fp, t in order:
    w = dw[fp]
    if av < tv: va.append((fp,w)); av += w
    elif at < tt: te.append((fp,w)); at += w
    else: tr.append((fp,w))
print("total windows(all docs, tmp vocab) =", total, " target val/test =", tv, tt)
for nm, lst in [("VAL", va), ("TEST", te), ("TRAIN", tr)]:
    print(f"\n{nm}: {len(lst)} docs, {sum(w for _,w in lst)} windows")
    for fp, w in lst: print(f"    {w:4d} win  {os.path.relpath(fp, bc.PROJECT_ROOT)}")

# exact overlap: 128-char shingles from val/test that literally occur inside TRAIN documents
train_text = "\n".join(t for fp, t in docs if any(fp==f for f,_ in tr))
print("\n=== substring leakage: does any 128-char val/test slice occur verbatim in a TRAIN document? ===")
for nm, lst in [("val", va), ("test", te)]:
    n_slice = n_hit = 0
    for fp, t in docs:
        if not any(fp==f for f,_ in lst): continue
        for i in range(0, max(1, len(t)-128), 128):
            s = t[i:i+128]
            if len(s) < 128: break
            n_slice += 1
            if s in train_text: n_hit += 1
    print(f"  {nm}: {n_hit}/{n_slice} 128-char slices found verbatim inside train documents")
