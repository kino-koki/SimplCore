# -*- coding: utf-8 -*-
import os
import sys, io, os, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
import build_corpus as bc
root = bc.PROJECT_ROOT
pats = [os.path.join(root, "*.txt"), os.path.join(root,"docs","**","*.md"),
        os.path.join(root,"java-high-performance","*.txt"), os.path.join(root,"simplcore","**","*.txt")]
files = []
for p in pats:
    for fp in glob.glob(p, recursive=True):
        try: t = open(fp, encoding="utf-8", errors="ignore").read().strip()
        except Exception: continue
        if len(t) > 500: files.append(fp)
files = sorted(set(files))
tot = 0
print("RAW SOURCE FILES USED BY build_corpus.collect_documents():")
for fp in files:
    sz = os.path.getsize(fp); tot += sz
    print("  %9d B  %s" % (sz, os.path.relpath(fp, root)))
print("\n  files=%d  total on-disk bytes=%d (%.1f KiB)" % (len(files), tot, tot/1024))
chars = sum(len(open(fp, encoding='utf-8', errors='ignore').read().strip()) for fp in files)
print("  total chars (stripped) = %d" % chars)

for name in ["corpus.json", "corpus512.json"]:
    p = os.path.join(root, "verify_sgc", "results", name)
    d = json.load(open(p)); v = d["vocab"]
    def tok(ws): return sum(len(a) for a, b in ws)
    lens = sorted(set(len(a) for a in d["train"][:50]))
    print("\n%s: vocab=%d train=%d val=%d test=%d" % (name, len(v), len(d["train"]), len(d["val"]), len(d["test"])))
    print("   tokens train/val/test = %d/%d/%d   window lens(first50)=%s" % (tok(d["train"]), tok(d["val"]), tok(d["test"]), lens))
    unk = v.index("<unk>")
    for sp in ("train","val","test"):
        ws = d[sp]; n = t = 0
        for a, b in ws[:400]:
            n += sum(1 for z in a if z == unk); t += len(a)
        print("   %s: <unk> rate = %.2f%%" % (sp, n/max(t,1)*100))
    print("   file on disk = %d B" % os.path.getsize(p))
