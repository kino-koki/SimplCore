import sys, io, json, os, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = r"G:\SimplCore-V 0.1-ALPHA\verify_sgc\results"
for name in ["corpus.json", "corpus512.json"]:
    d = json.load(open(os.path.join(R, name)))
    for sp in ("train","val","test"):
        ws = d[sp]
        cnt = collections.Counter((len(a), len(b)) for a,b in ws)
        print(name, sp, "n=", len(ws), "window shapes:", dict(cnt))
    print()
