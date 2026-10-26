import sys, io, os, re, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
root = r"G:\SimplCore-V 0.1-ALPHA"
pats = {"2.67%": re.compile(r"2\.67\s*%"), "12.89%": re.compile(r"12\.89\s*%"),
        "1.50%": re.compile(r"1\.50\s*%"), "门控消融": re.compile(r"门控消融"),
        "49%": re.compile(r"快\s*49|49\s*%")}
res = {k: [] for k in pats}
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".venv",".venv311",".venv21","libs","node_modules","__pycache__","third_party",".archive","wheels",".venv_sc",".trae-html-share-packages",".git",".tmp_audit")]
    for f in fn:
        if not f.lower().endswith((".py",".json",".md",".txt",".log",".html",".svg")): continue
        p = os.path.join(dp,f)
        try:
            if os.path.getsize(p) > 5_000_000: continue
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        for k, pat in pats.items():
            if pat.search(t): res[k].append(os.path.relpath(p, root))
for k, v in res.items():
    print(f"{k:10s} -> {v[:12]}")
