import sys, io, os, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
root = r"G:\SimplCore-V 0.1-ALPHA"
refs = ["few_step_theorem.py", "few_step_benchmark.py", "gated_sgc.py",
        "SGC_v2_门控消融_诚实报告.md", "v2_1200.json", "97.72"]
for r in refs:
    hits = glob.glob(os.path.join(root, "**", r), recursive=True)
    hits = [h for h in hits if "\\libs\\" not in h and "\\.venv" not in h]
    print(f"{r:38s} -> {[os.path.relpath(h, root) for h in hits] if hits else 'NOT FOUND'}")
# search for 97.72 anywhere in repo (excluding libs/venvs)
import re
pat = re.compile(r"97\.72")
found = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".venv",".venv311",".venv21","libs","node_modules","__pycache__","third_party",".archive","wheels",".venv_sc",".trae-html-share-packages",".git")]
    for f in fn:
        if not f.lower().endswith((".py",".json",".md",".txt",".log",".html",".svg",".tex")): continue
        p = os.path.join(dp,f)
        try:
            if os.path.getsize(p) > 5_000_000: continue
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        if pat.search(t): found.append(os.path.relpath(p, root))
print("\n'97.72' occurrences:", found)
# SGC v1 model_v2 reference
print("\nmodel_v2.py exists:", os.path.exists(os.path.join(root,"verify_sgc","model_v2.py")))
