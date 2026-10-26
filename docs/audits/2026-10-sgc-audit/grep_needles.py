import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
root = r"G:\SimplCore-V 0.1-ALPHA"
needles = ["0.5819", "0.8219", "0.594", "0.0063", "0.0139", "训练进行中", "0.671", "64.2 KB", "381", "236"]
skipd = {".venv", ".venv311", ".venv21", ".git", "node_modules", "__pycache__", "third_party", ".archive", "wheels", ".trae-html-share-packages"}
hits = 0
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in skipd]
    for f in fn:
        if not f.lower().endswith((".py", ".json", ".md", ".txt", ".log", ".ps1", ".html", ".svg")):
            continue
        p = os.path.join(dp, f)
        try:
            if os.path.getsize(p) > 8_000_000: continue
            t = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if "verify_sgc" not in p and ".tmp_audit" in p: continue
        for n in needles:
            if n in t:
                # print line numbers
                for i, line in enumerate(t.splitlines(), 1):
                    if n in line:
                        print(f"{n} :: {os.path.relpath(p, root)}:{i}: {line.strip()[:200]}")
                        hits += 1
                        break
print("hits", hits)
