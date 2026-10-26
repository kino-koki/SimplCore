import json, os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = r"G:\SimplCore-V 0.1-ALPHA\verify_sgc\results"
for f in sorted(os.listdir(R)):
    p = os.path.join(R, f)
    sz = os.path.getsize(p)
    if not f.endswith(".json") or sz > 200000:
        print(f"-- {f}  ({sz} bytes) [skipped content or non-json]")
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"-- {f}  LOAD ERROR {e}")
        continue
    s = json.dumps(d, ensure_ascii=False)
    print(f"-- {f} ({sz}B): {s[:2600]}")
