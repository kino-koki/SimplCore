import json, glob, os
rows = []
for f in sorted(glob.glob("verify_sgc/results/lingap_*.json")):
    d = json.load(open(f))
    rows.append((d["mixer"], d["G"], d["recall_last"], d["best"], d["chance"], os.path.basename(f)))
print("mixer    G   recall     best   chance  file")
for m, G, r, b, c, n in sorted(rows, key=lambda x: (x[1], x[0])):
    print("%-6s %4d  %7.4f  %7.4f  %7.4f  %s" % (m, G, r, b, c, n))
print()
print("broken-loss runs preserved:")
for f in sorted(glob.glob("verify_sgc/results/_broken_loss_2026_10/*.json")):
    d = json.load(open(f))
    print("  %-24s recall=%.4f (chance %.4f)" % (os.path.basename(f), d["recall_last"], d["chance"]))
