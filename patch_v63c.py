"""v6.3c: fix a clipped two-column heading, normalise ALL-CAPS to standard
prose, and disambiguate the table header. No scientific content changes."""
import subprocess
import sys

G = "/data/workspace/build_simplcore_pdf_v5.py"
s = open(G, encoding="utf-8").read()

REPL = [
    # clipped heading (was "... and a withdrawn training claim")
    ('    Hh("4.13  Fairness audit: two asymmetries removed, and a withdrawn "\n'
     '       "training claim", size=9.6, before=5.0, after=2.0)',
     '    Hh("4.13  Fairness audit and a withdrawn training claim",\n'
     '       size=9.6, before=5.0, after=2.0)'),
    # table header
    ('    _t = [["hops", "arm", "acc (5 seeds)", "analytic"]]',
     '    _t = [["hops", "arm", "acc (5 seeds)", "witness"]]'),
    # normalise ALL-CAPS in Sec 4.13 (WITHDRAWN is a register term, keep it)
    ('"one-hot ids. That was true of the IDS but not of the "\n'
     '      "COMPUTATION: only the TSC branch consumed the predecessor view "',
     '"one-hot ids. That was true of the ids but not of the "\n'
     '      "computation: only the TSC branch consumed the predecessor view "'),
    ('"Both are now removed: all eight arms receive "\n'
     '      "(U, Up, q, mask) with ONE fixed full-width orthonormal Hadamard "',
     '"Both are now removed: all eight arms receive "\n'
     '      "(U, Up, q, mask) with one fixed full-width orthonormal Hadamard "'),
    ('"and +0.8457 from the ASYMMETRIC setup. We "',
     '"and +0.8457 from the asymmetric setup. We "'),
    ('"<b>Theorem B itself is NOT "\n'
     '      "withdrawn</b>',
     '"<b>Theorem B itself is not "\n'
     '      "withdrawn</b>'),
    ('"We give EXPLICIT ZERO-TRAINING weight assignments on the same "',
     '"We give explicit zero-training weight assignments on the same "'),
    ('"The two function classes therefore COINCIDE on this task; the "\n'
     '      "observed training gap is OPTIMISATION / INDUCTIVE BIAS, not "',
     '"The two function classes therefore coincide on this task; the "\n'
     '      "observed training gap is optimisation / inductive bias, not "'),
    ('"theorem and NOT a superiority claim in either direction. "',
     '"theorem nor a superiority claim in either direction. "'),
    ('"NOT a new "\n      "theorem nor', '"neither a new "\n      "theorem nor'),
    ('"same-run reproduction, so "\n      "the Fisher-parity statement remains withdrawn. The additional reason is "\n'
     '      "now clearer: v6.2\'s low reference numbers came from an UNFAIR "',
     '"same-run reproduction, so "\n      "the Fisher-parity statement remains withdrawn. The additional reason is "\n'
     '      "now clearer: v6.2\'s low reference numbers came from an unfair "'),
    # Sec 5 / Sec 7 TOY -> toy
    ('"supplies same-condition TOY mechanism references, not "',
     '"supplies same-condition toy mechanism references, not "'),
    ('"adds small same-run TOY mechanism references for a reproduction "',
     '"adds small same-run toy mechanism references for a reproduction "'),
]

for old, new in REPL:
    n = s.count(old)
    if n == 0:
        print("  skip (not found):", old[:62].replace("\n", " "))
        continue
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
    print("  fixed:", old[:58].replace("\n", " "))

open(G, "w", encoding="utf-8").write(s)
r = subprocess.run([sys.executable, G], cwd="/data/workspace",
                   capture_output=True, text=True)
print((r.stdout or "")[-700:])
if r.returncode:
    print((r.stderr or "")[-2500:])
    sys.exit(1)
print("REBUILD OK")
