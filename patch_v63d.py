"""v6.3d: exact single-line replacements for the remaining ALL-CAPS."""
import subprocess
import sys

G = "/data/workspace/build_simplcore_pdf_v5.py"
s = open(G, encoding="utf-8").read()

REPL = [
    ('"the static-LTI and attention function classes COINCIDE on this task. "',
     '"the static-LTI and attention function classes coincide on this task. "'),
    ('"received the same one-hot ids. That was true of the IDS but not of "',
     '"received the same one-hot ids. That was true of the ids but not of "'),
    ('"the COMPUTATION: only the TSC branch consumed the predecessor view "',
     '"the computation: only the TSC branch consumed the predecessor view "'),
    ('"(U, Up, q, mask) with ONE fixed full-width orthonormal Hadamard "',
     '"(U, Up, q, mask) with one fixed full-width orthonormal Hadamard "'),
    ('"v6.2 reported 0.9512 and +0.8457 from the ASYMMETRIC setup. We "',
     '"v6.2 reported 0.9512 and +0.8457 from the asymmetric setup. We "'),
    ('"P("<b>Why this does NOT become an expressivity result either way.</b> "',
     '"P("<b>Why this does not become an expressivity result either way.</b> "'),
    ('"corroborated by end-to-end training. <b>Theorem B itself is NOT "',
     '"corroborated by end-to-end training. <b>Theorem B itself is not "'),
    ('"NOT a new theorem and NOT a superiority claim in either direction. "',
     '"neither a new theorem nor a superiority claim in either direction. "'),
    ('"record from the earlier single-hop run still has NO verifiable "',
     '"record from the earlier single-hop run still has no verifiable "'),
    ('"reference numbers came from an UNFAIR implementation, not from the "',
     '"reference numbers came from an unfair implementation, not from the "'),
    ('"or inferiority with respect to attention anywhere in this paper. "',
     '"or inferiority with respect to attention anywhere in this paper. "'),
    ('"Section 4.13 supplies same-condition TOY mechanism references, not "',
     '"Section 4.13 supplies same-condition toy mechanism references, not "'),
]

for old, new in REPL:
    if old == new:
        continue
    n = s.count(old)
    if n == 0:
        print("  skip (absent):", old[:64])
        continue
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
    print("  fixed:", old[:62])

open(G, "w", encoding="utf-8").write(s)
r = subprocess.run([sys.executable, G], cwd="/data/workspace",
                   capture_output=True, text=True)
print((r.stdout or "")[-600:])
if r.returncode:
    print((r.stderr or "")[-2500:])
    sys.exit(1)
print("REBUILD OK")
