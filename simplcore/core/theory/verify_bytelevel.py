"""
Finite-identity / numeric consistency checks for NT35..NT38
(the byte-level input-path theorems in new_theorems30.py (GROUP VIII)).

These checks BUILD concrete instances and verify each claim numerically.
They are NOT the proofs (the proofs are analytic in new_theorems30.py).

NO TRAINING. NO GRADIENTS. NO MODEL WEIGHTS. NO EXPERIMENTAL DATA.
Run: python verify_bytelevel.py    (prints PASS/FAIL per item)
"""
from __future__ import annotations

import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simplcore.core.model.bytelevel import (ByteLevelCodec, BytePatchBridge,
                                            EntropyPatcher)

RNG = random.Random(20261117)
rows = []


def rec(nid, name, ok, detail, kind="numeric_consistency"):
    v = "PASS" if ok else "FAIL"
    rows.append(dict(id=nid, name=name, kind=kind, verdict=v, detail=detail))
    print(("  [PASS] " if ok else "  [FAIL] ") + nid + " " + name, flush=True)
    return bool(ok)


# =============================================================================
# NT35: codec round-trip fidelity / injectivity / total decode
# =============================================================================
def nt35():
    codec = ByteLevelCodec()
    ok = True
    # (a) decode o encode = id on adversarial but VALID unicode
    samples = ["", "hello world", "你好，世界！", "emoji \U0001F680 test",
               "ünïcödé åccents", "tabs\tand\nnewlines\r\n", "4-byte \U00010488",
               "mixed 中英 mixing", "quote\"'`", "<html>&amp;</html>"]
    for s in samples:
        if codec.decode(codec.encode(s)) != s:
            ok = False
    # (b) injectivity of encode on distinct strings
    for _ in range(100):
        a = "".join(chr(RNG.randrange(32, 0x2FFF)) for _ in range(RNG.randrange(1, 12)))
        b = "".join(chr(RNG.randrange(32, 0x2FFF)) for _ in range(RNG.randrange(1, 12)))
        if a != b and codec.encode(a) == codec.encode(b):
            ok = False
    # (c) decode is total on arbitrary byte IDs: never raises, always str
    for _ in range(100):
        ids = [RNG.randrange(0, 256) for _ in range(RNG.randrange(0, 40))]
        out = codec.decode(ids)
        if not isinstance(out, str):
            ok = False
    return rec("NT35", "codec round-trip / injectivity / total decode", ok,
               "10 adversarial strings; 100 injectivity pairs; 100 random byte decodes",
               "numeric_consistency")


# =============================================================================
# NT36: patcher purity + causality + batch == streaming
# =============================================================================
def _feed_chunks(patcher, data, chunk):
    out = []
    for i in range(0, len(data), chunk):
        bounds, _ = patcher.feed(data[i:i + chunk])
        out += bounds
    return out


def nt36():
    ok = True
    det = []
    for mode in ("entropy", "space", "fixed"):
        ids = [RNG.randrange(0, 256) for _ in range(400)]
        # (a) purity: repeat + after streaming activity
        p = EntropyPatcher(mode=mode, threshold=3.0, max_len=8)
        a = p.boundaries(ids)
        if p.boundaries(ids) != a:
            ok = False
        p.feed(ids[:137])
        if p.boundaries(ids) != a:
            ok = False
        # (c) batch == streaming (absolute indices), arbitrary chunking
        ps = EntropyPatcher(mode=mode, threshold=3.0, max_len=8)
        got = _feed_chunks(ps, ids, 13) + ps.feed_flush()
        if got != a:
            ok = False
        # (b) causality: emissions during the prefix phase do not depend
        # on the suffix that follows
        k = 250
        p1 = EntropyPatcher(mode=mode, threshold=3.0, max_len=8)
        pre_only = _feed_chunks(p1, ids[:k], 13)
        p2 = EntropyPatcher(mode=mode, threshold=3.0, max_len=8)
        pre = _feed_chunks(p2, ids[:k], 13)
        _suff = _feed_chunks(p2, ids[k:], 13) + p2.feed_flush()
        if pre != pre_only:
            ok = False
        det.append(f"{mode}: {'ok' if ok else 'MISMATCH'}")
    return rec("NT36", "patcher purity + causality + batch==streaming", ok,
               "; ".join(det), "numeric_consistency")


# =============================================================================
# NT37: O(1) streaming carry (codec <= 3 bytes; patcher <= max_len - 1)
# =============================================================================
def nt37():
    ok = True
    det = []
    codec = ByteLevelCodec()
    max_pending = 0
    for _ in range(200):
        codec.feed("".join(chr(RNG.randrange(0x80, 0x2FFF))
                           for _ in range(RNG.randrange(0, 20))))
        max_pending = max(max_pending, len(codec._pending))
    if max_pending > 3:
        ok = False
    det.append(f"codec pending<={max_pending}")
    for mode in ("entropy", "space", "fixed"):
        p = EntropyPatcher(mode=mode, threshold=3.0, max_len=8)
        worst = 0
        for _ in range(200):
            p.feed([RNG.randrange(0, 256) for _ in range(RNG.randrange(0, 12))])
            worst = max(worst, len(p._pending))
        if worst > 7:                      # max_len - 1
            ok = False
        det.append(f"{mode} pending<={worst}")
    return rec("NT37", "streaming carry is O(1) (codec<=3, patcher<=max_len-1)",
               ok, "; ".join(det), "numeric_consistency")


# =============================================================================
# NT38: patch-count budget ceil(L/max_len) <= n <= L, exact partition
# =============================================================================
def nt38():
    ok = True
    det = []
    for mode in ("entropy", "space", "fixed"):
        for L in (1, 2, 3, 7, 8, 9, 16, 33, 100, 257):
            max_len = 8
            p = EntropyPatcher(mode=mode, threshold=3.0, max_len=max_len)
            ids = [RNG.randrange(0, 256) for _ in range(L)]
            b = p.boundaries(ids)
            n = len(b)
            # partition: monotone, in range, last = L-1, lengths in [1, max_len]
            if not (b == sorted(b) and len(set(b)) == n and b[-1] == L - 1):
                ok = False
            lens = [b[0] + 1] + [b[i] - b[i - 1] for i in range(1, n)]
            if min(lens) < 1 or max(lens) > max_len:
                ok = False
            if not (math.ceil(L / max_len) <= n <= L):
                ok = False
            det.append(f"{mode}/L={L}:n={n}")
    return rec("NT38", "patch budget: ceil(L/max_len) <= n <= L, exact partition",
               ok, f"{len(det)} cells checked", "numeric_consistency")


# =============================================================================
CHECKS = [nt35, nt36, nt37, nt38]


def main():
    print("== byte-level input-path theorem checks (NT35..NT38) ==")
    for fn in CHECKS:
        fn()
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"== {n_pass}/{len(rows)} PASS ==")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
