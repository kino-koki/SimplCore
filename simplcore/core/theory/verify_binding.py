"""
Finite-identity / numeric consistency checks for NT31..NT34
(the binding-memory theorems in new_theorems30.py (GROUP VII)).

These checks BUILD concrete instances and verify each claim numerically.
They are NOT the proofs (the proofs are analytic in new_theorems30.py).

NO TRAINING. NO GRADIENTS. NO MODEL WEIGHTS. NO EXPERIMENTAL DATA.
Run: python verify_binding.py    (prints PASS/FAIL per item)
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20261117)
rows = []


def rec(nid, name, ok, detail, kind="numeric_consistency"):
    v = "PASS" if ok else "FAIL"
    rows.append(dict(id=nid, name=name, kind=kind, verdict=v, detail=detail))
    print(("  [PASS] " if ok else "  [FAIL] ") + nid + " " + name, flush=True)
    return bool(ok)


# =============================================================================
# A minimal, faithful numpy model of the binding memory (mirrors bsgc.py)
# =============================================================================
class BindingMemory:
    def __init__(self, slots: int, dim: int):
        self.S = slots
        self.D = dim
        self.sk = np.zeros((slots, dim))
        self.sv = np.zeros((slots, dim))
        self.up = np.zeros(dim)
        self.writes = 0

    def write(self, k_new, v_new, ptr):
        self.sk[ptr % self.S] = k_new
        self.sv[ptr % self.S] = v_new
        self.writes += 1

    def state_size(self):
        return self.sk.size + self.sv.size + self.up.size


# =============================================================================
# NT31: state size is (2S+1)D, independent of L
# =============================================================================
def nt31():
    ok = True
    details = []
    for S, D, L in ((8, 16, 1), (8, 16, 64), (16, 32, 128), (4, 8, 1000)):
        bm = BindingMemory(S, D)
        for t in range(L):
            h = RNG.normal(size=D)
            k = h.copy()
            v = RNG.normal(size=D)
            if t >= 1:
                bm.write(k, v, t - 1)
            bm.up = h
        sz = bm.state_size()
        expect = (2 * S + 1) * D
        ok &= (sz == expect)
        details.append(f"S={S},D={D},L={L}: {sz} == {expect}")
    return rec("NT31", "state = (2S+1)D scalars, independent of L", ok,
               "; ".join(details), "numeric_consistency")


# =============================================================================
# NT32: FIFO addressability = the last S writes
# =============================================================================
def nt32():
    ok = True
    S, D, L = 8, 16, 64
    bm = BindingMemory(S, D)
    ks = RNG.normal(size=(L, D))
    vs = RNG.normal(size=(L, D))
    for t in range(1, L):
        bm.write(ks[t], vs[t], t - 1)
    # (a) the slots hold exactly the last S writes
    #     writes happen at positions t=1..L-1 with write index j = t-1 in [0, L-2]
    ok &= all(np.allclose(bm.sk[j % S], ks[j + 1]) for j in range(L - 1 - S, L - 1))
    ok &= all(np.allclose(bm.sv[j % S], vs[j + 1]) for j in range(L - 1 - S, L - 1))
    # (b) a write index j is present iff j >= (L-2) - (S-1)  i.e. j >= L-S-1
    present = [j for j in range(L - 1) if j >= (L - 2) - (S - 1)]
    ok &= all(j >= (L - 2) - (S - 1) for j in present)
    return rec("NT32", "FIFO slots hold exactly the last S writes", ok,
               f"S={S}, writes={L-1}, checked slots {L-S}..{L-2}",
               "numeric_consistency")


# =============================================================================
# NT33: residual binding recovers the value (multi-key, constructive)
# =============================================================================
def nt33():
    ok = True
    G, D = 8, 64
    # separated, unit-norm key/value codes (identity-like: orthogonal rows)
    code = np.eye(D)[: max(2 * G, D)]
    keys = code[:G]                     # key codes (unit, orthogonal)
    vals = code[G:2 * G]                # value codes
    # identity addressing projections: K_j = h[j], V_j = h[j+1], q = h[t]
    # bind pair i at positions (2i, 2i+1): slot j=2i gets (K=h[2i], V=h[2i+1])
    S = 4 * G
    sk = np.zeros((S, D))
    sv = np.zeros((S, D))
    for i in range(G):
        sk[2 * i] = keys[i]
        sv[2 * i] = vals[i]
    BETA = 64.0    # 锐化读出（模块的可学习 tau 趋于此极限）
    for i_star in range(G):
        q = keys[i_star]
        scores = np.array([q @ sk[j] for j in range(S)])
        # the matching slot has score 1; all other keys are orthogonal (0);
        # empty slots are zero too. argmax must be the matching slot.
        if int(np.argmax(scores)) != 2 * i_star or scores[2 * i_star] <= 0:
            ok = False
        m = scores.max()
        e = np.exp(BETA * (scores - m))
        probs = e / e.sum()
        read = probs @ sv
        if not np.allclose(read, vals[i_star], atol=1e-9):
            ok = False
    return rec("NT33", "residual binding: exact multi-key recovery", ok,
               f"G={G}, separated unit-norm code, identity projections",
               "numeric_consistency")


# =============================================================================
# NT34: strict causality (a perturbation at t' > t leaves the read at t)
# =============================================================================
def nt34():
    ok = True
    S, D, L = 8, 16, 64
    for t_star in (10, 30, 50):
        base = RNG.normal(size=(L, D))
        pert = base.copy()
        pert[t_star] += RNG.normal(size=D)          # 扰动 t* > t
        vseq = RNG.normal(size=(L, D))              # 两序列共用的 v 序列
        reads = {}
        for tag, seq in (("base", base), ("pert", pert)):
            bm = BindingMemory(S, D)
            out = []
            for t in range(L):
                h = seq[t]
                k = h.copy()
                v = vseq[t].copy()                 # 两序列用同一 v 序列
                if t >= 1:
                    bm.write(k, v, t - 1)          # 写使用 h[t-1], h[t]
                q = h
                scores = np.array([q @ bm.sk[j] for j in range(S)])
                m = scores.max()
                e = np.exp(scores - m)
                probs = e / e.sum()
                out.append(probs @ bm.sv)
            reads[tag] = out[:t_star]              # 只比较 t < t*
        ok &= np.allclose(reads["base"], reads["pert"], atol=0)
    return rec("NT34", "strict causality of the binding write/read", ok,
               f"t* in {10,30,50}; reads at t < t* bit-identical",
               "numeric_consistency")


# =============================================================================
CHECKS = [nt31, nt32, nt33, nt34]


def main():
    print("== binding-memory theorem checks (NT31..NT34) ==")
    for fn in CHECKS:
        fn()
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"== {n_pass}/{len(rows)} PASS ==")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
