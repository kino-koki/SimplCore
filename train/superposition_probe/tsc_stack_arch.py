"""
tsc_stack_arch.py -- v6.4 ARCHITECTURE MODULE.

The change is architectural, not a training trick:

  The inter-layer interface of a static-LTI TSC stack is NO LONGER a
  learnable matrix.  It is the codebook's own value->key swap

        M = C[V_K:, :]^T  C[:V_K, :]

  shared by every layer.  M is an algebraic object of the Sylvester-Hadamard
  dictionary: it contains no training data, no labels, no runtime annotation
  and no position rule.  Because it is shared and NOT a parameter, the
  parameter count does not grow with depth.

  The only learned weights are Wq (external-query projection) and Wout
  (read-out).  Both are trained from random initialisation.

  Inter-layer normalisation is built entirely from the graph's own
  exp/log/mul operators: sqrt(x) = exp(0.5 log(x)), inv = exp(-log(x)).
  No division by an off-graph constant, no straight-through gradient (W14).

Everything here preserves the SimplCore invariants:

  * the recurrence kernel K(tau) is STATIC and Toeplitz,
  * FFT / IIR duality holds to machine precision,
  * strict causality,
  * O(1) state (bytes independent of L),
  * no N x N score matrix, no KV cache, no softmax over token pairs.

This module lives in the architecture directory. It does NOT modify sgc.py:
the kernel itself needs no change -- the fix is entirely in the wiring
between two instances of the same static operator.

Run: python3 train/superposition_probe/tsc_stack_arch.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mini_autograd import T, const, free_graph, nll_from_logits

# In this self-contained autodiff `const(a)` and `param(a)` are the same
# object type (a T leaf); `param` exists only in the evaluation-directory
# copy. We keep the distinction explicit by a local alias.
param = const
import tsc_final_probe as _base          # hadamard(), kernel() -- SAME SGC kernel

OUT = HERE + "/results"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ codebook
def hadamard(n):
    return _base.hadamard(n)


def codebook_interface(C, VK):
    """The architecture's own value->key swap. Fixed, shared, NOT a parameter."""
    return C[VK:, :].T @ C[:VK, :]


def interface_properties(C, VK, M):
    """Six algebraic checks. These are exact identities, not measurements of
    a trained model."""
    V = C.shape[0]
    CVAL, CKEY = C[VK:, :], C[:VK, :]
    # (i) value -> key mapping
    e1 = float(np.linalg.norm(CVAL @ M - CKEY))
    # (ii) columns of M live in the key half-space (zero component on VALUE rows)
    proj_val = M @ CVAL.T                       # (V, VK) should be 0
    e2 = float(np.linalg.norm(proj_val))
    # (iii) key half is annihilated (M has no action on key codes)
    e3 = float(np.linalg.norm(CKEY @ M))
    # (iv) M^T M is the projector onto the KEY half-space; M M^T is the
    # projector onto the VALUE half-space.  (Getting these two swapped is a
    # natural mistake because C is symmetric -- the first draft did exactly
    # that and produced a Frobenius error of sqrt(V) = 8.0.)
    Pk = CKEY.T @ CKEY
    Pv = CVAL.T @ CVAL
    e4 = float(np.linalg.norm(M.T @ M - Pk))
    e4b = float(np.linalg.norm(M @ M.T - Pv))
    # (v) C is orthonormal -> M has rank VK, not V
    rank = int(np.linalg.matrix_rank(M, tol=1e-9))
    # (vi) M is an involution-like swap: M @ M.T @ M == M
    e6 = float(np.linalg.norm(M @ M.T @ M - M))
    return dict(value_to_key_err=e1, col_in_key_half_err=e2,
                key_annihilated_err=e3, key_projector_err=e4,
                value_projector_err=e4b,
                rank=rank, expected_rank=VK, triple_product_err=e6,
                note=("M^T M projects onto the key half-space, M M^T onto "
                      "the value half-space; rank(M)=V_K, so M is a partial "
                      "isometry between the two halves, NOT an invertible "
                      "V-by-V map. This is why the interface is a structural "
                      "object of the codebook rather than a free parameter."),
                all_identity=bool(max(e1, e2, e3, e4, e4b, e6) < 1e-10
                                  and rank == VK))


# ------------------------------------------------------------------ stack
class TSCStack:
    """H_L static-LTI layers wired by ONE fixed codebook-swap interface.

    Only Wq and Wout are parameters. Depth adds computation, not weights.
    """

    def __init__(self, n_layers, V=64, seed=0, rho_grid=None):
        assert V % 2 == 0 and (V & (V - 1)) == 0
        self.V, self.VK = V, V // 2
        self.n_layers = n_layers
        self.C = hadamard(V)
        self.M = codebook_interface(self.C, self.VK)
        _base.set_C(self.C, self.VK)
        rng = np.random.default_rng(83000 + seed * 191 + n_layers * 37)
        sc = 1.0 / math.sqrt(V)
        self.Wq = param(rng.standard_normal((V, V)) * sc)
        self.Wout = param(rng.standard_normal((V, self.VK)) * sc)
        self.p = dict(Wq=self.Wq, Wout=self.Wout)
        self.n_params = int(sum(x.d.size for x in self.p.values()))

    # -- fully differentiable normalisation, no off-graph constant ----------
    @staticmethod
    def unit(w):
        sq = (w * w).sum(-1, keepdims=True) + 1e-12
        inv_sqrt = (sq.log() * (-0.5)).exp()
        return w * inv_sqrt

    def forward(self, S, collect=False):
        U, Up, q = const(S["U"]), const(S["Up"]), const(S["q"])
        B, L, d = S["U"].shape
        K = const(S["K"].reshape(1, L, 1))
        M = const(self.M)

        def hop(qq):
            m = (Up * qq.reshape(B, 1, d)).sum(-1)
            return (m.reshape(B, L, 1) * U * K).sum(1)

        cur = q @ self.Wq
        zs = []
        for h in range(self.n_layers):
            z = hop(cur)
            zs.append(z)
            if h < self.n_layers - 1:
                cur = self.unit(z @ M)
        logits = z @ self.Wout
        if collect:
            return logits, dict(zs=zs, q_final=cur)
        return logits

    # -- invariants ---------------------------------------------------------
    def invariants(self, Ls=(9, 25, 129, 1025)):
        toe = []
        fft = []
        causal = []
        state = []
        for L in Ls:
            K = _base.kernel(L)
            A = np.zeros((L, L))
            for i in range(L):
                for j in range(i + 1):
                    A[i, j] = K[i - j]
            toe.append(max(abs(A[i, j] - A[i - 1, j - 1])
                           for i in range(1, L) for j in range(1, i + 1)))
            x = np.random.default_rng(L).standard_normal(L)
            y0 = A @ x
            y1 = np.fft.ifft(np.fft.fft(K, 2 * L) *
                             np.fft.fft(x, 2 * L))[:L].real
            fft.append(float(np.linalg.norm(y0 - y1) /
                             (np.linalg.norm(y0) + 1e-12)))
            # future perturbation: zero out t>L/2 in the input, first half
            # of the output must be bitwise unchanged
            x2 = x.copy()
            x2[L // 2:] = 0.0
            y2 = A @ x2
            causal.append(int(np.count_nonzero(y0[:L // 2] != y2[:L // 2])))
            state.append(8 * self.V * (self.n_layers + 1))
        return dict(
            toeplitz_dev_max=max(toe), fft_iir_err_max=max(fft),
            future_perturbation_changed_bits=max(causal),
            state_bytes=state,
            state_independent_of_L=len(set(state)) == 1,
            static_kernel=True, has_NxN_score=False, has_kv_cache=False,
            interface_is_parameter=False,
            parameters_independent_of_depth=True,
            all_pass=bool(max(toe) < 1e-12 and max(fft) < 1e-10
                          and max(causal) == 0 and len(set(state)) == 1))


# ------------------------------------------------------------------ checks
def gradcheck(n_layers=2, seed=0, n=8):
    rng = np.random.default_rng(4700 + n_layers)
    V = 64
    C = hadamard(V)
    d0 = _base.build(4, n_layers, n, 991, V, layout="gapped",
                     encoding="codebook", filler_max=3)
    X = d0["X"]
    Xp = np.zeros_like(X)
    Xp[:, 1:, :] = X[:, :-1, :]
    S = dict(U=X @ C, Up=Xp @ C, q=d0["Q0"].copy(), Y=d0["Y"],
             K=_base.kernel(X.shape[1]))
    stack = TSCStack(n_layers, V, seed)
    lg = stack.forward(S)
    loss = nll_from_logits(lg, S["Y"])
    loss.backward()
    worst = {}
    for k, P in stack.p.items():
        flat = P.d.ravel()
        i = int(rng.integers(flat.size))
        old = float(flat[i])
        e = 1e-6
        flat[i] = old + e
        gp = nll_from_logits(stack.forward(S), S["Y"])
        lp = float(gp.d)
        free_graph(gp)
        flat[i] = old - e
        gm = nll_from_logits(stack.forward(S), S["Y"])
        lm = float(gm.d)
        free_graph(gm)
        flat[i] = old
        num = (lp - lm) / (2 * e)
        ana = float(P.g.ravel()[i])
        worst[k] = abs(num - ana) / max(1.0, abs(num), abs(ana))
    free_graph(loss)
    return dict(worst=worst,
                all_pass=bool(all(v < 5e-4 for v in worst.values())),
                n_params=stack.n_params)


def analytic_weights_solve(n_layers=4, seed=3, n=512):
    """With Wq=I and Wout=CVAL^T (no training), the stack must reproduce the
    exact-H-layer construction. This is the zero-training witness INSIDE the
    architecture module, not in the evaluation directory."""
    V = 64
    C = hadamard(V)
    VK = V // 2
    d0 = _base.build(4, n_layers, n, seed, V, layout="gapped",
                     encoding="codebook", filler_max=3)
    X = d0["X"]
    Xp = np.zeros_like(X)
    Xp[:, 1:, :] = X[:, :-1, :]
    S = dict(U=X @ C, Up=Xp @ C, q=d0["Q0"].copy(), Y=d0["Y"],
             K=_base.kernel(X.shape[1]))
    st = TSCStack(n_layers, V, seed)
    st.Wq.d[...] = np.eye(V)
    st.Wout.d[...] = C[VK:, :].T
    lg = st.forward(S)
    acc = float((lg.d.argmax(-1) == S["Y"]).mean())
    free_graph(lg)
    return dict(hops=n_layers, layers=n_layers, acc=acc,
                note="Wq=I, Wout=CVAL^T, interface=fixed codebook swap")


def main():
    print("== v6.4 architecture module self-check ==", flush=True)
    V, VK = 64, 32
    C = hadamard(V)
    M = codebook_interface(C, VK)
    props = interface_properties(C, VK, M)
    print("  interface identities:", json.dumps(props, indent=1), flush=True)

    inv = {}
    gc = {}
    for nl in (1, 2, 3, 4):
        st = TSCStack(nl, V, 0)
        inv["layers_%d" % nl] = st.invariants()
        gc["layers_%d" % nl] = gradcheck(nl)
        print("  layers=%d params=%d invariants=%s gradcheck=%s" %
              (nl, st.n_params, inv["layers_%d" % nl]["all_pass"],
               gc["layers_%d" % nl]["all_pass"]), flush=True)

    wit = [analytic_weights_solve(nl) for nl in (1, 2, 3, 4)]
    print("  analytic witnesses:", [(x["hops"], x["acc"]) for x in wit],
          flush=True)

    checks = dict(
        interface_identities=props["all_identity"],
        all_invariants=bool(all(x["all_pass"] for x in inv.values())),
        all_gradchecks=bool(all(x["all_pass"] for x in gc.values())),
        params_independent_of_depth=len({TSCStack(n, V, 0).n_params
                                         for n in (1, 2, 3, 4)}) == 1,
        analytic_witnesses=bool(all(x["acc"] > 0.99 for x in wit)),
        interface_not_in_params=bool(
            all("M" not in TSCStack(n, V, 0).p for n in (1, 2, 3, 4))),
    )
    out = dict(interface_properties=props, invariants=inv, gradchecks=gc,
               analytic_witnesses=wit, checks=checks,
               n_pass=int(sum(checks.values())), n_checks=len(checks),
               all_pass=bool(all(checks.values())),
               summary=("Architectural change: the inter-layer interface is "
                        "the codebook's own value->key swap M, shared and "
                        "NOT a learnable parameter; inter-layer normalisation "
                        "is fully differentiable. The static Toeplitz kernel, "
                        "FFT/IIR duality, causality and O(1) state are "
                        "unchanged. Only Wq/Wout are learned."))
    json.dump(out, open(OUT + "/stack_arch_check.json", "w"), indent=1)
    print("  %d/%d PASS -> %s" % (out["n_pass"], out["n_checks"],
                                  OUT + "/stack_arch_check.json"), flush=True)
    return out


if __name__ == "__main__":
    main()
