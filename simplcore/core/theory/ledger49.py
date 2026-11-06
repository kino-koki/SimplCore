"""
Unified 49-item ledger (v4).  Single source of truth for classification.

  30 NEW THEOREMS  : NT01..NT30 (new_theorems15.py + new_theorems30.py)
   8 FALSIFIED     : F1..F8
   5 WITHDRAWN     : W1..W5 plus W7 (registered as a 6th row in the paper's
                     Appendix F; the 49-item audit counts 5 W-slots because W7
                     and F8 are the same underlying claim, adjudicated once)
   3 CONJECTURES   : C1, C3, C7
   3 EMPIRICAL     : E1, E2, E3  (no theorem can establish them)
  ----------------------------------------------------------------
  49 total, each adjudicated exactly once by verify49.py (49/49 PASS).

DISCIPLINE: PASS on a theorem item = finite-identity certification that the
statement is correctly formed with non-empty premises (the PROOF is analytic,
in the theorem modules).  PASS on a non-theorem item = correctly classified
with a reproducible witness / falsification spec, NOT 'statement is true'.

Nothing here is a measured accuracy promoted to a theorem.  No training.
"""
from __future__ import annotations

try:
    from .new_theorems30 import THEOREMS30
except ImportError:
    from new_theorems30 import THEOREMS30

# 19 non-theorem entries: (id, verdict, short_name, why_not_a_theorem)
NON_THEOREM = [
    ("F1", "FALSIFIED", "rho_max extends window", "NT07/NT27 interior optimum; rho->1 shortens resolvable length"),
    ("F2", "FALSIFIED", "SFSC preserves FFT duality", "time-varying a_t non-Toeplitz (NT12); FFT-proxy error 0.9720"),
    ("F3", "FALSIFIED", "phase resonance beats attention", "witness 0.8672 < 1.0000 at ~equal params"),
    ("F4", "FALSIFIED", "addressing <-> duality exclusive", "withdrawn via NT01/NT10; this withdrawal IS a core contribution"),
    ("F5", "FALSIFIED", "oscillatory basis is the advantage", "random code also 1.0000 (NT08); NT16 shows conjugate poles not strictly positive"),
    ("F6", "FALSIFIED", "dynamic scaling = spurious artifact", "same-sample prefixes; NT26 proves per-step peak causally pointwise-minimal"),
    ("F7", "FALSIFIED", "Shannon capacity limits retrieval", "1,280 required bits vs 225,280-bit state (176x headroom)"),
    ("F8", "FALSIFIED", "scalar monotone score is the gate", "NT20: gate is the UNIQUE multi-metric AND rule"),
    ("W1", "WITHDRAWN", "unconditional fp32 bitwise duality", "real-field only (NT09)"),
    ("W2", "WITHDRAWN", "periodic-axis superiority", "not replicated (O4)"),
    ("W3", "WITHDRAWN", "equivalent 4-7B parameters", "no supporting measurement"),
    ("W4", "WITHDRAWN", "NT13 unrestricted read-out", "exactly four premises required"),
    ("W5", "WITHDRAWN", "dynamic scaling spurious effects", "same-sample prefix controls (O6); W7/F8 register the T16 variant"),
    ("C1", "CONJECTURE", "stacking restores multi-hop", "no proof; NT11/NT28 give only necessary lower bounds; falsify with L=2,3 stacks"),
    ("C3", "CONJECTURE", "NT13 extends to deep nonlinear stacks", "no proof; cross-terms leave the fixed-summary premise; 2-layer counterexample spec"),
    ("C7", "CONJECTURE", "outside EVERY gated linear-attention variant", "no proof; NT03 settles only kernel families; exhibit a gated variant with same kernel"),
    ("E1", "EMPIRICAL_NOT_TESTABLE", ">=10M-token equal-compute LM parity", "requires >=10M tokens, >=5 seeds, paired CI, same-run Mamba-2/GLA/DeltaNet/Hyena/TS"),
    ("E2", "EMPIRICAL_NOT_TESTABLE", "RSI reaches SOTA", "NT24 gives a necessary rate bound, not capability; no theorem can establish"),
    ("E3", "EMPIRICAL_NOT_TESTABLE", "absolute throughput advantage", "O6 is an unoptimized reference implementation; only scaling is informative"),
]


def items():
    """All 49 statements as (id, verdict, name, group, note)."""
    out = []
    for t in THEOREMS30:
        out.append((t.nid, "NEW_THEOREM", t.title, t.group,
                    "complete analytic proof; " + t.use[:120]))
    for qid, v, name, why in NON_THEOREM:
        out.append((qid, v, name, "non-theorem", why))
    return out


def by_verdict() -> dict:
    d = {}
    for _, v, *_ in items():
        d[v] = d.get(v, 0) + 1
    return d


def summary() -> dict:
    return dict(total=49, theorems=len(THEOREMS30),
                non_theorems=len(NON_THEOREM),
                verdict_distribution=by_verdict(),
                honest_note=(
                    "49 statements cannot all be new non-trivial theorems: 8 are "
                    "false, 5 withdrawn, 3 unproved, 3 empirical. Inflating them "
                    "would repeat the theorem inflation this revision removes."))


__all__ = ["ITEMS", "NON_THEOREM", "items", "by_verdict", "summary"]

ITEMS = items()
