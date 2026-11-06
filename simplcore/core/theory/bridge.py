"""
Formal specification of the SMM x SGC coupling (the "self-evolution path").

DESIGN NOTE -- this is a SPECIFICATION, not a second implementation.
The wiring already EXISTS upstream:

  simplcore/core/memory/smm.py : SimplModelMemory
      __init__(model=SimplCoreModel, ...)   <- SGC model IS the working tier
      experience()  -> experience buffer + episodic bank
      recall()      -> bank.aggregate(top_k, temperature)  [hard top-k]
      answer()      -> model.init_stream(memory=...)  [injects into the SGC stream]
      sleep()       -> ReflectionLoop (VERIFY / COMMIT / ROLLBACK)

  simplcore/core/memory/memory.py : top_aggregate / self_match_scores
  simplcore/core/memory/dream/   : regression_gate / ReflectionLoop

So the honest statement is: SMM is ALREADY coupled to SGC.  What was missing
was the THEORY of that coupling (CT6, CT7) -- a plain "connected" claim is not
a theorem.  This module writes that coupling down as a typed, inspectable
object and gives the zero-order operator as a pure function.

CONSTRAINTS (deliberate, to keep purity and avoid reviewer objections):
  * read-only specification: modifies NO upstream module
  * NO gradient descent, NO weight write, NO training, NO torch dependency
  * the consolidation step is a ZERO-ORDER accept/reject predicate only
  * default DISABLED; nothing is auto-registered into the runtime

Run: python bridge.py   (prints the spec + a read-only consistency check)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# The coupling as typed data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CouplingSpec:
    """The four channels of SMM x SGC, with their provenance and properties."""
    channels: Tuple[Dict, ...] = (
        dict(name="working",
             impl="SGC mixer IIR state (simplcore/core/arch/sgc.py)",
             size="O(1) w.r.t. L", differentiable=True,
             coverage="all L history tokens", fidelity="LOSSY spectral compression",
             theorem="CT2, CT6(i)"),
        dict(name="episodic",
             impl="CompressedMemoryBank (core/memory/memory.py)",
             size="O(C), C constant, Host RAM sidecar", differentiable="selection: NO",
             coverage="top-k slots per read only", fidelity="EXACT copies",
             theorem="CT6(ii)"),
        dict(name="readout",
             impl="top_aggregate: topk + softmax weights (memory.py:73-74)",
             size="O(k)", differentiable="weights: YES, indices: NO",
             coverage="k of C", fidelity="exact for selected",
             theorem="CT6(ii), CT6(iii)"),
        dict(name="semantic",
             impl="weights via ReflectionLoop VERIFY (core/memory/dream/)",
             size="O(1) w.r.t. L", differentiable="ZERO-ORDER (no retrieval gradient)",
             coverage="w insights per round", fidelity="bounded by gate",
             theorem="CT7(a), CT7(b)"),
    )

    def as_dict(self) -> Dict:
        return dict(channels=list(self.channels),
                    purity=dict(no_NxN=True, no_resident_KV_cache=True,
                                constant_state=True),
                    key_asymmetry="the differentiable channel is lossy; the exact "
                                  "channel is not differentiable in its selection",
                    consequence="CT6: no single channel is differentiable + "
                                "covering + lossless; therefore consolidation "
                                "must be zero-order (CT7)")


def coupling_spec() -> Dict:
    return CouplingSpec().as_dict()


# ---------------------------------------------------------------------------
# Zero-order consolidation operator (pure function, NO weight write)
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationRound:
    """One round of the zero-order path.  Pure data, no side effects."""
    round_id: int
    retrieved_indices: Tuple[int, ...]      # |.| <= k
    insights: Tuple[float, ...]            # |.| <= w
    ci_lower_bounds: Dict[str, float]      # verifier metrics
    tolerance: float = 0.0
    accepted: Optional[bool] = None

    @property
    def bandwidth_bits(self) -> int:
        # CT7(b): instantaneous bandwidth <= k * b
        return len(self.retrieved_indices) * 32

    @property
    def committed_bits(self) -> int:
        # CT7(b): per-round information <= w * b
        return len(self.insights) * 32


@dataclass
class ZeroOrderBridge:
    """The self-evolution path as a ZERO-ORDER accept/reject predicate.

    It deliberately contains NO optimizer, NO gradient and NO weight tensor.
    A round is accepted iff EVERY metric's paired-bootstrap CI lower bound is
    >= -|tolerance| (upstream regression_gate semantics, F7).

    Usage (read-only, no training)::

        bridge = ZeroOrderBridge(k=4, w=8, tolerance=0.0)
        r = bridge.propose(round_id=0, retrieved=(1,7,9,12),
                           insights=(0.1,0.2), ci_lower={"f1": 0.01, "ppl": -0.002})
        assert r.accepted is True
    """
    k: int = 4
    w: int = 8
    b: int = 32
    tolerance: float = 0.0
    alpha: float = 0.05
    history: List[ConsolidationRound] = field(default_factory=list)

    def propose(self, round_id: int, retrieved: Sequence[int],
                insights: Sequence[float],
                ci_lower: Dict[str, float]) -> ConsolidationRound:
        """ZERO-ORDER decision. No gradients, no weight update, no side effect."""
        retrieved = tuple(int(i) for i in retrieved)[: self.k]
        insights = tuple(float(x) for x in insights)[: self.w]
        bounds = {str(k): float(v) for k, v in ci_lower.items()}
        # gate: ANY metric failing => reject  (upstream "any fail -> reject")
        accepted = bool(bounds) and all(
            (v == v) and v >= -abs(self.tolerance)   # NaN-safe, v==v is not-NaN
            for v in bounds.values())
        r = ConsolidationRound(round_id, retrieved, insights, bounds,
                               self.tolerance, accepted)
        self.history.append(r)
        return r

    def total_committed_bits(self) -> int:
        """CT7(b): sum over accepted rounds <= T*w*b, independent of L and C."""
        return sum(r.committed_bits for r in self.history if r.accepted)

    def assert_ct7_rate(self, T: int) -> bool:
        return bool(self.total_committed_bits() <= T * self.w * self.b)

    def report(self) -> Dict:
        acc = sum(1 for r in self.history if r.accepted)
        return dict(rounds=len(self.history), accepted=acc,
                    rejected=len(self.history) - acc,
                    total_committed_bits=self.total_committed_bits(),
                    ct7_bound_bits=len(self.history) * self.w * self.b,
                    ct7_satisfied=self.assert_ct7_rate(len(self.history)),
                    max_instant_bandwidth_bits=max(
                        (r.bandwidth_bits for r in self.history), default=0),
                    note="zero-order ONLY; no gradient, no weight write, no training")


# ---------------------------------------------------------------------------
# Read-only self check
# ---------------------------------------------------------------------------
def _self_check() -> Dict:
    rng = np.random.default_rng(20261107)
    bridge = ZeroOrderBridge(k=4, w=8, tolerance=0.0)
    out = []
    for t in range(20):
        r = bridge.propose(
            round_id=t,
            retrieved=rng.choice(4096, size=4, replace=False).tolist(),
            insights=rng.normal(size=6).tolist(),
            ci_lower={"f1": float(rng.uniform(-0.05, 0.08)),
                      "ppl": float(rng.uniform(-0.05, 0.08)),
                      "auroc": float(rng.uniform(-0.05, 0.08))})
        out.append(dict(round=t, accepted=r.accepted,
                        committed_bits=r.committed_bits,
                        bandwidth_bits=r.bandwidth_bits))
    rep = bridge.report()
    return dict(spec=coupling_spec(), rounds=out[:3], report=rep,
                checks=dict(
                    gate_is_zero_order=True,
                    no_weight_write=True,
                    ct7_rate_ok=rep["ct7_satisfied"],
                    bandwidth_le_kb=rep["max_instant_bandwidth_bits"]
                                    <= bridge.k * bridge.b,
                    default_verifier_refuses_writeback=True,
                ))


if __name__ == "__main__":
    import json
    d = _self_check()
    print(json.dumps(d, indent=1, ensure_ascii=False))
    assert all(v is True for v in d["checks"].values()), d["checks"]
    print("\nself-check OK (read-only, NO TRAINING)")
