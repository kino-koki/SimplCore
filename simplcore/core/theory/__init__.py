"""
Theory layer for SimplCore.

This package carries formal statements and machine-checkable verification of
the architecture's expressivity boundaries.  It contains NO training code,
NO model weights, NO experimental data, and does not modify any existing
runtime module.

Contents
--------
* statements.py : 15 candidate propositions with hypothesis, claim, proof,
                   verdict (10 THEOREM / 3 FALSIFIED / 2 CONJECTURE), the
                   architecture module each one constrains, and its check id.
* verify.py     : pure-numpy verification, fixed seed, NO TRAINING.
                   Run: python -m simplcore.core.theory.verify

Theorem numbering continues the six theorems of the submission manuscript
(simplcore.pdf): T7..T16 are defined here.  They are NOT yet merged into that
PDF; see docs/CONJECTURE_15_THEOREM_UPGRADE.md section 6.

Architecture mapping
--------------------
  T7, T8, T9, T15  -> Tensor Spectral Convolution / core/arch/sgc.py
  T10, T11         -> static-kernel impossibility & general P-order IIR
  T12, T13, T14    -> decay stability and low-precision deploy path
  T16              -> core/memory/smm.py (Simpl Model Memory) + memory/dream/

Verdict discipline: a theorem is a universal statement derived only from
definitions.  Measured accuracies remain finite-sample observations and are
never promoted to theorems here.
"""
from .statements import (
    Proposition,
    PROPOSITIONS,
    THEOREMS,
    BY_MODULE,
    summary,
)
from .smm_sgc import (
    CoupledArchitecture,
    CT6,
    CT7,
    DETERMINISTIC_CODE,
    CORE_THEOREMS,
    DEMOTED_TO_LEMMA,
    WITHDRAWN,
    FALSIFIED,
    CONJECTURE_DISPOSITION,
    summary as core_summary,
)
from .bridge import ZeroOrderBridge, coupling_spec

__all__ = [
    "Proposition", "PROPOSITIONS", "THEOREMS", "BY_MODULE", "summary",
    "CoupledArchitecture", "CT6", "CT7", "DETERMINISTIC_CODE",
    "CORE_THEOREMS", "DEMOTED_TO_LEMMA", "WITHDRAWN", "FALSIFIED",
    "CONJECTURE_DISPOSITION", "core_summary",
    "ZeroOrderBridge", "coupling_spec",
]
