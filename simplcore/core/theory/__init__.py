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

CURRENT SUBMISSION STATE (v4):
  simplcore.pdf now carries 30 de-duplicated theorems NT01..NT30 in six
  groups, complete proofs in Appendix A, and a 49-item audit ledger in
  Appendix G (30 theorems + 8 falsified + 5 withdrawn + 3 conjectures +
  3 empirical claims).  The historical T7..T16 ids in statements.py are
  retained ONLY as an audit trail; T16 is WITHDRAWN (W7).

  19 statements are deliberately NOT theorems.  Promoting them would be
  theorem inflation; see verify49.py / ledger49.py.

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
from .new_theorems15 import NT as THEOREMS15, Thm
from .new_theorems30 import THEOREMS30, NT16_30
from .ledger49 import ITEMS, NON_THEOREM, items, by_verdict, summary as ledger_summary

__all__ = [
    "Proposition", "PROPOSITIONS", "THEOREMS", "BY_MODULE", "summary",
    "CoupledArchitecture", "CT6", "CT7", "DETERMINISTIC_CODE",
    "CORE_THEOREMS", "DEMOTED_TO_LEMMA", "WITHDRAWN", "FALSIFIED",
    "CONJECTURE_DISPOSITION", "core_summary",
    "ZeroOrderBridge", "coupling_spec",
    "Thm", "THEOREMS15", "THEOREMS30", "NT16_30",
    "ITEMS", "NON_THEOREM", "items", "by_verdict", "ledger_summary",
]
