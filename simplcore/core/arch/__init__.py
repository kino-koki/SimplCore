"""SimplCore architecture implementations.

The research candidate is loaded lazily so importing lightweight configuration
and registry modules does not pull in PyTorch.
"""

__all__ = [
    "FluxLedgerBlock",
    "FluxLedgerMixer",
    "FluxLedgerState",
    "SimplCoreFluxLanguageModel",
]


def __getattr__(name: str):
    if name in __all__:
        from . import flux_ledger
        return getattr(flux_ledger, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
