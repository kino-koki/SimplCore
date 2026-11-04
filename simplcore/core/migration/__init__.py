"""Migration-side, zero-training repairs for TS2SC weight transfer.

见 :mod:`simplcore.core.migration.dc_bias` 与
``docs/TS2SC_coherence_audit.md``。
"""
from .dc_bias import (LayerwiseDCBias, apply_dc_bias, estimate_dc_bias,
                      estimate_layerwise_bias)

__all__ = ["estimate_dc_bias", "estimate_layerwise_bias",
           "apply_dc_bias", "LayerwiseDCBias"]
