from __future__ import annotations

from typing import Dict

from simplcore.mix.stacking.manifest import StackManifest


def build_calibration_plan(
    manifest: StackManifest,
    *,
    target_loss: float = 1.5,
    max_loss: float = 3.9,
) -> Dict[str, object]:
    """为 7B 堆叠后校准生成可追踪计划。"""
    sources = []
    for source in manifest.sources:
        data = source.to_dict()
        loss = data.get("loss")
        data["eligible_for_stacking"] = loss is None or float(loss) <= max_loss
        data["meets_target_loss"] = loss is not None and float(loss) < target_loss
        sources.append(data)

    return {
        "target_name": manifest.target_name,
        "loss_guard": {
            "target_loss": target_loss,
            "max_loss": max_loss,
            "on_exceed_max": "stop_and_rollback",
            "on_between_target_and_max": "continue_with_lr_decay_and_data_check",
        },
        "calibration_stages": [
            "shape_dtype_manifest_check",
            "sampled_forward_check",
            "loss_guarded_lora_or_layerwise_calibration",
            "checkpoint_export_after_loss_stabilizes",
        ],
        "sources": sources,
    }
