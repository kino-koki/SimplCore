import simplcore.tensor as torch

from simplcore.mix.stacking.manifest import SourceCheckpoint, StackManifest
from simplcore.mix.stacking.mapper import copy_matching_tensors_strict


def test_stack_manifest_records_source_and_target():
    source = SourceCheckpoint(path="small-000.pt", role="prelude", shard_id=0)
    manifest = StackManifest(target_name="mix-7b-text", sources=[source])
    data = manifest.to_dict()
    assert data["target_name"] == "mix-7b-text"
    assert data["sources"][0]["role"] == "prelude"


def test_copy_matching_tensors_strict_rejects_shape_mismatch():
    source = {"a.weight": torch.zeros(2, 2)}
    target = {"a.weight": torch.zeros(3, 3)}
    try:
        copy_matching_tensors_strict(source, target)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch should fail")


def test_copy_matching_tensors_strict_copies_equal_shapes():
    source = {"a.weight": torch.ones(2, 2)}
    target = {"a.weight": torch.zeros(2, 2)}
    copied = copy_matching_tensors_strict(source, target)
    assert copied == ["a.weight"]
    assert torch.equal(target["a.weight"], torch.ones(2, 2))
