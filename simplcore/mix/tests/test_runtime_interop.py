import json


def test_runtime_profile_targets_python311_and_java26():
    from simplcore.mix import SCRuntimeProfile

    profile = SCRuntimeProfile.default()

    assert profile.python_target == "3.11"
    assert profile.java_target == "26"
    assert profile.ecosystem_mode == "python_ecosystem"
    assert profile.performance_mode == "java26_hotpath"
    assert profile.target_tier == "T0"
    assert "simplcore_native" in profile.backends
    assert "simplcore_java26" in profile.backends


def test_simplcore_data_and_optimizer_aliases_are_public():
    from simplcore.mix import SCDataLoader, SCOptimizer, SCAdamW

    assert SCDataLoader.__name__ == "SCDataLoader"
    assert SCOptimizer.__name__ == "SCOptimizer"
    assert SCAdamW.__name__ == "SCAdamW"


def test_export_java_manifest_records_java26_and_python311(tmp_path):
    from simplcore.mix import SimplCoreMix, export_java26_manifest, mix_tiny_text

    model = SimplCoreMix(mix_tiny_text())
    manifest_path = tmp_path / "java26_manifest.json"

    manifest = export_java26_manifest(model, manifest_path)

    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == saved
    assert saved["format"] == "simplcore_mix_java26_manifest"
    assert saved["runtime"]["java_target"] == "26"
    assert saved["runtime"]["python_target"] == "3.11"
    assert saved["runtime"]["target_tier"] == "T0"
    assert saved["architecture"] == "SimplCore Mix"
    assert saved["backend"] == "simplcore_native"
    assert saved["tensor_count"] > 0
    assert saved["weights"][0]["name"]
