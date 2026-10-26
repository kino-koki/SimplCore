def test_self_owned_core_exports_simplcore_named_layers():
    from simplcore.mix.core import (
        SCBackend,
        SCEmbedding,
        SCFunctional,
        SCLinear,
        SCModule,
        SCParameter,
        SCRMSNorm,
    )

    assert SCBackend.default().name == "simplcore_native"
    assert SCModule.__name__ == "SCModule"
    assert SCLinear.__name__ == "SCLinear"
    assert SCEmbedding.__name__ == "SCEmbedding"
    assert SCRMSNorm.__name__ == "SCRMSNorm"
    assert SCParameter.__name__ == "SCParameter"
    assert hasattr(SCFunctional, "softmax")


def test_mix_model_uses_simplcore_module_identity():
    from simplcore.mix import SCModule, SimplCoreMix, mix_tiny_text

    model = SimplCoreMix(mix_tiny_text())
    info = model.get_architecture_info()

    assert isinstance(model, SCModule)
    assert info["architecture"] == "SimplCore Mix"
    assert info["backend"] == "simplcore_native"
    assert info["self_owned_surface"] is True


def test_core_model_files_do_not_use_public_pytorch_layer_names():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    checked = ["attention.py", "moe.py", "memory.py", "blocks.py", "model.py"]
    forbidden = [
        "class ModulatedBasisAttention(nn.Module)",
        "class ModulatedBasisMLAttention(nn.Module)",
        "class Expert(nn.Module)",
        "class PrototypeMoE(nn.Module)",
        "class WorkingMemoryBuffer(nn.Module)",
        "class LTIInjection(nn.Module)",
        "class LoRAAdapter(nn.Module)",
        "class ACTHalting(nn.Module)",
        "class PreludeBlock(nn.Module)",
        "class RecurrentBlock(nn.Module)",
        "class CodaBlock(nn.Module)",
        "nn.Linear",
        "nn.Parameter",
        "nn.Dropout",
        "nn.Embedding",
        "nn.ModuleList",
    ]

    violations = []
    for filename in checked:
        text = (root / filename).read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                violations.append(f"{filename}: {marker}")

    assert violations == []


def test_mix_recurrent_block_contains_simplcore_fusion_components():
    from simplcore.mix import SimplCoreMix, mix_tiny_text

    model = SimplCoreMix(mix_tiny_text())
    recurrent = model.recurrent
    info = model.get_architecture_info()

    assert hasattr(recurrent, "causal_intervention")
    assert hasattr(recurrent, "information_bottleneck")
    assert hasattr(recurrent, "meta_distiller")
    assert hasattr(recurrent, "plasticity_update")
    assert "DeepCIN深度因果干预网络 (DCGD+DDI+BCA)" in info["components"]["SimplCore"]
    assert "IB信息瓶颈" in info["components"]["SimplCore"]
    assert "MRD元表示蒸馏" in info["components"]["SimplCore"]
    assert "PUM可塑性更新模块" in info["components"]["SimplCore"]
