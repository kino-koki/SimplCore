def test_frontier_target_declares_t0_gates():
    from simplcore.mix import SCMilestoneSpec

    target = SCMilestoneSpec.default()

    assert target.current_tier == "T3+"
    assert target.target_tier == "T0"
    assert "150M完整训练" in target.required_gates
    assert "7B参数堆叠校准" in target.required_gates
    assert "中文Benchmark评测" in target.required_gates
    assert "消融实验" in target.required_gates


def test_model_architecture_info_reports_frontier_target():
    from simplcore.mix import SimplCoreMix, mix_tiny_text

    info = SimplCoreMix(mix_tiny_text()).get_architecture_info()

    assert info["current_tier"] == "T3+"
    assert info["target_tier"] == "T0"
    assert "frontier_gates" in info
    assert "150M完整训练" in info["frontier_gates"]
