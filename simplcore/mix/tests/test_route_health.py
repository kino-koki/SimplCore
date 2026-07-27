def test_route_has_pure_text_targets():
    from simplcore.mix import mix_150m_text, mix_7b_text, mix_14b_text

    assert mix_150m_text().cognitive_enabled is False
    assert mix_7b_text().cognitive_enabled is False
    assert mix_14b_text().cognitive_enabled is False
