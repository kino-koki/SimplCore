def test_mix_imports_core_symbols():
    from simplcore.mix import SimplCoreMix, SimplCoreMixConfig, mix_tiny_text

    cfg = mix_tiny_text()
    assert cfg.vocab_size > 0
    assert cfg.dim > 0
    assert SimplCoreMix is not None
    assert SimplCoreMixConfig is not None
