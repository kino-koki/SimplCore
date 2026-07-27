from simplcore.mix.stacking.build_targets import build_target_config


def test_build_7b_target_config():
    cfg = build_target_config("7b")
    assert cfg.vocab_size == 32000
    assert cfg.dim >= 4096
    assert cfg.n_experts >= 64


def test_build_14b_target_config():
    cfg = build_target_config("14b")
    assert cfg.vocab_size == 32000
    assert cfg.dim >= 5120
    assert cfg.n_experts >= 96
