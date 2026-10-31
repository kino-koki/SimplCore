# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Round-trip tests for the safetensors export path (`simplcore.core.export`)."""
from __future__ import annotations

import json
import os

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.export import (CONFIG_NAME, WEIGHTS_NAME,
                                   export_training_ckpt, load_pretrained,
                                   save_pretrained)
from simplcore.core.model import SimplCoreModel


def _cfg(tie: bool = True, mixer: str = "sgc", **kw) -> SimplCoreConfig:
    base = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                local_kernel=4, mixer=mixer, num_heads=4,
                tie_embeddings=tie)
    base.update(kw)
    return SimplCoreConfig(**base)


def _assert_same_model(a: SimplCoreModel, b: SimplCoreModel) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"tensor mismatch after round-trip: {k}"
    torch.manual_seed(7)
    x = torch.randint(0, a.config.vocab_size, (1, 16))
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("mixer", ["sgc", "srf"])
def test_round_trip(tmp_path, tie, mixer):
    torch.manual_seed(0)
    m = SimplCoreModel(_cfg(tie=tie, mixer=mixer, srf_heads=2, srf_chunk=8))
    out = str(tmp_path / "export")
    save_pretrained(m, out)
    assert os.path.exists(os.path.join(out, CONFIG_NAME))
    assert os.path.exists(os.path.join(out, WEIGHTS_NAME))
    m2 = load_pretrained(out)
    assert m2.config.tie_embeddings == tie
    _assert_same_model(m, m2)


def test_tied_head_key_is_dropped_from_file(tmp_path):
    """safetensors forbids shared tensors: the duplicate head key must not be stored."""
    torch.manual_seed(0)
    m = SimplCoreModel(_cfg(tie=True))
    out = str(tmp_path / "tied")
    save_pretrained(m, out)
    from safetensors import safe_open
    with safe_open(os.path.join(out, WEIGHTS_NAME), framework="pt") as f:
        keys = set(f.keys())
    assert "tok_emb.weight" in keys
    assert "head.weight" not in keys

    torch.manual_seed(0)
    m3 = SimplCoreModel(_cfg(tie=False))
    out3 = str(tmp_path / "untied")
    save_pretrained(m3, out3)
    with safe_open(os.path.join(out3, WEIGHTS_NAME), framework="pt") as f:
        keys3 = set(f.keys())
    assert "head.weight" in keys3 and "tok_emb.weight" in keys3


def test_export_from_training_checkpoint(tmp_path):
    """A real train.py checkpoint converts to a loadable, behaviourally-equal model."""
    from simplcore.core.train import build_param_groups, save_checkpoint

    torch.manual_seed(0)
    cfg = _cfg()
    m = SimplCoreModel(cfg)
    opt = torch.optim.AdamW(build_param_groups(m, 0.1), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    for _ in range(2):
        _, loss = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ckpt = str(tmp_path / "last.pt")
    save_checkpoint(ckpt, m, opt, step=2, tokens=64, cfg=cfg,
                    best_val=9.5, elapsed=1.5)

    out = str(tmp_path / "deploy")
    meta = export_training_ckpt(ckpt, out)
    assert meta["step"] == 2 and meta["tokens"] == 64
    assert meta["best_val"] == 9.5
    assert meta["params"] == m.num_params()

    recorded = json.load(open(os.path.join(out, "export_meta.json"), encoding="utf-8"))
    assert recorded["source_checkpoint"].endswith("last.pt")

    m2 = load_pretrained(out)
    _assert_same_model(m, m2)


def test_export_rejects_foreign_checkpoint(tmp_path):
    bad = tmp_path / "foreign.pt"
    torch.save({"state_dict": {}}, bad)
    with pytest.raises(ValueError):
        export_training_ckpt(str(bad), str(tmp_path / "out"))


def test_tokenizer_copy(tmp_path):
    torch.manual_seed(0)
    m = SimplCoreModel(_cfg())
    tok = tmp_path / "tokenizer.json"
    tok.write_text('{"version": "test"}', encoding="utf-8")
    out = str(tmp_path / "withtok")
    save_pretrained(m, out, tokenizer_json=str(tok))
    copied = os.path.join(out, "tokenizer.json")
    assert os.path.exists(copied)
    assert open(copied, encoding="utf-8").read() == '{"version": "test"}'
