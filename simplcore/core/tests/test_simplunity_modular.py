# -*- coding: utf-8 -*-
"""SimplUnity 模块化 / 后台并行 / 无回归门 / 经验 loader 的真实闭环测试。"""
from __future__ import annotations

import time

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.envlearn import WeightBudget
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharTokenizer
from simplcore.unity import (
    ExperienceLoader, ModuleBundle, UnityConfig, UnityModel,
    UnitySelfEvolution, for_preset,
)
from simplcore.unity.expert import Expert


def _base(**over) -> SimplCoreModel:
    torch.manual_seed(0)
    kw = dict(vocab_size=64, dim=32, num_layers=2, num_bases=4,
              ffn_ratio=2.0)
    kw.update(over)
    cfg = SimplCoreConfig(**kw)
    return SimplCoreModel(cfg).eval()


def _unity(base=None, **kw) -> UnityModel:
    base = base or _base()
    cfg = UnityConfig(num_experts=kw.pop("num_experts", 1),
                      expert_hidden=kw.pop("expert_hidden", 16),
                      expert_layers=kw.pop("expert_layers", 1),
                      top_k=kw.pop("top_k", 1),
                      expert_dropout=kw.pop("expert_dropout", 0.0),
                      **kw)
    return UnityModel(base, cfg, budget=WeightBudget(
        base_params=sum(p.numel() for p in base.parameters()),
        max_learned=10_000_000, total_cap=12_000_000))


def _batches(n: int = 2, seq: int = 8):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        x = torch.randint(0, 64, (1, seq), generator=g)
        out.append((x, x.clone()))
    return out


# ---------------------------------------------------------------------------
# 1. 模块包：导出 / 跨主干拼接 / 兼容性拒绝
# ---------------------------------------------------------------------------
def test_bundle_roundtrip_and_cross_backbone_attach(tmp_path) -> None:
    a = _unity(num_experts=2)
    bundle = a.export_expert(0, trained_tokens=1234, val_loss=1.5)
    assert bundle.kind == "unity_expert"
    assert bundle.metadata["trained_tokens"] == 1234
    path = tmp_path / "expert.pt"
    bundle.save(path)
    loaded = ModuleBundle.load(path)
    assert loaded.dim == a.dim and loaded.hidden == 16

    b = _unity(num_experts=1)
    idx = b.import_bundle(loaded, name="ported")
    assert idx == 1 and b.experts[idx].name == "ported"
    for (k1, v1), (k2, v2) in zip(a.experts[0].state_dict().items(),
                                  b.experts[idx].state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    assert b.expert_meta[idx]["trained_tokens"] == 1234


def test_bundle_rejects_incompatible_backbone(tmp_path) -> None:
    a = _unity(num_experts=1)
    bundle = a.export_expert(0)
    # 同 dim 但深度不同 -> base_fingerprint 不同
    b = _unity(base=_base(num_layers=3), num_experts=1)
    with pytest.raises(ValueError, match="不兼容"):
        b.import_bundle(bundle)


def test_budget_release_roundtrip() -> None:
    b = WeightBudget(base_params=100, max_learned=50, total_cap=200)
    b.allocate(30)
    assert b.learned_params == 30
    b.release(30)
    assert b.learned_params == 0


def test_save_load_learned_keeps_metadata(tmp_path) -> None:
    u = _unity(num_experts=1)
    u.expert_meta[0] = {"name": "x", "trained_tokens": 5, "val_loss": 1.2}
    path = tmp_path / "unity.pt"
    u.save_learned(path)
    v = UnityModel.load_learned(_base(), path)
    assert v.expert_meta[0]["trained_tokens"] == 5
    assert v.base_fingerprint == u.base_fingerprint


# ---------------------------------------------------------------------------
# 2. 后台并行 + 无回归门
# ---------------------------------------------------------------------------
def test_multi_worker_evolution_with_gate() -> None:
    u = _unity(num_experts=1)
    evo = UnitySelfEvolution(u, steps_per_domain=1, lr=1e-3, num_workers=2)
    evo.start()
    try:
        # 接受：门限放到极大
        evo.register_domain("accept", iter(_batches()),
                            eval_loader=_batches(), max_regression=1e9)
        # 拒绝：门限为负，候选必然"退化"
        evo.register_domain("reject", iter(_batches()),
                            eval_loader=_batches(), max_regression=-1e9)
        deadline = time.time() + 20
        while evo.num_pending > 0 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        evo.stop()
    st = evo.status()
    assert "accept" in st["done"]
    assert "reject" in st["rejected"]
    assert len(u.experts) == 2          # 只有 accept 被发布


def test_evolution_status_reports_budget() -> None:
    u = _unity(num_experts=1)
    evo = UnitySelfEvolution(u, num_workers=1)
    st = evo.status()
    assert st["workers"] == 1
    assert st["budget"]["learned_params"] >= 0
    assert st["rejected"] == []


# ---------------------------------------------------------------------------
# 3. 经验 -> loader
# ---------------------------------------------------------------------------
def test_experience_loader_next_token_shift() -> None:
    codec = CharTokenizer(limit=64)
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    loader = ExperienceLoader(codec, [text], seq_len=16, batch_size=2,
                              repeat=1)
    assert len(loader) >= 1
    x, y = next(iter(loader))
    assert x.shape == y.shape == (2, 16)
    ids = torch.tensor(codec.encode(text), dtype=torch.long)
    assert torch.equal(x[0], ids[:16])
    assert torch.equal(y[0], ids[1:17])


def test_experience_loader_skips_short_text() -> None:
    codec = CharTokenizer(limit=64)
    loader = ExperienceLoader(codec, ["x"], seq_len=8)
    assert len(loader) == 0
    assert list(loader) == []


# ---------------------------------------------------------------------------
# 4. 2B 的 Unity 预设（不实例化 2B 模型）
# ---------------------------------------------------------------------------
def test_unity_2b_preset_budget() -> None:
    cfg = for_preset("tuan-2b")
    r = cfg.resolve(2048)
    assert r.num_experts == 4 and r.top_k == 2
    assert r.expert_hidden == 4096 and r.expert_layers == 2
    # 解析估算：每层 3*dim*hidden，2 层，4 个专家
    per_expert = 2 * 3 * 2048 * 4096
    assert per_expert * 4 < 2_000_000_000
    assert 2_024_000_000 + per_expert * 4 < 3_000_000_000