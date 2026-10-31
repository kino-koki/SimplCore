# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长上下文外推 + 衰减消融工具守护测试.

守护 ``simplcore/core/extrapolation.py`` 的**管线机制**（不守收敛质量，
后者需 A100 规模训练）：
  * 外推探针：跨倍率评测，``stream_bytes`` 恒为常数（与 L 无关，O(1) 主张）；
  * 任务构造：不同 ``seq_len`` 的任务采样形状正确；
  * 衰减消融：不同 ``decay_max`` 产出不同模型、管线不崩；
  * 结果数据结构：字段完整、类型正确。

预算：全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import get_preset, SimplCoreConfig          # noqa: E402
from simplcore.core.extrapolation import (                          # noqa: E402
    extrapolation_probe,
    decay_ablation,
    sgc_model_factory,
    _make_task,
    ExtrapolationResult,
    DecayPoint,
)


def _tiny_cfg(dim=32, layers=2):
    cfg = get_preset("debug")
    cfg.dim = dim
    cfg.num_layers = layers
    cfg.num_heads = 4
    return cfg


# ---------------------------------------------------------------------------
# 1. 任务构造
# ---------------------------------------------------------------------------
def test_make_task_shapes():
    t = _make_task("selective-copy", seq_len=64)
    x, y = t.sample(8, torch.Generator().manual_seed(0))
    assert x.shape == (8, 64)
    t2 = _make_task("selective-copy", seq_len=128)
    x2, _ = t2.sample(8, torch.Generator().manual_seed(0))
    assert x2.shape == (8, 128)


def test_make_task_unknown_raises():
    with pytest.raises(ValueError):
        _make_task("no-such-task")


# ---------------------------------------------------------------------------
# 2. 外推探针：stream_bytes 恒为常数（O(1) 主张）
# ---------------------------------------------------------------------------
def test_extrapolation_stream_bytes_constant_across_scales():
    """不同外推倍率下，流式状态字节数不变（SGC 的 O(1) 状态主张）。"""
    cfg = _tiny_cfg()
    factory = sgc_model_factory(cfg)
    res = extrapolation_probe(factory, task_name="selective-copy",
                              train_len=32, scales=(1, 2, 4),
                              steps=20, batch_size=32, n_samples=64,
                              device="cpu", verbose=False,
                              solvability_check=False)
    assert len(res) == 3
    b = [r.stream_bytes for r in res]
    # 三个倍率下字节数必须完全相同（状态与 L 无关）
    assert b[0] == b[1] == b[2] > 0, f"stream_bytes not constant: {b}"


def test_extrapolation_result_fields():
    cfg = _tiny_cfg()
    factory = sgc_model_factory(cfg)
    res = extrapolation_probe(factory, task_name="selective-copy",
                              train_len=32, scales=(1,),
                              steps=10, batch_size=32, n_samples=32,
                              device="cpu", verbose=False, solvability_check=False)
    r = res[0]
    assert isinstance(r, ExtrapolationResult)
    for f in ("train_len", "scale", "eval_len", "task", "chance", "acc",
              "stream_bytes", "wall_ms"):
        assert hasattr(r, f)
    assert r.scale == 1
    assert r.eval_len == 32
    assert r.chance == pytest.approx(1.0 / 16.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. 衰减消融：不同 decay_max 管线不崩
# ---------------------------------------------------------------------------
def test_decay_ablation_runs_pipeline():
    cfg = _tiny_cfg()
    pts = decay_ablation(cfg, decay_maxes=(32, 128), task_name="selective-copy",
                         train_len=32, steps=10, batch_size=32, n_samples=32,
                         device="cpu", verbose=False)
    assert len(pts) == 2
    assert isinstance(pts[0], DecayPoint)
    for f in ("decay_max", "train_ppl", "extrap_ppl", "acc", "extrap_acc"):
        assert hasattr(pts[0], f)
    assert pts[0].decay_max == 32.0
    assert pts[1].decay_max == 128.0


# ---------------------------------------------------------------------------
# 4. 结果可 JSON 化
# ---------------------------------------------------------------------------
def test_results_json_serializable():
    import json
    from dataclasses import asdict
    cfg = _tiny_cfg()
    factory = sgc_model_factory(cfg)
    res = extrapolation_probe(factory, task_name="selective-copy",
                              train_len=32, scales=(1,),
                              steps=5, batch_size=32, n_samples=16,
                              device="cpu", verbose=False, solvability_check=False)
    # 必须能被 json.dumps 序列化（供 CLI 输出）
    s = json.dumps([asdict(r) for r in res])
    assert "stream_bytes" in s


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
