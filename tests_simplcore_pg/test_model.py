# -*- coding: utf-8 -*-
"""SimplCoreModel 测试：前向 / 参数量 / 生成 / 编译 / FX 图模式。"""
import torch

import pytest
import torch.fx

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model import SimplCoreModel
from simplcore_pg.exceptions import ModelError


def _nano():
    return SimplCoreConfig.get_nano_config()


def test_model_forward_logits_shape():
    model = SimplCoreModel(_nano())
    ids = torch.randint(0, 1000, (2, 12))
    logits = model(ids)
    assert logits.shape == (2, 12, 1000)


def test_model_param_count_matches_estimate():
    cfg = _nano()
    model = SimplCoreModel(cfg)
    est = cfg.estimate_parameters()["total"]
    # 估算不含 bias 的细微差异，允许 <5% 容差
    assert abs(model.num_parameters() - est) <= 0.05 * est + 1


def test_model_input_id_bounds():
    model = SimplCoreModel(_nano())
    with pytest.raises(ModelError):
        model(torch.tensor([[1000]]))  # 越界 vocab


def test_model_seqlen_guard():
    model = SimplCoreModel(_nano())
    with pytest.raises(ModelError):
        model(torch.randint(0, 1000, (1, 100)))  # > max_seq_len(64)


def test_model_mask_accepted():
    model = SimplCoreModel(_nano())
    ids = torch.randint(0, 1000, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.bool)
    logits = model(ids, attention_mask=mask)
    assert logits.shape == (1, 10, 1000)


def test_generate_greedy_shapes_and_termination():
    torch.manual_seed(0)
    model = SimplCoreModel(_nano())
    prefix = torch.randint(0, 1000, (1, 4))
    out = model.generate(prefix, max_new_tokens=5, top_k=0, temperature=1.0)
    assert out.shape[1] == 4 + 5
    # 全 pad 提前终止
    pad_cfg = SimplCoreConfig.get_nano_config()
    pad_cfg.pad_token_id = 0
    m2 = SimplCoreModel(pad_cfg)
    out2 = m2.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=10, temperature=1.0)
    assert out2.shape[1] <= 2 + 10


def test_generate_temperature_guard():
    model = SimplCoreModel(_nano())
    with pytest.raises(ModelError):
        model.generate(torch.randint(0, 1000, (1, 3)), temperature=0.0)


def test_compile_fallback_safe():
    model = SimplCoreModel(_nano())
    # 不论后端是否可用，compile 不应抛异常
    model.compile(mode="default", backend="inductor", smoke_input=(1, 4))
    ids = torch.randint(0, 1000, (1, 4))
    out = model(ids)
    assert out.shape == (1, 4, 1000)


def test_fx_trace_and_optimize():
    model = SimplCoreModel(_nano())
    ids = torch.randint(0, 1000, (1, 5))
    gm = model.trace_fx(example_input_ids=ids)
    assert isinstance(gm, torch.fx.GraphModule)
    counts = model.summarize_graph(gm)
    assert len(counts) > 0
    # 优化 pass 不破坏可运行性
    from simplcore_pg.fx_passes import optimize_for_inference
    gm2 = optimize_for_inference(gm)
    with torch.no_grad():
        out = gm2(ids)
    assert out.shape == (1, 5, 1000)


def test_configure_optimizers_groups():
    model = SimplCoreModel(_nano())
    opt = model.configure_optimizers(lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) == 2
