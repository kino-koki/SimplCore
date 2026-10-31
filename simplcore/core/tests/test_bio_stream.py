# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""生物计算主义流式 / 生成测试：验证混合计算在生成期（stream_step）也生效，
且流式路径与整段前向逐位一致（理论守卫），并覆盖 checkpoint/restore 与 unity 组合。
"""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.training_integration import build_enhanced_model
from simplcore.core.bio_computation import BioEnhancer


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """本文件会调用 ``torch.manual_seed``（会改写全局 RNG 状态）。

    套件里有一批确定性测试（BPE / 语料打包 / RNG 恢复）依赖全局 RNG 的既定状态，
    若本文件把种子改掉且不还原，就会把它们带崩。进出都还原 ⇒ 本文件零副作用。
    """
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)


def _tiny_cfg():
    return SimplCoreConfig(dim=32, num_layers=2, num_bases=8, vocab_size=64,
                           dropout=0.0, tie_embeddings=True)


def _toy_model():
    return SimplCoreModel(_tiny_cfg())


def test_stream_matches_forward_base():
    # 流式（逐 token）与整段前向的 logits 逐位一致：验证生成期混合计算与训练一致。
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config, field_dt=0.5, field_scale=1.0)
    bio.curriculum_frac = 0.0
    bio.eval()
    torch.manual_seed(0)
    x = torch.randint(0, 64, (2, 12))

    # 整段前向
    bio.field.reset(2, x.device, torch.float32)
    with torch.no_grad():
        full = bio.forward(x)                              # (B,L,V)

    # 逐 token 流式
    bio.field.reset(2, x.device, torch.float32)
    state = bio.init_stream(2, device=x.device, dtype=torch.float32)
    collected = []
    for t in range(x.shape[1]):
        logits, state = bio.stream_step(x[:, t], state)
        collected.append(logits)
    streamed = torch.stack(collected, dim=1)               # (B,L,V)

    assert torch.allclose(full, streamed, atol=1e-5), \
        f"流式与整段前向不一致，最大差 { (full-streamed).abs().max().item() }"


def test_generate_runs_and_uses_field():
    # 生成期连续场持续演化：场能量在生成中增长（混合计算真实生效），产出合法 token。
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config, field_dt=0.5, field_scale=1.0)
    bio.curriculum_frac = 0.0
    bio.eval()
    torch.manual_seed(1)
    idx = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        e0 = bio.field.field_energy().item()
        out = bio.generate(idx, max_new_tokens=8, temperature=0.8, top_k=20)
        e1 = bio.field.field_energy().item()
    assert out.shape == (1, 5 + 8)
    assert int(out.max()) < 64 and int(out.min()) >= 0
    # 场在生成中确实演化（能量变化）——混合计算不是恒等摆设
    assert e1 != e0


def test_generate_via_build_enhanced_model():
    # build_enhanced_model("bio") 产出的 em.model 也支持 generate / stream_step。
    model = _toy_model()
    em = build_enhanced_model(model, model.config, ["bio"])
    assert isinstance(em.model, BioEnhancer)
    em.model.eval()
    torch.manual_seed(2)
    idx = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        out = em.model.generate(idx, max_new_tokens=6, temperature=1.0)
    assert out.shape[1] == 10


def test_init_stream_resets_field():
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config)
    bio.eval()
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        # 先让场演化
        _, state = bio.stream_step(x[:, 0], bio.init_stream(1))
        assert bio.field.field_energy().item() > 0.0
        # 重新 init_stream 应复位到 0
        bio.init_stream(1)
        assert bio.field.field_energy().item() == 0.0


def test_checkpoint_restore_stream_roundtrip():
    # checkpoint/restore 后继续流式，logits 与不间断流式逐位一致。
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config, field_dt=0.5)
    bio.curriculum_frac = 0.0
    bio.eval()
    torch.manual_seed(3)
    x = torch.randint(0, 64, (1, 10))

    # 不间断：前 4 步 + 后 6 步
    bio.field.reset(1, x.device, torch.float32)
    s = bio.init_stream(1, device=x.device, dtype=torch.float32)
    cont = []
    for t in range(10):
        lg, s = bio.stream_step(x[:, t], s)
        cont.append(lg)
    cont = torch.stack(cont, dim=1)

    # 中断：前 4 步 checkpoint → restore → 后 6 步
    bio.field.reset(1, x.device, torch.float32)
    s2 = bio.init_stream(1, device=x.device, dtype=torch.float32)
    for t in range(4):
        _, s2 = bio.stream_step(x[:, t], s2)
    ckpt = bio.checkpoint_stream(s2)
    s3 = bio.restore_stream(ckpt)
    rest = []
    for t in range(4, 10):
        lg, s3 = bio.stream_step(x[:, t], s3)
        rest.append(lg)
    rest = torch.stack(rest, dim=1)

    assert torch.allclose(cont[:, 4:, :], rest, atol=1e-5), \
        f"checkpoint/restore 后流式不一致，最大差 { (cont[:,4:,:]-rest).abs().max().item() }"


def test_stream_with_unity_runs():
    # unity + bio 组合在生成期也跑（专家调制隐状态 + 连续场）。
    model = _toy_model()
    em = build_enhanced_model(model, model.config, ["unity", "bio"],
                              unity_experts=3, unity_topk=2)
    bio = em.extras["bio"]
    bio.curriculum_frac = 0.0
    bio.eval()
    torch.manual_seed(4)
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        state = bio.init_stream(1, device=x.device, dtype=torch.float32)
        coll = []
        for t in range(x.shape[1]):
            lg, state = bio.stream_step(x[:, t], state)
            coll.append(lg)
        streamed = torch.stack(coll, dim=1)
        # 与整段前向对照
        bio.field.reset(1, x.device, torch.float32)
        full = bio.forward(x)
    assert torch.allclose(full, streamed, atol=1e-5), \
        f"unity+bio 流式与整段前向不一致，最大差 { (full-streamed).abs().max().item() }"
    # 生成也能跑
    with torch.no_grad():
        out = bio.generate(x, max_new_tokens=4, temperature=1.0)
    assert out.shape[1] == 12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL BIO STREAM TESTS PASSED")
