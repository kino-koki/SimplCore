# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""生物计算主义（BioEnhancer）单元测试：验证论文三支柱真实可训练、且默认零回归。

覆盖：
1. 支柱 1 混合计算：连续场 O(1) 常驻态、与序列长度无关；前向可反传；
2. 支柱 2 尺度不可分：macro(课程进度)→meso(场 dt) 与 meso(场能量)→micro(门增益)
   的耦合系数真实参与梯度；curriculum_frac 改变时场 dt 随之变化；
3. 支柱 3 代谢约束：代谢账本记账、超额能耗产生效率损失、预算紧张收缩增益；
4. 集成：BioEnhancer 是 nn.Module，build_enhanced_model("bio") 接入训练闭环，
   参数可进优化器；与 unity 组合时 meso/macro 调节路由温度、不重复登记基座参数；
5. 零回归：bio 关闭（不进 enhance）时训练行为与基座逐位一致。
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
from simplcore.core.bio_enhance import (
    BioEnhancer, ContinuousField, MetabolicMeter, MultiscaleCoupling,
)


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """守住全局 RNG 状态：套件里有确定性测试（BPE / 打包 / RNG 恢复），
    本文件不得改写它们的起点种子。"""
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


# ---------------------------------------------------------------------------
# 支柱 1：混合计算（连续场 × 离散脉冲）
# ---------------------------------------------------------------------------
def test_continuous_field_runs_and_backprops():
    f = ContinuousField(dim=32, dt=0.5, scale=1.0)
    h = torch.randn(4, 16, 32)
    h2, phi = f.forward_sequence(h)
    assert h2.shape == h.shape
    # 场是 O(1) 常驻态：与序列长度无关
    assert f.phi.shape == (4, 32)
    # 离散脉冲注入确实改变了调制输出（否则门/脉冲全是恒等）
    assert not torch.allclose(h2, h)
    loss = h2.pow(2).mean()
    loss.backward()
    assert f.W.weight.grad is not None and f.spike.weight.grad is not None
    assert f.gain.weight.grad is not None


def test_continuous_field_state_is_o1_and_carries():
    f = ContinuousField(dim=32)
    h1 = torch.randn(2, 8, 32)
    _, phi1 = f.forward_sequence(h1)
    # 不同序列长度：常驻态形状不变（O(1)）
    h2 = torch.randn(2, 64, 32)
    _, phi2 = f.forward_sequence(h2)
    assert phi1.shape == phi2.shape == (2, 32)
    # 跨 chunk 携带：第二次调用的初始场是第一次的末态（detach 后）
    assert torch.allclose(phi2, f.phi)


def test_coupling_macro_to_meso_changes_dt():
    c = MultiscaleCoupling(dim=32, base_dt=0.5)
    dt_early = c.dt(0.0).item()
    # 默认偏置=base_dt，故早期≈0.5
    assert dt_early == 0.5  # 偏置初始化保证
    # 梯度能流向 macro_to_dt（macro → meso 耦合系数可训）
    dt_late = c.dt(1.0)
    dt_late.sum().backward()
    assert c.macro_to_dt.weight.grad is not None


def test_coupling_meso_to_micro_changes_gain():
    c = MultiscaleCoupling(dim=32)
    e = torch.tensor(2.0)
    g = c.gain_scale(e)
    assert g.item() == 1.0  # 偏置初始化保证
    g.sum().backward()
    assert c.meso_to_gain.weight.grad is not None


# ---------------------------------------------------------------------------
# 支柱 3：代谢约束
# ---------------------------------------------------------------------------
def test_metabolic_meter_records_and_penalizes():
    m = MetabolicMeter(budget=2.0, energy_weight=0.1)
    # 低能耗：无浪费、无效率损失（base_cost=1.0 + 小场，远低于预算 2.0）
    e_low = m.energy(32, torch.tensor(0.1))
    assert float(m.efficiency_loss(e_low)) == 0.0
    m.record(float(e_low), 0.1)
    # 高能耗（场能量大）：超过预算 → 效率损失 > 0、waste 记录
    e_high = m.energy(32, torch.tensor(50.0))
    loss_high = m.efficiency_loss(e_high)
    assert float(loss_high) > 0.0
    m.record(float(e_high), 50.0)
    assert m.waste > 0.0
    assert m.last_energy > m.budget


# ---------------------------------------------------------------------------
# 支柱 2+3 集成：BioEnhancer 端到端
# ---------------------------------------------------------------------------
def test_bio_enhancer_forward_and_trainable():
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config)
    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))
    logits, loss = bio(x, y)
    assert logits.shape[-1] == 64
    loss.backward()
    # 场与耦合器权重都应拿到梯度（三支柱可训）
    assert bio.field.W.weight.grad is not None
    assert bio.coupling.macro_to_dt.weight.grad is not None
    assert bio.coupling.meso_to_gain.weight.grad is not None
    # 代谢账本已记账
    assert bio.meter.last_energy > 0.0


def test_bio_curriculum_frac_modulates_field_dt():
    model = _toy_model()
    bio = BioEnhancer(model, cfg=model.config, field_dt=0.5)
    bio.curriculum_frac = 0.0
    dt0 = bio.coupling.dt(bio.curriculum_frac).item()
    bio.curriculum_frac = 1.0
    dt1 = bio.coupling.dt(bio.curriculum_frac).item()
    assert dt0 == 0.5
    # 把 macro→meso 耦合权重推离原点后，frac 改变应改变 dt（验证耦合真实生效）
    with torch.no_grad():
        bio.coupling.macro_to_dt.weight.fill_(0.8)
    bio.curriculum_frac = 0.0
    dt0b = bio.coupling.dt(bio.curriculum_frac).item()
    bio.curriculum_frac = 1.0
    dt1b = bio.coupling.dt(bio.curriculum_frac).item()
    assert dt0b != dt1b


def test_bio_in_build_enhanced_model():
    model = _toy_model()
    em = build_enhanced_model(model, model.config, ["bio"])
    assert "bio" in em.extras
    assert isinstance(em.model, BioEnhancer)
    # 参数过滤：基座（可训）+ 场 + 耦合器
    learned = em.model.num_params_learned()
    assert learned > 0
    x = torch.randint(0, 64, (2, 10))
    y = torch.randint(0, 64, (2, 10))
    logits, loss = em.train_forward(x, y)
    assert logits.shape[-1] == 64
    loss.backward()


def test_bio_with_unity_composes_without_double_base():
    model = _toy_model()
    # unity + bio 组合：bio 包裹 UnityModel，不重复登记基座参数
    em = build_enhanced_model(model, model.config, ["unity", "bio"],
                              unity_experts=3, unity_topk=2)
    assert "bio" in em.extras and "unity" in em.extras
    bio = em.extras["bio"]
    # 路由温度属性存在（meso/macro → micro 耦合入口）
    assert hasattr(bio.unity.router, "temperature")
    x = torch.randint(0, 64, (2, 10))
    y = torch.randint(0, 64, (2, 10))
    logits, loss = em.train_forward(x, y)
    assert logits.shape[-1] == 64
    loss.backward()
    # 基座在 UnityModel 内冻结 → 不应出现在可学参数里（避免重复）
    learned_names = [n for n, p in bio.named_parameters() if p.requires_grad]
    base_param_names = [n for n, _ in model.named_parameters()]
    assert not any(n in learned_names for n in base_param_names), \
        "基座参数不应进入 bio 可学集合（与 unity 内冻结重复）"


def test_bio_off_is_identity_to_base():
    # 不进 enhance：返回原基座，行为逐位一致
    model = _toy_model()
    em = build_enhanced_model(model, model.config, [])
    assert em.model is model
    x = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        logits_base, loss_base = model(x, x)
        logits_em, loss_em = em.train_forward(x, x)
    assert torch.allclose(logits_base, logits_em)
    assert torch.allclose(loss_base, loss_em)


def test_enhanced_model_params_share_device_with_base():
    """不变量：增强新增的模块（bio 的连续场/耦合器、unity 的路由/专家）必须与
    基座同设备。

    历史 bug：基座先 ``.to(device)``，增强模块在之后才构造 → 默认留在 CPU →
    GPU 上首次前向 device mismatch（CPU 训练掩盖了它）。``train.py`` 现在在
    ``model = em.model`` 之后补 ``.to(device)``；此测试守护该约束。
    """
    model = _toy_model()
    base_dev = model.tok_emb.weight.device
    for names in (["bio"], ["unity", "bio"]):
        m = _toy_model()
        em = build_enhanced_model(m, m.config, names, unity_experts=3, unity_topk=2)
        devs = {p.device for p in em.model.parameters()}
        assert devs == {base_dev}, \
            f"{names} 的增强模块与基座设备不一致：{devs} vs {base_dev}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL BIO TESTS PASSED")
