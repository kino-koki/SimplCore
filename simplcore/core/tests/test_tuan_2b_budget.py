# -*- coding: utf-8 -*-
"""`tuan-2b` 的 30 天 / 单 MI250 预算口径测试。

这些断言把"2B × 4M 能否在 1 个月跑完"钉在算术上：

* 参数口径是 2B（不是"标称 2B、实际 2.2B"）；
* 30 天 token 上限 ≈ 12.4B（6.1 tok/param），并明确不是 Chinchilla-optimal；
* 4M 上下文只增加 0.16% 的每 token FLOPs（长卷积的 log L 优势）；
* 单 GCD 显存（chunk=4096, micro=2）必须 < 64 GiB。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from simplcore.core.campaign import (
    BYTES_BF16, effective_flops, optimizer_state_bytes,
)
from simplcore.core.config import _CTX_2B_4M_30D, get_preset
from simplcore.core.scaling import (
    embedding_params, non_embedding_params, train_flops_per_token,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cfg():
    c = get_preset("tuan-2b")
    c.validate()
    return c


def test_param_count_is_two_billion(cfg) -> None:
    total = embedding_params(cfg) + non_embedding_params(cfg)
    assert 1.95e9 <= total <= 2.10e9, total
    # 嵌入占比必须低：大词表不该吃掉 2B 预算。
    assert embedding_params(cfg) / total < 0.10
    assert cfg.num_heads and cfg.dim % cfg.num_heads == 0


def test_thirty_day_token_budget_is_about_12_4b(cfg) -> None:
    flops = train_flops_per_token(cfg, 4096)
    tokens = effective_flops(30, mfu=0.25) / flops
    assert 12.0e9 <= tokens <= 13.5e9, tokens
    # 口径自检：6.1 tok/param 左右 —— 明显欠训，必须如实标注。
    params = embedding_params(cfg) + non_embedding_params(cfg)
    assert 5.5 <= tokens / params <= 7.0


def test_long_context_is_cheap_in_flops(cfg) -> None:
    """4M vs 4K：FFT 的 log L 项只增加 <1%（不是 attention 的线性增长）。"""
    f4k = train_flops_per_token(cfg, 4096)
    f4m = train_flops_per_token(cfg, 4_000_000)
    assert f4m / f4k - 1 < 0.01


def test_curriculum_spends_ten_percent_at_4m() -> None:
    stages = _CTX_2B_4M_30D["curriculum"]
    assert stages[-1]["seq_len"] == 4_000_000
    prev = stages[-2]["until_frac"]
    assert 0.08 <= (1.0 - prev) <= 0.12


def test_single_gcd_memory_fits_64gib(cfg) -> None:
    """每 GCD 是 64 GiB；DDP 复制完整模型，必须能放下。"""
    params = non_embedding_params(cfg)
    total = params + embedding_params(cfg)
    chunk, micro, act_tensors = 4096, 2, 8
    opt = optimizer_state_bytes(params)
    weights = total * BYTES_BF16
    grads = params * BYTES_BF16
    act = cfg.num_layers * chunk * cfg.dim * act_tensors * BYTES_BF16 * micro
    hbm = (opt + weights + grads + act) / (1024 ** 3)
    assert hbm < 64.0, hbm
    # 至少留 10% 给碎片/FFT workspace，否则 30 天长跑一定 OOM。
    assert hbm < 64.0 * 0.90, hbm


def test_launcher_and_plan_exist_and_are_consistent() -> None:
    script = ROOT / "scripts" / "train_tuan_2b_mi250.sh"
    plan = ROOT / "docs" / "TRAINING_2B_MI250_30D.md"
    assert script.exists() and plan.exists()
    text = script.read_text(encoding="utf-8")
    assert "--train-chunk" in text
    assert "--preset" in text and "tuan-2b" in text
    assert "4000000" in text                     # 4M 阶段
    assert "num-gcd" in text and "2" in text
    doc = plan.read_text(encoding="utf-8")
    assert "12.4B" in doc
    assert "Chinchilla" in doc
    assert "未实测" in doc or "不得把本文件当作实测结果" in doc