# -*- coding: utf-8 -*-
"""`tuan-2b` 的 24h / 单张 RTX 4090D-24GB 预算口径测试。

这些断言把"2B 全参 × 24h 能否在一张 24GB 卡跑完"钉在算术上：

* 参数口径是 2B（不是"标称 2B、实际 2.2B"）；
* 24h token 上限 ≈ 0.2B（~0.1 tok/param），并明确不是 Chinchilla-optimal；
* 4M 上下文只增加 <1% 的每 token FLOPs（长卷积的 log L 优势）；
* 24GB 显存（Adafactor 路径, chunk=4096）必须 < 23 GiB 实测线；
* AdamW 路径（16 B/参数）在 24GB 上不可行 —— 这是硬闸门。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from simplcore.core.campaign import (
    ADAFACTOR_BYTES_PER_PARAM, BYTES_BF16, effective_flops,
    optimizer_state_bytes,
)
from simplcore.core.config import _CTX_2B_4M, get_preset
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


def test_24h_token_budget_is_honest(cfg) -> None:
    """24h @ MFU 0.30 / util 0.65 / 157 TFLOPS：~0.2B tokens，欠训档如实标注。"""
    flops = train_flops_per_token(cfg, 4096)
    tokens = effective_flops(1, mfu=0.30) / flops
    assert 0.10e9 <= tokens <= 0.40e9, tokens
    # 口径自检：tokens/param 远低于 20×Chinchilla —— 欠训档，必须如实标注。
    params = embedding_params(cfg) + non_embedding_params(cfg)
    assert tokens / params < 0.25


def test_long_context_is_cheap_in_flops(cfg) -> None:
    """4M vs 4K：FFT 的 log L 项只增加 <1%（不是 attention 的线性增长）。"""
    f4k = train_flops_per_token(cfg, 4096)
    f4m = train_flops_per_token(cfg, 4_000_000)
    assert f4m / f4k - 1 < 0.01


def test_curriculum_spends_ten_percent_at_4m() -> None:
    stages = _CTX_2B_4M["curriculum"]
    assert stages[-1]["seq_len"] == 4_000_000
    prev = stages[-2]["until_frac"]
    assert 0.08 <= (1.0 - prev) <= 0.12


def test_4090d_24gib_memory_fits_with_adafactor(cfg) -> None:
    """Adafactor 路径（fp32 参数+梯度+因子状态 ≈8 B/参数）+ chunk 激活 < 23 GiB。

    注意 8 B/参数**已含** fp32 参数与梯度——不要在外面重复叠加 weights/grads。
    激活按单 micro-batch 计（梯度累积时每个 micro-batch 前向后立即 backward，
    激活不跨 micro-batch 叠加）。
    """
    total = non_embedding_params(cfg) + embedding_params(cfg)
    chunk, act_tensors = 4096, 8
    master_and_opt = total * ADAFACTOR_BYTES_PER_PARAM    # 8 B/参数已含参数+梯度
    for micro, cap in ((1, 23.0), (2, 23.0)):
        act = cfg.num_layers * chunk * cfg.dim * act_tensors * BYTES_BF16 * micro
        hbm = (master_and_opt + act) / (1024 ** 3)
        assert hbm < cap, (micro, hbm)
        # 至少留 4% 给碎片/FFT workspace/CUDA 上下文，否则 24h 长跑一定 OOM。
        assert hbm < 24.0 * 0.96, (micro, hbm)


def test_adamw_path_cannot_fit_24gib(cfg) -> None:
    """AdamW 16 B/参数 × 2B = 32 GiB > 24 GB —— 24GB 上硬性不可行。"""
    params = non_embedding_params(cfg)
    assert optimizer_state_bytes(params) > 24 * (1024 ** 3)


def test_launcher_and_plan_exist_and_are_consistent() -> None:
    script = ROOT / "scripts" / "train_tuan_2b_4090d.sh"
    plan = ROOT / "docs" / "RTX4090D_TRAINING.md"
    assert script.exists() and plan.exists()
    text = script.read_text(encoding="utf-8")
    assert "--train-chunk" in text
    assert "--preset" in text and "tuan-2b" in text
    assert "--time-budget-min" in text and "1440" in text   # 24h 硬停
    assert "--optimizer adafactor" in text                  # 24GB 唯一可行组合
    doc = plan.read_text(encoding="utf-8")
    assert "0.2B" in doc
    assert "Chinchilla" in doc
    assert "未实测" in doc or "欠训" in doc
