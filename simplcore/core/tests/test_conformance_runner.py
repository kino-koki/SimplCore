# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""后端一致性跑批器的守护测试。

核心方法论：**每个检查都必须被证明能抓住它声称抓住的缺陷**。
除正向（参考适配器全过）外，为 9 项检查逐一构造破坏性变异，
断言对应检查翻红 —— 这让跑批器本身成为被验证的对象。
"""
import json

import pytest
import torch

from simplcore.core.conformance_runner import (
    ConformanceReport,
    TorchReferenceAdapter,
    main,
    run_conformance,
)

SMALL = dict(length=512, chunk=128, batch=2, dim=16, num_bases=8, local_kernel=4)


def _run(adapter, **kw):
    cfg = dict(SMALL)
    cfg.update(kw)
    return run_conformance(adapter, **cfg)


def _result(rep: ConformanceReport, name: str):
    for r in rep.results:
        if r.name == name:
            return r
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 正向：参考适配器（数值 oracle 的适配器化）必须全过
# ---------------------------------------------------------------------------
def test_reference_adapter_passes_all():
    rep = _run(TorchReferenceAdapter())
    assert rep.all_passed, rep.format()


def test_reference_report_structure():
    rep = _run(TorchReferenceAdapter())
    names = [r.name for r in rep.results]
    assert len(names) == len(set(names)) == 9
    for must in ("P0-backend-conformance", "P0-stream-bitexact", "P0-chunked-carry",
                 "P0-frozen-mask", "P0-bf16-accum", "P0-shard-2gcd",
                 "P0-no-attention", "C-stream-serialize", "C-rollback"):
        assert must in names
    d = rep.as_dict()
    assert d["all_passed"] is True and len(d["results"]) == 9


# ---------------------------------------------------------------------------
# 负向：逐一破坏，对应检查必须翻红
# ---------------------------------------------------------------------------
def test_missing_op_caught():
    class _Missing(TorchReferenceAdapter):
        name = "missing-op"
        weight_decay_exempt = None          # 类型上存在但不可调用

    r = _result(_run(_Missing()), "P0-backend-conformance")
    assert r.passed is False and "weight_decay_exempt" in r.detail


def test_broken_chunked_carry_caught():
    class _BrokenScan(TorchReferenceAdapter):
        name = "broken-scan"

        def chunked_iir_scan(self, u, y1, y2, a, b, c):
            ys, y1n, y2n = super().chunked_iir_scan(u, y1, y2, a, b, c)
            return ys, y1n, torch.zeros_like(y2n)   # 跨块丢弃 y2

    r = _result(_run(_BrokenScan()), "P0-chunked-carry")
    assert r.passed is False


def test_broken_frozen_mask_caught():
    class _BrokenOpt(TorchReferenceAdapter):
        name = "broken-opt"

        def adamw_frozen_mask(self, *a, **k):
            p1, m1, v1 = super().adamw_frozen_mask(*a, **k)
            frozen = a[4]
            p1 = p1.clone()
            p1[frozen] = p1[frozen] - 1e-3          # 偷偷更新冻结位
            return p1, m1, v1

    r = _result(_run(_BrokenOpt()), "P0-frozen-mask")
    assert r.passed is False


def test_bf16_accumulation_discriminated():
    """纯 bf16 顺序累加在 K=4096 下相对误差 ~9%，必须被 1e-2 容差抓住；
    fp32 累加的参考实现必须通过 —— 这就是该容差的区分力证明。"""

    class _Bf16Accum(TorchReferenceAdapter):
        name = "bf16-accum"

        def gemm(self, a, b):
            af, bf = a.float(), b.float()
            s = torch.zeros(af.shape[0], bf.shape[1], dtype=torch.bfloat16)
            for k in range(af.shape[1]):            # 逐步 bf16 累加（错误示范）
                s = s + (af[:, k:k + 1] * bf[k:k + 1, :]).to(torch.bfloat16)
            return s

    rep_bad = _run(_Bf16Accum())
    assert _result(rep_bad, "P0-bf16-accum").passed is False
    rep_ref = _run(TorchReferenceAdapter())
    assert _result(rep_ref, "P0-bf16-accum").passed is True


def test_broken_dual_form_caught():
    class _BrokenDual(TorchReferenceAdapter):
        name = "broken-dual"

        def spectral_causal_conv(self, u, kernel):
            return super().spectral_causal_conv(u, kernel) * 1.001

    r = _result(_run(_BrokenDual()), "P0-stream-bitexact")
    assert r.passed is False


def test_broken_allreduce_caught():
    class _BrokenShard(TorchReferenceAdapter):
        name = "broken-shard"

        def shard_allreduce(self, grads):
            out = grads[0].to(torch.bfloat16).float()      # bf16 归约（错）
            for g in grads[1:]:
                out = out + g.to(torch.bfloat16).float()
            return out

    r = _result(_run(_BrokenShard()), "P0-shard-2gcd")
    assert r.passed is False


def test_forbidden_op_name_caught():
    class _Forbidden(TorchReferenceAdapter):
        name = "forbidden"

        def ops_registry(self):
            reg = super().ops_registry()
            reg["fused_flash_attention"] = lambda *a, **k: None
            return reg

    r = _result(_run(_Forbidden()), "P0-no-attention")
    assert r.passed is False and "fused_flash_attention" in r.detail


def test_lossy_serialize_caught():
    class _Lossy(TorchReferenceAdapter):
        name = "lossy"

        def stream_state_serialize(self, state):
            blob = super().stream_state_serialize(state)
            return blob[:-2] + b"\x00\x00"          # 尾部 2 字节失真

    r = _result(_run(_Lossy()), "C-stream-serialize")
    assert r.passed is False


def test_rollback_catches_state_drift():
    """回滚检查必须能抓住『快照恢复后状态与快照点不一致』的漂移。"""

    class _Drift(TorchReferenceAdapter):
        name = "drift"

        def stream_state_deserialize(self, blob):
            st = super().stream_state_deserialize(blob)
            st = {k: v.clone() for k, v in st.items()}
            st["y1"] = st["y1"] * 1.0001            # 恢复时引入漂移
            return st

    r = _result(_run(_Drift()), "C-rollback")
    assert r.passed is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_self_test_exit_zero(capsys):
    assert main(["--length", "256", "--chunk", "128"]) == 0
    out = capsys.readouterr().out
    assert "通过" in out and "P0-backend-conformance" in out


def test_cli_json(capsys):
    assert main(["--length", "256", "--chunk", "128", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["all_passed"] is True and d["adapter"] == "torch-reference"


def test_cli_rejects_non_adapter():
    from simplcore.core.conformance_runner import _load_adapter
    with pytest.raises((TypeError, AttributeError, ModuleNotFoundError)):
        _load_adapter("simplcore.core.config:get_preset")   # 不是适配器类
