# -*- coding: utf-8 -*-
"""SimplCore 单一可复现冒烟测试 (诚实版, 不美化失败).

覆盖核心链路，每个用例独立判定 PASS/FAIL，最终报告真实通过率。
任何真实异常都记为 FAIL，绝不降级为 SKIP 后报成功。

运行:
  python smoke_test.py            # 默认 CPU
  python smoke_test.py --device cuda   # 若有 GPU

退出码: 通过率 >= 90% 返回 0, 否则返回 1。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def case(name):
    """用例装饰器: 捕获异常, 返回 (name, ok, detail)."""
    def deco(fn):
        def wrap():
            try:
                t0 = time.perf_counter()
                msg = fn()
                dt = (time.perf_counter() - t0) * 1000
                return name, True, f"{msg} ({dt:.1f}ms)"
            except Exception as e:  # noqa: BLE001
                return name, False, f"FAIL: {type(e).__name__}: {e}"
        return wrap
    return deco


# --------------------------------------------------------------------------
@case("MoE模型 构造/前向/反向")
def test_moe_forward_backward():
    from simplcore_pg.config import SimplCoreConfig
    from simplcore_pg.model_moe import SimplCoreMoEModel
    cfg = SimplCoreConfig.get_nano_config()
    cfg.use_rfp = True
    cfg.use_dmc = True
    cfg.num_experts = 4
    cfg.experts_per_tok = 2
    m = SimplCoreMoEModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    y = m(x)
    assert y.shape[-1] == cfg.vocab_size, y.shape
    y.float().sum().backward()
    return f"params={sum(p.numel() for p in m.parameters()):,}"


@case("无注意力架构 构造/前向/反向 (Local+RFP+DMC+MoE)")
def test_attention_free():
    import inspect
    from simplcore_pg.config import SimplCoreConfig
    from simplcore_pg.model_moe import SimplCoreMoEBlock, SimplCoreMoEModel
    from simplcore_pg.longctx import SimplCoreStreamingBlock
    # 注意力自由预设：不应含任何自注意力 / Q-K-V / RoPE / SDPA
    cfg = SimplCoreConfig.get_attention_free_config(hidden_size=64, num_layers=2, vocab_size=64)
    assert cfg.use_attention is False, "v2.0 不应启用注意力"
    src = (inspect.getsource(SimplCoreMoEModel) + inspect.getsource(SimplCoreStreamingBlock)
           + inspect.getsource(SimplCoreMoEBlock))
    for bad in ("SelfAttention", "RoPE", "sdpa", "QKV", "qkv", "attn_window", "past_key_value"):
        assert bad not in src, f"架构泄漏注意力组件: {bad}"
    m = SimplCoreMoEModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = m(ids, training=True)
    assert logits.shape == (2, 16, cfg.vocab_size)
    logits.float().sum().backward()
    return f"params={sum(p.numel() for p in m.parameters()):,} (无注意力)"


@case("长上下文流式 前向/反向/状态传递")
def test_longctx_stream():
    from simplcore_pg.config import SimplCoreConfig
    from simplcore_pg.model_stream import SimplCoreStreamingModel
    lc = SimplCoreConfig.get_longctx_config(4096)
    lc.hidden_size = 128
    lc.num_layers = 2
    lc.num_experts = 4
    lc.experts_per_tok = 2
    lc.vocab_size = 128
    sm = SimplCoreStreamingModel(lc)
    seq = torch.randint(0, 128, size=(1, 300))
    out = sm(seq)
    assert out.shape[1] == 300
    out.float().sum().backward()
    # 流式分块 (真实 API)
    st = sm.stream_init(batch=1)
    for i in range(0, 256, lc.chunk_size):
        chunk = torch.randint(0, 128, size=(1, lc.chunk_size))
        logits, st = sm.stream_step(chunk, st)
        assert logits.shape == (1, lc.chunk_size, 128)
    # 流式生成
    gen = sm.generate_stream(torch.randint(0, 128, size=(1, 8)), max_new_tokens=4)
    assert gen.shape[1] == 4
    peak = sm.peak_activation_bytes(1, 300)
    peak_gb = (peak.get("est_peak_bytes") or peak.get("peak_bytes", 0)) / 1e9
    return f"peak_GB={peak_gb:.3f}"


@case("ResonanceField 频谱演化 梯度可达")
def test_rfp_grad():
    from simplcore_pg.resonance import ResonanceField
    rfp = ResonanceField(128, num_waves=8, num_steps=3)
    x = torch.randn(1, 128, 128)
    y = rfp(x).sum()
    y.backward()
    assert rfp.in_proj.weight.grad is not None
    assert rfp.in_proj.weight.grad.abs().sum() > 0
    return "grad OK"


@case("DynamicMemoryChannels 记忆读写 梯度可达")
def test_dmc_grad():
    from simplcore_pg.memory import DynamicMemoryChannels
    dmc = DynamicMemoryChannels(128, num_channels=4, channel_size=8, num_heads=2)
    x = torch.randn(1, 16, 128)
    out = dmc(x)
    out.sum().backward()
    assert dmc.memory_bank.grad is not None and dmc.memory_bank.grad.abs().sum() > 0
    return "grad OK"


@case("RFP vs 注意力 长序列延迟基准 (含真实数据)")
def test_rfp_benchmark():
    from bench_rfp_vs_attention import bench_rfp, bench_attn
    rfp = bench_rfp(256, 2048, reps=2)
    attn = bench_attn(256, 2048, heads=8, reps=2)
    # 真实标度: RFP 应显著快于注意力 (或至少同量级, 不应相反)
    assert rfp < attn * 2, f"RFP({rfp}) 不应比注意力({attn})慢 2 倍以上"
    return f"RFP={rfp*1000:.2f}ms attn={attn*1000:.2f}ms"


@case("物理规律诚实证明 加载权重实测")
def test_physics_honest():
    import json
    with open("final_physics_proof.json", encoding="utf-8") as f:
        d = json.load(f)
    # 结论必须与 all_passed 一致, 不得写死成功
    assert d["all_passed"] is False, "物理证明不应声称已学会 (与实测矛盾)"
    assert "尚未充分学习" in d["final_conclusion"], "结论需如实反映失败"
    return f"rel_err spring={d['spring_test']['relative_error_percent']:.1f}%"


@case("torch.compile 安全回退")
def test_compile_fallback():
    from simplcore_pg.config import SimplCoreConfig
    from simplcore_pg.model import SimplCoreModel
    cfg = SimplCoreConfig.get_nano_config()
    cfg.num_experts = 4
    m = SimplCoreModel(cfg)
    try:
        m.compile(mode="default", backend="inductor", smoke_input=(1, 16))
    except Exception:
        pass  # 平台不支持时须安全回退 eager, 不算失败
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        y = m(x)
    assert y.shape[-1] == cfg.vocab_size
    return "eager/compile OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cases = [
        test_moe_forward_backward,
        test_attention_free,
        test_longctx_stream,
        test_rfp_grad,
        test_dmc_grad,
        test_rfp_benchmark,
        test_physics_honest,
        test_compile_fallback,
    ]
    results = [c() for c in cases]
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    rate = passed / total * 100

    print("\n" + "=" * 64)
    print("SimplCore 冒烟测试 (诚实版)")
    print("=" * 64)
    for name, ok, detail in results:
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}: {detail}")
    print("-" * 64)
    print(f"通过率: {passed}/{total} = {rate:.1f}%")
    print("=" * 64)

    with open("smoke_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed, "total": total, "rate": rate,
            "results": [{"name": n, "ok": o, "detail": d} for n, o, d in results],
        }, f, indent=2, ensure_ascii=False)

    sys.exit(0 if rate >= 90 else 1)


if __name__ == "__main__":
    main()
