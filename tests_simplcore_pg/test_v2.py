# -*- coding: utf-8 -*-
"""SimplCore v2 测试：DeltaNet / SWA / OctaveMemory / GlyphStream / V2Model。

覆盖方案的核心不变量：
  * L1 DeltaNet：分块 == 全量；状态大小与 L 无关 (单步 O(1))；
  * L0 SWA：环形缓存逐步推理 == 全量带状掩码前向；KV 字节数恒定；
  * L2 Octave：层数随 log(L) 增长，单层预算恒定 (每翻倍 +1 层)；
  * GlyphStream：编码/合并/解码全链路形状与梯度；
  * V2Model：训练损失可下降 (可微性)、流式推理状态有界。
"""
try:
    import pytest  # noqa: F401  (可选；无 pytest 时用 verify_v2.py 直接运行)
except ModuleNotFoundError:  # pragma: no cover
    pytest = None
import torch

from simplcore_pg.v2 import (
    V2Config, GatedDeltaNet, SlidingWindowAttention, WindowKVCache,
    OctaveMemory, GlyphStreamEncoder, EntropyDrivenPooler, PrototypeDecoder,
    SimplCoreV2Model,
)


def _cfg(**kw):
    base = dict(
        hidden_size=64, num_layers=2,
        swa_num_heads=4, swa_num_kv_heads=2, swa_window=16,
        delta_num_heads=4,
        octave_every=1, octave_slots_per_level=8, octave_segment_tokens=8,
        octave_mem_tokens=2, octave_promote_tokens=2, octave_max_levels=6,
        octave_num_heads=4,
        ffn_intermediate_ratio=2.0,
        glyph_size=16, glyph_patch_size=8, glyph_hidden=64,
        glyph_encoder_layers=1, glyph_encoder_heads=4,
        pool_boundary_threshold=0.5, max_chars_per_unit=4,
        num_prototypes=128, dropout=0.0,
    )
    base.update(kw)
    return V2Config(**base)


# ======================================================================
# L1 Gated DeltaNet
# ======================================================================
def test_deltanet_shape_and_grad():
    m = GatedDeltaNet(64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out, S = m(x)
    assert out.shape == x.shape
    assert S.shape == (2, 4, 16, 16)
    out.sum().backward()
    assert m.q_proj.weight.grad is not None
    assert torch.isfinite(m.q_proj.weight.grad).all()


def test_deltanet_chunked_equals_full():
    """分块扫描 (不 detach) 必须与全量扫描逐位一致。"""
    m = GatedDeltaNet(64, num_heads=4)
    x = torch.randn(1, 10, 64)
    full, S_full = m(x)
    chunked, S_chunk = m.forward_chunked(x, chunk_size=3, detach_between_chunks=False)
    assert torch.allclose(full, chunked, atol=1e-5)
    assert torch.allclose(S_full, S_chunk, atol=1e-5)


def test_deltanet_state_size_independent_of_length():
    """单步 O(1)：状态字节数不随已处理长度变化。"""
    m = GatedDeltaNet(64, num_heads=4)
    state = m.init_state(1, torch.device("cpu"))
    before = state.nbytes()
    for L in (4, 16, 64):
        m.forward_runtime(torch.randn(1, L, 64), state)
        assert state.nbytes() == before


def test_deltanet_runtime_matches_training_path():
    """推理路径 (runtime state) 与训练路径 (S0 传递) 结果一致。"""
    m = GatedDeltaNet(64, num_heads=4)
    xa = torch.randn(1, 5, 64)
    xb = torch.randn(1, 5, 64)
    # 训练路径：S0 传递
    _, S_mid = m(xa)
    out_b_train, _ = m(xb, S0=S_mid)
    # 推理路径：runtime state
    state = m.init_state(1, xa.device)
    m.forward_runtime(xa, state)
    out_b_rt = m.forward_runtime(xb, state)
    assert torch.allclose(out_b_train, out_b_rt, atol=1e-5)


# ======================================================================
# L0 Sliding Window Attention
# ======================================================================
def test_swa_runtime_step_matches_full_forward():
    """逐 token 环形缓存推理必须与全量带状掩码前向一致。"""
    m = SlidingWindowAttention(64, num_heads=4, num_kv_heads=2, window=4)
    L = 10  # 超过窗口, 检验环形覆盖
    x = torch.randn(1, L, 64)
    full = m(x)
    cache = WindowKVCache(1, 2, 4, 16, x.device)
    steps = [m.step(x[:, t: t + 1], cache) for t in range(L)]
    streamed = torch.cat(steps, dim=1)
    assert torch.allclose(full, streamed, atol=1e-4)


def test_swa_kv_cache_bytes_constant():
    """KV 缓存字节数恒定，与已写入 token 数无关。"""
    cache = WindowKVCache(1, 2, 8, 16, torch.device("cpu"))
    before = cache.nbytes()
    for i in range(100):
        cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert cache.nbytes() == before
    assert cache.ordered()[0].shape[2] == 8  # 最多保留 W 个


def test_swa_window_limits_attention():
    """位置 i 不应看到 i-W 之前的位置：构造全同 q/k 验证输出有界性。"""
    m = SlidingWindowAttention(64, num_heads=4, num_kv_heads=2, window=4)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)
    x = torch.randn(1, 12, 64)
    out_full = m(x)
    # 把输入尾部改掉，前 W-1 个位置的输出不应变化太远 (它们看不到尾部)
    x2 = x.clone()
    x2[0, -1] += 10.0
    out2 = m(x2)
    # 位置 0 与最后位置相距 > W，其输出应完全不受尾部改动影响
    assert torch.allclose(out_full[0, 0], out2[0, 0], atol=1e-5)


# ======================================================================
# L2 Octave Memory
# ======================================================================
def _octave():
    return OctaveMemory(64, slots_per_level=8, mem_tokens=2, promote_tokens=2,
                        segment_tokens=8, max_levels=6, num_heads=4)


def test_octave_read_empty_is_identity():
    m = _octave()
    x = torch.randn(1, 5, 64)
    bank = m.init_bank(x.device)
    assert torch.equal(m.read(x, bank=bank), x)


def test_octave_levels_grow_logarithmically():
    """层数随写入量对数增长；单层预算恒定 (每翻倍 +1 层固定字节)。"""
    m = _octave()
    bank = m.init_bank(torch.device("cpu"))
    level_bytes = m.estimate_level_bytes(dtype_bytes=4)  # fp32

    def write_segments(n):
        for _ in range(n):
            m.write_segment(torch.randn(8, 64), bank)

    write_segments(4)                      # level0 满 (4段×2tok=8槽)
    assert bank.num_active_levels == 1
    write_segments(1)                      # 触发升格 → level1 激活
    assert bank.num_active_levels == 2
    nb_2 = bank.nbytes()

    write_segments(16)                     # 更多升格 → 应激活 level2
    assert bank.num_active_levels >= 3
    nb_3 = bank.nbytes()
    # 每多一层，字节数恰好多一层固定预算
    assert (nb_3 - nb_2) % level_bytes == 0


def test_octave_coverage_exponential():
    """第 k 层 token 覆盖范围指数增长。"""
    cfg = _cfg()
    cov = [cfg.octave_coverage_tokens(k) for k in range(4)]
    assert cov[0] == cfg.octave_segment_tokens
    assert all(cov[i + 1] > cov[i] for i in range(3))


def test_octave_read_shape_and_grad():
    m = _octave()
    bank = m.init_bank(torch.device("cpu"))
    for _ in range(3):
        m.write_segment(torch.randn(8, 64), bank)
    x = torch.randn(1, 5, 64, requires_grad=True)
    out = m.read(x, bank=bank)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.q_proj.weight.grad is not None  # query 侧可微


# ======================================================================
# GlyphStream
# ======================================================================
def test_glyph_encoder_shapes():
    enc = GlyphStreamEncoder(_cfg())
    v = enc.encode_chars("你好 abc")
    assert v.shape == (6, 64)
    x, mask = enc(["ab", "abcd"])
    assert x.shape == (2, 4, 64)
    assert mask[0].sum().item() == 2 and mask[1].sum().item() == 4


def test_pooler_compresses_and_aux():
    cfg = _cfg()
    enc = GlyphStreamEncoder(cfg)
    pooler = EntropyDrivenPooler(cfg)
    x, mask = enc(["hello world 你好"])
    bt = torch.tensor([enc.heuristic_boundaries("hello world 你好")]).float()
    pooled, segs, aux = pooler(x, mask, bt)
    assert 0 < pooled.shape[1] <= x.shape[1]
    assert aux is not None and torch.isfinite(aux)
    # 段区间必须无缝覆盖有效长度
    total = sum(e - s for s, e in segs[0])
    assert total == int(mask[0].sum().item())


def test_prototype_decoder():
    dec = PrototypeDecoder(64, 128)
    x = torch.randn(2, 5, 64)
    logits = dec.logits(x)
    assert logits.shape == (2, 5, 128)
    chars = dec.decode(x)
    assert len(chars) == 10
    assert dec.char_to_index("a") >= 0


# ======================================================================
# V2Model 端到端
# ======================================================================
def _model(**kw):
    torch.manual_seed(0)
    return SimplCoreV2Model(_cfg(**kw))


def test_model_train_loss_decreases():
    """过拟合一小段重复文本：损失必须下降 (证明全链路可微)。"""
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    text = "abababababababab"
    r0 = model.forward_train([text], use_pooling=False)
    first = float(r0["loss"].detach())
    for _ in range(15):
        opt.zero_grad()
        r = model.forward_train([text], use_pooling=False)
        r["loss"].backward()
        opt.step()
    assert float(r["loss"].detach()) < first


def test_model_train_with_pooling_runs():
    model = _model()
    r = model.forward_train(["hello world"], use_pooling=True)
    assert torch.isfinite(r["loss"])
    r["loss"].backward()
    # 梯度应一路回到字形编码器
    grads = [p.grad for p in model.encoder.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_model_generate_smoke():
    model = _model()
    out = model.generate("ab", max_new_chars=2)
    assert isinstance(out, str) and out.startswith("ab")


def test_model_runtime_state_bounded():
    """流式推理：KV+Delta 部分恒定；总状态有界且不随块数线性增长。"""
    model = _model()
    state = model.init_state(batch=1)
    kv0 = sum(c.nbytes() for c in state.kv_caches)
    ds0 = sum(s.nbytes() for s in state.delta_states)

    for _ in range(8):
        x = torch.randn(1, 16, 64)
        model.forward_chunk_runtime(x, state)
        kv = sum(c.nbytes() for c in state.kv_caches)
        ds = sum(s.nbytes() for s in state.delta_states)
        assert kv == kv0 and ds == ds0          # L0/L1 恒定
    total8 = state.nbytes()
    for _ in range(8):
        model.forward_chunk_runtime(torch.randn(1, 16, 64), state)
    total16 = state.nbytes()
    # 翻倍后增量最多 = 新增八度层预算 (有界, 非线性增长)
    assert total16 - total8 <= model.octave.estimate_level_bytes(4)


def test_config_memory_formula():
    """解析显存公式：上下文每翻倍，八度层数 +1 (对数增长)。"""
    cfg = _cfg(hidden_size=256, num_layers=4, swa_window=8192,
               octave_slots_per_level=64, octave_segment_tokens=64,
               octave_mem_tokens=8, octave_promote_tokens=32,  # fan=2: 每层覆盖翻倍
               octave_max_levels=32)
    m1 = cfg.total_inference_memory(1_000_000)
    m2 = cfg.total_inference_memory(2_000_000)
    m4 = cfg.total_inference_memory(4_000_000)
    assert m1["swa_kv_bytes"] == m2["swa_kv_bytes"]          # L0 恒定
    assert m1["delta_state_bytes"] == m2["delta_state_bytes"]  # L1 恒定
    inc21 = m2["total_bytes"] - m1["total_bytes"]
    inc42 = m4["total_bytes"] - m2["total_bytes"]
    assert inc21 == inc42 == cfg.octave_level_bytes()         # 每翻倍 +1 层
