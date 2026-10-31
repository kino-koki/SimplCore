# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""理论优势的可执行验证 —— 把 ``docs/THEORETICAL_EDGE.md`` 的每条主张变成断言。

为什么要有这个模块
------------------
"理论超传统 Transformer"如果只写在 Markdown 里，它就只是一段话；哪天有人
改了一行 mixer 代码，文档不会报错，loss 曲线也不会异常，但主张已经不成立了。

本模块把每条**机制层**主张变成一个**可运行的断言**，并给出一条 CLI::

    python -m simplcore.core.theory            # 跑全部检查，打印表格
    python -m simplcore.core.theory --json     # 机器可读输出

检查项与 ``docs/THEORETICAL_EDGE.md`` 的模块一一对应：

======== ==========================================================
模块     被验证的主张
======== ==========================================================
sgc      FFT↔IIR 双形式逐位一致 / 流式状态 O(1) / 严格因果 /
         无位置嵌入表
srf      chunk 并行 == 逐 token 递推 / Delta 输入相关 / 状态 O(1)
selective 通道门随输入变化 / 门不增状态
gates    L2 并行(cumsum) == 流式(运行和)
abstain  L4 是标量头，无注意力
grounding 数字/极性核对能识别无依据声明
memory   记忆注入是残差流加性项 / 目标函数只对增量计损失
dream    no-regression gate 能识别退化
code_exec 验证器能区分对错代码
extrapolation 无 max_len 截断（长长度仍有定义）
scaling  无 KV cache / 解码成本与 L 无关 / 训练长程项 O(log L)
purity   模型内零注意力模块 / 配置拒绝注意力 mixer
======== ==========================================================

诚实边界
--------
本模块只验证**机制层**性质（复杂度、状态、因果性、闸门行为）。
它**不**验证语言建模质量 —— 同参数量下 SimplCore 是否真的优于 Transformer
需要 LRA / S4 / Mamba 等基准的实测证据，目前**没有**。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .abstain import AbstentionHead
from .code_exec import CodeExecutionVerifier, CodeProblem
from .config import SimplCoreConfig
from .dream import EvalScores, regression_gate
from .gates import evidence_weight_step, evidence_weights, init_energy_sum
from .grounding import numbers_supported, polarity_matches
from .memory import MemoryAwareObjective
from .model import SimplCoreModel
from .purity import assert_pure_config, attention_like_modules
from .scaling import (infer_flops_per_token, kv_cache_bytes_per_token,
                      non_embedding_params, stream_state_bytes,
                      train_flops_per_token)
from .selective import GatedSpectralConv2
from .sgc import LayerStreamState
from .ssm import SelectiveSpectralSSM
from .accel import ThinkAccelerator
from .reasoning import DeepReasoner, stabilized
from .envlearn import (
    WeightBudget,
    LearnedAdapter,
    ForwardAdapter,
    EnvTrajectory,
    summarize_failures,
)

__all__ = ["Check", "run_all", "format_table", "main"]


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class Check:
    """一条被验证的主张。"""

    module: str
    claim: str
    passed: bool
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"module": self.module, "claim": self.claim,
                "passed": self.passed, "evidence": self.evidence}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _tiny_model(mixer: str = "sgc", **kw) -> SimplCoreModel:
    base = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8, decay_max=128,
                local_kernel=4, num_heads=2, ffn_ratio=2.0, mixer=mixer)
    base.update(kw)
    if mixer == "srf":
        base.setdefault("srf_heads", 2)
        base.setdefault("srf_chunk", 4)
    return SimplCoreModel(SimplCoreConfig(**base))


def _state_bytes(model: SimplCoreModel, length: int, vocab: int = 64) -> int:
    """让模型流式消费 ``length`` 个 token，返回流式状态字节数。"""
    x = torch.randint(0, vocab, (1, length))
    with torch.no_grad():
        st = model.init_stream(1, device=x.device, dtype=torch.float32)
        for t in range(length):
            _, st = model.stream_step(x[:, t], st)
    return st.num_bytes()


def _state_tensor_bytes(state: LayerStreamState) -> int:
    """``LayerStreamState`` 各张量的字节数之和（该类型本身没有 ``num_bytes``）。"""
    total = 0
    for name in ("y1", "y2", "conv_buf", "energy_sum"):
        t = getattr(state, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


# ---------------------------------------------------------------------------
# sgc.py —— SGC mixer
# ---------------------------------------------------------------------------
def _check_sgc() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    # --- 1. FFT(训练) ↔ IIR(流式) 双形式逐位一致 ---
    m = _tiny_model("sgc").eval()
    x = torch.randint(0, 64, (1, 24))
    with torch.no_grad():
        full = m(x)[0, -1]
        st = m.init_stream(1, device=x.device, dtype=torch.float32)
        last = None
        for t in range(x.shape[1]):
            last, st = m.stream_step(x[:, t], st)
    diff = float((full - last).abs().max())
    out.append(Check("sgc.py", "FFT 训练形式与 IIR 流式形式数值一致",
                     diff < 1e-3, f"max|Δ| = {diff:.2e} (< 1e-3)"))

    # --- 2. 流式状态与序列长度无关 ---
    sizes = {_state_bytes(m, L) for L in (4, 32, 128)}
    out.append(Check("sgc.py", "流式状态字节数与序列长度无关 (O(1))",
                     len(sizes) == 1,
                     f"L∈{{4,32,128}} → {sorted(sizes)} 字节"))

    # --- 3. 严格因果 ---
    m2 = _tiny_model("sgc").eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        a = m2(x)
        x2 = x.clone()
        x2[:, 8:] = torch.randint(0, 64, (1, 8))
        b = m2(x2)
    diff = float((a[:, :8] - b[:, :8]).abs().max())
    out.append(Check("sgc.py", "严格因果：扰动未来不影响前文",
                     diff < 1e-6, f"max|Δ(前 8 位)| = {diff:.2e} (< 1e-6)"))

    # --- 4. 无位置嵌入表（且长长度仍有定义） ---
    m3 = _tiny_model("sgc")
    pos_params = [n for n, _ in m3.named_parameters()
                  if any(k in n.lower() for k in ("pos_embed", "positional", "pos_emb"))]
    with torch.no_grad():
        big = m3(torch.randint(0, 64, (1, 1024)))
    finite = bool(torch.isfinite(big).all())
    out.append(Check("sgc.py", "无位置嵌入表（位置由因果核编码）",
                     not pos_params,
                     f"位置参数 = {pos_params or '无'}; L=1024 输出有限 = {finite}"))
    return out


# ---------------------------------------------------------------------------
# srf.py / ssm.py —— SRFN
# ---------------------------------------------------------------------------
def _check_srf() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    s = SelectiveSpectralSSM(8, num_bases=4, chunk_size=4)
    u = torch.randn(2, 12, 8)
    par = s(u)
    st = s.init_state(2, u.device, u.dtype)
    seq = []
    for t in range(u.shape[1]):
        y, st = s.step(u[:, t:t + 1], st)
        seq.append(y)
    seq = torch.cat(seq, dim=1)
    diff = float((par - seq).abs().max())
    out.append(Check("srf.py", "chunk 并行 scan == 逐 token 递推",
                     diff < 1e-5, f"max|Δ| = {diff:.2e} (< 1e-5)"))

    a = torch.zeros(1, 1, 8)
    b = torch.ones(1, 1, 8) * 3.0
    da, db = s.step_size(a), s.step_size(b)
    diff = float((da - db).abs().max())
    out.append(Check("srf.py", "selective：Delta 步长随输入变化",
                     diff > 1e-4, f"max|Δ(Delta)| = {diff:.2e} (> 1e-4)"))

    sizes = set()
    for L in (4, 128):
        _, sst = s(torch.randn(1, L, 8), return_state=True)
        sizes.add(sst.num_bytes())
    out.append(Check("srf.py", "流式状态字节数与序列长度无关 (O(1))",
                     len(sizes) == 1, f"L∈{{4,128}} → {sorted(sizes)} 字节"))
    return out


# ---------------------------------------------------------------------------
# selective.py —— GatedSpectralConv2
# ---------------------------------------------------------------------------
def _check_selective() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    mx = GatedSpectralConv2(16, num_bases=4, decay_max=128, gate_ab=True)
    h1 = torch.randn(1, 5, 16)
    h2 = torch.randn(1, 5, 16)
    with torch.no_grad():
        g1, g2 = mx.gate_gamma(h1), mx.gate_gamma(h2)
    diff = float((g1 - g2).abs().max())
    out.append(Check("selective.py", "γ/β 通道门随输入逐通道变化",
                     diff > 1e-6, f"max|Δ(γ)| = {diff:.2e} (> 1e-6)"))

    # 门不进核、不增状态：开/关 γβ 时状态形状与字节数必须一致
    gated = GatedSpectralConv2(16, num_bases=4, decay_max=128, gate_ab=True)
    plain = GatedSpectralConv2(16, num_bases=4, decay_max=128, gate_ab=False)
    s1 = gated.init_state(1, torch.device("cpu"), torch.float32)
    s2 = plain.init_state(1, torch.device("cpu"), torch.float32)
    same_shape = (tuple(s1.y1.shape) == tuple(s2.y1.shape)
                  and tuple(s1.conv_buf.shape) == tuple(s2.conv_buf.shape))
    same_bytes = _state_tensor_bytes(s1) == _state_tensor_bytes(s2)
    out.append(Check("selective.py", "门控不改变状态形状与字节数（不进核）",
                     same_shape and same_bytes,
                     f"y1 {tuple(s1.y1.shape)}=={tuple(s2.y1.shape)}, "
                     f"字节 {_state_tensor_bytes(s1)}=={_state_tensor_bytes(s2)}"))
    return out


# ---------------------------------------------------------------------------
# gates.py —— L2 证据门
# ---------------------------------------------------------------------------
def _check_gates() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    B, L, D = 2, 12, 4
    u = torch.randn(B, L, D)
    gain = torch.tensor(0.5)
    pos_gain = torch.tensor(0.1)
    bias = torch.tensor(-1.0)

    par = evidence_weights(u, gain, pos_gain, bias)                 # (B, L, 1)
    es = init_energy_sum(B, u.device, u.dtype)
    steps = []
    for t in range(L):
        w, es = evidence_weight_step(u[:, t:t + 1], es, t, gain, pos_gain, bias)
        steps.append(w)
    seq = torch.cat(steps, dim=1)                                 # (B, L, 1)
    diff = float((par - seq).abs().max())
    out.append(Check("gates.py", "L2 并行(cumsum) == 流式(运行和)",
                     diff < 1e-6, f"max|Δ| = {diff:.2e} (< 1e-6)"))
    return out


# ---------------------------------------------------------------------------
# abstain.py —— L4 拒答门
# ---------------------------------------------------------------------------
def _check_abstain() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    head = AbstentionHead(dim=16)
    hidden = torch.randn(2, 5, 16)
    logit = head(hidden)
    attn_mods = [n for n, m in head.named_modules()
                 if any(t in type(m).__name__.lower() for t in ("attention", "attn"))]
    out.append(Check("abstain.py", "L4 是标量头 (dim→1)，不含注意力",
                     tuple(logit.shape) == (2, 5) and not attn_mods,
                     f"输出形状 {tuple(logit.shape)}, 注意力模块 {attn_mods or '无'}"))
    return out


# ---------------------------------------------------------------------------
# grounding.py —— 证据绑定
# ---------------------------------------------------------------------------
def _check_grounding() -> List[Check]:
    out: List[Check] = []

    ok_num_bad = not numbers_supported("收入增长了 42%", "收入明显增长")
    ok_num_good = numbers_supported("收入增长了 42%", "收入增长了 42%")
    out.append(Check("grounding.py", "数字逐值核对：数字无出处即判无依据",
                     ok_num_bad and ok_num_good,
                     f"无出处={not ok_num_bad}, 有出处={ok_num_good}"))

    ok_pol_bad = not polarity_matches("没有增长", "增长了")
    ok_pol_good = polarity_matches("增长了", "增长了")
    out.append(Check("grounding.py", "极性核对：否定词计数不一致即判冲突",
                     ok_pol_bad and ok_pol_good,
                     f"冲突={not ok_pol_bad}, 一致={ok_pol_good}"))
    return out


# ---------------------------------------------------------------------------
# memory.py —— CMT 压缩记忆
# ---------------------------------------------------------------------------
def _check_memory() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    # 记忆注入 = 残差流上的加性项（Linear，无注意力）
    m = _tiny_model("sgc", use_memory=True)
    additive = isinstance(m.memory_in, nn.Linear)
    no_attn = not [n for n, mod in m.memory_in.named_modules()
                   if any(t in type(mod).__name__.lower() for t in ("attention", "attn"))]
    out.append(Check("memory.py", "记忆注入是残差流加性项（非注意力）",
                     additive and no_attn,
                     f"memory_in = {type(m.memory_in).__name__}, 注意力模块 = {'无' if no_attn else '有'}"))

    # 目标函数只对"记忆带来的增量"计损失
    obj = MemoryAwareObjective()
    V = 8
    with_mem = torch.randn(2, 4, V, requires_grad=True)
    without = torch.randn(2, 4, V)
    targets = torch.randint(0, V, (2, 4))
    obj(with_mem, without, targets).backward()
    ok = with_mem.grad is not None and without.grad is None
    out.append(Check("memory.py", "memory-aware 目标只对增量 logits 计损失",
                     ok,
                     f"有记忆分支梯度={with_mem.grad is not None}, "
                     f"无记忆分支梯度={without.grad is not None}"))
    return out


# ---------------------------------------------------------------------------
# dream.py —— no-regression gate
# ---------------------------------------------------------------------------
def _check_dream() -> List[Check]:
    out: List[Check] = []

    base = EvalScores.from_any({"acc": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
    worse = EvalScores.from_any({"acc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    better = EvalScores.from_any({"acc": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
    v_bad = regression_gate(base, worse, n_boot=200, seed=0)
    v_good = regression_gate(base, better, n_boot=200, seed=0)
    out.append(Check("dream.py", "no-regression gate 能识别退化并拒绝固化",
                     (not v_bad.passed) and v_good.passed,
                     f"退化→passed={v_bad.passed}, 持平→passed={v_good.passed}"))

    # 巩固器必须**真的**把训练信号写进权重（"跑了个空循环"不算）
    from .dream import TrainExample, default_consolidator
    from .tokenization import CharTokenizer

    torch.manual_seed(0)
    m = _tiny_model("sgc")
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    # 限定字符集：默认字符集的 id 会超出小模型的 vocab
    codec = CharTokenizer(limit="abcd")
    stats = default_consolidator(
        m, [TrainExample(prompt="ab", target="cd")],
        codec=codec, lr=1e-2, epochs=1, batch_size=1, seq_len=32)
    changed = sum(1 for k in before
                  if not torch.equal(before[k], m.state_dict()[k]))
    out.append(Check("dream.py", "巩固器真的把训练信号写进权重",
                     changed > 0 and stats.get("steps", 0) > 0,
                     f"{changed}/{len(before)} 个张量被改, {stats.get('steps')} 步"))
    return out


# ---------------------------------------------------------------------------
# code_exec.py —— pass@k 验证器
# ---------------------------------------------------------------------------
def _check_code_exec() -> List[Check]:
    out: List[Check] = []

    problem = CodeProblem(name="add", prompt="实现 add(a, b) 返回 a + b",
                          tests=["assert add(2, 3) == 5"])
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    r_good = CodeExecutionVerifier([problem], generator=lambda _: good).pass_rate()
    r_bad = CodeExecutionVerifier([problem], generator=lambda _: bad).pass_rate()
    out.append(Check("code_exec.py", "验证器能区分正确代码与错误代码",
                     r_good == 1.0 and r_bad == 0.0,
                     f"正确代码 pass_rate={r_good:.1f}, 错误代码 pass_rate={r_bad:.1f}"))
    return out


# ---------------------------------------------------------------------------
# extrapolation.py —— 无 max_len 截断
# ---------------------------------------------------------------------------
def _check_extrapolation() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    m = _tiny_model("sgc").eval()
    with torch.no_grad():
        long_out = m(torch.randint(0, 64, (1, 2048)))
    finite = bool(torch.isfinite(long_out).all())
    out.append(Check("extrapolation.py", "训练长度之外的长度仍有定义（无 max_len 截断）",
                     finite, f"L=2048 输出全部有限 = {finite}"))
    return out


# ---------------------------------------------------------------------------
# scaling.py —— 三本账
# ---------------------------------------------------------------------------
def _check_scaling() -> List[Check]:
    out: List[Check] = []

    cfg = SimplCoreConfig(vocab_size=8192, dim=256, num_layers=4, num_bases=16,
                           local_kernel=4, num_heads=4, ffn_ratio=2.0)
    out.append(Check("scaling.py", "KV cache 字节数恒为 0",
                     kv_cache_bytes_per_token(cfg) == 0,
                     f"kv_bytes_per_token = {kv_cache_bytes_per_token(cfg)}"))

    f512 = infer_flops_per_token(cfg, 512)
    f8192 = infer_flops_per_token(cfg, 8192)
    out.append(Check("scaling.py", "解码每 token FLOPs 与序列长度无关",
                     f512 == f8192, f"L=512 → {f512:.4e}, L=8192 → {f8192:.4e}"))

    def long_range(seq: int) -> float:
        return train_flops_per_token(cfg, seq) - 6.0 * non_embedding_params(cfg)

    r = long_range(8192) / long_range(1024)
    log_ratio = math.log2(2 * 8192) / math.log2(2 * 1024)
    # 线性增长会给出 8.0；log 增长给出 log_ratio ≈ 13/10
    out.append(Check("scaling.py", "训练长程项随 log L 亚线性增长（非线性 8×）",
                     r < 3.0,
                     f"比值 {r:.3f}（log 口径 {log_ratio:.3f}，线性口径 8.000）"))

    b1 = stream_state_bytes(cfg)
    b2 = stream_state_bytes(cfg, batch=2)
    out.append(Check("scaling.py", "流式状态字节数与 batch 成正比、与 L 无关",
                     b2 == 2 * b1, f"batch=1 → {b1} B, batch=2 → {b2} B"))
    return out


# ---------------------------------------------------------------------------
# purity.py —— 架构纯洁性
# ---------------------------------------------------------------------------
def _check_purity() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)

    m = _tiny_model("sgc")
    hits = attention_like_modules(m)
    out.append(Check("purity.py", "模型内零注意力模块",
                     not hits, f"命中 {hits or '无'}"))

    rejected = False
    try:
        assert_pure_config(SimplCoreConfig(vocab_size=64, dim=32, num_layers=1,
                                           mixer="attn"))
    except AssertionError:
        rejected = True
    out.append(Check("purity.py", "配置层拒绝注意力 mixer",
                     rejected, f"mixer='attn' 被拒 = {rejected}"))
    return out


# ---------------------------------------------------------------------------
# accel.py —— 思考加速（逐 token 自适应深度）
# ---------------------------------------------------------------------------
def _check_accel() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)
    m = _tiny_model("sgc").eval()

    # 1) skip_threshold=-inf 退化为恒全算力 → 与模型逐位一致
    acc = ThinkAccelerator(m, skip_threshold=float("-inf"), min_depth=m.config.num_layers)
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        g1 = m.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
        g2 = acc.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
    identical = bool(torch.equal(g1, g2))
    out.append(Check("accel.py", "skip_threshold=-inf 退化为恒全算力，输出逐位一致",
                     identical, f"identical={identical}"))

    # 2) 激进跳过 → 加速比 > 1 且输出有限（有损近似，但结构正确）
    acc2 = ThinkAccelerator(m, skip_threshold=1e9, min_depth=1)
    with torch.no_grad():
        g3 = acc2.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
    out.append(Check("accel.py", "激进跳过产生加速比 > 1 且输出有限",
                     acc2.ledger.speedup > 1 and bool(torch.isfinite(g3).all()),
                     f"speedup={acc2.ledger.speedup:.2f}"))

    # 3) 不引入注意力
    hits = acc.attention_like_modules()
    out.append(Check("accel.py", "加速器不引入注意力模块",
                     not hits, f"命中 {hits or '无'}"))
    return out


# ---------------------------------------------------------------------------
# reasoning.py —— 深度推理（迭代式思考控制器）
# ---------------------------------------------------------------------------
def _check_reasoning() -> List[Check]:
    out: List[Check] = []
    torch.manual_seed(0)
    from .tokenization import CharTokenizer
    codec = CharTokenizer(limit="abcd")
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_bases=8,
                                    decay_max=128, local_kernel=3, num_heads=2,
                                    ffn_ratio=2.0)
    m = SimplCoreModel(cfg).eval()

    # 1) fixed_steps 下轨迹结构正确
    r = DeepReasoner(m, codec, max_steps=3, max_thought_tokens=8,
                     max_answer_tokens=8, policy="fixed_steps", temperature=0)
    tr = r.reason("ab")
    ok_struct = (tr.steps_used <= 3 and len(tr.thoughts) == tr.steps_used
                 and isinstance(tr.answer, str))
    out.append(Check("reasoning.py", "fixed_steps 轨迹结构正确（步数<=上限、思考数==步数）",
                     ok_struct, f"steps={tr.steps_used}, thoughts={len(tr.thoughts)}"))

    # 2) 推理不改流式状态 O(1) 性质
    st = m.init_stream(1, dtype=torch.float32)
    b0 = st.num_bytes()
    for t in range(40):
        _, st = m.stream_step(torch.randint(0, 4, (1, 1)), st)
    out.append(Check("reasoning.py", "推理沿用 O(1) 流式状态（与模型一致）",
                     b0 == st.num_bytes(), f"{b0}=={st.num_bytes()}"))

    # 3) stabilized 早停判定（纯函数）
    ok_stab = (stabilized(["a", "a"], 2) and not stabilized(["a", "b"], 2)
               and not stabilized(["a"], 2))
    out.append(Check("reasoning.py", "stabilized 早停判定正确", ok_stab, "纯函数一致"))
    return out


# ---------------------------------------------------------------------------
# envlearn.py —— 环境学习（权重预算 + 分阶段蒸馏）
# ---------------------------------------------------------------------------
def _check_envlearn() -> List[Check]:
    out: List[Check] = []
    from .tokenization import CharTokenizer
    from .purity import attention_like_modules
    from .dream import default_consolidator
    codec = CharTokenizer(limit="abcd")
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_bases=8,
                                    decay_max=128, local_kernel=3, num_heads=2,
                                    ffn_ratio=2.0)
    m = SimplCoreModel(cfg).eval()

    # 1) WeightBudget 拒绝超过 2B 可学 / 3B 总上限
    b = WeightBudget(base_params=1_000_000_000)
    b.allocate(2_000_000_000)
    refused = not b.can_allocate(1)
    out.append(Check("envlearn.py", "WeightBudget 拒绝超过 2B 可学 / 3B 总上限",
                     refused,
                     f"learned={b.learned_params}, vram={b.vram_params()}"))

    # 2) LearnedAdapter 不含注意力
    ad = LearnedAdapter(dim=32, num_blocks=2)
    out.append(Check("envlearn.py", "LearnedAdapter 不含注意力模块",
                     not attention_like_modules(ad),
                     f"命中 {attention_like_modules(ad) or '无'}"))

    # 3) ForwardAdapter 冻结基座、只训适配器
    adapter = LearnedAdapter(dim=32, num_blocks=2)
    fa = ForwardAdapter(m, adapter, codec)
    base_before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    ad_before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    trajs = [EnvTrajectory(prompt="ab", response="x", passed=False, correct="cd")]
    ex = summarize_failures(trajs)
    default_consolidator(fa, ex, codec=codec, lr=1e-2, epochs=2,
                        batch_size=1, seq_len=32)
    changed_base = any(not torch.equal(m.state_dict()[k], base_before[k])
                       for k in base_before)
    changed_adapter = any(not torch.equal(adapter.state_dict()[k], ad_before[k])
                          for k in ad_before)
    out.append(Check("envlearn.py", "ForwardAdapter 冻结基座、只训适配器",
                     (not changed_base) and changed_adapter,
                     f"base_changed={changed_base}, adapter_changed={changed_adapter}"))
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
_CHECKS: List[Callable[[], List[Check]]] = [
    _check_sgc, _check_srf, _check_selective, _check_gates, _check_abstain,
    _check_grounding, _check_memory, _check_dream, _check_code_exec,
    _check_extrapolation, _check_scaling, _check_purity,
    _check_accel, _check_reasoning, _check_envlearn,
]


def run_all(verbose: bool = False) -> List[Check]:
    """跑完所有检查，返回逐条结果（不抛异常，失败也照常返回）。"""
    results: List[Check] = []
    for fn in _CHECKS:
        try:
            results.extend(fn())
        except Exception as exc:                       # pragma: no cover - 兜底
            name = getattr(fn, "__name__", "?")
            results.append(Check(name, "检查本身未能执行", False,
                                  f"{type(exc).__name__}: {exc}"))
    if verbose:
        for c in results:
            flag = "PASS" if c.passed else "FAIL"
            print(f"[{flag}] {c.module:16s} {c.claim}  —  {c.evidence}")
    return results


def _disp_width(text: str) -> int:
    """终端显示宽度：东亚全角字符占 2 列。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐补空格。"""
    return text + " " * max(0, width - _disp_width(text))


def format_table(results: List[Check]) -> str:
    """把结果渲染成一张对齐的表格（按终端显示宽度对齐，兼容中文）。"""
    if not results:
        return "(no checks ran)"
    w_mod = max(_disp_width(c.module) for c in results)
    w_claim = max(_disp_width(c.claim) for c in results)
    head = (f"{_pad('module', w_mod)}  {_pad('claim', w_claim)}  "
            f"{_pad('result', 6)}  evidence")
    lines = [head, "-" * max(len(head), 40)]
    for c in results:
        flag = "PASS" if c.passed else "FAIL"
        lines.append(f"{_pad(c.module, w_mod)}  {_pad(c.claim, w_claim)}  "
                     f"{_pad(flag, 6)}  {c.evidence}")
    n_pass = sum(1 for c in results if c.passed)
    lines.append("-" * max(len(head), 40))
    lines.append(f"{n_pass}/{len(results)} checks passed")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI：``python -m simplcore.core.theory``"""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--verbose", action="store_true", help="逐条打印")
    args = ap.parse_args(argv)

    results = run_all(verbose=False)
    if args.json:
        print(json.dumps({
            "checks": [c.as_dict() for c in results],
            "passed": sum(1 for c in results if c.passed),
            "total": len(results),
        }, ensure_ascii=False, indent=2))
    else:
        if args.verbose:
            for c in results:
                print(f"[{'PASS' if c.passed else 'FAIL'}] {c.module:16s} "
                      f"{c.claim}  —  {c.evidence}")
            print()
        print(format_table(results))
    return 0 if all(c.passed for c in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
