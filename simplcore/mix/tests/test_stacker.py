"""结构感知堆叠器测试（修复命门 9：堆叠测试覆盖为零）。

覆盖：
  - structural_stack 输出与目标模型形状完全一致
  - 报告覆盖全部参数（每参数有策略）
  - 多源 interleave：expert 槽位正确映射
  - 对称破缺噪声生效（扩展张量 ≠ 纯 repeat）
  - buffers（RoPE 频率）保持目标初始化，不被污染
"""
import simplcore.tensor as torch

from simplcore.mix.config import SimplCoreMixConfig
from simplcore.mix.model import SimplCoreMix
from simplcore.mix.stacking.stacker import structural_stack


def _small_cfg() -> SimplCoreMixConfig:
    return SimplCoreMixConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        n_basis_vectors=8,
        n_memory_channels=2,
        memory_dim=32,
        lora_rank=4,
        reasoning_enabled=False,
        confidence_gate_enabled=False,
    )


def _target_cfg() -> SimplCoreMixConfig:
    """放大的目标：dim×2, heads×2, experts×2, 层数×2"""
    return SimplCoreMixConfig(
        vocab_size=256,
        dim=128,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=256,
        max_loop_iters=4,
        prelude_layers=2,
        coda_layers=2,
        n_experts=8,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=256,
        n_basis_vectors=16,
        n_memory_channels=4,
        memory_dim=64,
        lora_rank=8,
        reasoning_enabled=False,
        confidence_gate_enabled=False,
    )


def _make_source_models(n: int, seed: int):
    torch.manual_seed(seed)
    return [SimplCoreMix(_small_cfg()) for _ in range(n)]


def test_stack_single_source_shapes_match():
    torch.manual_seed(0)
    src = _make_source_models(1, seed=0)[0]
    target = SimplCoreMix(_target_cfg())
    result = structural_stack(
        [src.state_dict()],
        target,
        noise_std=0.02,
        expert_mode="tile",
    )
    filled = result["state_dict"]
    # 所有张量 shape 与目标一致
    for key, tv in target.state_dict().items():
        assert key in filled, f"缺失 {key}"
        assert tuple(filled[key].shape) == tuple(tv.shape), f"shape 不符 {key}"
    # 模型能加载并前向
    target.load_state_dict(filled)
    tokens = torch.randint(0, 256, (1, 16))
    out = target(tokens, n_loops=1)
    assert out.shape == (1, 16, 256)
    # 报告完整
    report = result["report"]
    assert len(report.strategies) > 0


def _count_experts(state) -> int:
    """数 state_dict 里某个 MoE 的 expert 槽位数。"""
    import re
    idxs = set()
    for k in state:
        m = re.match(r"^(recurrent\.moe\.experts\.)(\d+)(\.|\b)", k)
        if m:
            idxs.add(int(m.group(2)))
    return len(idxs)


def test_stack_multi_source_interleave_experts():
    torch.manual_seed(0)
    srcs = _make_source_models(3, seed=1)
    target = SimplCoreMix(_target_cfg())
    # 给每个 source 的 expert gate 打唯一标记（值 = source_idx+1）
    states = []
    for si, s in enumerate(srcs):
        sd = s.state_dict()
        for k in list(sd):
            if ".experts." in k and k.endswith(".gate.weight"):
                sd[k] = torch.full_like(sd[k], float(si + 1))
        states.append(sd)
    result = structural_stack(states, target, noise_std=0.0, expert_mode="interleave")
    filled = result["state_dict"]
    assert _count_experts(states[0]) == 4
    assert _count_experts(filled) == 8
    # interleave 槽位: dst=M → source=(M%3), expert=(M//3)%4
    for M in range(8):
        s = M % 3
        key = f"recurrent.moe.experts.{M}.gate.weight"
        val = float(filled[key][0, 0].item())
        assert val == float(s + 1), f"槽位 {M} 应来自 source{s} (标记{s+1}), got {val}"


def test_noise_breaks_periodic_repeat():
    torch.manual_seed(0)
    src = _make_source_models(1, seed=2)[0]
    target = SimplCoreMix(_target_cfg())
    # 无噪声：扩展张量应是"块状重复"
    torch.manual_seed(0)
    result_no = structural_stack(
        [src.state_dict()], target, noise_std=0.0, expert_mode="tile"
    )["state_dict"]
    torch.manual_seed(7)
    result_yes = structural_stack(
        [src.state_dict()], target, noise_std=0.05, expert_mode="tile"
    )["state_dict"]
    # norm1.weight: (64,) → (128,)。noise=0 时精确重复（前半 == 后半）
    key = "prelude.0.norm1.weight"
    a = result_no[key]  # (128,)
    b = result_yes[key]
    assert torch.equal(a[:64], a[64:]), "noise=0 时 dim 扩展应是精确重复"
    # 对称破缺：加噪声后前半 != 后半，且整体 != a
    assert not torch.equal(b[:64], b[64:]), "对称破缺噪声应打破重复"
    assert not torch.equal(a, b), "加噪声后的扩展权重应不同于纯重复"


def test_buffers_preserved():
    torch.manual_seed(0)
    src = _make_source_models(1, seed=3)[0]
    target = SimplCoreMix(_target_cfg())
    result = structural_stack(
        [src.state_dict()], target, noise_std=0.02, expert_mode="tile"
    )["state_dict"]
    # freqs_cis 是 buffer，应保持目标模型的值（不被 source 污染）
    assert torch.equal(
        result["freqs_cis"], target.state_dict()["freqs_cis"]
    ), "buffer 不应被堆叠改动"


def test_missing_layer_falls_back_to_random_keep():
    # 目标层数 > 源层数时，新增层应保留目标随机初始化并记 warning
    torch.manual_seed(0)
    src = _make_source_models(1, seed=4)[0]
    target = SimplCoreMix(_target_cfg())  # target 2 层, src 1 层
    result = structural_stack(
        [src.state_dict()], target, noise_std=0.02, expert_mode="tile"
    )
    report = result["report"]
    # 至少有一条"源无此参数"的 warning（prelude.1.*）
    assert any("保留目标随机初始化" in w for w in report.warnings)
