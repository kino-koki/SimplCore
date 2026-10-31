"""序列长度课程单元测试（SimplCore 4M 超长上下文卖点的训练编排）。

覆盖：CLI 解析、context 解析、阶段查询、末段语义、零回归（无课程 = falsy）、
CLI 优先于 context。课程不影响模型结构，仅决定训练期数据窗口长度。
"""

import argparse

from simplcore.core.seqlen_curriculum import SeqLenCurriculum, build_curriculum


def test_cli_parse_and_stage_query():
    cur = SeqLenCurriculum.from_cli("4096:0.1,16384:0.25,65536:0.45,4000000", 100000)
    assert bool(cur) is True
    assert cur.max_seq_len == 4000000
    assert cur.seq_len_at(0) == 4096
    assert cur.seq_len_at(10000) == 16384
    assert cur.seq_len_at(30000) == 65536
    assert cur.seq_len_at(99999) == 4000000     # 末段跑到结束
    # 末段 until_step 必须为 None（跑到结束）
    assert cur.stages[-1].until_step is None
    print("[ok] cli parse + stage query (max=4M)")


def test_last_frac_optional():
    # 末段比例可省，默认 1.0（"8192" 为末段）
    cur = SeqLenCurriculum.from_cli("2048:0.5,8192", 1000)
    assert cur.seq_len_at(0) == 2048
    assert cur.seq_len_at(999) == 8192
    print("[ok] last frac optional")


def test_middle_must_have_frac():
    try:
        SeqLenCurriculum.from_cli("2048,8192", 1000)   # 中间段省比例 -> 报错
        raise AssertionError("中间段省比例应抛 ValueError")
    except ValueError:
        pass
    print("[ok] middle stage requires frac (ValueError)")


def test_context_parse():
    ctx = {
        "max_seq_len": 4000000,
        "curriculum": [
            {"seq_len": 4096, "until_frac": 0.1},
            {"seq_len": 16384, "until_frac": 0.25},
            {"seq_len": 4000000},
        ],
    }
    cur = SeqLenCurriculum.from_context(ctx, 100000)
    assert cur.seq_len_at(0) == 4096
    assert cur.seq_len_at(20000) == 16384
    assert cur.seq_len_at(90000) == 4000000
    print("[ok] context parse (max=4M)")


def test_no_curriculum_is_falsy_zero_regression():
    cur = SeqLenCurriculum.from_context(None, 100000)
    assert bool(cur) is False
    cur2 = SeqLenCurriculum.from_cli("", 100000)
    assert bool(cur2) is False
    # 仅 max_seq_len 无 curriculum -> 不启用课程（保持固定 seq_len）
    cur3 = SeqLenCurriculum.from_context({"max_seq_len": 4000000}, 100000)
    assert bool(cur3) is False
    print("[ok] no curriculum = falsy (zero regression)")


def test_cli_overrides_context():
    ctx = {"max_seq_len": 4000000,
           "curriculum": [{"seq_len": 4096, "until_frac": 0.1}, {"seq_len": 4000000}]}
    args = argparse.Namespace(seq_len_curriculum="1024:0.5,4096")
    cur = build_curriculum(args, type("C", (), {"context": ctx})(), 1000)
    assert cur.seq_len_at(0) == 1024
    assert cur.seq_len_at(999) == 4096
    print("[ok] CLI overrides preset context")


def test_monotonic_until_guard():
    # 截止必须递增
    try:
        SeqLenCurriculum.from_cli("4096:0.5,8192:0.3,4000000", 1000)
        raise AssertionError("非递增截止应抛 ValueError")
    except ValueError:
        pass
    print("[ok] monotonic until_step guard (ValueError)")


def test_short_run_rounding_collision_is_monotonic():
    """max_steps 很小（如 1）时 0.1/0.25 都会 round 到 0——这是取整碰撞，
    不是配置笔误；应单调化而不是启动即抛错。"""
    cur = SeqLenCurriculum.from_cli(
        "4096:0.1,16384:0.25,65536:0.45,4000000", 1)
    untils = [st.until_step for st in cur.stages]
    finite = [u for u in untils if u is not None]
    assert finite == sorted(set(finite)), f"截止未单调化: {untils}"
    assert untils[-1] is None
    print("[ok] short-run rounding collision is monotonic")


if __name__ == "__main__":
    test_cli_parse_and_stage_query()
    test_last_frac_optional()
    test_middle_must_have_frac()
    test_context_parse()
    test_no_curriculum_is_falsy_zero_regression()
    test_cli_overrides_context()
    test_monotonic_until_guard()
    test_short_run_rounding_collision_is_monotonic()
    print("\nALL SEQLEN CURRICULUM TESTS PASSED")
