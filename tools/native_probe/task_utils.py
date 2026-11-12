"""
task_utils.py — 共享任务与配置（纯数据层，无任何模型实现）。

bind1：序列 [k1 v1 k2 v2 ... kn vn q]，q == 某个 key 的 id，
目标输出对应 value；只在 t=L-1 算 CE。

兼容口径
========
`build(..., vocab=64)` 是默认路径，与历史所有结果逐位一致。
`vocab>64` 时 key 占 [0, V/2)，value 占 [V/2, V)，用于
「Mamba 级别」的大词表协议（Hadamard 要求 V 为 2 的幂）。
"""
import numpy as np

D_IN = 32
M_DEF = 16
C = 4
H = M_DEF * C
V_KEY = 32
V_VAL = 32
VOCAB = V_KEY + V_VAL
N_PAIRS = 4
L_TRAIN = 2 * N_PAIRS + 1
B = 16
STEPS = 800
LR = 3e-3
SEEDS = (0, 1, 2)


def split_vocab(vocab):
    """返回 (key 数量, value 数量, value 起始 id)。"""
    assert vocab >= 4 and vocab % 2 == 0, vocab
    return vocab // 2, vocab // 2, vocab // 2


def build(n_pairs, n_samples, seed, shuffle_pairs=False, vocab=VOCAB):
    """返回 one-hot 输入、前驱 one-hot（t=0 为零）、target id、元数据。"""
    vk, vv, v0 = split_vocab(vocab)
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    X = np.zeros((n_samples, L, vocab))
    Xp = np.zeros((n_samples, L, vocab))
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        keys = r.choice(vk, size=n_pairs, replace=False)
        vals = r.choice(vv, size=n_pairs, replace=False) + v0
        pairs = list(zip(keys, vals))
        if shuffle_pairs and n_pairs > 1:
            r.shuffle(pairs)
        j = int(r.integers(0, n_pairs))
        for p_, (k, v) in enumerate(pairs):
            X[i, 2 * p_, k] = 1.0
            X[i, 2 * p_ + 1, v] = 1.0
        q = pairs[j][0]
        X[i, L - 1, q] = 1.0
        Y[i] = pairs[j][1]
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(L=L, n_pairs=n_pairs, shuffle=shuffle_pairs,
                          seed=seed, vocab=vocab)


def build_multihop(n_pairs, n_samples, seed, vocab=VOCAB, hops=2):
    """bind2：query 给 k_j，答案是链上下一跳的 value（v_{j+1}），
    或 hops=3 时再跳一次。单层缩并读出是否够用，由实验判定。

    序列仍是 [k1 v1 ... kn vn q]；目标改为第 (j+hops-1) mod n 对的 value。
    注意这改变了「key→value 直接绑定」的定义，是**判别性硬度对照**，
    不是 MQAR 标准协议。
    """
    vk, vv, v0 = split_vocab(vocab)
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    X = np.zeros((n_samples, L, vocab))
    Xp = np.zeros((n_samples, L, vocab))
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        keys = r.choice(vk, size=n_pairs, replace=False)
        vals = r.choice(vv, size=n_pairs, replace=False) + v0
        pairs = list(zip(keys, vals))
        j = int(r.integers(0, n_pairs))
        for p_, (k, v) in enumerate(pairs):
            X[i, 2 * p_, k] = 1.0
            X[i, 2 * p_ + 1, v] = 1.0
        X[i, L - 1, pairs[j][0]] = 1.0
        Y[i] = pairs[(j + hops - 1) % n_pairs][1]
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(L=L, n_pairs=n_pairs, hops=hops,
                          seed=seed, vocab=vocab)


def trivial_baselines(Y, n_pairs, vocab=VOCAB):
    vk, vv, _ = split_vocab(vocab)
    return dict(chance=1.0 / vv, last_value=1.0 / n_pairs)
