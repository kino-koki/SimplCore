"""
task_utils.py — 共享任务与配置（纯数据层，无任何模型实现）。

bind1：序列 [k1 v1 k2 v2 ... kn vn q]，q == 某个 key 的 id，
目标输出对应 value；只在 t=L-1 算 CE。
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


def build(n_pairs, n_samples, seed, shuffle_pairs=False):
    """返回 one-hot 输入、前驱 one-hot（t=0 为零）、target id、元数据。"""
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    X = np.zeros((n_samples, L, VOCAB))
    Xp = np.zeros((n_samples, L, VOCAB))
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        keys = r.choice(V_KEY, size=n_pairs, replace=False)
        vals = r.choice(V_VAL, size=n_pairs, replace=False) + V_KEY
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
    return X, Xp, Y, dict(L=L, n_pairs=n_pairs, shuffle=shuffle_pairs, seed=seed)


def trivial_baselines(Y, n_pairs):
    return dict(chance=1.0 / V_VAL, last_value=1.0 / n_pairs)
