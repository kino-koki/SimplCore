"""
§5.5 X-axis-probe 白盒复现（v2：在线无限采样，避免过拟合）

关键修正：probe 任务必须**在线生成** batch（每步新样本）。固定小数据集会让
60k 参数的模型直接记住训练集，held-out 准确率反而低于 chance。
held-out 评估集对两个模型用**同一批**样本（同 seed），便于配对 bootstrap。

  pointing  精确指向   8-way  -> chance 0.125
  mqar      G=4 联想召回 8-way -> chance 0.125
  agg       聚合（多数类）二分类 -> chance 0.5
  count     精确计数（奇偶）二分类 -> chance 0.5
  markov    一阶马尔可夫 LM -> ppl

⚠ 规模声明：纯 numpy / 2 核 CPU 沙盒；序列长度、模型宽度、步数均小于论文。
   数字不可与论文直接比较，只能比对相对关系。
"""
import numpy as np
import nn as N


class Spec:
    def __init__(self, name, V, L, gen, chance, ncls, is_lm=False):
        self.name, self.V, self.L = name, V, L
        self.gen = gen                 # (n, rng) -> (X, Y|None)
        self.chance, self.ncls, self.is_lm = chance, ncls, is_lm


# ------------------------------------------------------------ 生成器
def _pointing(n, rng, G=4, Vc=8, L=65):
    """精确指向：G 个 (固定 label, content) 对散布序列中，末尾 query=某 label，
    目标=其后的 content。8-way, chance=0.125。
    vocab: 0..7=label, 8..15=content, 16=filler, 17=query
    """
    X = np.full((n, L), 16, dtype=np.int64)
    lab = rng.integers(0, 8, size=(n, G))
    val = 8 + rng.integers(0, Vc, size=(n, G))
    pos = np.sort(np.argsort(rng.random((n, L - 2)), axis=1)[:, :2 * G], axis=1)
    Y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for g in range(G):
            X[i, pos[i, 2 * g]] = lab[i, g]
            X[i, pos[i, 2 * g] + 1] = val[i, g]
        q = rng.integers(0, G)
        X[i, L - 2] = lab[i, q]
        X[i, L - 1] = 17
        Y[i] = val[i, q] - 8
    return X, Y


def _mqar(n, rng, G=4, Vk=64, Vv=8, L=65):
    """MQAR 联想召回：G 个 (key, value) 对，key 从 Vk=64 大词表随机采；
    末尾 query=某 key，目标=其 value。8-way, chance=0.125。
    vocab: 0..63=key, 64..71=value, 72=filler
    """
    V = Vk + Vv
    X = np.full((n, L), V, dtype=np.int64)
    keys = rng.integers(0, Vk, size=(n, G))
    vals = Vk + rng.integers(0, Vv, size=(n, G))
    pos = np.sort(np.argsort(rng.random((n, L - 1)), axis=1)[:, :2 * G], axis=1)
    Y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for g in range(G):
            X[i, pos[i, 2 * g]] = keys[i, g]
            X[i, pos[i, 2 * g] + 1] = vals[i, g]
        q = rng.integers(0, G)
        X[i, L - 1] = keys[i, q]
        Y[i] = vals[i, q] - Vk
    return X, Y


def _agg(n, rng, L=129):
    x = rng.integers(0, 2, size=(n, L))
    y = (x.sum(axis=1) > L / 2).astype(np.int64)
    return np.concatenate([x, np.full((n, 1), 2)], axis=1).astype(np.int64), y


def _count(n, rng, L=65):
    X = np.zeros((n, L), dtype=np.int64)
    k = rng.integers(1, 12, size=n)
    for i in range(n):
        X[i, rng.choice(L - 1, size=k[i], replace=False)] = 1
        X[i, L - 1] = 2
    return X, (k % 2).astype(np.int64)


def _markov_gen(T, V, L):
    def g(n, rng):
        X = np.zeros((n, L), dtype=np.int64)
        X[:, 0] = rng.integers(0, V, size=n)
        for i in range(n):
            for t in range(1, L):
                X[i, t] = rng.choice(V, p=T[X[i, t - 1]])
        return X.astype(np.int64), None
    return g


def _osc(n, rng, L=128, V=8, two=False, eps=0.15):
    """多尺度周期延续（物理振荡类归纳偏置测试台）

    隐藏相位钟：phase_t = (ph0 + w*t) mod p，观测 = 阶梯量化值 + eps 比例随机污染。
    目标 = **未被污染**的第 L 个值（输入只含 t=0..L-1，故无法靠复制作弊）。
    p 与 w 每条序列随机 -> 必须 in-context 推断周期。
    """
    X = np.zeros((n, L), dtype=np.int64)
    Y = np.zeros(n, dtype=np.int64)
    t = np.arange(L + 1)
    for i in range(n):
        if not two:
            p = int(rng.choice([7, 11, 13, 17, 19]))
            w = int(rng.choice([2, 3, 5]))
            ph0 = int(rng.integers(0, p))
            sig = (V * ((ph0 + w * t) % p)) / p
        else:
            p1 = int(rng.choice([7, 11, 13])); p2 = int(rng.choice([17, 19, 23]))
            s = rng.uniform(.7, 1.3) * np.sin(2 * np.pi * t / p1 + rng.uniform(0, 6.28)) \
                + rng.uniform(.7, 1.3) * np.sin(2 * np.pi * t / p2 + rng.uniform(0, 6.28))
            sig = (s + 2.6) / 5.2 * V
        q = np.clip(sig.astype(np.int64), 0, V - 1)
        X[i] = q[:L]
        if eps > 0:
            flip = rng.random(L) < eps
            X[i, flip] = rng.integers(0, V, size=int(flip.sum()))
        Y[i] = q[L]
    return X, Y


TASKS = {
    "osc":   lambda rng: Spec("osc", 8, 128, lambda n, r: _osc(n, r, eps=0.15), 0.125, 8),
    "osc0":  lambda rng: Spec("osc0", 8, 128, lambda n, r: _osc(n, r, eps=0.0), 0.125, 8),
    "osc05": lambda rng: Spec("osc05", 8, 128, lambda n, r: _osc(n, r, eps=0.05), 0.125, 8),
    "osc30": lambda rng: Spec("osc30", 8, 128, lambda n, r: _osc(n, r, eps=0.30), 0.125, 8),
    "osc2":  lambda rng: Spec("osc2", 8, 128, lambda n, r: _osc(n, r, two=True, eps=0.15), 0.125, 8),
    "pointing": lambda rng: Spec("pointing", 18, 65, _pointing, 0.125, 8),
    "mqar":     lambda rng: Spec("mqar", 73, 65, _mqar, 0.125, 8),
    "agg":      lambda rng: Spec("agg", 3, 130, _agg, 0.5, 2),
    "mqar129":   lambda rng: Spec("mqar129", 73, 129, lambda n, r: _mqar(n, r, L=129), 0.125, 8),
    "mqar257":   lambda rng: Spec("mqar257", 73, 257, lambda n, r: _mqar(n, r, L=257), 0.125, 8),
    "agg65":    lambda rng: Spec("agg65", 3, 66, lambda n, r: _agg(n, r, L=65), 0.5, 2),
    "count":    lambda rng: Spec("count", 3, 65, _count, 0.5, 2),
    "markov":   lambda rng: (lambda T: Spec("markov", 16, 65, _markov_gen(T, 16, 65),
                                            None, 16, is_lm=True))(
        rng.dirichlet(np.full(16, 0.5), size=16)),
}


# ------------------------------------------------------------ 训练 / 评估
def train_eval(model, spec, steps=800, bs=16, lr=3e-3, rng=None,
               n_eval=512, eval_every=100):
    rng = rng or np.random.default_rng(0)
    Xte, Yte = spec.gen(n_eval, np.random.default_rng(12345))   # 两模型共用
    opt = N.AdamW(model.params(), lr=lr, wd=0.0)
    hist = []
    for st in range(steps):
        xb, yb = spec.gen(bs, rng)
        lg = model.forward(xb)
        if spec.is_lm:
            loss = N.cross_entropy(N.flatten2d(N.slice_tail(lg, 1)),
                                   xb[:, 1:].ravel())
        else:
            loss = N.cross_entropy(N.last(lg), yb)
        loss.backward()
        opt.step()
        if (st + 1) % eval_every == 0 or st == 0:
            hist.append((st + 1, float(loss.v)))
    if spec.is_lm:
        nll, ntok = 0.0, 0
        for s in range(0, Xte.shape[0], bs):
            xb = Xte[s:s + bs]
            lg = model.forward(xb).v[:, :-1, :]
            z = lg - lg.max(axis=-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
            b, l = np.arange(xb.shape[0]), np.arange(xb.shape[1] - 1)
            nll += -lp[b[:, None], l[None, :], xb[:, 1:]].sum()
            ntok += xb.shape[0] * (xb.shape[1] - 1)
        return dict(ppl=float(np.exp(nll / ntok)), hist=hist)
    # 逐样本正确性（供配对 bootstrap）
    preds = []
    for s in range(0, Xte.shape[0], bs):
        xb = Xte[s:s + bs]
        pr = model.forward(xb).v[:, -1, :spec.ncls]
        preds.append(pr.argmax(axis=-1))
    preds = np.concatenate(preds)
    correct = (preds == Yte).astype(np.float64)
    return dict(acc=float(correct.mean()), hist=hist, correct=correct.tolist())


def bootstrap_ci(a, b, iters=4000, rng=None):
    """配对 bootstrap：返回 mean diff 与 95% CI（用于"噪声内"判定）。"""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    bs = d[rng.integers(0, d.size, size=(iters, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
