"""真实语料：字符级词表 + 训练/验证切分。

语料来源：github.com/kino-koki/SimplCore 真实包内 700 个
.md/.py/.txt/.json/.sh/.c/.h/.java 文件（570 万字符，含 36 万中文字符）。
与上游 docs/EDGE_TRAINING_REPORT.md 的「真实 repo 语料」设定一致。
"""
import numpy as np
import collections

V = 512
UNK = V - 1


def build_vocab(path="/data/workspace/train/corpus.txt"):
    text = open(path, encoding="utf-8").read()
    # 过滤 NUL / 控制字符，保留换行与制表
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    cnt = collections.Counter(text)
    chars = [ch for ch, _ in cnt.most_common(V - 1)]
    c2i = {ch: i for i, ch in enumerate(chars)}
    return text, c2i, chars


def encode(text, c2i):
    arr = np.fromiter((c2i.get(ch, UNK) for ch in text), dtype=np.int32,
                      count=len(text))
    return arr


def make_splits(arr, val_frac=0.05):
    n = len(arr)
    cut = int(n * (1 - val_frac))
    return arr[:cut], arr[cut:]


class Batcher:
    def __init__(self, arr, L, seed=0):
        self.arr, self.L, self.rng = arr, L, np.random.default_rng(seed)

    def batch(self, B):
        maxs = len(self.arr) - self.L - 1
        st = self.rng.integers(0, maxs, size=B)
        seq = np.stack([self.arr[s:s + self.L + 1] for s in st])
        return seq[:, :-1], seq[:, 1:]


def decode(ids, chars):
    return "".join(chars[i] if 0 <= i < len(chars) else "?" for i in ids)
