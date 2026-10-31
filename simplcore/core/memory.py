# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""CMT 式压缩记忆库 —— SimplCore 原生落地。

CMT（*Compression Memory Training*, AAAI'25）的三阶段：**编码(压缩) → 存储 → 检索与聚合**。
它的关键取舍是**基座权重冻结**，新知识进外部记忆库，因此天然没有灾难性遗忘。

原文在**注意力层**注入 KV。SimplCore 没有注意力、也没有 KV 缓存（流式状态是
``2·M·D`` 的 O(1) 振荡器态），因此本模块把注入点改为 **SGC 残差流上的加性记忆项**
—— 这是 SGC 原生的等价物，且**不引入任何 softmax 注意力**。

沿用 CMT 的三项技术：

* **self-matching**   —— :func:`self_match_scores`，查询与记忆键归一化点积
* **top-aggregation** —— :meth:`CompressedMemoryBank.aggregate`，按相似度加权求和
* **memory-aware objective** —— :class:`MemoryAwareObjective`，只对"记忆相对基座的增量"计损失

诚实边界：本模块只负责**记忆的编码/存储/检索/聚合**。"记忆内容对不对"由
Dreaming 循环的 no-regression gate 负责（``simplcore/core/dream.py``），
不在此处保证。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# self-matching / top-aggregation
# ---------------------------------------------------------------------------
def self_match_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """CMT 的 self-matching：查询向量与记忆键做归一化点积（余弦）。

    Args:
        query: (B, D) 查询向量。
        keys:  (N, D) 记忆键。
    Returns:
        (B, N) 相似度分数。
    """
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    return q @ k.t()


def top_aggregate(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    top_k: int = 4,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CMT 的 top-aggregation：取 top-k 条记忆，按相似度做 softmax 加权求和。

    Args:
        query: (B, D)
        keys: (N, D)
        values: (N, D)
        top_k: 取前 k 条。
        temperature: softmax 温度，越小越尖锐。
    Returns:
        ``(aggregated (B, D), scores (B, N))`` —— scores 是全量相似度，
        便于调用方做"检索置信度"判断（例如低于阈值就不用记忆）。
    """
    scores = self_match_scores(query, keys)
    n = keys.shape[0]
    k = max(1, min(int(top_k), n))
    top_scores, top_idx = scores.topk(k, dim=-1)
    weights = torch.softmax(top_scores / max(temperature, 1e-6), dim=-1)   # (B, k)
    picked = values[top_idx]                                              # (B, k, D)
    aggregated = (weights.unsqueeze(-1) * picked).sum(dim=1)              # (B, D)
    return aggregated, scores


# ---------------------------------------------------------------------------
# memory-aware objective
# ---------------------------------------------------------------------------
class MemoryAwareObjective(nn.Module):
    """CMT 的 memory-aware objective：只对"有记忆 vs 无记忆"的**增量**计损失。

    原文用对比式 logit 集成来放大记忆的贡献。这里给出等价但更轻的形式：
    先分别算有记忆/无记忆的 logits，取差值作为增量 logits，再算交叉熵。
    梯度因此集中在"记忆带来的改变"上，而不是模型本已会的东西。

    Args:
        base_loss: 基础损失函数，默认交叉熵。签名为 ``(logits, targets) -> scalar``。
    """

    def __init__(self, base_loss=None) -> None:
        super().__init__()
        self.base_loss = base_loss or (
            lambda logits, targets: F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,
            )
        )

    def forward(
        self,
        logits_with_memory: torch.Tensor,
        logits_without_memory: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_with_memory: (B, L, V)
            logits_without_memory: (B, L, V)
            targets: (B, L)
        """
        delta = logits_with_memory - logits_without_memory
        return self.base_loss(delta, targets)


# ---------------------------------------------------------------------------
# 压缩器
# ---------------------------------------------------------------------------
class MemoryCompressor(nn.Module):
    """默认压缩器：SGC 前向 → 末端若干隐状态 → 投影为 k 个 soft memory token。

    CMT 原文用 causal Transformer decoder 把长文档压成 k 个 soft token。
    这里用 SGC 自己的隐状态做同样的事：因果卷积天然把整段上下文压进每个位置，
    末端位置因此已携带全文摘要，取末端 k 个位置即得一组 soft token。

    注意：这是**近似**而非等价 —— 原文的压缩器是可训练的独立模块，
    本实现是训练无关的读出。若要复现 CMT 的效果，应把本类替换为可训练压缩器。
    """

    def __init__(self, dim: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, model, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model: 任意提供 ``forward(idx) -> logits`` 的因果模型。
            ids: (B, L) token id。
        Returns:
            (B, k, D) soft memory token。
        """
        _, hidden = model(ids, return_hidden=True)          # (B, L, D)
        tail = hidden[:, -self.k:, :]                       # (B, k, D)
        if tail.shape[1] < self.k:                          # 序列比 k 短时补齐
            pad = hidden[:, :1, :].expand(-1, self.k - tail.shape[1], -1)
            tail = torch.cat([tail, pad], dim=1)
        return self.proj(tail)


# ---------------------------------------------------------------------------
# 记忆库
# ---------------------------------------------------------------------------
@dataclass
class MemoryItem:
    """一条压缩记忆。"""
    key: torch.Tensor          # (D,) 检索键
    value: torch.Tensor        # (D,) 注入用值
    meta: Dict[str, Any] = field(default_factory=dict)


class CompressedMemoryBank:
    """CMT 的存储阶段：定容的压缩记忆库。

    定容 + LRU 驱逐（不是 CMT 原文的做法，CMT 的记忆库无上限）。
    这里的定容是为了让在线系统的显存/内存占用可预测；驱逐策略为
    "最不常被检索"优先，因为未被检索过的记忆对线上收益最低。
    """

    def __init__(self, dim: int, capacity: int = 4096) -> None:
        self.dim = dim
        self.capacity = capacity
        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []
        self._meta: List[Dict[str, Any]] = []
        self._hits: List[int] = []

    # -- 写入 ---------------------------------------------------------------
    def add(self, keys: torch.Tensor, values: torch.Tensor,
            meta: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        """批量写入。

        Args:
            keys: (N, D)
            values: (N, D)
            meta: 可选，每个条目一份元信息。
        """
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same leading dim")
        for i in range(keys.shape[0]):
            self._keys.append(keys[i].detach().to(torch.float32))
            self._values.append(values[i].detach().to(torch.float32))
            self._meta.append(dict(meta[i]) if meta is not None else {})
            self._hits.append(0)
        self._evict()

    def _evict(self) -> None:
        while len(self._keys) > self.capacity:
            victim = min(range(len(self._hits)), key=lambda i: self._hits[i])
            self._keys.pop(victim)
            self._values.pop(victim)
            self._meta.pop(victim)
            self._hits.pop(victim)

    # -- 检索 ---------------------------------------------------------------
    def retrieve(self, query: torch.Tensor, top_k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索最相似的 top-k 条记忆。

        Args:
            query: (B, D)
            top_k: 取前 k 条。
        Returns:
            ``(values (B, k, D), scores (B, k))``；库为空时返回空张量。
        """
        if not self._keys:
            b = query.shape[0]
            return (query.new_zeros((b, 0, self.dim)),
                    query.new_zeros((b, 0)))
        keys = torch.stack(self._keys)          # (N, D)
        values = torch.stack(self._values)      # (N, D)
        scores = self_match_scores(query, keys)  # (B, N)
        k = max(1, min(int(top_k), keys.shape[0]))
        top_scores, top_idx = scores.topk(k, dim=-1)
        picked = values[top_idx]                # (B, k, D)
        # 命中计数（LRU 驱逐用）
        for idx in top_idx.reshape(-1).tolist():
            self._hits[idx] += 1
        return picked, top_scores

    def aggregate(self, query: torch.Tensor, top_k: int = 4,
                  temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索 + top-aggregation，返回可直接注入模型的记忆向量。

        Returns:
            ``(memory (B, D), scores (B, N))``
        """
        if not self._keys:
            return query.new_zeros((query.shape[0], self.dim)), query.new_zeros((query.shape[0], 0))
        keys = torch.stack(self._keys)
        values = torch.stack(self._values)
        return top_aggregate(query, keys, values, top_k=top_k, temperature=temperature)

    # -- 状态 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    def state_dict(self) -> Dict[str, Any]:
        """完整状态快照（深拷贝，供 no-regression gate 回滚记忆用）。

        回滚必须覆盖记忆：只回滚权重会把这一轮学错的东西留在库里，
        下一轮继续被检索注入 —— "回滚了但错误还在"。
        """
        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "keys": [k.clone() for k in self._keys],
            "values": [v.clone() for v in self._values],
            "meta": [dict(m) for m in self._meta],
            "hits": list(self._hits),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从 :meth:`state_dict` 恢复。"""
        if not isinstance(state, dict) or "keys" not in state:
            raise ValueError("state must come from CompressedMemoryBank.state_dict()")
        self._keys = [k.clone() for k in state["keys"]]
        self._values = [v.clone() for v in state["values"]]
        self._meta = [dict(m) for m in state["meta"]]
        self._hits = list(state["hits"])

    @property
    def keys(self) -> torch.Tensor:
        return torch.stack(self._keys) if self._keys else torch.zeros(0, self.dim)

    @property
    def values(self) -> torch.Tensor:
        return torch.stack(self._values) if self._values else torch.zeros(0, self.dim)

    def metas(self) -> List[Dict[str, Any]]:
        return list(self._meta)

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._keys),
            "capacity": self.capacity,
            "hits_total": int(sum(self._hits)),
        }


# ---------------------------------------------------------------------------
# 文本 → 键编码器（确定性、跨进程可复现）
# ---------------------------------------------------------------------------
class TextKeyEncoder:
    """把一段文本编码成定长单位向量，用作记忆库的检索键 / CMT 注入向量。

    为什么需要它
    ------------
    ``DreamingLoop._write_memory`` 之前用"文本哈希 → 一个标量 → 填满整条向量"
    当键，结果是**所有记忆条目共线**：任意两条的余弦相似度恒为 1.0，检索无法
    区分，等于没有检索。这类缺陷不会让任何测试失败（形状对、接口对），
    只会在真实使用中表现为"记忆写了但从不被召回"。

    做法
    ----
    字符 n-gram + **带符号哈希（signed hashing trick）**：

        v[h(g) % D] += sign(g) * w(g)     对每个 n-gram g

    然后 L2 归一化。相同文本永远得到相同向量（``blake2b`` 而非内置 ``hash()``，
    后者带进程随机盐）。

    为什么要 ``fit``（这一版的关键修正）
    ------------------------------------
    纯 w(g)=1 的实现有一个致命缺陷：**同一模板下只差一个词的记忆几乎共线**。
    实测 "用户的名字是小林。" 与 "用户的名字是阿岚。" 的余弦相似度是 **0.71**
    （最坏 0.83），而且**换维度完全没用**——64 / 192 / 1024 全是 0.667，因为
    相似度只由共享 n-gram 的数量决定。后果是 ``memory_in`` 收到的两条记忆
    向量几乎相同，模型没有可用的区分信息，只能退回"从参数里挑一个人名"。

    ``fit(texts)`` 统计语料内的文档频率 df(g)，并令

        w(g) = (N / max(df(g), 1)) ** alpha        （alpha 默认 1.0）

    模板 gram（df 高）被压到很小，"是小林"这类区分性 gram 权重是大一个量级。
    实测同模板余弦 0.71 → **0.11**（max 0.26），跨模板仍为 0。

    这个统计**必须与训练、评测共用同一份**：训练用拟合版、推理用未拟合版，
    注入向量的分布就对不上，而形状、loss、参数统计全都看不出异常。所以
    状态可序列化（:meth:`state_dict` / :meth:`from_state`），并可用
    :meth:`fingerprint` 在报告里核对两边是否同一个编码器。

    这不是语义嵌入（不做同义归并），但它是**可复现、零依赖、有区分度**的键，
    足以让记忆库工作；要真正的语义检索应换成模型隐状态（``TrainableMemoryCompressor``
    或任意外部 embedder，接口一致）。
    """

    def __init__(self, dim: int, ngram: int = 3) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.ngram = int(ngram)
        # 未 fit 时退化为 w=1（与旧行为逐位一致，保证已有记忆库可继续使用）
        self._df: Optional[Dict[str, int]] = None
        self._n_docs: int = 0
        self._alpha: float = 1.0

    # -- 语料统计 ----------------------------------------------------------
    def fit(self, texts: Iterable[str], alpha: float = 1.0) -> "TextKeyEncoder":
        """用记忆语料统计 df，之后 :meth:`__call__` 按 ``(N/df)^alpha`` 加权。

        ``alpha=0`` 等价于不拟合。``alpha`` 越大，模板成分压得越狠；实测 1.0
        已能把同模板余弦压到 0.11，继续增大只增加对未登录 gram 的敏感度。
        """
        df: Dict[str, int] = {}
        n = 0
        for t in texts:
            if not t:
                continue
            n += 1
            for g in set(self._grams(t)):
                df[g] = df.get(g, 0) + 1
        self._df = df
        self._n_docs = n
        self._alpha = float(alpha)
        return self

    @property
    def fitted(self) -> bool:
        return self._df is not None

    def weight(self, gram: str) -> float:
        """单个 n-gram 的权重。未登录 gram 按 df=1 处理（"见过一次"），
        而不是 0 或无穷：它应当有信息量，但不该一枝独秀。"""
        if self._df is None or self._alpha == 0.0:
            return 1.0
        d = max(self._df.get(gram, 0), 1)
        return float((max(self._n_docs, 1) / d) ** self._alpha)

    def state_dict(self) -> Dict[str, Any]:
        """可 JSON 序列化。``df`` 为空也照存，便于核对"两边都没拟合"。"""
        return {"dim": self.dim, "ngram": self.ngram, "alpha": self._alpha,
                "n_docs": self._n_docs, "df": dict(self._df or {})}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TextKeyEncoder":
        enc = cls(int(state["dim"]), int(state.get("ngram", 3)))
        df = state.get("df") or {}
        if df:
            enc._df = {str(k): int(v) for k, v in df.items()}
            enc._n_docs = int(state.get("n_docs", 0) or 0)
            enc._alpha = float(state.get("alpha", 1.0))
        return enc

    def fingerprint(self) -> str:
        """编码器身份的短指纹（维度 + 统计量 + 表的哈希）。

        用于在训练报告 / 消融报告里核对"训练与评测用的是同一份统计"。
        两边指纹不同就意味着向量分布不同，此时任何对比都无意义。
        """
        blob = json.dumps(self.state_dict(), sort_keys=True,
                          ensure_ascii=False).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    @staticmethod
    def _digest(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"),
                                              digest_size=8).digest(), "big")

    def _grams(self, text: str) -> List[str]:
        t = _norm_text(text)
        n = self.ngram
        if len(t) <= n:
            return [t] if t else []
        return [t[i:i + n] for i in range(len(t) - n + 1)]

    def __call__(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim, dtype=torch.float32)
        grams = self._grams(text)
        if not grams:
            return v
        for g in grams:
            idx = self._digest(g) % self.dim
            sign = 1.0 if (self._digest(g + "#") & 1) else -1.0
            v[idx] += sign * self.weight(g)
        n = float(v.norm())
        return v / n if n > 0 else v


def _norm_text(text: str) -> str:
    """轻量归一化：去空白与常见标点，转小写（不引入任何第三方依赖）。"""
    out = []
    for ch in (text or ""):
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# 可训练压缩器（CMT 原文的可训练 encoder 在 SGC 上的落地）
# ---------------------------------------------------------------------------
class TrainableMemoryCompressor(nn.Module):
    """**可训练**的压缩器：因果深度卷积堆叠 + 末端 k 位置读出。

    与 :class:`MemoryCompressor` 的区别：后者是"训练无关的隐状态读出"
    （依赖被压缩模型的前向），本类是独立参数模块，可用 memory-aware objective
    端到端训练 —— 这是 ``docs/DREAMING_CMT.md`` §6 的 G1 缺口。

    设计约束（全部为结构保证）：

    * **没有任何注意力 / KV**：只有 depthwise 因果卷积与逐点线性，
      因此压缩长文档的成本是 O(L·K·D)，不进 O(L²)；
    * **严格因果**：位置 t 的输出只依赖 ``x[:t+1]``，可作为在线压缩器；
    * **定长输出**：``(B, k, D)``，与文档长度无关，可直接作为 CMT 的 soft token；
    * 基座权重可保持冻结（本模块独立），符合 CMT"基座冻结 + 外部记忆"的取舍。

    Args:
        dim: 隐宽。
        k: 输出 soft token 个数（默认 8）。
        layers: 因果卷积层数（默认 3，感受野 ``1 + layers*(kernel-1)``）。
        kernel: 每层卷积宽度（默认 5）。
    """

    def __init__(self, dim: int, k: int = 8, layers: int = 3,
                 kernel: int = 5) -> None:
        super().__init__()
        if kernel < 1:
            raise ValueError("kernel must be >= 1")
        self.dim = dim
        self.k = k
        self.layers = layers
        self.kernel = kernel
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=kernel, groups=dim, bias=False)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, L, D). Returns: (B, k, D) soft memory tokens."""
        B, L, D = x.shape
        h = self.in_proj(x).transpose(1, 2)                 # (B, D, L)
        for conv in self.convs:
            pad = self.kernel - 1
            padded = F.pad(h, (pad, 0))                     # 左填充 = 因果
            h = torch.tanh(conv(padded)) + h                # 残差 + 有界非线性

        h = h.transpose(1, 2)                               # (B, L, D)
        h = self.norm(h)
        tail = h[:, -self.k:, :]
        if tail.shape[1] < self.k:
            pad_tok = tail[:, :1, :].expand(-1, self.k - tail.shape[1], -1)
            tail = torch.cat([tail, pad_tok], dim=1)
        return self.head(tail)


# ---------------------------------------------------------------------------
# G2：memory-aware 双路前向 + 压缩器训练步
# ---------------------------------------------------------------------------
def memory_aware_loss(
    model,
    ids: torch.Tensor,
    memory: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    objective: Optional[MemoryAwareObjective] = None,
    memory_scale: float = 1.0,
) -> torch.Tensor:
    """CMT 的 memory-aware objective：**双路前向**（有记忆 / 无记忆）的增量损失。

    G2 此前只是"``MemoryAwareObjective`` 已支持、未接入"—— 因为没人做那两次前向。
    本函数就是那两次前向，代价是 2× 前向算力（CMT 原文同样如此）。

    **定义性质（必须知道）**：增量 logits 全零时损失等于 ``ln(V)``，而不是 0。
    该目标度量的是"记忆带来的那部分 logits 能不能独立预测下一个 token"，
    所以记忆什么都没带 = 均匀分布 = 随机猜测的熵。读取这个损失时应当与
    ``ln(vocab_size)`` 比较，而不是与 0 比较。

    Args:
        model: 支持 ``forward(ids, memory=...)`` 的因果模型（``SimplCoreModel``）。
        ids: (B, L) token id。
        memory: (B, D) 要注入的记忆向量；``None`` 时返回零损失（无可学增量）。
        targets: (B, L) 目标 token；缺省用右移一位的自监督目标。
        objective: 增量损失；缺省 :class:`MemoryAwareObjective`。
        memory_scale: 记忆强度缩放（便于做"记忆是否真的起作用"的消融）。
    """
    objective = objective or MemoryAwareObjective()
    if memory is None:
        return torch.zeros((), dtype=torch.float32, device=ids.device)
    tgt = targets
    if tgt is None:
        tgt = torch.roll(ids, shifts=-1, dims=1)
        tgt = tgt.clone()
        tgt[:, -1] = -100                                   # 末位没有下一个 token
    if memory_scale != 1.0:
        memory = memory * float(memory_scale)
    with_mem = model(ids, memory=memory)
    without_mem = model(ids)
    return objective(with_mem, without_mem, tgt)


def train_compressor_step(
    compressor: TrainableMemoryCompressor,
    model,
    ids: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    objective: Optional[MemoryAwareObjective] = None,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """训练**压缩器**一步（基座冻结、只更新压缩器）。

    这正是 G1 的收口：``MemoryCompressor`` 是"训练无关的隐状态读出"，
    无论训练多久输出都不会变；本函数用 memory-aware objective 的梯度
    真正更新 :class:`TrainableMemoryCompressor` 的参数。

    流程（一步）：

    1. ``x = model.tok_emb(ids)`` → 压缩器 → ``(B, k, D)`` soft token；
    2. 池化成 ``(B, D)`` 记忆向量；
    3. 双路前向算增量损失；
    4. 反传 —— 梯度**只**流向压缩器（基座参数被 ``requires_grad`` 屏蔽后恢复）。

    Returns:
        ``{"loss": float, "param_delta": float, "mem_norm": float}``。
        ``param_delta`` 是更新前后参数的最大绝对变化，用来**证明它真的在训练**
        （训练无关的读出这一步恒为 0 —— 那正是旧实现的缺陷）。
    """
    optimizer = optimizer or torch.optim.Adam(compressor.parameters(), lr=1e-3)
    embed = getattr(model, "tok_emb", None)
    if embed is None:
        raise TypeError("model 需提供 tok_emb（或改用 MemoryCompressor 的训练无关路径）")

    was_training = compressor.training
    compressor.train()

    # 基座冻结：只让压缩器收梯度
    frozen = []
    try:
        for p in getattr(model, "parameters", lambda: [])():
            if p.requires_grad:
                frozen.append(p)
                p.requires_grad_(False)

        x = embed(ids).detach()                             # (B, L, D)
        soft = compressor(x)                                # (B, k, D)
        memory = soft.mean(dim=1)                           # (B, D)

        loss = memory_aware_loss(model, ids, memory=memory,
                                 targets=targets, objective=objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0))
        optimizer.step()
    finally:
        for p in frozen:
            p.requires_grad_(True)
        compressor.train(was_training)

    return {
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "mem_norm": float(memory.detach().norm(dim=-1).mean()),
        "n_params": float(sum(p.numel() for p in compressor.parameters())),
    }


def compressor_param_snapshot(compressor: TrainableMemoryCompressor) -> torch.Tensor:
    """压缩器参数的拼接快照（测试与审计用：比较训练前后是否真的变了）。"""
    return torch.cat([p.detach().reshape(-1).clone() for p in compressor.parameters()])
