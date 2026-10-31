# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Long-range capability probes (SelectiveCopy / AssociativeRecall /
IntervalRecall), the small ``ProbeNet`` they are trained on, and the
``train_probe`` / ``evaluate_probe`` / ``probe_*`` entry points -- split out of
``eval.py``. All probes are constructively solvable: every report carries the
chance level and a fresh-test-set accuracy next to a demonstrated
solvability bar (a small ProbeNet trained on the exact task)."""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import (EvalBatch, SyntheticDataset, _jsonable, _next_token_targets,
                     resolve_device, resolve_dtype)
from .memprobe import mb
from ..model.model import SimplCoreModel


# ---------------------------------------------------------------------------
# Probe tasks (long-range capability probes)
# ---------------------------------------------------------------------------
def _finish(inputs: torch.Tensor, targets: torch.Tensor, device: Any, vocab_size: int):
    """Cast to the token dtype, move to the device, and assert every id is in range.

    The bounds assertion is not decoration: an out-of-range id silently produces a
    garbage embedding lookup or an autograd index error deep inside the model, and
    the earlier `inputs.to(device).long()` ordering bug meant a whole probe run
    could be scored against reinterpreted memory.  Failing loudly here is cheap.
    """
    device = torch.device(device)
    inputs = inputs.long().to(device)
    targets = targets.long().to(device)
    if bool(inputs.min() < 0 or inputs.max() >= vocab_size):
        raise AssertionError(
            f"probe batch has ids outside [0, {vocab_size}): min={int(inputs.min())} "
            f"max={int(inputs.max())}")
    return inputs, targets


class ProbeTask:
    """A synthetic long-range task with a declared answer position and chance level.

    Subclasses implement :meth:`sample` returning ``(inputs, targets)`` int64
    tensors of shape ``(B, task.seq_len)`` where every position that must not
    contribute to the loss carries the ignore index ``-100``.

    Attributes:
        name: task identifier.
        vocab_size: size of the token alphabet the probe model uses.
        seq_len: fixed sequence length.
        answer_pos: index (or tuple of indices) at which the answer is scored.
        chance: chance-level accuracy (``1 / n_choices``).
        uses_shift: ``True`` for plain next-token datasets (``answer_pos is None``).
    """

    name: str = "probe"
    vocab_size: int = 64
    seq_len: int = 64
    answer_pos: Optional[Any] = None

    @property
    def chance(self) -> float:
        raise NotImplementedError

    def sample(self, batch_size: int, generator: torch.Generator,
               device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        ap = self.answer_pos
        return {
            "name": self.name,
            "vocab_size": int(self.vocab_size),
            "seq_len": int(self.seq_len),
            "answer_pos": int(ap) if isinstance(ap, int) else (
                [int(a) for a in ap] if ap is not None else None),
            "chance": float(self.chance),
        }


class SelectiveCopyTask(ProbeTask):
    """Selective copy: copy the token that follows a marker, on demand.

    Layout::

        [ random tokens ... MARKER v ... random tokens ... QUERY  <-- answer = v ]

    The answer is one content token out of ``n_content``, so ``chance = 1 /
    n_content``.  This is the standard selective-copy / induction-head task: it
    requires a content-based (not position-based) lookup over a long range.
    """

    name = "selective-copy"

    def __init__(self, n_content: int = 16, seq_len: int = 64, gap: int = 8,
                 marker: Optional[int] = None, query: Optional[int] = None,
                 pad: int = 0, answer_pos: Optional[int] = None) -> None:
        if n_content < 2:
            raise ValueError("n_content must be >= 2")
        self.n_content = int(n_content)
        # marker/query/pad are ids *above* the content range, so no answer slot or
        # padding position can be confused with a legal content token.
        self.marker = int(marker) if marker is not None else int(n_content)
        self.query = int(query) if query is not None else int(n_content) + 1
        self.pad = int(pad) if pad > 0 else int(n_content) + 2
        self.vocab_size = max(self.marker, self.query, self.pad) + 1
        self.gap = int(gap)
        min_len = self.gap + 5
        self.seq_len = int(max(seq_len, min_len))
        if answer_pos is None:
            answer_pos = self.seq_len - 1          # the slot right after the query token
        if not (2 < int(answer_pos) < self.seq_len):
            raise ValueError("answer_pos must leave room for the query and the answer slot")
        self.answer_pos = int(answer_pos)

    @property
    def chance(self) -> float:
        return 1.0 / float(self.n_content)

    def sample(self, batch_size, generator, device="cpu"):
        B, T = int(batch_size), self.seq_len
        inputs = torch.randint(0, self.n_content, (B, T), generator=generator)
        # marker + value must sit before the query and be separated by G tokens
        hi = max(1, self.answer_pos - self.gap - 2)
        pos = torch.randint(1, hi + 1, (B,), generator=generator)
        val = torch.randint(0, self.n_content, (B,), generator=generator)
        rows = torch.arange(B, device=inputs.device)
        inputs[rows, pos] = self.marker
        inputs[rows, pos + 1] = val
        inputs[rows, self.answer_pos - 1] = self.query
        inputs[rows, self.answer_pos] = self.pad
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, self.answer_pos] = val
        # sanity: marker/query ids are outside the content alphabet and each occurs
        # exactly once per row, so every sequence has a unique, unambiguous answer
        assert self.marker >= self.n_content and self.query >= self.n_content
        assert int((inputs == self.marker).sum()) == B, "marker aliasing with content ids"
        assert int((inputs == self.query).sum()) == B, "query aliasing with content ids"
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"n_content": self.n_content, "marker": self.marker,
                  "query": self.query, "gap": self.gap})
        return d


class AssociativeRecallTask(ProbeTask):
    """Multi-query associative recall (MQAR).

    Layout::

        k0 v0 k1 v1 ... k_{n-1} v_{n-1}  QUERY q0 q1 ... q_{m-1}

    Layout::

        [pad, k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, a_0, ..., q_{m-1}, a_{m-1}]

    where each ``a_j`` is the answer slot of query ``q_j`` and holds the value
    bound to that key.  Keys and values live in disjoint id ranges, so the task is
    a pure key -> value lookup and ``chance = 1 / vocab_value``.  Queries are
    sampled with replacement from all ``num_kv`` keys, so a key may repeat.
    """

    name = "associative-recall"

    def __init__(self, num_kv: int = 8, num_queries: int = 4, vocab_key: int = 32,
                 vocab_value: int = 32, pad: int = 0) -> None:
        if num_kv < 1 or num_queries < 1:
            raise ValueError("num_kv and num_queries must be >= 1")
        if vocab_key < num_kv:
            raise ValueError("vocab_key must be >= num_kv")
        self.num_kv = int(num_kv)
        self.num_queries = int(num_queries)
        self.vocab_key = int(vocab_key)
        self.vocab_value = int(vocab_value)
        # pad id sits above both id ranges so it is never confused with a value
        self.key_base = 0
        self.value_base = self.vocab_key
        self.pad = int(pad) if pad > 0 else self.value_base + self.vocab_value
        self.vocab_size = max(self.pad, self.value_base + self.vocab_value) + 1
        # layout: [pad, k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, a_0, ..., q_{m-1}, a_{m-1}]
        # where a_j is the answer slot of query q_j and is always < seq_len.
        self.seq_len = 1 + 2 * self.num_kv + 2 * self.num_queries
        self.query_pos = tuple(1 + 2 * self.num_kv + 2 * j for j in range(self.num_queries))
        self.answer_pos = tuple(p + 1 for p in self.query_pos)
        assert self.answer_pos[-1] == self.seq_len - 1

    @property
    def chance(self) -> float:
        return 1.0 / float(self.vocab_value)

    def sample(self, batch_size, generator, device="cpu"):
        B = int(batch_size)
        T = self.seq_len
        # zeros, not empty: every slot is written below except the pad column, and
        # `torch.empty` recycles freed float storage, which is how uninitialised
        # int64 garbage (e.g. 4.16e18) leaked into the probe inputs.
        inputs = torch.zeros((B, T), dtype=torch.long)
        inputs[:, 0] = self.pad
        kv = self.num_kv
        # sample distinct keys per row, then values (values may repeat)
        keys = torch.empty((B, kv), dtype=torch.long)
        for b in range(B):
            keys[b] = torch.randperm(self.vocab_key, generator=generator)[:kv]
        keys = keys + self.key_base
        assert self.key_base == 0, "key ids must start at 0 so they cannot collide with pad"
        vals = torch.randint(0, self.vocab_value, (B, kv), generator=generator) + self.value_base
        inputs[:, 1:1 + 2 * kv:2] = keys
        inputs[:, 2:2 + 2 * kv:2] = vals
        qi = torch.randint(0, kv, (B, self.num_queries), generator=generator)
        qpos = torch.as_tensor(self.query_pos, dtype=torch.long)
        apos = torch.as_tensor(self.answer_pos, dtype=torch.long)
        inputs[:, qpos] = keys.gather(1, qi)
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, apos] = vals.gather(1, qi)
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"num_kv": self.num_kv, "num_queries": self.num_queries,
                  "vocab_key": self.vocab_key, "vocab_value": self.vocab_value,
                  "key_base": self.key_base, "value_base": self.value_base})
        return d


class IntervalRecallTask(ProbeTask):
    """Fixed query key, one value, then G distractors, then the query again.

    Layout::

        [pad, KEY, v, d_0 ... d_{G-1}, KEY, <-- label = v]

    The second occurrence of the single query key is at ``seq_len - 2``; the
    logits there are scored against the value.  The gap between the value and the
    repeated query is exactly ``G`` distractor tokens, so ``G`` is the only
    difficulty knob (``b = v - 1`` has distance ``G + 2``).

    The final (second) occurrence of the single query key is at the last
    position; the answer slot is the next one, which is why the tokenizer-free
    probe reserves one extra slot at the end.  The gap between the value and the
    query is ``G`` distractors, so ``G`` is the only difficulty knob.
    ``chance = 1 / n_values``.

    Design note (why the earlier G=64 / 300-step attempt returned pure chance):
    the answer slot is a *single* position whose target is one of ``n_values``
    classes.  With plain cross-entropy over a large vocabulary the gradient for
    any individual value token is ~1/n_values of the total, so 300 steps of a
    small model on a rare class is far too little signal -- the run stops at the
    unigram prior, which is exactly chance.  The fixes applied here are
    (i) a smaller value alphabet, (ii) more steps, and (iii) an
    answer-weighted squared-error objective (``loss="mse"``) that keeps a usable
    gradient on the correct class instead of being diluted by the softmax
    normaliser.  See ``--loss`` on the ``probe`` subcommand.
    """

    name = "interval-recall"

    def __init__(self, distractor_gap: int = 16, n_values: int = 8,
                 query_key: Optional[int] = None, pad: int = 0,
                 n_content: int = 4) -> None:
        if n_values < 2:
            raise ValueError("n_values must be >= 2")
        if distractor_gap < 0:
            raise ValueError("distractor_gap must be >= 0")
        if n_content < 1:
            raise ValueError("n_content must be >= 1 (distractors are random content)")
        self.G = int(distractor_gap)
        self.n_values = int(n_values)
        self.n_content = int(n_content)
        # Disjoint id ranges; pad is above everything so it is never a legal value.
        #   values     0        .. n_values - 1
        #   content    n_values .. n_values + n_content - 1   (distractors)
        #   key        n_values + n_content                   (the single query key)
        #   pad        n_values + n_content + 1
        self.value_base = 0
        self.content_base = self.n_values
        self.query_key = (int(query_key) if query_key is not None
                          else self.content_base + self.n_content)
        self.pad = int(pad) if pad > 0 else self.query_key + 1
        self.vocab_size = max(self.pad, self.query_key) + 1
        # layout: [content, KEY, value, d_0 ... d_{G-1}, KEY, <answer slot>]
        self.seq_len = self.G + 5
        # the second query key sits at seq_len - 2; the value is scored at seq_len - 1
        self.answer_pos = self.seq_len - 1
        assert self.answer_pos == self.G + 4

    @property
    def chance(self) -> float:
        return 1.0 / float(self.n_values)

    def sample(self, batch_size, generator, device="cpu"):
        B, T = int(batch_size), self.seq_len
        inputs = torch.full((B, T), self.pad, dtype=torch.long)
        vals = torch.randint(0, self.n_values, (B,), generator=generator)
        inputs[:, 1] = self.query_key
        inputs[:, 2] = vals + self.value_base
        # Distractors are *random content*, not repeated pad: with pad-only gaps a
        # model can solve the task by copying from a fixed offset, which tests
        # positional indexing rather than long-range key retrieval.  The content
        # alphabet (id 0 .. query_key-1) is disjoint from the query key and from
        # every value id, so a distractor can never be mistaken for the answer.
        if self.G > 0:
            inputs[:, 3:3 + self.G] = torch.randint(
                0, self.n_content, (B, self.G), generator=generator) + self.content_base
        inputs[:, 3 + self.G] = self.query_key
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, self.answer_pos] = vals + self.value_base
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"G": self.G, "n_values": self.n_values, "n_content": self.n_content,
                  "query_key": self.query_key, "value_base": self.value_base,
                  "content_base": self.content_base})
        return d


# ---------------------------------------------------------------------------
# Small, explicitly causal probe model
# ---------------------------------------------------------------------------
class _CausalDWConv(nn.Module):
    """Depthwise causal 1-D convolution (left-padded), O(L*K)."""

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.conv.weight.shape[-1]
        u = F.pad(x.transpose(1, 2), (k - 1, 0))
        return self.conv(u).transpose(1, 2)



class _ProbeMixer(nn.Module):
    """The probe's mixer: content gating + causal depthwise convolution path.

    This is a *clean-room* reimplementation for the probe models.  It is not the
    audited ``SpectralGlobalConv`` mixer: it is a plain causal depthwise conv
    (left-padded, so strictly lower-triangular) with an input-dependent content
    gate -- the same gating idea SimplCore uses, without the spectral global
    kernel.  The module contains no randomness (no dropout, fixed identities), so
    it is causal and deterministic -- both properties are asserted by
    ``tests/test_eval.py``.  Keeping the probe attention-free means the only
    difference vs. a real SimplCore mixer is the *global* reach, so a probe that
    matches SimplCore on a long-range task isolates the mixer as the variable.
    """

    def __init__(self, dim: int, global_kind: str = "conv", content_gate: bool = True,
                 num_heads: int = 4, local_kernel: int = 4) -> None:
        super().__init__()
        self.content_gate = content_gate
        self.kind = global_kind
        self.norm = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        # 探测用的全局路径一律是因果深度卷积 —— 不含任何 softmax 自注意力。
        self.global_mix = _CausalDWConv(dim, max(local_kernel, 3))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))
        return x + self.out_proj(self.global_mix(u))


class _ProbeBlock(nn.Module):
    def __init__(self, dim: int, global_kind: str, content_gate: bool,
                 num_heads: int, local_kernel: int, ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.mixer = _ProbeMixer(dim, global_kind, content_gate, num_heads, local_kernel)
        hidden = int(ffn_mult * dim)
        self.ffn_norm = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class ProbeNet(nn.Module):
    """Small deterministic causal sequence model used for probe solvability checks.

    Causal guarantees:

    * the input embedding is token-only (no learned positional table), so no
      absolute-position information can leak across positions;
    * the mixer path is a left-padded causal depthwise convolution with a content
      gate -- no softmax self-attention anywhere;
    * no dropout and no other stochastic layer.

    ``tok_emb``/``head`` mirror the ``SimplCoreModel`` attribute names so the exact
    same memory instrumentation runs on both.
    """

    def __init__(self, vocab_size: int, dim: int = 64, num_layers: int = 3,
                 global_kind: str = "conv", content_gate: bool = True,
                 num_heads: int = 4, local_kernel: int = 4,
                 ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.tok_emb = nn.Embedding(self.vocab_size, dim)
        self.blocks = nn.ModuleList([
            _ProbeBlock(dim, global_kind, content_gate, num_heads, local_kernel, ffn_mult)
            for _ in range(num_layers)])
        self.ln_f = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx.long())
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def _answer_logits(model: nn.Module, inputs: torch.Tensor, answer_pos: Any) -> torch.Tensor:
    """Logits used for scoring: ``(N, vocab)`` taken at ``answer_pos - 1``.

    `answer_pos` is uniformly the index of the **answer slot** (the label).  The
    autoregressive state that predicts it is the preceding position, so the logits
    come from ``answer_pos - 1``.
    """
    if isinstance(model, SimplCoreModel):
        # SimplCoreModel.forward takes (B, L) token ids (it adds the channel dim itself)
        logits = model(inputs)
    else:
        logits = model(inputs)
    if isinstance(answer_pos, int):
        return logits[:, answer_pos - 1, :]
    idx = torch.as_tensor([int(a) - 1 for a in answer_pos], device=logits.device,
                          dtype=torch.long)
    return logits.index_select(1, idx).reshape(-1, logits.size(-1))


def _answer_targets(targets: torch.Tensor, answer_pos: Any) -> torch.Tensor:
    """Targets scored by :func:`_answer_logits`: the token **at** ``answer_pos``."""
    if isinstance(answer_pos, int):
        return targets[:, answer_pos]
    idx = torch.as_tensor([int(a) for a in answer_pos], device=targets.device,
                          dtype=torch.long)
    return targets.index_select(1, idx).reshape(-1)


def _answer_loss(lg: torch.Tensor, tg: torch.Tensor, loss: str) -> torch.Tensor:
    """Probe objective restricted to unmasked scored positions.

    Positions whose target is ``-100`` are dropped *before* the objective is
    evaluated: ``F.one_hot`` rejects negative class indices, and counting them
    would dilute the gradient of the few positions that actually matter.
    """
    mask = tg != -100
    if int(mask.sum()) == 0:
        raise ValueError("no unmasked scored positions in this probe batch")
    sel_lg = lg[mask].float()
    sel_tg = tg[mask]
    if loss == "ce":
        return F.cross_entropy(sel_lg, sel_tg)
    if loss == "mse":
        return F.mse_loss(sel_lg, F.one_hot(sel_tg, sel_lg.size(-1)).float())
    raise ValueError(f"unknown probe loss {loss!r}; use 'mse' or 'ce'")


def evaluate_probe(model: nn.Module, task: ProbeTask, n_samples: int = 4096,
                   batch_size: int = 128, seed: int = 1234, device: Any = "cpu",
                   progress: bool = False) -> Dict[str, Any]:
    """Accuracy / cross-entropy of ``model`` on a **freshly sampled** test set.

    The returned ``probe_accuracy`` is the fraction of scored positions whose
    argmax equals the target; ``probe_loss`` is the per-position cross-entropy in
    nats; ``chance`` is the task's chance level (``1 / n_choices``), so
    ``lift_over_chance`` (the fraction of the gap from chance to perfect that the
    model actually closed) is directly interpretable.  ``progress`` is accepted
    for signature compatibility and unused.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    n_samples = int(n_samples)
    gen = torch.Generator().manual_seed(int(seed))
    flat_pos = task.answer_pos
    if flat_pos is None:
        raise ValueError("evaluate_probe needs a task with answer_pos (not a shift task)")
    n_pos = 1 if isinstance(flat_pos, int) else len(flat_pos)
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            b = min(batch_size, n_samples - start)
            inputs, targets = task.sample(b, gen, device=dev)
            lg = _answer_logits(model, inputs, flat_pos).float()
            tg = _answer_targets(targets, flat_pos)
            keep = tg != -100
            lg, tg = lg[keep], tg[keep]
            if tg.numel() == 0:
                continue
            loss_sum += float(F.cross_entropy(lg, tg, reduction="sum").item())
            correct += int((lg.argmax(-1) == tg).sum().item())
            total += int(tg.numel())
    acc = correct / max(total, 1)
    loss = loss_sum / max(total, 1)
    model.train()
    return {
        "probe_accuracy": acc,
        "probe_loss": loss,
        "probe_ppl": math.exp(min(loss, 700.0)),
        "chance": float(task.chance),
        "n_scored_positions": int(total),
        "n_sequences": int(n_samples),
        "n_positions_per_sequence": int(n_pos),
        "above_chance": bool(acc > task.chance * 1.5),
        "lift_over_chance": (acc - task.chance) / max(1e-12, 1.0 - task.chance),
        "test_set": "freshly sampled at evaluation time (generator seed %d)" % int(seed),
    }


def train_probe(task: ProbeTask, steps: int = 400, batch_size: int = 64, lr: float = 3e-3,
                dim: int = 64, num_layers: int = 3, num_heads: int = 4,
                global_kind: str = "conv", content_gate: bool = True,
                loss: str = "mse", eval_every: int = 50, eval_batches: int = 4,
                eval_batch_size: int = 64, seed: int = 0, device: Any = "cpu",
                log: bool = True, log_every: int = 0, warmup: int = 20,
                final_eval_samples: int = 1024, model_fn: Optional[Callable[[], nn.Module]] = None,
                progress: bool = True) -> Dict[str, Any]:
    """Train a small probe model on ``task`` and return its learning curve.

    Returns a dict with the task description, the probe-network config, the
    per-checkpoint ``curve`` (``step``, training loss, held-out accuracy and
    held-out cross-entropy), the final held-out accuracy on a fresh sample, and a
    ``solvable_at_this_budget`` flag comparing the best curve accuracy against the
    task's chance level.

    ``loss="mse"`` optimises the squared error at the scored positions.  For the
    copy probes the scored target is one of many classes, so this keeps a usable
    gradient on the correct class instead of the cross-entropy softmax
    normaliser diluting it (see :class:`IntervalRecallTask`).
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    if model_fn is None:
        model = ProbeNet(task.vocab_size, dim=dim, num_layers=num_layers,
                         global_kind=global_kind, content_gate=content_gate,
                         num_heads=num_heads)
    else:
        model = model_fn()
    model = model.to(dev).train()
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) *
        (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
    gen = torch.Generator().manual_seed(int(seed) + 991)
    curve: List[Dict[str, Any]] = []
    loss_name = loss.lower()
    log_every = int(log_every) if log_every > 0 else max(10, steps // 10)
    flat_pos = task.answer_pos
    assert flat_pos is not None, "train_probe needs a task with a scored position"

    def _checkpoint(step: int, train_loss: float) -> None:
        accs, ces = [], []
        with torch.no_grad():
            for j in range(eval_batches):
                inputs, targets = task.sample(eval_batch_size, gen, device=dev)
                lg = _answer_logits(model, inputs, flat_pos).float()
                tg = _answer_targets(targets, flat_pos)
                keep = tg != -100
                lg, tg = lg[keep], tg[keep]
                if tg.numel() == 0:
                    continue
                ces.append(float(F.cross_entropy(lg, tg, reduction="mean").item()))
                accs.append(float((lg.argmax(-1) == tg).float().mean().item()))
        curve.append({
            "step": int(step),
            "train_loss": float(train_loss),
            "eval_loss": statistics.fmean(ces),
            "eval_accuracy": statistics.fmean(accs),
        })

    t0 = time.perf_counter()
    step = 0
    running = float("nan")
    iterator = range(steps)
    if progress and steps > 200:
        try:  # pragma: no cover - cosmetic only
            from tqdm import tqdm
            iterator = tqdm(range(steps), desc=f"probe:{task.name}", unit="step")
        except Exception:
            iterator = range(steps)
    for step in iterator:
        inputs, targets = task.sample(batch_size, gen, device=dev)
        lg = _answer_logits(model, inputs, flat_pos)
        tg = _answer_targets(targets, flat_pos)
        l = _answer_loss(lg, tg, loss_name)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = float(l.detach().cpu())
        if step % log_every == 0:
            _checkpoint(step, running)
            if log:
                print(f"  step {step:5d}  train {running:.4f}  "
                      f"eval_acc {curve[-1]['eval_accuracy']:.4f}  "
                      f"eval_ce {curve[-1]['eval_loss']:.4f}", flush=True)
    train_seconds = time.perf_counter() - t0
    _checkpoint(steps, running)
    # the *same* held-out seed is used by compare_mixers, so the probe-network
    # accuracy reported next to each mixer is comparable across mixers
    final = evaluate_probe(model, task, n_samples=final_eval_samples, batch_size=eval_batch_size,
                           seed=int(seed) + 12345, device=dev)
    best = max(curve, key=lambda c: c["eval_accuracy"])
    return {
        "task": task.describe(),
        "probe_network": {
            "kind": type(model).__name__,
            "global_kind": global_kind,
            "content_gate": bool(content_gate),
            "dim": int(dim),
            "num_layers": int(num_layers),
            "n_params": n_params,
        },
        "training": {
            "steps": int(steps), "batch_size": int(batch_size), "lr": float(lr),
            "loss": loss_name, "seed": int(seed), "device": str(dev),
            "seconds": train_seconds,
        },
        "curve": curve,
        "final": final,
        "final_accuracy": final["probe_accuracy"],
        "final_loss": final["probe_loss"],
        "chance": float(task.chance),
        "best_eval_accuracy": best["eval_accuracy"],
        "best_step": best["step"],
        # bar: must beat 1.5x chance AND exceed it by >= 0.05 absolute, otherwise
        # a lucky small-n run would be called "solvable".
        "solvable_at_this_budget": bool(best["eval_accuracy"] >= max(1.5 * task.chance,
                                                                     task.chance + 0.05)),
    }


# ---------------------------------------------------------------------------
# Ready-made probe suites and honest reporting
# ---------------------------------------------------------------------------
def _make_probe_tasks(name: str, n_content: int = 16, n_values: int = 8,
                      num_kv: int = 8, num_queries: int = 4, gaps: Sequence[int] = (8, 16, 32),
                      max_gap: int = 64) -> List[ProbeTask]:
    if name in ("selective-copy", "selective_copy", "copy"):
        return [SelectiveCopyTask(n_content=n_content)]
    if name in ("associative-recall", "associative_recall", "mqar"):
        return [AssociativeRecallTask(num_kv=num_kv, num_queries=num_queries,
                                      vocab_key=num_kv, vocab_value=num_kv)]
    if name in ("interval-recall", "interval_recall"):
        # the reported set is content-free (distractors are pad), so accuracy is
        # a property of the *trained model*, not of this particular draw; the
        # solvability check below uses the content-rich max_gap variant.
        test = IntervalRecallTask(distractor_gap=max_gap, n_values=n_values)
        tasks = [IntervalRecallTask(distractor_gap=int(g), n_values=n_values) for g in gaps]
        tasks.append(test)
        return tasks
    raise ValueError(f"unknown probe task {name!r}")


def _probe_report(task: ProbeTask, trained: Dict[str, Any], untrained: Optional[Dict[str, Any]],
                  untrained_model_params: Optional[int] = None) -> Dict[str, Any]:
    trained_ok = bool(trained["solvable_at_this_budget"])
    untrained_ok = bool(untrained["solvable_at_this_budget"]) if untrained else None
    out: Dict[str, Any] = {
        "probe": task,                       # ProbeTask instance; _jsonable() handles it
        "task": task.describe(),
        "chance": float(task.chance),
        "trained": trained,
        "untrained_baseline": untrained,
        "solvability_check": {
            "trained_probe_network_above_chance": trained_ok,
            "untrained_probe_network_above_chance": untrained_ok,
            "untrained_probe_network_n_params": untrained_model_params,
            "bar": "best checkpointed eval accuracy >= max(1.5 * chance, chance + 0.05)",
        },
        "verdict": "probe_network_demonstrably_learned_the_task" if trained_ok else
                   "NOT_PROVEN_at_this_budget_the_probe_network_did_not_beat_chance",
    }
    if "G" in task.describe():
        out["G"] = task.describe()["G"]       # flattened so sweeps are easy to tabulate
    if "num_kv" in task.describe():
        out["num_kv"] = task.describe()["num_kv"]
    return out


def probe_selective_copy(model: Optional[nn.Module] = None, *, train: bool = False,
                         steps: int = 400, n_content: int = 16, seq_len: int = 64,
                         gap: int = 8, batch_size: int = 64, lr: float = 3e-3,
                         dim: int = 64, num_layers: int = 3, seed: int = 0,
                         device: Any = "cpu", n_samples: int = 2048,
                         loss: str = "mse", verbose: bool = True,
                         untrained_steps: int = 20, **kw: Any) -> Dict[str, Any]:
    """Selective copy: copy the token that follows a marked token.

    Returns ``{"task", "chance", "model_accuracy"|None, "solvability_check",
    "verdict", ...}``.  When ``train=True`` (or ``model is None``) a small
    :class:`ProbeNet` is trained on the task and its learning curve is included,
    so the reported number always sits next to a solvability check.
    """
    task = SelectiveCopyTask(n_content=n_content, seq_len=seq_len, gap=gap)
    do_train = bool(train)
    trained = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                          num_layers=num_layers, seed=seed, device=device, loss=loss,
                          final_eval_samples=n_samples, log=verbose) if do_train else None
    untrained = train_probe(task, steps=untrained_steps, batch_size=batch_size, lr=lr, dim=dim,
                            num_layers=num_layers, seed=seed, device=device, loss=loss,
                            final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out = _probe_report(task, trained, untrained) if trained else {
        "task": task.describe(), "chance": float(task.chance),
        "solvability_check": None, "verdict": "no_solvability_check_requested",
    }
    if model is not None:
        out["model_accuracy"] = evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234, device=device)["probe_accuracy"]
    if verbose:
        print(f"[{task.name}] chance={task.chance:.4f}", flush=True)
        if trained:
            print(f"  trained probe-net final acc={trained['final_accuracy']:.4f} "
                  f"best(step {trained['best_step']})={trained['best_eval_accuracy']:.4f} "
                  f"solvable={trained['solvable_at_this_budget']}", flush=True)
        if "model_accuracy" in out:
            print(f"  supplied model acc={out['model_accuracy']:.4f}", flush=True)
    return out


def probe_associative_recall(model: Optional[nn.Module] = None, *, train: bool = False,
                             steps: int = 500, num_kv: int = 8, num_queries: int = 4,
                             vocab_key: Optional[int] = None, vocab_value: Optional[int] = None,
                             batch_size: int = 64, lr: float = 3e-3, dim: int = 64,
                             num_layers: int = 3, seed: int = 0, device: Any = "cpu",
                             n_samples: int = 2048, loss: str = "mse", verbose: bool = True,
                             untrained_steps: int = 20, **kw: Any) -> Dict[str, Any]:
    """Multi-query associative recall (MQAR).

    ``chance = 1 / vocab_value``; every query is scored, and queries are sampled
    with replacement so a repeated key in one sequence is consistent by
    construction.
    """
    task = AssociativeRecallTask(
        num_kv=num_kv, num_queries=num_queries,
        vocab_key=vocab_key if vocab_key is not None else num_kv,
        vocab_value=vocab_value if vocab_value is not None else num_kv)
    do_train = bool(train)
    trained = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                          num_layers=num_layers, seed=seed, device=device, loss=loss,
                          final_eval_samples=n_samples, log=verbose) if do_train else None
    untrained = train_probe(task, steps=untrained_steps, batch_size=batch_size, lr=lr, dim=dim,
                            num_layers=num_layers, seed=seed, device=device, loss=loss,
                            final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out = _probe_report(task, trained, untrained) if trained else {
        "task": task.describe(), "chance": float(task.chance),
        "solvability_check": None, "verdict": "no_solvability_check_requested",
    }
    if model is not None:
        out["model_accuracy"] = evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234, device=device)["probe_accuracy"]
    if verbose:
        print(f"[{task.name}] chance=1/{task.vocab_value}={task.chance:.4f}", flush=True)
        if trained:
            print(f"  trained probe-net final acc={trained['final_accuracy']:.4f} "
                  f"best(step {trained['best_step']})={trained['best_eval_accuracy']:.4f} "
                  f"solvable={trained['solvable_at_this_budget']}", flush=True)
        if "model_accuracy" in out:
            print(f"  supplied model acc={out['model_accuracy']:.4f}", flush=True)
    return out


def probe_interval_recall(model: Optional[nn.Module] = None, *, train: bool = False,
                          steps: int = 800, gaps: Sequence[int] = (8, 16, 32, 64),
                          n_values: int = 8, batch_size: int = 64, lr: float = 3e-3,
                          dim: int = 64, num_layers: int = 3, seed: int = 0,
                          device: Any = "cpu", n_samples: int = 2048, loss: str = "mse",
                          verbose: bool = True, untrained_steps: int = 20,
                          train_all_gaps: bool = False, **kw: Any) -> Dict[str, Any]:
    """Interval recall: key, value, ``G`` distractors, key again -> value.

    Accuracy is reported **as a function of G** together with ``chance = 1 /
    n_values``.  By default the solvability training run covers the hardest
    requested gap only and ``trained`` is ``None`` for the easier gaps; set
    ``train_all_gaps=True`` for one curve per gap (slower).  Any gap with
    ``trained: null`` was **not** checked for solvability.
    """
    tasks = [IntervalRecallTask(distractor_gap=int(g), n_values=n_values) for g in gaps]
    per_gap: List[Dict[str, Any]] = []
    trained_hardest: Optional[Dict[str, Any]] = None
    for i, task in enumerate(tasks):
        hardest = (i == len(tasks) - 1)
        train_here = bool(train) and (hardest or train_all_gaps)
        t = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                        num_layers=num_layers, seed=seed, device=device, loss=loss,
                        final_eval_samples=n_samples, log=bool(verbose and train_here)) \
            if train_here else None
        rec = {
            "G": task.G,
            "seq_len": task.seq_len,
            "chance": float(task.chance),
            "n_values": int(task.n_values),
            "trained": t,
            "model_accuracy": (evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234 + i, device=device)["probe_accuracy"]
                               if model is not None else None),
        }
        if t is not None:
            trained_hardest = t
        per_gap.append(rec)
    untrained = train_probe(tasks[-1], steps=untrained_steps, batch_size=batch_size, lr=lr,
                            dim=dim, num_layers=num_layers, seed=seed, device=device,
                            loss=loss, final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out: Dict[str, Any] = {
        "probe": "interval-recall",
        "by_gap": per_gap,
        "chance": float(tasks[-1].chance),
        "n_values": int(n_values),
        "solvability_check": {
            "hardest_gap": int(tasks[-1].G),
            "trained_probe_network_above_chance":
                bool(trained_hardest["solvable_at_this_budget"]) if trained_hardest else None,
            "untrained_probe_network_above_chance":
                bool(untrained["solvable_at_this_budget"]) if untrained else None,
            "bar": "best checkpointed eval accuracy >= max(1.5 * chance, chance + 0.05)",
        },
        "verdict": ("probe_network_demonstrably_learned_the_task"
                    if (trained_hardest and trained_hardest["solvable_at_this_budget"])
                    else "NOT_PROVEN_at_this_budget_the_probe_network_did_not_beat_chance"),
    }
    if verbose:
        print(f"[interval-recall] chance=1/{n_values}={tasks[-1].chance:.4f}", flush=True)
        for rec in per_gap:
            acc = rec["model_accuracy"]
            t = rec["trained"]
            tail = f"  trained_net_final={t['final_accuracy']:.4f}" if t else ""
            print(f"  G={rec['G']:4d} seq_len={rec['seq_len']:4d} "
                  f"model_acc={('%.4f' % acc) if acc is not None else 'n/a'}{tail}", flush=True)
    return out


def probe_tasks(name: str) -> List[ProbeTask]:
    return _make_probe_tasks(name)


def probe_shift_batches(task: ProbeTask, batch_size: int, gen: torch.Generator,
                        device: Any) -> EvalBatch:
    inputs, _ = task.sample(batch_size, gen, device=device)
    return EvalBatch(inputs, _next_token_targets(inputs))


def probe_eval_dataset(task: ProbeTask, n_samples: int = 512,
                       seed: int = 1234) -> SyntheticDataset:
    """``(seq_len,)`` LongTensor dataset for :func:`evaluate_lm` (shift objective).

    Useful as a probe-style language-modelling check on a synthetic distribution
    when no tokenizer corpus is available.
    """
    return SyntheticDataset(task.vocab_size, task.seq_len, length=n_samples, seed=seed)


