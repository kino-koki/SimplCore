# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""非线性思考 —— 在 O(1) 流式状态之上做 latent 空间的迭代式精化。

为什么是"latent 递归"而不是"生成更多 token"
--------------------------------------------
DeepReasoner（``reasoning.py``）的思考发生在**文本层**：每一步思考都要真的
生成 token、再流回状态。这有代价——文本解码不可微、每步都要过完整采样循环。

本模块把思考搬进 **latent 空间**：拿到残差流的隐状态后，不再立刻输出 logits，
而是先用一个可学习的**门控非线性递归单元**迭代精化 z：

    z_{i+1} = z_i + down( silu(gate(z_i)) * up(norm(z_i)) )

再用一个**停止头**（halting head，ACT 式）决定何时收手：

    p_i = sigmoid(halting(z_i))          # 第 i 步的停止概率
    在 p_i >= halt_threshold 或 i == max_iters 时停

最后 logits = head(ln_f(z_T))。与 DeepReasoner 的互补关系：

* 文本层思考（reasoning.py）：可插工具、可核可读，但不可微、每步贵；
* latent 层思考（本模块）：**端到端可微**（训练时梯度能穿过全部迭代步）、
  单步只是 D×D 的非线性单元、停止步数自适应——但思考内容人类不可读。

设计约束（与全架构一致）
------------------------
1. **不引入任何注意力**：递归单元只有逐点线性 + SiLU 门 + LayerNorm。
2. **O(1) 状态不变**：迭代发生在"当前隐状态"上，不触碰任何 mixer 的递推态，
   流式常驻字节数与 ``SimplCoreModel.stream_bytes`` 完全一致。
3. **可退化**：``max_iters=0``（或 ``enabled=False``）时是**逐位一致**的直通
   ——logits 与 ``SimplCoreModel.stream_step`` 完全相同（theory 守卫）。
4. **思考算力可记账**：:class:`ThinkLedger` 记录"最大迭代预算 vs 实际迭代数"，
   加速比口径与 ``accel.ComputeLedger`` 一致（结构性、不依赖硬件计时）。

诚实边界
--------
1. 停止头零初始化 → p=0.5：未训练时迭代步数≈ ``halt_threshold`` 附近的启发式，
   不是"学会的思考"。机制正确 ≠ 质量达标（与 theory.py 的边界一致）。
2. latent 精化是**有损/无保证**的：迭代可能把好的隐状态改坏。评估它与直通
   的质量差需要真实任务，本模块只保证机制、可微性、可记账与纯洁性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NonlinearThinkCell", "HaltingHead", "ThinkLedger",
           "NonlinearThinker"]


# ---------------------------------------------------------------------------
# 递归单元
# ---------------------------------------------------------------------------
class NonlinearThinkCell(nn.Module):
    """门控非线性残差单元：latent 迭代的一步。

    结构（预范数、SwiGLU 式门控、零初始化残差）::

        z' = z + down( silu(gate(n)) * up(n) ),   n = norm(z)

    ``down`` 零初始化 → 初始为恒等映射：思考步初始不改基座行为，
    训练后自学"怎么改"。与 ``memory_in`` 的零初始化同一纪律。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 norm: str = "layernorm", eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        hidden = int(hidden) if hidden is not None else dim
        from .sgc import make_norm
        self.norm = make_norm(norm, dim, eps=eps)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        with torch.no_grad():
            self.down.weight.zero_()          # 恒等初始化

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = self.norm(z)
        return z + self.down(F.silu(self.gate(n)) * self.up(n))


class HaltingHead(nn.Module):
    """ACT 式停止头：z -> 停止概率 p ∈ (0,1)。

    零初始化 → p=0.5（未训练时既不偏向早停也不偏向跑满）。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()            # sigmoid(0)=0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) -> p: (...,)"""
        return torch.sigmoid(self.proj(z).squeeze(-1))


# ---------------------------------------------------------------------------
# 算力账本
# ---------------------------------------------------------------------------
@dataclass
class ThinkLedger:
    """latent 思考的算力账本（与 ``accel.ComputeLedger`` 同口径）。

    ``iters_budget`` 是每个 token 的迭代上限之和，``iters_run`` 是实际
    迭代数之和；``speedup = budget / run >= 1``。
    """

    iters_budget: int = 0
    iters_run: int = 0
    tokens: int = 0
    ponder_cost: float = 0.0          # Σ (1 - p_i)：思考步的"未停止质量"

    def record(self, iters_run: int, max_iters: int,
               ponder: float = 0.0) -> None:
        self.iters_run += int(iters_run)
        self.iters_budget += int(max_iters)
        self.tokens += 1
        self.ponder_cost += float(ponder)

    @property
    def speedup(self) -> float:
        return self.iters_budget / max(self.iters_run, 1)

    @property
    def mean_iters(self) -> float:
        return self.iters_run / max(self.tokens, 1)

    def reset(self) -> None:
        self.iters_budget = 0
        self.iters_run = 0
        self.tokens = 0
        self.ponder_cost = 0.0

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "iters_budget": self.iters_budget,
            "iters_run": self.iters_run,
            "speedup": round(self.speedup, 4),
            "mean_iters": round(self.mean_iters, 4),
            "ponder_cost": round(self.ponder_cost, 6),
        }


# ---------------------------------------------------------------------------
# 思考器
# ---------------------------------------------------------------------------
class NonlinearThinker(nn.Module):
    """包裹一个 SimplCoreModel，在流式路径上做 latent 迭代思考。

    Args:
        model: ``SimplCoreModel``（或同接口：tok_emb / blocks / ln_f / head）。
        cell: 可选自定义 latent 递归单元；缺省 :class:`NonlinearThinkCell`。
        halting: 可选自定义停止头；缺省 :class:`HaltingHead`。
        max_iters: 每次 token 的最大迭代步数（含 0；0 = 纯直通）。
        halt_threshold: 停止概率阈值，p_i >= 阈值即收手。
        ponder_weight: 训练时 ponder 损失权重（Σ(1-p_i) × 权重，鼓励早停）。
        enabled: 总开关；False 时与直通逐位一致。
    """

    def __init__(self, model: nn.Module, cell: Optional[nn.Module] = None,
                 halting: Optional[nn.Module] = None, max_iters: int = 2,
                 halt_threshold: float = 0.5, ponder_weight: float = 0.01,
                 enabled: bool = True) -> None:
        super().__init__()
        self.model = model
        dim = model.config.dim
        self.cell = cell if cell is not None else NonlinearThinkCell(dim)
        self.halting = halting if halting is not None else HaltingHead(dim)
        if max_iters < 0:
            raise ValueError("max_iters must be >= 0")
        self.max_iters = int(max_iters)
        if not (0.0 < halt_threshold <= 1.0):
            raise ValueError("halt_threshold must be in (0, 1]")
        self.halt_threshold = float(halt_threshold)
        self.ponder_weight = float(ponder_weight)
        self.enabled = bool(enabled)
        self.ledger = ThinkLedger()

    # -- 消费一个 token（复刻 stream_step，但返回 ln_f 前的残差流隐状态）------
    def _consume(self, token: torch.Tensor, state):
        """与 ``SimplCoreModel.stream_step`` 同序的同构前向，返回 (hidden, state)。

        hidden 是 **ln_f 之前**的残差流；logits = head(ln_f(hidden)) 与
        stream_step 的输出逐位一致——直通模式的逐位一致性由此保证。
        """
        model = self.model
        if token.dim() == 1:
            token = token.unsqueeze(1)
        x = model.tok_emb(token)                              # (B,1,D)
        mem = state.memory
        for blk, st in zip(model.blocks, state.layers):
            mx = blk.mixer
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, _ = mx.step(x, st)
            x = blk.ffn(x)
        state.length += 1
        return x, state

    # -- 流式：消费 + latent 思考 --------------------------------------------
    @torch.no_grad()
    def stream_step(self, token: torch.Tensor, state):
        """消费一个 token，返回思考后的下一 token logits。

        ``max_iters=0`` 或 ``enabled=False`` 时与 ``model.stream_step``
        **逐位一致**（theory 守卫）。
        """
        model = self.model
        hidden, state = self._consume(token, state)
        if not self.enabled or self.max_iters == 0:
            self.ledger.record(0, 0, 0.0)
            return model.head(model.ln_f(hidden[:, -1, :])), state

        z = hidden
        iters = 0
        ponder = 0.0
        for _ in range(self.max_iters):
            z = self.cell(z)
            iters += 1
            p = self.halting(z)                                # (B,1)
            ponder += float((1.0 - p.mean()).item())
            if bool((p >= self.halt_threshold).all()):
                break
        self.ledger.record(iters, self.max_iters, ponder)
        return model.head(model.ln_f(z[:, -1, :])), state

    # -- 训练（并行形式）：可微迭代 + ponder 正则 ----------------------------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None):
        """并行形式：整段前向 + latent 迭代精化 + CE(+ponder) 损失。

        Returns:
            ``(logits, loss)``；``targets=None`` 时 loss 为 None。
            ponder 损失 = ``ponder_weight * Σ_i (1 - p_i)``（只在
            ``max_iters>0`` 且 targets 存在时计入）。
        """
        model = self.model
        x = model.tok_emb(idx)
        mem = model._project_memory(memory)
        for blk in model.blocks:
            x = blk(x, mem)
        hidden = x                                             # 残差流

        ponder_loss = hidden.new_zeros(())
        z = hidden
        for _ in range(self.max_iters if self.enabled else 0):
            z = self.cell(z)
            p = self.halting(z)
            ponder_loss = ponder_loss + (1.0 - p).mean()

        logits = model.head(model.ln_f(z))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            if self.enabled and self.max_iters > 0:
                loss = loss + self.ponder_weight * ponder_loss
        return logits, loss

    # -- 生成 -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """与 ``SimplCoreModel.generate`` 同签名，走思考流式路径。"""
        was_training = self.model.training
        self.model.eval()
        B = idx.shape[0]
        state = self.model.init_stream(B, device=idx.device,
                                       dtype=self.model.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self.model._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
        if was_training:
            self.model.train()
        return out

    # -- 守卫 -----------------------------------------------------------------
    def attention_like_modules(self) -> List[Tuple[str, str]]:
        """返回思考器内所有类名含 attention/attn 的模块（应为空）。"""
        from .purity import attention_like_modules as _attn
        return list(_attn(self))

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        """思考不新增任何流式状态：与被包裹模型完全一致。"""
        return self.model.stream_bytes(batch, dtype)


# ---------------------------------------------------------------------------
# CLI 演示
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    """``python -m simplcore.core.nonlinear`` —— latent 思考机制演示。"""
    import argparse

    from .config import SimplCoreConfig
    from .model import SimplCoreModel

    ap = argparse.ArgumentParser(description="SimplCore 非线性思考演示（玩具模型）")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--len", type=int, default=16)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                          decay_max=128, local_kernel=3, num_heads=2,
                          ffn_ratio=2.0, mixer="sgc")
    m = SimplCoreModel(cfg).eval()
    th = NonlinearThinker(m, max_iters=args.iters,
                          halt_threshold=args.threshold).eval()

    idx = torch.randint(0, 64, (1, args.len))
    with torch.no_grad():
        # 直通对照
        th0 = NonlinearThinker(m, max_iters=0).eval()
        s0 = m.init_stream(1)
        lg0 = None
        for t in range(idx.shape[1]):
            lg0, s0 = th0.stream_step(idx[:, t], s0)
        s_plain = m.init_stream(1)
        lg_plain = None
        for t in range(idx.shape[1]):
            lg_plain, s_plain = m.stream_step(idx[:, t], s_plain)
        assert torch.equal(lg0, lg_plain), "直通必须与原模型逐位一致"

        # 思考路径
        s = m.init_stream(1)
        lg = None
        for t in range(idx.shape[1]):
            lg, s = th.stream_step(idx[:, t], s)
        delta = float((lg - lg_plain).abs().max())

    print(f"passthrough bit-exact : True")
    print(f"think logits max|Δ|   : {delta:.6f}")
    print(f"ledger                : {th.ledger.as_dict()}")
    print(f"attention modules     : {len(th.attention_like_modules())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
