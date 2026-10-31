# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 粘合模型：冻结 SGC 基座 + 路由门 + 多个小专家 → MoE 效果。

* 基座（``SimplCoreModel``）构造时**冻结**并置 eval；
* 前向：``h = base_hidden`` → 路由门选 top-k 专家 → ``h' = h + Σ w·Expert(h)`` → ``base.head``；
* 梯度只流向路由门 + 专家（基座无梯度），显存按 ``WeightBudget.learned_params`` 计；
* ``spawn_expert``：后台自进化入口——在预算守卫下追加全新小专家，注册进 ensemble，
  供「模型在真实环境学到新域、把经验写进权重时」并行训练（模型冻结、harness 生长）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import SimplCoreConfig
from ..core.envlearn import WeightBudget
from .bundle import ModuleBundle, base_fingerprint
from .config import UnityConfig
from .expert import Expert
from .router import ExpertRouter


class UnityModel(nn.Module):
    def __init__(self, base: nn.Module, cfg: UnityConfig,
                 base_params: Optional[int] = None,
                 budget: Optional[WeightBudget] = None) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        # 冻结基座：只让路由 + 专家承载梯度（显存只随可学权重增长）
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        dim = self._infer_dim(base)
        self.dim = dim
        r = cfg.resolve(dim)

        self.router = ExpertRouter(
            dim, r.num_experts, r.router_hidden, r.top_k,
            r.router_z_loss, r.load_balance_loss)

        self._budget = budget or WeightBudget(
            base_params=base_params or self._count_params(base),
            max_learned=2_000_000_000, total_cap=3_000_000_000)

        # 模块来源审计：必须在 _add_expert 之前初始化，否则构造期写 meta 会抛
        # AttributeError。base_fingerprint 用于跨主干拼接的兼容性校验。
        self.base_fingerprint = base_fingerprint(
            getattr(base, "config", None) or {})
        self.expert_meta: Dict[int, Dict[str, object]] = {}

        self.experts = nn.ModuleList()
        for i in range(r.num_experts):
            self._add_expert(r.expert_hidden, r.expert_layers, r.expert_dropout,
                             name=f"expert{i}")

        # 复用基座的词表头
        self.head = base.head

    # -- 构造辅助 -----------------------------------------------------------
    @staticmethod
    def _infer_dim(base: nn.Module) -> int:
        cfg = getattr(base, "config", None)
        if cfg is not None and getattr(cfg, "dim", None) is not None:
            return int(cfg.dim)
        emb = getattr(base, "tok_emb", None)
        if emb is not None:
            return int(emb.weight.shape[1])
        raise ValueError("无法推断基座维度：base 既无 .config.dim 也无 .tok_emb")

    @staticmethod
    def _count_params(base: nn.Module) -> int:
        return sum(p.numel() for p in base.parameters())

    def _add_expert(self, hidden: int, layers: int, dropout: float,
                   name: str) -> int:
        ep = Expert(self.dim, hidden, layers, dropout, name=name)
        cost = int(ep.num_params())
        if not self._budget.can_allocate(cost):
            raise ValueError(
                f"专家 '{name}' 需 {cost} 参数，超出 WeightBudget 剩余 "
                f"{self._budget.remaining_total()}（总上限 {self._budget.total_cap}）")
        self._budget.allocate(cost)
        self.experts.append(ep)
        idx = len(self.experts) - 1
        self.expert_meta[idx] = {"name": name}
        return idx

    # -- 自进化：后台并行训练新专家 ---------------------------------------
    def spawn_expert(self, name: Optional[str] = None, hidden: Optional[int] = None,
                     layers: Optional[int] = None, dropout: float = 0.0) -> int:
        """在预算守卫下追加一个全新小专家（零/小随机初始化）。

        对应「模型在真实环境学到新域、把经验写进权重时，后台同时训练新 Unity 模块」：
        调用方随后用该域**专属（不同）数据**通过 ``train_expert`` 训练它，再注册进 ensemble。
        返回新专家索引；预算不足时抛 ``ValueError``（harness 生长受 3B 上限约束）。
        """
        r = self.cfg.resolve(self.dim)
        name = name or f"expert{len(self.experts)}"
        return self._add_expert(hidden or r.expert_hidden, layers or r.expert_layers,
                               dropout, name)

    def train_expert(self, i: int, loader, optimizer, steps: int = 100,
                     device=None, amp_dtype=None) -> float:
        """在该域**专属数据**上训练第 ``i`` 个专家（基座 + 其他专家冻结，路由可选更新）。

        这是「每个专家训练数据必须不同」的落地点：调用方对每个专家传入不同域的 loader。
        返回该专家域上的平均 CE。
        """
        if i < 0 or i >= len(self.experts):
            raise IndexError(f"专家索引越界：{i} / {len(self.experts)}")
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.experts[i].parameters():
            p.requires_grad_(True)
        self.experts[i].train()
        self.base.eval()
        total, n = 0.0, 0
        for step, (x, y) in enumerate(loader):
            if step >= steps:
                break
            if device is not None:
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
                else torch.autocast("cpu", enabled=False)
            with ctx:
                # 强制只激活专家 i（domain 隔离 → 数据不同）
                _, loss = self.forward(x, y, domain=i)
            loss.backward()
            optimizer.step()
            total += float(loss) * y.numel()
            n += y.numel()
        # 恢复：非训练态下，base 冻结、专家可学（供主循环使用）
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.experts[i].parameters():
            p.requires_grad_(True)
        for p in self.router.parameters():
            p.requires_grad_(True)
        return total / max(n, 1)

    # -- 前向 ---------------------------------------------------------------
    def forward(self, idx, targets=None, domain=None, return_aux: bool = False,
                return_hidden: bool = False, return_expert_hidden: bool = False):
        """统一前向：``idx`` 为 token id 张量，(B, L)。

        * ``domain=None``：MoE 路由（top-k 自学）；
        * ``domain=int/(B,)/(B,L)``：强制路由到指定专家（隔离训练，数据不同）。
        返回：``(logits, loss)``（有 targets）或 ``logits`` / ``(logits, aux)``。
        ``return_hidden=True`` 额外回传**基座** ln_f 隐状态 ``h``；
        ``return_expert_hidden=True`` 额外回传**专家调制后**隐状态（供生物计算主义
        耦合器在其上施加连续场调制，与流式 ``stream_step_hidden`` 对齐），返回
        ``(logits, h_expert[, aux])``。
        """
        self.base.eval()
        out = self.base(idx, return_hidden=True)
        h = out[1] if isinstance(out, tuple) else out   # 基座 ln_f 后隐状态 (B,L,D)

        w, idxs, aux = self.router(h, domain=domain)

        # 仅在被路由选中的 token 上过对应专家（O(active)），再用可微的 index_add
        # 散射累加回全序列。避免「在无梯度张量上就地索引赋值」导致梯度丢失。
        h2 = h
        h_flat = h.reshape(-1, self.dim)
        contrib = torch.zeros_like(h_flat)
        for e in range(self.router.num_experts):
            we = w[..., e]                               # (B, L)
            mask = we > 0
            if not mask.any():
                continue
            active_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            h_sel = h_flat[active_idx]                   # (N, D)
            he = self.experts[e](h_sel)                  # (N, D) 可微
            we_sel = we.reshape(-1)[active_idx].unsqueeze(-1)  # (N, 1) 常量
            contrib = torch.index_add(contrib, 0, active_idx, we_sel * he)
        h2 = h2 + contrib.reshape(h.shape)

        logits = self.head(h2)
        # 生物计算主义耦合器要的隐状态：专家调制后（与流式对齐）；否则基座隐状态。
        h_ret = h2 if return_expert_hidden else h
        if return_hidden or return_expert_hidden:
            # 始终带 aux（lb/z）供生物计算主义耦合器消费；loss 由调用方按需重算。
            if targets is not None:
                return (logits, h_ret, aux)
            return (logits, h_ret, aux) if return_aux else (logits, h_ret)
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1), ignore_index=-100)
            aux_loss = (self.cfg.load_balance_loss * aux["lb"]
                        + self.cfg.router_z_loss * aux["z"])
            loss = ce + aux_loss
            return (logits, loss, aux) if return_aux else (logits, loss)
        return (logits, aux) if return_aux else logits

    # -- 流式：O(1) 递推（生成期也能跑 MoE + 生物连续场）-------------------
    def init_stream(self, batch, device=None, dtype=None):
        """初始化底层基座的 O(1) 流式状态。"""
        return self.base.init_stream(batch, device=device, dtype=dtype)

    def stream_step_hidden(self, token: torch.Tensor, state):
        """消费一个 token，返回**专家调制后**的 ln_f 隐状态 (B,1,D) 与更新后的状态。

        与 ``forward(return_expert_hidden=True)`` 的逐 token 等价形式，供
        ``BioEnhancer.stream_step`` 在其上施加连续场调制（段内 O(1)）。
        """
        base = self.base
        if token.dim() == 1:
            token = token.unsqueeze(1)
        x = base.tok_emb(token)                          # (B,1,D)
        mem = state.memory
        for blk, st in zip(base.blocks, state.layers):
            mx = blk.mixer
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, _ = mx.step(x, st)
            x = blk.ffn(x)
        state.length += 1
        h = base.ln_f(x)                                 # 基座 ln_f 隐状态
        w, idxs, aux = self.router(h, domain=None)
        h_flat = h.reshape(-1, self.dim)
        contrib = torch.zeros_like(h_flat)
        for e in range(self.router.num_experts):
            we = w[..., e]
            mask = we > 0
            if not mask.any():
                continue
            active_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            he = self.experts[e](h_flat[active_idx])
            contrib = torch.index_add(
                contrib, 0, active_idx,
                we.reshape(-1)[active_idx].unsqueeze(-1) * he)
        h2 = h + contrib.reshape(h.shape)
        return h2, state

    @torch.no_grad()
    def stream_step(self, token: torch.Tensor, state):
        """消费一个 token，返回下一 token 的 logits（O(1) 流式；含 MoE）。"""
        h2, st = self.stream_step_hidden(token, state)
        return self.head(h2)[:, -1, :], st

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """UnityMoE 条件下的自回归采样（O(1) 流式：prefill + 逐步路由）.

        与 ``SimplCoreModel.generate`` 的流式路径同约定；采样器复用基座的
        ``_sample``（温度/top-k/top-p 口径一致）。0.1-α 补齐——此前 Unity
        只有训练前向与 stream_step，没有统一生成入口。
        """
        was_training = self.training
        self.eval()
        state = self.init_stream(idx.shape[0], device=idx.device,
                                 dtype=next(self.parameters()).dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self.base._sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
        if was_training:
            self.train()
        return out

    # -- 预算 / 参数接口 ----------------------------------------------------
    def param_filter(self) -> List[torch.Tensor]:
        """仅返回可学参数（路由 + 专家），供优化器 / DDP 使用。"""
        return [p for p in self.parameters() if p.requires_grad]

    def num_params_learned(self) -> int:
        return sum(p.numel() for p in self.param_filter())

    def as_dict(self) -> Dict[str, object]:
        d = self._budget.as_dict()
        d["num_experts"] = len(self.experts)
        d["learned_params_actual"] = self.num_params_learned()
        return d

    # -- 持久化：只存可学部分（路由 + 专家），基座按预设恢复 ----------------
    def save_learned(self, path: str) -> None:
        """把路由 + 全部专家（可学部分）+ 预算落盘。基座不重复存（按预设恢复）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        from dataclasses import asdict
        torch.save({
            "dim": self.dim,
            "cfg": asdict(self.cfg),
            "router": self.router.state_dict(),
            "experts": [e.state_dict() for e in self.experts],
            "expert_meta": dict(self.expert_meta),
            "base_fingerprint": self.base_fingerprint,
            "budget": self._budget.as_dict(),
        }, path)

    @classmethod
    def load_learned(cls, base: nn.Module, path: str,
                     cfg: Optional[UnityConfig] = None) -> "UnityModel":
        """从 ``save_learned`` 的快照恢复路由 + 专家（基座由调用方持有）。

        新专家数按快照决定（``spawn_expert`` 长出的专家也能恢复）；预算账目同步还原。
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state["experts"]
        ucfg = cfg or UnityConfig(**state.get("cfg", {}))
        ucfg.num_experts = len(saved)
        um = cls(base, ucfg)  # 按快照专家数分配预算
        um.router.load_state_dict(state["router"])
        for i, sd in enumerate(saved):
            um.experts[i].load_state_dict(sd)
        b = state["budget"]
        um._budget.base_params = int(b["base_params"])
        um._budget.learned_params = int(b["learned_params"])
        um._budget.max_learned = int(b["max_learned"])
        um._budget.total_cap = int(b["total_cap"])
        um.expert_meta = dict(state.get("expert_meta", {}))
        um.base_fingerprint = str(state.get("base_fingerprint",
                                           um.base_fingerprint))
        return um

    # -- 模块级拆分 / 拼接 ---------------------------------------------------
    def export_expert(self, index: int, **metadata) -> ModuleBundle:
        """把第 ``index`` 个专家导出为可独立搬运的 :class:`ModuleBundle`。

        只读操作，不改变活动模型；bundle 记录 ``base_fingerprint``、来源索引与
        调用方附加的训练量/val loss，供跨主干拼接时审计。
        """
        i = int(index)
        if i < 0 or i >= len(self.experts):
            raise IndexError(f"专家索引越界：{i} / {len(self.experts)}")
        ep = self.experts[i]
        first = ep.layers[0]
        hidden = int(first.gate.out_features)
        dropout = float(getattr(getattr(first, "drop", None), "p", 0.0))
        meta = {"base_fingerprint": self.base_fingerprint, "source_index": i}
        meta.update(self.expert_meta.get(i, {}))
        meta.update(metadata)
        return ModuleBundle(
            name=ep.name, dim=self.dim, hidden=hidden,
            layers=len(ep.layers), dropout=dropout,
            state_dict={k: v.detach().cpu() for k, v in ep.state_dict().items()},
            metadata=meta)

    def import_bundle(self, bundle: ModuleBundle,
                      name: Optional[str] = None) -> int:
        """把外部 bundle 拼接到当前模型（预算守卫 + 结构指纹校验）。

        ``dim`` 必须一致；若 bundle 带 ``base_fingerprint`` 且本模型也有，
        指纹必须相同——否则拒绝，而不是让一个来自异构主干的专家静默污染路由。
        """
        if not bundle.compatible_with(self):
            raise ValueError(
                f"module '{bundle.name}' 与当前模型不兼容：bundle dim={bundle.dim} "
                f"kind={bundle.kind} fingerprint={bundle.metadata.get('base_fingerprint')} "
                f"vs model dim={self.dim} fingerprint={self.base_fingerprint}")
        idx = self.spawn_expert(name=name or bundle.name,
                                hidden=bundle.hidden, layers=bundle.layers,
                                dropout=bundle.dropout)
        self.experts[idx].load_state_dict(bundle.state_dict)
        self.expert_meta[idx] = dict(bundle.metadata)
        return idx

    # -- 无竞态后台训练：在独立候选专家上训练，完成后由调用方原子换入 ------
    def publish_expert(self, candidate: "Expert", name: Optional[str] = None) -> int:
        """把已训好的独立候选专家原子换入活动模型（后台生长的最终一步）。

        返回新专家索引；预算守卫由 ``spawn_expert`` 内部执行。
        """
        idx = self.spawn_expert(name=name)
        self.experts[idx].load_state_dict(candidate.state_dict())
        return idx


def train_standalone_expert(base: nn.Module, head: nn.Module, expert: nn.Module,
                            loader, optimizer, steps: int = 100,
                            device=None, amp_dtype=None) -> float:
    """在**独立的候选专家**上做域专属训练（不触碰活动模型参数，避免后台训练与
    推理前向的竞态）。

    对应「训练数据必须不同」：调用方对每个域传入不同 loader。返回该域平均 CE。
    """
    base.eval()
    expert.train()
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if step >= steps:
            break
        if device is not None:
            x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
            else torch.autocast("cpu", enabled=False)
        with ctx:
            with torch.no_grad():
                h = base(x, return_hidden=True)[1]
            he = expert(h)
            logits = head(h + he)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        total += float(loss) * y.numel()
        n += y.numel()
    return total / max(n, 1)

@torch.no_grad()
def evaluate_standalone(base: nn.Module, head: nn.Module, expert,
                        batches, device=None, amp_dtype=None) -> float:
    """在固定 batch 序列上评估 ``base(+expert)`` 的 token-weighted CE。

    GATE-3（无回归门）用同一口径比较"无专家基线"与"候选专家"：
    候选 CE 上升超过阈值就拒绝发布，而不是让后台训练静默污染活动模型。
    """
    batches = list(batches)
    if not batches:
        raise ValueError("evaluate_standalone requires non-empty batches")
    base.eval()
    if expert is not None:
        expert.eval()
    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
        else torch.autocast("cpu", enabled=False)
    total, n = 0.0, 0
    for x, y in batches:
        if device is not None:
            x, y = x.to(device), y.to(device)
        with ctx:
            h = base(x, return_hidden=True)[1]
            if expert is not None:
                h = h + expert(h)
            logits = head(h)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-100, reduction="sum")
        total += float(loss)
        n += int((y != -100).sum())
    return total / max(n, 1)
