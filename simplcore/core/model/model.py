# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore model: blocks, backbone, causal LM, streaming generation."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SimplCoreConfig
from ..arch.sgc import SGCStreamState, make_norm, recurrent_dtype
from ..arch.selective import GatedSpectralConv2
from ..arch.abstain import AbstentionHead


# ----------------------------------------------------------------------------
# Feed-forward
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """Pre-norm SwiGLU feed-forward with residual connection."""

    def __init__(self, dim: int, hidden: int, norm: str = "rmsnorm",
                 dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = make_norm(norm, dim, eps=eps)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))


# ----------------------------------------------------------------------------
# Block
# ----------------------------------------------------------------------------
class SimplCoreBlock(nn.Module):
    """Pre-norm block: mixer -> FFN, each with its own residual."""

    def __init__(self, config: SimplCoreConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # R3 分层多尺度衰减（opt-in）：scale≠1 时按层重置 SGC 家族的振荡网格
        scale = getattr(config, "decay_scale", 1.0)
        if scale != 1.0:
            from ..arch.sgc import reinit_oscillator_grid
            self._decay_scale_pending = config.decay_max * (scale ** layer_idx)
        else:
            self._decay_scale_pending = None
        if config.gate_ab:
            # GatedSpectralConv2：GLA/Mamba2 式 γ/β 通道独立双门 + 多头谱基。
            # 这是 `gate_ab` 开关唯一的落地路径 —— 之前该开关被模型忽略，
            # 导致 `analytic_params` 多算 2·D²+2·D、预算规划器口径失真。
            # gate_ab 是横切开关（覆盖 mixer 选择），不进注册表，在此处理。
            self.mixer = GatedSpectralConv2(
                dim=config.dim,
                num_bases=config.num_bases,
                decay_max=config.decay_max,
                local_kernel=1 if config.mixer == "sgc-nolocal" else config.local_kernel,
                num_heads=config.num_heads,
                content_gate=config.content_gate and config.mixer != "sgc-nogate",
                out_gate=config.out_gate,
                gate_ab=True,
                evidence_gate=config.evidence_gate,
                uncertainty_gate=config.uncertainty_gate,
                norm=config.norm,
                fft_dtype=config.resolve_fft_dtype(),
                eps=config.norm_eps,
            )
        else:
            # 常规 mixer：查注册表（simplcore/core/registry.py）构造，
            # 变体语义（sgc-nogate/nolocal）由 MixerSpec 在 build 内表达。
            from ..arch.registry import build_mixer
            self.mixer = build_mixer(config)
        self.ffn = SwiGLU(config.dim, config.hidden, norm=config.norm,
                          dropout=config.dropout, eps=config.norm_eps)
        # mixer 构造完成后消费 R3 分层 decay（SGC 家族；其它静默跳过）
        pending = getattr(self, "_decay_scale_pending", None)
        if pending is not None and pending != config.decay_max:
            from ..arch.sgc import reinit_oscillator_grid
            reinit_oscillator_grid(self.mixer, pending)
            self._decay_scale_pending = None

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        if memory is not None:
            # CMT 式记忆注入：把检索到的压缩记忆作为**残差流上的加性项**。
            # SGC 没有注意力、没有 KV 缓存，残差流是它原生的注入点。
            x = x + memory.unsqueeze(1)
        if self.config.grad_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.mixer, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False)
            return x
        x = self.mixer(x)
        return self.ffn(x)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _make_block_for_config(config: SimplCoreConfig, layer_idx: int) -> "SimplCoreBlock":
    """Build the block for layer ``layer_idx`` given ``config``.

    SimplCore ships only non-attention mixers (SGC / SGC-nogate / SGC-nolocal /
    SRF). Every mixer exposes an O(1) streaming state, so the model is always
    streamable — there is no attention baseline to route around.
    """
    return SimplCoreBlock(config, layer_idx)


class ConceptState(nn.Module):
    """NCM recurrence and optional causal next-concept prediction heads.

    It is intentionally separate from the core SGC/SNN token stream: the token
    model remains the primary predictor, while per-session concept state provides
    a causal latent path without storing batch state on the module.
    """

    def __init__(self, dim: int, concept_dim: int, alpha: float = 0.15,
                 beta: float = 0.7, label_classes: int = 0,
                 group_classes: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.concept_dim = int(concept_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.proj = nn.Linear(dim, concept_dim, bias=False)
        self.out = nn.Linear(concept_dim, dim, bias=False)
        self.future_predictor = nn.Linear(concept_dim, dim, bias=False)
        self.label_predictor = (
            nn.Linear(concept_dim, label_classes, bias=False)
            if label_classes > 0 else None)
        self.group_predictor = (
            nn.Linear(concept_dim, group_classes, bias=False)
            if group_classes > 0 else None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        future_key = prefix + "future_predictor.weight"
        if future_key not in state_dict:
            state_dict[future_key] = self.future_predictor.weight.detach()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch, self.concept_dim, device=device,
            dtype=recurrent_dtype(dtype))

    def step(self, x: torch.Tensor, state: torch.Tensor,
             collect_predictions: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError("ConceptState.step expects one token per batch")
            x = x[:, 0]
        if x.dim() != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"ConceptState input must have shape (batch, {self.dim})")
        if state.shape != (x.shape[0], self.concept_dim):
            raise ValueError(
                f"ConceptState state must have shape "
                f"({x.shape[0]}, {self.concept_dim})"
            )
        predictions: Dict[str, torch.Tensor] = {}
        if collect_predictions:
            predictions["future_latent"] = self.future_predictor(state.to(
                device=x.device, dtype=x.dtype))
            if self.label_predictor is not None:
                predictions["concept_labels"] = self.label_predictor(
                    state.to(device=x.device, dtype=x.dtype))
            if self.group_predictor is not None:
                predictions["token_groups"] = self.group_predictor(
                    state.to(device=x.device, dtype=x.dtype))
        concept = self.proj(x)
        state_dtype = recurrent_dtype(concept.dtype)
        updated = (1.0 - self.alpha) * state.to(
            device=x.device, dtype=state_dtype
        ) + self.alpha * concept.to(state_dtype)
        output = self.out(updated.to(dtype=concept.dtype))
        return output, updated, predictions

    def forward_sequence(self, x: torch.Tensor,
                         state: Optional[torch.Tensor] = None,
                         collect_predictions: bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    Dict[str, torch.Tensor]]:
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"ConceptState input must have shape (batch, seq, {self.dim})")
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        if x.shape[1] == 0:
            return x.new_empty(x.shape[0], 0, self.dim), state, {}

        projected = self.proj(x)
        state_dtype = recurrent_dtype(projected.dtype)
        projected = projected.to(state_dtype)
        initial_state = state.to(device=x.device, dtype=state_dtype)
        current = initial_state
        state_parts = []
        for token_concept in projected.unbind(dim=1):
            current = (1.0 - self.alpha) * current + self.alpha * token_concept
            state_parts.append(current.unsqueeze(1))
        state_sequence = torch.cat(state_parts, dim=1)
        output = self.out(state_sequence.to(dtype=x.dtype))

        predictions: Dict[str, torch.Tensor] = {}
        if collect_predictions:
            previous = torch.cat(
                (initial_state.unsqueeze(1), state_sequence[:, :-1]), dim=1)
            previous = previous.to(dtype=x.dtype)
            predictions["future_latent"] = self.future_predictor(previous)
            if self.label_predictor is not None:
                predictions["concept_labels"] = self.label_predictor(previous)
            if self.group_predictor is not None:
                predictions["token_groups"] = self.group_predictor(previous)
        return output, current, predictions

    def auxiliary_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        hidden: torch.Tensor,
        labels: Optional[torch.Tensor],
        groups: Optional[torch.Tensor],
        config: SimplCoreConfig,
    ) -> Dict[str, object]:
        losses: Dict[str, object] = {}
        future = predictions.get("future_latent")
        if future is not None:
            per_token = F.mse_loss(
                future.float(), hidden.detach().float(), reduction="none"
            ).mean(dim=-1)
            count = per_token.numel()
            loss_sum = per_token.sum()
            losses["future_latent_loss_sum"] = loss_sum
            losses["future_latent_count"] = count
            losses["future_latent_loss"] = loss_sum / max(count, 1)

        for name, targets, predictor, weight_name in (
            ("concept_labels", labels, self.label_predictor, "ncm_label_weight"),
            ("token_groups", groups, self.group_predictor, "ncm_group_weight"),
        ):
            if targets is None:
                continue
            if predictor is None:
                raise ValueError(
                    f"{name} targets were provided but the corresponding NCM "
                    "class count is not configured")
            logits = predictions[name]
            if targets.shape != logits.shape[:2]:
                raise ValueError(
                    f"{name} targets must have shape {tuple(logits.shape[:2])}, "
                    f"got {tuple(targets.shape)}")
            targets = targets.to(device=logits.device, dtype=torch.long)
            count = int((targets != -100).sum().item())
            if count:
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), ignore_index=-100, reduction="sum")
            else:
                loss_sum = logits.sum() * 0.0
            losses[f"{name}_loss_sum"] = loss_sum
            losses[f"{name}_count"] = count
            losses[f"{name}_loss"] = loss_sum / max(count, 1)

        weighted = hidden.sum() * 0.0
        for source, weight_name in (
            ("future_latent", "ncm_future_weight"),
            ("concept_labels", "ncm_label_weight"),
            ("token_groups", "ncm_group_weight"),
        ):
            loss_sum = losses.get(f"{source}_loss_sum")
            count = int(losses.get(f"{source}_count", 0))
            if loss_sum is not None and count:
                weighted = weighted + getattr(config, weight_name) * loss_sum / count
        losses["ncm_loss"] = weighted
        return losses


class SimplCoreModel(nn.Module):
    """SimplCore causal language model (non-attention backbone).

    There is deliberately **no positional-embedding table** and **no softmax
    self-attention**: the causal spectral convolution already encodes position,
    which is what makes length extrapolation and O(1)-state streaming well
    defined — and what removes the KV-cache that grows linearly with the
    sequence in Transformer decoders.

    NCM opt-in: if ``config.use_ncm`` is enabled, we add a small concept state
    parallel to the ordinary SGC/SNN stream. This is a semantic summary channel,
    not a Transformer-style attention cache; the model still behaves as a
    standard causal LM while the concept pathway provides a richer concept-level
    representation for chat and reasoning tasks.
    """

    def __init__(self, config: SimplCoreConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            _make_block_for_config(config, i) for i in range(config.num_layers)
        ])
        self.ln_f = make_norm(config.norm, config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        # L4 拒答门: 可选的 grounded 预测头（默认关闭；需外部证据标签才能训练）
        self.abstain_head = AbstentionHead(config.dim) if config.use_abstain else None
        self.ncm = ConceptState(
            config.dim,
            config.ncm_dim,
            config.ncm_alpha,
            config.ncm_beta,
            config.ncm_label_classes,
            config.ncm_group_classes,
        ) if config.use_ncm else None
        self.apply(self._init_weights)
        # CMT 记忆注入投影：零初始化 → 初始为恒等（不改变基座行为），训练后自学
        self.memory_in = nn.Linear(config.dim, config.dim, bias=False) if config.use_memory else None
        if self.memory_in is not None:
            with torch.no_grad():
                self.memory_in.weight.zero_()
        # scaled residual-output init (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.num_layers))


    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.init_std)

    # -- forward -------------------------------------------------------------
    def forward(self, idx_or_x, targets: Optional[torch.Tensor] = None,
                return_hidden: bool = False,
                text: Optional[str] = None, codec=None,
                return_abstain: bool = False,
                memory: Optional[torch.Tensor] = None,
                stream_state: Optional[SGCStreamState] = None,
                ncm_label_targets: Optional[torch.Tensor] = None,
                ncm_group_targets: Optional[torch.Tensor] = None,
                return_ncm_aux: bool = False):
        """统一前向入口：接受 token ID 张量或文本字符串（无词表模式）.

        * ``idx_or_x`` 为张量 → 传统路径：token ID → 嵌入 → 前向；
        * ``idx_or_x`` 为字符串（且提供 ``codec``）→ 无词表路径：
          ``codec.encode(text)`` → 张量 → 嵌入 → 前向 → 返回 logit；
          若 ``targets`` 也是字符串，同 codec 编码后计算 loss。

        这使"舍去传统 Transformer 词表"成为一等公民：同一套模型权重
        既可按 HF 词表 ID 流喂入，也可按字符/字节/BPE 文本直喂，
        不重新初始化词表参数。

        ⚠️ **恒等陷阱（`model(x, x)`）—— 第三方使用者必读**
        ----------------------------------------------------------------
        ``targets`` 与 ``logits`` **同位置对齐，本方法不做任何自动 shift**。
        位置 t 的输入包含 ``x[t]`` 本身，若 ``targets is x``，模型只需学会
        "复制当前 token" 即可把 loss 压到任意低——**不报错、loss 照样收敛
        到 1e-4 量级，只有看生成文本才会暴露**。

        正确写法（语言建模）：``targets`` 必须是**右移一位**的标签
        （``targets[..., t] == x[..., t+1]``），即::

            x, y = batch[:, :-1], batch[:, 1:]
            logits, loss = model(x, y)          # 预测下一个 token

        仓库内部测试里的 ``model(x, x)`` 全部是 **API 冒烟**（只验证前向/
        反传能跑通），不构成训练语义示范。
        """
        if isinstance(idx_or_x, str):
            if codec is None:
                raise ValueError("text input requires a codec")
            idx_or_x = torch.tensor([codec.encode(idx_or_x)],
                                     device=next(self.parameters()).device)
        if isinstance(targets, str):
            if codec is None:
                raise ValueError("text targets require a codec")
            targets = torch.tensor([codec.encode(targets)],
                                   device=next(self.parameters()).device)
        B, L = idx_or_x.shape
        if (ncm_label_targets is not None or ncm_group_targets is not None
                or return_ncm_aux) and self.ncm is None:
            raise ValueError("NCM targets/output require SimplCoreConfig(use_ncm=True)")
        if (ncm_label_targets is not None or ncm_group_targets is not None) \
                and not return_ncm_aux:
            raise ValueError("NCM label/group targets require return_ncm_aux=True")
        if return_ncm_aux and targets is None:
            raise ValueError("return_ncm_aux=True requires next-token targets")
        if return_ncm_aux and (return_hidden or return_abstain):
            raise ValueError("return_ncm_aux cannot be combined with hidden/abstain outputs")
        # 单块带状态前向：DDP 逐块训练必须经由 DDP.forward（reducer 才能
        # 在 no_sync / 最后一块同步之间正确切换），因此入口放在 forward。
        if stream_state is not None:
            if return_hidden or return_abstain or memory is not None:
                raise ValueError(
                    "stream_state 模式不支持 return_hidden/return_abstain/memory")
            result = self.forward_chunk_step(
                idx_or_x, targets, stream_state,
                ncm_label_targets=ncm_label_targets,
                ncm_group_targets=ncm_group_targets,
                return_ncm_aux=return_ncm_aux)
            if return_ncm_aux:
                logits, loss_sum, n_tok, _, aux = result
            else:
                logits, loss_sum, n_tok, _ = result
            if targets is None:
                return logits
            loss = loss_sum / max(n_tok, 1)
            if return_ncm_aux:
                return logits, loss, aux
            return logits, loss

        ncm_hidden = None
        ncm_predictions = None
        if self.ncm is not None:
            x = self.drop(self.tok_emb(idx_or_x))
            mem = self._project_memory(memory)
            for blk in self.blocks:
                x = blk(x, mem)
            ncm_hidden = x
            concept, _, ncm_predictions = self.ncm.forward_sequence(
                x, collect_predictions=return_ncm_aux)
            x = self.ln_f(x + concept * self.ncm.beta)
        else:
            x = self.drop(self.tok_emb(idx_or_x))
            mem = self._project_memory(memory)
            for blk in self.blocks:
                x = blk(x, mem)
            x = self.ln_f(x)
        logits = self.head(x)
        if return_abstain:
            if self.abstain_head is None:
                raise ValueError(
                    "return_abstain=True requires SimplCoreConfig(use_abstain=True)"
                )
            abstain_logit = self.abstain_head(x)
            if return_hidden:
                return logits, x, abstain_logit
            return logits, abstain_logit
        if return_hidden:
            return logits, x
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            if return_ncm_aux:
                aux = self.ncm.auxiliary_losses(
                    ncm_predictions, ncm_hidden,
                    ncm_label_targets, ncm_group_targets, self.config)
                return logits, loss + aux["ncm_loss"], aux
            return logits, loss
        return logits

    # -- streaming -----------------------------------------------------------
    def _project_memory(self, memory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """把检索到的压缩记忆投影到模型宽度。

        没有提供记忆时返回 ``None``（基座行为完全不变）。

        **提供了记忆但模型没开 ``use_memory`` 时必须报错**，不能静默丢弃。
        静默丢弃的表现是：训练照跑、loss 照降，但记忆行里的样本被当成
        "没有记忆"处理 —— 模型只能瞎猜，而这个问题在曲线和形状断言上
        都看不见。宁可让配置错误当场炸掉。
        """
        if memory is None:
            return None
        if self.memory_in is None:
            raise ValueError(
                "收到了 memory 但没有启用记忆注入（SimplCoreConfig(use_memory=False)）。"
                "请在配置里打开 use_memory，或在数据里去掉 memory 字段。"
            )
        return self.memory_in(memory)

    @property
    def supports_streaming(self) -> bool:
        """所有 mixer 都提供 O(1) 增量状态时才为 True。

        SimplCore 只搭载非注意力 mixer（SGC / SRF / GatedSpectralConv2），
        它们都用二阶 IIR 递推提供与序列长度无关的常驻状态，所以整套模型
        始终可流式。``stream_step`` 据此始终可用——这是相对 softmax 注意力
        （KV 缓存随序列线性膨胀）的核心理论优势。
        """
        for blk in self.blocks:
            mx = blk.mixer
            if not (hasattr(mx, "init_state") and hasattr(mx, "step")):
                return False
        return True

    def init_stream(self, batch: int, device=None, dtype=None,
                    memory: Optional[torch.Tensor] = None) -> SGCStreamState:
        device = device or self.tok_emb.weight.device
        dtype = dtype or self.tok_emb.weight.dtype
        states = []
        for blk in self.blocks:
            mx = blk.mixer
            assert hasattr(mx, "init_state") and hasattr(mx, "step"), \
                "SimplCore mixers must provide an O(1) init_state/step (no attention)"
            states.append(mx.init_state(batch, device, dtype))
        concept_state = (
            self.ncm.init_state(batch, device, dtype)
            if self.ncm is not None else None
        )
        return SGCStreamState(
            layers=states, length=0, memory=self._project_memory(memory),
            concept_state=concept_state)

    def stream_step(self, token: Optional[torch.Tensor], state: SGCStreamState,
                    inputs_embeds: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, SGCStreamState]:
        """Consume one token per batch element; returns logits for the *next* token.

        Every SimplCore mixer exposes an O(1) ``init_state``/``step`` recurrence,
        so per-token cost and memory are independent of the context length — the
        defining theoretical advantage over softmax-attention, whose KV cache
        grows linearly with the sequence.

        ``inputs_embeds``（0.1-α 多模态入口）：连续软 token (B,1,D)，提供时
        **替代** ``tok_emb(token)`` 直接进入递推（``token`` 可为 None）——
        模态前缀（图像 patch / 音频帧）由此以 O(1) 成本 prefill。默认 None，
        行为与旧版**逐位一致**。
        """
        if inputs_embeds is not None:
            x = inputs_embeds
            if x.dim() == 2:
                x = x.unsqueeze(1)                                     # (B,1,D)
        else:
            if token is None:
                raise ValueError("stream_step requires token or inputs_embeds")
            if token.dim() == 1:
                token = token.unsqueeze(1)
            x = self.tok_emb(token)                                # (B,1,D)
        mem = state.memory
        for blk, st in zip(self.blocks, state.layers):
            mx = blk.mixer
            assert st is not None, "streaming state must be initialised with init_stream()"
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, _ = mx.step(x, st)
            x = blk.ffn(x)
        if self.ncm is not None:
            if state.concept_state is None:
                state.concept_state = self.ncm.init_state(
                    x.shape[0], x.device, x.dtype)
            concept, state.concept_state, _ = self.ncm.step(
                x[:, -1, :], state.concept_state)
            x = x + concept.unsqueeze(1) * self.ncm.beta
        x = self.ln_f(x)
        state.length += 1
        return self.head(x[:, -1, :]), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        """常驻流式状态字节数（与序列长度无关）。

        逐层调用 mixer 的 ``streaming_state_bytes``；递推部分按 ``recurrent_dtype``
        计费——低精度计算时递推状态恒为 fp32，故账单以 fp32 元素大小为准，不随
        计算精度缩半（与 :func:`scaling.stream_state_bytes` 口径一致）。
        """
        total = 0
        for blk in self.blocks:
            mx = blk.mixer
            if hasattr(mx, "streaming_state_bytes"):
                total += mx.streaming_state_bytes(batch, dtype)
            elif hasattr(mx, "stream_bytes"):
                total += mx.stream_bytes(batch, dtype)
        if self.ncm is not None:
            total += batch * self.ncm.concept_dim * torch.empty(
                (), dtype=recurrent_dtype(dtype)).element_size()
        return total

    # -- chunked training ----------------------------------------------------
    @staticmethod
    def _detach_stream_state(state: "SGCStreamState") -> "SGCStreamState":
        """把流式状态整体 detach（截断 BPTT 的块间语义）。"""
        from ..arch.sgc import LayerStreamState, SGCStreamState
        layers = []
        for st in state.layers:
            if isinstance(st, LayerStreamState):
                layers.append(LayerStreamState(
                    y1=st.y1.detach(), y2=st.y2.detach(),
                    conv_buf=(st.conv_buf.detach()
                              if st.conv_buf is not None else None),
                    length=st.length,
                    energy_sum=(st.energy_sum.detach()
                                if st.energy_sum is not None else None),
                    spike_membrane=(st.spike_membrane.detach()
                                    if st.spike_membrane is not None else None)))
            else:
                layers.append(st)
        return SGCStreamState(
            layers=layers, length=state.length,
            memory=(state.memory.detach() if state.memory is not None else None),
            concept_state=(state.concept_state.detach()
                           if state.concept_state is not None else None))

    def forward_chunk_step(self, idx_chunk: torch.Tensor,
                           targets_chunk: Optional[torch.Tensor],
                           state: SGCStreamState,
                           ncm_label_targets: Optional[torch.Tensor] = None,
                           ncm_group_targets: Optional[torch.Tensor] = None,
                           return_ncm_aux: bool = False):
        """跑**一个** chunk，返回 ``(logits, loss_sum, n_tok, new_state)``。

        训练循环必须逐块调用它并**立即 backward**：只保留当前块的计算图，
        激活才是 ``O(chunk·D)``。若把多个 chunk 的 logits 拼起来再 backward，
        所有块的图会同时存活，显存退回 ``O(L)``。
        """
        if not self.supports_streaming:
            raise ValueError("forward_chunk_step 需要所有 mixer 提供 O(1) init_state/step")
        if return_ncm_aux and self.ncm is None:
            raise ValueError("return_ncm_aux=True requires use_ncm=True")
        if (ncm_label_targets is not None or ncm_group_targets is not None) \
                and not return_ncm_aux:
            raise ValueError("NCM label/group targets require return_ncm_aux=True")
        for blk in self.blocks:
            if not hasattr(blk.mixer, "forward_chunk"):
                raise ValueError(
                    f"forward_chunk_step 尚未为 mixer {type(blk.mixer).__name__} 提供"
                    "带状态的分块前向；请用 mixer='sgc' 系列")
        x = self.drop(self.tok_emb(idx_chunk))
        mem = state.memory
        for i, (blk, st) in enumerate(zip(self.blocks, state.layers)):
            if mem is not None:
                x = x + mem.unsqueeze(1)
            x, st = blk.mixer.forward_chunk(x, st)
            state.layers[i] = st
            x = blk.ffn(x)
        ncm_hidden = None
        ncm_predictions = None
        if self.ncm is not None:
            ncm_hidden = x
            concept, state.concept_state, ncm_predictions = self.ncm.forward_sequence(
                x, state.concept_state, collect_predictions=return_ncm_aux)
            x = x + concept * self.ncm.beta
        x = self.ln_f(x)
        logits = self.head(x)
        loss_sum = None
        n_tok = 0
        if targets_chunk is not None:
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets_chunk.reshape(-1),
                ignore_index=-100, reduction="sum")
            n_tok = int((targets_chunk != -100).sum().item())
        state.length += idx_chunk.shape[1]
        if return_ncm_aux:
            if targets_chunk is None:
                raise ValueError("return_ncm_aux=True requires next-token targets")
            aux = self.ncm.auxiliary_losses(
                ncm_predictions, ncm_hidden,
                ncm_label_targets, ncm_group_targets, self.config)
            return logits, loss_sum, n_tok, state, aux
        return logits, loss_sum, n_tok, state

    def forward_chunked(self, idx_or_x, targets: Optional[torch.Tensor] = None,
                        chunk_size: int = 4096, state: Optional[SGCStreamState] = None,
                        detach_state: bool = True):
        """分块前向的**便利接口**（评估/推理）：返回整段 logits。

        与 :meth:`forward` 数值等价（fp32 误差内）。注意它会把所有 chunk 的
        logits 保留在同一张图里，因此 **不适合带梯度的训练**——训练请逐块
        调用 :meth:`forward_chunk_step` 并立即 backward。

        Args:
            idx_or_x: (B, L) token id。
            targets: (B, L) 目标；提供时返回 ``(logits, loss)``。
            chunk_size: 每块的序列长度。
            state: 可选的初始流式状态；``None`` 时从零开始。
            detach_state: 块间是否 detach 状态。
        """
        if isinstance(idx_or_x, str):
            raise ValueError("forward_chunked 只接受 token id 张量；文本请先编码")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        B, L = idx_or_x.shape
        if L == 0:
            raise ValueError("empty sequence")
        if state is None:
            state = self.init_stream(B, device=idx_or_x.device,
                                     dtype=self.tok_emb.weight.dtype)
        logits_parts = []
        loss_sum = None
        n_tok = 0
        pos = 0
        while pos < L:
            end = min(pos + chunk_size, L)
            logits, part, n_part, state = self.forward_chunk_step(
                idx_or_x[:, pos:end],
                None if targets is None else targets[:, pos:end], state)
            logits_parts.append(logits)
            if part is not None:
                loss_sum = part if loss_sum is None else loss_sum + part
                n_tok += n_part
            pos = end
            if detach_state and pos < L:
                state = self._detach_stream_state(state)
        logits = torch.cat(logits_parts, dim=1)
        if targets is None:
            return logits
        loss = loss_sum / max(n_tok, 1)
        return logits, loss

    # -- streaming checkpoint/restore ----------------------------------------
    def checkpoint_stream(self, state: "SGCStreamState") -> dict:
        """把 O(1) 流式状态序列化成纯 dict（可 JSON/safetensors 持久化）.

        返回 ``{"length": int, "layers": [per-layer state dict], "memory": tensor|None}``。
        每层状态是 :class:`LayerStreamState`（SGC IIR 二阶递推）→ 转为 plain dict。
        调用方可将结果 ``torch.save`` / 写 safetensors 以保存"断点流"，
        恢复时用 :meth:`restore_stream` 重建。
        """
        layers: list = []
        for st in state.layers:
            if st is None:
                layers.append(None)
            elif isinstance(st, dict):
                # 兼容旧 schema：注意力时代遗留的 dict 状态原样透传
                layers.append(st)
            else:
                # LayerStreamState（SGC mixer）→ 字段导出
                layers.append({
                    "y1": st.y1, "y2": st.y2,
                    "conv_buf": st.conv_buf, "length": st.length,
                    # L2 证据门的前缀能量和：漏掉它会让 restore 之后的
                    # 闸门从 0 重新累积，流式输出与不中断的生成不再逐位一致。
                    "energy_sum": st.energy_sum,
                    "spike_membrane": st.spike_membrane,
                })
        # CMT 记忆注入：漏掉它会让带记忆的长会话在断点恢复后
        # 记忆注入静默消失——这正是 _project_memory 声明要杜绝的"静默丢弃"。
        return {
            "length": state.length,
            "layers": layers,
            "memory": state.memory,
            "concept_state": state.concept_state,
        }

    def restore_stream(self, ckpt: dict) -> "SGCStreamState":
        """从 :meth:`checkpoint_stream` 的 dict 重建 SGCStreamState.

        兼容 LayerStreamState 字段（自动还原为 LayerStreamState 实例）
        和 dict 状态原样透传；``memory``（投影后的 CMT 记忆）一并恢复。
        """
        from ..arch.sgc import SGCStreamState, LayerStreamState
        layers = []
        for item in ckpt.get("layers", []):
            if item is None:
                layers.append(None)
            elif isinstance(item, dict) and "y1" in item:
                layers.append(LayerStreamState(
                    y1=item["y1"], y2=item["y2"],
                    conv_buf=item.get("conv_buf"),
                    length=item.get("length", 0),
                    energy_sum=item.get("energy_sum"),
                    spike_membrane=item.get("spike_membrane")))
            else:
                layers.append(item)
        return SGCStreamState(layers=layers, length=ckpt.get("length", 0),
                              memory=ckpt.get("memory"),
                              concept_state=ckpt.get("concept_state"))

    # -- generation ----------------------------------------------------------
    @torch.no_grad()
    def generate_text(self, text: str, codec, max_new_tokens: int = 32,
                      temperature: float = 1.0, top_k: Optional[int] = None,
                      top_p: Optional[float] = None,
                      eos_token_id: Optional[int] = None) -> str:
        """无词表模式文本生成：text → codec.encode → 生成 → codec.decode → text.

        这是"舍去传统 Transformer 词表"的核心推理入口：同一套模型权重可切换
        任意编解码器（字符/字节/BPE），无需重新初始化词表参数，只需
        在推理时传入对应的 codec。

        生成在 ID 空间完成（O(1) 状态流式路径），最后统一 decode 回文本。
        若 codec 无 eos_token_id 概念（默认 CharTokenizer），传 ``eos_token_id``
        以提前停止；否则生成满 ``max_new_tokens`` 个 token。
        """
        from .tokenization import Tokenizer
        assert isinstance(codec, Tokenizer)
        idx = torch.tensor([codec.encode(text)], device=next(self.parameters()).device)
        out = self.generate(idx, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k,
                            top_p=top_p, eos_token_id=eos_token_id)
        gen_ids = out[0, idx.shape[1]:].tolist()
        return codec.decode(gen_ids)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, use_state: bool = True,
                 eos_token_id: Optional[int] = None,
                 logit_bias_fn: Optional[object] = None) -> torch.Tensor:
        """Autoregressive sampling.

        ``use_state=True`` uses the O(1) recurrent path: per-token cost and memory
        do not grow with the context length.

        ``logit_bias_fn``（v1.2，JitRL 式测试时调制钩子，默认 None）：
        每步在采样前以当前 ``(B, V)`` logits 调用，返回**加性偏置**
        （可广播到 logits 形状，如 ``(V,)``）或 ``None``（恒等）。
        供 ``jitrl`` 的经验优势偏置使用——**免训练的输出端策略调制**
        （KL 约束策略优化的加性闭式更新，见 JitRL arXiv:2601.18510）。
        默认 None 时所有路径与未接入**逐位一致**（测试锁定）；模型本身
        不依赖任何经验记忆实现。
        """
        was_training = self.training
        self.eval()
        B = idx.shape[0]

        def _mod(lg: torch.Tensor) -> torch.Tensor:
            if logit_bias_fn is None:
                return lg
            out = logit_bias_fn(lg)
            return lg if out is None else lg + out

        if not use_state:
            out = idx
            for _ in range(max_new_tokens):
                logits = _mod(self(out)[:, -1, :])
                nxt = self._sample(logits, temperature, top_k, top_p)
                out = torch.cat([out, nxt], dim=1)
                if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                    break
            if was_training:
                self.train()
            return out

        state = self.init_stream(B, device=idx.device, dtype=self.tok_emb.weight.dtype)
        logits = None
        for t in range(idx.shape[1]):
            logits, state = self.stream_step(idx[:, t], state)
        out = idx
        for _ in range(max_new_tokens):
            nxt = self._sample(_mod(logits), temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits, state = self.stream_step(nxt, state)
        if was_training:
            self.train()
        return out

    # -- weight snapshot / rollback (ReflectionLoop 用) ------------------------
    def snapshot(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        """完整权重快照，供 ``ReflectionLoop`` 的 no-regression gate 回滚。

        ``to_cpu=True``（默认）把快照搬到 CPU：1B 权重在 bf16 下约 2GB，
        留在显存会和训练态抢空间；回滚是低频操作，搬一次值得。
        """
        snap: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict().items():
            if not torch.is_tensor(v):
                continue
            snap[k] = v.detach().to("cpu", copy=True) if to_cpu else v.detach().clone()
        return snap

    def restore(self, snap: Dict[str, torch.Tensor]) -> None:
        """从 :meth:`snapshot` 就地恢复权重（拷贝回当前参数所在设备）。

        只覆盖快照里出现的键，其余保持不动；类型/设备按目标张量对齐。
        """
        if not snap:
            raise ValueError("restore 需要一个非空 snapshot")
        device = next(self.parameters()).device
        target = self.state_dict()
        with torch.no_grad():
            for k, v in snap.items():
                if k in target and torch.is_tensor(v):
                    target[k].copy_(v.to(device=device, dtype=target[k].dtype))

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int],
                top_p: Optional[float]) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0 < top_p < 1.0:
            srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(srt, dim=-1).cumsum(-1)
            remove = probs - F.softmax(srt, dim=-1) > top_p
            srt = srt.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, srt_idx, srt)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # -- introspection -------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def layer_summary(self) -> List[str]:
        return [repr(blk.mixer) for blk in self.blocks]


# Convenience alias: this model already carries the LM head.
SimplCoreForCausalLM = SimplCoreModel


if __name__ == "__main__":
    import sys

    from .config import get_preset, report_presets

    if "--report-presets" in sys.argv:
        raise SystemExit(report_presets())
    else:
        cfg = get_preset("toy")
        m = SimplCoreModel(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        logits, loss = m(x, x)
        loss.backward()
        print("toy smoke OK", tuple(logits.shape), float(loss))
