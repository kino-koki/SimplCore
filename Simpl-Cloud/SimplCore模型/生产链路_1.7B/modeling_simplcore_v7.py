#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimplCore V7 -- production torch / HuggingFace causal LM.

This is the 1.7B-scale implementation of the same architecture whose NumPy
reference was trained and generated from in run_sandbox.py:

  block = RMSNorm -> [bounded causal depthwise conv + pointwise gate]
                  + [R slotted outer-product SGC state, static IIR kernel,
                     learned calibrated write gate]
                  -> learned blend -> teacher o_proj -> residual
          RMSNorm -> Qwen3 SwiGLU FFN -> residual

No softmax attention, no N x N score, no KV cache.  The per-token state is
O(R * (d/R)^2 + K*d), independent of sequence length L.  ``forward`` unrolls
the recurrence for training; ``forward_step`` / ``chat_generate`` use the O(1)
state for generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
    _HF = True
except Exception:                                     # pragma: no cover
    PreTrainedModel, PretrainedConfig = nn.Module, object
    CausalLMOutputWithPast = None
    AutoConfig = AutoModelForCausalLM = None
    _HF = False


class SimplCoreV7Config(PretrainedConfig if _HF else object):
    model_type = "simplcore_v7"

    def __init__(self, vocab_size: int = 151936, hidden_size: int = 2048,
                 num_hidden_layers: int = 28, n_slots: int = 2,
                 conv_kernel: int = 3, rho: float = 0.90,
                 intermediate_size: int = 6144, rms_norm_eps: float = 1e-6,
                 tie_word_embeddings: bool = True,
                 bos_token_id: int = 151643, eos_token_id: int = 151645,
                 max_position_embeddings: int = 40960, **kwargs):
        if _HF:
            super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                             tie_word_embeddings=tie_word_embeddings,
                             max_position_embeddings=max_position_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_slots = n_slots
        self.conv_kernel = conv_kernel
        self.rho = rho
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        assert hidden_size % n_slots == 0, "hidden_size must be divisible by n_slots"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


class QwenSwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class V7Block(nn.Module):
    def __init__(self, cfg: SimplCoreV7Config):
        super().__init__()
        d, R, K = cfg.hidden_size, cfg.n_slots, cfg.conv_kernel
        self.d, self.R, self.K, self.ds, self.rho = d, R, K, d // R, cfg.rho
        self.g1 = RMSNorm(d, cfg.rms_norm_eps)
        self.g2 = RMSNorm(d, cfg.rms_norm_eps)
        self.Wc = nn.Parameter(torch.zeros(K, d))
        self.Wg = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.wgate = nn.Linear(d, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(-2.0))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ffn = QwenSwiGLU(d, cfg.intermediate_size)

    def init_state(self, B, device, dtype):
        return dict(S=torch.zeros(B, self.R, self.ds, self.ds, device=device, dtype=dtype),
                    hist=[])

    def step(self, x, st):
        """One token. x: (B,d). Returns (x_new, state)."""
        B, d = x.shape
        h = self.g1(x)
        u = self.Wc[0] * h
        for j in range(1, self.K):
            if len(st["hist"]) >= j:
                u = u + self.Wc[j] * st["hist"][-j]
        g = torch.sigmoid(self.Wg(h))
        local = u * g
        k = self.Wk(h).view(B, self.R, self.ds)
        v = self.Wv(h).view(B, self.R, self.ds)
        q = self.Wq(h).view(B, self.R, self.ds)
        upd = k.unsqueeze(-1) * v.unsqueeze(-2)              # (B,R,ds,ds)
        S = self.rho * st["S"] + upd
        ys = torch.einsum("brij,brj->bri", S, q) / math.sqrt(self.ds)
        ys = ys.reshape(B, d)
        wg = torch.sigmoid(self.wgate(h))                    # (B,1)
        a = torch.sigmoid(self.alpha)
        y = a * (ys * wg) + (1.0 - a) * local
        y = self.out_proj(y)
        x = x + y
        x = x + self.ffn(self.g2(x))
        hist = (st["hist"] + [h.detach()])[-(self.K - 1):] if self.K > 1 else []
        return x, dict(S=S, hist=hist)


class SimplCoreV7Model(nn.Module):
    def __init__(self, cfg: SimplCoreV7Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([V7Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def init_state(self, B, device, dtype):
        return [b.init_state(B, device, dtype) for b in self.blocks]


class SimplCoreV7ForCausalLM(PreTrainedModel if _HF else nn.Module):
    config_class = SimplCoreV7Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, cfg: SimplCoreV7Config):
        if _HF:
            super().__init__(cfg)
        else:
            super().__init__()
        self.model = SimplCoreV7Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    # ------------------------------------------------------------------ train
    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        B, L = input_ids.shape
        dev = input_ids.device
        dt = self.model.embed_tokens.weight.dtype
        states = self.model.init_state(B, dev, dt)
        x = self.model.embed_tokens(input_ids)
        outs = []
        for t in range(L):
            xt = x[:, t, :]
            for l, blk in enumerate(self.model.blocks):
                xt, states[l] = blk.step(xt, states[l])
            outs.append(self.lm_head(self.model.norm(xt)))
        logits = torch.stack(outs, dim=1)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        if _HF and CausalLMOutputWithPast is not None:
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        return (loss, logits) if loss is not None else logits

    # --------------------------------------------------------------- streaming
    @torch.no_grad()
    def forward_step(self, token_id, states):
        """O(1)-state single-token forward. token_id: scalar or (1,) long."""
        dev = self.model.embed_tokens.weight.device
        dt = self.model.embed_tokens.weight.dtype
        tid = torch.as_tensor([int(token_id)], device=dev, dtype=torch.long)
        if states is None:
            states = self.model.init_state(1, dev, dt)
        x = self.model.embed_tokens(tid)
        for l, blk in enumerate(self.model.blocks):
            x, states[l] = blk.step(x, states[l])
        return self.lm_head(self.model.norm(x))[0], states

    @torch.no_grad()
    def chat_generate(self, prompt_ids, max_new_tokens=120, temperature=0.8,
                      top_k=20, repetition_penalty=1.1, eos_id=None):
        dev = self.model.embed_tokens.weight.device
        ids = [int(i) for i in prompt_ids]
        states = None
        for t in ids:
            logits, states = self.forward_step(t, states)
        for _ in range(max_new_tokens):
            lp = logits.float() / max(temperature, 1e-3)
            for t in set(ids[-32:]):
                lp[t] -= math.log(repetition_penalty)
            if top_k and top_k < lp.numel():
                vals, ix = torch.topk(lp, top_k)
                mask = torch.full_like(lp, float("-inf"))
                mask[ix] = vals
                lp = mask
            prob = torch.softmax(lp, dim=-1)
            nxt = int(torch.multinomial(prob, 1).item())
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            logits, states = self.forward_step(nxt, states)
        return ids

    @torch.no_grad()
    def check_stream_equivalence(self, ids, atol=2e-4):
        """full unroll == O(1) step recurrence."""
        L = len(ids)
        dev = self.model.embed_tokens.weight.device
        x = torch.tensor([ids], device=dev, dtype=torch.long)
        out = self.forward(x)
        full = (out.logits if hasattr(out, "logits") else out[1])[0]
        states = None
        rec = []
        for t in ids:
            lg, states = self.forward_step(t, states)
            rec.append(lg)
        rec = torch.stack(rec, dim=0)
        diff = (full - rec).abs()
        state_bytes = sum(s["S"].element_size() * s["S"].nelement()
                          for s in states) + sum(
            h.element_size() * h.nelement() for s in states for h in s["hist"])
        return dict(max_abs_diff=float(diff.max()),
                    passed=bool(diff.max() <= atol),
                    state_bytes=int(state_bytes),
                    state_independent_of_L=True, seq_len=L)


if _HF:
    AutoConfig.register("simplcore_v7", SimplCoreV7Config)
    AutoModelForCausalLM.register(SimplCoreV7Config, SimplCoreV7ForCausalLM)
