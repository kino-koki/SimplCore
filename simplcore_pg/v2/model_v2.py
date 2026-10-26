# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCoreV2Model：无词表 + TriMemory 三级记忆的完整模型。

数据流::

    文本 ─GlyphStreamEncoder→ 字符向量 ─EntropyDrivenPooler→ 语义单元 (B,L',H)
        ─N×TriMemoryBlock→ 最终归一化 ─PrototypeDecoder→ 字符分布

训练目标 (next-unit 预测)：
  * 不开池化：逐字符 next-char CE；
  * 开池化：每语义单元预测**下一单元的首字符** (单元级 CE) +
    池化边界 BCE 辅助损失 (对齐全启发式边界)。

推理两条路径：
  * ``generate`` —— 原型级全量重编码循环 (O(L²)，仅用于验证与小规模演示；
    生产增量路径 = ``forward_chunk_runtime`` + 池化在线化，列为 P1 工程项)；
  * ``init_state`` + ``forward_chunk_runtime`` —— **分块流式推理**：
    恒定 KV 环 + 恒定 Delta 状态 + 对数增长八度库，单步显存与上下文无关。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm, init_weights
from .config_v2 import V2Config
from .glyph import GlyphStreamEncoder, EntropyDrivenPooler, PrototypeDecoder
from .octave import OctaveMemory
from .block_v2 import TriMemoryBlock, TriMemoryState


class SimplCoreV2Model(nn.Module):
    """SimplCore v2 完整模型 (无词表，token-free)。"""

    def __init__(self, config: V2Config, charset: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        H = config.hidden_size

        self.encoder = GlyphStreamEncoder(config, charset=charset)
        self.pooler = EntropyDrivenPooler(config)
        self.charset = self.encoder.charset

        # 全模型共享一个八度记忆模块 (读写权重共享, 参数高效)
        self.octave = OctaveMemory(
            H, slots_per_level=config.octave_slots_per_level,
            mem_tokens=config.octave_mem_tokens,
            promote_tokens=config.octave_promote_tokens,
            segment_tokens=config.octave_segment_tokens,
            max_levels=config.octave_max_levels,
            num_heads=config.octave_num_heads, eps=config.rms_norm_eps,
        )
        self.blocks = nn.ModuleList([
            TriMemoryBlock(config, has_octave=((i + 1) % config.octave_every == 0),
                           octave=self.octave)
            for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(H, eps=config.rms_norm_eps)
        self.decoder = PrototypeDecoder(H, config.num_prototypes,
                                        charset=self.charset, eps=config.rms_norm_eps)
        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    # 编码前端
    # ------------------------------------------------------------------
    def encode_text(
        self, texts: Sequence[str], use_pooling: bool = True
    ) -> Tuple[torch.Tensor, Optional[list], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """文本 -> (x (B,L,H), segments, pool_aux, boundary_targets)。"""
        x, mask = self.encoder(texts)
        if not use_pooling:
            return x, None, None, None
        bt = torch.zeros(x.shape[0], x.shape[1])
        for b, t in enumerate(texts):
            hb = self.encoder.heuristic_boundaries(t)
            if hb:
                bt[b, : len(hb)] = torch.tensor(hb, dtype=torch.float32)
        x, segs, pool_aux = self.pooler(x, mask, bt)
        return x, segs, pool_aux, bt

    # ------------------------------------------------------------------
    # 推理核 (训练态全量前向)
    # ------------------------------------------------------------------
    def _run_blocks(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        total_aux = None
        for block in self.blocks:
            x, aux = block(x, pos_offset=pos_offset, training=training)
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        return self.final_norm(x), total_aux

    # ------------------------------------------------------------------
    # 训练前向：next-unit CE
    # ------------------------------------------------------------------
    def forward_train(
        self,
        texts: Sequence[str],
        use_pooling: bool = True,
        boundary_aux_weight: float = 0.1,
    ) -> dict:
        """对一批文本做 next-unit 预测训练。

        Returns:
            dict(loss, ce_loss, pool_aux, moe_aux, num_units)
        """
        x, segs, pool_aux, _bt = self.encode_text(texts, use_pooling=use_pooling)
        h, moe_aux = self._run_blocks(x, training=True)
        logits = self.decoder(h)                              # (B, L', P)

        # ---- 构造目标 ----
        B, Lp, P = logits.shape
        targets = torch.full((B, Lp), -100, dtype=torch.long)
        for b, text in enumerate(texts):
            if use_pooling and segs is not None:
                units = segs[b]
                for u in range(len(units) - 1):
                    nxt_first = text[units[u + 1][0]]         # 下一单元首字符
                    targets[b, u] = self.decoder.char_to_index(nxt_first)
            else:
                for i in range(len(text) - 1):
                    targets[b, i] = self.decoder.char_to_index(text[i + 1])

        ce = F.cross_entropy(
            logits.reshape(B * Lp, P), targets.reshape(B * Lp), ignore_index=-100
        )
        loss = ce
        if pool_aux is not None:
            loss = loss + boundary_aux_weight * pool_aux
        if moe_aux is not None:
            loss = loss + moe_aux
        return {
            "loss": loss, "ce_loss": ce, "pool_aux": pool_aux, "moe_aux": moe_aux,
            "num_units": int((targets >= 0).sum().item()),
        }

    # ------------------------------------------------------------------
    # 流式推理 (分块; 状态显存 = 恒定 + log L)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_state(self, batch: int = 1, device: Optional[torch.device] = None,
                   dtype: torch.dtype = torch.float32) -> TriMemoryState:
        """创建推理状态：每层 KV 环 (恒定) + Delta 状态 (恒定) + 八度库 (log L)。"""
        device = device or next(self.parameters()).device
        state = TriMemoryState()
        for block in self.blocks:
            kv, ds = block.init_layer_states(batch, device, dtype)
            state.kv_caches.append(kv)
            state.delta_states.append(ds)
        state.octave_bank = self.octave.init_bank(device, dtype)
        return state

    @torch.no_grad()
    def forward_chunk_runtime(
        self,
        x: torch.Tensor,
        state: TriMemoryState,
        write_octave: bool = True,
    ) -> torch.Tensor:
        """流式处理一段连续单元 (B=1, n, H)，全部状态原地更新。

        八度库按 ``octave_segment_tokens`` 切段写入 (尾部不足一段也压缩)。
        """
        if x.shape[0] != 1:
            raise ValueError("流式推理路径当前仅支持 batch=1 (八度库为单条记忆)。")
        for i, block in enumerate(self.blocks):
            x = block.forward_runtime(x, state.kv_caches[i],
                                      state.delta_states[i], state.octave_bank)
        h = self.final_norm(x)
        if write_octave:
            seg = self.config.octave_segment_tokens
            for s in range(0, h.shape[1], seg):
                self.octave.write_segment(h[:, s: s + seg], state.octave_bank)
        return h

    @torch.no_grad()
    def prefill_text(self, text: str, state: TriMemoryState,
                     use_pooling: bool = True) -> torch.Tensor:
        """文本 → 编码 → 流式过推理核，返回最后位置的隐状态 (B=1, H)。"""
        x, _segs, _aux, _bt = self.encode_text([text], use_pooling=use_pooling)
        h = self.forward_chunk_runtime(x, state)
        return h[0, -1]

    # ------------------------------------------------------------------
    # 原型级生成 (全量重编码; 仅验证用)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, max_new_chars: int = 32,
                 use_pooling: bool = False) -> str:
        """贪心生成。每步重新编码全文本 —— 正确性优先的验证实现。"""
        text = prompt
        for _ in range(max_new_chars):
            x, _segs, _aux, _bt = self.encode_text([text], use_pooling=use_pooling)
            h, _ = self._run_blocks(x, training=False)
            idx = int(self.decoder.logits(h[0, -1]).argmax(dim=-1).item())
            ch = self.charset[idx] if 0 <= idx < len(self.charset) else ""
            if not ch or ch == "<unk>":
                break
            text += ch
        return text

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"H={self.config.hidden_size}, layers={self.config.num_layers}, "
                f"W={self.config.swa_window}, protos={self.config.num_prototypes}")
