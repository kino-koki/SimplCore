# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""held-out 评测循环（bf16 autocast，token 加权 loss）。"""
from __future__ import annotations



import math
from typing import Iterator, Optional

import torch

from .data_windows import unpack_lm_batch



# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: Optional[int] = None,
             chunk_size: int = 0) -> dict:
    """Token-weighted validation loss / perplexity.

    ``chunk_size`` > 0 时走分块前向（与训练同一套跨块状态，激活 O(chunk)），
    用于超长验证窗口；0 保持整段前向的旧行为。
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    ncm_sums = {}
    ncm_counts = {}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, labels, groups = unpack_lm_batch(batch)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if labels is not None else None
        groups = groups.to(device, non_blocking=True) if groups is not None else None
        valid_tokens = int((y != -100).sum().item())
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
            else torch.autocast("cpu", enabled=False)
        with ctx:
            if getattr(model, "ncm", None) is not None and chunk_size > 0:
                state = model.init_stream(
                    x.shape[0], device=x.device, dtype=model.tok_emb.weight.dtype)
                batch_nll = 0.0
                for pos in range(0, x.shape[1], chunk_size):
                    end = min(pos + chunk_size, x.shape[1])
                    result = model.forward_chunk_step(
                        x[:, pos:end], y[:, pos:end], state,
                        ncm_label_targets=(labels[:, pos:end]
                                           if labels is not None else None),
                        ncm_group_targets=(groups[:, pos:end]
                                           if groups is not None else None),
                        return_ncm_aux=True)
                    _, loss_sum, _, state, aux = result
                    batch_nll += float(loss_sum)
                    _accumulate_ncm_metrics(ncm_sums, ncm_counts, aux)
                loss = batch_nll / max(valid_tokens, 1)
            elif getattr(model, "ncm", None) is not None:
                _, combined_loss, aux = model(
                    x, y, ncm_label_targets=labels,
                    ncm_group_targets=groups, return_ncm_aux=True)
                loss = combined_loss - aux["ncm_loss"]
                _accumulate_ncm_metrics(ncm_sums, ncm_counts, aux)
            elif chunk_size > 0 and hasattr(model, "forward_chunked"):
                _, loss = model.forward_chunked(
                    x, y, chunk_size=chunk_size, detach_state=False)
            else:
                _, loss = model(x, y)
        total_nll += float(loss) * valid_tokens
        total_tokens += valid_tokens
    model.train()
    mean = total_nll / max(total_tokens, 1)
    result = {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": total_tokens}
    if ncm_counts:
        result["ncm_aux"] = {
            name: ncm_sums[name] / max(ncm_counts[name], 1)
            for name in ncm_counts
        }
    return result


def _accumulate_ncm_metrics(sums: dict, counts: dict, aux: dict) -> None:
    for source in ("future_latent", "concept_labels", "token_groups"):
        count = int(aux.get(f"{source}_count", 0))
        loss_sum = aux.get(f"{source}_loss_sum")
        if count and loss_sum is not None:
            sums[source] = sums.get(source, 0.0) + float(loss_sum.detach())
            counts[source] = counts.get(source, 0) + count
