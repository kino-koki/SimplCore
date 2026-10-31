# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""held-out 评测循环（bf16 autocast，token 加权 loss）。"""
from __future__ import annotations



import math
from typing import Iterator, Optional

import torch




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
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
            else torch.autocast("cpu", enabled=False)
        with ctx:
            if chunk_size > 0 and hasattr(model, "forward_chunked"):
                _, loss = model.forward_chunked(
                    x, y, chunk_size=chunk_size, detach_state=False)
            else:
                _, loss = model(x, y)
        n = y.numel()
        total_nll += float(loss) * n
        total_tokens += n
    model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": total_tokens}

