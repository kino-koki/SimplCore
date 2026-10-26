# -*- coding: utf-8 -*-
"""端到端测试：训练步、梯度裁剪、DMC 运行时记忆、NaN 守卫。"""
import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model import SimplCoreModel
from simplcore_pg.memory import MemoryState
from simplcore_pg.exceptions import (
    handle_gradient_clipping, check_tensor_health, GradientCheckpointError,
)


def test_training_step_backward():
    torch.manual_seed(0)
    cfg = SimplCoreConfig.get_nano_config()
    model = SimplCoreModel(cfg)
    opt = model.configure_optimizers(lr=1e-3)

    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), ids.reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    # 梯度健康检查 + 裁剪
    for p in model.parameters():
        if p.grad is not None:
            check_tensor_health(p.grad, name=p.__class__.__name__)
            p.grad = handle_gradient_clipping(p.grad, max_norm=1.0)
    opt.step()
    assert torch.isfinite(loss).all()


def test_overfit_tiny():
    """模型应当能在极小样本上过拟合 (loss 明显下降)。"""
    torch.manual_seed(1)
    cfg = SimplCoreConfig.get_nano_config()
    model = SimplCoreModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))

    losses = []
    for _ in range(25):
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.1


def test_dmc_runtime_memory_in_block():
    from simplcore_pg.block import SimplCoreBlock
    cfg = SimplCoreConfig.get_nano_config()
    block = SimplCoreBlock(cfg)
    st = MemoryState(cfg.num_channels, cfg.channel_size,
                     cfg.hidden_size // cfg.dmc_num_heads, device=torch.device("cpu"))
    x = torch.randn(1, 6, cfg.hidden_size)
    y = block(x, memory=st, training=False)
    assert y.shape == x.shape


def test_gradient_clip_after_nan_raises():
    grads = [torch.randn(3), torch.tensor([float("nan")])]
    raised = False
    for g in grads:
        try:
            handle_gradient_clipping(g, max_norm=1.0)
        except GradientCheckpointError:
            raised = True
    assert raised
