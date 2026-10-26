# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor DDP — 自研分布式数据并行（零 torch.nn.parallel 依赖）
=======================================================================

兼容 torch.nn.parallel.DistributedDataParallel 的最小实现：

  * 前向传播：直接调用包装模块（模型并行 / 张量并行在 backend 内完成）
  * 反向传播：在 backward() 后对梯度做 all-reduce（均值），保证多卡一致
  * 支持 find_unused_parameters 提示（原生引擎忽略，保留接口兼容）

注意：真正的张量并行 (Tensor Parallelism) 与流水线并行 (Pipeline Parallelism)
由 SimplCore 架构层的 SimplCoreBlock / topology_routing 在模型内部完成；
本 DDP 仅负责跨 rank 的数据并行梯度同步。
"""
from __future__ import annotations

from typing import Any, Optional

from ..tensor import Tensor, no_grad
from .. import distributed as dist


class DistributedDataParallel:
    """兼容 torch.nn.parallel.DistributedDataParallel。"""

    def __init__(
        self,
        module,
        device_ids: Optional[list] = None,
        output_device: Optional[Any] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
    ) -> None:
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers
        self.process_group = process_group
        self.find_unused_parameters = find_unused_parameters
        # 在分布式初始化时广播参数（保证多卡起点一致）
        if dist.is_initialized() and broadcast_buffers:
            self._broadcast_parameters()

    # ---- 参数广播 ----
    def _broadcast_parameters(self) -> None:
        rank = dist.get_rank(self.process_group)
        for p in self._params():
            if isinstance(p, Tensor) and p.data is not None:
                dist.broadcast(p, src=0, group=self.process_group)

    def _params(self):
        if hasattr(self.module, "parameters"):
            return list(self.module.parameters())
        return []

    # ---- 前向 ----
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---- 反向 + 梯度同步 ----
    def backward(self, loss: Tensor) -> None:
        """反向传播并对梯度做 all-reduce 均值同步。"""
        loss.backward()
        self._sync_gradients()

    def _sync_gradients(self) -> None:
        """对模型所有参数的梯度做 all-reduce（sum 后除以 world_size）。"""
        if not dist.is_initialized():
            return
        world = dist.get_world_size(self.process_group)
        if world <= 1:
            return
        for p in self._params():
            if isinstance(p, Tensor) and p.grad is not None:
                dist.all_reduce(p.grad, op="sum", group=self.process_group)
                # 转为均值（等价 DDP 的梯度平均语义）
                p.grad.data = p.grad.data / world

    # ---- 属性透传 ----
    @property
    def parameters(self):
        return self.module.parameters

    @property
    def named_parameters(self):
        return self.module.named_parameters

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        # 透传未在 DDP 上定义的属性到内部 module
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "module"), name)


__all__ = ["DistributedDataParallel"]
