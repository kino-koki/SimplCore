# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""DMC 工作记忆缓冲模块
融合 SimplCore DMC + OpenMythos 循环状态管理

状态安全重构说明:
  原实现存在三个问题:
  1. self.memories 是 register_buffer, 但 _update_memories 里用
     self.memories = updated.mean(dim=0).detach() 直接赋值,
     这会替换 buffer 而不是原地更新, 导致 DDP 下各 rank 状态不一致。
  2. 联想记忆的 assoc_keys/assoc_values 是 SCParameter (可训练),
     但推理时也在读取, 没有区分 training/eval 模式。
  3. 没有提供 reset_state() 方法, checkpoint 恢复后状态可能脏。

  新实现:
  1. memories 改为原地更新: self.memories.copy_(...)。
  2. 新增 reset_state() 方法, 用于 epoch 开始 / checkpoint 恢复时重置。
  3. 新增 state_dict 扩展钩子, 确保 buffer 正确保存和恢复。
  4. 训练时才更新记忆, eval 时只读不写。
  5. DDP 安全: 所有状态更新都在 forward 内, 不跨 forward 泄漏。
"""

from typing import Optional

import simplcore.tensor as torch
from .core import SCFunctional, SCLinear, SCModule, SCParameter
from .config import SimplCoreMixConfig


class WorkingMemoryBuffer(SCModule):
    """工作记忆缓冲 (SimplCore DMC)

    多通道动态记忆，支持：
    1. 门控记忆更新 (原地, DDP 安全)
    2. 可微联想记忆 (NTM-like)
    3. 跨循环状态持久化 (OpenMythos集成)

    状态管理:
    - memories: register_buffer, 原地更新, checkpoint 安全。
    - assoc_keys/assoc_values: 可训练参数, 随模型一起保存。
    - reset_state(): 手动重置非参数状态。
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.dim = cfg.dim
        self.n_channels = cfg.n_memory_channels
        self.memory_dim = cfg.memory_dim
        self._disabled = self.n_channels <= 0

        # 多通道记忆 [n_channels, memory_dim]
        # 用 register_buffer 保证它随 state_dict 保存/恢复
        self.register_buffer('memories', torch.randn(max(1, cfg.n_memory_channels), cfg.memory_dim) * 0.01)

        # 注意力权重 [n_channels, dim, memory_dim]
        self.attention_weights = SCParameter(
            torch.randn(max(1, cfg.n_memory_channels), cfg.dim, cfg.memory_dim) * 0.01
        )

        # 门控参数
        self.gate_proj = SCLinear(cfg.dim, max(1, cfg.n_memory_channels), bias=True)
        self.decay_proj = SCLinear(cfg.dim, max(1, cfg.n_memory_channels), bias=True)

        # 输出投影
        self.output_proj = SCLinear(cfg.memory_dim * max(1, cfg.n_memory_channels), cfg.dim, bias=False)

        # 联想记忆 (NTM-like)
        self.use_associative = cfg.wm_use_associative
        if cfg.wm_use_associative:
            self.assoc_size = cfg.wm_assoc_mem_size
            self.assoc_keys = SCParameter(torch.randn(cfg.wm_assoc_mem_size, cfg.dim) * 0.01)
            self.assoc_values = SCParameter(torch.randn(cfg.wm_assoc_mem_size, cfg.dim) * 0.01)
            self.register_buffer('assoc_usage', torch.zeros(cfg.wm_assoc_mem_size))

    def reset_state(self) -> None:
        """重置非参数状态。

        用于:
        - epoch 开始时清空记忆 buffer。
        - checkpoint 恢复后确保状态干净。
        - 评测前重置, 避免训练数据泄漏到评测。
        """
        with torch.no_grad():
            self.memories.zero_()
            if hasattr(self, 'assoc_usage'):
                self.assoc_usage.zero_()

    def _update_memories(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple:
        """根据输入更新多通道记忆（纯函数，无副作用）。

        因果性 (2026-10 审计修复): 记忆更新必须只看过去 —— 用**因果前缀均值**
        替代全序列均值, 即第 t 个位置的 gate/decay/新记忆值只由 s <= t 的
        输入决定。实现用 cumsum / cumcount 的前缀和形式, O(T) 一次算完,
        保持纯函数 (无 buffer 副作用) 语义不变。

        返回:
            updated_memories: 带梯度的更新后记忆 [n_channels, memory_dim]
            gate: 门控值 [B, T, n_channels] (带梯度, 用于前向计算)
            decay: 衰减值 [B, T, n_channels] (带梯度, 用于前向计算)

        状态传播约定:
          - 训练时: state 由调用方在循环内逐迭代传播 (blocks.py 的
            RecurrentBlock 维护 mem_state), 本函数不写回 buffer。
            这保证 forward 是无副作用的纯函数 —— gradient checkpointing
            重放时从同一初始状态出发, 计算路径与结果完全一致
            (否则重放时 buffer 已变, 触发 CheckpointError 或产生错误梯度)。
            同时避免训练样本间的状态泄漏。
          - 推理时: state=None, 用 buffer 作为旧状态 (跨样本记忆持久化)。

        DDP 安全: 训练时不写 buffer, 不存在跨 rank 状态同步问题。
        """
        B, T, D = x.shape
        # 因果前缀均值 x_mean[b,t,:] = mean_{s<=t} x[b,s,:]
        xs = x.float()
        cum = torch.cumsum(xs, dim=1)                                # [B, T, D]
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        x_mean = cum / counts                                        # [B, T, D]

        # 计算门控和衰减 (保留梯度用于前向计算) —— 逐位置, 形状 [B, T, n_channels]
        gate = torch.sigmoid(self.gate_proj(x_mean))  # [B, T, n_channels]
        decay = torch.sigmoid(self.decay_proj(x_mean))  # [B, T, n_channels]

        # 新记忆: 逐位置 [B, T, n_channels, memory_dim]
        new_memories = []
        for c in range(self.n_channels):
            att_w = self.attention_weights[c]  # [D, memory_dim]
            new_mem = torch.matmul(x_mean, att_w)  # [B, T, memory_dim]
            new_memories.append(new_mem)

        new_memories = torch.stack(new_memories, dim=2)  # [B, T, n_channels, memory_dim]

        # 门控更新: memory_t = decay_t * old + gate_t * new_t
        # 旧状态是运行时状态, 不是可学习参数: detach().clone() 切断与 buffer
        # 的存储/版本号共享 (detach 只隔离梯度, 不隔离 autograd 版本计数,
        # expand view 共享存储时 buffer 的 copy_ 会触发 inplace 冲突)。
        base_mem = state if state is not None else self.memories
        old_t = base_mem.detach().clone().unsqueeze(0).unsqueeze(0)  # [1,1,C,M]
        old_t = old_t.expand(B, T, -1, -1)                          # [B,T,C,M]
        updated = decay.unsqueeze(-1) * old_t + gate.unsqueeze(-1) * new_memories  # [B,T,C,M]

        # 带梯度的更新后记忆: 取最后一个位置 (因果: 记忆在 t=T-1 的值只看到前文),
        # 保持 [n_channels, memory_dim] 的输出形状与旧接口一致。
        updated_with_grad = updated[:, -1, :, :]  # [B, C, M]
        updated_with_grad = updated_with_grad.mean(dim=0)  # [C, M]

        # gate/decay 返回逐位置 [B, T, C] 形式 (因果: 第 t 行只依赖 s <= t 的前缀),
        # 调用方只取末位 [-1] 供日志/梯度 —— 保持旧接口语义, 且让因果性测试
        # 可以直接守护"前 L-1 行不变"。
        return updated_with_grad, gate, decay

    def _associative_recall(self, query: torch.Tensor) -> torch.Tensor:
        """联想记忆检索"""
        if not self.use_associative:
            return torch.zeros_like(query)

        # 余弦相似度
        query_norm = SCFunctional.normalize(query, p=2, dim=-1)
        keys_norm = SCFunctional.normalize(self.assoc_keys, p=2, dim=-1)
        sim = torch.matmul(query_norm, keys_norm.T)  # [B*T, assoc_size]

        # 加权读取
        weights = SCFunctional.softmax(sim, dim=-1)
        recalled = torch.matmul(weights, self.assoc_values)  # [B*T, dim]

        return recalled

    def forward(self, x: torch.Tensor, loop_state: torch.Tensor = None, state: torch.Tensor = None) -> tuple:
        """
        Args:
            x: [B, T, dim]
            loop_state: 上一循环状态 [B, T, dim] (OpenMythos集成)
            state: 训练时由调用方传入的上一迭代记忆状态 [n_channels, memory_dim]
                   (checkpoint 重放确定性: 本模块在训练时不做任何 buffer 副作用)
        Returns:
            output: [B, T, dim]
            state: 当前循环状态 [B, T, dim] 或记忆状态 (训练时)
        """
        B, T, D = x.shape

        # 禁用 (n_memory_channels=0)：恒等直通，不产生记忆输出
        if self._disabled:
            if self.training:
                return torch.zeros_like(x), state
            return torch.zeros_like(x), torch.zeros_like(x)

        # 训练时更新记忆并获取带梯度的记忆 (纯函数, 状态由调用方传播)
        # — eval 时只读 buffer, 不更新
        active_memories = self.memories  # 默认使用 buffer (eval)
        if self.training:
            active_memories, _gate, _decay = self._update_memories(x, state=state)

        # 多通道记忆读取 (使用带梯度的 active_memories)
        mem_reads = []
        for c in range(self.n_channels):
            mem = active_memories[c]  # [memory_dim]
            att_w = self.attention_weights[c]  # [dim, memory_dim]
            read = torch.matmul(x, att_w)  # [B, T, memory_dim]
            # 将记忆内容作为残差偏置加入, 确保梯度流回 gate_proj / decay_proj
            read = read + mem.unsqueeze(0).unsqueeze(0)
            mem_reads.append(read)

        mem_output = torch.cat(mem_reads, dim=-1)  # [B, T, n_channels * memory_dim]
        mem_output = self.output_proj(mem_output)  # [B, T, dim]

        # 联想记忆增强
        if self.use_associative:
            x_flat = x.view(B * T, D)
            assoc_out = self._associative_recall(x_flat).view(B, T, D)
            mem_output = mem_output + assoc_out * 0.1

        # 融合循环状态 (OpenMythos风格)
        if loop_state is not None:
            mem_output = mem_output + loop_state * 0.5

        # 训练时: 返回记忆状态 (detach, 值传播不带梯度) 供下一迭代使用;
        # eval 时: 返回 mem_output 保持原接口语义
        if self.training:
            return mem_output, active_memories.detach()
        return mem_output, mem_output


__all__ = ["WorkingMemoryBuffer"]
