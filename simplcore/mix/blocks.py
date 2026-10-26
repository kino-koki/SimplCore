# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 核心Block模块 — 三家融合版
Prelude / Recurrent Block / Coda

融合来源:
  OpenMythos: LTIInjection (谱半径<1保证) + LoRAAdapter (per-loop scale) 
              + loop_index_embedding + ACTHalting (remainder trick)
  SimplCore:  可调制基向量注意力 + 原型路由MoE + 工作记忆缓冲 + Hebbian在线适应
  Mix创新:    置信度门控 + 推理集成 + 组件间协同优化
"""
import simplcore.tensor as torch
from typing import Optional
from .core import (
    SCDropout,
    SCEmbedding,
    SCLinear,
    SCModule,
    SCParameter,
    SCReLU,
    SCRMSNorm,
    SCSequential,
    SCSigmoid,
)
from .utils import precompute_rope_freqs, apply_rope, get_causal_mask
from .attention import ModulatedBasisAttention, ModulatedBasisMLAttention
from .fusion import (
    StructuredGatedIntervention,
    SimplCoreGatedIntervention,
    GatedCompression,
    LowRankProjection,
    ResidualAdaptationModule,
)
from .moe import PrototypeMoE, Expert
from .memory import WorkingMemoryBuffer
from .config import SimplCoreMixConfig


# ---------------------------------------------------------------------------
# loop_index_embedding (OpenMythos)
# ---------------------------------------------------------------------------

def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """注入循环索引正弦信号到隐藏状态前loop_dim个通道 (OpenMythos)

    类似于RoPE对序列位置的作用，但应用于循环深度而非token位置。
    使共享权重的循环块在不同循环中执行功能上不同的操作。

    Args:
        h:        [B, T, dim] 隐藏状态
        loop_t:   当前循环索引 (0-based)
        loop_dim: 接收嵌入的通道数 (必须为偶数)
        theta:    正弦基频率
    Returns:
        h加上正弦偏置后的张量，形状不变
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # [loop_dim//2]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# LTIInjection — 数值稳定的线性时不变注入 (OpenMythos增强版)
# ---------------------------------------------------------------------------

class LTIInjection(SCModule):
    """LTI稳定输入注入 (OpenMythos Parcae, Prairie et al., 2026)

    循环隐藏状态演化:
        h_{t+1} = A · h_t + B · e + Transformer(h_t, e)

    通过ZOH离散化保证 ρ(A) < 1:
        A_continuous = Diag(-exp(log_A))       始终负定
        A_discrete   = exp(Δt · A_continuous)  逐元素，值在(0,1)

    log_A和log_dt为可学习参数，exp保证正值。
    在log空间计算避免0*inf=NaN，clamp保证float32有限。

    相比原版ParcaeStability的改进:
    - log空间计算，数值更稳定
    - 学习log_dt参数，自适应离散化步长
    - 更新规则包含transformer输出
    - clamp防止极端梯度
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.dim = cfg.dim
        self.log_A = SCParameter(torch.zeros(cfg.dim))
        self.log_dt = SCParameter(torch.tensor(cfg.lti_log_dt_init))
        self.B = SCParameter(torch.ones(cfg.dim) * cfg.lti_B_init)

    def get_A(self) -> torch.Tensor:
        """获取离散化A矩阵对角线，保证所有值在(0,1)"""
        # log空间: dt * A_c = -exp(log_dt + log_A)
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """h_{t+1} = A·h_t + B·e + transformer_out

        Args:
            h:               [B, T, dim] 当前隐藏状态
            e:               [B, T, dim] 编码输入 (Prelude输出，跨循环冻结)
            transformer_out: [B, T, dim] 当前循环的Transformer块输出
        Returns:
            [B, T, dim] 更新后的隐藏状态
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# LoRAAdapter — 深度自适应LoRA (OpenMythos)
# ---------------------------------------------------------------------------

class LoRAAdapter(SCModule):
    """深度自适应LoRA适配器 (OpenMythos, Bae et al., 2024)

    共享低秩down/up投影 + 每循环独立的scale向量。
    delta(x, t) = (down(x) * scale[t]) @ B

    支持深度外推: 推理时n_loops可超过训练max_loop_iters，
    超出范围的循环复用最后一个学习的scale。
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = SCLinear(dim, rank, bias=False)
        self.B = SCParameter(torch.randn(rank, dim) * 0.02)
        self.scale = SCEmbedding(max_loops, rank)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """计算LoRA delta，加到块输出上"""
        max_t = self.scale.num_embeddings - 1
        t_idx = min(loop_t, max_t)
        # 使用 arange 创建索引张量 (兼容双后端)
        s = self.scale(torch.tensor([t_idx], device=x.device, dtype=torch.int64).squeeze())
        down = self.down(x) * s  # [B, T, rank]
        return down @ self.B  # [B, T, dim]


# ---------------------------------------------------------------------------
# ACTHalting — 自适应计算时间停止 (OpenMythos)
# ---------------------------------------------------------------------------

class ACTHalting(SCModule):
    """ACT停止机制 (OpenMythos, Graves, 2016)

    每个循环迭代学习每位置的停止概率。
    收敛的位置(高累积停止概率)停止累积更新，
    仍在细化的位置继续计算。
    使用remainder trick: 累积概率+当前概率超过阈值时，
    将剩余概率质量作为最终权重。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.halt = SCLinear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """预测每位置的停止概率

        Args:
            h: [B, T, dim]
        Returns:
            [B, T] 停止概率，值在(0,1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# PreludeBlock — 前奏层 (SwiGLU + RFP注意力)
# ---------------------------------------------------------------------------

class PreludeBlock(SCModule):
    """前奏层：标准Transformer层，执行一次

    使用SwiGLU FFN (OpenMythos) + RFP注意力 (SimplCore)
    """
    def __init__(self, cfg: SimplCoreMixConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm1 = SCRMSNorm(cfg.dim)
        if cfg.attn_type == "mba_mla":
            self.attn = ModulatedBasisMLAttention(cfg)
        elif cfg.attn_type == "mska":
            from .mska import MultiScaleKernelAttention
            self.attn = MultiScaleKernelAttention(cfg)
        else:
            self.attn = ModulatedBasisAttention(cfg)

        # SwiGLU FFN (OpenMythos风格)
        self.norm2 = SCRMSNorm(cfg.dim)
        self.ffn = Expert(cfg.dim, cfg.dim * 4 // 3)
        self.dropout = SCDropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask, kv_cache=None):
        # 注意力 + 残差
        h = self.norm1(x)
        h = self.attn(h, freqs_cis, mask, kv_cache, f"prelude_{self.layer_idx}")
        x = x + self.dropout(h)

        # SwiGLU FFN + 残差
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ---------------------------------------------------------------------------
# RecurrentBlock — 三家融合核心循环块
# ---------------------------------------------------------------------------

class RecurrentBlock(SCModule):
    """循环块：核心推理层，循环T次 — 三家融合

    每次循环迭代 t:
        1. loop_index_embedding: 注入循环索引正弦信号 (OpenMythos)
        2. 组合归一化: norm(h_loop + e) (OpenMythos)
        3. RFP注意力: 基向量增强注意力 (SimplCore)
        4. LoRAAdapter: 深度自适应LoRA delta (OpenMythos)
        5. CCR-MoE: 原型路由混合专家 (SimplCore + OpenMythos MoE)
        6. 工作记忆缓冲: 多通道记忆+联想检索 (SimplCore)
        7. Hebbian在线适应: Hebbian推理时学习 (SimplCore)
        8. LTIInjection: 稳定更新 h = A·h + B·e + transformer_out (OpenMythos)
        9. ACTHalting: 累积停止概率+加权求和 (OpenMythos)

    特性:
    - 编码输入e在每步注入，防止跨循环漂移
    - ACT产生跨迭代隐藏状态的加权和
    - 推理时增加循环数 = 更深推理链 (深度外推)
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.act_threshold = cfg.act_threshold

        # 注意力层 (SimplCore MBA / MSKA)
        self.norm1 = SCRMSNorm(cfg.dim)
        if cfg.attn_type == "mba_mla":
            self.attn = ModulatedBasisMLAttention(cfg)
        elif cfg.attn_type == "mska":
            from .mska import MultiScaleKernelAttention
            self.attn = MultiScaleKernelAttention(cfg)
        else:
            self.attn = ModulatedBasisAttention(cfg)

        # MoE FFN (SimplCore CCR + OpenMythos MoE)
        self.norm2 = SCRMSNorm(cfg.dim)
        self.moe = PrototypeMoE(cfg)

        # 工作记忆缓冲 (SimplCore)
        self.dmc = WorkingMemoryBuffer(cfg)
        self.norm_dmc = SCRMSNorm(cfg.dim)

        # SGI 结构化门控干预 (SimplCore核心创新)
        # 实现: 分组有向注意力 + 可微Do干预 + 后门混杂调整
        self.structured_intervention = StructuredGatedIntervention(
            dim=cfg.dim,
            n_groups=cfg.cin_n_groups,
            top_k_parents=cfg.cin_top_k_parents,
            scale=cfg.cin_scale,
            use_confounder=cfg.cin_use_confounder,
        )

        # Java SimplCore 语义融合组件 (轻量版)
        fusion_dim = max(4, cfg.dim // 4)
        self.feature_compression = GatedCompression(cfg.dim, fusion_dim)
        self.feature_projection = LowRankProjection(cfg.dim, fusion_dim)
        self.residual_adaptation = ResidualAdaptationModule(cfg.dim)

        # LTI稳定注入 (OpenMythos)
        self.injection = LTIInjection(cfg)

        # LoRA深度适配 (OpenMythos)
        if cfg.loop_lora_enabled:
            self.lora_attn = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
            self.lora_moe = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

        # ACT停止 (OpenMythos)
        self.act = ACTHalting(cfg.dim)

        # 循环索引嵌入维度 (OpenMythos)
        self.loop_dim = max(2, int(cfg.dim * cfg.loop_dim_frac))

        # Java SimplCore 组件语义别名 (CIN/IB/MRD/PUM) —— 供外部/测试按旧命名访问
        self.causal_intervention = self.structured_intervention   # CIN: 结构化门控干预
        self.information_bottleneck = self.feature_compression    # IB:  门控压缩
        self.meta_distiller = self.feature_projection             # MRD: 低秩特征投影
        self.plasticity_update = self.residual_adaptation         # PUM: 残差适应模块

        # 组合归一化 (OpenMythos)
        self.norm_combined = SCRMSNorm(cfg.dim)

        # Hebbian在线适应 (SimplCore)
        # hebbian_weights 作为 buffer 而非 parameter:
        #   - 它通过 Hebbian 规则更新, 不参与梯度反传。
        #   - 作为 buffer 保证 checkpoint 保存/恢复。
        #   - 原地更新 (copy_) 保证 DDP 安全。
        if cfg.hebbian_enabled:
            self.hebbian_lr = cfg.hebbian_lr
            self.hebbian_decay = cfg.hebbian_decay
            self.register_buffer(
                "hebbian_weights", torch.eye(cfg.dim) * 0.001
            )

        self.dropout = SCDropout(cfg.dropout)

    def reset_state(self) -> None:
        """重置非参数状态 (SPM 可塑性矩阵、DMC 记忆)。

        用于:
        - epoch 开始时重置为初始状态。
        - checkpoint 恢复后确保状态干净。
        - 评测前重置, 避免训练状态泄漏到评测。
        """
        with torch.no_grad():
            if self.cfg.hebbian_enabled:
                # 重置为接近零的小矩阵, 保持初始语义
                self.hebbian_weights.copy_(torch.eye(self.dim, device=self.hebbian_weights.device) * 0.001)
            if hasattr(self.dmc, "reset_state"):
                self.dmc.reset_state()

    def _apply_hebbian_update(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        hebb_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Hebbian在线适应：Hebbian学习规则 (SimplCore) —— 纯函数, 无副作用。

        Args:
            x:  [B, T, dim] 输入 (循环块组合输入)
            out: [B, T, dim] 当前循环迭代输出
            hebb_state: 训练时由调用方传入的上一迭代可塑性状态 [dim, dim];
                        None 时使用 buffer (eval 路径)
        Returns:
            result: [B, T, dim] 施加可塑性后的输出
            new_state: 训练时的下一迭代状态 [dim, dim] (detach)

        状态传播约定:
          - 训练时: 状态在循环内局部传播, 不写回 hebbian_weights buffer。
            这保证 forward 是无副作用的纯函数 —— gradient checkpointing
            重放时从同一初始状态出发, 计算路径与结果完全一致
            (否则重放时 buffer 已变, 触发 CheckpointError 或产生错误梯度)。
            同时避免训练样本间的状态泄漏, DDP 下无跨 rank 状态同步问题。
          - 推理时: 用 buffer 并原地更新 (推理时在线适应, 原设计语义)。

        数值安全:
          hebbian_weights 参与 forward 计算 (matmul), 不能先更新再计算
          (autograd 会因 inplace 修改报错)。正确顺序:
            1. 用当前状态 detach 副本计算 plastic_out (保留计算图)。
            2. no_grad 下计算 Hebbian 更新量。
            3. 返回新状态, 由调用方 detach 传播 (不写 buffer)。
        """
        if not self.cfg.hebbian_enabled:
            return out, hebb_state

        if not self.training:
            # 推理时应用 hebbian_weights 并在线更新 buffer (保持原语义)
            plastic_out = torch.matmul(out, self.hebbian_weights.T)
            return out + plastic_out * 0.1, hebb_state

        base = hebb_state if hebb_state is not None else self.hebbian_weights
        B, T, D = out.shape
        # 因果前缀均值 (2026-10 审计修复): 第 t 个位置的 Hebbian 统计只由
        # s <= t 的 (out_s, x_s) 对构成, 全序列均值会把未来 token 的信号混进
        # 过去位置的权重更新。cumsum / cumcount 的前缀和形式, O(T)。
        x_mean = self._causal_prefix_mean(x)     # [B, T, D]
        out_mean = self._causal_prefix_mean(out)  # [B, T, D]

        # 1. 用当前可塑性状态的 detach 副本计算输出 (保留 autograd)
        hebbian_weights_current = base.detach().clone()
        plastic_out = torch.matmul(out, hebbian_weights_current.T)
        result = out + plastic_out * 0.1

        # 2. 计算 Hebbian 更新量 (no_grad, 不影响计算图), 返回新状态
        #    因果前缀统计: 逐位置外积取标量 (sum_t x_t * out_t), 再按时间维
        #    做前缀累积 sum_{s<=t}; Oja 项同理。末位 (t=T-1) 的累积值就是
        #    下一迭代的可塑性状态 —— 它只依赖全序列过去, 满足因果性。
        #    所有中间张量保持 [B, T] 形状 (NumPy 引擎无 4D bmm), reshape 到
        #    3D 走 matmul 批量计算。
        with torch.no_grad():
            B_t, T_t, D_t = out_mean.shape
            out_flat = out_mean.reshape(B_t * T_t, 1, D_t)      # [B*T, 1, D]
            x_flat = x_mean.reshape(B_t * T_t, D_t, 1)          # [B*T, D, 1]
            hebbian_t = torch.matmul(out_flat, x_flat)          # [B*T, 1, 1]
            hebbian_t = hebbian_t.reshape(B_t, T_t).mean(dim=0)  # [T]
            hebbian = hebbian_t.cumsum(dim=0)                   # [T]
            # Oja 项: 逐通道 (D, D) 形状, 取末位因果前缀值 (t=T-1)
            oja_term = (out_mean[:, -1, :] ** 2).mean(dim=0, keepdim=True)  # [D] 逐通道
            oja = self.hebbian_decay * hebbian_weights_current * oja_term    # [D, D]
            new_w = hebbian_weights_current + self.hebbian_lr * (
                hebbian[-1] - oja
            )

        return result, new_w.detach()

    @staticmethod
    def _causal_prefix_mean(x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D], 第 t 行 = mean_{s<=t} x[b,s,:]。"""
        B, T, D = x.shape
        cum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        return cum / counts

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """运行循环块，最多n_loops次迭代，带ACT提前停止

        Args:
            h:         [B, T, dim] Prelude输出的初始隐藏状态
            e:         [B, T, dim] 编码输入，每步注入
            freqs_cis: 预计算RoPE频率
            mask:      因果掩码或None
            n_loops:   循环次数，默认cfg.max_loop_iters
            kv_cache:  KV缓存字典
        Returns:
            ACT加权的隐藏状态跨迭代求和，[B, T, dim]
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        # ACT状态
        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        # 训练时运行时状态在循环内局部传播 (checkpoint 重放确定性, 无副作用):
        # - mem_state: DMC 记忆状态 [n_channels, memory_dim]
        # - hebb_state: 可塑性状态 [dim, dim]
        # None = 使用各自 buffer 的初始值 (首迭代)
        mem_state = None
        hebb_state = None

        for t in range(n_loops):
            # 1. 循环索引嵌入 (OpenMythos)
            h_loop = loop_index_embedding(h, t, self.loop_dim)

            # 2. 组合归一化: norm(h_loop + e) (OpenMythos)
            combined = self.norm_combined(h_loop + e)

            # 3. RFP注意力 (SimplCore)
            attn_out = self.attn(
                self.norm1(combined), freqs_cis, mask,
                kv_cache, f"recurrent_{t}"
            )
            # 4. LoRA深度适配 (OpenMythos)
            if self.cfg.loop_lora_enabled:
                attn_out = attn_out + self.lora_attn(attn_out, t)
            h_attn = combined + self.dropout(attn_out)

            # 5. CCR-MoE (SimplCore + OpenMythos)
            moe_out = self.moe(self.norm2(h_attn))
            if self.cfg.loop_lora_enabled:
                moe_out = moe_out + self.lora_moe(moe_out, t)
            h_moe = h_attn + self.dropout(moe_out)

            # 6. 工作记忆缓冲 (SimplCore) — 训练时状态在循环内传播
            mem_out, mem_state = self.dmc(
                self.norm_dmc(h_moe),
                loop_state=h if t > 0 else None,
                state=mem_state,
            )
            h_mem = h_moe + self.dropout(mem_out)

            # 7. 结构化门控干预: 分组有向注意力→可微干预→混杂调整
            h_mem = self.structured_intervention(h_mem, combined)
            h_mem = self.feature_compression(h_mem)
            h_mem = self.feature_projection(h_mem)
            h_mem = self.residual_adaptation(h_mem)

            # 8. SPM可塑性 (SimplCore) — 训练时状态在循环内传播
            h_mem, hebb_state = self._apply_hebbian_update(combined, h_mem, hebb_state)

            # 9. LTI稳定注入: h = A·h + B·e + transformer_out (OpenMythos)
            h = self.injection(h, e, h_mem)

            # 10. ACT停止 + 加权求和 (OpenMythos)
            p = self.act(h)  # [B, T]

            # 使用直通估计器: 前向用硬掩码, 反向用软梯度
            hard_halted = (cumulative_p >= self.act_threshold)
            soft_halted = torch.sigmoid((cumulative_p - self.act_threshold) * 10.0)
            # STE: 前向=hard, 反向=soft
            halted_mask = hard_halted.detach().float() + soft_halted - soft_halted.detach()
            still_running = 1.0 - halted_mask

            # remainder trick: 超过阈值时分配剩余概率质量
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running

            # 提前退出检查 (用 detach 的硬掩码, 不影响梯度)
            hard_halted_now = hard_halted | (cumulative_p.detach() >= self.act_threshold)
            # 训练时也允许物理截断 —— 这是本架构的数值稳定器:
            # 循环闭环增益 >1 (h 范数每迭代约 x2), 深循环不截断必然指数爆炸
            # (实测 loops=16 时 t>=8 后 h 范数 -> inf, loss=nan)。
            # gradient checkpointing 确定性由"循环内状态局部传播"保证
            # (DMC/Hebbian 训练时无副作用, 重放从同一初始状态出发,
            #  break 时机与原始 forward 完全一致, 不会 CheckpointError)。
            # 注意: DDP 下各 rank 若数据不同, break 时机可能不一致 ——
            # 多机训练建议固定 n_loops=max_loop_iters 并关闭 ACT。
            if hard_halted_now.all() and kv_cache is None:
                break
            halted = hard_halted_now

        return h_out


# ---------------------------------------------------------------------------
# CodaBlock — 尾声层 (SwiGLU + 置信度门控)
# ---------------------------------------------------------------------------

class CodaBlock(SCModule):
    """尾声层：标准Transformer层 + 置信度门控

    使用SwiGLU FFN (OpenMythos) + RFP注意力 (SimplCore)
    + 置信度门控 (Mix创新)

    命名说明:
      原名 "置信度门控" 容易暗示已具备事实性验证能力。
      实际上这里只是一个 learned confidence gate: 对隐藏状态做
      二值门控, 低置信度位置被抑制。这能降低低质量 token 的
      输出概率, 但不等同于真正的事实性置信度门控。
      真正的事实性评测需要外部知识库、检索增强、引用约束等,
      由 evaluation.py 的 FactualEvaluator 提供。
      这里更名为 "置信度门控" 以准确反映其能力边界。
    """

    def __init__(self, cfg: SimplCoreMixConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm1 = SCRMSNorm(cfg.dim)
        if cfg.attn_type == "mba_mla":
            self.attn = ModulatedBasisMLAttention(cfg)
        elif cfg.attn_type == "mska":
            from .mska import MultiScaleKernelAttention
            self.attn = MultiScaleKernelAttention(cfg)
        else:
            self.attn = ModulatedBasisAttention(cfg)

        # SwiGLU FFN (OpenMythos)
        self.norm2 = SCRMSNorm(cfg.dim)
        self.ffn = Expert(cfg.dim, cfg.dim * 4 // 3)

        # 置信度门控 (Mix创新)
        # 保留 gate_enabled / confidence_gate 字段名以兼容已有 checkpoint
        self.gate_enabled = cfg.confidence_gate_enabled
        if cfg.confidence_gate_enabled:
            self.confidence_gate = SCSequential(
                SCLinear(cfg.dim, cfg.dim // 4),
                SCReLU(),
                SCLinear(cfg.dim // 4, 1),
                SCSigmoid(),
            )

        self.dropout = SCDropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask, kv_cache=None):
        h = self.norm1(x)
        h = self.attn(h, freqs_cis, mask, kv_cache, f"coda_{self.layer_idx}")
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        # 置信度门控：低置信度位置抑制输出 (使用直通估计器保持梯度)
        if self.gate_enabled:
            confidence = self.confidence_gate(x)  # [B, T, 1]
            # STE: 前向用硬门控, 反向用软梯度
            hard_gate = (confidence > self.cfg.confidence_threshold).float()
            soft_gate = torch.sigmoid(
                (confidence - self.cfg.confidence_threshold) * 10.0
            )
            gate = hard_gate.detach() + soft_gate - soft_gate.detach()
            x = x * gate + x * (1 - gate) * 0.1

        return x
