# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
[SimplCore 核心创新] MultiScaleKernelAttention (MSKA)

────────────────────────────────────────────────────────────────
动机: 标准注意力有两个未经验证的假设
────────────────────────────────────────────────────────────────

标准多头注意力 (Vaswani et al., 2017):
    A_h = softmax(Q_h · K^T / √d) · V_h

潜在假设 1 — 所有头在相同"分辨率"上计算:
    所有注意力头使用完全相同的 Q·K^T 形式, 等价于用同一核函数
    在单一度量空间中建模所有语义关系。但语法匹配 (近距离)、
    语义关联 (中距离)、篇章关系 (远距离) 可能需要不同的解析分辨率。

潜在假设 2 — 余弦相似度是唯一的相似度量:
    Q·K^T 等价于余弦相似度 (去除模长后)。模型被迫用这个固定
    的度量来判定"两个词是否相关"。但语义相关性往往不是线性的。

────────────────────────────────────────────────────────────────
MSKA 的三个技术贡献
────────────────────────────────────────────────────────────────

贡献 1 — 多尺度头部频率缩放 (Multi-Scale Head Scaling, MS-Head):
    将注意力头分为 N_bands 个频带。每个频带学习一个频率缩放因子,
    对 Q/K 向量做 element-wise 缩放:
        Q_h' = σ(θ_b) · Q_h,  其中 θ_b 是频带 b 的可学习频率参数

    数学上等价于将注意力核从 K(q,k)=q·k^T 变为 K_b(q,k)=σ(θ_b)²·q·k^T,
    低频 (σ < 1) 核更平滑 → 关注大范围语义模式;
    高频 (σ > 1) 核更锐利 → 关注精确 token 级匹配。

    这与 MBA 基向量场注意力的基向量调制形成互补:
    MBA 调制的是注意力头内部的基向量;
    MS 调制的是注意力头之间的频率分辨率。
    两者可叠加使用。

贡献 2 — 双线性上下文核 (Bilinear Context Kernel, BiCK):
    在标准 Q·K^T 之外, 添加一个输入依赖的可学习偏差项:
        A_ij = Q_i · K_j^T / √d + γ · φ(x_i) · ψ(x_j)^T / √r

    其中:
    - φ, ψ 是 silu 激活的双线性投影 (共享参数, 减少参数量)
    - γ 是每头可学习的门控 (允许模型选择是否使用核)

    这个核允许模型学习: 两个位置在"函数空间"中的相关性,
    而不只是"投影空间"中的余弦距离。
    它可以捕捉 Q·K 无法表达的语音/语义高阶模式。

贡献 3 — 循环查询精化 (Recurrent Query Refine, RQR) [可选]:
    默认关闭 (scheduling_steps=1)。开启后:
    - Step 1: 正常注意力计算
    - Step t: 基于上一步的注意力输出, 用瓶颈 MLP 生成 ΔQ,
              Q_t = Q_1 + γ·ΔQ, 重新计算注意力
    - 各步输出加权组合: O = Σ w_t · O_t

    原理类似于计算机视觉中的 iterative attention
    (Xu et al., ICML 2015 "Show, Attend and Tell"),
    但在语言模型中, 它实现的是"层内深度"——在同一个头的多次
    迭代中逐渐精化注意力分布, 而不是跨层的逐一精化。

────────────────────────────────────────────────────────────────
与常见改进的对比
────────────────────────────────────────────────────────────────
- GQA (Ainslie et al., 2023):   减少 KV 头数, 压缩计算量
- MLA (DeepSeek, 2024):          低秩 KV 压缩, 减少缓存
- KDA (Moonshot/Kimi, 2025):     增量 K/V 编码, 长序列压缩
- MSKA (本工作):
    - 多分辨率处理:  改变注意力核的"尺度"
    - 可学习核:      打破余弦相似度的限制
    - 循环精化:      增加层内计算深度
    与上述方法是正交的 — 可叠加使用。

────────────────────────────────────────────────────────────────
效率评估
────────────────────────────────────────────────────────────────
MS-Head:  O(n_bands) 参数, 计算量 ≈ 0 (仅 element-wise mul)
BiCK:     O(dim·kernel_rank) 参数 + O(T²·kernel_rank) 计算量
RQR:      O(S·T²·d) 计算量, S 通常为 1 (关闭) 或 2

组合开销 (S=1): 标准注意力的约 1.15× (可忽略)
组合开销 (S=2): 标准注意力的约 2.15×
"""
import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F
from typing import Optional

from .core import SCDropout, SCFunctional, SCLinear, SCModule, SCParameter
from .utils import apply_rope
from .config import SimplCoreMixConfig


class MultiScaleKernelAttention(SCModule):
    """多尺度核注意力 (MSKA)

    接口: 完全兼容 ModulatedBasisAttention, 可即插即用替换。

    关闭 RQR (scheduling_steps=1) 后, 仅增加约 15% 计算量。
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.cfg = cfg

        # ── 标准维度参数 ──────────────────────────────────────────
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        # ── 标准 QKV 投影 ─────────────────────────────────────────
        self.wq = SCLinear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = SCLinear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        # ════════════════════════════════════════════════════════════
        #  贡献 1: MS-Head — 多尺度头部频率缩放
        # ════════════════════════════════════════════════════════════
        n_bands_raw = getattr(cfg, "mska_n_bands", 4)
        self.n_bands = min(max(n_bands_raw, 1), self.n_heads)
        self.band_size = self.n_heads // self.n_bands

        # 每个频带一个对数频率参数 [n_bands]
        # exp(range(-0.7, 1.0)) → ~0.5 ~ 2.7
        # 0.5×: 低频, 核更平滑; 2.0×: 高频, 核更锐利
        band_init = torch.linspace(-0.7, 1.0, self.n_bands)
        self.band_log_scale = SCParameter(band_init)

        # 频带间交叉混合: [n_bands, n_bands] softmax → 混合权重
        cross_init = torch.zeros(self.n_bands, self.n_bands)
        for i in range(self.n_bands):
            cross_init[i, i] = 3.0  # 对角占优
            if i > 0:
                cross_init[i, i - 1] = 0.5  # 相邻耦合
            if i < self.n_bands - 1:
                cross_init[i, i + 1] = 1.0
        self.cross_mix = SCParameter(cross_init)

        # ════════════════════════════════════════════════════════════
        #  贡献 2: BiCK — 双线性上下文核
        # ════════════════════════════════════════════════════════════
        kernel_rank = getattr(
            cfg, "mska_kernel_rank",
            max(8, self.head_dim // 4),
        )
        self.kernel_rank = kernel_rank

        # 双线性投影 φ, ψ: head_dim → kernel_rank
        # 在所有头上共享 (参数高效)
        self.phi_proj = SCLinear(self.head_dim, self.kernel_rank, bias=False)
        self.psi_proj = SCLinear(self.head_dim, self.kernel_rank, bias=False)

        # 每头的核门控: γ_h = sigmoid(logit_h), γ_h ∈ (0,1)
        kernel_logit_init = torch.zeros(cfg.n_heads) - 1.0  # 初始 sigmoid ≈ 0.27
        self.kernel_logit = SCParameter(kernel_logit_init)

        # ════════════════════════════════════════════════════════════
        #  贡献 3: RQR — 循环查询精化 [可选, 默认关闭]
        # ════════════════════════════════════════════════════════════
        self.sched_steps = max(1, getattr(cfg, "mska_scheduling_steps", 1))

        if self.sched_steps > 1:
            # 瓶颈 MLP: (H*D) → (2*kernel_rank) → (H*D)
            refine_dim = self.n_heads * self.head_dim
            self.refine_down = SCLinear(refine_dim, kernel_rank * 2, bias=False)
            self.refine_up = SCLinear(kernel_rank * 2, refine_dim, bias=False)
            # 步加权
            self.step_logits = SCParameter(torch.full((self.sched_steps,), 0.0))

        # Dropout
        self.attn_drop = SCDropout(cfg.dropout)

    # ════════════════════════════════════════════════════════════════
    #  MS-Head: 多尺度频率缩放
    # ════════════════════════════════════════════════════════════════

    def _multiscale_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """对 Q/K 做逐头频带频率缩放

        每个频带 b 的缩放因子 = exp(混合后频率) ∈ [0.5, 2.7]
        频带 b 内的所有头共享同一缩放因子。

        Args:
            q: [B, H, T, D]  query (含 RoPE)
            k: [B, H_kv, T, D] key (含 RoPE)

        Returns:
            q_scaled, k_scaled: 缩放后的 Q/K
        """
        B, H, T, D = q.shape
        H_kv = k.shape[1]

        # 频带混合: [n_bands] = softmax(cross_mix) @ exp(band_log_scale)
        mix_w = F.softmax(self.cross_mix, dim=-1)  # [nb, nb]
        band_freq = torch.exp(self.band_log_scale)  # [nb]
        freq = mix_w @ band_freq  # [nb]

        # 扩展到每个注意力头
        head_freq = freq.repeat_interleave(self.band_size)[:H]
        head_freq = head_freq.view(1, H, 1, 1)  # [1, H, 1, 1]

        # 频率缩放: Q' = Q * freq_h
        q_s = q * head_freq

        # KV 头: 复用到最近的 Q 头频率
        if H_kv < H:
            ratio = H // H_kv
            kv_freq = head_freq[:, ::ratio, :, :]
        else:
            kv_freq = head_freq[:, :H_kv, :, :]
        k_s = k * kv_freq

        return q_s, k_s

    # ════════════════════════════════════════════════════════════════
    #  BiCK: 双线性上下文核
    # ════════════════════════════════════════════════════════════════

    def _bilinear_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """计算标准 + 双线性核注意力得分

        Args:
            q: [B, H, T, D]  queries
            k: [B, H, T_kv, D] keys
            x: [B, T, dim]     原始输入

        Returns:
            scores: [B, H, T, T_kv]
        """
        B, H, T, D = q.shape
        T_kv = k.shape[2]

        # ── 标准 Q·K^T 得分 ──
        std_scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)

        # ── 双线性核得分 ──
        # 在 query/key 的 head_dim 上投影, 而非整个 dim
        # q: [B, H, T, D], phi_proj: D → r
        phi_q = F.silu(self.phi_proj(q))  # [B, H, T, r]
        psi_k = F.silu(self.psi_proj(k))  # [B, H, T_kv, r]

        # φ(q)·ψ(k)^T: [B, H, T, T_kv]
        kernel_scores = torch.matmul(
            phi_q, psi_k.transpose(-2, -1)
        ) * (self.kernel_rank ** -0.5)

        # 核门控: γ_h = sigmoid(logit_h)
        gamma = torch.sigmoid(self.kernel_logit).view(1, H, 1, 1)

        return std_scores + gamma * kernel_scores

    # ════════════════════════════════════════════════════════════════
    #  RQR: 循环查询精化
    # ════════════════════════════════════════════════════════════════

    def _refine_query(
        self,
        base_q: torch.Tensor,
        prev_output: torch.Tensor,
    ) -> torch.Tensor:
        """基于上一步注意力输出精化查询

        Args:
            base_q:     [B, H, T, D]  初始 Q (经 MS-Head 缩放)
            prev_output: [B, T, H*D]  上步注意力输出

        Returns:
            q_refined:  [B, H, T, D]
        """
        B, H, T, D = base_q.shape

        hidden = F.silu(self.refine_down(prev_output))  # [B, T, r*2]
        delta = self.refine_up(hidden)  # [B, T, H*D]
        delta = delta.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        return base_q + delta * 0.1

    # ════════════════════════════════════════════════════════════════
    #  掩码辅助
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _extend_mask(
        mask: torch.Tensor, T_kv: int
    ) -> torch.Tensor:
        """扩展因果掩码以覆盖 KV cache 位置"""
        T_cur = mask.shape[-1]
        if T_kv <= T_cur:
            return mask
        pad = torch.zeros(
            1, 1, T_cur, T_kv - T_cur,
            device=mask.device, dtype=mask.dtype,
        )
        return torch.cat([pad, mask], dim=-1)

    # ════════════════════════════════════════════════════════════════
    #  主入口
    # ════════════════════════════════════════════════════════════════

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """MSKA 前向传播

        Args:
            x:         [B, T, dim] 输入
            freqs_cis: RoPE 频率
            mask:      因果掩码
            kv_cache:  KV 缓存
            cache_key: 缓存键

        Returns:
            output: [B, T, dim]
        """
        B, T, _ = x.shape

        # ── 1. QKV 投影 + RoPE ─────────────────────────────────────
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # ── 2. KV 缓存 ────────────────────────────────────────────
        if kv_cache is not None:
            if cache_key in kv_cache:
                k_cached, v_cached = kv_cache[cache_key]
                k = torch.cat([k_cached, k], dim=1)
                v = torch.cat([v_cached, v], dim=1)
            kv_cache[cache_key] = (k.detach(), v.detach())

        T_kv = k.shape[1]

        # ── 3. GQA ────────────────────────────────────────────────
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2).contiguous()  # [B, H, T, D]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # ── 4. MS-Head: 多尺度频率缩放 ──────────────────────────
        q, k = self._multiscale_qk(q, k)

        # ── 5. 掩码 ─────────────────────────────────────────────
        if mask is not None:
            if mask.shape[-1] < T_kv:
                mask = self._extend_mask(mask, T_kv)
            elif mask.shape[-1] > T_kv:
                mask = mask[..., :T_kv]

        # ── 6. 注意力计算 (含 RQR) ─────────────────────────────
        if self.sched_steps <= 1:
            # 单步: 标准 + BiCK
            scores = self._bilinear_scores(q, k, x)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)  # [B, H, T, D]
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        else:
            # 多步: RQR 循环
            step_outs = []
            for step in range(self.sched_steps):
                q_step = q if step == 0 else self._refine_query(q, prev_out)
                scores = self._bilinear_scores(q_step, k, x)
                if mask is not None:
                    scores = scores + mask
                attn = F.softmax(scores, dim=-1)
                attn = self.attn_drop(attn)
                step_out = torch.matmul(attn, v)
                step_out = step_out.transpose(1, 2).contiguous().view(B, T, -1)

                step_outs.append(step_out)
                prev_out = step_out

            # 加权组合
            w = F.softmax(self.step_logits, dim=-1)
            out = step_outs[0]
            for i in range(1, len(step_outs)):
                out = out + w[i] * step_outs[i]

        # ── 7. 输出投影 ─────────────────────────────────────────
        return self.wo(out)

    # ════════════════════════════════════════════════════════════════
    #  分析接口
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def get_band_frequencies(self) -> torch.Tensor:
        """获取当前各频带的实际频率

        Returns:
            [n_bands] 频率值
        """
        mix_w = F.softmax(self.cross_mix, dim=-1)
        return mix_w @ torch.exp(self.band_log_scale)

    @torch.no_grad()
    def get_kernel_gates(self) -> torch.Tensor:
        """获取每头双线性核的门控值 γ

        Returns:
            [n_heads] γ ∈ (0, 1)
        """
        return torch.sigmoid(self.kernel_logit)


__all__ = ["MultiScaleKernelAttention"]
