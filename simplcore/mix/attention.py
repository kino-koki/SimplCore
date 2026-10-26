# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
MBA-MLA / MBA-GQA 注意力模块
融合 OpenMythos MLA/GQA + SimplCore 可调制基向量处理
"""
import simplcore.tensor as torch
from typing import Optional
from .core import SCDropout, SCFunctional, SCLinear, SCModule, SCParameter
from .utils import RMSNorm, apply_rope
from .config import SimplCoreMixConfig


class ModulatedBasisAttention(SCModule):
    """Modulated Basis Attention Attention

    将SimplCore的RFP频率调制基向量机制融入MLA/GQA注意力。
    核心创新：用可学习的共振基向量替代标准Q/K/V投影的一部分，
    通过频率缩放和相位偏移动态调制注意力模式。
    """

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.n_basis_vectors = cfg.n_basis_vectors

        # 标准投影
        self.wq = SCLinear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = SCLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = SCLinear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

        # 可调制基向量参数 (SimplCore创新)
        # 可学习的基向量 [n_heads, n_basis_vectors, head_dim]
        self.bases = SCParameter(
            torch.randn(cfg.n_heads, cfg.n_basis_vectors, self.head_dim) * 0.02
        )
        # 频率缩放和相位偏移 [n_heads]
        self.freq_scale = SCParameter(
            torch.ones(cfg.n_heads) * cfg.freq_scale_init
        )
        self.phase_shift = SCParameter(
            torch.zeros(cfg.n_heads) * cfg.phase_shift_init
        )
        # 基向量输出权重 [n_heads, n_basis_vectors, head_dim]
        self.base_output = SCParameter(
            torch.randn(cfg.n_heads, cfg.n_basis_vectors, self.head_dim) * 0.02
        )

        self.attn_drop = SCDropout(cfg.dropout)
        self.attn_temperature = cfg.attn_temperature

        # 输入自适应门控 (增强版RFP)
        self.modulation_gate = SCLinear(cfg.dim, cfg.n_heads, bias=True)

    def _compute_modulated_bases(self, x: torch.Tensor) -> torch.Tensor:
        """RFP核心：基于输入统计动态调制基向量

        Args:
            x: [B, T, dim]
        Returns:
            modulated_bases: [n_heads, n_basis_vectors, head_dim]
        """
        # 计算输入均值作为调制信号 [B, T, n_heads]
        gate = torch.sigmoid(self.modulation_gate(x))  # [B, T, n_heads]
        gate = gate.mean(dim=(0, 1))  # [n_heads]

        # 频率调制
        # modulated_base[j] = base[j] * cos(omega + j*0.01) + base_flip[j] * sin(phi + j*0.01)
        n_basis_vectors = self.n_basis_vectors
        head_dim = self.head_dim

        # 为每个head生成调制信号
        j_idx = torch.arange(n_basis_vectors, device=self.bases.device).float()
        j_idx = j_idx.view(1, n_basis_vectors, 1)  # [1, n_basis_vectors, 1]

        freq = self.freq_scale.view(self.n_heads, 1, 1)  # [n_heads, 1, 1]
        phase = self.phase_shift.view(self.n_heads, 1, 1)  # [n_heads, 1, 1]

        # 调制系数
        cos_mod = torch.cos(freq * 3.14159 + j_idx * 0.01)  # [n_heads, n_basis_vectors, 1]
        sin_mod = torch.sin(phase * 3.14159 + j_idx * 0.01)

        # 翻转索引
        bases_flip = torch.flip(self.bases, dims=[1])  # 沿n_basis_vectors维度翻转

        # 应用调制
        modulated = self.bases * cos_mod + bases_flip * sin_mod

        # 应用门控缩放
        gate_scale = gate.view(self.n_heads, 1, 1)
        modulated = modulated * (0.5 + 0.5 * gate_scale)

        return modulated

    def _apply_mba_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """应用RFP增强的注意力计算

        q, k, v: [B, H, T, head_dim]
        """
        B, H, T, D = q.shape

        # 调制基向量
        modulated_bases = self._compute_modulated_bases(
            q.transpose(1, 2).reshape(B, T, -1)[:, :1, :self.cfg.dim]
        )  # [n_heads, n_basis_vectors, head_dim]

        # 计算基向量与Q/K的基向量相似度
        # 将基向量扩展到batch
        bases = modulated_bases.unsqueeze(0)  # [1, H, n_basis_vectors, D]

        # Q与基的共振: [B, H, T, n_basis_vectors]
        q_resonance = torch.matmul(q, bases.transpose(-2, -1)) / (D ** 0.5)
        k_resonance = torch.matmul(k, bases.transpose(-2, -1)) / (D ** 0.5)

        # 标准注意力分数
        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 添加基向量增强 (低频增强)
        resonance_enhance = torch.matmul(q_resonance, k_resonance.transpose(-2, -1))
        resonance_enhance = resonance_enhance * 0.1  # 小幅增强，避免主导

        attn = attn + resonance_enhance

        if mask is not None:
            attn = attn + mask

        attn = self.attn_drop(SCFunctional.softmax(attn / self.attn_temperature, dim=-1))
        out = torch.matmul(attn, v)
        return out

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = self._apply_mba_attention(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class ModulatedBasisMLAttention(SCModule):
    """RFP增强的Multi-Latent Attention (DeepSeek-V2风格 + MBA)"""

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.n_basis_vectors = cfg.n_basis_vectors

        # Q压缩
        self.q_down = SCLinear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = SCLinear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = SCLinear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)

        # KV压缩
        self.kv_down = SCLinear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = SCLinear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )
        self.wo = SCLinear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = SCDropout(cfg.dropout)

        # RFP参数
        self.bases = SCParameter(
            torch.randn(cfg.n_heads, cfg.n_basis_vectors, self.q_head_dim) * 0.02
        )
        self.freq_scale = SCParameter(torch.ones(cfg.n_heads) * cfg.freq_scale_init)
        self.phase_shift = SCParameter(torch.zeros(cfg.n_heads) * cfg.phase_shift_init)
        self.modulation_gate = SCLinear(cfg.dim, cfg.n_heads, bias=True)
        self.attn_temperature = cfg.attn_temperature

    def _compute_modulated_bases(self, x: torch.Tensor = None) -> torch.Tensor:
        """调制基向量 — 输入自适应门控 (增强版RFP)

        Args:
            x: [B, T, dim] 输入张量, 用于计算调制门控
        Returns:
            modulated_bases: [n_heads, n_basis_vectors, q_head_dim]
        """
        j_idx = torch.arange(self.n_basis_vectors, device=self.bases.device).float()
        j_idx = j_idx.view(1, self.n_basis_vectors, 1)
        freq = self.freq_scale.view(self.n_heads, 1, 1)
        phase = self.phase_shift.view(self.n_heads, 1, 1)
        cos_mod = torch.cos(freq * 3.14159 + j_idx * 0.01)
        sin_mod = torch.sin(phase * 3.14159 + j_idx * 0.01)
        bases_flip = torch.flip(self.bases, dims=[1])
        modulated = self.bases * cos_mod + bases_flip * sin_mod

        # 输入自适应门控: 基于输入统计调制基向量幅度
        if x is not None:
            gate = torch.sigmoid(self.modulation_gate(x))  # [B, T, n_heads]
            gate = gate.mean(dim=(0, 1))  # [n_heads]
            gate_scale = gate.view(self.n_heads, 1, 1)
            modulated = modulated * (0.5 + 0.5 * gate_scale)

        return modulated

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]
        k_rope = kv_raw[..., self.kv_lora_rank :]

        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]
        v = kv[..., self.qk_nope_dim :]
        k = torch.cat([k_nope, k_rope], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RFP增强注意力
        modulated_bases = self._compute_modulated_bases(x).unsqueeze(0)
        q_res = torch.matmul(q, modulated_bases.transpose(-2, -1)) / (self.q_head_dim ** 0.5)
        k_res = torch.matmul(k, modulated_bases.transpose(-2, -1)) / (self.q_head_dim ** 0.5)

        scale = self.q_head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        res_enhance = torch.matmul(q_res, k_res.transpose(-2, -1)) * 0.1
        attn = attn + res_enhance

        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(SCFunctional.softmax(attn / self.attn_temperature, dim=-1))
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
