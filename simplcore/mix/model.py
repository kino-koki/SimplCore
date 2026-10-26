# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix 主模型 — 三家融合版
融合 SimplCore + OpenMythos RDT 的完整架构

架构流:
    Input
     ↓
    [Embedding]
     ↓
    [Prelude Layers] × N  (SwiGLU + MBA注意力，执行一次)
     ↓
    [Recurrent Block] × T (循环执行，每步融合9个子模块)
     ↓  h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
    [Coda Layers] × M     (SwiGLU + MBA注意力 + 置信度门控)
     ↓
    [Output Projection]
     ↓
    Logits

三家融合亮点:
  OpenMythos: RDT三段式骨架 + LTI稳定注入 + ACT停止 + LoRAAdapter + loop_index_embedding
  SimplCore:  可调制基向量注意力 + 原型路由MoE + 工作记忆缓冲 + Hebbian在线适应
  Mix创新:    置信度门控 + 组件协同优化 + SwiGLU统一FFN + 权重绑定
"""
import simplcore.tensor as torch
from simplcore.tensor.nn import functional as F
import simplcore.tensor.utils.checkpoint as torch_checkpoint
from typing import Optional, List
import math

from .config import SimplCoreMixConfig
from .core import SCEmbedding, SCLinear, SCModule, SCModuleList, SCRMSNorm
from .frontier import SCMilestoneSpec
from .utils import precompute_rope_freqs, get_causal_mask
from .blocks import (
    PreludeBlock,
    RecurrentBlock,
    CodaBlock,
    LTIInjection,
    LoRAAdapter,
    ACTHalting,
    loop_index_embedding,
)


class SimplCoreMix(SCModule):
    """SimplCore Mix 三家融合架构"""

    def __init__(self, cfg: SimplCoreMixConfig):
        super().__init__()
        self.cfg = cfg

        # Token嵌入
        self.tok_embeddings = SCEmbedding(cfg.vocab_size, cfg.dim)

        # Prelude层
        self.prelude = SCModuleList([
            PreludeBlock(cfg, i) for i in range(cfg.prelude_layers)
        ])

        # 循环块 (核心 — 内化管理循环+ACT)
        self.recurrent = RecurrentBlock(cfg)

        # Coda层
        self.coda = SCModuleList([
            CodaBlock(cfg, i) for i in range(cfg.coda_layers)
        ])

        # 输出投影
        self.norm_final = SCRMSNorm(cfg.dim)
        self.output = SCLinear(cfg.dim, cfg.vocab_size, bias=False)

        # 权重绑定 (OpenMythos)
        if cfg.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # RoPE频率预计算
        # GQA使用完整head_dim; MLA使用qk_rope_head_dim (解耦)
        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        )
        # MLA专用RoPE频率 (仅qk_rope_head_dim维度)
        self.register_buffer(
            "freqs_cis_mla",
            precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, SCLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, SCEmbedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_state(self) -> None:
        """重置所有非参数运行时状态。

        包括:
        - RecurrentBlock 的 SPM 可塑性矩阵。
        - WorkingMemoryBuffer 的记忆 buffer。

        使用场景:
        - 训练 epoch 开始时调用, 避免上一 epoch 的状态泄漏。
        - 从 checkpoint 恢复后调用, 确保状态干净。
        - 评测前调用, 避免训练状态影响评测结果。
        - DDP 下所有 rank 同步调用, 保证一致性。
        """
        if hasattr(self.recurrent, "reset_state"):
            self.recurrent.reset_state()

    def _get_freqs_cis(self, seq_len: int, device: torch.device, start_pos: int = 0):
        """获取RoPE频率，MLA和GQA使用不同维度"""
        if self.cfg.attn_type == "mba_mla":
            freqs = self.freqs_cis_mla
        else:
            freqs = self.freqs_cis
        return freqs[start_pos:start_pos + seq_len].to(device)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """因果掩码 (在CPU上创建后移至设备, 兼容DirectML)"""
        mask = torch.full((1, 1, seq_len, seq_len), -1e4)
        return torch.triu(mask, diagonal=1).to(device)

    def _should_checkpoint(self, kv_cache: Optional[dict]) -> bool:
        """训练时启用激活重计算，推理和 KV cache 路径保持原样。"""
        return bool(self.training and self.cfg.gradient_checkpointing and kv_cache is None)

    def _checkpoint(self, function, *args):
        """统一封装梯度检查点，显式使用非 reentrant 模式。"""
        return torch_checkpoint.checkpoint(function, *args, use_reentrant=False)

    def forward(
        self,
        tokens: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """前向传播: Prelude → Recurrent Block → Coda

        Args:
            tokens:    [B, T] 输入token IDs
            n_loops:   循环深度 (默认cfg.max_loop_iters，推理时可增大实现深度外推)
            kv_cache:  KV缓存字典 (传入空dict{}并跨decode步复用)
            start_pos: 第一个token在完整序列中的位置 (用于增量解码的RoPE选择)
        Returns:
            logits: [B, T, vocab_size]
        """
        T = tokens.shape[1]
        device = tokens.device

        # 嵌入
        x = self.tok_embeddings(tokens)  # [B, T, dim]

        # RoPE + mask
        freqs_cis = self._get_freqs_cis(T, device, start_pos)
        mask = self._causal_mask(T, device) if T > 1 else None

        # Prelude
        for layer in self.prelude:
            if self._should_checkpoint(kv_cache):
                x = self._checkpoint(lambda hidden, block=layer: block(hidden, freqs_cis, mask, None), x)
            else:
                x = layer(x, freqs_cis, mask, kv_cache)

        # 保存注入信号 (编码输入，跨循环冻结)
        e = x

        # Recurrent Block (内化管理循环 + ACT停止)
        if self._should_checkpoint(kv_cache):
            x = self._checkpoint(
                lambda hidden: self.recurrent(hidden, e, freqs_cis, mask, n_loops, None),
                x,
            )
        else:
            x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        # Coda
        for layer in self.coda:
            if self._should_checkpoint(kv_cache):
                x = self._checkpoint(lambda hidden, block=layer: block(hidden, freqs_cis, mask, None), x)
            else:
                x = layer(x, freqs_cis, mask, kv_cache)

        # 输出
        x = self.norm_final(x)
        logits = self.output(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 100,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """自回归生成 (KV缓存优化)

        步骤0处理完整prompt，后续步骤只传入最后生成的token，
        所有之前的key/value从kv_cache获取。

        Args:
            tokens:          [B, T] 起始token序列
            max_new_tokens:  最大生成token数
            n_loops:         推理时循环深度 (可大于训练深度实现深度外推)
            temperature:     采样温度
            top_k:           Top-K采样
            top_p:           Top-P (nucleus) 采样
        Returns:
            generated: [B, T+max_new_tokens]
        """
        self.eval()
        prompt_len = tokens.shape[1]
        kv_cache = {}
        generated = tokens.clone()

        for step in range(max_new_tokens):
            if step == 0:
                # 首步: 处理完整prompt
                cur_tokens = generated
                cur_start_pos = 0
            else:
                # 后续: 只处理最后一个token
                cur_tokens = generated[:, -1:]
                cur_start_pos = prompt_len + step - 1

            logits = self.forward(
                cur_tokens,
                n_loops=n_loops,
                kv_cache=kv_cache,
                start_pos=cur_start_pos,
            )

            logits = logits[:, -1, :] / temperature

            # Top-K过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-P过滤
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_param_count(self) -> dict:
        """获取参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        prelude_params = sum(p.numel() for p in self.prelude.parameters())
        recurrent_params = sum(p.numel() for p in self.recurrent.parameters())
        coda_params = sum(p.numel() for p in self.coda.parameters())
        embed_params = sum(p.numel() for p in self.tok_embeddings.parameters())
        output_params = sum(p.numel() for p in self.output.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "prelude": prelude_params,
            "recurrent": recurrent_params,
            "coda": coda_params,
            "embedding": embed_params,
            "output": output_params,
        }

    def get_architecture_info(self) -> dict:
        """获取架构信息 (三家融合详情)"""
        frontier = SCMilestoneSpec.default()
        return {
            "architecture": "SimplCore Mix",
            "backend": self.sc_backend_name(),
            "self_owned_surface": True,
            "current_tier": frontier.current_tier,
            "target_tier": frontier.target_tier,
            "frontier_gates": frontier.required_gates,
            "skeleton": "OpenMythos RDT (Prelude → Recurrent → Coda)",
            "components": {
                "OpenMythos": [
                    "LTIInjection (谱半径<1保证)",
                    "LoRAAdapter (per-loop scale + 深度外推)",
                    "loop_index_embedding (循环深度区分)",
                    "ACTHalting (remainder trick + 加权求和)",
                    "SwiGLU FFN (Prelude/Coda)",
                    "MLA Multi-Latent Attention",
                ],
                "SimplCore": [
                    "可调制基向量注意力 / MSKA多尺度核注意力 (频带缩放+双线性核+可配循环精化)",
                    "原型路由MoE (签名路由+置信度)",
                    "工作记忆缓冲通道 (多通道+联想检索)",
                    "Hebbian在线适应 (Hebbian推理时学习)",
                    "DeepCIN深度因果干预网络 (DCGD+DDI+BCA)",
                    "IB信息瓶颈",
                    "MRD元表示蒸馏",
                    "PUM可塑性更新模块",
                    "结构化门控干预(SGI) (DCGD+DDI+BCA)",
                    "门控压缩",
                    "低秩特征投影",
                    "残差适应模块",
                ],
                "Mix创新": [
                "置信度门控 (Coda层置信度过滤, 原名置信度门控)",
                    "组件协同优化 (LTI+SPM+DMC联合)",
                    "SwiGLU统一FFN (跨Prelude/Coda/Expert)",
                    "权重绑定 (embedding↔output)",
                    "MLA专用RoPE频率",
                ],
            },
            "loop_depth": self.cfg.max_loop_iters,
            "act_threshold": self.cfg.act_threshold,
            "weight_tying": self.cfg.weight_tying,
            "param_count": self.get_param_count()["total"],
        }
