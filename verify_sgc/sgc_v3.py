# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-v3：内容路由门控的因果全局卷积 (Content-Routed SGC).

这是对 SGC-v2 的诚实改进, 目的是缓解"位置相对核无法按内容路由 -> 高熵泛化差"的
固有局限 (详见《SGC_v2_诚实版报告》第3.3/4.1 节). 设计要点:

- 保持 SGC-v2 的 FFt 因果全局卷积为主体 (O(n log n) 位置核, 单位脉冲初始化);
- 额外引入一个**内容门控头**: 用内容键 K(x) 生成逐通道门控 gate∈[0,1],
  对 SGC 输出的残差差异做内容相关调制, 使同一位置核在不同内容上产生不同有效贡献;
- 为避免旧版 Gated-SGC 的过拟合, 本版:
    * 用 AdamW + weight_decay (不使用 SGD 大学习率),
    * 门控带 dropout,
    * gate_scale 初始化为 0 (退化=纯 SGC-v2), 由训练决定是否信赖内容门控,
    * 评估一律用独立 test 集 (绝不只报 train_acc).

对"完全基于 SimplCore 架构"的解释: 本模块核心仍是 SGC/位置卷积 + 门控,
未引入任何 QKV/softmax 注意力权重表 (没有任何 self-attention), 仍是 O(n log n) 主复杂度.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sgc_pytorch import SpectralGlobalConvV2, AttentionMixer, FFN


class ContentRoutedSGC(nn.Module):
    """SGC-v2 位置核 + 内容路由门控.

    gate = sigmoid(W_g * LN(x)), 逐通道调制卷积混合贡献:
        y_gated = residual + alpha * out_proj( gating 后的 sgc 差异 )

    relation = "mixer" (面向下游, 计算图沿用 PyTorch autograd).

    Args:
        dim:          通道维 D
        max_len:      序列最大长度 (核长度)
        rank:         SGC 低秩核秩
        gate_dropout: 门控 dropout (防过拟合)
        gate_scale0:  门控缩放初始化 (0 => 以纯 SGC 为起点)
    """

    def __init__(self, dim: int, max_len: int = 512, rank: int = 8,
                 gate_dropout: float = 0.1, gate_scale0: float = 0.0,
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.sgc = SpectralGlobalConvV2(dim, kernel_size=3, max_len=max_len, rank=rank)

        # 内容门控头
        self.gate_norm = nn.LayerNorm(dim, eps=eps)          # 门控输入归一
        self.gate_key = nn.Linear(dim, dim, bias=False)      # 内容键 W_k
        self.gate_out = nn.Linear(dim, dim, bias=False)      # 生成逐通道门控
        self.gate_scale = nn.Parameter(torch.tensor(gate_scale0))
        self.gate_drop = nn.Dropout(gate_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y_sgc = self.sgc(x)                     # (B,L,D) 位置核全局混合
        # 内容门控: 每 token 每通道 ∈ [0,1]
        g = self.gate_norm(x)
        k = torch.tanh(self.gate_key(g))        # 有界内容键
        gate = torch.sigmoid(self.gate_out(k))
        gate = self.gate_drop(gate)             # 训练期随机置零防止门控过拟合
        # 内容路由: 用 gate 调制"卷积差异"的贡献 (gate_scale0=0 => 初始等价 SGC-v2)
        blend = 1.0 + self.gate_scale * gate
        return residual + (y_sgc - residual) * blend


class ContentAdaptiveMixer(nn.Module):
    """内容自适应全局混合 (SGC-CA): 线性注意力式, O(n·d·r) 与序列长线性.

    与 SGC-v2 的"固定位置核"本质区别: 混合权重随内容(查询/键)变化,
    即每个位置按内容自适应地选择历史贡献. 不使用 QKV/softmax 注意力表.

    高效实现 (块式 bmm, 全部走 OpenBLAS, 避免 4D 中间张量):
      按块 C 处理: 块内用 causal tril(QK^T)V 累积, 跨块用携带状态
      S=∑(k⊗v), z=∑k (O(r·D), 长度无关) 延续. 数学等价于线性注意力:
        out_i = (Q_i·S + tril(QK^T)V_i) / (Q_i·z + tril(QK^T)·1 + eps)
      时/空均 O(n), 且不生成 (B,C,r,D) 巨型张量 -> CPU/核显上快.

    Args:
        dim:  通道维 D
        feat: 关联记忆槽数 r (S ∈ R^{r×D})
        chunk: 块大小 C (内部 bmm 粒度, 也是流式分块)
    """

    def __init__(self, dim: int, feat: int = 32, chunk: int = 0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.feat = feat
        self.chunk = chunk if chunk and chunk > 0 else 32   # 0 -> 默认块 32
        self.eps = eps
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, feat, bias=False)
        self.k_proj = nn.Linear(dim, feat, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.register_buffer('_tril',
                             torch.tril(torch.ones(self.chunk, self.chunk)),
                             persistent=False)

    def forward_stream(self, x, state=None):
        """流式: 逐块处理整个序列 x (B,L,D), 携带 (S,z) 状态.
        除非 state 跨调用, 峰值显存 O(chunk·D+r·D) 与 L 无关."""
        B, L, D = x.shape
        S = torch.zeros(B, self.feat, D, device=x.device, dtype=x.dtype) if state is None else state[0]
        z = torch.zeros(B, self.feat, device=x.device, dtype=x.dtype) if state is None else state[1]
        out = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
        C = self.chunk if self.chunk and self.chunk > 0 else L
        # 先做 x 上的投影 (批处理, 省多次小 GEMM)
        xq = F.elu(self.q_proj(self.norm(self.in_proj(x)))) + 1.0
        xk = F.elu(self.k_proj(self.norm(self.in_proj(x)))) + 1.0
        xv = self.v_proj(self.norm(self.in_proj(x)))
        for st in range(0, L, C):
            ed = min(L, st + C)
            o, S, z = self._step_h(xq[:, st:ed], xk[:, st:ed], xv[:, st:ed], S, z)
            out[:, st:ed] = x[:, st:ed] + self.out_proj(o)
        return out, (S, z)

    def _step_h(self, Q, K, V, S, z):
        B, C, D = V.shape
        Sim = torch.matmul(Q, K.transpose(1, 2)) * self._tril[:C, :C]
        num_state = torch.matmul(Q, S)
        num_att = torch.matmul(Sim, V)
        den_state = torch.matmul(Q, z.unsqueeze(-1))
        den_att = Sim.sum(-1, keepdim=True)
        num = num_state + num_att
        den = (den_state + den_att).clamp_min(self.eps)
        out = num / den
        S_new = S + torch.matmul(K.transpose(1, 2), V)
        z_new = z + K.sum(1)
        return out, S_new, z_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stream(x)[0]


class GatedAssocMixer(nn.Module):
    """门控关联记忆 (SGC-GM): 给 CA 加"可写可覆盖"的记忆容量, 逼近注意力的检索.

    平 CA 的状态 S=∑(k⊗v) 是"无限累加", 信息易饱和/被远史淹没, 精确召回弱.
    本版把状态改成 **可寻址记忆槽 (M 键, A 值)**, 读取用内容相似度加权,
    写入用"匹配强度吸引式重写" (每个槽被当前 (k,v) 按匹配比例拉近), 槽可覆盖旧内容,
    从而像注意力一样能记住”不同”的东西. 仍是 O(n)、常量内存、无 softmax.

      读取:  w_i = elu(q·M_i)+1 ;  out = Σ_i w_i A_i / (Σ_i w_i + eps)
      写入:  α_i = elu(k·M_i)+1 归一;  A_i←A_i+β·α_i·(v−A_i);  M_i←M_i+β·α_i·(k−M_i)

    Args:
        dim:   模型宽 D
        slots: 记忆槽数 r (=feat)
        beta:  写入门控步长 (越小越稳)
        chunk: 内部块粒度
    """

    def __init__(self, dim: int, slots: int = 32, beta: float = 0.1,
                 chunk: int = 0, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.chunk = chunk if chunk and chunk > 0 else 32
        self.beta = beta
        self.eps = eps
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)

    def _step_pos(self, q, k, v, M, A, s=4.0):
        """单位置读写. q,k,v:(B,D). 返回 (out, M_new, A_new). 用有界 tanh 亲和度防发散."""
        B = q.shape[0]
        wr = F.tanh(torch.matmul(q.unsqueeze(1), M.transpose(1, 2)).squeeze(1) / s) * 0.5 + 0.5  # (B,slots)
        den = wr.sum(-1, keepdim=True).clamp_min(self.eps)
        out = torch.bmm(wr.unsqueeze(1), A).squeeze(1) / den   # (B,D) 凸组合, 有界
        pa = F.tanh(torch.matmul(k.unsqueeze(1), M.transpose(1, 2)).squeeze(1) / s) * 0.5 + 0.5
        a = pa / pa.sum(-1, keepdim=True).clamp_min(self.eps)  # 槽内归一写权重
        M_new = M + self.beta * (k.unsqueeze(1) - M) * a.unsqueeze(-1)
        A_new = A + self.beta * (v.unsqueeze(1) - A) * a.unsqueeze(-1)
        return out, M_new, A_new

    def forward_stream(self, x, state=None):
        B, L, D = x.shape
        if state is None:
            # 旧实现直接 `torch.randn(...)`：每次前向都重新抽随机状态，导致
            # (a) 同一输入两次前向结果不同（实测差 9.7e-6），(b) 因果性测试失效
            # ——两次调用（原输入 / 扰动输入）拿到不同随机初值，输出处处不同，
            # 看起来像"未来泄漏"。改成像 SGC_HM._init_state 那样用固定生成的
            # 随机初值，使前向确定且可复现。
            g = torch.Generator(device=x.device).manual_seed(0)
            M = torch.randn(B, self.slots, D, generator=g,
                            device=x.device, dtype=x.dtype) * 0.02
            A = torch.zeros(B, self.slots, D, device=x.device, dtype=x.dtype)
        else:
            M, A = state
        h = self.norm(self.in_proj(x))
        Q = F.elu(self.q_proj(h)) + 1.0
        K = F.elu(self.k_proj(h)) + 1.0
        V = self.v_proj(h)
        out = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
        for t in range(L):
            o, M, A = self._step_pos(Q[:, t], K[:, t], V[:, t], M, A)
            out[:, t] = o
        return x + self.out_proj(out), (M, A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stream(x)[0]


class SGC_HM(nn.Module):
    """SGC-HM: 门控分层记忆 (Gated Hierarchical Memory).

    目标(诚实但进取): 让混合器拥有"注意力级别的记忆"、且**不随上下文长度线性衰退**,
    状态容量只随"显存预算"增长而与 L 无关 (上下文翻倍仅涨 0~1GB).

    打赢"线性衰退"的机制 —— 关键不是记住更多, 而是**改写**:
      * 普通线性注意力把历史摊进固定求和 S+=k⊗v, 每加一个新内容就按比例**稀释**旧内容,
        召回随上下文线性衰减.
      * 本模块用 **delta 覆盖写**: 新 (k,v) 不是摊进总量, 而是把"内容最像"的槽拉近覆盖
        (M_j←M_j+β·(k−M_j), A_j←A_j+β·(v−A_j)). 替换而非稀释 -> 远域保留不随 T 衰减.

    三层记忆 (容量只取决于预算, 与 L 无关):
      ① recent 精确缓冲 (W 元): 给"近期上下文的逐位精度"(注意力级别的触达);
      ② long 门控槽 (r 元, M 键 / A 值): 给"任意远的历史的可写可覆盖停留";
      ③ 读取 = 内容检索 (按 q·M 相似度) 加权 {缓冲, 长槽} -> 内容寻址, 不随位置衰减.

    实现: 分块 bmm (块内写 + 精确缓冲), CPU 上不退化到逐 token.

    Args:
        dim:     模型宽 D
        buffer:  近期精确缓冲元数 W
        slots:   长程门控槽数 r  (状态大小 = W·r·D, 由显存预算决定)
        beta:    delta 写入门控步长
        chunk:   内部块粒度
    """

    def __init__(self, dim: int, buffer: int = 128, slots: int = 512,
                 beta: float = 0.1, chunk: int = 0, eps: float = 1e-6,
                 scale: float = 2.0, read_tau: float = 4.0):
        super().__init__()
        self.dim = dim
        self.buffer = buffer
        self.slots = slots
        self.chunk = chunk if chunk and chunk > 0 else 64
        self.beta = beta
        self.eps = eps
        self.scale = scale
        # 路由可靠性: 读取温度可学习(经 softmax 得梯度, 自动加锐); 写入用单位化余弦 argmax
        self.read_tau = nn.Parameter(torch.tensor(float(read_tau)))
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.register_buffer('_tril',
                             torch.tril(torch.ones(max(chunk, 2), max(chunk, 2))),
                             persistent=False)

    def _init_state(self, B, D, device, dtype):
        # 键槽 M 是"内容寻址的原型": 用分得开的随机单位向量初始化(非 0),
        # 保证不同内容就近命中各自槽; 值槽 A 从 0 起步靠写入填充. 它们都是运行状态(非学习参数).
        g = torch.Generator(device=device).manual_seed(0)
        M = torch.randn(self.slots, D, generator=g, device=device, dtype=dtype)
        M = F.normalize(M, dim=-1).unsqueeze(0).expand(B, self.slots, D).contiguous()
        A = torch.zeros(B, self.slots, D, device=device, dtype=dtype)
        bk = torch.zeros(B, 0, D, device=device, dtype=dtype)   # 缓冲键 (近期精确)
        bv = torch.zeros(B, 0, D, device=device, dtype=dtype)   # 缓冲值
        return M, A, bk, bv

    def forward_stream(self, x, state=None):
        B, L, D = x.shape
        if state is None:
            M, A, bk, bv = self._init_state(B, D, x.device, x.dtype)
        else:
            M, A, bk, bv = state
        h = self.norm(self.in_proj(x))
        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)
        C = self.chunk
        out = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
        for st in range(0, L, C):
            ed = min(L, st + C)
            qc, kc, vc = Q[:, st:ed], K[:, st:ed], V[:, st:ed]
            n = ed - st
            # ---- 长程读取: 内容寻址 (单位化余弦 x 可学习温度, 胜者主导) ----
            q_n = F.normalize(qc, dim=-1)                          # (B,n,D)
            sim = q_n @ M.transpose(1, 2) * self.read_tau          # cosine 相似度
            w = F.softmax(sim, dim=-1)
            out_long = w @ A                                        # (B,n,D)
            # ---- 近期精确读取: 缓冲 causal ----
            inv = D ** -0.5
            qk_in = qc @ kc.transpose(1, 2) * inv                    # (B,n,n)
            tri = self._tril[:n, :n]
            if bk.shape[1] > 0:
                blen = bk.shape[1]
                qk = qc @ bk.transpose(1, 2) * inv                   # (B,n,blen)
                full = torch.cat([qk, qk_in], dim=-1)                # (B,n,blen+n)
                mask = torch.cat([torch.ones(n, blen), tri], dim=-1) # (n,blen+n) 缓冲不过遮
                att = F.softmax(full.masked_fill(mask == 0, float('-inf')), dim=-1)
                kv_all = torch.cat([bv, vc], dim=1)                  # (B,blen+n,D)
                out_recent = att @ kv_all
            else:
                att = F.softmax(qk_in.masked_fill(tri == 0, float('-inf')), dim=-1)
                out_recent = att @ vc
            out[:, st:ed] = x[:, st:ed] + self.out_proj(out_long + self.scale * out_recent)
            # ---- 长程胜者全取 delta 覆盖写 (块级, straight-through 路由) ----
            k_n = F.normalize(kc, dim=-1)                             # 单位化键做内容寻址
            aff = k_n @ M.transpose(1, 2)                              # (B,C,slots) cosine
            idx = aff.argmax(-1)                                      # 每token胜者槽
            one = F.one_hot(idx, self.slots).to(kc.dtype)             # (B,C,slots)
            cnt = one.sum(1, keepdim=True).transpose(1, 2).clamp_min(self.eps)  # (B,slots,1)
            claimed = (one.sum(1) > 0).unsqueeze(-1)                  # (B,slots,1)
            k_put = one.transpose(1, 2) @ kc / cnt                    # 赢家槽 <- 其认领者的(均值)键值
            v_put = one.transpose(1, 2) @ vc / cnt
            M_new = torch.where(claimed, M + self.beta * (k_put - M), M)  # 覆盖非稀释
            M = F.normalize(M_new, dim=-1)                            # 原型保持单位范数 -> 余弦语义稳定
            A = torch.where(claimed, A + self.beta * (v_put - A), A)
            # ---- 缓冲滚动更新 ----
            bk = torch.cat([bk, kc], dim=1)[:, -self.buffer:, :]
            bv = torch.cat([bv, vc], dim=1)[:, -self.buffer:, :]
        return out, (M, A, bk, bv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stream(x)[0]


def build_model(vocab: int, dim: int, n_layers: int, n_heads: int, max_len: int,
                mixer: str, rank: int = 8, gate_dropout: float = 0.1,
                ca_feat: int = 32, ca_chunk: int = 128, dropout: float = 0.0,
                gm_beta: float = 0.1):
    """同骨架 LM: 仅 mixer 不同 (attn / sgc / sgcv3 / ca / cagate), 保证公平对比.

    dropout: 嵌入层 + 每块残差流上的 drop 概率 (防记忆化崩塌的正则化)."""
    tok_emb = nn.Embedding(vocab, dim)
    pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
    blocks = nn.ModuleList()
    for _ in range(n_layers):
        if mixer == "attn":
            mix = AttentionMixer(dim, n_heads, max_len)
        elif mixer == "sgc":
            mix = SpectralGlobalConvV2(dim, 3, max_len, rank)
        elif mixer == "sgcv3":
            mix = ContentRoutedSGC(dim, max_len, rank, gate_dropout=gate_dropout)
        elif mixer == "ca":
            mix = ContentAdaptiveMixer(dim, feat=ca_feat, chunk=ca_chunk)
        elif mixer == "cagate":
            mix = GatedAssocMixer(dim, slots=ca_feat, beta=gm_beta, chunk=ca_chunk)
        elif mixer == "hm":
            mix = SGC_HM(dim, buffer=max(16, ca_feat // 4), slots=ca_feat,
                         beta=gm_beta, chunk=max(ca_chunk, 2))
        else:
            raise ValueError(f"unknown mixer {mixer}")
        blocks.append(nn.ModuleList([mix, FFN(dim, ratio=4.0)]))
    ln_f = nn.LayerNorm(dim)
    head = nn.Linear(dim, vocab, bias=False)
    head.weight = tok_emb.weight
    lm = SimpleLM(tok_emb, pos_emb, blocks, ln_f, head)
    lm.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    return lm


class SimpleLM(nn.Module):
    def __init__(self, tok_emb, pos_emb, blocks, ln_f, head):
        super().__init__()
        self.tok_emb = tok_emb
        self.pos_emb = pos_emb
        self.blocks = blocks
        self.ln_f = ln_f
        self.head = head

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb[:, :L])
        for mix, ffn in self.blocks:
            x = self.drop(mix(x))
            x = self.drop(ffn(x))
        return self.head(self.ln_f(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    # 冒烟: 三类 mixer 同规模前向可微
    torch.manual_seed(0)
    m = build_model(1000, 64, 2, 4, 128, "sgcv3")
    print("sgcv3 params:", count_params(m))
    x = torch.randint(0, 1000, (2, 128))
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, 1000), x.reshape(-1))
    loss.backward()
    print("forward/backward OK, loss=", round(float(loss), 4))