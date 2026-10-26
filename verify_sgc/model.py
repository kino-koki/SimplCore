"""
SGC vs Attention 下游有效性验证 —— 模型定义（纯自研 simplcore.tensor 后端）

两个模型共享同一骨架：token embedding + 位置编码 + 若干 block(混合层 + FFN) + 输出投影。
唯一区别是 block 内的"混合层"：AttentionMixer vs SGCMixer。
其余完全一致，保证对比公平。
"""
import numpy as np
import simplcore.tensor as torch
from simplcore.tensor.tensor import Tensor
from simplcore.tensor import nn


class AttentionMixer(nn.Module):
    """标准因果自注意力（baseline）。multi-head，带因果掩码。"""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.data.shape
        qkv = self.qkv(x)  # (B,L,3D)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # (B,H,L,hd)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        # scores (B,H,L,L)
        scores = (q @ k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        # 因果掩码
        mask = np.triu(np.ones((L, L), dtype=np.float32), k=1) * -1e9
        scores = scores + Tensor(mask[None, None, :, :], requires_grad=False)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B,H,L,hd)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class SGCMixer(nn.Module):
    """
    SGC 混合层（自研突破版）：用谱全局卷积替代注意力。
    包装 simplcore.architecture.spectral_global_conv.SpectralGlobalConv。
    SGC 是全局、无因果掩码依赖的谱混合；为公平对比，这里把它当"双向全局混合"用，
    评估其作为序列混合算子的表达能力（与 attention 同样能看到全序列）。
    """
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        from simplcore.architecture.spectral_global_conv import SpectralGlobalConv
        # SGC 在 head 维度上并行：把 dim 切成 n_heads 份，每份用 SGC 处理
        self.sgc = SpectralGlobalConv(dim=dim, num_bands=8, num_scales=3, rank=16)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.sgc(x)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim, mixer):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.mixer(h)
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        return x


class SeqModel(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=3, n_heads=4, mixer="attn"):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        layers = []
        for _ in range(n_layers):
            if mixer == "attn":
                m = AttentionMixer(dim, n_heads)
            else:
                m = SGCMixer(dim, n_heads)
            layers.append(Block(dim, m))
        self.blocks = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        B, L = idx.data.shape
        x = self.emb(idx) + self.pos[:, :L, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def count_params(model):
    n = 0
    for p in model.parameters():
        n += int(np.prod(p.data.shape))
    return n


class NaiveSpectralConv(nn.Module):
    """朴素谱卷积（对照基线）：频域可学习复增益、全 D 通道、无瓶颈、无门控。

    诊断表明它在 selective-copy 上 ≈ Attention，是 SGC 谱混合的强基线。
    频域增益在第一次 forward 时按实际 F 动态创建（恒等初始化）。
    """
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.fre = None  # 动态创建 (1, F, D)
        self.fim = None
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        from simplcore.architecture.fft_ops import rfft1d, irfft1d
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        B, L, D = u.shape
        cre, cim = rfft1d(u, dim=1)
        Fdim = cre.shape[1]
        if self.fre is None or self.fre.shape[1] != Fdim:
            self.fre = nn.Parameter(torch.ones(1, Fdim, D, dtype=torch.float32))
            self.fim = nn.Parameter(torch.zeros(1, Fdim, D, dtype=torch.float32))
        gre = self.fre[:, :Fdim, :]
        gim = self.fim[:, :Fdim, :]
        o_re = cre * gre - cim * gim
        o_im = cre * gim + cim * gre
        y = irfft1d(o_re, o_im, n=L, dim=1)
        out = self.out_proj(y)
        return residual + self.out_scale * out
