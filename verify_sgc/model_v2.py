"""
SGC v2 模型定义（纯自研 simplcore.tensor 后端）

在 model.py 基础上，新增 SGCMixerV2（使用 SpectralGlobalConvV2）。
其余骨架（embedding / 位置编码 / FFN / LayerNorm）与 Attention baseline 完全一致，
保证对比公平。SGC v2 = 谱全局混合（无瓶颈，恒等初始化）+ 轻量因果局部修正。
"""
import numpy as np
import simplcore.tensor as torch
from simplcore.tensor.tensor import Tensor
from simplcore.tensor import nn

from model import AttentionMixer, Block, SeqModel, count_params


class SGCMixerV2(nn.Module):
    """SGC v2 混合层：全局谱混合（固定 FFT 恒等往返，零谱参数）
    + 可学因果局部修正，门控融合。SpectralGlobalConvV2 内部已含残差。"""
    def __init__(self, dim, n_heads=4, max_len=64):
        super().__init__()
        from simplcore.architecture.spectral_global_conv import SpectralGlobalConvV2
        self.sgc = SpectralGlobalConvV2(dim=dim, kernel_size=3, max_len=max_len)

    def forward(self, x):
        return self.sgc(x)


def build_model(mixer, vocab_size, dim, n_layers, n_heads):
    """统一构建函数，mixer in {'attn','sgc','sgc2'}"""
    if mixer == "attn":
        m = lambda d, h: AttentionMixer(d, h)
    elif mixer == "sgc":
        from model import SGCMixer
        m = lambda d, h: SGCMixer(d, h)
    elif mixer == "sgc2":
        m = lambda d, h: SGCMixerV2(d, h, max_len=64)
    else:
        raise ValueError(mixer)
    # 用 SeqModel 的构造，但替换 mixer 类型：直接逐 block 构造以可控
    model = SeqModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                     n_heads=n_heads, mixer="attn")
    for blk in model.blocks:
        blk.mixer = m(dim, n_heads)
    return model
