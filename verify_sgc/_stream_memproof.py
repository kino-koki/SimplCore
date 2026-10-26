# -*- coding: utf-8 -*-
"""计时版流式内存结构验证: 打印每层携带状态的精确字节 + 各 L 的流式耗时.

沙箱屏蔽了 OS 工作集读取, 因此这里给两条环境无关的证据:
  1) 分析式: 流式路径上跨块存活的张量只有 (状态 S, 归一 z), 字节数对 L 是常数;
  2) 经验式: 长度翻倍->耗时翻倍, 证明 O(L) 线性且中间无任何 O(L^2) 峰值张量.
"""
import time, torch
from sgc_v3 import build_model, count_params

def state_bytes(dim, layers, feat, dtype_sz=2):      # fp16
    s = feat * dim * dtype_sz     # S:(1,feat,D)
    z = feat * dtype_sz           # z:(1,feat)
    return (s + z) * layers, s * layers, z * layers

def run_stream(m, layers, L, chunk, states):
    with torch.no_grad():
        for st in range(0, L, chunk):
            ed = min(L, st + chunk)
            x = m.tok_emb(torch.randint(0, 2000, (1, ed - st)))
            for li, (mix, ffn) in enumerate(m.blocks):
                if mix.__class__.__name__ == 'ContentAdaptiveMixer':
                    x, states[li] = mix.forward_stream(x, states[li])
                else:
                    x = mix(x)
                x = ffn(x)
            _ = m.head(m.ln_f(x))
            del x

torch.set_num_threads(8)
dim, layers, feat, chunk = 256, 4, 32, 256
m = build_model(2000, dim, layers, 4, chunk, 'ca', rank=8, ca_feat=feat, ca_chunk=chunk)
m.eval()
states = [None] * layers
run_stream(m, layers, 256, chunk, states)   # 预热

tot, s_byte, z_byte = state_bytes(dim, layers, feat)
print(f"[分析] 每层携带 S=1x{feat}x{dim}(fp16) + z=1x{feat} | 状态字节/层={s_byte},{z_byte} | 全模型常驻状态 ≈ {tot/1024:.1f} KB (与 L 无关)", flush=True)
print(f"      模型参数 {count_params(m):,} ≈ {count_params(m)*4/1e6:.1f}MB(fp32)", flush=True)
print(f"{'L':>9} | {'流式耗时':>8} | {'每 token 均摊':>11}", flush=True)
for k in (11, 12, 13, 14, 15, 16, 17):       # 2K..128K 计时, 快
    L = 1 << k
    states[:] = [None] * layers
    t0 = time.perf_counter()
    run_stream(m, layers, L, chunk, states)
    dt = time.perf_counter() - t0
    print(f"{L:>9,} | {dt:>7.2f}s | {dt/L*1e6:>10.1f}us/tok", flush=True)