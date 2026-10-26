# -*- coding: utf-8 -*-
"""流式长上下文推理 —— 实测 SGC-CA 的峰值内存随上下文长度的增长.

架构事实: (内容自适应)线性注意力不存历史 token, 只维护每层固定 O(D*r) 状态,
因此**流式/分块推理下峰值内存与上下文长度几乎无关**.
对照 softmax 注意力: KV cache O(L*D) 必然线性增长.

这里对 SGC-CA LM(dim,L,feat,chunk) 在若干上下文长度 L 上做全流式推理,
用 Windows API 采样进程工作集峰值, 输出"每倍增上下文的内存增量".
"""
import os, gc, time, ctypes, torch
from sgc_v3 import build_model, count_params

# ---- Windows 进程工作集读数 (正确的 PROCESS_MEMORY_COUNTERS) ----
import ctypes as _ct
from ctypes import wintypes as _wt
_WINMM = _ct.WinDLL('Kernel32', use_last_error=True)
class _PMC(_ct.Structure):
    _fields_ = [("cb", _wt.DWORD), ("PageFaultCount", _wt.DWORD),
                ("PeakWorkingSetSize", _ct.c_size_t), ("WorkingSetSize", _ct.c_size_t),
                ("QuotaPagedPoolUsage", _ct.c_size_t), ("QuotaNonPagedPoolUsage", _ct.c_size_t),
                ("PagefileUsage", _ct.c_size_t), ("PeakPagefileUsage", _ct.c_size_t)]
_psapi = _ct.WinDLL('psapi', use_last_error=True)
_GetCurrentProcess = _WINMM.GetCurrentProcess
_GetCurrentProcess.restype = _wt.HANDLE   # 关键: 伪句柄须按指针大小返回
_GetProcessMemoryInfo = _psapi.GetProcessMemoryInfo
_GetProcessMemoryInfo.argtypes = (_wt.HANDLE, _ct.POINTER(_PMC), _wt.DWORD)
_GetProcessMemoryInfo.restype = _ct.c_long

def rss_mb():
    pm = _PMC(); pm.cb = _ct.sizeof(pm)
    ok = _GetProcessMemoryInfo(_GetCurrentProcess(), _ct.byref(pm), _ct.sizeof(pm))
    return pm.WorkingSetSize / (1024.0 * 1024.0) if ok else 0.0

def build_streamer(dim, layers, feat, chunk):
    """随机初始化 SGC-CA 编解码 LM, 返回 (model, run_stream)."""
    vocab = 2000
    # SGC-CA 的混合不依赖位置, pos_emb 取小块即可(仅与 chunk 同量级), 避免 O(L) 权重
    m = build_model(vocab, dim, layers, 4, chunk, 'ca', rank=8, ca_feat=feat, ca_chunk=chunk)
    m.eval()
    states = [None] * layers
    def run_stream(L, max_rss):
        """全流式: 逐块喂 token, 状态跨块携带, 不产生 O(L) 输入张量."""
        nonlocal states
        states = [None] * layers
        with torch.no_grad():
            for st in range(0, L, chunk):
                ed = min(L, st + chunk)
                idx = torch.randint(0, vocab, (1, ed - st))
                x = m.tok_emb(idx)          # SGC-CA 混合不含位置项, 顺序由状态递归编码 -> 无需 pos_emb
                for li, (mix, ffn) in enumerate(m.blocks):
                    if mix.__class__.__name__ == 'ContentAdaptiveMixer':
                        x, states[li] = mix.forward_stream(x, states[li])
                    else:
                        x = mix(x)
                    x = ffn(x)
                _ = m.head(m.ln_f(x))          # 仅当前 chunk 的 logits
                del idx, x
                r = rss_mb()
                if r > max_rss:
                    max_rss = r
        return max_rss
    return m, run_stream

def main():
    torch.set_num_threads(8)
    dim = 256; layers = 4; feat = 32; chunk = 256
    model, run_stream = build_streamer(dim, layers, feat, chunk)
    nparams = count_params(model)
    print(f"模型: SGC-CA dim={dim} L={layers} feat={feat} chunk={chunk}", flush=True)
    print(f"参数量: {nparams:,} (fp32≈{nparams*4/1e6:.1f}MB)", flush=True)
    # 基线
    for _ in range(2):
        run_stream(256, 0.0)
    gc.collect(); base = rss_mb()
    print(f"基线 RSS: {base:.1f} MB", flush=True)
    lengths = [1 << k for k in range(11, 21)]  # 2048 .. 1048576
    print(f"{'L':>10} | {'峰值RSS':>10} | ΔRSS(总) | ΔRSS vs 前/2^?", flush=True)
    prev = None
    for L in lengths:
        t0 = time.perf_counter()
        peak = run_stream(L, base)
        dt = time.perf_counter() - t0
        d_tot = peak - base
        d_prev = peak - (prev if prev is not None else base)
        print(f"{L:>10,} | {peak:>8.1f}MB | {d_tot:+7.1f}MB | {d_prev:+7.1f}MB | {(time.perf_counter()-t0):.0f}s", flush=True)
        prev = peak

if __name__ == '__main__':
    main()