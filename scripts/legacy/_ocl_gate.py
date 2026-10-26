import sys, time, platform
import numpy as np

# ---- 1) 设备枚举 ----
import pyopencl as cl
print("=== OpenCL 平台/设备 ===", flush=True)
gpus = []
for p in cl.get_platforms():
    print("platform:", p.name, p.vendor, flush=True)
    for d in p.get_devices():
        typ = cl.device_type.to_string(d.type)
        print(f"  device: {d.name} | type={typ} | CU={d.max_compute_units} "
              f"| global={d.global_mem_size/2**30:.2f}GB | max_wg={d.max_work_group_size}", flush=True)
        if d.type & cl.device_type.GPU:
            gpus.append(d)
ctx = cl.Context(devices=[gpus[0]] if gpus else None)
compiler = "inteloclerr..."
print("picked:", gpus[0].name if gpus else "none", flush=True)

# ---- 2) CUDA-style 检查 CPU numpy 的 BLAS 基线 ----
def bench_np_matmul(N, iters=15, warmup=4):
    a = np.random.rand(N, N).astype(np.float32); b = np.random.rand(N, N).astype(np.float32)
    c = a @ b
    for _ in range(warmup): c = a @ b
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); a @ b; ts.append(time.perf_counter() - t0)
    ts.sort(); t = ts[len(ts)//2]
    return t, 2*N**3/t/1e9  # TFLOPS

# ---- 3) OpenCL tiled GEMM 内核 ----
KERNEL = r"""
#define TS 8
__kernel void gemm(const __global float* A, const __global float* B, __global float* C,
                   int M, int N, int K) {
    int row = get_global_id(0); // M
    int col = get_global_id(1); // N
    float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;
    int t;
    for (t=0; t+1<K; t+=2) {
        float a0 = A[row*K + t], a1 = A[row*K + t+1];
        float b0 = B[t*N + col], b1 = B[(t+1)*N + col];
        acc0 += a0*b0; acc0 += a1*b1;
    }
    for (; t<K; ++t) acc0 += A[row*K+t]*B[t*N+col];
    C[row*N + col] = acc0;
}
"""

def bench_ocl_matmul(ctx, N, iters=15, warmup=4):
    a = np.random.rand(N, N).astype(np.float32); b = np.random.rand(N, N).astype(np.float32)
    mf = cl.mem_flags
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build()
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    c = np.empty_like(a)
    ev = prg.gemm(q, (N, N), None, A_buf, B_buf, C_buf, np.int32(N), np.int32(N), np.int32(N))
    ev.wait()
    # 预加载内核
    for _ in range(warmup):
        prg.gemm(q, (N,N), None, A_buf, B_buf, C_buf, np.int32(N), np.int32(N), np.int32(N)).wait()
    def run():
        prg.gemm(q, (N,N), None, A_buf, B_buf, C_buf, np.int32(N), np.int32(N), np.int32(N)).wait()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); run(); ts.append(time.perf_counter()-t0)
    ts.sort(); t=ts[len(ts)//2]
    return t, 2*N**3/t/1e9

print("\n=== GEMM 对比（包含 host 同步往返，公平口径）===", flush=True)
print(f"{'N':>6} {'CPU numpy':>12} {'CPU GFLOPS':>11} {'OpenCL 逐元素':>14} {'OCL GFLOPS':>11}", flush=True)
for N in [512, 1024, 2048]:
    tcp, gcp = bench_np_matmul(N)
    toc, goc = bench_ocl_matmul(ctx, N, iters=8, warmup=3)
    print(f"{N:>6} {tcp*1000:>10.2f}ms {gcp:>10.1f} {toc*1000:>10.2f}ms {goc:>10.1f}", flush=True)

# ---- 4) 内存带宽（iGPU 天花板的真实约束）----
print("\n=== 内存带宽(Saxpy, 128MB buffers) ===", flush=True)
n = 1<<24  # 16M
x = np.random.rand(n).astype(np.float32); y = np.random.rand(n).astype(np.float32)
sax = r"__kernel void saxpy(__global float* y, const __global float* x, float a){ int i=get_global_id(0); y[i]+=a*x[i]; }"
prg2 = cl.Program(ctx, sax).build()
bY = cl.Buffer(ctx, cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR, hostbuf=y.copy())
bX = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
q = cl.CommandQueue(ctx)
for _ in range(3): prg2.saxpy(q,(n,),None,bY,bX,np.float32(0.5)).wait()
ts=[]
for _ in range(8):
    t0=time.perf_counter(); prg2.saxpy(q,(n,),None,bY,bX,np.float32(0.5)).wait(); ts.append(time.perf_counter()-t0)
ts.sort(); t=ts[len(ts)//2]
# 读写 x+y = 2*(128+128)MB
bytes_moved = 2*n*4*2
print(f"OpenCL saxpy: {t*1000:.1f}ms -> {bytes_moved/t/1e9:.1f} GB/s", flush=True)
# CPU numpy saxpy baseline
xc = x.copy(); yc = y.copy()
for _ in range(3): yc += 0.5*xc
ts=[]
for _ in range(8):
    t0=time.perf_counter(); yc += 0.5*xc; ts.append(time.perf_counter()-t0)
ts.sort(); tc=ts[len(ts)//2]
print(f"CPU  numpy  saxpy: {tc*1000:.1f}ms -> {2*n*4*2/tc/1e9:.1f} GB/s", flush=True)
print("BLAS backend:", platform.architecture(), flush=True)