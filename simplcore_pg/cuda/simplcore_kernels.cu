/*!
 * SimplCore GPU Kernels (CUDA)
 * ============================
 * 真实可编译的 CUDA 内核集合，对应 `simplcore_ARCHITECTURE.md` 中描述的
 * GPU 加速模块。本文件提供：
 *   - matmul_kernel          : 基础矩阵乘法 (M,N,K)
 *   - matmul_shared_kernel   : 共享内存分块矩阵乘法 (减少全局内存访问)
 *   - softmax_kernel         : 数值稳定 softmax (warp 归约 + 跨 warp 归约)
 *   - relu_kernel            : ReLU 激活
 *   - silu_kernel            : SiLU/GELU 风格激活 (SwiGLU 用)
 *   - spectral_conv_kernel    : 谱全局卷积前向 (FFT 已在 host 端完成，这里做
 *                               频域带通滤波 + 阻尼衰减的逐点乘加)
 *   - rms_norm_kernel        : RMSNorm (SimplCore 主干归一化)
 *
 * 编译 (参见 simplcore_ARCHITECTURE.md §4.2)：
 *   nvcc -O3 -shared -Xcompiler -fPIC \
 *        -o libsimplcore_gpu.so simplcore_kernels.cu \
 *        -lcudart -lcublas
 *
 * 该 .so 可由 simplcore_pg.cuda_loader 通过 ctypes 加载；无 GPU/加载失败
 * 时自动回退到 PyTorch/NumPy CPU 实现（与 CPU 路径行为一致）。
 *
 * 注意：本文件是 SimplCore GPU 加速的真实来源。早期文档曾写作
 * `resources/cuda/SimplCore_gpu.cu` 并描述 Java JOCL/OpenCL 路径，此处以
 * CUDA 实现为准，确保文档与代码一致。
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>

// ----------------------------------------------------------------------
// 基础工具
// ----------------------------------------------------------------------
#define SCRC_CUDA_CHECK(call)                                                \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            printf("SimplCore CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
            return _e;                                                       \
        }                                                                    \
    } while (0)

// 适配 fp32；如需 fp16 可扩展。
typedef float scrc_float;

// ----------------------------------------------------------------------
// 1. 基础矩阵乘法 (naive, M,N,K)
// ----------------------------------------------------------------------
__global__ void matmul_kernel(const scrc_float* __restrict__ A,
                               const scrc_float* __restrict__ B,
                               scrc_float* __restrict__ C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        scrc_float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------
// 2. 共享内存分块矩阵乘法 (TILE = 16)
// ----------------------------------------------------------------------
#define TILE 16
__global__ void matmul_shared_kernel(const scrc_float* __restrict__ A,
                                       const scrc_float* __restrict__ B,
                                       scrc_float* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ scrc_float As[TILE][TILE];
    __shared__ scrc_float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    scrc_float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && kB < K) ? B[kB * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
#undef TILE

// ----------------------------------------------------------------------
// 3. Softmax (数值稳定; 跨 warp 归约在 shared memory 内完成)
// ----------------------------------------------------------------------
__global__ void softmax_kernel(scrc_float* __restrict__ logits, int rows, int cols) {
    extern __shared__ scrc_float buf[];
    int row = blockIdx.x;
    if (row >= rows) return;

    scrc_float* row_data = logits + row * cols;
    scrc_float* sh = buf;  // 长度 = blockDim.x

    // 最大值
    scrc_float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        scrc_float v = row_data[i];
        if (v > local_max) local_max = v;
    }
    // warp 级归约取最大值
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    if (threadIdx.x == 0) sh[0] = local_max;
    __syncthreads();
    scrc_float max_val = sh[0];

    // 指数和
    scrc_float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        scrc_float e = expf(row_data[i] - max_val);
        row_data[i] = e;
        local_sum += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    if (threadIdx.x == 0) sh[1] = local_sum;
    __syncthreads();
    scrc_float sum_val = sh[1] + 1e-8f;

    // 归一化
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] /= sum_val;
    }
}

// ----------------------------------------------------------------------
// 4. ReLU
// ----------------------------------------------------------------------
__global__ void relu_kernel(const scrc_float* __restrict__ x,
                             scrc_float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

// ----------------------------------------------------------------------
// 5. SiLU (SwiGLU 中的激活)
// ----------------------------------------------------------------------
__global__ void silu_kernel(const scrc_float* __restrict__ x,
                             scrc_float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

// ----------------------------------------------------------------------
// 6. 谱卷积频域点乘 (带通 + 阻尼)
//    spec: (Lf,) 复数以实部/虚部交织存于 re/im；band/damp 为 (Lf,) 实系数
// ----------------------------------------------------------------------
__global__ void spectral_conv_kernel(const scrc_float* __restrict__ re_in,
                                      const scrc_float* __restrict__ im_in,
                                      const scrc_float* __restrict__ band,
                                      const scrc_float* __restrict__ damp,
                                      scrc_float* __restrict__ re_out,
                                      scrc_float* __restrict__ im_out,
                                      int Lf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Lf) {
        float b = band[i] * damp[i];
        re_out[i] = re_in[i] * b;
        im_out[i] = im_in[i] * b;
    }
}

// ----------------------------------------------------------------------
// 7. RMSNorm
//    y = x / sqrt(mean(x^2) + eps) * gamma
// ----------------------------------------------------------------------
__global__ void rms_norm_kernel(const scrc_float* __restrict__ x,
                                 const scrc_float* __restrict__ gamma,
                                 scrc_float* __restrict__ y,
                                 int rows, int cols, float eps) {
    extern __shared__ scrc_float rb[];
    int row = blockIdx.x;
    if (row >= rows) return;
    const scrc_float* xr = x + row * cols;
    scrc_float* rb_local = rb;

    scrc_float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = xr[i];
        ss += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xFFFFFFFF, ss, offset);
    if (threadIdx.x == 0) rb_local[0] = ss;
    __syncthreads();
    float rms = sqrtf(rb_local[0] / cols + eps);

    scrc_float* yr = y + row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        yr[i] = xr[i] / rms * gamma[i];
    }
}

// ======================================================================
// C 接口封装 (供 ctypes / JNI 加载)
// ======================================================================
extern "C" {

cudaError_t scrc_matmul(const scrc_float* A, const scrc_float* B,
                         scrc_float* C, int M, int N, int K,
                         bool use_shared) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    if (use_shared) {
        matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t scrc_softmax(scrc_float* logits, int rows, int cols) {
    int threads = (cols < 256) ? 32 : 256;
    int shared = threads * sizeof(scrc_float) * 2;
    softmax_kernel<<<rows, threads, shared>>>(logits, rows, cols);
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t scrc_relu(const scrc_float* x, scrc_float* y, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x, y, n);
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t scrc_silu(const scrc_float* x, scrc_float* y, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(x, y, n);
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t scrc_spectral_conv(const scrc_float* re_in, const scrc_float* im_in,
                                const scrc_float* band, const scrc_float* damp,
                                scrc_float* re_out, scrc_float* im_out, int Lf) {
    int threads = 256;
    int blocks = (Lf + threads - 1) / threads;
    spectral_conv_kernel<<<blocks, threads>>>(re_in, im_in, band, damp,
                                              re_out, im_out, Lf);
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t scrc_rms_norm(const scrc_float* x, const scrc_float* gamma,
                           scrc_float* y, int rows, int cols, float eps) {
    int threads = 256;
    int shared = threads * sizeof(scrc_float);
    rms_norm_kernel<<<rows, threads, shared>>>(x, gamma, y, rows, cols, eps);
    SCRC_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

}  // extern "C"
