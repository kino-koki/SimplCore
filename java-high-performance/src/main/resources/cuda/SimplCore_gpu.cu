/*!
 * SimplCore GPU Kernels (CUDA) - Java/JNI 加载版本
 * =================================================
 * 与 simplcore_pg/cuda/simplcore_kernels.cu 同源，供 Java 端的
 * GOEUatrixOperations 通过 JNI 加载 (libSimplCore_gpu.so / .dll)。
 *
 * 编译：
 *   nvcc -O3 -shared -Xcompiler -fPIC \
 *        -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" \
 *        -o libSimplCore_gpu.so SimplCore_gpu.cu -lcudart -lcublas
 *
 * 说明：早期架构文档把 Java 的 GPU 加速写作 CUDA (SimplCore_gpu.cu)，
 * 但实际交付使用的是 OpenCL/JOCL (resources/opencl/simplcore_kernels.cl)。
 * 这里补齐真实的 CUDA 实现，使两套描述都能落地；任一不可用时应优雅回退
 * 到 CPU/OpenCL 路径。
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>

typedef float scrc_float;

#define TILE 16

__global__ void matmul_shared_kernel(const scrc_float* A, const scrc_float* B,
                                      scrc_float* C, int M, int N, int K) {
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
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

extern "C" {
cudaError_t Java_com_qingxiaotuan_SimplCore_gpu_GOEUatrixOperations_matmul(
        const scrc_float* A, const scrc_float* B, scrc_float* C,
        int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t e = cudaGetLastError();
    return e;
}
}
