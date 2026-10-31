#define BLOCK_SIZE 16
#define WAVE_SIZE 64
#define VECTOR_SIZE 4
#define WARP_SIZE 32

#define float4_swizzle(v, idx) (v.s0)

__kernel void matmul_kernel(__global const float* A, __global const float* B, __global float* C,
                            int M, int K, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__kernel void matmul_shared_kernel(__global const float* A, __global const float* B, __global float* C,
                                   int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__kernel void matmul_vectorized_kernel(__global const float* A, __global const float* B, __global float* C,
                                       int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    int cols_per_thread = 4;
    int base_col = col * cols_per_thread;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (base_col + 0 < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + base_col + 0];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a = sharedA[ty][k];
            float b = sharedB[k][tx];
            sum0 += a * b;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && base_col + 0 < N) {
        C[row * N + base_col + 0] = sum0;
    }
}

__kernel void matmul_relu_kernel(__global const float* A, __global const float* B, __global float* C,
                                 int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = max(sum, 0.0f);
    }
}

__kernel void matmul_silu_kernel(__global const float* A, __global const float* B, __global float* C,
                                 int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        float sigmoid = 1.0f / (1.0f + exp(-sum));
        C[row * N + col] = sum * sigmoid;
    }
}

__kernel void layer_norm_fused_kernel(__global const float* input, __global float* output,
                                      __global const float* gamma, __global const float* beta,
                                      int batch_size, int hidden_size) {
    __local float mean_buf[256];
    __local float var_buf[256];
    
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);
    int batch_idx = gid / hidden_size;
    int elem_idx = gid % hidden_size;
    
    if (gid < batch_size * hidden_size) {
        int base = batch_idx * hidden_size;
        
        float sum = 0.0f;
        int offset = lid;
        while (offset < hidden_size) {
            sum += input[base + offset];
            offset += local_size;
        }
        
        mean_buf[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                mean_buf[lid] += mean_buf[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float mean = mean_buf[0] / (float)hidden_size;
        
        float var_sum = 0.0f;
        offset = lid;
        while (offset < hidden_size) {
            float diff = input[base + offset] - mean;
            var_sum += diff * diff;
            offset += local_size;
        }
        
        var_buf[lid] = var_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                var_buf[lid] += var_buf[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float var = var_buf[0] / (float)hidden_size;
        float inv_std = 1.0f / sqrt(var + 1e-6f);
        
        output[gid] = gamma[elem_idx] * (input[gid] - mean) * inv_std + beta[elem_idx];
    }
}

__kernel void rms_norm_fused_kernel(__global const float* input, __global float* output,
                                    __global const float* weight, float eps,
                                    int batch_size, int hidden_size) {
    __local float rms_buf[256];
    
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);
    int batch_idx = gid / hidden_size;
    int elem_idx = gid % hidden_size;
    
    if (gid < batch_size * hidden_size) {
        int base = batch_idx * hidden_size;
        
        float sum_sq = 0.0f;
        int offset = lid;
        while (offset < hidden_size) {
            float val = input[base + offset];
            sum_sq += val * val;
            offset += local_size;
        }
        
        rms_buf[lid] = sum_sq;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                rms_buf[lid] += rms_buf[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float rms = sqrt(rms_buf[0] / (float)hidden_size + eps);
        output[gid] = weight[elem_idx] * input[gid] / rms;
    }
}

__kernel void softmax_kernel(__global float* data, int size) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    __local float shared[256];
    
    shared[lid] = (gid < size) ? data[gid] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s && (gid + s) < size) {
            shared[lid] = max(shared[lid], shared[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float max_val = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    shared[lid] = (gid < size) ? exp(data[gid] - max_val) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s && (gid + s) < size) {
            shared[lid] += shared[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float sum = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (gid < size) {
        data[gid] = exp(data[gid] - max_val) / sum;
    }
}

__kernel void relu_kernel(__global float* x, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        x[idx] = max(x[idx], 0.0f);
    }
}

__kernel void silu_kernel(__global float* x, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float sigmoid = 1.0f / (1.0f + exp(-x[idx]));
        x[idx] = x[idx] * sigmoid;
    }
}

__kernel void relu_silu_kernel(__global float* x, int size, int mode) {
    int idx = get_global_id(0);
    if (idx < size) {
        if (mode == 0) {
            x[idx] = max(x[idx], 0.0f);
        } else {
            float sigmoid = 1.0f / (1.0f + exp(-x[idx]));
            x[idx] = x[idx] * sigmoid;
        }
    }
}

__kernel void add_kernel(__global const float* A, __global const float* B, __global float* C, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

__kernel void sub_kernel(__global const float* A, __global const float* B, __global float* C, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

__kernel void mul_kernel(__global const float* A, float scalar, __global float* C, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

__kernel void hadamard_kernel(__global const float* A, __global const float* B, __global float* C, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__kernel void dot_product_kernel(__global const float* a, __global const float* b, __global float* result, int size) {
    __local float sdata[WARP_SIZE];
    
    int idx = get_global_id(0);
    int tid = get_local_id(0);
    int gid = get_group_id(0);
    
    float sum = 0.0f;
    for (int i = idx; i < size; i += get_global_size(0)) {
        sum += a[i] * b[i];
    }
    
    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0) {
        result[gid] = sdata[0];
    }
}

__kernel void outer_product_kernel(__global const float* a, __global const float* b, __global float* C, int m, int n) {
    int i = get_global_id(1);
    int j = get_global_id(0);
    
    if (i < m && j < n) {
        C[i * n + j] = a[i] * b[j];
    }
}

__kernel void transpose_kernel(__global const float* input, __global float* output, int rows, int cols) {
    int i = get_global_id(1);
    int j = get_global_id(0);
    
    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

__kernel void norm_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    
    if (idx == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_sq += input[i] * input[i];
        }
        output[0] = sqrt(sum_sq);
    }
}

__kernel void add_bias_kernel(__global float* input, __global const float* bias, int batch_size, int hidden_size) {
    int idx = get_global_id(0);
    int elem_idx = idx % hidden_size;
    
    if (idx < batch_size * hidden_size) {
        input[idx] += bias[elem_idx];
    }
}

__kernel void gelu_kernel(__global float* x, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float val = x[idx];
        float cube = val * val * val;
        float cdf = 0.5f * (1.0f + tanh(0.79788456f * (val + 0.044715f * cube)));
        x[idx] = val * cdf;
    }
}

__kernel void matmul_bias_silu_kernel(__global const float* A, __global const float* B, 
                                      __global float* C, __global const float* bias,
                                      int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        float result = sum + bias[col];
        float sigmoid = 1.0f / (1.0f + exp(-result));
        C[row * N + col] = result * sigmoid;
    }
}

__kernel void matmul_bias_gelu_kernel(__global const float* A, __global const float* B, 
                                      __global float* C, __global const float* bias,
                                      int M, int K, int N) {
    __local float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __local float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * BLOCK_SIZE + ty;
    int col = get_group_id(0) * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sharedA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + ty < K) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        float val = sum + bias[col];
        float cube = val * val * val;
        float cdf = 0.5f * (1.0f + tanh(0.79788456f * (val + 0.044715f * cube)));
        C[row * N + col] = val * cdf;
    }
}

__kernel void rms_norm_kernel(__global const float* input, __global float* output,
                              __global const float* weight, float eps,
                              int batch_size, int hidden_size) {
    int idx = get_global_id(0);
    int batch_idx = idx / hidden_size;
    int elem_idx = idx % hidden_size;
    
    if (idx < batch_size * hidden_size) {
        float sum_sq = 0.0f;
        int base = batch_idx * hidden_size;
        
        for (int i = 0; i < hidden_size; i++) {
            float val = input[base + i];
            sum_sq += val * val;
        }
        
        float rms = sqrt(sum_sq / (float)hidden_size + eps);
        output[idx] = weight[elem_idx] * input[idx] / rms;
    }
}

__kernel void adam_kernel(__global float* params, __global const float* grads,
                          __global float* m, __global float* v,
                          float lr, float beta1, float beta2, float eps, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        float g = grads[idx];
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float m_hat = m[idx] / (1.0f - beta1);
        float v_hat = v[idx] / (1.0f - beta2);
        
        params[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

__kernel void dropout_kernel(__global const float* input, __global float* output,
                             float dropout_rate, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        float rand_val = (float)fabs(sin((float)(idx * 7919))) * 0.5f + 0.5f;
        if (rand_val < dropout_rate) {
            output[idx] = 0.0f;
        } else {
            output[idx] = input[idx] / (1.0f - dropout_rate);
        }
    }
}

__kernel void scaled_dot_product_kernel(__global const float* Q, __global const float* K,
                                        __global const float* V, __global float* output,
                                        float scale, int batch_size, int n_heads, int head_dim, int seq_len) {
    int idx = get_global_id(0);
    
    if (idx < batch_size * n_heads * seq_len * head_dim) {
        int batch_idx = idx / (n_heads * seq_len * head_dim);
        int head_idx = (idx / (seq_len * head_dim)) % n_heads;
        int seq_idx = (idx / head_dim) % seq_len;
        int dim_idx = idx % head_dim;
        
        int base = batch_idx * n_heads * seq_len * head_dim;
        int q_base = base + head_idx * seq_len * head_dim;
        int k_base = base + head_idx * seq_len * head_dim;
        int v_base = base + head_idx * seq_len * head_dim;
        
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += Q[q_base + seq_idx * head_dim + i] * K[k_base + i * seq_len + seq_idx];
        }
        score *= scale;
        
        float max_score = score;
        for (int j = 0; j < seq_len; j++) {
            float other_score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                other_score += Q[q_base + seq_idx * head_dim + i] * K[k_base + i * seq_len + j];
            }
            other_score *= scale;
            max_score = max(max_score, other_score);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float other_score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                other_score += Q[q_base + seq_idx * head_dim + i] * K[k_base + i * seq_len + j];
            }
            other_score *= scale;
            sum_exp += exp(other_score - max_score);
        }
        
        float output_val = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float other_score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                other_score += Q[q_base + seq_idx * head_dim + i] * K[k_base + i * seq_len + j];
            }
            other_score *= scale;
            float attn_weight = exp(other_score - max_score) / sum_exp;
            output_val += attn_weight * V[v_base + j * head_dim + dim_idx];
        }
        
        output[idx] = output_val;
    }
}

__kernel void fused_add_norm_kernel(__global const float* input, __global const float* residual,
                                    __global float* output, __global const float* weight,
                                    __global const float* bias, int batch_size, int hidden_size) {
    __local float mean_buf[256];
    __local float var_buf[256];
    
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int local_size = get_local_size(0);
    int batch_idx = gid / hidden_size;
    int elem_idx = gid % hidden_size;
    
    if (gid < batch_size * hidden_size) {
        int base = batch_idx * hidden_size;
        float added = input[gid] + residual[gid];
        
        float sum = 0.0f;
        int offset = lid;
        while (offset < hidden_size) {
            sum += input[base + offset] + residual[base + offset];
            offset += local_size;
        }
        
        mean_buf[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                mean_buf[lid] += mean_buf[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float mean = mean_buf[0] / (float)hidden_size;
        
        float var_sum = 0.0f;
        offset = lid;
        while (offset < hidden_size) {
            float diff = (input[base + offset] + residual[base + offset]) - mean;
            var_sum += diff * diff;
            offset += local_size;
        }
        
        var_buf[lid] = var_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) {
                var_buf[lid] += var_buf[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float var = var_buf[0] / (float)hidden_size;
        float inv_std = 1.0f / sqrt(var + 1e-6f);
        
        output[gid] = weight[elem_idx] * (added - mean) * inv_std + bias[elem_idx];
    }
}

__kernel void clip_grad_kernel(__global float* grads, float max_norm, int size) {
    __local float sdata[256];
    
    int idx = get_global_id(0);
    int tid = get_local_id(0);
    int local_size = get_local_size(0);
    
    float val = (idx < size) ? grads[idx] * grads[idx] : 0.0f;
    sdata[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0) {
        float norm = sqrt(sdata[0]);
        float clip_coef = min(max_norm / (norm + 1e-6f), 1.0f);
        for (int i = 0; i < local_size && i + get_group_id(0) * local_size < size; i++) {
            grads[get_group_id(0) * local_size + i] *= clip_coef;
        }
    }
}

__kernel void adamw_kernel(__global float* params, __global const float* grads,
                           __global float* m, __global float* v,
                           float lr, float beta1, float beta2, float eps, 
                           float weight_decay, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        float g = grads[idx];
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float m_hat = m[idx] / (1.0f - beta1);
        float v_hat = v[idx] / (1.0f - beta2);
        
        params[idx] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[idx]);
    }
}

__kernel void layer_norm_grad_kernel(__global const float* grad_out, __global const float* input,
                                     __global const float* normalized, __global const float* weight,
                                     __global const float* mean, __global const float* variance,
                                     __global float* grad_input, __global float* grad_weight,
                                     __global float* grad_bias, float eps,
                                     int batch_size, int hidden_size) {
    int idx = get_global_id(0);
    int batch_idx = idx / hidden_size;
    int elem_idx = idx % hidden_size;
    
    if (idx < batch_size * hidden_size) {
        int base = batch_idx * hidden_size;
        float w = weight[elem_idx];
        float g = grad_out[idx];
        
        grad_weight[elem_idx] += g * normalized[idx];
        grad_bias[elem_idx] += g;
        
        float inv_std = 1.0f / sqrt(variance[batch_idx] + eps);
        float x_minus_mean = input[idx] - mean[batch_idx];
        
        float grad_norm = g * w;
        float grad_var = grad_norm * x_minus_mean * (-0.5f) * pow(variance[batch_idx] + eps, -1.5f);
        float grad_mean = grad_norm * (-inv_std);
        
        grad_input[idx] = grad_norm * inv_std;
        
        float mean_grad = 0.0f;
        float var_grad = 0.0f;
        
        for (int i = 0; i < hidden_size; i++) {
            float x_i = input[base + i];
            float diff = x_i - mean[batch_idx];
            mean_grad += grad_norm * (-inv_std) + grad_var * (-2.0f / (float)hidden_size) * diff;
            var_grad += grad_norm * x_minus_mean * (-0.5f) * pow(variance[batch_idx] + eps, -1.5f);
        }
        
        grad_input[idx] += mean_grad * (1.0f / (float)hidden_size);
    }
}

__kernel void matmul_grad_kernel(__global const float* grad_out, __global const float* A, 
                                 __global const float* B, __global float* grad_A, __global float* grad_B,
                                 int M, int K, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < M && col < N) {
        float grad_sum_A = 0.0f;
        float grad_sum_B = 0.0f;
        
        for (int k = 0; k < K; k++) {
            grad_sum_A += grad_out[row * N + col] * B[k * N + col];
            grad_sum_B += A[row * K + k] * grad_out[row * N + col];
        }
        
        grad_A[row * K + col] += grad_sum_A;
        grad_B[row * N + col] += grad_sum_B;
    }
}

__kernel void gelu_grad_kernel(__global const float* grad_out, __global const float* input,
                               __global float* grad_input, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        float x = input[idx];
        float cube = x * x * x;
        float tanh_val = tanh(0.79788456f * (x + 0.044715f * cube));
        float cdf = 0.5f * (1.0f + tanh_val);
        
        float pdf = exp(-x * x / 2.0f) / sqrt(2.0f * 3.14159265f);
        float grad = cdf + x * pdf * 0.79788456f * (1.0f + 0.134145f * x * x);
        
        grad_input[idx] = grad_out[idx] * grad;
    }
}

__kernel void silu_grad_kernel(__global const float* grad_out, __global const float* input,
                               __global float* grad_input, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + exp(-x));
        float grad = sigmoid * (1.0f + x * (1.0f - sigmoid));
        
        grad_input[idx] = grad_out[idx] * grad;
    }
}

__kernel void softmax_grad_kernel(__global const float* grad_out, __global const float* softmax_out,
                                  __global float* grad_input, int batch_size, int vocab_size) {
    int idx = get_global_id(0);
    
    if (idx < batch_size * vocab_size) {
        int batch_idx = idx / vocab_size;
        int elem_idx = idx % vocab_size;
        int base = batch_idx * vocab_size;
        
        float p = softmax_out[idx];
        float grad_sum = 0.0f;
        
        for (int j = 0; j < vocab_size; j++) {
            float grad = (elem_idx == j ? p * (1.0f - p) : -p * softmax_out[base + j]);
            grad_sum += grad_out[base + j] * grad;
        }
        
        grad_input[idx] = grad_sum;
    }
}

__kernel void embedding_grad_kernel(__global const float* grad_out, __global float* grad_weight,
                                    __global const int* indices, int embedding_dim, int num_indices) {
    int idx = get_global_id(0);
    
    if (idx < num_indices * embedding_dim) {
        int i = idx / embedding_dim;
        int j = idx % embedding_dim;
        int word_idx = indices[i];
        
        grad_weight[word_idx * embedding_dim + j] += grad_out[idx];
    }
}

__kernel void cross_entropy_grad_kernel(__global const float* logits, __global const int* targets,
                                        __global float* grad_input, int batch_size, int vocab_size) {
    int idx = get_global_id(0);
    
    if (idx < batch_size * vocab_size) {
        int batch_idx = idx / vocab_size;
        int elem_idx = idx % vocab_size;
        int base = batch_idx * vocab_size;
        
        float max_val = logits[base];
        for (int i = 1; i < vocab_size; i++) {
            max_val = max(max_val, logits[base + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += exp(logits[base + i] - max_val);
        }
        
        float softmax_val = exp(logits[idx] - max_val) / sum_exp;
        grad_input[idx] = (softmax_val - (elem_idx == targets[batch_idx] ? 1.0f : 0.0f)) / (float)batch_size;
    }
}