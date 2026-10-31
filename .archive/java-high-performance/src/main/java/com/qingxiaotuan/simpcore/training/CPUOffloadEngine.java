package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import org.jocl.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import static org.jocl.CL.*;

/**
 * CPU卸载引擎 —— 实现ZeRO-1 + CPU Offloading混合策略
 * 
 * 核心策略：
 * 1. 模型参数：GPU ↔ CPU 按需传输（使用CL_MEM_USE_HOST_PTR零拷贝）
 * 2. 梯度：累积在CPU内存，量化压缩后传输到GPU
 * 3. 优化器状态（Adam的m/v）：完全驻留CPU，仅更新时同步到GPU
 * 4. 利用AMD APU统一内存特性，通过CL_MEM_USE_HOST_PTR实现零拷贝
 */
public class CPUOffloadEngine {

    private static final int OFFLOAD_BLOCK_SIZE = 1024 * 1024; // 1M elements per block
    private static final float COMPRESSION_THRESHOLD = 1e-6f;
    
    private final boolean useUnifiedMemory;
    private final int deviceId;
    private final cl_context context;
    private final cl_command_queue queue;
    
    private final ConcurrentHashMap<String, ParameterBlock[]> parameterBlocks = new ConcurrentHashMap<>();
    private final AtomicLong totalParameters = new AtomicLong(0);
    private final AtomicLong gpuMemoryUsed = new AtomicLong(0);
    
    private long maxGPUMemory;
    private boolean enableCompression;
    private boolean enableAsyncTransfer;
    
    private Path swapDirectory;
    private boolean useDiskSwap;
    
    public CPUOffloadEngine() {
        this.deviceId = GPUMatrixOperations.getCurrentDevice();
        this.context = GPUMatrixOperations.getContext();
        this.queue = GPUMatrixOperations.getCommandQueue(deviceId);
        this.useUnifiedMemory = true;
        this.enableCompression = false;
        this.enableAsyncTransfer = true;
        this.useDiskSwap = false;
        
        long[] globalMem = new long[1];
        clGetDeviceInfo(GPUMatrixOperations.getDeviceId(deviceId), 
            CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMem), null);
        this.maxGPUMemory = (long)(globalMem[0] * 0.8);
        
        System.out.println("[CPUOffload] Initialized. Max GPU memory: " + 
            (maxGPUMemory / (1024 * 1024)) + " MB");
        System.out.println("[CPUOffload] Unified memory: " + (useUnifiedMemory ? "enabled" : "disabled"));
    }
    
    public void setSwapDirectory(String path) {
        try {
            this.swapDirectory = Files.createTempDirectory("simpcore_swap_");
            this.useDiskSwap = true;
            System.out.println("[CPUOffload] Swap directory: " + swapDirectory);
        } catch (IOException e) {
            System.err.println("[CPUOffload] Failed to create swap directory: " + e.getMessage());
        }
    }
    
    public void enableCompression(boolean enable) {
        this.enableCompression = enable;
    }
    
    public void setMaxGPUMemory(long bytes) {
        this.maxGPUMemory = bytes;
    }
    
    /**
     * 注册一个参数张量并进行CPU卸载配置
     * @param paramName 参数名称
     * @param paramData 参数数据（CPU驻留）
     * @param numElements 参数元素数量
     */
    public void registerParameter(String paramName, float[] paramData, long numElements) {
        int numBlocks = (int)((numElements + OFFLOAD_BLOCK_SIZE - 1) / OFFLOAD_BLOCK_SIZE);
        ParameterBlock[] blocks = new ParameterBlock[numBlocks];
        
        long paramSizeBytes = numElements * 4;
        long gpuBudget = maxGPUMemory / 4; // 最多1/4 GPU内存用于参数
        
        for (int b = 0; b < numBlocks; b++) {
            int start = b * OFFLOAD_BLOCK_SIZE;
            int end = (int)Math.min(start + OFFLOAD_BLOCK_SIZE, numElements);
            int blockSize = end - start;
            
            boolean gpuResident = (b * blockSize * 4) < gpuBudget;
            
            ParameterBlock block = new ParameterBlock();
            block.name = paramName;
            block.blockIndex = b;
            block.start = start;
            block.size = blockSize;
            block.cpuData = paramData;
            block.cpuOffset = start;
            block.gpuResident = gpuResident;
            block.dirty = false;
            
            if (gpuResident && useUnifiedMemory) {
                block.gpuBuffer = clCreateBuffer(context, 
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    Sizeof.cl_float * blockSize, 
                    Pointer.to(paramData).withByteOffset(start * 4), null);
                gpuMemoryUsed.addAndGet(blockSize * 4L);
            }
            
            blocks[b] = block;
        }
        
        parameterBlocks.put(paramName, blocks);
        totalParameters.addAndGet(numElements);
        
        System.out.println("[CPUOffload] Registered param '" + paramName + "': " + 
            numElements + " elements, " + numBlocks + " blocks");
    }
    
    /**
     * 注册优化器状态（m/v）—— 始终驻留CPU
     */
    public OptimizerState registerOptimizerState(String paramName, float[] m, float[] v) {
        OptimizerState state = new OptimizerState();
        state.m = m;
        state.v = v;
        state.paramName = paramName;
        state.deviceId = deviceId;
        state.queue = queue;
        state.context = context;
        
        if (useDiskSwap && m.length > OFFLOAD_BLOCK_SIZE * 10) {
            try {
                state.swapFile = Files.createTempFile(swapDirectory, paramName + "_opt_", ".swap");
                state.useSwap = true;
            } catch (IOException e) {
                System.err.println("[CPUOffload] Failed to create swap for optimizer state: " + e.getMessage());
            }
        }
        
        return state;
    }
    
    /**
     * 确保参数块在GPU上，必要时从CPU传输
     */
    public cl_mem ensureOnGPU(String paramName, int elementOffset, int numElements) {
        ParameterBlock[] blocks = parameterBlocks.get(paramName);
        if (blocks == null) return null;
        
        int startBlock = elementOffset / OFFLOAD_BLOCK_SIZE;
        int endBlock = (elementOffset + numElements - 1) / OFFLOAD_BLOCK_SIZE;
        
        for (int b = startBlock; b <= endBlock && b < blocks.length; b++) {
            ParameterBlock block = blocks[b];
            if (!block.gpuResident) {
                evictBlock(findEvictionCandidate());
                transferToGPU(block);
            } else if (block.dirty && block.gpuBuffer == null) {
                transferToGPU(block);
            }
        }
        
        ParameterBlock firstBlock = blocks[startBlock];
        if (firstBlock.gpuBuffer != null) {
            return firstBlock.gpuBuffer;
        }
        return null;
    }
    
    /**
     * 从CPU传输参数块到GPU
     */
    private void transferToGPU(ParameterBlock block) {
        if (block.gpuBuffer != null) {
            clReleaseMemObject(block.gpuBuffer);
        }
        
        block.gpuBuffer = clCreateBuffer(context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * block.size,
            Pointer.to(block.cpuData).withByteOffset(block.cpuOffset * 4), null);
        block.gpuResident = true;
        block.dirty = false;
        gpuMemoryUsed.addAndGet(block.size * 4L);
    }
    
    /**
     * 从GPU回传参数块到CPU
     */
    public void syncToCPU(String paramName) {
        ParameterBlock[] blocks = parameterBlocks.get(paramName);
        if (blocks == null) return;
        
        for (ParameterBlock block : blocks) {
            if (block.gpuResident && block.gpuBuffer != null && block.dirty) {
                clEnqueueReadBuffer(queue, block.gpuBuffer, CL_TRUE, 0,
                    Sizeof.cl_float * block.size,
                    Pointer.to(block.cpuData).withByteOffset(block.cpuOffset * 4),
                    0, null, null);
                block.dirty = false;
            }
        }
    }
    
    /**
     * 将所有参数同步回CPU
     */
    public void syncAllToCPU() {
        for (String paramName : parameterBlocks.keySet()) {
            syncToCPU(paramName);
        }
        clFinish(queue);
    }
    
    /**
     * 查找可驱逐的块
     */
    private ParameterBlock findEvictionCandidate() {
        ParameterBlock candidate = null;
        for (ParameterBlock[] blocks : parameterBlocks.values()) {
            for (ParameterBlock block : blocks) {
                if (!block.gpuResident) continue;
                if (candidate == null || block.lastAccessTime < candidate.lastAccessTime) {
                    candidate = block;
                }
            }
        }
        return candidate;
    }
    
    /**
     * 驱逐GPU块到CPU
     */
    private void evictBlock(ParameterBlock block) {
        if (block == null || !block.gpuResident || block.gpuBuffer == null) return;
        
        if (block.dirty) {
            clEnqueueReadBuffer(queue, block.gpuBuffer, CL_TRUE, 0,
                Sizeof.cl_float * block.size,
                Pointer.to(block.cpuData).withByteOffset(block.cpuOffset * 4),
                0, null, null);
        }
        
        clReleaseMemObject(block.gpuBuffer);
        block.gpuBuffer = null;
        block.gpuResident = false;
        block.dirty = false;
        gpuMemoryUsed.addAndGet(-block.size * 4L);
    }
    
    /**
     * 使用ZeRO-1优化器更新（优化器状态驻留CPU）
     * @param weights 模型参数
     * @param gradients 梯度
     * @param m Adam一阶矩
     * @param v Adam二阶矩
     * @param lr 学习率
     * @param weightDecay 权重衰减
     * @param beta1 Adam beta1
     * @param beta2 Adam beta2
     * @param eps Adam eps
     * @param step 当前步数
     */
    public void zeroOffloadUpdate(float[] weights, float[] gradients, 
                                   float[] m, float[] v,
                                   float lr, float weightDecay,
                                   float beta1, float beta2, float eps, int step) {
        float biasCorrection1 = 1.0f - (float)Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float)Math.pow(beta2, step);
        float correctedLr = lr * (float)Math.sqrt(biasCorrection2) / biasCorrection1;
        
        int blockSize = 256 * 1024; // 256K elements per batch
        
        if (GPUMatrixOperations.isGpuAvailable() && useUnifiedMemory) {
            zeroOffloadUpdateGPU(weights, gradients, m, v, 
                correctedLr, weightDecay, beta1, beta2, eps, blockSize);
        } else {
            zeroOffloadUpdateCPU(weights, gradients, m, v, 
                correctedLr, weightDecay, beta1, beta2, eps);
        }
    }
    
    private void zeroOffloadUpdateCPU(float[] weights, float[] gradients,
                                       float[] m, float[] v,
                                       float lr, float weightDecay,
                                       float beta1, float beta2, float eps) {
        for (int i = 0; i < weights.length; i++) {
            float grad = gradients[i];
            
            if (Float.isNaN(grad) || Float.isInfinite(grad)) {
                grad = 0;
            }
            
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
            
            float denom = (float)Math.sqrt(v[i]) + eps;
            float update = lr * m[i] / denom;
            
            weights[i] -= update;
            
            if (weightDecay > 0) {
                weights[i] -= lr * weightDecay * weights[i];
            }
        }
    }
    
    private cl_kernel cachedZeroAdamKernel;
    private cl_program cachedZeroAdamProgram;
    
    private void ensureZeroAdamKernel() {
        if (cachedZeroAdamKernel != null) return;
        String kernelSource = 
            "__kernel void zero_adam_update(" +
            "  __global float* w, __global const float* g," +
            "  __global float* m, __global float* v," +
            "  float lr, float wd, float b1, float b2, float eps_val, int len) {" +
            "  int i = get_global_id(0);" +
            "  if (i >= len) return;" +
            "  float grad = g[i];" +
            "  if (!isnan(grad) && !isinf(grad)) {" +
            "    m[i] = b1 * m[i] + (1.0f - b1) * grad;" +
            "    v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;" +
            "    float denom = sqrt(v[i]) + eps_val;" +
            "    w[i] -= lr * m[i] / denom;" +
            "    if (wd > 0.0f) w[i] -= lr * wd * w[i];" +
            "  }" +
            "}";
        cachedZeroAdamProgram = clCreateProgramWithSource(context, 1, 
            new String[]{kernelSource}, null, null);
        clBuildProgram(cachedZeroAdamProgram, 0, null, null, null, null);
        cachedZeroAdamKernel = clCreateKernel(cachedZeroAdamProgram, "zero_adam_update", null);
    }

    private void zeroOffloadUpdateGPU(float[] weights, float[] gradients,
                                       float[] m, float[] v,
                                       float lr, float weightDecay,
                                       float beta1, float beta2, float eps,
                                       int blockSize) {
        ensureZeroAdamKernel();
        if (cachedZeroAdamKernel == null) {
            zeroOffloadUpdateCPU(weights, gradients, m, v, lr, weightDecay, beta1, beta2, eps);
            return;
        }
        
        int numBlocks = (weights.length + blockSize - 1) / blockSize;
        
        for (int b = 0; b < numBlocks; b++) {
            int start = b * blockSize;
            int end = Math.min(start + blockSize, weights.length);
            int size = end - start;
            
            cl_mem d_weights = clCreateBuffer(context, 
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * size, 
                Pointer.to(weights).withByteOffset(start * 4), null);
            cl_mem d_grad = clCreateBuffer(context, 
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * size, 
                Pointer.to(gradients).withByteOffset(start * 4), null);
            cl_mem d_m = clCreateBuffer(context, 
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * size, 
                Pointer.to(m).withByteOffset(start * 4), null);
            cl_mem d_v = clCreateBuffer(context, 
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * size, 
                Pointer.to(v).withByteOffset(start * 4), null);
            
            clSetKernelArg(cachedZeroAdamKernel, 0, Sizeof.cl_mem, Pointer.to(d_weights));
            clSetKernelArg(cachedZeroAdamKernel, 1, Sizeof.cl_mem, Pointer.to(d_grad));
            clSetKernelArg(cachedZeroAdamKernel, 2, Sizeof.cl_mem, Pointer.to(d_m));
            clSetKernelArg(cachedZeroAdamKernel, 3, Sizeof.cl_mem, Pointer.to(d_v));
            clSetKernelArg(cachedZeroAdamKernel, 4, Sizeof.cl_float, Pointer.to(new float[]{lr}));
            clSetKernelArg(cachedZeroAdamKernel, 5, Sizeof.cl_float, Pointer.to(new float[]{weightDecay}));
            clSetKernelArg(cachedZeroAdamKernel, 6, Sizeof.cl_float, Pointer.to(new float[]{beta1}));
            clSetKernelArg(cachedZeroAdamKernel, 7, Sizeof.cl_float, Pointer.to(new float[]{beta2}));
            clSetKernelArg(cachedZeroAdamKernel, 8, Sizeof.cl_float, Pointer.to(new float[]{eps}));
            clSetKernelArg(cachedZeroAdamKernel, 9, Sizeof.cl_int, Pointer.to(new int[]{size}));
            
            long[] globalWorkSize = new long[]{((size + 255) / 256) * 256};
            long[] localWorkSize = new long[]{256};
            clEnqueueNDRangeKernel(queue, cachedZeroAdamKernel, 1, null, globalWorkSize, localWorkSize, 
                0, null, null);
            
            clEnqueueReadBuffer(queue, d_weights, CL_TRUE, 0,
                Sizeof.cl_float * size,
                Pointer.to(weights).withByteOffset(start * 4), 0, null, null);
            
            clReleaseMemObject(d_weights);
            clReleaseMemObject(d_grad);
            clReleaseMemObject(d_m);
            clReleaseMemObject(d_v);
        }
    }
    
    /**
     * 梯度压缩（Top-k稀疏化）
     */
    public float[] compressGradients(float[] gradients, float compressionRatio) {
        if (!enableCompression || compressionRatio >= 1.0f) {
            return gradients;
        }
        
        int k = Math.max(1, (int)(gradients.length * compressionRatio));
        
        float[][] topK = new float[2][k];
        float[] values = new float[k];
        int[] indices = new int[k];
        
        for (int i = 0; i < k; i++) {
            values[i] = Float.NEGATIVE_INFINITY;
            indices[i] = -1;
        }
        
        for (int i = 0; i < gradients.length; i++) {
            float abs = Math.abs(gradients[i]);
            if (abs > values[0]) {
                values[0] = abs;
                indices[0] = i;
                for (int j = 0; j < k - 1 && values[j] > values[j + 1]; j++) {
                    float tmpVal = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = tmpVal;
                    int tmpIdx = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = tmpIdx;
                }
            }
        }
        
        float[] compressed = new float[gradients.length];
        for (int i = 0; i < k; i++) {
            if (indices[i] >= 0) {
                compressed[indices[i]] = gradients[indices[i]];
            }
        }
        
        return compressed;
    }
    
    /**
     * 获取GPU内存使用统计
     */
    public String getMemoryStats() {
        long totalParamBytes = totalParameters.get() * 4;
        long gpuBytes = gpuMemoryUsed.get();
        
        return String.format(
            "CPU Offload Stats:\n" +
            "  Total Parameters: %.2fB (%.2f GB)\n" +
            "  GPU Memory Used: %.2f MB / %.2f MB\n" +
            "  CPU Memory Used: %.2f GB\n" +
            "  Unified Memory: %s",
            totalParameters.get() / 1e9,
            totalParamBytes / (1024.0 * 1024 * 1024),
            gpuBytes / (1024.0 * 1024),
            maxGPUMemory / (1024.0 * 1024),
            (totalParamBytes - gpuBytes) / (1024.0 * 1024 * 1024),
            useUnifiedMemory ? "enabled" : "disabled"
        );
    }
    
    public void shutdown() {
        syncAllToCPU();
        
        for (ParameterBlock[] blocks : parameterBlocks.values()) {
            for (ParameterBlock block : blocks) {
                if (block.gpuBuffer != null) {
                    clReleaseMemObject(block.gpuBuffer);
                    block.gpuBuffer = null;
                }
            }
        }
        
        if (swapDirectory != null) {
            try {
                Files.walk(swapDirectory)
                    .sorted(java.util.Comparator.reverseOrder())
                    .map(java.nio.file.Path::toFile)
                    .forEach(File::delete);
            } catch (IOException e) {
                System.err.println("[CPUOffload] Failed to clean swap: " + e.getMessage());
            }
        }
        
        parameterBlocks.clear();
        System.out.println("[CPUOffload] Shutdown complete");
    }
    
    /**
     * 参数块描述
     */
    private static class ParameterBlock {
        String name;
        int blockIndex;
        int start;
        int size;
        float[] cpuData;
        int cpuOffset;
        boolean gpuResident;
        boolean dirty;
        cl_mem gpuBuffer;
        long lastAccessTime;
    }
    
    /**
     * 优化器状态描述
     */
    public static class OptimizerState {
        float[] m;
        float[] v;
        String paramName;
        Path swapFile;
        boolean useSwap;
        int deviceId;
        cl_command_queue queue;
        cl_context context;
        
        public float[] getM() { return m; }
        public float[] getV() { return v; }
        
        public void swapOut() {
            if (!useSwap || swapFile == null) return;
            try (FileChannel channel = FileChannel.open(swapFile, 
                    StandardOpenOption.WRITE, StandardOpenOption.CREATE)) {
                ByteBuffer buffer = ByteBuffer.allocate(m.length * 4 + v.length * 4);
                buffer.order(ByteOrder.nativeOrder());
                for (float f : m) buffer.putFloat(f);
                for (float f : v) buffer.putFloat(f);
                buffer.flip();
                channel.write(buffer);
                
                java.util.Arrays.fill(m, 0);
                java.util.Arrays.fill(v, 0);
            } catch (IOException e) {
                System.err.println("[OptimizerState] Swap out failed: " + e.getMessage());
            }
        }
        
        public void swapIn() {
            if (!useSwap || swapFile == null) return;
            try (FileChannel channel = FileChannel.open(swapFile, StandardOpenOption.READ)) {
                ByteBuffer buffer = ByteBuffer.allocate(m.length * 4 + v.length * 4);
                buffer.order(ByteOrder.nativeOrder());
                channel.read(buffer);
                buffer.flip();
                for (int i = 0; i < m.length; i++) m[i] = buffer.getFloat();
                for (int i = 0; i < v.length; i++) v[i] = buffer.getFloat();
            } catch (IOException e) {
                System.err.println("[OptimizerState] Swap in failed: " + e.getMessage());
            }
        }
    }
}
