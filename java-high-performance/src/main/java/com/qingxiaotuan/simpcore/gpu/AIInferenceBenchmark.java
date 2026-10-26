package com.qingxiaotuan.simpcore.gpu;

import org.jocl.*;
import java.util.Random;

import static org.jocl.CL.*;

public class AIInferenceBenchmark {

    private static final Random random = new Random(42);
    private static final int WARMUP_ITERATIONS = 10;
    private static final int MEASURE_ITERATIONS = 20;

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("SimplCore AI Inference Pipeline Benchmark");
        System.out.println("========================================\n");

        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("ERROR: GPU not available!");
            return;
        }

        System.out.println("=== System Information ===");
        System.out.println(GPUMatrixOperations.getDeviceInfo());

        System.out.println("\n=== End-to-End AI Task Benchmark (Optimized) ===");
        benchmarkOptimizedAITask();

        System.out.println("\n========================================");
        System.out.println("Benchmark Complete!");
        System.out.println("========================================");
    }

    private static void benchmarkOptimizedAITask() {
        int batchSize = 1;
        int seqLen = 64;
        int hiddenSize = 512;
        int mlpHidden = hiddenSize * 4;
        int seqHidden = batchSize * seqLen;

        float[] inputEmbedding = new float[seqHidden * hiddenSize];
        float[] attentionWeights = new float[seqHidden * seqLen];
        float[] mlpW1 = new float[hiddenSize * mlpHidden];
        float[] mlpW2 = new float[mlpHidden * hiddenSize];
        float[] mlpBias = new float[mlpHidden];
        float[] lnGamma = new float[hiddenSize];
        float[] lnBeta = new float[hiddenSize];

        for (int i = 0; i < inputEmbedding.length; i++) inputEmbedding[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < attentionWeights.length; i++) attentionWeights[i] = random.nextFloat();
        for (int i = 0; i < mlpW1.length; i++) mlpW1[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < mlpW2.length; i++) mlpW2[i] = random.nextFloat() * 2 - 1;
        for (int i = 0; i < mlpBias.length; i++) mlpBias[i] = random.nextFloat();
        for (int i = 0; i < lnGamma.length; i++) lnGamma[i] = 1.0f;
        for (int i = 0; i < lnBeta.length; i++) lnBeta[i] = 0.0f;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runAITaskCPU(inputEmbedding, attentionWeights, mlpW1, mlpW2, mlpBias, lnGamma, lnBeta, batchSize, seqLen, hiddenSize);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            runAITaskCPU(inputEmbedding, attentionWeights, mlpW1, mlpW2, mlpBias, lnGamma, lnBeta, batchSize, seqLen, hiddenSize);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        cl_context context = GPUMatrixOperations.getContext();
        cl_command_queue queue = GPUMatrixOperations.getCommandQueue(GPUMatrixOperations.getCurrentDevice());

        cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * inputEmbedding.length, Pointer.to(inputEmbedding), null);
        cl_mem d_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * attentionWeights.length, Pointer.to(attentionWeights), null);
        cl_mem d_mlpW1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * mlpW1.length, Pointer.to(mlpW1), null);
        cl_mem d_mlpW2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * mlpW2.length, Pointer.to(mlpW2), null);
        cl_mem d_mlpBias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * mlpBias.length, Pointer.to(mlpBias), null);
        cl_mem d_lnGamma = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * lnGamma.length, Pointer.to(lnGamma), null);
        cl_mem d_lnBeta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            Sizeof.cl_float * lnBeta.length, Pointer.to(lnBeta), null);

        cl_mem d_attended = clCreateBuffer(context, CL_MEM_READ_WRITE,
            Sizeof.cl_float * seqHidden * hiddenSize, null, null);
        cl_mem d_ln1Output = clCreateBuffer(context, CL_MEM_READ_WRITE,
            Sizeof.cl_float * seqHidden * hiddenSize, null, null);
        cl_mem d_mlpIn = clCreateBuffer(context, CL_MEM_READ_WRITE,
            Sizeof.cl_float * seqHidden * mlpHidden, null, null);
        cl_mem d_mlpOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
            Sizeof.cl_float * seqHidden * hiddenSize, null, null);
        cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            Sizeof.cl_float * seqHidden * hiddenSize, null, null);

        int BLOCK_SIZE = 32;

        cl_kernel matmulKernel = GPUMatrixOperations.kernels.get("matmul_shared_kernel");
        cl_kernel layerNormKernel = GPUMatrixOperations.kernels.get("layer_norm_kernel");
        cl_kernel addBiasKernel = GPUMatrixOperations.kernels.get("add_bias_kernel");
        cl_kernel geluKernel = GPUMatrixOperations.kernels.get("gelu_kernel");
        cl_kernel matmulBiasGeluKernel = GPUMatrixOperations.kernels.get("matmul_bias_gelu_kernel");

        float[] result = new float[seqHidden * hiddenSize];

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runOptimizedAITaskGPU(queue, matmulKernel, layerNormKernel, addBiasKernel, geluKernel,
                d_input, d_attn, d_mlpW1, d_mlpW2, d_mlpBias, d_lnGamma, d_lnBeta,
                d_attended, d_ln1Output, d_mlpIn, d_mlpOut, d_output,
                seqHidden, seqLen, hiddenSize, mlpHidden, BLOCK_SIZE, result);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            runOptimizedAITaskGPU(queue, matmulKernel, layerNormKernel, addBiasKernel, geluKernel,
                d_input, d_attn, d_mlpW1, d_mlpW2, d_mlpBias, d_lnGamma, d_lnBeta,
                d_attended, d_ln1Output, d_mlpIn, d_mlpOut, d_output,
                seqHidden, seqLen, hiddenSize, mlpHidden, BLOCK_SIZE, result);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            runDeepFusedAITaskGPU(queue, matmulKernel, layerNormKernel, matmulBiasGeluKernel,
                d_input, d_attn, d_mlpW1, d_mlpW2, d_mlpBias, d_lnGamma, d_lnBeta,
                d_attended, d_ln1Output, d_mlpIn, d_mlpOut, d_output,
                seqHidden, seqLen, hiddenSize, mlpHidden, BLOCK_SIZE, result);
        }

        long fusedTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            runDeepFusedAITaskGPU(queue, matmulKernel, layerNormKernel, matmulBiasGeluKernel,
                d_input, d_attn, d_mlpW1, d_mlpW2, d_mlpBias, d_lnGamma, d_lnBeta,
                d_attended, d_ln1Output, d_mlpIn, d_mlpOut, d_output,
                seqHidden, seqLen, hiddenSize, mlpHidden, BLOCK_SIZE, result);
            fusedTotal += System.nanoTime() - start;
        }
        double fusedAvg = fusedTotal / MEASURE_ITERATIONS / 1_000_000.0;

        clReleaseMemObject(d_input);
        clReleaseMemObject(d_attn);
        clReleaseMemObject(d_mlpW1);
        clReleaseMemObject(d_mlpW2);
        clReleaseMemObject(d_mlpBias);
        clReleaseMemObject(d_lnGamma);
        clReleaseMemObject(d_lnBeta);
        clReleaseMemObject(d_attended);
        clReleaseMemObject(d_ln1Output);
        clReleaseMemObject(d_mlpIn);
        clReleaseMemObject(d_mlpOut);
        clReleaseMemObject(d_output);

        double speedup = cpuAvg / gpuAvg;
        double fusedSpeedup = cpuAvg / fusedAvg;

        System.out.printf("End-to-End AI Task [BS=%d, Seq=%d, H=%d]:\n", batchSize, seqLen, hiddenSize);
        System.out.printf("  CPU:           %.2f ms\n", cpuAvg);
        System.out.printf("  GPU Optimized: %.2f ms, Speedup=%.2fx\n", gpuAvg, speedup);
        System.out.printf("  Deep Fused:    %.2f ms, Speedup=%.2fx\n", fusedAvg, fusedSpeedup);
        System.out.println();
    }

    private static float[] runAITaskCPU(float[] input, float[] attn, float[] mlpW1, float[] mlpW2,
                                        float[] mlpBias, float[] lnGamma, float[] lnBeta,
                                        int batchSize, int seqLen, int hiddenSize) {
        int seqHidden = batchSize * seqLen;
        int mlpHidden = hiddenSize * 4;

        float[] attended = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(attn, input, seqHidden, seqLen, hiddenSize);
        attended = com.qingxiaotuan.simpcore.math.MatrixOperations.layerNorm(attended, lnGamma, lnBeta, seqHidden, hiddenSize);

        float[] mlpIn = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(attended, mlpW1, seqHidden, hiddenSize, mlpHidden);
        mlpIn = com.qingxiaotuan.simpcore.math.MatrixOperations.addBias(mlpIn, mlpBias, seqHidden, mlpHidden);
        mlpIn = com.qingxiaotuan.simpcore.math.MatrixOperations.gelu(mlpIn);

        float[] output = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(mlpIn, mlpW2, seqHidden, mlpHidden, hiddenSize);
        output = com.qingxiaotuan.simpcore.math.MatrixOperations.layerNorm(output, lnGamma, lnBeta, seqHidden, hiddenSize);

        return output;
    }

    private static void runOptimizedAITaskGPU(cl_command_queue queue, cl_kernel matmulKernel, cl_kernel layerNormKernel,
                                              cl_kernel addBiasKernel, cl_kernel geluKernel,
                                              cl_mem d_input, cl_mem d_attn, cl_mem d_mlpW1, cl_mem d_mlpW2,
                                              cl_mem d_mlpBias, cl_mem d_lnGamma, cl_mem d_lnBeta,
                                              cl_mem d_attended, cl_mem d_ln1Output, cl_mem d_mlpIn,
                                              cl_mem d_mlpOut, cl_mem d_output,
                                              int seqHidden, int seqLen, int hiddenSize, int mlpHidden,
                                              int BLOCK_SIZE, float[] result) {
        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(d_attn));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(d_input));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(d_attended));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqLen}));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws1 = new long[]{
            ((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, gws1, null, 0, null, null);

        clSetKernelArg(layerNormKernel, 0, Sizeof.cl_mem, Pointer.to(d_attended));
        clSetKernelArg(layerNormKernel, 1, Sizeof.cl_mem, Pointer.to(d_ln1Output));
        clSetKernelArg(layerNormKernel, 2, Sizeof.cl_mem, Pointer.to(d_lnGamma));
        clSetKernelArg(layerNormKernel, 3, Sizeof.cl_mem, Pointer.to(d_lnBeta));
        clSetKernelArg(layerNormKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(layerNormKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws2 = new long[]{seqHidden * hiddenSize};
        long[] lws2 = new long[]{256};
        clEnqueueNDRangeKernel(queue, layerNormKernel, 1, null, gws2, lws2, 0, null, null);

        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(d_ln1Output));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(d_mlpW1));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{mlpHidden}));

        long[] gws3 = new long[]{
            ((mlpHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, gws3, null, 0, null, null);

        clSetKernelArg(addBiasKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(addBiasKernel, 1, Sizeof.cl_mem, Pointer.to(d_mlpBias));
        clSetKernelArg(addBiasKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(addBiasKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{mlpHidden}));

        long[] gws4 = new long[]{seqHidden * mlpHidden};
        clEnqueueNDRangeKernel(queue, addBiasKernel, 1, null, gws4, lws2, 0, null, null);

        clSetKernelArg(geluKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(geluKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{seqHidden * mlpHidden}));

        clEnqueueNDRangeKernel(queue, geluKernel, 1, null, gws4, lws2, 0, null, null);

        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(d_mlpW2));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(d_mlpOut));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{mlpHidden}));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws5 = new long[]{
            ((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, gws5, null, 0, null, null);

        clSetKernelArg(layerNormKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpOut));
        clSetKernelArg(layerNormKernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
        clSetKernelArg(layerNormKernel, 2, Sizeof.cl_mem, Pointer.to(d_lnGamma));
        clSetKernelArg(layerNormKernel, 3, Sizeof.cl_mem, Pointer.to(d_lnBeta));
        clSetKernelArg(layerNormKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(layerNormKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        clEnqueueNDRangeKernel(queue, layerNormKernel, 1, null, gws2, lws2, 0, null, null);

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
            Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);
    }

    private static void runDeepFusedAITaskGPU(cl_command_queue queue, cl_kernel matmulKernel, cl_kernel layerNormKernel,
                                              cl_kernel matmulBiasGeluKernel,
                                              cl_mem d_input, cl_mem d_attn, cl_mem d_mlpW1, cl_mem d_mlpW2,
                                              cl_mem d_mlpBias, cl_mem d_lnGamma, cl_mem d_lnBeta,
                                              cl_mem d_attended, cl_mem d_ln1Output, cl_mem d_mlpIn,
                                              cl_mem d_mlpOut, cl_mem d_output,
                                              int seqHidden, int seqLen, int hiddenSize, int mlpHidden,
                                              int BLOCK_SIZE, float[] result) {
        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(d_attn));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(d_input));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(d_attended));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqLen}));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws1 = new long[]{
            ((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, gws1, null, 0, null, null);

        clSetKernelArg(layerNormKernel, 0, Sizeof.cl_mem, Pointer.to(d_attended));
        clSetKernelArg(layerNormKernel, 1, Sizeof.cl_mem, Pointer.to(d_ln1Output));
        clSetKernelArg(layerNormKernel, 2, Sizeof.cl_mem, Pointer.to(d_lnGamma));
        clSetKernelArg(layerNormKernel, 3, Sizeof.cl_mem, Pointer.to(d_lnBeta));
        clSetKernelArg(layerNormKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(layerNormKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws2 = new long[]{seqHidden * hiddenSize};
        long[] lws2 = new long[]{256};
        clEnqueueNDRangeKernel(queue, layerNormKernel, 1, null, gws2, lws2, 0, null, null);

        clSetKernelArg(matmulBiasGeluKernel, 0, Sizeof.cl_mem, Pointer.to(d_ln1Output));
        clSetKernelArg(matmulBiasGeluKernel, 1, Sizeof.cl_mem, Pointer.to(d_mlpW1));
        clSetKernelArg(matmulBiasGeluKernel, 2, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(matmulBiasGeluKernel, 3, Sizeof.cl_mem, Pointer.to(d_mlpBias));
        clSetKernelArg(matmulBiasGeluKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulBiasGeluKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
        clSetKernelArg(matmulBiasGeluKernel, 6, Sizeof.cl_int, Pointer.to(new int[]{mlpHidden}));

        long[] gws3 = new long[]{
            ((mlpHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulBiasGeluKernel, 2, null, gws3, null, 0, null, null);

        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpIn));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(d_mlpW2));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(d_mlpOut));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{mlpHidden}));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        long[] gws5 = new long[]{
            ((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
            ((seqHidden + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
        };
        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, gws5, null, 0, null, null);

        clSetKernelArg(layerNormKernel, 0, Sizeof.cl_mem, Pointer.to(d_mlpOut));
        clSetKernelArg(layerNormKernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
        clSetKernelArg(layerNormKernel, 2, Sizeof.cl_mem, Pointer.to(d_lnGamma));
        clSetKernelArg(layerNormKernel, 3, Sizeof.cl_mem, Pointer.to(d_lnBeta));
        clSetKernelArg(layerNormKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{seqHidden}));
        clSetKernelArg(layerNormKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

        clEnqueueNDRangeKernel(queue, layerNormKernel, 1, null, gws2, lws2, 0, null, null);

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
            Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);
    }
}
