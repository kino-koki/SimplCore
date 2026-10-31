package com.qingxiaotuan.simpcore.gpu;

import java.util.Random;

public class GPUPowerBenchmark {

    private static final Random random = new Random(42);
    private static final int WARMUP_ITERATIONS = 5;
    private static final int MEASURE_ITERATIONS = 10;

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("SimplCore GPU Performance Benchmark");
        System.out.println("========================================\n");

        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("ERROR: GPU not available!");
            return;
        }

        System.out.println("=== System Information ===");
        System.out.println(GPUMatrixOperations.getDeviceInfo());

        System.out.println("\n=== Matrix Multiplication Benchmark ===");
        int[] matrixSizes = {64, 128, 256, 512, 1024};
        for (int size : matrixSizes) {
            benchmarkMatmul(size);
        }

        System.out.println("\n=== Neural Network Operators Benchmark ===");
        int[] vectorSizes = {1024, 4096, 16384, 65536};
        for (int size : vectorSizes) {
            benchmarkSoftmax(size);
            benchmarkReLU(size);
            benchmarkSiLU(size);
        }

        System.out.println("\n=== Vector Operations Benchmark ===");
        for (int size : new int[]{1024, 4096, 16384}) {
            benchmarkVectorAdd(size);
            benchmarkDotProduct(size);
        }

        System.out.println("\n========================================");
        System.out.println("Benchmark Complete!");
        System.out.println("========================================");
    }

    private static void benchmarkMatmul(int size) {
        float[] A = new float[size * size];
        float[] B = new float[size * size];
        for (int i = 0; i < size * size; i++) {
            A[i] = random.nextFloat();
            B[i] = random.nextFloat();
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.matmul(A, B, size, size, size);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.matmul(A, B, size, size, size);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, size, size, size);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, size, size, size);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;
        double gflops = (2.0 * size * size * size) / (gpuAvg * 1_000_000);

        System.out.printf("Matmul [%dx%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx, GFLOPS=%.1f\n",
                size, size, gpuAvg, cpuAvg, speedup, gflops);
    }

    private static void benchmarkSoftmax(int size) {
        float[] input = new float[size];
        for (int i = 0; i < size; i++) {
            input[i] = random.nextFloat() * 10 - 5;
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.softmax(input.clone());
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.softmax(input.clone());
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(input.clone());
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(input.clone());
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;

        System.out.printf("Softmax [%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx\n",
                size, gpuAvg, cpuAvg, speedup);
    }

    private static void benchmarkReLU(int size) {
        float[] input = new float[size];
        for (int i = 0; i < size; i++) {
            input[i] = random.nextFloat() * 20 - 10;
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.relu(input);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.relu(input);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.relu(input);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.relu(input);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;

        System.out.printf("ReLU [%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx\n",
                size, gpuAvg, cpuAvg, speedup);
    }

    private static void benchmarkSiLU(int size) {
        float[] input = new float[size];
        for (int i = 0; i < size; i++) {
            input[i] = random.nextFloat() * 20 - 10;
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.silu(input);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.silu(input);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.silu(input);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.silu(input);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;

        System.out.printf("SiLU [%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx\n",
                size, gpuAvg, cpuAvg, speedup);
    }

    private static void benchmarkVectorAdd(int size) {
        float[] A = new float[size];
        float[] B = new float[size];
        for (int i = 0; i < size; i++) {
            A[i] = random.nextFloat();
            B[i] = random.nextFloat();
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.add(A, B);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.add(A, B);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.add(A, B);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.add(A, B);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;

        System.out.printf("VectorAdd [%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx\n",
                size, gpuAvg, cpuAvg, speedup);
    }

    private static void benchmarkDotProduct(int size) {
        float[] A = new float[size];
        float[] B = new float[size];
        for (int i = 0; i < size; i++) {
            A[i] = random.nextFloat();
            B[i] = random.nextFloat();
        }

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GPUMatrixOperations.dotProduct(A, B);
        }

        long gpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            GPUMatrixOperations.dotProduct(A, B);
            gpuTotal += System.nanoTime() - start;
        }
        double gpuAvg = gpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(A, B);
        }

        long cpuTotal = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(A, B);
            cpuTotal += System.nanoTime() - start;
        }
        double cpuAvg = cpuTotal / MEASURE_ITERATIONS / 1_000_000.0;

        double speedup = cpuAvg / gpuAvg;

        System.out.printf("DotProduct [%d]: GPU=%.2f ms, CPU=%.2f ms, Speedup=%.2fx\n",
                size, gpuAvg, cpuAvg, speedup);
    }
}