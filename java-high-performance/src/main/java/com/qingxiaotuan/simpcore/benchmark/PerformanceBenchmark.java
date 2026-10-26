package com.qingxiaotuan.simpcore.benchmark;

import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public final class PerformanceBenchmark {

    private static final Random RANDOM = new Random(42);
    private static final int WARMUP_ITERATIONS = 10;
    private static final int BENCHMARK_ITERATIONS = 100;

    private PerformanceBenchmark() {}

    public static BenchmarkResult runAllBenchmarks() {
        List<BenchmarkResult> results = new ArrayList<>();
        
        results.add(benchmarkMatmul(64, 64, 64));
        results.add(benchmarkMatmul(128, 128, 128));
        results.add(benchmarkMatmul(256, 256, 256));
        results.add(benchmarkMatmul(512, 512, 512));
        
        results.add(benchmarkSoftmax(1024));
        results.add(benchmarkSoftmax(4096));
        
        results.add(benchmarkReLU(1000000));
        
        results.add(benchmarkDotProduct(10000));
        
        results.add(benchmarkMemoryAllocation());
        
        return aggregateResults(results);
    }

    public static BenchmarkResult benchmarkMatmul(int M, int K, int N) {
        float[] A = generateRandomArray(M * K);
        float[] B = generateRandomArray(K * N);
        
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.matmul(A, B, M, K, N);
        }
        
        long start = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            MatrixOperations.matmul(A, B, M, K, N);
        }
        long cpuTime = System.nanoTime() - start;
        
        float cpuMs = cpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        double cpuGFLOPS = calculateGFLOPS(M, K, N, cpuTime / BENCHMARK_ITERATIONS);
        
        Float gpuMs = null;
        Double gpuGFLOPS = null;
        
        if (GPUMatrixOperations.isGpuAvailable()) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                GPUMatrixOperations.matmul(A, B, M, K, N);
            }
            
            start = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                GPUMatrixOperations.matmul(A, B, M, K, N);
            }
            long gpuTime = System.nanoTime() - start;
            
            gpuMs = gpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
            gpuGFLOPS = calculateGFLOPS(M, K, N, gpuTime / BENCHMARK_ITERATIONS);
        }
        
        return new BenchmarkResult(
            "Matmul " + M + "x" + K + "x" + N,
            "Matrix Multiplication",
            cpuMs,
            gpuMs,
            cpuGFLOPS,
            gpuGFLOPS
        );
    }

    public static BenchmarkResult benchmarkSoftmax(int size) {
        float[] logits = generateRandomArray(size);
        
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.softmax(logits.clone());
        }
        
        long start = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            MatrixOperations.softmax(logits.clone());
        }
        long cpuTime = System.nanoTime() - start;
        
        float cpuMs = cpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        
        Float gpuMs = null;
        if (GPUMatrixOperations.isGpuAvailable()) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                GPUMatrixOperations.softmax(logits.clone());
            }
            
            start = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                GPUMatrixOperations.softmax(logits.clone());
            }
            long gpuTime = System.nanoTime() - start;
            
            gpuMs = gpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        }
        
        return new BenchmarkResult(
            "Softmax " + size,
            "Softmax Activation",
            cpuMs,
            gpuMs,
            null,
            null
        );
    }

    public static BenchmarkResult benchmarkReLU(int size) {
        float[] x = generateRandomArray(size);
        
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.relu(x.clone());
        }
        
        long start = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            MatrixOperations.relu(x.clone());
        }
        long cpuTime = System.nanoTime() - start;
        
        float cpuMs = cpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        
        Float gpuMs = null;
        if (GPUMatrixOperations.isGpuAvailable()) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                GPUMatrixOperations.relu(x.clone());
            }
            
            start = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                GPUMatrixOperations.relu(x.clone());
            }
            long gpuTime = System.nanoTime() - start;
            
            gpuMs = gpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        }
        
        return new BenchmarkResult(
            "ReLU " + size,
            "ReLU Activation",
            cpuMs,
            gpuMs,
            null,
            null
        );
    }

    public static BenchmarkResult benchmarkDotProduct(int size) {
        float[] a = generateRandomArray(size);
        float[] b = generateRandomArray(size);
        
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.dotProduct(a, b);
        }
        
        long start = System.nanoTime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            MatrixOperations.dotProduct(a, b);
        }
        long cpuTime = System.nanoTime() - start;
        
        float cpuMs = cpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        
        Float gpuMs = null;
        if (GPUMatrixOperations.isGpuAvailable()) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                GPUMatrixOperations.dotProduct(a, b);
            }
            
            start = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                GPUMatrixOperations.dotProduct(a, b);
            }
            long gpuTime = System.nanoTime() - start;
            
            gpuMs = gpuTime / (1_000_000.0 * BENCHMARK_ITERATIONS);
        }
        
        return new BenchmarkResult(
            "DotProduct " + size,
            "Dot Product",
            cpuMs,
            gpuMs,
            null,
            null
        );
    }

    public static BenchmarkResult benchmarkMemoryAllocation() {
        MemoryManager memoryManager = MemoryManager.getInstance();
        memoryManager.resetStats();
        
        int iterations = 10000;
        int size = 1024;
        
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            float[] array = memoryManager.allocateFloatArray(size);
            memoryManager.release(array);
        }
        long time = System.nanoTime() - start;
        
        float avgTimeNs = time / (float) iterations;
        MemoryManager.MemoryStats stats = memoryManager.getStats();
        
        return new BenchmarkResult(
            "MemoryAllocation",
            "Memory Allocation (" + iterations + " x " + size + ")",
            avgTimeNs / 1000.0f,
            null,
            null,
            null
        );
    }

    private static double calculateGFLOPS(int M, int K, int N, long nanoseconds) {
        double operations = 2.0 * M * K * N;
        double seconds = nanoseconds / 1_000_000_000.0;
        return operations / seconds / 1_000_000_000.0;
    }

    private static float[] generateRandomArray(int size) {
        float[] array = new float[size];
        for (int i = 0; i < size; i++) {
            array[i] = (RANDOM.nextFloat() - 0.5f) * 2.0f;
        }
        return array;
    }

    private static BenchmarkResult aggregateResults(List<BenchmarkResult> results) {
        float totalCpuMs = 0;
        float totalGpuMs = 0;
        int cpuCount = 0;
        int gpuCount = 0;
        
        for (BenchmarkResult result : results) {
            if (result.cpuTimeMs != null) {
                totalCpuMs += result.cpuTimeMs;
                cpuCount++;
            }
            if (result.gpuTimeMs != null) {
                totalGpuMs += result.gpuTimeMs;
                gpuCount++;
            }
        }
        
        return new BenchmarkResult(
            "Aggregate",
            "Overall Performance",
            cpuCount > 0 ? totalCpuMs / cpuCount : null,
            gpuCount > 0 ? totalGpuMs / gpuCount : null,
            null,
            null
        );
    }

    public static void printResults(List<BenchmarkResult> results) {
        System.out.println("\n=== SimpCore Performance Benchmark Results ===");
        System.out.println("==============================================");
        System.out.printf("%-30s %-15s %-15s %-15s %-15s%n",
            "Benchmark", "CPU Time (ms)", "GPU Time (ms)", "CPU GFLOPS", "GPU GFLOPS");
        System.out.println("----------------------------------------------");
        
        for (BenchmarkResult result : results) {
            System.out.printf("%-30s ", result.name);
            System.out.printf("%-15.4f ", result.cpuTimeMs != null ? result.cpuTimeMs : 0);
            System.out.printf("%-15.4f ", result.gpuTimeMs != null ? result.gpuTimeMs : 0);
            System.out.printf("%-15.2f ", result.cpuGFLOPS != null ? result.cpuGFLOPS : 0);
            System.out.printf("%-15.2f%n", result.gpuGFLOPS != null ? result.gpuGFLOPS : 0);
        }
        
        System.out.println("==============================================");
    }

    public static class BenchmarkResult {
        public final String name;
        public final String description;
        public final Float cpuTimeMs;
        public final Float gpuTimeMs;
        public final Double cpuGFLOPS;
        public final Double gpuGFLOPS;
        
        public BenchmarkResult(String name, String description,
                               Float cpuTimeMs, Float gpuTimeMs,
                               Double cpuGFLOPS, Double gpuGFLOPS) {
            this.name = name;
            this.description = description;
            this.cpuTimeMs = cpuTimeMs;
            this.gpuTimeMs = gpuTimeMs;
            this.cpuGFLOPS = cpuGFLOPS;
            this.gpuGFLOPS = gpuGFLOPS;
        }
        
        public double getSpeedup() {
            if (cpuTimeMs != null && gpuTimeMs != null && gpuTimeMs > 0) {
                return cpuTimeMs / gpuTimeMs;
            }
            return 1.0;
        }
    }
}