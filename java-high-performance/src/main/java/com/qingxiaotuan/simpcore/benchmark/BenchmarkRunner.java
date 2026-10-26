package com.qingxiaotuan.simpcore.benchmark;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryManager;

import java.util.Random;
import java.util.concurrent.*;

public class BenchmarkRunner {

    private static final Random RANDOM = new Random(42);
    private static final int WARMUP_ITERATIONS = 10;
    private static final int MEASURE_ITERATIONS = 50;

    public static BenchmarkResult runMatmulBenchmark(int M, int K, int N) {
        float[] A = generateRandomArray(M * K);
        float[] B = generateRandomArray(K * N);

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.matmul(A, B, M, K, N);
        }

        long totalTime = 0;
        float[] result = null;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            result = MatrixOperations.matmul(A, B, M, K, N);
            long end = System.nanoTime();
            totalTime += end - start;
        }

        double avgTimeMs = (totalTime / (double) MEASURE_ITERATIONS) / 1_000_000;
        double gflops = calculateGFLOPS(M, K, N, avgTimeMs);

        return new BenchmarkResult(
            "Matmul " + M + "x" + K + "x" + N,
            avgTimeMs,
            gflops,
            M * K * N * 2L,
            result.length * Float.BYTES
        );
    }

    public static BenchmarkResult runSoftmaxBenchmark(int size) {
        float[] logits = generateRandomArray(size);

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.softmax(logits);
        }

        long totalTime = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            MatrixOperations.softmax(logits);
            long end = System.nanoTime();
            totalTime += end - start;
        }

        double avgTimeMs = (totalTime / (double) MEASURE_ITERATIONS) / 1_000_000;

        return new BenchmarkResult(
            "Softmax " + size,
            avgTimeMs,
            size / avgTimeMs / 1_000_000,
            size,
            size * Float.BYTES
        );
    }

    public static BenchmarkResult runReLUABenchmark(int size) {
        float[] x = generateRandomArray(size);

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MatrixOperations.relu(x);
        }

        long totalTime = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            MatrixOperations.relu(x);
            long end = System.nanoTime();
            totalTime += end - start;
        }

        double avgTimeMs = (totalTime / (double) MEASURE_ITERATIONS) / 1_000_000;

        return new BenchmarkResult(
            "ReLU " + size,
            avgTimeMs,
            size / avgTimeMs / 1_000_000,
            size,
            size * Float.BYTES
        );
    }

    public static BenchmarkSuiteResult runAllBenchmarks() {
        BenchmarkSuiteResult suite = new BenchmarkSuiteResult();

        suite.addResult(runMatmulBenchmark(128, 128, 128));
        suite.addResult(runMatmulBenchmark(256, 256, 256));
        suite.addResult(runMatmulBenchmark(512, 512, 512));
        suite.addResult(runMatmulBenchmark(1024, 1024, 1024));
        suite.addResult(runMatmulBenchmark(2048, 2048, 2048));

        suite.addResult(runSoftmaxBenchmark(1024));
        suite.addResult(runSoftmaxBenchmark(4096));
        suite.addResult(runSoftmaxBenchmark(16384));

        suite.addResult(runReLUABenchmark(1_000_000));
        suite.addResult(runReLUABenchmark(10_000_000));

        return suite;
    }

    private static float[] generateRandomArray(int size) {
        float[] array = new float[size];
        for (int i = 0; i < size; i++) {
            array[i] = (RANDOM.nextFloat() - 0.5f) * 2;
        }
        return array;
    }

    private static double calculateGFLOPS(int M, int K, int N, double timeMs) {
        long operations = (long) M * K * N * 2;
        return operations / timeMs / 1_000_000_000;
    }

    public record BenchmarkResult(
        String name,
        double avgTimeMs,
        double throughput,
        long operations,
        long memoryBytes
    ) {
        public String format() {
            return String.format(
                "%-30s | Time: %.4f ms | Throughput: %.4f | Ops: %,d | Memory: %,d bytes",
                name, avgTimeMs, throughput, operations, memoryBytes
            );
        }
    }

    public static class BenchmarkSuiteResult {
        private final java.util.List<BenchmarkResult> results = new java.util.ArrayList<>();
        private long startTime;
        private long endTime;

        public void start() {
            startTime = System.currentTimeMillis();
        }

        public void end() {
            endTime = System.currentTimeMillis();
        }

        public void addResult(BenchmarkResult result) {
            results.add(result);
        }

        public java.util.List<BenchmarkResult> getResults() {
            return results;
        }

        public long getTotalTimeMs() {
            return endTime - startTime;
        }

        public void printReport() {
            System.out.println("=".repeat(120));
            System.out.println("SimpCore Java High Performance Benchmark Report");
            System.out.println("=".repeat(120));
            System.out.println();
            System.out.println("Benchmark Results:");
            System.out.println("-".repeat(120));
            System.out.printf("%-30s | %-15s | %-15s | %-15s | %-15s%n",
                "Benchmark", "Time (ms)", "Throughput", "Operations", "Memory (bytes)");
            System.out.println("-".repeat(120));
            for (BenchmarkResult result : results) {
                System.out.println(result.format());
            }
            System.out.println("-".repeat(120));
            System.out.printf("Total Benchmark Time: %d ms%n", getTotalTimeMs());
            System.out.println();

            MemoryManager.MemoryStats memStats = MemoryManager.getInstance().getStats();
            System.out.println("Memory Statistics:");
            System.out.println("-".repeat(120));
            System.out.printf("Allocated: %,d bytes (%.2f MB)%n",
                memStats.allocatedBytes(), memStats.allocatedBytes() / 1_048_576.0);
            System.out.printf("Peak: %,d bytes (%.2f MB)%n",
                memStats.peakBytes(), memStats.peakBytes() / 1_048_576.0);
            System.out.printf("JVM Total: %,d bytes (%.2f MB)%n",
                memStats.totalMemory(), memStats.totalMemory() / 1_048_576.0);
            System.out.printf("JVM Free: %,d bytes (%.2f MB)%n",
                memStats.freeMemory(), memStats.freeMemory() / 1_048_576.0);
            System.out.println("-".repeat(120));
        }
    }

    public static void main(String[] args) {
        BenchmarkSuiteResult suite = runAllBenchmarks();
        suite.printReport();
    }
}