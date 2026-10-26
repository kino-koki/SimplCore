package com.qingxiaotuan.simpcore.gpu;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 100, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 200, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class GPUPerformanceBenchmark {

    @Param({"64", "128", "256", "512", "1024"})
    private int matrixSize;

    @Param({"1024", "4096", "16384"})
    private int vectorSize;

    private float[] matrixA;
    private float[] matrixB;
    private float[] vector;

    private static final Random random = new Random(42);

    @Setup(Level.Trial)
    public void setup() {
        matrixA = new float[matrixSize * matrixSize];
        matrixB = new float[matrixSize * matrixSize];
        vector = new float[vectorSize];

        for (int i = 0; i < matrixSize * matrixSize; i++) {
            matrixA[i] = random.nextFloat();
            matrixB[i] = random.nextFloat();
        }

        for (int i = 0; i < vectorSize; i++) {
            vector[i] = random.nextFloat() * 20 - 10;
        }
    }

    @Benchmark
    @Group("matmul")
    @GroupThreads(1)
    public float[] matmulGPU() {
        return GPUMatrixOperations.matmul(matrixA, matrixB, matrixSize, matrixSize, matrixSize);
    }

    @Benchmark
    @Group("matmul")
    @GroupThreads(1)
    public float[] matmulCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(matrixA, matrixB, matrixSize, matrixSize, matrixSize);
    }

    @Benchmark
    @Group("softmax")
    @GroupThreads(1)
    public float[] softmaxGPU() {
        return GPUMatrixOperations.softmax(vector.clone());
    }

    @Benchmark
    @Group("softmax")
    @GroupThreads(1)
    public float[] softmaxCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(vector.clone());
    }

    @Benchmark
    @Group("relu")
    @GroupThreads(1)
    public float[] reluGPU() {
        return GPUMatrixOperations.relu(vector);
    }

    @Benchmark
    @Group("relu")
    @GroupThreads(1)
    public float[] reluCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.relu(vector);
    }

    @Benchmark
    @Group("silu")
    @GroupThreads(1)
    public float[] siluGPU() {
        return GPUMatrixOperations.silu(vector);
    }

    @Benchmark
    @Group("silu")
    @GroupThreads(1)
    public float[] siluCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(vector);
    }

    @Benchmark
    @Group("add")
    @GroupThreads(1)
    public float[] addGPU() {
        return GPUMatrixOperations.add(matrixA, matrixB);
    }

    @Benchmark
    @Group("add")
    @GroupThreads(1)
    public float[] addCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.add(matrixA, matrixB);
    }

    @Benchmark
    @Group("dotProduct")
    @GroupThreads(1)
    public float dotProductGPU() {
        return GPUMatrixOperations.dotProduct(matrixA, matrixB);
    }

    @Benchmark
    @Group("dotProduct")
    @GroupThreads(1)
    public float dotProductCPU() {
        return com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(matrixA, matrixB);
    }

    public static void main(String[] args) throws RunnerException {
        System.out.println("========================================");
        System.out.println("SimplCore GPU Performance Benchmark");
        System.out.println("========================================\n");

        boolean gpuAvailable = GPUMatrixOperations.isGpuAvailable();
        System.out.println("GPU Available: " + gpuAvailable);

        if (gpuAvailable) {
            System.out.println("Device Info:");
            System.out.println(GPUMatrixOperations.getDeviceInfo());
        } else {
            System.out.println("WARNING: GPU not available, only CPU benchmarks will run");
        }

        System.out.println("\nStarting benchmarks...\n");

        Options options = new OptionsBuilder()
                .include(GPUPerformanceBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();

        new Runner(options).run();

        System.out.println("\n========================================");
        System.out.println("Benchmark Complete!");
        System.out.println("========================================");
    }
}