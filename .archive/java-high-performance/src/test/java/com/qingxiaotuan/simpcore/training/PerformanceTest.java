package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import org.junit.Test;

public class PerformanceTest {

    @Test
    public void testMatmulPerformance() {
        System.out.println("Testing Matmul Performance...");
        
        int[] sizes = {64, 128, 256, 512};
        
        for (int size : sizes) {
            float[] A = new float[size * size];
            float[] B = new float[size * size];
            
            for (int i = 0; i < size * size; i++) {
                A[i] = (float) Math.random();
                B[i] = (float) Math.random();
            }
            
            long start = System.nanoTime();
            float[] C = MatrixOperations.matmul(A, B, size, size, size);
            long javaTime = System.nanoTime() - start;
            
            System.out.printf("Java Matmul %dx%d: %.2f ms%n", size, size, javaTime / 1_000_000.0);
        }
        
        System.out.println("Matmul performance test completed!");
    }

    @Test
    public void testVectorOperationsPerformance() {
        System.out.println("Testing Vector Operations Performance...");
        
        int[] sizes = {1000, 10000, 100000, 1000000};
        
        for (int size : sizes) {
            float[] A = new float[size];
            float[] B = new float[size];
            
            for (int i = 0; i < size; i++) {
                A[i] = (float) Math.random();
                B[i] = (float) Math.random();
            }
            
            long start = System.nanoTime();
            float[] C = MatrixOperations.add(A, B);
            long addTime = System.nanoTime() - start;
            
            start = System.nanoTime();
            float[] D = MatrixOperations.sub(A, B);
            long subTime = System.nanoTime() - start;
            
            start = System.nanoTime();
            float[] E = MatrixOperations.mul(A, 0.5f);
            long mulTime = System.nanoTime() - start;
            
            start = System.nanoTime();
            float[] F = MatrixOperations.relu(A);
            long reluTime = System.nanoTime() - start;
            
            System.out.printf("Vector Size %,d:%n", size);
            System.out.printf("  Add: %.2f ms%n", addTime / 1_000_000.0);
            System.out.printf("  Sub: %.2f ms%n", subTime / 1_000_000.0);
            System.out.printf("  Mul: %.2f ms%n", mulTime / 1_000_000.0);
            System.out.printf("  ReLU: %.2f ms%n", reluTime / 1_000_000.0);
        }
        
        System.out.println("Vector operations performance test completed!");
    }

    @Test
    public void testGPUMatmulPerformance() {
        System.out.println("Testing GPU Matmul Performance...");
        
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping test");
            return;
        }
        
        System.out.println(GPUMatrixOperations.getDeviceInfo());
        
        int[] sizes = {64, 128, 256};
        
        for (int size : sizes) {
            float[] A = new float[size * size];
            float[] B = new float[size * size];
            
            for (int i = 0; i < size * size; i++) {
                A[i] = (float) Math.random();
                B[i] = (float) Math.random();
            }
            
            float[] warmup = GPUMatrixOperations.matmul(A, B, size, size, size);
            
            long start = System.nanoTime();
            float[] C = GPUMatrixOperations.matmul(A, B, size, size, size);
            long gpuTime = System.nanoTime() - start;
            
            System.out.printf("GPU Matmul %dx%d: %.2f ms%n", size, size, gpuTime / 1_000_000.0);
        }
        
        System.out.println(GPUMatrixOperations.getPerformanceStats());
        
        System.out.println("GPU matmul performance test completed!");
    }

    @Test
    public void testTrainingPerformance() {
        System.out.println("Testing Training Performance...");
        
        int iterations = 10;
        
        com.qingxiaotuan.simpcore.model.Tokenizer tokenizer = new com.qingxiaotuan.simpcore.model.Tokenizer(1000);
        com.qingxiaotuan.simpcore.model.SimpCoreModel model = new com.qingxiaotuan.simpcore.model.SimpCoreModel(64, 2, 64, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);
        
        String text = "machine learning is a powerful technology that enables computers to learn from data and make predictions";
        
        long start = System.nanoTime();
        float totalLoss = 0;
        
        for (int i = 0; i < iterations; i++) {
            float loss = trainingLoop.trainStep(text, 1);
            totalLoss += loss;
        }
        
        long totalTime = System.nanoTime() - start;
        
        System.out.printf("Training Performance:%n");
        System.out.printf("  Iterations: %d%n", iterations);
        System.out.printf("  Total Time: %.2f ms%n", totalTime / 1_000_000.0);
        System.out.printf("  Average Time per Iteration: %.2f ms%n", totalTime / 1_000_000.0 / iterations);
        System.out.printf("  Average Loss: %.4f%n", totalLoss / iterations);
        
        System.out.println("Training performance test completed!");
    }

    @Test
    public void testMemoryOptimizer() {
        System.out.println("Testing Memory Optimizer...");
        
        com.qingxiaotuan.simpcore.memory.MemoryOptimizer memoryOptimizer = com.qingxiaotuan.simpcore.memory.MemoryOptimizer.getInstance();
        
        int[] sizes = {1024, 4096, 16384, 65536};
        
        for (int size : sizes) {
            float[] arr1 = memoryOptimizer.allocateFloatArray(size);
            float[] arr2 = memoryOptimizer.allocateFloatArray(size);
            
            memoryOptimizer.releaseFloatArray(arr1);
            memoryOptimizer.releaseFloatArray(arr2);
        }
        
        System.out.println("Memory Optimizer test completed!");
    }
}