package com.qingxiaotuan.simpcore.gpu;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GPUEnvironmentTest {

    @Test
    public void testGPUAvailability() {
        boolean available = GPUMatrixOperations.isGpuAvailable();
        System.out.println("GPU Available: " + available);
        
        if (available) {
            System.out.println("Device Count: " + GPUMatrixOperations.getDeviceCount());
            System.out.println("Current Device: " + GPUMatrixOperations.getCurrentDevice());
            System.out.println("\nDevice Info:");
            System.out.println(GPUMatrixOperations.getDeviceInfo());
        }
    }

    @Test
    public void testGPUMatrixOperations() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping GPU operations test");
            return;
        }

        int M = 64, K = 64, N = 64;
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        
        for (int i = 0; i < M * K; i++) {
            A[i] = (float) Math.random();
        }
        for (int i = 0; i < K * N; i++) {
            B[i] = (float) Math.random();
        }

        float[] result = GPUMatrixOperations.matmul(A, B, M, K, N);
        assertNotNull(result);
        assertEquals(M * N, result.length);
        
        System.out.println("Matrix multiplication completed successfully: " + M + "x" + K + " * " + K + "x" + N);
    }

    @Test
    public void testGPUSoftmax() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping softmax test");
            return;
        }

        float[] logits = new float[1024];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) (Math.random() * 10 - 5);
        }

        float[] result = GPUMatrixOperations.softmax(logits);
        assertNotNull(result);
        assertEquals(logits.length, result.length);

        float sum = 0;
        for (float val : result) {
            sum += val;
        }
        assertTrue(Math.abs(sum - 1.0f) < 0.01f, "Softmax sum should be approximately 1");
        
        System.out.println("Softmax completed successfully, sum: " + sum);
    }

    @Test
    public void testGPURelu() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping ReLU test");
            return;
        }

        float[] input = new float[1024];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) (Math.random() * 20 - 10);
        }

        float[] result = GPUMatrixOperations.relu(input);
        assertNotNull(result);
        assertEquals(input.length, result.length);

        for (float val : result) {
            assertTrue(val >= 0, "ReLU output should be non-negative");
        }
        
        System.out.println("ReLU completed successfully");
    }

    @Test
    public void testGPUAdd() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping add test");
            return;
        }

        float[] A = new float[1024];
        float[] B = new float[1024];
        for (int i = 0; i < 1024; i++) {
            A[i] = (float) Math.random();
            B[i] = (float) Math.random();
        }

        float[] result = GPUMatrixOperations.add(A, B);
        assertNotNull(result);
        assertEquals(1024, result.length);
        
        System.out.println("Vector addition completed successfully");
    }

    @Test
    public void testGPUDotProduct() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping dot product test");
            return;
        }

        float[] a = new float[1024];
        float[] b = new float[1024];
        for (int i = 0; i < 1024; i++) {
            a[i] = (float) Math.random();
            b[i] = (float) Math.random();
        }

        float result = GPUMatrixOperations.dotProduct(a, b);
        System.out.println("Dot product completed successfully: " + result);
    }

    @Test
    public void testGetGPUStats() {
        GPUMatrixOperations.GPUStats stats = GPUMatrixOperations.getGPUStats();
        assertNotNull(stats);
        
        System.out.println("GPU Stats:");
        System.out.println("Device Count: " + stats.getDeviceCount());
        
        String[] names = stats.getDeviceNames();
        for (int i = 0; i < names.length; i++) {
            System.out.println("Device " + i + ": " + names[i]);
            System.out.println("  Memory: " + stats.getMemoryTotal()[i] / (1024 * 1024) + " MB");
            System.out.println("  Compute Units: " + stats.getComputeCapability()[i]);
        }
    }

    @Test
    public void testGPUMemoryOperations() {
        if (!GPUMatrixOperations.isGpuAvailable()) {
            System.out.println("GPU not available, skipping memory operations test");
            return;
        }

        long handle = GPUMatrixOperations.allocatePinnedMemory(1024);
        assertNotEquals(-1, handle, "Memory allocation should succeed");

        float[] data = new float[1024];
        for (int i = 0; i < 1024; i++) {
            data[i] = (float) i;
        }

        GPUMatrixOperations.copyToGPU(data, handle, 1024);
        
        float[] result = new float[1024];
        GPUMatrixOperations.copyFromGPU(handle, result, 1024);
        
        for (int i = 0; i < 1024; i++) {
            assertEquals(data[i], result[i], 0.001f, "Data should be correctly copied");
        }

        GPUMatrixOperations.releasePinnedMemory(handle);
        
        System.out.println("Memory operations completed successfully");
    }
}