package com.qingxiaotuan.simpcore.gpu;

public class GPUValidation {

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("SimplCore GPU Validation Tool");
        System.out.println("========================================\n");

        System.out.println("[Step 1] Checking GPU availability...");
        boolean available = GPUMatrixOperations.isGpuAvailable();
        System.out.println("GPU Available: " + available);

        if (!available) {
            System.out.println("\n[ERROR] GPU not available!");
            System.out.println("Please check:");
            System.out.println("1. AMD Adrenalin driver is installed");
            System.out.println("2. OpenCL runtime is properly configured");
            System.out.println("3. JOCL native library is correctly loaded");
            return;
        }

        System.out.println("\n[Step 2] GPU Device Information");
        System.out.println("=================================");
        System.out.println(GPUMatrixOperations.getDeviceInfo());

        System.out.println("\n[Step 3] GPU Statistics");
        System.out.println("=======================");
        GPUMatrixOperations.GPUStats stats = GPUMatrixOperations.getGPUStats();
        System.out.println("Device Count: " + stats.getDeviceCount());
        String[] names = stats.getDeviceNames();
        for (int i = 0; i < names.length; i++) {
            System.out.println("  Device " + i + ": " + names[i]);
            System.out.println("    Memory: " + stats.getMemoryTotal()[i] / (1024 * 1024) + " MB");
            System.out.println("    Compute Units: " + stats.getComputeCapability()[i]);
        }

        System.out.println("\n[Step 4] Performing Quick Benchmark...");
        System.out.println("======================================");
        
        int[] sizes = {64, 128, 256, 512};
        
        for (int size : sizes) {
            System.out.println("\n--- Matrix Size: " + size + "x" + size + " ---");
            
            float[] A = new float[size * size];
            float[] B = new float[size * size];
            for (int i = 0; i < size * size; i++) {
                A[i] = (float) Math.random();
                B[i] = (float) Math.random();
            }

            long gpuStart = System.nanoTime();
            float[] gpuResult = GPUMatrixOperations.matmul(A, B, size, size, size);
            long gpuEnd = System.nanoTime();
            long gpuTime = gpuEnd - gpuStart;

            long cpuStart = System.nanoTime();
            float[] cpuResult = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, size, size, size);
            long cpuEnd = System.nanoTime();
            long cpuTime = cpuEnd - cpuStart;

            boolean match = true;
            for (int i = 0; i < size * size; i++) {
                if (Math.abs(gpuResult[i] - cpuResult[i]) > 0.001f) {
                    match = false;
                    break;
                }
            }

            System.out.println("GPU Time: " + String.format("%.2f", gpuTime / 1_000_000.0) + " ms");
            System.out.println("CPU Time: " + String.format("%.2f", cpuTime / 1_000_000.0) + " ms");
            System.out.println("Speedup: " + String.format("%.2fx", cpuTime / (double) gpuTime));
            System.out.println("Accuracy: " + (match ? "PASS" : "FAIL"));
        }

        System.out.println("\n[Step 5] Neural Network Operators Test");
        System.out.println("======================================");
        
        int nnSize = 4096;
        float[] logits = new float[nnSize];
        for (int i = 0; i < nnSize; i++) {
            logits[i] = (float) (Math.random() * 10 - 5);
        }

        // Softmax
        long smGpuStart = System.nanoTime();
        float[] smGpu = GPUMatrixOperations.softmax(logits.clone());
        long smGpuEnd = System.nanoTime();

        long smCpuStart = System.nanoTime();
        float[] smCpu = com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(logits.clone());
        long smCpuEnd = System.nanoTime();

        System.out.println("\nSoftmax (" + nnSize + " elements):");
        System.out.println("  GPU: " + String.format("%.2f", (smGpuEnd - smGpuStart) / 1_000_000.0) + " ms");
        System.out.println("  CPU: " + String.format("%.2f", (smCpuEnd - smCpuStart) / 1_000_000.0) + " ms");
        System.out.println("  Speedup: " + String.format("%.2fx", (smCpuEnd - smCpuStart) / (double) (smGpuEnd - smGpuStart)));

        // ReLU
        float[] reluInput = new float[nnSize];
        for (int i = 0; i < nnSize; i++) {
            reluInput[i] = (float) (Math.random() * 20 - 10);
        }

        long reluGpuStart = System.nanoTime();
        float[] reluGpu = GPUMatrixOperations.relu(reluInput);
        long reluGpuEnd = System.nanoTime();

        long reluCpuStart = System.nanoTime();
        float[] reluCpu = com.qingxiaotuan.simpcore.math.MatrixOperations.relu(reluInput);
        long reluCpuEnd = System.nanoTime();

        System.out.println("\nReLU (" + nnSize + " elements):");
        System.out.println("  GPU: " + String.format("%.2f", (reluGpuEnd - reluGpuStart) / 1_000_000.0) + " ms");
        System.out.println("  CPU: " + String.format("%.2f", (reluCpuEnd - reluCpuStart) / 1_000_000.0) + " ms");
        System.out.println("  Speedup: " + String.format("%.2fx", (reluCpuEnd - reluCpuStart) / (double) (reluGpuEnd - reluGpuStart)));

        System.out.println("\n========================================");
        System.out.println("Validation Complete!");
        System.out.println("========================================");
    }
}