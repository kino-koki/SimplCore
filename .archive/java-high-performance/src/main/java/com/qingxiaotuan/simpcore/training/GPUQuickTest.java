package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;

public class GPUQuickTest {
    public static void main(String[] args) {
        System.out.println("Testing GPU availability...");
        try {
            boolean avail = GPUMatrixOperations.isGpuAvailable();
            System.out.println("GPU Available: " + avail);
        } catch (Exception e) {
            System.out.println("GPU init error: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println("GPU quick test done.");
    }
}
