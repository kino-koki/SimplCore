package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.nn.LinearLayer;

public class SCSimpleTest {
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Simple Linear Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        int inFeatures = 4;
        int outFeatures = 3;
        
        System.out.printf("[1/4] Creating LinearLayer (in=%d, out=%d)...%n", inFeatures, outFeatures);
        LinearLayer layer = new LinearLayer(inFeatures, outFeatures);
        
        float[] input = {1, 2, 3, 4};
        System.out.printf("  Input: %s%n", java.util.Arrays.toString(input));
        
        System.out.println("\n[2/4] Forward pass...");
        float[] output = layer.forward(input);
        System.out.printf("  Output: %s%n", java.util.Arrays.toString(output));
        
        System.out.println("\n[3/4] Backward pass...");
        float[] gradOutput = {0.1f, 0.2f, 0.3f};
        float[] gradInput = layer.backward(gradOutput);
        System.out.printf("  Gradient input: %s%n", java.util.Arrays.toString(gradInput));
        
        System.out.println("\n[4/4] Parameter update...");
        layer.update(0.1f, 0);
        System.out.println("  Parameters updated successfully");
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Simple Test Complete!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}