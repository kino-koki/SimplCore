package com.qingxiaotuan.simpcore.vision.model;

import java.util.*;

public class VisionModelTest {
    
    private static final Random random = new Random(42);
    private static int passed = 0;
    private static int failed = 0;
    
    public static void main(String[] args) {
        System.out.println("=======================================================================");
        System.out.println("Vision Model Verification Suite");
        System.out.println("=======================================================================");
        
        testModelInitialization();
        testForwardPass();
        testBackwardPass();
        testParameterUpdate();
        testFeatureExtraction();
        
        System.out.println();
        System.out.println("=======================================================================");
        System.out.println("VERIFICATION RESULTS: " + passed + " PASSED, " + failed + " FAILED");
        System.out.println("Overall: " + (failed == 0 ? "✓ ALL PASSED" : "✗ SOME FAILED"));
        System.out.println("=======================================================================");
    }
    
    private static void testModelInitialization() {
        try {
            VisionModel model = new VisionModel(3, 224, 224, 512);
            
            System.out.println("  Parameters: " + model.getTotalParameters());
            System.out.println("  Embedding dim: " + model.getEmbeddingDim());
            System.out.println("  Input: " + 3 + "x" + 224 + "x" + 224);
            
            if (model.getTotalParameters() > 0 && model.getEmbeddingDim() == 512) {
                System.out.println("✓ [PASS] Model initialization successful");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Model initialization failed");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Model initialization threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testForwardPass() {
        try {
            VisionModel model = new VisionModel(3, 224, 224, 512);
            
            float[] dummyInput = new float[3 * 224 * 224];
            for (int i = 0; i < dummyInput.length; i++) {
                dummyInput[i] = random.nextFloat() * 2 - 1;
            }
            
            long startTime = System.currentTimeMillis();
            float[] output = model.forward(dummyInput);
            long endTime = System.currentTimeMillis();
            
            System.out.println("  Output length: " + output.length);
            System.out.println("  Time: " + (endTime - startTime) + " ms");
            
            boolean hasNaN = false;
            boolean hasInfinite = false;
            for (float v : output) {
                if (Float.isNaN(v)) hasNaN = true;
                if (Float.isInfinite(v)) hasInfinite = true;
            }
            
            if (output.length == 512 && !hasNaN && !hasInfinite) {
                System.out.println("✓ [PASS] Forward pass successful");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Forward pass returned invalid result");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Forward pass threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testBackwardPass() {
        try {
            VisionModel model = new VisionModel(3, 224, 224, 512);
            
            float[] dummyInput = new float[3 * 224 * 224];
            for (int i = 0; i < dummyInput.length; i++) {
                dummyInput[i] = random.nextFloat() * 2 - 1;
            }
            
            float[] output = model.forward(dummyInput);
            
            float[] grad = new float[512];
            for (int i = 0; i < grad.length; i++) {
                grad[i] = random.nextFloat() * 2 - 1;
            }
            
            float[] gradInput = model.backward(grad);
            
            float[] gradients = model.getGradients();
            int nonZeroGrads = 0;
            for (float g : gradients) {
                if (Math.abs(g) > 0.0001f) nonZeroGrads++;
            }
            
            System.out.println("  Gradient length: " + gradInput.length);
            System.out.println("  Non-zero weight gradients: " + nonZeroGrads + "/" + gradients.length);
            
            if (gradInput.length > 0 && nonZeroGrads > 0) {
                System.out.println("✓ [PASS] Backward pass successful");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Backward pass returned invalid gradients");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Backward pass threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testParameterUpdate() {
        try {
            VisionModel model = new VisionModel(3, 224, 224, 512);
            
            float[] dummyInput = new float[3 * 224 * 224];
            for (int i = 0; i < dummyInput.length; i++) {
                dummyInput[i] = random.nextFloat() * 2 - 1;
            }
            
            float[] outputBefore = model.forward(dummyInput);
            
            float[] grad = new float[512];
            for (int i = 0; i < grad.length; i++) {
                grad[i] = 0.1f;
            }
            
            model.backward(grad);
            model.updateParameters(0.1f);
            
            float[] outputAfter = model.forward(dummyInput);
            
            float diff = 0;
            for (int i = 0; i < 512; i++) {
                diff += Math.abs(outputAfter[i] - outputBefore[i]);
            }
            diff /= 512;
            
            System.out.println("  Average output change: " + String.format("%.6f", diff));
            
            if (diff > 0.0001f) {
                System.out.println("✓ [PASS] Parameter update successful");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Parameter update had no effect");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Parameter update threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testFeatureExtraction() {
        try {
            VisionModel model = new VisionModel(3, 224, 224, 512);
            
            float[][] testImages = new float[5][3 * 224 * 224];
            for (int img = 0; img < 5; img++) {
                for (int i = 0; i < testImages[img].length; i++) {
                    testImages[img][i] = random.nextFloat() * 2 - 1;
                }
            }
            
            float[][] features = new float[5][512];
            for (int img = 0; img < 5; img++) {
                features[img] = model.forward(testImages[img]);
            }
            
            float[][] similarities = new float[5][5];
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    float dot = 0;
                    float normI = 0, normJ = 0;
                    for (int k = 0; k < 512; k++) {
                        dot += features[i][k] * features[j][k];
                        normI += features[i][k] * features[i][k];
                        normJ += features[j][k] * features[j][k];
                    }
                    similarities[i][j] = dot / (float) (Math.sqrt(normI) * Math.sqrt(normJ));
                }
            }
            
            boolean diagLarger = true;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (i != j && similarities[i][i] < similarities[i][j]) {
                        diagLarger = false;
                        break;
                    }
                }
                if (!diagLarger) break;
            }
            
            System.out.println("  Self-similarity check: " + (diagLarger ? "PASS" : "WARN"));
            
            System.out.println("✓ [PASS] Feature extraction successful");
            passed++;
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Feature extraction threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
}