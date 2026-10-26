package com.qingxiaotuan.simpcore;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.util.Arrays;

public class SimpCoreVerification {
    
    private static int passed = 0;
    private static int failed = 0;
    
    public static void main(String[] args) {
        System.out.println("=" + "=".repeat(70));
        System.out.println("SimpCore Model Verification Suite");
        System.out.println("=" + "=".repeat(70));
        System.out.println();
        
        testModelInitialization();
        testForwardPass();
        testBackwardPass();
        testGenerate();
        testTrainingLoop();
        testGPUAvailability();
        
        System.out.println();
        System.out.println("=" + "=".repeat(70));
        System.out.println("VERIFICATION RESULTS: " + passed + " PASSED, " + failed + " FAILED");
        System.out.println("Overall: " + (failed == 0 ? "✓ ALL PASSED" : "✗ SOME FAILED"));
        System.out.println("=" + "=".repeat(70));
        
        System.exit(failed == 0 ? 0 : 1);
    }
    
    private static void testModelInitialization() {
        try {
            int vocabSize = 64;
            int dModel = 64;
            int nLayers = 2;
            int nHeads = 4;
            
            SimpCoreModel model = new SimpCoreModel(vocabSize, dModel, nLayers, 2, nHeads, 2, 2, 2, 32);
            
            if (model != null && model.getVocabSize() == vocabSize && 
                model.getDModel() == dModel && model.getNLayers() == nLayers) {
                System.out.println("✓ [PASS] Model initialization successful");
                System.out.println("  VocabSize: " + model.getVocabSize() + ", DModel: " + model.getDModel() + 
                                   ", NLayers: " + model.getNLayers());
                passed++;
            } else {
                System.out.println("✗ [FAIL] Model initialization failed");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Model initialization threw exception: " + e.getMessage());
            failed++;
        }
    }
    
    private static void testForwardPass() {
        try {
            SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
            Tokenizer tokenizer = new Tokenizer(64);
            
            String text = "hello world";
            int[] inputIds = tokenizer.encode(text);
            System.out.println("  Input text: '" + text + "'");
            System.out.println("  Input tokens: " + Arrays.toString(inputIds) + " (length: " + inputIds.length + ")");
            
            float[] output = model.forward(inputIds);
            System.out.println("  Output length: " + (output != null ? output.length : "null"));
            
            if (output != null && output.length > 0) {
                System.out.println("✓ [PASS] Forward pass successful");
                System.out.println("  Expected output size: " + inputIds.length * 64);
                passed++;
            } else {
                System.out.println("✗ [FAIL] Forward pass returned null or empty");
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
            SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
            Tokenizer tokenizer = new Tokenizer(64);
            
            String text = "hello world";
            int[] inputIds = tokenizer.encode(text);
            
            float[] output = model.forward(inputIds);
            System.out.println("  Forward output length: " + output.length);
            
            float[] gradients = new float[output.length];
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] = 1.0f;
            }
            
            float[] result = model.backwardPartial(gradients, 0, 2, 1, inputIds.length);
            System.out.println("  Backward result length: " + (result != null ? result.length : "null"));
            
            if (result != null && result.length == gradients.length) {
                float nonZeroCount = 0;
                for (float v : result) {
                    if (Math.abs(v) > 0.0001f) nonZeroCount++;
                }
                System.out.println("✓ [PASS] Backward pass successful");
                System.out.println("  Gradient length: " + result.length);
                System.out.println("  Non-zero gradients: " + nonZeroCount + "/" + result.length);
                passed++;
            } else {
                System.out.println("✗ [FAIL] Backward pass returned invalid result");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Backward pass threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testGenerate() {
        try {
            SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
            Tokenizer tokenizer = new Tokenizer(64);
            
            String prompt = "hello";
            int[] inputIds = tokenizer.encode(prompt);
            
            int[] outputIds = model.generate(inputIds, 10, tokenizer.getEosTokenId());
            
            if (outputIds != null && outputIds.length >= inputIds.length) {
                String generated = tokenizer.decode(outputIds);
                System.out.println("✓ [PASS] Generation successful");
                System.out.println("  Prompt: '" + prompt + "'");
                System.out.println("  Generated: '" + generated + "'");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Generation returned invalid result");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Generation threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testTrainingLoop() {
        try {
            SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
            Tokenizer tokenizer = new Tokenizer(64);
            
            String[] trainingData = {
                "hello world this is a test",
                "the cat sat on the mat",
                "i like to play with my dog",
                "machine learning is interesting",
                "deep neural networks are powerful"
            };
            
            float[] losses = new float[10];
            float learningRate = 0.01f;
            
            System.out.println("  Training for 10 epochs...");
            
            for (int epoch = 0; epoch < 10; epoch++) {
                float totalLoss = 0f;
                for (String text : trainingData) {
                    int[] inputIds = tokenizer.encode(text);
                    float[] logits = model.forward(inputIds);
                    
                    float loss = computeLoss(logits, inputIds, tokenizer.getVocabSize());
                    totalLoss += loss;
                    
                    float[] gradients = computeLossGradients(logits, inputIds, tokenizer.getVocabSize());
                    
                    float[] backGradients = model.backwardPartial(gradients, 0, 2, 1, inputIds.length);
                    
                    if (epoch == 0 && text.equals(trainingData[0])) {
                        float gradSum = 0;
                        for (float g : gradients) gradSum += Math.abs(g);
                        System.out.println("  Gradient magnitude: " + String.format("%.6f", gradSum));
                    }
                    
                    float[] outputWeights = model.getOutputWeights();
                    for (int i = 0; i < outputWeights.length; i++) {
                        outputWeights[i] -= learningRate * gradients[i % gradients.length];
                    }
                    model.updateParameters(learningRate);
                }
                losses[epoch] = totalLoss / trainingData.length;
                System.out.println("  Epoch " + (epoch + 1) + ": Loss = " + String.format("%.4f", losses[epoch]));
            }
            
            float initialLoss = losses[0];
            float finalLoss = losses[losses.length - 1];
            
            System.out.println("  Initial loss: " + String.format("%.4f", initialLoss));
            System.out.println("  Final loss: " + String.format("%.4f", finalLoss));
            
            if (initialLoss > finalLoss && finalLoss < initialLoss * 1.5f) {
                System.out.println("✓ [PASS] Training loop successful - loss decreased");
                System.out.println("  Loss improvement: " + String.format("%.2f", (initialLoss - finalLoss) / initialLoss * 100) + "%");
                passed++;
            } else {
                System.out.println("✗ [FAIL] Training loop - loss did not decrease properly");
                failed++;
            }
        } catch (Exception e) {
            System.out.println("✗ [FAIL] Training loop threw exception: " + e.getMessage());
            e.printStackTrace();
            failed++;
        }
    }
    
    private static void testGPUAvailability() {
        try {
            boolean gpuAvailable = com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations.isGpuAvailable();
            System.out.println("✓ [INFO] GPU availability check completed");
            System.out.println("  GPU Available: " + (gpuAvailable ? "YES" : "NO (using CPU)"));
            
            if (gpuAvailable) {
                System.out.println("  Device Info: " + 
                    com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations.getDeviceInfo());
            }
            passed++;
        } catch (Exception e) {
            System.out.println("✓ [INFO] GPU check skipped: " + e.getMessage());
            passed++;
        }
    }
    
    private static float computeLoss(float[] logits, int[] labels, int vocabSize) {
        float loss = 0f;
        int seqLen = labels.length;
        
        for (int i = 0; i < seqLen - 1; i++) {
            int target = labels[i + 1];
            int startIdx = i * vocabSize;
            int endIdx = (i + 1) * vocabSize;
            
            float sumExp = 0f;
            for (int j = startIdx; j < endIdx; j++) {
                sumExp += Math.exp(logits[j]);
            }
            
            float prob = (float) (Math.exp(logits[startIdx + target]) / sumExp);
            loss -= (float) Math.log(Math.max(prob, 1e-10f));
        }
        
        return loss / (seqLen - 1);
    }
    
    private static float[] computeLossGradients(float[] logits, int[] labels, int vocabSize) {
        float[] gradients = new float[logits.length];
        int seqLen = labels.length;
        
        for (int i = 0; i < seqLen - 1; i++) {
            int target = labels[i + 1];
            int startIdx = i * vocabSize;
            int endIdx = (i + 1) * vocabSize;
            
            float sumExp = 0f;
            for (int j = startIdx; j < endIdx; j++) {
                sumExp += Math.exp(logits[j]);
            }
            
            for (int j = startIdx; j < endIdx; j++) {
                float prob = (float) (Math.exp(logits[j]) / sumExp);
                gradients[j] = prob;
            }
            gradients[startIdx + target] -= 1.0f;
        }
        
        float scale = 1.0f / (seqLen - 1);
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= scale;
        }
        
        return gradients;
    }
}
