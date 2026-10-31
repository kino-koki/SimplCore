package com.qingxiaotuan.simpcore;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimpCoreModelTest {
    
    @Test
    public void testModelInitialization() {
        int vocabSize = 64;
        int dModel = 64;
        int nLayers = 2;
        int nHeads = 4;
        
        SimpCoreModel model = new SimpCoreModel(vocabSize, dModel, nLayers, 2, nHeads, 2, 2, 2, 32);
        
        assertNotNull(model);
        assertEquals(vocabSize, model.getVocabSize());
        assertEquals(dModel, model.getDModel());
        assertEquals(nLayers, model.getNLayers());
        
        System.out.println("✓ Model initialization successful");
    }
    
    @Test
    public void testForwardPass() {
        SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        
        String text = "hello world this is a test";
        int[] inputIds = tokenizer.encode(text);
        
        float[] output = model.forward(inputIds);
        
        assertNotNull(output);
        assertTrue(output.length > 0);
        
        System.out.println("✓ Forward pass successful");
        System.out.println("  Input length: " + inputIds.length);
        System.out.println("  Output length: " + output.length);
    }
    
    @Test
    public void testBackwardPass() {
        SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        
        String text = "hello world";
        int[] inputIds = tokenizer.encode(text);
        
        float[] output = model.forward(inputIds);
        float[] gradients = new float[output.length];
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] = 1.0f;
        }
        
        float[] result = model.backwardPartial(gradients, 0, 2);
        
        assertNotNull(result);
        assertEquals(gradients.length, result.length);
        
        System.out.println("✓ Backward pass successful");
        System.out.println("  Gradient length: " + gradients.length);
    }
    
    @Test
    public void testLayerNormOptimization() {
        SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
        
        float[] input = new float[64];
        for (int i = 0; i < 64; i++) {
            input[i] = (float) Math.sin(i);
        }
        
        float[] output = model.layerNorm(input, 0);
        
        assertNotNull(output);
        assertEquals(64, output.length);
        
        float sum = 0f;
        for (float val : output) {
            sum += val;
        }
        
        assertTrue(Math.abs(sum) < 1.0f);
        
        System.out.println("✓ Layer normalization successful");
        System.out.println("  Mean close to zero: " + String.format("%.6f", sum / 64));
    }
    
    @Test
    public void testGenerate() {
        SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        
        String prompt = "hello";
        int[] inputIds = tokenizer.encode(prompt);
        
        int[] outputIds = model.generate(inputIds, 10, tokenizer.getEosTokenId());
        
        assertNotNull(outputIds);
        assertTrue(outputIds.length >= inputIds.length);
        
        String generated = tokenizer.decode(outputIds);
        
        System.out.println("✓ Generation successful");
        System.out.println("  Prompt: '" + prompt + "'");
        System.out.println("  Generated: '" + generated + "'");
    }
    
    @Test
    public void testTrainingLoop() {
        SimpCoreModel model = new SimpCoreModel(64, 64, 2, 2, 4, 2, 2, 2, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        
        String[] trainingData = {
            "hello world this is a test",
            "the cat sat on the mat",
            "i like to play with my dog"
        };
        
        float initialLoss = Float.MAX_VALUE;
        float finalLoss = Float.MAX_VALUE;
        
        for (int epoch = 0; epoch < 3; epoch++) {
            for (String text : trainingData) {
                int[] inputIds = tokenizer.encode(text);
                float[] logits = model.forward(inputIds);
                
                float loss = computeLoss(logits, inputIds, tokenizer.getVocabSize());
                
                if (epoch == 0) {
                    initialLoss = Math.min(initialLoss, loss);
                }
                finalLoss = loss;
                
                float[] gradients = new float[logits.length];
                for (int i = 0; i < gradients.length; i++) {
                    gradients[i] = 0.01f;
                }
                model.backwardPartial(gradients, 0, 2);
            }
        }
        
        assertTrue(initialLoss > 0);
        assertTrue(finalLoss > 0);
        
        System.out.println("✓ Training loop successful");
        System.out.println("  Initial loss: " + String.format("%.4f", initialLoss));
        System.out.println("  Final loss: " + String.format("%.4f", finalLoss));
    }
    
    private float computeLoss(float[] logits, int[] labels, int vocabSize) {
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
    
    public static void main(String[] args) {
        SimpCoreModelTest test = new SimpCoreModelTest();
        
        System.out.println("=" + "=".repeat(60));
        System.out.println("SimpCore Model Test Suite");
        System.out.println("=" + "=".repeat(60));
        System.out.println();
        
        try {
            test.testModelInitialization();
            test.testForwardPass();
            test.testBackwardPass();
            test.testLayerNormOptimization();
            test.testGenerate();
            test.testTrainingLoop();
            
            System.out.println();
            System.out.println("=" + "=".repeat(60));
            System.out.println("ALL TESTS PASSED ✓");
            System.out.println("=" + "=".repeat(60));
        } catch (Exception e) {
            System.out.println();
            System.out.println("=" + "=".repeat(60));
            System.out.println("TEST FAILED ✗");
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.out.println("=" + "=".repeat(60));
        }
    }
}
