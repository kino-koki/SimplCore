package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.training.Optimizer;
import com.qingxiaotuan.simpcore.training.TrainingLoop;
import org.junit.Test;

import static org.junit.Assert.*;

public class ZYLAttentionIntegrationTest {

    @Test
    public void testModelWithZYLAttentionForward() {
        System.out.println("Testing SimpCoreModel with ZYL attention forward pass...");

        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(32, 2, 4, 1000, 32);

        model.setUseZYLAttention(true);
        assertTrue("Model should use ZYL attention", model.isUseZYLAttention());
        assertNotNull("ZYL layers should be initialized", model.getZylLayers());

        int[] inputIds = tokenizer.encode("hello world this is a test");
        if (inputIds.length < 2) {
            System.out.println("Skipping test - input too short");
            return;
        }

        int seqLen = Math.min(inputIds.length, model.getMaxSeqLen());
        float[] logits = model.forward(inputIds, 1, seqLen);

        assertNotNull("Logits should not be null", logits);
        assertEquals("Logits size should match", 1 * seqLen * model.getVocabSize(), logits.length);

        for (float v : logits) {
            assertTrue("Logits should be finite", Float.isFinite(v));
        }

        System.out.println("Model with ZYL attention forward pass test passed!");
    }

    @Test
    public void testModelWithZYLAttentionTraining() {
        System.out.println("Testing SimpCoreModel with ZYL attention training...");

        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(32, 2, 4, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);

        model.setUseZYLAttention(true);

        String[] texts = {"hello world this is a test", "machine learning is fun", "deep learning is powerful"};

        float initialLoss = Float.MAX_VALUE;
        float finalLoss = Float.MAX_VALUE;

        for (int iter = 0; iter < 50; iter++) {
            String text = texts[iter % texts.length];
            float loss = trainingLoop.trainStep(text, 1);

            if (iter == 0) initialLoss = loss;
            if (iter == 49) finalLoss = loss;

            assertTrue("Loss should be finite", Float.isFinite(loss));
        }

        System.out.printf("Initial Loss: %.4f, Final Loss: %.4f%n", initialLoss, finalLoss);

        assertTrue("Loss should decrease during training", finalLoss < initialLoss);

        System.out.println("Model with ZYL attention training test passed!");
    }

    @Test
    public void testModelWithZYLAttentionGeneration() {
        System.out.println("Testing SimpCoreModel with ZYL attention generation...");

        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(32, 2, 4, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);

        model.setUseZYLAttention(true);

        String[] texts = {"hello world", "machine learning"};
        for (int iter = 0; iter < 30; iter++) {
            trainingLoop.trainStep(texts[iter % texts.length], 1);
        }

        String prompt = "hello";
        String generated = trainingLoop.generate(prompt, 10);

        System.out.printf("Prompt: %s, Generated: %s%n", prompt, generated);

        assertNotNull("Generated text should not be null", generated);
        assertTrue("Generated text should not be empty", !generated.isEmpty());

        System.out.println("Model with ZYL attention generation test passed!");
    }

    @Test
    public void testMHAvsZYLAttentionSwitching() {
        System.out.println("Testing MHA vs ZYL attention switching...");

        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(32, 2, 4, 1000, 32);

        int[] inputIds = tokenizer.encode("hello world");
        if (inputIds.length < 2) {
            System.out.println("Skipping test - input too short");
            return;
        }

        int seqLen = Math.min(inputIds.length, model.getMaxSeqLen());

        model.setUseZYLAttention(false);
        float[] mhaLogits = model.forward(inputIds, 1, seqLen);

        model.setUseZYLAttention(true);
        float[] zylLogits = model.forward(inputIds, 1, seqLen);

        assertNotNull("MHA logits should not be null", mhaLogits);
        assertNotNull("ZYL logits should not be null", zylLogits);
        assertEquals("Logits sizes should match", mhaLogits.length, zylLogits.length);

        for (int i = 0; i < mhaLogits.length; i++) {
            assertTrue("MHA logits should be finite", Float.isFinite(mhaLogits[i]));
            assertTrue("ZYL logits should be finite", Float.isFinite(zylLogits[i]));
        }

        System.out.println("MHA vs ZYL attention switching test passed!");
    }

    @Test
    public void testZYLAttentionTrainingStability() {
        System.out.println("Testing ZYL attention training stability...");

        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(32, 2, 4, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);

        model.setUseZYLAttention(true);

        String[] texts = {"the quick brown fox jumps over the lazy dog",
                          "pack my box with five dozen liquor jugs",
                          "how vexingly quick daft zebras jump",
                          "bright vixens jump; dozy fowl quack"};

        float previousLoss = Float.MAX_VALUE;
        int stableCount = 0;

        for (int iter = 0; iter < 100; iter++) {
            String text = texts[iter % texts.length];
            float loss = trainingLoop.trainStep(text, 1);

            assertTrue("Loss should be finite", Float.isFinite(loss));
            assertTrue("Loss should be non-negative", loss >= 0);

            if (Math.abs(loss - previousLoss) < 1.0f) {
                stableCount++;
            }
            previousLoss = loss;
        }

        assertTrue("Training should stabilize", stableCount > 50);

        System.out.println("ZYL attention training stability test passed!");
    }
}