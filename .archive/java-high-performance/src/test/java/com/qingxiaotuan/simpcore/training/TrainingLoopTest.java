package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TrainingLoopTest {

    private TrainingLoop trainer;
    private SimpCoreModel model;
    private Tokenizer tokenizer;

    @BeforeEach
    public void setUp() {
        int vocabSize = 32;
        int dModel = 16;
        int nLayers = 1;
        
        tokenizer = new Tokenizer(vocabSize);
        model = new SimpCoreModel(vocabSize, dModel, nLayers, 4, 2, 2, 2, 1, 32);
        Optimizer optimizer = new Optimizer(0.001f, 1.0f);
        trainer = new TrainingLoop(model, tokenizer, optimizer);
    }

    @Test
    public void testTrainStep() {
        String text = "abcabcabc";
        float loss = trainer.trainStep(text, 1);
        
        assertFalse(Float.isNaN(loss));
        assertFalse(Float.isInfinite(loss));
        assertTrue(loss >= 0);
    }

    @Test
    public void testTrainEpoch() {
        String[] texts = {"abcabc", "defdef", "ghighi"};
        
        assertDoesNotThrow(() -> trainer.trainEpoch(texts, 10));
    }

    @Test
    public void testGenerate() {
        String prompt = "a";
        String generated = trainer.generate(prompt, 10);
        
        assertNotNull(generated);
        assertTrue(generated.length() > 0);
    }

    @Test
    public void testLossDecrease() {
        String text = "abcabcabc";
        float initialLoss = trainer.trainStep(text, 1);
        
        for (int i = 0; i < 50; i++) {
            trainer.trainStep(text, 1);
        }
        
        float finalLoss = trainer.trainStep(text, 1);
        
        assertTrue(finalLoss < initialLoss, "Loss should decrease after training");
    }

    @Test
    public void testGetModel() {
        assertNotNull(trainer.getModel());
        assertEquals(model, trainer.getModel());
    }

    @Test
    public void testGetTokenizer() {
        assertNotNull(trainer.getTokenizer());
        assertEquals(tokenizer, trainer.getTokenizer());
    }

    @Test
    public void testEmptyText() {
        float loss = trainer.trainStep("", 1);
        
        assertEquals(Float.MAX_VALUE, loss);
    }

    @Test
    public void testShortText() {
        float loss = trainer.trainStep("a", 1);
        
        assertEquals(Float.MAX_VALUE, loss);
    }
}