package com.qingxiaotuan.simpcore.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class SimpCoreModelIntegrationTest {

    private SimpCoreModel model;
    private Tokenizer tokenizer;

    @BeforeEach
    void setUp() {
        tokenizer = new Tokenizer(1000);
        model = new SimpCoreModel(1000, 64, 2, 32, 4, 8, 4, 2, 128);
    }

    @Test
    void testForwardPass() {
        String text = "Hello world";
        int[] inputIds = tokenizer.encode(text);
        
        float[] logits = model.forward(inputIds, 1, inputIds.length);
        
        assertNotNull(logits);
        assertEquals(inputIds.length * tokenizer.getVocabSize(), logits.length);
        
        boolean hasNonZero = Arrays.stream(logits).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testGenerate() {
        String prompt = "Hello";
        int[] inputIds = tokenizer.encode(prompt);
        
        int[] outputIds = model.generate(inputIds, 20, tokenizer.getEosTokenId());
        
        assertNotNull(outputIds);
        assertTrue(outputIds.length >= inputIds.length);
        
        String generatedText = tokenizer.decode(outputIds);
        assertNotNull(generatedText);
        assertTrue(generatedText.length() > 0);
    }

    @Test
    void testSaveLoad() throws Exception {
        String testFile = "test_model.bin";
        
        model.save(testFile);
        
        File file = new File(testFile);
        assertTrue(file.exists());
        assertTrue(file.length() > 0);
        
        SimpCoreModel loadedModel = SimpCoreModel.load(testFile);
        
        assertEquals(model.getVocabSize(), loadedModel.getVocabSize());
        assertEquals(model.getDModel(), loadedModel.getDModel());
        assertEquals(model.getNLayers(), loadedModel.getNLayers());
        assertEquals(model.getMaxSeqLen(), loadedModel.getMaxSeqLen());
        
        String text = "Test save and load";
        int[] inputIds = tokenizer.encode(text);
        
        float[] originalLogits = model.forward(inputIds, 1, inputIds.length);
        float[] loadedLogits = loadedModel.forward(inputIds, 1, inputIds.length);
        
        assertArrayEquals(originalLogits, loadedLogits, 1e-5f);
        
        file.delete();
    }

    @Test
    void testMultipleForwardPasses() {
        String text = "Repeat this test";
        int[] inputIds = tokenizer.encode(text);
        
        float[] logits1 = model.forward(inputIds, 1, inputIds.length);
        float[] logits2 = model.forward(inputIds, 1, inputIds.length);
        
        assertArrayEquals(logits1, logits2, 1e-5f);
    }

    @Test
    void testBatchForward() {
        int batchSize = 2;
        int seqLen = 8;
        int[] inputIds = new int[batchSize * seqLen];
        for (int i = 0; i < batchSize * seqLen; i++) {
            inputIds[i] = i % tokenizer.getVocabSize();
        }
        
        float[] logits = model.forward(inputIds, batchSize, seqLen);
        
        assertNotNull(logits);
        assertEquals(batchSize * seqLen * tokenizer.getVocabSize(), logits.length);
    }

    @Test
    void testGenerateWithEos() {
        String prompt = "Start";
        int[] inputIds = tokenizer.encode(prompt);
        
        int[] outputIds = model.generate(inputIds, 50, tokenizer.getEosTokenId());
        
        assertNotNull(outputIds);
        assertTrue(outputIds.length <= inputIds.length + 50);
    }

    @Test
    void testTokenizerEncodeDecode() {
        String text = "Hello World! This is a test.";
        int[] tokens = tokenizer.encode(text);
        String decoded = tokenizer.decode(tokens);
        
        assertNotNull(tokens);
        assertTrue(tokens.length > 0);
        assertNotNull(decoded);
        assertTrue(decoded.contains("Hello"));
    }
}