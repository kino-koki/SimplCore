package com.qingxiaotuan.simpcore.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimpCoreModelTest {

    private SimpCoreModel model;
    private Tokenizer tokenizer;

    @BeforeEach
    public void setUp() {
        int vocabSize = 32;
        int dModel = 16;
        int nLayers = 1;
        int nBases = 4;
        int nHeads = 2;
        int nChannels = 2;
        int nPaths = 2;
        int topK = 1;
        int maxSeqLen = 32;
        
        tokenizer = new Tokenizer(vocabSize);
        model = new SimpCoreModel(vocabSize, dModel, nLayers, nBases, nHeads, nChannels, nPaths, topK, maxSeqLen);
    }

    @Test
    public void testForwardShape() {
        String text = "abcabc";
        int[] inputIds = tokenizer.encode(text);
        float[] logits = model.forward(inputIds, 1, inputIds.length);
        
        assertNotNull(logits);
        assertEquals(inputIds.length * model.getVocabSize(), logits.length);
    }

    @Test
    public void testGenerate() {
        String prompt = "a";
        int[] inputIds = tokenizer.encode(prompt);
        int[] outputIds = model.generate(inputIds, 10, tokenizer.getEosTokenId());
        
        assertNotNull(outputIds);
        assertTrue(outputIds.length >= inputIds.length);
    }

    @Test
    public void testSaveLoad() {
        model.save("test_model.bin");
        SimpCoreModel loaded = SimpCoreModel.load("test_model.bin");
        
        assertNotNull(loaded);
        assertEquals(model.getVocabSize(), loaded.getVocabSize());
        assertEquals(model.getDModel(), loaded.getDModel());
        assertEquals(model.getNLayers(), loaded.getNLayers());
    }

    @Test
    public void testEmbeddingLookup() {
        int[] inputIds = {1, 2, 3};
        float[] embedding = model.getEmbeddingWeights();
        
        assertNotNull(embedding);
        assertEquals(model.getVocabSize() * model.getDModel(), embedding.length);
    }

    @Test
    public void testPositionalEncoding() {
        float[] posEncoding = model.getPosEncoding();
        
        assertNotNull(posEncoding);
        assertEquals(model.getMaxSeqLen() * model.getDModel(), posEncoding.length);
    }

    @Test
    public void testLayerNormWeights() {
        float[][] layerNormWeight = model.getLayerNormWeight();
        float[][] layerNormBias = model.getLayerNormBias();
        
        assertNotNull(layerNormWeight);
        assertNotNull(layerNormBias);
        assertEquals(model.getNLayers(), layerNormWeight.length);
        assertEquals(model.getNLayers(), layerNormBias.length);
    }

    @Test
    public void testFFNWeights() {
        float[][] ffnWeights1 = model.getFfnWeights1();
        float[][] ffnWeights2 = model.getFfnWeights2();
        
        assertNotNull(ffnWeights1);
        assertNotNull(ffnWeights2);
        assertEquals(model.getNLayers(), ffnWeights1.length);
        assertEquals(model.getNLayers(), ffnWeights2.length);
    }

    @Test
    public void testOutputWeights() {
        float[] outputWeights = model.getOutputWeights();
        
        assertNotNull(outputWeights);
        assertEquals(model.getDModel() * model.getVocabSize(), outputWeights.length);
    }

    @Test
    public void testArchitectureComponents() {
        assertNotNull(model.getRfpLayers());
        assertNotNull(model.getDmcLayers());
        assertNotNull(model.getTarLayers());
        assertNotNull(model.getCinLayers());
        assertNotNull(model.getIbLayers());
        assertNotNull(model.getMrdLayers());
        assertNotNull(model.getSpmLayers());
        
        assertEquals(model.getNLayers(), model.getRfpLayers().length);
        assertEquals(model.getNLayers(), model.getDmcLayers().length);
        assertEquals(model.getNLayers(), model.getTarLayers().length);
        assertEquals(model.getNLayers(), model.getCinLayers().length);
        assertEquals(model.getNLayers(), model.getIbLayers().length);
        assertEquals(model.getNLayers(), model.getMrdLayers().length);
        assertEquals(model.getNLayers(), model.getSpmLayers().length);
    }
}