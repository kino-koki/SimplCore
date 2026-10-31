package com.qingxiaotuan.simpcore.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class AdaptiveBasisLayerTest {

    private AdaptiveBasisLayer layer;
    private float[] testInput;

    @BeforeEach
    void setUp() {
        layer = new AdaptiveBasisLayer(128, 64, 4);
        testInput = new float[2 * 16 * 128];
        for (int i = 0; i < testInput.length; i++) {
            testInput[i] = (float) (Math.random() - 0.5) * 2;
        }
    }

    @Test
    void testForwardProducesOutput() {
        float[] output = layer.forward(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testForwardOptimizedProducesOutput() {
        float[] output = layer.forwardOptimized(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testOutputHasNonZeroValues() {
        float[] output = layer.forward(testInput, 2, 16);
        boolean hasNonZero = Arrays.stream(output).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testDModelAccessors() {
        assertEquals(128, layer.getDModel());
        assertEquals(64, layer.getNBases());
        assertEquals(4, layer.getNHeads());
    }

    @Test
    void testZeroInput() {
        float[] zeroInput = new float[2 * 16 * 128];
        float[] output = layer.forward(zeroInput, 2, 16);
        assertNotNull(output);
        assertEquals(zeroInput.length, output.length);
    }

    @Test
    void testDifferentBatchSizes() {
        for (int batch : new int[]{1, 2, 4}) {
            float[] input = new float[batch * 16 * 128];
            float[] output = layer.forward(input, batch, 16);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testDifferentSeqLens() {
        for (int seq : new int[]{8, 16, 32}) {
            float[] input = new float[2 * seq * 128];
            float[] output = layer.forward(input, 2, seq);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }
}