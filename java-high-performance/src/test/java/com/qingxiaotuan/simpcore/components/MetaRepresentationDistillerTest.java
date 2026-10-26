package com.qingxiaotuan.simpcore.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class MetaRepresentationDistillerTest {

    private MetaRepresentationDistiller distiller;
    private float[] testInput;

    @BeforeEach
    void setUp() {
        distiller = new MetaRepresentationDistiller(128, 32);
        testInput = new float[2 * 16 * 128];
        for (int i = 0; i < testInput.length; i++) {
            testInput[i] = (float) (Math.random() - 0.5) * 2;
        }
    }

    @Test
    void testForwardProducesOutput() {
        float[] output = distiller.forward(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testOutputHasNonZeroValues() {
        float[] output = distiller.forward(testInput, 2, 16);
        boolean hasNonZero = Arrays.stream(output).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testZeroInput() {
        float[] zeroInput = new float[2 * 16 * 128];
        float[] output = distiller.forward(zeroInput, 2, 16);
        assertNotNull(output);
        assertEquals(zeroInput.length, output.length);
    }

    @Test
    void testEncodeDecodeRoundTrip() {
        float[] x = new float[128];
        for (int i = 0; i < 128; i++) {
            x[i] = (float) (Math.random() - 0.5);
        }
        float[] latent = distiller.getLatentRepresentation(x);
        float[] reconstructed = distiller.reconstruct(latent);

        assertNotNull(latent);
        assertEquals(32, latent.length);
        assertNotNull(reconstructed);
        assertEquals(128, reconstructed.length);
    }

    @Test
    void testDistillationLoss() {
        float loss = distiller.computeDistillationLoss(testInput, 2, 16);
        assertTrue(loss >= 0);
    }

    @Test
    void testDifferentBatchSizes() {
        for (int batch : new int[]{1, 2, 4}) {
            float[] input = new float[batch * 16 * 128];
            float[] output = distiller.forward(input, batch, 16);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testDifferentSeqLens() {
        for (int seq : new int[]{8, 16, 32}) {
            float[] input = new float[2 * seq * 128];
            float[] output = distiller.forward(input, 2, seq);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testAccessors() {
        assertEquals(128, distiller.getDModel());
        assertEquals(32, distiller.getLatentDim());
        assertEquals(0.25f, distiller.getCompressionRatio(), 1e-6f);
    }

    @Test
    void testDeterministicOutput() {
        float[] output1 = distiller.forward(testInput, 2, 16);
        float[] output2 = distiller.forward(testInput, 2, 16);
        assertArrayEquals(output1, output2, 1e-5f);
    }
}