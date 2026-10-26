package com.qingxiaotuan.simpcore.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class TopologyAwareRouterTest {

    private TopologyAwareRouter router;
    private float[] testInput;

    @BeforeEach
    void setUp() {
        router = new TopologyAwareRouter(128, 8, 2);
        testInput = new float[2 * 16 * 128];
        for (int i = 0; i < testInput.length; i++) {
            testInput[i] = (float) (Math.random() - 0.5) * 2;
        }
    }

    @Test
    void testForwardProducesOutput() {
        float[] output = router.forward(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testOutputHasNonZeroValues() {
        float[] output = router.forward(testInput, 2, 16);
        boolean hasNonZero = Arrays.stream(output).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testZeroInput() {
        float[] zeroInput = new float[2 * 16 * 128];
        float[] output = router.forward(zeroInput, 2, 16);
        assertNotNull(output);
        assertEquals(zeroInput.length, output.length);
    }

    @Test
    void testDifferentBatchSizes() {
        for (int batch : new int[]{1, 2, 4}) {
            float[] input = new float[batch * 16 * 128];
            float[] output = router.forward(input, batch, 16);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testDifferentSeqLens() {
        for (int seq : new int[]{8, 16, 32}) {
            float[] input = new float[2 * seq * 128];
            float[] output = router.forward(input, 2, seq);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testAccessors() {
        assertEquals(128, router.getDModel());
        assertEquals(8, router.getNPaths());
        assertEquals(2, router.getTopK());
    }

    @Test
    void testDeterministicOutput() {
        float[] output1 = router.forward(testInput, 2, 16);
        float[] output2 = router.forward(testInput, 2, 16);
        assertArrayEquals(output1, output2, 1e-5f);
    }
}