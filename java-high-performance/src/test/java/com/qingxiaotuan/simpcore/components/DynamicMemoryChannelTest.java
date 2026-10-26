package com.qingxiaotuan.simpcore.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class DynamicMemoryChannelTest {

    private DynamicMemoryChannel channel;
    private float[] testInput;

    @BeforeEach
    void setUp() {
        channel = new DynamicMemoryChannel(128, 16, 128);
        testInput = new float[2 * 16 * 128];
        for (int i = 0; i < testInput.length; i++) {
            testInput[i] = (float) (Math.random() - 0.5) * 2;
        }
    }

    @Test
    void testForwardProducesOutput() {
        float[] output = channel.forward(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testOutputHasNonZeroValues() {
        float[] output = channel.forward(testInput, 2, 16);
        boolean hasNonZero = Arrays.stream(output).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testZeroInput() {
        float[] zeroInput = new float[2 * 16 * 128];
        float[] output = channel.forward(zeroInput, 2, 16);
        assertNotNull(output);
        assertEquals(zeroInput.length, output.length);
    }

    @Test
    void testMemoryStateAccess() {
        float[] memory = channel.getMemoryState(0);
        assertNotNull(memory);
        assertEquals(128, memory.length);
    }

    @Test
    void testInvalidMemoryStateAccess() {
        assertThrows(IllegalArgumentException.class, () -> channel.getMemoryState(-1));
        assertThrows(IllegalArgumentException.class, () -> channel.getMemoryState(16));
    }

    @Test
    void testResetMemory() {
        channel.forward(testInput, 2, 16);
        float[] memoryBefore = channel.getMemoryState(0);
        channel.resetMemory();
        float[] memoryAfter = channel.getMemoryState(0);
        assertNotEquals(Arrays.hashCode(memoryBefore), Arrays.hashCode(memoryAfter));
    }

    @Test
    void testDifferentBatchSizes() {
        for (int batch : new int[]{1, 2, 4}) {
            float[] input = new float[batch * 16 * 128];
            float[] output = channel.forward(input, batch, 16);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testDifferentSeqLens() {
        for (int seq : new int[]{8, 16, 32}) {
            float[] input = new float[2 * seq * 128];
            float[] output = channel.forward(input, 2, seq);
            assertNotNull(output);
            assertEquals(input.length, output.length);
        }
    }

    @Test
    void testAccessors() {
        assertEquals(128, channel.getDModel());
        assertEquals(16, channel.getNChannels());
        assertEquals(128, channel.getMemoryDim());
    }
}