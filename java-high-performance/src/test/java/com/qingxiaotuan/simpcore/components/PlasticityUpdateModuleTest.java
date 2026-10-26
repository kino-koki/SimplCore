package com.qingxiaotuan.simpcore.components;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import static org.junit.jupiter.api.Assertions.*;

class PlasticityUpdateModuleTest {

    private PlasticityUpdateModule module;
    private float[] testInput;

    @BeforeEach
    void setUp() {
        module = new PlasticityUpdateModule(128);
        testInput = new float[2 * 16 * 128];
        for (int i = 0; i < testInput.length; i++) {
            testInput[i] = (float) (Math.random() - 0.5) * 2;
        }
    }

    @Test
    void testForwardProducesOutput() {
        float[] output = module.forward(testInput, 2, 16);
        assertNotNull(output);
        assertEquals(testInput.length, output.length);
    }

    @Test
    void testOutputHasNonZeroValues() {
        float[] output = module.forward(testInput, 2, 16);
        boolean hasNonZero = Arrays.stream(output).anyMatch(v -> Math.abs(v) > 1e-8);
        assertTrue(hasNonZero);
    }

    @Test
    void testZeroInput() {
        float[] zeroInput = new float[2 * 16 * 128];
        float[] output = module.forward(zeroInput, 2, 16);
        assertNotNull(output);
        assertEquals(zeroInput.length, output.length);
    }

    @Test
    void testDisablePlasticity() {
        module.setEnabled(false);
        float[][] weightsBefore = module.getWeights();
        module.forward(testInput, 2, 16);
        float[][] weightsAfter = module.getWeights();

        for (int i = 0; i < 128; i++) {
            assertArrayEquals(weightsBefore[i], weightsAfter[i], 1e-8f);
        }
    }

    @Test
    void testEnablePlasticity() {
        assertTrue(module.isEnabled());
        module.setEnabled(false);
        assertFalse(module.isEnabled());
        module.setEnabled(true);
        assertTrue(module.isEnabled());
    }

    @Test
    void testWeightsAreBounded() {
        module.setEnabled(true);
        for (int i = 0; i < 200; i++) {
            module.forward(testInput, 2, 16);
        }
        float[][] weights = module.getWeights();
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                assertTrue(weights[i][j] >= -1.0f && weights[i][j] <= 1.0f);
            }
        }
    }

    @Test
    void testAccessors() {
        assertEquals(128, module.getDModel());
        assertEquals(100, module.getUpdateFreq());
        assertEquals(0.001f, module.getLearningRate(), 1e-6f);
    }
}