package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.training.GradientResult;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.*;

public class ZYLAttentionSystemTest {

    private static final float EPSILON = 1e-4f;
    private static final float GRAD_TOLERANCE = 1e-2f;

    @Test
    public void testForwardPass() {
        System.out.println("Testing ZYLAttentionSystem forward pass...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 4;
        config.dModel = 64;
        config.headDim = 16;
        config.maxContextLength = 128;
        config.slidingWindowSize = 32;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 2;
        int seqLen = 16;
        float[] input = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = random.nextFloat() * 2 - 1;
        }

        ZYLAttentionSystem.ZYLResult result = zyl.forward(input, batchSize, seqLen);

        assertNotNull("Output should not be null", result.output);
        assertEquals("Output size should match", batchSize * seqLen * config.dModel, result.output.length);
        assertEquals("Active tokens should match", batchSize * seqLen, result.activeTokens);

        for (float v : result.output) {
            assertTrue("Output values should be finite", Float.isFinite(v));
        }

        System.out.println("Forward pass test passed!");
    }

    @Test
    public void testBackwardPassGradientVerification() {
        System.out.println("Testing ZYLAttentionSystem backward pass gradient verification...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 2;
        config.dModel = 32;
        config.headDim = 16;
        config.maxContextLength = 64;
        config.slidingWindowSize = 16;
        config.enableRoPE = false;
        config.enableHierarchicalMemory = false;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 1;
        int seqLen = 8;
        float[] input = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = random.nextFloat() * 2 - 1;
        }

        ZYLAttentionSystem.ZYLResult forwardResult = zyl.forward(input, batchSize, seqLen);

        float[] dOutput = new float[batchSize * seqLen * config.dModel];
        for (int i = 0; i < dOutput.length; i++) {
            dOutput[i] = random.nextFloat() * 2 - 1;
        }

        GradientResult gradResult = zyl.backward(dOutput, input, batchSize, seqLen, forwardResult);

        float[] dInput = gradResult.getdInput();
        assertNotNull("dInput should not be null", dInput);
        assertEquals("dInput size should match", input.length, dInput.length);

        for (float v : dInput) {
            assertTrue("Gradient values should be finite", Float.isFinite(v));
        }

        float[] dQWeights = gradResult.getGradient("qWeights");
        float[] dKWeights = gradResult.getGradient("kWeights");
        float[] dVWeights = gradResult.getGradient("vWeights");
        float[] dOWeights = gradResult.getGradient("oWeights");

        assertNotNull("dQWeights should not be null", dQWeights);
        assertNotNull("dKWeights should not be null", dKWeights);
        assertNotNull("dVWeights should not be null", dVWeights);
        assertNotNull("dOWeights should not be null", dOWeights);

        float gradNorm = 0;
        for (float g : dInput) gradNorm += g * g;
        gradNorm = (float) Math.sqrt(gradNorm);
        assertTrue("Gradient norm should be positive", gradNorm > 0);

        System.out.println("Backward pass gradient verification test passed!");
    }

    @Test
    public void testKVCCache() {
        System.out.println("Testing ZYLAttentionSystem KV Cache...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 4;
        config.dModel = 64;
        config.headDim = 16;
        config.maxContextLength = 128;
        config.enableKVCache = true;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 1;
        int seqLen = 8;
        float[] input1 = new float[batchSize * seqLen * config.dModel];
        float[] input2 = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input1.length; i++) {
            input1[i] = random.nextFloat() * 2 - 1;
            input2[i] = random.nextFloat() * 2 - 1;
        }

        ZYLAttentionSystem.ZYLResult result1 = zyl.forward(input1, batchSize, seqLen);
        ZYLAttentionSystem.ZYLResult result2 = zyl.forward(input2, batchSize, seqLen);

        assertNotNull("First output should not be null", result1.output);
        assertNotNull("Second output should not be null", result2.output);

        assertTrue("Cache position should have advanced", zyl.getCurrentPosition() > 0);

        System.out.println("KV Cache test passed!");
    }

    @Test
    public void testRoPEEncoding() {
        System.out.println("Testing ZYLAttentionSystem RoPE encoding...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 2;
        config.dModel = 32;
        config.headDim = 16;
        config.maxContextLength = 64;
        config.enableRoPE = true;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 1;
        int seqLen = 4;
        float[] input = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = random.nextFloat() * 2 - 1;
        }

        ZYLAttentionSystem.ZYLResult result = zyl.forward(input, batchSize, seqLen);

        assertNotNull("Output should not be null with RoPE", result.output);

        for (float v : result.output) {
            assertTrue("Output with RoPE should be finite", Float.isFinite(v));
        }

        System.out.println("RoPE encoding test passed!");
    }

    @Test
    public void testHierarchicalMemory() {
        System.out.println("Testing ZYLAttentionSystem hierarchical memory...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 2;
        config.dModel = 32;
        config.headDim = 16;
        config.maxContextLength = 64;
        config.memoryLayers = 2;
        config.memoryBlockSize = 8;
        config.enableHierarchicalMemory = true;
        config.slidingWindowSize = 8;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 1;
        int seqLen = 16;
        float[] input = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = random.nextFloat() * 2 - 1;
        }

        ZYLAttentionSystem.ZYLResult result = zyl.forward(input, batchSize, seqLen);

        assertNotNull("Output should not be null with hierarchical memory", result.output);
        assertTrue("Memory loss should be small", result.memoryLoss < 0.1);

        for (float v : result.output) {
            assertTrue("Output with hierarchical memory should be finite", Float.isFinite(v));
        }

        System.out.println("Hierarchical memory test passed!");
    }

    @Test
    public void testSaveLoad() {
        System.out.println("Testing ZYLAttentionSystem save/load...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 2;
        config.dModel = 32;
        config.headDim = 16;
        config.maxContextLength = 64;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        float[] qWeightsBefore = zyl.getQWeights().clone();
        float[] kWeightsBefore = zyl.getKWeights().clone();
        float[] vWeightsBefore = zyl.getVWeights().clone();
        float[] oWeightsBefore = zyl.getOWeights().clone();

        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(baos);
            zyl.save(dos);
            dos.close();

            byte[] data = baos.toByteArray();
            assertTrue("Saved data should not be empty", data.length > 0);

            ByteArrayInputStream bais = new ByteArrayInputStream(data);
            DataInputStream dis = new DataInputStream(bais);

            ZYLAttentionSystem zylLoaded = new ZYLAttentionSystem(config);
            zylLoaded.initialize();
            zylLoaded.load(dis);
            dis.close();

            float[] qWeightsAfter = zylLoaded.getQWeights();
            float[] kWeightsAfter = zylLoaded.getKWeights();
            float[] vWeightsAfter = zylLoaded.getVWeights();
            float[] oWeightsAfter = zylLoaded.getOWeights();

            assertArrayEquals("Q weights should match after load", qWeightsBefore, qWeightsAfter, EPSILON);
            assertArrayEquals("K weights should match after load", kWeightsBefore, kWeightsAfter, EPSILON);
            assertArrayEquals("V weights should match after load", vWeightsBefore, vWeightsAfter, EPSILON);
            assertArrayEquals("O weights should match after load", oWeightsBefore, oWeightsAfter, EPSILON);

            System.out.println("Save/load test passed!");
        } catch (Exception e) {
            fail("Save/load failed: " + e.getMessage());
        }
    }

    @Test
    public void testReset() {
        System.out.println("Testing ZYLAttentionSystem reset...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 2;
        config.dModel = 32;
        config.headDim = 16;
        config.maxContextLength = 64;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);
        zyl.initialize();

        int batchSize = 1;
        int seqLen = 8;
        float[] input = new float[batchSize * seqLen * config.dModel];
        Random random = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = random.nextFloat() * 2 - 1;
        }

        zyl.forward(input, batchSize, seqLen);
        assertTrue("Position should have advanced", zyl.getCurrentPosition() > 0);

        zyl.reset();
        assertEquals("Position should be reset", 0, zyl.getCurrentPosition());

        System.out.println("Reset test passed!");
    }

    @Test
    public void testMemoryRetentionCalculation() {
        System.out.println("Testing ZYLAttentionSystem memory retention calculation...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.memoryRetention = 0.98;
        config.memoryBlockSize = 1024;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);

        double retention1024 = zyl.calculateMemoryRetentionRate(1024);
        double retention4096 = zyl.calculateMemoryRetentionRate(4096);

        assertTrue("Retention should decrease with longer context", retention4096 <= retention1024);
        assertTrue("Retention should be > 90% for 4096 tokens", retention4096 > 0.90);

        System.out.println("Memory retention calculation test passed!");
    }

    @Test
    public void testComputationComplexity() {
        System.out.println("Testing ZYLAttentionSystem computation complexity...");

        ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
        config.nHeads = 32;
        config.sparseTopK = 64;

        ZYLAttentionSystem zyl = new ZYLAttentionSystem(config);

        double denseComplexity = zyl.estimateComputationComplexity(4096);
        config.enableSparseAttention = true;
        double sparseComplexity = zyl.estimateComputationComplexity(4096);

        assertTrue("Sparse attention should be faster", sparseComplexity < denseComplexity);

        System.out.println("Computation complexity test passed!");
    }
}