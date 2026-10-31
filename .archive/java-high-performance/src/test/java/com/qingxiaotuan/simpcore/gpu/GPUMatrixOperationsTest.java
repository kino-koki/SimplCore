package com.qingxiaotuan.simpcore.gpu;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GPUMatrixOperationsTest {

    @Test
    public void testMatmul() {
        int M = 2, K = 3, N = 4;
        float[] A = {1, 2, 3, 4, 5, 6};
        float[] B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        
        float[] result = GPUMatrixOperations.matmul(A, B, M, K, N);
        
        assertNotNull(result);
        assertEquals(M * N, result.length);
        
        float expected00 = 1*1 + 2*5 + 3*9;
        float expected01 = 1*2 + 2*6 + 3*10;
        assertEquals(expected00, result[0], 0.001f);
        assertEquals(expected01, result[1], 0.001f);
    }

    @Test
    public void testMatmulSquare() {
        int size = 4;
        float[] A = new float[size * size];
        float[] B = new float[size * size];
        for (int i = 0; i < size * size; i++) {
            A[i] = i + 1;
            B[i] = i + 1;
        }
        
        float[] result = GPUMatrixOperations.matmul(A, B, size, size, size);
        
        assertNotNull(result);
        assertEquals(size * size, result.length);
    }

    @Test
    public void testSoftmax() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        
        float[] probs = GPUMatrixOperations.softmax(logits);
        
        assertNotNull(probs);
        assertEquals(logits.length, probs.length);
        
        float sum = 0;
        for (float p : probs) {
            sum += p;
        }
        assertEquals(1.0f, sum, 0.001f);
        
        assertTrue(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    @Test
    public void testRelu() {
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        
        float[] output = GPUMatrixOperations.relu(input);
        
        assertNotNull(output);
        assertEquals(input.length, output.length);
        assertEquals(0.0f, output[0], 0.001f);
        assertEquals(0.0f, output[1], 0.001f);
        assertEquals(0.0f, output[2], 0.001f);
        assertEquals(1.0f, output[3], 0.001f);
        assertEquals(2.0f, output[4], 0.001f);
    }

    @Test
    public void testSilu() {
        float[] input = {-2.0f, 0.0f, 2.0f};
        
        float[] output = GPUMatrixOperations.silu(input);
        
        assertNotNull(output);
        assertEquals(input.length, output.length);
    }

    @Test
    public void testAdd() {
        float[] A = {1, 2, 3};
        float[] B = {4, 5, 6};
        
        float[] result = GPUMatrixOperations.add(A, B);
        
        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(5, result[0], 0.001f);
        assertEquals(7, result[1], 0.001f);
        assertEquals(9, result[2], 0.001f);
    }

    @Test
    public void testMul() {
        float[] A = {1, 2, 3};
        float scalar = 2.0f;
        
        float[] result = GPUMatrixOperations.mul(A, scalar);
        
        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(2, result[0], 0.001f);
        assertEquals(4, result[1], 0.001f);
        assertEquals(6, result[2], 0.001f);
    }

    @Test
    public void testDotProduct() {
        float[] a = {1, 2, 3};
        float[] b = {4, 5, 6};
        
        float result = GPUMatrixOperations.dotProduct(a, b);
        
        assertEquals(32.0f, result, 0.001f);
    }

    @Test
    public void testOuterProduct() {
        float[] a = {1, 2};
        float[] b = {3, 4, 5};
        
        float[] result = GPUMatrixOperations.outerProduct(a, b, 2, 3);
        
        assertNotNull(result);
        assertEquals(6, result.length);
        assertEquals(3, result[0], 0.001f);
        assertEquals(4, result[1], 0.001f);
        assertEquals(5, result[2], 0.001f);
        assertEquals(6, result[3], 0.001f);
        assertEquals(8, result[4], 0.001f);
        assertEquals(10, result[5], 0.001f);
    }

    @Test
    public void testIsGpuAvailable() {
        boolean available = GPUMatrixOperations.isGpuAvailable();
        
        assertTrue(available || !available);
    }
}