package com.qingxiaotuan.simpcore.memory;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MemoryOptimizerTest {

    @Test
    public void testGetInstance() {
        MemoryOptimizer optimizer1 = MemoryOptimizer.getInstance();
        MemoryOptimizer optimizer2 = MemoryOptimizer.getInstance();
        
        assertSame(optimizer1, optimizer2);
    }

    @Test
    public void testAllocateAndRelease() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[] array1 = optimizer.allocateFloatArray(100);
        float[] array2 = optimizer.allocateFloatArray(200);
        
        assertNotNull(array1);
        assertNotNull(array2);
        assertEquals(100, array1.length);
        assertEquals(200, array2.length);
        
        optimizer.releaseFloatArray(array1);
        optimizer.releaseFloatArray(array2);
    }

    @Test
    public void testAllocateMatrix() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[][] matrix = optimizer.allocateMatrix(10, 20);
        
        assertNotNull(matrix);
        assertEquals(10, matrix.length);
        for (int i = 0; i < 10; i++) {
            assertEquals(20, matrix[i].length);
        }
        
        optimizer.releaseMatrix(matrix);
    }

    @Test
    public void testPoolReuse() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[] array1 = optimizer.allocateFloatArray(100);
        optimizer.releaseFloatArray(array1);
        
        float[] array2 = optimizer.allocateFloatArray(100);
        
        assertSame(array1, array2);
    }

    @Test
    public void testDifferentSizes() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[] small = optimizer.allocateFloatArray(50);
        float[] large = optimizer.allocateFloatArray(200);
        
        assertNotSame(small, large);
        
        optimizer.releaseFloatArray(small);
        optimizer.releaseFloatArray(large);
    }

    @Test
    public void testGetStats() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        MemoryOptimizer.MemoryStats stats = optimizer.getStats();
        
        assertNotNull(stats);
        
        float[] array = optimizer.allocateFloatArray(100);
        optimizer.releaseFloatArray(array);
        
        stats = optimizer.getStats();
        
        assertTrue(stats.totalAllocations() >= 1);
        assertTrue(stats.hitRatePercent() >= 0);
        assertTrue(stats.hitRatePercent() <= 100);
    }

    @Test
    public void testScopedFloatArray() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[] array = null;
        try (MemoryOptimizer.ScopedFloatArray scoped = new MemoryOptimizer.ScopedFloatArray(100)) {
            array = scoped.get();
            assertNotNull(array);
            assertEquals(100, array.length);
        }
        
        assertTrue(optimizer.getPool(100).contains(array));
    }

    @Test
    public void testScopedMatrix() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[][] matrix = null;
        try (MemoryOptimizer.ScopedMatrix scoped = new MemoryOptimizer.ScopedMatrix(10, 20)) {
            matrix = scoped.get();
            assertNotNull(matrix);
            assertEquals(10, matrix.length);
        }
        
        assertTrue(optimizer.getMatrixPool(10, 20).contains(matrix));
    }

    @Test
    public void testReset() {
        MemoryOptimizer optimizer = MemoryOptimizer.getInstance();
        
        float[] array = optimizer.allocateFloatArray(100);
        optimizer.releaseFloatArray(array);
        
        optimizer.reset();
        
        float[] newArray = optimizer.allocateFloatArray(100);
        
        assertNotSame(array, newArray);
    }
}