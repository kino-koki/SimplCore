package com.qingxiaotuan.simpcore.memory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

public final class MemoryOptimizer {

    private static final MemoryOptimizer INSTANCE = new MemoryOptimizer();
    
    private final ConcurrentHashMap<Integer, ConcurrentLinkedQueue<float[]>> floatArrayPool;
    private final ConcurrentHashMap<Integer, ConcurrentLinkedQueue<float[][]>> floatMatrixPool;
    private final AtomicInteger allocations;
    private final AtomicInteger poolHits;
    private final int maxPoolSize;

    private MemoryOptimizer() {
        this.floatArrayPool = new ConcurrentHashMap<>();
        this.floatMatrixPool = new ConcurrentHashMap<>();
        this.allocations = new AtomicInteger(0);
        this.poolHits = new AtomicInteger(0);
        this.maxPoolSize = 1000;
    }

    public static MemoryOptimizer getInstance() {
        return INSTANCE;
    }

    public float[] allocateFloatArray(int size) {
        allocations.incrementAndGet();
        ConcurrentLinkedQueue<float[]> pool = floatArrayPool.computeIfAbsent(size, k -> new ConcurrentLinkedQueue<>());
        
        float[] array = pool.poll();
        if (array != null) {
            poolHits.incrementAndGet();
            return array;
        }
        
        return new float[size];
    }

    public void releaseFloatArray(float[] array) {
        if (array == null) return;
        
        ConcurrentLinkedQueue<float[]> pool = floatArrayPool.get(array.length);
        if (pool != null && pool.size() < maxPoolSize) {
            for (int i = 0; i < array.length; i++) {
                array[i] = 0.0f;
            }
            pool.offer(array);
        }
    }

    public float[][] allocateFloatMatrix(int rows, int cols) {
        allocations.incrementAndGet();
        int key = rows * 10000 + cols;
        ConcurrentLinkedQueue<float[][]> pool = floatMatrixPool.computeIfAbsent(key, k -> new ConcurrentLinkedQueue<>());
        
        float[][] matrix = pool.poll();
        if (matrix != null) {
            poolHits.incrementAndGet();
            return matrix;
        }
        
        return new float[rows][cols];
    }

    public void releaseFloatMatrix(float[][] matrix) {
        if (matrix == null) return;
        
        int rows = matrix.length;
        int cols = rows > 0 ? matrix[0].length : 0;
        int key = rows * 10000 + cols;
        
        ConcurrentLinkedQueue<float[][]> pool = floatMatrixPool.get(key);
        if (pool != null && pool.size() < maxPoolSize) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = 0.0f;
                }
            }
            pool.offer(matrix);
        }
    }

    public float[] allocateAligned(int size) {
        return allocateFloatArray(size);
    }

    public void releaseAligned(float[] array) {
        releaseFloatArray(array);
    }

    public float[] copyAndRelease(float[] source) {
        if (source == null) return null;
        
        float[] copy = allocateFloatArray(source.length);
        System.arraycopy(source, 0, copy, 0, source.length);
        releaseFloatArray(source);
        
        return copy;
    }

    public void clearPool(int size) {
        floatArrayPool.remove(size);
    }

    public void clearAllPools() {
        floatArrayPool.clear();
        floatMatrixPool.clear();
    }

    public MemoryStats getStats() {
        int totalPooled = 0;
        for (ConcurrentLinkedQueue<float[]> pool : floatArrayPool.values()) {
            totalPooled += pool.size();
        }
        for (ConcurrentLinkedQueue<float[][]> pool : floatMatrixPool.values()) {
            totalPooled += pool.size();
        }
        
        int totalAllocations = allocations.get();
        int hits = poolHits.get();
        double hitRate = totalAllocations > 0 ? (double) hits / totalAllocations * 100 : 0;
        
        return new MemoryStats(totalAllocations, hits, hitRate, totalPooled);
    }

    public void resetStats() {
        allocations.set(0);
        poolHits.set(0);
    }

    public record MemoryStats(
        int totalAllocations,
        int poolHits,
        double hitRatePercent,
        int totalPooledObjects
    ) {}

    public static class ScopedFloatArray implements AutoCloseable {
        private final float[] array;
        
        public ScopedFloatArray(int size) {
            this.array = MemoryOptimizer.getInstance().allocateFloatArray(size);
        }
        
        public float[] get() {
            return array;
        }
        
        @Override
        public void close() {
            MemoryOptimizer.getInstance().releaseFloatArray(array);
        }
    }

    public static class ScopedFloatMatrix implements AutoCloseable {
        private final float[][] matrix;
        
        public ScopedFloatMatrix(int rows, int cols) {
            this.matrix = MemoryOptimizer.getInstance().allocateFloatMatrix(rows, cols);
        }
        
        public float[][] get() {
            return matrix;
        }
        
        @Override
        public void close() {
            MemoryOptimizer.getInstance().releaseFloatMatrix(matrix);
        }
    }
}