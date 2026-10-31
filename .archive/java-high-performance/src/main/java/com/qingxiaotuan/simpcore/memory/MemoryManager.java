package com.qingxiaotuan.simpcore.memory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public final class MemoryManager {

    private static final MemoryManager INSTANCE = new MemoryManager();

    private final AtomicLong allocatedBytes = new AtomicLong(0);
    private final AtomicLong peakBytes = new AtomicLong(0);
    private final Map<String, float[]> cachedArrays = new ConcurrentHashMap<>();
    private final Map<String, Long> arraySizes = new ConcurrentHashMap<>();

    private MemoryManager() {}

    public static MemoryManager getInstance() {
        return INSTANCE;
    }

    public float[] allocateFloatArray(int size) {
        float[] array = new float[size];
        long bytes = (long) size * Float.BYTES;
        allocatedBytes.addAndGet(bytes);
        updatePeak();
        return array;
    }

    public float[][] allocateFloatMatrix(int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        long bytes = (long) rows * cols * Float.BYTES;
        allocatedBytes.addAndGet(bytes);
        updatePeak();
        return matrix;
    }

    public void release(float[] array) {
        if (array != null) {
            long bytes = (long) array.length * Float.BYTES;
            allocatedBytes.addAndGet(-bytes);
        }
    }

    public void cacheArray(String key, float[] data) {
        if (cachedArrays.containsKey(key)) {
            releaseCached(key);
        }
        cachedArrays.put(key, data.clone());
        arraySizes.put(key, (long) data.length * Float.BYTES);
        allocatedBytes.addAndGet((long) data.length * Float.BYTES);
        updatePeak();
    }

    public float[] getCachedArray(String key) {
        float[] data = cachedArrays.get(key);
        return data != null ? data.clone() : null;
    }

    public void releaseCached(String key) {
        Long size = arraySizes.remove(key);
        if (size != null) {
            allocatedBytes.addAndGet(-size);
        }
        cachedArrays.remove(key);
    }

    public void clearCache() {
        for (String key : cachedArrays.keySet()) {
            releaseCached(key);
        }
    }

    public MemoryStats getStats() {
        return new MemoryStats(
            allocatedBytes.get(),
            peakBytes.get(),
            Runtime.getRuntime().totalMemory(),
            Runtime.getRuntime().freeMemory(),
            cachedArrays.size()
        );
    }

    private void updatePeak() {
        long current = allocatedBytes.get();
        long peak = peakBytes.get();
        while (current > peak) {
            if (peakBytes.compareAndSet(peak, current)) {
                break;
            }
            peak = peakBytes.get();
        }
    }

    public void resetStats() {
        allocatedBytes.set(0);
        peakBytes.set(0);
    }

    public record MemoryStats(
        long allocatedBytes,
        long peakBytes,
        long totalMemory,
        long freeMemory,
        int cachedSegments
    ) {
        public long usedMemory() {
            return totalMemory - freeMemory;
        }

        public double utilizationPercent() {
            return (double) usedMemory() / totalMemory * 100;
        }
    }
}