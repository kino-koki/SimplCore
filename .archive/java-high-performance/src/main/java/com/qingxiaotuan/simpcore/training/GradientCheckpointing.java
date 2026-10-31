package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.autograd.*;
import java.util.*;
import java.util.concurrent.*;

public class GradientCheckpointing {
    
    private final List<Tensor> checkpoints;
    private final List<Integer> checkpointIndices;
    private final int checkpointInterval;
    private volatile boolean enabled;
    
    private boolean useSelectiveCheckpointing;
    private Set<Integer> expensiveOps;
    
    private long totalForwardTimeNs;
    private long totalRecomputeTimeNs;
    private int recomputeCount;
    
    public GradientCheckpointing(int checkpointInterval) {
        if (checkpointInterval < 1) {
            throw new IllegalArgumentException("checkpointInterval must be >= 1");
        }
        this.checkpointInterval = checkpointInterval;
        this.checkpoints = new ArrayList<>();
        this.checkpointIndices = new ArrayList<>();
        this.enabled = true;
        this.useSelectiveCheckpointing = false;
        this.expensiveOps = new HashSet<>();
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            clear();
        }
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public void enableSelectiveCheckpointing(Set<Integer> expensiveOps) {
        this.useSelectiveCheckpointing = true;
        this.expensiveOps = expensiveOps != null ? expensiveOps : new HashSet<>();
    }
    
    public void disableSelectiveCheckpointing() {
        this.useSelectiveCheckpointing = false;
        this.expensiveOps.clear();
    }
    
    public Tensor forward(Tensor input, CheckpointableBlock block, int nLoops) {
        if (!enabled) {
            return block.forward(input, nLoops);
        }
        
        clear();
        
        Tensor current = input;
        
        for (int t = 0; t < nLoops; t++) {
            boolean shouldCheckpoint = shouldCheckpointAt(t);
            
            if (shouldCheckpoint) {
                checkpoints.add(current.clone());
                checkpointIndices.add(t);
            }
            
            long startTime = System.nanoTime();
            current = block.forward(current);
            totalForwardTimeNs += System.nanoTime() - startTime;
        }
        
        return current;
    }
    
    private boolean shouldCheckpointAt(int t) {
        if (useSelectiveCheckpointing) {
            return t % checkpointInterval == 0 || expensiveOps.contains(t);
        }
        return t % checkpointInterval == 0;
    }
    
    public Tensor backward(Tensor gradOutput, CheckpointableBlock block, int nLoops) {
        if (!enabled) {
            throw new IllegalStateException("Backward requires checkpointing to be enabled");
        }
        
        Tensor grad = gradOutput;
        Tensor currentInput = null;
        
        for (int t = nLoops - 1; t >= 0; t--) {
            int checkpointIdx = findNearestCheckpoint(t);
            
            if (checkpointIdx >= 0 && checkpointIndices.get(checkpointIdx) == t) {
                currentInput = checkpoints.get(checkpointIdx);
            } else if (checkpointIdx >= 0) {
                int startIdx = checkpointIndices.get(checkpointIdx);
                Tensor checkpoint = checkpoints.get(checkpointIdx);
                currentInput = checkpoint.clone();
                
                long startTime = System.nanoTime();
                for (int step = startIdx; step < t; step++) {
                    currentInput = block.forwardWithoutGrad(currentInput);
                }
                long recomputeTime = System.nanoTime() - startTime;
                totalRecomputeTimeNs += recomputeTime;
                recomputeCount++;
            }
            
            grad = block.backward(currentInput, grad);
        }
        
        return grad;
    }
    
    public Future<Tensor> backwardAsync(Tensor gradOutput, CheckpointableBlock block, 
                                        int nLoops, ExecutorService executor) {
        return executor.submit(() -> backward(gradOutput, block, nLoops));
    }
    
    private int findNearestCheckpoint(int t) {
        for (int i = checkpointIndices.size() - 1; i >= 0; i--) {
            if (checkpointIndices.get(i) <= t) {
                return i;
            }
        }
        return -1;
    }
    
    public int getCheckpointCount() {
        return checkpoints.size();
    }
    
    public long getMemorySavedBytes(int nLoops, int hiddenSize, int batchSize) {
        if (!enabled) return 0;
        
        long totalActivations = (long) nLoops * batchSize * hiddenSize * 4;
        long checkpointActivations = (long) checkpoints.size() * batchSize * hiddenSize * 4;
        return totalActivations - checkpointActivations;
    }
    
    public float getComputeOverhead(int nLoops) {
        if (!enabled || nLoops == 0) return 0;
        
        int recomputeSteps = 0;
        for (int t = 0; t < nLoops; t++) {
            int checkpointIdx = findNearestCheckpoint(t);
            if (checkpointIdx >= 0 && checkpointIndices.get(checkpointIdx) != t) {
                recomputeSteps++;
            }
        }
        return (float) recomputeSteps / nLoops;
    }
    
    public float getActualRecomputeRatio() {
        if (totalForwardTimeNs == 0) return 0;
        return (float) totalRecomputeTimeNs / totalForwardTimeNs;
    }
    
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("enabled", enabled);
        stats.put("checkpointCount", checkpoints.size());
        stats.put("checkpointInterval", checkpointInterval);
        stats.put("selectiveCheckpointing", useSelectiveCheckpointing);
        stats.put("recomputeCount", recomputeCount);
        stats.put("recomputeRatio", getActualRecomputeRatio());
        stats.put("forwardTimeMs", totalForwardTimeNs / 1_000_000.0);
        stats.put("recomputeTimeMs", totalRecomputeTimeNs / 1_000_000.0);
        return stats;
    }
    
    public void resetStats() {
        totalForwardTimeNs = 0;
        totalRecomputeTimeNs = 0;
        recomputeCount = 0;
    }
    
    public void clear() {
        checkpoints.clear();
        checkpointIndices.clear();
    }
    
    public interface CheckpointableBlock {
        Tensor forward(Tensor input);
        Tensor forward(Tensor input, int nLoops);
        Tensor forwardWithoutGrad(Tensor input);
        Tensor backward(Tensor input, Tensor gradOutput);
    }
}