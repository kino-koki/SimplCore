package com.qingxiaotuan.simpcore.compute;

import java.util.*;
import java.util.concurrent.*;

public class AdaptiveComputeEngine {
    
    public static class ComputePlan {
        public final int computeLevel;
        public final int batchSize;
        public final int maxTokens;
        public final int parallelism;
        public final boolean useGPU;
        
        public ComputePlan(int computeLevel, int batchSize, int maxTokens, int parallelism, boolean useGPU) {
            this.computeLevel = computeLevel;
            this.batchSize = batchSize;
            this.maxTokens = maxTokens;
            this.parallelism = parallelism;
            this.useGPU = useGPU;
        }
    }
    
    private final ExecutorService executor;
    private final ScheduledExecutorService monitor;
    private volatile int currentLoad;
    private volatile int targetComputeLevel;
    private volatile boolean autoScaleEnabled;
    private final int maxComputeLevel;
    
    public AdaptiveComputeEngine(int maxComputeLevel) {
        this.maxComputeLevel = maxComputeLevel;
        this.executor = Executors.newFixedThreadPool(maxComputeLevel * 4);
        this.monitor = Executors.newScheduledThreadPool(1);
        this.currentLoad = 0;
        this.targetComputeLevel = 1;
        this.autoScaleEnabled = true;
        startLoadMonitoring();
    }
    
    public AdaptiveComputeEngine() {
        this(5);
    }
    
    private void startLoadMonitoring() {
        monitor.scheduleAtFixedRate(() -> {
            if (autoScaleEnabled) {
                adjustComputeLevel();
            }
        }, 0, 10, TimeUnit.SECONDS);
    }
    
    private synchronized void adjustComputeLevel() {
        if (currentLoad < 20) {
            targetComputeLevel = Math.max(1, targetComputeLevel - 1);
        } else if (currentLoad < 50) {
            targetComputeLevel = 2;
        } else if (currentLoad < 70) {
            targetComputeLevel = 3;
        } else if (currentLoad < 90) {
            targetComputeLevel = 4;
        } else {
            targetComputeLevel = maxComputeLevel;
        }
    }
    
    public ComputePlan generatePlan(String input) {
        int complexity = analyzeComplexity(input);
        
        int batchSize = Math.max(1, 8 - complexity);
        int maxTokens = Math.max(100, 1024 - (complexity * 100));
        int parallelism = Math.min(8, complexity + 1);
        boolean useGPU = complexity >= 3;
        
        return new ComputePlan(targetComputeLevel, batchSize, maxTokens, parallelism, useGPU);
    }
    
    private int analyzeComplexity(String input) {
        int complexity = 1;
        
        if (input.length() > 1000) complexity++;
        if (input.length() > 5000) complexity++;
        if (input.length() > 10000) complexity++;
        
        if (input.contains("数学") || input.contains("计算") || input.contains("证明")) complexity++;
        if (input.contains("代码") || input.contains("编程") || input.contains("算法")) complexity++;
        if (input.contains("分析") || input.contains("推理") || input.contains("逻辑")) complexity++;
        
        return Math.min(maxComputeLevel, complexity);
    }
    
    public Future<String> executeWithPlan(String input, ComputePlan plan) {
        incrementLoad();
        
        return executor.submit(() -> {
            try {
                Thread.sleep(plan.computeLevel * 50);
                return "[Compute Level " + plan.computeLevel + "] Processed: " + input.substring(0, Math.min(50, input.length())) + "...";
            } finally {
                decrementLoad();
            }
        });
    }
    
    private synchronized void incrementLoad() {
        currentLoad = Math.min(100, currentLoad + 5);
    }
    
    private synchronized void decrementLoad() {
        currentLoad = Math.max(0, currentLoad - 5);
    }
    
    public void setAutoScaleEnabled(boolean enabled) {
        this.autoScaleEnabled = enabled;
    }
    
    public boolean isAutoScaleEnabled() {
        return autoScaleEnabled;
    }
    
    public synchronized int getCurrentLoad() {
        return currentLoad;
    }
    
    public int getTargetComputeLevel() {
        return targetComputeLevel;
    }
    
    public void shutdown() {
        monitor.shutdown();
        executor.shutdown();
    }
    
    public String getStatus() {
        return String.format(
            "AdaptiveCompute: load=%d%%, level=%d/%d, autoScale=%s",
            currentLoad, targetComputeLevel, maxComputeLevel, autoScaleEnabled
        );
    }
}