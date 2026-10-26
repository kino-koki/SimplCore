package com.qingxiaotuan.simpcore.training;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class DistributedTrainingFramework {
    
    private static final int DEFAULT_GRAD_ACCUM_STEPS = 1;
    private static final int CHECKPOINT_INTERVAL = 1000;
    
    private final List<Node> nodes;
    private final int worldSize;
    private final int rank;
    
    private final GradientTracker gradientTracker;
    private final List<PipelineStage> pipelineStages;
    
    private final ExecutorService asyncExecutor;
    
    private final int gradientAccumulationSteps;
    private int currentAccumStep;
    
    private final AtomicLong totalSamplesProcessed = new AtomicLong(0);
    private final AtomicLong totalSteps = new AtomicLong(0);
    private final AtomicLong totalAllReduceTimeNs = new AtomicLong(0);
    private final AtomicInteger allReduceCount = new AtomicInteger(0);
    
    private final boolean useMixedPrecision;
    private final float lossScale;
    private final float minLossScale;
    private final AtomicInteger scaleAdjustCounter = new AtomicInteger(0);
    
    private float currentLossScale;
    private volatile boolean trainingPaused = false;
    
    private final Map<String, Float> layerLearningRates;
    private final Map<String, Long> layerUpdateTimestamps;
    
    public static class Node {
        final int id;
        final String host;
        final int gpuCount;
        final long vram;
        final boolean isMaster;
        final List<Integer> peers;
        
        Node(int id, String host, int gpuCount, long vram, boolean isMaster) {
            this.id = id;
            this.host = host;
            this.gpuCount = gpuCount;
            this.vram = vram;
            this.isMaster = isMaster;
            this.peers = new ArrayList<>();
        }
    }
    
    public static class AllReduceResult {
        final float[] values;
        final float latency;
        
        AllReduceResult(float[] values, float latency) {
            this.values = values;
            this.latency = latency;
        }
    }
    
    public static class GradientTracker {
        private final Map<String, float[]> gradients;
        private final Map<String, Long> updateCounts;
        private final float clipNorm;
        private final Object lock = new Object();
        
        GradientTracker(float clipNorm) {
            this.gradients = new HashMap<>();
            this.updateCounts = new HashMap<>();
            this.clipNorm = clipNorm;
        }
        
        void accumulate(String layerName, float[] grad) {
            synchronized (lock) {
                float[] current = gradients.get(layerName);
                if (current == null) {
                    current = new float[grad.length];
                    gradients.put(layerName, current);
                }
                
                for (int i = 0; i < grad.length; i++) {
                    current[i] += grad[i];
                }
                
                updateCounts.merge(layerName, 1L, Long::sum);
            }
        }
        
        float[] getAverage(String layerName) {
            synchronized (lock) {
                float[] grad = gradients.get(layerName);
                if (grad == null) return null;
                
                Long count = updateCounts.get(layerName);
                if (count == null || count == 0) return null;
                
                float[] avg = new float[grad.length];
                float norm = 0;
                
                for (int i = 0; i < grad.length; i++) {
                    avg[i] = grad[i] / count;
                    norm += avg[i] * avg[i];
                }
                norm = (float) Math.sqrt(norm);
                
                if (norm > clipNorm && norm > 0) {
                    float factor = clipNorm / norm;
                    for (int i = 0; i < avg.length; i++) {
                        avg[i] *= factor;
                    }
                }
                
                return avg;
            }
        }
        
        void reset() {
            synchronized (lock) {
                for (float[] grad : gradients.values()) {
                    Arrays.fill(grad, 0);
                }
                updateCounts.clear();
            }
        }
        
        void reset(String layerName) {
            synchronized (lock) {
                float[] grad = gradients.get(layerName);
                if (grad != null) {
                    Arrays.fill(grad, 0);
                }
                updateCounts.remove(layerName);
            }
        }
        
        int getAccumulatedCount(String layerName) {
            synchronized (lock) {
                Long count = updateCounts.get(layerName);
                return count != null ? count.intValue() : 0;
            }
        }
    }
    
    public static class PipelineStage {
        final int stageId;
        final int startLayer;
        final int endLayer;
        final int nodeId;
        int microBatchSize;
        
        PipelineStage(int stageId, int startLayer, int endLayer, int nodeId) {
            this.stageId = stageId;
            this.startLayer = startLayer;
            this.endLayer = endLayer;
            this.nodeId = nodeId;
            this.microBatchSize = 1;
        }
        
        int getLayerCount() {
            return endLayer - startLayer;
        }
    }
    
    public DistributedTrainingFramework(int worldSize) {
        this(worldSize, 0, DEFAULT_GRAD_ACCUM_STEPS, false);
    }
    
    public DistributedTrainingFramework(int worldSize, int rank, 
                                        int gradientAccumulationSteps,
                                        boolean useMixedPrecision) {
        this.worldSize = worldSize;
        this.rank = rank;
        this.gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
        this.currentAccumStep = 0;
        this.useMixedPrecision = useMixedPrecision;
        this.lossScale = 65536.0f;
        this.minLossScale = 1.0f;
        this.currentLossScale = useMixedPrecision ? lossScale : 1.0f;
        
        this.nodes = new ArrayList<>();
        this.gradientTracker = new GradientTracker(1.0f);
        this.pipelineStages = new ArrayList<>();
        
        this.layerLearningRates = new ConcurrentHashMap<>();
        this.layerUpdateTimestamps = new ConcurrentHashMap<>();
        
        this.asyncExecutor = Executors.newFixedThreadPool(4,
            new ThreadFactoryBuilder("dtf-async-"));
        
        initializeNodes();
    }
    
    private void initializeNodes() {
        for (int i = 0; i < worldSize; i++) {
            boolean isMaster = (i == 0);
            nodes.add(new Node(i, "node" + i + ".local", 8, 80L, isMaster));
        }
        
        for (Node node : nodes) {
            for (int i = 0; i < nodes.size(); i++) {
                if (i != node.id) {
                    node.peers.add(i);
                }
            }
        }
    }
    
    public void setupPipelineParallelism(int totalLayers) {
        setupPipelineParallelism(totalLayers, 1);
    }
    
    public void setupPipelineParallelism(int totalLayers, int microBatchSize) {
        pipelineStages.clear();
        int layersPerStage = (int) Math.ceil((double) totalLayers / worldSize);
        
        for (int i = 0; i < worldSize; i++) {
            int start = i * layersPerStage;
            int end = Math.min(start + layersPerStage, totalLayers);
            PipelineStage stage = new PipelineStage(i, start, end, i);
            stage.microBatchSize = microBatchSize;
            pipelineStages.add(stage);
        }
    }
    
    public AllReduceResult allReduce(float[] values) {
        long startTime = System.nanoTime();
        
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i] * worldSize;
        }
        
        for (Node node : nodes) {
            if (node.id != rank) {
                for (int i = 0; i < values.length; i++) {
                    result[i] += values[i] * 0.1f;
                }
            }
        }
        
        for (int i = 0; i < result.length; i++) {
            result[i] /= worldSize;
        }
        
        long endTime = System.nanoTime();
        float latency = (endTime - startTime) / 1_000_000_000.0f;
        
        totalAllReduceTimeNs.addAndGet(endTime - startTime);
        allReduceCount.incrementAndGet();
        
        return new AllReduceResult(result, latency);
    }
    
    public Future<AllReduceResult> allReduceAsync(float[] values) {
        return asyncExecutor.submit(() -> allReduce(values));
    }
    
    public void broadcast(float[] data) {
        for (Node node : nodes) {
            if (node.id != rank) {
                float[] received = new float[data.length];
                System.arraycopy(data, 0, received, 0, data.length);
            }
        }
    }
    
    public void updateGradients(String layerName, float[] grad) {
        if (useMixedPrecision) {
            for (int i = 0; i < grad.length; i++) {
                grad[i] *= currentLossScale;
            }
        }
        
        gradientTracker.accumulate(layerName, grad);
        layerUpdateTimestamps.put(layerName, System.currentTimeMillis());
        
        if (gradientTracker.getAccumulatedCount(layerName) >= worldSize * gradientAccumulationSteps) {
            float[] avgGrad = gradientTracker.getAverage(layerName);
            
            if (avgGrad != null) {
                if (useMixedPrecision) {
                    for (int i = 0; i < avgGrad.length; i++) {
                        avgGrad[i] /= currentLossScale;
                    }
                    
                    if (hasInfOrNaN(avgGrad)) {
                        currentLossScale = Math.max(minLossScale, currentLossScale / 2.0f);
                        gradientTracker.reset(layerName);
                        return;
                    } else {
                        scaleAdjustCounter.incrementAndGet();
                        if (scaleAdjustCounter.get() >= 2000) {
                            currentLossScale = Math.min(lossScale, currentLossScale * 2.0f);
                            scaleAdjustCounter.set(0);
                        }
                    }
                }
                
                broadcast(avgGrad);
            }
            
            gradientTracker.reset(layerName);
            currentAccumStep++;
        }
    }
    
    private boolean hasInfOrNaN(float[] arr) {
        for (float v : arr) {
            if (Float.isInfinite(v) || Float.isNaN(v)) {
                return true;
            }
        }
        return false;
    }
    
    public void setLayerLearningRate(String layerName, float lr) {
        layerLearningRates.put(layerName, lr);
    }
    
    public float getLayerLearningRate(String layerName, float defaultLr) {
        return layerLearningRates.getOrDefault(layerName, defaultLr);
    }
    
    public float computeBatchLoss(float[] logits, int[] targets, int batchSize) {
        float totalLoss = 0;
        
        for (int b = 0; b < batchSize; b++) {
            int offset = b * logits.length / batchSize;
            int target = targets[b];
            
            float maxLogit = Float.NEGATIVE_INFINITY;
            int logitPerSample = logits.length / batchSize;
            
            for (int i = 0; i < logitPerSample; i++) {
                maxLogit = Math.max(maxLogit, logits[offset + i]);
            }
            
            float sumExp = 0;
            for (int i = 0; i < logitPerSample; i++) {
                sumExp += (float) Math.exp(logits[offset + i] - maxLogit);
            }
            
            float loss = maxLogit + (float) Math.log(sumExp) - logits[offset + target];
            totalLoss += loss;
        }
        
        return totalLoss / batchSize;
    }
    
    public void syncModel(float[] modelParams) {
        allReduce(modelParams);
        broadcast(modelParams);
    }
    
    public Future<Void> syncModelAsync(float[] modelParams) {
        return asyncExecutor.submit(() -> {
            syncModel(modelParams);
            return null;
        });
    }
    
    public boolean shouldCheckpoint(int step) {
        return step > 0 && step % CHECKPOINT_INTERVAL == 0;
    }
    
    public void pauseTraining() {
        trainingPaused = true;
    }
    
    public void resumeTraining() {
        trainingPaused = false;
    }
    
    public boolean isTrainingPaused() {
        return trainingPaused;
    }
    
    public int getGradientAccumulationSteps() {
        return gradientAccumulationSteps;
    }
    
    public float getCurrentLossScale() {
        return currentLossScale;
    }
    
    public Map<String, Object> getTrainingStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalSamplesProcessed", totalSamplesProcessed.get());
        stats.put("totalSteps", totalSteps.get());
        stats.put("allReduceCount", allReduceCount.get());
        stats.put("avgAllReduceTimeMs", 
            allReduceCount.get() > 0 ? totalAllReduceTimeNs.get() / 1_000_000.0 / allReduceCount.get() : 0);
        stats.put("currentLossScale", currentLossScale);
        stats.put("gradientAccumulationSteps", gradientAccumulationSteps);
        stats.put("mixedPrecision", useMixedPrecision);
        return stats;
    }
    
    public void recordSample(int count) {
        totalSamplesProcessed.addAndGet(count);
    }
    
    public void recordStep() {
        totalSteps.incrementAndGet();
    }
    
    public void printClusterStatus() {
        System.out.println("Cluster Status:");
        System.out.println("  World Size: " + worldSize);
        System.out.println("  Rank: " + rank);
        System.out.println("  Gradient Accumulation Steps: " + gradientAccumulationSteps);
        System.out.println("  Mixed Precision: " + useMixedPrecision);
        if (useMixedPrecision) {
            System.out.println("  Loss Scale: " + currentLossScale);
        }
        System.out.println("  Nodes:");
        
        for (Node node : nodes) {
            System.out.println("    - Node " + node.id + ": " + node.host + 
                " (GPUs: " + node.gpuCount + ", VRAM: " + node.vram + "GB, Master: " + node.isMaster + ")");
        }
        
        System.out.println("  Pipeline Stages:");
        for (PipelineStage stage : pipelineStages) {
            System.out.println("    - Stage " + stage.stageId + ": Layers " + stage.startLayer + 
                "-" + stage.endLayer + " (" + stage.getLayerCount() + " layers, Node " + 
                stage.nodeId + ", microBatch=" + stage.microBatchSize + ")");
        }
    }
    
    public void simulateTraining(int steps) {
        System.out.println("\nSimulating distributed training...");
        Random r = new Random(42);
        
        for (int step = 0; step < steps; step++) {
            recordStep();
            
            float[] dummyGrad = new float[1024];
            for (int i = 0; i < dummyGrad.length; i++) {
                dummyGrad[i] = (float) r.nextGaussian() * 0.01f;
            }
            
            updateGradients("layer_" + (step % 10), dummyGrad);
            recordSample(32);
            
            if (step % 100 == 0) {
                AllReduceResult ar = allReduce(new float[]{1.0f, 2.0f, 3.0f});
                System.out.printf("  Step %d: AllReduce latency = %.4f ms, loss_scale=%.1f%n", 
                    step, ar.latency * 1000, currentLossScale);
            }
            
            if (shouldCheckpoint(step)) {
                System.out.println("  Step " + step + ": Checkpoint triggered");
            }
        }
        
        System.out.println("Distributed training simulation complete!");
        System.out.println("Stats: " + getTrainingStats());
    }
    
    public void shutdown() {
        asyncExecutor.shutdown();
        try {
            asyncExecutor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Distributed Training Framework");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        int worldSize = 8;
        int gradAccumSteps = 4;
        boolean useMixedPrecision = true;
        
        DistributedTrainingFramework dtf = new DistributedTrainingFramework(
            worldSize, 0, gradAccumSteps, useMixedPrecision);
        
        dtf.setupPipelineParallelism(240, 4);
        
        dtf.printClusterStatus();
        
        dtf.simulateTraining(500);
        
        dtf.shutdown();
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Framework Ready!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
    
    private static class ThreadFactoryBuilder implements ThreadFactory {
        private final String prefix;
        private final AtomicInteger counter = new AtomicInteger(0);
        
        ThreadFactoryBuilder(String prefix) {
            this.prefix = prefix;
        }
        
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, prefix + counter.incrementAndGet());
            t.setDaemon(true);
            return t;
        }
    }
}