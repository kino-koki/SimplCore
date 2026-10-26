package com.qingxiaotuan.simpcore.vision.inference;

import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer.FeatureResult;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.ProcessedFrame;
import java.util.*;
import java.util.concurrent.*;

public class InferenceDecisionLayer {
    
    private final List<InferenceEngine> engines;
    private final ExecutorService executor;
    private final ScheduledExecutorService scheduler;
    private final int parallelism;
    private volatile boolean running = true;
    
    public InferenceDecisionLayer() {
        this.engines = new ArrayList<>();
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(parallelism);
        this.scheduler = Executors.newScheduledThreadPool(2);
    }
    
    public InferenceDecisionLayer(int parallelism) {
        this.engines = new ArrayList<>();
        this.parallelism = parallelism;
        this.executor = Executors.newFixedThreadPool(parallelism);
        this.scheduler = Executors.newScheduledThreadPool(2);
    }
    
    public void addEngine(InferenceEngine engine) {
        engines.add(engine);
    }
    
    public void removeEngine(InferenceEngine engine) {
        engines.remove(engine);
    }
    
    public void clearEngines() {
        engines.clear();
    }
    
    public InferenceResult infer(ProcessedFrame frame, FeatureResult features) {
        InferenceResult result = new InferenceResult(frame.getTimestamp());
        
        for (InferenceEngine engine : engines) {
            if (!running) break;
            long startTime = System.nanoTime();
            InferenceOutput output = engine.infer(frame, features);
            long endTime = System.nanoTime();
            
            result.addOutput(engine.getName(), output);
            result.addProcessingTime(engine.getName(), (endTime - startTime) / 1_000_000.0);
        }
        
        return result;
    }
    
    public Future<InferenceResult> inferAsync(ProcessedFrame frame, FeatureResult features) {
        return executor.submit(() -> infer(frame, features));
    }
    
    public void shutdown() {
        running = false;
        executor.shutdown();
        scheduler.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            scheduler.shutdownNow();
        }
    }
    
    public int getEngineCount() {
        return engines.size();
    }
    
    public interface InferenceEngine {
        InferenceOutput infer(ProcessedFrame frame, FeatureResult features);
        String getName();
        boolean isEnabled();
        void setEnabled(boolean enabled);
        int getPriority();
        void setPriority(int priority);
    }
    
    public static class InferenceOutput {
        private String type;
        private List<Object> results;
        private Map<String, Object> metadata;
        private boolean successful;
        private String errorMessage;
        
        public InferenceOutput(String type) {
            this.type = type;
            this.results = new ArrayList<>();
            this.metadata = new HashMap<>();
            this.successful = true;
        }
        
        public void addResult(Object result) {
            results.add(result);
        }
        
        public void addResults(List<Object> results) {
            this.results.addAll(results);
        }
        
        public void addMetadata(String key, Object value) {
            metadata.put(key, value);
        }
        
        public void markFailed(String errorMessage) {
            this.successful = false;
            this.errorMessage = errorMessage;
        }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public List<Object> getResults() { return results; }
        public void setResults(List<Object> results) { this.results = results; }
        
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        
        public boolean isSuccessful() { return successful; }
        public void setSuccessful(boolean successful) { this.successful = successful; }
        
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        
        public int getResultCount() { return results.size(); }
    }
    
    public static class InferenceResult {
        private long timestamp;
        private Map<String, InferenceOutput> outputs;
        private Map<String, Double> processingTimes;
        private Map<String, Object> metadata;
        
        public InferenceResult(long timestamp) {
            this.timestamp = timestamp;
            this.outputs = new HashMap<>();
            this.processingTimes = new HashMap<>();
            this.metadata = new HashMap<>();
        }
        
        public void addOutput(String engineName, InferenceOutput output) {
            outputs.put(engineName, output);
        }
        
        public InferenceOutput getOutput(String engineName) {
            return outputs.get(engineName);
        }
        
        public Map<String, InferenceOutput> getAllOutputs() {
            return outputs;
        }
        
        public void addProcessingTime(String engineName, double timeMs) {
            processingTimes.put(engineName, timeMs);
        }
        
        public Map<String, Double> getProcessingTimes() { return processingTimes; }
        
        public void addMetadata(String key, Object value) { metadata.put(key, value); }
        
        public Map<String, Object> getMetadata() { return metadata; }
        
        public long getTimestamp() { return timestamp; }
        
        public double getTotalProcessingTime() {
            return processingTimes.values().stream().mapToDouble(Double::doubleValue).sum();
        }
        
        public boolean hasDetections() {
            return outputs.values().stream().anyMatch(o -> o.isSuccessful() && o.getResultCount() > 0);
        }
    }
}