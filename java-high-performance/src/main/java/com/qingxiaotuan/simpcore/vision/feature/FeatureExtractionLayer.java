package com.qingxiaotuan.simpcore.vision.feature;

import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.ProcessedFrame;
import java.util.*;
import java.util.concurrent.*;

public class FeatureExtractionLayer {
    
    private final List<FeatureExtractor> extractors;
    private final ExecutorService executor;
    private final int parallelism;
    private volatile boolean running = true;
    
    public FeatureExtractionLayer() {
        this.extractors = new ArrayList<>();
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(parallelism);
    }
    
    public FeatureExtractionLayer(int parallelism) {
        this.extractors = new ArrayList<>();
        this.parallelism = parallelism;
        this.executor = Executors.newFixedThreadPool(parallelism);
    }
    
    public void addExtractor(FeatureExtractor extractor) {
        extractors.add(extractor);
    }
    
    public void removeExtractor(FeatureExtractor extractor) {
        extractors.remove(extractor);
    }
    
    public void clearExtractors() {
        extractors.clear();
    }
    
    public FeatureResult extract(ProcessedFrame frame) {
        FeatureResult result = new FeatureResult(frame.getTimestamp());
        
        for (FeatureExtractor extractor : extractors) {
            if (!running) break;
            long startTime = System.nanoTime();
            FeatureDescriptor descriptor = extractor.extract(frame);
            long endTime = System.nanoTime();
            
            result.addDescriptor(extractor.getName(), descriptor);
            result.addProcessingTime(extractor.getName(), (endTime - startTime) / 1_000_000.0);
        }
        
        return result;
    }
    
    public Future<FeatureResult> extractAsync(ProcessedFrame frame) {
        return executor.submit(() -> extract(frame));
    }
    
    public void shutdown() {
        running = false;
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
    
    public int getExtractorCount() {
        return extractors.size();
    }
    
    public interface FeatureExtractor {
        FeatureDescriptor extract(ProcessedFrame frame);
        String getName();
        boolean isEnabled();
        void setEnabled(boolean enabled);
        int getFeatureDim();
    }
    
    public static class FeatureDescriptor {
        private float[] features;
        private int[] keypoints;
        private float[] keypointScores;
        private float[] descriptors;
        private String type;
        private int dimension;
        
        public FeatureDescriptor() {
            this.features = new float[0];
            this.keypoints = new int[0];
            this.keypointScores = new float[0];
            this.descriptors = new float[0];
            this.type = "unknown";
            this.dimension = 0;
        }
        
        public FeatureDescriptor(float[] features, String type) {
            this.features = features;
            this.type = type;
            this.dimension = features.length;
            this.keypoints = new int[0];
            this.keypointScores = new float[0];
            this.descriptors = new float[0];
        }
        
        public FeatureDescriptor(int[] keypoints, float[] descriptors, String type, int dimension) {
            this.keypoints = keypoints;
            this.descriptors = descriptors;
            this.type = type;
            this.dimension = dimension;
            this.features = new float[0];
            this.keypointScores = new float[0];
        }
        
        public float[] getFeatures() { return features; }
        public void setFeatures(float[] features) { 
            this.features = features; 
            this.dimension = features.length;
        }
        
        public int[] getKeypoints() { return keypoints; }
        public void setKeypoints(int[] keypoints) { this.keypoints = keypoints; }
        
        public float[] getKeypointScores() { return keypointScores; }
        public void setKeypointScores(float[] keypointScores) { this.keypointScores = keypointScores; }
        
        public float[] getDescriptors() { return descriptors; }
        public void setDescriptors(float[] descriptors) { this.descriptors = descriptors; }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public int getDimension() { return dimension; }
        public void setDimension(int dimension) { this.dimension = dimension; }
        
        public int getKeypointCount() { return keypoints.length / 2; }
        
        public float[] getKeypoint(int index) {
            if (index < 0 || index >= getKeypointCount()) return null;
            return new float[]{keypoints[index * 2], keypoints[index * 2 + 1]};
        }
    }
    
    public static class FeatureResult {
        private long timestamp;
        private Map<String, FeatureDescriptor> descriptors;
        private Map<String, Double> processingTimes;
        private Map<String, Object> metadata;
        
        public FeatureResult(long timestamp) {
            this.timestamp = timestamp;
            this.descriptors = new HashMap<>();
            this.processingTimes = new HashMap<>();
            this.metadata = new HashMap<>();
        }
        
        public void addDescriptor(String extractorName, FeatureDescriptor descriptor) {
            descriptors.put(extractorName, descriptor);
        }
        
        public FeatureDescriptor getDescriptor(String extractorName) {
            return descriptors.get(extractorName);
        }
        
        public Map<String, FeatureDescriptor> getAllDescriptors() {
            return descriptors;
        }
        
        public void addProcessingTime(String extractorName, double timeMs) {
            processingTimes.put(extractorName, timeMs);
        }
        
        public Map<String, Double> getProcessingTimes() { return processingTimes; }
        
        public void addMetadata(String key, Object value) { metadata.put(key, value); }
        
        public Map<String, Object> getMetadata() { return metadata; }
        
        public long getTimestamp() { return timestamp; }
        
        public double getTotalProcessingTime() {
            return processingTimes.values().stream().mapToDouble(Double::doubleValue).sum();
        }
    }
}