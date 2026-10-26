package com.qingxiaotuan.simpcore.vision.preprocessing;

import com.qingxiaotuan.simpcore.vision.capture.CameraSource.Frame;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import java.util.*;
import java.util.concurrent.*;

public class PreprocessingPipeline {
    
    private final List<ImageProcessor> processors;
    private final ExecutorService executor;
    private final int parallelism;
    private volatile boolean running = true;
    
    public PreprocessingPipeline() {
        this.processors = new ArrayList<>();
        this.parallelism = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(parallelism);
    }
    
    public PreprocessingPipeline(int parallelism) {
        this.processors = new ArrayList<>();
        this.parallelism = parallelism;
        this.executor = Executors.newFixedThreadPool(parallelism);
    }
    
    public void addProcessor(ImageProcessor processor) {
        processors.add(processor);
    }
    
    public void removeProcessor(ImageProcessor processor) {
        processors.remove(processor);
    }
    
    public void clearProcessors() {
        processors.clear();
    }
    
    public ProcessedFrame process(Frame frame) {
        ProcessedFrame processed = new ProcessedFrame(frame);
        
        for (ImageProcessor processor : processors) {
            if (!running) break;
            long startTime = System.nanoTime();
            processed = processor.process(processed);
            long endTime = System.nanoTime();
            
            processed.addProcessingTime(processor.getClass().getSimpleName(), 
                (endTime - startTime) / 1_000_000.0);
        }
        
        return processed;
    }
    
    public Future<ProcessedFrame> processAsync(Frame frame) {
        return executor.submit(() -> process(frame));
    }
    
    public List<Future<ProcessedFrame>> processBatch(List<Frame> frames) {
        List<Future<ProcessedFrame>> futures = new ArrayList<>();
        for (Frame frame : frames) {
            futures.add(processAsync(frame));
        }
        return futures;
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
    
    public int getProcessorCount() {
        return processors.size();
    }
    
    public interface ImageProcessor {
        ProcessedFrame process(ProcessedFrame frame);
        String getName();
        boolean isEnabled();
        void setEnabled(boolean enabled);
    }
    
    public static class ProcessedFrame {
        private byte[] data;
        private int width;
        private int height;
        private int stride;
        private String format;
        private long timestamp;
        private float[] intrinsics;
        private float[] extrinsics;
        private byte[] depthData;
        private Map<String, Double> processingTimes;
        private Map<String, Object> metadata;
        
        public ProcessedFrame(Frame frame) {
            this.data = Arrays.copyOf(frame.getData(), frame.getBufferSize());
            this.width = frame.getWidth();
            this.height = frame.getHeight();
            this.stride = frame.getStride();
            this.format = frame.getFormat();
            this.timestamp = frame.getTimestamp();
            this.intrinsics = frame.getIntrinsics() != null ? 
                Arrays.copyOf(frame.getIntrinsics(), frame.getIntrinsics().length) : null;
            this.extrinsics = frame.getExtrinsics() != null ? 
                Arrays.copyOf(frame.getExtrinsics(), frame.getExtrinsics().length) : null;
            this.depthData = frame.getDepthData() != null ? 
                Arrays.copyOf(frame.getDepthData(), frame.getDepthData().length) : null;
            this.processingTimes = new HashMap<>();
            this.metadata = new HashMap<>();
        }
        
        public ProcessedFrame(byte[] data, int width, int height, String format) {
            this.data = data;
            this.width = width;
            this.height = height;
            this.format = format;
            this.stride = width * (format.equals("RGB") || format.equals("BGR") ? 3 : 
                                  format.equals("RGBA") || format.equals("BGRA") ? 4 : 1);
            this.timestamp = System.currentTimeMillis();
            this.processingTimes = new HashMap<>();
            this.metadata = new HashMap<>();
        }
        
        public byte[] getData() { return data; }
        public void setData(byte[] data) { this.data = data; }
        
        public int getWidth() { return width; }
        public void setWidth(int width) { this.width = width; }
        
        public int getHeight() { return height; }
        public void setHeight(int height) { this.height = height; }
        
        public int getStride() { return stride; }
        public void setStride(int stride) { this.stride = stride; }
        
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
        
        public float[] getIntrinsics() { return intrinsics; }
        public void setIntrinsics(float[] intrinsics) { this.intrinsics = intrinsics; }
        
        public float[] getExtrinsics() { return extrinsics; }
        public void setExtrinsics(float[] extrinsics) { this.extrinsics = extrinsics; }
        
        public byte[] getDepthData() { return depthData; }
        public void setDepthData(byte[] depthData) { this.depthData = depthData; }
        
        public Map<String, Double> getProcessingTimes() { return processingTimes; }
        public void addProcessingTime(String processorName, double timeMs) {
            processingTimes.put(processorName, timeMs);
        }
        
        public Map<String, Object> getMetadata() { return metadata; }
        public void addMetadata(String key, Object value) { metadata.put(key, value); }
        
        public int getBufferSize() {
            return width * height * (format.equals("RGB") || format.equals("BGR") ? 3 : 
                                   format.equals("RGBA") || format.equals("BGRA") ? 4 : 1);
        }
    }
}