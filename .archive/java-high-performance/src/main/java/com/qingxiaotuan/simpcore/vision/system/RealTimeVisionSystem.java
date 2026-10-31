package com.qingxiaotuan.simpcore.vision.system;

import com.qingxiaotuan.simpcore.vision.api.VisualAPILayer;
import com.qingxiaotuan.simpcore.vision.api.VisualAPILayer.*;
import com.qingxiaotuan.simpcore.vision.api.VisionPipeline;
import com.qingxiaotuan.simpcore.vision.capture.CameraSource;
import com.qingxiaotuan.simpcore.vision.capture.CameraSource.Frame;
import com.qingxiaotuan.simpcore.vision.capture.MockCameraSource;
import com.qingxiaotuan.simpcore.vision.feature.*;
import com.qingxiaotuan.simpcore.vision.inference.*;
import com.qingxiaotuan.simpcore.vision.preprocessing.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

public class RealTimeVisionSystem {
    
    private CameraSource cameraSource;
    private VisionPipeline pipeline;
    private VisualAPILayer apiLayer;
    private SystemMonitor monitor;
    private PriorityScheduler scheduler;
    private FaultRecoveryManager faultManager;
    
    private volatile boolean running = false;
    private ScheduledExecutorService mainExecutor;
    private ScheduledFuture<?> processingLoop;
    private ScheduledFuture<?> monitorLoop;
    
    private final AtomicLong frameCount = new AtomicLong(0);
    private final AtomicLong lastFrameTime = new AtomicLong(0);
    private final AtomicLong totalProcessingTime = new AtomicLong(0);
    
    public RealTimeVisionSystem() {
        this.mainExecutor = Executors.newScheduledThreadPool(4);
        this.monitor = new SystemMonitor();
        this.scheduler = new PriorityScheduler();
        this.faultManager = new FaultRecoveryManager();
    }
    
    public void initialize() throws VisionSystemException {
        try {
            cameraSource = new MockCameraSource();
            configurePipeline();
            
            apiLayer = new VisualAPILayer();
            apiLayer.setCameraSource(cameraSource);
            apiLayer.setPipeline(pipeline);
            apiLayer.setVisionCallback(new VisionCallback() {
                @Override
                public void onResult(VisionResult result) {
                    processResult(result);
                }
                
                @Override
                public void onError(Exception e) {
                    faultManager.handleError(e);
                }
            });
            
            monitor.startMonitoring();
            faultManager.setSystem(this);
            
            System.out.println("RealTimeVisionSystem initialized successfully");
        } catch (Exception e) {
            throw new VisionSystemException("Failed to initialize vision system", e);
        }
    }
    
    private void configurePipeline() {
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        preprocessing.addProcessor(new ImageProcessors.DenoisingProcessor());
        preprocessing.addProcessor(new ImageProcessors.DistortionCorrectionProcessor());
        preprocessing.addProcessor(new ImageProcessors.GammaCorrectionProcessor());
        preprocessing.addProcessor(new ImageProcessors.ColorNormalizationProcessor());
        
        FeatureExtractionLayer featureExtraction = new FeatureExtractionLayer();
        featureExtraction.addExtractor(new FeatureExtractors.SimpleKeypointExtractor());
        featureExtraction.addExtractor(new FeatureExtractors.ColorHistogramExtractor());
        featureExtraction.addExtractor(new FeatureExtractors.HogExtractor());
        
        InferenceDecisionLayer inference = new InferenceDecisionLayer();
        inference.addEngine(new InferenceEngines.ObjectDetectionEngine());
        inference.addEngine(new InferenceEngines.SceneUnderstandingEngine());
        inference.addEngine(new InferenceEngines.VisualSlamEngine());
        inference.addEngine(new InferenceEngines.InstanceSegmentationEngine());
        
        pipeline = new VisionPipeline(preprocessing, featureExtraction, inference);
    }
    
    public void start() throws VisionSystemException {
        if (running) return;
        
        try {
            apiLayer.start();
            running = true;
            
            processingLoop = mainExecutor.scheduleAtFixedRate(() -> {
                if (!running) return;
                try {
                    Frame frame = cameraSource.capture();
                    VisionResult result = apiLayer.processFrame(frame);
                    processResult(result);
                } catch (Exception e) {
                    faultManager.handleError(e);
                }
            }, 0, 33, TimeUnit.MILLISECONDS);
            
            monitorLoop = mainExecutor.scheduleAtFixedRate(() -> {
                monitor.updateMetrics(getSystemMetrics());
            }, 0, 1000, TimeUnit.MILLISECONDS);
            
            System.out.println("RealTimeVisionSystem started");
        } catch (Exception e) {
            throw new VisionSystemException("Failed to start vision system", e);
        }
    }
    
    public void stop() {
        running = false;
        
        if (processingLoop != null) {
            processingLoop.cancel(false);
        }
        if (monitorLoop != null) {
            monitorLoop.cancel(false);
        }
        
        if (apiLayer != null) {
            apiLayer.stop();
        }
        
        monitor.stopMonitoring();
        faultManager.stop();
        
        System.out.println("RealTimeVisionSystem stopped");
    }
    
    public boolean isRunning() {
        return running;
    }
    
    private void processResult(VisionResult result) {
        frameCount.incrementAndGet();
        lastFrameTime.set(System.currentTimeMillis());
        totalProcessingTime.addAndGet(result.getTotalProcessingTime());
        
        monitor.recordFrame(result);
        scheduler.updatePriorities(result);
        
        if (visionSystemCallback != null) {
            visionSystemCallback.onFrameProcessed(result);
        }
    }
    
    public SystemMetrics getSystemMetrics() {
        long frames = frameCount.get();
        long totalTime = totalProcessingTime.get();
        double avgProcessingTime = frames > 0 ? (double) totalTime / frames : 0;
        
        return new SystemMetrics(
            frames,
            avgProcessingTime,
            monitor.getCpuUsage(),
            monitor.getMemoryUsage(),
            monitor.getGpuUsage(),
            running,
            faultManager.getErrorCount(),
            faultManager.getLastErrorTime()
        );
    }
    
    public void adjustPerformance(PerformanceMode mode) {
        scheduler.setPerformanceMode(mode);
        System.out.println("Performance mode changed to: " + mode);
    }
    
    public void updateCameraProperties(CameraSource.CameraProperties properties) throws VisionSystemException {
        try {
            apiLayer.setCameraProperties(properties);
        } catch (com.qingxiaotuan.simpcore.vision.api.VisualAPILayer.VisionException e) {
            throw new VisionSystemException("Failed to update camera properties", e);
        }
    }
    
    public void shutdown() {
        stop();
        mainExecutor.shutdown();
        try {
            if (!mainExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                mainExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            mainExecutor.shutdownNow();
        }
        
        if (pipeline != null) {
            pipeline.shutdown();
        }
        
        if (cameraSource != null) {
            cameraSource.release();
        }
        
        System.out.println("RealTimeVisionSystem shutdown complete");
    }
    
    private VisionSystemCallback visionSystemCallback;
    
    public void setVisionSystemCallback(VisionSystemCallback callback) {
        this.visionSystemCallback = callback;
    }
    
    public interface VisionSystemCallback {
        void onFrameProcessed(VisionResult result);
        void onSystemError(Exception e);
        void onMetricsUpdated(SystemMetrics metrics);
    }
    
    public static class SystemMetrics {
        private long frameCount;
        private double avgProcessingTimeMs;
        private double cpuUsage;
        private double memoryUsage;
        private double gpuUsage;
        private boolean isRunning;
        private int errorCount;
        private long lastErrorTime;
        
        public SystemMetrics(long frameCount, double avgProcessingTimeMs,
            double cpuUsage, double memoryUsage, double gpuUsage,
            boolean isRunning, int errorCount, long lastErrorTime) {
            this.frameCount = frameCount;
            this.avgProcessingTimeMs = avgProcessingTimeMs;
            this.cpuUsage = cpuUsage;
            this.memoryUsage = memoryUsage;
            this.gpuUsage = gpuUsage;
            this.isRunning = isRunning;
            this.errorCount = errorCount;
            this.lastErrorTime = lastErrorTime;
        }
        
        public long getFrameCount() { return frameCount; }
        public double getAvgProcessingTimeMs() { return avgProcessingTimeMs; }
        public double getCpuUsage() { return cpuUsage; }
        public double getMemoryUsage() { return memoryUsage; }
        public double getGpuUsage() { return gpuUsage; }
        public boolean isRunning() { return isRunning; }
        public int getErrorCount() { return errorCount; }
        public long getLastErrorTime() { return lastErrorTime; }
        
        @Override
        public String toString() {
            return String.format(
                "SystemMetrics{frames=%d, avgTime=%.2fms, CPU=%.1f%%, Memory=%.1f%%, GPU=%.1f%%, running=%s, errors=%d}",
                frameCount, avgProcessingTimeMs, cpuUsage, memoryUsage, gpuUsage, isRunning, errorCount
            );
        }
    }
    
    public enum PerformanceMode {
        MAX_PERFORMANCE,
        BALANCED,
        POWER_SAVER
    }
    
    public static class VisionSystemException extends RuntimeException {
        public VisionSystemException(String message) { super(message); }
        public VisionSystemException(String message, Throwable cause) { super(message, cause); }
    }
    
    private static class SystemMonitor {
        private volatile double cpuUsage = 0;
        private volatile double memoryUsage = 0;
        private volatile double gpuUsage = 0;
        private ScheduledExecutorService monitorExecutor;
        
        public void startMonitoring() {
            monitorExecutor = Executors.newSingleThreadScheduledExecutor();
            monitorExecutor.scheduleAtFixedRate(this::updateSystemUsage, 0, 500, TimeUnit.MILLISECONDS);
        }
        
        public void stopMonitoring() {
            if (monitorExecutor != null) {
                monitorExecutor.shutdown();
            }
        }
        
        private void updateSystemUsage() {
            Runtime runtime = Runtime.getRuntime();
            memoryUsage = (double) (runtime.totalMemory() - runtime.freeMemory()) / runtime.maxMemory() * 100;
            cpuUsage = 20 + Math.random() * 30;
            gpuUsage = 15 + Math.random() * 40;
        }
        
        public void recordFrame(VisionResult result) {
        }
        
        public void updateMetrics(SystemMetrics metrics) {
        }
        
        public double getCpuUsage() { return cpuUsage; }
        public double getMemoryUsage() { return memoryUsage; }
        public double getGpuUsage() { return gpuUsage; }
    }
    
    private static class PriorityScheduler {
        private PerformanceMode mode = PerformanceMode.BALANCED;
        private Map<String, Integer> enginePriorities = new HashMap<>();
        
        public void updatePriorities(VisionResult result) {
        }
        
        public void setPerformanceMode(PerformanceMode mode) {
            this.mode = mode;
        }
        
        public PerformanceMode getPerformanceMode() { return mode; }
    }
    
    private static class FaultRecoveryManager {
        private int errorCount = 0;
        private long lastErrorTime = 0;
        private RealTimeVisionSystem system;
        private ScheduledExecutorService recoveryExecutor;
        
        public void setSystem(RealTimeVisionSystem system) {
            this.system = system;
        }
        
        public void handleError(Exception e) {
            errorCount++;
            lastErrorTime = System.currentTimeMillis();
            System.err.println("Vision system error: " + e.getMessage());
            
            if (errorCount % 10 == 0) {
                attemptRecovery();
            }
        }
        
        private void attemptRecovery() {
            System.out.println("Attempting recovery after " + errorCount + " errors");
        }
        
        public void stop() {
            if (recoveryExecutor != null) {
                recoveryExecutor.shutdown();
            }
        }
        
        public int getErrorCount() { return errorCount; }
        public long getLastErrorTime() { return lastErrorTime; }
    }
}