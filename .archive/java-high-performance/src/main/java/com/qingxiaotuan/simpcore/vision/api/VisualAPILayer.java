package com.qingxiaotuan.simpcore.vision.api;

import com.qingxiaotuan.simpcore.vision.capture.CameraSource;
import com.qingxiaotuan.simpcore.vision.capture.CameraSource.*;
import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer.FeatureResult;
import com.qingxiaotuan.simpcore.vision.inference.InferenceDecisionLayer.InferenceResult;
import com.qingxiaotuan.simpcore.vision.inference.InferenceEngines.*;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.ProcessedFrame;
import java.util.*;
import java.util.concurrent.*;

public class VisualAPILayer {
    
    private CameraSource cameraSource;
    private VisionPipeline pipeline;
    private ExecutorService callbackExecutor;
    private VisionCallback visionCallback;
    private volatile boolean running = false;
    private ScheduledFuture<?> processingFuture;
    
    public VisualAPILayer() {
        this.callbackExecutor = Executors.newSingleThreadExecutor();
    }
    
    public void setCameraSource(CameraSource cameraSource) {
        this.cameraSource = cameraSource;
    }
    
    public void setPipeline(VisionPipeline pipeline) {
        this.pipeline = pipeline;
    }
    
    public void setVisionCallback(VisionCallback callback) {
        this.visionCallback = callback;
    }
    
    public void start() throws VisionException {
        if (running) return;
        if (cameraSource == null) {
            throw new VisionException("Camera source not set");
        }
        if (pipeline == null) {
            throw new VisionException("Vision pipeline not set");
        }
        
        running = true;
        
        try {
            cameraSource.initialize();
            cameraSource.setCallback(new FrameCallback() {
                @Override
                public void onFrame(Frame frame) {
                    if (!running) return;
                    processFrameAsync(frame);
                }
                
                @Override
                public void onError(CameraException e) {
                    if (visionCallback != null) {
                        visionCallback.onError(e);
                    }
                }
            });
            cameraSource.start();
        } catch (CameraException e) {
            throw new VisionException("Failed to start camera", e);
        }
        
        System.out.println("VisualAPILayer started");
    }
    
    public void stop() {
        running = false;
        
        if (processingFuture != null) {
            processingFuture.cancel(false);
        }
        
        if (cameraSource != null) {
            cameraSource.stop();
        }
        
        System.out.println("VisualAPILayer stopped");
    }
    
    public boolean isRunning() {
        return running && cameraSource != null && cameraSource.isRunning();
    }
    
    public VisionResult processFrame(Frame frame) {
        if (pipeline == null) {
            throw new VisionException("Vision pipeline not set");
        }
        
        long startTime = System.currentTimeMillis();
        
        ProcessedFrame processedFrame = pipeline.preprocess(frame);
        long preprocessTime = System.currentTimeMillis() - startTime;
        
        FeatureResult features = pipeline.extractFeatures(processedFrame);
        long featureTime = System.currentTimeMillis() - startTime - preprocessTime;
        
        InferenceResult inference = pipeline.infer(processedFrame, features);
        long inferenceTime = System.currentTimeMillis() - startTime - preprocessTime - featureTime;
        
        VisionResult result = new VisionResult(
            frame.getTimestamp(),
            frame.getWidth(),
            frame.getHeight(),
            processedFrame,
            features,
            inference
        );
        
        result.setProcessingTime(VisionResult.Stage.PREPROCESS, preprocessTime);
        result.setProcessingTime(VisionResult.Stage.FEATURE_EXTRACTION, featureTime);
        result.setProcessingTime(VisionResult.Stage.INFERENCE, inferenceTime);
        result.setTotalProcessingTime(System.currentTimeMillis() - startTime);
        
        return result;
    }
    
    private void processFrameAsync(Frame frame) {
        callbackExecutor.submit(() -> {
            try {
                VisionResult result = processFrame(frame);
                if (visionCallback != null) {
                    visionCallback.onResult(result);
                }
            } catch (Exception e) {
                if (visionCallback != null) {
                    visionCallback.onError(e);
                }
            }
        });
    }
    
    public List<DetectionResult> getDetections(VisionResult result) {
        List<DetectionResult> detections = new ArrayList<>();
        
        InferenceResult inference = result.getInferenceResult();
        if (inference == null) return detections;
        
        var objectOutput = inference.getOutput("ObjectDetectionEngine");
        if (objectOutput != null && objectOutput.isSuccessful()) {
            for (Object obj : objectOutput.getResults()) {
                if (obj instanceof ObjectDetectionEngine.Detection) {
                    ObjectDetectionEngine.Detection det = (ObjectDetectionEngine.Detection) obj;
                    detections.add(new DetectionResult(
                        det.getClassName(),
                        det.getConfidence(),
                        det.getBoundingBox().getX1(),
                        det.getBoundingBox().getY1(),
                        det.getBoundingBox().getWidth(),
                        det.getBoundingBox().getHeight(),
                        det.getTimestamp()
                    ));
                }
            }
        }
        
        return detections;
    }
    
    public SceneResult getScene(VisionResult result) {
        InferenceResult inference = result.getInferenceResult();
        if (inference == null) return null;
        
        var sceneOutput = inference.getOutput("SceneUnderstandingEngine");
        if (sceneOutput != null && sceneOutput.isSuccessful()) {
            for (Object obj : sceneOutput.getResults()) {
                if (obj instanceof SceneUnderstandingEngine.SceneResult) {
                    SceneUnderstandingEngine.SceneResult sr = 
                        (SceneUnderstandingEngine.SceneResult) obj;
                    return new SceneResult(sr.getSceneType(), sr.getConfidence(), sr.getKeywords());
                }
            }
        }
        
        return null;
    }
    
    public PoseResult getPose(VisionResult result) {
        InferenceResult inference = result.getInferenceResult();
        if (inference == null) return null;
        
        var slamOutput = inference.getOutput("VisualSlamEngine");
        if (slamOutput != null && slamOutput.isSuccessful()) {
            for (Object obj : slamOutput.getResults()) {
                if (obj instanceof VisualSlamEngine.SlamResult) {
                    VisualSlamEngine.SlamResult sr = (VisualSlamEngine.SlamResult) obj;
                    float[] pos = sr.getPose().getPosition();
                    float[] rot = sr.getPose().getRotation();
                    return new PoseResult(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2],
                        sr.getTrajectoryLength(), sr.getMapPointCount());
                }
            }
        }
        
        return null;
    }
    
    public List<InstanceResult> getInstances(VisionResult result) {
        List<InstanceResult> instances = new ArrayList<>();
        
        InferenceResult inference = result.getInferenceResult();
        if (inference == null) return instances;
        
        var segOutput = inference.getOutput("InstanceSegmentationEngine");
        if (segOutput != null && segOutput.isSuccessful()) {
            for (Object obj : segOutput.getResults()) {
                if (obj instanceof InstanceSegmentationEngine.InstanceSegment) {
                    InstanceSegmentationEngine.InstanceSegment seg = 
                        (InstanceSegmentationEngine.InstanceSegment) obj;
                    instances.add(new InstanceResult(
                        seg.getClassName(),
                        seg.getConfidence(),
                        seg.getBoundingBox().getX1(),
                        seg.getBoundingBox().getY1(),
                        seg.getBoundingBox().getWidth(),
                        seg.getBoundingBox().getHeight(),
                        seg.getMask(),
                        seg.getTimestamp()
                    ));
                }
            }
        }
        
        return instances;
    }
    
    public void shutdown() {
        stop();
        callbackExecutor.shutdown();
        try {
            if (!callbackExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                callbackExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            callbackExecutor.shutdownNow();
        }
    }
    
    public CameraProperties getCameraProperties() {
        if (cameraSource == null) return null;
        return cameraSource.getProperties();
    }
    
    public void setCameraProperties(CameraProperties properties) throws VisionException {
        if (cameraSource == null) {
            throw new VisionException("Camera source not set");
        }
        try {
            cameraSource.setProperties(properties);
        } catch (CameraException e) {
            throw new VisionException("Failed to set camera properties", e);
        }
    }
    
    public interface VisionCallback {
        void onResult(VisionResult result);
        void onError(Exception e);
    }
    
    public static class VisionException extends RuntimeException {
        public VisionException(String message) { super(message); }
        public VisionException(String message, Throwable cause) { super(message, cause); }
    }
    
    public static class VisionResult {
        private long timestamp;
        private int width;
        private int height;
        private ProcessedFrame processedFrame;
        private FeatureResult featureResult;
        private InferenceResult inferenceResult;
        private Map<Stage, Long> processingTimes;
        private long totalProcessingTime;
        
        public enum Stage {
            PREPROCESS,
            FEATURE_EXTRACTION,
            INFERENCE
        }
        
        public VisionResult(long timestamp, int width, int height, 
            ProcessedFrame processedFrame, FeatureResult featureResult, InferenceResult inferenceResult) {
            this.timestamp = timestamp;
            this.width = width;
            this.height = height;
            this.processedFrame = processedFrame;
            this.featureResult = featureResult;
            this.inferenceResult = inferenceResult;
            this.processingTimes = new EnumMap<>(Stage.class);
        }
        
        public long getTimestamp() { return timestamp; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public ProcessedFrame getProcessedFrame() { return processedFrame; }
        public FeatureResult getFeatureResult() { return featureResult; }
        public InferenceResult getInferenceResult() { return inferenceResult; }
        
        public Map<Stage, Long> getProcessingTimes() { return processingTimes; }
        public void setProcessingTime(Stage stage, long timeMs) { 
            processingTimes.put(stage, timeMs); 
        }
        
        public long getTotalProcessingTime() { return totalProcessingTime; }
        public void setTotalProcessingTime(long totalProcessingTime) { 
            this.totalProcessingTime = totalProcessingTime; 
        }
        
        public boolean hasDetections() {
            return inferenceResult != null && inferenceResult.hasDetections();
        }
    }
    
    public static class DetectionResult {
        private String className;
        private float confidence;
        private float x, y, width, height;
        private long timestamp;
        
        public DetectionResult(String className, float confidence, 
            float x, float y, float width, float height, long timestamp) {
            this.className = className;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.timestamp = timestamp;
        }
        
        public String getClassName() { return className; }
        public float getConfidence() { return confidence; }
        public float getX() { return x; }
        public float getY() { return y; }
        public float getWidth() { return width; }
        public float getHeight() { return height; }
        public long getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format("DetectionResult{%s, confidence=%.2f, x=%.1f, y=%.1f, w=%.1f, h=%.1f}",
                className, confidence, x, y, width, height);
        }
    }
    
    public static class SceneResult {
        private String sceneType;
        private float confidence;
        private List<String> keywords;
        
        public SceneResult(String sceneType, float confidence, List<String> keywords) {
            this.sceneType = sceneType;
            this.confidence = confidence;
            this.keywords = keywords;
        }
        
        public String getSceneType() { return sceneType; }
        public float getConfidence() { return confidence; }
        public List<String> getKeywords() { return keywords; }
        
        @Override
        public String toString() {
            return String.format("SceneResult{%s, confidence=%.2f, keywords=%s}",
                sceneType, confidence, keywords);
        }
    }
    
    public static class PoseResult {
        private float x, y, z;
        private float rx, ry, rz;
        private int trajectoryLength;
        private int mapPointCount;
        
        public PoseResult(float x, float y, float z, float rx, float ry, float rz,
            int trajectoryLength, int mapPointCount) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.rx = rx;
            this.ry = ry;
            this.rz = rz;
            this.trajectoryLength = trajectoryLength;
            this.mapPointCount = mapPointCount;
        }
        
        public float getX() { return x; }
        public float getY() { return y; }
        public float getZ() { return z; }
        public float getRx() { return rx; }
        public float getRy() { return ry; }
        public float getRz() { return rz; }
        public int getTrajectoryLength() { return trajectoryLength; }
        public int getMapPointCount() { return mapPointCount; }
        
        @Override
        public String toString() {
            return String.format("PoseResult{x=%.2f, y=%.2f, z=%.2f, rx=%.2f, ry=%.2f, rz=%.2f, traj=%d, points=%d}",
                x, y, z, rx, ry, rz, trajectoryLength, mapPointCount);
        }
    }
    
    public static class InstanceResult {
        private String className;
        private float confidence;
        private float x, y, width, height;
        private byte[] mask;
        private long timestamp;
        
        public InstanceResult(String className, float confidence, 
            float x, float y, float width, float height, byte[] mask, long timestamp) {
            this.className = className;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.mask = mask;
            this.timestamp = timestamp;
        }
        
        public String getClassName() { return className; }
        public float getConfidence() { return confidence; }
        public float getX() { return x; }
        public float getY() { return y; }
        public float getWidth() { return width; }
        public float getHeight() { return height; }
        public byte[] getMask() { return mask; }
        public long getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format("InstanceResult{%s, confidence=%.2f, x=%.1f, y=%.1f, w=%.1f, h=%.1f}",
                className, confidence, x, y, width, height);
        }
    }
}