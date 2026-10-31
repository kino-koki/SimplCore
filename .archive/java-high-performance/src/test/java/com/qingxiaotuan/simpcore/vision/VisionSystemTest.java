package com.qingxiaotuan.simpcore.vision;

import com.qingxiaotuan.simpcore.vision.api.*;
import com.qingxiaotuan.simpcore.vision.api.VisualAPILayer.*;
import com.qingxiaotuan.simpcore.vision.capture.*;
import com.qingxiaotuan.simpcore.vision.capture.CameraSource.*;
import com.qingxiaotuan.simpcore.vision.feature.*;
import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer.*;
import com.qingxiaotuan.simpcore.vision.inference.*;
import com.qingxiaotuan.simpcore.vision.preprocessing.*;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.*;
import com.qingxiaotuan.simpcore.vision.system.RealTimeVisionSystem;
import org.junit.jupiter.api.*;
import java.util.*;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class VisionSystemTest {
    
    @Test
    @Order(1)
    @DisplayName("CameraSource接口测试")
    public void testCameraSource() throws CameraException {
        CameraSource camera = new MockCameraSource();
        
        camera.initialize();
        assertFalse(camera.isRunning());
        
        camera.start();
        assertTrue(camera.isRunning());
        
        Frame frame = camera.capture();
        assertNotNull(frame);
        assertTrue(frame.getWidth() > 0);
        assertTrue(frame.getHeight() > 0);
        assertNotNull(frame.getData());
        
        camera.stop();
        assertFalse(camera.isRunning());
        
        camera.release();
    }
    
    @Test
    @Order(2)
    @DisplayName("MockCameraSource属性测试")
    public void testMockCameraProperties() throws CameraException {
        CameraSource camera = new MockCameraSource();
        
        CameraProperties props = camera.getProperties();
        assertNotNull(props);
        assertEquals(640, props.getWidth());
        assertEquals(480, props.getHeight());
        assertEquals(30, props.getFps());
        
        List<CameraProperties> available = camera.getAvailableProperties();
        assertTrue(available.size() > 0);
        
        CameraProperties newProps = new CameraProperties();
        newProps.setWidth(1280);
        newProps.setHeight(720);
        newProps.setFps(30);
        
        camera.setProperties(newProps);
        CameraProperties updated = camera.getProperties();
        assertEquals(1280, updated.getWidth());
        assertEquals(720, updated.getHeight());
        
        camera.release();
    }
    
    @Test
    @Order(3)
    @DisplayName("图像预处理管道测试")
    public void testPreprocessingPipeline() throws CameraException {
        CameraSource camera = new MockCameraSource();
        camera.initialize();
        camera.start();
        
        Frame frame = camera.capture();
        
        PreprocessingPipeline pipeline = new PreprocessingPipeline();
        pipeline.addProcessor(new ImageProcessors.DenoisingProcessor());
        pipeline.addProcessor(new ImageProcessors.GammaCorrectionProcessor());
        pipeline.addProcessor(new ImageProcessors.ColorNormalizationProcessor());
        
        ProcessedFrame processed = pipeline.process(frame);
        
        assertNotNull(processed);
        assertEquals(frame.getWidth(), processed.getWidth());
        assertEquals(frame.getHeight(), processed.getHeight());
        assertNotNull(processed.getData());
        assertTrue(processed.getProcessingTimes().size() > 0);
        
        camera.release();
        pipeline.shutdown();
    }
    
    @Test
    @Order(4)
    @DisplayName("图像处理器功能测试")
    public void testImageProcessors() throws CameraException {
        CameraSource camera = new MockCameraSource();
        camera.initialize();
        camera.start();
        
        Frame frame = camera.capture();
        ProcessedFrame processed = new ProcessedFrame(frame);
        
        ImageProcessors.DenoisingProcessor denoiser = new ImageProcessors.DenoisingProcessor();
        processed = denoiser.process(processed);
        assertTrue(processed.getMetadata().containsKey("denoised"));
        
        ImageProcessors.EdgeEnhancementProcessor edgeEnhancer = new ImageProcessors.EdgeEnhancementProcessor();
        processed = edgeEnhancer.process(processed);
        assertTrue(processed.getMetadata().containsKey("edgeEnhanced"));
        
        ImageProcessors.GammaCorrectionProcessor gamma = new ImageProcessors.GammaCorrectionProcessor();
        gamma.setGamma(1.5f);
        processed = gamma.process(processed);
        assertTrue(processed.getMetadata().containsKey("gammaCorrected"));
        
        ImageProcessors.ResizeProcessor resizer = new ImageProcessors.ResizeProcessor(320, 240);
        processed = resizer.process(processed);
        assertEquals(320, processed.getWidth());
        assertEquals(240, processed.getHeight());
        
        camera.release();
    }
    
    @Test
    @Order(5)
    @DisplayName("特征提取层测试")
    public void testFeatureExtractionLayer() throws CameraException {
        CameraSource camera = new MockCameraSource();
        camera.initialize();
        camera.start();
        
        Frame frame = camera.capture();
        
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        ProcessedFrame processed = preprocessing.process(frame);
        
        FeatureExtractionLayer extractor = new FeatureExtractionLayer();
        extractor.addExtractor(new FeatureExtractors.SimpleKeypointExtractor());
        extractor.addExtractor(new FeatureExtractors.ColorHistogramExtractor());
        
        FeatureResult features = extractor.extract(processed);
        
        assertNotNull(features);
        assertTrue(features.getAllDescriptors().size() >= 2);
        
        FeatureDescriptor keypointDesc = features.getDescriptor("SimpleKeypointExtractor");
        assertNotNull(keypointDesc);
        assertTrue(keypointDesc.getKeypointCount() >= 0);
        
        FeatureDescriptor colorDesc = features.getDescriptor("ColorHistogramExtractor");
        assertNotNull(colorDesc);
        assertEquals(512, colorDesc.getDimension());
        
        camera.release();
        extractor.shutdown();
    }
    
    @Test
    @Order(6)
    @DisplayName("推理决策层测试")
    public void testInferenceDecisionLayer() throws CameraException {
        CameraSource camera = new MockCameraSource();
        camera.initialize();
        camera.start();
        
        Frame frame = camera.capture();
        
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        ProcessedFrame processed = preprocessing.process(frame);
        
        FeatureExtractionLayer featureExtraction = new FeatureExtractionLayer();
        featureExtraction.addExtractor(new FeatureExtractors.SimpleKeypointExtractor());
        FeatureResult features = featureExtraction.extract(processed);
        
        InferenceDecisionLayer inference = new InferenceDecisionLayer();
        inference.addEngine(new InferenceEngines.ObjectDetectionEngine());
        inference.addEngine(new InferenceEngines.SceneUnderstandingEngine());
        inference.addEngine(new InferenceEngines.VisualSlamEngine());
        
        InferenceDecisionLayer.InferenceResult result = inference.infer(processed, features);
        
        assertNotNull(result);
        assertTrue(result.getAllOutputs().size() >= 3);
        
        camera.release();
        featureExtraction.shutdown();
        inference.shutdown();
    }
    
    @Test
    @Order(7)
    @DisplayName("视觉API层测试")
    public void testVisualAPILayer() throws VisualAPILayer.VisionException, CameraException {
        CameraSource camera = new MockCameraSource();
        
        VisionPipeline pipeline = new VisionPipeline();
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        pipeline.setPreprocessingPipeline(preprocessing);
        
        FeatureExtractionLayer featureExtraction = new FeatureExtractionLayer();
        featureExtraction.addExtractor(new FeatureExtractors.SimpleKeypointExtractor());
        pipeline.setFeatureExtractionLayer(featureExtraction);
        
        InferenceDecisionLayer inference = new InferenceDecisionLayer();
        inference.addEngine(new InferenceEngines.ObjectDetectionEngine());
        inference.addEngine(new InferenceEngines.SceneUnderstandingEngine());
        pipeline.setInferenceDecisionLayer(inference);
        
        VisualAPILayer api = new VisualAPILayer();
        api.setCameraSource(camera);
        api.setPipeline(pipeline);
        
        api.start();
        assertTrue(api.isRunning());
        
        Frame frame = camera.capture();
        VisionResult result = api.processFrame(frame);
        
        assertNotNull(result);
        assertEquals(frame.getWidth(), result.getWidth());
        assertEquals(frame.getHeight(), result.getHeight());
        
        List<DetectionResult> detections = api.getDetections(result);
        assertNotNull(detections);
        
        SceneResult scene = api.getScene(result);
        assertNotNull(scene);
        
        api.stop();
        assertFalse(api.isRunning());
        
        api.shutdown();
        featureExtraction.shutdown();
        inference.shutdown();
        camera.release();
    }
    
    @Test
    @Order(8)
    @DisplayName("视觉API结果提取测试")
    public void testApiResultExtraction() throws VisualAPILayer.VisionException, CameraException {
        CameraSource camera = new MockCameraSource();
        
        VisionPipeline pipeline = new VisionPipeline();
        
        InferenceDecisionLayer inference = new InferenceDecisionLayer();
        inference.addEngine(new InferenceEngines.VisualSlamEngine());
        inference.addEngine(new InferenceEngines.InstanceSegmentationEngine());
        pipeline.setInferenceDecisionLayer(inference);
        
        VisualAPILayer api = new VisualAPILayer();
        api.setCameraSource(camera);
        api.setPipeline(pipeline);
        
        api.start();
        
        Frame frame = camera.capture();
        VisionResult result = api.processFrame(frame);
        
        PoseResult pose = api.getPose(result);
        assertNotNull(pose);
        
        List<InstanceResult> instances = api.getInstances(result);
        assertNotNull(instances);
        
        api.stop();
        api.shutdown();
        inference.shutdown();
        camera.release();
    }
    
    @Test
    @Order(9)
    @DisplayName("实时视觉系统初始化测试")
    public void testRealTimeVisionSystem() throws RealTimeVisionSystem.VisionSystemException {
        RealTimeVisionSystem system = new RealTimeVisionSystem();
        
        system.initialize();
        assertFalse(system.isRunning());
        
        system.start();
        assertTrue(system.isRunning());
        
        RealTimeVisionSystem.SystemMetrics metrics = system.getSystemMetrics();
        assertNotNull(metrics);
        assertTrue(metrics.isRunning());
        
        system.stop();
        assertFalse(system.isRunning());
        
        system.shutdown();
    }
    
    @Test
    @Order(10)
    @DisplayName("实时视觉系统性能模式测试")
    public void testPerformanceMode() throws RealTimeVisionSystem.VisionSystemException {
        RealTimeVisionSystem system = new RealTimeVisionSystem();
        
        system.initialize();
        system.start();
        
        system.adjustPerformance(RealTimeVisionSystem.PerformanceMode.MAX_PERFORMANCE);
        system.adjustPerformance(RealTimeVisionSystem.PerformanceMode.BALANCED);
        system.adjustPerformance(RealTimeVisionSystem.PerformanceMode.POWER_SAVER);
        
        system.stop();
        system.shutdown();
    }
    
    @Test
    @Order(11)
    @DisplayName("实时视觉系统指标监控测试")
    public void testSystemMetrics() throws RealTimeVisionSystem.VisionSystemException, InterruptedException {
        RealTimeVisionSystem system = new RealTimeVisionSystem();
        
        system.initialize();
        system.start();
        
        Thread.sleep(1500);
        
        RealTimeVisionSystem.SystemMetrics metrics = system.getSystemMetrics();
        assertNotNull(metrics);
        assertTrue(metrics.getFrameCount() > 0);
        assertTrue(metrics.getAvgProcessingTimeMs() >= 0);
        
        system.stop();
        system.shutdown();
    }
    
    @Test
    @Order(12)
    @DisplayName("全管线集成测试")
    public void testFullPipelineIntegration() throws CameraException, VisualAPILayer.VisionException {
        CameraSource camera = new MockCameraSource();
        
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        preprocessing.addProcessor(new ImageProcessors.DenoisingProcessor());
        preprocessing.addProcessor(new ImageProcessors.GammaCorrectionProcessor());
        
        FeatureExtractionLayer featureExtraction = new FeatureExtractionLayer();
        featureExtraction.addExtractor(new FeatureExtractors.SimpleKeypointExtractor());
        featureExtraction.addExtractor(new FeatureExtractors.HogExtractor());
        
        InferenceDecisionLayer inference = new InferenceDecisionLayer();
        inference.addEngine(new InferenceEngines.ObjectDetectionEngine());
        inference.addEngine(new InferenceEngines.SceneUnderstandingEngine());
        inference.addEngine(new InferenceEngines.VisualSlamEngine());
        inference.addEngine(new InferenceEngines.InstanceSegmentationEngine());
        
        VisionPipeline pipeline = new VisionPipeline(preprocessing, featureExtraction, inference);
        
        VisualAPILayer api = new VisualAPILayer();
        api.setCameraSource(camera);
        api.setPipeline(pipeline);
        
        api.start();
        
        Frame frame = camera.capture();
        VisionResult result = api.processFrame(frame);
        
        assertNotNull(result);
        assertTrue(result.getProcessingTimes().size() > 0);
        assertTrue(result.getTotalProcessingTime() >= 0);
        
        List<DetectionResult> detections = api.getDetections(result);
        SceneResult scene = api.getScene(result);
        PoseResult pose = api.getPose(result);
        
        assertNotNull(detections);
        assertNotNull(scene);
        assertNotNull(pose);
        
        api.stop();
        api.shutdown();
        pipeline.shutdown();
        camera.release();
    }
    
    @Test
    @Order(13)
    @DisplayName("异步处理测试")
    public void testAsyncProcessing() throws CameraException, ExecutionException, InterruptedException {
        CameraSource camera = new MockCameraSource();
        camera.initialize();
        camera.start();
        
        PreprocessingPipeline preprocessing = new PreprocessingPipeline();
        Future<ProcessedFrame> future = preprocessing.processAsync(camera.capture());
        
        ProcessedFrame processed = future.get(5, TimeUnit.SECONDS);
        assertNotNull(processed);
        
        camera.release();
        preprocessing.shutdown();
    }
    
    @Test
    @Order(14)
    @DisplayName("边界情况测试")
    public void testEdgeCases() {
        CameraSource camera = new MockCameraSource();
        
        assertThrows(CameraException.class, () -> {
            camera.capture();
        });
        
        VisualAPILayer api = new VisualAPILayer();
        assertThrows(VisualAPILayer.VisionException.class, () -> {
            api.start();
        });
        
        api.shutdown();
        camera.release();
    }
    
    @Test
    @Order(15)
    @DisplayName("资源释放测试")
    public void testResourceCleanup() throws CameraException, VisualAPILayer.VisionException {
        CameraSource camera = new MockCameraSource();
        
        VisionPipeline pipeline = new VisionPipeline();
        VisualAPILayer api = new VisualAPILayer();
        api.setCameraSource(camera);
        api.setPipeline(pipeline);
        
        api.start();
        api.stop();
        api.shutdown();
        
        pipeline.shutdown();
        camera.release();
        
        RealTimeVisionSystem system = new RealTimeVisionSystem();
        system.initialize();
        system.start();
        system.stop();
        system.shutdown();
    }
}