package com.qingxiaotuan.simpcore.vision.api;

import com.qingxiaotuan.simpcore.vision.capture.CameraSource.Frame;
import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer;
import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer.FeatureResult;
import com.qingxiaotuan.simpcore.vision.inference.InferenceDecisionLayer;
import com.qingxiaotuan.simpcore.vision.inference.InferenceDecisionLayer.InferenceResult;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.ProcessedFrame;

public class VisionPipeline {
    
    private PreprocessingPipeline preprocessingPipeline;
    private FeatureExtractionLayer featureExtractionLayer;
    private InferenceDecisionLayer inferenceDecisionLayer;
    
    public VisionPipeline() {
        this.preprocessingPipeline = new PreprocessingPipeline();
        this.featureExtractionLayer = new FeatureExtractionLayer();
        this.inferenceDecisionLayer = new InferenceDecisionLayer();
    }
    
    public VisionPipeline(PreprocessingPipeline preprocessingPipeline,
        FeatureExtractionLayer featureExtractionLayer,
        InferenceDecisionLayer inferenceDecisionLayer) {
        this.preprocessingPipeline = preprocessingPipeline;
        this.featureExtractionLayer = featureExtractionLayer;
        this.inferenceDecisionLayer = inferenceDecisionLayer;
    }
    
    public ProcessedFrame preprocess(Frame frame) {
        return preprocessingPipeline.process(frame);
    }
    
    public FeatureResult extractFeatures(ProcessedFrame frame) {
        return featureExtractionLayer.extract(frame);
    }
    
    public InferenceResult infer(ProcessedFrame frame, FeatureResult features) {
        return inferenceDecisionLayer.infer(frame, features);
    }
    
    public PreprocessingPipeline getPreprocessingPipeline() {
        return preprocessingPipeline;
    }
    
    public void setPreprocessingPipeline(PreprocessingPipeline preprocessingPipeline) {
        this.preprocessingPipeline = preprocessingPipeline;
    }
    
    public FeatureExtractionLayer getFeatureExtractionLayer() {
        return featureExtractionLayer;
    }
    
    public void setFeatureExtractionLayer(FeatureExtractionLayer featureExtractionLayer) {
        this.featureExtractionLayer = featureExtractionLayer;
    }
    
    public InferenceDecisionLayer getInferenceDecisionLayer() {
        return inferenceDecisionLayer;
    }
    
    public void setInferenceDecisionLayer(InferenceDecisionLayer inferenceDecisionLayer) {
        this.inferenceDecisionLayer = inferenceDecisionLayer;
    }
    
    public void shutdown() {
        preprocessingPipeline.shutdown();
        featureExtractionLayer.shutdown();
        inferenceDecisionLayer.shutdown();
    }
    
    public int getPreprocessorCount() {
        return preprocessingPipeline.getProcessorCount();
    }
    
    public int getExtractorCount() {
        return featureExtractionLayer.getExtractorCount();
    }
    
    public int getEngineCount() {
        return inferenceDecisionLayer.getEngineCount();
    }
}