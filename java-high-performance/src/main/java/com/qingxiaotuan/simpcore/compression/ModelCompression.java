package com.qingxiaotuan.simpcore.compression;

import java.util.*;

public class ModelCompression {
    
    public static class CompressionResult {
        public final String compressedModelId;
        public final long originalSize;
        public final long compressedSize;
        public final float compressionRatio;
        public final float accuracyLoss;
        public final String method;
        
        public CompressionResult(String compressedModelId, long originalSize, long compressedSize, 
                                float compressionRatio, float accuracyLoss, String method) {
            this.compressedModelId = compressedModelId;
            this.originalSize = originalSize;
            this.compressedSize = compressedSize;
            this.compressionRatio = compressionRatio;
            this.accuracyLoss = accuracyLoss;
            this.method = method;
        }
    }
    
    public static class QuantizationConfig {
        public int bits;
        public boolean useGroupedQuantization;
        public int groupSize;
        public boolean enableActivationQuantization;
        
        public QuantizationConfig(int bits) {
            this.bits = bits;
            this.useGroupedQuantization = true;
            this.groupSize = 128;
            this.enableActivationQuantization = false;
        }
    }
    
    private final Map<String, Object> compressedModels;
    private volatile boolean enabled;
    
    public ModelCompression() {
        this.compressedModels = new HashMap<>();
        this.enabled = true;
    }
    
    public CompressionResult quantize(Object model, String modelId, QuantizationConfig config) {
        if (!enabled) {
            return new CompressionResult(modelId, 0, 0, 1.0f, 0.0f, "disabled");
        }
        
        long originalSize = estimateModelSize(model);
        float ratio = 32.0f / config.bits;
        long compressedSize = (long) (originalSize / ratio);
        float accuracyLoss = calculateAccuracyLoss(config.bits);
        
        compressedModels.put(modelId + "_quantized", model);
        
        return new CompressionResult(
            modelId + "_quantized",
            originalSize,
            compressedSize,
            ratio,
            accuracyLoss,
            "quantization_" + config.bits + "bit"
        );
    }
    
    public CompressionResult sparseify(Object model, String modelId, float sparsityRatio) {
        if (!enabled) {
            return new CompressionResult(modelId, 0, 0, 1.0f, 0.0f, "disabled");
        }
        
        long originalSize = estimateModelSize(model);
        long compressedSize = (long) (originalSize * (1 - sparsityRatio));
        float accuracyLoss = sparsityRatio * 0.05f;
        
        compressedModels.put(modelId + "_sparse", model);
        
        return new CompressionResult(
            modelId + "_sparse",
            originalSize,
            compressedSize,
            1.0f / (1 - sparsityRatio),
            accuracyLoss,
            "sparseify_" + (int)(sparsityRatio * 100) + "%"
        );
    }
    
    public CompressionResult distill(Object teacherModel, Object studentModel, String modelId) {
        if (!enabled) {
            return new CompressionResult(modelId, 0, 0, 1.0f, 0.0f, "disabled");
        }
        
        long teacherSize = estimateModelSize(teacherModel);
        long studentSize = estimateModelSize(studentModel);
        float ratio = (float) teacherSize / studentSize;
        float accuracyLoss = 0.02f;
        
        compressedModels.put(modelId + "_distilled", studentModel);
        
        return new CompressionResult(
            modelId + "_distilled",
            teacherSize,
            studentSize,
            ratio,
            accuracyLoss,
            "distillation"
        );
    }
    
    public CompressionResult applyMixedCompression(Object model, String modelId, 
                                                    QuantizationConfig qConfig, 
                                                    float sparsityRatio) {
        CompressionResult qResult = quantize(model, modelId, qConfig);
        CompressionResult sResult = sparseify(model, modelId + "_quantized", sparsityRatio);
        
        float combinedRatio = qResult.compressionRatio * sResult.compressionRatio;
        float combinedLoss = qResult.accuracyLoss + sResult.accuracyLoss;
        
        return new CompressionResult(
            modelId + "_mixed",
            qResult.originalSize,
            (long) (qResult.originalSize / combinedRatio),
            combinedRatio,
            combinedLoss,
            "mixed"
        );
    }
    
    public Object getCompressedModel(String compressedModelId) {
        return compressedModels.get(compressedModelId);
    }
    
    public Set<String> listCompressedModels() {
        return compressedModels.keySet();
    }
    
    public void removeCompressedModel(String compressedModelId) {
        compressedModels.remove(compressedModelId);
    }
    
    private long estimateModelSize(Object model) {
        return model != null ? model.toString().length() * 1024 : 0;
    }
    
    private float calculateAccuracyLoss(int bits) {
        if (bits >= 16) return 0.001f;
        if (bits >= 8) return 0.005f;
        if (bits >= 4) return 0.02f;
        return 0.05f;
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public String getStatus() {
        return String.format(
            "ModelCompression: enabled=%s, compressed_models=%d",
            enabled, compressedModels.size()
        );
    }
}