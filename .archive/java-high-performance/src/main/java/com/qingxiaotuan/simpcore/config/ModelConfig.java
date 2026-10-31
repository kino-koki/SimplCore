package com.qingxiaotuan.simpcore.config;

import java.util.HashMap;
import java.util.Map;

public class ModelConfig {

    private final int vocabSize;
    private final int dModel;
    private final int nLayers;
    private final int nHeads;
    private final int nBases;
    private final int nChannels;
    private final int nPaths;
    private final int topK;
    private final int maxSeqLen;
    private final int latentDim;
    private final float compressionRatio;
    private final TrainingConfig trainingConfig;
    private final InferenceConfig inferenceConfig;
    private final HardwareConfig hardwareConfig;

    private ModelConfig(Builder builder) {
        this.vocabSize = builder.vocabSize;
        this.dModel = builder.dModel;
        this.nLayers = builder.nLayers;
        this.nHeads = builder.nHeads;
        this.nBases = builder.nBases;
        this.nChannels = builder.nChannels;
        this.nPaths = builder.nPaths;
        this.topK = builder.topK;
        this.maxSeqLen = builder.maxSeqLen;
        this.latentDim = builder.latentDim;
        this.compressionRatio = builder.compressionRatio;
        this.trainingConfig = builder.trainingConfig;
        this.inferenceConfig = builder.inferenceConfig;
        this.hardwareConfig = builder.hardwareConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNLayers() { return nLayers; }
    public int getNHeads() { return nHeads; }
    public int getNBases() { return nBases; }
    public int getNChannels() { return nChannels; }
    public int getNPaths() { return nPaths; }
    public int getTopK() { return topK; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public int getLatentDim() { return latentDim; }
    public float getCompressionRatio() { return compressionRatio; }
    public TrainingConfig getTrainingConfig() { return trainingConfig; }
    public InferenceConfig getInferenceConfig() { return inferenceConfig; }
    public HardwareConfig getHardwareConfig() { return hardwareConfig; }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("vocabSize", vocabSize);
        map.put("dModel", dModel);
        map.put("nLayers", nLayers);
        map.put("nHeads", nHeads);
        map.put("nBases", nBases);
        map.put("nChannels", nChannels);
        map.put("nPaths", nPaths);
        map.put("topK", topK);
        map.put("maxSeqLen", maxSeqLen);
        map.put("latentDim", latentDim);
        map.put("compressionRatio", compressionRatio);
        map.put("training", trainingConfig.toMap());
        map.put("inference", inferenceConfig.toMap());
        map.put("hardware", hardwareConfig.toMap());
        return map;
    }

    public static class Builder {
        private int vocabSize = 10000;
        private int dModel = 512;
        private int nLayers = 6;
        private int nHeads = 8;
        private int nBases = 4;
        private int nChannels = 4;
        private int nPaths = 8;
        private int topK = 2;
        private int maxSeqLen = 512;
        private int latentDim = 128;
        private float compressionRatio = 0.8f;
        private TrainingConfig trainingConfig = TrainingConfig.builder().build();
        private InferenceConfig inferenceConfig = InferenceConfig.builder().build();
        private HardwareConfig hardwareConfig = HardwareConfig.builder().build();

        public Builder vocabSize(int val) { vocabSize = val; return this; }
        public Builder dModel(int val) { dModel = val; return this; }
        public Builder nLayers(int val) { nLayers = val; return this; }
        public Builder nHeads(int val) { nHeads = val; return this; }
        public Builder nBases(int val) { nBases = val; return this; }
        public Builder nChannels(int val) { nChannels = val; return this; }
        public Builder nPaths(int val) { nPaths = val; return this; }
        public Builder topK(int val) { topK = val; return this; }
        public Builder maxSeqLen(int val) { maxSeqLen = val; return this; }
        public Builder latentDim(int val) { latentDim = val; return this; }
        public Builder compressionRatio(float val) { compressionRatio = val; return this; }
        public Builder trainingConfig(TrainingConfig val) { trainingConfig = val; return this; }
        public Builder inferenceConfig(InferenceConfig val) { inferenceConfig = val; return this; }
        public Builder hardwareConfig(HardwareConfig val) { hardwareConfig = val; return this; }

        public ModelConfig build() { return new ModelConfig(this); }
    }

    public static class TrainingConfig {
        private final float learningRate;
        private final float weightDecay;
        private final float momentum;
        private final float gradClip;
        private final int batchSize;
        private final int gradientAccumulationSteps;
        private final int warmupSteps;
        private final int totalSteps;
        private final String lrSchedulerType;
        private final boolean mixedPrecision;

        private TrainingConfig(TrainingConfigBuilder builder) {
            this.learningRate = builder.learningRate;
            this.weightDecay = builder.weightDecay;
            this.momentum = builder.momentum;
            this.gradClip = builder.gradClip;
            this.batchSize = builder.batchSize;
            this.gradientAccumulationSteps = builder.gradientAccumulationSteps;
            this.warmupSteps = builder.warmupSteps;
            this.totalSteps = builder.totalSteps;
            this.lrSchedulerType = builder.lrSchedulerType;
            this.mixedPrecision = builder.mixedPrecision;
        }

        public static TrainingConfigBuilder builder() { return new TrainingConfigBuilder(); }

        public float getLearningRate() { return learningRate; }
        public float getWeightDecay() { return weightDecay; }
        public float getMomentum() { return momentum; }
        public float getGradClip() { return gradClip; }
        public int getBatchSize() { return batchSize; }
        public int getGradientAccumulationSteps() { return gradientAccumulationSteps; }
        public int getWarmupSteps() { return warmupSteps; }
        public int getTotalSteps() { return totalSteps; }
        public String getLrSchedulerType() { return lrSchedulerType; }
        public boolean isMixedPrecision() { return mixedPrecision; }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("learningRate", learningRate);
            map.put("weightDecay", weightDecay);
            map.put("momentum", momentum);
            map.put("gradClip", gradClip);
            map.put("batchSize", batchSize);
            map.put("gradientAccumulationSteps", gradientAccumulationSteps);
            map.put("warmupSteps", warmupSteps);
            map.put("totalSteps", totalSteps);
            map.put("lrSchedulerType", lrSchedulerType);
            map.put("mixedPrecision", mixedPrecision);
            return map;
        }

        public static class TrainingConfigBuilder {
            private float learningRate = 0.001f;
            private float weightDecay = 0.01f;
            private float momentum = 0.9f;
            private float gradClip = 1.0f;
            private int batchSize = 8;
            private int gradientAccumulationSteps = 1;
            private int warmupSteps = 1000;
            private int totalSteps = 100000;
            private String lrSchedulerType = "cosine";
            private boolean mixedPrecision = false;

            public TrainingConfigBuilder learningRate(float val) { learningRate = val; return this; }
            public TrainingConfigBuilder weightDecay(float val) { weightDecay = val; return this; }
            public TrainingConfigBuilder momentum(float val) { momentum = val; return this; }
            public TrainingConfigBuilder gradClip(float val) { gradClip = val; return this; }
            public TrainingConfigBuilder batchSize(int val) { batchSize = val; return this; }
            public TrainingConfigBuilder gradientAccumulationSteps(int val) { gradientAccumulationSteps = val; return this; }
            public TrainingConfigBuilder warmupSteps(int val) { warmupSteps = val; return this; }
            public TrainingConfigBuilder totalSteps(int val) { totalSteps = val; return this; }
            public TrainingConfigBuilder lrSchedulerType(String val) { lrSchedulerType = val; return this; }
            public TrainingConfigBuilder mixedPrecision(boolean val) { mixedPrecision = val; return this; }

            public TrainingConfig build() { return new TrainingConfig(this); }
        }
    }

    public static class InferenceConfig {
        private final int maxNewTokens;
        private final float temperature;
        private final int topK;
        private final float topP;
        private final boolean useKVCache;
        private final int cacheMaxSize;

        private InferenceConfig(InferenceConfigBuilder builder) {
            this.maxNewTokens = builder.maxNewTokens;
            this.temperature = builder.temperature;
            this.topK = builder.topK;
            this.topP = builder.topP;
            this.useKVCache = builder.useKVCache;
            this.cacheMaxSize = builder.cacheMaxSize;
        }

        public static InferenceConfigBuilder builder() { return new InferenceConfigBuilder(); }

        public int getMaxNewTokens() { return maxNewTokens; }
        public float getTemperature() { return temperature; }
        public int getTopK() { return topK; }
        public float getTopP() { return topP; }
        public boolean isUseKVCache() { return useKVCache; }
        public int getCacheMaxSize() { return cacheMaxSize; }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("maxNewTokens", maxNewTokens);
            map.put("temperature", temperature);
            map.put("topK", topK);
            map.put("topP", topP);
            map.put("useKVCache", useKVCache);
            map.put("cacheMaxSize", cacheMaxSize);
            return map;
        }

        public static class InferenceConfigBuilder {
            private int maxNewTokens = 100;
            private float temperature = 1.0f;
            private int topK = 50;
            private float topP = 0.9f;
            private boolean useKVCache = true;
            private int cacheMaxSize = 10000;

            public InferenceConfigBuilder maxNewTokens(int val) { maxNewTokens = val; return this; }
            public InferenceConfigBuilder temperature(float val) { temperature = val; return this; }
            public InferenceConfigBuilder topK(int val) { topK = val; return this; }
            public InferenceConfigBuilder topP(float val) { topP = val; return this; }
            public InferenceConfigBuilder useKVCache(boolean val) { useKVCache = val; return this; }
            public InferenceConfigBuilder cacheMaxSize(int val) { cacheMaxSize = val; return this; }

            public InferenceConfig build() { return new InferenceConfig(this); }
        }
    }

    public static class HardwareConfig {
        private final int numThreads;
        private final boolean useGPU;
        private final int gpuDeviceId;
        private final boolean useSIMD;
        private final int memoryLimitMB;

        private HardwareConfig(HardwareConfigBuilder builder) {
            this.numThreads = builder.numThreads;
            this.useGPU = builder.useGPU;
            this.gpuDeviceId = builder.gpuDeviceId;
            this.useSIMD = builder.useSIMD;
            this.memoryLimitMB = builder.memoryLimitMB;
        }

        public static HardwareConfigBuilder builder() { return new HardwareConfigBuilder(); }

        public int getNumThreads() { return numThreads; }
        public boolean isUseGPU() { return useGPU; }
        public int getGpuDeviceId() { return gpuDeviceId; }
        public boolean isUseSIMD() { return useSIMD; }
        public int getMemoryLimitMB() { return memoryLimitMB; }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("numThreads", numThreads);
            map.put("useGPU", useGPU);
            map.put("gpuDeviceId", gpuDeviceId);
            map.put("useSIMD", useSIMD);
            map.put("memoryLimitMB", memoryLimitMB);
            return map;
        }

        public static class HardwareConfigBuilder {
            private int numThreads = Runtime.getRuntime().availableProcessors();
            private boolean useGPU = true;
            private int gpuDeviceId = 0;
            private boolean useSIMD = true;
            private int memoryLimitMB = 8192;

            public HardwareConfigBuilder numThreads(int val) { numThreads = val; return this; }
            public HardwareConfigBuilder useGPU(boolean val) { useGPU = val; return this; }
            public HardwareConfigBuilder gpuDeviceId(int val) { gpuDeviceId = val; return this; }
            public HardwareConfigBuilder useSIMD(boolean val) { useSIMD = val; return this; }
            public HardwareConfigBuilder memoryLimitMB(int val) { memoryLimitMB = val; return this; }

            public HardwareConfig build() { return new HardwareConfig(this); }
        }
    }
}