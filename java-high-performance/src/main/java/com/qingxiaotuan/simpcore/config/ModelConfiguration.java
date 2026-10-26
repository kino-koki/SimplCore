package com.qingxiaotuan.simpcore.config;

import java.util.*;

public class ModelConfiguration {
    
    private long targetParams;
    private int vocabSize;
    private int dModel;
    private int nLayers;
    private int nHeads;
    private int dFF;
    private int maxSeqLen;
    private float paramDensity;
    private boolean useSparse;
    private float hallucinationControl;
    private int verificationLayers;
    private int attentionSpanFactor;
    
    public static final Map<String, Long> PRESET_SCALES = new LinkedHashMap<>();
    static {
        PRESET_SCALES.put("tiny-1M", 1_000_000L);
        PRESET_SCALES.put("small-10M", 10_000_000L);
        PRESET_SCALES.put("base-100M", 100_000_000L);
        PRESET_SCALES.put("medium-1B", 1_000_000_000L);
        PRESET_SCALES.put("large-7B", 7_000_000_000L);
        PRESET_SCALES.put("xlarge-13B", 13_000_000_000L);
        PRESET_SCALES.put("xxlarge-33B", 33_000_000_000L);
        PRESET_SCALES.put("giant-70B", 70_000_000_000L);
        PRESET_SCALES.put("huge-100B", 100_000_000_000L);
        PRESET_SCALES.put("massive-500B", 500_000_000_000L);
        PRESET_SCALES.put("colossal-1T", 1_000_000_000_000L);
        PRESET_SCALES.put("titan-5T", 5_000_000_000_000L);
        PRESET_SCALES.put("cosmic-12T", 12_000_000_000_000L);
        PRESET_SCALES.put("universal-20T", 20_000_000_000_000L);
    }
    
    public ModelConfiguration(long targetParams) {
        this.targetParams = targetParams;
        this.paramDensity = 1.0f;
        this.useSparse = false;
        this.hallucinationControl = 0.5f;
        this.verificationLayers = 2;
        this.attentionSpanFactor = 1;
        autoScale();
    }
    
    public ModelConfiguration(String presetName) {
        this(PRESET_SCALES.getOrDefault(presetName, 1_000_000L));
    }
    
    private void autoScale() {
        double params = targetParams;
        
        if (params <= 10_000_000) {
            vocabSize = 512;
            dModel = 64;
            nLayers = 4;
            nHeads = 4;
            dFF = 256;
            maxSeqLen = 256;
        } else if (params <= 100_000_000) {
            vocabSize = 2048;
            dModel = 128;
            nLayers = 8;
            nHeads = 8;
            dFF = 512;
            maxSeqLen = 512;
        } else if (params <= 1_000_000_000) {
            vocabSize = 8192;
            dModel = 256;
            nLayers = 12;
            nHeads = 8;
            dFF = 1024;
            maxSeqLen = 1024;
        } else if (params <= 7_000_000_000L) {
            vocabSize = 32768;
            dModel = 4096;
            nLayers = 32;
            nHeads = 32;
            dFF = 11008;
            maxSeqLen = 2048;
        } else if (params <= 13_000_000_000L) {
            vocabSize = 32768;
            dModel = 5120;
            nLayers = 40;
            nHeads = 40;
            dFF = 13824;
            maxSeqLen = 2048;
        } else if (params <= 33_000_000_000L) {
            vocabSize = 32768;
            dModel = 6144;
            nLayers = 60;
            nHeads = 48;
            dFF = 16384;
            maxSeqLen = 4096;
        } else if (params <= 70_000_000_000L) {
            vocabSize = 65536;
            dModel = 8192;
            nLayers = 80;
            nHeads = 64;
            dFF = 22016;
            maxSeqLen = 4096;
        } else if (params <= 100_000_000_000L) {
            vocabSize = 65536;
            dModel = 10240;
            nLayers = 100;
            nHeads = 80;
            dFF = 28672;
            maxSeqLen = 8192;
        } else if (params <= 500_000_000_000L) {
            vocabSize = 131072;
            dModel = 16384;
            nLayers = 128;
            nHeads = 128;
            dFF = 49152;
            maxSeqLen = 16384;
        } else if (params <= 1_000_000_000_000L) {
            vocabSize = 131072;
            dModel = 24576;
            nLayers = 160;
            nHeads = 192;
            dFF = 73728;
            maxSeqLen = 32768;
        } else if (params <= 5_000_000_000_000L) {
            vocabSize = 262144;
            dModel = 32768;
            nLayers = 200;
            nHeads = 256;
            dFF = 98304;
            maxSeqLen = 65536;
        } else if (params <= 12_000_000_000_000L) {
            vocabSize = 262144;
            dModel = 40960;
            nLayers = 240;
            nHeads = 320;
            dFF = 122880;
            maxSeqLen = 131072;
        } else {
            vocabSize = 524288;
            dModel = 49152;
            nLayers = 280;
            nHeads = 384;
            dFF = 147456;
            maxSeqLen = 262144;
            hallucinationControl = 0.95f;
            verificationLayers = 8;
            attentionSpanFactor = 4;
        }
        
        if (useSparse) {
            dModel = (int) (dModel * 1.5);
            paramDensity = 0.4f;
        }
        
        if (targetParams >= 1_000_000_000_000L) {
            hallucinationControl = Math.min(0.9f + (float) Math.log10(targetParams / 1_000_000_000_000L) * 0.02f, 0.99f);
            verificationLayers = Math.min(2 + (int) (targetParams / 500_000_000_000L), 16);
            attentionSpanFactor = Math.min(1 + (int) (targetParams / 500_000_000_000L), 8);
        }
    }
    
    public long calculateActualParams() {
        long embParams = (long) vocabSize * dModel;
        long attnParams = (long) nLayers * (4L * dModel * dModel + 4L * dModel);
        long ffnParams = (long) nLayers * (2L * dModel * dFF + dFF + dModel);
        long lnParams = (long) nLayers * 2L * dModel * 2;
        
        return (long) ((embParams + attnParams + ffnParams + lnParams) * paramDensity);
    }
    
    public ModelConfiguration withVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
        return this;
    }
    
    public ModelConfiguration withDModel(int dModel) {
        this.dModel = dModel;
        return this;
    }
    
    public ModelConfiguration withNLayers(int nLayers) {
        this.nLayers = nLayers;
        return this;
    }
    
    public ModelConfiguration withNHeads(int nHeads) {
        this.nHeads = nHeads;
        return this;
    }
    
    public ModelConfiguration withMaxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }
    
    public ModelConfiguration withSparse(boolean useSparse) {
        this.useSparse = useSparse;
        autoScale();
        return this;
    }
    
    public long getTargetParams() { return targetParams; }
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNLayers() { return nLayers; }
    public int getNHeads() { return nHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public float getParamDensity() { return paramDensity; }
    public boolean isSparse() { return useSparse; }
    public float getHallucinationControl() { return hallucinationControl; }
    public int getVerificationLayers() { return verificationLayers; }
    public int getAttentionSpanFactor() { return attentionSpanFactor; }
    
    public String getPresetName() {
        for (Map.Entry<String, Long> entry : PRESET_SCALES.entrySet()) {
            if (entry.getValue() == targetParams) {
                return entry.getKey();
            }
        }
        return "custom-" + targetParams;
    }
    
    public String toSummary() {
        return String.format(
            "ModelConfig: %s | Target: %.1fB | Actual: %.1fB | dModel=%d | Layers=%d | Heads=%d | SeqLen=%d | HallucinationControl=%.2f | VerificationLayers=%d",
            getPresetName(),
            targetParams / 1_000_000_000.0,
            calculateActualParams() / 1_000_000_000.0,
            dModel, nLayers, nHeads, maxSeqLen, hallucinationControl, verificationLayers
        );
    }
    
    public static List<String> listPresets() {
        return new ArrayList<>(PRESET_SCALES.keySet());
    }
}