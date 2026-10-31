package com.qingxiaotuan.simpcore.model;

public class RDTConfig {
    public int vocabSize;
    public int dim;
    public int nHeads;
    public int nKvHeads;
    public int maxSeqLen;
    public int maxLoopIters;
    public int preludeLayers;
    public int codaLayers;
    
    public String attnType;
    public int kvLoraRank;
    public int qLoraRank;
    public int qkRopeHeadDim;
    public int qkNopeHeadDim;
    public int vHeadDim;
    
    public int nExperts;
    public int nSharedExperts;
    public int nExpertsPerTok;
    public int expertDim;
    
    public float actThreshold;
    public float ltiSpectralBound;
    public float dropoutRate;
    
    public RDTConfig() {}
    
    public RDTConfig(
        int vocabSize, int dim, int nHeads, int nKvHeads,
        int maxSeqLen, int maxLoopIters,
        int preludeLayers, int codaLayers,
        String attnType,
        int nExperts, int nSharedExperts, int nExpertsPerTok, int expertDim,
        float actThreshold, float ltiSpectralBound, float dropoutRate
    ) {
        this.vocabSize = vocabSize;
        this.dim = dim;
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.maxSeqLen = maxSeqLen;
        this.maxLoopIters = maxLoopIters;
        this.preludeLayers = preludeLayers;
        this.codaLayers = codaLayers;
        this.attnType = attnType;
        this.nExperts = nExperts;
        this.nSharedExperts = nSharedExperts;
        this.nExpertsPerTok = nExpertsPerTok;
        this.expertDim = expertDim;
        this.actThreshold = actThreshold;
        this.ltiSpectralBound = ltiSpectralBound;
        this.dropoutRate = dropoutRate;
        this.kvLoraRank = dim / 4;
        this.qLoraRank = dim / 2;
        this.qkRopeHeadDim = 64;
        this.qkNopeHeadDim = 128;
        this.vHeadDim = 128;
    }
    
    public static RDTConfig mythos_1b() {
        return new RDTConfig(32000, 2048, 16, 4, 4096, 16, 2, 2, "mla", 64, 2, 4, 2048, 0.99f, 0.95f, 0.1f);
    }
    
    public static RDTConfig mythos_3b() {
        return new RDTConfig(32000, 3072, 24, 6, 4096, 16, 2, 2, "mla", 64, 2, 4, 4096, 0.99f, 0.95f, 0.1f);
    }
    
    public static RDTConfig mythos_10b() {
        return new RDTConfig(32000, 4096, 32, 8, 8192, 24, 3, 3, "mla", 128, 4, 4, 5632, 0.99f, 0.95f, 0.1f);
    }
    
    public static RDTConfig mythos_50b() {
        return new RDTConfig(64000, 6144, 48, 12, 8192, 32, 4, 4, "mla", 256, 8, 4, 9728, 0.99f, 0.95f, 0.1f);
    }
    
    public static RDTConfig mythos_100b() {
        return new RDTConfig(100000, 8192, 64, 16, 1000000, 32, 4, 4, "mla", 256, 8, 4, 13568, 0.99f, 0.95f, 0.05f);
    }
    
    public static RDTConfig mythos_550b() {
        return new RDTConfig(100000, 12288, 96, 24, 1000000, 48, 6, 6, "mla", 512, 16, 6, 23040, 0.99f, 0.95f, 0.05f);
    }
    
    public static RDTConfig mythos_v100_48x() {
        return new RDTConfig(100000, 4096, 32, 8, 8192, 32, 4, 4, "mla", 192, 8, 4, 8192, 0.99f, 0.95f, 0.05f);
    }
    
    public static RDTConfig qingxiaotuan_100m() {
        // vocab=65536 (翻倍), dim=512, 12头MLA, 12专家(expertDim=1536)
        // 参数量: embedding+lmHead ~67M + 其他层 ~33M ≈ 100M
        return new RDTConfig(65536, 512, 12, 2, 2048, 8, 2, 2, "mla", 12, 3, 2, 1536, 0.99f, 0.95f, 0.1f);
    }
    
    public static RDTConfig prototype_mini() {
        return new RDTConfig(1000, 128, 4, 1, 128, 4, 1, 1, "mla", 4, 1, 2, 64, 0.99f, 0.95f, 0.1f);
    }
    
    public long estimateParams() {
        long params = 0;
        params += (long) vocabSize * dim;
        params += (long) dim * 3;
        int headDim = dim / nHeads;
        params += (long) nHeads * headDim * (headDim + dim) * 3;
        params += (long) nExperts * expertDim * dim * 2;
        params += (long) nSharedExperts * expertDim * dim * 2;
        params += (long) dim * dim * 2;
        params += (long) dim * vocabSize;
        return params;
    }
}
