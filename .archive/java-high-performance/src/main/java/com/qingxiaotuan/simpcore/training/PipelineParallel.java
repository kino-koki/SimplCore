package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.distributed.CommunicationLayer;
import com.qingxiaotuan.simpcore.model.RDTTransformerBlock;
import com.qingxiaotuan.simpcore.model.LTIInjection;

public class PipelineParallel {
    private RDTTransformerBlock[] preludeBlocks;
    private RDTTransformerBlock recurrentBlock;
    private RDTTransformerBlock[] codaBlocks;
    private LTIInjection ltiInjection;
    
    private int rank;
    private int worldSize;
    private int stage;
    private int nStages;
    
    private int preludeGPUs;
    private int recurrentGPUs;
    private int codaGPUs;
    
    private int microBatchSize;
    
    private CommunicationLayer comm;
    
    public PipelineParallel(RDTTransformerBlock[] preludeBlocks, 
                          RDTTransformerBlock recurrentBlock,
                          RDTTransformerBlock[] codaBlocks,
                          LTIInjection ltiInjection,
                          int rank, int worldSize, int microBatchSize) {
        this.preludeBlocks = preludeBlocks;
        this.recurrentBlock = recurrentBlock;
        this.codaBlocks = codaBlocks;
        this.ltiInjection = ltiInjection;
        this.rank = rank;
        this.worldSize = worldSize;
        this.microBatchSize = microBatchSize;
        
        nStages = 3;
        preludeGPUs = worldSize / 3;
        recurrentGPUs = worldSize / 3;
        codaGPUs = worldSize - preludeGPUs - recurrentGPUs;
        
        if (rank < preludeGPUs) {
            stage = 0;
        } else if (rank < preludeGPUs + recurrentGPUs) {
            stage = 1;
        } else {
            stage = 2;
        }
        
        comm = new CommunicationLayer(rank, worldSize);
    }
    
    public void init() throws Exception {
        comm.init();
    }
    
    public float[] forward(float[] x, int nLoops) {
        if (stage == 0) {
            return runPrelude(x);
        } else if (stage == 1) {
            return runRecurrent(x, nLoops);
        } else {
            return runCoda(x);
        }
    }
    
    private float[] runPrelude(float[] x) {
        for (RDTTransformerBlock block : preludeBlocks) {
            x = block.forward(x);
        }
        
        int nextRank = preludeGPUs;
        comm.sendAsync(nextRank, x);
        
        return x;
    }
    
    private float[] runRecurrent(float[] x, int nLoops) {
        int prevRank = rank - 1;
        if (prevRank < 0) prevRank = 0;
        comm.receiveBlocking(prevRank, x);
        
        float[] e = x;
        
        for (int t = 0; t < nLoops; t++) {
            float[] injection = ltiInjection.inject(x, e);
            float[] residual = x;
            
            float[] blockOut = recurrentBlock.forward(x);
            
            for (int i = 0; i < x.length; i++) {
                x[i] = injection[i] + blockOut[i];
            }
            
            x = elementwiseAdd(x, residual);
        }
        
        int nextRank = preludeGPUs + recurrentGPUs;
        if (rank == preludeGPUs + recurrentGPUs - 1) {
            nextRank = preludeGPUs + recurrentGPUs;
        } else {
            nextRank = rank + 1;
        }
        comm.sendAsync(nextRank, x);
        
        return x;
    }
    
    private float[] runCoda(float[] x) {
        int prevRank = rank - 1;
        if (prevRank < preludeGPUs) prevRank = preludeGPUs;
        comm.receiveBlocking(prevRank, x);
        
        for (RDTTransformerBlock block : codaBlocks) {
            x = block.forward(x);
        }
        
        return x;
    }
    
    private float[] elementwiseAdd(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    public int getStage() {
        return stage;
    }
    
    public int getStageGPUs() {
        if (stage == 0) return preludeGPUs;
        if (stage == 1) return recurrentGPUs;
        return codaGPUs;
    }
}
