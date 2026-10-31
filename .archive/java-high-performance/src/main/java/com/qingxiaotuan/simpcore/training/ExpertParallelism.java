package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.distributed.CommunicationLayer;
import com.qingxiaotuan.simpcore.model.MoELayer;

public class ExpertParallelism {
    private MoELayer[] experts;
    private int nExperts;
    private int nSharedExperts;
    private int nExpertsPerTok;
    private int expertDim;
    private int dim;
    
    private int rank;
    private int worldSize;
    private int expertsPerDevice;
    private int startExpertIdx;
    
    private float[] routerWeights;
    private float[] routerBias;
    
    private CommunicationLayer comm;
    
    public ExpertParallelism(int dim, int expertDim, int nExperts, int nSharedExperts, 
                            int nExpertsPerTok, int rank, int worldSize) {
        this.dim = dim;
        this.expertDim = expertDim;
        this.nExperts = nExperts;
        this.nSharedExperts = nSharedExperts;
        this.nExpertsPerTok = nExpertsPerTok;
        this.rank = rank;
        this.worldSize = worldSize;
        
        expertsPerDevice = nExperts / worldSize;
        if (rank < nExperts % worldSize) {
            expertsPerDevice++;
        }
        
        startExpertIdx = rank * (nExperts / worldSize);
        if (rank >= nExperts % worldSize) {
            startExpertIdx += nExperts % worldSize;
        }
        
        experts = new MoELayer[expertsPerDevice];
        for (int i = 0; i < expertsPerDevice; i++) {
            experts[i] = new MoELayer(dim, expertDim, 1, nSharedExperts, 1);
        }
        
        routerWeights = new float[dim * nExperts];
        routerBias = new float[nExperts];
        
        float scale = (float) (1.0 / Math.sqrt(dim));
        for (int i = 0; i < dim * nExperts; i++) {
            routerWeights[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        comm = new CommunicationLayer(rank, worldSize);
    }
    
    public void init() throws Exception {
        comm.init();
    }
    
    public float[] forward(float[] x) {
        float[] routerLogits = computeRouterLogits(x);
        
        int[] selectedExperts = selectTopK(routerLogits, nExpertsPerTok);
        float[] selectedWeights = new float[nExpertsPerTok];
        float[] softmaxProbs = softmax(routerLogits);
        for (int i = 0; i < nExpertsPerTok; i++) {
            selectedWeights[i] = softmaxProbs[selectedExperts[i]];
        }
        
        float[] result = new float[dim];
        
        for (int i = 0; i < nExpertsPerTok; i++) {
            int expertIdx = selectedExperts[i];
            float weight = selectedWeights[i];
            
            if (expertIdx >= startExpertIdx && expertIdx < startExpertIdx + expertsPerDevice) {
                int localIdx = expertIdx - startExpertIdx;
                float[] expertOut = experts[localIdx].forward(x);
                for (int j = 0; j < dim; j++) {
                    result[j] += expertOut[j] * weight;
                }
            } else {
                float[] expertOut = communicateWithRemoteExpert(expertIdx, x);
                for (int j = 0; j < dim; j++) {
                    result[j] += expertOut[j] * weight;
                }
            }
        }
        
        for (int i = 0; i < nSharedExperts; i++) {
            float[] sharedOut = experts[0].forward(x);
            for (int j = 0; j < dim; j++) {
                result[j] += sharedOut[j] / nSharedExperts;
            }
        }
        
        return result;
    }
    
    private float[] computeRouterLogits(float[] x) {
        float[] logits = new float[nExperts];
        for (int i = 0; i < nExperts; i++) {
            float dot = 0.0f;
            for (int j = 0; j < dim; j++) {
                dot += routerWeights[i * dim + j] * x[j];
            }
            logits[i] = dot + routerBias[i];
        }
        return logits;
    }
    
    private float[] communicateWithRemoteExpert(int expertIdx, float[] x) {
        int targetRank = expertIdx / expertsPerDevice;
        
        comm.sendBlocking(targetRank, x);
        float[] result = new float[dim];
        comm.receiveBlocking(targetRank, result);
        
        return result;
    }
    
    public int getExpertsPerDevice() {
        return expertsPerDevice;
    }
    
    public int getStartExpertIdx() {
        return startExpertIdx;
    }
    
    public float getMemoryUsage() {
        float expertMemory = expertsPerDevice * (long) dim * expertDim * 4 * 2 / (1024f * 1024f);
        float routerMemory = (routerWeights.length + routerBias.length) * 4 / (1024f * 1024f);
        return expertMemory + routerMemory;
    }
    
    private int[] selectTopK(float[] probs, int k) {
        int[] indices = new int[k];
        float[] values = new float[k];
        
        for (int i = 0; i < k; i++) {
            indices[i] = -1;
            values[i] = Float.NEGATIVE_INFINITY;
        }
        
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > values[k - 1]) {
                int j = k - 1;
                while (j > 0 && probs[i] > values[j - 1]) {
                    values[j] = values[j - 1];
                    indices[j] = indices[j - 1];
                    j--;
                }
                values[j] = probs[i];
                indices[j] = i;
            }
        }
        
        return indices;
    }
    
    private float[] softmax(float[] x) {
        float max = x[0];
        for (float v : x) {
            if (v > max) max = v;
        }
        
        float sum = 0.0f;
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = (float) Math.exp(x[i] - max);
            sum += result[i];
        }
        
        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
}
