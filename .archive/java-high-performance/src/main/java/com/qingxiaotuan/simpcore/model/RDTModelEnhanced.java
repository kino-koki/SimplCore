package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

public class RDTModelEnhanced {
    private RDTConfig config;
    
    private float[] embedding;
    private float[] lmHeadWeight;
    private float[] lmHeadBias;
    
    private RDTTransformerBlock[] preludeBlocks;
    private RDTTransformerBlock recurrentBlock;
    private RDTTransformerBlock[] codaBlocks;
    
    private LTIInjection ltiInjection;
    private ACTModule actModule;
    private DeepWiseLoRA loraAdapter;
    
    private float[] driftHistory;
    private float[] normHistory;
    private float driftWarningThreshold;
    private float normWarningThreshold;
    private boolean driftWarningTriggered;
    
    public RDTModelEnhanced(RDTConfig config) {
        this.config = config;
        this.driftWarningThreshold = 0.1f;
        this.normWarningThreshold = 10.0f;
        initModel();
    }
    
    private void initModel() {
        embedding = new float[config.vocabSize * config.dim];
        lmHeadWeight = new float[config.dim * config.vocabSize];
        lmHeadBias = new float[config.vocabSize];
        
        float scale = (float) (1.0 / Math.sqrt(config.dim));
        for (int i = 0; i < config.vocabSize * config.dim; i++) {
            embedding[i] = (float) (Math.random() - 0.5) * 2 * scale;
            lmHeadWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        preludeBlocks = new RDTTransformerBlock[config.preludeLayers];
        for (int i = 0; i < config.preludeLayers; i++) {
            preludeBlocks[i] = new RDTTransformerBlock(
                config.dim, config.nHeads, false, 
                0, 0, 0, 0
            );
        }
        
        recurrentBlock = new RDTTransformerBlock(
            config.dim, config.nHeads, true,
            config.nExperts, config.nSharedExperts, 
            config.nExpertsPerTok, config.expertDim
        );
        
        codaBlocks = new RDTTransformerBlock[config.codaLayers];
        for (int i = 0; i < config.codaLayers; i++) {
            codaBlocks[i] = new RDTTransformerBlock(
                config.dim, config.nHeads, false,
                0, 0, 0, 0
            );
        }
        
        ltiInjection = new LTIInjection(config.dim, config.ltiSpectralBound);
        actModule = new ACTModule(config.dim, config.actThreshold);
        loraAdapter = new DeepWiseLoRA(config.dim, config.dim / 16);
        
        driftHistory = new float[1000];
        normHistory = new float[1000];
    }
    
    public float[] forward(int[] tokens, int nLoops) {
        return forward(tokens, nLoops, false);
    }
    
    public float[] forward(int[] tokens, int nLoops, boolean enableDriftMonitoring) {
        float[] x = embed(tokens);
        
        for (RDTTransformerBlock block : preludeBlocks) {
            x = block.forward(x);
        }
        
        float[] e = x;
        float initialNorm = computeL2Norm(x);
        
        if (nLoops <= 0) {
            nLoops = config.maxLoopIters;
        }
        
        driftWarningTriggered = false;
        float previousNorm = initialNorm;
        normHistory[0] = initialNorm;
        
        for (int t = 0; t < nLoops; t++) {
            float[] injection = ltiInjection.inject(x, e);
            float[] residual = x;
            
            float[] blockOut = recurrentBlock.forward(x);
            blockOut = loraAdapter.apply(blockOut);
            
            for (int i = 0; i < config.dim; i++) {
                x[i] = injection[i] + blockOut[i];
            }
            
            x = elementwiseAdd(x, residual);
            x = applyLayerNorm(x);
            x = clipNorm(x, 5.0f);
            x = applyLoopPositionalEncoding(x, t, nLoops);
            
            if (enableDriftMonitoring && t < 1000) {
                float currentNorm = computeL2Norm(x);
                normHistory[t + 1] = currentNorm;
                
                float drift = Math.abs(currentNorm - previousNorm) / previousNorm;
                driftHistory[t] = drift;
                
                if (drift > driftWarningThreshold) {
                    driftWarningTriggered = true;
                    System.out.println("[WARN] Drift detected at iteration " + t + ": " + 
                                   String.format("%.4f", drift) + " (threshold: " + driftWarningThreshold + ")");
                }
                
                if (currentNorm > normWarningThreshold) {
                    System.out.println("[WARN] Norm explosion at iteration " + t + ": " + 
                                   String.format("%.4f", currentNorm) + " (threshold: " + normWarningThreshold + ")");
                }
                
                previousNorm = currentNorm;
            }
        }
        
        for (RDTTransformerBlock block : codaBlocks) {
            x = block.forward(x);
        }
        
        float[] logits = MatrixOperations.matmul(lmHeadWeight, x, config.vocabSize, config.dim, 1);
        for (int i = 0; i < config.vocabSize; i++) {
            logits[i] += lmHeadBias[i];
        }
        
        return logits;
    }
    
    public StabilityReport runLongTermStabilityTest(int[] tokens, int nSteps) {
        float[] x = embed(tokens);
        
        for (RDTTransformerBlock block : preludeBlocks) {
            x = block.forward(x);
        }
        
        float[] e = x;
        float initialNorm = computeL2Norm(x);
        
        float[] norms = new float[nSteps];
        float[] drifts = new float[nSteps - 1];
        norms[0] = initialNorm;
        
        boolean stable = true;
        int instabilityPoint = -1;
        
        for (int t = 0; t < nSteps; t++) {
            float[] injection = ltiInjection.inject(x, e);
            float[] residual = x;
            
            float[] blockOut = recurrentBlock.forward(x);
            blockOut = loraAdapter.apply(blockOut);
            
            for (int i = 0; i < config.dim; i++) {
                x[i] = injection[i] + blockOut[i];
            }
            
            x = elementwiseAdd(x, residual);
            x = applyLayerNorm(x);
            x = clipNorm(x, 5.0f);
            x = applyLoopPositionalEncoding(x, t, nSteps);
            
            if (t > 0) {
                norms[t] = computeL2Norm(x);
                drifts[t - 1] = Math.abs(norms[t] - norms[t - 1]) / norms[t - 1];
                
                if (drifts[t - 1] > driftWarningThreshold * 5 && instabilityPoint == -1) {
                    instabilityPoint = t;
                    stable = false;
                }
                
                if (norms[t] > normWarningThreshold * 10) {
                    instabilityPoint = t;
                    stable = false;
                    break;
                }
            }
        }
        
        return new StabilityReport(norms, drifts, stable, instabilityPoint);
    }
    
    public float computeL2Norm(float[] x) {
        float sum = 0.0f;
        for (float v : x) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }
    
    public float[] getDriftHistory() {
        return driftHistory;
    }
    
    public float[] getNormHistory() {
        return normHistory;
    }
    
    public boolean isDriftWarningTriggered() {
        return driftWarningTriggered;
    }
    
    public void setDriftWarningThreshold(float threshold) {
        this.driftWarningThreshold = threshold;
    }
    
    public float getSpectralRadius() {
        return ltiInjection.getSpectralRadius();
    }
    
    public long countParameters() {
        long params = 0;
        params += (long) config.vocabSize * config.dim;
        params += (long) config.dim * config.vocabSize;
        params += config.preludeLayers * (long) config.dim * config.dim * 8;
        params += (long) config.dim * config.dim * 8;
        params += config.codaLayers * (long) config.dim * config.dim * 8;
        params += (long) config.nExperts * config.expertDim * config.dim * 2;
        params += (long) config.nSharedExperts * config.expertDim * config.dim * 2;
        params += (long) config.dim * config.dim * 2;
        params += loraAdapter.countParameters();
        return params;
    }
    
    public void tuneSpectralRadius(float targetRadius) {
        ltiInjection = new LTIInjection(config.dim, targetRadius);
    }
    
    public DeepWiseLoRA getLoraAdapter() {
        return loraAdapter;
    }
    
    private float[] embed(int[] tokens) {
        float[] result = new float[config.dim];
        
        for (int token : tokens) {
            if (token >= 0 && token < config.vocabSize) {
                int offset = token * config.dim;
                for (int i = 0; i < config.dim; i++) {
                    result[i] += embedding[offset + i];
                }
            }
        }
        
        return result;
    }
    
    private float[] applyLoopPositionalEncoding(float[] x, int t, int totalLoops) {
        float[] posEnc = new float[config.dim];
        for (int i = 0; i < config.dim; i += 2) {
            float theta = (float) (1.0 / Math.pow(10000.0, i / (double) config.dim));
            posEnc[i] = (float) Math.sin(t * theta);
            posEnc[i + 1] = (float) Math.cos(t * theta);
        }
        
        float[] result = new float[config.dim];
        for (int i = 0; i < config.dim; i++) {
            result[i] = x[i] + posEnc[i];
        }
        
        return result;
    }
    
    private float[] elementwiseAdd(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    private float[] applyLayerNorm(float[] x) {
        float[] result = new float[config.dim];
        
        float mean = 0.0f;
        for (float v : x) {
            mean += v;
        }
        mean /= config.dim;
        
        float var = 0.0f;
        for (float v : x) {
            var += (v - mean) * (v - mean);
        }
        var /= config.dim;
        
        float eps = 1e-5f;
        float std = (float) Math.sqrt(var + eps);
        
        for (int i = 0; i < config.dim; i++) {
            result[i] = (x[i] - mean) / std;
        }
        
        return result;
    }
    
    private float[] clipNorm(float[] x, float maxNorm) {
        float norm = computeL2Norm(x);
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            float[] result = new float[config.dim];
            for (int i = 0; i < config.dim; i++) {
                result[i] = x[i] * scale;
            }
            return result;
        }
        return x;
    }
    
    public static class StabilityReport {
        public float[] norms;
        public float[] drifts;
        public boolean stable;
        public int instabilityPoint;
        
        public StabilityReport(float[] norms, float[] drifts, boolean stable, int instabilityPoint) {
            this.norms = norms;
            this.drifts = drifts;
            this.stable = stable;
            this.instabilityPoint = instabilityPoint;
        }
        
        public float getMaxNorm() {
            float max = norms[0];
            for (float n : norms) {
                if (n > max) max = n;
            }
            return max;
        }
        
        public float getAvgDrift() {
            float sum = 0.0f;
            for (float d : drifts) {
                sum += d;
            }
            return sum / drifts.length;
        }
        
        public float getMaxDrift() {
            float max = drifts[0];
            for (float d : drifts) {
                if (d > max) max = d;
            }
            return max;
        }
        
        public String generateReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("Stability Report:\n");
            sb.append("  Stable: ").append(stable ? "YES" : "NO").append("\n");
            if (!stable) {
                sb.append("  Instability detected at step: ").append(instabilityPoint).append("\n");
            }
            sb.append("  Max norm: ").append(String.format("%.4f", getMaxNorm())).append("\n");
            sb.append("  Avg drift: ").append(String.format("%.6f", getAvgDrift())).append("\n");
            sb.append("  Max drift: ").append(String.format("%.6f", getMaxDrift())).append("\n");
            return sb.toString();
        }
    }
}
