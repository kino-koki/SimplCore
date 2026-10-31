package com.qingxiaotuan.simpcore.model;

public class MLAAttention {
    private int dim;
    private int nHeads;
    private int headDim;
    private int kvRank;
    private int compressionRatio;
    
    private float[] qWeight;
    private float[] kWeight;
    private float[] vWeight;
    private float[] oWeight;
    
    private float[] kDownWeight;
    private float[] vDownWeight;
    private float[] kUpWeight;
    private float[] vUpWeight;
    
    private float[] cachedK;
    private float[] cachedV;
    private int cacheSize;
    private int currentCachePos;
    
    public MLAAttention(int dim, int nHeads, int kvRank, int maxCacheSize) {
        this.dim = dim;
        this.nHeads = nHeads;
        this.headDim = dim / nHeads;
        this.kvRank = kvRank;
        this.compressionRatio = headDim / kvRank;
        
        this.cacheSize = maxCacheSize;
        this.currentCachePos = 0;
        
        initWeights();
        initCache();
    }
    
    private void initWeights() {
        float scale = (float) (1.0 / Math.sqrt(dim));
        
        qWeight = new float[dim * dim];
        kWeight = new float[dim * dim];
        vWeight = new float[dim * dim];
        oWeight = new float[dim * dim];
        
        kDownWeight = new float[dim * kvRank];
        vDownWeight = new float[dim * kvRank];
        kUpWeight = new float[kvRank * dim];
        vUpWeight = new float[kvRank * dim];
        
        for (int i = 0; i < dim * dim; i++) {
            qWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            kWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            vWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            oWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        float loraScale = (float) (1.0 / Math.sqrt(kvRank));
        for (int i = 0; i < dim * kvRank; i++) {
            kDownWeight[i] = (float) (Math.random() - 0.5) * 2 * loraScale;
            vDownWeight[i] = (float) (Math.random() - 0.5) * 2 * loraScale;
        }
        
        for (int i = 0; i < kvRank * dim; i++) {
            kUpWeight[i] = (float) (Math.random() - 0.5) * 2 * loraScale;
            vUpWeight[i] = (float) (Math.random() - 0.5) * 2 * loraScale;
        }
    }
    
    private void initCache() {
        int compressedDim = nHeads * kvRank;
        cachedK = new float[cacheSize * compressedDim];
        cachedV = new float[cacheSize * compressedDim];
    }
    
    public float[] forward(float[] x) {
        float[] q = matmul(qWeight, x, dim, dim, 1);
        float[] k = matmul(kWeight, x, dim, dim, 1);
        float[] v = matmul(vWeight, x, dim, dim, 1);
        
        float[] compressedK = compressKV(k, kDownWeight, kUpWeight);
        float[] compressedV = compressKV(v, vDownWeight, vUpWeight);
        
        addToCache(compressedK, compressedV);
        
        float[] output = attention(q, compressedK, compressedV);
        
        return matmul(oWeight, output, dim, dim, 1);
    }
    
    private float[] compressKV(float[] x, float[] downWeight, float[] upWeight) {
        float[] down = matmul(downWeight, x, kvRank, dim, 1);
        
        down = applyGelu(down);
        
        float[] up = matmul(upWeight, down, dim, kvRank, 1);
        
        return up;
    }
    
    private float[] attention(float[] q, float[] k, float[] v) {
        float[] output = new float[dim];
        
        int compressedDim = nHeads * kvRank;
        int effectiveCacheSize = currentCachePos;
        
        for (int h = 0; h < nHeads; h++) {
            float[] qh = new float[headDim];
            for (int i = 0; i < headDim; i++) {
                qh[i] = q[h * headDim + i];
            }
            
            float[] scores = new float[effectiveCacheSize];
            float maxScore = Float.NEGATIVE_INFINITY;
            
            for (int t = 0; t < effectiveCacheSize; t++) {
                float score = 0.0f;
                for (int i = 0; i < kvRank; i++) {
                    float ki = k[t * compressedDim + h * kvRank + i];
                    float qi = qh[i];
                    score += ki * qi;
                }
                score /= (float) Math.sqrt(kvRank);
                scores[t] = score;
                if (score > maxScore) maxScore = score;
            }
            
            float[] probs = new float[effectiveCacheSize];
            float sumProbs = 0.0f;
            for (int t = 0; t < effectiveCacheSize; t++) {
                probs[t] = (float) Math.exp(scores[t] - maxScore);
                sumProbs += probs[t];
            }
            for (int t = 0; t < effectiveCacheSize; t++) {
                probs[t] /= sumProbs;
            }
            
            for (int i = 0; i < headDim; i++) {
                float val = 0.0f;
                for (int t = 0; t < effectiveCacheSize; t++) {
                    float vi = v[t * compressedDim + h * kvRank + Math.min(i, kvRank - 1)];
                    val += probs[t] * vi;
                }
                output[h * headDim + i] = val;
            }
        }
        
        return output;
    }
    
    private void addToCache(float[] k, float[] v) {
        int compressedDim = nHeads * kvRank;
        
        if (currentCachePos < cacheSize) {
            System.arraycopy(k, 0, cachedK, currentCachePos * compressedDim, compressedDim);
            System.arraycopy(v, 0, cachedV, currentCachePos * compressedDim, compressedDim);
            currentCachePos++;
        } else {
            System.arraycopy(cachedK, compressedDim, cachedK, 0, (cacheSize - 1) * compressedDim);
            System.arraycopy(cachedV, compressedDim, cachedV, 0, (cacheSize - 1) * compressedDim);
            System.arraycopy(k, 0, cachedK, (cacheSize - 1) * compressedDim, compressedDim);
            System.arraycopy(v, 0, cachedV, (cacheSize - 1) * compressedDim, compressedDim);
        }
    }
    
    private float[] matmul(float[] w, float[] x, int outDim, int inDim, int batch) {
        float[] result = new float[outDim * batch];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < outDim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < inDim; j++) {
                    sum += w[i * inDim + j] * x[b * inDim + j];
                }
                result[b * outDim + i] = sum;
            }
        }
        return result;
    }
    
    private float[] applyGelu(float[] x) {
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = x[i] * 0.5f * (1.0f + (float) Math.tanh(Math.sqrt(2.0 / Math.PI) * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
        }
        return result;
    }
    
    public long getOriginalCacheSize(int seqLen) {
        return (long) seqLen * nHeads * headDim * 2;
    }
    
    public long getCompressedCacheSize(int seqLen) {
        return (long) seqLen * nHeads * kvRank * 2;
    }
    
    public int getCompressionRatio() {
        return compressionRatio;
    }
    
    public void clearCache() {
        currentCachePos = 0;
    }
    
    public long countParameters() {
        long params = 0;
        params += (long) dim * dim * 4;
        params += (long) dim * kvRank * 4;
        return params;
    }
}
