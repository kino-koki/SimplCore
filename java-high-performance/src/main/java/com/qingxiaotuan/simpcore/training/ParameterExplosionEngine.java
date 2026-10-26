package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.components.MultiHeadAttention;
import com.qingxiaotuan.simpcore.model.SimpCoreModel;

import java.util.*;

public class ParameterExplosionEngine {
    
    private final SimpCoreModel model;
    private final int expansionRank;
    private final float sparsityRatio;
    private final Random random;
    
    private Map<String, LowRankFactorization> factorizations;
    private Map<String, float[]> expansionMasks;
    private boolean expanded;
    
    private static class LowRankFactorization {
        float[][] U;
        float[][] S;
        float[][] V;
        float[][] original;
        int originalRows;
        int originalCols;
        int expandedRank;
        
        LowRankFactorization(float[][] orig, int rank) {
            this.original = orig;
            this.originalRows = orig.length;
            this.originalCols = orig[0].length;
            this.expandedRank = rank;
            initFactorization();
        }
        
        void initFactorization() {
            int rank = Math.min(expandedRank, Math.min(originalRows, originalCols));
            
            U = new float[originalRows][rank];
            S = new float[rank][rank];
            V = new float[rank][originalCols];
            
            Random rand = new Random(42);
            for (int i = 0; i < originalRows; i++) {
                for (int j = 0; j < rank; j++) {
                    U[i][j] = (float) (rand.nextGaussian() * Math.sqrt(1.0 / rank));
                }
            }
            
            for (int i = 0; i < rank; i++) {
                S[i][i] = (float) (1.0 / (1.0 + i));
            }
            
            for (int i = 0; i < rank; i++) {
                for (int j = 0; j < originalCols; j++) {
                    V[i][j] = (float) (rand.nextGaussian() * Math.sqrt(1.0 / rank));
                }
            }
            
            float[][] reconstructed = reconstruct();
            float alpha = 0.1f;
            for (int i = 0; i < originalRows; i++) {
                for (int j = 0; j < originalCols; j++) {
                    U[i][0] += alpha * (original[i][j] - reconstructed[i][j]);
                }
            }
        }
        
        float[][] reconstruct() {
            float[][] result = new float[originalRows][originalCols];
            int rank = U[0].length;
            for (int i = 0; i < originalRows; i++) {
                for (int j = 0; j < originalCols; j++) {
                    float sum = 0;
                    for (int k = 0; k < rank; k++) {
                        sum += U[i][k] * S[k][k] * V[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }
        
        float[][] getEffectiveWeights() {
            float[][] base = reconstruct();
            float[][] result = new float[originalRows][originalCols];
            for (int i = 0; i < originalRows; i++) {
                for (int j = 0; j < originalCols; j++) {
                    result[i][j] = base[i][j] + original[i][j] * 0.9f;
                }
            }
            return result;
        }
        
        long countEffectiveParams() {
            return (long) originalRows * expandedRank + (long) expandedRank * expandedRank + (long) expandedRank * originalCols;
        }
    }
    
    public ParameterExplosionEngine(SimpCoreModel model, int expansionRank, float sparsityRatio) {
        this.model = model;
        this.expansionRank = expansionRank;
        this.sparsityRatio = sparsityRatio;
        this.random = new Random(42);
        this.factorizations = new HashMap<>();
        this.expansionMasks = new HashMap<>();
        this.expanded = false;
    }
    
    public void applyParameterExplosion() {
        if (expanded) return;
        
        factorizeEmbeddingWeights();
        factorizeOutputWeights();
        factorizeAttentionWeights();
        factorizeFFNWeights();
        
        expanded = true;
        System.out.println("[ParameterExplosion] Applied | rank=" + expansionRank + 
                           " | sparse=" + sparsityRatio + " | effectiveParams=" + countEffectiveParams());
    }
    
    private void factorizeEmbeddingWeights() {
        float[] original = model.getEmbeddingWeights();
        int vocabSize = model.getVocabSize();
        int dModel = model.getDModel();
        
        float[][] origMatrix = new float[vocabSize][dModel];
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < dModel; j++) {
                origMatrix[i][j] = original[i * dModel + j];
            }
        }
        
        LowRankFactorization lrf = new LowRankFactorization(origMatrix, expansionRank);
        factorizations.put("embedding", lrf);
        
        float[] mask = new float[vocabSize * dModel];
        Arrays.fill(mask, 1.0f);
        applySparsity(mask);
        expansionMasks.put("embedding", mask);
    }
    
    private void factorizeOutputWeights() {
        float[] original = model.getOutputWeights();
        int dModel = model.getDModel();
        int vocabSize = model.getVocabSize();
        
        float[][] origMatrix = new float[dModel][vocabSize];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSize; j++) {
                origMatrix[i][j] = original[i * vocabSize + j];
            }
        }
        
        LowRankFactorization lrf = new LowRankFactorization(origMatrix, expansionRank);
        factorizations.put("output", lrf);
        
        float[] mask = new float[dModel * vocabSize];
        Arrays.fill(mask, 1.0f);
        applySparsity(mask);
        expansionMasks.put("output", mask);
    }
    
    private void factorizeAttentionWeights() {
        int nLayers = model.getNLayers();
        int dModel = model.getDModel();
        
        MultiHeadAttention[] mhaLayers = model.getMhaLayers();
        if (mhaLayers == null || mhaLayers.length == 0) return;
        
        int nHeads = mhaLayers[0].getNHeads();
        int headDim = dModel / nHeads;
        
        for (int l = 0; l < nLayers; l++) {
            MultiHeadAttention mha = mhaLayers[l];
            
            factorize2D("qWeights_l" + l, mha.getQWeights(), nHeads, dModel, headDim);
            factorize2D("kWeights_l" + l, mha.getKWeightsShared(), mha.getKWeightsShared().length, dModel, headDim);
            factorize2D("vWeights_l" + l, mha.getVWeightsShared(), mha.getVWeightsShared().length, dModel, headDim);
            factorize2D("oWeights_l" + l, mha.getOWeights(), nHeads, headDim, dModel);
        }
    }
    
    private void factorizeFFNWeights() {
        int nLayers = model.getNLayers();
        int dModel = model.getDModel();
        
        float[][] ffnWeights1 = model.getFfnWeights1();
        float[][] ffnWeights2 = model.getFfnWeights2();
        
        for (int l = 0; l < nLayers; l++) {
            factorize1D("ffnW1_l" + l, ffnWeights1[l], dModel, dModel * 4);
            factorize1D("ffnW2_l" + l, ffnWeights2[l], dModel * 4, dModel);
        }
    }
    
    private void factorize2D(String name, float[][] original, int rows, int inDim, int outDim) {
        LowRankFactorization lrf = new LowRankFactorization(original, expansionRank);
        factorizations.put(name, lrf);
        
        float[] mask = new float[rows * inDim * outDim];
        Arrays.fill(mask, 1.0f);
        applySparsity(mask);
        expansionMasks.put(name, mask);
    }
    
    private void factorize1D(String name, float[] original, int rows, int cols) {
        float[][] origMatrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                origMatrix[i][j] = original[i * cols + j];
            }
        }
        
        LowRankFactorization lrf = new LowRankFactorization(origMatrix, expansionRank);
        factorizations.put(name, lrf);
        
        float[] mask = new float[rows * cols];
        Arrays.fill(mask, 1.0f);
        applySparsity(mask);
        expansionMasks.put(name, mask);
    }
    
    private void applySparsity(float[] mask) {
        for (int i = 0; i < mask.length; i++) {
            if (random.nextFloat() < sparsityRatio) {
                mask[i] = 0;
            }
        }
    }
    
    public void applySparseActivation(float[] input, String paramName) {
        float[] mask = expansionMasks.get(paramName);
        if (mask == null) return;
        
        for (int i = 0; i < Math.min(input.length, mask.length); i++) {
            input[i] *= mask[i];
        }
    }
    
    public void pruneUnimportantWeights(float threshold) {
        for (Map.Entry<String, float[]> entry : expansionMasks.entrySet()) {
            float[] mask = entry.getValue();
            LowRankFactorization lrf = factorizations.get(entry.getKey());
            
            if (lrf != null) {
                for (int i = 0; i < lrf.U.length; i++) {
                    for (int j = 0; j < lrf.U[i].length; j++) {
                        if (Math.abs(lrf.U[i][j]) < threshold) lrf.U[i][j] = 0;
                    }
                }
                for (int i = 0; i < lrf.S.length; i++) {
                    if (Math.abs(lrf.S[i][i]) < threshold) lrf.S[i][i] = 0;
                }
                for (int i = 0; i < lrf.V.length; i++) {
                    for (int j = 0; j < lrf.V[i].length; j++) {
                        if (Math.abs(lrf.V[i][j]) < threshold) lrf.V[i][j] = 0;
                    }
                }
            }
            
            if (mask != null) {
                for (int i = 0; i < mask.length; i++) {
                    if (mask[i] == 0) mask[i] = 0;
                }
            }
        }
    }
    
    public void densifyGradients(float[] gradients, String paramName) {
        float[] mask = expansionMasks.get(paramName);
        if (mask == null) return;
        
        for (int i = 0; i < Math.min(gradients.length, mask.length); i++) {
            if (mask[i] == 0) {
                gradients[i] = 0;
            }
        }
    }
    
    public void updateLowRankFactors(String paramName, float[] gradients, float lr) {
        LowRankFactorization lrf = factorizations.get(paramName);
        if (lrf == null) return;
        
        int rows = lrf.originalRows;
        int cols = lrf.originalCols;
        int rank = lrf.expandedRank;
        
        float[][] gradMatrix = new float[rows][cols];
        int idx = 0;
        for (int i = 0; i < rows && idx < gradients.length; i++) {
            for (int j = 0; j < cols && idx < gradients.length; j++) {
                gradMatrix[i][j] = gradients[idx++];
            }
        }
        
        float[][] UUpdate = new float[rows][rank];
        float[][] VUpdate = new float[rank][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < rank; k++) {
                float sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum += gradMatrix[i][j] * lrf.V[k][j] * lrf.S[k][k];
                }
                UUpdate[i][k] = lr * sum;
            }
        }
        
        for (int k = 0; k < rank; k++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0;
                for (int i = 0; i < rows; i++) {
                    sum += lrf.U[i][k] * lrf.S[k][k] * gradMatrix[i][j];
                }
                VUpdate[k][j] = lr * sum;
            }
        }
        
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < rank; k++) {
                lrf.U[i][k] += UUpdate[i][k];
            }
        }
        for (int k = 0; k < rank; k++) {
            for (int j = 0; j < cols; j++) {
                lrf.V[k][j] += VUpdate[k][j];
            }
        }
    }
    
    public long countEffectiveParams() {
        long count = 0;
        for (LowRankFactorization lrf : factorizations.values()) {
            count += lrf.countEffectiveParams();
        }
        return count;
    }
    
    public long countNonZeroParams() {
        long count = 0;
        for (float[] mask : expansionMasks.values()) {
            for (float m : mask) {
                if (m > 0) count++;
            }
        }
        return count;
    }
    
    public float getEffectiveExpansionRatio() {
        long total = countEffectiveParams();
        if (total == 0) return 0;
        return (float) countNonZeroParams() / total;
    }
    
    public boolean isExpanded() {
        return expanded;
    }
    
    public Map<String, LowRankFactorization> getFactorizations() {
        return factorizations;
    }
    
    public Map<String, float[]> getExpansionMasks() {
        return expansionMasks;
    }
    
    public String getStatistics() {
        return String.format("ParameterExplosion[expanded=%s, rank=%d, sparse=%.2f, effectiveParams=%d, nonZero=%d, ratio=%.2f]",
                expanded, expansionRank, sparsityRatio, countEffectiveParams(), countNonZeroParams(), getEffectiveExpansionRatio());
    }
}