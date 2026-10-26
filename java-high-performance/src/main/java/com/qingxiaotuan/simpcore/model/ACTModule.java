package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

public class ACTModule {
    private float[] haltingWeight;
    private float[] haltingBias;
    private int dim;
    private float threshold;
    
    public ACTModule(int dim, float threshold) {
        this.dim = dim;
        this.threshold = threshold;
        initWeights();
    }
    
    private void initWeights() {
        haltingWeight = new float[dim];
        haltingBias = new float[1];
        
        for (int i = 0; i < dim; i++) {
            haltingWeight[i] = (float) (Math.random() - 0.5) * 2 / dim;
        }
        haltingBias[0] = 0.0f;
    }
    
    public float computeHaltingProbability(float[] hiddenState) {
        float logit = MatrixOperations.dotProduct(haltingWeight, hiddenState) + haltingBias[0];
        return sigmoid(logit);
    }
    
    private float sigmoid(float x) {
        if (x > 0) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        } else {
            float expX = (float) Math.exp(x);
            return expX / (1.0f + expX);
        }
    }
    
    public int determineLoopCount(float[] initialState) {
        float[] h = initialState;
        float haltingCumulative = 0.0f;
        
        for (int t = 0; t < 100; t++) {
            float p = computeHaltingProbability(h);
            haltingCumulative += p;
            
            if (haltingCumulative >= threshold) {
                return t + 1;
            }
            
            h = MatrixOperations.relu(h);
        }
        
        return 100;
    }
    
    public float computePenalty(int actualLoops, float[] finalState) {
        float p = computeHaltingProbability(finalState);
        float remaining = Math.max(0, threshold - p);
        return (float) (remaining * actualLoops * 0.01);
    }
    
    public void updateParameters(float learningRate) {
        for (int i = 0; i < haltingWeight.length; i++) {
            haltingWeight[i] += (float) (Math.random() - 0.5) * learningRate;
        }
        haltingBias[0] += (float) (Math.random() - 0.5) * learningRate;
    }
}
