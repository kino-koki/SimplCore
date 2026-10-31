package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

public class DeepWiseLoRA {
    private int dim;
    private int rank;
    private float scaling;
    
    private float[] loraA;
    private float[] loraB;
    
    private float[] loraA_grad;
    private float[] loraB_grad;
    
    private float learningRate;
    private float[] loraA_m;
    private float[] loraA_v;
    private float[] loraB_m;
    private float[] loraB_v;
    private float beta1;
    private float beta2;
    private float epsilon;
    
    public DeepWiseLoRA(int dim, int rank) {
        this.dim = dim;
        this.rank = rank;
        this.scaling = (float) rank;
        this.learningRate = 3e-4f;
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.epsilon = 1e-8f;
        
        initWeights();
        initOptimizer();
    }
    
    private void initWeights() {
        float scale = (float) (1.0 / Math.sqrt(rank));
        loraA = new float[dim * rank];
        loraB = new float[rank * dim];
        
        for (int i = 0; i < dim * rank; i++) {
            loraA[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        for (int i = 0; i < rank * dim; i++) {
            loraB[i] = 0.0f;
        }
        
        loraA_grad = new float[dim * rank];
        loraB_grad = new float[rank * dim];
    }
    
    private void initOptimizer() {
        loraA_m = new float[dim * rank];
        loraA_v = new float[dim * rank];
        loraB_m = new float[rank * dim];
        loraB_v = new float[rank * dim];
    }
    
    public float[] apply(float[] x) {
        float[] intermediate = new float[rank];
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int i = 0; i < dim; i++) {
                sum += x[i] * loraA[i * rank + r];
            }
            intermediate[r] = sum;
        }
        
        float[] delta = new float[dim];
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                sum += intermediate[r] * loraB[r * dim + i];
            }
            delta[i] = sum * scaling;
        }
        
        float[] result = new float[dim];
        for (int i = 0; i < dim; i++) {
            result[i] = x[i] + delta[i];
        }
        
        return result;
    }
    
    public void update(float[] grad) {
        for (int i = 0; i < rank; i++) {
            for (int j = 0; j < dim; j++) {
                loraB_grad[i * dim + j] += grad[j];
            }
        }
        
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < rank; j++) {
                float gradSum = 0.0f;
                for (int k = 0; k < dim; k++) {
                    gradSum += grad[k] * loraB[j * dim + k];
                }
                loraA_grad[i * rank + j] += gradSum * scaling;
            }
        }
    }
    
    public void step() {
        adamStep(loraA, loraA_grad, loraA_m, loraA_v);
        adamStep(loraB, loraB_grad, loraB_m, loraB_v);
        
        for (int i = 0; i < loraA_grad.length; i++) {
            loraA_grad[i] = 0.0f;
        }
        for (int i = 0; i < loraB_grad.length; i++) {
            loraB_grad[i] = 0.0f;
        }
    }
    
    private void adamStep(float[] params, float[] grads, float[] m, float[] v) {
        for (int i = 0; i < params.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
            
            float mHat = m[i] / (1 - beta1);
            float vHat = v[i] / (1 - beta2);
            
            params[i] -= learningRate * mHat / (float) (Math.sqrt(vHat) + epsilon);
        }
    }
    
    public long countParameters() {
        return (long) dim * rank * 2;
    }
    
    public int getRank() {
        return rank;
    }
    
    public void setLearningRate(float lr) {
        this.learningRate = lr;
    }
    
    public float getScaling() {
        return scaling;
    }
}
