package com.qingxiaotuan.simpcore.training;

import java.util.HashMap;
import java.util.Map;

public class AdamWOptimizer {
    
    private volatile float learningRate;
    private final float weightDecay;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float gradClip;
    private LearningRateScheduler scheduler;
    
    private final Map<float[], float[]> mCache;
    private final Map<float[], float[]> vCache;
    private int step;
    
    private float[] currentGradients;
    private int gradientAccumulationSteps;
    private int accumulationCount;

    public AdamWOptimizer(float learningRate) {
        this(learningRate, 0.01f, 0.9f, 0.999f, 1e-8f, 1.0f);
    }

    public AdamWOptimizer(float learningRate, float weightDecay) {
        this(learningRate, weightDecay, 0.9f, 0.999f, 1e-8f, 1.0f);
    }

    public AdamWOptimizer(float learningRate, float weightDecay, float beta1, float beta2) {
        this(learningRate, weightDecay, beta1, beta2, 1e-8f, 1.0f);
    }

    public AdamWOptimizer(float learningRate, float weightDecay, float beta1, float beta2, float eps, float gradClip) {
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.gradClip = gradClip;
        this.mCache = new HashMap<>();
        this.vCache = new HashMap<>();
        this.step = 0;
        this.gradientAccumulationSteps = 1;
        this.accumulationCount = 0;
    }

    public void setScheduler(LearningRateScheduler scheduler) {
        this.scheduler = scheduler;
    }

    public void setGradientAccumulationSteps(int steps) {
        this.gradientAccumulationSteps = Math.max(1, steps);
    }

    public void accumulateGradients(float[] gradients) {
        if (currentGradients == null || currentGradients.length != gradients.length) {
            currentGradients = new float[gradients.length];
        }
        for (int i = 0; i < gradients.length; i++) {
            currentGradients[i] += gradients[i];
        }
        accumulationCount++;
    }

    public boolean isAccumulationReady() {
        return accumulationCount >= gradientAccumulationSteps;
    }

    public void resetAccumulation() {
        if (currentGradients != null) {
            for (int i = 0; i < currentGradients.length; i++) {
                currentGradients[i] = 0;
            }
        }
        accumulationCount = 0;
    }

    public void update(float[] weights, float[] gradients) {
        step++;
        float lr = getCurrentLearningRate();
        
        float[] gradToUse = gradients;
        if (gradientAccumulationSteps > 1) {
            gradToUse = new float[gradients.length];
            for (int i = 0; i < gradients.length; i++) {
                gradToUse[i] = gradients[i] / gradientAccumulationSteps;
            }
        }

        float norm = 0;
        for (float g : gradToUse) {
            norm += g * g;
        }
        norm = (float) Math.sqrt(norm);
        float scale = 1.0f;
        if (norm > gradClip) {
            scale = gradClip / norm;
        }

        float[] m = mCache.computeIfAbsent(weights, k -> new float[weights.length]);
        float[] v = vCache.computeIfAbsent(weights, k -> new float[weights.length]);

        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < weights.length; i++) {
            float grad = gradToUse[i] * scale;
            
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;
            
            float update = mHat / ((float) Math.sqrt(vHat) + eps);
            
            if (weightDecay > 0) {
                weights[i] -= lr * weightDecay * weights[i];
            }
            weights[i] -= lr * update;
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
        }

        if (scheduler != null) {
            scheduler.step();
        }
    }

    public void updateWithAccumulated(float[] weights) {
        if (currentGradients == null) {
            return;
        }
        
        step++;
        float lr = getCurrentLearningRate();
        
        float norm = 0;
        for (float g : currentGradients) {
            norm += g * g;
        }
        norm = (float) Math.sqrt(norm);
        float scale = 1.0f;
        if (norm > gradClip) {
            scale = gradClip / norm;
        }

        float[] m = mCache.computeIfAbsent(weights, k -> new float[weights.length]);
        float[] v = vCache.computeIfAbsent(weights, k -> new float[weights.length]);
        
        float effectiveBatch = accumulationCount;

        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < weights.length; i++) {
            float grad = (currentGradients[i] / effectiveBatch) * scale;
            
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;
            
            float update = mHat / ((float) Math.sqrt(vHat) + eps);
            
            if (weightDecay > 0) {
                weights[i] -= lr * weightDecay * weights[i];
            }
            weights[i] -= lr * update;
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
        }

        resetAccumulation();
        
        if (scheduler != null) {
            scheduler.step();
        }
    }

    public void update(float[][] weights, float[][] gradients) {
        for (int i = 0; i < weights.length; i++) {
            update(weights[i], gradients[i]);
        }
    }

    public void update(float[] weights, float[] gradients, int offset, int length) {
        step++;
        float lr = getCurrentLearningRate();
        
        float[] gradToUse = new float[length];
        for (int i = 0; i < length; i++) {
            gradToUse[i] = gradients[offset + i];
        }
        
        if (gradientAccumulationSteps > 1) {
            for (int i = 0; i < length; i++) {
                gradToUse[i] /= gradientAccumulationSteps;
            }
        }

        float norm = 0;
        for (float g : gradToUse) {
            norm += g * g;
        }
        norm = (float) Math.sqrt(norm);
        float scale = 1.0f;
        if (norm > gradClip) {
            scale = gradClip / norm;
        }

        float[] m = mCache.computeIfAbsent(weights, k -> new float[weights.length]);
        float[] v = vCache.computeIfAbsent(weights, k -> new float[weights.length]);

        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < length; i++) {
            float grad = gradToUse[i] * scale;
            
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;
            
            float update = mHat / ((float) Math.sqrt(vHat) + eps);
            
            if (weightDecay > 0) {
                weights[i] -= lr * weightDecay * weights[i];
            }
            weights[i] -= lr * update;
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
        }

        if (scheduler != null) {
            scheduler.step();
        }
    }

    public void accumulateGradients(float[] gradients, int offset, int length) {
        if (currentGradients == null || currentGradients.length < offset + length) {
            currentGradients = new float[Math.max(currentGradients != null ? currentGradients.length : 0, offset + length)];
        }
        for (int i = 0; i < length; i++) {
            currentGradients[offset + i] += gradients[offset + i];
        }
        accumulationCount++;
    }

    public float getCurrentLearningRate() {
        if (scheduler != null) {
            return scheduler.getLearningRate();
        }
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getBaseLearningRate() {
        return learningRate;
    }

    public float getWeightDecay() {
        return weightDecay;
    }

    public float getBeta1() {
        return beta1;
    }

    public float getBeta2() {
        return beta2;
    }

    public float getEps() {
        return eps;
    }

    public float getGradClip() {
        return gradClip;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public int getGradientAccumulationSteps() {
        return gradientAccumulationSteps;
    }

    public void clearCache() {
        mCache.clear();
        vCache.clear();
    }

    public int getCacheSize() {
        return mCache.size();
    }
}
