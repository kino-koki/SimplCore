package com.qingxiaotuan.simpcore.training;

import java.util.HashMap;
import java.util.Map;

public class Optimizer {

    private volatile float learningRate;
    private final float weightDecay;
    private float momentum;
    private final float gradClip;
    private LearningRateScheduler scheduler;
    private final Map<float[], float[]> velocityCache;
    private float[] currentGradients;
    private int gradientAccumulationSteps;
    private int accumulationCount;

    public Optimizer(float learningRate) {
        this(learningRate, 0.0f, 0.0f, 1.0f);
    }

    public Optimizer(float learningRate, float weightDecay, float momentum) {
        this(learningRate, weightDecay, momentum, 1.0f);
    }

    public Optimizer(float learningRate, float weightDecay, float momentum, float gradClip) {
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.gradClip = gradClip;
        this.velocityCache = new HashMap<>();
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

        float[] velocity = velocityCache.computeIfAbsent(weights, k -> new float[weights.length]);

        for (int i = 0; i < weights.length; i++) {
            float grad = gradToUse[i] * scale;
            if (weightDecay > 0) {
                grad += weightDecay * weights[i];
            }
            
            if (momentum > 0) {
                velocity[i] = momentum * velocity[i] + grad;
                weights[i] -= lr * velocity[i];
            } else {
                weights[i] -= lr * grad;
            }
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
        }

    }

    /**
     * Step the learning rate scheduler exactly once.
     * This should be called ONCE per training step (not per parameter update).
     */
    public void stepScheduler() {
        if (scheduler != null) {
            scheduler.step();
        }
    }

    public void updateWithAccumulated(float[] weights) {
        if (currentGradients == null) {
            return;
        }
        
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

        float[] velocity = velocityCache.computeIfAbsent(weights, k -> new float[weights.length]);
        float effectiveBatch = accumulationCount;

        for (int i = 0; i < weights.length; i++) {
            float grad = (currentGradients[i] / effectiveBatch) * scale;
            if (weightDecay > 0) {
                grad += weightDecay * weights[i];
            }
            
            if (momentum > 0) {
                velocity[i] = momentum * velocity[i] + grad;
                weights[i] -= lr * velocity[i];
            } else {
                weights[i] -= lr * grad;
            }
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
        }

        resetAccumulation();
    }

    public void update(float[][] weights, float[][] gradients) {
        for (int i = 0; i < weights.length; i++) {
            update(weights[i], gradients[i]);
        }
    }

    public void update(float[] weights, float[] gradients, int offset, int length) {
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

        float[] velocity = velocityCache.computeIfAbsent(weights, k -> new float[weights.length]);

        for (int i = 0; i < length; i++) {
            float grad = gradToUse[i] * scale;
            if (weightDecay > 0) {
                grad += weightDecay * weights[i];
            }
            
            if (momentum > 0) {
                velocity[i] = momentum * velocity[i] + grad;
                weights[i] -= lr * velocity[i];
            } else {
                weights[i] -= lr * grad;
            }
            
            if (Float.isNaN(weights[i]) || Float.isInfinite(weights[i])) {
                weights[i] = 0;
            }
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

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    /**
     * Apply gradient clipping in-place to a gradients array.
     * @param grads the gradient values to clip (modified in-place)
     * @param maxNorm the maximum allowable norm
     */
    public static void clipGradients(float[] grads, float maxNorm) {
        float normSq = 0f;
        for (float g : grads) {
            normSq += g * g;
        }
        float norm = (float) Math.sqrt(normSq);
        if (norm > maxNorm && norm > 1e-12f) {
            float scale = maxNorm / norm;
            for (int i = 0; i < grads.length; i++) {
                grads[i] *= scale;
            }
        }
    }

    /**
     * Apply a cosine annealing schedule to the internal learning rate.
     * This creates a {@link LearningRateScheduler.CosineAnnealingScheduler}
     * with optional linear warmup and replaces the current scheduler.
     *
     * @param step        current training step
     * @param totalSteps  total number of training steps
     * @param warmupSteps number of linear warmup steps
     * @param peakLr      peak learning rate after warmup
     * @param minLr       minimum learning rate at end of decay
     */
    public void applyCosineSchedule(int step, int totalSteps, int warmupSteps, float peakLr, float minLr) {
        LearningRateScheduler cosineScheduler = new LearningRateScheduler.CosineAnnealingScheduler(
            peakLr, totalSteps - warmupSteps, minLr);
        LearningRateScheduler warmupScheduler = new LearningRateScheduler.LinearWarmupScheduler(
            peakLr, warmupSteps, cosineScheduler);
        setScheduler(warmupScheduler);
        // fast-forward to the given step
        for (int i = 0; i < step; i++) {
            warmupScheduler.step();
        }
    }

    /**
     * Apply L2 weight decay to a weight array in-place.
     * weight -= decay * weight
     *
     * @param weights the weight array to decay (modified in-place)
     * @param decay   the weight decay coefficient
     */
    public static void applyWeightDecay(float[] weights, float decay) {
        if (decay <= 0f) return;
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= decay * weights[i];
        }
    }

    public float getBaseLearningRate() {
        return learningRate;
    }

    public float getWeightDecay() {
        return weightDecay;
    }

    public float getMomentum() {
        return momentum;
    }

    public float getGradClip() {
        return gradClip;
    }

    public int getGradientAccumulationSteps() {
        return gradientAccumulationSteps;
    }

    public void clearVelocityCache() {
        velocityCache.clear();
    }

    public int getVelocityCacheSize() {
        return velocityCache.size();
    }
}