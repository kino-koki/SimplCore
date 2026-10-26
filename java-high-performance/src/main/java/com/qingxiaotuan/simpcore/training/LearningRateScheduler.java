package com.qingxiaotuan.simpcore.training;

public abstract class LearningRateScheduler {

    protected float baseLearningRate;
    protected int currentStep;

    public LearningRateScheduler(float baseLearningRate) {
        this.baseLearningRate = baseLearningRate;
        this.currentStep = 0;
    }

    public abstract float getLearningRate();

    public void step() {
        currentStep++;
    }

    public void reset() {
        currentStep = 0;
    }

    public int getCurrentStep() {
        return currentStep;
    }

    public float getBaseLearningRate() {
        return baseLearningRate;
    }

    public static class ConstantScheduler extends LearningRateScheduler {
        public ConstantScheduler(float learningRate) {
            super(learningRate);
        }

        @Override
        public float getLearningRate() {
            return baseLearningRate;
        }
    }

    public static class StepDecayScheduler extends LearningRateScheduler {
        private final int stepSize;
        private final float gamma;

        public StepDecayScheduler(float baseLearningRate, int stepSize, float gamma) {
            super(baseLearningRate);
            this.stepSize = stepSize;
            this.gamma = gamma;
        }

        @Override
        public float getLearningRate() {
            int decaySteps = currentStep / stepSize;
            return baseLearningRate * (float) Math.pow(gamma, decaySteps);
        }
    }

    public static class CosineAnnealingScheduler extends LearningRateScheduler {
        private final int totalSteps;
        private final float minLearningRate;

        public CosineAnnealingScheduler(float baseLearningRate, int totalSteps) {
            this(baseLearningRate, totalSteps, 0.0f);
        }

        public CosineAnnealingScheduler(float baseLearningRate, int totalSteps, float minLearningRate) {
            super(baseLearningRate);
            this.totalSteps = totalSteps;
            this.minLearningRate = minLearningRate;
        }

        @Override
        public float getLearningRate() {
            if (currentStep >= totalSteps) {
                return minLearningRate;
            }
            float progress = (float) currentStep / totalSteps;
            float cosineDecay = 0.5f * (1 + (float) Math.cos(Math.PI * progress));
            return minLearningRate + (baseLearningRate - minLearningRate) * cosineDecay;
        }
    }

    public static class LinearWarmupScheduler extends LearningRateScheduler {
        private final int warmupSteps;
        private final LearningRateScheduler innerScheduler;

        public LinearWarmupScheduler(float baseLearningRate, int warmupSteps, LearningRateScheduler innerScheduler) {
            super(baseLearningRate);
            this.warmupSteps = warmupSteps;
            this.innerScheduler = innerScheduler;
        }

        @Override
        public float getLearningRate() {
            if (currentStep < warmupSteps) {
                return baseLearningRate * ((float) currentStep / warmupSteps);
            }
            return innerScheduler.getLearningRate();
        }

        @Override
        public void step() {
            super.step();
            if (currentStep >= warmupSteps) {
                innerScheduler.step();
            }
        }

        @Override
        public void reset() {
            super.reset();
            innerScheduler.reset();
        }
    }

    public static class CosineAnnealingWarmRestarts extends LearningRateScheduler {
        private final int T_0;
        private final int T_mult;
        private final float minLearningRate;

        public CosineAnnealingWarmRestarts(float baseLearningRate, int T_0) {
            this(baseLearningRate, T_0, 1, 0.0f);
        }

        public CosineAnnealingWarmRestarts(float baseLearningRate, int T_0, int T_mult, float minLearningRate) {
            super(baseLearningRate);
            this.T_0 = T_0;
            this.T_mult = T_mult;
            this.minLearningRate = minLearningRate;
        }

        @Override
        public float getLearningRate() {
            int T_i = T_0;
            int current = currentStep;
            
            while (current >= T_i) {
                current -= T_i;
                T_i *= T_mult;
            }
            
            float progress = (float) current / T_i;
            float cosineDecay = 0.5f * (1 + (float) Math.cos(Math.PI * progress));
            return minLearningRate + (baseLearningRate - minLearningRate) * cosineDecay;
        }
    }

    public static class ExponentialDecayScheduler extends LearningRateScheduler {
        private final float decayRate;
        private final int decaySteps;

        public ExponentialDecayScheduler(float baseLearningRate, float decayRate, int decaySteps) {
            super(baseLearningRate);
            this.decayRate = decayRate;
            this.decaySteps = decaySteps;
        }

        @Override
        public float getLearningRate() {
            return baseLearningRate * (float) Math.exp(-decayRate * (currentStep / (float) decaySteps));
        }
    }
    
    public static class PolynomialDecayScheduler extends LearningRateScheduler {
        private final int totalSteps;
        private final float power;
        private final float endLearningRate;
        
        public PolynomialDecayScheduler(float baseLearningRate, int totalSteps) {
            this(baseLearningRate, totalSteps, 1.0f, 0.0f);
        }
        
        public PolynomialDecayScheduler(float baseLearningRate, int totalSteps, float power, float endLearningRate) {
            super(baseLearningRate);
            this.totalSteps = totalSteps;
            this.power = power;
            this.endLearningRate = endLearningRate;
        }
        
        @Override
        public float getLearningRate() {
            if (currentStep >= totalSteps) {
                return endLearningRate;
            }
            float progress = (float) currentStep / totalSteps;
            float decay = (float) Math.pow(1 - progress, power);
            return endLearningRate + (baseLearningRate - endLearningRate) * decay;
        }
    }
    
    public static class ReduceLROnPlateauScheduler extends LearningRateScheduler {
        private final float factor;
        private final int patience;
        private final float minLearningRate;
        private final float threshold;
        private int badSteps;
        private float bestLoss;
        
        public ReduceLROnPlateauScheduler(float baseLearningRate) {
            this(baseLearningRate, 0.1f, 10, 1e-7f, 0.0f);
        }
        
        public ReduceLROnPlateauScheduler(float baseLearningRate, float factor, int patience, float threshold, float minLearningRate) {
            super(baseLearningRate);
            this.factor = factor;
            this.patience = patience;
            this.threshold = threshold;
            this.minLearningRate = minLearningRate;
            this.badSteps = 0;
            this.bestLoss = Float.MAX_VALUE;
        }
        
        public void step(float loss) {
            currentStep++;
            
            if (loss < bestLoss - threshold) {
                bestLoss = loss;
                badSteps = 0;
            } else {
                badSteps++;
                if (badSteps >= patience) {
                    float newLr = Math.max(minLearningRate, baseLearningRate * factor);
                    baseLearningRate = newLr;
                    badSteps = 0;
                }
            }
        }
        
        @Override
        public void step() {
            currentStep++;
        }
        
        @Override
        public float getLearningRate() {
            return baseLearningRate;
        }
        
        public void resetBestLoss() {
            bestLoss = Float.MAX_VALUE;
            badSteps = 0;
        }
    }
    
    public static class CyclicLR extends LearningRateScheduler {
        private final float maxLearningRate;
        private final int stepSizeUp;
        private final int stepSizeDown;
        private final float gamma;
        
        public CyclicLR(float baseLearningRate, float maxLearningRate, int stepSize) {
            this(baseLearningRate, maxLearningRate, stepSize, stepSize, 1.0f);
        }
        
        public CyclicLR(float baseLearningRate, float maxLearningRate, int stepSizeUp, int stepSizeDown, float gamma) {
            super(baseLearningRate);
            this.maxLearningRate = maxLearningRate;
            this.stepSizeUp = stepSizeUp;
            this.stepSizeDown = stepSizeDown;
            this.gamma = gamma;
        }
        
        @Override
        public float getLearningRate() {
            int cycleLength = stepSizeUp + stepSizeDown;
            int cycle = currentStep / cycleLength;
            int x = currentStep % cycleLength;
            
            float scale = (float) Math.pow(gamma, cycle);
            
            if (x < stepSizeUp) {
                float progress = (float) x / stepSizeUp;
                return (baseLearningRate + (maxLearningRate - baseLearningRate) * progress) * scale;
            } else {
                float progress = (float) (x - stepSizeUp) / stepSizeDown;
                return (maxLearningRate - (maxLearningRate - baseLearningRate) * progress) * scale;
            }
        }
    }
    
    public static class NoamScheduler extends LearningRateScheduler {
        private final int warmupSteps;
        private final int modelSize;
        
        public NoamScheduler(int modelSize, int warmupSteps) {
            super(1.0f);
            this.modelSize = modelSize;
            this.warmupSteps = warmupSteps;
        }
        
        @Override
        public float getLearningRate() {
            float factor = (float) Math.pow(modelSize, -0.5);
            float warmupFactor = (float) Math.pow(currentStep + 1, -0.5) * Math.min(
                (float) (currentStep + 1) * (float) Math.pow(warmupSteps, -1.5),
                (float) Math.pow(warmupSteps, -0.5)
            );
            return factor * warmupFactor;
        }
    }
}