package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * PlasticityUpdateModule with three-factor learning rule and modulatory signals.
 * 
 * Innovation:
 * - Three-factor rule: ΔW = modulatory_signal * (pre @ post) where modulatory_signal is a global signal
 * - Modulatory parameter gates plasticity strength
 * - Short-term trace (STP): fast decaying trace of recent activity
 * - Long-term trace (LTP): slow accumulating trace for important patterns
 * - setModulationSignal(float signal) for external control signal
 * - Eligibility trace: e(t) = gamma * e(t-1) + pre @ post
 */
public class PlasticityUpdateModule {

    private final int dModel;
    private final float[][] plasticWeights;
    private final float[][] gradientAccumulator;
    private final int updateFreq;
    private final float learningRate;
    private final float gamma;
    private int stepCount;
    private boolean enabled;
    
    // Supervised learning mode
    private boolean supervisedMode;
    private float[][] supervisedGradientAccumulator;
    private int supervisedStepCount;
    private static final int SUPERVISED_UPDATE_FREQ = 5;

    // Three-Factor Plasticity parameters
    private float modulationSignal;             // global control signal
    private float[][] eligibilityTrace;         // e(t) = gamma * e(t-1) + pre @ post
    private final float eligibilityDecay;       // decay rate for eligibility trace
    private float[][] stpTrace;                 // Short-term trace (fast decay)
    private float[][] ltpTrace;                 // Long-term trace (slow accumulation)
    private final float stpDecay;               // fast decay (0.9)
    private final float ltpDecay;               // slow decay (0.99)
    private static final float STP_DECAY_RATE = 0.9f;
    private static final float LTP_DECAY_RATE = 0.99f;
    private static final float MODULATION_DEFAULT = 1.0f;

    public PlasticityUpdateModule(int dModel) {
        this.dModel = dModel;
        this.updateFreq = 100;
        this.learningRate = 0.001f;
        this.gamma = 0.1f;
        this.stepCount = 0;
        this.enabled = true;
        this.supervisedMode = false;
        this.supervisedStepCount = 0;
        
        // Three-factor plasticity
        this.modulationSignal = MODULATION_DEFAULT;
        this.eligibilityDecay = 0.95f;
        this.stpDecay = STP_DECAY_RATE;
        this.ltpDecay = LTP_DECAY_RATE;

        Random random = new Random(42);
        this.plasticWeights = new float[dModel][dModel];
        this.gradientAccumulator = new float[dModel][dModel];
        this.supervisedGradientAccumulator = new float[dModel][dModel];
        
        // Three-factor traces
        this.eligibilityTrace = new float[dModel][dModel];
        this.stpTrace = new float[dModel][dModel];
        this.ltpTrace = new float[dModel][dModel];

        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                plasticWeights[i][j] = (random.nextFloat() - 0.5f) * 0.01f;
            }
        }
    }

    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] y = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float sum = 0f;
                    for (int j = 0; j < dModel; j++) {
                        sum += plasticWeights[i][j] * xSlice[j];
                    }
                    y[i] = sum;
                }

                for (int j = 0; j < dModel; j++) {
                    output[xOffset + j] = xSlice[j] + 0.1f * y[j];
                }

                if (enabled && s == seqLen - 1) {
                    // Use three-factor plasticity rule
                    updateWeightsThreeFactor(xSlice, y);
                }
            }
        }

        return output;
    }

    /**
     * Three-factor plasticity update rule.
     * ΔW = modulation_signal * eligibility_trace
     * 
     * Eligibility trace: e(t) = eligibilityDecay * e(t-1) + pre @ post (Hebbian)
     * Then modulated by global control signal.
     */
    private void updateWeightsThreeFactor(float[] x, float[] y) {
        // Compute Hebbian pre @ post
        float[][] hebbian = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                hebbian[i][j] = y[i] * x[j];
            }
        }

        // Update eligibility trace: e(t) = eligibilityDecay * e(t-1) + pre @ post
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                eligibilityTrace[i][j] = eligibilityDecay * eligibilityTrace[i][j] + hebbian[i][j];
            }
        }

        // Update STP trace (fast decay - captures recent activity)
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                stpTrace[i][j] = stpDecay * stpTrace[i][j] + hebbian[i][j];
            }
        }

        // Update LTP trace (slow decay - accumulates important patterns)
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                ltpTrace[i][j] = ltpDecay * ltpTrace[i][j] + hebbian[i][j];
            }
        }

        // Oja's normalization term
        float yVar = 0f;
        for (int i = 0; i < dModel; i++) {
            yVar += y[i] * y[i];
        }
        yVar /= dModel;

        // Apply three-factor rule: ΔW = modulation * eligibility - Oja_forget
        float effectiveLR = learningRate * modulationSignal;
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                float ojaForget = gamma * plasticWeights[i][j] * yVar;
                gradientAccumulator[i][j] += eligibilityTrace[i][j] - ojaForget;
            }
        }

        stepCount++;

        if (stepCount >= updateFreq) {
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    float delta = gradientAccumulator[i][j] / stepCount;
                    plasticWeights[i][j] += effectiveLR * delta;
                    plasticWeights[i][j] = Math.max(-1f, Math.min(1f, plasticWeights[i][j]));
                    gradientAccumulator[i][j] = 0f;
                }
            }
            stepCount = 0;
        }
    }

    /**
     * Set the modulation signal (global control signal).
     * This gates the three-factor plasticity update.
     * 
     * @param signal the modulation value (typically 0.0 to 2.0, default 1.0)
     */
    public void setModulationSignal(float signal) {
        this.modulationSignal = Math.max(0f, signal);
    }

    /**
     * Get current modulation signal value.
     */
    public float getModulationSignal() {
        return modulationSignal;
    }

    /**
     * Get the eligibility trace matrix (read-only diagnostic).
     */
    public float[][] getEligibilityTrace() {
        float[][] copy = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            System.arraycopy(eligibilityTrace[i], 0, copy[i], 0, dModel);
        }
        return copy;
    }

    /**
     * Get the STP trace (short-term trace).
     */
    public float[][] getSTPTrace() {
        float[][] copy = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            System.arraycopy(stpTrace[i], 0, copy[i], 0, dModel);
        }
        return copy;
    }

    /**
     * Get the LTP trace (long-term trace).
     */
    public float[][] getLTPTrace() {
        float[][] copy = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            System.arraycopy(ltpTrace[i], 0, copy[i], 0, dModel);
        }
        return copy;
    }

    public void setSupervisedMode(boolean supervised) {
        this.supervisedMode = supervised;
        if (supervised) {
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    supervisedGradientAccumulator[i][j] = 0;
                }
            }
            supervisedStepCount = 0;
        }
    }
    
    public boolean isSupervisedMode() { return supervisedMode; }
    
    public void accumulateSupervisedGradient(float[] flatGradients) {
        if (!supervisedMode || flatGradients == null) return;
        
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                int idx = i * dModel + j;
                if (idx < flatGradients.length) {
                    supervisedGradientAccumulator[i][j] += flatGradients[idx];
                }
            }
        }
        supervisedStepCount++;
        
        if (supervisedStepCount >= SUPERVISED_UPDATE_FREQ) {
            applySupervisedGradients();
        }
    }
    
    private void applySupervisedGradients() {
        float supervisedLr = learningRate * 0.1f;
        
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                float avgGrad = supervisedGradientAccumulator[i][j] / supervisedStepCount;
                plasticWeights[i][j] += supervisedLr * avgGrad;
                plasticWeights[i][j] = Math.max(-1f, Math.min(1f, plasticWeights[i][j]));
                supervisedGradientAccumulator[i][j] = 0;
            }
        }
        supervisedStepCount = 0;
    }

    /**
     * Reset all plasticity traces (eligibility, STP, LTP).
     */
    public void resetTraces() {
        for (int i = 0; i < dModel; i++) {
            Arrays.fill(eligibilityTrace[i], 0f);
            Arrays.fill(stpTrace[i], 0f);
            Arrays.fill(ltpTrace[i], 0f);
        }
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public float[][] getWeights() {
        float[][] copy = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            System.arraycopy(plasticWeights[i], 0, copy[i], 0, dModel);
        }
        return copy;
    }

    public int getDModel() { return dModel; }
    public int getUpdateFreq() { return updateFreq; }
    public float getLearningRate() { return learningRate; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] row : plasticWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : gradientAccumulator) for (float f : row) dos.writeFloat(f);
        // Save three-factor traces
        for (float[] row : eligibilityTrace) for (float f : row) dos.writeFloat(f);
        for (float[] row : stpTrace) for (float f : row) dos.writeFloat(f);
        for (float[] row : ltpTrace) for (float f : row) dos.writeFloat(f);
        dos.writeInt(updateFreq);
        dos.writeFloat(learningRate);
        dos.writeFloat(gamma);
        dos.writeInt(stepCount);
        dos.writeBoolean(enabled);
        dos.writeFloat(modulationSignal);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dPlasticWeights = new float[dModel * dModel];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                float[] y = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float sum = 0f;
                    for (int j = 0; j < dModel; j++) {
                        sum += plasticWeights[i][j] * xSlice[j];
                    }
                    y[i] = sum;
                }

                float[] dY = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    dY[j] = dOutSlice[j] * 0.1f;
                }

                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dY[i] * plasticWeights[i][j];
                        dPlasticWeights[i * dModel + j] += xSlice[j] * dY[i];
                    }
                }

                for (int j = 0; j < dModel; j++) {
                    dInput[xOffset + j] += dOutSlice[j];
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("plasticWeights", dPlasticWeights);
        return result;
    }

    public float[][] getPlasticWeights() { return plasticWeights; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                plasticWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                gradientAccumulator[i][j] = dis.readFloat();
            }
        }
        // Load three-factor traces
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                eligibilityTrace[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                stpTrace[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                ltpTrace[i][j] = dis.readFloat();
            }
        }
        dis.readInt();
        dis.readFloat();
        dis.readFloat();
        stepCount = dis.readInt();
        enabled = dis.readBoolean();
        modulationSignal = dis.readFloat();
    }
}