package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * CausalInterventionNetwork with True Counterfactual Reasoning.
 * 
 * Innovation:
 * - Three parallel paths: factual, counterfactual, and confounded
 * - Factual: W_factual @ x (what actually happens)
 * - Counterfactual: W_counterfactual @ x (what WOULD happen under different conditions)
 * - Confounded: original path (with spurious correlations)
 * - Output = factual + (counterfactual - confounded) * intervention_strength
 * - Simplified implementation of Pearl's do-calculus
 * - New method: counterfactualReason(float[] x, float[] counterfactualInput)
 */
public class CausalInterventionNetwork {

    private final int dModel;
    private final float[][] interventionWeights;
    private final float[][] confounderWeights;
    private final float[] interventionThreshold;
    
    // Counterfactual reasoning paths
    private final float[][] factualWeights;         // W_factual
    private final float[][] counterfactualWeights;  // W_counterfactual
    private final float[] counterfactualBias;

    public CausalInterventionNetwork(int dModel) {
        this.dModel = dModel;

        Random random = new Random(42);
        this.interventionWeights = new float[dModel][dModel];
        this.confounderWeights = new float[dModel][dModel];
        this.interventionThreshold = new float[dModel];
        
        // Counterfactual paths
        this.factualWeights = new float[dModel][dModel];
        this.counterfactualWeights = new float[dModel][dModel];
        this.counterfactualBias = new float[dModel];

        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                interventionWeights[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                confounderWeights[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                factualWeights[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                counterfactualWeights[i][j] = (random.nextFloat() - 0.5f) * 0.08f;
            }
            interventionThreshold[i] = 0.3f + random.nextFloat() * 0.4f;
            counterfactualBias[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
    }

    /**
     * Forward pass with three parallel causal paths.
     * Output = factual + (counterfactual - confounded) * intervention_strength
     */
    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                // Path 1: Factual - what actually happens
                float[] factual = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float sum = 0f;
                    for (int j = 0; j < dModel; j++) {
                        sum += factualWeights[i][j] * xSlice[j];
                    }
                    factual[i] = sum;
                }

                // Path 2: Confounded - the biased path (spurious correlations)
                float[] confounded = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float sum = 0f;
                    for (int j = 0; j < dModel; j++) {
                        sum += confounderWeights[i][j] * xSlice[j];
                    }
                    confounded[i] = sum;
                }

                // Path 3: Counterfactual - what WOULD happen under intervention
                float[] counterfactual = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float sum = counterfactualBias[i];
                    for (int j = 0; j < dModel; j++) {
                        sum += counterfactualWeights[i][j] * xSlice[j];
                    }
                    counterfactual[i] = sum;
                }

                // Compute intervention strength per dimension
                float[] doOperator = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float intervention = 0f;
                    for (int j = 0; j < dModel; j++) {
                        intervention += interventionWeights[i][j] * xSlice[j];
                    }

                    float interventionStrength = (float) (1.0 / (1.0 + Math.exp(-(intervention - interventionThreshold[i]))));
                    doOperator[i] = xSlice[i] * (1 - interventionStrength) + intervention * interventionStrength;
                }

                // Causal output: factual + counterfactual adjustment
                float[] causalOutput = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float interventionStrength = (float) (1.0 / (1.0 + Math.exp(-(
                        MatrixOperations.dotProduct(interventionWeights[i], xSlice) - interventionThreshold[i]))));
                    // Output = factual + (counterfactual - confounded) * intervention_strength
                    causalOutput[i] = factual[i] + (counterfactual[i] - confounded[i]) * 0.5f * interventionStrength;
                }

                System.arraycopy(causalOutput, 0, output, xOffset, dModel);
            }
        }

        return output;
    }

    /**
     * Perform counterfactual reasoning with a specific counterfactual input.
     * 
     * @param x the factual input
     * @param counterfactualInput the counterfactual input (what-if scenario)
     * @return counterfactual output
     */
    public float[] counterfactualReason(float[] x, float[] counterfactualInput) {
        if (x.length != dModel || counterfactualInput.length != dModel) {
            throw new IllegalArgumentException("Input dimensions must match dModel");
        }

        float[] factual = new float[dModel];
        float[] counterfactual = new float[dModel];
        float[] confounded = new float[dModel];

        // Compute factual path on original input
        for (int i = 0; i < dModel; i++) {
            float sum = 0f;
            for (int j = 0; j < dModel; j++) {
                sum += factualWeights[i][j] * x[j];
            }
            factual[i] = sum;
        }

        // Compute counterfactual path on counterfactual input
        for (int i = 0; i < dModel; i++) {
            float sum = counterfactualBias[i];
            for (int j = 0; j < dModel; j++) {
                sum += counterfactualWeights[i][j] * counterfactualInput[j];
            }
            counterfactual[i] = sum;
        }

        // Compute confounded path on original input
        for (int i = 0; i < dModel; i++) {
            float sum = 0f;
            for (int j = 0; j < dModel; j++) {
                sum += confounderWeights[i][j] * x[j];
            }
            confounded[i] = sum;
        }

        // Intervention strength based on factual input
        float[] result = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            float intervention = 0f;
            for (int j = 0; j < dModel; j++) {
                intervention += interventionWeights[i][j] * x[j];
            }
            float interventionStrength = (float) (1.0 / (1.0 + Math.exp(-(intervention - interventionThreshold[i]))));
            result[i] = factual[i] + (counterfactual[i] - confounded[i]) * interventionStrength;
        }

        return result;
    }

    /**
     * Original forward with explicit intervention vector (backward compatible).
     */
    public float[] forwardWithCounterfactual(float[] x, float[] intervention, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] counterfactual = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float doValue = xSlice[i] * (1 - intervention[i]) + intervention[i] * 0.5f;

                    float confounder = 0f;
                    for (int j = 0; j < dModel; j++) {
                        confounder += confounderWeights[i][j] * xSlice[j];
                    }

                    counterfactual[i] = doValue - confounder;
                }

                System.arraycopy(counterfactual, 0, output, xOffset, dModel);
            }
        }

        return output;
    }

    public int getDModel() { return dModel; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] row : interventionWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : confounderWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : factualWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : counterfactualWeights) for (float f : row) dos.writeFloat(f);
        for (float f : interventionThreshold) dos.writeFloat(f);
        for (float f : counterfactualBias) dos.writeFloat(f);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dInterventionWeights = new float[dModel * dModel];
        float[] dConfounderWeights = new float[dModel * dModel];
        float[] dInterventionThreshold = new float[dModel];
        float[] dFactualWeights = new float[dModel * dModel];
        float[] dCounterfactualWeights = new float[dModel * dModel];
        float[] dCounterfactualBias = new float[dModel];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                // Recompute all paths
                float[] factual = new float[dModel];
                float[] confounded = new float[dModel];
                float[] counterfactual = new float[dModel];
                float[] intervention = new float[dModel];
                float[] interventionStrength = new float[dModel];

                for (int i = 0; i < dModel; i++) {
                    float factSum = 0f, confSum = 0f, cfSum = counterfactualBias[i], intSum = 0f;
                    for (int j = 0; j < dModel; j++) {
                        factSum += factualWeights[i][j] * xSlice[j];
                        confSum += confounderWeights[i][j] * xSlice[j];
                        cfSum += counterfactualWeights[i][j] * xSlice[j];
                        intSum += interventionWeights[i][j] * xSlice[j];
                    }
                    factual[i] = factSum;
                    confounded[i] = confSum;
                    counterfactual[i] = cfSum;
                    intervention[i] = intSum;
                    interventionStrength[i] = (float) (1.0 / (1.0 + Math.exp(-(intSum - interventionThreshold[i]))));
                }

                // Gradients
                for (int i = 0; i < dModel; i++) {
                    float dCausal = dOutSlice[i];

                    // dFactual contribution: direct + through interventionStrength
                    float dFact = dCausal;

                    // d(counterfactual - confounded) * interventionStrength
                    float cfAdjust = dCausal * 0.5f * interventionStrength[i];
                    float confAdjust = -dCausal * 0.5f * interventionStrength[i];

                    // dInterventionStrength
                    float dStrength = dCausal * 0.5f * (counterfactual[i] - confounded[i]);
                    float dsigmoid = interventionStrength[i] * (1 - interventionStrength[i]);
                    float dInterventionInput = dStrength * dsigmoid;

                    // Backprop to factual weights
                    for (int j = 0; j < dModel; j++) {
                        dFactualWeights[i * dModel + j] += xSlice[j] * dFact;
                        dInput[xOffset + j] += dFact * factualWeights[i][j];
                    }

                    // Backprop to counterfactual weights
                    for (int j = 0; j < dModel; j++) {
                        dCounterfactualWeights[i * dModel + j] += xSlice[j] * cfAdjust;
                        dInput[xOffset + j] += cfAdjust * counterfactualWeights[i][j];
                    }
                    dCounterfactualBias[i] += cfAdjust;

                    // Backprop to confounder weights
                    for (int j = 0; j < dModel; j++) {
                        dConfounderWeights[i * dModel + j] += xSlice[j] * confAdjust;
                        dInput[xOffset + j] += confAdjust * confounderWeights[i][j];
                    }

                    // Backprop through intervention
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dInterventionInput * interventionWeights[i][j];
                        dInterventionWeights[i * dModel + j] += xSlice[j] * dInterventionInput;
                    }
                    dInterventionThreshold[i] += -dStrength * dsigmoid;
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("interventionWeights", dInterventionWeights);
        result.addGradient("confounderWeights", dConfounderWeights);
        result.addGradient("interventionThreshold", dInterventionThreshold);
        result.addGradient("factualWeights", dFactualWeights);
        result.addGradient("counterfactualWeights", dCounterfactualWeights);
        result.addGradient("counterfactualBias", dCounterfactualBias);
        return result;
    }

    // Getters (backward compatible)
    public float[][] getInterventionWeights() { return interventionWeights; }
    public float[][] getConfounderWeights() { return confounderWeights; }
    public float[] getInterventionThreshold() { return interventionThreshold; }
    
    // New getters
    public float[][] getFactualWeights() { return factualWeights; }
    public float[][] getCounterfactualWeights() { return counterfactualWeights; }
    public float[] getCounterfactualBias() { return counterfactualBias; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                interventionWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                confounderWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                factualWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                counterfactualWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            interventionThreshold[i] = dis.readFloat();
        }
        for (int i = 0; i < dModel; i++) {
            counterfactualBias[i] = dis.readFloat();
        }
    }
}
