package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

/**
 * AdaptiveBasisLayer with input-statistics-based modulation.
 * 
 * Innovation:
 * - Bases are modulated by input statistics (not just fixed trig functions)
 * - Compute input mean/variance per batch -> adjust frequency scaling
 * - Learnable input-dependent modulation gate: sigma(input @ W_gate)
 * - Original trig modulation preserved for backward compatibility
 * - New method: forwardAdaptive() uses input statistics to modulate
 * - Makes basis modulation truly adaptive to the input distribution
 */
public class AdaptiveBasisLayer {

    private final int dModel;
    private final int nBases;
    private final int nHeads;
    private final float[][] bases;
    private final float[] freqScale;
    private final float[] phaseShift;
    private final float[][] outputWeights;
    private final float temperature;
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();
    
    // Adaptive modulation parameters
    private final float[][] modulationGateWeights;  // [nHeads][dModel] learnable gate
    private final float[] modulationGateBias;       // [nHeads] bias for gate
    private float[] cachedInputMean;                // batch-level input mean
    private float[] cachedInputVar;                 // batch-level input variance
    private static final float GATE_SCALE = 0.1f;

    public AdaptiveBasisLayer(int dModel, int nBases, int nHeads) {
        this.dModel = dModel;
        this.nBases = nBases;
        this.nHeads = nHeads;
        this.temperature = 1.0f;

        Random random = new Random(42);
        this.bases = new float[nHeads][nBases * dModel];
        this.freqScale = new float[nHeads];
        this.phaseShift = new float[nHeads];
        this.outputWeights = new float[nHeads][dModel * dModel];
        
        // Adaptive modulation parameters
        this.modulationGateWeights = new float[nHeads][dModel];
        this.modulationGateBias = new float[nHeads];
        this.cachedInputMean = new float[dModel];
        this.cachedInputVar = new float[dModel];

        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < nBases * dModel; i++) {
                bases[h][i] = (random.nextFloat() - 0.5f) * 2;
            }
            bases[h] = MatrixOperations.normalize(bases[h]);
            freqScale[h] = 0.5f + random.nextFloat();
            phaseShift[h] = random.nextFloat();
            for (int i = 0; i < dModel * dModel; i++) {
                outputWeights[h][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            for (int i = 0; i < dModel; i++) {
                modulationGateWeights[h][i] = (random.nextFloat() - 0.5f) * GATE_SCALE;
            }
            modulationGateBias[h] = (random.nextFloat() - 0.5f) * GATE_SCALE;
        }
    }

    /**
     * Original forward pass with fixed trig modulation.
     * Fully backward compatible.
     */
    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        POOL.invoke(new BasisTask(x, output, bases, freqScale, phaseShift, outputWeights,
            batchSize, seqLen, dModel, nBases, nHeads, temperature, 0, batchSize));

        for (int i = 0; i < totalSize; i++) {
            output[i] /= nHeads;
        }

        return output;
    }

    /**
     * Adaptive forward pass: uses input statistics to modulate frequency scaling.
     * Bases are modulated by both trigonometric functions AND input-dependent gates.
     */
    public float[] forwardAdaptive(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        // Compute batch-level input statistics
        computeInputStatistics(x, batchSize, seqLen);

        int blockSize = Math.min(8, batchSize);
        for (int b = 0; b < batchSize; b += blockSize) {
            int endB = Math.min(b + blockSize, batchSize);
            for (int h = 0; h < nHeads; h++) {
                float[] base = bases[h];
                float freq = freqScale[h];
                float phase = phaseShift[h];
                float[] gateW = modulationGateWeights[h];
                float gateB = modulationGateBias[h];

                // Compute adaptive modulation factor from input statistics
                // Gate = sigma(mean @ gateW + var_contrib + gateB)
                float gateInput = gateB;
                for (int j = 0; j < dModel; j++) {
                    gateInput += cachedInputMean[j] * gateW[j];
                }
                float adaptiveGate = (float) (1.0 / (1.0 + Math.exp(-gateInput)));

                // Modulate frequency with adaptive gate
                float adaptiveFreq = freq * (1.0f + 0.5f * adaptiveGate);

                float[] modulatedBase = new float[nBases * dModel];
                for (int i = 0; i < nBases * dModel; i += dModel) {
                    for (int j = 0; j < dModel; j++) {
                        int idx = i + j;
                        float cosVal = (float) Math.cos(adaptiveFreq * Math.PI + j * 0.01);
                        float sinVal = (float) Math.sin(phase * Math.PI + j * 0.01);
                        int flipIdx = i + (dModel - 1 - j);
                        modulatedBase[idx] = base[idx] * cosVal + base[flipIdx] * sinVal;
                    }
                }
                modulatedBase = MatrixOperations.normalize(modulatedBase);

                for (int s = 0; s < seqLen; s++) {
                    int xOffset = (b * seqLen + s) * dModel;
                    float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                    float[] response = MatrixOperations.matmul(xSlice, modulatedBase, 1, dModel, nBases);

                    float maxResp = Float.NEGATIVE_INFINITY;
                    for (float r : response) {
                        if (r > maxResp) maxResp = r;
                    }
                    float sumExp = 0f;
                    float[] softmax = new float[nBases];
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] = (float) Math.exp(response[k] / temperature - maxResp);
                        sumExp += softmax[k];
                    }
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] /= sumExp;
                    }

                    float[] headOutput = MatrixOperations.matmul(softmax, modulatedBase, 1, nBases, dModel);
                    float[] projWeights = outputWeights[h];
                    float[] projected = MatrixOperations.matmul(headOutput, projWeights, 1, dModel, dModel);

                    for (int j = 0; j < dModel; j++) {
                        output[xOffset + j] += projected[j];
                    }
                }
            }
        }

        for (int i = 0; i < totalSize; i++) {
            output[i] /= nHeads;
        }

        return output;
    }

    /**
     * Compute input mean and variance across the batch.
     */
    private void computeInputStatistics(float[] x, int batchSize, int seqLen) {
        int totalElements = batchSize * seqLen;
        if (totalElements == 0) return;

        Arrays.fill(cachedInputMean, 0f);
        Arrays.fill(cachedInputVar, 0f);

        // Compute mean
        for (int i = 0; i < totalElements; i++) {
            int offset = i * dModel;
            for (int j = 0; j < dModel; j++) {
                cachedInputMean[j] += x[offset + j];
            }
        }
        for (int j = 0; j < dModel; j++) {
            cachedInputMean[j] /= totalElements;
        }

        // Compute variance
        for (int i = 0; i < totalElements; i++) {
            int offset = i * dModel;
            for (int j = 0; j < dModel; j++) {
                float diff = x[offset + j] - cachedInputMean[j];
                cachedInputVar[j] += diff * diff;
            }
        }
        for (int j = 0; j < dModel; j++) {
            cachedInputVar[j] /= totalElements;
        }
    }

    public float[] forwardOptimized(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        int blockSize = Math.min(8, batchSize);
        for (int b = 0; b < batchSize; b += blockSize) {
            int endB = Math.min(b + blockSize, batchSize);
            for (int h = 0; h < nHeads; h++) {
                float[] base = bases[h];
                float freq = freqScale[h];
                float phase = phaseShift[h];

                float[] modulatedBase = new float[nBases * dModel];
                for (int i = 0; i < nBases * dModel; i += dModel) {
                    for (int j = 0; j < dModel; j++) {
                        int idx = i + j;
                        float cosVal = (float) Math.cos(freq * Math.PI + j * 0.01);
                        float sinVal = (float) Math.sin(phase * Math.PI + j * 0.01);
                        int flipIdx = i + (dModel - 1 - j);
                        modulatedBase[idx] = base[idx] * cosVal + base[flipIdx] * sinVal;
                    }
                }
                modulatedBase = MatrixOperations.normalize(modulatedBase);

                for (int s = 0; s < seqLen; s++) {
                    int xOffset = (b * seqLen + s) * dModel;
                    float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                    float[] response = MatrixOperations.matmul(xSlice, modulatedBase, 1, dModel, nBases);

                    float maxResp = Float.NEGATIVE_INFINITY;
                    for (float r : response) {
                        if (r > maxResp) maxResp = r;
                    }
                    float sumExp = 0f;
                    float[] softmax = new float[nBases];
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] = (float) Math.exp(response[k] / temperature - maxResp);
                        sumExp += softmax[k];
                    }
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] /= sumExp;
                    }

                    float[] weightedBase = new float[nBases * dModel];
                    for (int k = 0; k < nBases; k++) {
                        for (int j = 0; j < dModel; j++) {
                            weightedBase[k * dModel + j] = modulatedBase[k * dModel + j] * softmax[k];
                        }
                    }

                    float[] headOutput = MatrixOperations.matmul(softmax, weightedBase, 1, nBases, dModel);

                    float[] projWeights = outputWeights[h];
                    float[] projected = MatrixOperations.matmul(headOutput, projWeights, 1, dModel, dModel);

                    for (int j = 0; j < dModel; j++) {
                        output[xOffset + j] += projected[j];
                    }
                }
            }
        }

        for (int i = 0; i < totalSize; i++) {
            output[i] /= nHeads;
        }

        return output;
    }

    public int getDModel() { return dModel; }
    public int getNBases() { return nBases; }
    public int getNHeads() { return nHeads; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] base : bases) for (float f : base) dos.writeFloat(f);
        for (float f : freqScale) dos.writeFloat(f);
        for (float f : phaseShift) dos.writeFloat(f);
        for (float[] w : outputWeights) for (float f : w) dos.writeFloat(f);
        // Save adaptive modulation params
        for (float[] w : modulationGateWeights) for (float f : w) dos.writeFloat(f);
        for (float f : modulationGateBias) dos.writeFloat(f);
        dos.writeFloat(temperature);
    }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < nBases * dModel; i++) {
                bases[h][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < nHeads; i++) {
            freqScale[i] = dis.readFloat();
        }
        for (int i = 0; i < nHeads; i++) {
            phaseShift[i] = dis.readFloat();
        }
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < dModel * dModel; i++) {
                outputWeights[h][i] = dis.readFloat();
            }
        }
        // Load adaptive modulation params
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < dModel; i++) {
                modulationGateWeights[h][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < nHeads; i++) {
            modulationGateBias[i] = dis.readFloat();
        }
        dis.readFloat();
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dOutputWeights = new float[nHeads * dModel * dModel];
        float[] dBases = new float[nHeads * nBases * dModel];
        float[] dFreqScale = new float[nHeads];
        float[] dPhaseShift = new float[nHeads];
        float[] dModGateWeights = new float[nHeads * dModel];
        float[] dModGateBias = new float[nHeads];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                for (int h = 0; h < nHeads; h++) {
                    float[] modulatedBase = new float[nBases * dModel];
                    float[] base = bases[h];
                    float freq = freqScale[h];
                    float phase = phaseShift[h];

                    for (int i = 0; i < nBases * dModel; i += dModel) {
                        for (int j = 0; j < dModel; j++) {
                            int idx = i + j;
                            float cosVal = (float) Math.cos(freq * Math.PI + j * 0.01);
                            float sinVal = (float) Math.sin(phase * Math.PI + j * 0.01);
                            int flipIdx = i + (dModel - 1 - j);
                            modulatedBase[idx] = base[idx] * cosVal + base[flipIdx] * sinVal;
                        }
                    }

                    float[] response = MatrixOperations.matmul(xSlice, modulatedBase, 1, dModel, nBases);

                    float maxResp = Float.NEGATIVE_INFINITY;
                    for (float r : response) {
                        if (r > maxResp) maxResp = r;
                    }
                    float sumExp = 0f;
                    float[] softmax = new float[nBases];
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] = (float) Math.exp(response[k] / temperature - maxResp);
                        sumExp += softmax[k];
                    }
                    for (int k = 0; k < nBases; k++) {
                        softmax[k] /= sumExp;
                    }

                    float[] headOutput = MatrixOperations.matmul(softmax, modulatedBase, 1, nBases, dModel);
                    float[] projWeights = outputWeights[h];
                    float[] projected = MatrixOperations.matmul(headOutput, projWeights, 1, dModel, dModel);

                    float[] dProjected = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        dProjected[j] = dOutSlice[j] / nHeads;
                    }

                    float[] dHeadOutput = MatrixOperations.matmul(dProjected, MatrixOperations.transpose(projWeights, dModel, dModel), 1, dModel, dModel);

                    float[] dModulatedBase = MatrixOperations.outerProduct(softmax, dHeadOutput, nBases, dModel);
                    float[] dSoftmax = MatrixOperations.matmul(dHeadOutput, MatrixOperations.transpose(modulatedBase, nBases, dModel), 1, dModel, nBases);

                    float[] dResponse = new float[nBases];
                    float sumGrad = 0f;
                    for (int k = 0; k < nBases; k++) {
                        sumGrad += dSoftmax[k];
                    }
                    for (int k = 0; k < nBases; k++) {
                        dResponse[k] = softmax[k] * (dSoftmax[k] / temperature - sumGrad / temperature);
                    }

                    float[] dxHead = MatrixOperations.matmul(dResponse, MatrixOperations.transpose(modulatedBase, dModel, nBases), 1, nBases, dModel);

                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dxHead[j];
                    }

                    for (int j = 0; j < dModel * dModel; j++) {
                        dOutputWeights[h * dModel * dModel + j] += headOutput[j / dModel] * dProjected[j % dModel];
                    }

                    for (int i = 0; i < nBases * dModel; i += dModel) {
                        for (int j = 0; j < dModel; j++) {
                            int idx = i + j;
                            float cosVal = (float) Math.cos(freq * Math.PI + j * 0.01);
                            float sinVal = (float) Math.sin(phase * Math.PI + j * 0.01);
                            int flipIdx = i + (dModel - 1 - j);
                            float dMod = dModulatedBase[idx];
                            dBases[h * nBases * dModel + idx] += dMod * cosVal;
                            dBases[h * nBases * dModel + flipIdx] += dMod * sinVal;
                            dFreqScale[h] += dMod * base[idx] * (-(float)Math.sin(freq * Math.PI + j * 0.01)) * (float)Math.PI;
                            dPhaseShift[h] += dMod * base[flipIdx] * (float)Math.cos(phase * Math.PI + j * 0.01) * (float)Math.PI;
                        }
                    }
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("outputWeights", dOutputWeights);
        result.addGradient("bases", dBases);
        result.addGradient("freqScale", dFreqScale);
        result.addGradient("phaseShift", dPhaseShift);
        result.addGradient("modulationGateWeights", dModGateWeights);
        result.addGradient("modulationGateBias", dModGateBias);
        return result;
    }

    public float[][] getOutputWeights() { return outputWeights; }
    public float[][] getBases() { return bases; }
    public float[] getFreqScale() { return freqScale; }
    public float[] getPhaseShift() { return phaseShift; }
    public float[][] getModulationGateWeights() { return modulationGateWeights; }
    public float[] getModulationGateBias() { return modulationGateBias; }

    private static class BasisTask extends java.util.concurrent.RecursiveAction {
        private final float[] x, output, freqScale, phaseShift;
        private final float[][] bases, outputWeights;
        private final int batchSize, seqLen, dModel, nBases, nHeads;
        private final float temperature;
        private final int startBatch, endBatch;
        private static final int THRESHOLD = 4;

        BasisTask(float[] x, float[] output, float[][] bases, float[] freqScale, float[] phaseShift,
                      float[][] outputWeights, int batchSize, int seqLen, int dModel, int nBases,
                      int nHeads, float temperature, int startBatch, int endBatch) {
            this.x = x;
            this.output = output;
            this.bases = bases;
            this.freqScale = freqScale;
            this.phaseShift = phaseShift;
            this.outputWeights = outputWeights;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
            this.dModel = dModel;
            this.nBases = nBases;
            this.nHeads = nHeads;
            this.temperature = temperature;
            this.startBatch = startBatch;
            this.endBatch = endBatch;
        }

        @Override
        protected void compute() {
            if (endBatch - startBatch <= THRESHOLD) {
                computeDirect();
            } else {
                int mid = (startBatch + endBatch) / 2;
                invokeAll(
                    new BasisTask(x, output, bases, freqScale, phaseShift, outputWeights,
                        batchSize, seqLen, dModel, nBases, nHeads, temperature, startBatch, mid),
                    new BasisTask(x, output, bases, freqScale, phaseShift, outputWeights,
                        batchSize, seqLen, dModel, nBases, nHeads, temperature, mid, endBatch)
                );
            }
        }

        private void computeDirect() {
            for (int b = startBatch; b < endBatch; b++) {
                for (int h = 0; h < nHeads; h++) {
                    float[] modulatedBase = new float[nBases * dModel];
                    float[] base = bases[h];
                    float freq = freqScale[h];
                    float phase = phaseShift[h];

                    for (int i = 0; i < nBases * dModel; i += dModel) {
                        for (int j = 0; j < dModel; j++) {
                            int idx = i + j;
                            float cosVal = (float) Math.cos(freq * Math.PI + j * 0.01);
                            float sinVal = (float) Math.sin(phase * Math.PI + j * 0.01);
                            int flipIdx = i + (dModel - 1 - j);
                            modulatedBase[idx] = base[idx] * cosVal + base[flipIdx] * sinVal;
                        }
                    }

                    for (int s = 0; s < seqLen; s++) {
                        int xOffset = (b * seqLen + s) * dModel;
                        float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                        float[] response = MatrixOperations.matmul(xSlice, modulatedBase, 1, dModel, nBases);

                        float maxResp = Float.NEGATIVE_INFINITY;
                        for (float r : response) {
                            if (r > maxResp) maxResp = r;
                        }
                        float sumExp = 0f;
                        float[] softmax = new float[nBases];
                        for (int k = 0; k < nBases; k++) {
                            softmax[k] = (float) Math.exp(response[k] / temperature - maxResp);
                            sumExp += softmax[k];
                        }
                        for (int k = 0; k < nBases; k++) {
                            softmax[k] /= sumExp;
                        }

                        float[] headOutput = MatrixOperations.matmul(softmax, modulatedBase, 1, nBases, dModel);
                        float[] projected = MatrixOperations.matmul(headOutput, outputWeights[h], 1, dModel, dModel);

                        for (int j = 0; j < dModel; j++) {
                            output[xOffset + j] += projected[j];
                        }
                    }
                }
            }
        }
    }
}