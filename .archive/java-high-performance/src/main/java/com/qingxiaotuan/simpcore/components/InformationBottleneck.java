package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * InformationBottleneck with Variational Information Bottleneck (VIB).
 * 
 * Innovation:
 * - Instead of hard top-K, use learned Gaussian parameters
 * - Encoder outputs mu and log_var for each dimension
 * - Reparameterization trick: z = mu + eps * exp(log_var/2)
 * - KL divergence between encoder distribution and prior N(0,I)
 * - Loss = BCE + beta * KL (beta is the trade-off parameter)
 * - computeVIBLoss() method returns the KL term
 * - Proper implementation of VIB framework (Alemi et al. 2017)
 */
public class InformationBottleneck {

    private final int dModel;
    private final float[] bottleneckThreshold;
    private final float[] importanceWeights;
    private final float compressionRatio;
    
    // Variational Information Bottleneck parameters
    private final float[][] muWeights;       // [dModel][dModel] mean encoder
    private final float[][] logVarWeights;   // [dModel][dModel] log variance encoder
    private final float[] muBias;
    private final float[] logVarBias;
    private float beta;                      // KL trade-off parameter
    private static final float BETA_DEFAULT = 1e-4f;
    
    // Cached for backward
    private float[] cachedMu;
    private float[] cachedLogVar;
    private float[] cachedEps;
    private float[] cachedZ;
    private boolean vibEnabled;

    public InformationBottleneck(int dModel, float compressionRatio) {
        this.dModel = dModel;
        this.compressionRatio = compressionRatio;
        this.beta = BETA_DEFAULT;
        this.vibEnabled = false;

        Random random = new Random(42);
        this.bottleneckThreshold = new float[dModel];
        this.importanceWeights = new float[dModel];
        
        // VIB parameters
        this.muWeights = new float[dModel][dModel];
        this.logVarWeights = new float[dModel][dModel];
        this.muBias = new float[dModel];
        this.logVarBias = new float[dModel];
        this.cachedMu = new float[dModel];
        this.cachedLogVar = new float[dModel];
        this.cachedEps = new float[dModel];
        this.cachedZ = new float[dModel];

        for (int i = 0; i < dModel; i++) {
            bottleneckThreshold[i] = 0.01f + random.nextFloat() * 0.02f;
            importanceWeights[i] = 0.5f + random.nextFloat() * 0.5f;
            muBias[i] = (random.nextFloat() - 0.5f) * 0.01f;
            logVarBias[i] = (random.nextFloat() - 0.5f) * 0.01f - 5f; // start with small variance
            for (int j = 0; j < dModel; j++) {
                muWeights[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                logVarWeights[i][j] = (random.nextFloat() - 0.5f) * 0.01f;
            }
        }
    }
    
    /**
     * Enable VIB mode. When true, forward() uses variational reparameterization.
     * When false, uses the original magnitude-based top-K selection.
     */
    public void setVIBEnabled(boolean enabled) {
        this.vibEnabled = enabled;
    }
    
    public boolean isVIBEnabled() {
        return vibEnabled;
    }
    
    /**
     * Set the beta parameter for KL trade-off.
     * Higher beta = stronger compression.
     */
    public void setBeta(float beta) {
        this.beta = Math.max(1e-6f, beta);
    }
    
    public float getBeta() {
        return beta;
    }

    /**
     * Forward pass. Uses VIB when vibEnabled is true, otherwise uses
     * original magnitude-based selection.
     */
    public float[] forward(float[] x, int batchSize, int seqLen) {
        if (vibEnabled) {
            return forwardVIB(x, batchSize, seqLen);
        }
        return forwardTopK(x, batchSize, seqLen);
    }

    /**
     * Original magnitude-based top-K feature selection (backward compatible).
     */
    private float[] forwardTopK(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        int keepDim = (int) (dModel * compressionRatio);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] importance = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    importance[i] = Math.abs(xSlice[i]) * importanceWeights[i];
                }

                int[] indices = new int[dModel];
                for (int i = 0; i < dModel; i++) {
                    indices[i] = i;
                }

                for (int i = 0; i < dModel - 1; i++) {
                    for (int j = i + 1; j < dModel; j++) {
                        if (importance[indices[j]] > importance[indices[i]]) {
                            int temp = indices[i];
                            indices[i] = indices[j];
                            indices[j] = temp;
                        }
                    }
                }

                float[] compressed = new float[dModel];
                for (int i = 0; i < keepDim; i++) {
                    int idx = indices[i];
                    compressed[idx] = xSlice[idx];
                }

                for (int i = keepDim; i < dModel; i++) {
                    int idx = indices[i];
                    if (Math.abs(xSlice[idx]) > bottleneckThreshold[idx]) {
                        compressed[idx] = xSlice[idx] * 0.1f;
                    }
                }

                System.arraycopy(compressed, 0, output, xOffset, dModel);
            }
        }

        return output;
    }

    /**
     * VIB forward: encoder outputs mu and log_var, uses reparameterization trick.
     * z = mu + eps * exp(log_var/2)
     */
    private float[] forwardVIB(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];
        Random random = new Random();

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                // Encoder: compute mu and log_var
                float[] mu = new float[dModel];
                float[] logVar = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float muSum = muBias[i];
                    float lvSum = logVarBias[i];
                    for (int j = 0; j < dModel; j++) {
                        muSum += muWeights[i][j] * xSlice[j];
                        lvSum += logVarWeights[i][j] * xSlice[j];
                    }
                    mu[i] = muSum;
                    logVar[i] = lvSum; // log(sigma^2)
                }

                // Reparameterization trick: z = mu + eps * exp(logVar/2)
                float[] eps = new float[dModel];
                float[] z = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    eps[i] = (float) random.nextGaussian();
                    z[i] = mu[i] + eps[i] * (float) Math.exp(logVar[i] * 0.5f);
                }

                // Cache for backward and KL computation
                System.arraycopy(mu, 0, cachedMu, 0, dModel);
                System.arraycopy(logVar, 0, cachedLogVar, 0, dModel);
                System.arraycopy(eps, 0, cachedEps, 0, dModel);
                System.arraycopy(z, 0, cachedZ, 0, dModel);

                System.arraycopy(z, 0, output, xOffset, dModel);
            }
        }

        return output;
    }

    /**
     * Compute the KL divergence term for VIB loss.
     * KL = 0.5 * sum(exp(logVar) + mu^2 - 1 - logVar)
     * This should be added to the total loss: loss = task_loss + beta * KL
     */
    public float computeVIBLoss() {
        if (!vibEnabled) return 0f;
        
        float kl = 0f;
        for (int i = 0; i < dModel; i++) {
            kl += (float) Math.exp(cachedLogVar[i]) + cachedMu[i] * cachedMu[i] - 1 - cachedLogVar[i];
        }
        return 0.5f * kl;
    }

    /**
     * Forward pass with differentiable masks (backward compatible).
     */
    public float[] forwardWithGradients(float[] x, int batchSize, int seqLen) {
        if (vibEnabled) {
            return forwardVIB(x, batchSize, seqLen);
        }
        
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        int keepDim = (int) (dModel * compressionRatio);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] importance = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    importance[i] = Math.abs(xSlice[i]) * importanceWeights[i];
                }

                int[] indices = new int[dModel];
                for (int i = 0; i < dModel; i++) {
                    indices[i] = i;
                }

                for (int i = 0; i < dModel - 1; i++) {
                    for (int j = i + 1; j < dModel; j++) {
                        if (importance[indices[j]] > importance[indices[i]]) {
                            int temp = indices[i];
                            indices[i] = indices[j];
                            indices[j] = temp;
                        }
                    }
                }

                float[] compressed = new float[dModel];
                float[] mask = new float[dModel];

                for (int i = 0; i < keepDim; i++) {
                    int idx = indices[i];
                    compressed[idx] = xSlice[idx];
                    mask[idx] = 1.0f;
                }

                for (int i = keepDim; i < dModel; i++) {
                    int idx = indices[i];
                    if (Math.abs(xSlice[idx]) > bottleneckThreshold[idx]) {
                        compressed[idx] = xSlice[idx] * 0.1f;
                        mask[idx] = 0.1f;
                    } else {
                        mask[idx] = 0f;
                    }
                }

                for (int j = 0; j < dModel; j++) {
                    output[xOffset + j] = compressed[j];
                }
            }
        }

        return output;
    }

    public int getDModel() { return dModel; }
    public float getCompressionRatio() { return compressionRatio; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float f : bottleneckThreshold) dos.writeFloat(f);
        for (float f : importanceWeights) dos.writeFloat(f);
        // Save VIB params
        for (float[] row : muWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : logVarWeights) for (float f : row) dos.writeFloat(f);
        for (float f : muBias) dos.writeFloat(f);
        for (float f : logVarBias) dos.writeFloat(f);
        dos.writeFloat(beta);
        dos.writeFloat(compressionRatio);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        if (vibEnabled) {
            return backwardVIB(dOutput, x, batchSize, seqLen);
        }
        return backwardTopK(dOutput, x, batchSize, seqLen);
    }

    /**
     * VIB backward pass with gradients through reparameterization.
     */
    private GradientResult backwardVIB(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dMuWeights = new float[dModel * dModel];
        float[] dLogVarWeights = new float[dModel * dModel];
        float[] dMuBias = new float[dModel];
        float[] dLogVarBias = new float[dModel];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                // Forward recompute
                float[] mu = new float[dModel];
                float[] logVar = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    float muSum = muBias[i];
                    float lvSum = logVarBias[i];
                    for (int j = 0; j < dModel; j++) {
                        muSum += muWeights[i][j] * xSlice[j];
                        lvSum += logVarWeights[i][j] * xSlice[j];
                    }
                    mu[i] = muSum;
                    logVar[i] = lvSum;
                }

                // dL/dz = dOutSlice (since output = z)
                // dL/dmu = dL/dz + dKL/dmu
                // dL/d(logVar) = dL/dz * dz/d(logVar) + dKL/d(logVar)
                // where dz/d(logVar) = eps * 0.5 * exp(logVar/2)
                // and dKL/dmu = beta * mu, dKL/d(logVar) = beta * 0.5 * (exp(logVar) - 1)

                float[] dMu = new float[dModel];
                float[] dLogVar = new float[dModel];
                
                Random random = new Random();
                for (int i = 0; i < dModel; i++) {
                    float eps = (float) random.nextGaussian();
                    float std = (float) Math.exp(logVar[i] * 0.5f);
                    
                    // Gradient from output: dz/dmu = 1, dz/d(logVar) = eps * 0.5 * exp(logVar/2)
                    dMu[i] = dOutSlice[i] + beta * mu[i];  // + KL grad
                    dLogVar[i] = dOutSlice[i] * eps * 0.5f * std + beta * 0.5f * ((float)Math.exp(logVar[i]) - 1);  // + KL grad
                }

                // Backprop to weights
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dMuWeights[i * dModel + j] += xSlice[j] * dMu[i];
                        dLogVarWeights[i * dModel + j] += xSlice[j] * dLogVar[i];
                    }
                    dMuBias[i] += dMu[i];
                    dLogVarBias[i] += dLogVar[i];
                    
                    // Input gradient
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dMu[i] * muWeights[i][j] + dLogVar[i] * logVarWeights[i][j];
                    }
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("muWeights", dMuWeights);
        result.addGradient("logVarWeights", dLogVarWeights);
        result.addGradient("muBias", dMuBias);
        result.addGradient("logVarBias", dLogVarBias);
        return result;
    }

    /**
     * Original top-K backward pass.
     */
    private GradientResult backwardTopK(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dImportanceWeights = new float[dModel];
        float[] dBottleneckThreshold = new float[dModel];

        int keepDim = (int) (dModel * compressionRatio);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                float[] importance = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    importance[i] = Math.abs(xSlice[i]) * importanceWeights[i];
                }

                int[] indices = new int[dModel];
                for (int i = 0; i < dModel; i++) {
                    indices[i] = i;
                }

                for (int i = 0; i < dModel - 1; i++) {
                    for (int j = i + 1; j < dModel; j++) {
                        if (importance[indices[j]] > importance[indices[i]]) {
                            int temp = indices[i];
                            indices[i] = indices[j];
                            indices[j] = temp;
                        }
                    }
                }

                float[] mask = new float[dModel];
                for (int i = 0; i < keepDim; i++) {
                    int idx = indices[i];
                    mask[idx] = 1.0f;
                }

                for (int i = keepDim; i < dModel; i++) {
                    int idx = indices[i];
                    if (Math.abs(xSlice[idx]) > bottleneckThreshold[idx]) {
                        mask[idx] = 0.1f;
                    } else {
                        mask[idx] = 0f;
                    }
                }

                for (int i = 0; i < dModel; i++) {
                    dInput[xOffset + i] = dOutSlice[i] * mask[i];
                }

                for (int i = 0; i < dModel; i++) {
                    float sign = xSlice[i] >= 0 ? 1f : -1f;
                    dImportanceWeights[i] += dOutSlice[i] * mask[i] * sign;
                    if (Math.abs(xSlice[i]) == bottleneckThreshold[i]) {
                        dBottleneckThreshold[i] += dOutSlice[i] * 0.1f;
                    }
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("importanceWeights", dImportanceWeights);
        result.addGradient("bottleneckThreshold", dBottleneckThreshold);
        result.addGradient("muWeights", new float[dModel * dModel]);
        result.addGradient("logVarWeights", new float[dModel * dModel]);
        result.addGradient("muBias", new float[dModel]);
        result.addGradient("logVarBias", new float[dModel]);
        return result;
    }

    public float[] getImportanceWeights() { return importanceWeights; }
    public float[] getBottleneckThreshold() { return bottleneckThreshold; }
    
    // VIB getters
    public float[][] getMuWeights() { return muWeights; }
    public float[][] getLogVarWeights() { return logVarWeights; }
    public float[] getMuBias() { return muBias; }
    public float[] getLogVarBias() { return logVarBias; }
    
    public float[] getCachedMu() { return cachedMu; }
    public float[] getCachedLogVar() { return cachedLogVar; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int i = 0; i < dModel; i++) {
            bottleneckThreshold[i] = dis.readFloat();
        }
        for (int i = 0; i < dModel; i++) {
            importanceWeights[i] = dis.readFloat();
        }
        // Load VIB params
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                muWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                logVarWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel; i++) {
            muBias[i] = dis.readFloat();
        }
        for (int i = 0; i < dModel; i++) {
            logVarBias[i] = dis.readFloat();
        }
        beta = dis.readFloat();
        dis.readFloat();
    }
}
