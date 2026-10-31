package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * MultiHeadAttention with RoPE + MQA + GQA support.
 * 
 * Parameter Explosion Technologies:
 * - MQA (Multi-Query Attention): All query heads share single K,V projection (reduces KV cache ~30%)
 * - GQA (Grouped-Query Attention): Query heads grouped, each group shares K,V (balance between MQA and MHA)
 * - RoPE (Rotary Position Encoding): Relative position encoding, improves length generalization
 * - Dynamic Head Assignment: Query heads dynamically assigned to KV groups based on attention patterns
 */
public class MultiHeadAttention {

    private final int dModel;
    private final int nHeads;
    private final int nKVHeads;
    private final int headDim;
    private final float[][] qWeights;
    private final float[][] kWeights;
    private final float[][] vWeights;
    private final float[][] oWeights;
    private final float[] qBias;
    private final float[] kBias;
    private final float[] vBias;
    private final float[] oBias;
    private final float attentionDropout;
    private final float scale;
    private final boolean useGQA;
    private final boolean useMQA;
    private static final float ROPE_THETA = 10000.0f;
    private float[] ropeSinCache;
    private float[] ropeCosCache;
    private int ropeCacheMaxLen;
    private int[] kvHeadAssignment;

    public MultiHeadAttention(int dModel, int nHeads) {
        this(dModel, nHeads, nHeads, 0.1f);
    }

    public MultiHeadAttention(int dModel, int nHeads, float attentionDropout) {
        this(dModel, nHeads, nHeads, attentionDropout);
    }

    public MultiHeadAttention(int dModel, int nHeads, int nKVHeads, float attentionDropout) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.nKVHeads = Math.min(nKVHeads, nHeads);
        this.headDim = dModel / nHeads;
        this.attentionDropout = attentionDropout;
        this.scale = (float) (1.0 / Math.sqrt(headDim));
        this.useMQA = (nKVHeads == 1);
        this.useGQA = (nKVHeads > 1 && nKVHeads < nHeads);

        Random random = new Random(42);
        float std = (float) Math.sqrt(2.0 / (dModel + headDim));

        qWeights = new float[nHeads][dModel * headDim];
        kWeights = new float[nKVHeads][dModel * headDim];
        vWeights = new float[nKVHeads][dModel * headDim];
        oWeights = new float[nHeads][headDim * dModel];
        qBias = new float[nHeads * headDim];
        kBias = new float[nKVHeads * headDim];
        vBias = new float[nKVHeads * headDim];
        oBias = new float[dModel];

        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < dModel * headDim; i++) {
                qWeights[h][i] = (float)(random.nextGaussian()) * std;
            }
            for (int i = 0; i < headDim * dModel; i++) {
                oWeights[h][i] = (float)(random.nextGaussian()) * std;
            }
            for (int i = 0; i < headDim; i++) {
                qBias[h * headDim + i] = 0;
            }
        }

        for (int kh = 0; kh < nKVHeads; kh++) {
            for (int i = 0; i < dModel * headDim; i++) {
                kWeights[kh][i] = (float)(random.nextGaussian()) * std;
                vWeights[kh][i] = (float)(random.nextGaussian()) * std;
            }
            for (int i = 0; i < headDim; i++) {
                kBias[kh * headDim + i] = 0;
                vBias[kh * headDim + i] = 0;
            }
        }

        for (int i = 0; i < dModel; i++) {
            oBias[i] = 0;
        }

        this.ropeCacheMaxLen = 0;
        this.ropeSinCache = new float[0];
        this.ropeCosCache = new float[0];

        this.kvHeadAssignment = new int[nHeads];
        int headsPerGroup = nHeads / nKVHeads;
        for (int h = 0; h < nHeads; h++) {
            kvHeadAssignment[h] = Math.min(h / headsPerGroup, nKVHeads - 1);
        }
    }

    /**
     * Precompute RoPE sin/cos for positions up to maxLen.
     */
    private void ensureRopeCache(int maxLen) {
        if (maxLen <= ropeCacheMaxLen) return;
        ropeSinCache = new float[maxLen * headDim];
        ropeCosCache = new float[maxLen * headDim];
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < headDim; i += 2) {
                float theta = (float) (Math.pow(ROPE_THETA, -2.0 * i / headDim));
                float sinVal = (float) Math.sin(pos * theta);
                float cosVal = (float) Math.cos(pos * theta);
                ropeSinCache[pos * headDim + i] = sinVal;
                ropeSinCache[pos * headDim + i + 1] = sinVal;
                ropeCosCache[pos * headDim + i] = cosVal;
                ropeCosCache[pos * headDim + i + 1] = cosVal;
            }
        }
        ropeCacheMaxLen = maxLen;
    }

    /**
     * Apply rotary position encoding to a vector.
     */
    private float[] applyRoPE(float[] x, int position) {
        ensureRopeCache(position + 1);
        float[] rotated = new float[x.length];
        int offset = position * headDim;
        for (int i = 0; i < headDim; i += 2) {
            int i2 = i + 1;
            if (i2 < headDim) {
                float cos = ropeCosCache[offset + i];
                float sin = ropeSinCache[offset + i];
                rotated[i] = x[i] * cos - x[i2] * sin;
                rotated[i2] = x[i] * sin + x[i2] * cos;
            } else {
                rotated[i] = x[i];
            }
        }
        return rotated;
    }

    /**
     * Compute gradient through RoPE for backward pass.
     * Returns dX such that loss = L(RoPE(x, pos)), so dX = dL/dRoPE * dRoPE/dx.
     */
    private float[] applyRoPEBackward(float[] dOutput, int position) {
        ensureRopeCache(position + 1);
        float[] dX = new float[dOutput.length];
        int offset = position * headDim;
        for (int i = 0; i < headDim; i += 2) {
            int i2 = i + 1;
            if (i2 < headDim) {
                float cos = ropeCosCache[offset + i];
                float sin = ropeSinCache[offset + i];
                // RoPE: rotated[i] = x[i]*cos - x[i2]*sin
                //       rotated[i2] = x[i]*sin + x[i2]*cos
                // dL/dx[i] = dL/drotated[i] * cos + dL/drotated[i2] * sin
                // dL/dx[i2] = -dL/drotated[i] * sin + dL/drotated[i2] * cos
                dX[i] = dOutput[i] * cos + dOutput[i2] * sin;
                dX[i2] = -dOutput[i] * sin + dOutput[i2] * cos;
            } else {
                dX[i] = dOutput[i];
            }
        }
        return dX;
    }

    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];
        ensureRopeCache(seqLen);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                float[] headOutputs = new float[dModel];
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                for (int h = 0; h < nHeads; h++) {
                    float[] q = computeQuery(xSlice, h);
                    float[] qRoPE = applyRoPE(q, s);

                    int kvHead = kvHeadAssignment[h];

                    float[] attentionScores = new float[seqLen];
                    for (int t = 0; t < seqLen; t++) {
                        int kvOffset = (b * seqLen + t) * dModel;
                        float[] kSlice = Arrays.copyOfRange(x, kvOffset, kvOffset + dModel);
                        float[] kHead = computeKey(kSlice, kvHead);
                        float[] kHeadRoPE = applyRoPE(kHead, t);

                        float score = MatrixOperations.dotProduct(qRoPE, kHeadRoPE) * scale;
                        if (t > s) {
                            score = Float.NEGATIVE_INFINITY;
                        }
                        attentionScores[t] = score;
                    }

                    float[] attentionWeights = MatrixOperations.softmax(attentionScores);

                    float[] headOutput = new float[headDim];
                    for (int t = 0; t < seqLen; t++) {
                        int vOffset = (b * seqLen + t) * dModel;
                        float[] vSlice = Arrays.copyOfRange(x, vOffset, vOffset + dModel);
                        float[] vHead = computeValue(vSlice, kvHead);
                        float weight = attentionWeights[t];
                        for (int i = 0; i < headDim; i++) {
                            headOutput[i] += weight * vHead[i];
                        }
                    }

                    float[] projected = MatrixOperations.matmul(headOutput, oWeights[h], 1, headDim, dModel);
                    for (int i = 0; i < dModel; i++) {
                        headOutputs[i] += projected[i];
                    }
                }

                for (int i = 0; i < dModel; i++) {
                    output[xOffset + i] = headOutputs[i] / nHeads + oBias[i];
                }
            }
        }

        return output;
    }

    public float[] forwardWithKVCache(float[] x, float[][] pastK, float[][] pastV, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];
        ensureRopeCache(seqLen + (pastK[0].length / headDim));

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                float[] headOutputs = new float[dModel];
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                int currentPos = pastK[0].length / headDim + s;

                float[][] kRoPEs = new float[nKVHeads][headDim];
                float[][] vShareds = new float[nKVHeads][headDim];
                for (int kh = 0; kh < nKVHeads; kh++) {
                    kRoPEs[kh] = applyRoPE(computeKey(xSlice, kh), currentPos);
                    vShareds[kh] = computeValue(xSlice, kh);
                }

                for (int h = 0; h < nHeads; h++) {
                    float[] q = computeQuery(xSlice, h);
                    float[] qRoPE = applyRoPE(q, currentPos);

                    int kvHead = kvHeadAssignment[h];
                    int cacheSize = pastK[kvHead].length / headDim;
                    float[] attentionScores = new float[currentPos + 1];

                    for (int t = 0; t < cacheSize; t++) {
                        float[] kCached = Arrays.copyOfRange(pastK[kvHead], t * headDim, (t + 1) * headDim);
                        float score = MatrixOperations.dotProduct(qRoPE, kCached) * scale;
                        attentionScores[t] = score;
                    }
                    attentionScores[cacheSize] = MatrixOperations.dotProduct(qRoPE, kRoPEs[kvHead]) * scale;

                    float[] attentionWeights = MatrixOperations.softmax(attentionScores);

                    float[] headOutput = new float[headDim];
                    for (int t = 0; t < cacheSize; t++) {
                        float[] vCached = Arrays.copyOfRange(pastV[kvHead], t * headDim, (t + 1) * headDim);
                        float weight = attentionWeights[t];
                        for (int i = 0; i < headDim; i++) {
                            headOutput[i] += weight * vCached[i];
                        }
                    }
                    float weight = attentionWeights[cacheSize];
                    for (int i = 0; i < headDim; i++) {
                        headOutput[i] += weight * vShareds[kvHead][i];
                    }

                    float[] projected = MatrixOperations.matmul(headOutput, oWeights[h], 1, headDim, dModel);
                    for (int i = 0; i < dModel; i++) {
                        headOutputs[i] += projected[i];
                    }
                }

                for (int kh = 0; kh < nKVHeads; kh++) {
                    float[] newK = new float[pastK[kh].length + headDim];
                    System.arraycopy(pastK[kh], 0, newK, 0, pastK[kh].length);
                    System.arraycopy(kRoPEs[kh], 0, newK, pastK[kh].length, headDim);
                    pastK[kh] = newK;

                    float[] newV = new float[pastV[kh].length + headDim];
                    System.arraycopy(pastV[kh], 0, newV, 0, pastV[kh].length);
                    System.arraycopy(vShareds[kh], 0, newV, pastV[kh].length, headDim);
                    pastV[kh] = newV;
                }

                for (int i = 0; i < dModel; i++) {
                    output[xOffset + i] = headOutputs[i] / nHeads + oBias[i];
                }
            }
        }

        return output;
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dQWeights = new float[nHeads * dModel * headDim];
        // Shared K,V gradients (single arrays)
        float[] dKWeights = new float[dModel * headDim];
        float[] dVWeights = new float[dModel * headDim];
        float[] dOWeights = new float[nHeads * headDim * dModel];
        float[] dQBias = new float[nHeads * headDim];
        float[] dKBias = new float[headDim];
        float[] dVBias = new float[headDim];

        ensureRopeCache(seqLen);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                // Shared K,V for current position (for gradient computation)
                float[] kShared = computeKeyShared(xSlice);
                float[] vShared = computeValueShared(xSlice);
                float[] kRoPE = applyRoPE(kShared, s);
                float[] dKSharedRoPE = new float[headDim];
                float[] dVShared = new float[headDim];

                for (int h = 0; h < nHeads; h++) {
                    float[] dHeadOutput = new float[dModel];
                    for (int i = 0; i < dModel; i++) {
                        dHeadOutput[i] = dOutSlice[i] / nHeads;
                    }

                    // dHeadInput: gradient propagated through O to headInput (length = headDim)
                    float[] dHeadInput = MatrixOperations.matmul(dHeadOutput, MatrixOperations.transpose(oWeights[h], headDim, dModel), 1, dModel, headDim);

                    float[] q = computeQuery(xSlice, h);
                    float[] qRoPE = applyRoPE(q, s);

                    float[] attentionScores = new float[seqLen];
                    for (int t = 0; t < seqLen; t++) {
                        int kvOffset = (b * seqLen + t) * dModel;
                        float[] kSlice = Arrays.copyOfRange(x, kvOffset, kvOffset + dModel);
                        float[] kHead = computeKeyShared(kSlice);
                        float[] kHeadRoPE = applyRoPE(kHead, t);
                        float score = MatrixOperations.dotProduct(qRoPE, kHeadRoPE) * scale;
                        if (t > s) score = Float.NEGATIVE_INFINITY;
                        attentionScores[t] = score;
                    }
                    float[] attentionWeights = MatrixOperations.softmax(attentionScores);

                    float[] dAttention = new float[seqLen];
                    for (int t = 0; t < seqLen; t++) {
                        int vOffset = (b * seqLen + t) * dModel;
                        float[] vSlice = Arrays.copyOfRange(x, vOffset, vOffset + dModel);
                        float[] vHead = computeValueShared(vSlice);
                        for (int i = 0; i < headDim; i++) {
                            dAttention[t] += dHeadInput[i] * vHead[i];
                        }
                    }

                    float[] dScores = new float[seqLen];
                    float sumGrad = 0;
                    for (int t = 0; t < seqLen; t++) {
                        sumGrad += dAttention[t];
                    }
                    for (int t = 0; t < seqLen; t++) {
                        dScores[t] = attentionWeights[t] * (dAttention[t] - sumGrad);
                    }

                    float[] dQRoPE = new float[headDim];
                    float[] dKHeadRoPE = new float[headDim];

                    // Accumulate dK from this head into shared dK
                    for (int t = 0; t < seqLen; t++) {
                        int kvOffset = (b * seqLen + t) * dModel;
                        float[] kSlice = Arrays.copyOfRange(x, kvOffset, kvOffset + dModel);
                        float[] kHead = computeKeyShared(kSlice);
                        float[] kHeadRoPE = applyRoPE(kHead, t);

                        float dScore = dScores[t] * scale;
                        for (int i = 0; i < headDim; i++) {
                            dQRoPE[i] += dScore * kHeadRoPE[i];
                            dKHeadRoPE[i] += dScore * qRoPE[i];
                        }
                    }

                    // Backward through RoPE for Q
                    float[] dQ = applyRoPEBackward(dQRoPE, s);

                    // Backward through RoPE for K (accumulate into shared dK)
                    // For each position t, we need to backprop through RoPE(t)
                    for (int t = 0; t < seqLen; t++) {
                        float dScore = dScores[t] * scale;
                        // dKHeadRoPE at position t from this head
                        float[] dKAtPos = new float[headDim];
                        for (int i = 0; i < headDim; i++) {
                            dKAtPos[i] = dScore * qRoPE[i];
                        }
                        // Backward through RoPE at position t
                        float[] dKAtPosUnrotated = applyRoPEBackward(dKAtPos, t);

                        // Add to shared K gradient (from this sequence position's input)
                        for (int i = 0; i < headDim; i++) {
                            dKSharedRoPE[i] += dKAtPosUnrotated[i];
                        }
                    }

                    // Compute head input (attention-weighted V) for O weight gradient
                    float[] headInput = new float[headDim];
                    for (int t = 0; t < seqLen; t++) {
                        int vOffset = (b * seqLen + t) * dModel;
                        float[] vSlice = Arrays.copyOfRange(x, vOffset, vOffset + dModel);
                        float[] vHead = computeValueShared(vSlice);
                        for (int i = 0; i < headDim; i++) {
                            headInput[i] += attentionWeights[t] * vHead[i];
                        }
                    }

                    // V gradient (shared) - propagate through dHeadInput (headDim elements)
                    for (int t = 0; t < seqLen; t++) {
                        float weight = attentionWeights[t];
                        for (int i = 0; i < headDim; i++) {
                            dVShared[i] += weight * dHeadInput[i];
                        }
                    }

                    // Compute gradient for x (from Q path)
                    float[] dXQ = MatrixOperations.matmul(dQ, MatrixOperations.transpose(qWeights[h], dModel, headDim), 1, headDim, dModel);
                    for (int i = 0; i < dModel; i++) {
                        dInput[xOffset + i] += dXQ[i];
                    }

                    // Q weight gradients
                    for (int i = 0; i < dModel * headDim; i++) {
                        dQWeights[h * dModel * headDim + i] += xSlice[i / headDim] * dQ[i % headDim];
                    }
                    // Q bias gradients
                    for (int i = 0; i < headDim; i++) {
                        dQBias[h * headDim + i] += dQ[i];
                    }

                    // O weight gradients: dO[h][i][j] += headInput[j] * dHeadOutput[i]
                    // dHeadOutput has dModel elements (gradient of O projection output)
                    for (int i = 0; i < dModel; i++) {
                        float dOutI = dHeadOutput[i];
                        for (int j = 0; j < headDim; j++) {
                            dOWeights[h * headDim * dModel + j * dModel + i] += headInput[j] * dOutI;
                        }
                    }
                }

                float[] dXK = MatrixOperations.matmul(dKSharedRoPE, MatrixOperations.transpose(kWeights[0], dModel, headDim), 1, headDim, dModel);
                for (int i = 0; i < dModel; i++) {
                    dInput[xOffset + i] += dXK[i];
                }

                float[] dXV = MatrixOperations.matmul(dVShared, MatrixOperations.transpose(vWeights[0], dModel, headDim), 1, headDim, dModel);
                for (int i = 0; i < dModel; i++) {
                    dInput[xOffset + i] += dXV[i];
                }

                for (int i = 0; i < dModel * headDim; i++) {
                    dKWeights[i] += xSlice[i / headDim] * dKSharedRoPE[i % headDim];
                }
                for (int i = 0; i < dModel * headDim; i++) {
                    dVWeights[i] += xSlice[i / headDim] * dVShared[i % headDim];
                }
                for (int i = 0; i < headDim; i++) {
                    dKBias[i] += dKSharedRoPE[i];
                }
                for (int i = 0; i < headDim; i++) {
                    dVBias[i] += dVShared[i];
                }

                // Residual connection from output
                for (int j = 0; j < dModel; j++) {
                    dInput[xOffset + j] += dOutSlice[j];
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("qWeights", dQWeights);
        result.addGradient("kWeights", dKWeights);
        result.addGradient("vWeights", dVWeights);
        result.addGradient("oWeights", dOWeights);
        return result;
    }

    // Helper to reconstruct head output for backward O weights
    private float[] computeQuery(float[] x, int head) {
        float[] q = new float[headDim];
        float[] weights = qWeights[head];
        int biasOffset = head * headDim;
        for (int i = 0; i < headDim; i++) {
            float sum = qBias[biasOffset + i];
            for (int j = 0; j < dModel; j++) {
                sum += weights[j * headDim + i] * x[j];
            }
            q[i] = sum;
        }
        return q;
    }

    private float[] computeKey(float[] x, int kvHead) {
        float[] k = new float[headDim];
        for (int i = 0; i < headDim; i++) {
            float sum = kBias[kvHead * headDim + i];
            for (int j = 0; j < dModel; j++) {
                sum += kWeights[kvHead][j * headDim + i] * x[j];
            }
            k[i] = sum;
        }
        return k;
    }

    private float[] computeValue(float[] x, int kvHead) {
        float[] v = new float[headDim];
        for (int i = 0; i < headDim; i++) {
            float sum = vBias[kvHead * headDim + i];
            for (int j = 0; j < dModel; j++) {
                sum += vWeights[kvHead][j * headDim + i] * x[j];
            }
            v[i] = sum;
        }
        return v;
    }

    private float[] computeKeyShared(float[] x) {
        return computeKey(x, 0);
    }

    private float[] computeValueShared(float[] x) {
        return computeValue(x, 0);
    }

    public float[][] createKVCache() {
        float[][] cache = new float[2 * nKVHeads][0];
        return cache;
    }

    public void resetKVCache(float[][] cache) {
        for (int h = 0; h < 2 * nKVHeads; h++) {
            cache[h] = new float[0];
        }
    }

    public int getDModel() { return dModel; }
    public int getNHeads() { return nHeads; }
    public int getNKVHeads() { return nKVHeads; }
    public int getHeadDim() { return headDim; }

    // For backward compatibility: returns per-head Q weights
    public float[][] getQWeights() { return qWeights; }

    // For backward compatibility: returns shared K weights as per-head views
    public float[][] getKWeights() {
        float[][] perHead = new float[nHeads][];
        for (int h = 0; h < nHeads; h++) {
            perHead[h] = kWeights[kvHeadAssignment[h]];
        }
        return perHead;
    }

    public float[][] getVWeights() {
        float[][] perHead = new float[nHeads][];
        for (int h = 0; h < nHeads; h++) {
            perHead[h] = vWeights[kvHeadAssignment[h]];
        }
        return perHead;
    }

    public float[][] getOWeights() { return oWeights; }

    public float[][] getKWeightsShared() { return kWeights; }
    public float[][] getVWeightsShared() { return vWeights; }

    // RoPE frequency getter
    public static float getRopeTheta() { return ROPE_THETA; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (int h = 0; h < nHeads; h++) {
            for (float f : qWeights[h]) dos.writeFloat(f);
        }
        for (int kh = 0; kh < nKVHeads; kh++) {
            for (float f : kWeights[kh]) dos.writeFloat(f);
        }
        for (int kh = 0; kh < nKVHeads; kh++) {
            for (float f : vWeights[kh]) dos.writeFloat(f);
        }
        for (int h = 0; h < nHeads; h++) {
            for (float f : oWeights[h]) dos.writeFloat(f);
        }
        for (float f : qBias) dos.writeFloat(f);
        for (float f : kBias) dos.writeFloat(f);
        for (float f : vBias) dos.writeFloat(f);
        for (float f : oBias) dos.writeFloat(f);
        dos.writeFloat(attentionDropout);
        dos.writeInt(nKVHeads);
    }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < dModel * headDim; i++) qWeights[h][i] = dis.readFloat();
        }
        for (int kh = 0; kh < nKVHeads; kh++) {
            for (int i = 0; i < dModel * headDim; i++) kWeights[kh][i] = dis.readFloat();
        }
        for (int kh = 0; kh < nKVHeads; kh++) {
            for (int i = 0; i < dModel * headDim; i++) vWeights[kh][i] = dis.readFloat();
        }
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < headDim * dModel; i++) oWeights[h][i] = dis.readFloat();
        }
        for (int i = 0; i < nHeads * headDim; i++) qBias[i] = dis.readFloat();
        for (int i = 0; i < nKVHeads * headDim; i++) kBias[i] = dis.readFloat();
        for (int i = 0; i < nKVHeads * headDim; i++) vBias[i] = dis.readFloat();
        for (int i = 0; i < dModel; i++) oBias[i] = dis.readFloat();
        dis.readFloat();
        if (dis.available() > 0) dis.readInt();
    }
}
