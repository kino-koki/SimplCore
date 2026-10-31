package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * DynamicMemoryChannel with Differentiable Associative Memory.
 * 
 * Innovation:
 * - Separate associative memory matrix (key-value store, like NTM but simpler)
 * - Content-based addressing: compute cosine similarity between query and keys
 * - Weighted read from values based on similarity
 * - Least-used memory overwrite for writes (like DNC)
 * - Memory size: 64 slots, each of dimension dModel
 * - New method: associativeRecall(float[] query) for direct memory lookup
 */
public class DynamicMemoryChannel {

    private final int dModel;
    private final int nChannels;
    private final int memoryDim;
    private final float[][] memories;
    private final float[][] attentionWeights;
    private final float[] gateWeights;
    private final float[] decayRates;

    // Differentiable Associative Memory (NTM-like)
    private static final int ASSOC_MEM_SIZE = 64;
    private final float[][] assocKeys;      // [ASSOC_MEM_SIZE][dModel]
    private final float[][] assocValues;    // [ASSOC_MEM_SIZE][dModel]
    private final float[] assocUsage;       // [ASSOC_MEM_SIZE] usage counter for LRU eviction
    private static final float ASSOC_SHIFT = 0.01f;

    public DynamicMemoryChannel(int dModel, int nChannels, int memoryDim) {
        this.dModel = dModel;
        this.nChannels = nChannels;
        this.memoryDim = memoryDim;

        Random random = new Random(42);
        this.memories = new float[nChannels][memoryDim];
        this.attentionWeights = new float[nChannels][dModel * memoryDim];
        this.gateWeights = new float[nChannels];
        this.decayRates = new float[nChannels];

        // Initialize associative memory
        this.assocKeys = new float[ASSOC_MEM_SIZE][dModel];
        this.assocValues = new float[ASSOC_MEM_SIZE][dModel];
        this.assocUsage = new float[ASSOC_MEM_SIZE];

        for (int c = 0; c < nChannels; c++) {
            for (int i = 0; i < memoryDim; i++) {
                memories[c][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            for (int i = 0; i < dModel * memoryDim; i++) {
                attentionWeights[c][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            gateWeights[c] = 0.5f + random.nextFloat() * 0.3f;
            decayRates[c] = 0.95f + random.nextFloat() * 0.04f;
        }

        // Initialize associative memory with small random values
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            for (int j = 0; j < dModel; j++) {
                assocKeys[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                assocValues[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
            }
            assocUsage[i] = 0f;
        }
    }

    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];
        float[] residual = Arrays.copyOf(x, totalSize);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] context = new float[dModel];
                float gateSum = 0f;

                for (int c = 0; c < nChannels; c++) {
                    float[] attnWeights = attentionWeights[c];
                    float[] memory = memories[c];

                    float[] key = MatrixOperations.matmul(xSlice, attnWeights, 1, dModel, memoryDim);
                    float similarity = MatrixOperations.dotProduct(key, memory);
                    float attention = (float) Math.exp(similarity / 0.1f);

                    for (int j = 0; j < dModel && j < memoryDim; j++) {
                        context[j] += attention * memory[j];
                    }
                    gateSum += attention;
                }

                if (gateSum > 0) {
                    for (int j = 0; j < dModel; j++) {
                        context[j] /= gateSum;
                    }
                }

                // Also perform associative recall
                float[] assocRead = associativeRecall(xSlice);
                for (int j = 0; j < dModel; j++) {
                    context[j] += 0.3f * assocRead[j];
                }

                for (int j = 0; j < dModel; j++) {
                    output[xOffset + j] = residual[xOffset + j] + 0.5f * context[j];
                }

                updateMemories(xSlice);
                // Update associative memory (write if similarity is high enough)
                writeAssociativeMemory(xSlice, xSlice);
            }
        }

        return output;
    }

    /**
     * Content-based addressing: compute cosine similarity between query and keys.
     * Returns weighted read from values based on similarity.
     */
    public float[] associativeRecall(float[] query) {
        float[] readVector = new float[dModel];
        float[] similarities = new float[ASSOC_MEM_SIZE];
        float maxSim = Float.NEGATIVE_INFINITY;

        // Compute cosine similarities
        float queryNorm = MatrixOperations.norm(query);
        if (queryNorm < 1e-8f) queryNorm = 1e-8f;

        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            float keyNorm = MatrixOperations.norm(assocKeys[i]);
            if (keyNorm < 1e-8f) keyNorm = 1e-8f;
            float dot = MatrixOperations.dotProduct(query, assocKeys[i]);
            similarities[i] = dot / (queryNorm * keyNorm);
            if (similarities[i] > maxSim) maxSim = similarities[i];
        }

        // Softmax weighting
        float sumExp = 0f;
        float[] weights = new float[ASSOC_MEM_SIZE];
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            weights[i] = (float) Math.exp((similarities[i] - maxSim) / 0.1f);
            sumExp += weights[i];
        }
        if (sumExp > 0) {
            for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
                weights[i] /= sumExp;
            }
        }

        // Weighted read
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            for (int j = 0; j < dModel; j++) {
                readVector[j] += weights[i] * assocValues[i][j];
            }
        }

        return readVector;
    }

    /**
     * Least-used memory overwrite for writes (like DNC).
     * Writes to the least-used slot based on usage tracking.
     */
    private void writeAssociativeMemory(float[] key, float[] value) {
        // Find least-used slot
        int leastUsedIdx = 0;
        float minUsage = Float.MAX_VALUE;
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            if (assocUsage[i] < minUsage) {
                minUsage = assocUsage[i];
                leastUsedIdx = i;
            }
        }

        // Compute similarity to existing keys - if close enough, update existing
        float queryNorm = MatrixOperations.norm(key);
        if (queryNorm < 1e-8f) queryNorm = 1e-8f;

        int closestIdx = -1;
        float closestSim = 0.5f; // threshold
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            float keyNorm = MatrixOperations.norm(assocKeys[i]);
            if (keyNorm < 1e-8f) keyNorm = 1e-8f;
            float dot = MatrixOperations.dotProduct(key, assocKeys[i]);
            float sim = dot / (queryNorm * keyNorm);
            if (sim > closestSim) {
                closestSim = sim;
                closestIdx = i;
            }
        }

        int writeIdx;
        if (closestIdx >= 0) {
            // Update existing entry
            writeIdx = closestIdx;
        } else {
            // Write to least-used slot
            writeIdx = leastUsedIdx;
        }

        // Write with momentum
        float writeRate = 0.1f;
        for (int j = 0; j < dModel; j++) {
            assocKeys[writeIdx][j] = (1 - writeRate) * assocKeys[writeIdx][j] + writeRate * key[j];
            assocValues[writeIdx][j] = (1 - writeRate) * assocValues[writeIdx][j] + writeRate * value[j];
        }

        // Bump usage and decay all
        assocUsage[writeIdx] += 1.0f;
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            assocUsage[i] *= 0.99f;
        }
    }

    private void updateMemories(float[] x) {
        for (int c = 0; c < nChannels; c++) {
            float[] attnWeights = attentionWeights[c];
            float[] newMemory = MatrixOperations.matmul(x, attnWeights, 1, dModel, memoryDim);

            float decay = decayRates[c];
            float gate = gateWeights[c];

            for (int i = 0; i < memoryDim; i++) {
                memories[c][i] = decay * memories[c][i] + gate * newMemory[i];
            }

            float norm = MatrixOperations.norm(memories[c]);
            if (norm > 1.0f) {
                for (int i = 0; i < memoryDim; i++) {
                    memories[c][i] /= norm;
                }
            }
        }
    }

    public float[] getMemoryState(int channel) {
        if (channel < 0 || channel >= nChannels) {
            throw new IllegalArgumentException("Invalid channel index");
        }
        return Arrays.copyOf(memories[channel], memoryDim);
    }

    public void resetMemory() {
        Random random = new Random(42);
        for (int c = 0; c < nChannels; c++) {
            for (int i = 0; i < memoryDim; i++) {
                memories[c][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
        }
        // Reset associative memory
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            for (int j = 0; j < dModel; j++) {
                assocKeys[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
                assocValues[i][j] = (random.nextFloat() - 0.5f) * 0.05f;
            }
            assocUsage[i] = 0f;
        }
    }

    public int getDModel() { return dModel; }
    public int getNChannels() { return nChannels; }
    public int getMemoryDim() { return memoryDim; }
    public int getAssocMemSize() { return ASSOC_MEM_SIZE; }

    // Getters for associative memory
    public float[][] getAssocKeys() { return assocKeys; }
    public float[][] getAssocValues() { return assocValues; }
    public float[] getAssocUsage() { return assocUsage; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] m : memories) for (float f : m) dos.writeFloat(f);
        for (float[] w : attentionWeights) for (float f : w) dos.writeFloat(f);
        for (float f : gateWeights) dos.writeFloat(f);
        for (float f : decayRates) dos.writeFloat(f);
        // Save associative memory
        for (float[] k : assocKeys) for (float f : k) dos.writeFloat(f);
        for (float[] v : assocValues) for (float f : v) dos.writeFloat(f);
        for (float f : assocUsage) dos.writeFloat(f);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dAttentionWeights = new float[nChannels * dModel * memoryDim];
        float[] dGateWeights = new float[nChannels];
        float[] dDecayRates = new float[nChannels];
        float[] dAssocKeys = new float[ASSOC_MEM_SIZE * dModel];
        float[] dAssocValues = new float[ASSOC_MEM_SIZE * dModel];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                float[] context = new float[dModel];
                float gateSum = 0f;

                for (int c = 0; c < nChannels; c++) {
                    float[] attnWeights = attentionWeights[c];
                    float[] memory = memories[c];

                    float[] key = MatrixOperations.matmul(xSlice, attnWeights, 1, dModel, memoryDim);
                    float similarity = MatrixOperations.dotProduct(key, memory);
                    float attention = (float) Math.exp(similarity / 0.1f);

                    for (int j = 0; j < dModel && j < memoryDim; j++) {
                        context[j] += attention * memory[j];
                    }
                    gateSum += attention;
                }

                // Associative recall gradient
                float[] assocRead = associativeRecall(xSlice);

                float[] dContext = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    dContext[j] = dOutSlice[j] * 0.5f;
                    if (gateSum > 0) {
                        dContext[j] /= gateSum;
                    }
                }

                // Gradients through associative recall
                // Compute cosine similarity gradient
                float queryNorm = MatrixOperations.norm(xSlice);
                if (queryNorm < 1e-8f) queryNorm = 1e-8f;
                float[] dAssocRead = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    dAssocRead[j] = dContext[j] * 0.3f;
                }

                // Backprop through associative recall
                float[] dQueryAssoc = new float[dModel];
                for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
                    float keyNorm = MatrixOperations.norm(assocKeys[i]);
                    if (keyNorm < 1e-8f) keyNorm = 1e-8f;
                    float dot = MatrixOperations.dotProduct(xSlice, assocKeys[i]);
                    float sim = dot / (queryNorm * keyNorm);

                    float expSim = (float) Math.exp(sim / 0.1f);
                    // Simplified gradient: weight contribution to read
                    float weight = expSim / (gateSum * ASSOC_MEM_SIZE + 1e-8f);
                    for (int j = 0; j < dModel; j++) {
                        dQueryAssoc[j] += dAssocRead[j] * weight * assocKeys[i][j] / queryNorm;
                        dAssocKeys[i * dModel + j] += dAssocRead[j] * weight * xSlice[j] / keyNorm;
                        dAssocValues[i * dModel + j] += dAssocRead[j] * weight;
                    }
                }

                for (int c = 0; c < nChannels; c++) {
                    float[] attnWeights = attentionWeights[c];
                    float[] memory = memories[c];

                    float[] key = MatrixOperations.matmul(xSlice, attnWeights, 1, dModel, memoryDim);
                    float similarity = MatrixOperations.dotProduct(key, memory);
                    float attention = (float) Math.exp(similarity / 0.1f);

                    float[] dAttention = new float[dModel];
                    for (int j = 0; j < dModel && j < memoryDim; j++) {
                        dAttention[j] = dContext[j] * memory[j];
                    }

                    float dSimilarity = MatrixOperations.dotProduct(dAttention, memory) * attention / 0.1f;
                    float[] dKey = new float[memoryDim];
                    for (int j = 0; j < memoryDim; j++) {
                        dKey[j] = dSimilarity * memory[j];
                    }

                    float[] dXChannel = MatrixOperations.matmul(dKey, MatrixOperations.transpose(attnWeights, dModel, memoryDim), 1, memoryDim, dModel);
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dXChannel[j] + dQueryAssoc[j];
                    }

                    for (int j = 0; j < dModel * memoryDim; j++) {
                        dAttentionWeights[c * dModel * memoryDim + j] += xSlice[j / memoryDim] * dKey[j % memoryDim];
                    }

                    float[] newMemory = MatrixOperations.matmul(xSlice, attnWeights, 1, dModel, memoryDim);
                    for (int j = 0; j < memoryDim; j++) {
                        dGateWeights[c] += dContext[j] * attention * newMemory[j];
                        dDecayRates[c] += dContext[j] * attention * memories[c][j];
                    }
                }

                for (int j = 0; j < dModel; j++) {
                    dInput[xOffset + j] += dOutSlice[j];
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("attentionWeights", dAttentionWeights);
        result.addGradient("gateWeights", dGateWeights);
        result.addGradient("decayRates", dDecayRates);
        result.addGradient("assocKeys", dAssocKeys);
        result.addGradient("assocValues", dAssocValues);
        return result;
    }

    public float[][] getAttentionWeights() { return attentionWeights; }
    public float[] getGateWeights() { return gateWeights; }
    public float[] getDecayRates() { return decayRates; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int c = 0; c < nChannels; c++) {
            for (int i = 0; i < memoryDim; i++) {
                memories[c][i] = dis.readFloat();
            }
        }
        for (int c = 0; c < nChannels; c++) {
            for (int i = 0; i < dModel * memoryDim; i++) {
                attentionWeights[c][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < nChannels; i++) {
            gateWeights[i] = dis.readFloat();
        }
        for (int i = 0; i < nChannels; i++) {
            decayRates[i] = dis.readFloat();
        }
        // Load associative memory
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            for (int j = 0; j < dModel; j++) {
                assocKeys[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            for (int j = 0; j < dModel; j++) {
                assocValues[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < ASSOC_MEM_SIZE; i++) {
            assocUsage[i] = dis.readFloat();
        }
    }
}
