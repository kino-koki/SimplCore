package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;

import java.io.*;
import java.util.*;

public class ZYLAttentionSystem {

    public static class ZYLConfig {
        public int maxContextLength = 2_000_000;
        public int nHeads = 32;
        public int dModel = 2048;
        public int headDim = 64;
        public int memoryLayers = 4;
        public int memoryBlockSize = 1024;
        public double memoryRetention = 0.98;
        public int sparseTopK = 64;
        public int slidingWindowSize = 4096;
        public boolean enableHierarchicalMemory = true;
        public boolean enableSparseAttention = true;
        public boolean enableRoPE = true;
        public boolean enableKVCache = true;
        public float dropoutRate = 0.1f;

        public void print() {
            System.out.println("=== ZYL Attention Configuration ===");
            System.out.println("Max context length: " + maxContextLength);
            System.out.println("Number of heads: " + nHeads);
            System.out.println("Model dimension: " + dModel);
            System.out.println("Head dimension: " + headDim);
            System.out.println("Memory layers: " + memoryLayers);
            System.out.println("Memory block size: " + memoryBlockSize);
            System.out.println("Memory retention: " + String.format("%.2f%%", memoryRetention * 100));
            System.out.println("Sparse top-K: " + sparseTopK);
            System.out.println("Sliding window: " + slidingWindowSize);
            System.out.println("RoPE enabled: " + enableRoPE);
            System.out.println("KV Cache enabled: " + enableKVCache);
            System.out.println("Dropout rate: " + dropoutRate);
        }
    }

    public static class ZYLResult {
        public float[] output;
        public float[][] attentionWeights;
        public float[][][] qMatrices;
        public float[][][] kMatrices;
        public float[][][] vMatrices;
        public float[][][] kCache;
        public float[][][] vCache;
        public int activeTokens;
        public double memoryLoss;

        public void clear() {
            output = null;
            attentionWeights = null;
            qMatrices = null;
            kMatrices = null;
            vMatrices = null;
            kCache = null;
            vCache = null;
        }
    }

    public static class HierarchicalMemory {
        public float[][] layers;
        public int[] layerSizes;
        public float[] importanceScores;
        public float[] memoryWeights;
        public int totalCapacity;
        public int currentPosition;

        public HierarchicalMemory(int layers, int baseSize, int dim) {
            this.layers = new float[layers][];
            this.layerSizes = new int[layers];
            this.importanceScores = new float[layers];
            this.memoryWeights = new float[layers];
            this.currentPosition = 0;

            int total = 0;
            for (int i = 0; i < layers; i++) {
                int size = baseSize * (int) Math.pow(2, i);
                this.layers[i] = new float[size * dim];
                this.layerSizes[i] = size;
                this.importanceScores[i] = 1.0f / (i + 1);
                this.memoryWeights[i] = (float) (1.0 / Math.pow(1.5, i));
                total += size;
            }
            this.totalCapacity = total;
        }

        public void update(float[] newContent, int seqLen, int dim, double retention) {
            float[] importance = computeImportance(newContent, seqLen, dim);

            int offset = 0;
            for (int layer = 0; layer < layers.length && offset < seqLen; layer++) {
                int take = Math.min(layerSizes[layer], seqLen - offset);
                int destIdx = (currentPosition % layerSizes[layer]) * dim;

                for (int i = 0; i < take * dim; i++) {
                    int srcIdx = offset * dim + i;
                    int memIdx = destIdx + (i % dim);
                    if (memIdx >= layers[layer].length) {
                        memIdx = memIdx % layers[layer].length;
                    }
                    layers[layer][memIdx] = layers[layer][memIdx] * (1 - (float) retention)
                            + newContent[srcIdx] * (float) retention * importance[offset + (i / dim)];
                }

                offset += take;
            }

            currentPosition += seqLen;
        }

        public float[] retrieve(int queryPosition, int dim, int k) {
            float[] retrieved = new float[k * dim];
            Arrays.fill(retrieved, 0);

            int count = 0;
            float totalWeight = 0;
            for (float w : memoryWeights) totalWeight += w;

            for (int layer = layers.length - 1; layer >= 0 && count < k; layer--) {
                int take = Math.min(layerSizes[layer], k - count);
                float weight = memoryWeights[layer] / totalWeight;

                int startIdx = ((queryPosition / layerSizes[layer]) % layerSizes[layer]) * dim;
                for (int i = 0; i < take * dim; i++) {
                    int srcIdx = (startIdx + i) % layers[layer].length;
                    retrieved[count * dim + i] = layers[layer][srcIdx] * weight;
                }
                count += take;
            }

            return retrieved;
        }

        public float[] retrieveWithAttention(float[] query, int dim, int k) {
            float[] retrieved = new float[k * dim];
            Arrays.fill(retrieved, 0);

            List<float[]> candidates = new ArrayList<>();
            List<Integer> candidateLayers = new ArrayList<>();

            for (int layer = layers.length - 1; layer >= 0; layer--) {
                int samples = Math.min(layerSizes[layer], k / layers.length);
                for (int s = 0; s < samples; s++) {
                    int idx = ((currentPosition + s) % layerSizes[layer]) * dim;
                    float[] vec = new float[dim];
                    for (int d = 0; d < dim; d++) {
                        vec[d] = layers[layer][(idx + d) % layers[layer].length];
                    }
                    candidates.add(vec);
                    candidateLayers.add(layer);
                }
            }

            if (candidates.isEmpty()) return retrieved;

            float[] scores = new float[candidates.size()];
            float maxScore = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < candidates.size(); i++) {
                scores[i] = MatrixOperations.dotProduct(query, candidates.get(i));
                if (scores[i] > maxScore) maxScore = scores[i];
            }

            float[] weights = new float[candidates.size()];
            float sumWeights = 0;
            for (int i = 0; i < candidates.size(); i++) {
                weights[i] = (float) Math.exp(scores[i] - maxScore);
                sumWeights += weights[i];
            }
            for (int i = 0; i < candidates.size(); i++) {
                weights[i] /= sumWeights;
            }

            int take = Math.min(k, candidates.size());
            Integer[] indices = new Integer[candidates.size()];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            Arrays.sort(indices, (a, b) -> Float.compare(weights[b], weights[a]));

            for (int i = 0; i < take; i++) {
                int idx = indices[i];
                float weight = weights[idx];
                float[] vec = candidates.get(idx);
                for (int d = 0; d < dim; d++) {
                    retrieved[i * dim + d] += vec[d] * weight;
                }
            }

            return retrieved;
        }

        private float[] computeImportance(float[] content, int seqLen, int dim) {
            float[] importance = new float[seqLen];
            for (int i = 0; i < seqLen; i++) {
                float norm = 0;
                for (int d = 0; d < dim; d++) {
                    norm += content[i * dim + d] * content[i * dim + d];
                }
                importance[i] = (float) Math.sqrt(norm);
            }

            float maxImportance = 0;
            for (float v : importance) {
                if (v > maxImportance) maxImportance = v;
            }
            if (maxImportance > 0) {
                for (int i = 0; i < importance.length; i++) {
                    importance[i] /= maxImportance;
                }
            }

            return importance;
        }

        public float[] backward(float[] dRetrieved, int queryPosition, int dim, int k) {
            float[] dMemory = new float[dim];
            Arrays.fill(dMemory, 0);

            float totalWeight = 0;
            for (float w : memoryWeights) totalWeight += w;

            int count = 0;
            for (int layer = layers.length - 1; layer >= 0 && count < k; layer--) {
                int take = Math.min(layerSizes[layer], k - count);
                float weight = memoryWeights[layer] / totalWeight;

                int startIdx = ((queryPosition / layerSizes[layer]) % layerSizes[layer]) * dim;
                for (int i = 0; i < take * dim; i++) {
                    int memIdx = (startIdx + i) % layers[layer].length;
                    layers[layer][memIdx] -= 0.001f * dRetrieved[count * dim + i] * weight;
                    dMemory[i % dim] += dRetrieved[count * dim + i] * weight;
                }
                count += take;
            }

            return dMemory;
        }

        public void save(DataOutputStream dos) throws IOException {
            dos.writeInt(layers.length);
            for (int i = 0; i < layers.length; i++) {
                dos.writeInt(layerSizes[i]);
                for (float v : layers[i]) {
                    dos.writeFloat(v);
                }
            }
            dos.writeFloat(currentPosition);
        }

        public void load(DataInputStream dis) throws IOException {
            int numLayers = dis.readInt();
            if (numLayers != layers.length) throw new IOException("Memory layers count mismatch");
            
            for (int i = 0; i < numLayers; i++) {
                int size = dis.readInt();
                if (size != layerSizes[i]) throw new IOException("Memory layer " + i + " size mismatch");
                
                int totalSize = size * layers[i].length / layerSizes[i];
                for (int j = 0; j < totalSize; j++) {
                    layers[i][j] = dis.readFloat();
                }
            }
            currentPosition = dis.readInt();
        }
    }

    private final ZYLConfig config;
    private HierarchicalMemory memory;

    private float[] qWeights;
    private float[] kWeights;
    private float[] vWeights;
    private float[] oWeights;
    private float[] positionBias;
    private float[] ropeSin;
    private float[] ropeCos;

    private float[][][] kvCache;
    private int cachePosition = 0;

    private int currentPosition = 0;
    private Random random = new Random(42);

    public ZYLAttentionSystem(ZYLConfig config) {
        this.config = config;
        this.memory = new HierarchicalMemory(config.memoryLayers, config.memoryBlockSize, config.headDim);
    }

    public void initialize() {
        int qkvSize = config.nHeads * config.dModel * config.headDim;
        int oSize = config.nHeads * config.headDim * config.dModel;

        qWeights = new float[qkvSize];
        kWeights = new float[qkvSize];
        vWeights = new float[qkvSize];
        oWeights = new float[oSize];
        positionBias = new float[config.nHeads * config.slidingWindowSize];

        float scale = (float) Math.sqrt(2.0 / (config.dModel + config.headDim));
        for (int i = 0; i < qkvSize; i++) {
            qWeights[i] = (float) (random.nextGaussian() * scale);
            kWeights[i] = (float) (random.nextGaussian() * scale);
            vWeights[i] = (float) (random.nextGaussian() * scale);
        }
        float oScale = (float) Math.sqrt(2.0 / (config.headDim + config.dModel));
        for (int i = 0; i < oSize; i++) {
            oWeights[i] = (float) (random.nextGaussian() * oScale);
        }

        for (int h = 0; h < config.nHeads; h++) {
            for (int i = 0; i < config.slidingWindowSize; i++) {
                positionBias[h * config.slidingWindowSize + i] = (float) (-i * i * 0.0001);
            }
        }

        if (config.enableRoPE) {
            int ropeSize = config.maxContextLength * config.headDim / 2;
            ropeSin = new float[ropeSize];
            ropeCos = new float[ropeSize];
            for (int pos = 0; pos < config.maxContextLength; pos++) {
                for (int i = 0; i < config.headDim / 2; i++) {
                    double angle = pos / Math.pow(10000, 2.0 * i / config.headDim);
                    ropeSin[pos * config.headDim / 2 + i] = (float) Math.sin(angle);
                    ropeCos[pos * config.headDim / 2 + i] = (float) Math.cos(angle);
                }
            }
        }

        if (config.enableKVCache) {
            kvCache = new float[2][config.nHeads][config.maxContextLength * config.headDim];
        }

        memory = new HierarchicalMemory(config.memoryLayers, config.memoryBlockSize, config.headDim);
    }

    public ZYLResult forward(float[] input, int batchSize, int seqLen) {
        ZYLResult result = new ZYLResult();
        int totalSize = batchSize * seqLen * config.dModel;
        float scale = (float) (1.0 / Math.sqrt(config.headDim));

        float[] output = new float[totalSize];
        Arrays.fill(output, 0);

        float[][] attentionWeights = new float[config.nHeads][batchSize * seqLen * seqLen];
        float[][][] qMatrices = new float[config.nHeads][batchSize * seqLen][config.headDim];
        float[][][] kMatrices = new float[config.nHeads][batchSize * seqLen][config.headDim];
        float[][][] vMatrices = new float[config.nHeads][batchSize * seqLen][config.headDim];

        for (int h = 0; h < config.nHeads; h++) {
            int qOffset = h * config.dModel * config.headDim;
            int kOffset = h * config.dModel * config.headDim;
            int vOffset = h * config.dModel * config.headDim;
            int oOffset = h * config.headDim * config.dModel;

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int inputOffset = (b * seqLen + s) * config.dModel;
                    float[] x = new float[config.dModel];
                    for (int d = 0; d < config.dModel; d++) {
                        x[d] = input[inputOffset + d];
                    }

                    float[] q = new float[config.headDim];
                    float[] k = new float[config.headDim];
                    float[] v = new float[config.headDim];

                    for (int i = 0; i < config.dModel; i++) {
                        for (int j = 0; j < config.headDim; j++) {
                            q[j] += x[i] * qWeights[qOffset + i * config.headDim + j];
                            k[j] += x[i] * kWeights[kOffset + i * config.headDim + j];
                            v[j] += x[i] * vWeights[vOffset + i * config.headDim + j];
                        }
                    }

                    if (config.enableRoPE) {
                        int pos = currentPosition + s;
                        applyRoPE(q, pos);
                        applyRoPE(k, pos);
                    }

                    if (config.enableKVCache) {
                        int cacheIdx = h * config.maxContextLength * config.headDim + cachePosition * config.headDim;
                        for (int j = 0; j < config.headDim; j++) {
                            kvCache[0][h][cacheIdx + j] = k[j];
                            kvCache[1][h][cacheIdx + j] = v[j];
                        }
                    }

                    qMatrices[h][b * seqLen + s] = q;
                    kMatrices[h][b * seqLen + s] = k;
                    vMatrices[h][b * seqLen + s] = v;

                    int startWindow = Math.max(0, s - config.slidingWindowSize);
                    int effectiveLen = s - startWindow + 1;

                    float[] scores = new float[effectiveLen + (config.enableHierarchicalMemory ? config.sparseTopK / 2 : 0)];
                    Arrays.fill(scores, Float.NEGATIVE_INFINITY);

                    for (int t = startWindow; t <= s; t++) {
                        int tIdx = t - startWindow;
                        float score = MatrixOperations.dotProduct(q, kMatrices[h][b * seqLen + t]) * scale;
                        int relPos = s - t;
                        if (relPos >= 0 && relPos < config.slidingWindowSize) {
                            score += positionBias[h * config.slidingWindowSize + relPos];
                        }
                        scores[tIdx] = score;
                    }

                    float[] memoryK = null;
                    float[] memoryV = null;
                    if (config.enableHierarchicalMemory && currentPosition > config.slidingWindowSize) {
                        int memoryTokens = config.sparseTopK / 2;
                        memoryK = memory.retrieve(currentPosition + s, config.headDim, memoryTokens);
                        memoryV = memory.retrieve(currentPosition + s + 1, config.headDim, memoryTokens);

                        for (int m = 0; m < memoryTokens; m++) {
                            float[] memK = new float[config.headDim];
                            float[] memV = new float[config.headDim];
                            for (int d = 0; d < config.headDim; d++) {
                                memK[d] = memoryK[m * config.headDim + d];
                                memV[d] = memoryV[m * config.headDim + d];
                            }
                            float score = MatrixOperations.dotProduct(q, memK) * scale;
                            scores[effectiveLen + m] = score;
                        }
                    }

                    float[] weights = softmax(scores);

                    float[] outputVec = new float[config.headDim];
                    Arrays.fill(outputVec, 0);

                    for (int t = startWindow; t <= s; t++) {
                        int tIdx = t - startWindow;
                        float w = weights[tIdx];
                        float[] vVec = vMatrices[h][b * seqLen + t];
                        for (int d = 0; d < config.headDim; d++) {
                            outputVec[d] += w * vVec[d];
                        }
                        if (tIdx < seqLen) {
                            attentionWeights[h][(b * seqLen + s) * seqLen + tIdx] = w;
                        }
                    }

                    if (config.enableHierarchicalMemory && currentPosition > config.slidingWindowSize && memoryV != null) {
                        int memoryTokens = config.sparseTopK / 2;
                        for (int m = 0; m < memoryTokens; m++) {
                            float w = weights[effectiveLen + m];
                            for (int d = 0; d < config.headDim; d++) {
                                outputVec[d] += w * memoryV[m * config.headDim + d];
                            }
                        }
                    }

                    float[] projected = new float[config.dModel];
                    for (int i = 0; i < config.headDim; i++) {
                        for (int j = 0; j < config.dModel; j++) {
                            projected[j] += outputVec[i] * oWeights[oOffset + i * config.dModel + j];
                        }
                    }

                    if (config.dropoutRate > 0 && currentPosition < 1000) {
                        for (int j = 0; j < config.dModel; j++) {
                            if (random.nextFloat() < config.dropoutRate) {
                                projected[j] = 0;
                            }
                        }
                    }

                    int outputOffset = (b * seqLen + s) * config.dModel;
                    for (int j = 0; j < config.dModel; j++) {
                        output[outputOffset + j] += projected[j];
                    }
                }
            }
        }

        if (config.enableHierarchicalMemory) {
            float[] memoryContent = new float[seqLen * config.headDim];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int h = 0; h < config.nHeads; h++) {
                        float[] v = vMatrices[h][b * seqLen + s];
                        for (int d = 0; d < config.headDim; d++) {
                            memoryContent[s * config.headDim + d] += v[d] / config.nHeads;
                        }
                    }
                }
            }
            memory.update(memoryContent, seqLen, config.headDim, config.memoryRetention);
        }

        cachePosition += seqLen;
        currentPosition += seqLen;

        result.output = output;
        result.attentionWeights = attentionWeights;
        result.qMatrices = qMatrices;
        result.kMatrices = kMatrices;
        result.vMatrices = vMatrices;
        result.activeTokens = seqLen * batchSize;
        result.memoryLoss = 1.0 - config.memoryRetention;

        return result;
    }

    public GradientResult backward(float[] dOutput, float[] input, int batchSize, int seqLen, ZYLResult forwardResult) {
        int totalSize = batchSize * seqLen * config.dModel;
        float[] dInput = new float[totalSize];
        Arrays.fill(dInput, 0);

        float[] dQWeights = new float[config.nHeads * config.dModel * config.headDim];
        float[] dKWeights = new float[config.nHeads * config.dModel * config.headDim];
        float[] dVWeights = new float[config.nHeads * config.dModel * config.headDim];
        float[] dOWeights = new float[config.nHeads * config.headDim * config.dModel];
        float[] dPositionBias = new float[config.nHeads * config.slidingWindowSize];

        for (int h = 0; h < config.nHeads; h++) {
            int qOffset = h * config.dModel * config.headDim;
            int kOffset = h * config.dModel * config.headDim;
            int vOffset = h * config.dModel * config.headDim;
            int oOffset = h * config.headDim * config.dModel;

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int outputOffset = (b * seqLen + s) * config.dModel;

                    float[] dOut = new float[config.dModel];
                    for (int d = 0; d < config.dModel; d++) {
                        dOut[d] = dOutput[outputOffset + d] / config.nHeads;
                    }

                    float[] dProjected = new float[config.headDim];
                    for (int i = 0; i < config.headDim; i++) {
                        for (int j = 0; j < config.dModel; j++) {
                            dProjected[i] += dOut[j] * oWeights[oOffset + i * config.dModel + j];
                        }
                    }

                    for (int i = 0; i < config.headDim; i++) {
                        for (int j = 0; j < config.dModel; j++) {
                            dOWeights[oOffset + i * config.dModel + j] += dOut[j] * forwardResult.vMatrices[h][b * seqLen + s][i];
                        }
                    }

                    int inputOffset = (b * seqLen + s) * config.dModel;
                    float[] x = new float[config.dModel];
                    for (int d = 0; d < config.dModel; d++) {
                        x[d] = input[inputOffset + d];
                    }

                    float[] q = forwardResult.qMatrices[h][b * seqLen + s];
                    float[] k = forwardResult.kMatrices[h][b * seqLen + s];
                    float[] v = forwardResult.vMatrices[h][b * seqLen + s];

                    int startWindow = Math.max(0, s - config.slidingWindowSize);
                    int effectiveLen = s - startWindow + 1;

                    float[] scores = new float[effectiveLen];
                    float scale = (float) (1.0 / Math.sqrt(config.headDim));
                    for (int t = startWindow; t <= s; t++) {
                        int tIdx = t - startWindow;
                        scores[tIdx] = MatrixOperations.dotProduct(q, forwardResult.kMatrices[h][b * seqLen + t]) * scale;
                    }
                    float[] weights = softmax(scores);

                    float[] dAttention = new float[effectiveLen];
                    for (int t = startWindow; t <= s; t++) {
                        int tIdx = t - startWindow;
                        for (int d = 0; d < config.headDim; d++) {
                            dAttention[tIdx] += dProjected[d] * forwardResult.vMatrices[h][b * seqLen + t][d];
                        }
                    }

                    float[] dScores = new float[effectiveLen];
                    float sumGrad = 0;
                    for (int t = 0; t < effectiveLen; t++) {
                        sumGrad += dAttention[t];
                    }
                    for (int t = 0; t < effectiveLen; t++) {
                        dScores[t] = weights[t] * (dAttention[t] - sumGrad);
                    }

                    float[] dQ = new float[config.headDim];
                    float[] dK = new float[config.headDim];
                    float[] dV = new float[config.headDim];

                    for (int t = startWindow; t <= s; t++) {
                        int tIdx = t - startWindow;
                        float dScore = dScores[tIdx] * scale;
                        float weight = weights[tIdx];

                        float[] kT = forwardResult.kMatrices[h][b * seqLen + t];
                        float[] vT = forwardResult.vMatrices[h][b * seqLen + t];

                        for (int d = 0; d < config.headDim; d++) {
                            dQ[d] += dScore * kT[d];
                            dK[d] += dScore * q[d];
                            dV[d] += weight * dProjected[d];
                        }

                        int relPos = s - t;
                        if (relPos >= 0 && relPos < config.slidingWindowSize) {
                            dPositionBias[h * config.slidingWindowSize + relPos] += dScore;
                        }
                    }

                    if (config.enableRoPE) {
                        int pos = currentPosition - seqLen + s;
                        applyRoPEBackward(dQ, pos);
                        applyRoPEBackward(dK, pos);
                    }

                    for (int i = 0; i < config.dModel; i++) {
                        for (int j = 0; j < config.headDim; j++) {
                            dQWeights[qOffset + i * config.headDim + j] += x[i] * dQ[j];
                            dKWeights[kOffset + i * config.headDim + j] += x[i] * dK[j];
                            dVWeights[vOffset + i * config.headDim + j] += x[i] * dV[j];
                        }
                    }

                    float[] dXQ = new float[config.dModel];
                    float[] dXK = new float[config.dModel];
                    float[] dXV = new float[config.dModel];

                    for (int i = 0; i < config.dModel; i++) {
                        for (int j = 0; j < config.headDim; j++) {
                            dXQ[i] += dQ[j] * qWeights[qOffset + i * config.headDim + j];
                            dXK[i] += dK[j] * kWeights[kOffset + i * config.headDim + j];
                            dXV[i] += dV[j] * vWeights[vOffset + i * config.headDim + j];
                        }
                    }

                    for (int d = 0; d < config.dModel; d++) {
                        dInput[inputOffset + d] += dXQ[d] + dXK[d] + dXV[d];
                    }
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("qWeights", dQWeights);
        result.addGradient("kWeights", dKWeights);
        result.addGradient("vWeights", dVWeights);
        result.addGradient("oWeights", dOWeights);
        result.addGradient("positionBias", dPositionBias);

        return result;
    }

    private void applyRoPE(float[] vec, int pos) {
        if (!config.enableRoPE || ropeSin == null) return;

        for (int i = 0; i < config.headDim / 2; i++) {
            int idx = pos * config.headDim / 2 + i;
            if (idx >= ropeSin.length) break;

            float sin = ropeSin[idx];
            float cos = ropeCos[idx];

            float t1 = vec[2 * i];
            float t2 = vec[2 * i + 1];

            vec[2 * i] = t1 * cos - t2 * sin;
            vec[2 * i + 1] = t1 * sin + t2 * cos;
        }
    }

    private void applyRoPEBackward(float[] dVec, int pos) {
        if (!config.enableRoPE || ropeSin == null) return;

        for (int i = 0; i < config.headDim / 2; i++) {
            int idx = pos * config.headDim / 2 + i;
            if (idx >= ropeSin.length) break;

            float sin = ropeSin[idx];
            float cos = ropeCos[idx];

            float dt1 = dVec[2 * i];
            float dt2 = dVec[2 * i + 1];

            dVec[2 * i] = dt1 * cos + dt2 * sin;
            dVec[2 * i + 1] = -dt1 * sin + dt2 * cos;
        }
    }

    private float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) max = v;
        }

        float[] exp = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }

        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = exp[i] / sum;
        }

        return result;
    }

    public void reset() {
        currentPosition = 0;
        cachePosition = 0;
        memory = new HierarchicalMemory(config.memoryLayers, config.memoryBlockSize, config.headDim);
        if (kvCache != null) {
            for (int h = 0; h < config.nHeads; h++) {
                Arrays.fill(kvCache[0][h], 0);
                Arrays.fill(kvCache[1][h], 0);
            }
        }
    }

    public void save(DataOutputStream dos) throws IOException {
        dos.writeInt(qWeights.length);
        for (float v : qWeights) dos.writeFloat(v);
        dos.writeInt(kWeights.length);
        for (float v : kWeights) dos.writeFloat(v);
        dos.writeInt(vWeights.length);
        for (float v : vWeights) dos.writeFloat(v);
        dos.writeInt(oWeights.length);
        for (float v : oWeights) dos.writeFloat(v);
        dos.writeInt(positionBias.length);
        for (float v : positionBias) dos.writeFloat(v);
        memory.save(dos);
        dos.writeInt(currentPosition);
        dos.writeInt(cachePosition);
    }

    public void load(DataInputStream dis) throws IOException {
        int qSize = dis.readInt();
        if (qSize != qWeights.length) throw new IOException("Q weights size mismatch");
        for (int i = 0; i < qSize; i++) qWeights[i] = dis.readFloat();
        
        int kSize = dis.readInt();
        if (kSize != kWeights.length) throw new IOException("K weights size mismatch");
        for (int i = 0; i < kSize; i++) kWeights[i] = dis.readFloat();
        
        int vSize = dis.readInt();
        if (vSize != vWeights.length) throw new IOException("V weights size mismatch");
        for (int i = 0; i < vSize; i++) vWeights[i] = dis.readFloat();
        
        int oSize = dis.readInt();
        if (oSize != oWeights.length) throw new IOException("O weights size mismatch");
        for (int i = 0; i < oSize; i++) oWeights[i] = dis.readFloat();
        
        int pbSize = dis.readInt();
        if (pbSize != positionBias.length) throw new IOException("Position bias size mismatch");
        for (int i = 0; i < pbSize; i++) positionBias[i] = dis.readFloat();
        
        memory.load(dis);
        currentPosition = dis.readInt();
        cachePosition = dis.readInt();
    }

    public ZYLConfig getConfig() { return config; }
    public HierarchicalMemory getMemory() { return memory; }
    public int getCurrentPosition() { return currentPosition; }

    public float[] getQWeights() { return qWeights; }
    public float[] getKWeights() { return kWeights; }
    public float[] getVWeights() { return vWeights; }
    public float[] getOWeights() { return oWeights; }
    public float[] getPositionBias() { return positionBias; }

    public double calculateMemoryRetentionRate(int contextLength) {
        double retention = 1.0;
        for (int i = 0; i < contextLength / config.memoryBlockSize; i++) {
            retention *= config.memoryRetention;
        }
        return retention;
    }

    public double estimateComputationComplexity(int contextLength) {
        if (!config.enableSparseAttention) {
            return (double) contextLength * contextLength * config.nHeads;
        }
        return (double) contextLength * config.sparseTopK * config.nHeads;
    }
}