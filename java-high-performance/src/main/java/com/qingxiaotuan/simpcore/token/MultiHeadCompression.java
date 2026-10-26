package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

import java.util.*;

public class MultiHeadCompression {

    public static class MHCConfig {
        public int compressionRatio = 2;
        public int minHeads = 4;
        public int maxHeads = 64;
        public double threshold = 0.1;
        public boolean enableAdaptive = true;
        public boolean enableSparsity = true;
        public int blockSize = 64;

        public void print() {
            System.out.println("=== MHC Configuration ===");
            System.out.println("Compression ratio: " + compressionRatio);
            System.out.println("Min heads: " + minHeads);
            System.out.println("Max heads: " + maxHeads);
            System.out.println("Threshold: " + threshold);
            System.out.println("Adaptive: " + enableAdaptive);
            System.out.println("Sparsity: " + enableSparsity);
            System.out.println("Block size: " + blockSize);
        }
    }

    public static class MHCResult {
        public float[] compressedOutput;
        public float[] headImportance;
        public int activeHeads;
        public double compressionRatio;
        public double reconstructionError;

        public void print() {
            System.out.println("=== MHC Result ===");
            System.out.println("Active heads: " + activeHeads);
            System.out.println("Compression ratio: " + String.format("%.1f%%", compressionRatio * 100));
            System.out.println("Reconstruction error: " + String.format("%.4f", reconstructionError));
        }
    }

    private final MHCConfig config;
    private float[][] projectionMatrix;
    private float[][] reconstructionMatrix;
    private float[] headBias;

    public MultiHeadCompression(MHCConfig config) {
        this.config = config;
    }

    public void initialize(int nHeads, int headDim, int dModel) {
        int compressedDim = dModel / config.compressionRatio;
        
        projectionMatrix = new float[nHeads][headDim * compressedDim];
        reconstructionMatrix = new float[nHeads][compressedDim * headDim];
        headBias = new float[nHeads];

        Random random = new Random(42);
        for (int h = 0; h < nHeads; h++) {
            for (int i = 0; i < headDim * compressedDim; i++) {
                projectionMatrix[h][i] = (float) (random.nextGaussian() / Math.sqrt(headDim));
            }
            for (int i = 0; i < compressedDim * headDim; i++) {
                reconstructionMatrix[h][i] = (float) (random.nextGaussian() / Math.sqrt(compressedDim));
            }
            headBias[h] = (float) random.nextGaussian() * 0.01f;
        }
    }

    public MHCResult compress(float[] input, int batchSize, int seqLen, int nHeads, int headDim, int dModel) {
        MHCResult result = new MHCResult();
        int totalSize = batchSize * seqLen * dModel;
        int compressedDim = dModel / config.compressionRatio;
        int compressedSize = batchSize * seqLen * compressedDim;

        float[] headOutputs = new float[nHeads * batchSize * seqLen * headDim];
        float[] headNorms = new float[nHeads];

        for (int h = 0; h < nHeads; h++) {
            float norm = 0;
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int inputOffset = (b * seqLen + s) * dModel;
                    int headOffset = h * batchSize * seqLen * headDim + (b * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++) {
                        int sourceIdx = inputOffset + (h * headDim + d) % dModel;
                        headOutputs[headOffset + d] = input[sourceIdx];
                        norm += input[sourceIdx] * input[sourceIdx];
                    }
                }
            }
            headNorms[h] = (float) Math.sqrt(norm / (batchSize * seqLen * headDim));
        }

        List<Integer> activeHeadIndices = new ArrayList<>();
        for (int h = 0; h < nHeads; h++) {
            if (headNorms[h] > config.threshold) {
                activeHeadIndices.add(h);
            }
        }

        if (activeHeadIndices.size() < config.minHeads) {
            activeHeadIndices.clear();
            for (int h = 0; h < Math.min(config.minHeads, nHeads); h++) {
                activeHeadIndices.add(h);
            }
        }

        if (activeHeadIndices.size() > config.maxHeads) {
            activeHeadIndices.sort((a, b) -> Float.compare(headNorms[b], headNorms[a]));
            activeHeadIndices = activeHeadIndices.subList(0, config.maxHeads);
        }

        result.activeHeads = activeHeadIndices.size();
        result.compressionRatio = 1.0 - (double) result.activeHeads / nHeads;

        float[] headWeights = new float[nHeads];
        float totalWeight = 0;
        for (int h : activeHeadIndices) {
            headWeights[h] = headNorms[h];
            totalWeight += headNorms[h];
        }
        for (int h = 0; h < nHeads; h++) {
            headWeights[h] /= totalWeight;
        }
        result.headImportance = headWeights;

        float[] compressed = new float[compressedSize];
        Arrays.fill(compressed, 0);

        for (int h : activeHeadIndices) {
            float weight = headWeights[h];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int headOffset = h * batchSize * seqLen * headDim + (b * seqLen + s) * headDim;
                    int compressedOffset = (b * seqLen + s) * compressedDim;

                    float[] headSlice = new float[headDim];
                    for (int d = 0; d < headDim; d++) {
                        headSlice[d] = headOutputs[headOffset + d];
                    }

                    float[] projected = MatrixOperations.matmul(headSlice, projectionMatrix[h], 1, headDim, compressedDim);
                    for (int d = 0; d < compressedDim; d++) {
                        compressed[compressedOffset + d] += weight * projected[d];
                    }
                }
            }
        }

        if (config.enableSparsity) {
            applySparsity(compressed, compressedSize, compressedDim);
        }

        result.compressedOutput = compressed;

        float[] reconstructed = reconstruct(compressed, batchSize, seqLen, nHeads, headDim, dModel, headWeights);
        double error = 0;
        for (int i = 0; i < totalSize; i++) {
            error += (input[i] - reconstructed[i]) * (input[i] - reconstructed[i]);
        }
        result.reconstructionError = error / totalSize;

        return result;
    }

    public float[] reconstruct(float[] compressed, int batchSize, int seqLen, int nHeads, int headDim, int dModel, float[] headWeights) {
        int totalSize = batchSize * seqLen * dModel;
        int compressedDim = dModel / config.compressionRatio;

        float[] reconstructed = new float[totalSize];
        Arrays.fill(reconstructed, 0);

        for (int h = 0; h < nHeads; h++) {
            if (headWeights[h] < config.threshold) continue;

            float weight = headWeights[h];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int compressedOffset = (b * seqLen + s) * compressedDim;
                    int outputOffset = (b * seqLen + s) * dModel;

                    float[] compressedSlice = new float[compressedDim];
                    for (int d = 0; d < compressedDim; d++) {
                        compressedSlice[d] = compressed[compressedOffset + d];
                    }

                    float[] decompressed = MatrixOperations.matmul(compressedSlice, reconstructionMatrix[h], 1, compressedDim, headDim);
                    for (int d = 0; d < headDim; d++) {
                        int targetIdx = outputOffset + (h * headDim + d) % dModel;
                        reconstructed[targetIdx] += weight * (decompressed[d] + headBias[h]);
                    }
                }
            }
        }

        return reconstructed;
    }

    private void applySparsity(float[] data, int size, int dim) {
        int blocks = size / dim;
        for (int b = 0; b < blocks; b++) {
            int offset = b * dim;
            float norm = 0;
            for (int d = 0; d < dim; d++) {
                norm += data[offset + d] * data[offset + d];
            }
            norm = (float) Math.sqrt(norm);
            if (norm < config.threshold) {
                for (int d = 0; d < dim; d++) {
                    data[offset + d] = 0;
                }
            }
        }
    }

    public float[] backward(float[] dCompressed, float[] compressed, int batchSize, int seqLen, int nHeads, int headDim, int dModel, float[] headWeights) {
        int totalSize = batchSize * seqLen * dModel;
        int compressedDim = dModel / config.compressionRatio;

        float[] dInput = new float[totalSize];
        Arrays.fill(dInput, 0);

        for (int h = 0; h < nHeads; h++) {
            if (headWeights[h] < config.threshold) continue;

            float weight = headWeights[h];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int dCompressedOffset = (b * seqLen + s) * compressedDim;
                    int inputOffset = (b * seqLen + s) * dModel;

                    float[] dCompressedSlice = new float[compressedDim];
                    for (int d = 0; d < compressedDim; d++) {
                        dCompressedSlice[d] = dCompressed[dCompressedOffset + d];
                    }

                    float[] dDecompressed = MatrixOperations.matmul(dCompressedSlice, MatrixOperations.transpose(reconstructionMatrix[h], compressedDim, headDim), 1, compressedDim, headDim);
                    for (int d = 0; d < headDim; d++) {
                        int targetIdx = inputOffset + (h * headDim + d) % dModel;
                        dInput[targetIdx] += weight * dDecompressed[d];
                    }
                }
            }
        }

        return dInput;
    }

    public MHCConfig getConfig() { return config; }
    public float[][] getProjectionMatrix() { return projectionMatrix; }
    public float[][] getReconstructionMatrix() { return reconstructionMatrix; }
}