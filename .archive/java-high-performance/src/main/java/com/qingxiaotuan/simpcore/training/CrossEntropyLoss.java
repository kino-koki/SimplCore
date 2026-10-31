package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;

public class CrossEntropyLoss {

    public static float compute(float[] logits, int[] targets, int batchSize, int seqLen, int vocabSize) {
        float totalLoss = 0.0f;
        int count = 0;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int target = targets[b * seqLen + s];
                if (target < 0 || target >= vocabSize) continue;

                int logitOffset = (b * seqLen + s) * vocabSize;
                float[] slice = memoryOptimizer.allocateFloatArray(vocabSize);
                System.arraycopy(logits, logitOffset, slice, 0, vocabSize);

                float[] probs = MatrixOperations.softmax(slice);
                float prob = probs[target];
                totalLoss -= (float) Math.log(Math.max(prob, 1e-10f));
                count++;
                
                memoryOptimizer.releaseFloatArray(slice);
                memoryOptimizer.releaseFloatArray(probs);
            }
        }

        return count > 0 ? totalLoss / count : 0.0f;
    }

    public static float[] computeGradients(float[] logits, int[] targets, int batchSize, int seqLen, int vocabSize) {
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] grads = memoryOptimizer.allocateFloatArray(logits.length);
        int count = 0;

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int target = targets[b * seqLen + s];
                if (target < 0 || target >= vocabSize) continue;

                int logitOffset = (b * seqLen + s) * vocabSize;
                float[] slice = memoryOptimizer.allocateFloatArray(vocabSize);
                System.arraycopy(logits, logitOffset, slice, 0, vocabSize);

                float[] probs = MatrixOperations.softmax(slice);
                for (int i = 0; i < vocabSize; i++) {
                    grads[logitOffset + i] = probs[i];
                }
                grads[logitOffset + target] -= 1.0f;
                count++;
                
                memoryOptimizer.releaseFloatArray(slice);
                memoryOptimizer.releaseFloatArray(probs);
            }
        }

        float scale = count > 0 ? 1.0f / count : 0.0f;
        for (int i = 0; i < grads.length; i++) {
            grads[i] *= scale;
        }

        return grads;
    }

    public static float computeWithMask(float[] logits, int[] targets, int[] mask, int batchSize, int seqLen, int vocabSize) {
        float totalLoss = 0.0f;
        int count = 0;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int target = targets[b * seqLen + s];
                int maskVal = mask[b * seqLen + s];
                if (maskVal == 0 || target < 0 || target >= vocabSize) continue;

                int logitOffset = (b * seqLen + s) * vocabSize;
                float[] slice = memoryOptimizer.allocateFloatArray(vocabSize);
                System.arraycopy(logits, logitOffset, slice, 0, vocabSize);

                float[] probs = MatrixOperations.softmax(slice);
                float prob = probs[target];
                totalLoss -= (float) Math.log(Math.max(prob, 1e-10f));
                count++;
                
                memoryOptimizer.releaseFloatArray(slice);
                memoryOptimizer.releaseFloatArray(probs);
            }
        }

        return count > 0 ? totalLoss / count : 0.0f;
    }
}