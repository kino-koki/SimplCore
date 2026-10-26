package com.qingxiaotuan.simpcore.training;

public class CausalMask {

    private final float[][] mask;
    private final int seqLen;

    public CausalMask(int seqLen) {
        this.seqLen = seqLen;
        this.mask = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                mask[i][j] = j <= i ? 1.0f : 0.0f;
            }
        }
    }

    public float get(int i, int j) {
        if (i < 0 || i >= seqLen || j < 0 || j >= seqLen) {
            return 0.0f;
        }
        return mask[i][j];
    }

    public float[][] getMatrix() {
        return mask;
    }

    public int getSeqLen() {
        return seqLen;
    }

    public float[] applyToVector(float[] vector, int position) {
        float[] result = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            if (i <= position) {
                result[i] = vector[i];
            } else {
                result[i] = 0.0f;
            }
        }
        return result;
    }

    public static float[] applyCausalMask(float[] attention, int seqLen) {
        float[] masked = new float[attention.length];
        int numHeads = attention.length / (seqLen * seqLen);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    int idx = h * seqLen * seqLen + i * seqLen + j;
                    masked[idx] = j <= i ? attention[idx] : Float.NEGATIVE_INFINITY;
                }
            }
        }
        return masked;
    }
}