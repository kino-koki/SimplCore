package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

public class LTIInjection {
    private float[] A;
    private float[] B;
    private int dim;
    private float spectralBound;
    
    public LTIInjection(int dim, float spectralBound) {
        this.dim = dim;
        this.spectralBound = spectralBound;
        initWeights();
    }
    
    private void initWeights() {
        A = new float[dim * dim];
        B = new float[dim * dim];
        
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                A[i * dim + j] = (float) (Math.random() - 0.5) * 2 / dim;
                B[i * dim + j] = (float) (Math.random() - 0.5) * 2 / dim;
            }
        }
        
        enforceSpectralBound();
    }
    
    private void enforceSpectralBound() {
        float[] Atemp = new float[dim * dim];
        System.arraycopy(A, 0, Atemp, 0, A.length);
        
        float spectralNorm = estimateSpectralNorm(Atemp);
        float scale = spectralBound / spectralNorm;
        
        for (int i = 0; i < A.length; i++) {
            A[i] *= scale;
        }
    }
    
    private float estimateSpectralNorm(float[] mat) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) Math.random();
        }
        
        float norm = (float) Math.sqrt(MatrixOperations.dotProduct(v, v));
        for (int i = 0; i < dim; i++) {
            v[i] /= norm;
        }
        
        for (int iter = 0; iter < 20; iter++) {
            float[] Av = MatrixOperations.matmul(A, v, dim, dim, 1);
            norm = (float) Math.sqrt(MatrixOperations.dotProduct(Av, Av));
            for (int i = 0; i < dim; i++) {
                v[i] = Av[i] / norm;
            }
        }
        
        return norm;
    }
    
    public float[] inject(float[] h, float[] e) {
        float[] Ah = MatrixOperations.matmul(A, h, dim, dim, 1);
        float[] Be = MatrixOperations.matmul(B, e, dim, dim, 1);
        float[] result = new float[dim];
        
        for (int i = 0; i < dim; i++) {
            result[i] = Ah[i] + Be[i];
        }
        
        return result;
    }
    
    public float[] getA() {
        return A;
    }
    
    public float[] getB() {
        return B;
    }
    
    public float getSpectralRadius() {
        return estimateSpectralNorm(A);
    }
    
    public void updateParameters(float learningRate) {
        for (int i = 0; i < A.length; i++) {
            A[i] += (float) (Math.random() - 0.5) * learningRate;
        }
        for (int i = 0; i < B.length; i++) {
            B[i] += (float) (Math.random() - 0.5) * learningRate;
        }
        enforceSpectralBound();
    }
}
