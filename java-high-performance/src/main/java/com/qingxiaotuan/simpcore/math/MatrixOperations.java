package com.qingxiaotuan.simpcore.math;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class MatrixOperations {

    private static final ForkJoinPool COMPUTE_POOL = ForkJoinPool.commonPool();
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final boolean VECTOR_ENABLED = true;

    private MatrixOperations() {}

    public static float[] matmul(float[] A, float[] B, int M, int K, int N) {
        if (VECTOR_ENABLED && K >= SPECIES.length()) {
            return matmulVectorized(A, B, M, K, N);
        }
        float[] C = new float[M * N];
        COMPUTE_POOL.invoke(new MatmulTask(A, B, C, M, K, N, 0, M));
        return C;
    }

    private static final ThreadLocal<float[]> TL_BT = new ThreadLocal<>();

    private static float[] matmulVectorized(float[] A, float[] B, int M, int K, int N) {
        float[] C = new float[M * N];
        
        // Transpose B: B(KxN) -> BT(NxK) so inner loop accesses contiguous memory
        // C[i][j] = sum_k A[i][k] * B[k][j] = sum_k A[i][k] * BT[j][k]
        float[] BT = getOrCreateTransposed(K, N);
        for (int j = 0; j < N; j++) {
            int btOff = j * K;
            for (int k = 0; k < K; k++) {
                BT[btOff + k] = B[k * N + j];
            }
        }
        
        for (int i = 0; i < M; i++) {
            int aOff = i * K;
            int cOff = i * N;
            for (int j = 0; j < N; j++) {
                int btOff = j * K;
                float sum = 0f;
                int k = 0;
                
                for (; k + SPECIES.length() <= K; k += SPECIES.length()) {
                    FloatVector aVec = FloatVector.fromArray(SPECIES, A, aOff + k);
                    FloatVector bVec = FloatVector.fromArray(SPECIES, BT, btOff + k);
                    sum += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);
                }
                
                for (; k < K; k++) {
                    sum += A[aOff + k] * BT[btOff + k];
                }
                
                C[cOff + j] = sum;
            }
        }
        
        return C;
    }

    private static float[] getOrCreateTransposed(int K, int N) {
        float[] bt = TL_BT.get();
        int needed = N * K;
        if (bt == null || bt.length < needed) {
            bt = new float[needed];
            TL_BT.set(bt);
        }
        return bt;
    }

    public static float[] matmulSIMD(float[] A, float[] B, int M, int K, int N) {
        return matmulVectorized(A, B, M, K, N);
    }

    public static float[] add(float[] A, float[] B) {
        if (A.length != B.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        if (VECTOR_ENABLED) {
            return addVectorized(A, B);
        }
        float[] result = new float[A.length];
        COMPUTE_POOL.invoke(new VectorTask(A, B, result, 0, A.length));
        return result;
    }

    private static float[] addVectorized(float[] A, float[] B) {
        float[] result = new float[A.length];
        int i = 0;
        
        for (; i + SPECIES.length() <= A.length; i += SPECIES.length()) {
            FloatVector aVec = FloatVector.fromArray(SPECIES, A, i);
            FloatVector bVec = FloatVector.fromArray(SPECIES, B, i);
            aVec.add(bVec).intoArray(result, i);
        }
        
        for (; i < A.length; i++) {
            result[i] = A[i] + B[i];
        }
        
        return result;
    }

    public static float[] sub(float[] A, float[] B) {
        if (A.length != B.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        if (VECTOR_ENABLED) {
            return subVectorized(A, B);
        }
        float[] result = new float[A.length];
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] - B[i];
        }
        return result;
    }

    private static float[] subVectorized(float[] A, float[] B) {
        float[] result = new float[A.length];
        int i = 0;
        
        for (; i + SPECIES.length() <= A.length; i += SPECIES.length()) {
            FloatVector aVec = FloatVector.fromArray(SPECIES, A, i);
            FloatVector bVec = FloatVector.fromArray(SPECIES, B, i);
            aVec.sub(bVec).intoArray(result, i);
        }
        
        for (; i < A.length; i++) {
            result[i] = A[i] - B[i];
        }
        
        return result;
    }

    public static float[] mul(float[] A, float scalar) {
        if (VECTOR_ENABLED) {
            return mulVectorized(A, scalar);
        }
        float[] result = new float[A.length];
        COMPUTE_POOL.invoke(new ScalarMulTask(A, scalar, result, 0, A.length));
        return result;
    }

    private static float[] mulVectorized(float[] A, float scalar) {
        float[] result = new float[A.length];
        FloatVector scalarVec = FloatVector.broadcast(SPECIES, scalar);
        int i = 0;
        
        for (; i + SPECIES.length() <= A.length; i += SPECIES.length()) {
            FloatVector aVec = FloatVector.fromArray(SPECIES, A, i);
            aVec.mul(scalarVec).intoArray(result, i);
        }
        
        for (; i < A.length; i++) {
            result[i] = A[i] * scalar;
        }
        
        return result;
    }

    public static float[] transpose(float[] A, int rows, int cols) {
        float[] result = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = A[i * cols + j];
            }
        }
        return result;
    }

    public static float[] relu(float[] x) {
        if (VECTOR_ENABLED) {
            return reluVectorized(x);
        }
        float[] result = new float[x.length];
        COMPUTE_POOL.invoke(new ReluTask(x, result, 0, x.length));
        return result;
    }

    private static float[] reluVectorized(float[] x) {
        float[] result = new float[x.length];
        FloatVector zero = FloatVector.broadcast(SPECIES, 0f);
        int i = 0;
        
        for (; i + SPECIES.length() <= x.length; i += SPECIES.length()) {
            FloatVector xVec = FloatVector.fromArray(SPECIES, x, i);
            xVec.max(zero).intoArray(result, i);
        }
        
        for (; i < x.length; i++) {
            result[i] = Math.max(0f, x[i]);
        }
        
        return result;
    }

    public static float dotProduct(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        if (VECTOR_ENABLED) {
            return dotProductVectorized(a, b);
        }
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static float dotProductVectorized(float[] a, float[] b) {
        float sum = 0f;
        int i = 0;
        
        for (; i + SPECIES.length() <= a.length; i += SPECIES.length()) {
            FloatVector aVec = FloatVector.fromArray(SPECIES, a, i);
            FloatVector bVec = FloatVector.fromArray(SPECIES, b, i);
            sum += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);
        }
        
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }

    public static float norm(float[] x) {
        float sum = 0f;
        for (float v : x) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    public static float[] normalize(float[] x) {
        float n = norm(x);
        if (n < 1e-8f) {
            return Arrays.copyOf(x, x.length);
        }
        return mul(x, 1f / n);
    }

    public static float[] outerProduct(float[] a, float[] b, int m, int n) {
        float[] result = new float[m * n];
        for (int i = 0; i < m; i++) {
            float ai = a[i];
            for (int j = 0; j < n; j++) {
                result[i * n + j] = ai * b[j];
            }
        }
        return result;
    }

    public static float mean(float[] x) {
        return sum(x) / x.length;
    }

    public static float[] addBias(float[] input, float[] bias, int batchSize, int hiddenSize) {
        float[] result = Arrays.copyOf(input, input.length);
        for (int b = 0; b < batchSize; b++) {
            int base = b * hiddenSize;
            for (int i = 0; i < hiddenSize; i++) {
                result[base + i] += bias[i];
            }
        }
        return result;
    }

    public static float[] gelu(float[] x) {
        if (VECTOR_ENABLED) {
            return geluVectorized(x);
        }
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float cdf = 0.5f * (1.0f + (float) Math.tanh(0.79788456f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
            result[i] = x[i] * cdf;
        }
        return result;
    }
    
    private static float[] geluVectorized(float[] x) {
        float[] result = new float[x.length];
        FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
        FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
        FloatVector coef1 = FloatVector.broadcast(SPECIES, 0.79788456f);
        FloatVector coef2 = FloatVector.broadcast(SPECIES, 0.044715f);
        
        int i = 0;
        for (; i + SPECIES.length() <= x.length; i += SPECIES.length()) {
            FloatVector xVec = FloatVector.fromArray(SPECIES, x, i);
            FloatVector xCube = xVec.mul(xVec).mul(xVec);
            FloatVector tanhArg = coef1.mul(xVec.add(coef2.mul(xCube)));
            
            float[] tanhResult = new float[SPECIES.length()];
            for (int j = 0; j < SPECIES.length(); j++) {
                tanhResult[j] = (float) Math.tanh(tanhArg.lane(j));
            }
            FloatVector tanhVec = FloatVector.fromArray(SPECIES, tanhResult, 0);
            
            FloatVector cdf = half.mul(one.add(tanhVec));
            xVec.mul(cdf).intoArray(result, i);
        }
        
        for (; i < x.length; i++) {
            float cdf = 0.5f * (1.0f + (float) Math.tanh(0.79788456f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
            result[i] = x[i] * cdf;
        }
        
        return result;
    }
    
    public static float[] silu(float[] x) {
        if (VECTOR_ENABLED) {
            return siluVectorized(x);
        }
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float sigmoid = 1f / (1f + (float) Math.exp(-x[i]));
            result[i] = x[i] * sigmoid;
        }
        return result;
    }
    
    private static float[] siluVectorized(float[] x) {
        float[] result = new float[x.length];
        FloatVector one = FloatVector.broadcast(SPECIES, 1.0f);
        
        int i = 0;
        for (; i + SPECIES.length() <= x.length; i += SPECIES.length()) {
            FloatVector xVec = FloatVector.fromArray(SPECIES, x, i);
            
            float[] sigmoidResult = new float[SPECIES.length()];
            for (int j = 0; j < SPECIES.length(); j++) {
                sigmoidResult[j] = 1f / (1f + (float) Math.exp(-xVec.lane(j)));
            }
            FloatVector sigmoidVec = FloatVector.fromArray(SPECIES, sigmoidResult, 0);
            
            xVec.mul(sigmoidVec).intoArray(result, i);
        }
        
        for (; i < x.length; i++) {
            float sigmoid = 1f / (1f + (float) Math.exp(-x[i]));
            result[i] = x[i] * sigmoid;
        }
        
        return result;
    }
    
    public static float[] layerNorm(float[] input, float[] gamma, float[] beta, int batchSize, int hiddenSize) {
        if (VECTOR_ENABLED) {
            return layerNormVectorized(input, gamma, beta, batchSize, hiddenSize);
        }
        float[] result = new float[input.length];
        for (int b = 0; b < batchSize; b++) {
            int base = b * hiddenSize;
            float sum = 0.0f;
            for (int i = 0; i < hiddenSize; i++) {
                sum += input[base + i];
            }
            float mean = sum / hiddenSize;
            
            float varSum = 0.0f;
            for (int i = 0; i < hiddenSize; i++) {
                float diff = input[base + i] - mean;
                varSum += diff * diff;
            }
            float var = varSum / hiddenSize;
            float invStd = 1.0f / (float) Math.sqrt(var + 1e-6);
            
            for (int i = 0; i < hiddenSize; i++) {
                result[base + i] = gamma[i] * (input[base + i] - mean) * invStd + beta[i];
            }
        }
        return result;
    }
    
    private static float[] layerNormVectorized(float[] input, float[] gamma, float[] beta, int batchSize, int hiddenSize) {
        float[] result = new float[input.length];
        FloatVector epsVec = FloatVector.broadcast(SPECIES, 1e-6f);
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * hiddenSize;
            
            float mean = 0f;
            int i = 0;
            for (; i + SPECIES.length() <= hiddenSize; i += SPECIES.length()) {
                FloatVector vec = FloatVector.fromArray(SPECIES, input, base + i);
                mean += vec.reduceLanes(VectorOperators.ADD);
            }
            for (; i < hiddenSize; i++) {
                mean += input[base + i];
            }
            mean /= hiddenSize;
            
            FloatVector meanVec = FloatVector.broadcast(SPECIES, mean);
            
            float varSum = 0f;
            i = 0;
            for (; i + SPECIES.length() <= hiddenSize; i += SPECIES.length()) {
                FloatVector vec = FloatVector.fromArray(SPECIES, input, base + i);
                FloatVector diff = vec.sub(meanVec);
                varSum += diff.mul(diff).reduceLanes(VectorOperators.ADD);
            }
            for (; i < hiddenSize; i++) {
                float diff = input[base + i] - mean;
                varSum += diff * diff;
            }
            float var = varSum / hiddenSize;
            float invStd = 1.0f / (float) Math.sqrt(var + 1e-6);
            FloatVector invStdVec = FloatVector.broadcast(SPECIES, invStd);
            
            i = 0;
            for (; i + SPECIES.length() <= hiddenSize; i += SPECIES.length()) {
                FloatVector inputVec = FloatVector.fromArray(SPECIES, input, base + i);
                FloatVector gammaVec = FloatVector.fromArray(SPECIES, gamma, i);
                FloatVector betaVec = FloatVector.fromArray(SPECIES, beta, i);
                
                FloatVector normalized = inputVec.sub(meanVec).mul(invStdVec);
                normalized.mul(gammaVec).add(betaVec).intoArray(result, base + i);
            }
            for (; i < hiddenSize; i++) {
                result[base + i] = gamma[i] * (input[base + i] - mean) * invStd + beta[i];
            }
        }
        
        return result;
    }
    
    public static float[] hadamardProduct(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        if (VECTOR_ENABLED) {
            return hadamardProductVectorized(a, b);
        }
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    private static float[] hadamardProductVectorized(float[] a, float[] b) {
        float[] result = new float[a.length];
        int i = 0;
        
        for (; i + SPECIES.length() <= a.length; i += SPECIES.length()) {
            FloatVector aVec = FloatVector.fromArray(SPECIES, a, i);
            FloatVector bVec = FloatVector.fromArray(SPECIES, b, i);
            aVec.mul(bVec).intoArray(result, i);
        }
        
        for (; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        
        return result;
    }
    
    public static float sum(float[] x) {
        if (VECTOR_ENABLED) {
            return sumVectorized(x);
        }
        float sum = 0f;
        for (float v : x) {
            sum += v;
        }
        return sum;
    }
    
    private static float sumVectorized(float[] x) {
        float sum = 0f;
        int i = 0;
        
        for (; i + SPECIES.length() <= x.length; i += SPECIES.length()) {
            FloatVector vec = FloatVector.fromArray(SPECIES, x, i);
            sum += vec.reduceLanes(VectorOperators.ADD);
        }
        
        for (; i < x.length; i++) {
            sum += x[i];
        }
        
        return sum;
    }
    
    public static float[] clamp(float[] x, float min, float max) {
        if (VECTOR_ENABLED) {
            return clampVectorized(x, min, max);
        }
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.max(min, Math.min(max, x[i]));
        }
        return result;
    }
    
    private static float[] clampVectorized(float[] x, float min, float max) {
        float[] result = new float[x.length];
        FloatVector minVec = FloatVector.broadcast(SPECIES, min);
        FloatVector maxVec = FloatVector.broadcast(SPECIES, max);
        
        int i = 0;
        for (; i + SPECIES.length() <= x.length; i += SPECIES.length()) {
            FloatVector xVec = FloatVector.fromArray(SPECIES, x, i);
            xVec.max(minVec).min(maxVec).intoArray(result, i);
        }
        
        for (; i < x.length; i++) {
            result[i] = Math.max(min, Math.min(max, x[i]));
        }
        
        return result;
    }
    
    public static float[] softmax(float[] logits) {
        if (VECTOR_ENABLED) {
            return softmaxVectorized(logits);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > max) max = v;
        }
        
        float[] exp = new float[logits.length];
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        return mul(exp, 1f / sum);
    }
    
    private static float[] softmaxVectorized(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        int i = 0;
        for (; i + SPECIES.length() <= logits.length; i += SPECIES.length()) {
            FloatVector vec = FloatVector.fromArray(SPECIES, logits, i);
            float vecMax = vec.reduceLanes(VectorOperators.MAX);
            if (vecMax > max) max = vecMax;
        }
        for (; i < logits.length; i++) {
            if (logits[i] > max) max = logits[i];
        }
        
        FloatVector maxVec = FloatVector.broadcast(SPECIES, max);
        float[] exp = new float[logits.length];
        
        i = 0;
        float sum = 0f;
        for (; i + SPECIES.length() <= logits.length; i += SPECIES.length()) {
            FloatVector vec = FloatVector.fromArray(SPECIES, logits, i);
            float[] expLane = new float[SPECIES.length()];
            for (int j = 0; j < SPECIES.length(); j++) {
                expLane[j] = (float) Math.exp(vec.lane(j) - max);
                sum += expLane[j];
            }
            FloatVector expVec = FloatVector.fromArray(SPECIES, expLane, 0);
            expVec.intoArray(exp, i);
        }
        for (; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        return mul(exp, 1f / sum);
    }

    private static class MatmulTask extends RecursiveAction {
        private final float[] A, B, C;
        private final int M, K, N;
        private final int startRow, endRow;
        private static final int THRESHOLD = 64;

        MatmulTask(float[] A, float[] B, float[] C, int M, int K, int N, int startRow, int endRow) {
            this.A = A;
            this.B = B;
            this.C = C;
            this.M = M;
            this.K = K;
            this.N = N;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected void compute() {
            if (endRow - startRow <= THRESHOLD) {
                computeDirect();
            } else {
                int mid = (startRow + endRow) / 2;
                invokeAll(
                    new MatmulTask(A, B, C, M, K, N, startRow, mid),
                    new MatmulTask(A, B, C, M, K, N, mid, endRow)
                );
            }
        }

        private void computeDirect() {
            for (int i = startRow; i < endRow; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }

    private static class VectorTask extends RecursiveAction {
        private final float[] A, B, result;
        private final int start, end;
        private static final int THRESHOLD = 4096;

        VectorTask(float[] A, float[] B, float[] result, int start, int end) {
            this.A = A;
            this.B = B;
            this.result = result;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                for (int i = start; i < end; i++) {
                    result[i] = A[i] + B[i];
                }
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new VectorTask(A, B, result, start, mid),
                    new VectorTask(A, B, result, mid, end)
                );
            }
        }
    }

    private static class ScalarMulTask extends RecursiveAction {
        private final float[] A;
        private final float scalar;
        private final float[] result;
        private final int start, end;
        private static final int THRESHOLD = 4096;

        ScalarMulTask(float[] A, float scalar, float[] result, int start, int end) {
            this.A = A;
            this.scalar = scalar;
            this.result = result;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                for (int i = start; i < end; i++) {
                    result[i] = A[i] * scalar;
                }
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new ScalarMulTask(A, scalar, result, start, mid),
                    new ScalarMulTask(A, scalar, result, mid, end)
                );
            }
        }
    }

    private static class ReluTask extends RecursiveAction {
        private final float[] x, result;
        private final int start, end;
        private static final int THRESHOLD = 4096;

        ReluTask(float[] x, float[] result, int start, int end) {
            this.x = x;
            this.result = result;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            if (end - start <= THRESHOLD) {
                for (int i = start; i < end; i++) {
                    result[i] = Math.max(0f, x[i]);
                }
            } else {
                int mid = (start + end) / 2;
                invokeAll(
                    new ReluTask(x, result, start, mid),
                    new ReluTask(x, result, mid, end)
                );
            }
        }
    }

    public static float[] rmsNorm(float[] input, float[] weight, int batchSize, int hiddenSize) {
        float[] result = new float[input.length];
        float eps = 1e-6f;
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * hiddenSize;
            
            float sumSq = 0.0f;
            for (int i = 0; i < hiddenSize; i++) {
                float val = input[base + i];
                sumSq += val * val;
            }
            float rms = (float) Math.sqrt(sumSq / hiddenSize + eps);
            float invRms = 1.0f / rms;
            
            for (int i = 0; i < hiddenSize; i++) {
                result[base + i] = input[base + i] * invRms * weight[i];
            }
        }
        return result;
    }

    public static void adamUpdate(float[] params, float[] grads, float[] m, float[] v,
                                  float lr, float beta1, float beta2, float eps, int length) {
        for (int i = 0; i < length; i++) {
            float g = grads[i];
            
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            
            float m_hat = m[i] / (1.0f - beta1);
            float v_hat = v[i] / (1.0f - beta2);
            
            params[i] -= lr * m_hat / ((float) Math.sqrt(v_hat) + eps);
        }
    }

    public static float[] dropout(float[] input, float dropoutRate, boolean training, int batchSize, int hiddenSize) {
        if (!training || dropoutRate == 0.0f) {
            return Arrays.copyOf(input, input.length);
        }
        
        float[] result = new float[input.length];
        float scale = 1.0f / (1.0f - dropoutRate);
        
        for (int i = 0; i < input.length; i++) {
            if (Math.random() < dropoutRate) {
                result[i] = 0.0f;
            } else {
                result[i] = input[i] * scale;
            }
        }
        return result;
    }

    public static float[] scaledDotProduct(float[] Q, float[] K, float[] V, int batchSize, int seqLen, int headDim) {
        float[] output = new float[batchSize * seqLen * headDim];
        float scale = 1.0f / (float) Math.sqrt(headDim);
        
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    float score = 0.0f;
                    for (int d = 0; d < headDim; d++) {
                        score += Q[b * seqLen * headDim + i * headDim + d] * 
                                 K[b * seqLen * headDim + j * headDim + d];
                    }
                    score *= scale;
                    
                    for (int d = 0; d < headDim; d++) {
                        output[b * seqLen * headDim + i * headDim + d] += 
                            score * V[b * seqLen * headDim + j * headDim + d];
                    }
                }
            }
        }
        return output;
    }

    public static void clipGradients(float[] grads, float maxNorm, int length) {
        float norm = norm(grads);
        if (norm > maxNorm) {
            float scale = maxNorm / (norm + 1e-6f);
            for (int i = 0; i < length; i++) {
                grads[i] *= scale;
            }
        }
    }
}