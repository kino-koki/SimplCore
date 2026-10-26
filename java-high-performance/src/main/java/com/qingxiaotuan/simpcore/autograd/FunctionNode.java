package com.qingxiaotuan.simpcore.autograd;

import java.util.*;

public abstract class FunctionNode {
    
    public final Tensor[] inputs;
    public final Tensor output;
    
    public FunctionNode(Tensor[] inputs, Tensor output) {
        this.inputs = inputs;
        this.output = output;
    }
    
    public abstract void backward(float[] gradOutput);
    
    public static class AddNode extends FunctionNode {
        public AddNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            if (inputs[0].requiresGrad) {
                if (inputs[0].grad == null) inputs[0].grad = new float[inputs[0].numel];
                for (int i = 0; i < gradOutput.length; i++) {
                    inputs[0].grad[i] += gradOutput[i];
                }
            }
            if (inputs.length > 1 && inputs[1].requiresGrad) {
                if (inputs[1].grad == null) inputs[1].grad = new float[inputs[1].numel];
                for (int i = 0; i < gradOutput.length; i++) {
                    inputs[1].grad[i] += gradOutput[i];
                }
            }
        }
    }
    
    public static class MulNode extends FunctionNode {
        public MulNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            if (inputs[0].requiresGrad) {
                if (inputs[0].grad == null) inputs[0].grad = new float[inputs[0].numel];
                for (int i = 0; i < gradOutput.length; i++) {
                    inputs[0].grad[i] += gradOutput[i] * inputs[1].data[i];
                }
            }
            if (inputs.length > 1 && inputs[1].requiresGrad) {
                if (inputs[1].grad == null) inputs[1].grad = new float[inputs[1].numel];
                for (int i = 0; i < gradOutput.length; i++) {
                    inputs[1].grad[i] += gradOutput[i] * inputs[0].data[i];
                }
            }
        }
    }
    
    public static class MatMulNode extends FunctionNode {
        private final boolean transposeA;
        private final boolean transposeB;
        
        public MatMulNode(Tensor[] inputs, Tensor output, boolean transposeA, boolean transposeB) {
            super(inputs, output);
            this.transposeA = transposeA;
            this.transposeB = transposeB;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor A = inputs[0];
            Tensor B = inputs[1];
            
            int m = transposeA ? A.shape[1] : A.shape[0];
            int k = transposeA ? A.shape[0] : A.shape[1];
            int n = transposeB ? B.shape[0] : B.shape[1];
            
            if (A.requiresGrad) {
                if (A.grad == null) A.grad = new float[A.numel];
                float[] gradA = new float[A.numel];
                
                float[] B_T = new float[B.numel];
                if (transposeB) {
                    System.arraycopy(B.data, 0, B_T, 0, B.numel);
                } else {
                    for (int i = 0; i < B.shape[0]; i++) {
                        for (int j = 0; j < B.shape[1]; j++) {
                            B_T[j * B.shape[0] + i] = B.data[i * B.shape[1] + j];
                        }
                    }
                }
                
                for (int i = 0; i < m; i++) {
                    for (int p = 0; p < k; p++) {
                        for (int j = 0; j < n; j++) {
                            int outIdx = i * n + j;
                            int bIdx = p * n + j;
                            int aIdx = transposeA ? p * m + i : i * k + p;
                            gradA[aIdx] += gradOutput[outIdx] * B_T[bIdx];
                        }
                    }
                }
                
                for (int i = 0; i < A.numel; i++) {
                    A.grad[i] += gradA[i];
                }
            }
            
            if (B.requiresGrad) {
                if (B.grad == null) B.grad = new float[B.numel];
                float[] gradB = new float[B.numel];
                
                float[] A_T = new float[A.numel];
                if (transposeA) {
                    System.arraycopy(A.data, 0, A_T, 0, A.numel);
                } else {
                    for (int i = 0; i < A.shape[0]; i++) {
                        for (int j = 0; j < A.shape[1]; j++) {
                            A_T[j * A.shape[0] + i] = A.data[i * A.shape[1] + j];
                        }
                    }
                }
                
                for (int p = 0; p < k; p++) {
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < m; i++) {
                            int outIdx = i * n + j;
                            int aIdx = p * m + i;
                            int bIdx = transposeB ? j * k + p : p * n + j;
                            gradB[bIdx] += gradOutput[outIdx] * A_T[aIdx];
                        }
                    }
                }
                
                for (int i = 0; i < B.numel; i++) {
                    B.grad[i] += gradB[i];
                }
            }
        }
    }
    
    public static class LayerNormNode extends FunctionNode {
        private final float[] normalized;
        private final float[] mean;
        private final float[] variance;
        private final float eps;
        
        public LayerNormNode(Tensor[] inputs, Tensor output, float[] normalized, float[] mean, float[] variance, float eps) {
            super(inputs, output);
            this.normalized = normalized;
            this.mean = mean;
            this.variance = variance;
            this.eps = eps;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            Tensor weight = inputs.length > 1 ? inputs[1] : null;
            Tensor bias = inputs.length > 2 ? inputs[2] : null;
            
            int batchSize = input.shape[0];
            int hiddenSize = input.shape[1];
            
            float[] gradX = new float[input.numel];
            float[] gradW = weight != null && weight.requiresGrad ? new float[weight.numel] : null;
            float[] gradB = bias != null && bias.requiresGrad ? new float[bias.numel] : null;
            
            for (int b = 0; b < batchSize; b++) {
                int base = b * hiddenSize;
                
                float meanGrad = 0;
                float varGrad = 0;
                
                for (int i = 0; i < hiddenSize; i++) {
                    float w = weight != null ? weight.data[i] : 1.0f;
                    float g = gradOutput[base + i];
                    
                    if (gradW != null) {
                        gradW[i] += g * normalized[base + i];
                    }
                    if (gradB != null) {
                        gradB[i] += g;
                    }
                    
                    float xMinusMean = input.data[base + i] - mean[b];
                    float invStd = (float) (1.0 / Math.sqrt(variance[b] + eps));
                    
                    float gradNorm = g * w;
                    float gradVarTerm = gradNorm * xMinusMean * (-0.5f) * (float) Math.pow(variance[b] + eps, -1.5);
                    float gradMeanTerm = gradNorm * (-invStd);
                    
                    gradX[base + i] = gradNorm * invStd;
                    
                    for (int j = 0; j < hiddenSize; j++) {
                        gradX[base + i] += gradVarTerm * (2.0f / hiddenSize) * (input.data[base + j] - mean[b]) * (i == j ? 1 : 0);
                    }
                    gradX[base + i] += gradMeanTerm * (1.0f / hiddenSize);
                    
                    meanGrad += gradMeanTerm + gradVarTerm * (-2.0f / hiddenSize) * (input.data[base + i] - mean[b]);
                    varGrad += gradVarTerm;
                }
                
                for (int i = 0; i < hiddenSize; i++) {
                    gradX[base + i] += meanGrad * (1.0f / hiddenSize);
                }
            }
            
            if (input.requiresGrad) {
                if (input.grad == null) input.grad = new float[input.numel];
                for (int i = 0; i < input.numel; i++) {
                    input.grad[i] += gradX[i];
                }
            }
            
            if (gradW != null) {
                for (int i = 0; i < weight.numel; i++) {
                    weight.grad[i] += gradW[i];
                }
            }
            
            if (gradB != null) {
                for (int i = 0; i < bias.numel; i++) {
                    bias.grad[i] += gradB[i];
                }
            }
        }
    }
    
    public static class GELUNode extends FunctionNode {
        public GELUNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (input.requiresGrad) {
                if (input.grad == null) input.grad = new float[input.numel];
                
                for (int i = 0; i < input.numel; i++) {
                    float x = input.data[i];
                    float cdf = 0.5f * (1.0f + (float) Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715f * x * x * x)));
                    float pdf = (float) (Math.exp(-x * x / 2.0) / Math.sqrt(2.0 * Math.PI));
                    float grad = cdf + x * pdf * (float) Math.sqrt(2.0 / Math.PI) * (1.0f + 0.134145f * x * x);
                    input.grad[i] += gradOutput[i] * grad;
                }
            }
        }
    }
    
    public static class ReLUNode extends FunctionNode {
        public ReLUNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (input.requiresGrad) {
                if (input.grad == null) input.grad = new float[input.numel];
                
                for (int i = 0; i < input.numel; i++) {
                    input.grad[i] += gradOutput[i] * (input.data[i] > 0 ? 1.0f : 0.0f);
                }
            }
        }
    }
    
    public static class SiLUNode extends FunctionNode {
        public SiLUNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (input.requiresGrad) {
                if (input.grad == null) input.grad = new float[input.numel];
                
                for (int i = 0; i < input.numel; i++) {
                    float x = input.data[i];
                    float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
                    float grad = sigmoid * (1.0f + x * (1.0f - sigmoid));
                    input.grad[i] += gradOutput[i] * grad;
                }
            }
        }
    }
    
    public static class SoftmaxNode extends FunctionNode {
        private final int dim;
        
        public SoftmaxNode(Tensor[] inputs, Tensor output, int dim) {
            super(inputs, output);
            this.dim = dim;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            int batchSize = input.shape[0];
            int vocabSize = input.shape[1];
            
            for (int b = 0; b < batchSize; b++) {
                int base = b * vocabSize;
                
                for (int i = 0; i < vocabSize; i++) {
                    float p = output.data[base + i];
                    float gradSum = 0;
                    
                    for (int j = 0; j < vocabSize; j++) {
                        float grad = (i == j ? p * (1 - p) : -p * output.data[base + j]);
                        gradSum += gradOutput[base + j] * grad;
                    }
                    
                    input.grad[base + i] += gradSum;
                }
            }
        }
    }
    
    public static class LogSoftmaxNode extends FunctionNode {
        private final int dim;
        
        public LogSoftmaxNode(Tensor[] inputs, Tensor output, int dim) {
            super(inputs, output);
            this.dim = dim;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            int batchSize = input.shape[0];
            int vocabSize = input.shape[1];
            
            for (int b = 0; b < batchSize; b++) {
                int base = b * vocabSize;
                
                float[] softmax = new float[vocabSize];
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < vocabSize; i++) {
                    maxVal = Math.max(maxVal, input.data[base + i]);
                }
                
                float sum = 0;
                for (int i = 0; i < vocabSize; i++) {
                    softmax[i] = (float) Math.exp(input.data[base + i] - maxVal);
                    sum += softmax[i];
                }
                for (int i = 0; i < vocabSize; i++) {
                    softmax[i] /= sum;
                }
                
                for (int i = 0; i < vocabSize; i++) {
                    float gradSum = gradOutput[base + i];
                    for (int j = 0; j < vocabSize; j++) {
                        gradSum -= gradOutput[base + j] * softmax[j];
                    }
                    input.grad[base + i] += gradSum;
                }
            }
        }
    }
    
    public static class EmbeddingNode extends FunctionNode {
        private final int[] indices;
        
        public EmbeddingNode(Tensor[] inputs, Tensor output, int[] indices) {
            super(inputs, output);
            this.indices = indices;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor weight = inputs[0];
            
            if (!weight.requiresGrad) return;
            if (weight.grad == null) weight.grad = new float[weight.numel];
            
            int embeddingDim = weight.shape[1];
            
            for (int i = 0; i < indices.length; i++) {
                int idx = indices[i];
                int base = idx * embeddingDim;
                
                for (int j = 0; j < embeddingDim; j++) {
                    weight.grad[base + j] += gradOutput[i * embeddingDim + j];
                }
            }
        }
    }
    
    public static class DropoutNode extends FunctionNode {
        private final float dropoutRate;
        private final boolean[] mask;
        
        public DropoutNode(Tensor[] inputs, Tensor output, float dropoutRate, boolean[] mask) {
            super(inputs, output);
            this.dropoutRate = dropoutRate;
            this.mask = mask;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            float scale = 1.0f / (1.0f - dropoutRate);
            
            for (int i = 0; i < input.numel; i++) {
                input.grad[i] += gradOutput[i] * (mask[i] ? scale : 0.0f);
            }
        }
    }
    
    public static class SumNode extends FunctionNode {
        private final int dim;
        
        public SumNode(Tensor[] inputs, Tensor output, int dim) {
            super(inputs, output);
            this.dim = dim;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            int[] shape = input.shape;
            int[] strides = new int[shape.length];
            strides[strides.length - 1] = 1;
            for (int i = strides.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            
            if (dim == -1 || dim == input.ndim) {
                for (int i = 0; i < input.numel; i++) {
                    input.grad[i] += gradOutput[0];
                }
            } else {
                int reducedSize = 1;
                for (int i = 0; i < shape.length; i++) {
                    if (i != dim) reducedSize *= shape[i];
                }
                
                for (int i = 0; i < input.numel; i++) {
                    int[] indices = new int[shape.length];
                    int idx = i;
                    for (int j = 0; j < shape.length; j++) {
                        indices[j] = idx / strides[j];
                        idx %= strides[j];
                    }
                    
                    int gradIdx = 0;
                    for (int j = 0; j < shape.length; j++) {
                        if (j != dim) gradIdx = gradIdx * (j < dim ? shape[j] : shape[j]) + indices[j];
                    }
                    
                    input.grad[i] += gradOutput[gradIdx];
                }
            }
        }
    }
    
    public static class MeanNode extends FunctionNode {
        private final int dim;
        private final int keepDim;
        
        public MeanNode(Tensor[] inputs, Tensor output, int dim, int keepDim) {
            super(inputs, output);
            this.dim = dim;
            this.keepDim = keepDim;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            int reduceSize = 1;
            if (dim == -1) {
                reduceSize = input.numel;
            } else {
                reduceSize = input.shape[dim];
            }
            float scale = 1.0f / reduceSize;
            
            if (dim == -1) {
                for (int i = 0; i < input.numel; i++) {
                    input.grad[i] += gradOutput[0] * scale;
                }
            } else {
                int[] shape = input.shape;
                int[] strides = new int[shape.length];
                strides[strides.length - 1] = 1;
                for (int i = strides.length - 2; i >= 0; i--) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                
                int reducedSize = 1;
                for (int i = 0; i < shape.length; i++) {
                    if (i != dim) reducedSize *= shape[i];
                }
                
                for (int i = 0; i < input.numel; i++) {
                    int[] indices = new int[shape.length];
                    int idx = i;
                    for (int j = 0; j < shape.length; j++) {
                        indices[j] = idx / strides[j];
                        idx %= strides[j];
                    }
                    
                    int gradIdx = 0;
                    int currentStride = 1;
                    for (int j = shape.length - 1; j >= 0; j--) {
                        if (j != dim) {
                            gradIdx += indices[j] * currentStride;
                            currentStride *= shape[j];
                        }
                    }
                    
                    input.grad[i] += gradOutput[gradIdx] * scale;
                }
            }
        }
    }
    
    public static class ReshapeNode extends FunctionNode {
        public ReshapeNode(Tensor[] inputs, Tensor output) {
            super(inputs, output);
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            System.arraycopy(gradOutput, 0, input.grad, 0, input.numel);
        }
    }
    
    public static class TransposeNode extends FunctionNode {
        private final int dim0;
        private final int dim1;
        
        public TransposeNode(Tensor[] inputs, Tensor output, int dim0, int dim1) {
            super(inputs, output);
            this.dim0 = dim0;
            this.dim1 = dim1;
        }
        
        @Override
        public void backward(float[] gradOutput) {
            Tensor input = inputs[0];
            
            if (!input.requiresGrad) return;
            if (input.grad == null) input.grad = new float[input.numel];
            
            int[] shape = input.shape;
            int[] newShape = Arrays.copyOf(shape, shape.length);
            int temp = newShape[dim0];
            newShape[dim0] = newShape[dim1];
            newShape[dim1] = temp;
            
            int[] strides = new int[shape.length];
            strides[strides.length - 1] = 1;
            for (int i = strides.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            
            int[] newStrides = new int[newShape.length];
            newStrides[newStrides.length - 1] = 1;
            for (int i = newStrides.length - 2; i >= 0; i--) {
                newStrides[i] = newStrides[i + 1] * newShape[i + 1];
            }
            
            for (int i = 0; i < input.numel; i++) {
                int[] indices = new int[shape.length];
                int idx = i;
                for (int j = 0; j < shape.length; j++) {
                    indices[j] = idx / strides[j];
                    idx %= strides[j];
                }
                
                int[] transposedIndices = Arrays.copyOf(indices, indices.length);
                int tempIdx = transposedIndices[dim0];
                transposedIndices[dim0] = transposedIndices[dim1];
                transposedIndices[dim1] = tempIdx;
                
                int newIndex = 0;
                for (int j = 0; j < newShape.length; j++) {
                    newIndex += transposedIndices[j] * newStrides[j];
                }
                
                input.grad[i] += gradOutput[newIndex];
            }
        }
    }
}