package com.qingxiaotuan.simpcore.autograd;

import java.util.*;

public final class AutogradEngine {
    
    private AutogradEngine() {}
    
    public static Tensor add(Tensor a, Tensor b) {
        if (!a.requiresGrad && !b.requiresGrad) {
            float[] resultData = new float[a.numel];
            for (int i = 0; i < a.numel; i++) {
                resultData[i] = a.data[i] + b.data[i];
            }
            return new Tensor(resultData, a.shape);
        }
        
        float[] resultData = new float[a.numel];
        for (int i = 0; i < a.numel; i++) {
            resultData[i] = a.data[i] + b.data[i];
        }
        
        Tensor result = new Tensor(resultData, a.shape);
        result.requiresGrad = a.requiresGrad || b.requiresGrad;
        result.gradFn = new FunctionNode.AddNode(new Tensor[]{a, b}, result);
        
        return result;
    }
    
    public static Tensor sub(Tensor a, Tensor b) {
        if (!a.requiresGrad && !b.requiresGrad) {
            float[] resultData = new float[a.numel];
            for (int i = 0; i < a.numel; i++) {
                resultData[i] = a.data[i] - b.data[i];
            }
            return new Tensor(resultData, a.shape);
        }
        
        float[] resultData = new float[a.numel];
        for (int i = 0; i < a.numel; i++) {
            resultData[i] = a.data[i] - b.data[i];
        }
        
        float[] bNeg = new float[b.numel];
        for (int i = 0; i < b.numel; i++) {
            bNeg[i] = -b.data[i];
        }
        Tensor bNegTensor = new Tensor(bNeg, b.shape);
        bNegTensor.requiresGrad = b.requiresGrad;
        
        Tensor result = new Tensor(resultData, a.shape);
        result.requiresGrad = a.requiresGrad || b.requiresGrad;
        result.gradFn = new FunctionNode.AddNode(new Tensor[]{a, bNegTensor}, result);
        
        return result;
    }
    
    public static Tensor mul(Tensor a, Tensor b) {
        if (!a.requiresGrad && !b.requiresGrad) {
            float[] resultData = new float[a.numel];
            for (int i = 0; i < a.numel; i++) {
                resultData[i] = a.data[i] * b.data[i];
            }
            return new Tensor(resultData, a.shape);
        }
        
        float[] resultData = new float[a.numel];
        for (int i = 0; i < a.numel; i++) {
            resultData[i] = a.data[i] * b.data[i];
        }
        
        Tensor result = new Tensor(resultData, a.shape);
        result.requiresGrad = a.requiresGrad || b.requiresGrad;
        result.gradFn = new FunctionNode.MulNode(new Tensor[]{a, b}, result);
        
        return result;
    }
    
    public static Tensor div(Tensor a, Tensor b) {
        if (!a.requiresGrad && !b.requiresGrad) {
            float[] resultData = new float[a.numel];
            for (int i = 0; i < a.numel; i++) {
                resultData[i] = a.data[i] / b.data[i];
            }
            return new Tensor(resultData, a.shape);
        }
        
        float[] resultData = new float[a.numel];
        float[] bInv = new float[b.numel];
        for (int i = 0; i < a.numel; i++) {
            bInv[i] = 1.0f / b.data[i];
            resultData[i] = a.data[i] * bInv[i];
        }
        
        Tensor bInvTensor = new Tensor(bInv, b.shape);
        bInvTensor.requiresGrad = b.requiresGrad;
        
        Tensor result = new Tensor(resultData, a.shape);
        result.requiresGrad = a.requiresGrad || b.requiresGrad;
        result.gradFn = new FunctionNode.MulNode(new Tensor[]{a, bInvTensor}, result);
        
        return result;
    }
    
    public static Tensor matmul(Tensor a, Tensor b) {
        return matmul(a, b, false, false);
    }
    
    public static Tensor matmul(Tensor a, Tensor b, boolean transposeA, boolean transposeB) {
        int m = transposeA ? a.shape[1] : a.shape[0];
        int k = transposeA ? a.shape[0] : a.shape[1];
        int n = transposeB ? b.shape[0] : b.shape[1];
        
        float[] resultData = new float[m * n];
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    int aIdx = transposeA ? p * a.shape[1] + i : i * a.shape[1] + p;
                    int bIdx = transposeB ? j * b.shape[1] + p : p * b.shape[1] + j;
                    sum += a.data[aIdx] * b.data[bIdx];
                }
                resultData[i * n + j] = sum;
            }
        }
        
        Tensor result = new Tensor(resultData, m, n);
        result.requiresGrad = a.requiresGrad || b.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.MatMulNode(new Tensor[]{a, b}, result, transposeA, transposeB);
        }
        
        return result;
    }
    
    public static Tensor layerNorm(Tensor input, Tensor weight, Tensor bias) {
        return layerNorm(input, weight, bias, 1e-5f);
    }
    
    public static Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
        int batchSize = input.shape[0];
        int hiddenSize = input.shape[1];
        
        float[] resultData = new float[input.numel];
        float[] normalized = new float[input.numel];
        float[] mean = new float[batchSize];
        float[] variance = new float[batchSize];
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * hiddenSize;
            
            float sum = 0;
            for (int i = 0; i < hiddenSize; i++) {
                sum += input.data[base + i];
            }
            mean[b] = sum / hiddenSize;
            
            float sumSq = 0;
            for (int i = 0; i < hiddenSize; i++) {
                float diff = input.data[base + i] - mean[b];
                sumSq += diff * diff;
            }
            variance[b] = sumSq / hiddenSize;
            
            float invStd = (float) (1.0 / Math.sqrt(variance[b] + eps));
            
            for (int i = 0; i < hiddenSize; i++) {
                float normVal = (input.data[base + i] - mean[b]) * invStd;
                normalized[base + i] = normVal;
                
                float w = weight != null ? weight.data[i] : 1.0f;
                float bVal = bias != null ? bias.data[i] : 0.0f;
                resultData[base + i] = normVal * w + bVal;
            }
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad || 
                             (weight != null && weight.requiresGrad) || 
                             (bias != null && bias.requiresGrad);
        
        if (result.requiresGrad) {
            List<Tensor> inputs = new ArrayList<>();
            inputs.add(input);
            if (weight != null) inputs.add(weight);
            if (bias != null) inputs.add(bias);
            result.gradFn = new FunctionNode.LayerNormNode(
                inputs.toArray(new Tensor[0]), result, normalized, mean, variance, eps);
        }
        
        return result;
    }
    
    public static Tensor gelu(Tensor input) {
        float[] resultData = new float[input.numel];
        
        for (int i = 0; i < input.numel; i++) {
            float x = input.data[i];
            resultData[i] = 0.5f * x * (1.0f + (float) Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715f * x * x * x)));
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.GELUNode(new Tensor[]{input}, result);
        }
        
        return result;
    }
    
    public static Tensor relu(Tensor input) {
        float[] resultData = new float[input.numel];
        
        for (int i = 0; i < input.numel; i++) {
            resultData[i] = Math.max(0, input.data[i]);
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.ReLUNode(new Tensor[]{input}, result);
        }
        
        return result;
    }
    
    public static Tensor silu(Tensor input) {
        float[] resultData = new float[input.numel];
        
        for (int i = 0; i < input.numel; i++) {
            float x = input.data[i];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
            resultData[i] = x * sigmoid;
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.SiLUNode(new Tensor[]{input}, result);
        }
        
        return result;
    }
    
    public static Tensor softmax(Tensor input) {
        return softmax(input, -1);
    }
    
    public static Tensor softmax(Tensor input, int dim) {
        if (dim < 0) {
            dim = input.ndim + dim;
        }
        
        int batchSize = input.shape[0];
        int vocabSize = input.shape[1];
        
        float[] resultData = new float[input.numel];
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * vocabSize;
            
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                maxVal = Math.max(maxVal, input.data[base + i]);
            }
            
            float sum = 0;
            for (int i = 0; i < vocabSize; i++) {
                resultData[base + i] = (float) Math.exp(input.data[base + i] - maxVal);
                sum += resultData[base + i];
            }
            
            for (int i = 0; i < vocabSize; i++) {
                resultData[base + i] /= sum;
            }
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.SoftmaxNode(new Tensor[]{input}, result, dim);
        }
        
        return result;
    }
    
    public static Tensor logSoftmax(Tensor input) {
        return logSoftmax(input, -1);
    }
    
    public static Tensor logSoftmax(Tensor input, int dim) {
        if (dim < 0) {
            dim = input.ndim + dim;
        }
        
        int batchSize = input.shape[0];
        int vocabSize = input.shape[1];
        
        float[] resultData = new float[input.numel];
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * vocabSize;
            
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                maxVal = Math.max(maxVal, input.data[base + i]);
            }
            
            float sum = 0;
            for (int i = 0; i < vocabSize; i++) {
                sum += (float) Math.exp(input.data[base + i] - maxVal);
            }
            float logSum = maxVal + (float) Math.log(sum);
            
            for (int i = 0; i < vocabSize; i++) {
                resultData[base + i] = input.data[base + i] - logSum;
            }
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.LogSoftmaxNode(new Tensor[]{input}, result, dim);
        }
        
        return result;
    }
    
    public static Tensor embedding(Tensor weight, int[] indices) {
        int vocabSize = weight.shape[0];
        int embeddingDim = weight.shape[1];
        
        float[] resultData = new float[indices.length * embeddingDim];
        
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            int base = idx * embeddingDim;
            
            for (int j = 0; j < embeddingDim; j++) {
                resultData[i * embeddingDim + j] = weight.data[base + j];
            }
        }
        
        Tensor result = new Tensor(resultData, indices.length, embeddingDim);
        result.requiresGrad = weight.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.EmbeddingNode(new Tensor[]{weight}, result, indices);
        }
        
        return result;
    }
    
    public static Tensor dropout(Tensor input, float dropoutRate) {
        return dropout(input, dropoutRate, true);
    }
    
    public static Tensor dropout(Tensor input, float dropoutRate, boolean training) {
        if (!training || dropoutRate == 0.0f) {
            return input.clone();
        }
        
        float[] resultData = new float[input.numel];
        boolean[] mask = new boolean[input.numel];
        float scale = 1.0f / (1.0f - dropoutRate);
        Random random = new Random();
        
        for (int i = 0; i < input.numel; i++) {
            if (random.nextFloat() < dropoutRate) {
                mask[i] = false;
                resultData[i] = 0.0f;
            } else {
                mask[i] = true;
                resultData[i] = input.data[i] * scale;
            }
        }
        
        Tensor result = new Tensor(resultData, input.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.DropoutNode(new Tensor[]{input}, result, dropoutRate, mask);
        }
        
        return result;
    }
    
    public static Tensor sum(Tensor input) {
        return sum(input, -1);
    }
    
    public static Tensor sum(Tensor input, int dim) {
        if (dim < 0) {
            dim = input.ndim + dim;
        }
        
        if (dim == -1 || dim == input.ndim) {
            float sumVal = 0;
            for (float val : input.data) {
                sumVal += val;
            }
            Tensor result = new Tensor(new float[]{sumVal}, new int[]{});
            result.requiresGrad = input.requiresGrad;
            
            if (result.requiresGrad) {
                result.gradFn = new FunctionNode.SumNode(new Tensor[]{input}, result, dim);
            }
            return result;
        }
        
        int[] newShape = new int[input.ndim - 1];
        int idx = 0;
        for (int i = 0; i < input.ndim; i++) {
            if (i != dim) {
                newShape[idx++] = input.shape[i];
            }
        }
        
        int reducedSize = 1;
        for (int dimVal : newShape) {
            reducedSize *= dimVal;
        }
        
        float[] resultData = new float[reducedSize];
        
        int[] strides = new int[input.shape.length];
        strides[strides.length - 1] = 1;
        for (int i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * input.shape[i + 1];
        }
        
        for (int i = 0; i < input.numel; i++) {
            int[] indices = new int[input.shape.length];
            int idxCopy = i;
            for (int j = 0; j < input.shape.length; j++) {
                indices[j] = idxCopy / strides[j];
                idxCopy %= strides[j];
            }
            
            int gradIdx = 0;
            int currentStride = 1;
            for (int j = input.shape.length - 1; j >= 0; j--) {
                if (j != dim) {
                    gradIdx += indices[j] * currentStride;
                    currentStride *= input.shape[j];
                }
            }
            
            resultData[gradIdx] += input.data[i];
        }
        
        Tensor result = new Tensor(resultData, newShape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.SumNode(new Tensor[]{input}, result, dim);
        }
        
        return result;
    }
    
    public static Tensor mean(Tensor input) {
        return mean(input, -1);
    }
    
    public static Tensor mean(Tensor input, int dim) {
        Tensor sumResult = sum(input, dim);
        
        if (!input.requiresGrad) {
            return sumResult;
        }
        
        int reduceSize = 1;
        if (dim == -1) {
            reduceSize = input.numel;
        } else {
            reduceSize = input.shape[dim];
        }
        float scale = 1.0f / reduceSize;
        
        float[] resultData = new float[sumResult.numel];
        for (int i = 0; i < sumResult.numel; i++) {
            resultData[i] = sumResult.data[i] * scale;
        }
        
        Tensor result = new Tensor(resultData, sumResult.shape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.MeanNode(new Tensor[]{input}, result, dim, 0);
        }
        
        return result;
    }
    
    public static Tensor reshape(Tensor input, int... newShape) {
        Tensor result = input.reshape(newShape);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.ReshapeNode(new Tensor[]{input}, result);
        }
        
        return result;
    }
    
    public static Tensor transpose(Tensor input, int dim0, int dim1) {
        Tensor result = input.transpose(dim0, dim1);
        result.requiresGrad = input.requiresGrad;
        
        if (result.requiresGrad) {
            result.gradFn = new FunctionNode.TransposeNode(new Tensor[]{input}, result, dim0, dim1);
        }
        
        return result;
    }
    
    public static Tensor permute(Tensor input, int... dims) {
        return input.permute(dims);
    }
    
    public static Tensor squeeze(Tensor input) {
        return input.squeeze();
    }
    
    public static Tensor squeeze(Tensor input, int dim) {
        return input.squeeze(dim);
    }
    
    public static Tensor unsqueeze(Tensor input, int dim) {
        return input.unsqueeze(dim);
    }
    
    public static Tensor expand(Tensor input, int... newShape) {
        return input.expand(newShape);
    }
    
    public static Tensor clone(Tensor input) {
        return input.clone();
    }
    
    public static Tensor detach(Tensor input) {
        return input.detach();
    }
    
    public static float mseLoss(Tensor output, Tensor target) {
        float[] diff = new float[output.numel];
        for (int i = 0; i < output.numel; i++) {
            diff[i] = output.data[i] - target.data[i];
        }
        
        float mse = 0;
        for (float val : diff) {
            mse += val * val;
        }
        return mse / output.numel;
    }
    
    public static Tensor mseLossTensor(Tensor output, Tensor target) {
        float[] diff = new float[output.numel];
        float[] squaredDiff = new float[output.numel];
        for (int i = 0; i < output.numel; i++) {
            diff[i] = output.data[i] - target.data[i];
            squaredDiff[i] = diff[i] * diff[i];
        }
        
        float sum = 0;
        for (float val : squaredDiff) {
            sum += val;
        }
        float mse = sum / output.numel;
        
        Tensor result = new Tensor(new float[]{mse}, new int[]{});
        result.requiresGrad = output.requiresGrad;
        
        if (result.requiresGrad) {
            float[] grad = new float[output.numel];
            for (int i = 0; i < output.numel; i++) {
                grad[i] = 2.0f * diff[i] / output.numel;
            }
            result.grad = grad;
        }
        
        return result;
    }
    
    public static float crossEntropyLoss(Tensor logits, int[] targets) {
        int batchSize = logits.shape[0];
        int vocabSize = logits.shape[1];
        
        float loss = 0;
        for (int b = 0; b < batchSize; b++) {
            int base = b * vocabSize;
            int target = targets[b];
            
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                maxVal = Math.max(maxVal, logits.data[base + i]);
            }
            
            float sum = 0;
            for (int i = 0; i < vocabSize; i++) {
                sum += (float) Math.exp(logits.data[base + i] - maxVal);
            }
            
            loss += maxVal + (float) Math.log(sum) - logits.data[base + target];
        }
        
        return loss / batchSize;
    }
    
    public static Tensor crossEntropyLossTensor(Tensor logits, int[] targets) {
        int batchSize = logits.shape[0];
        int vocabSize = logits.shape[1];
        
        float loss = 0;
        float[] grad = logits.requiresGrad ? new float[logits.numel] : null;
        
        for (int b = 0; b < batchSize; b++) {
            int base = b * vocabSize;
            int target = targets[b];
            
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                maxVal = Math.max(maxVal, logits.data[base + i]);
            }
            
            float sum = 0;
            float[] exp = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                exp[i] = (float) Math.exp(logits.data[base + i] - maxVal);
                sum += exp[i];
            }
            
            loss += maxVal + (float) Math.log(sum) - logits.data[base + target];
            
            if (grad != null) {
                for (int i = 0; i < vocabSize; i++) {
                    grad[base + i] = exp[i] / sum;
                }
                grad[base + target] -= 1.0f;
                for (int i = 0; i < vocabSize; i++) {
                    grad[base + i] /= batchSize;
                }
            }
        }
        
        loss /= batchSize;
        
        Tensor result = new Tensor(new float[]{loss}, new int[]{});
        result.requiresGrad = logits.requiresGrad;
        
        if (result.requiresGrad) {
            result.grad = grad;
        }
        
        return result;
    }
    
    public static void clipGradients(Tensor[] tensors, float maxNorm) {
        float totalNorm = 0;
        for (Tensor tensor : tensors) {
            if (tensor.grad != null) {
                for (float val : tensor.grad) {
                    totalNorm += val * val;
                }
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Tensor tensor : tensors) {
                if (tensor.grad != null) {
                    for (int i = 0; i < tensor.grad.length; i++) {
                        tensor.grad[i] *= scale;
                    }
                }
            }
        }
    }
    
    public static void zeroGrad(Tensor[] tensors) {
        for (Tensor tensor : tensors) {
            tensor.zeroGrad();
        }
    }
    
    public static Tensor cat(Tensor[] tensors, int dim) {
        if (dim < 0) {
            dim = tensors[0].ndim + dim;
        }
        
        int[] shape = tensors[0].shape;
        int catSize = 0;
        for (Tensor t : tensors) {
            catSize += t.shape[dim];
        }
        
        int[] newShape = Arrays.copyOf(shape, shape.length);
        newShape[dim] = catSize;
        
        int totalNumel = 1;
        for (int dimVal : newShape) {
            totalNumel *= dimVal;
        }
        
        float[] resultData = new float[totalNumel];
        
        int[] strides = new int[shape.length];
        strides[strides.length - 1] = 1;
        for (int i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        int offset = 0;
        for (Tensor t : tensors) {
            for (int i = 0; i < t.numel; i++) {
                int[] indices = new int[t.shape.length];
                int idx = i;
                for (int j = 0; j < t.shape.length; j++) {
                    indices[j] = idx / strides[j];
                    idx %= strides[j];
                }
                
                indices[dim] += offset;
                
                int newIndex = 0;
                for (int j = 0; j < newShape.length; j++) {
                    newIndex = newIndex * newShape[j] + indices[j];
                }
                
                resultData[newIndex] = t.data[i];
            }
            offset += t.shape[dim];
        }
        
        Tensor result = new Tensor(resultData, newShape);
        boolean requiresGrad = false;
        for (Tensor t : tensors) {
            if (t.requiresGrad) {
                requiresGrad = true;
                break;
            }
        }
        result.requiresGrad = requiresGrad;
        
        return result;
    }
    
    public static Tensor stack(Tensor[] tensors, int dim) {
        if (dim < 0) {
            dim = tensors[0].ndim + dim;
        }
        
        int[] shape = tensors[0].shape;
        int[] newShape = new int[shape.length + 1];
        
        int idx = 0;
        for (int i = 0; i < newShape.length; i++) {
            if (i == dim) {
                newShape[i] = tensors.length;
            } else {
                newShape[i] = shape[idx++];
            }
        }
        
        int totalNumel = 1;
        for (int dimVal : newShape) {
            totalNumel *= dimVal;
        }
        
        float[] resultData = new float[totalNumel];
        
        int[] strides = new int[shape.length];
        strides[strides.length - 1] = 1;
        for (int i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        for (int tIdx = 0; tIdx < tensors.length; tIdx++) {
            Tensor t = tensors[tIdx];
            for (int i = 0; i < t.numel; i++) {
                int[] indices = new int[t.shape.length];
                int idxCopy = i;
                for (int j = 0; j < t.shape.length; j++) {
                    indices[j] = idxCopy / strides[j];
                    idxCopy %= strides[j];
                }
                
                int[] newIndices = new int[newShape.length];
                int newIdx = 0;
                for (int j = 0; j < newShape.length; j++) {
                    if (j == dim) {
                        newIndices[j] = tIdx;
                    } else {
                        newIndices[j] = indices[newIdx++];
                    }
                }
                
                int newIndex = 0;
                for (int j = 0; j < newShape.length; j++) {
                    newIndex = newIndex * newShape[j] + newIndices[j];
                }
                
                resultData[newIndex] = t.data[i];
            }
        }
        
        Tensor result = new Tensor(resultData, newShape);
        boolean requiresGrad = false;
        for (Tensor t : tensors) {
            if (t.requiresGrad) {
                requiresGrad = true;
                break;
            }
        }
        result.requiresGrad = requiresGrad;
        
        return result;
    }
}