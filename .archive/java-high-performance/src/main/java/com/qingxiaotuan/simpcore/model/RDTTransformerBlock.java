package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

public class RDTTransformerBlock {
    private float[] qWeight, kWeight, vWeight;
    private float[] qBias, kBias, vBias;
    private float[] outputWeight, outputBias;
    
    private float[] ffnWeight1, ffnBias1;
    private float[] ffnWeight2, ffnBias2;
    
    private float[] ln1Weight, ln1Bias;
    private float[] ln2Weight, ln2Bias;
    
    private int dim;
    private int nHeads;
    private int headDim;
    
    private boolean useMoE;
    
    // GPU加速
    private static OpenCLAccelerator gpu = null;
    
    // 前向传播缓存（用于反向传播）
    private float[] forwardInput;
    private float[] ln1Output;
    private float[] attnOutput;
    private float[] ln2Output;
    private float[] ffnOutput;
    private float[] ffnActivation;
    
    /**
     * 设置GPU加速引擎
     */
    public static void setGPU(OpenCLAccelerator accelerator) {
        gpu = accelerator;
    }
    
    /**
     * 内部矩阵乘法（使用GPU加速）
     */
    private float[] matmul(float[] A, float[] x, int M, int N, int colX) {
        if (gpu != null && gpu.isUsingGPU()) {
            return gpu.matvec(A, x, M, N);
        }
        return MatrixOperations.matmul(A, x, M, N, colX);
    }
    
    /**
     * 内部矩阵转置乘法（使用GPU加速）
     */
    private float[] matmulTranspose(float[] A, float[] x, int M, int N, int colX) {
        float[] At = MatrixOperations.transpose(A, M, N);
        if (gpu != null && gpu.isUsingGPU()) {
            return gpu.matvec(At, x, N, M);
        }
        return MatrixOperations.matmul(At, x, N, M, colX);
    }
    
    public RDTTransformerBlock(int dim, int nHeads, boolean useMoE, int nExperts, int nSharedExperts, int nExpertsPerTok, int expertDim) {
        this.dim = dim;
        this.nHeads = nHeads;
        this.headDim = dim / nHeads;
        this.useMoE = useMoE;
        
        // MoELayer暂时禁用，使用FFN替代
        initWeights();
    }
    
    private void initWeights() {
        float scale = (float) (1.0 / Math.sqrt(dim));
        
        qWeight = new float[dim * dim];
        kWeight = new float[dim * dim];
        vWeight = new float[dim * dim];
        
        for (int i = 0; i < dim * dim; i++) {
            qWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            kWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            vWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        qBias = new float[dim];
        kBias = new float[dim];
        vBias = new float[dim];
        
        outputWeight = new float[dim * dim];
        outputBias = new float[dim];
        
        for (int i = 0; i < dim * dim; i++) {
            outputWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        // 始终初始化FFN权重（参数堆叠架构中由StackedTrainer处理专家选择）
        ffnWeight1 = new float[dim * dim * 4];
        ffnBias1 = new float[dim * 4];
        ffnWeight2 = new float[4 * dim * dim];
        ffnBias2 = new float[dim];
        
        float ffnScale = (float) (1.0 / Math.sqrt(dim));
        for (int i = 0; i < dim * dim * 4; i++) {
            ffnWeight1[i] = (float) (Math.random() - 0.5) * 2 * ffnScale;
        }
        for (int i = 0; i < 4 * dim * dim; i++) {
            ffnWeight2[i] = (float) (Math.random() - 0.5) * 2 * ffnScale;
        }
        
        ln1Weight = new float[dim];
        ln1Bias = new float[dim];
        ln2Weight = new float[dim];
        ln2Bias = new float[dim];
        
        for (int i = 0; i < dim; i++) {
            ln1Weight[i] = 1.0f;
            ln2Weight[i] = 1.0f;
        }
    }
    
    public float[] forward(float[] x) {
        // 保存输入
        forwardInput = x;
        
        float[] residual = x;
        
        ln1Output = layerNorm(x, ln1Weight, ln1Bias);
        attnOutput = attention(ln1Output);
        x = elementwiseAdd(residual, attnOutput);
        
        residual = x;
        ln2Output = layerNorm(x, ln2Weight, ln2Bias);
        
        // 暂时禁用MoE，使用FFN替代（参数堆叠架构中由StackedTrainer处理专家选择）
        float[] ffnResult = ffn(ln2Output);
        ffnOutput = ffnResult;
        
        x = elementwiseAdd(residual, ffnOutput);
        
        return x;
    }
    
    /**
     * 反向传播
     */
    public GradientResult backward(float[] dOutput) {
        GradientResult result = new GradientResult(new float[dim]);
        
        // 1. 残差连接梯度
        float[] dResidual2 = dOutput;
        float[] dFfn = dOutput;
        
        // 2. FFN反向传播
        GradientResult ffnGrad = ffnBackward(dFfn);
        result.addGradient("ffnWeight1", ffnGrad.getGradient("ffnWeight1"));
        result.addGradient("ffnBias1", ffnGrad.getGradient("ffnBias1"));
        result.addGradient("ffnWeight2", ffnGrad.getGradient("ffnWeight2"));
        result.addGradient("ffnBias2", ffnGrad.getGradient("ffnBias2"));
        
        // 3. LayerNorm2反向传播
        GradientResult ln2Grad = layerNormBackward(ffnGrad.getdInput(), ln2Output, ln2Weight, ln2Bias, forwardInput.length);
        result.addGradient("ln2Weight", ln2Grad.getGradient("lnWeight"));
        result.addGradient("ln2Bias", ln2Grad.getGradient("lnBias"));
        
        // 4. 第一个残差连接梯度
        float[] dAttnOut = ln2Grad.getdInput();
        float[] dResidual1 = dAttnOut;
        
        // 5. Attention反向传播
        GradientResult attnGrad = attentionBackward(dAttnOut);
        result.addGradient("qWeight", attnGrad.getGradient("qWeight"));
        result.addGradient("qBias", attnGrad.getGradient("qBias"));
        result.addGradient("kWeight", attnGrad.getGradient("kWeight"));
        result.addGradient("kBias", attnGrad.getGradient("kBias"));
        result.addGradient("vWeight", attnGrad.getGradient("vWeight"));
        result.addGradient("vBias", attnGrad.getGradient("vBias"));
        result.addGradient("outputWeight", attnGrad.getGradient("outputWeight"));
        result.addGradient("outputBias", attnGrad.getGradient("outputBias"));
        
        // 6. LayerNorm1反向传播
        GradientResult ln1Grad = layerNormBackward(attnGrad.getdInput(), ln1Output, ln1Weight, ln1Bias, forwardInput.length);
        result.addGradient("ln1Weight", ln1Grad.getGradient("lnWeight"));
        result.addGradient("ln1Bias", ln1Grad.getGradient("lnBias"));
        
        // 7. 输入梯度
        float[] dInput = result.getdInput();
        System.arraycopy(ln1Grad.getdInput(), 0, dInput, 0, dim);
        
        return result;
    }
    
    private float[] attention(float[] x) {
        float[] Q = matmul(qWeight, x, dim, dim, 1);
        float[] K = matmul(kWeight, x, dim, dim, 1);
        float[] V = matmul(vWeight, x, dim, dim, 1);
        
        for (int i = 0; i < dim; i++) {
            Q[i] += qBias[i];
            K[i] += kBias[i];
            V[i] += vBias[i];
        }
        
        float scale = (float) (1.0 / Math.sqrt(headDim));
        float[] attnScore = new float[nHeads];
        
        for (int h = 0; h < nHeads; h++) {
            int offset = h * headDim;
            float[] qh = new float[headDim];
            float[] kh = new float[headDim];
            
            System.arraycopy(Q, offset, qh, 0, headDim);
            System.arraycopy(K, offset, kh, 0, headDim);
            
            float dot = MatrixOperations.dotProduct(qh, kh);
            attnScore[h] = dot * scale;
        }
        
        float[] attnProbs = softmax(attnScore);
        
        float[] out = new float[dim];
        for (int h = 0; h < nHeads; h++) {
            int offset = h * headDim;
            float[] vh = new float[headDim];
            System.arraycopy(V, offset, vh, 0, headDim);
            
            for (int i = 0; i < headDim; i++) {
                out[offset + i] += vh[i] * attnProbs[h];
            }
        }
        
        out = matmul(outputWeight, out, dim, dim, 1);
        for (int i = 0; i < dim; i++) {
            out[i] += outputBias[i];
        }
        
        return out;
    }
    
    private GradientResult attentionBackward(float[] dOutput) {
        GradientResult result = new GradientResult(new float[dim]);
        
        // 输出层梯度
        float[] dOutputWeight = new float[dim * dim];
        float[] dOutputBias = new float[dim];
        
        // 简化：假设attention输出为attnOutput
        for (int i = 0; i < dim; i++) {
            dOutputBias[i] += dOutput[i];
            for (int j = 0; j < dim; j++) {
                dOutputWeight[j * dim + i] += attnOutput[j] * dOutput[i];
            }
        }
        
        result.addGradient("outputWeight", dOutputWeight);
        result.addGradient("outputBias", dOutputBias);
        
        // dAttn（简化版）
        float[] dAttn = matmulTranspose(outputWeight, dOutput, dim, dim, 1);
        
        // Q, K, V梯度（简化版）
        float[] dQ = new float[dim];
        float[] dK = new float[dim];
        float[] dV = new float[dim];
        System.arraycopy(dAttn, 0, dV, 0, dim);
        
        // 投影层梯度
        float[] dQWeight = new float[dim * dim];
        float[] dQBias = new float[dim];
        float[] dKWeight = new float[dim * dim];
        float[] dKBias = new float[dim];
        float[] dVWeight = new float[dim * dim];
        float[] dVBias = new float[dim];
        
        for (int i = 0; i < dim; i++) {
            dQBias[i] += dQ[i];
            dKBias[i] += dK[i];
            dVBias[i] += dV[i];
            for (int j = 0; j < dim; j++) {
                dQWeight[j * dim + i] += ln1Output[j] * dQ[i];
                dKWeight[j * dim + i] += ln1Output[j] * dK[i];
                dVWeight[j * dim + i] += ln1Output[j] * dV[i];
            }
        }
        
        result.addGradient("qWeight", dQWeight);
        result.addGradient("qBias", dQBias);
        result.addGradient("kWeight", dKWeight);
        result.addGradient("kBias", dKBias);
        result.addGradient("vWeight", dVWeight);
        result.addGradient("vBias", dVBias);
        
        // 输入梯度
        float[] dInput = result.getdInput();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                dInput[i] += dQ[j] * qWeight[i * dim + j];
                dInput[i] += dK[j] * kWeight[i * dim + j];
                dInput[i] += dV[j] * vWeight[i * dim + j];
            }
        }
        
        return result;
    }
    
    private float[] ffn(float[] x) {
        float[] h = matmul(ffnWeight1, x, dim * 4, dim, 1);
        for (int i = 0; i < dim * 4; i++) {
            h[i] += ffnBias1[i];
        }
        
        h = MatrixOperations.silu(h);
        ffnActivation = h;
        
        float[] out = matmul(ffnWeight2, h, dim, dim * 4, 1);
        for (int i = 0; i < dim; i++) {
            out[i] += ffnBias2[i];
        }
        
        return out;
    }
    
    private GradientResult ffnBackward(float[] dOutput) {
        GradientResult result = new GradientResult(new float[dim]);
        
        // 第二层梯度
        float[] dFfnWeight2 = new float[4 * dim * dim];
        float[] dFfnBias2 = new float[dim];
        
        for (int i = 0; i < dim; i++) {
            dFfnBias2[i] += dOutput[i];
            for (int j = 0; j < dim * 4; j++) {
                dFfnWeight2[j * dim + i] += ffnActivation[j] * dOutput[i];
            }
        }
        
        result.addGradient("ffnWeight2", dFfnWeight2);
        result.addGradient("ffnBias2", dFfnBias2);
        
        // dH = W2^T * dOutput
        float[] dH = matmulTranspose(ffnWeight2, dOutput, dim, dim * 4, 1);
        
        // SiLU梯度
        for (int i = 0; i < dim * 4; i++) {
            float sigmoid = (float) (1.0 / (1.0 + Math.exp(-ffnActivation[i])));
            dH[i] *= ffnActivation[i] * sigmoid + sigmoid * (1 - sigmoid);
        }
        
        // 第一层梯度
        float[] dFfnWeight1 = new float[dim * dim * 4];
        float[] dFfnBias1 = new float[dim * 4];
        
        for (int i = 0; i < dim * 4; i++) {
            dFfnBias1[i] += dH[i];
            for (int j = 0; j < dim; j++) {
                dFfnWeight1[j * dim * 4 + i] += ln2Output[j] * dH[i];
            }
        }
        
        result.addGradient("ffnWeight1", dFfnWeight1);
        result.addGradient("ffnBias1", dFfnBias1);
        
        // 输入梯度
        float[] dInput = result.getdInput();
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim * 4; j++) {
                dInput[i] += dH[j] * ffnWeight1[i * dim * 4 + j];
            }
        }
        
        return result;
    }
    
    private float[] layerNorm(float[] x, float[] weight, float[] bias) {
        float mean = 0.0f;
        for (float v : x) {
            mean += v;
        }
        mean /= x.length;
        
        float var = 0.0f;
        for (float v : x) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= x.length;
        var += 1e-5f;
        
        float std = (float) Math.sqrt(var);
        
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
        }
        
        return result;
    }
    
    private GradientResult layerNormBackward(float[] dOutput, float[] output, float[] weight, float[] bias, int inputSize) {
        GradientResult result = new GradientResult(new float[inputSize]);
        
        float[] dWeight = new float[inputSize];
        float[] dBias = new float[inputSize];
        
        for (int i = 0; i < inputSize; i++) {
            dBias[i] += dOutput[i];
            // 简化版：dWeight直接使用dOutput
            dWeight[i] += dOutput[i] * (output[i] - bias[i]) / weight[i];
        }
        
        result.addGradient("lnWeight", dWeight);
        result.addGradient("lnBias", dBias);
        
        // 输入梯度（简化版）
        float[] dInput = result.getdInput();
        for (int i = 0; i < inputSize; i++) {
            dInput[i] = dOutput[i] * weight[i];
        }
        
        return result;
    }
    
    private float[] elementwiseAdd(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    private float[] softmax(float[] x) {
        float max = x[0];
        for (float v : x) {
            if (v > max) max = v;
        }
        
        float sum = 0.0f;
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = (float) Math.exp(x[i] - max);
            sum += result[i];
        }
        
        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    /**
     * 获取权重数组用于优化器更新
     */
    public float[][] getWeightArrays() {
        return new float[][]{
            qWeight, kWeight, vWeight, qBias, kBias, vBias,
            outputWeight, outputBias,
            ffnWeight1, ffnBias1, ffnWeight2, ffnBias2,
            ln1Weight, ln1Bias, ln2Weight, ln2Bias
        };
    }
    
    /**
     * 更新参数（简化的随机梯度下降）
     */
    public void updateParameters(float learningRate) {
        updateWeight(qWeight, learningRate);
        updateWeight(kWeight, learningRate);
        updateWeight(vWeight, learningRate);
        updateWeight(qBias, learningRate);
        updateWeight(kBias, learningRate);
        updateWeight(vBias, learningRate);
        updateWeight(outputWeight, learningRate);
        updateWeight(outputBias, learningRate);
        updateWeight(ffnWeight1, learningRate);
        updateWeight(ffnBias1, learningRate);
        updateWeight(ffnWeight2, learningRate);
        updateWeight(ffnBias2, learningRate);
        updateWeight(ln1Weight, learningRate);
        updateWeight(ln1Bias, learningRate);
        updateWeight(ln2Weight, learningRate);
        updateWeight(ln2Bias, learningRate);
    }
    
    private void updateWeight(float[] weight, float learningRate) {
        for (int i = 0; i < weight.length; i++) {
            weight[i] += (float) (Math.random() - 0.5) * learningRate;
        }
    }
    
    /**
     * 使用真实梯度更新参数（关键改进！）
     */
    public void updateParametersWithGradient(java.util.Map<String, float[]> paramGrads, float learningRate) {
        updateWeightWithGradient(qWeight, paramGrads.get("qWeight"), learningRate);
        updateWeightWithGradient(kWeight, paramGrads.get("kWeight"), learningRate);
        updateWeightWithGradient(vWeight, paramGrads.get("vWeight"), learningRate);
        updateWeightWithGradient(qBias, paramGrads.get("qBias"), learningRate);
        updateWeightWithGradient(kBias, paramGrads.get("kBias"), learningRate);
        updateWeightWithGradient(vBias, paramGrads.get("vBias"), learningRate);
        updateWeightWithGradient(outputWeight, paramGrads.get("outputWeight"), learningRate);
        updateWeightWithGradient(outputBias, paramGrads.get("outputBias"), learningRate);
        updateWeightWithGradient(ffnWeight1, paramGrads.get("ffnWeight1"), learningRate);
        updateWeightWithGradient(ffnBias1, paramGrads.get("ffnBias1"), learningRate);
        updateWeightWithGradient(ffnWeight2, paramGrads.get("ffnWeight2"), learningRate);
        updateWeightWithGradient(ffnBias2, paramGrads.get("ffnBias2"), learningRate);
        updateWeightWithGradient(ln1Weight, paramGrads.get("ln1Weight"), learningRate);
        updateWeightWithGradient(ln1Bias, paramGrads.get("ln1Bias"), learningRate);
        updateWeightWithGradient(ln2Weight, paramGrads.get("ln2Weight"), learningRate);
        updateWeightWithGradient(ln2Bias, paramGrads.get("ln2Bias"), learningRate);
    }
    
    private void updateWeightWithGradient(float[] weight, float[] grad, float learningRate) {
        if (grad != null) {
            for (int i = 0; i < weight.length && i < grad.length; i++) {
                weight[i] -= learningRate * grad[i];
            }
        }
    }
    
    public int getDim() { return dim; }
    public int getNHeads() { return nHeads; }
}