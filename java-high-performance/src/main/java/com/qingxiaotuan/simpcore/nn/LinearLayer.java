package com.qingxiaotuan.simpcore.nn;

import java.util.*;

public class LinearLayer {
    float[] weights;
    float[] bias;
    float[] weightGrad;
    float[] biasGrad;
    float[] lastInput;
    int inFeatures;
    int outFeatures;
    
    public LinearLayer(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weights = new float[inFeatures * outFeatures];
        this.bias = new float[outFeatures];
        this.weightGrad = new float[inFeatures * outFeatures];
        this.biasGrad = new float[outFeatures];
        this.lastInput = null;
        
        Random r = new Random(42);
        float scale = (float) Math.sqrt(1.0 / inFeatures);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (r.nextGaussian()) * scale;
        }
    }
    
    public float[] forward(float[] input) {
        this.lastInput = input.clone();
        int seqLen = input.length / inFeatures;
        float[] output = new float[seqLen * outFeatures];
        
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < outFeatures; i++) {
                output[t * outFeatures + i] = bias[i];
                for (int j = 0; j < inFeatures; j++) {
                    output[t * outFeatures + i] += input[t * inFeatures + j] * weights[j * outFeatures + i];
                }
            }
        }
        
        return output;
    }
    
    public float[] backward(float[] gradOutput) {
        int seqLen = gradOutput.length / outFeatures;
        float[] gradInput = new float[seqLen * inFeatures];
        
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < inFeatures; j++) {
                for (int i = 0; i < outFeatures; i++) {
                    gradInput[t * inFeatures + j] += gradOutput[t * outFeatures + i] * weights[j * outFeatures + i];
                }
            }
            
            for (int i = 0; i < outFeatures; i++) {
                biasGrad[i] += gradOutput[t * outFeatures + i];
                for (int j = 0; j < inFeatures; j++) {
                    weightGrad[j * outFeatures + i] += gradOutput[t * outFeatures + i] * lastInput[t * inFeatures + j];
                }
            }
        }
        
        return gradInput;
    }
    
    public void update(float lr, float weightDecay) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= lr * (weightGrad[i] + weightDecay * weights[i]);
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] -= lr * biasGrad[i];
        }
        Arrays.fill(weightGrad, 0);
        Arrays.fill(biasGrad, 0);
    }

    /**
     * 使用Adam优化器更新参数。
     *
     * @param optimizer Adam优化器实例
     */
    public void updateWithAdam(com.qingxiaotuan.simpcore.training.AdamOptimizer optimizer) {
        optimizer.update(weights, weightGrad);
        optimizer.update(bias, biasGrad);
        Arrays.fill(weightGrad, 0);
        Arrays.fill(biasGrad, 0);
    }
    
    public long getParameterCount() {
        return (long) inFeatures * outFeatures + outFeatures;
    }
    
    public void save(java.io.DataOutputStream dos) throws Exception {
        for (float v : weights) dos.writeFloat(v);
        for (float v : bias) dos.writeFloat(v);
    }
    
    public void load(java.io.DataInputStream dis) throws Exception {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = dis.readFloat();
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = dis.readFloat();
        }
    }
}