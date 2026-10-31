package com.qingxiaotuan.simpcore.vision.model;

import java.util.*;

public class VisionModel {
    
    private static final Random random = new Random(42);
    
    private int inputChannels;
    private int inputHeight;
    private int inputWidth;
    private int embeddingDim;
    
    private ConvLayer[] convLayers;
    private BatchNormLayer[] bnLayers;
    private LinearLayer finalLinear;
    
    private float[] weights;
    private float[] gradients;
    private int paramCount;
    
    public VisionModel(int inputChannels, int inputHeight, int inputWidth, int embeddingDim) {
        this.inputChannels = inputChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.embeddingDim = embeddingDim;
        
        buildModel();
        initWeights();
    }
    
    private void buildModel() {
        List<ConvLayer> convList = new ArrayList<>();
        List<BatchNormLayer> bnList = new ArrayList<>();
        
        convList.add(new ConvLayer(3, 32, 3, 2, 1));
        bnList.add(new BatchNormLayer(32));
        
        convList.add(new ConvLayer(32, 64, 3, 1, 1));
        bnList.add(new BatchNormLayer(64));
        
        convList.add(new ConvLayer(64, 128, 3, 2, 1));
        bnList.add(new BatchNormLayer(128));
        
        convList.add(new ConvLayer(128, 128, 3, 1, 1));
        bnList.add(new BatchNormLayer(128));
        
        convList.add(new ConvLayer(128, 256, 3, 2, 1));
        bnList.add(new BatchNormLayer(256));
        
        convList.add(new ConvLayer(256, 256, 3, 1, 1));
        bnList.add(new BatchNormLayer(256));
        
        convList.add(new ConvLayer(256, 512, 3, 2, 1));
        bnList.add(new BatchNormLayer(512));
        
        convList.add(new ConvLayer(512, 512, 3, 1, 1));
        bnList.add(new BatchNormLayer(512));
        
        convList.add(new ConvLayer(512, 512, 3, 1, 1));
        bnList.add(new BatchNormLayer(512));
        
        convList.add(new ConvLayer(512, 768, 3, 2, 1));
        bnList.add(new BatchNormLayer(768));
        
        convLayers = convList.toArray(new ConvLayer[0]);
        bnLayers = bnList.toArray(new BatchNormLayer[0]);
        
        finalLinear = new LinearLayer(768, embeddingDim);
        
        calculateParamCount();
    }
    
    private void initWeights() {
        int totalParams = getTotalParameters();
        weights = new float[totalParams];
        gradients = new float[totalParams];
        
        int offset = 0;
        
        for (ConvLayer conv : convLayers) {
            int params = conv.getParamCount();
            for (int i = 0; i < params; i++) {
                weights[offset + i] = (random.nextFloat() - 0.5f) * (float) Math.sqrt(2.0 / conv.getFanIn());
            }
            offset += params;
        }
        
        for (BatchNormLayer bn : bnLayers) {
            int params = bn.getParamCount();
            for (int i = 0; i < params; i++) {
                if (i < bn.getChannels()) {
                    weights[offset + i] = 1.0f;
                } else {
                    weights[offset + i] = 0.0f;
                }
            }
            offset += params;
        }
        
        int params = finalLinear.getParamCount();
        for (int i = 0; i < params; i++) {
            weights[offset + i] = (random.nextFloat() - 0.5f) * (float) Math.sqrt(2.0 / finalLinear.getFanIn());
        }
    }
    
    private void calculateParamCount() {
        paramCount = 0;
        for (ConvLayer conv : convLayers) {
            paramCount += conv.getParamCount();
        }
        for (BatchNormLayer bn : bnLayers) {
            paramCount += bn.getParamCount();
        }
        paramCount += finalLinear.getParamCount();
    }
    
    public float[] forward(float[] input) {
        float[] x = input;
        int offset = 0;
        int h = inputHeight;
        int w = inputWidth;
        
        for (int i = 0; i < convLayers.length; i++) {
            x = convLayers[i].forward(x, h, w, weights, offset);
            offset += convLayers[i].getParamCount();
            
            x = bnLayers[i].forward(x, weights, offset);
            offset += bnLayers[i].getParamCount();
            
            x = relu(x);
            
            if (convLayers[i].getStride() == 2) {
                h = (h + 2 * convLayers[i].getPadding() - convLayers[i].getKernelSize()) / 2 + 1;
                w = (w + 2 * convLayers[i].getPadding() - convLayers[i].getKernelSize()) / 2 + 1;
            }
        }
        
        x = globalAvgPool(x, h, w, 768);
        x = finalLinear.forward(x, weights, offset);
        
        return x;
    }
    
    public float[] backward(float[] gradOutput) {
        float[] grad = gradOutput;
        int offset = getTotalParameters() - finalLinear.getParamCount();
        
        grad = finalLinear.backward(grad, gradients, offset, weights);
        offset -= finalLinear.getParamCount();
        
        grad = globalAvgPoolBackward(grad, inputHeight, inputWidth);
        
        for (int i = convLayers.length - 1; i >= 0; i--) {
            grad = reluBackward(grad);
            
            grad = bnLayers[i].backward(grad, gradients, offset);
            offset -= bnLayers[i].getParamCount();
            
            grad = convLayers[i].backward(grad, gradients, offset);
            offset -= convLayers[i].getParamCount();
        }
        
        return grad;
    }
    
    public void updateParameters(float learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradients[i];
        }
        Arrays.fill(gradients, 0);
    }
    
    private float[] relu(float[] x) {
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.max(0, x[i]);
        }
        return result;
    }
    
    private float[] reluBackward(float[] grad) {
        return grad;
    }
    
    private float[] globalAvgPool(float[] x, int h, int w, int c) {
        float[] result = new float[c];
        int size = h * w;
        
        for (int i = 0; i < c; i++) {
            float sum = 0;
            for (int j = 0; j < size; j++) {
                sum += x[j * c + i];
            }
            result[i] = sum / size;
        }
        
        return result;
    }
    
    private float[] globalAvgPoolBackward(float[] grad, int h, int w) {
        int c = grad.length;
        int size = (h / 32) * (w / 32);
        float[] result = new float[size * c];
        
        for (int i = 0; i < c; i++) {
            float val = grad[i] / size;
            for (int j = 0; j < size; j++) {
                result[j * c + i] = val;
            }
        }
        
        return result;
    }
    
    public int getTotalParameters() {
        return paramCount;
    }
    
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    public float[] getWeights() {
        return weights;
    }
    
    public float[] getGradients() {
        return gradients;
    }
    
    public static class ConvLayer {
        private int inChannels;
        private int outChannels;
        private int kernelSize;
        private int stride;
        private int padding;
        
        public ConvLayer(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }
        
        public int getParamCount() {
            return kernelSize * kernelSize * inChannels * outChannels;
        }
        
        public int getFanIn() {
            return kernelSize * kernelSize * inChannels;
        }
        
        public int getStride() {
            return stride;
        }
        
        public int getPadding() {
            return padding;
        }
        
        public int getKernelSize() {
            return kernelSize;
        }
        
        public float[] forward(float[] input, int h, int w, float[] weights, int offset) {
            int outH = (h + 2 * padding - kernelSize) / stride + 1;
            int outW = (w + 2 * padding - kernelSize) / stride + 1;
            float[] output = new float[outH * outW * outChannels];
            
            Arrays.fill(output, 0);
            
            for (int oc = 0; oc < outChannels; oc++) {
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int kh = 0; kh < kernelSize; kh++) {
                        for (int kw = 0; kw < kernelSize; kw++) {
                            float wVal = weights[offset + oc * kernelSize * kernelSize * inChannels + 
                                                  kh * kernelSize * inChannels + 
                                                  kw * inChannels + ic];
                            
                            for (int oh = 0; oh < outH; oh++) {
                                for (int ow = 0; ow < outW; ow++) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        int inIdx = (ih * w + iw) * inChannels + ic;
                                        int outIdx = (oh * outW + ow) * outChannels + oc;
                                        output[outIdx] += input[inIdx] * wVal;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return output;
        }
        
        public float[] backward(float[] gradOutput, float[] gradients, int offset) {
            return gradOutput;
        }
    }
    
    public static class BatchNormLayer {
        private int channels;
        
        public BatchNormLayer(int channels) {
            this.channels = channels;
        }
        
        public int getParamCount() {
            return channels * 2;
        }
        
        public int getChannels() {
            return channels;
        }
        
        public float[] forward(float[] input, float[] weights, int offset) {
            float[] result = new float[input.length];
            int size = input.length / channels;
            
            for (int c = 0; c < channels; c++) {
                float mean = 0;
                for (int i = 0; i < size; i++) {
                    mean += input[i * channels + c];
                }
                mean /= size;
                
                float var = 0;
                for (int i = 0; i < size; i++) {
                    var += (input[i * channels + c] - mean) * (input[i * channels + c] - mean);
                }
                var /= size;
                var += 1e-5f;
                var = (float) Math.sqrt(var);
                
                float gamma = weights[offset + c];
                float beta = weights[offset + channels + c];
                
                for (int i = 0; i < size; i++) {
                    result[i * channels + c] = gamma * (input[i * channels + c] - mean) / var + beta;
                }
            }
            
            return result;
        }
        
        public float[] backward(float[] gradOutput, float[] gradients, int offset) {
            return gradOutput;
        }
    }
    
    public static class LinearLayer {
        private int inFeatures;
        private int outFeatures;
        
        public LinearLayer(int inFeatures, int outFeatures) {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
        }
        
        public int getParamCount() {
            return inFeatures * outFeatures;
        }
        
        public int getFanIn() {
            return inFeatures;
        }
        
        public float[] forward(float[] input, float[] weights, int offset) {
            float[] output = new float[outFeatures];
            Arrays.fill(output, 0);
            
            for (int i = 0; i < outFeatures; i++) {
                for (int j = 0; j < inFeatures; j++) {
                    output[i] += input[j] * weights[offset + i * inFeatures + j];
                }
            }
            
            return output;
        }
        
        public float[] backward(float[] gradOutput, float[] gradients, int offset, float[] weights) {
            float[] gradInput = new float[inFeatures];
            
            for (int i = 0; i < outFeatures; i++) {
                for (int j = 0; j < inFeatures; j++) {
                    gradients[offset + i * inFeatures + j] += gradOutput[i];
                    gradInput[j] += gradOutput[i] * weights[offset + i * inFeatures + j];
                }
            }
            
            return gradInput;
        }
    }
    
    public static void main(String[] args) {
        VisionModel model = new VisionModel(3, 224, 224, 512);
        System.out.println("=== Vision Model Summary ===");
        System.out.println("Total parameters: " + model.getTotalParameters());
        System.out.println("Embedding dimension: " + model.getEmbeddingDim());
        System.out.println("Input: " + model.inputChannels + "x" + model.inputHeight + "x" + model.inputWidth);
        
        float[] dummyInput = new float[3 * 224 * 224];
        for (int i = 0; i < dummyInput.length; i++) {
            dummyInput[i] = random.nextFloat();
        }
        
        long startTime = System.currentTimeMillis();
        float[] output = model.forward(dummyInput);
        long endTime = System.currentTimeMillis();
        
        System.out.println();
        System.out.println("Forward pass test:");
        System.out.println("Output length: " + output.length);
        System.out.println("Time: " + (endTime - startTime) + " ms");
        System.out.println("First 5 values: " + Arrays.toString(Arrays.copyOf(output, 5)));
        
        float[] grad = new float[model.getEmbeddingDim()];
        for (int i = 0; i < grad.length; i++) {
            grad[i] = 1.0f;
        }
        
        startTime = System.currentTimeMillis();
        float[] gradInput = model.backward(grad);
        endTime = System.currentTimeMillis();
        
        System.out.println();
        System.out.println("Backward pass test:");
        System.out.println("Gradient length: " + gradInput.length);
        System.out.println("Time: " + (endTime - startTime) + " ms");
    }
}