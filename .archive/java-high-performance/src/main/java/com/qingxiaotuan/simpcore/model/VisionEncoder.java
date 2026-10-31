package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

/**
 * Vision Encoder - 视觉编码器
 * 基于卷积层的图像特征提取架构
 */
public class VisionEncoder {
    
    private final int dModel;
    private final int inputChannels;
    
    // 卷积层参数
    private int[][] filterSizes;    // 每个卷积层的滤波器大小
    private int[][] strides;        // 每个卷积层的步长
    private int[][] padding;        // 每个卷积层的填充
    private int[] outputChannels;   // 每个卷积层的输出通道数
    
    // 权重参数
    private float[][][][][] convWeights;  // [layer][outCh][inCh][kh][kw]
    private float[][] convBiases;          // [layer][outCh]
    
    // 注意力层
    private MultiHeadAttention attention;
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 缓存
    private float[][][][] layerOutputs;
    
    public VisionEncoder(int dModel, int inputChannels) {
        this.dModel = dModel;
        this.inputChannels = inputChannels;
        initArchitecture();
        initWeights();
        initAttention();
        initializeGPU();
    }
    
    private void initArchitecture() {
        // 4层卷积架构
        int nLayers = 4;
        filterSizes = new int[nLayers][2];
        strides = new int[nLayers][2];
        padding = new int[nLayers][2];
        outputChannels = new int[nLayers];
        
        // Layer 1: 64通道, 3x3卷积, stride 2
        filterSizes[0] = new int[]{3, 3};
        strides[0] = new int[]{2, 2};
        padding[0] = new int[]{1, 1};
        outputChannels[0] = 64;
        
        // Layer 2: 128通道, 3x3卷积, stride 2
        filterSizes[1] = new int[]{3, 3};
        strides[1] = new int[]{2, 2};
        padding[1] = new int[]{1, 1};
        outputChannels[1] = 128;
        
        // Layer 3: 256通道, 3x3卷积, stride 2
        filterSizes[2] = new int[]{3, 3};
        strides[2] = new int[]{2, 2};
        padding[2] = new int[]{1, 1};
        outputChannels[2] = 256;
        
        // Layer 4: 512通道, 3x3卷积, stride 2
        filterSizes[3] = new int[]{3, 3};
        strides[3] = new int[]{2, 2};
        padding[3] = new int[]{1, 1};
        outputChannels[3] = 512;
    }
    
    private void initWeights() {
        int nLayers = filterSizes.length;
        convWeights = new float[nLayers][][][][];
        convBiases = new float[nLayers][];
        
        int prevChannels = inputChannels;
        for (int l = 0; l < nLayers; l++) {
            int outCh = outputChannels[l];
            int inCh = prevChannels;
            int kh = filterSizes[l][0];
            int kw = filterSizes[l][1];
            
            float scale = (float) (1.0 / Math.sqrt(inCh * kh * kw));
            convWeights[l] = new float[outCh][inCh][kh][kw];
            convBiases[l] = new float[outCh];
            
            for (int oc = 0; oc < outCh; oc++) {
                for (int ic = 0; ic < inCh; ic++) {
                    for (int i = 0; i < kh; i++) {
                        for (int j = 0; j < kw; j++) {
                            convWeights[l][oc][ic][i][j] = (float) (Math.random() - 0.5) * 2 * scale;
                        }
                    }
                }
            }
            
            prevChannels = outCh;
        }
    }
    
    private void initAttention() {
        attention = new MultiHeadAttention(dModel, 8);
    }
    
    private void initializeGPU() {
        try {
            gpu = OpenCLAccelerator.getInstance();
            useGPU = gpu.isUsingGPU() && gpu.isInitialized();
        } catch (Exception e) {
            useGPU = false;
        }
    }
    
    /**
     * 前向传播
     * @param image 输入图像 [height, width, channels]
     * @return 视觉特征 [numPatches, dModel]
     */
    public float[][] forward(float[][][] image) {
        int nLayers = filterSizes.length;
        layerOutputs = new float[nLayers][][][];
        
        float[][][] current = image;
        
        // 卷积层前向传播
        for (int l = 0; l < nLayers; l++) {
            int inCh = current[0][0].length;
            int outCh = outputChannels[l];
            int kh = filterSizes[l][0];
            int kw = filterSizes[l][1];
            int sh = strides[l][0];
            int sw = strides[l][1];
            int ph = padding[l][0];
            int pw = padding[l][1];
            
            // 计算输出尺寸
            int h = current.length;
            int w = current[0].length;
            int outH = (h + 2 * ph - kh) / sh + 1;
            int outW = (w + 2 * pw - kw) / sw + 1;
            
            // 应用卷积
            float[][][] output = new float[outH][outW][outCh];
            
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    for (int oc = 0; oc < outCh; oc++) {
                        float sum = 0;
                        for (int ic = 0; ic < inCh; ic++) {
                            for (int i = 0; i < kh; i++) {
                                for (int j = 0; j < kw; j++) {
                                    int ih = oh * sh + i - ph;
                                    int iw = ow * sw + j - pw;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        sum += current[ih][iw][ic] * convWeights[l][oc][ic][i][j];
                                    }
                                }
                            }
                        }
                        output[oh][ow][oc] = ReLU(sum + convBiases[l][oc]);
                    }
                }
            }
            
            current = output;
            layerOutputs[l] = output;
        }
        
        // 将特征图展平为序列
        int h = current.length;
        int w = current[0].length;
        int ch = current[0][0].length;
        int numPatches = h * w;
        
        float[][] patches = new float[numPatches][ch];
        int idx = 0;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                patches[idx++] = current[i][j];
            }
        }
        
        // 如果通道数不等于dModel，进行线性投影
        if (ch != dModel) {
            patches = projectToDModel(patches, ch, dModel);
        }
        
        // 应用自注意力
        float[][] result = attention.forward(patches);
        
        return result;
    }
    
    /**
     * 将特征投影到dModel维度
     */
    private float[][] projectToDModel(float[][] input, int inDim, int outDim) {
        int n = input.length;
        float[][] output = new float[n][outDim];
        
        // 创建投影矩阵
        float[] projWeight = new float[inDim * outDim];
        float[] projBias = new float[outDim];
        float scale = (float) (1.0 / Math.sqrt(inDim));
        
        for (int i = 0; i < inDim * outDim; i++) {
            projWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < outDim; j++) {
                float sum = 0;
                for (int k = 0; k < inDim; k++) {
                    sum += input[i][k] * projWeight[k * outDim + j];
                }
                output[i][j] = sum + projBias[j];
            }
        }
        
        return output;
    }
    
    /**
     * ReLU激活函数
     */
    private float ReLU(float x) {
        return Math.max(0, x);
    }
    
    /**
     * 获取视觉特征的CLS标记（第一个token）
     */
    public float[] getCLSToken(float[][] features) {
        if (features.length > 0) {
            return features[0];
        }
        return new float[dModel];
    }
    
    public MultiHeadAttention getAttention() { return attention; }
    public int getOutputChannels(int layer) { return outputChannels[layer]; }
    public int getDModel() { return dModel; }
}
