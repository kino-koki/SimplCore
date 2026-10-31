package com.qingxiaotuan.simpcore.nn;

import java.util.*;

public class MultiHeadAttention {
    
    public static class AttentionResult {
        public final float[] output;
        public final float[][] attentionWeights;
        public final float[] q, k, v;
        public final float[][] headsOutput;
        
        public AttentionResult(float[] output, float[][] attentionWeights, float[] q, float[] k, float[] v, float[][] headsOutput) {
            this.output = output;
            this.attentionWeights = attentionWeights;
            this.q = q;
            this.k = k;
            this.v = v;
            this.headsOutput = headsOutput;
        }
    }
    
    private final int nHeads;
    private final int dModel;
    private final int dHead;
    private final float[] wQ;
    private final float[] wK;
    private final float[] wV;
    private final float[] wO;
    private final float[] bQ;
    private final float[] bK;
    private final float[] bV;
    private final float[] bO;
    private final float[] wQGrad;
    private final float[] wKGrad;
    private final float[] wVGrad;
    private final float[] wOGrad;
    private final float[] bQGrad;
    private final float[] bKGrad;
    private final float[] bVGrad;
    private final float[] bOGrad;
    private final float scale;
    private float[] lastInput;
    private int lastSeqLen;
    
    public MultiHeadAttention(int dModel, int nHeads) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dHead = dModel / nHeads;
        this.scale = (float) Math.sqrt(dHead);
        
        int paramSize = dModel * dModel;
        
        wQ = new float[paramSize];
        wK = new float[paramSize];
        wV = new float[paramSize];
        wO = new float[paramSize];
        bQ = new float[dModel];
        bK = new float[dModel];
        bV = new float[dModel];
        bO = new float[dModel];
        
        wQGrad = new float[paramSize];
        wKGrad = new float[paramSize];
        wVGrad = new float[paramSize];
        wOGrad = new float[paramSize];
        bQGrad = new float[dModel];
        bKGrad = new float[dModel];
        bVGrad = new float[dModel];
        bOGrad = new float[dModel];
        
        Random r = new Random(42);
        float weightScale = (float) Math.sqrt(1.0 / dModel);
        
        initWeights(wQ, weightScale, r);
        initWeights(wK, weightScale, r);
        initWeights(wV, weightScale, r);
        initWeights(wO, weightScale, r);
    }
    
    private void initWeights(float[] weights, float scale, Random r) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (r.nextDouble() * 2 - 1) * scale;
        }
    }
    
    public AttentionResult forward(float[] input, int seqLen) {
        this.lastInput = input.clone();
        this.lastSeqLen = seqLen;
        
        float[] q = linear(input, wQ, bQ, seqLen);
        float[] k = linear(input, wK, bK, seqLen);
        float[] v = linear(input, wV, bV, seqLen);
        
        float[][] headsOutput = new float[nHeads][seqLen * dHead];
        float[][] allWeights = new float[nHeads][seqLen * seqLen];
        
        for (int h = 0; h < nHeads; h++) {
            float[] qHead = extractHead(q, h, seqLen);
            float[] kHead = extractHead(k, h, seqLen);
            float[] vHead = extractHead(v, h, seqLen);
            
            float[] attnWeights = computeAttentionWeights(qHead, kHead, seqLen);
            float[] attnOutput = applyAttention(vHead, attnWeights, seqLen);
            
            headsOutput[h] = attnOutput;
            allWeights[h] = attnWeights;
        }
        
        float[] concatenated = concatenateHeads(headsOutput, seqLen);
        float[] output = linear(concatenated, wO, bO, seqLen);
        
        return new AttentionResult(output, allWeights, q, k, v, headsOutput);
    }
    
    public float[] backward(float[] gradOutput, AttentionResult result) {
        int seqLen = lastSeqLen;
        int actualSeqLen = gradOutput.length / dModel;
        
        float[] gradConcatenated = linearBackward(gradOutput, wO, bO, actualSeqLen, wOGrad, bOGrad);
        
        float[][] gradHeads = splitHeads(gradConcatenated, seqLen);
        
        float[] gradQ = new float[seqLen * dModel];
        float[] gradK = new float[seqLen * dModel];
        float[] gradV = new float[seqLen * dModel];
        
        for (int h = 0; h < nHeads; h++) {
            float[] qHead = extractHead(result.q, h, seqLen);
            float[] kHead = extractHead(result.k, h, seqLen);
            float[] vHead = extractHead(result.v, h, seqLen);
            
            float[] gradHead = attentionBackward(gradHeads[h], qHead, kHead, vHead, result.attentionWeights[h], seqLen);
            
            float[] gradQHead = new float[seqLen * dHead];
            float[] gradKHead = new float[seqLen * dHead];
            float[] gradVHead = new float[seqLen * dHead];
            
            for (int t = 0; t < seqLen; t++) {
                for (int i = 0; i < dHead; i++) {
                    gradQHead[t * dHead + i] = gradHead[t * dHead * 3 + i];
                    gradKHead[t * dHead + i] = gradHead[t * dHead * 3 + dHead + i];
                    gradVHead[t * dHead + i] = gradHead[t * dHead * 3 + dHead * 2 + i];
                }
            }
            
            mergeHead(gradQHead, gradQ, h, seqLen);
            mergeHead(gradKHead, gradK, h, seqLen);
            mergeHead(gradVHead, gradV, h, seqLen);
        }
        
        float[] gradInput = new float[seqLen * dModel];
        
        float[] gradQInput = linearBackward(gradQ, wQ, bQ, seqLen, wQGrad, bQGrad);
        float[] gradKInput = linearBackward(gradK, wK, bK, seqLen, wKGrad, bKGrad);
        float[] gradVInput = linearBackward(gradV, wV, bV, seqLen, wVGrad, bVGrad);
        
        for (int i = 0; i < gradInput.length; i++) {
            gradInput[i] = gradQInput[i] + gradKInput[i] + gradVInput[i];
        }
        
        return gradInput;
    }
    
    private float[] attentionBackward(float[] gradOutput, float[] q, float[] k, float[] v, float[] attnWeights, int seqLen) {
        float[] gradQ = new float[seqLen * dHead];
        float[] gradK = new float[seqLen * dHead];
        float[] gradV = new float[seqLen * dHead];
        
        float[] gradAttn = new float[seqLen * seqLen];
        for (int t = 0; t < seqLen; t++) {
            for (int s = 0; s < seqLen; s++) {
                for (int i = 0; i < dHead; i++) {
                    gradAttn[t * seqLen + s] += gradOutput[t * dHead + i] * v[s * dHead + i];
                }
            }
        }
        
        for (int t = 0; t < seqLen; t++) {
            for (int s = 0; s < seqLen; s++) {
                float g = gradAttn[t * seqLen + s];
                float p = attnWeights[t * seqLen + s];
                
                for (int i = 0; i < dHead; i++) {
                    gradQ[t * dHead + i] += g * p * k[s * dHead + i] / scale;
                    gradK[s * dHead + i] += g * p * q[t * dHead + i] / scale;
                }
            }
        }
        
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dHead; i++) {
                for (int s = 0; s < seqLen; s++) {
                    gradV[s * dHead + i] += gradOutput[t * dHead + i] * attnWeights[t * seqLen + s];
                }
            }
        }
        
        float[] result = new float[seqLen * dHead * 3];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dHead; i++) {
                result[t * dHead * 3 + i] = gradQ[t * dHead + i];
                result[t * dHead * 3 + dHead + i] = gradK[t * dHead + i];
                result[t * dHead * 3 + dHead * 2 + i] = gradV[t * dHead + i];
            }
        }
        
        return result;
    }
    
    private float[] linear(float[] input, float[] weights, float[] bias, int seqLen) {
        float[] output = new float[seqLen * dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dModel; i++) {
                float sum = bias[i];
                for (int j = 0; j < dModel; j++) {
                    sum += input[t * dModel + j] * weights[j * dModel + i];
                }
                output[t * dModel + i] = sum;
            }
        }
        return output;
    }
    
    private float[] linearBackward(float[] gradOutput, float[] weights, float[] bias, int seqLen, float[] weightGrad, float[] biasGrad) {
        float[] gradInput = new float[seqLen * dModel];
        
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    gradInput[t * dModel + j] += gradOutput[t * dModel + i] * weights[j * dModel + i];
                }
            }
            
            for (int i = 0; i < dModel; i++) {
                biasGrad[i] += gradOutput[t * dModel + i];
                for (int j = 0; j < dModel; j++) {
                    weightGrad[j * dModel + i] += gradOutput[t * dModel + i] * lastInput[t * dModel + j];
                }
            }
        }
        
        return gradInput;
    }
    
    private float[] extractHead(float[] input, int headIdx, int seqLen) {
        float[] head = new float[seqLen * dHead];
        int offset = headIdx * dHead;
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dHead; i++) {
                head[t * dHead + i] = input[t * dModel + offset + i];
            }
        }
        return head;
    }
    
    private void mergeHead(float[] head, float[] output, int headIdx, int seqLen) {
        int offset = headIdx * dHead;
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dHead; i++) {
                output[t * dModel + offset + i] += head[t * dHead + i];
            }
        }
    }
    
    private float[][] splitHeads(float[] input, int seqLen) {
        float[][] heads = new float[nHeads][seqLen * dHead];
        for (int h = 0; h < nHeads; h++) {
            heads[h] = extractHead(input, h, seqLen);
        }
        return heads;
    }
    
    private float[] computeAttentionWeights(float[] q, float[] k, int seqLen) {
        float[] scores = new float[seqLen * seqLen];
        
        for (int t = 0; t < seqLen; t++) {
            for (int s = 0; s < seqLen; s++) {
                float score = 0;
                for (int i = 0; i < dHead; i++) {
                    score += q[t * dHead + i] * k[s * dHead + i];
                }
                scores[t * seqLen + s] = score / scale;
            }
        }
        
        for (int t = 0; t < seqLen; t++) {
            for (int s = t + 1; s < seqLen; s++) {
                scores[t * seqLen + s] = Float.NEGATIVE_INFINITY;
            }
        }
        
        return softmax2D(scores, seqLen);
    }
    
    private float[] softmax2D(float[] scores, int seqLen) {
        float[] result = new float[seqLen * seqLen];
        for (int t = 0; t < seqLen; t++) {
            float maxScore = scores[t * seqLen];
            for (int s = 0; s < seqLen; s++) {
                if (scores[t * seqLen + s] > maxScore) {
                    maxScore = scores[t * seqLen + s];
                }
            }
            
            float sum = 0;
            for (int s = 0; s < seqLen; s++) {
                result[t * seqLen + s] = (float) Math.exp(scores[t * seqLen + s] - maxScore);
                sum += result[t * seqLen + s];
            }
            
            for (int s = 0; s < seqLen; s++) {
                result[t * seqLen + s] /= sum;
            }
        }
        return result;
    }
    
    private float[] applyAttention(float[] v, float[] weights, int seqLen) {
        float[] output = new float[seqLen * dHead];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dHead; i++) {
                float sum = 0;
                for (int s = 0; s < seqLen; s++) {
                    sum += weights[t * seqLen + s] * v[s * dHead + i];
                }
                output[t * dHead + i] = sum;
            }
        }
        return output;
    }
    
    private float[] concatenateHeads(float[][] heads, int seqLen) {
        float[] result = new float[seqLen * dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int h = 0; h < nHeads; h++) {
                int offset = h * dHead;
                for (int i = 0; i < dHead; i++) {
                    result[t * dModel + offset + i] = heads[h][t * dHead + i];
                }
            }
        }
        return result;
    }
    
    public void update(float lr, float weightDecay) {
        applyGradients(wQ, wQGrad, lr, weightDecay);
        applyGradients(wK, wKGrad, lr, weightDecay);
        applyGradients(wV, wVGrad, lr, weightDecay);
        applyGradients(wO, wOGrad, lr, weightDecay);
        
        applyBiasGradients(bQ, bQGrad, lr);
        applyBiasGradients(bK, bKGrad, lr);
        applyBiasGradients(bV, bVGrad, lr);
        applyBiasGradients(bO, bOGrad, lr);
        
        Arrays.fill(wQGrad, 0);
        Arrays.fill(wKGrad, 0);
        Arrays.fill(wVGrad, 0);
        Arrays.fill(wOGrad, 0);
        Arrays.fill(bQGrad, 0);
        Arrays.fill(bKGrad, 0);
        Arrays.fill(bVGrad, 0);
        Arrays.fill(bOGrad, 0);
    }

    /**
     * 使用Adam优化器更新所有参数（wQ/wK/wV/wO/bQ/bK/bV/bO）。
     *
     * @param optimizer Adam优化器实例
     */
    public void updateWithAdam(com.qingxiaotuan.simpcore.training.AdamOptimizer optimizer) {
        optimizer.update(wQ, wQGrad);
        optimizer.update(wK, wKGrad);
        optimizer.update(wV, wVGrad);
        optimizer.update(wO, wOGrad);
        optimizer.update(bQ, bQGrad);
        optimizer.update(bK, bKGrad);
        optimizer.update(bV, bVGrad);
        optimizer.update(bO, bOGrad);

        Arrays.fill(wQGrad, 0);
        Arrays.fill(wKGrad, 0);
        Arrays.fill(wVGrad, 0);
        Arrays.fill(wOGrad, 0);
        Arrays.fill(bQGrad, 0);
        Arrays.fill(bKGrad, 0);
        Arrays.fill(bVGrad, 0);
        Arrays.fill(bOGrad, 0);
    }
    
    private void applyGradients(float[] weights, float[] grads, float lr, float weightDecay) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= lr * (grads[i] + weightDecay * weights[i]);
        }
    }
    
    private void applyBiasGradients(float[] bias, float[] grads, float lr) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] -= lr * grads[i];
        }
    }
    
    public long getParameterCount() {
        return (long) dModel * dModel * 4 + dModel * 4;
    }
    
    public int getDModel() { return dModel; }
    public int getNHeads() { return nHeads; }
    public int getDHead() { return dHead; }
    
    public float[] getWeights() {
        float[] allWeights = new float[wQ.length * 4];
        System.arraycopy(wQ, 0, allWeights, 0, wQ.length);
        System.arraycopy(wK, 0, allWeights, wQ.length, wK.length);
        System.arraycopy(wV, 0, allWeights, wQ.length * 2, wV.length);
        System.arraycopy(wO, 0, allWeights, wQ.length * 3, wO.length);
        return allWeights;
    }
    
    public void save(java.io.DataOutputStream dos) throws Exception {
        for (float v : wQ) dos.writeFloat(v);
        for (float v : wK) dos.writeFloat(v);
        for (float v : wV) dos.writeFloat(v);
        for (float v : wO) dos.writeFloat(v);
        for (float v : bQ) dos.writeFloat(v);
        for (float v : bK) dos.writeFloat(v);
        for (float v : bV) dos.writeFloat(v);
        for (float v : bO) dos.writeFloat(v);
    }
    
    public void load(java.io.DataInputStream dis) throws Exception {
        for (int i = 0; i < wQ.length; i++) wQ[i] = dis.readFloat();
        for (int i = 0; i < wK.length; i++) wK[i] = dis.readFloat();
        for (int i = 0; i < wV.length; i++) wV[i] = dis.readFloat();
        for (int i = 0; i < wO.length; i++) wO[i] = dis.readFloat();
        for (int i = 0; i < bQ.length; i++) bQ[i] = dis.readFloat();
        for (int i = 0; i < bK.length; i++) bK[i] = dis.readFloat();
        for (int i = 0; i < bV.length; i++) bV[i] = dis.readFloat();
        for (int i = 0; i < bO.length; i++) bO[i] = dis.readFloat();
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Multi-Head Attention Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        int dModel = 64;
        int nHeads = 8;
        int seqLen = 10;
        
        System.out.println("[1/3] Creating attention layer...");
        MultiHeadAttention attention = new MultiHeadAttention(dModel, nHeads);
        System.out.printf("  dModel: %d, nHeads: %d, dHead: %d%n", dModel, nHeads, dModel / nHeads);
        System.out.printf("  Parameters: %.2fM%n", attention.getParameterCount() / 1_000_000.0);
        
        System.out.println("\n[2/3] Forward pass...");
        float[] input = new float[seqLen * dModel];
        Random r = new Random(42);
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) r.nextGaussian() * 0.1f;
        }
        
        AttentionResult result = attention.forward(input, seqLen);
        System.out.printf("  Input shape: (%d, %d)%n", seqLen, dModel);
        System.out.printf("  Output shape: (%d, %d)%n", seqLen, dModel);
        System.out.printf("  Output norm: %.4f%n", computeNorm(result.output));
        
        System.out.println("\n[3/3] Backward pass...");
        float[] gradOutput = new float[seqLen * dModel];
        for (int i = 0; i < gradOutput.length; i++) {
            gradOutput[i] = (float) r.nextGaussian() * 0.1f;
        }
        
        float[] gradInput = attention.backward(gradOutput, result);
        System.out.printf("  Gradient norm: %.4f%n", computeNorm(gradInput));
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Multi-Head Attention Test Passed!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
    
    private static float computeNorm(float[] arr) {
        float sum = 0;
        for (float v : arr) sum += v * v;
        return (float) Math.sqrt(sum);
    }
}