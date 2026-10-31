package com.qingxiaotuan.simpcore.nn;

import java.util.*;

public class Transformer {
    
    public static class TransformerConfig {
        public int dModel = 64;
        public int nLayers = 2;
        public int nHeads = 8;
        public int vocabSize = 256;
        public int seqLen = 32;
        public float dropout = 0.1f;
        
        public TransformerConfig dModel(int val) { dModel = val; return this; }
        public TransformerConfig nLayers(int val) { nLayers = val; return this; }
        public TransformerConfig nHeads(int val) { nHeads = val; return this; }
        public TransformerConfig vocabSize(int val) { vocabSize = val; return this; }
        public TransformerConfig seqLen(int val) { seqLen = val; return this; }
        public TransformerConfig dropout(float val) { dropout = val; return this; }
    }
    
    private final TransformerConfig config;
    private final float[] embeddings;
    private final float[] embGrad;
    private final MultiHeadAttention[] attentions;
    private final LinearLayer[] ffnLayers1;
    private final LinearLayer[] ffnLayers2;
    private final float[] lnGamma;
    private final float[] lnBeta;
    private final float[] lnGammaGrad;
    private final float[] lnBetaGrad;
    private final LinearLayer outputLayer;
    private final float[] posEncoding;
    private final Dropout[] attnDropouts;
    private final Dropout[] ffnDropouts;
    private boolean trainingMode = true;

    public Transformer(TransformerConfig config) {
        this.config = config;
        
        this.embeddings = new float[config.vocabSize * config.dModel];
        this.embGrad = new float[config.vocabSize * config.dModel];
        initEmbeddings();
        
        this.attentions = new MultiHeadAttention[config.nLayers];
        this.ffnLayers1 = new LinearLayer[config.nLayers];
        this.ffnLayers2 = new LinearLayer[config.nLayers];
        
        for (int i = 0; i < config.nLayers; i++) {
            attentions[i] = new MultiHeadAttention(config.dModel, config.nHeads);
            ffnLayers1[i] = new LinearLayer(config.dModel, config.dModel * 4);
            ffnLayers2[i] = new LinearLayer(config.dModel * 4, config.dModel);
        }
        
        this.lnGamma = new float[config.dModel];
        this.lnBeta = new float[config.dModel];
        this.lnGammaGrad = new float[config.dModel];
        this.lnBetaGrad = new float[config.dModel];
        Arrays.fill(lnGamma, 1.0f);
        Arrays.fill(lnBeta, 0.0f);
        
        this.outputLayer = new LinearLayer(config.dModel, config.vocabSize);

        this.posEncoding = new float[config.seqLen * config.dModel];
        initPosEncoding();

        // 初始化Dropout层
        this.attnDropouts = new Dropout[config.nLayers];
        this.ffnDropouts = new Dropout[config.nLayers];
        for (int i = 0; i < config.nLayers; i++) {
            attnDropouts[i] = new Dropout(config.dropout, 42 + i);
            ffnDropouts[i] = new Dropout(config.dropout, 100 + i);
        }
    }

    /**
     * 设置训练/推理模式。
     *
     * @param training true=训练模式，false=推理模式
     */
    public void setTraining(boolean training) {
        this.trainingMode = training;
        for (int i = 0; i < config.nLayers; i++) {
            attnDropouts[i].setTraining(training);
            ffnDropouts[i].setTraining(training);
        }
    }
    
    private void initPosEncoding() {
        for (int pos = 0; pos < config.seqLen; pos++) {
            for (int i = 0; i < config.dModel; i += 2) {
                float angle = pos / (float) Math.pow(10000, (2 * i) / (float) config.dModel);
                posEncoding[pos * config.dModel + i] = (float) Math.sin(angle);
                if (i + 1 < config.dModel) {
                    posEncoding[pos * config.dModel + i + 1] = (float) Math.cos(angle);
                }
            }
        }
    }
    
    private void initEmbeddings() {
        Random r = new Random(42);
        float scale = (float) Math.sqrt(1.0 / config.dModel);
        for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] = (float) (r.nextDouble() * 2 - 1) * scale;
        }
    }
    
    public float[] forward(int[] tokens) {
        int seqLen = Math.min(tokens.length, config.seqLen);
        float[] hidden = new float[seqLen * config.dModel];
        
        for (int t = 0; t < seqLen; t++) {
            int token = tokens[t];
            for (int i = 0; i < config.dModel; i++) {
                hidden[t * config.dModel + i] = embeddings[token * config.dModel + i] + posEncoding[t * config.dModel + i];
            }
        }
        
        for (int l = 0; l < config.nLayers; l++) {
            float[] attnInput = layerNorm(hidden, seqLen);
            MultiHeadAttention.AttentionResult attnResult = attentions[l].forward(attnInput, seqLen);
            float[] attnOutput = attnResult.output;
            // 应用Attention Dropout
            attnOutput = attnDropouts[l].forward(attnOutput);

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i] + attnOutput[i];
            }

            float[] ffnInput = layerNorm(hidden, seqLen);
            float[] ffnHidden = relu(ffnLayers1[l].forward(ffnInput));
            float[] ffnOutput = ffnLayers2[l].forward(ffnHidden);
            // 应用FFN Dropout
            ffnOutput = ffnDropouts[l].forward(ffnOutput);

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i] + ffnOutput[i];
            }
        }
        
        float[] finalNorm = layerNorm(hidden, seqLen);
        float[] logits = outputLayer.forward(finalNorm);
        
        return logits;
    }
    
    private float[] layerNorm(float[] input, int seqLen) {
        float[] output = new float[seqLen * config.dModel];
        float eps = 1e-5f;
        
        for (int t = 0; t < seqLen; t++) {
            float mean = 0;
            for (int i = 0; i < config.dModel; i++) {
                mean += input[t * config.dModel + i];
            }
            mean /= config.dModel;
            
            float var = 0;
            for (int i = 0; i < config.dModel; i++) {
                float diff = input[t * config.dModel + i] - mean;
                var += diff * diff;
            }
            var /= config.dModel;
            
            if (var < eps) {
                var = eps;
            }
            
            float sqrtVar = (float) Math.sqrt(var + eps);
            
            for (int i = 0; i < config.dModel; i++) {
                float x = input[t * config.dModel + i];
                output[t * config.dModel + i] = lnGamma[i] * (x - mean) / sqrtVar + lnBeta[i];
            }
        }
        
        return output;
    }
    
    private float[] relu(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
    
    private float[] reluBackward(float[] gradOutput, float[] input) {
        float[] gradInput = new float[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            int idx = Math.min(i, input.length - 1);
            gradInput[i] = input[idx] > 0 ? gradOutput[i] : 0;
        }
        return gradInput;
    }
    
    private float[] layerNormBackward(float[] gradOutput, float[] input, int seqLen) {
        int actualSeqLen = gradOutput.length / config.dModel;
        float[] gradInput = new float[actualSeqLen * config.dModel];
        float eps = 1e-5f;
        
        for (int t = 0; t < actualSeqLen; t++) {
            float mean = 0;
            int inputIdx = Math.min(t, seqLen - 1);
            for (int i = 0; i < config.dModel; i++) {
                mean += input[inputIdx * config.dModel + i];
            }
            mean /= config.dModel;
            
            float var = 0;
            for (int i = 0; i < config.dModel; i++) {
                var += (input[inputIdx * config.dModel + i] - mean) * (input[inputIdx * config.dModel + i] - mean);
            }
            var /= config.dModel;
            
            float sqrtVar = (float) Math.sqrt(var + eps);
            
            for (int i = 0; i < config.dModel; i++) {
                float x = input[inputIdx * config.dModel + i];
                float dx = (x - mean) / sqrtVar;
                
                lnGammaGrad[i] += gradOutput[t * config.dModel + i] * dx;
                lnBetaGrad[i] += gradOutput[t * config.dModel + i];
                
                float dMean = -gradOutput[t * config.dModel + i] * lnGamma[i] / sqrtVar;
                float dVar = gradOutput[t * config.dModel + i] * lnGamma[i] * dx * (-0.5f) / (var + eps);
                
                gradInput[t * config.dModel + i] += gradOutput[t * config.dModel + i] * lnGamma[i] / sqrtVar;
                gradInput[t * config.dModel + i] += dMean / config.dModel;
                gradInput[t * config.dModel + i] += dVar * 2 * (x - mean) / config.dModel;
            }
        }
        
        return gradInput;
    }
    
    public float trainStep(int[] inputTokens, int[] targetTokens, float lr, float weightDecay) {
        int seqLen = Math.min(inputTokens.length, config.seqLen);
        
        float[] hidden = new float[seqLen * config.dModel];
        float[][] layerInputs = new float[config.nLayers][];
        float[][] layerHidden = new float[config.nLayers][];
        MultiHeadAttention.AttentionResult[] attnResults = new MultiHeadAttention.AttentionResult[config.nLayers];
        
        for (int t = 0; t < seqLen; t++) {
            int token = inputTokens[t];
            for (int i = 0; i < config.dModel; i++) {
                hidden[t * config.dModel + i] = embeddings[token * config.dModel + i];
            }
        }
        
        for (int l = 0; l < config.nLayers; l++) {
            layerInputs[l] = hidden.clone();
            float[] attnInput = layerNorm(hidden, seqLen);
            attnResults[l] = attentions[l].forward(attnInput, seqLen);
            // 训练时应用Attention Dropout
            float[] attnOut = trainingMode ? attnDropouts[l].forward(attnResults[l].output) : attnResults[l].output;

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i] + attnOut[i];
            }

            layerHidden[l] = hidden.clone();
            float[] ffnInput = layerNorm(hidden, seqLen);
            float[] ffnHidden = relu(ffnLayers1[l].forward(ffnInput));
            float[] ffnOutput = ffnLayers2[l].forward(ffnHidden);
            // 训练时应用FFN Dropout
            float[] ffnOut = trainingMode ? ffnDropouts[l].forward(ffnOutput) : ffnOutput;

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i] + ffnOut[i];
            }
        }
        
        if (hasNaN(hidden)) {
            System.out.println("ERROR: NaN in hidden");
            return Float.NaN;
        }
        
        float[] finalNorm = layerNorm(hidden, seqLen);
        float[] logits = outputLayer.forward(finalNorm);
        
        if (hasNaN(logits)) {
            System.out.println("ERROR: NaN in logits");
            return Float.NaN;
        }
        
        float loss = 0;
        int count = 0;
        float[] gradLogits = new float[seqLen * config.vocabSize];
        Arrays.fill(gradLogits, 0);
        
        for (int t = 0; t < seqLen; t++) {
            if (t >= targetTokens.length) break;
            
            int target = targetTokens[t];
            
            float maxLogit = logits[t * config.vocabSize];
            for (int i = 1; i < config.vocabSize; i++) {
                if (logits[t * config.vocabSize + i] > maxLogit) {
                    maxLogit = logits[t * config.vocabSize + i];
                }
            }
            
            float[] softmax = new float[config.vocabSize];
            float sum = 0;
            for (int i = 0; i < config.vocabSize; i++) {
                softmax[i] = (float) Math.exp(logits[t * config.vocabSize + i] - maxLogit);
                sum += softmax[i];
            }
            for (int i = 0; i < config.vocabSize; i++) {
                softmax[i] /= sum;
            }
            
            loss += (float) -Math.log(Math.max(1e-10f, softmax[target]));
            count++;
            
            for (int i = 0; i < config.vocabSize; i++) {
                gradLogits[t * config.vocabSize + i] = softmax[i];
            }
            gradLogits[t * config.vocabSize + target] -= 1.0f;
        }
        
        loss /= count;
        
        if (hasNaN(gradLogits)) {
            System.out.println("ERROR: NaN in gradLogits");
            return Float.NaN;
        }
        
        float[] gradFinalNorm = outputLayer.backward(gradLogits);
        clipGradient(gradFinalNorm, 1.0f);
        if (hasNaN(gradFinalNorm)) {
            System.out.println("ERROR: NaN in gradFinalNorm");
            return Float.NaN;
        }
        float[] gradHidden = layerNormBackward(gradFinalNorm, hidden, seqLen);
        clipGradient(gradHidden, 1.0f);
        if (hasNaN(gradHidden)) {
            System.out.println("ERROR: NaN in gradHidden after first layerNormBackward");
            return Float.NaN;
        }
        
        for (int l = config.nLayers - 1; l >= 0; l--) {
            float[] gradFFN = gradHidden.clone();
            // FFN Dropout反向传播
            if (trainingMode) {
                gradFFN = ffnDropouts[l].backward(gradFFN);
                clipGradient(gradFFN, 1.0f);
            }
            float[] gradFFNOutput = ffnLayers2[l].backward(gradFFN);
            clipGradient(gradFFNOutput, 1.0f);
            float[] gradFFNHidden = reluBackward(gradFFNOutput, ffnLayers1[l].lastInput);
            clipGradient(gradFFNHidden, 1.0f);
            float[] gradFFNInput = ffnLayers1[l].backward(gradFFNHidden);
            clipGradient(gradFFNInput, 1.0f);

            float[] gradAfterFFN = layerNormBackward(gradFFNInput, layerHidden[l], seqLen);
            clipGradient(gradAfterFFN, 1.0f);

            float[] gradAttn = gradAfterFFN.clone();
            for (int i = 0; i < gradAttn.length; i++) {
                gradAttn[i] += gradHidden[i];
            }
            // Attention Dropout反向传播
            if (trainingMode) {
                gradAttn = attnDropouts[l].backward(gradAttn);
                clipGradient(gradAttn, 1.0f);
            }

            float[] gradAttnOutput = attentions[l].backward(gradAttn, attnResults[l]);
            clipGradient(gradAttnOutput, 1.0f);
            gradHidden = layerNormBackward(gradAttnOutput, layerInputs[l], seqLen);
            clipGradient(gradHidden, 1.0f);
        }
        
        if (hasNaN(gradHidden)) {
            System.out.println("ERROR: NaN in gradHidden");
            System.out.println("  gradHidden length: " + gradHidden.length);
            for (int i = 0; i < Math.min(10, gradHidden.length); i++) {
                System.out.println("  gradHidden[" + i + "]: " + gradHidden[i]);
            }
            return Float.NaN;
        }
        
        for (int t = 0; t < seqLen; t++) {
            int token = inputTokens[t];
            for (int i = 0; i < config.dModel; i++) {
                embGrad[token * config.dModel + i] += gradHidden[t * config.dModel + i];
            }
        }
        
        if (hasNaN(embGrad)) {
            System.out.println("ERROR: NaN in embGrad before clip");
            return Float.NaN;
        }
        clipGradients(1.0f);
        if (hasNaN(embGrad)) {
            System.out.println("ERROR: NaN in embGrad after clip");
            return Float.NaN;
        }
        updateParameters(lr, weightDecay);
        
        if (hasNaN(embeddings)) {
            System.out.println("ERROR: NaN in embeddings after update");
            return Float.NaN;
        }
        
        float maxEmb = 0;
        for (float v : embeddings) {
            maxEmb = Math.max(maxEmb, Math.abs(v));
        }
        if (maxEmb > 100) {
            System.out.println("WARNING: Embeddings have large values: " + maxEmb);
        }
        
        for (int l = 0; l < config.nLayers; l++) {
            if (hasNaN(attentions[l].getWeights())) {
                System.out.println("ERROR: NaN in attention weights after update");
                return Float.NaN;
            }
        }
        
        return loss;
    }
    
    private boolean hasNaN(float[] arr) {
        for (float v : arr) {
            if (Float.isNaN(v) || Float.isInfinite(v)) return true;
        }
        return false;
    }
    
    private void clipGradients(float maxNorm) {
        float norm = 0;
        for (float g : embGrad) norm += g * g;
        norm = (float) Math.sqrt(norm);
        
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            for (int i = 0; i < embGrad.length; i++) {
                embGrad[i] *= scale;
            }
        }
    }
    
    private void clipGradient(float[] grad, float maxNorm) {
        float norm = 0;
        for (float g : grad) norm += g * g;
        norm = (float) Math.sqrt(norm);
        
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            for (int i = 0; i < grad.length; i++) {
                grad[i] *= scale;
            }
        }
    }
    
    private void updateParameters(float lr, float weightDecay) {
        // 当lr=0且weightDecay=0时，跳过更新和梯度清零，保留梯度供Adam使用
        if (lr == 0.0f && weightDecay == 0.0f) {
            return;
        }
        for (int i = 0; i < embeddings.length; i++) {
            embeddings[i] -= lr * (embGrad[i] + weightDecay * embeddings[i]);
        }
        Arrays.fill(embGrad, 0);

        for (int l = 0; l < config.nLayers; l++) {
            attentions[l].update(lr, weightDecay);
            ffnLayers1[l].update(lr, weightDecay);
            ffnLayers2[l].update(lr, weightDecay);
        }

        for (int i = 0; i < config.dModel; i++) {
            lnGamma[i] -= lr * (lnGammaGrad[i] + weightDecay * lnGamma[i]);
            lnBeta[i] -= lr * lnBetaGrad[i];
        }
        Arrays.fill(lnGammaGrad, 0);
        Arrays.fill(lnBetaGrad, 0);

        outputLayer.update(lr, weightDecay);
    }

    /**
     * 使用Adam优化器更新参数。
     *
     * @param optimizer Adam优化器实例
     */
    public void updateParametersWithAdam(com.qingxiaotuan.simpcore.training.AdamOptimizer optimizer) {
        // 更新embeddings
        optimizer.update(embeddings, embGrad);
        Arrays.fill(embGrad, 0);

        // 更新每层参数
        for (int l = 0; l < config.nLayers; l++) {
            attentions[l].updateWithAdam(optimizer);
            ffnLayers1[l].updateWithAdam(optimizer);
            ffnLayers2[l].updateWithAdam(optimizer);
        }

        // 更新LayerNorm参数
        optimizer.update(lnGamma, lnGammaGrad);
        optimizer.update(lnBeta, lnBetaGrad);
        Arrays.fill(lnGammaGrad, 0);
        Arrays.fill(lnBetaGrad, 0);

        // 更新输出层
        outputLayer.updateWithAdam(optimizer);
    }

    /**
     * 使用Adam优化器的trainStep。
     *
     * @param inputTokens 输入tokens
     * @param targetTokens 目标tokens
     * @param optimizer Adam优化器
     * @return 训练损失
     */
    public float trainStepWithAdam(int[] inputTokens, int[] targetTokens,
                                    com.qingxiaotuan.simpcore.training.AdamOptimizer optimizer) {
        float loss = trainStepForwardBackward(inputTokens, targetTokens);
        if (Float.isNaN(loss)) return loss;

        // 使用Adam更新参数
        updateParametersWithAdam(optimizer);

        if (hasNaN(embeddings)) {
            System.out.println("ERROR: NaN in embeddings after Adam update");
            return Float.NaN;
        }

        return loss;
    }

    /**
     * 仅执行前向和反向传播，不更新参数。
     * 用于分离梯度计算和参数更新。
     *
     * @param inputTokens 输入tokens
     * @param targetTokens 目标tokens
     * @return 训练损失
     */
    public float trainStepForwardBackward(int[] inputTokens, int[] targetTokens) {
        return trainStep(inputTokens, targetTokens, 0.0f, 0.0f);
    }
    
    public int predictNext(int[] tokens) {
        float[] logits = forward(tokens);
        int lastPos = Math.min(tokens.length - 1, config.seqLen - 1);
        
        int maxIdx = 0;
        float maxVal = logits[lastPos * config.vocabSize];
        for (int i = 1; i < config.vocabSize; i++) {
            if (logits[lastPos * config.vocabSize + i] > maxVal) {
                maxVal = logits[lastPos * config.vocabSize + i];
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }
    
    public int getVocabSize() {
        return config.vocabSize;
    }
    
    public long getParameterCount() {
        long count = (long) config.vocabSize * config.dModel;
        
        for (int l = 0; l < config.nLayers; l++) {
            count += attentions[l].getParameterCount();
            count += ffnLayers1[l].getParameterCount();
            count += ffnLayers2[l].getParameterCount();
        }
        
        count += outputLayer.getParameterCount();
        
        return count;
    }
    
    public TransformerConfig getConfig() { return config; }
    
    public void save(String filePath) throws Exception {
        try (java.io.DataOutputStream dos = new java.io.DataOutputStream(
                new java.io.FileOutputStream(filePath))) {
            dos.writeInt(config.dModel);
            dos.writeInt(config.nLayers);
            dos.writeInt(config.nHeads);
            dos.writeInt(config.vocabSize);
            dos.writeInt(config.seqLen);
            
            for (float v : embeddings) dos.writeFloat(v);
            for (float v : lnGamma) dos.writeFloat(v);
            for (float v : lnBeta) dos.writeFloat(v);
            
            for (int l = 0; l < config.nLayers; l++) {
                attentions[l].save(dos);
                ffnLayers1[l].save(dos);
                ffnLayers2[l].save(dos);
            }
            
            outputLayer.save(dos);
        }
    }
    
    public static Transformer load(String filePath) throws Exception {
        try (java.io.DataInputStream dis = new java.io.DataInputStream(
                new java.io.FileInputStream(filePath))) {
            int dModel = dis.readInt();
            int nLayers = dis.readInt();
            int nHeads = dis.readInt();
            int vocabSize = dis.readInt();
            int seqLen = dis.readInt();
            
            TransformerConfig config = new TransformerConfig()
                .dModel(dModel)
                .nLayers(nLayers)
                .nHeads(nHeads)
                .vocabSize(vocabSize)
                .seqLen(seqLen);
            
            Transformer model = new Transformer(config);
            
            for (int i = 0; i < model.embeddings.length; i++) {
                model.embeddings[i] = dis.readFloat();
            }
            for (int i = 0; i < model.lnGamma.length; i++) {
                model.lnGamma[i] = dis.readFloat();
            }
            for (int i = 0; i < model.lnBeta.length; i++) {
                model.lnBeta[i] = dis.readFloat();
            }
            
            for (int l = 0; l < config.nLayers; l++) {
                model.attentions[l].load(dis);
                model.ffnLayers1[l].load(dis);
                model.ffnLayers2[l].load(dis);
            }
            
            model.outputLayer.load(dis);
            
            return model;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Transformer Model Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        TransformerConfig config = new TransformerConfig()
            .dModel(64)
            .nLayers(2)
            .nHeads(8)
            .vocabSize(128)
            .seqLen(32);
        
        System.out.println("[1/3] Creating Transformer...");
        Transformer model = new Transformer(config);
        System.out.printf("  Architecture: dModel=%d, nLayers=%d, nHeads=%d%n", 
            config.dModel, config.nLayers, config.nHeads);
        System.out.printf("  Parameters: %.2fM%n", model.getParameterCount() / 1_000_000.0);
        
        System.out.println("\n[2/3] Forward pass test...");
        int[] tokens = new int[10];
        Random r = new Random(42);
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = r.nextInt(config.vocabSize);
        }
        
        float[] logits = model.forward(tokens);
        System.out.printf("  Input tokens: %d%n", tokens.length);
        System.out.printf("  Output logits shape: (%d, %d)%n", tokens.length, config.vocabSize);
        System.out.printf("  Logits norm: %.4f%n", computeNorm(logits));
        
        System.out.println("\n[3/3] Prediction test...");
        int nextToken = model.predictNext(tokens);
        System.out.printf("  Predicted next token: %d%n", nextToken);
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Transformer Test Passed!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
    
    private static float computeNorm(float[] arr) {
        float sum = 0;
        for (float v : arr) sum += v * v;
        return (float) Math.sqrt(sum);
    }
}