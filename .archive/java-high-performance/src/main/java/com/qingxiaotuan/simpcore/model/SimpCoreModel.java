package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.cognition.*;
import com.qingxiaotuan.simpcore.components.*;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import com.qingxiaotuan.simpcore.token.ZYLAttentionSystem;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

public class SimpCoreModel {

    private final int vocabSize;
    private final int dModel;
    private final int nLayers;
    private final int maxSeqLen;
    private float[] embeddingWeights;
    private float[] posEncoding;
    private MultiHeadAttention[] mhaLayers;
    private ZYLAttentionSystem[] zylLayers;
    private boolean useZYLAttention = false;
    private AdaptiveBasisLayer[] ablLayers;
    private DynamicMemoryChannel[] dmcLayers;
    private TopologyAwareRouter[] tarLayers;
    private CausalInterventionNetwork[] cinLayers;
    private InformationBottleneck[] ibLayers;
    private MetaRepresentationDistiller[] mrdLayers;
    private PlasticityUpdateModule[] pumLayers;
    private float[][] ffnWeights1;
    private float[][] ffnWeights2;
    private float[] outputWeights;
    private float[][] layerNormWeight;
    private float[][] layerNormBias;
    private final Random random;
    private boolean inferenceMode;
    
    private float[][] dFFNWeights1;
    private float[][] dFFNWeights2;
    private float[] dOutputWeights;
    
    // 认知架构组件
    private boolean enableCognition;
    private HierarchicalPerceptionModule hierarchicalPerception;
    private WorkingMemoryController workingMemory;
    private CognitiveReasoner cognitiveReasoner;
    private SchemaLearningModule schemaLearning;

    public SimpCoreModel(int vocabSize, int dModel, int nLayers, int nBases, int nHeads,
                         int nChannels, int nPaths, int topK, int maxSeqLen) {
        this(vocabSize, dModel, nLayers, nBases, nHeads, nChannels, nPaths, topK, maxSeqLen, false);
    }

    public SimpCoreModel(int vocabSize, int dModel, int nLayers, int nBases, int nHeads,
                         int nChannels, int nPaths, int topK, int maxSeqLen, boolean enableCognition) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.nLayers = nLayers;
        this.maxSeqLen = maxSeqLen;
        this.enableCognition = enableCognition;
        this.random = new Random(42);
        initModel(nBases, nHeads, nChannels, nPaths, topK);
        if (enableCognition) {
            initCognitionComponents();
        }
    }

    public SimpCoreModel(int dModel, int nLayers, int nHeads, int vocabSize, int maxSeqLen) {
        this(vocabSize, dModel, nLayers, nHeads, nHeads, 4, 8, 2, maxSeqLen);
    }

    private void initModel(int nBases, int nHeads, int nChannels, int nPaths, int topK) {
        embeddingWeights = new float[vocabSize * dModel];
        for (int i = 0; i < vocabSize * dModel; i++) {
            embeddingWeights[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }

        posEncoding = new float[maxSeqLen * dModel];
        initPosEncoding();

        mhaLayers = new MultiHeadAttention[nLayers];
        ablLayers = new AdaptiveBasisLayer[nLayers];
        dmcLayers = new DynamicMemoryChannel[nLayers];
        tarLayers = new TopologyAwareRouter[nLayers];
        cinLayers = new CausalInterventionNetwork[nLayers];
        ibLayers = new InformationBottleneck[nLayers];
        mrdLayers = new MetaRepresentationDistiller[nLayers];
        pumLayers = new PlasticityUpdateModule[nLayers];

        for (int l = 0; l < nLayers; l++) {
            mhaLayers[l] = new MultiHeadAttention(dModel, nHeads);
            ablLayers[l] = new AdaptiveBasisLayer(dModel, nBases, nHeads);
            dmcLayers[l] = new DynamicMemoryChannel(dModel, nChannels, dModel);
            tarLayers[l] = new TopologyAwareRouter(dModel, nPaths, topK);
            cinLayers[l] = new CausalInterventionNetwork(dModel);
            ibLayers[l] = new InformationBottleneck(dModel, 0.8f);
            mrdLayers[l] = new MetaRepresentationDistiller(dModel, dModel / 4);
            pumLayers[l] = new PlasticityUpdateModule(dModel);
        }

        ffnWeights1 = new float[nLayers][dModel * dModel * 4];
        ffnWeights2 = new float[nLayers][dModel * 4 * dModel];
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < dModel * dModel * 4; i++) {
                ffnWeights1[l][i] = (random.nextFloat() - 0.5f) * 0.05f;
            }
            for (int i = 0; i < dModel * 4 * dModel; i++) {
                ffnWeights2[l][i] = (random.nextFloat() - 0.5f) * 0.05f;
            }
        }

        outputWeights = new float[dModel * vocabSize];
        for (int i = 0; i < dModel * vocabSize; i++) {
            outputWeights[i] = (random.nextFloat() - 0.5f) * 0.05f;
        }

        layerNormWeight = new float[nLayers][dModel];
        layerNormBias = new float[nLayers][dModel];
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < dModel; i++) {
                layerNormWeight[l][i] = 1.0f;
                layerNormBias[l][i] = 0.0f;
            }
        }

        this.inferenceMode = false;
        
        dFFNWeights1 = new float[nLayers][dModel * dModel * 4];
        dFFNWeights2 = new float[nLayers][dModel * 4 * dModel];
        dOutputWeights = new float[dModel * vocabSize];
    }
    
    private void initCognitionComponents() {
        hierarchicalPerception = new HierarchicalPerceptionModule(dModel);
        workingMemory = new WorkingMemoryController(dModel);
        cognitiveReasoner = new CognitiveReasoner(dModel);
        schemaLearning = new SchemaLearningModule(dModel);
    }

    private void initPosEncoding() {
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dModel; i += 2) {
                posEncoding[pos * dModel + i] = (float) Math.sin(pos / Math.pow(10000, i / (double) dModel));
                posEncoding[pos * dModel + i + 1] = (float) Math.cos(pos / Math.pow(10000, i / (double) dModel));
            }
        }
    }

    public float[] forward(int[] inputIds) {
        return forward(inputIds, 1, inputIds.length);
    }
    
    public float[] forward(int[] inputIds, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] hidden = memoryOptimizer.allocateFloatArray(totalSize);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = inputIds[b * seqLen + s];
                int offset = (b * seqLen + s) * dModel;
                for (int i = 0; i < dModel; i++) {
                    int posIdx = Math.min(s, maxSeqLen - 1);
                    hidden[offset + i] = embeddingWeights[tokenId * dModel + i] + posEncoding[posIdx * dModel + i];
                }
            }
        }

        // 认知增强：通过分层感知模块处理嵌入
        if (enableCognition && hierarchicalPerception != null) {
            float[] perceived = hierarchicalPerception.forward(hidden, batchSize, seqLen);
            System.arraycopy(perceived, 0, hidden, 0, totalSize);
            memoryOptimizer.releaseFloatArray(perceived);
        }

        for (int l = 0; l < nLayers; l++) {
            float[] mhaOutput;
            if (useZYLAttention && zylLayers != null) {
                mhaOutput = zylLayers[l].forward(hidden, batchSize, seqLen).output;
            } else {
                mhaOutput = mhaLayers[l].forward(hidden, batchSize, seqLen);
            }
            
            float[] attentionResidual = new float[batchSize * seqLen * dModel];
            for (int i = 0; i < batchSize * seqLen * dModel; i++) {
                attentionResidual[i] = hidden[i] + mhaOutput[i];
            }
            
            float[] mhaNorm = new float[batchSize * seqLen * dModel];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = attentionResidual[offset + j];
                    }
                    float[] normalized = layerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        mhaNorm[offset + j] = normalized[j];
                    }
                }
            }
            
            float[] rfpOutput = ablLayers[l].forward(mhaNorm, batchSize, seqLen);
            float[] dmcOutput = dmcLayers[l].forward(rfpOutput, batchSize, seqLen);
            float[] tarOutput = tarLayers[l].forward(dmcOutput, batchSize, seqLen);
            float[] cinOutput = cinLayers[l].forward(tarOutput, batchSize, seqLen);
            float[] ibOutput = ibLayers[l].forward(cinOutput, batchSize, seqLen);
            float[] ffnOutput = applyFFN(ibOutput, ffnWeights1[l], ffnWeights2[l], batchSize, seqLen);
            float[] mrdOutput = mrdLayers[l].forward(ffnOutput, batchSize, seqLen);
            float[] spmOutput = pumLayers[l].forward(mrdOutput, batchSize, seqLen);

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] layerInput = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        layerInput[j] = mhaNorm[offset + j] + spmOutput[offset + j];
                    }
                    float[] normalized = layerNorm(layerInput, l);
                    for (int j = 0; j < dModel; j++) {
                        hidden[offset + j] = normalized[j];
                    }
                }
            }
            
            memoryOptimizer.releaseFloatArray(mhaOutput);
            memoryOptimizer.releaseFloatArray(rfpOutput);
            memoryOptimizer.releaseFloatArray(dmcOutput);
            memoryOptimizer.releaseFloatArray(tarOutput);
            memoryOptimizer.releaseFloatArray(cinOutput);
            memoryOptimizer.releaseFloatArray(ibOutput);
            memoryOptimizer.releaseFloatArray(ffnOutput);
            memoryOptimizer.releaseFloatArray(mrdOutput);
            memoryOptimizer.releaseFloatArray(spmOutput);
        }

        // 认知增强：通过推理器和模式学习处理Transformer输出
        if (enableCognition && cognitiveReasoner != null && workingMemory != null) {
            // 执行多步推理（利用工作记忆）
            float[] reasoned = cognitiveReasoner.reason(hidden, cognitiveReasoner.getDefaultSteps(), workingMemory);
            System.arraycopy(reasoned, 0, hidden, 0, totalSize);
            
            // 工作记忆衰减和巩固
            workingMemory.decay();
            workingMemory.consolidate();
            
            // 模式学习增强
            if (schemaLearning != null) {
                float[] schemed = schemaLearning.forward(hidden);
                System.arraycopy(schemed, 0, hidden, 0, totalSize);
            }
        }

        float[] logits = MatrixOperations.matmul(hidden, outputWeights, batchSize * seqLen, dModel, vocabSize);
        memoryOptimizer.releaseFloatArray(hidden);
        return logits;
    }

    private float[] applyFFN(float[] x, float[] w1, float[] w2, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        
        float[] hidden = memoryOptimizer.allocateFloatArray(batchSize * seqLen * dModel * 4);

        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] rowData = memoryOptimizer.allocateFloatArray(dModel);
            System.arraycopy(x, i * dModel, rowData, 0, dModel);
            float[] result = MatrixOperations.matmul(rowData, w1, 1, dModel, dModel * 4);
            for (int j = 0; j < dModel * 4; j++) {
                hidden[i * dModel * 4 + j] = (float) (result[j] * (1.0 / (1.0 + Math.exp(-result[j]))));
            }
            memoryOptimizer.releaseFloatArray(rowData);
            memoryOptimizer.releaseFloatArray(result);
        }

        float[] output = memoryOptimizer.allocateFloatArray(totalSize);
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] rowData = memoryOptimizer.allocateFloatArray(dModel * 4);
            System.arraycopy(hidden, i * dModel * 4, rowData, 0, dModel * 4);
            float[] result = MatrixOperations.matmul(rowData, w2, 1, dModel * 4, dModel);
            System.arraycopy(result, 0, output, i * dModel, dModel);
            memoryOptimizer.releaseFloatArray(rowData);
            memoryOptimizer.releaseFloatArray(result);
        }
        
        memoryOptimizer.releaseFloatArray(hidden);
        return output;
    }

    private static final jdk.incubator.vector.VectorSpecies<Float> VECTOR_SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
    
    private float[] layerNorm(float[] x, int layer) {
        float mean = 0f;
        int i = 0;
        
        for (; i + VECTOR_SPECIES.length() <= dModel; i += VECTOR_SPECIES.length()) {
            jdk.incubator.vector.FloatVector vec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, x, i);
            mean += vec.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        }
        for (; i < dModel; i++) {
            mean += x[i];
        }
        mean /= dModel;
        
        float var = 0f;
        jdk.incubator.vector.FloatVector meanVec = jdk.incubator.vector.FloatVector.broadcast(VECTOR_SPECIES, mean);
        i = 0;
        
        for (; i + VECTOR_SPECIES.length() <= dModel; i += VECTOR_SPECIES.length()) {
            jdk.incubator.vector.FloatVector vec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, x, i);
            jdk.incubator.vector.FloatVector diff = vec.sub(meanVec);
            var += diff.mul(diff).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        }
        for (; i < dModel; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= dModel;
        
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] output = memoryOptimizer.allocateFloatArray(dModel);
        float invStd = 1.0f / (float) Math.sqrt(var + 1e-5f);
        jdk.incubator.vector.FloatVector invStdVec = jdk.incubator.vector.FloatVector.broadcast(VECTOR_SPECIES, invStd);
        
        i = 0;
        for (; i + VECTOR_SPECIES.length() <= dModel; i += VECTOR_SPECIES.length()) {
            jdk.incubator.vector.FloatVector xVec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, x, i);
            jdk.incubator.vector.FloatVector wVec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, layerNormWeight[layer], i);
            jdk.incubator.vector.FloatVector bVec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, layerNormBias[layer], i);
            
            jdk.incubator.vector.FloatVector normalized = xVec.sub(meanVec).mul(invStdVec);
            normalized.mul(wVec).add(bVec).intoArray(output, i);
        }
        for (; i < dModel; i++) {
            output[i] = (x[i] - mean) * invStd * layerNormWeight[layer][i] + layerNormBias[layer][i];
        }
        
        return output;
    }

    public int[] generate(int[] inputIds, int maxNewTokens, int eosTokenId) {
        int currentLen = inputIds.length;
        int[] outputIds = Arrays.copyOf(inputIds, currentLen + maxNewTokens);

        boolean prevInferenceMode = this.inferenceMode;
        this.inferenceMode = true;
        for (int l = 0; l < nLayers; l++) {
            pumLayers[l].setEnabled(false);
            dmcLayers[l].resetMemory();
        }

        try {
            for (int t = 0; t < maxNewTokens; t++) {
                int seqLen = Math.min(currentLen + t, maxSeqLen);
                int startIdx = currentLen + t - seqLen;
                int[] inputSlice = Arrays.copyOfRange(outputIds, startIdx, currentLen + t);

                float[] logits = forward(inputSlice, 1, seqLen);
                float[] lastLogits = Arrays.copyOfRange(logits, (seqLen - 1) * vocabSize, seqLen * vocabSize);

                float[] probs = MatrixOperations.softmax(lastLogits);

                int nextToken = sampleToken(probs);
                outputIds[currentLen + t] = nextToken;

                if (nextToken == eosTokenId) {
                    return Arrays.copyOf(outputIds, currentLen + t + 1);
                }
            }
        } finally {
            this.inferenceMode = prevInferenceMode;
            for (int l = 0; l < nLayers; l++) {
                pumLayers[l].setEnabled(!prevInferenceMode);
            }
        }

        return Arrays.copyOf(outputIds, currentLen + maxNewTokens);
    }

    private int sampleToken(float[] probs) {
        float r = random.nextFloat();
        float cumsum = 0;
        for (int i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (cumsum >= r) {
                return i;
            }
        }
        return probs.length - 1;
    }

    public void save(String filePath) {
        try {
            java.io.DataOutputStream dos = new java.io.DataOutputStream(
                new java.io.FileOutputStream(filePath));
            save(dos);
            dos.close();
        } catch (java.io.IOException e) {
            throw new RuntimeException("Save failed", e);
        }
    }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        dos.writeInt(vocabSize);
        dos.writeInt(dModel);
        dos.writeInt(nLayers);
        dos.writeInt(maxSeqLen);
        dos.writeInt(ablLayers[0].getNBases());
        dos.writeInt(ablLayers[0].getNHeads());
        dos.writeInt(dmcLayers[0].getNChannels());
        dos.writeInt(tarLayers[0].getNPaths());
        dos.writeInt(tarLayers[0].getTopK());

        for (float f : embeddingWeights) dos.writeFloat(f);
        for (float f : posEncoding) dos.writeFloat(f);

        for (int l = 0; l < nLayers; l++) {
            mhaLayers[l].save(dos);
            ablLayers[l].save(dos);
            dmcLayers[l].save(dos);
            tarLayers[l].save(dos);
            cinLayers[l].save(dos);
            ibLayers[l].save(dos);
            mrdLayers[l].save(dos);
            pumLayers[l].save(dos);
        }

        for (float[] w : ffnWeights1) for (float f : w) dos.writeFloat(f);
        for (float[] w : ffnWeights2) for (float f : w) dos.writeFloat(f);
        for (float f : outputWeights) dos.writeFloat(f);
        for (float[] w : layerNormWeight) for (float f : w) dos.writeFloat(f);
        for (float[] w : layerNormBias) for (float f : w) dos.writeFloat(f);
    }

    public static SimpCoreModel load(String filePath) {
        try {
            java.io.DataInputStream dis = new java.io.DataInputStream(
                new java.io.FileInputStream(filePath));
            SimpCoreModel model = loadFromStream(dis);
            dis.close();
            return model;
        } catch (java.io.IOException e) {
            throw new RuntimeException("Load failed", e);
        }
    }

    public static SimpCoreModel loadFromStream(java.io.DataInputStream dis) throws java.io.IOException {
        int vocabSize = dis.readInt();
        int dModel = dis.readInt();
        int nLayers = dis.readInt();
        int maxSeqLen = dis.readInt();
        int nBases = dis.readInt();
        int nHeads = dis.readInt();
        int nChannels = dis.readInt();
        int nPaths = dis.readInt();
        int topK = dis.readInt();

        SimpCoreModel model = new SimpCoreModel(vocabSize, dModel, nLayers, nBases, nHeads, nChannels, nPaths, topK, maxSeqLen);

        for (int i = 0; i < model.embeddingWeights.length; i++) {
            model.embeddingWeights[i] = dis.readFloat();
        }
        for (int i = 0; i < model.posEncoding.length; i++) {
            model.posEncoding[i] = dis.readFloat();
        }

        for (int l = 0; l < nLayers; l++) {
            model.mhaLayers[l].load(dis);
            model.ablLayers[l].load(dis);
            model.dmcLayers[l].load(dis);
            model.tarLayers[l].load(dis);
            model.cinLayers[l].load(dis);
            model.ibLayers[l].load(dis);
            model.mrdLayers[l].load(dis);
            model.pumLayers[l].load(dis);
        }

        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < model.ffnWeights1[l].length; i++) {
                model.ffnWeights1[l][i] = dis.readFloat();
            }
        }
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < model.ffnWeights2[l].length; i++) {
                model.ffnWeights2[l][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < model.outputWeights.length; i++) {
            model.outputWeights[i] = dis.readFloat();
        }
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < dModel; i++) {
                model.layerNormWeight[l][i] = dis.readFloat();
            }
        }
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < dModel; i++) {
                model.layerNormBias[l][i] = dis.readFloat();
            }
        }

        return model;
    }

    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNLayers() { return nLayers; }
    public int getMaxSeqLen() { return maxSeqLen; }

    public boolean isInferenceMode() { return inferenceMode; }

    public void setInferenceMode(boolean inferenceMode) {
        this.inferenceMode = inferenceMode;
        for (int l = 0; l < nLayers; l++) {
            pumLayers[l].setEnabled(!inferenceMode);
        }
    }

    public boolean isUseZYLAttention() { return useZYLAttention; }

    public void setUseZYLAttention(boolean useZYL) {
        this.useZYLAttention = useZYL;
        if (useZYL && zylLayers == null) {
            initializeZYLAttention();
        }
    }

    private void initializeZYLAttention() {
        zylLayers = new ZYLAttentionSystem[nLayers];
        for (int l = 0; l < nLayers; l++) {
            ZYLAttentionSystem.ZYLConfig config = new ZYLAttentionSystem.ZYLConfig();
            config.maxContextLength = maxSeqLen;
            config.nHeads = mhaLayers[0].getNHeads();
            config.dModel = dModel;
            config.headDim = mhaLayers[0].getHeadDim();
            config.slidingWindowSize = Math.min(4096, maxSeqLen);
            zylLayers[l] = new ZYLAttentionSystem(config);
            zylLayers[l].initialize();
        }
    }

    public float[] getEmbeddingWeights() { return embeddingWeights; }
    public float[] getPosEncoding() { return posEncoding; }
    public float[] getOutputWeights() { return outputWeights; }
    public float[][] getFfnWeights1() { return ffnWeights1; }
    public float[][] getFfnWeights2() { return ffnWeights2; }
    public float[][] getLayerNormWeight() { return layerNormWeight; }
    public float[][] getLayerNormBias() { return layerNormBias; }

    public MultiHeadAttention[] getMhaLayers() { return mhaLayers; }
    public ZYLAttentionSystem[] getZylLayers() { return zylLayers; }
    public AdaptiveBasisLayer[] getAblLayers() { return ablLayers; }
    public DynamicMemoryChannel[] getDmcLayers() { return dmcLayers; }
    public TopologyAwareRouter[] getTarLayers() { return tarLayers; }
    public CausalInterventionNetwork[] getCinLayers() { return cinLayers; }
    public InformationBottleneck[] getIbLayers() { return ibLayers; }
    public MetaRepresentationDistiller[] getMrdLayers() { return mrdLayers; }
    public PlasticityUpdateModule[] getPumLayers() { return pumLayers; }
    
    // 认知架构组件getters
    public boolean isEnableCognition() { return enableCognition; }
    public HierarchicalPerceptionModule getHierarchicalPerception() { return hierarchicalPerception; }
    public WorkingMemoryController getWorkingMemory() { return workingMemory; }
    public CognitiveReasoner getCognitiveReasoner() { return cognitiveReasoner; }
    public SchemaLearningModule getSchemaLearning() { return schemaLearning; }
    
    public void setEnableCognition(boolean enableCognition) {
        this.enableCognition = enableCognition;
        if (enableCognition && hierarchicalPerception == null) {
            initCognitionComponents();
        }
    }

    public float[] forwardPartial(float[] hidden, int startLayer, int endLayer) {
        int batchSize = hidden.length / (maxSeqLen * dModel);
        int seqLen = maxSeqLen;
        int totalSize = batchSize * seqLen * dModel;
        
        for (int l = startLayer; l < endLayer && l < nLayers; l++) {
            float[] mhaOutput;
            if (useZYLAttention && zylLayers != null) {
                mhaOutput = zylLayers[l].forward(hidden, batchSize, seqLen).output;
            } else {
                mhaOutput = mhaLayers[l].forward(hidden, batchSize, seqLen);
            }
            
            float[] attentionResidual = new float[totalSize];
            for (int i = 0; i < totalSize; i++) {
                attentionResidual[i] = hidden[i] + mhaOutput[i];
            }
            
            float[] mhaNorm = new float[totalSize];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = attentionResidual[offset + j];
                    }
                    float[] normalized = layerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        mhaNorm[offset + j] = normalized[j];
                    }
                }
            }
            
            float[] rfpOutput = ablLayers[l].forward(mhaNorm, batchSize, seqLen);
            float[] dmcOutput = dmcLayers[l].forward(rfpOutput, batchSize, seqLen);
            float[] tarOutput = tarLayers[l].forward(dmcOutput, batchSize, seqLen);
            float[] cinOutput = cinLayers[l].forward(tarOutput, batchSize, seqLen);
            float[] ibOutput = ibLayers[l].forward(cinOutput, batchSize, seqLen);
            float[] ffnOutput = applyFFN(ibOutput, ffnWeights1[l], ffnWeights2[l], batchSize, seqLen);
            float[] mrdOutput = mrdLayers[l].forward(ffnOutput, batchSize, seqLen);
            float[] spmOutput = pumLayers[l].forward(mrdOutput, batchSize, seqLen);

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] layerInput = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        layerInput[j] = mhaNorm[offset + j] + spmOutput[offset + j];
                    }
                    float[] normalized = layerNorm(layerInput, l);
                    for (int j = 0; j < dModel; j++) {
                        hidden[offset + j] = normalized[j];
                    }
                }
            }
        }
        
        return hidden;
    }

    public float[] backwardPartial(float[] gradients, int startLayer, int endLayer) {
        int totalSize = gradients.length;
        if (totalSize == 0 || totalSize % dModel != 0) {
            return gradients;
        }
        int seqLen = (int) Math.sqrt((double) totalSize / dModel);
        if (seqLen == 0) {
            return gradients;
        }
        int batchSize = totalSize / (seqLen * dModel);
        return backwardPartial(gradients, startLayer, endLayer, batchSize, seqLen);
    }
    
    public float[] backwardPartial(float[] gradients, int startLayer, int endLayer, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] dHidden = memoryOptimizer.allocateFloatArray(totalSize);
        System.arraycopy(gradients, 0, dHidden, 0, totalSize);
        
        for (int l = 0; l < nLayers; l++) {
            Arrays.fill(dFFNWeights1[l], 0);
            Arrays.fill(dFFNWeights2[l], 0);
        }
        Arrays.fill(dOutputWeights, 0);
        
        for (int l = Math.min(endLayer - 1, nLayers - 1); l >= startLayer; l--) {
            float[] dLayerNorm2 = new float[totalSize];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = dHidden[offset + j];
                    }
                    float[] dNorm = backwardLayerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        dLayerNorm2[offset + j] = dNorm[j];
                    }
                }
            }
            
            float[] dSPM = pumLayers[l].backward(dLayerNorm2, new float[totalSize], batchSize, seqLen).getdInput();
            float[] dMRD = mrdLayers[l].backward(dSPM, new float[totalSize], batchSize, seqLen).getdInput();
            
            GradientResult ffnGrad = backwardFFN(dMRD, ffnWeights1[l], ffnWeights2[l], l, batchSize, seqLen);
            float[] dIB = ibLayers[l].backward(ffnGrad.getdInput(), new float[totalSize], batchSize, seqLen).getdInput();
            float[] dCIN = cinLayers[l].backward(dIB, new float[totalSize], batchSize, seqLen).getdInput();
            float[] dTAR = tarLayers[l].backward(dCIN, new float[totalSize], batchSize, seqLen).getdInput();
            float[] dDMC = dmcLayers[l].backward(dTAR, new float[totalSize], batchSize, seqLen).getdInput();
            float[] dRFP = ablLayers[l].backward(dDMC, new float[totalSize], batchSize, seqLen).getdInput();
            
            float[] dLayerNorm1 = new float[totalSize];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = dRFP[offset + j];
                    }
                    float[] dNorm = backwardLayerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        dLayerNorm1[offset + j] = dNorm[j];
                    }
                }
            }
            
            float[] dMHA;
            if (useZYLAttention && zylLayers != null) {
                dMHA = dLayerNorm1;
            } else {
                float[] dummyInput = new float[totalSize];
                GradientResult mhaGrad = mhaLayers[l].backward(dLayerNorm1, dummyInput, batchSize, seqLen);
                dMHA = mhaGrad.getdInput();
            }
            
            for (int i = 0; i < totalSize; i++) {
                dHidden[i] = dMHA[i];
            }
        }
        
        float[] result = new float[totalSize];
        System.arraycopy(dHidden, 0, result, 0, totalSize);
        memoryOptimizer.releaseFloatArray(dHidden);
        return result;
    }
    
    public void updateParameters(float learningRate) {
        for (int l = 0; l < nLayers; l++) {
            for (int i = 0; i < ffnWeights1[l].length; i++) {
                ffnWeights1[l][i] -= learningRate * dFFNWeights1[l][i];
            }
            for (int i = 0; i < ffnWeights2[l].length; i++) {
                ffnWeights2[l][i] -= learningRate * dFFNWeights2[l][i];
            }
        }
        
        for (int i = 0; i < outputWeights.length; i++) {
            outputWeights[i] -= learningRate * dOutputWeights[i];
        }
    }
    
    public void updateParametersDirect(float[] gradients, float learningRate, int seqLen) {
        int batchSize = gradients.length / (seqLen * vocabSize);
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen - 1; s++) {
                int gradOffset = (b * seqLen + s) * vocabSize;
                int hiddenOffset = (b * seqLen + s) * dModel;
                
                for (int i = 0; i < vocabSize; i++) {
                    for (int j = 0; j < dModel; j++) {
                        int idx = i * dModel + j;
                        outputWeights[idx] -= learningRate * gradients[gradOffset + i] * 
                            (hiddenOffset + j < gradients.length ? gradients[hiddenOffset + j] : 1);
                    }
                }
            }
        }
    }
    
    private float[] backwardLayerNorm(float[] dOutput, int layer) {
        float[] result = new float[dModel];
        
        jdk.incubator.vector.FloatVector dOutVec;
        jdk.incubator.vector.FloatVector wVec;
        
        int i = 0;
        for (; i + VECTOR_SPECIES.length() <= dModel; i += VECTOR_SPECIES.length()) {
            dOutVec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, dOutput, i);
            wVec = jdk.incubator.vector.FloatVector.fromArray(VECTOR_SPECIES, layerNormWeight[layer], i);
            dOutVec.mul(wVec).intoArray(result, i);
        }
        for (; i < dModel; i++) {
            result[i] = dOutput[i] * layerNormWeight[layer][i];
        }
        
        return result;
    }
    
    private GradientResult backwardFFN(float[] dOutput, float[] w1, float[] w2, int layer, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] rowData = new float[dModel];
            System.arraycopy(dOutput, i * dModel, rowData, 0, dModel);
            float[] dW2 = MatrixOperations.matmul(rowData, w2, 1, dModel, dModel * 4);
            for (int j = 0; j < dModel * 4; j++) {
                dFFNWeights2[layer][i * dModel * 4 + j] += dW2[j];
            }
        }
        
        float[] dInput = new float[totalSize];
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] rowData = new float[dModel * 4];
            for (int j = 0; j < dModel * 4; j++) {
                rowData[j] = dFFNWeights2[layer][i * dModel * 4 + j];
            }
            float[] dW1 = MatrixOperations.matmul(rowData, w1, 1, dModel * 4, dModel);
            System.arraycopy(dW1, 0, dInput, i * dModel, dModel);
            for (int j = 0; j < dModel * dModel * 4; j++) {
                dFFNWeights1[layer][j] += rowData[j / dModel] * dOutput[(i * dModel) + (j % dModel)];
            }
        }
        
        GradientResult result = new GradientResult(dInput);
        return result;
    }

    public float[] forward(float[] input, int batchSize) {
        int seqLen = input.length / (batchSize * dModel);
        return forward(input, batchSize, seqLen);
    }

    public float[] forward(float[] input, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] hidden = memoryOptimizer.allocateFloatArray(totalSize);
        
        System.arraycopy(input, 0, hidden, 0, Math.min(input.length, totalSize));
        
        for (int l = 0; l < nLayers; l++) {
            float[] mhaOutput;
            if (useZYLAttention && zylLayers != null) {
                mhaOutput = zylLayers[l].forward(hidden, batchSize, seqLen).output;
            } else {
                mhaOutput = mhaLayers[l].forward(hidden, batchSize, seqLen);
            }
            
            float[] attentionResidual = new float[totalSize];
            for (int i = 0; i < totalSize; i++) {
                attentionResidual[i] = hidden[i] + mhaOutput[i];
            }
            
            float[] mhaNorm = new float[totalSize];
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = attentionResidual[offset + j];
                    }
                    float[] normalized = layerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        mhaNorm[offset + j] = normalized[j];
                    }
                }
            }
            
            float[] rfpOutput = ablLayers[l].forward(mhaNorm, batchSize, seqLen);
            float[] dmcOutput = dmcLayers[l].forward(rfpOutput, batchSize, seqLen);
            float[] tarOutput = tarLayers[l].forward(dmcOutput, batchSize, seqLen);
            float[] cinOutput = cinLayers[l].forward(tarOutput, batchSize, seqLen);
            float[] ibOutput = ibLayers[l].forward(cinOutput, batchSize, seqLen);
            float[] ffnOutput = applyFFN(ibOutput, ffnWeights1[l], ffnWeights2[l], batchSize, seqLen);
            float[] mrdOutput = mrdLayers[l].forward(ffnOutput, batchSize, seqLen);
            float[] spmOutput = pumLayers[l].forward(mrdOutput, batchSize, seqLen);

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] layerInput = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        layerInput[j] = mhaNorm[offset + j] + spmOutput[offset + j];
                    }
                    float[] normalized = layerNorm(layerInput, l);
                    for (int j = 0; j < dModel; j++) {
                        hidden[offset + j] = normalized[j];
                    }
                }
            }
        }

        float[] logits = MatrixOperations.matmul(hidden, outputWeights, batchSize * seqLen, dModel, vocabSize);
        memoryOptimizer.releaseFloatArray(hidden);
        return logits;
    }
}