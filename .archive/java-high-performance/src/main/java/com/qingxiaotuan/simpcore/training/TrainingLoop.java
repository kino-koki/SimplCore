package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import com.qingxiaotuan.simpcore.training.GradientResult;
import com.qingxiaotuan.simpcore.components.*;
import com.qingxiaotuan.simpcore.token.ZYLAttentionSystem;
import java.util.Map;

public class TrainingLoop {

    private final SimpCoreModel model;
    private final Tokenizer tokenizer;
    private final Optimizer optimizer;
    private final int dModel;
    private float[] gradEmbeddingWeights;
    private float[][] gradFFNWeights1;
    private float[][] gradFFNWeights2;
    private float[] gradOutputWeights;
    private float[][] gradLayerNormWeight;
    private float[][] gradLayerNormBias;
    private int totalSteps;
    private int currentStep;
    private float totalLoss;
    private int lossCount;
    private boolean mixedPrecision;
    private float scaleFactor;

    public TrainingLoop(SimpCoreModel model, Tokenizer tokenizer, Optimizer optimizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.optimizer = optimizer;
        this.dModel = model.getDModel();
        this.totalSteps = 0;
        this.currentStep = 0;
        this.totalLoss = 0;
        this.lossCount = 0;
        this.mixedPrecision = false;
        this.scaleFactor = 1.0f;
        initGradients();
    }

    public void enableMixedPrecision(boolean enabled) {
        this.mixedPrecision = enabled;
        this.scaleFactor = enabled ? 256.0f : 1.0f;
    }

    public boolean isMixedPrecisionEnabled() {
        return mixedPrecision;
    }

    private void initGradients() {
        int vocabSize = model.getVocabSize();
        int dModel = model.getDModel();
        int nLayers = model.getNLayers();
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();

        gradEmbeddingWeights = memoryOptimizer.allocateFloatArray(vocabSize * dModel);
        gradFFNWeights1 = new float[nLayers][];
        gradFFNWeights2 = new float[nLayers][];
        for (int l = 0; l < nLayers; l++) {
            gradFFNWeights1[l] = memoryOptimizer.allocateFloatArray(dModel * dModel * 4);
            gradFFNWeights2[l] = memoryOptimizer.allocateFloatArray(dModel * 4 * dModel);
        }
        gradOutputWeights = memoryOptimizer.allocateFloatArray(dModel * vocabSize);
        gradLayerNormWeight = new float[nLayers][];
        gradLayerNormBias = new float[nLayers][];
        for (int l = 0; l < nLayers; l++) {
            gradLayerNormWeight[l] = memoryOptimizer.allocateFloatArray(dModel);
            gradLayerNormBias[l] = memoryOptimizer.allocateFloatArray(dModel);
        }
    }

    public float trainStep(String text, int batchSize) {
        return trainStep(text, batchSize, false);
    }

    public float trainStep(String text, int batchSize, boolean accumulate) {
        int[] tokens = tokenizer.encode(text);
        if (tokens.length < 2) return Float.MAX_VALUE;

        int maxLen = model.getMaxSeqLen();
        int seqLen = Math.min(tokens.length - 1, maxLen);
        
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        int[] inputIds = new int[seqLen];
        int[] targetIds = new int[seqLen];

        for (int s = 0; s < seqLen; s++) {
            inputIds[s] = tokens[s];
            targetIds[s] = tokens[s + 1];
        }

        float[] logits = model.forward(inputIds, batchSize, seqLen);
        
        float loss = CrossEntropyLoss.compute(logits, targetIds, batchSize, seqLen, tokenizer.getVocabSize());
        trackLoss(loss);

        float[] dLogits = CrossEntropyLoss.computeGradients(logits, targetIds, batchSize, seqLen, tokenizer.getVocabSize());
        
        if (mixedPrecision) {
            for (int i = 0; i < dLogits.length; i++) {
                dLogits[i] *= scaleFactor;
            }
        }

        backward(dLogits, inputIds, batchSize, seqLen, accumulate);

        memoryOptimizer.releaseFloatArray(logits);
        memoryOptimizer.releaseFloatArray(dLogits);
        
        currentStep++;
        totalSteps++;
        
        return loss;
    }

    private void trackLoss(float loss) {
        totalLoss += loss;
        lossCount++;
    }

    public float getAverageLoss() {
        if (lossCount == 0) return Float.MAX_VALUE;
        return totalLoss / lossCount;
    }

    public void resetLossTracking() {
        totalLoss = 0;
        lossCount = 0;
    }

    private void backward(float[] dLogits, int[] inputIds, int batchSize, int seqLen, boolean accumulate) {
        int vocabSize = model.getVocabSize();
        int nLayers = model.getNLayers();
        int totalSize = batchSize * seqLen * dModel;
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();

        float[][] layerHiddenStates = new float[nLayers + 1][];
        float[][] layerOutputs = new float[nLayers][];
        float[][][] componentOutputs = new float[nLayers][8][];
        float[][] mhaOutputs = new float[nLayers][];
        ZYLAttentionSystem.ZYLResult[][] zylResults = model.isUseZYLAttention() ? new ZYLAttentionSystem.ZYLResult[nLayers][batchSize] : null;
        
        for (int l = 0; l <= nLayers; l++) {
            layerHiddenStates[l] = memoryOptimizer.allocateFloatArray(totalSize);
        }
        for (int l = 0; l < nLayers; l++) {
            layerOutputs[l] = memoryOptimizer.allocateFloatArray(totalSize);
            mhaOutputs[l] = memoryOptimizer.allocateFloatArray(totalSize);
            for (int c = 0; c < 8; c++) {
                componentOutputs[l][c] = memoryOptimizer.allocateFloatArray(totalSize);
            }
        }

        float[] initialHidden = memoryOptimizer.allocateFloatArray(totalSize);
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = inputIds[b * seqLen + s];
                int offset = (b * seqLen + s) * dModel;
                for (int i = 0; i < dModel; i++) {
                    initialHidden[offset + i] = model.getEmbeddingWeights()[tokenId * dModel + i]
                            + model.getPosEncoding()[s * dModel + i];
                }
            }
        }
        System.arraycopy(initialHidden, 0, layerHiddenStates[0], 0, totalSize);
        memoryOptimizer.releaseFloatArray(initialHidden);

        for (int l = 0; l < nLayers; l++) {
            float[] x = layerHiddenStates[l];
            
            float[] mhaOutput;
            if (model.isUseZYLAttention() && model.getZylLayers() != null) {
                ZYLAttentionSystem.ZYLResult zylResult = model.getZylLayers()[l].forward(x, batchSize, seqLen);
                mhaOutput = zylResult.output;
                if (zylResults != null) {
                    for (int b = 0; b < batchSize; b++) {
                        zylResults[l][b] = zylResult;
                    }
                }
            } else {
                mhaOutput = model.getMhaLayers()[l].forward(x, batchSize, seqLen);
            }
            System.arraycopy(mhaOutput, 0, mhaOutputs[l], 0, totalSize);
            
            float[] attentionResidual = memoryOptimizer.allocateFloatArray(totalSize);
            for (int i = 0; i < totalSize; i++) {
                attentionResidual[i] = x[i] + mhaOutput[i];
            }
            
            float[] mhaNorm = memoryOptimizer.allocateFloatArray(totalSize);
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = attentionResidual[offset + j];
                    }
                    float[] normalized = layerNorm(slice, l);
                    for (int j = 0; j < dModel; j++) {
                        mhaNorm[offset + j] = normalized[j];
                    }
                    memoryOptimizer.releaseFloatArray(slice);
                    memoryOptimizer.releaseFloatArray(normalized);
                }
            }
            System.arraycopy(mhaNorm, 0, componentOutputs[l][0], 0, totalSize);
            memoryOptimizer.releaseFloatArray(attentionResidual);
            
            float[] rfpOutput = model.getRfpLayers()[l].forward(mhaNorm, batchSize, seqLen);
            System.arraycopy(rfpOutput, 0, componentOutputs[l][1], 0, totalSize);
            
            float[] dmcOutput = model.getDmcLayers()[l].forward(rfpOutput, batchSize, seqLen);
            System.arraycopy(dmcOutput, 0, componentOutputs[l][2], 0, totalSize);
            
            float[] tarOutput = model.getTarLayers()[l].forward(dmcOutput, batchSize, seqLen);
            System.arraycopy(tarOutput, 0, componentOutputs[l][3], 0, totalSize);
            
            float[] cinOutput = model.getCinLayers()[l].forward(tarOutput, batchSize, seqLen);
            System.arraycopy(cinOutput, 0, componentOutputs[l][4], 0, totalSize);
            
            float[] ibOutput = model.getIbLayers()[l].forward(cinOutput, batchSize, seqLen);
            System.arraycopy(ibOutput, 0, componentOutputs[l][5], 0, totalSize);
            
            float[] ffnOutput = applyFFN(ibOutput, model.getFfnWeights1()[l], model.getFfnWeights2()[l], batchSize, seqLen);
            System.arraycopy(ffnOutput, 0, componentOutputs[l][6], 0, totalSize);
            
            float[] mrdOutput = model.getMrdLayers()[l].forward(ffnOutput, batchSize, seqLen);
            System.arraycopy(mrdOutput, 0, componentOutputs[l][7], 0, totalSize);
            
            float[] spmOutput = model.getSpmLayers()[l].forward(mrdOutput, batchSize, seqLen);

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] layerInput = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        layerInput[j] = mhaNorm[offset + j] + spmOutput[offset + j];
                    }
                    float[] normalized = layerNorm(layerInput, l);
                    for (int j = 0; j < dModel; j++) {
                        layerHiddenStates[l + 1][offset + j] = normalized[j];
                        layerOutputs[l][offset + j] = spmOutput[offset + j];
                    }
                    memoryOptimizer.releaseFloatArray(layerInput);
                    memoryOptimizer.releaseFloatArray(normalized);
                }
            }
            memoryOptimizer.releaseFloatArray(mhaOutput);
            memoryOptimizer.releaseFloatArray(mhaNorm);
        }

        for (int i = 0; i < gradOutputWeights.length; i++) {
            gradOutputWeights[i] = 0;
        }

        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] hiddenSlice = memoryOptimizer.allocateFloatArray(dModel);
            System.arraycopy(layerHiddenStates[nLayers], i * dModel, hiddenSlice, 0, dModel);

            float[] logitGrad = memoryOptimizer.allocateFloatArray(vocabSize);
            System.arraycopy(dLogits, i * vocabSize, logitGrad, 0, vocabSize);

            float[] outer = MatrixOperations.outerProduct(hiddenSlice, logitGrad, dModel, vocabSize);
            for (int j = 0; j < dModel * vocabSize; j++) {
                gradOutputWeights[j] += outer[j];
            }
            memoryOptimizer.releaseFloatArray(hiddenSlice);
            memoryOptimizer.releaseFloatArray(logitGrad);
            memoryOptimizer.releaseFloatArray(outer);
        }

        float[] dHidden = memoryOptimizer.allocateFloatArray(totalSize);
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] slice = memoryOptimizer.allocateFloatArray(vocabSize);
            System.arraycopy(dLogits, i * vocabSize, slice, 0, vocabSize);
            float[] grad = MatrixOperations.matmul(slice, model.getOutputWeights(), 1, vocabSize, dModel);
            System.arraycopy(grad, 0, dHidden, i * dModel, dModel);
            memoryOptimizer.releaseFloatArray(slice);
            memoryOptimizer.releaseFloatArray(grad);
        }

        for (int l = nLayers - 1; l >= 0; l--) {
            float[] dNext = memoryOptimizer.allocateFloatArray(totalSize);

            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;

                    float[] layerInput = memoryOptimizer.allocateFloatArray(dModel);
                    float[] prevHidden = memoryOptimizer.allocateFloatArray(dModel);
                    float[] layerOut = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        layerInput[j] = layerHiddenStates[l][offset + j] + layerOutputs[l][offset + j];
                        prevHidden[j] = layerHiddenStates[l][offset + j];
                        layerOut[j] = layerOutputs[l][offset + j];
                    }

                    float mean = 0;
                    for (int j = 0; j < dModel; j++) mean += layerInput[j];
                    mean /= dModel;

                    float var = 0;
                    for (int j = 0; j < dModel; j++) var += (layerInput[j] - mean) * (layerInput[j] - mean);
                    var /= dModel;
                    float std = (float) Math.sqrt(var + 1e-5);

                    float[] dNormalized = memoryOptimizer.allocateFloatArray(dModel);
                    float[] dScale = memoryOptimizer.allocateFloatArray(dModel);
                    float[] dBias = memoryOptimizer.allocateFloatArray(dModel);

                    for (int j = 0; j < dModel; j++) {
                        dNormalized[j] = dHidden[offset + j] * model.getLayerNormWeight()[l][j];
                        dScale[j] = dHidden[offset + j] * (layerInput[j] - mean) / std;
                        dBias[j] = dHidden[offset + j];
                    }

                    float dMean = 0;
                    float dVar = 0;
                    for (int j = 0; j < dModel; j++) {
                        dMean -= dNormalized[j] / std;
                        dVar += dNormalized[j] * (layerInput[j] - mean) / (2 * std * std * std);
                    }

                    float[] dLayerInput = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        dLayerInput[j] = dNormalized[j] / std + dVar * 2 * (layerInput[j] - mean) / dModel + dMean / dModel;
                    }

                    for (int j = 0; j < dModel; j++) {
                        dNext[offset + j] = dLayerInput[j];
                    }

                    for (int j = 0; j < dModel; j++) {
                        gradLayerNormWeight[l][j] += dScale[j];
                        gradLayerNormBias[l][j] += dBias[j];
                    }
                    
                    memoryOptimizer.releaseFloatArray(layerInput);
                    memoryOptimizer.releaseFloatArray(prevHidden);
                    memoryOptimizer.releaseFloatArray(layerOut);
                    memoryOptimizer.releaseFloatArray(dNormalized);
                    memoryOptimizer.releaseFloatArray(dScale);
                    memoryOptimizer.releaseFloatArray(dBias);
                    memoryOptimizer.releaseFloatArray(dLayerInput);
                }
            }

            GradientResult dSPM = model.getSpmLayers()[l].backward(dNext, componentOutputs[l][7], batchSize, seqLen);
            updateComponentWeights(model.getSpmLayers()[l], dSPM, accumulate);
            GradientResult dMRD = model.getMrdLayers()[l].backward(dSPM.getdInput(), componentOutputs[l][6], batchSize, seqLen);
            updateComponentWeights(model.getMrdLayers()[l], dMRD, accumulate);
            
            float[] ibInput = componentOutputs[l][5];
            float[] ffnInput = memoryOptimizer.allocateFloatArray(totalSize);
            System.arraycopy(ibInput, 0, ffnInput, 0, totalSize);

            for (int i = 0; i < gradFFNWeights1[l].length; i++) gradFFNWeights1[l][i] = 0;
            for (int i = 0; i < gradFFNWeights2[l].length; i++) gradFFNWeights2[l][i] = 0;

            float[] dFFN = memoryOptimizer.allocateFloatArray(totalSize);
            System.arraycopy(dMRD.getdInput(), 0, dFFN, 0, totalSize);
            float[] hiddenFFN = memoryOptimizer.allocateFloatArray(batchSize * seqLen * dModel * 4);
            for (int i = 0; i < batchSize * seqLen; i++) {
                float[] rowData = memoryOptimizer.allocateFloatArray(dModel);
                System.arraycopy(ffnInput, i * dModel, rowData, 0, dModel);
                float[] result = MatrixOperations.matmul(rowData, model.getFfnWeights1()[l], 1, dModel, dModel * 4);
                for (int j = 0; j < dModel * 4; j++) {
                    float sigmoid = 1f / (1f + (float) Math.exp(-result[j]));
                    hiddenFFN[i * dModel * 4 + j] = result[j] * sigmoid;
                }
                memoryOptimizer.releaseFloatArray(rowData);
                memoryOptimizer.releaseFloatArray(result);
            }

            for (int i = 0; i < batchSize * seqLen; i++) {
                float[] dOut = memoryOptimizer.allocateFloatArray(dModel);
                System.arraycopy(dFFN, i * dModel, dOut, 0, dModel);
                float[] rowData = memoryOptimizer.allocateFloatArray(dModel * 4);
                System.arraycopy(hiddenFFN, i * dModel * 4, rowData, 0, dModel * 4);

                float[] gradW2 = MatrixOperations.outerProduct(rowData, dOut, dModel * 4, dModel);
                for (int j = 0; j < dModel * 4 * dModel; j++) {
                    gradFFNWeights2[l][j] += gradW2[j];
                }

                float[] dAct = MatrixOperations.matmul(dOut, model.getFfnWeights2()[l], 1, dModel, dModel * 4);

                float[] ffn1 = memoryOptimizer.allocateFloatArray(dModel * 4);
                float[] inputRow = memoryOptimizer.allocateFloatArray(dModel);
                System.arraycopy(ffnInput, i * dModel, inputRow, 0, dModel);
                float[] result = MatrixOperations.matmul(inputRow, model.getFfnWeights1()[l], 1, dModel, dModel * 4);
                System.arraycopy(result, 0, ffn1, 0, dModel * 4);

                for (int j = 0; j < dModel * 4; j++) {
                    float sigmoid = 1f / (1f + (float) Math.exp(-ffn1[j]));
                    float dsigmoid = sigmoid * (1 - sigmoid);
                    dAct[j] *= (ffn1[j] * dsigmoid + sigmoid);
                }

                float[] gradW1 = MatrixOperations.outerProduct(inputRow, dAct, dModel, dModel * 4);
                for (int j = 0; j < dModel * dModel * 4; j++) {
                    gradFFNWeights1[l][j] += gradW1[j];
                }
                
                memoryOptimizer.releaseFloatArray(dOut);
                memoryOptimizer.releaseFloatArray(rowData);
                memoryOptimizer.releaseFloatArray(gradW2);
                memoryOptimizer.releaseFloatArray(dAct);
                memoryOptimizer.releaseFloatArray(ffn1);
                memoryOptimizer.releaseFloatArray(inputRow);
                memoryOptimizer.releaseFloatArray(result);
                memoryOptimizer.releaseFloatArray(gradW1);
            }

            if (mixedPrecision) {
                float invScale = 1.0f / scaleFactor;
                for (int i = 0; i < gradFFNWeights1[l].length; i++) gradFFNWeights1[l][i] *= invScale;
                for (int i = 0; i < gradFFNWeights2[l].length; i++) gradFFNWeights2[l][i] *= invScale;
                for (int i = 0; i < gradLayerNormWeight[l].length; i++) {
                    gradLayerNormWeight[l][i] *= invScale;
                    gradLayerNormBias[l][i] *= invScale;
                }
            }

            if (!accumulate) {
                optimizer.update(model.getFfnWeights1()[l], gradFFNWeights1[l]);
                optimizer.update(model.getFfnWeights2()[l], gradFFNWeights2[l]);
                optimizer.update(model.getLayerNormWeight()[l], gradLayerNormWeight[l]);
                optimizer.update(model.getLayerNormBias()[l], gradLayerNormBias[l]);
            } else {
                optimizer.accumulateGradients(gradFFNWeights1[l]);
                optimizer.accumulateGradients(gradFFNWeights2[l]);
                optimizer.accumulateGradients(gradLayerNormWeight[l]);
                optimizer.accumulateGradients(gradLayerNormBias[l]);
            }

            memoryOptimizer.releaseFloatArray(ffnInput);
            memoryOptimizer.releaseFloatArray(dFFN);
            memoryOptimizer.releaseFloatArray(hiddenFFN);

            GradientResult dIB = model.getIbLayers()[l].backward(dMRD.getdInput(), componentOutputs[l][4], batchSize, seqLen);
            updateComponentWeights(model.getIbLayers()[l], dIB, accumulate);
            GradientResult dCIN = model.getCinLayers()[l].backward(dIB.getdInput(), componentOutputs[l][3], batchSize, seqLen);
            updateComponentWeights(model.getCinLayers()[l], dCIN, accumulate);
            GradientResult dTAR = model.getTarLayers()[l].backward(dCIN.getdInput(), componentOutputs[l][2], batchSize, seqLen);
            updateComponentWeights(model.getTarLayers()[l], dTAR, accumulate);
            GradientResult dDMC = model.getDmcLayers()[l].backward(dTAR.getdInput(), componentOutputs[l][1], batchSize, seqLen);
            updateComponentWeights(model.getDmcLayers()[l], dDMC, accumulate);
            GradientResult dRFP = model.getRfpLayers()[l].backward(dDMC.getdInput(), componentOutputs[l][0], batchSize, seqLen);
            updateComponentWeights(model.getRfpLayers()[l], dRFP, accumulate);
            
            float[] dMhaNorm = dRFP.getdInput();
            float[] dAttentionResidual = memoryOptimizer.allocateFloatArray(totalSize);
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int offset = (b * seqLen + s) * dModel;
                    float[] slice = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        slice[j] = mhaOutputs[l][offset + j] + layerHiddenStates[l][offset + j];
                    }
                    float mean = 0;
                    for (int j = 0; j < dModel; j++) mean += slice[j];
                    mean /= dModel;
                    float var = 0;
                    for (int j = 0; j < dModel; j++) var += (slice[j] - mean) * (slice[j] - mean);
                    var /= dModel;
                    float std = (float) Math.sqrt(var + 1e-5);
                    float[] dNormalized = memoryOptimizer.allocateFloatArray(dModel);
                    for (int j = 0; j < dModel; j++) {
                        dNormalized[j] = dMhaNorm[offset + j] * model.getLayerNormWeight()[l][j];
                    }
                    float dMean = 0, dVar = 0;
                    for (int j = 0; j < dModel; j++) {
                        dMean -= dNormalized[j] / std;
                        dVar += dNormalized[j] * (slice[j] - mean) / (2 * std * std * std);
                    }
                    for (int j = 0; j < dModel; j++) {
                        dAttentionResidual[offset + j] = dNormalized[j] / std + dVar * 2 * (slice[j] - mean) / dModel + dMean / dModel;
                    }
                    memoryOptimizer.releaseFloatArray(slice);
                    memoryOptimizer.releaseFloatArray(dNormalized);
                }
            }
            
            float[] dMHA = memoryOptimizer.allocateFloatArray(totalSize);
            for (int i = 0; i < totalSize; i++) {
                dMHA[i] = dAttentionResidual[i];
            }
            
            float[] dX = memoryOptimizer.allocateFloatArray(totalSize);
            System.arraycopy(dAttentionResidual, 0, dX, 0, totalSize);
            
            GradientResult mhaResult;
            if (model.isUseZYLAttention() && model.getZylLayers() != null && zylResults != null) {
                ZYLAttentionSystem.ZYLResult zylResult = zylResults[l][0];
                mhaResult = model.getZylLayers()[l].backward(dMHA, layerHiddenStates[l], batchSize, seqLen, zylResult);
                updateZYLComponentWeights(model.getZylLayers()[l], mhaResult, accumulate);
            } else {
                mhaResult = model.getMhaLayers()[l].backward(dMHA, layerHiddenStates[l], batchSize, seqLen);
                updateComponentWeights(model.getMhaLayers()[l], mhaResult, accumulate);
            }
            float[] mhaGrads = mhaResult.getdInput();
            for (int i = 0; i < totalSize; i++) {
                dX[i] += mhaGrads[i];
            }
            
            memoryOptimizer.releaseFloatArray(dAttentionResidual);
            memoryOptimizer.releaseFloatArray(dMHA);
            memoryOptimizer.releaseFloatArray(mhaGrads);

            System.arraycopy(dX, 0, dHidden, 0, totalSize);
            memoryOptimizer.releaseFloatArray(dNext);
            memoryOptimizer.releaseFloatArray(dX);
        }

        for (int i = 0; i < gradEmbeddingWeights.length; i++) {
            gradEmbeddingWeights[i] = 0;
        }

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = inputIds[b * seqLen + s];
                int offset = (b * seqLen + s) * dModel;
                for (int j = 0; j < dModel; j++) {
                    gradEmbeddingWeights[tokenId * dModel + j] += dHidden[offset + j];
                }
            }
        }

        if (mixedPrecision) {
            float invScale = 1.0f / scaleFactor;
            for (int i = 0; i < gradOutputWeights.length; i++) gradOutputWeights[i] *= invScale;
            for (int i = 0; i < gradEmbeddingWeights.length; i++) gradEmbeddingWeights[i] *= invScale;
        }

        if (!accumulate) {
            optimizer.update(model.getOutputWeights(), gradOutputWeights);
            optimizer.update(model.getEmbeddingWeights(), gradEmbeddingWeights);
        } else {
            optimizer.accumulateGradients(gradOutputWeights);
            optimizer.accumulateGradients(gradEmbeddingWeights);
        }
        
        for (int l = 0; l <= nLayers; l++) {
            memoryOptimizer.releaseFloatArray(layerHiddenStates[l]);
        }
        for (int l = 0; l < nLayers; l++) {
            memoryOptimizer.releaseFloatArray(layerOutputs[l]);
            for (int c = 0; c < 7; c++) {
                memoryOptimizer.releaseFloatArray(componentOutputs[l][c]);
            }
        }
        memoryOptimizer.releaseFloatArray(dHidden);
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

    private float[] layerNorm(float[] x, int layer) {
        float mean = 0;
        for (int i = 0; i < model.getDModel(); i++) {
            mean += x[i];
        }
        mean /= model.getDModel();

        float var = 0;
        for (int i = 0; i < model.getDModel(); i++) {
            var += (x[i] - mean) * (x[i] - mean);
        }
        var /= model.getDModel();

        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        float[] output = memoryOptimizer.allocateFloatArray(model.getDModel());
        float std = (float) Math.sqrt(var + 1e-5);
        for (int i = 0; i < model.getDModel(); i++) {
            output[i] = (x[i] - mean) / std * model.getLayerNormWeight()[layer][i] + model.getLayerNormBias()[layer][i];
        }
        return output;
    }

    private void updateComponentWeights(AdaptiveBasisLayer abl, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dOutputWeights = grads.get("outputWeights");
        float[] dBases = grads.get("bases");
        float[] dFreqScale = grads.get("freqScale");
        float[] dPhaseShift = grads.get("phaseShift");

        if (dOutputWeights != null) {
            for (int h = 0; h < abl.getNHeads(); h++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dOutputWeights, h * dModel * dModel, dModel * dModel);
                } else {
                    optimizer.update(abl.getOutputWeights()[h], dOutputWeights, h * dModel * dModel, dModel * dModel);
                }
            }
        }
        if (dBases != null) {
            for (int h = 0; h < abl.getNHeads(); h++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dBases, h * abl.getNBases() * dModel, abl.getNBases() * dModel);
                } else {
                    optimizer.update(abl.getBases()[h], dBases, h * abl.getNBases() * dModel, abl.getNBases() * dModel);
                }
            }
        }
        if (dFreqScale != null && !accumulate) optimizer.update(abl.getFreqScale(), dFreqScale);
        if (dPhaseShift != null && !accumulate) optimizer.update(abl.getPhaseShift(), dPhaseShift);
    }

    private void updateComponentWeights(DynamicMemoryChannel dmc, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dAttentionWeights = grads.get("attentionWeights");
        float[] dGateWeights = grads.get("gateWeights");
        float[] dDecayRates = grads.get("decayRates");

        if (dAttentionWeights != null) {
            for (int c = 0; c < dmc.getNChannels(); c++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dAttentionWeights, c * dModel * dmc.getMemoryDim(), dModel * dmc.getMemoryDim());
                } else {
                    optimizer.update(dmc.getAttentionWeights()[c], dAttentionWeights, c * dModel * dmc.getMemoryDim(), dModel * dmc.getMemoryDim());
                }
            }
        }
        if (dGateWeights != null && !accumulate) optimizer.update(dmc.getGateWeights(), dGateWeights);
        if (dDecayRates != null && !accumulate) optimizer.update(dmc.getDecayRates(), dDecayRates);
    }

    private void updateComponentWeights(TopologyAwareRouter tar, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dPathWeights = grads.get("pathWeights");
        float[] dSignatureWeights = grads.get("signatureWeights");
        float[] dRoutingWeights = grads.get("routingWeights");

        if (dPathWeights != null) {
            for (int p = 0; p < tar.getNPaths(); p++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dPathWeights, p * dModel * dModel, dModel * dModel);
                } else {
                    optimizer.update(tar.getPathWeights()[p], dPathWeights, p * dModel * dModel, dModel * dModel);
                }
            }
        }
        if (dSignatureWeights != null && !accumulate) optimizer.update(tar.getSignatureWeights(), dSignatureWeights);
        if (dRoutingWeights != null && !accumulate) optimizer.update(tar.getRoutingWeights(), dRoutingWeights);
    }

    private void updateComponentWeights(CausalInterventionNetwork cin, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dInterventionWeights = grads.get("interventionWeights");
        float[] dConfounderWeights = grads.get("confounderWeights");
        float[] dInterventionThreshold = grads.get("interventionThreshold");

        if (dInterventionWeights != null && !accumulate) {
            for (int i = 0; i < dModel; i++) {
                optimizer.update(cin.getInterventionWeights()[i], dInterventionWeights, i * dModel, dModel);
            }
        }
        if (dConfounderWeights != null && !accumulate) {
            for (int i = 0; i < dModel; i++) {
                optimizer.update(cin.getConfounderWeights()[i], dConfounderWeights, i * dModel, dModel);
            }
        }
        if (dInterventionThreshold != null && !accumulate) optimizer.update(cin.getInterventionThreshold(), dInterventionThreshold);
    }

    private void updateComponentWeights(InformationBottleneck ib, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dImportanceWeights = grads.get("importanceWeights");
        float[] dBottleneckThreshold = grads.get("bottleneckThreshold");

        if (dImportanceWeights != null && !accumulate) optimizer.update(ib.getImportanceWeights(), dImportanceWeights);
        if (dBottleneckThreshold != null && !accumulate) optimizer.update(ib.getBottleneckThreshold(), dBottleneckThreshold);
    }

    private void updateComponentWeights(MetaRepresentationDistiller mrd, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dEncoderWeights = grads.get("encoderWeights");
        float[] dDecoderWeights = grads.get("decoderWeights");
        float[] dLatentBias = grads.get("latentBias");

        if (dEncoderWeights != null && !accumulate) {
            for (int i = 0; i < dModel; i++) {
                optimizer.update(mrd.getEncoderWeights()[i], dEncoderWeights, i * mrd.getLatentDim(), mrd.getLatentDim());
            }
        }
        if (dDecoderWeights != null && !accumulate) {
            for (int i = 0; i < mrd.getLatentDim(); i++) {
                optimizer.update(mrd.getDecoderWeights()[i], dDecoderWeights, i * dModel, dModel);
            }
        }
        if (dLatentBias != null && !accumulate) optimizer.update(mrd.getLatentBias(), dLatentBias);
    }

    private void updateComponentWeights(PlasticityUpdateModule pum, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dPlasticWeights = grads.get("plasticWeights");

        if (dPlasticWeights != null) {
            for (int i = 0; i < dModel; i++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dPlasticWeights, i * dModel, dModel);
                } else {
                    optimizer.update(pum.getPlasticWeights()[i], dPlasticWeights, i * dModel, dModel);
                }
            }
        }
    }

    public void trainEpoch(String[] texts, int iterations) {
        for (int iter = 0; iter < iterations; iter++) {
            String text = texts[iter % texts.length];
            float loss = trainStep(text, 1);
            if (iter % 10 == 0) {
                System.out.printf("Iteration %d: Loss = %.4f, LR = %.6f%n", 
                    iter, loss, optimizer.getCurrentLearningRate());
            }
        }
    }

    public String generate(String prompt, int maxNewTokens) {
        int[] inputIds = tokenizer.encode(prompt);
        int[] outputIds = model.generate(inputIds, maxNewTokens, tokenizer.getEosTokenId());
        return tokenizer.decode(outputIds);
    }

    public SimpCoreModel getModel() {
        return model;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public int getCurrentStep() {
        return currentStep;
    }

    private void updateComponentWeights(MultiHeadAttention mha, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dQWeights = grads.get("qWeights");
        float[] dKWeights = grads.get("kWeights");
        float[] dVWeights = grads.get("vWeights");
        float[] dOWeights = grads.get("oWeights");

        int nHeads = mha.getNHeads();
        int headDim = mha.getHeadDim();

        if (dQWeights != null) {
            for (int h = 0; h < nHeads; h++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dQWeights, h * dModel * headDim, dModel * headDim);
                } else {
                    optimizer.update(mha.getQWeights()[h], dQWeights, h * dModel * headDim, dModel * headDim);
                }
            }
        }
        if (dKWeights != null) {
            float[][] kWeightsShared = mha.getKWeightsShared();
            if (accumulate) {
                optimizer.accumulateGradients(dKWeights, 0, dKWeights.length);
            } else {
                for (int kh = 0; kh < kWeightsShared.length; kh++) {
                    float[] kRow = kWeightsShared[kh];
                    int gradOffset = kh * kRow.length;
                    if (gradOffset + kRow.length <= dKWeights.length) {
                        optimizer.update(kRow, dKWeights, gradOffset, kRow.length);
                    }
                }
            }
        }
        if (dVWeights != null) {
            float[][] vWeightsShared = mha.getVWeightsShared();
            if (accumulate) {
                optimizer.accumulateGradients(dVWeights, 0, dVWeights.length);
            } else {
                for (int kh = 0; kh < vWeightsShared.length; kh++) {
                    float[] vRow = vWeightsShared[kh];
                    int gradOffset = kh * vRow.length;
                    if (gradOffset + vRow.length <= dVWeights.length) {
                        optimizer.update(vRow, dVWeights, gradOffset, vRow.length);
                    }
                }
            }
        }
        if (dOWeights != null) {
            for (int h = 0; h < nHeads; h++) {
                if (accumulate) {
                    optimizer.accumulateGradients(dOWeights, h * headDim * dModel, headDim * dModel);
                } else {
                    optimizer.update(mha.getOWeights()[h], dOWeights, h * headDim * dModel, headDim * dModel);
                }
            }
        }
    }

    private void updateZYLComponentWeights(ZYLAttentionSystem zyl, GradientResult grad, boolean accumulate) {
        Map<String, float[]> grads = grad.getParameterGradients();
        float[] dQWeights = grads.get("qWeights");
        float[] dKWeights = grads.get("kWeights");
        float[] dVWeights = grads.get("vWeights");
        float[] dOWeights = grads.get("oWeights");
        float[] dPositionBias = grads.get("positionBias");

        int nHeads = zyl.getConfig().nHeads;
        int headDim = zyl.getConfig().headDim;

        if (dQWeights != null) {
            if (accumulate) {
                optimizer.accumulateGradients(dQWeights);
            } else {
                optimizer.update(zyl.getQWeights(), dQWeights);
            }
        }
        if (dKWeights != null) {
            if (accumulate) {
                optimizer.accumulateGradients(dKWeights);
            } else {
                optimizer.update(zyl.getKWeights(), dKWeights);
            }
        }
        if (dVWeights != null) {
            if (accumulate) {
                optimizer.accumulateGradients(dVWeights);
            } else {
                optimizer.update(zyl.getVWeights(), dVWeights);
            }
        }
        if (dOWeights != null) {
            if (accumulate) {
                optimizer.accumulateGradients(dOWeights);
            } else {
                optimizer.update(zyl.getOWeights(), dOWeights);
            }
        }
        if (dPositionBias != null) {
            if (accumulate) {
                optimizer.accumulateGradients(dPositionBias);
            } else {
                optimizer.update(zyl.getPositionBias(), dPositionBias);
            }
        }
    }
}