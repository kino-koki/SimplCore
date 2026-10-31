package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * TopologyAwareRouter with Meta-Cognitive Routing and uncertainty estimation.
 * 
 * Innovation:
 * - Each expert computes its own confidence score
 * - Router weights by confidence (not just top-K logits)
 * - Uncertainty path for low-confidence inputs (learned default transform)
 * - Gumbel-Softmax during training for exploration, deterministic top-K during inference
 * - Tracks routing statistics for load balancing
 */
public class TopologyAwareRouter {

    private final int dModel;
    private final int nPaths;
    private final int topK;
    private final float[][] pathWeights;
    private final float[] signatureWeights;
    private final float[] routingWeights;
    
    // Meta-Cognitive: expert confidence weights
    private final float[] expertConfidenceWeights;  // [nPaths] per-expert confidence
    private final float[] uncertaintyPathWeights;    // [dModel * dModel] default transform
    private final float uncertaintyThreshold;        // threshold for uncertainty path
    
    // Gumbel-Softmax temperature for stochastic routing
    private float gumbelTemperature;
    private boolean trainingMode;  // true = stochastic (Gumbel), false = deterministic
    
    // Routing statistics for load balancing
    private final int[] routingCounts;
    private int totalRoutingDecisions;
    private boolean trackStatistics;

    public TopologyAwareRouter(int dModel, int nPaths, int topK) {
        this.dModel = dModel;
        this.nPaths = nPaths;
        this.topK = topK;

        Random random = new Random(42);
        this.pathWeights = new float[nPaths][dModel * dModel];
        this.signatureWeights = new float[dModel * dModel];
        this.routingWeights = new float[dModel * nPaths];
        
        // Meta-cognitive parameters
        this.expertConfidenceWeights = new float[nPaths];
        this.uncertaintyPathWeights = new float[dModel * dModel];
        this.uncertaintyThreshold = 0.3f;
        this.gumbelTemperature = 1.0f;
        this.trainingMode = false;
        
        // Routing statistics
        this.routingCounts = new int[nPaths];
        this.totalRoutingDecisions = 0;
        this.trackStatistics = true;

        for (int p = 0; p < nPaths; p++) {
            for (int i = 0; i < dModel * dModel; i++) {
                pathWeights[p][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            expertConfidenceWeights[p] = 0.5f + random.nextFloat() * 0.5f;
        }

        for (int i = 0; i < dModel * dModel; i++) {
            signatureWeights[i] = (random.nextFloat() - 0.5f) * 0.1f;
            uncertaintyPathWeights[i] = (random.nextFloat() - 0.5f) * 0.05f;
        }

        for (int i = 0; i < dModel * nPaths; i++) {
            routingWeights[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }
    }

    /**
     * Set training mode: true enables Gumbel-Softmax stochastic routing,
     * false enables deterministic top-K routing.
     */
    public void setTrainingMode(boolean training) {
        this.trainingMode = training;
    }

    public boolean isTrainingMode() {
        return trainingMode;
    }

    /**
     * Set Gumbel-Softmax temperature (higher = more exploration).
     */
    public void setGumbelTemperature(float temp) {
        this.gumbelTemperature = Math.max(0.1f, temp);
    }

    /**
     * Enable/disable routing statistics tracking.
     */
    public void setTrackStatistics(boolean track) {
        this.trackStatistics = track;
    }

    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[] signature = computeSignature(xSlice);
                float[] pathLogits = computePathLogits(signature);
                
                // Compute confidence scores
                float[] confidenceScores = computeConfidenceScores(pathLogits);
                
                // Check if max confidence exceeds threshold
                float maxConf = Float.NEGATIVE_INFINITY;
                for (float c : confidenceScores) {
                    if (c > maxConf) maxConf = c;
                }

                float[] combinedOutput = new float[dModel];

                if (maxConf < uncertaintyThreshold) {
                    // Uncertainty path: use learned default transform
                    combinedOutput = MatrixOperations.matmul(xSlice, uncertaintyPathWeights, 1, dModel, dModel);
                } else {
                    // Normal routing with confidence weighting
                    int[] selectedPaths;
                    if (trainingMode) {
                        selectedPaths = selectPathsGumbel(pathLogits);
                    } else {
                        selectedPaths = selectPathsDeterministic(confidenceScores);
                    }
                    
                    float[] pathScores = confidenceScores;
                    float scoreSum = 0f;

                    for (int k = 0; k < topK; k++) {
                        int pathIdx = selectedPaths[k];
                        float score = pathScores[pathIdx];

                        float[] pathWeight = pathWeights[pathIdx];
                        float[] pathOutput = MatrixOperations.matmul(xSlice, pathWeight, 1, dModel, dModel);

                        for (int j = 0; j < dModel; j++) {
                            combinedOutput[j] += score * pathOutput[j];
                        }
                        scoreSum += score;
                    }

                    if (scoreSum > 0) {
                        for (int j = 0; j < dModel; j++) {
                            combinedOutput[j] /= scoreSum;
                        }
                    }

                    // Update routing statistics
                    if (trackStatistics) {
                        for (int k = 0; k < topK; k++) {
                            routingCounts[selectedPaths[k]]++;
                        }
                        totalRoutingDecisions++;
                    }
                }

                for (int j = 0; j < dModel; j++) {
                    output[xOffset + j] = xSlice[j] + combinedOutput[j];
                }
            }
        }

        return output;
    }

    /**
     * Compute expert confidence scores from routing logits.
     * Confidence = sigmoid(logit * expert_confidence_weight)
     */
    private float[] computeConfidenceScores(float[] logits) {
        float[] scores = new float[nPaths];
        for (int i = 0; i < nPaths; i++) {
            float weightedLogit = logits[i] * expertConfidenceWeights[i];
            scores[i] = (float) (1.0 / (1.0 + Math.exp(-weightedLogit)));
        }
        return scores;
    }

    private float[] computeSignature(float[] x) {
        float[] signature = MatrixOperations.matmul(x, signatureWeights, 1, dModel, dModel);
        signature = MatrixOperations.relu(signature);
        return MatrixOperations.normalize(signature);
    }

    private float[] computePathLogits(float[] signature) {
        return MatrixOperations.matmul(signature, routingWeights, 1, dModel, nPaths);
    }

    /**
     * Gumbel-Softmax routing for stochastic exploration during training.
     */
    private int[] selectPathsGumbel(float[] logits) {
        Random random = new Random();
        float[] gumbelLogits = new float[nPaths];
        for (int i = 0; i < nPaths; i++) {
            float u = random.nextFloat() + 1e-10f;
            float gumbel = (float) (-Math.log(-Math.log(u)));
            gumbelLogits[i] = (logits[i] + gumbel) / gumbelTemperature;
        }

        return topKIndices(gumbelLogits);
    }

    /**
     * Deterministic top-K routing for inference.
     */
    private int[] selectPathsDeterministic(float[] scores) {
        return topKIndices(scores);
    }

    private int[] topKIndices(float[] values) {
        int[] indices = new int[nPaths];
        for (int i = 0; i < nPaths; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < nPaths - 1; i++) {
            for (int j = i + 1; j < nPaths; j++) {
                if (values[indices[j]] > values[indices[i]]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        return Arrays.copyOf(indices, topK);
    }

    /**
     * Get load balancing statistics.
     */
    public float[] getRoutingDistribution() {
        float[] dist = new float[nPaths];
        if (totalRoutingDecisions == 0) return dist;
        for (int i = 0; i < nPaths; i++) {
            dist[i] = (float) routingCounts[i] / totalRoutingDecisions;
        }
        return dist;
    }

    /**
     * Compute load balancing loss (encourages uniform routing).
     */
    public float computeLoadBalancingLoss() {
        float[] dist = getRoutingDistribution();
        float uniform = 1.0f / nPaths;
        float loss = 0f;
        for (int i = 0; i < nPaths; i++) {
            float diff = dist[i] - uniform;
            loss += diff * diff;
        }
        return loss * nPaths; // scaled by nPaths
    }

    public void resetRoutingStatistics() {
        Arrays.fill(routingCounts, 0);
        totalRoutingDecisions = 0;
    }

    // Backward compatible getters
    public int getDModel() { return dModel; }
    public int getNPaths() { return nPaths; }
    public int getTopK() { return topK; }

    // New getters
    public float[] getExpertConfidenceWeights() { return expertConfidenceWeights; }
    public float[] getUncertaintyPathWeights() { return uncertaintyPathWeights; }
    public float getUncertaintyThreshold() { return uncertaintyThreshold; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] w : pathWeights) for (float f : w) dos.writeFloat(f);
        for (float f : signatureWeights) dos.writeFloat(f);
        for (float f : routingWeights) dos.writeFloat(f);
        // Save meta-cognitive params
        for (float f : expertConfidenceWeights) dos.writeFloat(f);
        for (float f : uncertaintyPathWeights) dos.writeFloat(f);
        dos.writeFloat(uncertaintyThreshold);
        dos.writeFloat(gumbelTemperature);
        dos.writeBoolean(trainingMode);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dPathWeights = new float[nPaths * dModel * dModel];
        float[] dSignatureWeights = new float[dModel * dModel];
        float[] dRoutingWeights = new float[dModel * nPaths];
        float[] dExpertConfidence = new float[nPaths];
        float[] dUncertaintyWeights = new float[dModel * dModel];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                float[] signature = computeSignature(xSlice);
                float[] rawLogits = computePathLogits(signature);
                float[] confidenceScores = computeConfidenceScores(rawLogits);

                float maxConf = Float.NEGATIVE_INFINITY;
                for (float c : confidenceScores) {
                    if (c > maxConf) maxConf = c;
                }

                if (maxConf < uncertaintyThreshold) {
                    // Uncertainty path gradient
                    float[] dCombined = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        dCombined[j] = dOutSlice[j];
                    }
                    for (int j = 0; j < dModel * dModel; j++) {
                        dUncertaintyWeights[j] += xSlice[j / dModel] * dCombined[j % dModel];
                    }
                    float[] dXUncertainty = MatrixOperations.matmul(dCombined, 
                        MatrixOperations.transpose(uncertaintyPathWeights, dModel, dModel), 1, dModel, dModel);
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dXUncertainty[j];
                    }
                } else {
                    int[] selectedPaths = selectPathsDeterministic(confidenceScores);
                    float[] pathScores = confidenceScores;
                    float scoreSum = 0f;
                    for (int k = 0; k < topK; k++) {
                        scoreSum += pathScores[selectedPaths[k]];
                    }

                    float[] dCombined = new float[dModel];
                    for (int j = 0; j < dModel; j++) {
                        dCombined[j] = dOutSlice[j];
                    }

                    float[] dSignature = new float[dModel];
                    float[] dLogits = new float[nPaths];

                    for (int k = 0; k < topK; k++) {
                        int pathIdx = selectedPaths[k];
                        float score = pathScores[pathIdx];
                        float weight = scoreSum > 0 ? score / scoreSum : 1f / topK;

                        float[] pathWeight = pathWeights[pathIdx];
                        float[] pathOutput = MatrixOperations.matmul(xSlice, pathWeight, 1, dModel, dModel);

                        for (int j = 0; j < dModel; j++) {
                            float dPathOutput = dCombined[j] * weight;
                            float[] dXPath = MatrixOperations.matmul(new float[]{dPathOutput}, 
                                MatrixOperations.transpose(pathWeight, dModel, dModel), 1, 1, dModel);
                            for (int jj = 0; jj < dModel; jj++) {
                                dInput[xOffset + jj] += dXPath[jj];
                            }

                            for (int jj = 0; jj < dModel * dModel; jj++) {
                                dPathWeights[pathIdx * dModel * dModel + jj] += xSlice[jj / dModel] * dPathOutput;
                            }

                            dLogits[pathIdx] += dCombined[j] * pathOutput[j] * 
                                (scoreSum > 0 ? (1 - weight) / scoreSum : 0);
                        }
                    }

                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dOutSlice[j];
                    }

                    // Backprop through confidence scores
                    for (int i = 0; i < nPaths; i++) {
                        float sigmoid = confidenceScores[i];
                        float dSig = sigmoid * (1 - sigmoid);
                        dLogits[i] *= dSig * expertConfidenceWeights[i];
                        dExpertConfidence[i] += dLogits[i] * rawLogits[i];
                    }

                    float[] dSigRaw = MatrixOperations.matmul(dLogits, 
                        MatrixOperations.transpose(routingWeights, dModel, nPaths), 1, nPaths, dModel);
                    for (int j = 0; j < dModel; j++) {
                        float reluGrad = signature[j] > 0 ? 1 : 0;
                        dSignature[j] += dSigRaw[j] * reluGrad;
                    }

                    float[] dSigNorm = new float[dModel];
                    float sigNorm = MatrixOperations.norm(signature);
                    if (sigNorm > 1e-8f) {
                        float dot = MatrixOperations.dotProduct(dSignature, signature);
                        float sigNorm3 = sigNorm * sigNorm * sigNorm;
                        for (int j = 0; j < dModel; j++) {
                            dSigNorm[j] = dSignature[j] / sigNorm - dot * signature[j] / sigNorm3;
                        }
                    }

                    float[] dxSig = MatrixOperations.matmul(dSigNorm, 
                        MatrixOperations.transpose(signatureWeights, dModel, dModel), 1, dModel, dModel);
                    for (int j = 0; j < dModel; j++) {
                        dInput[xOffset + j] += dxSig[j];
                    }

                    for (int j = 0; j < dModel * dModel; j++) {
                        dSignatureWeights[j] += xSlice[j / dModel] * dSigNorm[j % dModel];
                    }

                    for (int j = 0; j < dModel * nPaths; j++) {
                        dRoutingWeights[j] += signature[j / nPaths] * dLogits[j % nPaths];
                    }
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("pathWeights", dPathWeights);
        result.addGradient("signatureWeights", dSignatureWeights);
        result.addGradient("routingWeights", dRoutingWeights);
        result.addGradient("expertConfidenceWeights", dExpertConfidence);
        result.addGradient("uncertaintyPathWeights", dUncertaintyWeights);
        return result;
    }

    public float[][] getPathWeights() { return pathWeights; }
    public float[] getSignatureWeights() { return signatureWeights; }
    public float[] getRoutingWeights() { return routingWeights; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int p = 0; p < nPaths; p++) {
            for (int i = 0; i < dModel * dModel; i++) {
                pathWeights[p][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < dModel * dModel; i++) {
            signatureWeights[i] = dis.readFloat();
        }
        for (int i = 0; i < dModel * nPaths; i++) {
            routingWeights[i] = dis.readFloat();
        }
        // Load meta-cognitive params
        for (int i = 0; i < nPaths; i++) {
            expertConfidenceWeights[i] = dis.readFloat();
        }
        for (int i = 0; i < dModel * dModel; i++) {
            uncertaintyPathWeights[i] = dis.readFloat();
        }
        dis.readFloat();
        dis.readFloat();
        trainingMode = dis.readBoolean();
    }
}
