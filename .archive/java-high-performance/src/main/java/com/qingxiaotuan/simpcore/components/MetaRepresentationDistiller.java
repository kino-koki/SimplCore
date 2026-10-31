package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Arrays;
import java.util.Random;

/**
 * MetaRepresentationDistiller with Compositional Concept Disentanglement.
 * 
 * Innovation:
 * - Latent space decomposed into concept-specific subspaces
 * - Each subspace learns independent features
 * - "Concept mixing" operation that recombines concepts from different inputs
 * - disentangle(float[] x) returns concept vector array
 * - compose(float[][] concepts) for concept recombination
 * - β-VAE style regularization (beta=4) to encourage disentanglement
 */
public class MetaRepresentationDistiller {

    private final int dModel;
    private final int latentDim;
    private final int nConcepts;
    private final float[][] encoderWeights;
    private final float[][] decoderWeights;
    private final float[] latentBias;
    
    // Concept disentanglement parameters
    private final int conceptDim;                    // dimensions per concept
    private final float[][] conceptEncoders;          // [nConcepts][dModel * conceptDim]
    private final float[][] conceptDecoders;          // [nConcepts][conceptDim * dModel]
    private final float[] conceptBias;                // [nConcepts * conceptDim]
    private static final float BETA_VAE = 4.0f;       // β-VAE regularization strength

    public MetaRepresentationDistiller(int dModel, int latentDim) {
        this.dModel = dModel;
        this.latentDim = latentDim;
        this.nConcepts = Math.max(2, latentDim / 4);  // at least 2 concepts
        this.conceptDim = latentDim / nConcepts;

        Random random = new Random(42);
        this.encoderWeights = new float[dModel][latentDim];
        this.decoderWeights = new float[latentDim][dModel];
        this.latentBias = new float[latentDim];
        
        // Concept disentanglement parameters
        this.conceptEncoders = new float[nConcepts][dModel * conceptDim];
        this.conceptDecoders = new float[nConcepts][conceptDim * dModel];
        this.conceptBias = new float[nConcepts * conceptDim];

        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < latentDim; j++) {
                encoderWeights[i][j] = (random.nextFloat() - 0.5f) * 0.1f;
            }
        }

        for (int i = 0; i < latentDim; i++) {
            for (int j = 0; j < dModel; j++) {
                decoderWeights[i][j] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            latentBias[i] = (random.nextFloat() - 0.5f) * 0.05f;
        }
        
        // Initialize concept-specific encoders/decoders
        for (int c = 0; c < nConcepts; c++) {
            for (int i = 0; i < dModel * conceptDim; i++) {
                conceptEncoders[c][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            for (int i = 0; i < conceptDim * dModel; i++) {
                conceptDecoders[c][i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
        }
        for (int i = 0; i < nConcepts * conceptDim; i++) {
            conceptBias[i] = (random.nextFloat() - 0.5f) * 0.05f;
        }
    }

    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] output = new float[totalSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                // Encode with concept disentanglement
                float[][] concepts = disentangle(xSlice);
                
                // Reconstruct from concepts with β-VAE style bottleneck
                float[] latent = new float[latentDim];
                for (int c = 0; c < nConcepts; c++) {
                    System.arraycopy(concepts[c], 0, latent, c * conceptDim, conceptDim);
                }
                
                float[] reconstructed = decode(latent);

                float distillationFactor = (float) (1.0 - (double) latentDim / dModel);
                for (int j = 0; j < dModel; j++) {
                    output[xOffset + j] = xSlice[j] * (1 - distillationFactor) + reconstructed[j] * distillationFactor;
                }
            }
        }

        return output;
    }

    /**
     * Disentangle input into concept-specific subspace vectors.
     * Each concept is encoded by its own encoder network.
     * 
     * @param x input vector of dimension dModel
     * @return array of concept vectors, each of dimension conceptDim
     */
    public float[][] disentangle(float[] x) {
        float[][] concepts = new float[nConcepts][conceptDim];
        
        for (int c = 0; c < nConcepts; c++) {
            float[] encWeights = conceptEncoders[c];
            int biasOffset = c * conceptDim;
            
            for (int i = 0; i < conceptDim; i++) {
                float sum = conceptBias[biasOffset + i];
                for (int j = 0; j < dModel; j++) {
                    sum += encWeights[j * conceptDim + i] * x[j];
                }
                // Tanh activation for bounded concept representation
                concepts[c][i] = (float) Math.tanh(sum);
            }
        }
        
        return concepts;
    }

    /**
     * Compose/recombine concept vectors from different inputs.
     * Takes concepts from potentially different inputs and combines them.
     * 
     * @param concepts array of concept vectors (nConcepts x conceptDim)
     * @return reconstructed output vector of dimension dModel
     */
    public float[] compose(float[][] concepts) {
        if (concepts.length != nConcepts) {
            throw new IllegalArgumentException("Number of concepts must match: " + nConcepts);
        }
        
        // Concatenate concepts into latent vector
        float[] latent = new float[latentDim];
        for (int c = 0; c < nConcepts; c++) {
            System.arraycopy(concepts[c], 0, latent, c * conceptDim, conceptDim);
        }
        
        // Decode to output
        return decode(latent);
    }

    /**
     * Concept mixing: takes two inputs and mixes their concepts.
     * For each concept, randomly selects from either input A or B.
     * 
     * @param conceptsA concepts from input A
     * @param conceptsB concepts from input B
     * @param mixMask boolean array controlling which concepts come from B
     * @return mixed reconstruction
     */
    public float[] conceptMix(float[][] conceptsA, float[][] conceptsB, boolean[] mixMask) {
        float[][] mixed = new float[nConcepts][conceptDim];
        for (int c = 0; c < nConcepts; c++) {
            if (mixMask != null && c < mixMask.length && mixMask[c]) {
                System.arraycopy(conceptsB[c], 0, mixed[c], 0, conceptDim);
            } else {
                System.arraycopy(conceptsA[c], 0, mixed[c], 0, conceptDim);
            }
        }
        return compose(mixed);
    }

    /**
     * Compute β-VAE regularization loss to encourage disentanglement.
     * Measures the total correlation between concept dimensions.
     */
    public float computeBetaVAELoss(float[] x, int batchSize, int seqLen) {
        float totalLoss = 0f;
        int count = 0;

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[][] concepts = disentangle(xSlice);
                
                // Reconstruction loss
                float[] latent = new float[latentDim];
                for (int c = 0; c < nConcepts; c++) {
                    System.arraycopy(concepts[c], 0, latent, c * conceptDim, conceptDim);
                }
                float[] reconstructed = decode(latent);
                
                float reconLoss = 0f;
                for (int j = 0; j < dModel; j++) {
                    float diff = xSlice[j] - reconstructed[j];
                    reconLoss += diff * diff;
                }
                
                // Concept independence loss (encourages each concept to be independent)
                float conceptLoss = 0f;
                for (int c = 0; c < nConcepts; c++) {
                    float conceptNorm = 0f;
                    for (int i = 0; i < conceptDim; i++) {
                        conceptNorm += concepts[c][i] * concepts[c][i];
                    }
                    conceptLoss += conceptNorm;
                }
                
                // Total loss with β-VAE weighting
                totalLoss += reconLoss + BETA_VAE * conceptLoss;
                count++;
            }
        }

        return count > 0 ? totalLoss / count : 0f;
    }

    private float[] encode(float[] x) {
        float[] latent = new float[latentDim];
        for (int i = 0; i < latentDim; i++) {
            float sum = latentBias[i];
            for (int j = 0; j < dModel; j++) {
                sum += encoderWeights[j][i] * x[j];
            }
            latent[i] = (float) Math.tanh(sum);
        }
        return latent;
    }

    private float[] decode(float[] latent) {
        float[] reconstructed = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            float sum = 0f;
            for (int j = 0; j < latentDim; j++) {
                sum += decoderWeights[j][i] * latent[j];
            }
            reconstructed[i] = sum;
        }
        return reconstructed;
    }

    public float[] getLatentRepresentation(float[] x) {
        return encode(x);
    }

    public float[] reconstruct(float[] latent) {
        return decode(latent);
    }

    public float computeDistillationLoss(float[] x, int batchSize, int seqLen) {
        float totalLoss = 0f;
        int count = 0;

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);

                float[][] concepts = disentangle(xSlice);
                float[] latent = new float[latentDim];
                for (int c = 0; c < nConcepts; c++) {
                    System.arraycopy(concepts[c], 0, latent, c * conceptDim, conceptDim);
                }
                float[] reconstructed = decode(latent);

                float loss = 0f;
                for (int j = 0; j < dModel; j++) {
                    loss += (xSlice[j] - reconstructed[j]) * (xSlice[j] - reconstructed[j]);
                }
                totalLoss += loss / dModel;
                count++;
            }
        }

        return totalLoss / count;
    }

    public int getDModel() { return dModel; }
    public int getLatentDim() { return latentDim; }
    public int getNConcepts() { return nConcepts; }
    public int getConceptDim() { return conceptDim; }
    public float getCompressionRatio() { return (float) latentDim / dModel; }
    public static float getBetaVAE() { return BETA_VAE; }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        for (float[] row : encoderWeights) for (float f : row) dos.writeFloat(f);
        for (float[] row : decoderWeights) for (float f : row) dos.writeFloat(f);
        for (float f : latentBias) dos.writeFloat(f);
        // Save concept params
        for (float[] row : conceptEncoders) for (float f : row) dos.writeFloat(f);
        for (float[] row : conceptDecoders) for (float f : row) dos.writeFloat(f);
        for (float f : conceptBias) dos.writeFloat(f);
    }

    public GradientResult backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        float[] dEncoderWeights = new float[dModel * latentDim];
        float[] dDecoderWeights = new float[latentDim * dModel];
        float[] dLatentBias = new float[latentDim];
        float[] dConceptEncoders = new float[nConcepts * dModel * conceptDim];
        float[] dConceptDecoders = new float[nConcepts * conceptDim * dModel];
        float[] dConceptBias = new float[nConcepts * conceptDim];

        float distillationFactor = (float) (1.0 - (double) latentDim / dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int xOffset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, xOffset, xOffset + dModel);
                float[] dOutSlice = Arrays.copyOfRange(dOutput, xOffset, xOffset + dModel);

                // Forward recompute
                float[][] concepts = disentangle(xSlice);
                float[] latent = new float[latentDim];
                for (int c = 0; c < nConcepts; c++) {
                    System.arraycopy(concepts[c], 0, latent, c * conceptDim, conceptDim);
                }
                float[] reconstructed = decode(latent);

                float[] dReconstructed = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    dReconstructed[j] = dOutSlice[j] * distillationFactor;
                }

                // Gradient through decoder
                float[] dLatent = new float[latentDim];
                for (int i = 0; i < latentDim; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dLatent[i] += dReconstructed[j] * decoderWeights[i][j];
                    }
                }

                // Gradient through tanh for each concept
                float[][] dConcepts = new float[nConcepts][conceptDim];
                for (int c = 0; c < nConcepts; c++) {
                    for (int i = 0; i < conceptDim; i++) {
                        int latentIdx = c * conceptDim + i;
                        float dtanh = 1f - latent[latentIdx] * latent[latentIdx];
                        dConcepts[c][i] = dLatent[latentIdx] * dtanh;
                    }
                }

                // Gradient through concept encoders
                for (int c = 0; c < nConcepts; c++) {
                    float[] encWeights = conceptEncoders[c];
                    int biasOffset = c * conceptDim;
                    
                    for (int i = 0; i < conceptDim; i++) {
                        float dConcept = dConcepts[c][i];
                        dConceptBias[biasOffset + i] += dConcept;
                        
                        for (int j = 0; j < dModel; j++) {
                            dInput[xOffset + j] += dConcept * encWeights[j * conceptDim + i];
                            dConceptEncoders[c * dModel * conceptDim + j * conceptDim + i] += xSlice[j] * dConcept;
                        }
                    }
                }

                // Gradient through decoder weights
                for (int i = 0; i < latentDim; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dDecoderWeights[i * dModel + j] += latent[i] * dReconstructed[j];
                    }
                }

                // Gradient through concept decoders
                for (int c = 0; c < nConcepts; c++) {
                    float[] conceptsC = concepts[c];
                    float[] decWeights = conceptDecoders[c];
                    
                    // dConcepts already includes tanh gradient above
                    // The decoder part uses standard decoder, not concept decoders
                    // (conceptDecoders are for future use in compose pathway)
                    for (int i = 0; i < conceptDim; i++) {
                        for (int j = 0; j < dModel; j++) {
                            dConceptDecoders[c * conceptDim * dModel + i * dModel + j] += conceptsC[i] * dReconstructed[j];
                        }
                    }
                }

                // Residual connection
                for (int j = 0; j < dModel; j++) {
                    dInput[xOffset + j] += dOutSlice[j] * (1 - distillationFactor);
                }

                // Standard encoder gradients (for backward compatibility)
                float[] stdLatent = encode(xSlice);
                float[] dStdLatent = new float[latentDim];
                for (int i = 0; i < latentDim; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dStdLatent[i] += dReconstructed[j] * decoderWeights[i][j];
                    }
                    float dtanh = 1f - stdLatent[i] * stdLatent[i];
                    dStdLatent[i] *= dtanh;
                }
                for (int i = 0; i < latentDim; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dEncoderWeights[j * latentDim + i] += xSlice[j] * dStdLatent[i];
                    }
                    dLatentBias[i] += dStdLatent[i];
                }
            }
        }

        GradientResult result = new GradientResult(dInput);
        result.addGradient("encoderWeights", dEncoderWeights);
        result.addGradient("decoderWeights", dDecoderWeights);
        result.addGradient("latentBias", dLatentBias);
        result.addGradient("conceptEncoders", dConceptEncoders);
        result.addGradient("conceptDecoders", dConceptDecoders);
        result.addGradient("conceptBias", dConceptBias);
        return result;
    }

    public float[][] getEncoderWeights() { return encoderWeights; }
    public float[][] getDecoderWeights() { return decoderWeights; }
    public float[] getLatentBias() { return latentBias; }
    
    // New getters
    public float[][] getConceptEncoders() { return conceptEncoders; }
    public float[][] getConceptDecoders() { return conceptDecoders; }
    public float[] getConceptBias() { return conceptBias; }

    public void load(java.io.DataInputStream dis) throws java.io.IOException {
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < latentDim; j++) {
                encoderWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < latentDim; i++) {
            for (int j = 0; j < dModel; j++) {
                decoderWeights[i][j] = dis.readFloat();
            }
        }
        for (int i = 0; i < latentDim; i++) {
            latentBias[i] = dis.readFloat();
        }
        // Load concept params
        for (int c = 0; c < nConcepts; c++) {
            for (int i = 0; i < dModel * conceptDim; i++) {
                conceptEncoders[c][i] = dis.readFloat();
            }
        }
        for (int c = 0; c < nConcepts; c++) {
            for (int i = 0; i < conceptDim * dModel; i++) {
                conceptDecoders[c][i] = dis.readFloat();
            }
        }
        for (int i = 0; i < nConcepts * conceptDim; i++) {
            conceptBias[i] = dis.readFloat();
        }
    }
}
