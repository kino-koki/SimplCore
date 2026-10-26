package com.qingxiaotuan.simpcore.vision.model;

import java.util.*;

public class VisionTrainer {
    
    private static final Random random = new Random(42);
    private final VisionModel model;
    private final int numClasses;
    private final int imageSize;
    private final int imageChannels;
    
    private float learningRate;
    private int epochs;
    private int batchSize;
    
    private List<float[]> trainImages;
    private List<Integer> trainLabels;
    
    private float[] classifierWeights;
    private float[] classifierBias;
    private float[] dClassifierWeights;
    private float[] dClassifierBias;
    
    public VisionTrainer(VisionModel model, int numClasses, int imageSize, int imageChannels) {
        this.model = model;
        this.numClasses = numClasses;
        this.imageSize = imageSize;
        this.imageChannels = imageChannels;
        this.learningRate = 0.01f;
        this.epochs = 10;
        this.batchSize = 4;
        
        int embedDim = model.getEmbeddingDim();
        classifierWeights = new float[numClasses * embedDim];
        classifierBias = new float[numClasses];
        dClassifierWeights = new float[numClasses * embedDim];
        dClassifierBias = new float[numClasses];
        
        for (int i = 0; i < numClasses * embedDim; i++) {
            classifierWeights[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }
    }
    
    public void generateSyntheticDataset(int samplesPerClass) {
        trainImages = new ArrayList<>();
        trainLabels = new ArrayList<>();
        
        System.out.println("Generating synthetic dataset...");
        System.out.println("  Classes: " + numClasses);
        System.out.println("  Samples per class: " + samplesPerClass);
        System.out.println("  Total samples: " + (numClasses * samplesPerClass));
        System.out.println("  Image size: " + imageChannels + "x" + imageSize + "x" + imageSize);
        
        for (int cls = 0; cls < numClasses; cls++) {
            for (int i = 0; i < samplesPerClass; i++) {
                float[] image = generateClassImage(cls, i);
                trainImages.add(image);
                trainLabels.add(cls);
            }
        }
        
        shuffleDataset();
        System.out.println("  Dataset generated successfully!");
        System.out.println();
    }
    
    private float[] generateClassImage(int classIdx, int sampleIdx) {
        float[] image = new float[imageChannels * imageSize * imageSize];
        Arrays.fill(image, -0.8f);
        
        float jitterX = (random.nextFloat() - 0.5f) * imageSize * 0.1f;
        float jitterY = (random.nextFloat() - 0.5f) * imageSize * 0.1f;
        float scaleJitter = 1.0f + (random.nextFloat() - 0.5f) * 0.2f;
        
        float cx = imageSize * 0.5f + jitterX;
        float cy = imageSize * 0.5f + jitterY;
        float size = imageSize * 0.3f * scaleJitter;
        
        switch (classIdx % 10) {
            case 0:
                drawCircle(image, cx, cy, size, 0.9f, 0.2f, 0.2f);
                break;
            case 1:
                drawSquare(image, cx, cy, size, 0.2f, 0.8f, 0.2f);
                break;
            case 2:
                drawTriangle(image, cx, cy, size, 0.2f, 0.3f, 0.9f);
                break;
            case 3:
                drawCross(image, cx, cy, size, 0.9f, 0.9f, 0.2f);
                break;
            case 4:
                drawDiamond(image, cx, cy, size, 0.9f, 0.5f, 0.1f);
                break;
            case 5:
                drawRing(image, cx, cy, size, size * 0.55f, 0.8f, 0.3f, 0.8f);
                break;
            case 6:
                drawHorizontalStripes(image, size, 0.5f, 0.5f, 0.5f);
                break;
            case 7:
                drawVerticalStripes(image, size, 0.9f, 0.6f, 0.3f);
                break;
            case 8:
                drawGrid(image, size, 0.7f, 0.7f, 0.7f);
                break;
            default:
                drawDotPattern(image, size, 0.4f, 0.8f, 0.6f);
                break;
        }
        
        addNoise(image, 0.08f);
        
        return image;
    }
    
    private void drawCircle(float[] image, float cx, float cy, float radius, float r, float g, float b) {
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist < radius) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawRing(float[] image, float cx, float cy, float outerR, float innerR, float r, float g, float b) {
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                if (dist < outerR && dist > innerR) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawSquare(float[] image, float cx, float cy, float size, float r, float g, float b) {
        float half = size / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= cx - half && x <= cx + half && y >= cy - half && y <= cy + half) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawTriangle(float[] image, float cx, float cy, float size, float r, float g, float b) {
        float half = size / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = x - cx;
                float dy = cy - y;
                if (dy >= 0 && dy <= size && Math.abs(dx) <= (1 - dy / size) * half) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawDiamond(float[] image, float cx, float cy, float size, float r, float g, float b) {
        float half = size / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = Math.abs(x - cx);
                float dy = Math.abs(y - cy);
                if (dx / half + dy / half <= 1) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawCross(float[] image, float cx, float cy, float size, float r, float g, float b) {
        float thickness = size * 0.25f;
        float half = size / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = Math.abs(x - cx);
                float dy = Math.abs(y - cy);
                if ((dx < thickness / 2 && dy < half) || (dy < thickness / 2 && dx < half)) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawHorizontalStripes(float[] image, float size, float r, float g, float b) {
        int stripeHeight = Math.max(2, (int) (imageSize / 10.0));
        float offset = (imageSize - size) / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= offset && x <= imageSize - offset && y >= offset && y <= imageSize - offset) {
                    int stripeIdx = (int) ((y - offset) / stripeHeight);
                    if (stripeIdx % 2 == 0) {
                        int idx = (y * imageSize + x) * imageChannels;
                        image[idx] = r;
                        image[idx + 1] = g;
                        image[idx + 2] = b;
                    }
                }
            }
        }
    }
    
    private void drawVerticalStripes(float[] image, float size, float r, float g, float b) {
        int stripeWidth = Math.max(2, (int) (imageSize / 10.0));
        float offset = (imageSize - size) / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= offset && x <= imageSize - offset && y >= offset && y <= imageSize - offset) {
                    int stripeIdx = (int) ((x - offset) / stripeWidth);
                    if (stripeIdx % 2 == 0) {
                        int idx = (y * imageSize + x) * imageChannels;
                        image[idx] = r;
                        image[idx + 1] = g;
                        image[idx + 2] = b;
                    }
                }
            }
        }
    }
    
    private void drawGrid(float[] image, float size, float r, float g, float b) {
        int cellSize = Math.max(3, (int) (imageSize / 8.0));
        float offset = (imageSize - size) / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= offset && x <= imageSize - offset && y >= offset && y <= imageSize - offset) {
                    int cellX = (int) ((x - offset) / cellSize);
                    int cellY = (int) ((y - offset) / cellSize);
                    if ((cellX + cellY) % 2 == 0) {
                        int idx = (y * imageSize + x) * imageChannels;
                        image[idx] = r;
                        image[idx + 1] = g;
                        image[idx + 2] = b;
                    }
                }
            }
        }
    }
    
    private void drawDotPattern(float[] image, float size, float r, float g, float b) {
        int dotSpacing = Math.max(4, (int) (imageSize / 12.0));
        float dotRadius = dotSpacing * 0.35f;
        float offset = (imageSize - size) / 2;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= offset && x <= imageSize - offset && y >= offset && y <= imageSize - offset) {
                    int dx = (int) ((x - offset) % dotSpacing) - dotSpacing / 2;
                    int dy = (int) ((y - offset) % dotSpacing) - dotSpacing / 2;
                    if (dx * dx + dy * dy < dotRadius * dotRadius) {
                        int idx = (y * imageSize + x) * imageChannels;
                        image[idx] = r;
                        image[idx + 1] = g;
                        image[idx + 2] = b;
                    }
                }
            }
        }
    }
    
    private void addNoise(float[] image, float amount) {
        for (int i = 0; i < image.length; i++) {
            image[i] += (random.nextFloat() - 0.5f) * 2 * amount;
            image[i] = Math.max(-1.0f, Math.min(1.0f, image[i]));
        }
    }
    
    private void shuffleDataset() {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < trainImages.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        List<float[]> newImages = new ArrayList<>();
        List<Integer> newLabels = new ArrayList<>();
        for (int idx : indices) {
            newImages.add(trainImages.get(idx));
            newLabels.add(trainLabels.get(idx));
        }
        trainImages = newImages;
        trainLabels = newLabels;
    }
    
    public void train() {
        System.out.println("=======================================================================");
        System.out.println("VISION MODEL TRAINING");
        System.out.println("=======================================================================");
        System.out.println("Model parameters: " + model.getTotalParameters());
        System.out.println("Classifier params: " + (classifierWeights.length + classifierBias.length));
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Epochs: " + epochs);
        System.out.println("Batch size: " + batchSize);
        System.out.println("Training samples: " + trainImages.size());
        System.out.println("=======================================================================");
        System.out.println();
        
        float[] losses = new float[epochs];
        float[] accuracies = new float[epochs];
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            
            float totalLoss = 0;
            int correct = 0;
            int total = 0;
            int numBatches = 0;
            
            Arrays.fill(dClassifierWeights, 0);
            Arrays.fill(dClassifierBias, 0);
            
            for (int i = 0; i < trainImages.size(); i++) {
                float[] image = trainImages.get(i);
                int label = trainLabels.get(i);
                
                float[] features = model.forward(image);
                
                float[] logits = computeLogits(features);
                
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < numClasses; c++) {
                    maxLogit = Math.max(maxLogit, logits[c]);
                }
                
                float sumExp = 0;
                for (int c = 0; c < numClasses; c++) {
                    sumExp += Math.exp(logits[c] - maxLogit);
                }
                
                float loss = -logits[label] + maxLogit + (float) Math.log(sumExp);
                totalLoss += loss;
                
                int predicted = 0;
                float maxPred = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < numClasses; c++) {
                    if (logits[c] > maxPred) {
                        maxPred = logits[c];
                        predicted = c;
                    }
                }
                if (predicted == label) correct++;
                total++;
                
                float[] gradFeatures = new float[features.length];
                for (int c = 0; c < numClasses; c++) {
                    float prob = (float) Math.exp(logits[c] - maxLogit) / sumExp;
                    float gradLogit = prob - (c == label ? 1 : 0);
                    
                    dClassifierBias[c] += gradLogit;
                    
                    for (int f = 0; f < features.length; f++) {
                        dClassifierWeights[c * features.length + f] += gradLogit * features[f];
                        gradFeatures[f] += gradLogit * classifierWeights[c * features.length + f];
                    }
                }
                
                model.backward(gradFeatures);
                
                if ((i + 1) % batchSize == 0 || i == trainImages.size() - 1) {
                    float scale = 1.0f / batchSize;
                    for (int j = 0; j < dClassifierWeights.length; j++) {
                        classifierWeights[j] -= learningRate * dClassifierWeights[j] * scale;
                    }
                    for (int j = 0; j < dClassifierBias.length; j++) {
                        classifierBias[j] -= learningRate * dClassifierBias[j] * scale;
                    }
                    
                    model.updateParameters(learningRate);
                    
                    Arrays.fill(dClassifierWeights, 0);
                    Arrays.fill(dClassifierBias, 0);
                    
                    numBatches++;
                }
            }
            
            totalLoss /= total;
            losses[epoch] = totalLoss;
            accuracies[epoch] = (float) correct / total * 100;
            
            long epochTime = System.currentTimeMillis() - epochStartTime;
            
            System.out.println(String.format(
                "Epoch %3d/%d | Loss: %.4f | Acc: %.1f%% | Time: %d ms",
                epoch + 1, epochs, totalLoss, accuracies[epoch], epochTime
            ));
            
            if (epoch > 0 && losses[epoch] > losses[epoch - 1]) {
                learningRate *= 0.95f;
                System.out.println("  -> Reducing learning rate to " + String.format("%.6f", learningRate));
            }
            
            shuffleDataset();
        }
        
        System.out.println();
        System.out.println("=======================================================================");
        System.out.println("TRAINING COMPLETE");
        System.out.println("=======================================================================");
        System.out.println("Initial loss: " + String.format("%.4f", losses[0]));
        System.out.println("Final loss: " + String.format("%.4f", losses[epochs - 1]));
        System.out.println("Initial accuracy: " + String.format("%.1f", accuracies[0]) + "%");
        System.out.println("Final accuracy: " + String.format("%.1f", accuracies[epochs - 1]) + "%");
        if (losses[0] > 0) {
            System.out.println("Loss improvement: " + 
                String.format("%.2f", (losses[0] - losses[epochs - 1]) / losses[0] * 100) + "%");
        }
        System.out.println("=======================================================================");
    }
    
    private float[] computeLogits(float[] features) {
        float[] logits = new float[numClasses];
        int embedDim = features.length;
        
        for (int c = 0; c < numClasses; c++) {
            float sum = classifierBias[c];
            for (int f = 0; f < embedDim; f++) {
                sum += classifierWeights[c * embedDim + f] * features[f];
            }
            logits[c] = sum;
        }
        
        return logits;
    }
    
    public void setLearningRate(float lr) { this.learningRate = lr; }
    public void setEpochs(int epochs) { this.epochs = epochs; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    
    public static void main(String[] args) {
        int imgSize = 64;
        int embedDim = 128;
        int numClasses = 10;
        
        VisionModel model = new VisionModel(3, imgSize, imgSize, embedDim);
        VisionTrainer trainer = new VisionTrainer(model, numClasses, imgSize, 3);
        
        trainer.setLearningRate(0.005f);
        trainer.setEpochs(15);
        trainer.setBatchSize(8);
        
        trainer.generateSyntheticDataset(15);
        
        trainer.train();
    }
}