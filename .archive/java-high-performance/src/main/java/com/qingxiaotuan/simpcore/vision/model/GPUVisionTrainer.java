package com.qingxiaotuan.simpcore.vision.model;

import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;
import java.util.*;

public class GPUVisionTrainer {
    
    private static final Random random = new Random(42);
    private final VisionModel model;
    private final int numClasses;
    private final int imageSize;
    private final int imageChannels;
    
    private float learningRate;
    private int epochs;
    private int batchSize;
    private boolean useGPU;
    
    private List<float[]> trainImages;
    private List<Integer> trainLabels;
    
    private float[] classifierWeights;
    private float[] classifierBias;
    private float[] dClassifierWeights;
    private float[] dClassifierBias;
    
    public GPUVisionTrainer(VisionModel model, int numClasses, int imageSize, int imageChannels) {
        this.model = model;
        this.numClasses = numClasses;
        this.imageSize = imageSize;
        this.imageChannels = imageChannels;
        this.learningRate = 0.01f;
        this.epochs = 10;
        this.batchSize = 4;
        this.useGPU = false;
        
        int embedDim = model.getEmbeddingDim();
        classifierWeights = new float[numClasses * embedDim];
        classifierBias = new float[numClasses];
        dClassifierWeights = new float[numClasses * embedDim];
        dClassifierBias = new float[numClasses];
        
        for (int i = 0; i < numClasses * embedDim; i++) {
            classifierWeights[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }
    }
    
    public void checkGPU() {
        System.out.println("Checking GPU availability...");
        long startTime = System.currentTimeMillis();
        
        try {
            boolean available = GPUMatrixOperations.isGpuAvailable();
            if (available) {
                useGPU = true;
                String deviceInfo = GPUMatrixOperations.getDeviceInfo();
                System.out.println("  ✓ GPU available!");
                System.out.println("  Device: " + deviceInfo);
            } else {
                useGPU = false;
                System.out.println("  ✗ GPU not available, falling back to CPU");
            }
        } catch (Exception e) {
            useGPU = false;
            System.out.println("  ✗ GPU check failed: " + e.getMessage());
            System.out.println("  Falling back to CPU");
        }
        
        long time = System.currentTimeMillis() - startTime;
        System.out.println("  GPU check completed in " + time + " ms");
        System.out.println();
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
                drawShape(image, cx, cy, size, 0.9f, 0.2f, 0.2f, "circle");
                break;
            case 1:
                drawShape(image, cx, cy, size, 0.2f, 0.8f, 0.2f, "square");
                break;
            case 2:
                drawShape(image, cx, cy, size, 0.2f, 0.3f, 0.9f, "triangle");
                break;
            case 3:
                drawShape(image, cx, cy, size, 0.9f, 0.9f, 0.2f, "cross");
                break;
            case 4:
                drawShape(image, cx, cy, size, 0.9f, 0.5f, 0.1f, "diamond");
                break;
            case 5:
                drawShape(image, cx, cy, size, 0.8f, 0.3f, 0.8f, "ring");
                break;
            case 6:
                drawPattern(image, size, 0.5f, 0.5f, 0.5f, "horizontal");
                break;
            case 7:
                drawPattern(image, size, 0.9f, 0.6f, 0.3f, "vertical");
                break;
            case 8:
                drawPattern(image, size, 0.7f, 0.7f, 0.7f, "grid");
                break;
            default:
                drawPattern(image, size, 0.4f, 0.8f, 0.6f, "dots");
                break;
        }
        
        addNoise(image, 0.08f);
        
        return image;
    }
    
    private void drawShape(float[] image, float cx, float cy, float size, float r, float g, float b, String type) {
        float half = size / 2;
        
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);
                boolean draw = false;
                
                switch (type) {
                    case "circle":
                        draw = dist < size;
                        break;
                    case "ring":
                        draw = dist < size && dist > size * 0.55f;
                        break;
                    case "square":
                        draw = Math.abs(dx) < half && Math.abs(dy) < half;
                        break;
                    case "triangle":
                        draw = (cy - y) >= 0 && (cy - y) <= size && Math.abs(dx) <= (1 - (cy - y) / size) * half;
                        break;
                    case "diamond":
                        draw = Math.abs(dx) / half + Math.abs(dy) / half <= 1;
                        break;
                    case "cross":
                        float thickness = size * 0.25f;
                        draw = ((Math.abs(dx) < thickness / 2 && Math.abs(dy) < half) || 
                                (Math.abs(dy) < thickness / 2 && Math.abs(dx) < half));
                        break;
                }
                
                if (draw) {
                    int idx = (y * imageSize + x) * imageChannels;
                    image[idx] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                }
            }
        }
    }
    
    private void drawPattern(float[] image, float size, float r, float g, float b, String type) {
        float offset = (imageSize - size) / 2;
        
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                if (x >= offset && x <= imageSize - offset && y >= offset && y <= imageSize - offset) {
                    boolean draw = false;
                    
                    switch (type) {
                        case "horizontal":
                            int stripeH = Math.max(2, imageSize / 10);
                            draw = ((int) ((y - offset) / stripeH)) % 2 == 0;
                            break;
                        case "vertical":
                            int stripeW = Math.max(2, imageSize / 10);
                            draw = ((int) ((x - offset) / stripeW)) % 2 == 0;
                            break;
                        case "grid":
                            int cell = Math.max(3, imageSize / 8);
                            int cellX = (int) ((x - offset) / cell);
                            int cellY = (int) ((y - offset) / cell);
                            draw = (cellX + cellY) % 2 == 0;
                            break;
                        case "dots":
                            int dotSpacing = Math.max(4, imageSize / 12);
                            int dx = (int) ((x - offset) % dotSpacing) - dotSpacing / 2;
                            int dy = (int) ((y - offset) % dotSpacing) - dotSpacing / 2;
                            draw = dx * dx + dy * dy < dotSpacing * dotSpacing * 0.12f;
                            break;
                    }
                    
                    if (draw) {
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
        System.out.println("VISION MODEL TRAINING (" + (useGPU ? "GPU MODE" : "CPU MODE") + ")");
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
                    updateClassifier(scale);
                    model.updateParameters(learningRate);
                    
                    Arrays.fill(dClassifierWeights, 0);
                    Arrays.fill(dClassifierBias, 0);
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
            
            if (epoch > 0 && losses[epoch] > losses[epoch - 1] * 1.02f) {
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
    
    private void updateClassifier(float scale) {
        for (int j = 0; j < dClassifierWeights.length; j++) {
            classifierWeights[j] -= learningRate * dClassifierWeights[j] * scale;
        }
        for (int j = 0; j < dClassifierBias.length; j++) {
            classifierBias[j] -= learningRate * dClassifierBias[j] * scale;
        }
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
        GPUVisionTrainer trainer = new GPUVisionTrainer(model, numClasses, imgSize, 3);
        
        trainer.checkGPU();
        
        trainer.setLearningRate(0.005f);
        trainer.setEpochs(15);
        trainer.setBatchSize(8);
        
        trainer.generateSyntheticDataset(15);
        
        trainer.train();
    }
}