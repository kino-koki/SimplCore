package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.nn.Transformer;
import com.qingxiaotuan.simpcore.nn.Transformer.TransformerConfig;
import com.qingxiaotuan.simpcore.tokenizer.BPETokenizer;

import java.util.*;

public class SCBasicTraining {
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Basic Transformer Training");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        String trainingText = "abcdefghijklmnopqrstuvwxyz";
        
        System.out.println("[1/5] Loading training data...");
        System.out.printf("  Training text: %s%n", trainingText);
        
        System.out.println("\n[2/5] Training BPE tokenizer...");
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(trainingText, 10);
        int vocabSize = tokenizer.getVocabSize();
        System.out.printf("  Vocabulary size: %d%n", vocabSize);
        
        System.out.println("\n[3/5] Creating Transformer model...");
        TransformerConfig config = new TransformerConfig()
            .dModel(32)
            .nLayers(1)
            .nHeads(4)
            .vocabSize(vocabSize)
            .seqLen(10);
        
        Transformer model = new Transformer(config);
        System.out.printf("  Architecture: dModel=%d, nLayers=%d, nHeads=%d%n", 
            config.dModel, config.nLayers, config.nHeads);
        System.out.printf("  Parameters: %.2fM%n", model.getParameterCount() / 1_000_000.0);
        
        System.out.println("\n[4/5] Training...");
        int[] tokens = tokenizer.encode(trainingText);
        int epochs = 1000;
        float lr = 0.1f;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0;
            
            for (int i = 0; i < tokens.length - 2; i++) {
                int[] inputTokens = new int[]{tokens[i]};
                int[] targetTokens = new int[]{tokens[i + 1]};
                
                float loss = model.trainStep(inputTokens, targetTokens, lr, 0);
                totalLoss += loss;
            }
            
            if (epoch % 200 == 0) {
                System.out.printf("  Epoch %d/%d - Loss: %.4f%n", epoch, epochs, totalLoss / (tokens.length - 2));
            }
        }
        
        System.out.println("\n[5/5] Testing prediction...");
        int correct = 0;
        for (int i = 0; i < tokens.length - 2; i++) {
            int[] inputTokens = new int[]{tokens[i]};
            int predicted = model.predictNext(inputTokens);
            
            if (predicted == tokens[i + 1]) {
                correct++;
            }
        }
        
        System.out.printf("  Prediction accuracy: %d/%d (%.2f%%)%n", 
            correct, tokens.length - 2, (double) correct / (tokens.length - 2) * 100);
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                  Basic Training Complete!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}