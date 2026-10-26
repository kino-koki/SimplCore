package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.nn.Transformer;
import com.qingxiaotuan.simpcore.nn.Transformer.TransformerConfig;

public class SCTransformerTest {
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Transformer Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        int vocabSize = 100;
        int seqLen = 5;
        
        System.out.println("[1/4] Creating Transformer model...");
        TransformerConfig config = new TransformerConfig()
            .dModel(16)
            .nLayers(1)
            .nHeads(2)
            .vocabSize(vocabSize)
            .seqLen(seqLen);
        
        Transformer model = new Transformer(config);
        System.out.printf("  Architecture: dModel=%d, nLayers=%d, nHeads=%d, vocab=%d, seqLen=%d%n", 
            config.dModel, config.nLayers, config.nHeads, config.vocabSize, config.seqLen);
        System.out.printf("  Parameters: %.2fM%n", model.getParameterCount() / 1_000_000.0);
        
        int[] inputTokens = {1, 2, 3, 4, 5};
        int[] targetTokens = {2, 3, 4, 5, 6};
        
        System.out.printf("\n[2/4] Input: %s%n", java.util.Arrays.toString(inputTokens));
        System.out.printf("      Target: %s%n", java.util.Arrays.toString(targetTokens));
        
        System.out.println("\n[3/4] Training step...");
        float loss = model.trainStep(inputTokens, targetTokens, 0.01f, 0);
        System.out.printf("  Loss: %.4f%n", loss);
        
        System.out.println("\n[4/4] Testing forward pass...");
        float[] logits = model.forward(inputTokens);
        System.out.printf("  Logits shape: %d (seqLen=%d, vocabSize=%d)%n", 
            logits.length, seqLen, vocabSize);
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Transformer Test Complete!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}