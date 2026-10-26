package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;

import java.io.PrintWriter;

public class AdvancedOnlineLearningTest {

    public static void main(String[] args) throws Exception {
        PrintWriter log = new PrintWriter(System.out, true);
        
        log.println("=== Advanced Online Learning System Test ===");
        log.println();
        
        Tokenizer tokenizer = new Tokenizer(128);
        SimpCoreModel model = new SimpCoreModel(128, 64, 2, 1, 4, 1, 1, 1, 32);
        
        AdvancedOnlineLearningManager manager = new AdvancedOnlineLearningManager(model, 5e-5f, 4, 5, 10);
        
        log.println("Model: D_MODEL=64, N_LAYERS=2, N_HEADS=4");
        log.println("Online LR: 5e-5");
        log.println("Replay interval: 5");
        log.println("EWC update interval: 10");
        log.println();
        
        String[] testData = {
            "hello world this is a test",
            "deep learning is amazing",
            "AI changes everything",
            "the quick brown fox jumps",
            "i love machine learning",
            "natural language processing",
            "transformer architecture",
            "attention is all you need",
            "neural networks are powerful",
            "reinforcement learning is cool"
        };
        
        log.println("[Phase 1] Base Training (30 steps)");
        log.println("----------------------------------");
        TrainingLoop loop = new TrainingLoop(model, tokenizer, new Optimizer(0.001f, 0.01f, 0.9f, Float.MAX_VALUE));
        
        for (int step = 1; step <= 30; step++) {
            String input = testData[(step - 1) % testData.length];
            float loss = loop.trainStep(input, 1);
            
            if (step % 10 == 0) {
                log.printf("  Step %d: loss=%.4f%n", step, loss);
            }
        }
        
        log.println();
        log.println("[Phase 2] Advanced Online Learning (40 steps)");
        log.println("---------------------------------------------");
        
        for (int step = 1; step <= 40; step++) {
            String input = testData[(step - 1) % testData.length];
            int[] tokens = tokenizer.encode(input);
            
            boolean learned = manager.processInteraction(tokens);
            
            if (step % 5 == 0) {
                log.printf("  Step %d: avgLoss=%.4f, recentLoss=%.4f, ewc=%s, memory=%s%n",
                    step, manager.getAverageLoss(), manager.getRecentLoss(),
                    manager.getEwcSystem().isInitialized() ? "ON" : "OFF",
                    manager.getMemoryBuffer().getTotalSize());
            }
        }
        
        log.println();
        log.println("[Phase 3] Generation Test");
        log.println("-------------------------");
        
        String prompt = "hello";
        int[] promptTokens = tokenizer.encode(prompt);
        int[] generatedTokens = model.generate(promptTokens, 10, 0);
        String generated = tokenizer.decode(generatedTokens);
        
        log.printf("  Prompt: \"%s\"%n", prompt);
        log.printf("  Generated: \"%s\"%n", generated);
        
        log.println();
        log.println("[Phase 4] EWC Protection Check");
        log.println("------------------------------");
        
        if (manager.getEwcSystem().isInitialized()) {
            log.printf("  EWC initialized: YES%n");
            log.printf("  Fisher matrix size: %d%n", manager.getEwcSystem().getFisherMatrix().size());
            log.printf("  Samples for Fisher: %d%n", manager.getEwcSystem().getSamplesForFisher());
            log.printf("  Top 5 important params: %s%n", manager.getEwcSystem().getImportantParams(5));
        } else {
            log.println("  EWC initialized: NO");
        }
        
        log.println();
        log.println("[Phase 5] Memory Buffer Statistics");
        log.println("----------------------------------");
        log.println("  " + manager.getMemoryBuffer().getStatistics());
        
        log.println();
        log.println("=== Test Complete ===");
        log.println(manager.getStatistics());
        
        log.close();
    }
}