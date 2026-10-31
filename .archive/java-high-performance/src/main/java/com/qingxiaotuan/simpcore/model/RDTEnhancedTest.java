package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.orchestration.DynamicOrchestrator;
import com.qingxiaotuan.simpcore.antihallucination.AntiHallucinationSystem;

public class RDTEnhancedTest {
    public static void main(String[] args) {
        System.out.println("=" + "=".repeat(80));
        System.out.println("  RDT MODEL ENHANCEMENT TEST SUITE");
        System.out.println("=" + "=".repeat(80));
        
        testRDTModelEnhanced();
        testDeepWiseLoRA();
        testMLAAttention();
        testOrchestration();
        testAntiHallucination();
        
        System.out.println("=" + "=".repeat(80));
        System.out.println("  ALL TESTS COMPLETED");
        System.out.println("=" + "=".repeat(80));
    }
    
    private static void testRDTModelEnhanced() {
        System.out.println("\n--- [1] RDT Model Enhanced Test ---");
        
        RDTConfig config = RDTConfig.prototype_mini();
        RDTModelEnhanced model = new RDTModelEnhanced(config);
        
        int[] tokens = {1, 2, 3, 4, 5};
        
        System.out.println("  Running forward pass with drift monitoring...");
        float[] logits = model.forward(tokens, 8, true);
        System.out.println("  Output logits: " + logits.length + " dimensions");
        System.out.println("  Drift warning triggered: " + model.isDriftWarningTriggered());
        
        System.out.println("  Running long-term stability test (1000 steps)...");
        RDTModelEnhanced.StabilityReport report = model.runLongTermStabilityTest(tokens, 1000);
        System.out.println("  Stability report:\n" + report.generateReport());
        
        System.out.println("  Parameters: " + model.countParameters());
        System.out.println("  Spectral radius: " + model.getSpectralRadius());
    }
    
    private static void testDeepWiseLoRA() {
        System.out.println("\n--- [2] Deep-Wise LoRA Test ---");
        
        int dim = 128;
        int rank = 8;
        DeepWiseLoRA lora = new DeepWiseLoRA(dim, rank);
        
        float[] x = new float[dim];
        for (int i = 0; i < dim; i++) {
            x[i] = (float) (Math.random() - 0.5);
        }
        
        float[] output = lora.apply(x);
        System.out.println("  Input dim: " + dim + ", Rank: " + rank);
        System.out.println("  Output dim: " + output.length);
        System.out.println("  LoRA parameters: " + lora.countParameters());
        
        float[] grad = new float[dim];
        for (int i = 0; i < dim; i++) {
            grad[i] = (float) (Math.random() - 0.5) * 0.01f;
        }
        
        lora.update(grad);
        lora.step();
        System.out.println("  LoRA weights updated successfully");
    }
    
    private static void testMLAAttention() {
        System.out.println("\n--- [3] MLA Attention Test ---");
        
        int dim = 128;
        int nHeads = 4;
        int kvRank = 16;
        int maxCache = 1024;
        
        MLAAttention mla = new MLAAttention(dim, nHeads, kvRank, maxCache);
        
        float[] x = new float[dim];
        for (int i = 0; i < dim; i++) {
            x[i] = (float) (Math.random() - 0.5);
        }
        
        float[] output = mla.forward(x);
        System.out.println("  Dim: " + dim + ", Heads: " + nHeads + ", KV Rank: " + kvRank);
        System.out.println("  Compression ratio: " + mla.getCompressionRatio() + "x");
        
        long originalSize = mla.getOriginalCacheSize(1024);
        long compressedSize = mla.getCompressedCacheSize(1024);
        System.out.println("  Original KV cache (1024 seq): " + originalSize / 1024 + " KB");
        System.out.println("  Compressed KV cache (1024 seq): " + compressedSize / 1024 + " KB");
        System.out.println("  Compression: " + String.format("%.1fx", (float) originalSize / compressedSize));
        
        System.out.println("  MLA parameters: " + mla.countParameters());
    }
    
    private static void testOrchestration() {
        System.out.println("\n--- [4] Dynamic Orchestration Test ---");
        
        int dim = 128;
        DynamicOrchestrator orchestrator = new DynamicOrchestrator(dim, 100);
        
        float[] mathActivation = new float[dim];
        float[] logicActivation = new float[dim];
        float[] probabilityActivation = new float[dim];
        
        for (int i = 0; i < dim; i++) {
            mathActivation[i] = (float) (Math.random() - 0.5 + 0.3);
            logicActivation[i] = (float) (Math.random() - 0.5 + 0.2);
            probabilityActivation[i] = (float) (Math.random() - 0.5 + 0.25);
        }
        
        orchestrator.registerCapabilityModule("math", mathActivation, "calculation", "数学计算能力");
        orchestrator.registerCapabilityModule("logic", logicActivation, "reasoning", "逻辑推理能力");
        orchestrator.registerCapabilityModule("probability", probabilityActivation, "calculation", "概率演算能力");
        
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++) {
            query[i] = (float) (Math.random() - 0.5);
        }
        
        for (int i = 0; i < 10; i++) {
            float[] result = orchestrator.orchestrate(query, 3, 1.0f);
        }
        
        DynamicOrchestrator.LatencyReport latencyReport = orchestrator.getLatencyReport();
        System.out.println("  " + latencyReport.toString());
    }
    
    private static void testAntiHallucination() {
        System.out.println("\n--- [5] Anti-Hallucination System Test ---");
        
        AntiHallucinationSystem antiHallucination = new AntiHallucinationSystem(128);
        
        String[] testInputs = {
            "帮我",
            "地球是圆的吗？",
            "中国首都是哪里？",
            "2024年北京奥运会什么时候举行？",
            "请解释什么是机器学习，并给出具体的应用案例。"
        };
        
        for (String input : testInputs) {
            AntiHallucinationSystem.ProcessingResult result = antiHallucination.processInput(input);
            System.out.println("  Input: \"" + input + "\"");
            System.out.println("    Decision: " + result.finalDecision);
            if (result.layer1Result != null) {
                System.out.println("    Completeness: " + result.layer1Result.completenessScore + "/100");
            }
            if (result.rejectionReason != null) {
                System.out.println("    Reason: " + result.rejectionReason);
            }
            System.out.println();
        }
        
        String input = "解释相对论";
        String output = "相对论可能是爱因斯坦提出的一种物理学理论，据说它改变了我们对时空的理解。";
        var outputResult = antiHallucination.validateOutput(input, output);
        System.out.println("  Output validation:");
        System.out.println("    Valid: " + outputResult.valid);
        System.out.println("    Hallucination detected: " + outputResult.hallucinationDetected);
        System.out.println("    Hallucination score: " + outputResult.hallucinationScore);
        System.out.println("    Issues: " + String.join(", ", outputResult.issues));
    }
}
