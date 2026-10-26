package com.qingxiaotuan.simpcore.antihallucination;

public class AntiHallucinationTest {
    public static void main(String[] args) {
        System.out.println("=" + "=".repeat(80));
        System.out.println("  ANTI-HALLUCINATION FOUR-LAYER GATE SYSTEM TEST");
        System.out.println("=" + "=".repeat(80));
        
        testLayer1InputStructure();
        testLayer2KnowledgeModuleGate();
        testLayer3InputValidation();
        testLayer4BoundaryConstraint();
        testFullSystemIntegration();
        
        System.out.println("=" + "=".repeat(80));
        System.out.println("  ALL ANTI-HALLUCINATION TESTS COMPLETED");
        System.out.println("=" + "=".repeat(80));
    }
    
    private static void testLayer1InputStructure() {
        System.out.println("\n--- [Layer 1] Input Structure Gate ---");
        
        InputStructureGate gate = new InputStructureGate();
        
        String[] testCases = {
            "帮我",
            "地球是圆的吗？",
            "2024年北京奥运会什么时候举行？",
            "请解释什么是机器学习"
        };
        
        for (String input : testCases) {
            InputStructureGate.InputResult result = gate.validate(input);
            System.out.println("  Input: \"" + input + "\"");
            System.out.println("    Passed: " + result.passed + " | Completeness: " + result.completenessScore + "/100");
            if (!result.passed) {
                System.out.println("    Reason: " + result.rejectionReason);
            }
            System.out.println("    Intent: " + result.intent + " | Entities: " + (result.entities != null ? result.entities.size() : 0));
            System.out.println();
        }
    }
    
    private static void testLayer2KnowledgeModuleGate() {
        System.out.println("\n--- [Layer 2] Knowledge Module Gate ---");
        
        KnowledgeModuleGate gate = new KnowledgeModuleGate();
        
        String[][] testCases = {
            {"admin", "security_cve", "read"},
            {"guest", "security_cve", "read"},
            {"expert", "code_audit", "write"},
            {"analyst", "financial_data", "read"},
            {"guest", "medical_reference", "read"},
            {"admin", "general_knowledge", "write"}
        };
        
        for (String[] tc : testCases) {
            String role = tc[0];
            String module = tc[1];
            String action = tc[2];
            
            KnowledgeModuleGate.AccessDecision result = gate.checkAccess(role, module, action);
            System.out.println("  Role: " + role + " -> Module: " + module + " (" + action + ")");
            System.out.println("    Allowed: " + result.allowed + " | Reason: " + result.reason);
            if (result.highRiskBlock) {
                System.out.println("    [HIGH RISK BLOCKED]");
            }
        }
        
        System.out.println("\n  Testing hot-update access policy:");
        gate.updateAccessPolicy("guest", "general_knowledge", KnowledgeModuleGate.AccessLevel.FULL);
        var updatedResult = gate.checkAccess("guest", "general_knowledge", "write");
        System.out.println("    Guest -> general_knowledge (write) after update: " + updatedResult.allowed);
        
        System.out.println("\n  Audit logs (last 5):");
        var logs = gate.getRecentAuditLogs(5);
        for (var log : logs) {
            System.out.println("    " + log.toString());
        }
    }
    
    private static void testLayer3InputValidation() {
        System.out.println("\n--- [Layer 3] Input Validation Gate ---");
        
        InputValidationGate gate = new InputValidationGate();
        
        String[] inputs = {
            "解释相对论",
            "2+2等于多少",
            "代码审计发现漏洞"
        };
        
        String[] outputs = {
            "相对论可能是爱因斯坦提出的理论，据说改变了时空观",
            "2+2=4",
            "这段代码存在严重的安全漏洞，所有系统都受到威胁"
        };
        
        for (int i = 0; i < inputs.length; i++) {
            float[] logits = new float[1000];
            for (int j = 0; j < 1000; j++) {
                logits[j] = (float) (Math.random() - 0.5) * 2;
            }
            
            InputValidationGate.ValidationResult result = gate.validateOutput(inputs[i], outputs[i], logits);
            System.out.println("  Input: \"" + inputs[i] + "\"");
            System.out.println("  Output: \"" + outputs[i] + "\"");
            System.out.println("    Passed: " + result.passed + " | Intercepted: " + result.intercepted);
            System.out.println("    KL Divergence: " + String.format("%.4f", result.klDivergence));
            System.out.println("    Logical Gap: " + String.format("%.4f", result.logicalGapScore));
            System.out.println("    Uncertainty: " + String.format("%.4f", result.uncertaintyScore));
            
            if (!result.uncertainTerms.isEmpty()) {
                System.out.println("    Uncertain terms:");
                for (var term : result.uncertainTerms) {
                    System.out.println("      - " + term.term + " (" + term.category + ")");
                }
            }
            System.out.println();
        }
        
        System.out.println("  Review queue size: " + gate.getReviewQueueSize());
        
        var reviewItem = gate.processNextReview();
        if (reviewItem != null) {
            System.out.println("  Processed review item: " + reviewItem.id);
            System.out.println("    Priority: " + reviewItem.priority + " | Status: " + reviewItem.status);
            System.out.println("    Note: " + reviewItem.reviewNote);
        }
    }
    
    private static void testLayer4BoundaryConstraint() {
        System.out.println("\n--- [Layer 4] Boundary Constraint Gate ---");
        
        BoundaryConstraintGate gate = new BoundaryConstraintGate();
        
        String[] testCases = {
            "请帮我写一首诗",
            "审计这段代码有没有安全漏洞",
            "分析CVE-2024-12345漏洞的影响",
            "创作一个科幻故事",
            "计算1+1等于多少"
        };
        
        for (String input : testCases) {
            BoundaryConstraintGate.BoundaryResult result = gate.checkBoundary(input, null);
            System.out.println("  Input: \"" + input + "\"");
            System.out.println("    Scene: " + result.sceneType);
            System.out.println("    Passed: " + result.passed + " | Blocked: " + result.blocked);
            System.out.println("    Strict mode required: " + result.strictModeRequired);
            if (result.blocked) {
                System.out.println("    Error:\n" + result.errorMessage);
            }
            System.out.println();
        }
        
        System.out.println("  Strict mode enforcement test:");
        var strictResult = gate.enforceStrictMode("审计代码漏洞", "```python\nprint('hello')\n```");
        System.out.println("    Code validation passed: " + strictResult.passed);
        for (var v : strictResult.validations) {
            System.out.println("    - " + v.validationType + ": " + v.passed + " (" + v.details + ")");
        }
    }
    
    private static void testFullSystemIntegration() {
        System.out.println("\n--- [Full System] Integration Test ---");
        
        AntiHallucinationSystem system = new AntiHallucinationSystem(128);
        
        System.out.println("  Test 1: Open-ended creative request");
        var result1 = system.processInput("请帮我写一首关于春天的诗");
        System.out.println("    Decision: " + result1.finalDecision);
        if (result1.errorMessage != null) {
            System.out.println("    Error: " + result1.errorMessage);
        }
        
        System.out.println("\n  Test 2: Deterministic code audit request");
        var result2 = system.processInput("分析这段Java代码是否存在SQL注入漏洞");
        System.out.println("    Decision: " + result2.finalDecision);
        System.out.println("    Strict mode required: " + result2.layer4Result.strictModeRequired);
        
        System.out.println("\n  Test 3: Knowledge access check (analyst role)");
        var access1 = system.checkKnowledgeAccess("security_cve", "read");
        System.out.println("    Analyst -> security_cve (read): " + access1.allowed);
        
        System.out.println("\n  Test 4: Knowledge access check (guest role - high risk)");
        system.setRole("guest");
        var access2 = system.checkKnowledgeAccess("security_cve", "read");
        System.out.println("    Guest -> security_cve (read): " + access2.allowed + " | " + access2.reason);
        
        System.out.println("\n  Test 5: Output validation chain");
        float[] logits = new float[1000];
        for (int i = 0; i < 1000; i++) {
            logits[i] = (float) (Math.random() - 0.5) * 2;
        }
        var validation = system.validateOutput("解释机器学习", 
            "机器学习可能是一种AI技术，它让计算机从数据中学习", logits);
        System.out.println("    Final decision: " + validation.finalDecision);
        if (validation.layer3Result != null) {
            System.out.println("    KL Divergence: " + String.format("%.4f", validation.layer3Result.klDivergence));
            System.out.println("    Logical Gap: " + String.format("%.4f", validation.layer3Result.logicalGapScore));
            System.out.println("    Uncertainty: " + String.format("%.4f", validation.layer3Result.uncertaintyScore));
        }
    }
}
