package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;

public class AntiHallucinationSystem {
    private InputStructureGate layer1;
    private KnowledgeModuleGate layer2;
    private InputValidationGate layer3;
    private BoundaryConstraintGate layer4;
    
    private String currentRole;
    
    public AntiHallucinationSystem(int dim) {
        this.layer1 = new InputStructureGate();
        this.layer2 = new KnowledgeModuleGate();
        this.layer3 = new InputValidationGate();
        this.layer4 = new BoundaryConstraintGate();
        this.currentRole = "analyst";
    }
    
    public void setRole(String role) {
        this.currentRole = role;
    }
    
    public InputStructureGate getInputStructureGate() { return layer1; }
    public BoundaryConstraintGate getBoundaryConstraintGate() { return layer4; }
    public InputValidationGate getInputValidationGate() { return layer3; }
    public KnowledgeModuleGate getKnowledgeModuleGate() { return layer2; }
    
    public ProcessingResult processInput(String input) {
        ProcessingResult result = new ProcessingResult();
        
        result.layer1Result = layer1.validate(input);
        if (!result.layer1Result.passed) {
            result.finalDecision = ProcessingResult.Decision.REJECT;
            result.rejectionReason = result.layer1Result.rejectionReason;
            return result;
        }
        
        result.layer4Result = layer4.checkBoundary(input, result.layer1Result);
        if (!result.layer4Result.passed) {
            result.finalDecision = ProcessingResult.Decision.BLOCK;
            result.rejectionReason = result.layer4Result.rejectionReason;
            result.errorMessage = result.layer4Result.errorMessage;
            return result;
        }
        
        result.finalDecision = ProcessingResult.Decision.ACCEPT;
        return result;
    }
    
    public ValidationChainResult validateOutput(String input, String output, float[] logits) {
        ValidationChainResult result = new ValidationChainResult();
        
        result.layer1Result = layer1.validate(input);
        
        result.layer4Result = layer4.checkBoundary(input, result.layer1Result);
        if (!result.layer4Result.passed) {
            result.finalDecision = ValidationChainResult.Decision.BLOCK;
            result.errorMessage = result.layer4Result.errorMessage;
            return result;
        }
        
        if (result.layer4Result.strictModeRequired) {
            var strictResult = layer4.enforceStrictMode(input, output);
            result.strictModeResult = strictResult;
            if (!strictResult.passed) {
                result.finalDecision = ValidationChainResult.Decision.INTERCEPT;
                result.rejectionReason = strictResult.rejectionReason;
                return result;
            }
        }
        
        result.layer3Result = layer3.validateOutput(input, output, logits);
        if (!result.layer3Result.passed) {
            result.finalDecision = ValidationChainResult.Decision.INTERCEPT;
            result.rejectionReason = "验证未通过";
            return result;
        }
        
        result.finalDecision = ValidationChainResult.Decision.APPROVE;
        return result;
    }
    
    public KnowledgeModuleGate.AccessDecision checkKnowledgeAccess(String moduleId, String action) {
        return layer2.checkAccess(currentRole, moduleId, action);
    }
    
    public void updateAccessPolicy(String roleId, String moduleId, KnowledgeModuleGate.AccessLevel level) {
        layer2.updateAccessPolicy(roleId, moduleId, level);
    }
    
    public InputValidationGate.ReviewItem processNextReview() {
        return layer3.processNextReview();
    }
    
    public int getReviewQueueSize() {
        return layer3.getReviewQueueSize();
    }
    
    public static class ProcessingResult {
        public InputStructureGate.InputResult layer1Result;
        public BoundaryConstraintGate.BoundaryResult layer4Result;
        public Decision finalDecision;
        public String rejectionReason;
        public String errorMessage;
        
        public enum Decision {
            ACCEPT, REJECT, BLOCK
        }
    }
    
    public static class ValidationChainResult {
        public InputStructureGate.InputResult layer1Result;
        public BoundaryConstraintGate.BoundaryResult layer4Result;
        public BoundaryConstraintGate.StrictModeEnforcer.StrictResult strictModeResult;
        public InputValidationGate.ValidationResult layer3Result;
        public Decision finalDecision;
        public String rejectionReason;
        public String errorMessage;
        
        public enum Decision {
            APPROVE, INTERCEPT, BLOCK
        }
    }
}
