package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;
import java.util.regex.*;

public class InputValidationGate {
    private static final float KL_DIVERGENCE_THRESHOLD = 0.5f;
    private static final float LOGICAL_GAP_THRESHOLD = 0.3f;
    private static final float UNCERTAINTY_THRESHOLD = 0.4f;
    
    private Set<String> absoluteTerms;
    private Set<String> overgeneralizationTerms;
    private Set<String> unfoundedTerms;
    private Map<String, float[]> trainingDataStats;
    private Queue<ReviewItem> reviewQueue;
    
    public InputValidationGate() {
        this.absoluteTerms = new HashSet<>(Arrays.asList(
            "所有", "全部", "完全", "绝对", "必须", "一定", "肯定",
            "毫无", "从未", "永远", "始终", "必然", "必定", "绝不可能"
        ));
        
        this.overgeneralizationTerms = new HashSet<>(Arrays.asList(
            "普遍", "大多数", "通常", "一般来说", "基本上", "大体上",
            "总体而言", "整体来看", "往往", "常常", "总是"
        ));
        
        this.unfoundedTerms = new HashSet<>(Arrays.asList(
            "可能", "也许", "大概", "据说", "据称", "有人说",
            "应该是", "似乎", "好像", "仿佛", "感觉", "认为"
        ));
        
        this.trainingDataStats = initTrainingDataStats();
        this.reviewQueue = new LinkedList<>();
    }
    
    private Map<String, float[]> initTrainingDataStats() {
        Map<String, float[]> stats = new HashMap<>();
        
        float[] commonDistribution = new float[1000];
        for (int i = 0; i < 1000; i++) {
            commonDistribution[i] = (float) Math.exp(-i * 0.01);
        }
        normalize(commonDistribution);
        stats.put("common", commonDistribution);
        
        float[] factualDistribution = new float[1000];
        for (int i = 0; i < 1000; i++) {
            factualDistribution[i] = (float) Math.exp(-i * 0.005);
        }
        normalize(factualDistribution);
        stats.put("factual", factualDistribution);
        
        return stats;
    }
    
    public ValidationResult validateOutput(String input, String output, float[] outputLogits) {
        ValidationResult result = new ValidationResult();
        result.input = input;
        result.output = output;
        result.checks = new ArrayList<>();
        
        KLCheckResult klResult = checkFactConsistency(outputLogits);
        result.klDivergence = klResult.klDivergence;
        result.factConsistent = klResult.passed;
        result.checks.add(new CheckItem("KL散度校验", klResult.passed, 
            String.format("KL散度: %.4f (阈值: %.2f)", klResult.klDivergence, KL_DIVERGENCE_THRESHOLD)));
        
        LogicalGapResult gapResult = checkLogicalGap(input, output);
        result.logicalGapScore = gapResult.gapScore;
        result.logicallyComplete = gapResult.passed;
        result.checks.add(new CheckItem("逻辑断层检测", gapResult.passed, 
            String.format("逻辑连贯性: %.4f (阈值: %.2f)", gapResult.gapScore, LOGICAL_GAP_THRESHOLD)));
        
        UncertaintyResult uncertaintyResult = checkInferenceRationality(output);
        result.uncertaintyScore = uncertaintyResult.score;
        result.inferenceRational = uncertaintyResult.passed;
        result.uncertainTerms = uncertaintyResult.uncertainTerms;
        result.checks.add(new CheckItem("推断合理性审查", uncertaintyResult.passed, 
            String.format("不确定性评分: %.4f (阈值: %.2f)", uncertaintyResult.score, UNCERTAINTY_THRESHOLD)));
        
        result.passed = klResult.passed && gapResult.passed && uncertaintyResult.passed;
        
        if (!result.passed) {
            result.intercepted = true;
            triggerReview(result);
        }
        
        return result;
    }
    
    private KLCheckResult checkFactConsistency(float[] logits) {
        KLCheckResult result = new KLCheckResult();
        
        if (logits == null || logits.length == 0) {
            result.klDivergence = 0.0f;
            result.passed = true;
            return result;
        }
        
        float[] targetDistribution = trainingDataStats.get("factual");
        float[] outputDistribution = softmax(logits);
        
        float kl = computeKLDivergence(outputDistribution, targetDistribution);
        result.klDivergence = kl;
        result.passed = kl < KL_DIVERGENCE_THRESHOLD;
        
        return result;
    }
    
    private LogicalGapResult checkLogicalGap(String input, String output) {
        LogicalGapResult result = new LogicalGapResult();
        
        Set<String> inputKeywords = extractKeywords(input);
        Set<String> outputKeywords = extractKeywords(output);
        
        if (inputKeywords.isEmpty()) {
            result.gapScore = 1.0f;
            result.passed = true;
            return result;
        }
        
        int intersection = 0;
        for (String keyword : inputKeywords) {
            if (outputKeywords.contains(keyword)) {
                intersection++;
            } else {
                for (String outputKeyword : outputKeywords) {
                    if (output.contains(keyword) || keyword.contains(outputKeyword)) {
                        intersection++;
                        break;
                    }
                }
            }
        }
        
        float coherence = (float) intersection / inputKeywords.size();
        result.gapScore = coherence;
        result.passed = coherence > LOGICAL_GAP_THRESHOLD;
        
        return result;
    }
    
    private UncertaintyResult checkInferenceRationality(String output) {
        UncertaintyResult result = new UncertaintyResult();
        result.uncertainTerms = new ArrayList<>();
        
        int absoluteCount = 0;
        int overgenCount = 0;
        int unfoundedCount = 0;
        
        for (String term : absoluteTerms) {
            if (output.contains(term)) {
                absoluteCount++;
                result.uncertainTerms.add(new UncertainTerm(term, "绝对化表述"));
            }
        }
        
        for (String term : overgeneralizationTerms) {
            if (output.contains(term)) {
                overgenCount++;
                result.uncertainTerms.add(new UncertainTerm(term, "过度概括"));
            }
        }
        
        for (String term : unfoundedTerms) {
            if (output.contains(term)) {
                unfoundedCount++;
                result.uncertainTerms.add(new UncertainTerm(term, "缺乏依据"));
            }
        }
        
        int totalTerms = countWords(output);
        float uncertainty = (float) (absoluteCount + overgenCount + unfoundedCount) / Math.max(totalTerms, 1);
        
        result.score = uncertainty;
        result.passed = uncertainty < UNCERTAINTY_THRESHOLD;
        
        return result;
    }
    
    private void triggerReview(ValidationResult result) {
        ReviewItem item = new ReviewItem();
        item.id = "REV-" + System.currentTimeMillis();
        item.timestamp = java.time.LocalDateTime.now();
        item.input = result.input;
        item.output = result.output;
        item.validationResult = result;
        item.priority = calculatePriority(result);
        item.status = ReviewStatus.PENDING;
        
        reviewQueue.add(item);
    }
    
    private ReviewPriority calculatePriority(ValidationResult result) {
        int failedChecks = 0;
        if (!result.factConsistent) failedChecks++;
        if (!result.logicallyComplete) failedChecks++;
        if (!result.inferenceRational) failedChecks++;
        
        if (failedChecks >= 3) {
            return ReviewPriority.HIGH;
        } else if (failedChecks >= 2) {
            return ReviewPriority.MEDIUM;
        } else {
            return ReviewPriority.LOW;
        }
    }
    
    public ReviewItem processNextReview() {
        if (reviewQueue.isEmpty()) {
            return null;
        }
        
        ReviewItem item = reviewQueue.poll();
        
        if (item.priority == ReviewPriority.HIGH) {
            item.status = ReviewStatus.IN_PROGRESS;
            
            if (item.validationResult.klDivergence > KL_DIVERGENCE_THRESHOLD * 2) {
                item.status = ReviewStatus.REJECTED;
                item.reviewNote = "KL散度过高，事实严重偏离";
            } else {
                item.status = ReviewStatus.APPROVED;
                item.reviewNote = "复核通过，建议附带不确定性说明";
            }
        } else {
            item.status = ReviewStatus.AUTO_APPROVED;
            item.reviewNote = "低优先级自动复核通过";
        }
        
        return item;
    }
    
    public int getReviewQueueSize() {
        return reviewQueue.size();
    }
    
    private float[] softmax(float[] logits) {
        float[] result = new float[logits.length];
        float max = Float.NEGATIVE_INFINITY;
        for (float l : logits) {
            if (l > max) max = l;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }
        
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    private void normalize(float[] distribution) {
        float sum = 0.0f;
        for (float v : distribution) {
            sum += v;
        }
        for (int i = 0; i < distribution.length; i++) {
            distribution[i] /= sum;
        }
    }
    
    private float computeKLDivergence(float[] p, float[] q) {
        int minLen = Math.min(p.length, q.length);
        float kl = 0.0f;
        
        for (int i = 0; i < minLen; i++) {
            if (p[i] > 0 && q[i] > 0) {
                kl += p[i] * (float) Math.log(p[i] / q[i]);
            }
        }
        
        return kl;
    }
    
    private Set<String> extractKeywords(String text) {
        Set<String> keywords = new HashSet<>();
        
        Pattern pattern = Pattern.compile("[\\u4e00-\\u9fa5]+|[a-zA-Z]{2,}");
        Matcher matcher = pattern.matcher(text);
        
        while (matcher.find()) {
            String match = matcher.group();
            keywords.add(match);
            
            if (match.length() >= 2) {
                for (int i = 0; i < match.length(); i++) {
                    keywords.add(String.valueOf(match.charAt(i)));
                }
            }
        }
        
        return keywords;
    }
    
    private int countWords(String text) {
        int count = 0;
        boolean inWord = false;
        for (char c : text.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                if (!inWord) {
                    count++;
                    inWord = true;
                }
            } else {
                inWord = false;
            }
        }
        return count;
    }
    
    public static class ValidationResult {
        public String input;
        public String output;
        public boolean passed;
        public boolean intercepted;
        
        public float klDivergence;
        public boolean factConsistent;
        
        public float logicalGapScore;
        public boolean logicallyComplete;
        
        public float uncertaintyScore;
        public boolean inferenceRational;
        public List<UncertainTerm> uncertainTerms;
        
        public List<CheckItem> checks;
    }
    
    public static class CheckItem {
        public String name;
        public boolean passed;
        public String details;
        
        public CheckItem(String name, boolean passed, String details) {
            this.name = name;
            this.passed = passed;
            this.details = details;
        }
    }
    
    public static class KLCheckResult {
        public float klDivergence;
        public boolean passed;
    }
    
    public static class LogicalGapResult {
        public float gapScore;
        public boolean passed;
    }
    
    public static class UncertaintyResult {
        public float score;
        public boolean passed;
        public List<UncertainTerm> uncertainTerms;
    }
    
    public static class UncertainTerm {
        public String term;
        public String category;
        
        public UncertainTerm(String term, String category) {
            this.term = term;
            this.category = category;
        }
    }
    
    public static class ReviewItem {
        public String id;
        public java.time.LocalDateTime timestamp;
        public String input;
        public String output;
        public ValidationResult validationResult;
        public ReviewPriority priority;
        public ReviewStatus status;
        public String reviewNote;
    }
    
    public enum ReviewPriority {
        HIGH, MEDIUM, LOW
    }
    
    public enum ReviewStatus {
        PENDING, IN_PROGRESS, APPROVED, REJECTED, AUTO_APPROVED
    }
}
