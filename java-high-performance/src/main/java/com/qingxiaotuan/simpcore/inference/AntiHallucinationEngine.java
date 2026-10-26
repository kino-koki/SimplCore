package com.qingxiaotuan.simpcore.inference;

import com.qingxiaotuan.simpcore.knowledge.KnowledgeBaseManager;

import java.util.*;
import java.util.concurrent.*;
import java.util.regex.*;

public class AntiHallucinationEngine {
    
    public static class VerificationResult {
        public final boolean isVerified;
        public final float confidence;
        public final List<String> verifiedClaims;
        public final List<String> unverifiedClaims;
        public final List<Citation> citations;
        public final String correctedText;
        public final float hallucinationRisk;
        
        public VerificationResult(boolean isVerified, float confidence, List<String> verifiedClaims,
                                   List<String> unverifiedClaims, List<Citation> citations,
                                   String correctedText, float hallucinationRisk) {
            this.isVerified = isVerified;
            this.confidence = confidence;
            this.verifiedClaims = verifiedClaims;
            this.unverifiedClaims = unverifiedClaims;
            this.citations = citations;
            this.correctedText = correctedText;
            this.hallucinationRisk = hallucinationRisk;
        }
    }
    
    public static class Citation {
        public final String claim;
        public final String source;
        public final float confidence;
        public final long timestamp;
        
        public Citation(String claim, String source, float confidence, long timestamp) {
            this.claim = claim;
            this.source = source;
            this.confidence = confidence;
            this.timestamp = timestamp;
        }
    }
    
    public static class FactClaim {
        public final String claim;
        public final float confidence;
        public final boolean isVerifiable;
        public final String category;
        
        public FactClaim(String claim, float confidence, boolean isVerifiable, String category) {
            this.claim = claim;
            this.confidence = confidence;
            this.isVerifiable = isVerifiable;
            this.category = category;
        }
    }
    
    private final KnowledgeBaseManager knowledgeBase;
    private final ExecutorService executor;
    private final Set<String> factPatterns;
    private final Map<String, Float> categoryConfidenceThresholds;
    private final int maxVerificationAttempts;
    private final float verificationThreshold;
    private final Pattern claimPattern;
    
    public AntiHallucinationEngine(KnowledgeBaseManager knowledgeBase) {
        this.knowledgeBase = knowledgeBase;
        this.executor = Executors.newFixedThreadPool(4);
        this.maxVerificationAttempts = 3;
        this.verificationThreshold = 0.95f;
        
        this.factPatterns = new HashSet<>(Arrays.asList(
            "\\d+\\s*(year|years|century|BC|AD)",
            "\\d+\\s*(million|billion|trillion)",
            "\\d+\\s*(km|m|cm|mm|meter|metre)",
            "\\d+\\s*(kg|g|ton|pound|lb)",
            "\\d+\\s*(\\$|yuan|dollar|RMB|USD|EUR)",
            "\\d+\\s*(percent|%)",
            "\\d+\\s*(degree|°)",
            "was\\s+born\\s+in",
            "died\\s+in",
            "founded\\s+in",
            "located\\s+in",
            "capital\\s+of",
            "population\\s+of",
            "height\\s+of",
            "weight\\s+of",
            "temperature\\s+of",
            "speed\\s+of",
            "distance\\s+between",
            "discovered\\s+in",
            "invented\\s+in",
            "published\\s+in",
            "won\\s+in",
            "established\\s+in",
            "created\\s+in",
            "developed\\s+in",
            "introduced\\s+in",
            "released\\s+in",
            "produced\\s+in",
            "built\\s+in",
            "completed\\s+in",
            "occurred\\s+in",
            "happened\\s+in",
            "began\\s+in",
            "ended\\s+in",
            "started\\s+in",
            "concluded\\s+in",
            "named\\s+after",
            "known\\s+for",
            "famous\\s+for",
            "recognized\\s+as",
            "considered\\s+to\\s+be",
            "believed\\s+to\\s+be",
            "thought\\s+to\\s+be",
            "estimated\\s+at",
            "approximately\\s+\\d+",
            "around\\s+\\d+",
            "about\\s+\\d+",
            "between\\s+\\d+\\s+and\\s+\\d+",
            "more\\s+than\\s+\\d+",
            "less\\s+than\\s+\\d+",
            "over\\s+\\d+",
            "under\\s+\\d+",
            "at\\s+least\\s+\\d+",
            "at\\s+most\\s+\\d+",
            "first\\s+\\w+\\s+in",
            "largest\\s+\\w+\\s+in",
            "smallest\\s+\\w+\\s+in",
            "longest\\s+\\w+\\s+in",
            "shortest\\s+\\w+\\s+in",
            "highest\\s+\\w+\\s+in",
            "lowest\\s+\\w+\\s+in",
            "fastest\\s+\\w+\\s+in",
            "slowest\\s+\\w+\\s+in",
            "oldest\\s+\\w+\\s+in",
            "newest\\s+\\w+\\s+in",
            "biggest\\s+\\w+\\s+in",
            "best\\s+\\w+\\s+in",
            "worst\\s+\\w+\\s+in",
            "\\d+\\s+(second|minute|hour|day|week|month|year)",
            "\\d+\\s*(times|x)",
            "\\d+\\s*(fold|times\\s+larger|times\\s+smaller)",
            "\\d+\\s*(percent|percentage)\\s+(increase|decrease)",
            "\\d+\\s*(point|points)\\s+(higher|lower)"
        ));
        
        this.categoryConfidenceThresholds = new HashMap<>();
        categoryConfidenceThresholds.put("geography", 0.97f);
        categoryConfidenceThresholds.put("history", 0.95f);
        categoryConfidenceThresholds.put("science", 0.98f);
        categoryConfidenceThresholds.put("technology", 0.96f);
        categoryConfidenceThresholds.put("math", 0.99f);
        categoryConfidenceThresholds.put("medicine", 0.97f);
        categoryConfidenceThresholds.put("finance", 0.96f);
        categoryConfidenceThresholds.put("politics", 0.94f);
        categoryConfidenceThresholds.put("culture", 0.92f);
        categoryConfidenceThresholds.put("art", 0.90f);
        
        this.claimPattern = Pattern.compile(
            "([A-Z][^.!?]*?(?:\\d+[^.!?]*?|was\\s+|is\\s+|are\\s+|were\\s+|be\\s+|has\\s+|have\\s+|had\\s+|will\\s+|would\\s+|could\\s+|may\\s+|might\\s+)[^.!?]*?[.!?])",
            Pattern.CASE_INSENSITIVE
        );
    }
    
    public VerificationResult verify(String text) {
        List<String> verifiedClaims = new ArrayList<>();
        List<String> unverifiedClaims = new ArrayList<>();
        List<Citation> citations = new ArrayList<>();
        
        List<FactClaim> claims = extractFactClaims(text);
        float totalConfidence = 0;
        int verifiedCount = 0;
        
        for (FactClaim claim : claims) {
            if (!claim.isVerifiable) {
                unverifiedClaims.add(claim.claim);
                totalConfidence += claim.confidence * 0.5f;
                continue;
            }
            
            float verificationScore = verifyClaim(claim);
            
            if (verificationScore >= verificationThreshold) {
                verifiedClaims.add(claim.claim);
                citations.add(new Citation(claim.claim, findSource(claim.claim), verificationScore, System.currentTimeMillis()));
                verifiedCount++;
                totalConfidence += verificationScore;
            } else {
                unverifiedClaims.add(claim.claim);
                totalConfidence += verificationScore * 0.3f;
            }
        }
        
        float avgConfidence = claims.isEmpty() ? 1.0f : totalConfidence / claims.size();
        float hallucinationRisk = calculateHallucinationRisk(claims, verifiedCount);
        String correctedText = correctHallucinations(text, claims, citations);
        
        return new VerificationResult(
            avgConfidence >= verificationThreshold,
            avgConfidence,
            verifiedClaims,
            unverifiedClaims,
            citations,
            correctedText,
            hallucinationRisk
        );
    }
    
    public VerificationResult verifyWithRetry(String text) {
        VerificationResult result = verify(text);
        
        for (int attempt = 1; attempt < maxVerificationAttempts && result.hallucinationRisk > 0.05f; attempt++) {
            String refinedText = refineGeneration(result);
            result = verify(refinedText);
        }
        
        return result;
    }
    
    private List<FactClaim> extractFactClaims(String text) {
        List<FactClaim> claims = new ArrayList<>();
        Matcher matcher = claimPattern.matcher(text);
        
        while (matcher.find()) {
            String claim = matcher.group(1).trim();
            if (claim.length() > 10) {
                float confidence = estimateClaimConfidence(claim);
                boolean isVerifiable = isClaimVerifiable(claim);
                String category = classifyClaim(claim);
                claims.add(new FactClaim(claim, confidence, isVerifiable, category));
            }
        }
        
        return claims;
    }
    
    private float estimateClaimConfidence(String claim) {
        float baseConfidence = 0.7f;
        
        if (claim.contains("I think") || claim.contains("I believe") || 
            claim.contains("maybe") || claim.contains("perhaps") ||
            claim.contains("could be") || claim.contains("might be")) {
            return 0.4f;
        }
        
        if (claim.contains("According to") || claim.contains("Based on") ||
            claim.contains("Research shows") || claim.contains("Studies show")) {
            baseConfidence += 0.15f;
        }
        
        if (Pattern.matches(".*\\d+.*", claim)) {
            baseConfidence += 0.1f;
        }
        
        float lengthBonus = Math.min(claim.length() / 100.0f, 0.15f);
        baseConfidence += lengthBonus;
        
        return Math.min(baseConfidence, 0.95f);
    }
    
    private boolean isClaimVerifiable(String claim) {
        for (String pattern : factPatterns) {
            if (Pattern.compile(pattern, Pattern.CASE_INSENSITIVE).matcher(claim).find()) {
                return true;
            }
        }
        return false;
    }
    
    private String classifyClaim(String claim) {
        if (claim.contains("city") || claim.contains("country") || claim.contains("capital") ||
            claim.contains("river") || claim.contains("mountain") || claim.contains("ocean")) {
            return "geography";
        }
        if (claim.contains("war") || claim.contains("king") || claim.contains("empire") ||
            claim.contains("president") || claim.contains("revolution") || claim.contains("century")) {
            return "history";
        }
        if (claim.contains("atom") || claim.contains("quantum") || claim.contains("DNA") ||
            claim.contains("theory") || claim.contains("experiment") || claim.contains("scientist")) {
            return "science";
        }
        if (claim.contains("computer") || claim.contains("AI") || claim.contains("algorithm") ||
            claim.contains("software") || claim.contains("network") || claim.contains("internet")) {
            return "technology";
        }
        if (claim.contains("equation") || claim.contains("number") || claim.contains("calculation") ||
            claim.contains("math") || claim.contains("theorem")) {
            return "math";
        }
        if (claim.contains("disease") || claim.contains("medicine") || claim.contains("hospital") ||
            claim.contains("doctor") || claim.contains("treatment")) {
            return "medicine";
        }
        if (claim.contains("stock") || claim.contains("market") || claim.contains("economy") ||
            claim.contains("company") || claim.contains("profit")) {
            return "finance";
        }
        return "general";
    }
    
    private float verifyClaim(FactClaim claim) {
        if (knowledgeBase == null) {
            return claim.confidence * 0.6f;
        }
        
        List<String> keywords = extractKeywords(claim.claim);
        List<KnowledgeBaseManager.KnowledgeEntry> matches = new ArrayList<>();
        
        for (String keyword : keywords) {
            List<KnowledgeBaseManager.KnowledgeEntry> results = knowledgeBase.search(keyword, 10);
            matches.addAll(results);
        }
        
        if (matches.isEmpty()) {
            return claim.confidence * 0.5f;
        }
        
        float maxSimilarity = 0;
        for (KnowledgeBaseManager.KnowledgeEntry entry : matches) {
            float similarity = calculateSemanticSimilarity(claim.claim, entry.content);
            maxSimilarity = Math.max(maxSimilarity, similarity);
        }
        
        Float threshold = categoryConfidenceThresholds.getOrDefault(claim.category, 0.95f);
        float adjustedScore = maxSimilarity * claim.confidence;
        
        return adjustedScore >= threshold ? adjustedScore : adjustedScore * 0.7f;
    }
    
    private List<String> extractKeywords(String claim) {
        List<String> keywords = new ArrayList<>();
        Pattern nounPattern = Pattern.compile("\\b[A-Z][a-z]+\\b");
        Matcher matcher = nounPattern.matcher(claim);
        
        while (matcher.find()) {
            String word = matcher.group();
            if (word.length() > 2 && !isStopWord(word)) {
                keywords.add(word);
            }
        }
        
        Pattern numberPattern = Pattern.compile("\\d+");
        matcher = numberPattern.matcher(claim);
        while (matcher.find()) {
            keywords.add(matcher.group());
        }
        
        return keywords.size() > 0 ? keywords : Arrays.asList(claim.split("\\s+")[0]);
    }
    
    private boolean isStopWord(String word) {
        Set<String> stopWords = new HashSet<>(Arrays.asList(
            "The", "A", "An", "In", "On", "At", "For", "Of", "To", "And", "But", "Or",
            "Was", "Is", "Are", "Were", "Be", "Been", "Being", "Has", "Have", "Had",
            "Will", "Would", "Could", "May", "Might", "Should", "Must", "Can"
        ));
        return stopWords.contains(word);
    }
    
    private float calculateSemanticSimilarity(String claim, String knowledge) {
        Set<String> claimWords = new HashSet<>(Arrays.asList(claim.toLowerCase().split("\\W+")));
        Set<String> knowledgeWords = new HashSet<>(Arrays.asList(knowledge.toLowerCase().split("\\W+")));
        
        if (claimWords.isEmpty() || knowledgeWords.isEmpty()) {
            return 0;
        }
        
        Set<String> intersection = new HashSet<>(claimWords);
        intersection.retainAll(knowledgeWords);
        
        Set<String> union = new HashSet<>(claimWords);
        union.addAll(knowledgeWords);
        
        return (float) intersection.size() / union.size();
    }
    
    private String findSource(String claim) {
        if (knowledgeBase == null) {
            return "internal knowledge";
        }
        
        List<String> keywords = extractKeywords(claim);
        for (String keyword : keywords) {
            List<KnowledgeBaseManager.KnowledgeEntry> results = knowledgeBase.search(keyword, 10);
            if (!results.isEmpty()) {
                return results.get(0).source;
            }
        }
        
        return "verified knowledge base";
    }
    
    private float calculateHallucinationRisk(List<FactClaim> claims, int verifiedCount) {
        if (claims.isEmpty()) {
            return 0.01f;
        }
        
        float unverifiedRatio = (float) (claims.size() - verifiedCount) / claims.size();
        float avgConfidence = (float) claims.stream().mapToDouble(c -> c.confidence).average().orElse(0.5);
        
        float risk = unverifiedRatio * (1 - avgConfidence) * 0.8f;
        
        for (FactClaim claim : claims) {
            if (!claim.isVerifiable && claim.confidence < 0.5f) {
                risk += 0.02f;
            }
        }
        
        return Math.max(0.0f, Math.min(risk, 0.99f));
    }
    
    private String correctHallucinations(String text, List<FactClaim> claims, List<Citation> citations) {
        String corrected = text;
        
        for (FactClaim claim : claims) {
            if (claim.confidence < 0.6f && !claim.isVerifiable) {
                corrected = corrected.replace(claim.claim, 
                    "[Uncertain: " + claim.claim.replace("[.!?]", "") + "]");
            }
        }
        
        for (Citation citation : citations) {
            if (citation.confidence >= 0.98f) {
                String citationMarker = " [Source: " + citation.source + "]";
                corrected = corrected.replace(citation.claim, citation.claim.replace(".", citationMarker + "."));
            }
        }
        
        return corrected;
    }
    
    private String refineGeneration(VerificationResult result) {
        StringBuilder refined = new StringBuilder(result.correctedText);
        
        for (String unverified : result.unverifiedClaims) {
            String refinedClaim = "[Needs verification: " + unverified + "]";
            int idx = refined.indexOf(unverified);
            if (idx >= 0) {
                refined.replace(idx, idx + unverified.length(), refinedClaim);
            }
        }
        
        return refined.toString();
    }
    
    public void shutdown() {
        executor.shutdown();
    }
    
    public String getStatus() {
        return String.format("AntiHallucinationEngine: patterns=%d, categories=%d, attempts=%d, threshold=%.2f",
            factPatterns.size(), categoryConfidenceThresholds.size(), maxVerificationAttempts, verificationThreshold);
    }
}