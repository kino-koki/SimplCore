package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.util.*;

public class RedundantTokenDetector {

    public enum RedundancyType {
        DUPLICATE(1, "重复Token", "连续出现的相同Token"),
        LOW_INFO(2, "低信息Token", "信息量低于阈值的Token"),
        COMPRESSIBLE(3, "可压缩Token", "可通过编码压缩的Token"),
        STRUCTURAL(4, "结构冗余", "语法结构导致的冗余"),
        FORMATTING(5, "格式冗余", "格式化符号导致的冗余");

        private final int priority;
        private final String name;
        private final String description;

        RedundancyType(int priority, String name, String description) {
            this.priority = priority;
            this.name = name;
            this.description = description;
        }

        public int getPriority() { return priority; }
        public String getName() { return name; }
        public String getDescription() { return description; }
    }

    public static class RedundancyReport {
        public List<RedundancyItem> items = new ArrayList<>();
        public Map<RedundancyType, Integer> typeCount = new HashMap<>();
        public double totalRedundancyRatio;
        public double estimatedCompressionRatio;

        public void addItem(RedundancyItem item) {
            items.add(item);
            typeCount.put(item.type, typeCount.getOrDefault(item.type, 0) + 1);
        }

        public void print() {
            System.out.println("=== Redundant Token Detection Report ===");
            System.out.println("Total redundancy ratio: " + String.format("%.1f%%", totalRedundancyRatio * 100));
            System.out.println("Estimated compression ratio: " + String.format("%.1f%%", estimatedCompressionRatio * 100));
            System.out.println("\nRedundancy by type:");
            Arrays.stream(RedundancyType.values())
                    .sorted((a, b) -> Integer.compare(b.getPriority(), a.getPriority()))
                    .forEach(type -> System.out.println("  " + type.getName() + ": " + typeCount.getOrDefault(type, 0)));
            System.out.println("\nTop redundant tokens:");
            items.stream()
                    .sorted((a, b) -> Double.compare(b.redundancyScore, a.redundancyScore))
                    .limit(10)
                    .forEach(item -> System.out.println("  " + item.token + " [" + item.type.getName() + "] score: " + String.format("%.3f", item.redundancyScore)));
        }
    }

    public static class RedundancyItem {
        public String token;
        public RedundancyType type;
        public int position;
        public double redundancyScore;
        public double semanticPreservation;
        public String replacement;

        public RedundancyItem(String token, RedundancyType type, int position, double score, double preservation, String replacement) {
            this.token = token;
            this.type = type;
            this.position = position;
            this.redundancyScore = score;
            this.semanticPreservation = preservation;
            this.replacement = replacement;
        }
    }

    private static final Set<String> ARTICLES = Set.of("the", "a", "an");
    private static final Set<String> CONNECTORS = Set.of("and", "but", "or", "so", "yet", "for", "nor");
    private static final Set<String> PREPOSITIONS = Set.of("of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under");
    private static final Set<String> AUXILIARIES = Set.of("is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can");

    public static RedundancyReport detectRedundancy(int[] tokenIds, Tokenizer tokenizer, double semanticThreshold) {
        RedundancyReport report = new RedundancyReport();
        Map<String, Integer> tokenPositions = new HashMap<>();
        List<String> tokenStrings = new ArrayList<>();

        for (int id : tokenIds) {
            String token = tokenizer.idToToken.get(id);
            tokenStrings.add(token != null ? token : "<unk>");
        }

        for (int i = 0; i < tokenStrings.size(); i++) {
            String token = tokenStrings.get(i);

            if (token.equals("<bos>") || token.equals("<eos>") || token.equals("<pad>")) {
                continue;
            }

            double redundancyScore = 0;
            RedundancyType type = null;
            double preservation = 1.0;
            String replacement = null;

            if (i > 0 && token.equals(tokenStrings.get(i - 1))) {
                redundancyScore = 0.9;
                type = RedundancyType.DUPLICATE;
                preservation = 0.95;
                replacement = "";
            }

            else if (ARTICLES.contains(token.toLowerCase())) {
                redundancyScore = 0.7;
                type = RedundancyType.LOW_INFO;
                preservation = 0.85;
                replacement = "";
            }

            else if (CONNECTORS.contains(token.toLowerCase())) {
                redundancyScore = 0.6;
                type = RedundancyType.LOW_INFO;
                preservation = 0.9;
                replacement = "";
            }

            else if (PREPOSITIONS.contains(token.toLowerCase())) {
                redundancyScore = 0.55;
                type = RedundancyType.LOW_INFO;
                preservation = 0.88;
                replacement = "";
            }

            else if (AUXILIARIES.contains(token.toLowerCase())) {
                redundancyScore = 0.5;
                type = RedundancyType.LOW_INFO;
                preservation = 0.92;
                replacement = "";
            }

            else if (token.length() == 1 && Character.isWhitespace(token.charAt(0))) {
                redundancyScore = 0.4;
                type = RedundancyType.FORMATTING;
                preservation = 0.98;
                replacement = "";
            }

            else if (token.matches("[\\p{Punct}]+") && token.length() > 1) {
                redundancyScore = 0.35;
                type = RedundancyType.FORMATTING;
                preservation = 0.96;
                replacement = token.substring(0, 1);
            }

            else if (token.matches("\\d{4,}")) {
                redundancyScore = 0.3;
                type = RedundancyType.COMPRESSIBLE;
                preservation = 0.99;
                replacement = "NUM";
            }

            if (type != null && preservation >= semanticThreshold) {
                report.addItem(new RedundancyItem(token, type, i, redundancyScore, preservation, replacement));
            }
        }

        report.totalRedundancyRatio = (double) report.items.size() / tokenStrings.size();
        report.estimatedCompressionRatio = report.totalRedundancyRatio * 0.7;

        return report;
    }

    public static int[] applyCompression(int[] tokenIds, Tokenizer tokenizer, RedundancyReport report) {
        List<Integer> compressed = new ArrayList<>();
        Set<Integer> removedPositions = new HashSet<>();

        for (RedundancyItem item : report.items) {
            removedPositions.add(item.position);
        }

        for (int i = 0; i < tokenIds.length; i++) {
            if (!removedPositions.contains(i)) {
                compressed.add(tokenIds[i]);
            }
        }

        return compressed.stream().mapToInt(Integer::intValue).toArray();
    }

    public static double calculateSemanticSimilarity(String original, String compressed) {
        Set<String> originalWords = new HashSet<>(Arrays.asList(original.toLowerCase().split("\\s+")));
        Set<String> compressedWords = new HashSet<>(Arrays.asList(compressed.toLowerCase().split("\\s+")));

        if (originalWords.isEmpty()) return 1.0;

        Set<String> intersection = new HashSet<>(originalWords);
        intersection.retainAll(compressedWords);

        return (double) intersection.size() / originalWords.size();
    }
}