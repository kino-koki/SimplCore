package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.util.*;

public class EfficientTokenEncoder {

    public enum ContentType {
        NATURAL_LANGUAGE(0.3),
        CODE(0.5),
        STRUCTURED_DATA(0.7),
        NUMERIC(0.8),
        MIXED(0.4);

        private final double compressionFactor;

        ContentType(double factor) {
            this.compressionFactor = factor;
        }

        public double getCompressionFactor() { return compressionFactor; }
    }

    public static class EncodingResult {
        public int[] encodedTokens;
        public double compressionRatio;
        public double semanticFidelity;
        public ContentType detectedType;
        public Map<String, Object> encodingMetrics = new HashMap<>();

        public void print() {
            System.out.println("=== Encoding Result ===");
            System.out.println("Content type: " + detectedType);
            System.out.println("Compression ratio: " + String.format("%.1f%%", compressionRatio * 100));
            System.out.println("Semantic fidelity: " + String.format("%.2f%%", semanticFidelity * 100));
            System.out.println("Original tokens: " + encodingMetrics.get("originalCount"));
            System.out.println("Encoded tokens: " + encodingMetrics.get("encodedCount"));
        }
    }

    private static final Set<String> CODE_KEYWORDS = Set.of("public", "private", "protected", "static", "final", "void", "int", "float", "double", "boolean", "char", "class", "interface", "extends", "implements", "override", "return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "throw", "try", "catch", "finally", "new", "instanceof", "this", "super", "null", "true", "false", "import", "package", "enum", "abstract", "native", "synchronized", "volatile", "transient");
    private static final Set<String> DATA_KEYWORDS = Set.of("json", "xml", "yaml", "csv", "table", "field", "value", "key", "row", "column", "schema", "struct", "array", "object");

    public static ContentType detectContentType(String text) {
        String lower = text.toLowerCase();
        int codeScore = 0;
        int dataScore = 0;
        int numericScore = 0;
        int textScore = 0;

        for (String kw : CODE_KEYWORDS) {
            if (lower.contains(kw)) codeScore++;
        }
        for (String kw : DATA_KEYWORDS) {
            if (lower.contains(kw)) dataScore++;
        }

        int digitCount = 0;
        int letterCount = 0;
        int specialCount = 0;
        for (char c : text.toCharArray()) {
            if (Character.isDigit(c)) digitCount++;
            else if (Character.isLetter(c)) letterCount++;
            else if (c == '{' || c == '}' || c == '[' || c == ']' || c == '<' || c == '>' || c == '"') specialCount++;
        }

        if (digitCount > letterCount * 2) numericScore = 10;
        if (specialCount > letterCount / 2) dataScore += 5;
        if (codeScore > 5) codeScore += 5;

        int total = codeScore + dataScore + numericScore + textScore;
        if (numericScore >= total * 0.4) return ContentType.NUMERIC;
        if (dataScore >= total * 0.4) return ContentType.STRUCTURED_DATA;
        if (codeScore >= total * 0.4) return ContentType.CODE;
        if (letterCount > digitCount * 3) return ContentType.NATURAL_LANGUAGE;
        return ContentType.MIXED;
    }

    public static EncodingResult encode(int[] tokens, Tokenizer tokenizer, ContentType type, double fidelityThreshold) {
        EncodingResult result = new EncodingResult();
        result.detectedType = type;

        List<String> tokenStrings = new ArrayList<>();
        for (int id : tokens) {
            String token = tokenizer.idToToken.get(id);
            tokenStrings.add(token != null ? token : "<unk>");
        }

        List<Integer> encoded = new ArrayList<>();
        Map<String, Integer> freqMap = new HashMap<>();
        for (String token : tokenStrings) {
            freqMap.put(token, freqMap.getOrDefault(token, 0) + 1);
        }

        List<String> frequentTokens = new ArrayList<>();
        freqMap.entrySet().stream()
                .filter(e -> e.getValue() > 1)
                .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
                .limit(200)
                .forEach(e -> frequentTokens.add(e.getKey()));

        Map<String, Integer> tokenToCode = new HashMap<>();
        int codeId = tokenizer.getVocabSize();
        for (String token : frequentTokens) {
            tokenToCode.put(token, codeId++);
        }

        for (int i = 0; i < tokenStrings.size(); i++) {
            String token = tokenStrings.get(i);

            if (token.equals("<bos>") || token.equals("<eos>") || token.equals("<pad>")) {
                encoded.add(tokens[i]);
                continue;
            }

            if (tokenToCode.containsKey(token) && freqMap.get(token) > 2) {
                encoded.add(tokenToCode.get(token));
            } else if (type == ContentType.CODE && isCodeToken(token)) {
                encoded.add(encodeCodeToken(token, tokenizer));
            } else if (type == ContentType.STRUCTURED_DATA && isStructuredToken(token)) {
                encoded.add(encodeStructuredToken(token, tokenizer));
            } else if (type == ContentType.NUMERIC && isNumericToken(token)) {
                encoded.add(encodeNumericToken(token, tokenizer));
            } else {
                encoded.add(tokens[i]);
            }
        }

        result.encodedTokens = encoded.stream().mapToInt(Integer::intValue).toArray();
        result.compressionRatio = 1.0 - (double) encoded.size() / tokens.length;
        result.semanticFidelity = Math.min(1.0, 1.0 - result.compressionRatio * (1.0 - type.getCompressionFactor()));

        result.encodingMetrics.put("originalCount", tokens.length);
        result.encodingMetrics.put("encodedCount", encoded.size());
        result.encodingMetrics.put("dictionarySize", tokenToCode.size());
        result.encodingMetrics.put("frequencyCompression", frequentTokens.size());

        return result;
    }

    private static boolean isCodeToken(String token) {
        return CODE_KEYWORDS.contains(token) || token.matches("[a-zA-Z_][a-zA-Z0-9_]*") || token.matches("[{}()\\[\\]<>]");
    }

    private static boolean isStructuredToken(String token) {
        return token.matches("[{}]") || token.matches("\"[^\"]*\"") || token.matches("[,:]");
    }

    private static boolean isNumericToken(String token) {
        return token.matches("-?\\d+\\.?\\d*") || token.matches("[0-9]+");
    }

    private static int encodeCodeToken(String token, Tokenizer tokenizer) {
        if (CODE_KEYWORDS.contains(token)) {
            return tokenizer.getVocabSize() + CODE_KEYWORDS.indexOf(token);
        }
        int hash = token.hashCode() % 1000;
        return tokenizer.getVocabSize() + 100 + hash;
    }

    private static int encodeStructuredToken(String token, Tokenizer tokenizer) {
        if (token.equals("{")) return tokenizer.getVocabSize() + 2000;
        if (token.equals("}")) return tokenizer.getVocabSize() + 2001;
        if (token.equals("[")) return tokenizer.getVocabSize() + 2002;
        if (token.equals("]")) return tokenizer.getVocabSize() + 2003;
        if (token.equals("\"")) return tokenizer.getVocabSize() + 2004;
        if (token.equals(":")) return tokenizer.getVocabSize() + 2005;
        if (token.equals(",")) return tokenizer.getVocabSize() + 2006;
        return tokenizer.getVocabSize() + 2007;
    }

    private static int encodeNumericToken(String token, Tokenizer tokenizer) {
        try {
            double num = Double.parseDouble(token);
            if (num == Math.floor(num) && num >= 0 && num <= 1000) {
                return tokenizer.getVocabSize() + 3000 + (int) num;
            }
            int magnitude = (int) Math.log10(Math.abs(num));
            return tokenizer.getVocabSize() + 4000 + magnitude;
        } catch (NumberFormatException e) {
            return tokenizer.getVocabSize() + 4100;
        }
    }

    public static int[] decode(int[] encodedTokens, Tokenizer tokenizer) {
        List<Integer> decoded = new ArrayList<>();

        for (int token : encodedTokens) {
            if (token < tokenizer.getVocabSize()) {
                decoded.add(token);
            } else if (token >= tokenizer.getVocabSize() + 2000 && token < tokenizer.getVocabSize() + 2008) {
                int offset = token - (tokenizer.getVocabSize() + 2000);
                String[] structTokens = {"{", "}", "[", "]", "\"", ":", ","};
                if (offset < structTokens.length) {
                    decoded.add(tokenizer.tokenToId.get(structTokens[offset]));
                } else {
                    decoded.add(tokenizer.getUnkTokenId());
                }
            } else {
                decoded.add(tokenizer.getUnkTokenId());
            }
        }

        return decoded.stream().mapToInt(Integer::intValue).toArray();
    }

    public static double calculateEncodingEfficiency(String text, Tokenizer tokenizer) {
        int[] original = tokenizer.encode(text);
        ContentType type = detectContentType(text);
        EncodingResult result = encode(original, tokenizer, type, 0.9);

        String decoded = tokenizer.decode(decode(result.encodedTokens, tokenizer));
        double similarity = RedundantTokenDetector.calculateSemanticSimilarity(text, decoded);

        return result.compressionRatio * similarity;
    }
}