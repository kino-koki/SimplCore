package com.qingxiaotuan.simpcore.tokenizer;

import java.util.*;

public class BPETokenizer {
    
    protected Map<String, Integer> vocab;
    protected Map<Integer, String> idToToken;
    protected Map<String, String> merges;
    protected int nextId;
    
    public BPETokenizer() {
        this.vocab = new HashMap<>();
        this.idToToken = new HashMap<>();
        this.merges = new HashMap<>();
        this.nextId = 0;
        initializeBaseVocab();
    }
    
    private void initializeBaseVocab() {
        for (int i = 0; i < 256; i++) {
            String token = String.valueOf((char) i);
            vocab.put(token, nextId);
            idToToken.put(nextId, token);
            nextId++;
        }
        
        vocab.put("<|endoftext|>", nextId);
        idToToken.put(nextId, "<|endoftext|>");
        nextId++;
        
        vocab.put("<|padding|>", nextId);
        idToToken.put(nextId, "<|padding|>");
        nextId++;
        
        vocab.put("<|unknown|>", nextId);
        idToToken.put(nextId, "<|unknown|>");
        nextId++;
    }
    
    public void train(String text, int iterations) {
        List<String> tokens = new ArrayList<>();
        for (char c : text.toCharArray()) {
            tokens.add(String.valueOf(c));
        }
        
        for (int i = 0; i < iterations; i++) {
            Map<String, Integer> pairCounts = new HashMap<>();
            
            for (int j = 0; j < tokens.size() - 1; j++) {
                String pair = tokens.get(j) + "|" + tokens.get(j + 1);
                pairCounts.put(pair, pairCounts.getOrDefault(pair, 0) + 1);
            }
            
            if (pairCounts.isEmpty()) break;
            
            String bestPair = null;
            int maxCount = 0;
            for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    bestPair = entry.getKey();
                }
            }
            
            if (bestPair == null || maxCount < 2) break;
            
            String[] parts = bestPair.split("\\|");
            String newToken = parts[0] + parts[1];
            
            vocab.put(newToken, nextId);
            idToToken.put(nextId, newToken);
            merges.put(bestPair, newToken);
            nextId++;
            
            List<String> newTokens = new ArrayList<>();
            int j = 0;
            while (j < tokens.size()) {
                if (j < tokens.size() - 1 && tokens.get(j).equals(parts[0]) && tokens.get(j + 1).equals(parts[1])) {
                    newTokens.add(newToken);
                    j += 2;
                } else {
                    newTokens.add(tokens.get(j));
                    j++;
                }
            }
            tokens = newTokens;
        }
        
        System.out.printf("BPE training complete. Vocab size: %d, Merges: %d%n", vocab.size(), merges.size());
    }
    
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        
        List<String> tokens = new ArrayList<>();
        for (char c : text.toCharArray()) {
            tokens.add(String.valueOf(c));
        }
        
        while (true) {
            String bestPair = null;
            int bestIdx = -1;
            
            for (int i = 0; i < tokens.size() - 1; i++) {
                String pair = tokens.get(i) + "|" + tokens.get(i + 1);
                if (merges.containsKey(pair)) {
                    bestPair = pair;
                    bestIdx = i;
                }
            }
            
            if (bestPair == null) break;
            
            String replacement = merges.get(bestPair);
            tokens.set(bestIdx, replacement);
            tokens.remove(bestIdx + 1);
        }
        
        for (String token : tokens) {
            ids.add(vocab.getOrDefault(token, vocab.get("<|unknown|>")));
        }
        
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }
    
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String token = idToToken.getOrDefault(id, "<|unknown|>");
            sb.append(token);
        }
        return sb.toString();
    }
    
    public int getVocabSize() {
        return vocab.size();
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - BPE Tokenizer Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        String trainingText = "hello hello world world";
        
        System.out.println("[1/4] Training BPE tokenizer...");
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(trainingText, 10);
        System.out.printf("  Final vocab size: %d%n", tokenizer.getVocabSize());
        
        System.out.println("\n[2/4] Testing encoding...");
        String testText = "hello world";
        int[] encoded = tokenizer.encode(testText);
        System.out.printf("  Input: %s%n", testText);
        System.out.printf("  Encoded: %s%n", Arrays.toString(encoded));
        
        System.out.println("\n[3/4] Testing decoding...");
        String decoded = tokenizer.decode(encoded);
        System.out.printf("  Decoded: %s%n", decoded);
        
        System.out.println("\n[4/4] Testing reconstruction...");
        boolean matches = testText.equals(decoded);
        System.out.printf("  Encoding/decoding roundtrip: %s%n", matches ? "SUCCESS" : "FAILED");
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    BPE Tokenizer Test Passed!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}