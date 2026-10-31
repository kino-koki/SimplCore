package com.qingxiaotuan.simpcore.model;

import java.util.*;
import java.io.*;

public class Tokenizer {

    private final Map<String, Integer> tokenToId;
    private final List<String> idToToken;
    private final int maxTokenLength;
    private final String bosToken = "<bos>";
    private final String eosToken = "<eos>";
    private final String padToken = "<pad>";
    private final String unkToken = "<unk>";

    public Tokenizer(int vocabSize) {
        this.tokenToId = new HashMap<>();
        this.idToToken = new ArrayList<>();
        this.maxTokenLength = 10;

        int id = 0;
        tokenToId.put(bosToken, id);
        idToToken.add(bosToken);
        id++;

        tokenToId.put(eosToken, id);
        idToToken.add(eosToken);
        id++;

        tokenToId.put(padToken, id);
        idToToken.add(padToken);
        id++;

        tokenToId.put(unkToken, id);
        idToToken.add(unkToken);
        id++;

        for (char c = 'a'; c <= 'z'; c++) {
            String token = String.valueOf(c);
            if (id < vocabSize) {
                tokenToId.put(token, id);
                idToToken.add(token);
                id++;
            }
        }

        for (char c = 'A'; c <= 'Z'; c++) {
            String token = String.valueOf(c);
            if (id < vocabSize) {
                tokenToId.put(token, id);
                idToToken.add(token);
                id++;
            }
        }

        for (char c = '0'; c <= '9'; c++) {
            String token = String.valueOf(c);
            if (id < vocabSize) {
                tokenToId.put(token, id);
                idToToken.add(token);
                id++;
            }
        }

        String[] specialChars = {" ", ".", ",", "!", "?", ":", ";", "-", "_", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "\"", "'", "\\", "/", "<", ">", "=", "+", "~", "`", "|"};
        for (String ch : specialChars) {
            if (id < vocabSize) {
                tokenToId.put(ch, id);
                idToToken.add(ch);
                id++;
            }
        }

        while (id < vocabSize) {
            String extraToken = "<extra_" + (id - 4 - 26 - 26 - 10 - 33) + ">";
            tokenToId.put(extraToken, id);
            idToToken.add(extraToken);
            id++;
        }
    }

    private String generateRandomToken(int seed) {
        StringBuilder sb = new StringBuilder();
        Random rnd = new Random(seed);
        int length = 2 + rnd.nextInt(5);
        for (int i = 0; i < length; i++) {
            sb.append((char) ('a' + rnd.nextInt(26)));
        }
        return sb.toString();
    }

    public int[] encode(String text) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(tokenToId.get(bosToken));

        int i = 0;
        while (i < text.length()) {
            boolean found = false;
            for (int len = Math.min(maxTokenLength, text.length() - i); len >= 1; len--) {
                String substr = text.substring(i, i + len);
                if (tokenToId.containsKey(substr)) {
                    tokens.add(tokenToId.get(substr));
                    i += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                tokens.add(tokenToId.get(unkToken));
                i++;
            }
        }

        tokens.add(tokenToId.get(eosToken));
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int tokenId : tokens) {
            if (tokenId >= 0 && tokenId < idToToken.size()) {
                String token = idToToken.get(tokenId);
                if (!token.equals(bosToken) && !token.equals(eosToken) && !token.equals(padToken) && !token.equals(unkToken) && !token.startsWith("<extra_")) {
                    sb.append(token);
                }
            }
        }
        return sb.toString();
    }

    public int getVocabSize() {
        return tokenToId.size();
    }

    public int getBosTokenId() {
        return tokenToId.get(bosToken);
    }

    public int getEosTokenId() {
        return tokenToId.get(eosToken);
    }

    public int getPadTokenId() {
        return tokenToId.get(padToken);
    }

    public int getUnkTokenId() {
        return tokenToId.get(unkToken);
    }

    public void save(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(new HashMap<>(tokenToId));
            oos.writeObject(new ArrayList<>(idToToken));
        }
    }

    @SuppressWarnings("unchecked")
    public static Tokenizer load(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            Map<String, Integer> tokenMap = (Map<String, Integer>) ois.readObject();
            List<String> idList = (List<String>) ois.readObject();

            Tokenizer tokenizer = new Tokenizer(0);
            tokenizer.tokenToId.clear();
            tokenizer.tokenToId.putAll(tokenMap);
            tokenizer.idToToken.clear();
            tokenizer.idToToken.addAll(idList);

            return tokenizer;
        }
    }
}