package com.qingxiaotuan.simpcore.data;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DataPipeline<T> {

    private List<DataProcessor<T>> processors;
    private DataSource<T> dataSource;
    private int batchSize;
    private boolean shuffle;
    private Random random;

    public DataPipeline() {
        this.processors = new ArrayList<>();
        this.batchSize = 32;
        this.shuffle = false;
        this.random = new Random(42);
    }

    public DataPipeline<T> setDataSource(DataSource<T> dataSource) {
        this.dataSource = dataSource;
        return this;
    }

    public DataPipeline<T> addProcessor(DataProcessor<T> processor) {
        this.processors.add(processor);
        return this;
    }

    public DataPipeline<T> setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public DataPipeline<T> setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    public DataPipeline<T> setSeed(long seed) {
        this.random.setSeed(seed);
        return this;
    }

    public Iterator<List<T>> iterator() {
        List<T> data = dataSource.load();
        
        for (DataProcessor<T> processor : processors) {
            data = processor.process(data);
        }
        
        if (shuffle) {
            Collections.shuffle(data, random);
        }
        
        return new BatchIterator(data, batchSize);
    }

    public List<List<T>> collect() {
        List<List<T>> batches = new ArrayList<>();
        Iterator<List<T>> iterator = iterator();
        while (iterator.hasNext()) {
            batches.add(iterator.next());
        }
        return batches;
    }

    public int size() {
        return (int) Math.ceil((double) dataSource.load().size() / batchSize);
    }

    private static class BatchIterator<T> implements Iterator<List<T>> {
        private final List<T> data;
        private final int batchSize;
        private int index;

        public BatchIterator(List<T> data, int batchSize) {
            this.data = data;
            this.batchSize = batchSize;
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < data.size();
        }

        @Override
        public List<T> next() {
            int end = Math.min(index + batchSize, data.size());
            List<T> batch = data.subList(index, end);
            index = end;
            return new ArrayList<>(batch);
        }
    }

    public interface DataSource<T> {
        List<T> load();
        int size();
    }

    public interface DataProcessor<T> {
        List<T> process(List<T> data);
    }

    public static class TextTokenizer implements DataProcessor<String> {
        private final Tokenizer tokenizer;
        private final int maxLength;

        public TextTokenizer(Tokenizer tokenizer, int maxLength) {
            this.tokenizer = tokenizer;
            this.maxLength = maxLength;
        }

        @Override
        public List<String> process(List<String> data) {
            return data.stream()
                .map(text -> {
                    int[] tokens = tokenizer.encode(text);
                    if (tokens.length > maxLength) {
                        int[] truncated = new int[maxLength];
                        System.arraycopy(tokens, 0, truncated, 0, maxLength);
                        return tokenizer.decode(truncated);
                    }
                    return text;
                })
                .collect(Collectors.toList());
        }
    }

    public static class TextNormalizer implements DataProcessor<String> {
        @Override
        public List<String> process(List<String> data) {
            return data.stream()
                .map(text -> text.toLowerCase().trim())
                .map(text -> text.replaceAll("[^a-zA-Z0-9\\s]", ""))
                .collect(Collectors.toList());
        }
    }

    public static class ImageResizer implements DataProcessor<float[]> {
        private final int targetWidth;
        private final int targetHeight;

        public ImageResizer(int targetWidth, int targetHeight) {
            this.targetWidth = targetWidth;
            this.targetHeight = targetHeight;
        }

        @Override
        public List<float[]> process(List<float[]> data) {
            return data.stream()
                .map(image -> resizeImage(image, targetWidth, targetHeight))
                .collect(Collectors.toList());
        }

        private float[] resizeImage(float[] image, int width, int height) {
            float[] resized = new float[width * height * 3];
            return resized;
        }
    }

    public static class ImageNormalizer implements DataProcessor<float[]> {
        @Override
        public List<float[]> process(List<float[]> data) {
            return data.stream()
                .map(image -> {
                    float[] normalized = new float[image.length];
                    for (int i = 0; i < image.length; i++) {
                        normalized[i] = image[i] / 255.0f;
                    }
                    return normalized;
                })
                .collect(Collectors.toList());
        }
    }

    public static class DataAugmenter<T> implements DataProcessor<T> {
        private final List<Function<T, T>> augmentations;
        private final float probability;

        public DataAugmenter(List<Function<T, T>> augmentations, float probability) {
            this.augmentations = augmentations;
            this.probability = probability;
        }

        @Override
        public List<T> process(List<T> data) {
            Random rand = new Random();
            return data.stream()
                .map(item -> {
                    T augmented = item;
                    for (Function<T, T> augmentation : augmentations) {
                        if (rand.nextFloat() < probability) {
                            augmented = augmentation.apply(augmented);
                        }
                    }
                    return augmented;
                })
                .collect(Collectors.toList());
        }
    }

    public static class PaddingProcessor implements DataProcessor<int[]> {
        private final int maxLength;
        private final int padValue;

        public PaddingProcessor(int maxLength, int padValue) {
            this.maxLength = maxLength;
            this.padValue = padValue;
        }

        @Override
        public List<int[]> process(List<int[]> data) {
            return data.stream()
                .map(sequence -> {
                    int[] padded = new int[maxLength];
                    Arrays.fill(padded, padValue);
                    System.arraycopy(sequence, 0, padded, 0, Math.min(sequence.length, maxLength));
                    return padded;
                })
                .collect(Collectors.toList());
        }
    }

    public static class LabelEncoder<T> implements DataProcessor<T> {
        private final Map<T, Integer> labelMap;

        public LabelEncoder(List<T> labels) {
            this.labelMap = new HashMap<>();
            int index = 0;
            for (T label : labels) {
                if (!labelMap.containsKey(label)) {
                    labelMap.put(label, index++);
                }
            }
        }

        @Override
        public List<T> process(List<T> data) {
            return data;
        }

        public int encode(T label) {
            return labelMap.getOrDefault(label, -1);
        }

        public T decode(int index) {
            for (Map.Entry<T, Integer> entry : labelMap.entrySet()) {
                if (entry.getValue() == index) {
                    return entry.getKey();
                }
            }
            return null;
        }

        public int numClasses() {
            return labelMap.size();
        }
    }

    public static class CsvDataSource implements DataSource<String[]> {
        private final String filePath;
        private final boolean hasHeader;

        public CsvDataSource(String filePath, boolean hasHeader) {
            this.filePath = filePath;
            this.hasHeader = hasHeader;
        }

        @Override
        public List<String[]> load() {
            List<String[]> rows = new ArrayList<>();
            return rows;
        }

        @Override
        public int size() {
            return load().size();
        }
    }

    public static class TextFileDataSource implements DataSource<String> {
        private final String filePath;

        public TextFileDataSource(String filePath) {
            this.filePath = filePath;
        }

        @Override
        public List<String> load() {
            List<String> lines = new ArrayList<>();
            return lines;
        }

        @Override
        public int size() {
            return load().size();
        }
    }

    public static class Tokenizer {
        private final Map<String, Integer> vocab;
        private final Map<Integer, String> idxToToken;
        private final String bosToken = "<bos>";
        private final String eosToken = "<eos>";
        private final String padToken = "<pad>";
        private final String unkToken = "<unk>";

        public Tokenizer() {
            this.vocab = new HashMap<>();
            this.idxToToken = new HashMap<>();
            addSpecialTokens();
        }

        private void addSpecialTokens() {
            vocab.put(bosToken, 0);
            vocab.put(eosToken, 1);
            vocab.put(padToken, 2);
            vocab.put(unkToken, 3);
            idxToToken.put(0, bosToken);
            idxToToken.put(1, eosToken);
            idxToToken.put(2, padToken);
            idxToToken.put(3, unkToken);
        }

        public void fit(List<String> texts) {
            for (String text : texts) {
                String[] tokens = text.split("\\s+");
                for (String token : tokens) {
                    if (!vocab.containsKey(token)) {
                        int idx = vocab.size();
                        vocab.put(token, idx);
                        idxToToken.put(idx, token);
                    }
                }
            }
        }

        public int[] encode(String text) {
            List<Integer> tokens = new ArrayList<>();
            tokens.add(vocab.get(bosToken));
            for (String token : text.split("\\s+")) {
                tokens.add(vocab.getOrDefault(token, vocab.get(unkToken)));
            }
            tokens.add(vocab.get(eosToken));
            return tokens.stream().mapToInt(Integer::intValue).toArray();
        }

        public String decode(int[] tokens) {
            StringBuilder sb = new StringBuilder();
            for (int token : tokens) {
                String tokenStr = idxToToken.getOrDefault(token, unkToken);
                if (!tokenStr.equals(bosToken) && !tokenStr.equals(eosToken) && !tokenStr.equals(padToken)) {
                    sb.append(tokenStr).append(" ");
                }
            }
            return sb.toString().trim();
        }

        public int getVocabSize() {
            return vocab.size();
        }
    }
}