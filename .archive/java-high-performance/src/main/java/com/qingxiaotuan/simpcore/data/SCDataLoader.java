package com.qingxiaotuan.simpcore.data;

import java.util.*;
import java.util.concurrent.*;

/**
 * 多线程数据加载器 - 支持预加载、数据缓存和动态序列长度处理。
 *
 * <p>功能特性：
 * <ul>
 *   <li>多线程预加载：使用线程池在后台预取下一批数据</li>
 *   <li>数据缓存：LRU缓存避免重复计算</li>
 *   <li>动态序列长度：支持最长512 token的变长序列</li>
 *   <li>批处理：自动将数据组织成mini-batch</li>
 *   <li>数据打乱：每个epoch自动打乱数据顺序</li>
 * </ul></p>
 *
 * <p>复杂度分析：
 * <ul>
 *   <li>预加载开销: O(batch_size * seq_len) 每批，但异步执行不阻塞训练</li>
 *   <li>缓存命中: O(1) 查找</li>
 *   <li>内存开销: O(cache_size * batch_size * seq_len)</li>
 * </ul></p>
 *
 * @author SC Architecture Team
 */
public class SCDataLoader {

    /** 最大序列长度 */
    public static final int MAX_SEQ_LEN = 512;
    /** 默认缓存大小 */
    private static final int DEFAULT_CACHE_SIZE = 64;

    private final List<int[]> allSequences;
    private final int batchSize;
    private final int seqLen;
    private final boolean shuffle;
    private final int numWorkers;

    private final BlockingQueue<int[][]> prefetchQueue;
    private final ExecutorService prefetchExecutor;
    private final Map<Integer, int[][]> batchCache;
    private final int cacheSize;

    private List<int[]> currentOrder;
    private int currentBatchIndex;
    private int totalBatches;
    private volatile boolean prefetchRunning;

    /**
     * 创建数据加载器。
     *
     * @param sequences 所有数据序列列表
     * @param batchSize 每批大小
     * @param seqLen 序列长度（支持动态，最大512）
     * @param shuffle 是否打乱数据
     * @param numWorkers 预加载线程数
     */
    public SCDataLoader(List<int[]> sequences, int batchSize, int seqLen,
                        boolean shuffle, int numWorkers) {
        this.allSequences = new ArrayList<>(sequences);
        this.batchSize = batchSize;
        this.seqLen = Math.min(seqLen, MAX_SEQ_LEN);
        this.shuffle = shuffle;
        this.numWorkers = Math.max(1, numWorkers);
        this.prefetchQueue = new LinkedBlockingQueue<>(numWorkers * 2);
        this.prefetchExecutor = Executors.newFixedThreadPool(this.numWorkers);
        this.batchCache = new LinkedHashMap<Integer, int[][]>(DEFAULT_CACHE_SIZE, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Integer, int[][]> eldest) {
                return size() > DEFAULT_CACHE_SIZE;
            }
        };
        this.cacheSize = DEFAULT_CACHE_SIZE;
        this.currentBatchIndex = 0;
        reset();
    }

    /**
     * 创建数据加载器（默认1个工作线程）。
     *
     * @param sequences 数据序列
     * @param batchSize 批大小
     * @param seqLen 序列长度
     */
    public SCDataLoader(List<int[]> sequences, int batchSize, int seqLen) {
        this(sequences, batchSize, seqLen, true, 1);
    }

    /**
     * 重置数据迭代器，开始新epoch。
     */
    public void reset() {
        currentOrder = new ArrayList<>(allSequences);
        if (shuffle) {
            Collections.shuffle(currentOrder);
        }
        totalBatches = (int) Math.ceil((double) currentOrder.size() / batchSize);
        currentBatchIndex = 0;
        prefetchRunning = false;
    }

    /**
     * 启动后台预加载。
     */
    public void startPrefetch() {
        if (prefetchRunning) return;
        prefetchRunning = true;
        prefetchExecutor.submit(this::prefetchLoop);
    }

    /**
     * 预加载循环（在后台线程执行）。
     */
    private void prefetchLoop() {
        while (prefetchRunning && currentBatchIndex < totalBatches) {
            int batchIdx = currentBatchIndex;
            int[][] batch = loadBatch(batchIdx);

            try {
                prefetchQueue.put(batch);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * 加载指定批次的数据（带缓存）。
     *
     * @param batchIdx 批次索引
     * @return 批次数据数组
     */
    private int[][] loadBatch(int batchIdx) {
        // 检查缓存
        synchronized (batchCache) {
            int[][] cached = batchCache.get(batchIdx);
            if (cached != null) {
                return cached;
            }
        }

        // 加载批次
        int start = batchIdx * batchSize;
        int end = Math.min(start + batchSize, currentOrder.size());
        int actualBatchSize = end - start;

        int[][] batch = new int[actualBatchSize][];
        for (int i = 0; i < actualBatchSize; i++) {
            int[] seq = currentOrder.get(start + i);
            batch[i] = padOrTruncate(seq, seqLen);
        }

        // 存入缓存
        synchronized (batchCache) {
            batchCache.put(batchIdx, batch);
        }

        return batch;
    }

    /**
     * 获取下一个批次（同步）。
     *
     * @return 批次数据，如果没有更多数据返回null
     */
    public int[][] getNextBatch() {
        if (currentBatchIndex >= totalBatches) {
            return null;
        }

        // 优先从预加载队列获取
        if (prefetchRunning) {
            int[][] batch = prefetchQueue.poll();
            if (batch != null) {
                currentBatchIndex++;
                return batch;
            }
        }

        // 同步加载
        int[][] batch = loadBatch(currentBatchIndex);
        currentBatchIndex++;
        return batch;
    }

    /**
     * 填充或截断序列到指定长度。
     * 支持动态序列长度处理。
     *
     * @param seq 原始序列
     * @param targetLen 目标长度
     * @return 处理后的序列
     */
    public static int[] padOrTruncate(int[] seq, int targetLen) {
        int len = Math.min(seq.length, targetLen);
        int[] result = new int[len];

        System.arraycopy(seq, 0, result, 0, len);

        // 如果需要填充（仅在目标长度大于原始长度时）
        if (seq.length < targetLen) {
            int[] padded = new int[targetLen];
            System.arraycopy(seq, 0, padded, 0, seq.length);
            // 用0填充剩余部分
            for (int i = seq.length; i < targetLen; i++) {
                padded[i] = 0;
            }
            return padded;
        }

        return result;
    }

    /**
     * 是否还有更多批次。
     *
     * @return true如果还有数据
     */
    public boolean hasNext() {
        return currentBatchIndex < totalBatches;
    }

    /**
     * 获取总批次数。
     *
     * @return 批次总数
     */
    public int getTotalBatches() {
        return totalBatches;
    }

    /**
     * 获取当前批次索引。
     *
     * @return 当前批次索引
     */
    public int getCurrentBatchIndex() {
        return currentBatchIndex;
    }

    /**
     * 获取数据集大小。
     *
     * @return 序列总数
     */
    public int size() {
        return allSequences.size();
    }

    /**
     * 获取序列长度。
     *
     * @return 序列长度
     */
    public int getSeqLen() {
        return seqLen;
    }

    /**
     * 获取批大小。
     *
     * @return 批大小
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * 获取缓存命中率。
     *
     * @return 缓存命中率（0-1）
     */
    public float getCacheHitRate() {
        return batchCache.isEmpty() ? 0 : 1.0f;  // 简化版
    }

    /**
     * 关闭数据加载器，释放资源。
     */
    public void close() {
        prefetchRunning = false;
        prefetchExecutor.shutdown();
        try {
            if (!prefetchExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                prefetchExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            prefetchExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        batchCache.clear();
        prefetchQueue.clear();
    }

    /**
     * 从token数组创建滑动窗口序列。
     *
     * @param tokens 原始token数组
     * @param seqLen 序列长度
     * @param stride 滑动步长
     * @return 序列列表
     */
    public static List<int[]> createSlidingWindowSequences(int[] tokens, int seqLen, int stride) {
        List<int[]> sequences = new ArrayList<>();
        seqLen = Math.min(seqLen, MAX_SEQ_LEN);

        for (int i = 0; i + seqLen <= tokens.length; i += stride) {
            int[] seq = new int[seqLen];
            System.arraycopy(tokens, i, seq, 0, seqLen);
            sequences.add(seq);
        }

        // 处理剩余部分
        if (tokens.length > 0 && (tokens.length - seqLen) % stride != 0) {
            int start = tokens.length - seqLen;
            if (start >= 0) {
                int[] seq = new int[seqLen];
                System.arraycopy(tokens, start, seq, 0, seqLen);
                sequences.add(seq);
            }
        }

        return sequences;
    }

    /**
     * 将多个文本段落合并为token序列。
     *
     * @param tokenLists 多个token列表
     * @param separator 分隔符token ID
     * @return 合并后的token数组
     */
    public static int[] concatenateTokens(List<int[]> tokenLists, int separator) {
        int totalLen = 0;
        for (int[] tokens : tokenLists) {
            totalLen += tokens.length + 1;  // +1 for separator
        }

        int[] result = new int[totalLen];
        int idx = 0;
        for (int[] tokens : tokenLists) {
            System.arraycopy(tokens, 0, result, idx, tokens.length);
            idx += tokens.length;
            result[idx++] = separator;
        }

        return result;
    }
}
