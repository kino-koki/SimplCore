package com.qingxiaotuan.simpcore.concurrency;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public final class ThreadPoolManager {

    private static final ThreadPoolManager INSTANCE = new ThreadPoolManager();

    private final ThreadPoolExecutor computePool;
    private final ThreadPoolExecutor ioPool;
    private final ThreadPoolExecutor memoryPool;

    private ThreadPoolManager() {
        int cpuCores = Runtime.getRuntime().availableProcessors();

        computePool = new ThreadPoolExecutor(
            cpuCores,
            cpuCores * 2,
            60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(10000),
            new NamedThreadFactory("compute"),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );

        ioPool = new ThreadPoolExecutor(
            cpuCores,
            cpuCores * 4,
            30L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(10000),
            new NamedThreadFactory("io"),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );

        memoryPool = new ThreadPoolExecutor(
            Math.max(1, cpuCores / 2),
            cpuCores,
            30L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1000),
            new NamedThreadFactory("memory"),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );
    }

    public static ThreadPoolManager getInstance() {
        return INSTANCE;
    }

    public ThreadPoolExecutor getComputePool() {
        return computePool;
    }

    public ThreadPoolExecutor getIoPool() {
        return ioPool;
    }

    public ThreadPoolExecutor getMemoryPool() {
        return memoryPool;
    }

    public <T> CompletableFuture<T> submitCompute(Callable<T> task) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return task.call();
            } catch (Exception e) {
                throw new CompletionException(e);
            }
        }, computePool);
    }

    public CompletableFuture<Void> submitCompute(Runnable task) {
        return CompletableFuture.runAsync(task, computePool);
    }

    public <T> CompletableFuture<T> submitIo(Callable<T> task) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return task.call();
            } catch (Exception e) {
                throw new CompletionException(e);
            }
        }, ioPool);
    }

    public void shutdown() {
        computePool.shutdown();
        ioPool.shutdown();
        memoryPool.shutdown();
        try {
            if (!computePool.awaitTermination(60, TimeUnit.SECONDS)) {
                computePool.shutdownNow();
            }
            if (!ioPool.awaitTermination(60, TimeUnit.SECONDS)) {
                ioPool.shutdownNow();
            }
            if (!memoryPool.awaitTermination(60, TimeUnit.SECONDS)) {
                memoryPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            computePool.shutdownNow();
            ioPool.shutdownNow();
            memoryPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public PoolStats getStats() {
        return new PoolStats(
            computePool.getActiveCount(),
            computePool.getPoolSize(),
            computePool.getCompletedTaskCount(),
            ioPool.getActiveCount(),
            ioPool.getPoolSize(),
            ioPool.getCompletedTaskCount()
        );
    }

    public record PoolStats(
        int computeActive,
        int computePoolSize,
        long computeCompleted,
        int ioActive,
        int ioPoolSize,
        long ioCompleted
    ) {}

    private static class NamedThreadFactory implements ThreadFactory {
        private final String prefix;
        private final AtomicInteger counter = new AtomicInteger(0);

        NamedThreadFactory(String prefix) {
            this.prefix = prefix;
        }

        @Override
        public Thread newThread(Runnable r) {
            Thread thread = new Thread(r, prefix + "-" + counter.incrementAndGet());
            thread.setDaemon(true);
            thread.setPriority(Thread.NORM_PRIORITY);
            return thread;
        }
    }
}