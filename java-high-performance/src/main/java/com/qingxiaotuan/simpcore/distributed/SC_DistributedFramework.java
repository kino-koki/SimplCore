package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.training.*;
import com.qingxiaotuan.simpcore.math.MatrixOperations;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

interface SCTrainingDataPipeline {
    String nextBatch(int size);
}

public class SC_DistributedFramework {

    public static class Worker {
        public final int id;
        public final String gpuType;
        public final SimpCoreModel model;
        public final Tokenizer tokenizer;
        public final Optimizer optimizer;
        public final SCTrainingDataPipeline dataPipeline;
        public volatile float localLoss;
        public volatile boolean active;
        
        public Worker(int id, String gpuType, SimpCoreModel model, Tokenizer tokenizer,
                      Optimizer optimizer, SCTrainingDataPipeline pipeline) {
            this.id = id;
            this.gpuType = gpuType;
            this.model = model;
            this.tokenizer = tokenizer;
            this.optimizer = optimizer;
            this.dataPipeline = pipeline;
            this.localLoss = Float.MAX_VALUE;
            this.active = true;
        }
    }

    private final List<Worker> workers = new ArrayList<>();
    private final ExecutorService workerPool;
    private final CyclicBarrier stepBarrier;
    private final AtomicInteger activeWorkers;
    private final AtomicInteger completedSteps;
    private volatile boolean running;
    private volatile float globalLoss;
    private final int worldSize;

    public SC_DistributedFramework(int worldSize) {
        this.worldSize = worldSize;
        this.workerPool = Executors.newFixedThreadPool(worldSize);
        this.stepBarrier = new CyclicBarrier(worldSize);
        this.activeWorkers = new AtomicInteger(worldSize);
        this.completedSteps = new AtomicInteger(0);
        this.running = false;
        System.out.println("[SC-Distributed] Framework initialized: " + worldSize + " workers");
    }

    public void addWorker(int id, String gpuType, SimpCoreModel model, Tokenizer tokenizer,
                          Optimizer optimizer, SCTrainingDataPipeline pipeline) {
        workers.add(new Worker(id, gpuType, model, tokenizer, optimizer, pipeline));
    }

    public void train(int totalSteps, int batchSize, PrintWriter log) {
        running = true;
        log.println("[SC-Distributed] Starting distributed training: " + totalSteps + " steps");
        
        for (Worker w : workers) {
            log.println("  Worker " + w.id + ": " + w.gpuType);
        }

        long startTime = System.currentTimeMillis();
        for (int step = 1; step <= totalSteps && running; step++) {
            globalLoss = 0;
            final int stepNum = step;

            for (Worker worker : workers) {
                if (!worker.active) continue;
                
                workerPool.submit(() -> {
                    try {
                        String input = worker.dataPipeline.nextBatch(batchSize);
                        TrainingLoop loop = new TrainingLoop(worker.model, worker.tokenizer, worker.optimizer);
                        float loss = loop.trainStep(input, 1);
                        synchronized (this) {
                            globalLoss += loss;
                        }
                    } catch (Exception e) {
                        System.err.println("[Worker " + worker.id + "] Error: " + e.getMessage());
                        worker.active = false;
                        activeWorkers.decrementAndGet();
                    }
                    
                    try {
                        stepBarrier.await(30, TimeUnit.SECONDS);
                    } catch (Exception e) {}
                });
            }

            try {
                stepBarrier.await(60, TimeUnit.SECONDS);
            } catch (Exception e) {
                break;
            }

            globalLoss /= Math.max(1, activeWorkers.get());

            if (step % 10 == 0) {
                long elapsed = System.currentTimeMillis() - startTime;
                double stepsPerSec = step / (elapsed / 1000.0);
                log.printf("[Step %d] Loss=%.6f | Active=%d/%d | %.2f steps/s%n", 
                    step, globalLoss, activeWorkers.get(), worldSize, stepsPerSec);
                log.flush();
            }

            completedSteps.incrementAndGet();
        }

        running = false;
        long elapsed = System.currentTimeMillis() - startTime;
        log.println("[SC-Distributed] Training completed: " + completedSteps.get() + " steps in " + (elapsed/1000) + "s");
    }

    public void shutdown() {
        running = false;
        workerPool.shutdown();
    }

    public int getCompletedSteps() { return completedSteps.get(); }
    public float getGlobalLoss() { return globalLoss; }

    public static class ParameterExplosionConfig {
        public int dModel = 512;
        public int nLayers = 8;
        public int nHeads = 8;
        public int nKVHeads = 2;
        public int vocabSize = 64000;
        public int maxSeqLen = 4096;
        public boolean useMQA = true;
        public boolean useGQA = false;
        public int ffnMultiplier = 4;
        
        public long calculateParams() {
            long params = 0;
            params += (long) vocabSize * dModel;
            params += (long) nLayers * dModel * dModel * 3;
            params += (long) nLayers * dModel * dModel;
            params += (long) nLayers * dModel * dModel * ffnMultiplier * 2;
            params += (long) nLayers * dModel * 4;
            params += (long) dModel * vocabSize;
            return params;
        }
        
        public void print(PrintWriter log) {
            log.println("=== Parameter Explosion Config ===");
            log.println("  dModel: " + dModel);
            log.println("  nLayers: " + nLayers);
            log.println("  nHeads: " + nHeads);
            log.println("  nKVHeads: " + nKVHeads);
            log.println("  vocabSize: " + vocabSize);
            log.println("  maxSeqLen: " + maxSeqLen);
            log.println("  MQA: " + useMQA);
            log.println("  GQA: " + useGQA);
            log.println("  FFN Multiplier: " + ffnMultiplier);
            log.println("  Total Params: " + String.format("%.2fB", calculateParams() / 1e9));
        }
    }

    public static void main(String[] args) throws Exception {
        PrintWriter log = new PrintWriter(new FileOutputStream("sc_distributed_log.txt"), true);
        
        ParameterExplosionConfig peConfig = new ParameterExplosionConfig();
        peConfig.dModel = 128;
        peConfig.nLayers = 4;
        peConfig.nHeads = 4;
        peConfig.vocabSize = 128;
        peConfig.maxSeqLen = 32;
        peConfig.print(log);

        log.println("\n[SC-Distributed] Building workers...");
        SCTrainingDataPipeline pipeline = new SCTrainingDataPipeline() {
            final String[] data = {"hello world this is a test", "deep learning is amazing", 
                "AI changes everything", "the quick brown fox", "i love machine learning"};
            int idx = 0;
            @Override public String nextBatch(int size) { return data[(idx++) % data.length]; }
        };

        SC_DistributedFramework framework = new SC_DistributedFramework(2);
        
        for (int i = 0; i < 2; i++) {
            Tokenizer tokenizer = new Tokenizer(peConfig.vocabSize);
            SimpCoreModel model = new SimpCoreModel(peConfig.vocabSize, peConfig.dModel, 
                peConfig.nLayers, 1, peConfig.nHeads, 1, 1, 1, peConfig.maxSeqLen);
            Optimizer opt = new Optimizer(0.001f, 0.01f, 0.9f, Float.MAX_VALUE);
            framework.addWorker(i, "GPU-" + i, model, tokenizer, opt, pipeline);
        }

        log.println("\n[SC-Distributed] Starting training...");
        framework.train(20, 1, log);
        
        log.println("\n[SC-Distributed] Training finished. Final loss: " + framework.getGlobalLoss());
        log.close();
    }
}