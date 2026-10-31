package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.training.Optimizer;
import com.qingxiaotuan.simpcore.gpu.GPUDeviceManager;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class AdvancedDistributedTrainingFramework {

    public static class TrainingConfig {
        public int worldSize = 1;
        public int rank = 0;
        public String masterHost = "localhost";
        public int port = DistributedCommunicationLayer.DEFAULT_PORT;
        public int epochs = 10;
        public int batchSize = 32;
        public float learningRate = 0.001f;
        public float weightDecay = 0.01f;
        public int gradientAccumulationSteps = 1;
        public boolean useGradientClipping = true;
        public float maxGradientNorm = 1.0f;
        public ParallelStrategy parallelStrategy = ParallelStrategy.DATA_PARALLEL;
        public boolean enableMixedPrecision = false;
        public int logInterval = 10;
        public int saveInterval = 100;
    }

    public enum ParallelStrategy {
        DATA_PARALLEL,
        MODEL_PARALLEL,
        PIPELINE_PARALLEL,
        HYBRID
    }

    public static class WorkerStats {
        public final int rank;
        public final String gpuName;
        public volatile float loss;
        public volatile float accuracy;
        public volatile int stepsCompleted;
        public volatile long bytesProcessed;
        public volatile boolean active;

        public WorkerStats(int rank, String gpuName) {
            this.rank = rank;
            this.gpuName = gpuName;
            this.loss = Float.MAX_VALUE;
            this.accuracy = 0;
            this.stepsCompleted = 0;
            this.bytesProcessed = 0;
            this.active = true;
        }
    }

    private final TrainingConfig config;
    private final DistributedCommunicationLayer commLayer;
    private final Map<Integer, WorkerStats> workerStats = new ConcurrentHashMap<>();
    
    private volatile boolean running = false;
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicLong totalSamplesProcessed = new AtomicLong(0);
    
    private final ExecutorService trainingPool;
    private final ScheduledExecutorService statsScheduler;
    
    private SimpCoreModel globalModel;
    private Optimizer globalOptimizer;
    
    public AdvancedDistributedTrainingFramework(TrainingConfig config) {
        this.config = config;
        this.commLayer = new DistributedCommunicationLayer(config.rank, config.worldSize, config.masterHost, config.port);
        
        int poolSize = config.parallelStrategy == ParallelStrategy.DATA_PARALLEL 
            ? config.worldSize 
            : Math.max(2, config.worldSize / 2);
        this.trainingPool = Executors.newFixedThreadPool(poolSize);
        this.statsScheduler = Executors.newScheduledThreadPool(2);
    }

    public void initialize(SimpCoreModel model, Optimizer optimizer) throws Exception {
        this.globalModel = model;
        this.globalOptimizer = optimizer;
        
        commLayer.initialize();
        
        if (config.rank == 0) {
            initializeWorkers();
            startStatsCollection();
        }
        
        GPUDeviceManager.initialize();
        System.out.println("[Distributed] Framework initialized on rank " + config.rank);
    }

    private void initializeWorkers() {
        for (int i = 0; i < config.worldSize; i++) {
            if (i != config.rank) {
                String gpuName = "Unknown";
                if (GPUDeviceManager.isGPUAvailable()) {
                    GPUDeviceManager.DeviceInfo device = GPUDeviceManager.getSelectedDevice();
                    if (device != null) {
                        gpuName = device.name;
                    }
                }
                workerStats.put(i, new WorkerStats(i, gpuName));
            }
        }
    }

    private void startStatsCollection() {
        statsScheduler.scheduleAtFixedRate(() -> {
            if (!running) return;
            
            StringBuilder sb = new StringBuilder();
            sb.append("[Distributed] Global Step ").append(globalStep.get()).append("\n");
            sb.append("  Samples: ").append(totalSamplesProcessed.get()).append("\n");
            
            for (WorkerStats stats : workerStats.values()) {
                sb.append("  Worker ").append(stats.rank).append(": ");
                sb.append("Loss=").append(String.format("%.4f", stats.loss)).append(", ");
                sb.append("Active=").append(stats.active).append("\n");
            }
            
            System.out.println(sb.toString());
        }, 10, config.logInterval, TimeUnit.SECONDS);
    }

    public void train(Dataset dataset) {
        running = true;
        
        if (config.rank == 0) {
            trainMaster(dataset);
        } else {
            trainWorker(dataset);
        }
        
        running = false;
    }

    private void trainMaster(Dataset dataset) {
        int totalBatches = dataset.size() / config.batchSize;
        int batchesPerWorker = totalBatches / config.worldSize;
        
        List<Future<TrainingResult>> futures = new ArrayList<>();
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            dataset.shuffle();
            
            for (int batchIdx = 0; batchIdx < batchesPerWorker; batchIdx++) {
                int globalBatchIdx = batchIdx * config.worldSize;
                
                for (int workerRank : workerStats.keySet()) {
                    if (!workerStats.get(workerRank).active) continue;
                    
                    int start = globalBatchIdx * config.batchSize;
                    int end = start + config.batchSize;
                    Dataset batch = dataset.slice(start, end);
                    
                    futures.add(trainingPool.submit(() -> {
                        return executeWorkerStep(workerRank, batch);
                    }));
                }
                
                aggregateAndUpdate(futures);
                futures.clear();
                
                if (globalStep.incrementAndGet() % config.logInterval == 0) {
                    System.out.printf("[Distributed] Epoch %d, Step %d: Loss = %.6f%n",
                        epoch, globalStep.get(), workerStats.values().stream()
                            .mapToDouble(s -> s.loss).average().orElse(0));
                }
                
                if (globalStep.get() % config.saveInterval == 0) {
                    saveCheckpoint(epoch);
                }
            }
        }
    }

    private TrainingResult executeWorkerStep(int workerRank, Dataset batch) {
        try {
            float loss = computeLoss(globalModel, batch);
            float[] gradients = computeGradients(globalModel, batch);
            
            commLayer.allReduce(gradients, DistributedCommunicationLayer.ReduceOp.AVG);
            
            globalOptimizer.update(globalModel.getOutputWeights(), gradients);
            
            WorkerStats stats = workerStats.get(workerRank);
            if (stats != null) {
                stats.loss = loss;
                stats.stepsCompleted++;
                stats.bytesProcessed += batch.size() * config.batchSize * 4;
            }
            
            return new TrainingResult(loss, 0, workerRank);
        } catch (Exception e) {
            System.err.println("[Distributed] Worker " + workerRank + " failed: " + e.getMessage());
            WorkerStats stats = workerStats.get(workerRank);
            if (stats != null) {
                stats.active = false;
            }
            return new TrainingResult(Float.MAX_VALUE, 0, workerRank);
        }
    }

    private void aggregateAndUpdate(List<Future<TrainingResult>> futures) {
        float totalLoss = 0;
        int count = 0;
        
        for (Future<TrainingResult> future : futures) {
            try {
                TrainingResult result = future.get(60, TimeUnit.SECONDS);
                totalLoss += result.averageLoss;
                count++;
                
                WorkerStats stats = workerStats.get(result.workerId);
                if (stats != null) {
                    stats.loss = result.averageLoss;
                }
            } catch (Exception e) {
                System.err.println("[Distributed] Aggregation failed: " + e.getMessage());
            }
        }
        
        if (count > 0) {
            float avgLoss = totalLoss / count;
            for (WorkerStats stats : workerStats.values()) {
                if (stats.active) {
                    stats.loss = avgLoss;
                }
            }
        }
        
        broadcastModel();
    }

    private void broadcastModel() {
        if (config.worldSize > 1) {
            float[] modelParams = globalModel.getOutputWeights();
            commLayer.broadcast(modelParams);
        }
    }

    private void trainWorker(Dataset dataset) {
        int totalBatches = dataset.size() / config.batchSize;
        int batchesPerWorker = totalBatches / config.worldSize;
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            for (int batchIdx = 0; batchIdx < batchesPerWorker && running; batchIdx++) {
                int start = batchIdx * config.batchSize;
                int end = start + config.batchSize;
                Dataset batch = dataset.slice(start, end);
                
                float loss = computeLoss(globalModel, batch);
                float[] gradients = computeGradients(globalModel, batch);
                
                commLayer.allReduce(gradients, DistributedCommunicationLayer.ReduceOp.AVG);
                globalOptimizer.update(globalModel.getOutputWeights(), gradients);
                
                globalStep.incrementAndGet();
                
                if (batchIdx % config.logInterval == 0) {
                    System.out.printf("[Worker %d] Epoch %d, Batch %d: Loss = %.6f%n",
                        config.rank, epoch, batchIdx, loss);
                }
            }
        }
    }

    public void trainWithModelParallelism(Dataset dataset) {
        if (config.parallelStrategy != ParallelStrategy.MODEL_PARALLEL) {
            train(dataset);
            return;
        }
        
        running = true;
        
        int layersPerWorker = globalModel.getNLayers() / config.worldSize;
        int startLayer = config.rank * layersPerWorker;
        int endLayer = Math.min(startLayer + layersPerWorker, globalModel.getNLayers());
        
        System.out.println("[Distributed] Model Parallelism: Worker " + config.rank + 
            " handles layers " + startLayer + "-" + endLayer);
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            dataset.shuffle();
            
            for (int i = 0; i < dataset.size(); i += config.batchSize) {
                int end = Math.min(i + config.batchSize, dataset.size());
                Dataset batch = dataset.slice(i, end);
                
                float[] input = prepareBatch(batch);
                
                if (config.rank == 0) {
                    input = globalModel.forwardPartial(input, startLayer, endLayer);
                    for (int targetRank = 1; targetRank < config.worldSize; targetRank++) {
                        commLayer.broadcast(input);
                    }
                } else {
                    CompletableFuture<float[]> receiveFuture = new CompletableFuture<>();
                    try {
                        input = receiveFuture.get(30, TimeUnit.SECONDS);
                        input = globalModel.forwardPartial(input, startLayer, endLayer);
                        
                        if (config.rank < config.worldSize - 1) {
                            commLayer.broadcast(input);
                        }
                    } catch (Exception e) {
                        System.err.println("[Distributed] Model parallel forward failed: " + e.getMessage());
                    }
                }
                
                if (config.rank == config.worldSize - 1) {
                    float loss = computeLossFromOutput(input, batch);
                    float[] gradients = computeGradientsFromOutput(input, batch);
                    
                    for (int sourceRank = config.worldSize - 2; sourceRank >= 0; sourceRank--) {
                        commLayer.broadcast(gradients);
                    }
                    
                    globalOptimizer.update(globalModel.getOutputWeights(), gradients);
                } else {
                    CompletableFuture<float[]> gradFuture = new CompletableFuture<>();
                    try {
                        float[] gradients = gradFuture.get(30, TimeUnit.SECONDS);
                        gradients = globalModel.backwardPartial(gradients, endLayer, startLayer);
                        
                        if (config.rank > 0) {
                            commLayer.broadcast(gradients);
                        }
                    } catch (Exception e) {
                        System.err.println("[Distributed] Model parallel backward failed: " + e.getMessage());
                    }
                }
                
                globalStep.incrementAndGet();
                totalSamplesProcessed.addAndGet(batch.size());
            }
        }
        
        running = false;
    }

    public void trainWithPipelineParallelism(Dataset dataset) {
        if (config.parallelStrategy != ParallelStrategy.PIPELINE_PARALLEL) {
            train(dataset);
            return;
        }
        
        running = true;
        
        int stages = config.worldSize;
        int microBatches = 4;
        
        System.out.println("[Distributed] Pipeline Parallelism: " + stages + " stages, " + microBatches + " micro-batches");
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            for (int i = 0; i < dataset.size(); i += config.batchSize) {
                List<Dataset> microBatchesList = splitIntoMicroBatches(dataset.slice(i, Math.min(i + config.batchSize, dataset.size())), microBatches);
                
                for (int microIdx = 0; microIdx < microBatches; microIdx++) {
                    Dataset microBatch = microBatchesList.get(microIdx);
                    
                    int stage = microIdx % stages;
                    
                    if (stage == config.rank) {
                        float[] input = prepareBatch(microBatch);
                        input = processPipelineStage(input, stage);
                        
                        if (stage < stages - 1) {
                            commLayer.broadcast(input);
                        }
                    }
                    
                    if (stage > 0 && stage - 1 == config.rank) {
                        CompletableFuture<float[]> receiveFuture = new CompletableFuture<>();
                        try {
                            float[] input = receiveFuture.get(30, TimeUnit.SECONDS);
                            input = processPipelineStage(input, stage);
                            
                            if (stage < stages - 1) {
                                commLayer.broadcast(input);
                            }
                        } catch (Exception e) {
                            System.err.println("[Distributed] Pipeline stage failed: " + e.getMessage());
                        }
                    }
                    
                    if (stage == stages - 1 && stage == config.rank) {
                        float loss = computeLossFromOutput(input, microBatch);
                        float[] gradients = computeGradientsFromOutput(input, microBatch);
                        globalOptimizer.update(globalModel.getOutputWeights(), gradients);
                        
                        globalStep.incrementAndGet();
                    }
                }
            }
        }
        
        running = false;
    }

    private List<Dataset> splitIntoMicroBatches(Dataset batch, int numMicroBatches) {
        List<Dataset> microBatches = new ArrayList<>();
        int microSize = batch.size() / numMicroBatches;
        
        for (int i = 0; i < numMicroBatches; i++) {
            int start = i * microSize;
            int end = (i == numMicroBatches - 1) ? batch.size() : start + microSize;
            microBatches.add(batch.slice(start, end));
        }
        
        return microBatches;
    }

    private float[] processPipelineStage(float[] input, int stage) {
        int layersPerStage = globalModel.getNLayers() / config.worldSize;
        int startLayer = stage * layersPerStage;
        int endLayer = Math.min(startLayer + layersPerStage, globalModel.getNLayers());
        
        return globalModel.forwardPartial(input, startLayer, endLayer);
    }

    private float[] prepareBatch(Dataset batch) {
        List<float[]> data = batch.getData();
        float[] input = new float[data.size() * data.get(0).length];
        
        for (int i = 0; i < data.size(); i++) {
            System.arraycopy(data.get(i), 0, input, i * data.get(0).length, data.get(0).length);
        }
        
        return input;
    }

    private float computeLoss(SimpCoreModel model, Dataset batch) {
        float[] input = prepareBatch(batch);
        float[] output = model.forward(input, batch.size());
        
        float loss = 0;
        List<Integer> labels = batch.getLabels();
        for (int i = 0; i < labels.size(); i++) {
            int label = labels.get(i);
            float predicted = output[i * model.getVocabSize() + label];
            loss -= Math.log(Math.max(predicted, 1e-10f));
        }
        
        return loss / labels.size();
    }

    private float[] computeGradients(SimpCoreModel model, Dataset batch) {
        float[] input = prepareBatch(batch);
        float[] output = model.forward(input, batch.size());
        
        int vocabSize = model.getVocabSize();
        float[] gradients = new float[output.length];
        
        List<Integer> labels = batch.getLabels();
        
        for (int i = 0; i < labels.size(); i++) {
            int start = i * vocabSize;
            float[] logits = new float[vocabSize];
            System.arraycopy(output, start, logits, 0, vocabSize);
            
            float[] softmax = softmax(logits);
            int label = labels.get(i);
            
            for (int j = 0; j < vocabSize; j++) {
                gradients[start + j] = softmax[j] - (j == label ? 1.0f : 0.0f);
            }
        }
        
        return gradients;
    }

    private float computeLossFromOutput(float[] output, Dataset batch) {
        float loss = 0;
        List<Integer> labels = batch.getLabels();
        int vocabSize = globalModel.getVocabSize();
        
        for (int i = 0; i < labels.size(); i++) {
            int label = labels.get(i);
            float predicted = output[i * vocabSize + label];
            loss -= Math.log(Math.max(predicted, 1e-10f));
        }
        
        return loss / labels.size();
    }

    private float[] computeGradientsFromOutput(float[] output, Dataset batch) {
        int vocabSize = globalModel.getVocabSize();
        float[] gradients = new float[output.length];
        
        List<Integer> labels = batch.getLabels();
        
        for (int i = 0; i < labels.size(); i++) {
            int start = i * vocabSize;
            float[] logits = new float[vocabSize];
            System.arraycopy(output, start, logits, 0, vocabSize);
            
            float[] softmax = softmax(logits);
            int label = labels.get(i);
            
            for (int j = 0; j < vocabSize; j++) {
                gradients[start + j] = softmax[j] - (j == label ? 1.0f : 0.0f);
            }
        }
        
        return gradients;
    }
    
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            max = Math.max(max, val);
        }
        
        float[] exp = new float[logits.length];
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        float[] result = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            result[i] = exp[i] / sum;
        }
        
        return result;
    }

    public void saveCheckpoint(int epoch) {
        if (config.rank != 0) return;
        
        try {
            String checkpointPath = String.format("checkpoint_epoch_%d_rank_%d.bin", epoch, config.rank);
            globalModel.save(checkpointPath);
            System.out.println("[Distributed] Saved checkpoint: " + checkpointPath);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to save checkpoint: " + e.getMessage());
        }
    }

    public void loadCheckpoint(String path) {
        try {
            globalModel.load(path);
            System.out.println("[Distributed] Loaded checkpoint: " + path);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to load checkpoint: " + e.getMessage());
        }
    }

    public void shutdown() {
        running = false;
        
        trainingPool.shutdown();
        statsScheduler.shutdown();
        
        try {
            if (!trainingPool.awaitTermination(10, TimeUnit.SECONDS)) {
                trainingPool.shutdownNow();
            }
            if (!statsScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                statsScheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            trainingPool.shutdownNow();
            statsScheduler.shutdownNow();
        }
        
        commLayer.shutdown();
        System.out.println("[Distributed] Framework shutdown complete");
    }

    public int getGlobalStep() {
        return globalStep.get();
    }

    public long getTotalSamplesProcessed() {
        return totalSamplesProcessed.get();
    }

    public WorkerStats getWorkerStats(int rank) {
        return workerStats.get(rank);
    }

    public Map<Integer, WorkerStats> getAllWorkerStats() {
        return Collections.unmodifiableMap(workerStats);
    }

    public static class Dataset {
        private final List<float[]> data;
        private final List<Integer> labels;
        
        public Dataset() {
            this.data = new ArrayList<>();
            this.labels = new ArrayList<>();
        }
        
        public void add(float[] item, int label) {
            data.add(item);
            labels.add(label);
        }
        
        public int size() {
            return data.size();
        }
        
        public Dataset slice(int start, int end) {
            Dataset slice = new Dataset();
            for (int i = start; i < end && i < data.size(); i++) {
                slice.add(data.get(i), labels.get(i));
            }
            return slice;
        }
        
        public void shuffle() {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < data.size(); i++) {
                indices.add(i);
            }
            Collections.shuffle(indices);
            
            List<float[]> newData = new ArrayList<>();
            List<Integer> newLabels = new ArrayList<>();
            for (int idx : indices) {
                newData.add(data.get(idx));
                newLabels.add(labels.get(idx));
            }
            
            data.clear();
            data.addAll(newData);
            labels.clear();
            labels.addAll(newLabels);
        }
        
        public List<float[]> getData() {
            return data;
        }
        
        public List<Integer> getLabels() {
            return labels;
        }
    }

    public static class TrainingResult {
        public final float averageLoss;
        public final float accuracy;
        public final int workerId;
        
        public TrainingResult(float averageLoss, float accuracy, int workerId) {
            this.averageLoss = averageLoss;
            this.accuracy = accuracy;
            this.workerId = workerId;
        }
    }
}