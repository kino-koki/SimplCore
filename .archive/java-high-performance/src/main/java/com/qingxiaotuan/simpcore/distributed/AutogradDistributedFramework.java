package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.autograd.*;
import com.qingxiaotuan.simpcore.gpu.GPUDeviceManager;
import com.qingxiaotuan.simpcore.model.SimpCoreModel;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class AutogradDistributedFramework {

    public static class TrainingConfig {
        public int worldSize = 1;
        public int rank = 0;
        public String masterHost = "localhost";
        public int port = DistributedCommunicationLayer.DEFAULT_PORT;
        public int epochs = 10;
        public int batchSize = 32;
        public int microBatchSize = 8;
        public float learningRate = 0.001f;
        public float weightDecay = 0.01f;
        public int gradientAccumulationSteps = 1;
        public boolean useGradientClipping = true;
        public float maxGradientNorm = 1.0f;
        public ParallelStrategy parallelStrategy = ParallelStrategy.DATA_PARALLEL;
        public boolean enableMixedPrecision = false;
        public int logInterval = 10;
        public int saveInterval = 100;
        public boolean useGPU = true;
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
        public volatile float gradNorm;
        public volatile float learningRate;
        public volatile int stepsCompleted;
        public volatile long bytesProcessed;
        public volatile boolean active;
        public volatile float gpuUtilization;
        public volatile long gpuMemoryUsed;

        public WorkerStats(int rank, String gpuName) {
            this.rank = rank;
            this.gpuName = gpuName;
            this.loss = Float.MAX_VALUE;
            this.gradNorm = 0;
            this.learningRate = 0;
            this.stepsCompleted = 0;
            this.bytesProcessed = 0;
            this.active = true;
            this.gpuUtilization = 0;
            this.gpuMemoryUsed = 0;
        }
    }

    public static class TrainingMetrics {
        public final List<Float> lossHistory = new ArrayList<>();
        public final List<Float> gradNormHistory = new ArrayList<>();
        public final List<Float> lrHistory = new ArrayList<>();
        public final List<Long> stepTimes = new ArrayList<>();
        public final List<Float> gpuUtilHistory = new ArrayList<>();
        public final List<Long> memoryUsageHistory = new ArrayList<>();
        public volatile float currentLoss = Float.MAX_VALUE;
        public volatile float currentGradNorm = 0;
        public volatile int globalStep = 0;
        public volatile long samplesProcessed = 0;

        public synchronized void record(float loss, float gradNorm, float lr, long stepTime, 
                                        float gpuUtil, long memoryUsage) {
            lossHistory.add(loss);
            gradNormHistory.add(gradNorm);
            lrHistory.add(lr);
            stepTimes.add(stepTime);
            gpuUtilHistory.add(gpuUtil);
            memoryUsageHistory.add(memoryUsage);
            currentLoss = loss;
            currentGradNorm = gradNorm;
        }

        public synchronized float getAverageLoss(int recentCount) {
            if (lossHistory.isEmpty()) return Float.MAX_VALUE;
            int start = Math.max(0, lossHistory.size() - recentCount);
            double sum = 0;
            int count = 0;
            for (int i = start; i < lossHistory.size(); i++) {
                sum += lossHistory.get(i);
                count++;
            }
            return (float) (sum / count);
        }

        public synchronized String getSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Training Metrics Summary:\n");
            sb.append("  Global Step: ").append(globalStep).append("\n");
            sb.append("  Samples Processed: ").append(samplesProcessed).append("\n");
            sb.append("  Current Loss: ").append(String.format("%.6f", currentLoss)).append("\n");
            sb.append("  Current Gradient Norm: ").append(String.format("%.4f", currentGradNorm)).append("\n");
            if (!lossHistory.isEmpty()) {
                sb.append("  Average Loss (last 100): ").append(String.format("%.6f", getAverageLoss(100))).append("\n");
            }
            if (!stepTimes.isEmpty()) {
                double avgTime = stepTimes.stream().mapToLong(Long::longValue).average().orElse(0);
                sb.append("  Average Step Time: ").append(String.format("%.2f ms", avgTime)).append("\n");
            }
            return sb.toString();
        }
    }

    private final TrainingConfig config;
    private final DistributedCommunicationLayer commLayer;
    private final Map<Integer, WorkerStats> workerStats = new ConcurrentHashMap<>();
    private final TrainingMetrics metrics = new TrainingMetrics();
    
    private volatile boolean running = false;
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicLong totalSamplesProcessed = new AtomicLong(0);
    private final AtomicLong totalTrainingTime = new AtomicLong(0);
    
    private final ExecutorService trainingPool;
    private final ScheduledExecutorService statsScheduler;
    
    private float[] globalGradients;
    private float[] globalParameters;
    private int parameterCount = 0;
    
    public AutogradDistributedFramework(TrainingConfig config) {
        this.config = config;
        this.commLayer = new DistributedCommunicationLayer(config.rank, config.worldSize, config.masterHost, config.port);
        
        int poolSize = config.parallelStrategy == ParallelStrategy.DATA_PARALLEL 
            ? Math.max(2, config.worldSize) 
            : Math.max(4, config.worldSize * 2);
        this.trainingPool = Executors.newFixedThreadPool(poolSize);
        this.statsScheduler = Executors.newScheduledThreadPool(2);
    }

    public void initialize(SimpCoreModel model) throws Exception {
        commLayer.initialize();
        
        if (config.rank == 0) {
            initializeWorkers();
            startStatsCollection();
        }
        
        GPUDeviceManager.initialize();
        
        parameterCount = model.getDModel() * model.getVocabSize();
        globalGradients = new float[parameterCount];
        globalParameters = new float[parameterCount];
        System.arraycopy(model.getOutputWeights(), 0, globalParameters, 0, parameterCount);
        
        System.out.println("[Distributed] Autograd framework initialized on rank " + config.rank);
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
            
            float gpuUtil = 0;
            long gpuMemory = 0;
            if (GPUDeviceManager.isGPUAvailable()) {
                GPUDeviceManager.DeviceInfo device = GPUDeviceManager.getSelectedDevice();
                if (device != null) {
                    gpuMemory = device.memoryTotal;
                }
            }
            
            metrics.gpuUtilHistory.add(gpuUtil);
            metrics.memoryUsageHistory.add(gpuMemory);
            
            if (globalStep.get() % config.logInterval == 0) {
                System.out.println(metrics.getSummary());
            }
        }, 5, config.logInterval, TimeUnit.SECONDS);
    }

    public void train(Dataset dataset) {
        running = true;
        
        if (config.parallelStrategy == ParallelStrategy.MODEL_PARALLEL) {
            trainWithModelParallelism(dataset);
        } else if (config.parallelStrategy == ParallelStrategy.PIPELINE_PARALLEL) {
            trainWithPipelineParallelism(dataset);
        } else {
            trainDataParallel(dataset);
        }
        
        running = false;
    }

    private void trainDataParallel(Dataset dataset) {
        int totalBatches = dataset.size() / config.batchSize;
        int batchesPerWorker = totalBatches / config.worldSize;
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            dataset.shuffle();
            
            for (int batchIdx = 0; batchIdx < batchesPerWorker && running; batchIdx++) {
                long stepStartTime = System.nanoTime();
                
                int start = batchIdx * config.batchSize * config.worldSize + config.rank * config.batchSize;
                int end = start + config.batchSize;
                Dataset batch = dataset.slice(start, end);
                
                Tensor loss = executeTrainingStep(batch);
                
                float[] gradients = computeGradientsFromTensor(loss);
                
                commLayer.allReduce(gradients, DistributedCommunicationLayer.ReduceOp.AVG);
                
                updateParameters(gradients, config.learningRate);
                
                long stepTime = (System.nanoTime() - stepStartTime) / 1_000_000;
                
                int step = globalStep.incrementAndGet();
                totalSamplesProcessed.addAndGet(batch.size());
                
                WorkerStats stats = workerStats.get(config.rank);
                if (stats != null) {
                    stats.loss = loss.data[0];
                    stats.gradNorm = computeGradientNorm(gradients);
                    stats.stepsCompleted++;
                    stats.learningRate = config.learningRate;
                }
                
                metrics.globalStep = step;
                metrics.samplesProcessed = totalSamplesProcessed.get();
                metrics.record(loss.data[0], stats != null ? stats.gradNorm : 0, 
                              config.learningRate, stepTime, 0, 0);
                
                if (step % config.logInterval == 0) {
                    System.out.printf("[Rank %d] Epoch %d, Step %d: Loss = %.6f, GradNorm = %.4f, Time = %.2f ms%n",
                        config.rank, epoch, step, loss.data[0], 
                        stats != null ? stats.gradNorm : 0, stepTime);
                }
                
                loss = null;
                gradients = null;
            }
        }
    }

    private Tensor executeTrainingStep(Dataset batch) {
        List<float[]> data = batch.getData();
        List<Integer> labels = batch.getLabels();
        int batchSize = data.size();
        int inputDim = data.get(0).length;
        
        float[] inputData = new float[batchSize * inputDim];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(data.get(i), 0, inputData, i * inputDim, inputDim);
        }
        
        int[] targetArray = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            targetArray[i] = labels.get(i);
        }
        
        Tensor input = new Tensor(inputData, batchSize, inputDim);
        input.requiresGrad = false;
        
        Tensor weight = new Tensor(globalParameters.clone(), inputDim, 10);
        weight.requiresGrad = true;
        
        Tensor logits = AutogradEngine.matmul(input, weight);
        
        return AutogradEngine.crossEntropyLossTensor(logits, targetArray);
    }

    private float[] computeGradientsFromTensor(Tensor loss) {
        loss.backward();
        
        float[] gradients = new float[parameterCount];
        Arrays.fill(gradients, 0);
        
        collectGradients(loss, gradients);
        
        return gradients;
    }

    private void collectGradients(Tensor tensor, float[] gradients) {
        if (tensor.gradFn != null) {
            for (Tensor input : tensor.gradFn.inputs) {
                if (input.requiresGrad && input.grad != null) {
                    System.arraycopy(input.grad, 0, gradients, 0, Math.min(input.grad.length, gradients.length));
                }
                collectGradients(input, gradients);
            }
        }
    }

    private float computeGradientNorm(float[] gradients) {
        float sum = 0;
        for (float g : gradients) {
            sum += g * g;
        }
        return (float) Math.sqrt(sum);
    }

    private void updateParameters(float[] gradients, float lr) {
        for (int i = 0; i < parameterCount; i++) {
            globalParameters[i] -= lr * gradients[i];
        }
    }

    public void trainWithModelParallelism(Dataset dataset) {
        running = true;
        
        int layersPerWorker = 4;
        int startLayer = config.rank * layersPerWorker;
        int endLayer = Math.min(startLayer + layersPerWorker, 12);
        
        System.out.println("[Distributed] Model Parallelism: Worker " + config.rank + 
            " handles layers " + startLayer + "-" + endLayer);
        
        int totalBatches = dataset.size() / config.batchSize;
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            dataset.shuffle();
            
            for (int batchIdx = 0; batchIdx < totalBatches && running; batchIdx++) {
                int start = batchIdx * config.batchSize;
                int end = Math.min(start + config.batchSize, dataset.size());
                Dataset batch = dataset.slice(start, end);
                
                float[] inputData = prepareBatchData(batch);
                Tensor inputTensor = new Tensor(inputData, config.batchSize, 64);
                inputTensor.requiresGrad = true;
                
                Tensor output = forwardPassModelParallel(inputTensor, startLayer, endLayer);
                
                float[] targetArray = prepareTargets(batch);
                Tensor loss = AutogradEngine.crossEntropyLossTensor(output, convertToIntArray(targetArray));
                
                float[] gradients = backwardPassModelParallel(loss, startLayer, endLayer);
                
                if (config.rank == 0) {
                    updateParameters(gradients, config.learningRate);
                    broadcastParameters();
                } else {
                    receiveParameters();
                }
                
                int step = globalStep.incrementAndGet();
                
                if (step % config.logInterval == 0) {
                    System.out.printf("[Rank %d] Epoch %d, Step %d: Loss = %.6f%n",
                        config.rank, epoch, step, loss.data[0]);
                }
            }
        }
        
        running = false;
    }

    private Tensor forwardPassModelParallel(Tensor input, int startLayer, int endLayer) {
        Tensor x = input;
        
        for (int l = startLayer; l < endLayer; l++) {
            int hiddenDim = 256;
            Tensor w1 = new Tensor(new float[x.numel * hiddenDim], x.shape[1], hiddenDim);
            w1.requiresGrad = true;
            Tensor w2 = new Tensor(new float[hiddenDim * 10], hiddenDim, 10);
            w2.requiresGrad = true;
            
            Tensor hidden = AutogradEngine.matmul(x, w1);
            hidden = AutogradEngine.silu(hidden);
            x = AutogradEngine.matmul(hidden, w2);
            
            if (l < endLayer - 1) {
                x = AutogradEngine.layerNorm(x, null, null);
            }
        }
        
        return x;
    }

    private float[] backwardPassModelParallel(Tensor loss, int startLayer, int endLayer) {
        loss.backward();
        
        float[] gradients = new float[parameterCount];
        
        if (config.rank == config.worldSize - 1) {
            gradients = collectAllGradients(loss);
        } else {
            float[] gradFromNext = receiveGradientFromNext();
            propagateGradients(loss, gradFromNext);
            gradients = collectAllGradients(loss);
            sendGradientToPrev(gradients);
        }
        
        return gradients;
    }

    private float[] collectAllGradients(Tensor tensor) {
        List<Float> allGradients = new ArrayList<>();
        collectGradientsRecursive(tensor, allGradients);
        
        float[] result = new float[Math.min(allGradients.size(), parameterCount)];
        for (int i = 0; i < result.length; i++) {
            result[i] = allGradients.get(i);
        }
        return result;
    }

    private void collectGradientsRecursive(Tensor tensor, List<Float> gradients) {
        if (tensor.requiresGrad && tensor.grad != null) {
            for (float g : tensor.grad) {
                gradients.add(g);
            }
        }
        if (tensor.gradFn != null) {
            for (Tensor input : tensor.gradFn.inputs) {
                collectGradientsRecursive(input, gradients);
            }
        }
    }

    private float[] receiveGradientFromNext() {
        float[] grad = new float[parameterCount];
        try {
            commLayer.receive(grad);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to receive gradient: " + e.getMessage());
        }
        return grad;
    }

    private void sendGradientToPrev(float[] gradients) {
        try {
            commLayer.send(gradients, config.rank - 1);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to send gradient: " + e.getMessage());
        }
    }

    private void broadcastParameters() {
        try {
            commLayer.broadcast(globalParameters);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to broadcast parameters: " + e.getMessage());
        }
    }

    private void receiveParameters() {
        try {
            commLayer.receive(globalParameters);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to receive parameters: " + e.getMessage());
        }
    }

    public void trainWithPipelineParallelism(Dataset dataset) {
        running = true;
        
        int stages = config.worldSize;
        int microBatches = Math.max(1, config.batchSize / config.microBatchSize);
        
        System.out.println("[Distributed] Pipeline Parallelism: " + stages + " stages, " + microBatches + " micro-batches");
        
        int totalBatches = dataset.size() / config.batchSize;
        
        float[][] accumulatedGradients = new float[stages][parameterCount];
        int[] processedMicroBatches = new int[stages];
        
        for (int epoch = 0; epoch < config.epochs && running; epoch++) {
            dataset.shuffle();
            
            for (int batchIdx = 0; batchIdx < totalBatches && running; batchIdx++) {
                int start = batchIdx * config.batchSize;
                int end = Math.min(start + config.batchSize, dataset.size());
                Dataset batch = dataset.slice(start, end);
                
                List<Dataset> microBatchesList = splitIntoMicroBatches(batch, microBatches);
                
                for (int microIdx = 0; microIdx < microBatches && running; microIdx++) {
                    Dataset microBatch = microBatchesList.get(microIdx);
                    
                    int stage = microIdx % stages;
                    
                    if (stage == config.rank) {
                        float[] inputData = prepareBatchData(microBatch);
                        Tensor input = new Tensor(inputData, config.microBatchSize, 64);
                        input.requiresGrad = true;
                        
                        Tensor output = processPipelineStage(input, stage);
                        
                        if (stage < stages - 1) {
                            sendActivationToNextStage(output.data);
                        } else {
                            float[] targetArray = prepareTargets(microBatch);
                            Tensor loss = AutogradEngine.crossEntropyLossTensor(output, convertToIntArray(targetArray));
                            
                            float[] gradients = computeGradientsFromTensor(loss);
                            accumulateGradients(accumulatedGradients, gradients, stage);
                            processedMicroBatches[stage]++;
                            
                            if (processedMicroBatches[stage] >= microBatches) {
                                float[] avgGradients = averageGradients(accumulatedGradients[stage], microBatches);
                                updateParameters(avgGradients, config.learningRate);
                                processedMicroBatches[stage] = 0;
                                Arrays.fill(accumulatedGradients[stage], 0);
                            }
                            
                            sendGradientToPrevStage(gradients);
                        }
                    }
                    
                    if (stage > 0 && stage - 1 == config.rank) {
                        float[] inputData = receiveActivationFromPrevStage();
                        Tensor input = new Tensor(inputData, config.microBatchSize, 64);
                        input.requiresGrad = true;
                        
                        Tensor output = processPipelineStage(input, stage);
                        
                        if (stage < stages - 1) {
                            sendActivationToNextStage(output.data);
                        } else {
                            float[] targetArray = prepareTargets(microBatch);
                            Tensor loss = AutogradEngine.crossEntropyLossTensor(output, convertToIntArray(targetArray));
                            
                            float[] gradients = computeGradientsFromTensor(loss);
                            accumulateGradients(accumulatedGradients, gradients, stage);
                            processedMicroBatches[stage]++;
                            
                            if (processedMicroBatches[stage] >= microBatches) {
                                float[] avgGradients = averageGradients(accumulatedGradients[stage], microBatches);
                                updateParameters(avgGradients, config.learningRate);
                                processedMicroBatches[stage] = 0;
                                Arrays.fill(accumulatedGradients[stage], 0);
                            }
                            
                            sendGradientToPrevStage(gradients);
                        }
                    }
                    
                    if (stage > 0 && stage == config.rank) {
                        float[] gradFromNext = receiveGradientFromNextStage();
                        if (gradFromNext != null) {
                            accumulateGradients(accumulatedGradients, gradFromNext, stage);
                            processedMicroBatches[stage]++;
                            
                            if (processedMicroBatches[stage] >= microBatches) {
                                float[] avgGradients = averageGradients(accumulatedGradients[stage], microBatches);
                                updateParameters(avgGradients, config.learningRate);
                                processedMicroBatches[stage] = 0;
                                Arrays.fill(accumulatedGradients[stage], 0);
                            }
                            
                            if (stage > 0) {
                                sendGradientToPrevStage(gradFromNext);
                            }
                        }
                    }
                }
                
                globalStep.incrementAndGet();
            }
        }
        
        running = false;
    }

    private Tensor processPipelineStage(Tensor input, int stage) {
        int hiddenDim = 256;
        int layersPerStage = 2;
        
        Tensor x = input;
        for (int l = 0; l < layersPerStage; l++) {
            Tensor w1 = new Tensor(new float[x.numel * hiddenDim], x.shape[1], hiddenDim);
            w1.requiresGrad = true;
            Tensor w2 = new Tensor(new float[hiddenDim * (stage < config.worldSize - 1 ? hiddenDim : 10)], 
                                  hiddenDim, stage < config.worldSize - 1 ? hiddenDim : 10);
            w2.requiresGrad = true;
            
            Tensor hidden = AutogradEngine.matmul(x, w1);
            hidden = AutogradEngine.silu(hidden);
            x = AutogradEngine.matmul(hidden, w2);
            
            x = AutogradEngine.layerNorm(x, null, null);
        }
        
        return x;
    }

    private void sendActivationToNextStage(float[] activation) {
        try {
            commLayer.send(activation, config.rank + 1);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to send activation: " + e.getMessage());
        }
    }

    private float[] receiveActivationFromPrevStage() {
        float[] activation = new float[config.microBatchSize * 64];
        try {
            commLayer.receive(activation);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to receive activation: " + e.getMessage());
        }
        return activation;
    }

    private void sendGradientToPrevStage(float[] gradients) {
        if (config.rank > 0) {
            try {
                commLayer.send(gradients, config.rank - 1);
            } catch (Exception e) {
                System.err.println("[Distributed] Failed to send gradient: " + e.getMessage());
            }
        }
    }

    private float[] receiveGradientFromNextStage() {
        if (config.rank < config.worldSize - 1) {
            float[] gradients = new float[parameterCount];
            try {
                commLayer.receive(gradients);
                return gradients;
            } catch (Exception e) {
                System.err.println("[Distributed] Failed to receive gradient: " + e.getMessage());
            }
        }
        return null;
    }

    private void accumulateGradients(float[][] accumulated, float[] gradients, int stage) {
        for (int i = 0; i < Math.min(gradients.length, accumulated[stage].length); i++) {
            accumulated[stage][i] += gradients[i];
        }
    }

    private float[] averageGradients(float[] gradients, int count) {
        float[] result = new float[gradients.length];
        for (int i = 0; i < gradients.length; i++) {
            result[i] = gradients[i] / count;
        }
        return result;
    }

    private float[] prepareBatchData(Dataset batch) {
        List<float[]> data = batch.getData();
        int inputDim = data.get(0).length;
        float[] inputData = new float[data.size() * inputDim];
        for (int i = 0; i < data.size(); i++) {
            System.arraycopy(data.get(i), 0, inputData, i * inputDim, inputDim);
        }
        return inputData;
    }

    private float[] prepareTargets(Dataset batch) {
        List<Integer> labels = batch.getLabels();
        float[] targets = new float[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            targets[i] = labels.get(i);
        }
        return targets;
    }

    private int[] convertToIntArray(float[] floatArray) {
        int[] intArray = new int[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            intArray[i] = (int) floatArray[i];
        }
        return intArray;
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

    public void saveCheckpoint(int epoch) {
        if (config.rank != 0) return;
        
        try {
            String checkpointPath = String.format("autograd_checkpoint_epoch_%d.bin", epoch);
            java.io.DataOutputStream dos = new java.io.DataOutputStream(
                new java.io.FileOutputStream(checkpointPath));
            
            dos.writeInt(globalStep.get());
            dos.writeLong(totalSamplesProcessed.get());
            dos.writeInt(parameterCount);
            
            for (float param : globalParameters) {
                dos.writeFloat(param);
            }
            
            dos.close();
            System.out.println("[Distributed] Saved autograd checkpoint: " + checkpointPath);
        } catch (Exception e) {
            System.err.println("[Distributed] Failed to save checkpoint: " + e.getMessage());
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
        System.out.println("[Distributed] Autograd framework shutdown complete");
    }

    public TrainingMetrics getMetrics() {
        return metrics;
    }

    public WorkerStats getWorkerStats(int rank) {
        return workerStats.get(rank);
    }

    public int getGlobalStep() {
        return globalStep.get();
    }

    public long getTotalSamplesProcessed() {
        return totalSamplesProcessed.get();
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
}