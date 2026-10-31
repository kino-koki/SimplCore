package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.autograd.*;
import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.training.Optimizer;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class DistributedTrainingFramework {

    private final ClusterManager clusterManager;
    private final DataParallelStrategy dataParallel;
    private final ModelParallelStrategy modelParallel;
    private final AutoBalancer loadBalancer;
    private final FaultTolerance tolerance;
    private final CommunicationLayer commLayer;
    
    private ExecutorService workerPool;
    private AtomicBoolean running;
    private AtomicInteger epochCounter;

    public DistributedTrainingFramework() {
        this.clusterManager = new ClusterManager();
        this.dataParallel = new DataParallelStrategy();
        this.modelParallel = new ModelParallelStrategy();
        this.loadBalancer = new AutoBalancer();
        this.tolerance = new FaultTolerance();
        this.commLayer = new CommunicationLayer();
        
        this.running = new AtomicBoolean(false);
        this.epochCounter = new AtomicInteger(0);
    }

    public void initialize(int numWorkers, String[] hosts) {
        clusterManager.initialize(numWorkers, hosts);
        commLayer.initialize(clusterManager.getWorkers());
        workerPool = Executors.newFixedThreadPool(numWorkers);
        System.out.println("[Distributed] Initialized with " + numWorkers + " workers");
    }

    public void train(Dataset dataset, SimpCoreModel model, Optimizer optimizer,
                      int epochs, int batchSize) {
        
        running.set(true);
        
        for (int epoch = 0; epoch < epochs && running.get(); epoch++) {
            epochCounter.set(epoch);
            
            List<Future<TrainingResult>> futures = new ArrayList<>();
            List<Dataset> shards = dataParallel.distribute(dataset, clusterManager.getWorkerCount());
            
            for (int i = 0; i < clusterManager.getWorkerCount(); i++) {
                final int workerId = i;
                final Dataset shard = shards.get(i);
                
                futures.add(workerPool.submit(() -> {
                    return trainOnWorker(workerId, model, optimizer, shard, batchSize);
                }));
            }
            
            TrainingResult aggregated = aggregateResults(futures);
            updateGlobalModel(model, aggregated);
            
            loadBalancer.rebalance(clusterManager, aggregated);
            
            if (epoch % 10 == 0) {
                System.out.printf("[Distributed] Epoch %d: Loss = %.6f, Perplexity = %.2f%n",
                    epoch, aggregated.averageLoss, aggregated.perplexity);
            }
        }
        
        running.set(false);
    }

    private TrainingResult trainOnWorker(int workerId, SimpCoreModel model, 
                                         Optimizer optimizer, Dataset dataset, int batchSize) {
        float totalLoss = 0;
        int totalTokens = 0;
        
        try {
            for (int i = 0; i < dataset.size(); i += batchSize) {
                int end = Math.min(i + batchSize, dataset.size());
                Dataset batch = dataset.slice(i, end);
                
                float loss = computeBatchLoss(model, batch);
                totalLoss += loss;
                totalTokens += batch.size();
                
                float[] gradients = computeGradients(model, batch);
                commLayer.allReduce(gradients, CommunicationLayer.ReduceOp.AVG);
                
                optimizer.update(model.getOutputWeights(), gradients);
            }
        } catch (Exception e) {
            tolerance.handleFailure(workerId, e);
            throw new RuntimeException("Worker " + workerId + " failed", e);
        }
        
        float avgLoss = totalLoss / (dataset.size() / (float) batchSize);
        float perplexity = (float) Math.exp(avgLoss);
        
        return new TrainingResult(avgLoss, perplexity, workerId);
    }

    private float computeBatchLoss(SimpCoreModel model, Dataset batch) {
        float totalLoss = 0;
        
        for (int i = 0; i < batch.size(); i++) {
            float[] input = batch.getData().get(i);
            int target = batch.getLabels().get(i);
            
            float[] logits = model.forward(input);
            
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float logit : logits) {
                if (logit > maxLogit) maxLogit = logit;
            }
            
            float sumExp = 0;
            for (float logit : logits) {
                sumExp += (float) Math.exp(logit - maxLogit);
            }
            
            float loss = maxLogit + (float) Math.log(sumExp) - logits[target];
            totalLoss += loss;
        }
        
        return totalLoss / batch.size();
    }

    private float[] computeGradients(SimpCoreModel model, Dataset batch) {
        int paramCount = model.getOutputWeights().length;
        float[] gradients = new float[paramCount];
        
        for (int i = 0; i < batch.size(); i++) {
            float[] input = batch.getData().get(i);
            int target = batch.getLabels().get(i);
            
            float[] logits = model.forward(input);
            
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (float logit : logits) {
                if (logit > maxLogit) maxLogit = logit;
            }
            
            float sumExp = 0;
            float[] expLogits = new float[logits.length];
            for (int j = 0; j < logits.length; j++) {
                expLogits[j] = (float) Math.exp(logits[j] - maxLogit);
                sumExp += expLogits[j];
            }
            
            for (int j = 0; j < logits.length; j++) {
                float grad = expLogits[j] / sumExp;
                if (j == target) {
                    grad -= 1.0f;
                }
                
                for (int k = 0; k < input.length; k++) {
                    int idx = j * input.length + k;
                    if (idx < gradients.length) {
                        gradients[idx] += grad * input[k];
                    }
                }
            }
        }
        
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= batch.size();
        }
        
        return gradients;
    }

    private TrainingResult aggregateResults(List<Future<TrainingResult>> futures) {
        float totalLoss = 0;
        float totalPerplexity = 0;
        int count = 0;
        
        for (Future<TrainingResult> future : futures) {
            try {
                TrainingResult result = future.get();
                totalLoss += result.averageLoss;
                totalPerplexity += result.perplexity;
                count++;
            } catch (InterruptedException | ExecutionException e) {
                tolerance.handleFailure(-1, e);
            }
        }
        
        return new TrainingResult(totalLoss / count, totalPerplexity / count, -1);
    }

    private void updateGlobalModel(SimpCoreModel globalModel, TrainingResult aggregated) {
        commLayer.broadcast(globalModel);
    }

    public void shutdown() {
        running.set(false);
        if (workerPool != null) {
            workerPool.shutdown();
        }
        clusterManager.shutdown();
        commLayer.shutdown();
    }

    public ClusterManager getClusterManager() {
        return clusterManager;
    }

    public AutoBalancer getLoadBalancer() {
        return loadBalancer;
    }

    public FaultTolerance getFaultTolerance() {
        return tolerance;
    }

    public static class ClusterManager {
        private List<WorkerNode> workers;
        private AtomicInteger nextWorkerId;
        
        public void initialize(int numWorkers, String[] hosts) {
            workers = new ArrayList<>();
            nextWorkerId = new AtomicInteger(0);
            
            for (int i = 0; i < numWorkers; i++) {
                workers.add(new WorkerNode(i, hosts[i], "localhost"));
            }
        }
        
        public List<WorkerNode> getWorkers() {
            return workers;
        }
        
        public int getWorkerCount() {
            return workers.size();
        }
        
        public WorkerNode getWorker(int id) {
            return workers.get(id);
        }
        
        public void shutdown() {
            for (WorkerNode worker : workers) {
                worker.disconnect();
            }
            workers.clear();
        }
    }

    public static class WorkerNode {
        private final int id;
        private final String host;
        private final String ip;
        private volatile boolean connected;
        
        public WorkerNode(int id, String host, String ip) {
            this.id = id;
            this.host = host;
            this.ip = ip;
            this.connected = true;
        }
        
        public int getId() { return id; }
        public String getHost() { return host; }
        public String getIp() { return ip; }
        public boolean isConnected() { return connected; }
        public void disconnect() { connected = false; }
    }

    public static class DataParallelStrategy {
        public List<Dataset> distribute(Dataset dataset, int numWorkers) {
            List<Dataset> shards = new ArrayList<>();
            int shardSize = dataset.size() / numWorkers;
            
            for (int i = 0; i < numWorkers; i++) {
                int start = i * shardSize;
                int end = (i == numWorkers - 1) ? dataset.size() : (i + 1) * shardSize;
                shards.add(dataset.slice(start, end));
            }
            
            return shards;
        }
    }

    public static class ModelParallelStrategy {
        public List<SimpCoreModel> split(SimpCoreModel model, int numWorkers) {
            List<SimpCoreModel> parts = new ArrayList<>();
            for (int i = 0; i < numWorkers; i++) {
                parts.add(model.clone());
            }
            return parts;
        }
    }

    public static class AutoBalancer {
        public void rebalance(ClusterManager clusterManager, TrainingResult result) {
        }
    }

    public static class FaultTolerance {
        private final ConcurrentHashMap<Integer, Integer> failureCount;
        
        public FaultTolerance() {
            failureCount = new ConcurrentHashMap<>();
        }
        
        public void handleFailure(int workerId, Exception e) {
            failureCount.merge(workerId, 1, Integer::sum);
            System.err.println("[FaultTolerance] Worker " + workerId + " failed: " + e.getMessage());
        }
        
        public int getFailureCount(int workerId) {
            return failureCount.getOrDefault(workerId, 0);
        }
    }

    public static class CommunicationLayer {
        private List<WorkerNode> workers;
        private com.qingxiaotuan.simpcore.distributed.CommunicationLayer internalComm;
        
        public void initialize(List<WorkerNode> workers) {
            this.workers = workers;
            if (!workers.isEmpty()) {
                internalComm = new com.qingxiaotuan.simpcore.distributed.CommunicationLayer(
                    0, workers.size());
            }
        }
        
        public void allReduce(float[] gradients, ReduceOp op) {
            if (internalComm != null && workers.size() > 1) {
                internalComm.allReduce(gradients, 
                    com.qingxiaotuan.simpcore.distributed.CommunicationLayer.ReduceOp.SUM);
            }
        }
        
        public void broadcast(SimpCoreModel model) {
        }
        
        public void send(int targetId, float[] data) {
        }
        
        public float[] receive(int sourceId) {
            return new float[0];
        }
        
        public void shutdown() {
            if (internalComm != null) {
                internalComm.shutdown();
            }
            workers.clear();
        }
        
        public enum ReduceOp {
            SUM, AVG, MAX, MIN
        }
    }

    public static class Dataset {
        private final List<float[]> data;
        private final List<Integer> labels;
        
        public Dataset() {
            data = new ArrayList<>();
            labels = new ArrayList<>();
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
            for (int i = start; i < end; i++) {
                slice.add(data.get(i), labels.get(i));
            }
            return slice;
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
        public final float perplexity;
        public final int workerId;
        
        public TrainingResult(float averageLoss, float perplexity, int workerId) {
            this.averageLoss = averageLoss;
            this.perplexity = perplexity;
            this.workerId = workerId;
        }
    }
}