package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.autograd.*;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class TensorParallel {
    
    private final int rank;
    private final int worldSize;
    private final CommunicationLayer comm;
    
    private final ExecutorService computeOverlapPool;
    
    private final int[] rankOffsets;
    
    public TensorParallel(int rank, int worldSize, CommunicationLayer comm) {
        this.rank = rank;
        this.worldSize = worldSize;
        this.comm = comm;
        this.computeOverlapPool = Executors.newFixedThreadPool(2,
            new ThreadFactoryBuilder("tp-overlap-"));
        this.rankOffsets = computeRankOffsets(worldSize);
    }
    
    private static int[] computeRankOffsets(int worldSize) {
        int[] offsets = new int[worldSize];
        offsets[0] = 0;
        for (int r = 1; r < worldSize; r++) {
            offsets[r] = -1;
        }
        return offsets;
    }
    
    public Tensor columnParallelLinear(Tensor input, Tensor weight, Tensor bias) {
        int inputSize = input.shape[1];
        int outputSize = weight.shape[1];
        
        int baseLocalSize = outputSize / worldSize;
        int remainder = outputSize % worldSize;
        
        int startCol = rank * baseLocalSize + Math.min(rank, remainder);
        int actualLocalOutputSize = baseLocalSize + (rank < remainder ? 1 : 0);
        int endCol = startCol + actualLocalOutputSize;
        
        float[] localWeightData = new float[inputSize * actualLocalOutputSize];
        for (int i = 0; i < inputSize; i++) {
            System.arraycopy(weight.data, i * outputSize + startCol,
                           localWeightData, i * actualLocalOutputSize, actualLocalOutputSize);
        }
        
        Tensor localWeight = new Tensor(localWeightData, inputSize, actualLocalOutputSize);
        
        Tensor localOutput = AutogradEngine.matmul(input, localWeight);
        
        if (bias != null) {
            float[] localBiasData = new float[actualLocalOutputSize];
            System.arraycopy(bias.data, startCol, localBiasData, 0, actualLocalOutputSize);
            Tensor localBias = new Tensor(localBiasData, actualLocalOutputSize);
            localOutput = AutogradEngine.add(localOutput, localBias);
        }
        
        return localOutput;
    }
    
    public Tensor rowParallelLinear(Tensor input, Tensor weight, Tensor bias) {
        int inputSize = input.shape[1];
        int outputSize = weight.shape[1];
        
        int baseLocalSize = inputSize / worldSize;
        int remainder = inputSize % worldSize;
        
        int startRow = rank * baseLocalSize + Math.min(rank, remainder);
        int actualLocalInputSize = baseLocalSize + (rank < remainder ? 1 : 0);
        int endRow = startRow + actualLocalInputSize;
        
        float[] localInputData = new float[input.shape[0] * actualLocalInputSize];
        for (int i = 0; i < input.shape[0]; i++) {
            System.arraycopy(input.data, i * inputSize + startRow,
                           localInputData, i * actualLocalInputSize, actualLocalInputSize);
        }
        
        Tensor localInput = new Tensor(localInputData, input.shape[0], actualLocalInputSize);
        
        float[] localWeightData = new float[actualLocalInputSize * outputSize];
        for (int i = 0; i < actualLocalInputSize; i++) {
            System.arraycopy(weight.data, (startRow + i) * outputSize,
                           localWeightData, i * outputSize, outputSize);
        }
        
        Tensor localWeight = new Tensor(localWeightData, actualLocalInputSize, outputSize);
        
        Tensor localOutput = AutogradEngine.matmul(localInput, localWeight);
        
        float[] outputData = new float[localOutput.numel];
        System.arraycopy(localOutput.data, 0, outputData, 0, localOutput.numel);
        
        comm.allReduce(outputData, CommunicationLayer.ReduceOp.SUM);
        
        Tensor output = new Tensor(outputData, localOutput.shape);
        
        if (bias != null) {
            output = AutogradEngine.add(output, bias);
        }
        
        return output;
    }
    
    public Future<Tensor> rowParallelLinearAsync(Tensor input, Tensor weight, Tensor bias) {
        return computeOverlapPool.submit(() -> rowParallelLinear(input, weight, bias));
    }
    
    public Tensor allGather(Tensor input) {
        if (worldSize == 1) return input;
        
        int batchSize = input.shape[0];
        int localHiddenSize = input.shape[1];
        int globalHiddenSize = localHiddenSize * worldSize;
        
        float[][] allParts = new float[worldSize][batchSize * localHiddenSize];
        System.arraycopy(input.data, 0, allParts[rank], 0, input.numel);
        
        comm.allGather(input.data, allParts);
        
        float[] resultData = new float[batchSize * globalHiddenSize];
        for (int r = 0; r < worldSize; r++) {
            System.arraycopy(allParts[r], 0, resultData, r * batchSize * localHiddenSize,
                           batchSize * localHiddenSize);
        }
        
        return new Tensor(resultData, batchSize, globalHiddenSize);
    }
    
    public Tensor scatter(Tensor input) {
        if (worldSize == 1) return input;
        
        int batchSize = input.shape[0];
        int globalHiddenSize = input.shape[1];
        
        int baseLocalSize = globalHiddenSize / worldSize;
        int remainder = globalHiddenSize % worldSize;
        
        int startCol = rank * baseLocalSize + Math.min(rank, remainder);
        int actualLocalHiddenSize = baseLocalSize + (rank < remainder ? 1 : 0);
        int endCol = startCol + actualLocalHiddenSize;
        
        float[] localData = new float[batchSize * actualLocalHiddenSize];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(input.data, i * globalHiddenSize + startCol,
                           localData, i * actualLocalHiddenSize, actualLocalHiddenSize);
        }
        
        return new Tensor(localData, batchSize, actualLocalHiddenSize);
    }
    
    public Tensor split(Tensor input, int dim) {
        if (worldSize == 1) return input;
        
        int[] shape = input.shape;
        int totalSize = shape[dim];
        
        int baseLocalSize = totalSize / worldSize;
        int remainder = totalSize % worldSize;
        
        int startIdx = rank * baseLocalSize + Math.min(rank, remainder);
        int actualLocalSize = baseLocalSize + (rank < remainder ? 1 : 0);
        int endIdx = startIdx + actualLocalSize;
        
        int[] localShape = shape.clone();
        localShape[dim] = actualLocalSize;
        
        int[] strides = computeStrides(shape);
        int[] localStrides = computeStrides(localShape);
        
        int localNumel = 1;
        for (int s : localShape) {
            localNumel *= s;
        }
        
        float[] localData = new float[localNumel];
        
        if (dim == 0) {
            int srcOffset = startIdx * strides[0];
            System.arraycopy(input.data, srcOffset, localData, 0, localNumel);
        } else if (dim == 1 && shape.length == 2) {
            int rows = shape[0];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(input.data,
                    i * shape[1] + startIdx,
                    localData,
                    i * actualLocalSize,
                    actualLocalSize);
            }
        } else {
            int[] indices = new int[shape.length];
            for (int i = 0; i < input.numel; i++) {
                int temp = i;
                for (int j = 0; j < shape.length; j++) {
                    indices[j] = temp / strides[j];
                    temp %= strides[j];
                }
                
                if (indices[dim] >= startIdx && indices[dim] < endIdx) {
                    int localIndex = 0;
                    for (int j = 0; j < localShape.length; j++) {
                        int localIdx = (j == dim) ? indices[j] - startIdx : indices[j];
                        localIndex += localIdx * localStrides[j];
                    }
                    localData[localIndex] = input.data[i];
                }
            }
        }
        
        return new Tensor(localData, localShape);
    }
    
    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[strides.length - 1] = 1;
        for (int i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    
    public Tensor fusedColumnParallelLinear(Tensor input, Tensor weight, Tensor bias, 
                                           Tensor nextWeight, Tensor nextBias) {
        Tensor localOutput = columnParallelLinear(input, weight, bias);
        
        Tensor localInput = localOutput;
        if (nextWeight != null) {
            int localInputSize = localOutput.shape[1];
            int nextOutputSize = nextWeight.shape[1];
            
            int baseLocalSize = nextOutputSize / worldSize;
            int remainder = nextOutputSize % worldSize;
            
            int startCol = rank * baseLocalSize + Math.min(rank, remainder);
            int actualLocalOutputSize = baseLocalSize + (rank < remainder ? 1 : 0);
            
            float[] localNextWeightData = new float[localInputSize * actualLocalOutputSize];
            for (int i = 0; i < localInputSize; i++) {
                System.arraycopy(nextWeight.data, i * nextOutputSize + startCol,
                               localNextWeightData, i * actualLocalOutputSize, actualLocalOutputSize);
            }
            
            Tensor localNextWeight = new Tensor(localNextWeightData, localInputSize, actualLocalOutputSize);
            localInput = AutogradEngine.matmul(localOutput, localNextWeight);
            
            if (nextBias != null) {
                float[] localNextBiasData = new float[actualLocalOutputSize];
                System.arraycopy(nextBias.data, startCol, localNextBiasData, 0, actualLocalOutputSize);
                Tensor localNextBias = new Tensor(localNextBiasData, actualLocalOutputSize);
                localInput = AutogradEngine.add(localInput, localNextBias);
            }
        }
        
        return localInput;
    }
    
    public int getRank() {
        return rank;
    }
    
    public int getWorldSize() {
        return worldSize;
    }
    
    public void shutdown() {
        computeOverlapPool.shutdown();
        try {
            computeOverlapPool.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    private static class ThreadFactoryBuilder implements ThreadFactory {
        private final String prefix;
        private final AtomicInteger counter = new AtomicInteger(0);
        
        ThreadFactoryBuilder(String prefix) {
            this.prefix = prefix;
        }
        
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, prefix + counter.incrementAndGet());
            t.setDaemon(true);
            return t;
        }
    }
}