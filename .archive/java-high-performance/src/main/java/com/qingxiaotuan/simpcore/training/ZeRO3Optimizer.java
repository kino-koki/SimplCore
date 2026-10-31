package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.distributed.CommunicationLayer;
import com.qingxiaotuan.simpcore.distributed.CommunicationLayer.ReduceOp;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class ZeRO3Optimizer {
    
    private final float[] params;
    
    private final float[] localParams;
    private final float[] localGrads;
    
    private final int rank;
    private final int worldSize;
    private final int paramSize;
    private final int localSize;
    private final int paramOffset;
    
    private volatile float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final float gradClip;
    
    private final AtomicInteger step = new AtomicInteger(0);
    
    private volatile boolean cpuOffloadEnabled;
    private volatile boolean nvmeOffloadEnabled;
    
    private final float[] m;
    private final float[] v;
    private float[] cpuM;
    private float[] cpuV;
    private String nvmePath;
    
    private final CommunicationLayer comm;
    private final ExecutorService offloadPool;
    
    private final int[] rankOffsets;
    private final int[] rankSizes;
    
    private volatile boolean initialized = false;
    
    public ZeRO3Optimizer(float[] params, int rank, int worldSize, float lr) {
        this(params, rank, worldSize, lr, 0.9f, 0.999f, 1e-8f, 0.01f, 1.0f);
    }
    
    public ZeRO3Optimizer(float[] params, int rank, int worldSize, float lr, 
                          float beta1, float beta2, float eps, float weightDecay, float gradClip) {
        this.params = params;
        this.rank = rank;
        this.worldSize = worldSize;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.gradClip = gradClip;
        
        this.paramSize = params.length;
        
        this.rankOffsets = new int[worldSize];
        this.rankSizes = new int[worldSize];
        computeRankInfo();
        
        this.localSize = rankSizes[rank];
        this.paramOffset = rankOffsets[rank];
        
        this.localParams = new float[localSize];
        this.localGrads = new float[localSize];
        System.arraycopy(params, paramOffset, localParams, 0, localSize);
        
        this.m = new float[localSize];
        this.v = new float[localSize];
        
        this.cpuOffloadEnabled = false;
        this.nvmeOffloadEnabled = false;
        
        this.comm = new CommunicationLayer(rank, worldSize);
        this.offloadPool = Executors.newFixedThreadPool(2, 
            new ThreadFactoryBuilder("zero-offload-"));
    }
    
    private void computeRankInfo() {
        int baseSize = paramSize / worldSize;
        int remainder = paramSize % worldSize;
        
        int offset = 0;
        for (int r = 0; r < worldSize; r++) {
            rankOffsets[r] = offset;
            rankSizes[r] = baseSize + (r < remainder ? 1 : 0);
            offset += rankSizes[r];
        }
    }
    
    public void init() throws Exception {
        comm.init();
        initialized = true;
    }
    
    public void setGradients(float[] grads) {
        if (grads.length < paramOffset + localSize) {
            throw new IllegalArgumentException("Gradients array too small");
        }
        System.arraycopy(grads, paramOffset, localGrads, 0, localSize);
    }
    
    public Future<Void> setGradientsAsync(float[] grads) {
        return offloadPool.submit(() -> {
            setGradients(grads);
            return null;
        });
    }
    
    public void step() {
        int currentStep = step.incrementAndGet();
        
        if (cpuOffloadEnabled) {
            loadOptimizerStateFromCPU();
        }
        
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, currentStep);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, currentStep);
        
        float globalGradNorm = computeGlobalGradNorm();
        float scale = globalGradNorm > gradClip ? gradClip / (globalGradNorm + 1e-6f) : 1.0f;
        
        for (int i = 0; i < localSize; i++) {
            float grad = localGrads[i] * scale;
            
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
            
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;
            
            float update = mHat / ((float) Math.sqrt(vHat) + eps);
            
            if (weightDecay > 0) {
                localParams[i] -= lr * weightDecay * localParams[i];
            }
            localParams[i] -= lr * update;
            
            if (Float.isNaN(localParams[i]) || Float.isInfinite(localParams[i])) {
                localParams[i] = 0.0f;
            }
        }
        
        if (cpuOffloadEnabled) {
            saveOptimizerStateToCPU();
        }
        
        allGatherParams();
    }
    
    private float computeGlobalGradNorm() {
        float localNorm = 0;
        for (float g : localGrads) {
            localNorm += g * g;
        }
        
        if (worldSize == 1) {
            return (float) Math.sqrt(localNorm);
        }
        
        float[] normArr = new float[]{localNorm};
        comm.allReduce(normArr, ReduceOp.SUM);
        return (float) Math.sqrt(normArr[0]);
    }
    
    public Future<Void> stepAsync() {
        return offloadPool.submit(() -> {
            step();
            return null;
        });
    }
    
    private void loadOptimizerStateFromCPU() {
        if (cpuM != null && cpuV != null) {
            System.arraycopy(cpuM, 0, m, 0, localSize);
            System.arraycopy(cpuV, 0, v, 0, localSize);
        }
    }
    
    private void saveOptimizerStateToCPU() {
        if (cpuM == null) cpuM = new float[localSize];
        if (cpuV == null) cpuV = new float[localSize];
        System.arraycopy(m, 0, cpuM, 0, localSize);
        System.arraycopy(v, 0, cpuV, 0, localSize);
    }
    
    private void allGatherParams() {
        if (worldSize == 1) {
            System.arraycopy(localParams, 0, params, paramOffset, localSize);
            return;
        }
        
        allGatherRing();
    }
    
    private void allGatherRing() {
        float[] sendBuf = Arrays.copyOf(localParams, localSize);
        float[] recvBuf = new float[Math.max(localSize, getMaxRankSize())];
        
        int sendTo = (rank + 1) % worldSize;
        int recvFrom = (rank - 1 + worldSize) % worldSize;
        
        for (int i = 0; i < worldSize - 1; i++) {
            int recvSize = rankSizes[recvFrom];
            if (recvBuf.length < recvSize) {
                recvBuf = new float[recvSize];
            }
            
            final float[] finalSendBuf = sendBuf;
            Future<?> sendFuture = offloadPool.submit(() -> 
                comm.sendBlocking(sendTo, finalSendBuf));
            
            comm.receiveBlocking(recvFrom, recvBuf);
            
            int recvOffset = rankOffsets[recvFrom];
            System.arraycopy(recvBuf, 0, params, recvOffset, recvSize);
            
            try {
                sendFuture.get(30, TimeUnit.SECONDS);
            } catch (Exception e) {
                throw new RuntimeException("Ring all-gather send failed", e);
            }
            
            float[] temp = sendBuf;
            sendBuf = recvBuf;
            recvBuf = temp;
        }
        
        System.arraycopy(localParams, 0, params, paramOffset, localSize);
    }
    
    private int getMaxRankSize() {
        int max = 0;
        for (int s : rankSizes) {
            if (s > max) max = s;
        }
        return max;
    }
    
    public void allReduceGradients() {
        if (worldSize == 1) return;
        comm.allReduce(localGrads, ReduceOp.SUM);
    }
    
    public Future<Void> allReduceGradientsAsync() {
        return offloadPool.submit(() -> {
            allReduceGradients();
            return null;
        });
    }
    
    public void setLearningRate(float lr) {
        this.lr = lr;
    }
    
    public float getLearningRate() {
        return lr;
    }
    
    public float getMemoryUsageMB() {
        float gpuMemory = (float) (localParams.length + localGrads.length + m.length + v.length) * 4 / (1024f * 1024f);
        if (cpuOffloadEnabled && cpuM != null) {
            gpuMemory -= (float) (m.length + v.length) * 4 / (1024f * 1024f);
        }
        return gpuMemory;
    }
    
    public float getCPUMemoryUsageMB() {
        if (!cpuOffloadEnabled || cpuM == null) return 0;
        return (float) (cpuM.length + cpuV.length) * 4 / (1024f * 1024f);
    }
    
    public int getLocalSize() {
        return localSize;
    }
    
    public int getParamOffset() {
        return paramOffset;
    }
    
    public void enableCPUOffload() {
        this.cpuOffloadEnabled = true;
        if (cpuM == null) cpuM = new float[localSize];
        if (cpuV == null) cpuV = new float[localSize];
    }
    
    public void disableCPUOffload() {
        if (cpuOffloadEnabled) {
            loadOptimizerStateFromCPU();
        }
        this.cpuOffloadEnabled = false;
    }
    
    public void enableNVMeOffload(String path) {
        this.nvmeOffloadEnabled = true;
        this.nvmePath = path;
        
        File dir = new File(path).getParentFile();
        if (dir != null && !dir.exists()) {
            dir.mkdirs();
        }
    }
    
    public void disableNVMeOffload() {
        this.nvmeOffloadEnabled = false;
    }
    
    public Future<Void> saveToNVMe() {
        if (!nvmeOffloadEnabled || nvmePath == null) {
            return CompletableFuture.completedFuture(null);
        }
        
        return offloadPool.submit(() -> {
            try (FileChannel channel = new FileOutputStream(nvmePath).getChannel()) {
                ByteBuffer header = ByteBuffer.allocate(16);
                header.putInt(paramSize);
                header.putInt(localSize);
                header.putInt(paramOffset);
                header.putInt(rank);
                header.flip();
                channel.write(header);
                
                ByteBuffer buffer = ByteBuffer.allocate(localSize * 4);
                for (float f : localParams) {
                    buffer.putFloat(f);
                }
                buffer.flip();
                channel.write(buffer);
                
                if (!cpuOffloadEnabled) {
                    buffer.clear();
                    for (float val : m) buffer.putFloat(val);
                    buffer.flip();
                    channel.write(buffer);
                    
                    buffer.clear();
                    for (float val : v) buffer.putFloat(val);
                    buffer.flip();
                    channel.write(buffer);
                }
            } catch (IOException e) {
                throw new RuntimeException("NVMe save failed", e);
            }
            return null;
        });
    }
    
    public Future<Void> loadFromNVMe() {
        if (!nvmeOffloadEnabled || nvmePath == null) {
            return CompletableFuture.completedFuture(null);
        }
        
        return offloadPool.submit(() -> {
            try (FileChannel channel = new FileInputStream(nvmePath).getChannel()) {
                ByteBuffer header = ByteBuffer.allocate(16);
                channel.read(header);
                header.flip();
                
                int savedParamSize = header.getInt();
                int savedLocalSize = header.getInt();
                int savedOffset = header.getInt();
                int savedRank = header.getInt();
                
                if (savedParamSize != paramSize || savedLocalSize != localSize ||
                    savedOffset != paramOffset || savedRank != rank) {
                    throw new IOException("NVMe checkpoint mismatch");
                }
                
                ByteBuffer buffer = ByteBuffer.allocate(localSize * 4);
                channel.read(buffer);
                buffer.flip();
                for (int i = 0; i < localSize; i++) {
                    localParams[i] = buffer.getFloat();
                }
                
                if (!cpuOffloadEnabled) {
                    buffer.clear();
                    channel.read(buffer);
                    buffer.flip();
                    for (int i = 0; i < localSize; i++) {
                        m[i] = buffer.getFloat();
                    }
                    
                    buffer.clear();
                    channel.read(buffer);
                    buffer.flip();
                    for (int i = 0; i < localSize; i++) {
                        v[i] = buffer.getFloat();
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException("NVMe load failed", e);
            }
            return null;
        });
    }
    
    public void shutdown() {
        if (initialized) {
            comm.shutdown();
        }
        offloadPool.shutdown();
        try {
            offloadPool.awaitTermination(5, TimeUnit.SECONDS);
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