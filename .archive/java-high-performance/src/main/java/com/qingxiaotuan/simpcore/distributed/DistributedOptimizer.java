package com.qingxiaotuan.simpcore.distributed;

import com.qingxiaotuan.simpcore.training.Optimizer;

import java.io.*;
import java.net.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class DistributedOptimizer {

    private final Optimizer baseOptimizer;
    private final int worldSize;
    private final int rank;
    private final ExecutorService syncService;
    private final List<Future<?>> pendingFutures;
    private final AtomicInteger pendingCount;
    private final String[] hostnames;
    private final int basePort;
    private ServerSocket serverSocket;
    private final ConcurrentHashMap<Integer, Socket> connectionCache;

    public DistributedOptimizer(Optimizer baseOptimizer, int worldSize, int rank) {
        this(baseOptimizer, worldSize, rank, "localhost", 23456);
    }

    public DistributedOptimizer(Optimizer baseOptimizer, int worldSize, int rank, String hostname, int basePort) {
        this.baseOptimizer = baseOptimizer;
        this.worldSize = worldSize;
        this.rank = rank;
        this.basePort = basePort;
        this.hostnames = new String[worldSize];
        for (int i = 0; i < worldSize; i++) {
            hostnames[i] = hostname;
        }
        this.syncService = Executors.newFixedThreadPool(worldSize * 2);
        this.pendingFutures = new ArrayList<>();
        this.pendingCount = new AtomicInteger(0);
        this.connectionCache = new ConcurrentHashMap<>();
        
        if (worldSize > 1) {
            startServer();
            establishConnections();
        }
    }

    private void startServer() {
        try {
            serverSocket = new ServerSocket(basePort + rank);
            syncService.submit(() -> {
                while (!serverSocket.isClosed()) {
                    try {
                        Socket clientSocket = serverSocket.accept();
                        DataInputStream dis = new DataInputStream(clientSocket.getInputStream());
                        int sourceRank = dis.readInt();
                        connectionCache.put(sourceRank, clientSocket);
                    } catch (IOException e) {
                        if (!serverSocket.isClosed()) {
                            e.printStackTrace();
                        }
                    }
                }
            });
            Thread.sleep(500);
        } catch (Exception e) {
            throw new RuntimeException("Failed to start server on port " + (basePort + rank), e);
        }
    }

    private void establishConnections() {
        for (int i = 0; i < worldSize; i++) {
            if (i != rank) {
                int target = i;
                syncService.submit(() -> {
                    try {
                        Socket socket = new Socket(hostnames[target], basePort + target);
                        DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
                        dos.writeInt(rank);
                        connectionCache.put(target, socket);
                    } catch (IOException e) {
                        throw new RuntimeException("Failed to connect to rank " + target, e);
                    }
                });
            }
        }
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    public void update(float[] weights, float[] gradients) {
        if (worldSize > 1) {
            allReduce(gradients);
        }
        baseOptimizer.update(weights, gradients);
    }

    public void update(float[][] weights, float[][] gradients) {
        for (int i = 0; i < weights.length; i++) {
            update(weights[i], gradients[i]);
        }
    }

    private void allReduce(float[] gradients) {
        if (worldSize <= 1) return;

        float[] totalGradients = new float[gradients.length];
        System.arraycopy(gradients, 0, totalGradients, 0, gradients.length);

        for (int i = 0; i < worldSize; i++) {
            if (i != rank) {
                int source = i;
                pendingFutures.add(syncService.submit(() -> {
                    float[] received = receiveGradientsFrom(source);
                    for (int j = 0; j < gradients.length; j++) {
                        totalGradients[j] += received[j];
                    }
                }));
                pendingCount.incrementAndGet();
            }
        }

        for (int i = 0; i < worldSize; i++) {
            if (i != rank) {
                int target = i;
                float[] toSend = gradients.clone();
                pendingFutures.add(syncService.submit(() -> {
                    sendGradientsTo(target, toSend);
                }));
                pendingCount.incrementAndGet();
            }
        }

        waitForPending();

        float invWorldSize = 1.0f / worldSize;
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] = totalGradients[i] * invWorldSize;
        }
    }

    private float[] receiveGradientsFrom(int sourceRank) {
        try {
            Socket socket = connectionCache.get(sourceRank);
            if (socket == null || socket.isClosed()) {
                socket = new Socket(hostnames[sourceRank], basePort + sourceRank);
                DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
                dos.writeInt(rank);
                connectionCache.put(sourceRank, socket);
            }
            
            DataInputStream dis = new DataInputStream(socket.getInputStream());
            int length = dis.readInt();
            float[] gradients = new float[length];
            for (int i = 0; i < length; i++) {
                gradients[i] = dis.readFloat();
            }
            return gradients;
        } catch (IOException e) {
            throw new RuntimeException("Failed to receive gradients from rank " + sourceRank, e);
        }
    }

    private void sendGradientsTo(int targetRank, float[] gradients) {
        try {
            Socket socket = connectionCache.get(targetRank);
            if (socket == null || socket.isClosed()) {
                socket = new Socket(hostnames[targetRank], basePort + targetRank);
                DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
                dos.writeInt(rank);
                connectionCache.put(targetRank, socket);
            }
            
            DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
            dos.writeInt(gradients.length);
            for (float grad : gradients) {
                dos.writeFloat(grad);
            }
            dos.flush();
        } catch (IOException e) {
            throw new RuntimeException("Failed to send gradients to rank " + targetRank, e);
        }
    }

    public void waitForPending() {
        for (Future<?> future : pendingFutures) {
            try {
                future.get(30, TimeUnit.SECONDS);
            } catch (InterruptedException | ExecutionException | TimeoutException e) {
                throw new RuntimeException("Distributed sync failed", e);
            }
        }
        pendingFutures.clear();
        pendingCount.set(0);
    }

    public void broadcastWeights(float[] weights) {
        if (worldSize <= 1) return;
        
        if (rank == 0) {
            for (int i = 1; i < worldSize; i++) {
                int target = i;
                float[] toSend = weights.clone();
                pendingFutures.add(syncService.submit(() -> {
                    sendWeightsTo(target, toSend);
                }));
            }
        } else {
            float[] received = receiveWeightsFrom(0);
            System.arraycopy(received, 0, weights, 0, weights.length);
        }
        waitForPending();
    }

    private float[] receiveWeightsFrom(int sourceRank) {
        try {
            Socket socket = connectionCache.get(sourceRank);
            if (socket == null || socket.isClosed()) {
                socket = new Socket(hostnames[sourceRank], basePort + sourceRank);
                DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
                dos.writeInt(rank);
                connectionCache.put(sourceRank, socket);
            }
            
            DataInputStream dis = new DataInputStream(socket.getInputStream());
            int length = dis.readInt();
            float[] weights = new float[length];
            for (int i = 0; i < length; i++) {
                weights[i] = dis.readFloat();
            }
            return weights;
        } catch (IOException e) {
            throw new RuntimeException("Failed to receive weights from rank " + sourceRank, e);
        }
    }

    private void sendWeightsTo(int targetRank, float[] weights) {
        try {
            Socket socket = connectionCache.get(targetRank);
            if (socket == null || socket.isClosed()) {
                socket = new Socket(hostnames[targetRank], basePort + targetRank);
                DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
                dos.writeInt(rank);
                connectionCache.put(targetRank, socket);
            }
            
            DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
            dos.writeInt(weights.length);
            for (float w : weights) {
                dos.writeFloat(w);
            }
            dos.flush();
        } catch (IOException e) {
            throw new RuntimeException("Failed to send weights to rank " + targetRank, e);
        }
    }

    public float getCurrentLearningRate() {
        return baseOptimizer.getCurrentLearningRate();
    }

    public float getBaseLearningRate() {
        return baseOptimizer.getBaseLearningRate();
    }

    public float getWeightDecay() {
        return baseOptimizer.getWeightDecay();
    }

    public float getMomentum() {
        return baseOptimizer.getMomentum();
    }

    public float getGradClip() {
        return baseOptimizer.getGradClip();
    }

    public int getWorldSize() {
        return worldSize;
    }

    public int getRank() {
        return rank;
    }

    public void shutdown() {
        for (Socket socket : connectionCache.values()) {
            try {
                socket.close();
            } catch (IOException ignored) {}
        }
        connectionCache.clear();
        
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {}
        }
        
        syncService.shutdown();
        try {
            if (!syncService.awaitTermination(60, TimeUnit.SECONDS)) {
                syncService.shutdownNow();
            }
        } catch (InterruptedException e) {
            syncService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public boolean isMaster() {
        return rank == 0;
    }
}