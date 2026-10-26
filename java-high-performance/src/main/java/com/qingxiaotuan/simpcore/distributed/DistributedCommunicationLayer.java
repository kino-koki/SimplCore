package com.qingxiaotuan.simpcore.distributed;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class DistributedCommunicationLayer {

    public static final int DEFAULT_PORT = 29500;
    public static final int BUFFER_SIZE = 64 * 1024;
    public static final int HEARTBEAT_INTERVAL = 5000;
    public static final int HEARTBEAT_TIMEOUT = 30000;

    private final int rank;
    private final int worldSize;
    private final String masterHost;
    private final int port;
    
    private ServerSocket serverSocket;
    private final Map<Integer, Socket> peerSockets = new ConcurrentHashMap<>();
    private final Map<Integer, DataOutputStream> peerOutputStreams = new ConcurrentHashMap<>();
    private final Map<Integer, DataInputStream> peerInputStreams = new ConcurrentHashMap<>();
    private final Map<Integer, String> peerAddresses = new ConcurrentHashMap<>();
    
    private final ExecutorService sendPool = Executors.newCachedThreadPool();
    private final ExecutorService receivePool = Executors.newCachedThreadPool();
    
    private final ConcurrentHashMap<Integer, CompletableFuture<float[]>> pendingRequests = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, List<float[]>> allReduceBuffers = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, ReduceOp> allReduceOps = new ConcurrentHashMap<>();
    
    private final AtomicLong messageCounter = new AtomicLong(0);
    private volatile boolean running = true;
    
    private final ScheduledExecutorService heartbeatScheduler = Executors.newScheduledThreadPool(2);
    private final ConcurrentHashMap<Integer, Long> lastHeartbeat = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, Integer> failureCount = new ConcurrentHashMap<>();

    public DistributedCommunicationLayer(int rank, int worldSize, String masterHost) {
        this(rank, worldSize, masterHost, DEFAULT_PORT);
    }

    public DistributedCommunicationLayer(int rank, int worldSize, String masterHost, int port) {
        this.rank = rank;
        this.worldSize = worldSize;
        this.masterHost = masterHost;
        this.port = port;
    }

    public void initialize() throws IOException {
        if (rank == 0) {
            startMaster();
        } else {
            connectToMaster();
        }
        startHeartbeat();
        System.out.println("[Distributed] Communication layer initialized: rank=" + rank + ", worldSize=" + worldSize);
    }

    private void startMaster() throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("[Distributed] Master listening on port " + port);
        
        receivePool.submit(() -> {
            while (running) {
                try {
                    Socket client = serverSocket.accept();
                    DataInputStream in = new DataInputStream(new BufferedInputStream(client.getInputStream()));
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(client.getOutputStream()));
                    
                    int clientRank = in.readInt();
                    peerSockets.put(clientRank, client);
                    peerInputStreams.put(clientRank, in);
                    peerOutputStreams.put(clientRank, out);
                    peerAddresses.put(clientRank, client.getInetAddress().getHostAddress());
                    System.out.println("[Distributed] Master accepted connection from rank " + clientRank);
                    
                    startReceiver(clientRank, in);
                } catch (IOException e) {
                    if (running) {
                        System.err.println("[Distributed] Master accept error: " + e.getMessage());
                    }
                }
            }
        });
    }

    private void connectToMaster() throws IOException {
        Socket socket = new Socket(masterHost, port);
        DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
        
        out.writeInt(rank);
        out.flush();
        
        peerSockets.put(0, socket);
        peerInputStreams.put(0, in);
        peerOutputStreams.put(0, out);
        
        startReceiver(0, in);
        
        int peerCount = in.readInt();
        for (int i = 0; i < peerCount; i++) {
            int peerRank = in.readInt();
            if (peerRank != rank) {
                Socket peerSocket = new Socket(masterHost, port + peerRank);
                DataInputStream peerIn = new DataInputStream(new BufferedInputStream(peerSocket.getInputStream()));
                DataOutputStream peerOut = new DataOutputStream(new BufferedOutputStream(peerSocket.getOutputStream()));
                
                peerOut.writeInt(rank);
                peerOut.flush();
                
                peerSockets.put(peerRank, peerSocket);
                peerInputStreams.put(peerRank, peerIn);
                peerOutputStreams.put(peerRank, peerOut);
                
                startReceiver(peerRank, peerIn);
                System.out.println("[Distributed] Rank " + rank + " connected to rank " + peerRank);
            }
        }
    }

    private void startReceiver(int peerRank, DataInputStream in) {
        receivePool.submit(() -> {
            while (running) {
                try {
                    int messageType = in.readInt();
                    int length = in.readInt();
                    
                    byte[] data = new byte[length];
                    in.readFully(data);
                    
                    processMessage(peerRank, messageType, data);
                } catch (IOException e) {
                    if (running) {
                        handleDisconnection(peerRank);
                    }
                    break;
                }
            }
        });
    }

    private void startHeartbeat() {
        heartbeatScheduler.scheduleAtFixedRate(() -> {
            for (int peerRank : peerSockets.keySet()) {
                sendHeartbeat(peerRank);
            }
        }, 0, HEARTBEAT_INTERVAL, TimeUnit.MILLISECONDS);
        
        heartbeatScheduler.scheduleAtFixedRate(() -> {
            long now = System.currentTimeMillis();
            for (Map.Entry<Integer, Long> entry : lastHeartbeat.entrySet()) {
                if (now - entry.getValue() > HEARTBEAT_TIMEOUT) {
                    System.err.println("[Distributed] Heartbeat timeout for rank " + entry.getKey());
                    handleDisconnection(entry.getKey());
                }
            }
        }, HEARTBEAT_INTERVAL, HEARTBEAT_INTERVAL, TimeUnit.MILLISECONDS);
    }

    private void sendHeartbeat(int peerRank) {
        try {
            DataOutputStream out = peerOutputStreams.get(peerRank);
            if (out != null) {
                out.writeInt(MessageType.HEARTBEAT.ordinal());
                out.writeInt(4);
                out.writeInt(rank);
                out.flush();
            }
        } catch (Exception e) {
        }
    }

    private void handleDisconnection(int peerRank) {
        Socket socket = peerSockets.remove(peerRank);
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException e) {}
        }
        peerInputStreams.remove(peerRank);
        peerOutputStreams.remove(peerRank);
        failureCount.merge(peerRank, 1, Integer::sum);
        System.err.println("[Distributed] Rank " + peerRank + " disconnected");
    }

    private void processMessage(int fromRank, int messageType, byte[] data) {
        switch (MessageType.values()[messageType]) {
            case HEARTBEAT:
                lastHeartbeat.put(fromRank, System.currentTimeMillis());
                break;
            case ALL_REDUCE_REQUEST:
                handleAllReduceRequest(fromRank, data);
                break;
            case ALL_REDUCE_RESPONSE:
                handleAllReduceResponse(fromRank, data);
                break;
            case BROADCAST:
                handleBroadcast(fromRank, data);
                break;
            case SCATTER:
                handleScatter(fromRank, data);
                break;
            case GATHER:
                handleGather(fromRank, data);
                break;
        }
    }

    public void allReduce(float[] data, ReduceOp op) {
        if (worldSize == 1) return;
        
        long requestId = messageCounter.incrementAndGet();
        CompletableFuture<float[]> future = new CompletableFuture<>();
        pendingRequests.put((int) requestId, future);
        
        allReduceBuffers.put((int) requestId, Collections.synchronizedList(new ArrayList<>()));
        allReduceOps.put((int) requestId, op);
        
        try {
            for (int peerRank : peerSockets.keySet()) {
                DataOutputStream out = peerOutputStreams.get(peerRank);
                if (out != null) {
                    out.writeInt(MessageType.ALL_REDUCE_REQUEST.ordinal());
                    out.writeInt(4 + 4 + data.length * 4);
                    out.writeInt((int) requestId);
                    out.writeInt(op.ordinal());
                    for (float val : data) {
                        out.writeFloat(val);
                    }
                    out.flush();
                }
            }
            
            float[] result = future.get(60, TimeUnit.SECONDS);
            System.arraycopy(result, 0, data, 0, data.length);
        } catch (Exception e) {
            throw new RuntimeException("AllReduce failed", e);
        } finally {
            pendingRequests.remove((int) requestId);
            allReduceBuffers.remove((int) requestId);
            allReduceOps.remove((int) requestId);
        }
    }

    private void handleAllReduceRequest(int fromRank, byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        int requestId = buffer.getInt();
        int opOrdinal = buffer.getInt();
        
        float[] received = new float[(data.length - 8) / 4];
        for (int i = 0; i < received.length; i++) {
            received[i] = buffer.getFloat();
        }
        
        try {
            DataOutputStream out = peerOutputStreams.get(fromRank);
            if (out != null) {
                out.writeInt(MessageType.ALL_REDUCE_RESPONSE.ordinal());
                out.writeInt(4 + received.length * 4);
                out.writeInt(requestId);
                for (float val : received) {
                    out.writeFloat(val);
                }
                out.flush();
            }
        } catch (IOException e) {
            System.err.println("[Distributed] Failed to send allReduce response: " + e.getMessage());
        }
    }

    private void handleAllReduceResponse(int fromRank, byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        int requestId = buffer.getInt();
        
        CompletableFuture<float[]> future = pendingRequests.get(requestId);
        if (future == null) return;
        
        float[] received = new float[(data.length - 4) / 4];
        for (int i = 0; i < received.length; i++) {
            received[i] = buffer.getFloat();
        }
        
        List<float[]> buffers = allReduceBuffers.get(requestId);
        if (buffers == null) return;
        
        buffers.add(received);
        
        if (buffers.size() == peerSockets.size()) {
            float[] result = new float[received.length];
            ReduceOp op = allReduceOps.get(requestId);
            if (op == null) op = ReduceOp.AVG;
            
            for (int i = 0; i < result.length; i++) {
                switch (op) {
                    case SUM:
                        result[i] = 0.0f;
                        for (float[] buf : buffers) {
                            result[i] += buf[i];
                        }
                        break;
                    case AVG:
                        result[i] = 0.0f;
                        for (float[] buf : buffers) {
                            result[i] += buf[i];
                        }
                        result[i] /= worldSize;
                        break;
                    case MAX:
                        result[i] = Float.NEGATIVE_INFINITY;
                        for (float[] buf : buffers) {
                            result[i] = Math.max(result[i], buf[i]);
                        }
                        break;
                    case MIN:
                        result[i] = Float.POSITIVE_INFINITY;
                        for (float[] buf : buffers) {
                            result[i] = Math.min(result[i], buf[i]);
                        }
                        break;
                }
            }
            future.complete(result);
        }
    }

    public void broadcast(float[] data) {
        if (worldSize == 1) return;
        
        for (int peerRank : peerSockets.keySet()) {
            sendPool.submit(() -> {
                try {
                    DataOutputStream out = peerOutputStreams.get(peerRank);
                    if (out != null) {
                        out.writeInt(MessageType.BROADCAST.ordinal());
                        out.writeInt(data.length * 4);
                        for (float val : data) {
                            out.writeFloat(val);
                        }
                        out.flush();
                    }
                } catch (IOException e) {
                    System.err.println("[Distributed] Broadcast to rank " + peerRank + " failed: " + e.getMessage());
                }
            });
        }
    }

    private void handleBroadcast(int fromRank, byte[] data) {
        float[] received = new float[data.length / 4];
        ByteBuffer buffer = ByteBuffer.wrap(data);
        for (int i = 0; i < received.length; i++) {
            received[i] = buffer.getFloat();
        }
        
        if (rank != 0) {
            for (int peerRank : peerSockets.keySet()) {
                if (peerRank != fromRank) {
                    try {
                        DataOutputStream out = peerOutputStreams.get(peerRank);
                        if (out != null) {
                            out.writeInt(MessageType.BROADCAST.ordinal());
                            out.writeInt(data.length);
                            out.write(data);
                            out.flush();
                        }
                    } catch (IOException e) {
                        System.err.println("[Distributed] Broadcast forward failed: " + e.getMessage());
                    }
                }
            }
        }
    }

    public void scatter(float[][] data, int[] sizes) {
        if (worldSize == 1) return;
        
        int offset = 0;
        for (int peerRank : peerSockets.keySet()) {
            int size = sizes[peerRank];
            float[] chunk = new float[size];
            System.arraycopy(data[0], offset, chunk, 0, size);
            offset += size;
            
            sendPool.submit(() -> {
                try {
                    DataOutputStream out = peerOutputStreams.get(peerRank);
                    if (out != null) {
                        out.writeInt(MessageType.SCATTER.ordinal());
                        out.writeInt(4 + chunk.length * 4);
                        out.writeInt(chunk.length);
                        for (float val : chunk) {
                            out.writeFloat(val);
                        }
                        out.flush();
                    }
                } catch (IOException e) {
                    System.err.println("[Distributed] Scatter to rank " + peerRank + " failed: " + e.getMessage());
                }
            });
        }
    }

    private void handleScatter(int fromRank, byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        int size = buffer.getInt();
        float[] chunk = new float[size];
        for (int i = 0; i < chunk.length; i++) {
            chunk[i] = buffer.getFloat();
        }
        
        synchronized (this) {
            pendingRequests.put(fromRank, CompletableFuture.completedFuture(chunk));
        }
    }

    public float[] gather(float[] data) {
        if (worldSize == 1) return data;
        
        float[][] allData = new float[worldSize][data.length];
        System.arraycopy(data, 0, allData[rank], 0, data.length);
        
        for (int peerRank : peerSockets.keySet()) {
            try {
                DataOutputStream out = peerOutputStreams.get(peerRank);
                if (out != null) {
                    out.writeInt(MessageType.GATHER.ordinal());
                    out.writeInt(data.length * 4);
                    for (float val : data) {
                        out.writeFloat(val);
                    }
                    out.flush();
                }
            } catch (IOException e) {
                System.err.println("[Distributed] Gather to rank " + peerRank + " failed: " + e.getMessage());
            }
        }
        
        for (int peerRank : peerSockets.keySet()) {
            CompletableFuture<float[]> future = new CompletableFuture<>();
            pendingRequests.put(peerRank + 1000, future);
            
            try {
                float[] peerData = future.get(30, TimeUnit.SECONDS);
                System.arraycopy(peerData, 0, allData[peerRank], 0, peerData.length);
            } catch (Exception e) {
                System.err.println("[Distributed] Gather from rank " + peerRank + " failed: " + e.getMessage());
            }
        }
        
        float[] result = new float[data.length * worldSize];
        for (int i = 0; i < worldSize; i++) {
            System.arraycopy(allData[i], 0, result, i * data.length, data.length);
        }
        
        return result;
    }

    private void handleGather(int fromRank, byte[] data) {
        float[] received = new float[data.length / 4];
        ByteBuffer buffer = ByteBuffer.wrap(data);
        for (int i = 0; i < received.length; i++) {
            received[i] = buffer.getFloat();
        }
        
        synchronized (this) {
            pendingRequests.put(fromRank + 1000, CompletableFuture.completedFuture(received));
        }
    }

    public void shutdown() {
        running = false;
        
        for (Socket socket : peerSockets.values()) {
            try {
                socket.close();
            } catch (IOException e) {}
        }
        
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {}
        }
        
        sendPool.shutdown();
        receivePool.shutdown();
        heartbeatScheduler.shutdown();
        
        System.out.println("[Distributed] Communication layer shutdown");
    }

    public int getRank() {
        return rank;
    }

    public int getWorldSize() {
        return worldSize;
    }

    public boolean isPeerConnected(int peerRank) {
        return peerSockets.containsKey(peerRank);
    }

    public int getFailureCount(int peerRank) {
        return failureCount.getOrDefault(peerRank, 0);
    }

    public enum MessageType {
        HEARTBEAT,
        ALL_REDUCE_REQUEST,
        ALL_REDUCE_RESPONSE,
        BROADCAST,
        SCATTER,
        GATHER
    }

    public enum ReduceOp {
        SUM,
        AVG,
        MAX,
        MIN
    }
}