package com.qingxiaotuan.simpcore.distributed;

import java.io.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class CommunicationLayer {
    
    private static final int DEFAULT_PORT = 12345;
    private static final int BUFFER_SIZE = 1 << 20;
    private static final int MAX_RETRIES = 60;
    private static final int RETRY_DELAY_MS = 1000;
    
    private final int rank;
    private final int worldSize;
    private final String masterHost;
    private final int masterPort;
    
    private ServerSocketChannel serverChannel;
    private final Map<Integer, SocketChannel> rankToChannel = new ConcurrentHashMap<>();
    
    private final ExecutorService sendPool;
    private final ExecutorService recvPool;
    
    private final AtomicLong bytesSent = new AtomicLong(0);
    private final AtomicLong bytesReceived = new AtomicLong(0);
    private final AtomicLong messagesSent = new AtomicLong(0);
    private final AtomicLong messagesReceived = new AtomicLong(0);
    
    private volatile boolean running = false;
    private final CyclicBarrier barrier;
    
    public CommunicationLayer(int rank, int worldSize) {
        this(rank, worldSize, "localhost", DEFAULT_PORT);
    }
    
    public CommunicationLayer(int rank, int worldSize, String masterHost, int masterPort) {
        if (rank < 0 || rank >= worldSize) {
            throw new IllegalArgumentException("Invalid rank: " + rank + " / " + worldSize);
        }
        this.rank = rank;
        this.worldSize = worldSize;
        this.masterHost = masterHost;
        this.masterPort = masterPort;
        this.sendPool = Executors.newFixedThreadPool(Math.min(4, worldSize),
            new ThreadFactoryBuilder("comm-send-"));
        this.recvPool = Executors.newFixedThreadPool(Math.min(4, worldSize),
            new ThreadFactoryBuilder("comm-recv-"));
        this.barrier = worldSize > 1 ? new CyclicBarrier(worldSize) : null;
    }
    
    public void init() throws Exception {
        if (worldSize == 1) {
            running = true;
            return;
        }
        
        serverChannel = ServerSocketChannel.open();
        serverChannel.socket().setReuseAddress(true);
        serverChannel.socket().bind(new InetSocketAddress(masterPort + rank));
        serverChannel.configureBlocking(false);
        
        new Thread(this::acceptLoop, "comm-accept-" + rank).start();
        
        for (int other = 0; other < worldSize; other++) {
            if (other != rank) {
                connectToRank(other);
            }
        }
        
        int expected = worldSize - 1;
        int waited = 0;
        while (rankToChannel.size() < expected && waited < MAX_RETRIES * RETRY_DELAY_MS) {
            Thread.sleep(100);
            waited += 100;
        }
        
        if (rankToChannel.size() < expected) {
            throw new RuntimeException(String.format("Failed to establish all connections: %d/%d",
                rankToChannel.size(), expected));
        }
        
        running = true;
        barrier();
    }
    
    private void acceptLoop() {
        try {
            while (running) {
                SocketChannel channel = serverChannel.accept();
                if (channel == null) {
                    Thread.sleep(10);
                    continue;
                }
                
                channel.configureBlocking(true);
                channel.socket().setTcpNoDelay(true);
                channel.socket().setReceiveBufferSize(BUFFER_SIZE);
                channel.socket().setSendBufferSize(BUFFER_SIZE);
                
                ByteBuffer headerBuf = ByteBuffer.allocate(4);
                while (headerBuf.hasRemaining()) {
                    channel.read(headerBuf);
                }
                headerBuf.flip();
                int clientRank = headerBuf.getInt();
                
                ByteBuffer response = ByteBuffer.allocate(4);
                response.putInt(rank);
                response.flip();
                while (response.hasRemaining()) {
                    channel.write(response);
                }
                
                rankToChannel.put(clientRank, channel);
                
            }
        } catch (ClosedByInterruptException e) {
            // normal shutdown
        } catch (IOException e) {
            if (running) {
                System.err.println("[COMM] Accept error on rank " + rank + ": " + e.getMessage());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    private void connectToRank(int otherRank) throws InterruptedException {
        int attempts = 0;
        while (attempts < MAX_RETRIES && !rankToChannel.containsKey(otherRank)) {
            try {
                SocketChannel channel = SocketChannel.open();
                channel.configureBlocking(true);
                channel.socket().setTcpNoDelay(true);
                channel.socket().setReceiveBufferSize(BUFFER_SIZE);
                channel.socket().setSendBufferSize(BUFFER_SIZE);
                
                channel.connect(new InetSocketAddress(masterHost, masterPort + otherRank));
                
                ByteBuffer headerBuf = ByteBuffer.allocate(4);
                headerBuf.putInt(rank);
                headerBuf.flip();
                while (headerBuf.hasRemaining()) {
                    channel.write(headerBuf);
                }
                
                ByteBuffer responseBuf = ByteBuffer.allocate(4);
                while (responseBuf.hasRemaining()) {
                    channel.read(responseBuf);
                }
                responseBuf.flip();
                int remoteRank = responseBuf.getInt();
                
                if (remoteRank == otherRank) {
                    rankToChannel.put(otherRank, channel);
                    return;
                }
                
            } catch (ConnectException e) {
                attempts++;
                Thread.sleep(RETRY_DELAY_MS);
            } catch (IOException e) {
                attempts++;
                Thread.sleep(RETRY_DELAY_MS);
            }
        }
    }
    
    public void allReduce(float[] data, ReduceOp op) {
        if (worldSize == 1) return;
        
        if (worldSize <= 8) {
            treeAllReduce(data, op);
        } else {
            ringAllReduce(data, op);
        }
    }
    
    public void ringAllReduce(float[] data, ReduceOp op) {
        if (worldSize == 1) return;
        
        int leftNeighbor = (rank - 1 + worldSize) % worldSize;
        int rightNeighbor = (rank + 1) % worldSize;
        
        float[] sendBuf = Arrays.copyOf(data, data.length);
        float[] recvBuf = new float[data.length];
        
        for (int step = 0; step < worldSize - 1; step++) {
            final float[] finalSendBuf = sendBuf;
            
            Future<?> sendFuture = sendPool.submit(() -> 
                sendRaw(rightNeighbor, finalSendBuf));
            
            receiveRaw(leftNeighbor, recvBuf);
            
            applyReduce(sendBuf, recvBuf, op);
            
            try {
                sendFuture.get(30, TimeUnit.SECONDS);
            } catch (Exception e) {
                throw new RuntimeException("Ring all-reduce send failed", e);
            }
            
            float[] temp = sendBuf;
            sendBuf = recvBuf;
            recvBuf = temp;
        }
        
        System.arraycopy(sendBuf, 0, data, 0, data.length);
    }
    
    private void treeAllReduce(float[] data, ReduceOp op) {
        if (worldSize == 1) return;
        
        float[] temp = new float[data.length];
        
        for (int mask = 1; mask < worldSize; mask <<= 1) {
            int peer = rank ^ mask;
            
            if (peer < worldSize) {
                if ((rank & mask) == 0) {
                    receiveRaw(peer, temp);
                    applyReduce(data, temp, op);
                } else {
                    sendRaw(peer, data);
                    break;
                }
            }
        }
        
        for (int mask = highestPowerOfTwo(worldSize) >> 1; mask > 0; mask >>= 1) {
            int peer = rank ^ mask;
            if (peer < worldSize && (rank & mask) == 0) {
                sendRaw(peer, data);
            }
        }
    }
    
    private static int highestPowerOfTwo(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
    
    public Future<Void> asyncAllReduce(float[] data, ReduceOp op) {
        return sendPool.submit(() -> {
            allReduce(data, op);
            return null;
        });
    }
    
    public void broadcast(float[] data, int root) {
        if (worldSize == 1) return;
        
        if (rank == root) {
            List<Future<?>> futures = new ArrayList<>();
            for (int target = 0; target < worldSize; target++) {
                if (target != root) {
                    final int t = target;
                    futures.add(sendPool.submit(() -> sendRaw(t, data)));
                }
            }
            for (Future<?> f : futures) {
                try {
                    f.get(30, TimeUnit.SECONDS);
                } catch (Exception e) {
                    throw new RuntimeException("Broadcast send failed", e);
                }
            }
        } else {
            receiveRaw(root, data);
        }
    }
    
    public void allGather(float[] sendData, float[][] recvData) {
        if (worldSize == 1) {
            System.arraycopy(sendData, 0, recvData[0], 0, sendData.length);
            return;
        }
        
        recvData[rank] = Arrays.copyOf(sendData, sendData.length);
        
        for (int step = 0; step < worldSize - 1; step++) {
            int sendTo = (rank + 1) % worldSize;
            int recvFrom = (rank - 1 + worldSize) % worldSize;
            
            float[] toSend = recvData[(rank - step + worldSize) % worldSize];
            float[] toRecv = recvData[(rank - step - 1 + worldSize) % worldSize];
            
            final float[] finalSend = toSend;
            Future<?> sendFuture = sendPool.submit(() -> sendRaw(sendTo, finalSend));
            
            receiveRaw(recvFrom, toRecv);
            
            try {
                sendFuture.get(30, TimeUnit.SECONDS);
            } catch (Exception e) {
                throw new RuntimeException("All-gather send failed", e);
            }
        }
    }
    
    private void sendRaw(int targetRank, float[] data) {
        SocketChannel channel = getChannel(targetRank);
        
        try {
            ByteBuffer header = ByteBuffer.allocate(4);
            header.putInt(data.length);
            header.flip();
            while (header.hasRemaining()) {
                channel.write(header);
            }
            
            ByteBuffer body = ByteBuffer.allocate(data.length * 4);
            body.asFloatBuffer().put(data);
            while (body.hasRemaining()) {
                channel.write(body);
            }
            
            bytesSent.addAndGet((long) data.length * 4 + 4);
            messagesSent.incrementAndGet();
            
        } catch (IOException e) {
            throw new RuntimeException("Send failed to rank " + targetRank, e);
        }
    }
    
    private void receiveRaw(int sourceRank, float[] data) {
        SocketChannel channel = getChannel(sourceRank);
        
        try {
            ByteBuffer header = ByteBuffer.allocate(4);
            while (header.hasRemaining()) {
                int read = channel.read(header);
                if (read == -1) throw new EOFException("Connection closed from rank " + sourceRank);
            }
            header.flip();
            int length = header.getInt();
            
            if (length != data.length) {
                throw new RuntimeException(String.format(
                    "Length mismatch from rank %d: expected %d, got %d",
                    sourceRank, data.length, length));
            }
            
            ByteBuffer body = ByteBuffer.allocate(length * 4);
            while (body.hasRemaining()) {
                int read = channel.read(body);
                if (read == -1) throw new EOFException("Connection closed from rank " + sourceRank);
            }
            body.flip();
            body.asFloatBuffer().get(data);
            
            bytesReceived.addAndGet((long) length * 4 + 4);
            messagesReceived.incrementAndGet();
            
        } catch (IOException e) {
            throw new RuntimeException("Receive failed from rank " + sourceRank, e);
        }
    }
    
    public void sendBlocking(int targetRank, float[] data) {
        sendRaw(targetRank, data);
    }
    
    public Future<Void> sendAsync(int targetRank, float[] data) {
        return sendPool.submit(() -> {
            sendRaw(targetRank, data);
            return null;
        });
    }
    
    public void receiveBlocking(int sourceRank, float[] data) {
        receiveRaw(sourceRank, data);
    }
    
    public Future<float[]> receiveAsync(int sourceRank, int length) {
        return recvPool.submit(() -> {
            float[] data = new float[length];
            receiveRaw(sourceRank, data);
            return data;
        });
    }
    
    private SocketChannel getChannel(int targetRank) {
        SocketChannel channel = rankToChannel.get(targetRank);
        if (channel == null) {
            throw new IllegalStateException("No connection to rank " + targetRank);
        }
        return channel;
    }
    
    public void barrier() {
        if (worldSize == 1) return;
        
        try {
            barrier.await(60, TimeUnit.SECONDS);
        } catch (Exception e) {
            throw new RuntimeException("Barrier failed", e);
        }
    }
    
    private void applyReduce(float[] result, float[] other, ReduceOp op) {
        switch (op) {
            case SUM:
                for (int i = 0; i < result.length; i++) {
                    result[i] += other[i];
                }
                break;
            case AVG:
                for (int i = 0; i < result.length; i++) {
                    result[i] += other[i];
                }
                break;
            case MAX:
                for (int i = 0; i < result.length; i++) {
                    result[i] = Math.max(result[i], other[i]);
                }
                break;
            case MIN:
                for (int i = 0; i < result.length; i++) {
                    result[i] = Math.min(result[i], other[i]);
                }
                break;
        }
    }
    
    public int getRank() {
        return rank;
    }
    
    public int getWorldSize() {
        return worldSize;
    }
    
    public boolean isMaster() {
        return rank == 0;
    }
    
    public Map<String, Long> getStats() {
        Map<String, Long> stats = new LinkedHashMap<>();
        stats.put("bytesSent", bytesSent.get());
        stats.put("bytesReceived", bytesReceived.get());
        stats.put("messagesSent", messagesSent.get());
        stats.put("messagesReceived", messagesReceived.get());
        return stats;
    }
    
    public void resetStats() {
        bytesSent.set(0);
        bytesReceived.set(0);
        messagesSent.set(0);
        messagesReceived.set(0);
    }
    
    public void shutdown() {
        running = false;
        
        for (SocketChannel channel : rankToChannel.values()) {
            try {
                channel.close();
            } catch (IOException e) {
                // ignore
            }
        }
        rankToChannel.clear();
        
        if (serverChannel != null) {
            try {
                serverChannel.close();
            } catch (IOException e) {
                // ignore
            }
        }
        
        sendPool.shutdownNow();
        recvPool.shutdownNow();
        
        try {
            sendPool.awaitTermination(3, TimeUnit.SECONDS);
            recvPool.awaitTermination(3, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public enum ReduceOp {
        SUM, AVG, MAX, MIN
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
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: CommunicationLayer <rank> <worldSize>");
            System.exit(1);
        }
        
        int rank = Integer.parseInt(args[0]);
        int worldSize = Integer.parseInt(args[1]);
        
        CommunicationLayer comm = new CommunicationLayer(rank, worldSize);
        
        try {
            comm.init();
            System.out.println("[COMM] Rank " + rank + " initialized, connections: " 
                + comm.rankToChannel.size());
            
            float[] data = new float[100];
            for (int i = 0; i < data.length; i++) {
                data[i] = rank + i * 0.01f;
            }
            
            System.out.println("[COMM] Rank " + rank + " before allReduce: data[0]=" + data[0]);
            
            comm.allReduce(data, ReduceOp.SUM);
            
            float expected = 0;
            for (int r = 0; r < worldSize; r++) expected += r;
            System.out.println("[COMM] Rank " + rank + " after allReduce: data[0]=" + data[0] 
                + " (expected: " + expected + ")");
            
            comm.barrier();
            
            if (rank == 0) {
                System.out.println("[COMM] Stats: " + comm.getStats());
            }
            
        } finally {
            comm.shutdown();
        }
    }
}