package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.autograd.Tensor;
import com.qingxiaotuan.simpcore.distributed.CommunicationLayer;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.util.zip.*;

public class AsyncCheckpointer {
    
    private static final int MAGIC_NUMBER = 0x53433031;
    private static final int VERSION = 1;
    private static final int MAX_CHECKPOINTS = 10;
    
    private final String checkpointDir;
    private final int rank;
    private final int worldSize;
    private final CommunicationLayer comm;
    
    private final ExecutorService checkpointPool;
    private final ExecutorService restorePool;
    
    private final AtomicLong checkpointId = new AtomicLong(0);
    private final ReentrantLock checkpointLock = new ReentrantLock();
    private final ReentrantLock restoreLock = new ReentrantLock();
    
    private final Queue<Long> checkpointHistory = new ConcurrentLinkedQueue<>();
    
    private final boolean useIncrementalCheckpointing;
    private final boolean useCompression;
    private long lastCheckpointHash;
    
    private final AtomicLong totalBytesWritten = new AtomicLong(0);
    private final AtomicLong totalCheckpoints = new AtomicLong(0);
    
    public AsyncCheckpointer(String checkpointDir, int rank, int worldSize, CommunicationLayer comm) {
        this(checkpointDir, rank, worldSize, comm, false, false);
    }
    
    public AsyncCheckpointer(String checkpointDir, int rank, int worldSize, CommunicationLayer comm,
                            boolean useIncrementalCheckpointing, boolean useCompression) {
        this.checkpointDir = checkpointDir;
        this.rank = rank;
        this.worldSize = worldSize;
        this.comm = comm;
        this.useIncrementalCheckpointing = useIncrementalCheckpointing;
        this.useCompression = useCompression;
        this.lastCheckpointHash = 0;
        
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        this.checkpointPool = Executors.newFixedThreadPool(2,
            new ThreadFactoryBuilder("ckpt-save-"));
        this.restorePool = Executors.newFixedThreadPool(2,
            new ThreadFactoryBuilder("ckpt-restore-"));
        
        loadCheckpointId();
    }
    
    private void loadCheckpointId() {
        File[] files = new File(checkpointDir).listFiles((d, name) -> 
            name.startsWith("checkpoint_") && name.endsWith("_rank_" + rank + ".bin"));
        if (files != null) {
            long maxId = 0;
            for (File f : files) {
                try {
                    String[] parts = f.getName().split("_");
                    long id = Long.parseLong(parts[1]);
                    if (id > maxId) maxId = id;
                } catch (NumberFormatException ignored) {}
            }
            checkpointId.set(maxId);
        }
    }
    
    public Future<CheckpointResult> saveAsync(List<Tensor> parameters, int epoch, int step) {
        return checkpointPool.submit(() -> {
            checkpointLock.lock();
            try {
                long id = checkpointId.incrementAndGet();
                String filename = String.format("checkpoint_%d_epoch_%d_step_%d_rank_%d.bin", 
                    id, epoch, step, rank);
                String path = checkpointDir + File.separator + filename;
                
                long startTime = System.currentTimeMillis();
                long bytesWritten;
                
                List<Tensor> paramsToSave = parameters;
                List<Integer> changedIndices = null;
                
                if (useIncrementalCheckpointing && lastCheckpointHash != 0) {
                    changedIndices = computeChangedIndices(parameters);
                    if (changedIndices.isEmpty()) {
                        return new CheckpointResult(id, epoch, step, path, 0);
                    }
                }
                
                bytesWritten = writeCheckpoint(path, id, epoch, step, paramsToSave, changedIndices);
                
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                
                totalBytesWritten.addAndGet(bytesWritten);
                totalCheckpoints.incrementAndGet();
                
                lastCheckpointHash = computeParamsHash(parameters);
                checkpointHistory.add(id);
                
                cleanupOldCheckpoints();
                
                if (comm != null && worldSize > 1) {
                    comm.barrier();
                }
                
                if (rank == 0) {
                    saveMetadata(id, epoch, step, paramsToSave.size(), duration, bytesWritten);
                }
                
                return new CheckpointResult(id, epoch, step, path, duration);
                
            } finally {
                checkpointLock.unlock();
            }
        });
    }
    
    private long writeCheckpoint(String path, long id, int epoch, int step,
                                List<Tensor> parameters, List<Integer> changedIndices) throws IOException {
        long bytesWritten = 0;
        
        try (FileChannel channel = new FileOutputStream(path).getChannel()) {
            ByteBuffer header = ByteBuffer.allocate(32);
            header.putInt(MAGIC_NUMBER);
            header.putInt(VERSION);
            header.putLong(id);
            header.putInt(epoch);
            header.putInt(step);
            header.putInt(parameters.size());
            header.flip();
            bytesWritten += channel.write(header);
            
            OutputStream os = Channels.newOutputStream(channel);
            if (useCompression) {
                os = new GZIPOutputStream(os, 65536);
            }
            
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(os, 65536));
            
            for (Tensor param : parameters) {
                dos.writeInt(param.shape.length);
                for (int dim : param.shape) {
                    dos.writeInt(dim);
                }
                dos.writeInt(param.numel);
                
                byte[] floatBytes = new byte[param.numel * 4];
                ByteBuffer.wrap(floatBytes).asFloatBuffer().put(param.data);
                dos.write(floatBytes);
                bytesWritten += 4 + param.shape.length * 4 + 4 + param.numel * 4;
            }
            
            dos.flush();
            if (os instanceof GZIPOutputStream) {
                ((GZIPOutputStream) os).finish();
            }
            
            long checksum = computeParamsHash(parameters);
            ByteBuffer footer = ByteBuffer.allocate(8);
            footer.putLong(checksum);
            footer.flip();
            bytesWritten += channel.write(footer);
        }
        
        return bytesWritten;
    }
    
    private List<Integer> computeChangedIndices(List<Tensor> parameters) {
        List<Integer> changed = new ArrayList<>();
        long currentHash = computeParamsHash(parameters);
        
        if (currentHash != lastCheckpointHash) {
            for (int i = 0; i < parameters.size(); i++) {
                changed.add(i);
            }
        }
        return changed;
    }
    
    private long computeParamsHash(List<Tensor> parameters) {
        CRC32 crc = new CRC32();
        for (Tensor param : parameters) {
            byte[] bytes = new byte[param.numel * 4];
            ByteBuffer.wrap(bytes).asFloatBuffer().put(param.data);
            crc.update(bytes);
        }
        return crc.getValue();
    }
    
    public Future<List<Tensor>> restoreAsync(long checkpointId) {
        return restorePool.submit(() -> {
            restoreLock.lock();
            try {
                String path = findCheckpointPath(checkpointId);
                if (path == null) {
                    throw new FileNotFoundException("Checkpoint " + checkpointId + " not found for rank " + rank);
                }
                
                List<Tensor> parameters = new ArrayList<>();
                long storedChecksum = 0;
                
                try (FileChannel channel = new FileInputStream(path).getChannel()) {
                    ByteBuffer header = ByteBuffer.allocate(32);
                    channel.read(header);
                    header.flip();
                    
                    int magic = header.getInt();
                    int version = header.getInt();
                    
                    if (magic != MAGIC_NUMBER) {
                        throw new IOException("Invalid checkpoint file: bad magic number");
                    }
                    if (version != VERSION) {
                        throw new IOException("Unsupported checkpoint version: " + version);
                    }
                    
                    long id = header.getLong();
                    int epoch = header.getInt();
                    int step = header.getInt();
                    int numParams = header.getInt();
                    
                    InputStream is = Channels.newInputStream(channel);
                    if (useCompression) {
                        is = new GZIPInputStream(is, 65536);
                    }
                    DataInputStream dis = new DataInputStream(new BufferedInputStream(is, 65536));
                    
                    for (int i = 0; i < numParams; i++) {
                        int ndim = dis.readInt();
                        int[] shape = new int[ndim];
                        for (int j = 0; j < ndim; j++) {
                            shape[j] = dis.readInt();
                        }
                        int numel = dis.readInt();
                        
                        byte[] floatBytes = new byte[numel * 4];
                        dis.readFully(floatBytes);
                        
                        float[] data = new float[numel];
                        ByteBuffer.wrap(floatBytes).asFloatBuffer().get(data);
                        
                        parameters.add(new Tensor(data, shape));
                    }
                    
                    ByteBuffer footer = ByteBuffer.allocate(8);
                    channel.read(footer);
                    footer.flip();
                    storedChecksum = footer.getLong();
                }
                
                long computedChecksum = computeParamsHash(parameters);
                if (storedChecksum != computedChecksum) {
                    throw new IOException("Checkpoint checksum mismatch: stored=" + storedChecksum 
                        + ", computed=" + computedChecksum);
                }
                
                lastCheckpointHash = storedChecksum;
                
                if (comm != null && worldSize > 1) {
                    comm.barrier();
                }
                
                return parameters;
                
            } finally {
                restoreLock.unlock();
            }
        });
    }
    
    private String findCheckpointPath(long checkpointId) {
        File dir = new File(checkpointDir);
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith("checkpoint_" + checkpointId + "_") && 
            name.endsWith("_rank_" + rank + ".bin"));
        
        if (files != null && files.length > 0) {
            return files[0].getAbsolutePath();
        }
        return null;
    }
    
    private void saveMetadata(long id, int epoch, int step, int numParams, 
                             long duration, long bytesWritten) {
        String metadataPath = checkpointDir + File.separator + "metadata.json";
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(metadataPath, true))) {
            writer.write(String.format(
                "{\"id\":%d,\"epoch\":%d,\"step\":%d,\"numParams\":%d,\"duration\":%d,\"bytes\":%d,\"timestamp\":%d}\n",
                id, epoch, step, numParams, duration, bytesWritten, System.currentTimeMillis()));
        } catch (IOException e) {
            System.err.println("[CKPT] Failed to save metadata: " + e.getMessage());
        }
    }
    
    public List<CheckpointInfo> listCheckpoints() {
        List<CheckpointInfo> checkpoints = new ArrayList<>();
        File dir = new File(checkpointDir);
        
        if (!dir.exists()) {
            return checkpoints;
        }
        
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith("checkpoint_") && name.endsWith("_rank_0.bin"));
        if (files == null) {
            return checkpoints;
        }
        
        Arrays.sort(files, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
        
        for (File file : files) {
            String name = file.getName();
            try {
                String[] parts = name.split("_");
                long id = Long.parseLong(parts[1]);
                int epoch = Integer.parseInt(parts[3]);
                int step = Integer.parseInt(parts[5]);
                
                checkpoints.add(new CheckpointInfo(id, epoch, step, file.lastModified(), file.length()));
            } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                continue;
            }
        }
        
        return checkpoints;
    }
    
    public CheckpointInfo getLatestCheckpoint() {
        List<CheckpointInfo> checkpoints = listCheckpoints();
        return checkpoints.isEmpty() ? null : checkpoints.get(0);
    }
    
    public void cleanupOldCheckpoints() {
        List<CheckpointInfo> checkpoints = listCheckpoints();
        
        if (checkpoints.size() <= MAX_CHECKPOINTS) {
            return;
        }
        
        for (int i = MAX_CHECKPOINTS; i < checkpoints.size(); i++) {
            CheckpointInfo info = checkpoints.get(i);
            
            for (int r = 0; r < worldSize; r++) {
                final int rankNum = r;
                File dir = new File(checkpointDir);
                File[] files = dir.listFiles((d, name) -> 
                    name.startsWith("checkpoint_" + info.id + "_") && 
                    name.endsWith("_rank_" + rankNum + ".bin"));
                
                if (files != null) {
                    for (File f : files) {
                        f.delete();
                    }
                }
            }
        }
    }
    
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalCheckpoints", totalCheckpoints.get());
        stats.put("totalBytesWritten", totalBytesWritten.get());
        stats.put("currentCheckpointId", checkpointId.get());
        stats.put("incrementalCheckpointing", useIncrementalCheckpointing);
        stats.put("compression", useCompression);
        return stats;
    }
    
    public void shutdown() {
        checkpointPool.shutdown();
        restorePool.shutdown();
        
        try {
            checkpointPool.awaitTermination(10, TimeUnit.SECONDS);
            restorePool.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public static class CheckpointResult {
        public final long id;
        public final int epoch;
        public final int step;
        public final String path;
        public final long durationMs;
        
        public CheckpointResult(long id, int epoch, int step, String path, long durationMs) {
            this.id = id;
            this.epoch = epoch;
            this.step = step;
            this.path = path;
            this.durationMs = durationMs;
        }
    }
    
    public static class CheckpointInfo {
        public final long id;
        public final int epoch;
        public final int step;
        public final long timestamp;
        public final long sizeBytes;
        
        public CheckpointInfo(long id, int epoch, int step, long timestamp, long sizeBytes) {
            this.id = id;
            this.epoch = epoch;
            this.step = step;
            this.timestamp = timestamp;
            this.sizeBytes = sizeBytes;
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