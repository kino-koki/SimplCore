package com.qingxiaotuan.simpcore.gpu;

import org.jocl.*;

import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.jocl.CL.*;

public final class GPUBufferManager {

    private static final int BUFFER_POOL_SIZE = 64;
    private static final int MAX_BUFFER_SIZE = 16 * 1024 * 1024;

    private static final ConcurrentHashMap<Integer, GPUBuffer> bufferPool = new ConcurrentHashMap<>();
    private static final AtomicInteger bufferIdCounter = new AtomicInteger(0);
    private static final ConcurrentHashMap<Long, FloatBuffer> pinnedMemory = new ConcurrentHashMap<>();

    private static boolean unifiedMemorySupported = false;
    private static boolean svmSupported = false;

    static {
        try {
            detectMemoryFeatures();
        } catch (Exception e) {
            System.err.println("[GPU] BufferManager initialization failed: " + e.getMessage());
        }
    }

    private GPUBufferManager() {}

    private static void detectMemoryFeatures() {
        if (!GPUMatrixOperations.isGpuAvailable()) return;
        
        try {
            cl_device_id device = GPUMatrixOperations.getDeviceId(0);
            int[] unifiedMemory = new int[1];
            clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, Sizeof.cl_int, Pointer.to(unifiedMemory), null);
            unifiedMemorySupported = unifiedMemory[0] == 1;
            
            System.out.println("[GPU] Unified Memory Support: " + (unifiedMemorySupported ? "Enabled" : "Disabled"));
            
            int[] svmCapabilities = new int[1];
            clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, Sizeof.cl_int, Pointer.to(svmCapabilities), null);
            svmSupported = (svmCapabilities[0] & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0;
            
            System.out.println("[GPU] SVM Support: " + (svmSupported ? "Enabled" : "Disabled"));
        } catch (Exception e) {
            System.err.println("[GPU] Memory feature detection failed: " + e.getMessage());
        }
    }

    public static GPUBuffer allocate(int size) {
        return allocate(size, MemoryType.DEFAULT);
    }

    public static GPUBuffer allocate(int size, MemoryType memoryType) {
        int id = bufferIdCounter.incrementAndGet();
        FloatBuffer hostBuffer = FloatBuffer.allocate(size);
        
        long memoryFlags = CL_MEM_READ_WRITE;
        if (unifiedMemorySupported && memoryType != MemoryType.EXPLICIT_COPY) {
            memoryFlags |= CL_MEM_USE_HOST_PTR;
        } else {
            memoryFlags |= CL_MEM_COPY_HOST_PTR;
        }
        
        cl_mem deviceBuffer = clCreateBuffer(
            GPUMatrixOperations.getContext(),
            memoryFlags,
            Sizeof.cl_float * size,
            Pointer.to(hostBuffer),
            null
        );
        
        GPUBuffer buffer = new GPUBuffer(id, deviceBuffer, hostBuffer, size, memoryType);
        bufferPool.put(id, buffer);
        
        return buffer;
    }

    public static GPUBuffer wrap(float[] data) {
        return wrap(data, MemoryType.DEFAULT);
    }

    public static GPUBuffer wrap(float[] data, MemoryType memoryType) {
        int id = bufferIdCounter.incrementAndGet();
        FloatBuffer hostBuffer = FloatBuffer.wrap(data);
        
        long memoryFlags = CL_MEM_READ_WRITE;
        if (unifiedMemorySupported && memoryType != MemoryType.EXPLICIT_COPY) {
            memoryFlags |= CL_MEM_USE_HOST_PTR;
        } else {
            memoryFlags |= CL_MEM_COPY_HOST_PTR;
        }
        
        cl_mem deviceBuffer = clCreateBuffer(
            GPUMatrixOperations.getContext(),
            memoryFlags,
            Sizeof.cl_float * data.length,
            Pointer.to(hostBuffer),
            null
        );
        
        GPUBuffer buffer = new GPUBuffer(id, deviceBuffer, hostBuffer, data.length, memoryType);
        bufferPool.put(id, buffer);
        
        return buffer;
    }

    public static void release(GPUBuffer buffer) {
        if (buffer != null && bufferPool.containsKey(buffer.id)) {
            bufferPool.remove(buffer.id);
            clReleaseMemObject(buffer.deviceBuffer);
        }
    }

    public static void copyToDevice(GPUBuffer buffer) {
        if (buffer == null) return;
        
        if (unifiedMemorySupported && buffer.memoryType != MemoryType.EXPLICIT_COPY) {
            return;
        }
        
        clEnqueueWriteBuffer(
            GPUBuffer.getCommandQueue(),
            buffer.deviceBuffer,
            CL_TRUE,
            0,
            Sizeof.cl_float * buffer.size,
            Pointer.to(buffer.hostBuffer),
            0,
            null,
            null
        );
    }

    public static void copyToHost(GPUBuffer buffer) {
        if (buffer == null) return;
        
        if (unifiedMemorySupported && buffer.memoryType != MemoryType.EXPLICIT_COPY) {
            return;
        }
        
        clEnqueueReadBuffer(
            GPUBuffer.getCommandQueue(),
            buffer.deviceBuffer,
            CL_TRUE,
            0,
            Sizeof.cl_float * buffer.size,
            Pointer.to(buffer.hostBuffer),
            0,
            null,
            null
        );
    }

    public static void copyToDeviceAsync(GPUBuffer buffer, cl_event event) {
        if (buffer == null) return;
        
        if (unifiedMemorySupported && buffer.memoryType != MemoryType.EXPLICIT_COPY) {
            return;
        }
        
        clEnqueueWriteBuffer(
            GPUBuffer.getCommandQueue(),
            buffer.deviceBuffer,
            CL_FALSE,
            0,
            Sizeof.cl_float * buffer.size,
            Pointer.to(buffer.hostBuffer),
            0,
            null,
            event
        );
    }

    public static void copyToHostAsync(GPUBuffer buffer, cl_event event) {
        if (buffer == null) return;
        
        if (unifiedMemorySupported && buffer.memoryType != MemoryType.EXPLICIT_COPY) {
            return;
        }
        
        clEnqueueReadBuffer(
            GPUBuffer.getCommandQueue(),
            buffer.deviceBuffer,
            CL_FALSE,
            0,
            Sizeof.cl_float * buffer.size,
            Pointer.to(buffer.hostBuffer),
            0,
            null,
            event
        );
    }

    public static void copyBetweenBuffers(GPUBuffer src, GPUBuffer dst) {
        if (src == null || dst == null) return;
        
        long size = Math.min(src.size, dst.size) * Sizeof.cl_float;
        clEnqueueCopyBuffer(
            GPUBuffer.getCommandQueue(),
            src.deviceBuffer,
            dst.deviceBuffer,
            0,
            0,
            size,
            0,
            null,
            null
        );
    }

    public static int getActiveBufferCount() {
        return bufferPool.size();
    }

    public static boolean isUnifiedMemorySupported() {
        return unifiedMemorySupported;
    }

    public enum MemoryType {
        DEFAULT,
        EXPLICIT_COPY,
        PINNED,
        SVM
    }

    public static final class GPUBuffer {
        
        public final int id;
        public final cl_mem deviceBuffer;
        public final FloatBuffer hostBuffer;
        public final int size;
        public final MemoryType memoryType;
        
        private GPUBuffer(int id, cl_mem deviceBuffer, FloatBuffer hostBuffer, int size, MemoryType memoryType) {
            this.id = id;
            this.deviceBuffer = deviceBuffer;
            this.hostBuffer = hostBuffer;
            this.size = size;
            this.memoryType = memoryType;
        }
        
        public float[] getHostData() {
            return hostBuffer.array();
        }
        
        public void setHostData(float[] data) {
            hostBuffer.clear();
            hostBuffer.put(data);
            hostBuffer.flip();
        }
        
        public void ensureOnDevice() {
            GPUBufferManager.copyToDevice(this);
        }
        
        public void ensureOnHost() {
            GPUBufferManager.copyToHost(this);
        }
        
        public static cl_command_queue getCommandQueue() {
            return GPUMatrixOperations.getCommandQueue(0);
        }
    }
}