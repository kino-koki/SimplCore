package com.qingxiaotuan.simpcore.gpu;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.jocl.CL.*;

public final class GPUMatrixOperations {

    private static final String KERNEL_FILE = "/opencl/simpcore_kernels.cl";
    private static final int WARP_SIZE = 32;
    private static final int MAX_BATCH_SIZE = 64;
    private static int BLOCK_SIZE = 16; // adjusted at runtime for AMD/Intel GPUs

    private static boolean initialized = false;
    private static volatile boolean gpuAvailable = false;
    private static volatile boolean gpuErrorReported = false;
    private static int deviceCount = 0;
    private static int currentDevice = 0;

    private static cl_platform_id platformId = null;
    private static cl_device_id[] deviceIds = null;
    private static cl_context context = null;
    private static cl_command_queue[] commandQueues = null;
    private static cl_command_queue[] asyncCommandQueues = null;
    private static cl_program program = null;

    static final ConcurrentHashMap<String, cl_kernel> kernels = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Long, cl_mem> deviceBuffers = new ConcurrentHashMap<>();
    private static final AtomicInteger bufferIdCounter = new AtomicInteger(0);
    
    private static final AtomicLong totalGPUBytesTransferred = new AtomicLong(0);
    private static final AtomicLong totalGPUOperations = new AtomicLong(0);
    private static final AtomicLong totalGPUMilliseconds = new AtomicLong(0);

    static {
        gpuAvailable = false;
        Thread initThread = new Thread(() -> {
            try {
                long start = System.currentTimeMillis();
                initOpenCL();
                long elapsed = System.currentTimeMillis() - start;
                gpuAvailable = true;
                
                String deviceName = getString(deviceIds[0], CL_DEVICE_NAME);
                String platformName = getString(platformId, CL_PLATFORM_NAME);
                String deviceType = getString(deviceIds[0], CL_DEVICE_TYPE);
                System.out.println("[GPU] OpenCL initialized successfully in " + elapsed + " ms");
                System.out.println("[GPU] Platform: " + platformName);
                System.out.println("[GPU] Device: " + deviceName + " (" + deviceType + ")");
            } catch (UnsatisfiedLinkError e) {
                System.out.println("[GPU] OpenCL init failed: Native library not found - " + e.getMessage());
            } catch (Throwable t) {
                System.out.println("[GPU] OpenCL init failed: " + t.getClass().getName() + " - " + t.getMessage());
                t.printStackTrace();
            }
        }, "gpu-init");
        initThread.setDaemon(true);
        initThread.start();
        try {
            initThread.join(15000);
        } catch (InterruptedException e) {
            // timeout, proceed with CPU fallback
        }
        if (initThread.isAlive()) {
            initThread.interrupt();
            System.out.println("[GPU] OpenCL init timed out (15s) — falling back to CPU");
        }
        if (!gpuAvailable) {
            System.out.println("[GPU] Using CPU path for all matrix operations");
        }
    }
    
    public static boolean tryReinitialize() {
        if (gpuAvailable) return true;
        try {
            initOpenCL();
            gpuAvailable = true;
            String deviceName = getString(deviceIds[0], CL_DEVICE_NAME);
            System.out.println("[GPU] Reinitialization successful on " + deviceName);
            return true;
        } catch (Throwable t) {
            System.out.println("[GPU] Reinitialization failed: " + t.getMessage());
            return false;
        }
    }

    private GPUMatrixOperations() {}

    private static void initOpenCL() throws Exception {
        CL.setExceptionsEnabled(true);

        int[] numPlatforms = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);

        if (numPlatforms[0] == 0) {
            throw new Exception("No OpenCL platforms found");
        }

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(numPlatforms[0], platforms, null);

        platformId = platforms[0];

        for (cl_platform_id platform : platforms) {
            String platformName = getString(platform, CL_PLATFORM_NAME);
            if (platformName.contains("NVIDIA") || platformName.contains("AMD") || 
                platformName.contains("Intel") || platformName.contains("Advanced Micro Devices")) {
                platformId = platform;
                break;
            }
        }

        int[] numDevices = new int[1];
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, null, numDevices);

        if (numDevices[0] == 0) {
            System.out.println("[GPU] No GPU devices found, falling back to all devices");
            clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        }

        if (numDevices[0] == 0) {
            throw new Exception("No OpenCL devices found");
        }

        deviceCount = numDevices[0];
        deviceIds = new cl_device_id[deviceCount];
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds, null);

        context = clCreateContext(null, deviceCount, deviceIds, null, null, null);

        commandQueues = new cl_command_queue[deviceCount];
        asyncCommandQueues = new cl_command_queue[deviceCount];
        for (int i = 0; i < deviceCount; i++) {
            commandQueues[i] = clCreateCommandQueue(context, deviceIds[i], CL_QUEUE_PROFILING_ENABLE, null);
            asyncCommandQueues[i] = clCreateCommandQueue(context, deviceIds[i], 0, null);
        }

        // AMD GPUs may need smaller block size — detect and adjust
        String deviceName = getString(deviceIds[0], CL_DEVICE_NAME);
        boolean isAmd = deviceName.contains("AMD") || deviceName.contains("Advanced Micro Devices");
        boolean isIntel = deviceName.contains("Intel");
        if (isAmd || isIntel) {
            BLOCK_SIZE = 8;
            System.out.println("[GPU] Adjusted BLOCK_SIZE to " + BLOCK_SIZE + " for " + deviceName);
        }

        String kernelSource = loadKernelSource();
        program = clCreateProgramWithSource(context, 1, new String[]{kernelSource}, null, null);

        int buildErr = clBuildProgram(program, deviceCount, deviceIds, "-cl-fast-relaxed-math -cl-mad-enable", null, null);
        if (buildErr != CL_SUCCESS) {
            byte[] log = new byte[8192];
            clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 8192, Pointer.to(log), null);
            String buildLog = new String(log, 0, log.length).trim();
            throw new Exception("OpenCL kernel build failed: " + buildLog);
        }

        String[] kernelNames = {
            "matmul_kernel", "matmul_shared_kernel", "matmul_vectorized_kernel", "matmul_relu_kernel", "matmul_silu_kernel",
            "matmul_bias_silu_kernel", "matmul_bias_gelu_kernel",
            "layer_norm_kernel", "layer_norm_fused_kernel", "rms_norm_kernel", "rms_norm_fused_kernel",
            "softmax_kernel", "relu_kernel", "silu_kernel", "gelu_kernel",
            "relu_silu_kernel", "add_kernel", "sub_kernel", "mul_kernel",
            "hadamard_kernel", "dot_product_kernel", "outer_product_kernel",
            "transpose_kernel", "norm_kernel", "add_bias_kernel",
            "adam_kernel", "adamw_kernel", "dropout_kernel", "scaled_dot_product_kernel",
            "fused_add_norm_kernel", "clip_grad_kernel",
            "layer_norm_grad_kernel", "matmul_grad_kernel", "gelu_grad_kernel",
            "silu_grad_kernel", "softmax_grad_kernel", "embedding_grad_kernel",
            "cross_entropy_grad_kernel"
        };

        for (String name : kernelNames) {
            try {
                cl_kernel kernel = clCreateKernel(program, name, null);
                kernels.put(name, kernel);
            } catch (Exception e) {
                System.err.println("[GPU] Failed to create kernel: " + name + " - " + e.getMessage());
            }
        }

        System.out.println("[GPU] OpenCL initialized successfully");
        System.out.println("[GPU] Platform: " + getString(platformId, CL_PLATFORM_NAME));
        System.out.println("[GPU] Device: " + getString(deviceIds[0], CL_DEVICE_NAME));
        System.out.println("[GPU] Compute Units: " + getComputeUnits(deviceIds[0]));
    }

    private static String loadKernelSource() throws Exception {
        InputStream is = GPUMatrixOperations.class.getResourceAsStream(KERNEL_FILE);
        if (is == null) {
            return generateDefaultKernels();
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            return sb.toString();
        }
    }

    private static String generateDefaultKernels() {
        return """
            __kernel void matmul_kernel(__global const float* A, __global const float* B, __global float* C, int M, int K, int N) {
                int row = get_global_id(1);
                int col = get_global_id(0);
                if (row < M && col < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[row * K + k] * B[k * N + col];
                    }
                    C[row * N + col] = sum;
                }
            }
            
            __kernel void matmul_shared_kernel(__global const float* A, __global const float* B, __global float* C, int M, int K, int N, __local float* sharedA, __local float* sharedB) {
                int tx = get_local_id(0);
                int ty = get_local_id(1);
                int row = get_global_id(1);
                int col = get_global_id(0);
                float sum = 0.0f;
                int TILE = 16;
                
                for (int t = 0; t < K / TILE + (K % TILE != 0); t++) {
                    if (row < M && t * TILE + tx < K) {
                        sharedA[ty * TILE + tx] = A[row * K + t * TILE + tx];
                    } else {
                        sharedA[ty * TILE + tx] = 0.0f;
                    }
                    if (col < N && t * TILE + ty < K) {
                        sharedB[ty * TILE + tx] = B[(t * TILE + ty) * N + col];
                    } else {
                        sharedB[ty * TILE + tx] = 0.0f;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    for (int k = 0; k < TILE; k++) {
                        sum += sharedA[ty * TILE + k] * sharedB[k * TILE + tx];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                if (row < M && col < N) {
                    C[row * N + col] = sum;
                }
            }
            
            __kernel void softmax_kernel(__global float* logits, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    float max_val = logits[0];
                    for (int i = 1; i < length; i++) {
                        if (logits[i] > max_val) max_val = logits[i];
                    }
                    float sum_exp = 0.0f;
                    for (int i = 0; i < length; i++) {
                        sum_exp += exp(logits[i] - max_val);
                    }
                    logits[idx] = exp(logits[idx] - max_val) / sum_exp;
                }
            }
            
            __kernel void relu_kernel(__global float* x, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    x[idx] = max(x[idx], 0.0f);
                }
            }
            
            __kernel void silu_kernel(__global float* x, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    float sigmoid = 1.0f / (1.0f + exp(-x[idx]));
                    x[idx] = x[idx] * sigmoid;
                }
            }
            
            __kernel void add_kernel(__global const float* A, __global const float* B, __global float* C, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    C[idx] = A[idx] + B[idx];
                }
            }
            
            __kernel void sub_kernel(__global const float* A, __global const float* B, __global float* C, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    C[idx] = A[idx] - B[idx];
                }
            }
            
            __kernel void mul_kernel(__global const float* A, float scalar, __global float* C, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    C[idx] = A[idx] * scalar;
                }
            }
            
            __kernel void dot_product_kernel(__global const float* a, __global const float* b, __global float* result, int length, __local float* shared) {
                int idx = get_local_id(0);
                shared[idx] = 0.0f;
                for (int i = idx; i < length; i += 32) {
                    shared[idx] += a[i] * b[i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (idx == 0) {
                    float sum = 0.0f;
                    for (int i = 0; i < 32; i++) {
                        sum += shared[i];
                    }
                    result[0] = sum;
                }
            }
            
            __kernel void outer_product_kernel(__global const float* a, __global const float* b, __global float* C, int m, int n) {
                int row = get_global_id(1);
                int col = get_global_id(0);
                if (row < m && col < n) {
                    C[row * n + col] = a[row] * b[col];
                }
            }
            
            __kernel void transpose_kernel(__global const float* A, __global float* B, int rows, int cols) {
                int row = get_global_id(1);
                int col = get_global_id(0);
                if (row < rows && col < cols) {
                    B[col * rows + row] = A[row * cols + col];
                }
            }
            
            __kernel void norm_kernel(__global const float* x, __global float* result, int length) {
                int idx = get_global_id(0);
                if (idx == 0) {
                    float sum = 0.0f;
                    for (int i = 0; i < length; i++) {
                        sum += x[i] * x[i];
                    }
                    result[0] = sqrt(sum);
                }
            }
            
            __kernel void hadamard_kernel(__global const float* a, __global const float* b, __global float* c, int length) {
                int idx = get_global_id(0);
                if (idx < length) {
                    c[idx] = a[idx] * b[idx];
                }
            }
            """;
    }

    private static String getString(cl_platform_id platform, int paramName) {
        long size[] = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);
        byte buffer[] = new byte[(int) size[0]];
        clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1);
    }

    private static String getString(cl_device_id device, int paramName) {
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte buffer[] = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1);
    }

    private static int getComputeUnits(cl_device_id device) {
        try {
            int[] units = new int[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_int, Pointer.to(units), null);
            return units[0];
        } catch (Exception e) {
            return 0;
        }
    }

    public static boolean isGpuAvailable() {
        return gpuAvailable;
    }

    public static int getDeviceCount() {
        return deviceCount;
    }

    public static int getCurrentDevice() {
        return currentDevice;
    }

    public static cl_device_id getDeviceId(int index) {
        if (index >= 0 && index < deviceCount) {
            return deviceIds[index];
        }
        return null;
    }

    public static cl_context getContext() {
        return context;
    }

    public static cl_command_queue getCommandQueue(int index) {
        if (index >= 0 && index < deviceCount) {
            return commandQueues[index];
        }
        return null;
    }

    public static void setDevice(int deviceId) {
        if (deviceId >= 0 && deviceId < deviceCount) {
            currentDevice = deviceId;
        }
    }

    public static float[] matmul(float[] A, float[] B, int M, int K, int N) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
        }
        return matmulGPU(A, B, M, K, N);
    }

    private static float[] matmulGPU(float[] A, float[] B, int M, int K, int N) {
        long startTime = System.nanoTime();
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[M * N];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            totalGPUBytesTransferred.addAndGet((long)(A.length + B.length) * 4);

            cl_kernel kernel;
            if (K > BLOCK_SIZE && M > BLOCK_SIZE && N > BLOCK_SIZE) {
                kernel = kernels.get("matmul_shared_kernel");
            } else {
                kernel = kernels.get("matmul_kernel");
            }

            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{N}));

            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            totalGPUBytesTransferred.addAndGet((long)result.length * 4);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            if (!gpuErrorReported) {
                System.err.println("[GPU] matmul failed: " + e.getMessage() + " - falling back to CPU");
                gpuErrorReported = true;
            }
            gpuAvailable = false;
            return com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
        }

        totalGPUOperations.incrementAndGet();
        totalGPUMilliseconds.addAndGet((System.nanoTime() - startTime) / 1_000_000);
        return result;
    }

    public static float[] softmax(float[] logits) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(logits);
        }
        return softmaxGPU(logits);
    }

    private static float[] softmaxGPU(float[] logits) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = logits.clone();

        try {
            cl_mem d_logits = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * result.length, Pointer.to(result), null);

            cl_kernel kernel = kernels.get("softmax_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(logits);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_logits));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{result.length}));

            long[] globalWorkSize = new long[]{(result.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_logits, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_logits);

        } catch (Exception e) {
            if (!gpuErrorReported) {
                System.err.println("[GPU] softmax failed: " + e.getMessage() + " - falling back to CPU");
                gpuErrorReported = true;
            }
            gpuAvailable = false;
            return com.qingxiaotuan.simpcore.math.MatrixOperations.softmax(logits);
        }

        return result;
    }

    public static float[] relu(float[] x) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.relu(x);
        }
        return reluGPU(x);
    }

    private static float[] reluGPU(float[] x) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[x.length];

        try {
            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * x.length, Pointer.to(x), null);
            cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("relu_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.relu(x);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_result));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{x.length}));

            long[] globalWorkSize = new long[]{(x.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueCopyBuffer(queue, d_x, d_result, 0, 0,
                Sizeof.cl_float * x.length, 0, null, null);

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_x);
            clReleaseMemObject(d_result);

        } catch (Exception e) {
            if (!gpuErrorReported) {
                System.err.println("[GPU] relu failed: " + e.getMessage() + " - falling back to CPU");
                gpuErrorReported = true;
            }
            gpuAvailable = false;
            return com.qingxiaotuan.simpcore.math.MatrixOperations.relu(x);
        }

        return result;
    }

    public static float[] silu(float[] x) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(x);
        }
        return siluGPU(x);
    }

    private static float[] siluGPU(float[] x) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[x.length];

        try {
            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * x.length, Pointer.to(x), null);
            cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("silu_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(x);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_result));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{x.length}));

            long[] globalWorkSize = new long[]{(x.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueCopyBuffer(queue, d_x, d_result, 0, 0,
                Sizeof.cl_float * x.length, 0, null, null);

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_x);
            clReleaseMemObject(d_result);

        } catch (Exception e) {
            if (!gpuErrorReported) {
                System.err.println("[GPU] silu failed: " + e.getMessage() + " - falling back to CPU");
                gpuErrorReported = true;
            }
            gpuAvailable = false;
            return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(x);
        }

        return result;
    }

    public static float[] add(float[] A, float[] B) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.add(A, B);
        }
        return addGPU(A, B);
    }

    private static float[] addGPU(float[] A, float[] B) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[A.length];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("add_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.add(A, B);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{A.length}));

            long[] globalWorkSize = new long[]{(A.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] add failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.add(A, B);
        }

        return result;
    }

    public static float[] sub(float[] A, float[] B) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.sub(A, B);
        }
        return subGPU(A, B);
    }

    private static float[] subGPU(float[] A, float[] B) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[A.length];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("sub_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.sub(A, B);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{A.length}));

            long[] globalWorkSize = new long[]{(A.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] sub failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.sub(A, B);
        }

        return result;
    }

    public static float[] mul(float[] A, float scalar) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.mul(A, scalar);
        }
        return mulGPU(A, scalar);
    }

    private static float[] mulGPU(float[] A, float scalar) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[A.length];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("mul_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.mul(A, scalar);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{scalar}));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{A.length}));

            long[] globalWorkSize = new long[]{(A.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] mul failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.mul(A, scalar);
        }

        return result;
    }

    public static float dotProduct(float[] a, float[] b) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(a, b);
        }
        return dotProductGPU(a, b);
    }

    private static float dotProductGPU(float[] a, float[] b) {
        cl_command_queue queue = commandQueues[currentDevice];
        
        int numGroups = Math.max(1, (a.length + WARP_SIZE - 1) / WARP_SIZE);
        float[] partialSums = new float[numGroups];

        try {
            cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * a.length, Pointer.to(a), null);
            cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * b.length, Pointer.to(b), null);
            cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * numGroups, null, null);

            cl_kernel kernel = kernels.get("dot_product_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(a, b);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_b));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_result));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{a.length}));

            long[] globalWorkSize = new long[]{numGroups * WARP_SIZE};
            long[] localWorkSize = new long[]{WARP_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0,
                Sizeof.cl_float * numGroups, Pointer.to(partialSums), 0, null, null);

            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseMemObject(d_result);

            float totalSum = 0.0f;
            for (float partial : partialSums) {
                totalSum += partial;
            }
            return totalSum;

        } catch (Exception e) {
            System.err.println("[GPU] dotProduct failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.dotProduct(a, b);
        }
    }

    public static float[] outerProduct(float[] a, float[] b, int m, int n) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.outerProduct(a, b, m, n);
        }
        return outerProductGPU(a, b, m, n);
    }

    private static float[] outerProductGPU(float[] a, float[] b, int m, int n) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[m * n];

        try {
            cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * m, Pointer.to(a), null);
            cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, Pointer.to(b), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("outer_product_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.outerProduct(a, b, m, n);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_b));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

            long[] globalWorkSize = new long[]{n, m};
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            if (!gpuErrorReported) {
                System.err.println("[GPU] outerProduct failed: " + e.getMessage() + " - falling back to CPU");
                gpuErrorReported = true;
            }
            gpuAvailable = false;
            return com.qingxiaotuan.simpcore.math.MatrixOperations.outerProduct(a, b, m, n);
        }

        return result;
    }

    public static float[] transpose(float[] A, int rows, int cols) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.transpose(A, rows, cols);
        }
        return transposeGPU(A, rows, cols);
    }

    private static float[] transposeGPU(float[] A, int rows, int cols) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[rows * cols];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("transpose_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.transpose(A, rows, cols);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{rows}));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{cols}));

            long[] globalWorkSize = new long[]{cols, rows};
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_B, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);

        } catch (Exception e) {
            System.err.println("[GPU] transpose failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.transpose(A, rows, cols);
        }

        return result;
    }

    public static float[] hadamardProduct(float[] a, float[] b) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.hadamardProduct(a, b);
        }
        return hadamardProductGPU(a, b);
    }

    private static float[] hadamardProductGPU(float[] a, float[] b) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[a.length];

        try {
            cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * a.length, Pointer.to(a), null);
            cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * b.length, Pointer.to(b), null);
            cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("hadamard_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.hadamardProduct(a, b);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_a));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_b));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_c));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{a.length}));

            long[] globalWorkSize = new long[]{(a.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseMemObject(d_c);

        } catch (Exception e) {
            System.err.println("[GPU] hadamardProduct failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.hadamardProduct(a, b);
        }

        return result;
    }

    public static float norm(float[] x) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.norm(x);
        }
        return normGPU(x);
    }

    private static float normGPU(float[] x) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[]{0.0f};

        try {
            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * x.length, Pointer.to(x), null);
            cl_mem d_result = clCreateBuffer(context, CL_MEM_READ_WRITE,
                Sizeof.cl_float, Pointer.to(result), null);

            cl_kernel kernel = kernels.get("norm_kernel");
            if (kernel == null) {
                return com.qingxiaotuan.simpcore.math.MatrixOperations.norm(x);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_x));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_result));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{x.length}));

            long[] globalWorkSize = new long[]{1};
            long[] localWorkSize = new long[]{1};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0,
                Sizeof.cl_float, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_x);
            clReleaseMemObject(d_result);

        } catch (Exception e) {
            System.err.println("[GPU] norm failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.norm(x);
        }

        return result[0];
    }

    public static long allocatePinnedMemory(int size) {
        if (!gpuAvailable) return -1;
        try {
            cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                Sizeof.cl_float * size, null, null);
            long id = bufferIdCounter.incrementAndGet();
            deviceBuffers.put(id, buffer);
            return id;
        } catch (Exception e) {
            return -1;
        }
    }

    public static void releasePinnedMemory(long handle) {
        if (gpuAvailable && handle != -1) {
            cl_mem buffer = deviceBuffers.remove(handle);
            if (buffer != null) {
                clReleaseMemObject(buffer);
            }
        }
    }

    public static void copyToGPU(float[] hostData, long devicePointer, int size) {
        if (!gpuAvailable || devicePointer == -1) return;
        cl_mem buffer = deviceBuffers.get(devicePointer);
        if (buffer != null) {
            cl_command_queue queue = commandQueues[currentDevice];
            clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
                Sizeof.cl_float * size, Pointer.to(hostData), 0, null, null);
        }
    }

    public static void copyFromGPU(long devicePointer, float[] hostData, int size) {
        if (!gpuAvailable || devicePointer == -1) return;
        cl_mem buffer = deviceBuffers.get(devicePointer);
        if (buffer != null) {
            cl_command_queue queue = commandQueues[currentDevice];
            clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                Sizeof.cl_float * size, Pointer.to(hostData), 0, null, null);
        }
    }

    public static String getDeviceInfo() {
        if (!gpuAvailable) return "GPU not available";
        try {
            cl_device_id device = deviceIds[currentDevice];
            StringBuilder info = new StringBuilder();
            info.append("Device: ").append(getString(device, CL_DEVICE_NAME)).append("\n");
            info.append("Vendor: ").append(getString(device, CL_DEVICE_VENDOR)).append("\n");
            
            int[] computeUnits = new int[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_int, Pointer.to(computeUnits), null);
            info.append("Compute Units: ").append(computeUnits[0]).append("\n");
            
            long[] globalMem = new long[1];
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMem), null);
            info.append("Global Memory: ").append(globalMem[0] / (1024 * 1024)).append(" MB\n");
            
            int[] clockFreq = new int[1];
            clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, Sizeof.cl_int, Pointer.to(clockFreq), null);
            info.append("Clock Frequency: ").append(clockFreq[0]).append(" MHz\n");
            
            info.append("Platform: ").append(getString(platformId, CL_PLATFORM_NAME));
            
            return info.toString();
        } catch (Exception e) {
            return "Error getting device info: " + e.getMessage();
        }
    }

    public static void synchronize() {
        if (gpuAvailable) {
            clFinish(commandQueues[currentDevice]);
        }
    }

    public static void shutdown() {
        if (gpuAvailable) {
            for (cl_kernel kernel : kernels.values()) {
                clReleaseKernel(kernel);
            }
            kernels.clear();

            for (cl_mem buffer : deviceBuffers.values()) {
                clReleaseMemObject(buffer);
            }
            deviceBuffers.clear();

            for (int i = 0; i < deviceCount; i++) {
                clReleaseCommandQueue(commandQueues[i]);
                clReleaseCommandQueue(asyncCommandQueues[i]);
            }

            clReleaseProgram(program);
            clReleaseContext(context);

            gpuAvailable = false;
        }
    }

    public static String getPerformanceStats() {
        long bytes = totalGPUBytesTransferred.get();
        long ops = totalGPUOperations.get();
        long ms = totalGPUMilliseconds.get();
        
        StringBuilder sb = new StringBuilder();
        sb.append("GPU Performance Statistics:\n");
        sb.append("  Total Operations: ").append(ops).append("\n");
        sb.append("  Total Bytes Transferred: ").append(bytes / (1024 * 1024)).append(" MB\n");
        sb.append("  Total Time: ").append(ms).append(" ms\n");
        if (ms > 0) {
            sb.append("  Average Time per Operation: ").append(String.format("%.2f", (double)ms / ops)).append(" ms\n");
            sb.append("  Throughput: ").append(String.format("%.2f", (double)bytes / (1024 * 1024 * (ms / 1000.0)))).append(" MB/s");
        }
        return sb.toString();
    }

    public static void resetPerformanceStats() {
        totalGPUBytesTransferred.set(0);
        totalGPUOperations.set(0);
        totalGPUMilliseconds.set(0);
    }

    public static class GPUStats {
        private final int deviceCount;
        private final String[] deviceNames;
        private final long[] memoryTotal;
        private final long[] memoryFree;
        private final int[] computeCapability;

        public GPUStats(int deviceCount, String[] deviceNames,
                        long[] memoryTotal, long[] memoryFree, int[] computeCapability) {
            this.deviceCount = deviceCount;
            this.deviceNames = deviceNames;
            this.memoryTotal = memoryTotal;
            this.memoryFree = memoryFree;
            this.computeCapability = computeCapability;
        }

        public int getDeviceCount() { return deviceCount; }
        public String[] getDeviceNames() { return deviceNames; }
        public long[] getMemoryTotal() { return memoryTotal; }
        public long[] getMemoryFree() { return memoryFree; }
        public int[] getComputeCapability() { return computeCapability; }
    }

    public static GPUStats getGPUStats() {
        if (!gpuAvailable) {
            return new GPUStats(0, new String[0], new long[0], new long[0], new int[0]);
        }

        String[] deviceNames = new String[deviceCount];
        long[] memoryTotal = new long[deviceCount];
        long[] memoryFree = new long[deviceCount];
        int[] computeCapability = new int[deviceCount];

        try {
            for (int i = 0; i < deviceCount; i++) {
                cl_device_id device = deviceIds[i];
                deviceNames[i] = getString(device, CL_DEVICE_NAME);

                long[] globalMem = new long[1];
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMem), null);
                memoryTotal[i] = globalMem[0];

                int[] maxUnits = new int[1];
                clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_int, Pointer.to(maxUnits), null);
                computeCapability[i] = maxUnits[0];
            }
        } catch (Exception e) {
            System.err.println("[GPU] Error getting stats: " + e.getMessage());
        }

        return new GPUStats(deviceCount, deviceNames, memoryTotal, memoryFree, computeCapability);
    }

    public static float[] matmulRelu(float[] A, float[] B, int M, int K, int N) {
        if (!gpuAvailable) {
            float[] result = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
            return com.qingxiaotuan.simpcore.math.MatrixOperations.relu(result);
        }
        return matmulReluGPU(A, B, M, K, N);
    }

    private static float[] matmulReluGPU(float[] A, float[] B, int M, int K, int N) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[M * N];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("matmul_relu_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_A);
                clReleaseMemObject(d_B);
                clReleaseMemObject(d_C);
                return matmulRelu(A, B, M, K, N);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{N}));

            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] matmulRelu failed: " + e.getMessage());
            return matmulRelu(A, B, M, K, N);
        }

        return result;
    }

    public static float[] matmulSilu(float[] A, float[] B, int M, int K, int N) {
        if (!gpuAvailable) {
            float[] result = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
            return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(result);
        }
        return matmulSiluGPU(A, B, M, K, N);
    }

    private static float[] matmulSiluGPU(float[] A, float[] B, int M, int K, int N) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[M * N];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("matmul_silu_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_A);
                clReleaseMemObject(d_B);
                clReleaseMemObject(d_C);
                return matmulSilu(A, B, M, K, N);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{N}));

            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};

            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] matmulSilu failed: " + e.getMessage());
            return matmulSilu(A, B, M, K, N);
        }

        return result;
    }

    public static float[] layerNorm(float[] input, float[] gamma, float[] beta, int batchSize, int hiddenSize) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.layerNorm(input, gamma, beta, batchSize, hiddenSize);
        }
        return layerNormGPU(input, gamma, beta, batchSize, hiddenSize);
    }

    private static float[] layerNormGPU(float[] input, float[] gamma, float[] beta, int batchSize, int hiddenSize) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[input.length];

        try {
            cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
            cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);
            cl_mem d_gamma = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * gamma.length, Pointer.to(gamma), null);
            cl_mem d_beta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * beta.length, Pointer.to(beta), null);

            cl_kernel kernel = kernels.get("layer_norm_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_input);
                clReleaseMemObject(d_output);
                clReleaseMemObject(d_gamma);
                clReleaseMemObject(d_beta);
                return layerNorm(input, gamma, beta, batchSize, hiddenSize);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_gamma));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_beta));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_input);
            clReleaseMemObject(d_output);
            clReleaseMemObject(d_gamma);
            clReleaseMemObject(d_beta);

        } catch (Exception e) {
            System.err.println("[GPU] layerNorm failed: " + e.getMessage());
            return layerNorm(input, gamma, beta, batchSize, hiddenSize);
        }

        return result;
    }

    public static float[] addBias(float[] input, float[] bias, int batchSize, int hiddenSize) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.addBias(input, bias, batchSize, hiddenSize);
        }
        return addBiasGPU(input, bias, batchSize, hiddenSize);
    }

    private static float[] addBiasGPU(float[] input, float[] bias, int batchSize, int hiddenSize) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[input.length];

        try {
            cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
            cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * bias.length, Pointer.to(bias), null);

            cl_kernel kernel = kernels.get("add_bias_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_input);
                clReleaseMemObject(d_bias);
                return addBias(input, bias, batchSize, hiddenSize);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_bias));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_input);
            clReleaseMemObject(d_bias);

        } catch (Exception e) {
            System.err.println("[GPU] addBias failed: " + e.getMessage());
            return addBias(input, bias, batchSize, hiddenSize);
        }

        return result;
    }

    public static float[] gelu(float[] x) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.gelu(x);
        }
        return geluGPU(x);
    }

    private static float[] geluGPU(float[] x) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[x.length];

        try {
            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * x.length, Pointer.to(x), null);

            cl_kernel kernel = kernels.get("gelu_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_x);
                return gelu(x);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_x));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{x.length}));

            long[] globalWorkSize = new long[]{(x.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_x, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_x);

        } catch (Exception e) {
            System.err.println("[GPU] gelu failed: " + e.getMessage());
            return gelu(x);
        }

        return result;
    }

    public static float[] hadamard(float[] A, float[] B) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.hadamardProduct(A, B);
        }
        return hadamardGPU(A, B);
    }

    private static float[] hadamardGPU(float[] A, float[] B) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[A.length];

        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("hadamard_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_A);
                clReleaseMemObject(d_B);
                clReleaseMemObject(d_C);
                return hadamard(A, B);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{A.length}));

            long[] globalWorkSize = new long[]{(A.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);

        } catch (Exception e) {
            System.err.println("[GPU] hadamard failed: " + e.getMessage());
            return hadamard(A, B);
        }

        return result;
    }

    public static void matmulGPU(cl_mem d_A, cl_mem d_B, cl_mem d_C, int M, int K, int N) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel;
            if (K > BLOCK_SIZE && M > BLOCK_SIZE && N > BLOCK_SIZE) {
                kernel = kernels.get("matmul_shared_kernel");
            } else {
                kernel = kernels.get("matmul_kernel");
            }
            
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{N}));
            
            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};
            
            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);
            
        } catch (Exception e) {
            System.err.println("[GPU] matmulGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static void reluGPU(cl_mem d_x, int length) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel = kernels.get("relu_kernel");
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_x));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{length}));
            
            long[] globalWorkSize = new long[]{(length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};
            
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);
                
        } catch (Exception e) {
            System.err.println("[GPU] reluGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static void siluGPU(cl_mem d_x, int length) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel = kernels.get("silu_kernel");
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_x));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{length}));
            
            long[] globalWorkSize = new long[]{(length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};
            
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);
                
        } catch (Exception e) {
            System.err.println("[GPU] siluGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static void softmaxGPU(cl_mem d_logits, int length) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel = kernels.get("softmax_kernel");
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_logits));
            clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{length}));
            
            long[] globalWorkSize = new long[]{(length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};
            
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);
                
        } catch (Exception e) {
            System.err.println("[GPU] softmaxGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static void layerNormGPU(cl_mem d_input, cl_mem d_output, cl_mem d_gamma, 
                                    cl_mem d_beta, int batchSize, int hiddenSize) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel = kernels.get("layer_norm_kernel");
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_gamma));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_beta));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            
            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};
            
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);
                
        } catch (Exception e) {
            System.err.println("[GPU] layerNormGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static void addBiasGPU(cl_mem d_input, cl_mem d_bias, int batchSize, int hiddenSize) {
        if (!gpuAvailable) return;
        cl_command_queue queue = commandQueues[currentDevice];
        
        try {
            cl_kernel kernel = kernels.get("add_bias_kernel");
            if (kernel == null) return;
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_bias));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));
            
            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};
            
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);
                
        } catch (Exception e) {
            System.err.println("[GPU] addBiasGPU(cl_mem) failed: " + e.getMessage());
        }
    }

    public static float[] matmulBiasSilu(float[] A, float[] B, float[] bias, int M, int K, int N) {
        if (!gpuAvailable) {
            float[] result = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
            result = com.qingxiaotuan.simpcore.math.MatrixOperations.addBias(result, bias, M, N);
            return com.qingxiaotuan.simpcore.math.MatrixOperations.silu(result);
        }
        return matmulBiasSiluGPU(A, B, bias, M, K, N);
    }

    private static float[] matmulBiasSiluGPU(float[] A, float[] B, float[] bias, int M, int K, int N) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[M * N];
        
        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);
            cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * bias.length, Pointer.to(bias), null);
                
            cl_kernel kernel = kernels.get("matmul_bias_silu_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_A);
                clReleaseMemObject(d_B);
                clReleaseMemObject(d_C);
                clReleaseMemObject(d_bias);
                return matmulBiasSilu(A, B, bias, M, K, N);
            }
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_bias));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{N}));
            
            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};
            
            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);
                
            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);
                
            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);
            clReleaseMemObject(d_bias);
            
        } catch (Exception e) {
            System.err.println("[GPU] matmulBiasSilu failed: " + e.getMessage());
            return matmulBiasSilu(A, B, bias, M, K, N);
        }
        
        return result;
    }

    public static float[] matmulBiasGelu(float[] A, float[] B, float[] bias, int M, int K, int N) {
        if (!gpuAvailable) {
            float[] result = com.qingxiaotuan.simpcore.math.MatrixOperations.matmul(A, B, M, K, N);
            result = com.qingxiaotuan.simpcore.math.MatrixOperations.addBias(result, bias, M, N);
            return com.qingxiaotuan.simpcore.math.MatrixOperations.gelu(result);
        }
        return matmulBiasGeluGPU(A, B, bias, M, K, N);
    }

    private static float[] matmulBiasGeluGPU(float[] A, float[] B, float[] bias, int M, int K, int N) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[M * N];
        
        try {
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * A.length, Pointer.to(A), null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * B.length, Pointer.to(B), null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);
            cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                Sizeof.cl_float * bias.length, Pointer.to(bias), null);
                
            cl_kernel kernel = kernels.get("matmul_bias_gelu_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_A);
                clReleaseMemObject(d_B);
                clReleaseMemObject(d_C);
                clReleaseMemObject(d_bias);
                return matmulBiasGelu(A, B, bias, M, K, N);
            }
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_bias));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{K}));
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{N}));
            
            long[] globalWorkSize = new long[]{
                ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
                ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
            };
            long[] localWorkSize = new long[]{BLOCK_SIZE, BLOCK_SIZE};
            
            clEnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);
                
            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);
                
            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);
            clReleaseMemObject(d_bias);
            
        } catch (Exception e) {
            System.err.println("[GPU] matmulBiasGelu failed: " + e.getMessage());
            return matmulBiasGelu(A, B, bias, M, K, N);
        }
        
        return result;
    }

    public static float[] rmsNorm(float[] input, float[] weight, float eps, int batchSize, int hiddenSize) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.rmsNorm(input, weight, batchSize, hiddenSize);
        }
        return rmsNormGPU(input, weight, eps, batchSize, hiddenSize);
    }

    private static float[] rmsNormGPU(float[] input, float[] weight, float eps, int batchSize, int hiddenSize) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[input.length];

        try {
            cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
            cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);
            cl_mem d_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * weight.length, Pointer.to(weight), null);

            cl_kernel kernel = kernels.get("rms_norm_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_input);
                clReleaseMemObject(d_output);
                clReleaseMemObject(d_weight);
                return rmsNorm(input, weight, eps, batchSize, hiddenSize);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_weight));
            clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[]{eps}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_input);
            clReleaseMemObject(d_output);
            clReleaseMemObject(d_weight);

        } catch (Exception e) {
            System.err.println("[GPU] rmsNorm failed: " + e.getMessage());
            return rmsNorm(input, weight, eps, batchSize, hiddenSize);
        }

        return result;
    }

    public static void adamUpdate(float[] params, float[] grads, float[] m, float[] v,
                                  float lr, float beta1, float beta2, float eps, int length) {
        if (!gpuAvailable) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.adamUpdate(params, grads, m, v, lr, beta1, beta2, eps, length);
            return;
        }
        adamUpdateGPU(params, grads, m, v, lr, beta1, beta2, eps, length);
    }

    private static void adamUpdateGPU(float[] params, float[] grads, float[] m, float[] v,
                                      float lr, float beta1, float beta2, float eps, int length) {
        cl_command_queue queue = commandQueues[currentDevice];

        try {
            cl_mem d_params = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * params.length, Pointer.to(params), null);
            cl_mem d_grads = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * grads.length, Pointer.to(grads), null);
            cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * m.length, Pointer.to(m), null);
            cl_mem d_v = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * v.length, Pointer.to(v), null);

            cl_kernel kernel = kernels.get("adam_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_params);
                clReleaseMemObject(d_grads);
                clReleaseMemObject(d_m);
                clReleaseMemObject(d_v);
                com.qingxiaotuan.simpcore.math.MatrixOperations.adamUpdate(params, grads, m, v, lr, beta1, beta2, eps, length);
                return;
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_params));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_grads));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_m));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_v));
            clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{lr}));
            clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{beta1}));
            clSetKernelArg(kernel, 6, Sizeof.cl_float, Pointer.to(new float[]{beta2}));
            clSetKernelArg(kernel, 7, Sizeof.cl_float, Pointer.to(new float[]{eps}));
            clSetKernelArg(kernel, 8, Sizeof.cl_int, Pointer.to(new int[]{length}));

            long[] globalWorkSize = new long[]{(length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_params, CL_TRUE, 0,
                Sizeof.cl_float * params.length, Pointer.to(params), 0, null, null);
            clEnqueueReadBuffer(queue, d_m, CL_TRUE, 0,
                Sizeof.cl_float * m.length, Pointer.to(m), 0, null, null);
            clEnqueueReadBuffer(queue, d_v, CL_TRUE, 0,
                Sizeof.cl_float * v.length, Pointer.to(v), 0, null, null);

            clReleaseMemObject(d_params);
            clReleaseMemObject(d_grads);
            clReleaseMemObject(d_m);
            clReleaseMemObject(d_v);

        } catch (Exception e) {
            System.err.println("[GPU] adamUpdate failed: " + e.getMessage());
            com.qingxiaotuan.simpcore.math.MatrixOperations.adamUpdate(params, grads, m, v, lr, beta1, beta2, eps, length);
        }
    }

    public static float[] dropout(float[] input, float dropoutRate, boolean training, int batchSize, int hiddenSize) {
        if (!gpuAvailable || !training) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.dropout(input, dropoutRate, training, batchSize, hiddenSize);
        }
        return dropoutGPU(input, dropoutRate);
    }

    private static float[] dropoutGPU(float[] input, float dropoutRate) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[input.length];

        try {
            cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
            cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("dropout_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_input);
                clReleaseMemObject(d_output);
                return com.qingxiaotuan.simpcore.math.MatrixOperations.dropout(input, dropoutRate, true, 1, input.length);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[]{dropoutRate}));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{input.length}));

            long[] globalWorkSize = new long[]{(input.length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_input);
            clReleaseMemObject(d_output);

        } catch (Exception e) {
            System.err.println("[GPU] dropout failed: " + e.getMessage());
            return com.qingxiaotuan.simpcore.math.MatrixOperations.dropout(input, dropoutRate, true, 1, input.length);
        }

        return result;
    }

    public static float[] scaledDotProduct(float[] Q, float[] K, float[] V, int batchSize, int nHeads, int headDim) {
        if (!gpuAvailable) {
            return com.qingxiaotuan.simpcore.math.MatrixOperations.scaledDotProduct(Q, K, V, batchSize, nHeads, headDim);
        }
        return scaledDotProductGPU(Q, K, V, batchSize, nHeads, headDim);
    }

    private static float[] scaledDotProductGPU(float[] Q, float[] K, float[] V, int batchSize, int nHeads, int headDim) {
        cl_command_queue queue = commandQueues[currentDevice];
        int seqLen = Q.length / (batchSize * nHeads * headDim);
        float[] result = new float[Q.length];

        try {
            cl_mem d_Q = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * Q.length, Pointer.to(Q), null);
            cl_mem d_K = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * K.length, Pointer.to(K), null);
            cl_mem d_V = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * V.length, Pointer.to(V), null);
            cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);

            cl_kernel kernel = kernels.get("scaled_dot_product_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_Q);
                clReleaseMemObject(d_K);
                clReleaseMemObject(d_V);
                clReleaseMemObject(d_output);
                return scaledDotProduct(Q, K, V, batchSize, nHeads, headDim);
            }

            float scale = (float) (1.0 / Math.sqrt(headDim));
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_Q));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_K));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_V));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{scale}));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{nHeads}));
            clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{headDim}));
            clSetKernelArg(kernel, 8, Sizeof.cl_int, Pointer.to(new int[]{seqLen}));

            long[] globalWorkSize = new long[]{batchSize * nHeads * seqLen * headDim};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_Q);
            clReleaseMemObject(d_K);
            clReleaseMemObject(d_V);
            clReleaseMemObject(d_output);

        } catch (Exception e) {
            System.err.println("[GPU] scaledDotProduct failed: " + e.getMessage());
            return scaledDotProduct(Q, K, V, batchSize, nHeads, headDim);
        }

        return result;
    }

    public static float[] fusedAddNorm(float[] input, float[] residual, float[] weight, float[] bias,
                                       int batchSize, int hiddenSize) {
        if (!gpuAvailable) {
            float[] added = com.qingxiaotuan.simpcore.math.MatrixOperations.add(input, residual);
            return com.qingxiaotuan.simpcore.math.MatrixOperations.layerNorm(added, weight, bias, batchSize, hiddenSize);
        }
        return fusedAddNormGPU(input, residual, weight, bias, batchSize, hiddenSize);
    }

    private static float[] fusedAddNormGPU(float[] input, float[] residual, float[] weight, float[] bias,
                                           int batchSize, int hiddenSize) {
        cl_command_queue queue = commandQueues[currentDevice];
        float[] result = new float[input.length];

        try {
            cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
            cl_mem d_residual = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * residual.length, Pointer.to(residual), null);
            cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * result.length, null, null);
            cl_mem d_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * weight.length, Pointer.to(weight), null);
            cl_mem d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * bias.length, Pointer.to(bias), null);

            cl_kernel kernel = kernels.get("fused_add_norm_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_input);
                clReleaseMemObject(d_residual);
                clReleaseMemObject(d_output);
                clReleaseMemObject(d_weight);
                clReleaseMemObject(d_bias);
                return fusedAddNorm(input, residual, weight, bias, batchSize, hiddenSize);
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_input));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(d_residual));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(d_output));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(d_weight));
            clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(d_bias));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
            clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{hiddenSize}));

            long[] globalWorkSize = new long[]{batchSize * hiddenSize};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                Sizeof.cl_float * result.length, Pointer.to(result), 0, null, null);

            clReleaseMemObject(d_input);
            clReleaseMemObject(d_residual);
            clReleaseMemObject(d_output);
            clReleaseMemObject(d_weight);
            clReleaseMemObject(d_bias);

        } catch (Exception e) {
            System.err.println("[GPU] fusedAddNorm failed: " + e.getMessage());
            return fusedAddNorm(input, residual, weight, bias, batchSize, hiddenSize);
        }

        return result;
    }

    public static void clipGradients(float[] grads, float maxNorm, int length) {
        if (!gpuAvailable) {
            com.qingxiaotuan.simpcore.math.MatrixOperations.clipGradients(grads, maxNorm, length);
            return;
        }
        clipGradientsGPU(grads, maxNorm, length);
    }

    private static void clipGradientsGPU(float[] grads, float maxNorm, int length) {
        cl_command_queue queue = commandQueues[currentDevice];

        try {
            cl_mem d_grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * grads.length, Pointer.to(grads), null);

            cl_kernel kernel = kernels.get("clip_grad_kernel");
            if (kernel == null) {
                clReleaseMemObject(d_grads);
                com.qingxiaotuan.simpcore.math.MatrixOperations.clipGradients(grads, maxNorm, length);
                return;
            }

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(d_grads));
            clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{maxNorm}));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{length}));

            long[] globalWorkSize = new long[]{(length + 255) / 256 * 256};
            long[] localWorkSize = new long[]{256};

            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, localWorkSize,
                0, null, null);

            clEnqueueReadBuffer(queue, d_grads, CL_TRUE, 0,
                Sizeof.cl_float * grads.length, Pointer.to(grads), 0, null, null);

            clReleaseMemObject(d_grads);

        } catch (Exception e) {
            System.err.println("[GPU] clipGradients failed: " + e.getMessage());
            com.qingxiaotuan.simpcore.math.MatrixOperations.clipGradients(grads, maxNorm, length);
        }
    }
}