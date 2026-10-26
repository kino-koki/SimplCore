package com.qingxiaotuan.simpcore.gpu;

import org.jocl.*;

import static org.jocl.CL.*;

/**
 * SimplCore OpenCL GPU加速引擎
 * 完全基于自研架构，通过JOCL实现GPU加速
 * 支持AMD Radeon Graphics 680M等GPU
 */
public class OpenCLAccelerator {
    
    // 单例实例
    private static OpenCLAccelerator instance;
    
    private cl_platform_id[] platforms;
    private cl_device_id[] devices;
    private cl_context context;
    private cl_command_queue queue;
    
    private cl_program matrixMultProgram;
    private cl_kernel matrixMultKernel;
    
    private cl_program reluProgram;
    private cl_kernel reluKernel;
    
    private cl_program addProgram;
    private cl_kernel addKernel;
    
    private cl_program matvecProgram;
    private cl_kernel matvecKernel;
    
    private boolean initialized = false;
    private boolean useGPU = false;
    
    /**
     * 初始化OpenCL加速引擎
     */
    public void initialize() {
        try {
            // 初始化JOCL
            CL.setExceptionsEnabled(true);
            
            // 获取平台
            int numPlatforms[] = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);
            if (numPlatforms[0] == 0) {
                System.out.println("[OpenCL] 未找到OpenCL平台，使用CPU模式");
                return;
            }
            
            platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(numPlatforms[0], platforms, null);
            
            // 获取GPU设备
            int numDevices[] = new int[1];
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, null, numDevices);
            
            if (numDevices[0] == 0) {
                System.out.println("[OpenCL] 未找到GPU设备，尝试CPU设备");
                clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, null, numDevices);
                if (numDevices[0] == 0) {
                    System.out.println("[OpenCL] 未找到任何OpenCL设备");
                    return;
                }
            }
            
            devices = new cl_device_id[numDevices[0]];
            clGetDeviceIDs(platforms[0], 
                numDevices[0] > 0 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 
                numDevices[0], devices, null);
            
            // 创建上下文
            context = clCreateContext(null, 1, devices, null, null, null);
            
            // 创建命令队列
            queue = clCreateCommandQueueWithProperties(context, devices[0], null, null);
            
            // 获取设备信息
            byte deviceName[] = new byte[1024];
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, deviceName.length, 
                Pointer.to(deviceName), null);
            String name = new String(deviceName).trim();
            
            long maxMem[] = new long[1];
            clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, 
                Sizeof.cl_ulong, Pointer.to(maxMem), null);
            
            // AMD GPU设备名称可能是gfx1035等格式，只要是GPU设备类型就启用GPU加速
            useGPU = true;
            
            System.out.printf("[OpenCL] 设备: %s\n", name);
            System.out.printf("[OpenCL] 内存: %.2f GB\n", maxMem[0] / (1024.0 * 1024.0 * 1024.0));
            System.out.printf("[OpenCL] 模式: %s\n", useGPU ? "GPU加速" : "CPU计算");
            
            // 编译OpenCL内核
            compileKernels();
            
            initialized = true;
            System.out.println("[OpenCL] 初始化完成");
            
        } catch (Exception e) {
            System.out.println("[OpenCL] 初始化失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 编译OpenCL内核
     */
    private void compileKernels() {
        // 矩阵乘法内核
        String matrixMultSource = 
            "__kernel void matrixMult(__global const float *A, __global const float *B, " +
            "__global float *C, int M, int N, int K) {\n" +
            "    int row = get_global_id(0);\n" +
            "    int col = get_global_id(1);\n" +
            "    float sum = 0.0f;\n" +
            "    for (int i = 0; i < K; i++) {\n" +
            "        sum += A[row * K + i] * B[i * N + col];\n" +
            "    }\n" +
            "    C[row * N + col] = sum;\n" +
            "}\n";
        
        // ReLU内核
        String reluSource = 
            "__kernel void relu(__global float *data, int size) {\n" +
            "    int idx = get_global_id(0);\n" +
            "    if (idx < size) {\n" +
            "        data[idx] = max(data[idx], 0.0f);\n" +
            "    }\n" +
            "}\n";
        
        // 向量加法内核
        String addSource = 
            "__kernel void add(__global const float *a, __global const float *b, " +
            "__global float *c, int size) {\n" +
            "    int idx = get_global_id(0);\n" +
            "    if (idx < size) {\n" +
            "        c[idx] = a[idx] + b[idx];\n" +
            "    }\n" +
            "}\n";
        
        // 矩阵-向量乘法内核
        String matvecSource = 
            "__kernel void matvec(__global const float *A, __global const float *x, " +
            "__global float *y, int M, int N) {\n" +
            "    int row = get_global_id(0);\n" +
            "    float sum = 0.0f;\n" +
            "    for (int i = 0; i < N; i++) {\n" +
            "        sum += A[row * N + i] * x[i];\n" +
            "    }\n" +
            "    y[row] = sum;\n" +
            "}\n";
        
        try {
            // 编译矩阵乘法
            matrixMultProgram = clCreateProgramWithSource(context, 1, 
                new String[]{matrixMultSource}, null, null);
            clBuildProgram(matrixMultProgram, 1, devices, "", null, null);
            matrixMultKernel = clCreateKernel(matrixMultProgram, "matrixMult", null);
            
            // 编译ReLU
            reluProgram = clCreateProgramWithSource(context, 1, 
                new String[]{reluSource}, null, null);
            clBuildProgram(reluProgram, 1, devices, "", null, null);
            reluKernel = clCreateKernel(reluProgram, "relu", null);
            
            // 编译加法
            addProgram = clCreateProgramWithSource(context, 1, 
                new String[]{addSource}, null, null);
            clBuildProgram(addProgram, 1, devices, "", null, null);
            addKernel = clCreateKernel(addProgram, "add", null);
            
            // 编译矩阵-向量乘法
            matvecProgram = clCreateProgramWithSource(context, 1, 
                new String[]{matvecSource}, null, null);
            clBuildProgram(matvecProgram, 1, devices, "", null, null);
            matvecKernel = clCreateKernel(matvecProgram, "matvec", null);
            
            System.out.println("[OpenCL] 内核编译完成");
            
        } catch (Exception e) {
            System.out.println("[OpenCL] 内核编译失败: " + e.getMessage());
            // 打印编译日志
            if (matrixMultProgram != null) {
                byte log[] = new byte[4096];
                clGetProgramBuildInfo(matrixMultProgram, devices[0], 
                    CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
                System.out.println("编译日志: " + new String(log).trim());
            }
        }
    }
    
    /**
     * 矩阵乘法 C = A * B
     */
    public float[] matrixMultiply(float[] A, float[] B, int M, int N, int K) {
        if (!initialized || !useGPU) {
            return cpuMatrixMultiply(A, B, M, N, K);
        }
        
        try {
            float[] C = new float[M * N];
            
            // 创建缓冲区
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * A.length, null, null);
            cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * B.length, null, null);
            cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * C.length, null, null);
            
            // 复制数据到GPU
            clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, 
                Sizeof.cl_float * A.length, Pointer.to(A), 0, null, null);
            clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, 
                Sizeof.cl_float * B.length, Pointer.to(B), 0, null, null);
            
            // 设置内核参数
            clSetKernelArg(matrixMultKernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(matrixMultKernel, 1, Sizeof.cl_mem, Pointer.to(d_B));
            clSetKernelArg(matrixMultKernel, 2, Sizeof.cl_mem, Pointer.to(d_C));
            clSetKernelArg(matrixMultKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(matrixMultKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{N}));
            clSetKernelArg(matrixMultKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{K}));
            
            // 执行内核
            long globalWorkSize[] = new long[]{M, N};
            clEnqueueNDRangeKernel(queue, matrixMultKernel, 2, null, 
                globalWorkSize, null, 0, null, null);
            
            // 复制结果回CPU
            clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, 
                Sizeof.cl_float * C.length, Pointer.to(C), 0, null, null);
            
            // 释放资源
            clReleaseMemObject(d_A);
            clReleaseMemObject(d_B);
            clReleaseMemObject(d_C);
            
            return C;
            
        } catch (Exception e) {
            System.out.println("[OpenCL] 矩阵乘法失败: " + e.getMessage());
            return cpuMatrixMultiply(A, B, M, N, K);
        }
    }
    
    /**
     * CPU矩阵乘法（备用）
     */
    private float[] cpuMatrixMultiply(float[] A, float[] B, int M, int N, int K) {
        float[] C = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return C;
    }
    
    /**
     * ReLU激活函数
     */
    public void relu(float[] data) {
        if (!initialized || !useGPU) {
            cpuRelu(data);
            return;
        }
        
        try {
            // 创建缓冲区
            cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                Sizeof.cl_float * data.length, null, null);
            
            // 复制数据到GPU
            clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, 
                Sizeof.cl_float * data.length, Pointer.to(data), 0, null, null);
            
            // 设置内核参数
            clSetKernelArg(reluKernel, 0, Sizeof.cl_mem, Pointer.to(d_data));
            clSetKernelArg(reluKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{data.length}));
            
            // 执行内核
            long globalWorkSize[] = new long[]{data.length};
            clEnqueueNDRangeKernel(queue, reluKernel, 1, null, 
                globalWorkSize, null, 0, null, null);
            
            // 复制结果回CPU
            clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, 
                Sizeof.cl_float * data.length, Pointer.to(data), 0, null, null);
            
            // 释放资源
            clReleaseMemObject(d_data);
            
        } catch (Exception e) {
            System.out.println("[OpenCL] ReLU失败: " + e.getMessage());
            cpuRelu(data);
        }
    }
    
    /**
     * CPU ReLU（备用）
     */
    private void cpuRelu(float[] data) {
        for (int i = 0; i < data.length; i++) {
            if (data[i] < 0) data[i] = 0;
        }
    }
    
    /**
     * 向量加法
     */
    public float[] add(float[] a, float[] b) {
        if (!initialized || !useGPU) {
            return cpuAdd(a, b);
        }
        
        try {
            float[] c = new float[a.length];
            
            cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * a.length, null, null);
            cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * b.length, null, null);
            cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * c.length, null, null);
            
            clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, 
                Sizeof.cl_float * a.length, Pointer.to(a), 0, null, null);
            clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, 
                Sizeof.cl_float * b.length, Pointer.to(b), 0, null, null);
            
            clSetKernelArg(addKernel, 0, Sizeof.cl_mem, Pointer.to(d_a));
            clSetKernelArg(addKernel, 1, Sizeof.cl_mem, Pointer.to(d_b));
            clSetKernelArg(addKernel, 2, Sizeof.cl_mem, Pointer.to(d_c));
            clSetKernelArg(addKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{a.length}));
            
            long globalWorkSize[] = new long[]{a.length};
            clEnqueueNDRangeKernel(queue, addKernel, 1, null, 
                globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, 
                Sizeof.cl_float * c.length, Pointer.to(c), 0, null, null);
            
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseMemObject(d_c);
            
            return c;
            
        } catch (Exception e) {
            System.out.println("[OpenCL] 向量加法失败: " + e.getMessage());
            return cpuAdd(a, b);
        }
    }
    
    /**
     * CPU向量加法（备用）
     */
    private float[] cpuAdd(float[] a, float[] b) {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }
    
    /**
     * 矩阵-向量乘法
     */
    public float[] matvec(float[] A, float[] x, int M, int N) {
        if (!initialized || !useGPU) {
            return cpuMatvec(A, x, M, N);
        }
        
        try {
            float[] y = new float[M];
            
            cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * A.length, null, null);
            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                Sizeof.cl_float * x.length, null, null);
            cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * y.length, null, null);
            
            clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, 
                Sizeof.cl_float * A.length, Pointer.to(A), 0, null, null);
            clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, 
                Sizeof.cl_float * x.length, Pointer.to(x), 0, null, null);
            
            clSetKernelArg(matvecKernel, 0, Sizeof.cl_mem, Pointer.to(d_A));
            clSetKernelArg(matvecKernel, 1, Sizeof.cl_mem, Pointer.to(d_x));
            clSetKernelArg(matvecKernel, 2, Sizeof.cl_mem, Pointer.to(d_y));
            clSetKernelArg(matvecKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{M}));
            clSetKernelArg(matvecKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{N}));
            
            long globalWorkSize[] = new long[]{M};
            clEnqueueNDRangeKernel(queue, matvecKernel, 1, null, 
                globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, 
                Sizeof.cl_float * y.length, Pointer.to(y), 0, null, null);
            
            clReleaseMemObject(d_A);
            clReleaseMemObject(d_x);
            clReleaseMemObject(d_y);
            
            return y;
            
        } catch (Exception e) {
            System.out.println("[OpenCL] 矩阵-向量乘法失败: " + e.getMessage());
            return cpuMatvec(A, x, M, N);
        }
    }
    
    /**
     * CPU矩阵-向量乘法（备用）
     */
    private float[] cpuMatvec(float[] A, float[] x, int M, int N) {
        float[] y = new float[M];
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
        return y;
    }
    
    /**
     * 是否使用GPU加速
     */
    public boolean isUsingGPU() {
        return useGPU;
    }
    
    /**
     * 是否已初始化
     */
    public boolean isInitialized() {
        return initialized;
    }
    
    /**
     * 释放资源
     */
    public void release() {
        if (initialized) {
            try {
                clReleaseKernel(matrixMultKernel);
                clReleaseProgram(matrixMultProgram);
                clReleaseKernel(reluKernel);
                clReleaseProgram(reluProgram);
                clReleaseKernel(addKernel);
                clReleaseProgram(addProgram);
                clReleaseKernel(matvecKernel);
                clReleaseProgram(matvecProgram);
                clReleaseCommandQueue(queue);
                clReleaseContext(context);
                System.out.println("[OpenCL] 资源释放完成");
            } catch (Exception e) {
                // 忽略释放错误
            }
            initialized = false;
        }
    }
    
    /**
     * 获取单例实例
     */
    public static OpenCLAccelerator getInstance() {
        if (instance == null) {
            instance = new OpenCLAccelerator();
            instance.initialize();
        }
        return instance;
    }
}