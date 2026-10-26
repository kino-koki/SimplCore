package com.qingxiaotuan.simpcore.gpu;

import java.lang.reflect.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class JCudaOptimizationLayer {

    public static class JCudaConfig {
        public boolean useJCuda = true;
        public boolean useCuDNN = true;
        public boolean useCuBLAS = true;
        public int maxBatchSize = 64;
        public int workspaceSizeMB = 64;
        public boolean enableTensorCore = true;
        public boolean enableFP16 = false;
        public boolean enableFP8 = false;
    }

    private final JCudaConfig config;
    private volatile boolean initialized = false;
    private volatile boolean jcudaAvailable = false;
    
    private Object cudaContext;
    private Object cublasHandle;
    private Object cudnnHandle;
    
    private final ConcurrentHashMap<Long, Object> deviceMemory = new ConcurrentHashMap<>();
    private final AtomicLong memoryIdCounter = new AtomicLong(0);
    
    private final ExecutorService memoryPool = Executors.newFixedThreadPool(2);
    
    public JCudaOptimizationLayer(JCudaConfig config) {
        this.config = config;
    }

    public boolean initialize() {
        if (initialized) return jcudaAvailable;
        
        try {
            Class.forName("jcuda.driver.CUdeviceptr");
            Class.forName("jcuda.driver.CudaDriver");
            jcudaAvailable = true;
        } catch (ClassNotFoundException e) {
            System.out.println("[JCuda] JCuda not available, falling back to OpenCL");
            jcudaAvailable = false;
            initialized = true;
            return false;
        }
        
        try {
            initCudaDriver();
            
            if (config.useCuBLAS) {
                initCuBLAS();
            }
            
            if (config.useCuDNN) {
                initCuDNN();
            }
            
            initialized = true;
            System.out.println("[JCuda] JCuda optimization layer initialized successfully");
            return true;
        } catch (Exception e) {
            System.err.println("[JCuda] Failed to initialize JCuda: " + e.getMessage());
            jcudaAvailable = false;
            initialized = true;
            return false;
        }
    }

    private void initCudaDriver() throws Exception {
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method initializeMethod = cudaDriverClass.getMethod("initialize");
        initializeMethod.invoke(null);
        
        Method deviceGetMethod = cudaDriverClass.getMethod("cuDeviceGet", int[].class, int.class);
        int[] device = new int[1];
        deviceGetMethod.invoke(null, device, 0);
        
        Method contextCreateMethod = cudaDriverClass.getMethod("cuCtxCreate", int[].class, int.class, int.class);
        int[] context = new int[1];
        contextCreateMethod.invoke(null, context, 0, device[0]);
        
        cudaContext = context[0];
        
        System.out.println("[JCuda] CUDA context created on device " + device[0]);
    }

    private void initCuBLAS() throws Exception {
        Class<?> cublasClass = Class.forName("jcuda.cublas.Cublas");
        Method createMethod = cublasClass.getMethod("cublasCreate_v2", long[].class);
        long[] handle = new long[1];
        createMethod.invoke(null, handle);
        cublasHandle = handle[0];
        
        if (config.enableTensorCore) {
            Class<?> cublasPointerModeClass = Class.forName("jcuda.cublas.CUBLAS_POINTER_MODE");
            Object hostMode = Enum.valueOf(cublasPointerModeClass, "CUBLAS_POINTER_MODE_HOST");
            
            Method setPointerModeMethod = cublasClass.getMethod("cublasSetPointerMode_v2", long.class, Object.class);
            setPointerModeMethod.invoke(null, cublasHandle, hostMode);
            
            System.out.println("[JCuda] CuBLAS initialized with Tensor Core support");
        }
    }

    private void initCuDNN() throws Exception {
        Class<?> cudnnClass = Class.forName("jcuda.cudnn.Cudnn");
        Method createMethod = cudnnClass.getMethod("cudnnCreate", long[].class);
        long[] handle = new long[1];
        createMethod.invoke(null, handle);
        cudnnHandle = handle[0];
        
        System.out.println("[JCuda] CuDNN initialized");
    }

    public float[] matmul(float[] A, float[] B, int M, int K, int N) {
        if (!jcudaAvailable || !config.useCuBLAS) {
            return GPUMatrixOperations.matmul(A, B, M, K, N);
        }
        
        try {
            long ptrA = allocateDeviceMemory(A.length * 4);
            long ptrB = allocateDeviceMemory(B.length * 4);
            float[] C = new float[M * N];
            long ptrC = allocateDeviceMemory(C.length * 4);
            
            copyToDevice(ptrA, A);
            copyToDevice(ptrB, B);
            
            callCublasMatmul(ptrA, ptrB, ptrC, M, K, N);
            
            copyFromDevice(C, ptrC);
            
            freeDeviceMemory(ptrA);
            freeDeviceMemory(ptrB);
            freeDeviceMemory(ptrC);
            
            return C;
        } catch (Exception e) {
            System.err.println("[JCuda] CuBLAS matmul failed, falling back: " + e.getMessage());
            return GPUMatrixOperations.matmul(A, B, M, K, N);
        }
    }

    private void callCublasMatmul(long ptrA, long ptrB, long ptrC, int M, int K, int N) throws Exception {
        Class<?> cublasClass = Class.forName("jcuda.cublas.Cublas");
        Class<?> cublasOperationClass = Class.forName("jcuda.cublas.CUBLAS_OP");
        
        Object noTrans = Enum.valueOf(cublasOperationClass, "CUBLAS_OP_N");
        
        float alpha = 1.0f;
        float beta = 0.0f;
        
        Method gemmMethod = cublasClass.getMethod(
            "cublasSgemm_v2",
            long.class,
            Object.class,
            Object.class,
            int.class,
            int.class,
            int.class,
            float[].class,
            long.class,
            int.class,
            long.class,
            int.class,
            float[].class,
            long.class,
            int.class
        );
        
        gemmMethod.invoke(null,
            cublasHandle,
            noTrans,
            noTrans,
            N,
            M,
            K,
            new float[]{alpha},
            ptrB,
            N,
            ptrA,
            K,
            new float[]{beta},
            ptrC,
            N
        );
    }

    public float[] layerNorm(float[] input, float[] gamma, float[] beta, int batchSize, int hiddenSize) {
        if (!jcudaAvailable || !config.useCuDNN) {
            return GPUMatrixOperations.layerNorm(input, gamma, beta, batchSize, hiddenSize);
        }
        
        try {
            float[] output = new float[input.length];
            
            callCudnnLayerNorm(input, output, gamma, beta, batchSize, hiddenSize);
            
            return output;
        } catch (Exception e) {
            System.err.println("[JCuda] CuDNN layerNorm failed, falling back: " + e.getMessage());
            return GPUMatrixOperations.layerNorm(input, gamma, beta, batchSize, hiddenSize);
        }
    }

    private void callCudnnLayerNorm(float[] input, float[] output, float[] gamma, float[] beta,
                                    int batchSize, int hiddenSize) throws Exception {
        Class<?> cudnnClass = Class.forName("jcuda.cudnn.Cudnn");
        Class<?> cudnnTensorDescriptorClass = Class.forName("jcuda.cudnn.cudnnTensorDescriptor");
        Class<?> cudnnNormModeClass = Class.forName("jcuda.cudnn.cudnnNormMode");
        
        Constructor<?> descConstructor = cudnnTensorDescriptorClass.getConstructor();
        Object inputDesc = descConstructor.newInstance();
        Object outputDesc = descConstructor.newInstance();
        
        int[] inputDims = {batchSize, 1, 1, hiddenSize};
        int[] strides = {hiddenSize, hiddenSize, hiddenSize, 1};
        
        Method setTensor4dMethod = cudnnClass.getMethod("cudnnSetTensor4dDescriptor",
            cudnnTensorDescriptorClass,
            int.class,
            int.class,
            int[].class,
            int[].class
        );
        
        setTensor4dMethod.invoke(null, inputDesc, 0, 3, inputDims, strides);
        setTensor4dMethod.invoke(null, outputDesc, 0, 3, inputDims, strides);
        
        int[] normDims = {hiddenSize};
        
        Object normMode = Enum.valueOf(cudnnNormModeClass, "CUDNN_NORM_HIDDEN");
        
        float epsilon = 1e-5f;
        
        Method layerNormForwardMethod = cudnnClass.getMethod("cudnnLayerNormalizationForward",
            long.class,
            Object.class,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class,
            cudnnTensorDescriptorClass,
            int[].class,
            float[].class,
            float[].class,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class,
            float[].class,
            float[].class
        );
        
        layerNormForwardMethod.invoke(null,
            cudnnHandle,
            normMode,
            new float[]{1.0f},
            new float[]{0.0f},
            inputDesc,
            input,
            outputDesc,
            output,
            normDims,
            gamma,
            beta,
            epsilon,
            null,
            0,
            null,
            0
        );
        
        Method destroyDescMethod = cudnnClass.getMethod("cudnnDestroyTensorDescriptor", cudnnTensorDescriptorClass);
        destroyDescMethod.invoke(null, inputDesc);
        destroyDescMethod.invoke(null, outputDesc);
    }

    public float[] gelu(float[] input) {
        if (!jcudaAvailable || !config.useCuDNN) {
            return GPUMatrixOperations.gelu(input);
        }
        
        try {
            float[] output = new float[input.length];
            
            callCudnnActivation(input, output, "GELU");
            
            return output;
        } catch (Exception e) {
            System.err.println("[JCuda] CuDNN gelu failed, falling back: " + e.getMessage());
            return GPUMatrixOperations.gelu(input);
        }
    }

    public float[] silu(float[] input) {
        if (!jcudaAvailable || !config.useCuDNN) {
            return GPUMatrixOperations.silu(input);
        }
        
        try {
            float[] output = new float[input.length];
            
            callCudnnActivation(input, output, "SIGMOID");
            
            for (int i = 0; i < input.length; i++) {
                output[i] = input[i] * output[i];
            }
            
            return output;
        } catch (Exception e) {
            System.err.println("[JCuda] CuDNN silu failed, falling back: " + e.getMessage());
            return GPUMatrixOperations.silu(input);
        }
    }

    private void callCudnnActivation(float[] input, float[] output, String activation) throws Exception {
        Class<?> cudnnClass = Class.forName("jcuda.cudnn.Cudnn");
        Class<?> cudnnTensorDescriptorClass = Class.forName("jcuda.cudnn.cudnnTensorDescriptor");
        Class<?> cudnnActivationModeClass = Class.forName("jcuda.cudnn.cudnnActivationMode");
        
        Constructor<?> descConstructor = cudnnTensorDescriptorClass.getConstructor();
        Object inputDesc = descConstructor.newInstance();
        Object outputDesc = descConstructor.newInstance();
        
        int[] dims = {1, 1, 1, input.length};
        int[] strides = {input.length, input.length, input.length, 1};
        
        Method setTensor4dMethod = cudnnClass.getMethod("cudnnSetTensor4dDescriptor",
            cudnnTensorDescriptorClass,
            int.class,
            int.class,
            int[].class,
            int[].class
        );
        
        setTensor4dMethod.invoke(null, inputDesc, 0, 3, dims, strides);
        setTensor4dMethod.invoke(null, outputDesc, 0, 3, dims, strides);
        
        Object activationMode = Enum.valueOf(cudnnActivationModeClass, "CUDNN_ACTIVATION_" + activation);
        
        float reluCeiling = 0.0f;
        
        Method activationForwardMethod = cudnnClass.getMethod("cudnnActivationForward",
            long.class,
            Object.class,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class,
            float[].class,
            float.class
        );
        
        activationForwardMethod.invoke(null,
            cudnnHandle,
            activationMode,
            new float[]{1.0f},
            inputDesc,
            input,
            new float[]{0.0f},
            outputDesc,
            output,
            reluCeiling
        );
        
        Method destroyDescMethod = cudnnClass.getMethod("cudnnDestroyTensorDescriptor", cudnnTensorDescriptorClass);
        destroyDescMethod.invoke(null, inputDesc);
        destroyDescMethod.invoke(null, outputDesc);
    }

    public float[] softmax(float[] input, int batchSize, int seqLen, int hiddenSize) {
        if (!jcudaAvailable || !config.useCuDNN) {
            return GPUMatrixOperations.softmax(input, batchSize, seqLen, hiddenSize);
        }
        
        try {
            float[] output = new float[input.length];
            
            callCudnnSoftmax(input, output, batchSize * seqLen, hiddenSize);
            
            return output;
        } catch (Exception e) {
            System.err.println("[JCuda] CuDNN softmax failed, falling back: " + e.getMessage());
            return GPUMatrixOperations.softmax(input, batchSize, seqLen, hiddenSize);
        }
    }

    private void callCudnnSoftmax(float[] input, float[] output, int n, int d) throws Exception {
        Class<?> cudnnClass = Class.forName("jcuda.cudnn.Cudnn");
        Class<?> cudnnTensorDescriptorClass = Class.forName("jcuda.cudnn.cudnnTensorDescriptor");
        Class<?> cudnnSoftmaxModeClass = Class.forName("jcuda.cudnn.cudnnSoftmaxMode");
        Class<?> cudnnSoftmaxAlgorithmClass = Class.forName("jcuda.cudnn.cudnnSoftmaxAlgorithm");
        
        Constructor<?> descConstructor = cudnnTensorDescriptorClass.getConstructor();
        Object inputDesc = descConstructor.newInstance();
        Object outputDesc = descConstructor.newInstance();
        
        int[] dims = {n, d, 1, 1};
        int[] strides = {d, 1, 1, 1};
        
        Method setTensor4dMethod = cudnnClass.getMethod("cudnnSetTensor4dDescriptor",
            cudnnTensorDescriptorClass,
            int.class,
            int.class,
            int[].class,
            int[].class
        );
        
        setTensor4dMethod.invoke(null, inputDesc, 0, 3, dims, strides);
        setTensor4dMethod.invoke(null, outputDesc, 0, 3, dims, strides);
        
        Object softmaxMode = Enum.valueOf(cudnnSoftmaxModeClass, "CUDNN_SOFTMAX_MODE_INSTANCE");
        Object softmaxAlgo = Enum.valueOf(cudnnSoftmaxAlgorithmClass, "CUDNN_SOFTMAX_ACCURATE");
        
        Method softmaxForwardMethod = cudnnClass.getMethod("cudnnSoftmaxForward",
            long.class,
            Object.class,
            Object.class,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class,
            float[].class,
            cudnnTensorDescriptorClass,
            float[].class
        );
        
        softmaxForwardMethod.invoke(null,
            cudnnHandle,
            softmaxAlgo,
            softmaxMode,
            new float[]{1.0f},
            inputDesc,
            input,
            new float[]{0.0f},
            outputDesc,
            output
        );
        
        Method destroyDescMethod = cudnnClass.getMethod("cudnnDestroyTensorDescriptor", cudnnTensorDescriptorClass);
        destroyDescMethod.invoke(null, inputDesc);
        destroyDescMethod.invoke(null, outputDesc);
    }

    public float[] adamUpdate(float[] weights, float[] gradients, float[] m, float[] v,
                              float lr, float beta1, float beta2, float epsilon, int step) {
        if (!jcudaAvailable) {
            return adamUpdateCPU(weights, gradients, m, v, lr, beta1, beta2, epsilon, step);
        }
        
        try {
            float[] output = new float[weights.length];
            
            callJCudaAdam(weights, gradients, m, v, output, lr, beta1, beta2, epsilon, step);
            
            return output;
        } catch (Exception e) {
            System.err.println("[JCuda] Adam update failed, falling back: " + e.getMessage());
            return adamUpdateCPU(weights, gradients, m, v, lr, beta1, beta2, epsilon, step);
        }
    }

    private void callJCudaAdam(float[] weights, float[] gradients, float[] m, float[] v,
                               float[] output, float lr, float beta1, float beta2, 
                               float epsilon, int step) throws Exception {
        long ptrW = allocateDeviceMemory(weights.length * 4);
        long ptrG = allocateDeviceMemory(gradients.length * 4);
        long ptrM = allocateDeviceMemory(m.length * 4);
        long ptrV = allocateDeviceMemory(v.length * 4);
        long ptrO = allocateDeviceMemory(output.length * 4);
        
        copyToDevice(ptrW, weights);
        copyToDevice(ptrG, gradients);
        copyToDevice(ptrM, m);
        copyToDevice(ptrV, v);
        
        String kernelSource = "__global__ void adam_kernel(float* w, float* g, float* m, float* v, float* o, " +
            "float lr, float b1, float b2, float eps, int step) {\n" +
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    m[idx] = b1 * m[idx] + (1.0f - b1) * g[idx];\n" +
            "    v[idx] = b2 * v[idx] + (1.0f - b2) * g[idx] * g[idx];\n" +
            "    float m_hat = m[idx] / (1.0f - powf(b1, step + 1));\n" +
            "    float v_hat = v[idx] / (1.0f - powf(b2, step + 1));\n" +
            "    o[idx] = w[idx] - lr * m_hat / (sqrtf(v_hat) + eps);\n" +
            "}\n";
        
        compileAndLaunchKernel(kernelSource, "adam_kernel", weights.length, 
            ptrW, ptrG, ptrM, ptrV, ptrO, lr, beta1, beta2, epsilon, step);
        
        copyFromDevice(output, ptrO);
        
        freeDeviceMemory(ptrW);
        freeDeviceMemory(ptrG);
        freeDeviceMemory(ptrM);
        freeDeviceMemory(ptrV);
        freeDeviceMemory(ptrO);
    }

    private float[] adamUpdateCPU(float[] weights, float[] gradients, float[] m, float[] v,
                                  float lr, float beta1, float beta2, float epsilon, int step) {
        float[] output = new float[weights.length];
        float b1t = (float) Math.pow(beta1, step + 1);
        float b2t = (float) Math.pow(beta2, step + 1);
        
        for (int i = 0; i < weights.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];
            
            float mHat = m[i] / (1 - b1t);
            float vHat = v[i] / (1 - b2t);
            
            output[i] = weights[i] - lr * mHat / ((float) Math.sqrt(vHat) + epsilon);
        }
        
        return output;
    }

    private long allocateDeviceMemory(int sizeBytes) throws Exception {
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method mallocMethod = cudaDriverClass.getMethod("cuMemAlloc", long[].class, long.class);
        
        long[] ptr = new long[1];
        mallocMethod.invoke(null, ptr, sizeBytes);
        
        long id = memoryIdCounter.incrementAndGet();
        deviceMemory.put(id, ptr[0]);
        
        return ptr[0];
    }

    private void freeDeviceMemory(long ptr) throws Exception {
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method freeMethod = cudaDriverClass.getMethod("cuMemFree", long.class);
        freeMethod.invoke(null, ptr);
        
        deviceMemory.values().removeIf(v -> (long) v == ptr);
    }

    private void copyToDevice(long dst, float[] src) throws Exception {
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method copyMethod = cudaDriverClass.getMethod("cuMemcpyHtoD", long.class, float[].class, long.class);
        copyMethod.invoke(null, dst, src, src.length * 4L);
    }

    private void copyFromDevice(float[] dst, long src) throws Exception {
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method copyMethod = cudaDriverClass.getMethod("cuMemcpyDtoH", float[].class, long.class, long.class);
        copyMethod.invoke(null, dst, src, dst.length * 4L);
    }

    private void compileAndLaunchKernel(String source, String kernelName, int numElements,
                                        Object... args) throws Exception {
        Class<?> nvrtcClass = null;
        try {
            nvrtcClass = Class.forName("jcuda.nvrtc.JNvrtc");
        } catch (ClassNotFoundException e) {
            throw new UnsupportedOperationException("NVRTC not available");
        }
        
        Class<?> nvrtcProgramClass = Class.forName("jcuda.nvrtc.nvrtcProgram");
        Constructor<?> programConstructor = nvrtcProgramClass.getConstructor();
        Object program = programConstructor.newInstance();
        
        Method createProgramMethod = nvrtcClass.getMethod("nvrtcCreateProgram",
            nvrtcProgramClass, String.class, String.class, int.class, Object[].class, Object[].class);
        createProgramMethod.invoke(null, program, source, null, 0, null, null);
        
        String[] options = {"-arch=compute_70", "--use_fast_math"};
        Method compileProgramMethod = nvrtcClass.getMethod("nvrtcCompileProgram",
            nvrtcProgramClass, int.class, String[].class);
        compileProgramMethod.invoke(null, program, options.length, options);
        
        Method getPTXMethod = nvrtcClass.getMethod("nvrtcGetPTX", nvrtcProgramClass);
        String ptx = (String) getPTXMethod.invoke(null, program);
        
        Method destroyProgramMethod = nvrtcClass.getMethod("nvrtcDestroyProgram", nvrtcProgramClass);
        destroyProgramMethod.invoke(null, program);
        
        Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
        Method moduleLoadMethod = cudaDriverClass.getMethod("cuModuleLoadDataEx", int[].class, String.class, int.class, Object.class);
        int[] module = new int[1];
        moduleLoadMethod.invoke(null, module, ptx, 0, null);
        
        Method moduleGetFunctionMethod = cudaDriverClass.getMethod("cuModuleGetFunction", int[].class, int.class, String.class);
        int[] kernel = new int[1];
        moduleGetFunctionMethod.invoke(null, kernel, module[0], kernelName);
        
        int blockSize = 256;
        int gridSize = (numElements + blockSize - 1) / blockSize;
        
        Object[] kernelParams = args;
        
        Method launchKernelMethod = cudaDriverClass.getMethod("cuLaunchKernel",
            int.class, int.class, int.class, int.class, int.class, int.class, int.class, int.class,
            long.class, Object[].class, Object[].class);
        
        launchKernelMethod.invoke(null,
            kernel[0],
            gridSize, 1, 1,
            blockSize, 1, 1,
            0,
            0,
            0,
            kernelParams,
            null
        );
        
        Method moduleUnloadMethod = cudaDriverClass.getMethod("cuModuleUnload", int.class);
        moduleUnloadMethod.invoke(null, module[0]);
    }

    public void shutdown() {
        if (!initialized) return;
        
        try {
            if (cublasHandle != null) {
                Class<?> cublasClass = Class.forName("jcuda.cublas.Cublas");
                Method destroyMethod = cublasClass.getMethod("cublasDestroy_v2", long.class);
                destroyMethod.invoke(null, cublasHandle);
            }
            
            if (cudnnHandle != null) {
                Class<?> cudnnClass = Class.forName("jcuda.cudnn.Cudnn");
                Method destroyMethod = cudnnClass.getMethod("cudnnDestroy", long.class);
                destroyMethod.invoke(null, cudnnHandle);
            }
            
            if (cudaContext != null) {
                Class<?> cudaDriverClass = Class.forName("jcuda.driver.CudaDriver");
                Method ctxDestroyMethod = cudaDriverClass.getMethod("cuCtxDestroy", int.class);
                ctxDestroyMethod.invoke(null, cudaContext);
            }
        } catch (Exception e) {
            System.err.println("[JCuda] Error during shutdown: " + e.getMessage());
        }
        
        memoryPool.shutdown();
        initialized = false;
        jcudaAvailable = false;
        
        System.out.println("[JCuda] JCuda optimization layer shutdown");
    }

    public boolean isJCudaAvailable() {
        return jcudaAvailable;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public boolean isUsingCuDNN() {
        return jcudaAvailable && config.useCuDNN && cudnnHandle != null;
    }

    public boolean isUsingCuBLAS() {
        return jcudaAvailable && config.useCuBLAS && cublasHandle != null;
    }

    public long getDeviceMemoryAllocated() {
        return deviceMemory.size() * 4L * 1024L * 1024L;
    }

    public JCudaConfig getConfig() {
        return config;
    }
}