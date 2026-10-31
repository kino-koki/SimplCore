package com.qingxiaotuan.simpcore.gpu;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public final class GPUDeviceManager {

    public enum GPUVendor {
        NVIDIA, AMD, INTEL, OTHER
    }

    public enum GPUBackend {
        CUDA, OPENCL, NONE
    }

    public static class DeviceInfo {
        public final int id;
        public final String name;
        public final GPUVendor vendor;
        public final GPUBackend backend;
        public final long memoryTotal;
        public final int computeUnits;
        public final int maxClockFrequency;
        public final boolean unifiedMemory;

        public DeviceInfo(int id, String name, GPUVendor vendor, GPUBackend backend,
                          long memoryTotal, int computeUnits, int maxClockFrequency,
                          boolean unifiedMemory) {
            this.id = id;
            this.name = name;
            this.vendor = vendor;
            this.backend = backend;
            this.memoryTotal = memoryTotal;
            this.computeUnits = computeUnits;
            this.maxClockFrequency = maxClockFrequency;
            this.unifiedMemory = unifiedMemory;
        }

        public double getPerformanceScore() {
            double memoryScore = memoryTotal / (1024.0 * 1024.0 * 1024.0);
            double computeScore = computeUnits * maxClockFrequency / 1000.0;
            double vendorBonus = vendor == GPUVendor.NVIDIA ? 1.5 : 1.0;
            return (memoryScore + computeScore) * vendorBonus;
        }

        @Override
        public String toString() {
            return String.format("Device[%d]: %s | Vendor: %s | Backend: %s | Memory: %.2f GB | CU: %d | Clock: %d MHz",
                    id, name, vendor, backend, memoryTotal / (1024.0 * 1024.0 * 1024.0),
                    computeUnits, maxClockFrequency);
        }
    }

    private static final AtomicBoolean initialized = new AtomicBoolean(false);
    private static final List<DeviceInfo> devices = new ArrayList<>();
    private static volatile DeviceInfo selectedDevice = null;
    private static volatile GPUBackend selectedBackend = GPUBackend.NONE;

    private GPUDeviceManager() {}

    public static void initialize() {
        if (initialized.compareAndSet(false, true)) {
            detectDevices();
            selectBestDevice();
        }
    }

    private static void detectDevices() {
        devices.clear();

        boolean cudaAvailable = tryDetectCUDA();
        
        if (!cudaAvailable) {
            tryDetectOpenCL();
        }
    }

    private static boolean tryDetectCUDA() {
        try {
            Class.forName("jcuda.runtime.CudaRuntime");
            System.out.println("[GPU] CUDA runtime detected");
            
            int deviceCount = getCUDADeviceCountReflect();
            if (deviceCount > 0) {
                for (int i = 0; i < deviceCount; i++) {
                    String name = getCUDADeviceNameReflect(i);
                    long memory = getCUDADeviceMemoryReflect(i);
                    int smCount = getCUDADeviceSMCountReflect(i);
                    int clockRate = getCUDADeviceClockRateReflect(i);

                    DeviceInfo device = new DeviceInfo(i, name, GPUVendor.NVIDIA, GPUBackend.CUDA,
                            memory, smCount, clockRate, false);
                    devices.add(device);
                    System.out.println("[GPU] Found CUDA device: " + device);
                }
                return true;
            }
        } catch (ClassNotFoundException e) {
            System.out.println("[GPU] JCuda not available, trying OpenCL");
        } catch (Exception e) {
            System.out.println("[GPU] CUDA detection failed: " + e.getMessage());
        }
        return false;
    }

    private static int getCUDADeviceCountReflect() {
        try {
            Class<?> jcudaClass = Class.forName("jcuda.runtime.JCuda");
            java.lang.reflect.Method method = jcudaClass.getMethod("cudaGetDeviceCount");
            return (int) method.invoke(null);
        } catch (Exception e) {
            return 0;
        }
    }

    private static String getCUDADeviceNameReflect(int index) {
        try {
            Class<?> jcudaClass = Class.forName("jcuda.runtime.JCuda");
            byte[] name = new byte[256];
            java.lang.reflect.Method method = jcudaClass.getMethod("cudaDeviceGetName", byte[].class, int.class, int.class);
            method.invoke(null, name, 256, index);
            return new String(name).trim();
        } catch (Exception e) {
            return "Unknown CUDA Device";
        }
    }

    private static long getCUDADeviceMemoryReflect(int index) {
        try {
            Class<?> jcudaClass = Class.forName("jcuda.runtime.JCuda");
            long[] mem = new long[1];
            java.lang.reflect.Method method = jcudaClass.getMethod("cudaDeviceGetMemInfo", long[].class, long[].class);
            method.invoke(null, null, mem);
            return mem[0];
        } catch (Exception e) {
            return 0;
        }
    }

    private static int getCUDADeviceSMCountReflect(int index) {
        try {
            Class<?> propClass = Class.forName("jcuda.runtime.cudaDeviceProp");
            Object prop = propClass.getDeclaredConstructor().newInstance();
            Class<?> jcudaClass = Class.forName("jcuda.runtime.JCuda");
            java.lang.reflect.Method method = jcudaClass.getMethod("cudaGetDeviceProperties", propClass, int.class);
            method.invoke(null, prop, index);
            java.lang.reflect.Field field = propClass.getField("multiProcessorCount");
            return field.getInt(prop);
        } catch (Exception e) {
            return 0;
        }
    }

    private static int getCUDADeviceClockRateReflect(int index) {
        try {
            Class<?> propClass = Class.forName("jcuda.runtime.cudaDeviceProp");
            Object prop = propClass.getDeclaredConstructor().newInstance();
            Class<?> jcudaClass = Class.forName("jcuda.runtime.JCuda");
            java.lang.reflect.Method method = jcudaClass.getMethod("cudaGetDeviceProperties", propClass, int.class);
            method.invoke(null, prop, index);
            java.lang.reflect.Field field = propClass.getField("clockRate");
            return field.getInt(prop) / 1000;
        } catch (Exception e) {
            return 0;
        }
    }

    private static void tryDetectOpenCL() {
        try {
            Class.forName("org.jocl.CL");
            System.out.println("[GPU] OpenCL runtime detected");
            
            int deviceCount = getOpenCLDeviceCountReflect();
            for (int i = 0; i < deviceCount; i++) {
                String name = getOpenCLDeviceNameReflect(i);
                GPUVendor vendor = parseVendor(name);
                long memory = getOpenCLDeviceMemoryReflect(i);
                int computeUnits = getOpenCLComputeUnitsReflect(i);
                int clockRate = getOpenCLDeviceClockRateReflect(i);
                boolean unifiedMemory = getOpenCLUnifiedMemoryReflect(i);

                DeviceInfo device = new DeviceInfo(i, name, vendor, GPUBackend.OPENCL,
                        memory, computeUnits, clockRate, unifiedMemory);
                devices.add(device);
                System.out.println("[GPU] Found OpenCL device: " + device);
            }
        } catch (ClassNotFoundException e) {
            System.out.println("[GPU] OpenCL not available, using CPU");
        } catch (Exception e) {
            System.out.println("[GPU] OpenCL detection failed: " + e.getMessage());
        }
    }

    private static GPUVendor parseVendor(String name) {
        String lower = name.toLowerCase();
        if (lower.contains("nvidia")) return GPUVendor.NVIDIA;
        if (lower.contains("amd") || lower.contains("advanced micro devices")) return GPUVendor.AMD;
        if (lower.contains("intel")) return GPUVendor.INTEL;
        return GPUVendor.OTHER;
    }

    private static int getOpenCLDeviceCountReflect() {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int totalDevices = 0;
            for (int p = 0; p < numPlatforms[0]; p++) {
                Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                int CL_DEVICE_TYPE_GPU = 4;
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                totalDevices += numDevices[0];
            }
            return totalDevices;
        } catch (Exception e) {
            return 0;
        }
    }

    private static String getOpenCLDeviceNameReflect(int index) {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
            
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int currentIndex = 0;
            int CL_DEVICE_TYPE_GPU = 4;
            int CL_DEVICE_NAME = 4992;
            
            for (int p = 0; p < numPlatforms[0]; p++) {
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                
                for (int d = 0; d < numDevices[0]; d++) {
                    if (currentIndex == index) {
                        long[] size = new long[1];
                        java.lang.reflect.Method getInfo = clClass.getMethod("clGetDeviceInfo", 
                            deviceIdClass, int.class, long.class, Object.class, long[].class);
                        getInfo.invoke(null, devices[d], CL_DEVICE_NAME, 0, null, size);
                        
                        byte[] nameBytes = new byte[(int) size[0]];
                        getInfo.invoke(null, devices[d], CL_DEVICE_NAME, size[0], nameBytes, null);
                        return new String(nameBytes).trim();
                    }
                    currentIndex++;
                }
            }
        } catch (Exception e) {}
        return "Unknown OpenCL Device";
    }

    private static long getOpenCLDeviceMemoryReflect(int index) {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
            
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int currentIndex = 0;
            int CL_DEVICE_TYPE_GPU = 4;
            int CL_DEVICE_GLOBAL_MEM_SIZE = 4915;
            
            for (int p = 0; p < numPlatforms[0]; p++) {
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                
                for (int d = 0; d < numDevices[0]; d++) {
                    if (currentIndex == index) {
                        long[] mem = new long[1];
                        java.lang.reflect.Method getInfo = clClass.getMethod("clGetDeviceInfo", 
                            deviceIdClass, int.class, long.class, Object.class, long[].class);
                        getInfo.invoke(null, devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, 8, mem, null);
                        return mem[0];
                    }
                    currentIndex++;
                }
            }
        } catch (Exception e) {}
        return 0;
    }

    private static int getOpenCLComputeUnitsReflect(int index) {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
            
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int currentIndex = 0;
            int CL_DEVICE_TYPE_GPU = 4;
            int CL_DEVICE_MAX_COMPUTE_UNITS = 4917;
            
            for (int p = 0; p < numPlatforms[0]; p++) {
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                
                for (int d = 0; d < numDevices[0]; d++) {
                    if (currentIndex == index) {
                        int[] units = new int[1];
                        java.lang.reflect.Method getInfo = clClass.getMethod("clGetDeviceInfo", 
                            deviceIdClass, int.class, long.class, Object.class, long[].class);
                        getInfo.invoke(null, devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, 4, units, null);
                        return units[0];
                    }
                    currentIndex++;
                }
            }
        } catch (Exception e) {}
        return 0;
    }

    private static int getOpenCLDeviceClockRateReflect(int index) {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
            
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int currentIndex = 0;
            int CL_DEVICE_TYPE_GPU = 4;
            int CL_DEVICE_MAX_CLOCK_FREQUENCY = 4927;
            
            for (int p = 0; p < numPlatforms[0]; p++) {
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                
                for (int d = 0; d < numDevices[0]; d++) {
                    if (currentIndex == index) {
                        int[] freq = new int[1];
                        java.lang.reflect.Method getInfo = clClass.getMethod("clGetDeviceInfo", 
                            deviceIdClass, int.class, long.class, Object.class, long[].class);
                        getInfo.invoke(null, devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, freq, null);
                        return freq[0];
                    }
                    currentIndex++;
                }
            }
        } catch (Exception e) {}
        return 0;
    }

    private static boolean getOpenCLUnifiedMemoryReflect(int index) {
        try {
            Class<?> clClass = Class.forName("org.jocl.CL");
            Class<?> platformIdClass = Class.forName("org.jocl.cl_platform_id");
            Class<?> deviceIdClass = Class.forName("org.jocl.cl_device_id");
            
            Object[] platforms = new Object[16];
            int[] numPlatforms = new int[1];
            java.lang.reflect.Method getPlatforms = clClass.getMethod("clGetPlatformIDs", int.class, platformIdClass.arrayType(), int[].class);
            getPlatforms.invoke(null, 16, platforms, numPlatforms);
            
            int currentIndex = 0;
            int CL_DEVICE_TYPE_GPU = 4;
            int CL_DEVICE_HOST_UNIFIED_MEMORY = 4956;
            
            for (int p = 0; p < numPlatforms[0]; p++) {
                Object[] devices = new Object[64];
                int[] numDevices = new int[1];
                java.lang.reflect.Method getDevices = clClass.getMethod("clGetDeviceIDs", 
                    platformIdClass, int.class, int.class, deviceIdClass.arrayType(), int[].class);
                getDevices.invoke(null, platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, numDevices);
                
                for (int d = 0; d < numDevices[0]; d++) {
                    if (currentIndex == index) {
                        int[] unified = new int[1];
                        java.lang.reflect.Method getInfo = clClass.getMethod("clGetDeviceInfo", 
                            deviceIdClass, int.class, long.class, Object.class, long[].class);
                        getInfo.invoke(null, devices[d], CL_DEVICE_HOST_UNIFIED_MEMORY, 4, unified, null);
                        return unified[0] == 1;
                    }
                    currentIndex++;
                }
            }
        } catch (Exception e) {}
        return false;
    }

    private static void selectBestDevice() {
        if (devices.isEmpty()) {
            selectedBackend = GPUBackend.NONE;
            System.out.println("[GPU] No GPU devices found, using CPU");
            return;
        }

        DeviceInfo best = devices.get(0);
        for (DeviceInfo device : devices) {
            if (device.backend == GPUBackend.CUDA && best.backend != GPUBackend.CUDA) {
                best = device;
            } else if (device.getPerformanceScore() > best.getPerformanceScore()) {
                best = device;
            }
        }

        selectedDevice = best;
        selectedBackend = best.backend;
        
        System.out.println("[GPU] Selected best device: " + best);
        System.out.println("[GPU] Using backend: " + selectedBackend);
    }

    public static List<DeviceInfo> getDevices() {
        return new ArrayList<>(devices);
    }

    public static DeviceInfo getSelectedDevice() {
        return selectedDevice;
    }

    public static GPUBackend getSelectedBackend() {
        return selectedBackend;
    }

    public static boolean isCUDAAvailable() {
        return selectedBackend == GPUBackend.CUDA;
    }

    public static boolean isGPUAvailable() {
        return selectedBackend != GPUBackend.NONE;
    }

    public static void selectDevice(int index) {
        if (index >= 0 && index < devices.size()) {
            selectedDevice = devices.get(index);
            selectedBackend = selectedDevice.backend;
            System.out.println("[GPU] Selected device: " + selectedDevice);
        }
    }

    public static String getDeviceSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== GPU Device Summary ===\n");
        sb.append("Total devices: ").append(devices.size()).append("\n");
        sb.append("Selected backend: ").append(selectedBackend).append("\n");
        
        for (DeviceInfo device : devices) {
            sb.append(device).append("\n");
            sb.append("  Performance Score: ").append(String.format("%.2f", device.getPerformanceScore())).append("\n");
            if (device == selectedDevice) {
                sb.append("  [SELECTED]\n");
            }
        }
        return sb.toString();
    }
}