package com.qingxiaotuan.simpcore.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

public class GPUDiagnostic {
    
    public static void main(String[] args) {
        System.out.println("=" + "=".repeat(60));
        System.out.println("GPU Diagnostic Tool");
        System.out.println("=" + "=".repeat(60));
        
        try {
            System.out.println("\n1. Testing JOCL native library loading...");
            CL.setExceptionsEnabled(true);
            System.out.println("   ✓ JOCL native library loaded successfully");
            
            System.out.println("\n2. Querying OpenCL platforms...");
            int[] numPlatforms = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);
            System.out.println("   Found " + numPlatforms[0] + " platform(s)");
            
            if (numPlatforms[0] == 0) {
                System.out.println("   ✗ No OpenCL platforms found!");
                System.out.println("   Check: AMD Radeon Software / OpenCL driver installation");
                return;
            }
            
            cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(numPlatforms[0], platforms, null);
            
            for (int i = 0; i < platforms.length; i++) {
                String platformName = getString(platforms[i], CL_PLATFORM_NAME);
                String platformVendor = getString(platforms[i], CL_PLATFORM_VENDOR);
                String platformVersion = getString(platforms[i], CL_PLATFORM_VERSION);
                
                System.out.println("\n   Platform " + (i+1) + ":");
                System.out.println("     Name: " + platformName);
                System.out.println("     Vendor: " + platformVendor);
                System.out.println("     Version: " + platformVersion);
                
                System.out.println("     Devices:");
                        int[] numDevices = new int[1];
                        try {
                            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, null, numDevices);
                            if (numDevices[0] == 0) {
                                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, null, numDevices);
                            }
                            cl_device_id[] devices = new cl_device_id[numDevices[0]];
                            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices[0], devices, null);
                            
                            for (int j = 0; j < devices.length; j++) {
                                String deviceName = getString(devices[j], CL_DEVICE_NAME);
                                String deviceType = getDeviceType(devices[j]);
                                long[] globalMem = new long[1];
                                clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMem), null);
                                int[] computeUnits = new int[1];
                                clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_int, Pointer.to(computeUnits), null);
                                
                                System.out.println("       - " + deviceName);
                                System.out.println("         Type: " + deviceType);
                                System.out.println("         Memory: " + (globalMem[0] / (1024*1024)) + " MB");
                                System.out.println("         Compute Units: " + computeUnits[0]);
                                
                                if (deviceName.contains("AMD") || deviceName.contains("Radeon") || deviceName.contains("680")) {
                                    System.out.println("         ✅ This is your AMD Radeon 680M!");
                                }
                            }
                        } catch (Exception e) {
                            System.out.println("       Error: " + e.getMessage());
                            e.printStackTrace();
                        }
            }
            
            System.out.println("\n" + "=".repeat(60));
            System.out.println("GPU Diagnostic PASSED");
            System.out.println("=".repeat(60));
            
        } catch (UnsatisfiedLinkError e) {
            System.out.println("\n   ✗ Native library loading failed:");
            System.out.println("     " + e.getMessage());
            System.out.println("\n   Possible fixes:");
            System.out.println("     1. Install AMD Radeon Software (includes OpenCL driver)");
            System.out.println("     2. Ensure OpenCL.dll is in PATH");
            System.out.println("     3. Check JOCL version compatibility");
        } catch (Exception e) {
            System.out.println("\n   ✗ Error: " + e.getClass().getName());
            System.out.println("     " + e.getMessage());
            e.printStackTrace();
        }
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
    
    private static String getDeviceType(cl_device_id device) {
        long[] type = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_TYPE, Sizeof.cl_long, Pointer.to(type), null);
        long t = type[0];
        if (t == CL_DEVICE_TYPE_GPU) return "GPU";
        if (t == CL_DEVICE_TYPE_CPU) return "CPU";
        if (t == CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
        return "Other";
    }
}