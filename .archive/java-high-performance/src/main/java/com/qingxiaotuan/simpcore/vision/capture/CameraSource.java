package com.qingxiaotuan.simpcore.vision.capture;

import java.util.List;
import java.util.Map;

public interface CameraSource {
    
    void initialize() throws CameraException;
    
    void start() throws CameraException;
    
    void stop();
    
    boolean isRunning();
    
    Frame capture() throws CameraException;
    
    void setCallback(FrameCallback callback);
    
    CameraProperties getProperties();
    
    void setProperties(CameraProperties properties) throws CameraException;
    
    List<CameraProperties> getAvailableProperties();
    
    void release();
    
    interface FrameCallback {
        void onFrame(Frame frame);
        void onError(CameraException e);
    }
    
    class CameraException extends Exception {
        public CameraException(String message) { super(message); }
        public CameraException(String message, Throwable cause) { super(message, cause); }
    }
    
    class CameraProperties {
        private int width;
        private int height;
        private int fps;
        private int bitDepth;
        private boolean hdrEnabled;
        private boolean depthEnabled;
        private String format;
        private Map<String, Object> extraParams;
        
        public CameraProperties() {
            this.width = 1920;
            this.height = 1080;
            this.fps = 30;
            this.bitDepth = 8;
            this.hdrEnabled = false;
            this.depthEnabled = false;
            this.format = "RGB";
        }
        
        public int getWidth() { return width; }
        public void setWidth(int width) { this.width = width; }
        
        public int getHeight() { return height; }
        public void setHeight(int height) { this.height = height; }
        
        public int getFps() { return fps; }
        public void setFps(int fps) { this.fps = fps; }
        
        public int getBitDepth() { return bitDepth; }
        public void setBitDepth(int bitDepth) { this.bitDepth = bitDepth; }
        
        public boolean isHdrEnabled() { return hdrEnabled; }
        public void setHdrEnabled(boolean hdrEnabled) { this.hdrEnabled = hdrEnabled; }
        
        public boolean isDepthEnabled() { return depthEnabled; }
        public void setDepthEnabled(boolean depthEnabled) { this.depthEnabled = depthEnabled; }
        
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        
        public Map<String, Object> getExtraParams() { return extraParams; }
        public void setExtraParams(Map<String, Object> extraParams) { this.extraParams = extraParams; }
        
        public int getPixelCount() { return width * height; }
        
        public int getBufferSize() {
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            return width * height * channels * (bitDepth / 8);
        }
    }
    
    class Frame {
        private byte[] data;
        private int width;
        private int height;
        private int stride;
        private String format;
        private long timestamp;
        private byte[] depthData;
        private float[] intrinsics;
        private float[] extrinsics;
        
        public Frame(byte[] data, int width, int height, String format) {
            this.data = data;
            this.width = width;
            this.height = height;
            this.format = format;
            this.timestamp = System.currentTimeMillis();
            this.stride = width * (format.equals("RGB") || format.equals("BGR") ? 3 : 
                                  format.equals("RGBA") || format.equals("BGRA") ? 4 : 1);
        }
        
        public byte[] getData() { return data; }
        public void setData(byte[] data) { this.data = data; }
        
        public int getWidth() { return width; }
        public void setWidth(int width) { this.width = width; }
        
        public int getHeight() { return height; }
        public void setHeight(int height) { this.height = height; }
        
        public int getStride() { return stride; }
        public void setStride(int stride) { this.stride = stride; }
        
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
        
        public byte[] getDepthData() { return depthData; }
        public void setDepthData(byte[] depthData) { this.depthData = depthData; }
        
        public float[] getIntrinsics() { return intrinsics; }
        public void setIntrinsics(float[] intrinsics) { this.intrinsics = intrinsics; }
        
        public float[] getExtrinsics() { return extrinsics; }
        public void setExtrinsics(float[] extrinsics) { this.extrinsics = extrinsics; }
        
        public int getPixelCount() { return width * height; }
        
        public int getBufferSize() {
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            return width * height * channels;
        }
    }
}