package com.qingxiaotuan.simpcore.vision.capture;

import com.qingxiaotuan.simpcore.vision.capture.CameraSource.*;
import java.util.*;
import java.util.concurrent.*;

public class MockCameraSource implements CameraSource {
    
    private volatile boolean running = false;
    private CameraProperties properties;
    private FrameCallback callback;
    private ScheduledExecutorService executor;
    private ScheduledFuture<?> captureFuture;
    private final Random random;
    
    public MockCameraSource() {
        this.properties = new CameraProperties();
        this.random = new Random(42);
    }
    
    public MockCameraSource(CameraProperties properties) {
        this.properties = properties;
        this.random = new Random(42);
    }
    
    @Override
    public void initialize() throws CameraException {
        System.out.println("MockCameraSource initialized with: " + 
            properties.getWidth() + "x" + properties.getHeight() + "@" + properties.getFps() + "fps");
    }
    
    @Override
    public void start() throws CameraException {
        if (running) return;
        running = true;
        
        executor = Executors.newSingleThreadScheduledExecutor();
        long intervalMs = 1000 / properties.getFps();
        
        captureFuture = executor.scheduleAtFixedRate(() -> {
            try {
                Frame frame = capture();
                if (callback != null) {
                    callback.onFrame(frame);
                }
            } catch (CameraException e) {
                if (callback != null) {
                    callback.onError(e);
                }
            }
        }, 0, intervalMs, TimeUnit.MILLISECONDS);
        
        System.out.println("MockCameraSource started");
    }
    
    @Override
    public void stop() {
        running = false;
        if (captureFuture != null) {
            captureFuture.cancel(false);
        }
        if (executor != null) {
            executor.shutdown();
        }
        System.out.println("MockCameraSource stopped");
    }
    
    @Override
    public boolean isRunning() {
        return running;
    }
    
    @Override
    public Frame capture() throws CameraException {
        if (!running) {
            throw new CameraException("Camera is not running");
        }
        
        int bufferSize = properties.getBufferSize();
        byte[] data = new byte[bufferSize];
        
        generateTestFrame(data);
        
        Frame frame = new Frame(data, properties.getWidth(), properties.getHeight(), properties.getFormat());
        frame.setTimestamp(System.nanoTime() / 1000);
        
        if (properties.isDepthEnabled()) {
            byte[] depthData = new byte[properties.getWidth() * properties.getHeight()];
            generateDepthData(depthData);
            frame.setDepthData(depthData);
        }
        
        float[] intrinsics = new float[]{
            (float) properties.getWidth() / 2, 0, (float) properties.getWidth() / 2,
            0, (float) properties.getHeight() / 2, (float) properties.getHeight() / 2,
            0, 0, 1
        };
        frame.setIntrinsics(intrinsics);
        
        return frame;
    }
    
    private void generateTestFrame(byte[] data) {
        int width = properties.getWidth();
        int height = properties.getHeight();
        String format = properties.getFormat();
        int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                      format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels;
                
                byte r = (byte) ((x + y) % 256);
                byte g = (byte) ((x - y + 256) % 256);
                byte b = (byte) ((x * y) % 256);
                
                if (format.startsWith("RGB")) {
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                } else if (format.startsWith("BGR")) {
                    data[idx] = b;
                    data[idx + 1] = g;
                    data[idx + 2] = r;
                }
                
                if (channels == 4) {
                    data[idx + 3] = (byte) 255;
                }
            }
        }
        
        addNoise(data, 0.1f);
    }
    
    private void generateDepthData(byte[] depthData) {
        int width = properties.getWidth();
        int height = properties.getHeight();
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float distance = 1.0f + (float) Math.sqrt(Math.pow(x - width/2, 2) + Math.pow(y - height/2, 2)) / 100.0f;
                depthData[idx] = (byte) Math.min(255, (int) (distance * 25.5));
            }
        }
    }
    
    private void addNoise(byte[] data, float noiseLevel) {
        for (int i = 0; i < data.length; i++) {
            int noise = (int) ((random.nextDouble() - 0.5) * 2 * noiseLevel * 255);
            data[i] = (byte) Math.max(0, Math.min(255, data[i] + noise));
        }
    }
    
    @Override
    public void setCallback(FrameCallback callback) {
        this.callback = callback;
    }
    
    @Override
    public CameraProperties getProperties() {
        return properties;
    }
    
    @Override
    public void setProperties(CameraProperties properties) throws CameraException {
        if (running) {
            throw new CameraException("Cannot change properties while camera is running");
        }
        this.properties = properties;
    }
    
    @Override
    public List<CameraProperties> getAvailableProperties() {
        List<CameraProperties> props = new ArrayList<>();
        
        CameraProperties p1 = new CameraProperties();
        p1.setWidth(640); p1.setHeight(480); p1.setFps(60);
        props.add(p1);
        
        CameraProperties p2 = new CameraProperties();
        p2.setWidth(1280); p2.setHeight(720); p2.setFps(30);
        props.add(p2);
        
        CameraProperties p3 = new CameraProperties();
        p3.setWidth(1920); p3.setHeight(1080); p3.setFps(30);
        props.add(p3);
        
        CameraProperties p4 = new CameraProperties();
        p4.setWidth(3840); p4.setHeight(2160); p4.setFps(30); p4.setHdrEnabled(true);
        props.add(p4);
        
        return props;
    }
    
    @Override
    public void release() {
        stop();
        callback = null;
        System.out.println("MockCameraSource released");
    }
}