package com.qingxiaotuan.simpcore.vision.capture;

import org.bytedeco.javacv.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class RealCameraSource implements CameraSource {
    
    private FFmpegFrameGrabber grabber;
    private CameraProperties properties;
    private FrameCallback callback;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread captureThread;
    private int cameraIndex;
    private List<CameraProperties> availableProperties;
    
    public RealCameraSource() {
        this(0);
    }
    
    public RealCameraSource(int cameraIndex) {
        this.cameraIndex = cameraIndex;
        this.properties = new CameraProperties();
        discoverAvailableCameras();
    }
    
    private void discoverAvailableCameras() {
        availableProperties = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            try (FFmpegFrameGrabber testGrabber = new FFmpegFrameGrabber(i)) {
                testGrabber.setOption("rtbufsize", "10M");
                testGrabber.start();
                CameraProperties props = new CameraProperties();
                props.setWidth(testGrabber.getImageWidth());
                props.setHeight(testGrabber.getImageHeight());
                props.setFps((int) testGrabber.getFrameRate());
                props.setFormat("RGB");
                availableProperties.add(props);
                testGrabber.stop();
            } catch (Exception e) {
                break;
            }
        }
        if (availableProperties.isEmpty()) {
            CameraProperties defaultProps = new CameraProperties();
            availableProperties.add(defaultProps);
        }
    }
    
    @Override
    public void initialize() throws CameraException {
        try {
            grabber = new FFmpegFrameGrabber(cameraIndex);
            grabber.setOption("rtbufsize", "10M");
            grabber.setOption("stimeout", "5000000");
            
            grabber.setImageWidth(properties.getWidth());
            grabber.setImageHeight(properties.getHeight());
            grabber.setFrameRate(properties.getFps());
            
            grabber.start();
            
            int actualWidth = grabber.getImageWidth();
            int actualHeight = grabber.getImageHeight();
            double actualFps = grabber.getFrameRate();
            
            properties.setWidth(actualWidth);
            properties.setHeight(actualHeight);
            properties.setFps((int) actualFps);
            
        } catch (Exception e) {
            throw new CameraException("Failed to initialize camera", e);
        }
    }
    
    @Override
    public void start() throws CameraException {
        if (running.get()) return;
        
        running.set(true);
        captureThread = new Thread(this::captureLoop, "CameraCapture-" + cameraIndex);
        captureThread.setDaemon(true);
        captureThread.start();
    }
    
    private void captureLoop() {
        Java2DFrameConverter converter = new Java2DFrameConverter();
        OpenCVFrameConverter.ToMat matConverter = new OpenCVFrameConverter.ToMat();
        
        while (running.get()) {
            try {
                Frame frame = grabber.grab();
                if (frame == null) {
                    Thread.sleep(33);
                    continue;
                }
                
                Frame capturedFrame = convertFrame(frame);
                if (callback != null) {
                    callback.onFrame(capturedFrame);
                }
                
            } catch (Exception e) {
                if (running.get() && callback != null) {
                    callback.onError(new CameraException("Capture error", e));
                }
                break;
            }
        }
    }
    
    private Frame convertFrame(org.bytedeco.javacv.Frame inputFrame) {
        Java2DFrameConverter converter = new Java2DFrameConverter();
        java.awt.image.BufferedImage bufferedImage = converter.convert(inputFrame);
        
        if (bufferedImage == null) {
            throw new RuntimeException("Failed to convert frame");
        }
        
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        String format = "RGB";
        int channels = 3;
        
        byte[] data = new byte[width * height * channels];
        int[] pixels = bufferedImage.getRGB(0, 0, width, height, null, 0, width);
        
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            data[i * 3] = (byte) ((pixel >> 16) & 0xFF);
            data[i * 3 + 1] = (byte) ((pixel >> 8) & 0xFF);
            data[i * 3 + 2] = (byte) (pixel & 0xFF);
        }
        
        return new Frame(data, width, height, format);
    }
    
    @Override
    public void stop() {
        running.set(false);
        if (captureThread != null) {
            try {
                captureThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    @Override
    public boolean isRunning() {
        return running.get() && grabber != null;
    }
    
    @Override
    public Frame capture() throws CameraException {
        if (!running.get()) {
            throw new CameraException("Camera not running");
        }
        
        try {
            Frame frame = grabber.grab();
            if (frame == null) {
                return null;
            }
            return convertFrame(frame);
        } catch (Exception e) {
            throw new CameraException("Failed to capture frame", e);
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
        this.properties = properties;
        if (grabber != null) {
            stop();
            initialize();
            start();
        }
    }
    
    @Override
    public List<CameraProperties> getAvailableProperties() {
        return availableProperties;
    }
    
    @Override
    public void release() {
        stop();
        if (grabber != null) {
            try {
                grabber.stop();
                grabber.release();
            } catch (Exception e) {
                // ignore
            }
            grabber = null;
        }
    }
    
    public int getCameraIndex() {
        return cameraIndex;
    }
    
    public void setCameraIndex(int cameraIndex) {
        this.cameraIndex = cameraIndex;
        discoverAvailableCameras();
    }
}
