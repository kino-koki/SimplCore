package com.qingxiaotuan.simpcore.vision.preprocessing;

import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.*;
import java.util.*;

public class ImageProcessors {
    
    public static class DenoisingProcessor implements ImageProcessor {
        private boolean enabled = true;
        private int kernelSize = 3;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = new byte[input.length];
            
            int halfKernel = kernelSize / 2;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = (y * width + x) * channels;
                    
                    for (int c = 0; c < channels; c++) {
                        int sum = 0;
                        int count = 0;
                        
                        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                
                                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                    sum += input[(ny * width + nx) * channels + c] & 0xFF;
                                    count++;
                                }
                            }
                        }
                        
                        output[idx + c] = (byte) (sum / count);
                    }
                }
            }
            
            frame.setData(output);
            frame.addMetadata("denoised", true);
            return frame;
        }
        
        @Override
        public String getName() { return "DenoisingProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public int getKernelSize() { return kernelSize; }
        public void setKernelSize(int kernelSize) { 
            this.kernelSize = kernelSize; 
        }
    }
    
    public static class EdgeEnhancementProcessor implements ImageProcessor {
        private boolean enabled = true;
        private float strength = 1.0f;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = Arrays.copyOf(input, input.length);
            
            float[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            float[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = (y * width + x) * channels;
                    
                    for (int c = 0; c < channels; c++) {
                        float gx = 0, gy = 0;
                        
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                int nidx = (ny * width + nx) * channels + c;
                                float pixel = (input[nidx] & 0xFF) / 255.0f;
                                gx += pixel * sobelX[ky + 1][kx + 1];
                                gy += pixel * sobelY[ky + 1][kx + 1];
                            }
                        }
                        
                        float magnitude = (float) Math.sqrt(gx * gx + gy * gy);
                        int original = input[idx + c] & 0xFF;
                        int enhanced = (int) (original + magnitude * strength * 50);
                        output[idx + c] = (byte) Math.max(0, Math.min(255, enhanced));
                    }
                }
            }
            
            frame.setData(output);
            frame.addMetadata("edgeEnhanced", true);
            return frame;
        }
        
        @Override
        public String getName() { return "EdgeEnhancementProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public float getStrength() { return strength; }
        public void setStrength(float strength) { this.strength = strength; }
    }
    
    public static class GammaCorrectionProcessor implements ImageProcessor {
        private boolean enabled = true;
        private float gamma = 1.0f;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            byte[] input = frame.getData();
            byte[] output = new byte[input.length];
            
            float invGamma = 1.0f / gamma;
            
            for (int i = 0; i < input.length; i++) {
                float normalized = (input[i] & 0xFF) / 255.0f;
                float corrected = (float) Math.pow(normalized, invGamma);
                output[i] = (byte) (corrected * 255);
            }
            
            frame.setData(output);
            frame.addMetadata("gammaCorrected", true);
            frame.addMetadata("gammaValue", gamma);
            return frame;
        }
        
        @Override
        public String getName() { return "GammaCorrectionProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public float getGamma() { return gamma; }
        public void setGamma(float gamma) { this.gamma = gamma; }
    }
    
    public static class ResizeProcessor implements ImageProcessor {
        private boolean enabled = true;
        private int targetWidth;
        private int targetHeight;
        
        public ResizeProcessor(int targetWidth, int targetHeight) {
            this.targetWidth = targetWidth;
            this.targetHeight = targetHeight;
        }
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = new byte[targetWidth * targetHeight * channels];
            
            float scaleX = (float) width / targetWidth;
            float scaleY = (float) height / targetHeight;
            
            for (int ty = 0; ty < targetHeight; ty++) {
                for (int tx = 0; tx < targetWidth; tx++) {
                    int idx = (ty * targetWidth + tx) * channels;
                    
                    float sx = tx * scaleX;
                    float sy = ty * scaleY;
                    
                    int x0 = (int) sx;
                    int y0 = (int) sy;
                    int x1 = Math.min(x0 + 1, width - 1);
                    int y1 = Math.min(y0 + 1, height - 1);
                    
                    float wx = sx - x0;
                    float wy = sy - y0;
                    
                    for (int c = 0; c < channels; c++) {
                        int p00 = input[(y0 * width + x0) * channels + c] & 0xFF;
                        int p01 = input[(y0 * width + x1) * channels + c] & 0xFF;
                        int p10 = input[(y1 * width + x0) * channels + c] & 0xFF;
                        int p11 = input[(y1 * width + x1) * channels + c] & 0xFF;
                        
                        float val = (1 - wx) * (1 - wy) * p00 +
                                   wx * (1 - wy) * p01 +
                                   (1 - wx) * wy * p10 +
                                   wx * wy * p11;
                        
                        output[idx + c] = (byte) Math.max(0, Math.min(255, (int) val));
                    }
                }
            }
            
            frame.setData(output);
            frame.setWidth(targetWidth);
            frame.setHeight(targetHeight);
            frame.setStride(targetWidth * channels);
            frame.addMetadata("resized", true);
            frame.addMetadata("originalWidth", width);
            frame.addMetadata("originalHeight", height);
            return frame;
        }
        
        @Override
        public String getName() { return "ResizeProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public int getTargetWidth() { return targetWidth; }
        public void setTargetWidth(int targetWidth) { this.targetWidth = targetWidth; }
        
        public int getTargetHeight() { return targetHeight; }
        public void setTargetHeight(int targetHeight) { this.targetHeight = targetHeight; }
    }
    
    public static class ColorNormalizationProcessor implements ImageProcessor {
        private boolean enabled = true;
        private float[] mean = {0.485f, 0.456f, 0.406f};
        private float[] std = {0.229f, 0.224f, 0.225f};
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            float[] normalized = new float[width * height * channels];
            
            for (int i = 0; i < width * height; i++) {
                for (int c = 0; c < Math.min(channels, 3); c++) {
                    float val = (input[i * channels + c] & 0xFF) / 255.0f;
                    normalized[i * channels + c] = (val - mean[c]) / std[c];
                }
                if (channels == 4) {
                    normalized[i * channels + 3] = (input[i * channels + 3] & 0xFF) / 255.0f;
                }
            }
            
            byte[] output = new byte[input.length];
            for (int i = 0; i < normalized.length; i++) {
                float clamped = Math.max(-2, Math.min(2, normalized[i]));
                output[i] = (byte) ((clamped + 2) / 4 * 255);
            }
            
            frame.setData(output);
            frame.addMetadata("colorNormalized", true);
            return frame;
        }
        
        @Override
        public String getName() { return "ColorNormalizationProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public float[] getMean() { return mean; }
        public void setMean(float[] mean) { this.mean = mean; }
        
        public float[] getStd() { return std; }
        public void setStd(float[] std) { this.std = std; }
    }
    
    public static class DistortionCorrectionProcessor implements ImageProcessor {
        private boolean enabled = true;
        private float[] distortionCoeffs = {0, 0, 0, 0};
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = new byte[input.length];
            
            float cx = width / 2.0f;
            float cy = height / 2.0f;
            float k1 = distortionCoeffs[0];
            float k2 = distortionCoeffs[1];
            float p1 = distortionCoeffs[2];
            float p2 = distortionCoeffs[3];
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float dx = (x - cx) / cx;
                    float dy = (y - cy) / cy;
                    float r2 = dx * dx + dy * dy;
                    float r4 = r2 * r2;
                    
                    float xDistorted = dx * (1 + k1 * r2 + k2 * r4) + 
                                      2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx);
                    float yDistorted = dy * (1 + k1 * r2 + k2 * r4) + 
                                      p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy;
                    
                    int srcX = (int) (cx + xDistorted * cx);
                    int srcY = (int) (cy + yDistorted * cy);
                    
                    int idx = (y * width + x) * channels;
                    
                    if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                        int srcIdx = (srcY * width + srcX) * channels;
                        System.arraycopy(input, srcIdx, output, idx, channels);
                    } else {
                        Arrays.fill(output, idx, idx + channels, (byte) 0);
                    }
                }
            }
            
            frame.setData(output);
            frame.addMetadata("distortionCorrected", true);
            return frame;
        }
        
        @Override
        public String getName() { return "DistortionCorrectionProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public float[] getDistortionCoeffs() { return distortionCoeffs; }
        public void setDistortionCoeffs(float[] distortionCoeffs) { 
            this.distortionCoeffs = distortionCoeffs; 
        }
    }
    
    public static class HistogramEqualizationProcessor implements ImageProcessor {
        private boolean enabled = true;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = new byte[input.length];
            
            for (int c = 0; c < channels; c++) {
                int[] histogram = new int[256];
                int pixelCount = width * height;
                
                for (int i = 0; i < pixelCount; i++) {
                    int idx = i * channels + c;
                    histogram[input[idx] & 0xFF]++;
                }
                
                int[] cumulative = new int[256];
                cumulative[0] = histogram[0];
                for (int i = 1; i < 256; i++) {
                    cumulative[i] = cumulative[i - 1] + histogram[i];
                }
                
                float scale = 255.0f / pixelCount;
                int[] equalized = new int[256];
                for (int i = 0; i < 256; i++) {
                    equalized[i] = (int) (cumulative[i] * scale);
                }
                
                for (int i = 0; i < pixelCount; i++) {
                    int idx = i * channels + c;
                    output[idx] = (byte) equalized[input[idx] & 0xFF];
                }
            }
            
            frame.setData(output);
            frame.addMetadata("histogramEqualized", true);
            return frame;
        }
        
        @Override
        public String getName() { return "HistogramEqualizationProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
    }
    
    public static class BlurProcessor implements ImageProcessor {
        private boolean enabled = true;
        private int kernelSize = 5;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = new byte[input.length];
            
            int halfKernel = kernelSize / 2;
            float kernelSum = kernelSize * kernelSize;
            
            for (int y = halfKernel; y < height - halfKernel; y++) {
                for (int x = halfKernel; x < width - halfKernel; x++) {
                    int idx = (y * width + x) * channels;
                    
                    for (int c = 0; c < channels; c++) {
                        float sum = 0;
                        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                sum += input[(ny * width + nx) * channels + c] & 0xFF;
                            }
                        }
                        output[idx + c] = (byte) (sum / kernelSum);
                    }
                }
            }
            
            System.arraycopy(input, 0, output, 0, halfKernel * width * channels);
            System.arraycopy(input, (height - halfKernel) * width * channels, 
                           output, (height - halfKernel) * width * channels, 
                           halfKernel * width * channels);
            
            frame.setData(output);
            frame.addMetadata("blurred", true);
            return frame;
        }
        
        @Override
        public String getName() { return "BlurProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public int getKernelSize() { return kernelSize; }
        public void setKernelSize(int kernelSize) { this.kernelSize = kernelSize; }
    }
    
    public static class SharpenProcessor implements ImageProcessor {
        private boolean enabled = true;
        private float amount = 1.0f;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] input = frame.getData();
            byte[] output = Arrays.copyOf(input, input.length);
            
            float[][] kernel = {
                {0, -1, 0},
                {-1, 5 + amount, -1},
                {0, -1, 0}
            };
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = (y * width + x) * channels;
                    
                    for (int c = 0; c < channels; c++) {
                        float sum = 0;
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                sum += (input[(ny * width + nx) * channels + c] & 0xFF) * kernel[ky + 1][kx + 1];
                            }
                        }
                        output[idx + c] = (byte) Math.max(0, Math.min(255, (int) sum));
                    }
                }
            }
            
            frame.setData(output);
            frame.addMetadata("sharpened", true);
            return frame;
        }
        
        @Override
        public String getName() { return "SharpenProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        public float getAmount() { return amount; }
        public void setAmount(float amount) { this.amount = amount; }
    }
    
    public static class GrayscaleProcessor implements ImageProcessor {
        private boolean enabled = true;
        
        @Override
        public ProcessedFrame process(ProcessedFrame frame) {
            if (!enabled) return frame;
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            
            if (format.equals("GRAY")) return frame;
            
            byte[] input = frame.getData();
            byte[] output = new byte[width * height];
            
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            for (int i = 0; i < width * height; i++) {
                int r = input[i * channels] & 0xFF;
                int g = input[i * channels + 1] & 0xFF;
                int b = input[i * channels + 2] & 0xFF;
                int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);
                output[i] = (byte) gray;
            }
            
            frame.setData(output);
            frame.setFormat("GRAY");
            frame.setStride(width);
            frame.addMetadata("grayscale", true);
            return frame;
        }
        
        @Override
        public String getName() { return "GrayscaleProcessor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
    }
}