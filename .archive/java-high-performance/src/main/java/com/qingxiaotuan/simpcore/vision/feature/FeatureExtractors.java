package com.qingxiaotuan.simpcore.vision.feature;

import com.qingxiaotuan.simpcore.vision.feature.FeatureExtractionLayer.*;
import com.qingxiaotuan.simpcore.vision.model.VisionModel;
import com.qingxiaotuan.simpcore.vision.preprocessing.PreprocessingPipeline.ProcessedFrame;
import java.util.*;

public class FeatureExtractors {
    
    public static class SimpleKeypointExtractor implements FeatureExtractor {
        private boolean enabled = true;
        private int maxKeypoints = 200;
        private int threshold = 20;
        private final int featureDim = 128;
        
        @Override
        public FeatureDescriptor extract(ProcessedFrame frame) {
            if (!enabled) return new FeatureDescriptor();
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] data = frame.getData();
            
            List<int[]> keypoints = new ArrayList<>();
            List<Float> scores = new ArrayList<>();
            
            int step = 4;
            int halfWindow = 8;
            
            for (int y = halfWindow; y < height - halfWindow; y += step) {
                for (int x = halfWindow; x < width - halfWindow; x += step) {
                    float response = computeHarrisResponse(data, x, y, width, height, channels);
                    
                    if (response > threshold && isLocalMax(data, x, y, width, height, channels, step)) {
                        keypoints.add(new int[]{x, y});
                        scores.add(response);
                        
                        if (keypoints.size() >= maxKeypoints) break;
                    }
                }
                if (keypoints.size() >= maxKeypoints) break;
            }
            
            int[] keypointArray = new int[keypoints.size() * 2];
            float[] scoreArray = new float[scores.size()];
            float[] descriptors = new float[keypoints.size() * featureDim];
            
            for (int i = 0; i < keypoints.size(); i++) {
                keypointArray[i * 2] = keypoints.get(i)[0];
                keypointArray[i * 2 + 1] = keypoints.get(i)[1];
                scoreArray[i] = scores.get(i);
                
                float[] desc = computeDescriptor(data, keypoints.get(i)[0], keypoints.get(i)[1], 
                    width, height, channels);
                System.arraycopy(desc, 0, descriptors, i * featureDim, featureDim);
            }
            
            FeatureDescriptor fd = new FeatureDescriptor(keypointArray, descriptors, 
                "simple_keypoint", featureDim);
            fd.setKeypointScores(scoreArray);
            
            return fd;
        }
        
        private float computeHarrisResponse(byte[] data, int x, int y, int width, int height, int channels) {
            float[] dx = new float[9];
            float[] dy = new float[9];
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int idx = ((y + ky) * width + (x + kx)) * channels;
                    int idxX = ((y + ky) * width + (x + kx + 1)) * channels;
                    int idxY = ((y + ky + 1) * width + (x + kx)) * channels;
                    
                    float val = (data[idx] & 0xFF) / 255.0f;
                    float valX = (x + kx + 1 < width) ? (data[idxX] & 0xFF) / 255.0f : val;
                    float valY = (y + ky + 1 < height) ? (data[idxY] & 0xFF) / 255.0f : val;
                    
                    dx[ky + 1 + (kx + 1) * 3] = valX - val;
                    dy[ky + 1 + (kx + 1) * 3] = valY - val;
                }
            }
            
            float sumIx2 = 0, sumIy2 = 0, sumIxy = 0;
            for (int i = 0; i < 9; i++) {
                sumIx2 += dx[i] * dx[i];
                sumIy2 += dy[i] * dy[i];
                sumIxy += dx[i] * dy[i];
            }
            
            float det = sumIx2 * sumIy2 - sumIxy * sumIxy;
            float trace = sumIx2 + sumIy2;
            return det - 0.04f * trace * trace;
        }
        
        private boolean isLocalMax(byte[] data, int x, int y, int width, int height, int channels, int step) {
            float centerResponse = computeHarrisResponse(data, x, y, width, height, channels);
            
            for (int ky = -step; ky <= step; ky += step) {
                for (int kx = -step; kx <= step; kx += step) {
                    if (ky == 0 && kx == 0) continue;
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx > 0 && nx < width && ny > 0 && ny < height) {
                        float neighborResponse = computeHarrisResponse(data, nx, ny, width, height, channels);
                        if (neighborResponse > centerResponse) return false;
                    }
                }
            }
            return true;
        }
        
        private float[] computeDescriptor(byte[] data, int x, int y, int width, int height, int channels) {
            float[] desc = new float[featureDim];
            int binSize = 4;
            int numBins = 8;
            
            for (int by = 0; by < 4; by++) {
                for (int bx = 0; bx < 4; bx++) {
                    float[] hist = new float[numBins];
                    
                    for (int dy = 0; dy < binSize; dy++) {
                        for (int dx = 0; dx < binSize; dx++) {
                            int px = x + bx * binSize + dx - 8;
                            int py = y + by * binSize + dy - 8;
                            
                            if (px > 0 && px < width - 1 && py > 0 && py < height - 1) {
                                int idx = (py * width + px) * channels;
                                float val = (data[idx] & 0xFF) / 255.0f;
                                float valX = (data[(py * width + px + 1) * channels] & 0xFF) / 255.0f;
                                float valY = (data[((py + 1) * width + px) * channels] & 0xFF) / 255.0f;
                                
                                float gx = valX - val;
                                float gy = valY - val;
                                float angle = (float) Math.atan2(gy, gx);
                                if (angle < 0) angle += (float) Math.PI * 2;
                                
                                int bin = (int) (angle / ((float) Math.PI * 2) * numBins);
                                float mag = (float) Math.sqrt(gx * gx + gy * gy);
                                hist[bin] += mag;
                            }
                        }
                    }
                    
                    for (int i = 0; i < numBins; i++) {
                        desc[(by * 4 + bx) * numBins + i] = hist[i];
                    }
                }
            }
            
            float norm = 0;
            for (float v : desc) norm += v * v;
            norm = (float) Math.sqrt(norm);
            for (int i = 0; i < desc.length; i++) desc[i] /= norm;
            
            return desc;
        }
        
        @Override
        public String getName() { return "SimpleKeypointExtractor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        @Override
        public int getFeatureDim() { return featureDim; }
        
        public int getMaxKeypoints() { return maxKeypoints; }
        public void setMaxKeypoints(int maxKeypoints) { this.maxKeypoints = maxKeypoints; }
    }
    
    public static class ColorHistogramExtractor implements FeatureExtractor {
        private boolean enabled = true;
        private int binsPerChannel = 8;
        private final int featureDim = 8 * 8 * 8;
        
        @Override
        public FeatureDescriptor extract(ProcessedFrame frame) {
            if (!enabled) return new FeatureDescriptor();
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] data = frame.getData();
            float[] hist = new float[featureDim];
            Arrays.fill(hist, 0);
            
            boolean isRGB = format.startsWith("RGB");
            
            for (int i = 0; i < width * height; i++) {
                int r = data[i * channels] & 0xFF;
                int g = data[i * channels + 1] & 0xFF;
                int b = data[i * channels + 2] & 0xFF;
                
                if (!isRGB) {
                    int temp = r;
                    r = b;
                    b = temp;
                }
                
                int rBin = Math.min(binsPerChannel - 1, r * binsPerChannel / 256);
                int gBin = Math.min(binsPerChannel - 1, g * binsPerChannel / 256);
                int bBin = Math.min(binsPerChannel - 1, b * binsPerChannel / 256);
                
                int idx = rBin * binsPerChannel * binsPerChannel + gBin * binsPerChannel + bBin;
                hist[idx]++;
            }
            
            float total = width * height;
            for (int i = 0; i < hist.length; i++) {
                hist[i] /= total;
            }
            
            return new FeatureDescriptor(hist, "color_histogram");
        }
        
        @Override
        public String getName() { return "ColorHistogramExtractor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        @Override
        public int getFeatureDim() { return featureDim; }
        
        public int getBinsPerChannel() { return binsPerChannel; }
        public void setBinsPerChannel(int binsPerChannel) { 
            this.binsPerChannel = binsPerChannel; 
        }
    }
    
    public static class HogExtractor implements FeatureExtractor {
        private boolean enabled = true;
        private int cellSize = 8;
        private int blockSize = 2;
        private int numBins = 9;
        private final int featureDim = 3780;
        
        @Override
        public FeatureDescriptor extract(ProcessedFrame frame) {
            if (!enabled) return new FeatureDescriptor();
            
            int width = frame.getWidth();
            int height = frame.getHeight();
            String format = frame.getFormat();
            int channels = format.equals("RGB") || format.equals("BGR") ? 3 : 
                          format.equals("RGBA") || format.equals("BGRA") ? 4 : 1;
            
            byte[] data = frame.getData();
            
            int cellsX = width / cellSize;
            int cellsY = height / cellSize;
            
            float[][] gradientX = new float[height][width];
            float[][] gradientY = new float[height][width];
            float[][] magnitude = new float[height][width];
            float[][] orientation = new float[height][width];
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = (y * width + x) * channels;
                    float val = (data[idx] & 0xFF) / 255.0f;
                    float valX = (x + 1 < width) ? (data[(y * width + x + 1) * channels] & 0xFF) / 255.0f : val;
                    float valY = (y + 1 < height) ? (data[((y + 1) * width + x) * channels] & 0xFF) / 255.0f : val;
                    
                    gradientX[y][x] = valX - val;
                    gradientY[y][x] = valY - val;
                    magnitude[y][x] = (float) Math.sqrt(gradientX[y][x] * gradientX[y][x] + 
                                                         gradientY[y][x] * gradientY[y][x]);
                    orientation[y][x] = (float) Math.atan2(gradientY[y][x], gradientX[y][x]);
                    if (orientation[y][x] < 0) orientation[y][x] += (float) Math.PI;
                }
            }
            
            float[][][] cellHists = new float[cellsY][cellsX][numBins];
            
            for (int cy = 0; cy < cellsY; cy++) {
                for (int cx = 0; cx < cellsX; cx++) {
                    for (int dy = 0; dy < cellSize; dy++) {
                        for (int dx = 0; dx < cellSize; dx++) {
                            int y = cy * cellSize + dy;
                            int x = cx * cellSize + dx;
                            
                            float mag = magnitude[y][x];
                            float orient = orientation[y][x];
                            
                            int bin = (int) (orient / (float) Math.PI * numBins);
                            bin = Math.min(numBins - 1, Math.max(0, bin));
                            
                            cellHists[cy][cx][bin] += mag;
                        }
                    }
                }
            }
            
            List<Float> features = new ArrayList<>();
            
            for (int by = 0; by <= cellsY - blockSize; by++) {
                for (int bx = 0; bx <= cellsX - blockSize; bx++) {
                    float[] blockVec = new float[blockSize * blockSize * numBins];
                    int idx = 0;
                    
                    for (int cy = 0; cy < blockSize; cy++) {
                        for (int cx = 0; cx < blockSize; cx++) {
                            for (int b = 0; b < numBins; b++) {
                                blockVec[idx++] = cellHists[by + cy][bx + cx][b];
                            }
                        }
                    }
                    
                    float norm = 0;
                    for (float v : blockVec) norm += v * v;
                    norm = (float) Math.sqrt(norm + 0.001f);
                    for (float v : blockVec) features.add(v / norm);
                }
            }
            
            float[] featureArray = new float[features.size()];
            for (int i = 0; i < features.size(); i++) {
                featureArray[i] = features.get(i);
            }
            
            return new FeatureDescriptor(featureArray, "hog");
        }
        
        @Override
        public String getName() { return "HogExtractor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        @Override
        public int getFeatureDim() { return featureDim; }
        
        public int getCellSize() { return cellSize; }
        public void setCellSize(int cellSize) { this.cellSize = cellSize; }
    }
    
    public static class CnnFeatureExtractor implements FeatureExtractor {
        private boolean enabled = true;
        private String modelPath;
        private boolean modelLoaded = false;
        private VisionModel visionModel;
        private final int featureDim = 512;
        private static final int TARGET_SIZE = 224;
        
        public CnnFeatureExtractor(String modelPath) {
            this.modelPath = modelPath;
        }
        
        @Override
        public FeatureDescriptor extract(ProcessedFrame frame) {
            if (!enabled) return new FeatureDescriptor();
            
            if (!modelLoaded) {
                loadModel();
            }
            
            byte[] data = frame.getData();
            int width = frame.getWidth();
            int height = frame.getHeight();
            
            float[] resizedInput = resizeAndNormalize(data, width, height);
            
            float[] features = visionModel.forward(resizedInput);
            
            float norm = 0;
            for (float v : features) norm += v * v;
            norm = (float) Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < features.length; i++) features[i] /= norm;
            }
            
            return new FeatureDescriptor(features, "cnn");
        }
        
        private float[] resizeAndNormalize(byte[] data, int width, int height) {
            float[] result = new float[3 * TARGET_SIZE * TARGET_SIZE];
            
            float scaleX = (float) width / TARGET_SIZE;
            float scaleY = (float) height / TARGET_SIZE;
            
            for (int ty = 0; ty < TARGET_SIZE; ty++) {
                for (int tx = 0; tx < TARGET_SIZE; tx++) {
                    int sx = Math.min((int) (tx * scaleX), width - 1);
                    int sy = Math.min((int) (ty * scaleY), height - 1);
                    
                    int idx = (sy * width + sx) * 4;
                    int outIdx = (ty * TARGET_SIZE + tx) * 3;
                    
                    result[outIdx] = ((data[idx] & 0xFF) / 255.0f - 0.5f) * 2.0f;
                    result[outIdx + 1] = ((data[idx + 1] & 0xFF) / 255.0f - 0.5f) * 2.0f;
                    result[outIdx + 2] = ((data[idx + 2] & 0xFF) / 255.0f - 0.5f) * 2.0f;
                }
            }
            
            return result;
        }
        
        private void loadModel() {
            visionModel = new VisionModel(3, TARGET_SIZE, TARGET_SIZE, featureDim);
            modelLoaded = true;
        }
        
        @Override
        public String getName() { return "CnnFeatureExtractor"; }
        
        @Override
        public boolean isEnabled() { return enabled; }
        
        @Override
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        
        @Override
        public int getFeatureDim() { return featureDim; }
        
        public boolean isModelLoaded() { return modelLoaded; }
        public void reloadModel() { modelLoaded = false; }
        public VisionModel getVisionModel() { return visionModel; }
    }
}