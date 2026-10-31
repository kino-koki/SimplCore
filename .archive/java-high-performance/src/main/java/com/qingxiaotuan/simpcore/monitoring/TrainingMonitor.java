package com.qingxiaotuan.simpcore.monitoring;

import com.qingxiaotuan.simpcore.gpu.GPUDeviceManager;
import com.qingxiaotuan.simpcore.distributed.AutogradDistributedFramework;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class TrainingMonitor {

    public static class MonitorConfig {
        public int sampleIntervalMs = 100;
        public int historySize = 10000;
        public int alertThresholdLoss = 100;
        public float alertThresholdGradNorm = 10.0f;
        public boolean enableConsoleOutput = true;
        public boolean enableFileLogging = false;
        public String logFilePath = "training.log";
        public int consoleOutputInterval = 10;
        public boolean enableAnomalyDetection = true;
        public float anomalyThreshold = 3.0f;
        public boolean enableProgressBar = true;
    }

    public static class MetricSnapshot {
        public final long timestamp;
        public final int globalStep;
        public final float loss;
        public final float gradNorm;
        public final float learningRate;
        public final long stepTimeMs;
        public final float gpuUtilization;
        public final long gpuMemoryUsed;
        public final long gpuMemoryTotal;
        public final float cpuUtilization;
        public final long memoryUsed;
        public final long memoryTotal;
        public final double throughput;
        public final long samplesProcessed;
        public final long epoch;

        public MetricSnapshot(long timestamp, int globalStep, float loss, float gradNorm, float learningRate,
                              long stepTimeMs, float gpuUtilization, long gpuMemoryUsed, long gpuMemoryTotal,
                              float cpuUtilization, long memoryUsed, long memoryTotal, double throughput,
                              long samplesProcessed, long epoch) {
            this.timestamp = timestamp;
            this.globalStep = globalStep;
            this.loss = loss;
            this.gradNorm = gradNorm;
            this.learningRate = learningRate;
            this.stepTimeMs = stepTimeMs;
            this.gpuUtilization = gpuUtilization;
            this.gpuMemoryUsed = gpuMemoryUsed;
            this.gpuMemoryTotal = gpuMemoryTotal;
            this.cpuUtilization = cpuUtilization;
            this.memoryUsed = memoryUsed;
            this.memoryTotal = memoryTotal;
            this.throughput = throughput;
            this.samplesProcessed = samplesProcessed;
            this.epoch = epoch;
        }

        @Override
        public String toString() {
            return String.format(
                "Step=%d | Loss=%.6f | GradNorm=%.4f | LR=%.6f | Time=%.2fms | " +
                "GPU=%.1f%% (%d/%d MB) | CPU=%.1f%% | Memory=%d/%d MB | Throughput=%.2f/s",
                globalStep, loss, gradNorm, learningRate, stepTimeMs,
                gpuUtilization, gpuMemoryUsed / 1024 / 1024, gpuMemoryTotal / 1024 / 1024,
                cpuUtilization, memoryUsed / 1024 / 1024, memoryTotal / 1024 / 1024, throughput
            );
        }
    }

    public static class LossCurve {
        private final Deque<Float> lossHistory;
        private final Deque<Integer> stepHistory;
        private final int maxSize;
        private volatile float currentLoss;
        private volatile float minLoss;
        private volatile float maxLoss;
        private volatile float avgLoss;

        public LossCurve(int maxSize) {
            this.maxSize = maxSize;
            this.lossHistory = new LinkedList<>();
            this.stepHistory = new LinkedList<>();
            this.currentLoss = Float.MAX_VALUE;
            this.minLoss = Float.MAX_VALUE;
            this.maxLoss = Float.MIN_VALUE;
            this.avgLoss = Float.MAX_VALUE;
        }

        public synchronized void add(int step, float loss) {
            lossHistory.addLast(loss);
            stepHistory.addLast(step);
            
            if (lossHistory.size() > maxSize) {
                lossHistory.removeFirst();
                stepHistory.removeFirst();
            }
            
            currentLoss = loss;
            minLoss = Math.min(minLoss, loss);
            maxLoss = Math.max(maxLoss, loss);
            
            double sum = 0;
            for (float l : lossHistory) {
                sum += l;
            }
            avgLoss = (float) (sum / lossHistory.size());
        }

        public synchronized float[] getLossArray() {
            return lossHistory.stream().mapToFloat(Float::floatValue).toArray();
        }

        public synchronized int[] getStepArray() {
            return stepHistory.stream().mapToInt(Integer::intValue).toArray();
        }

        public float getCurrentLoss() { return currentLoss; }
        public float getMinLoss() { return minLoss; }
        public float getMaxLoss() { return maxLoss; }
        public float getAvgLoss() { return avgLoss; }
        public int size() { return lossHistory.size(); }
    }

    public static class GradientNormTracker {
        private final Deque<Float> gradNormHistory;
        private final Deque<Integer> stepHistory;
        private final int maxSize;
        private volatile float currentGradNorm;
        private volatile float avgGradNorm;
        private volatile float maxGradNorm;
        private volatile boolean exploding;

        public GradientNormTracker(int maxSize) {
            this.maxSize = maxSize;
            this.gradNormHistory = new LinkedList<>();
            this.stepHistory = new LinkedList<>();
            this.currentGradNorm = 0;
            this.avgGradNorm = 0;
            this.maxGradNorm = 0;
            this.exploding = false;
        }

        public synchronized void add(int step, float gradNorm) {
            gradNormHistory.addLast(gradNorm);
            stepHistory.addLast(step);
            
            if (gradNormHistory.size() > maxSize) {
                gradNormHistory.removeFirst();
                stepHistory.removeFirst();
            }
            
            currentGradNorm = gradNorm;
            maxGradNorm = Math.max(maxGradNorm, gradNorm);
            
            double sum = 0;
            for (float g : gradNormHistory) {
                sum += g;
            }
            avgGradNorm = (float) (sum / gradNormHistory.size());
            
            exploding = gradNorm > 1000 || (gradNormHistory.size() > 10 && gradNorm > avgGradNorm * 10);
        }

        public float getCurrentGradNorm() { return currentGradNorm; }
        public float getAvgGradNorm() { return avgGradNorm; }
        public float getMaxGradNorm() { return maxGradNorm; }
        public boolean isExploding() { return exploding; }
    }

    public static class HardwareMonitor {
        private final AtomicReference<Float> gpuUtilization = new AtomicReference<>(0f);
        private final AtomicReference<Long> gpuMemoryUsed = new AtomicReference<>(0L);
        private final AtomicReference<Long> gpuMemoryTotal = new AtomicReference<>(0L);
        
        private final AtomicReference<Float> cpuUtilization = new AtomicReference<>(0f);
        private final AtomicReference<Long> memoryUsed = new AtomicReference<>(0L);
        private final AtomicReference<Long> memoryTotal = new AtomicReference<>(0L);
        
        private final AtomicReference<Double> throughput = new AtomicReference<>(0.0);
        private final AtomicLong samplesProcessed = new AtomicLong(0);
        private final AtomicLong lastThroughputTime = new AtomicLong(System.currentTimeMillis());

        public void updateGPU(float utilization, long memoryUsed, long memoryTotal) {
            gpuUtilization.set(utilization);
            this.gpuMemoryUsed.set(memoryUsed);
            this.gpuMemoryTotal.set(memoryTotal);
        }

        public void updateCPU(float utilization, long memoryUsed, long memoryTotal) {
            cpuUtilization.set(utilization);
            this.memoryUsed.set(memoryUsed);
            this.memoryTotal.set(memoryTotal);
        }

        public void updateThroughput(long samples) {
            long now = System.currentTimeMillis();
            long elapsed = now - lastThroughputTime.get();
            
            if (elapsed > 0) {
                throughput.set((samples * 1000.0) / elapsed);
            }
            
            samplesProcessed.addAndGet(samples);
            lastThroughputTime.set(now);
        }

        public float getGpuUtilization() { return gpuUtilization.get(); }
        public long getGpuMemoryUsed() { return gpuMemoryUsed.get(); }
        public long getGpuMemoryTotal() { return gpuMemoryTotal.get(); }
        public float getCpuUtilization() { return cpuUtilization.get(); }
        public long getMemoryUsed() { return memoryUsed.get(); }
        public long getMemoryTotal() { return memoryTotal.get(); }
        public double getThroughput() { return throughput.get(); }
        public long getSamplesProcessed() { return samplesProcessed.get(); }
    }

    private final MonitorConfig config;
    private final LossCurve lossCurve;
    private final GradientNormTracker gradNormTracker;
    private final HardwareMonitor hardwareMonitor;
    
    private final ScheduledExecutorService monitoringPool;
    private final ScheduledExecutorService statsPool;
    
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicLong epoch = new AtomicLong(0);
    private final AtomicLong startTime = new AtomicLong(System.currentTimeMillis());
    private final AtomicLong totalStepTime = new AtomicLong(0);
    
    private volatile boolean running = false;
    private volatile float currentLoss = Float.MAX_VALUE;
    private volatile float currentGradNorm = 0;
    private volatile float currentLearningRate = 0;
    
    private final List<MetricSnapshot> recentSnapshots = Collections.synchronizedList(new ArrayList<>());

    public TrainingMonitor(MonitorConfig config) {
        this.config = config;
        this.lossCurve = new LossCurve(config.historySize);
        this.gradNormTracker = new GradientNormTracker(config.historySize);
        this.hardwareMonitor = new HardwareMonitor();
        
        this.monitoringPool = Executors.newScheduledThreadPool(2);
        this.statsPool = Executors.newScheduledThreadPool(1);
    }

    public void start() {
        running = true;
        startTime.set(System.currentTimeMillis());
        
        monitoringPool.scheduleAtFixedRate(this::sampleHardware, 0, config.sampleIntervalMs, TimeUnit.MILLISECONDS);
        
        statsPool.scheduleAtFixedRate(this::logStats, config.consoleOutputInterval * 1000, 
                                      config.consoleOutputInterval * 1000, TimeUnit.MILLISECONDS);
        
        System.out.println("[Monitor] Training monitor started");
    }

    public void stop() {
        running = false;
        
        monitoringPool.shutdown();
        statsPool.shutdown();
        
        try {
            if (!monitoringPool.awaitTermination(5, TimeUnit.SECONDS)) {
                monitoringPool.shutdownNow();
            }
            if (!statsPool.awaitTermination(5, TimeUnit.SECONDS)) {
                statsPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            monitoringPool.shutdownNow();
            statsPool.shutdownNow();
        }
        
        System.out.println("[Monitor] Training monitor stopped");
    }

    private void sampleHardware() {
        if (!running) return;
        
        if (GPUDeviceManager.isGPUAvailable()) {
            GPUDeviceManager.DeviceInfo device = GPUDeviceManager.getSelectedDevice();
            if (device != null) {
                hardwareMonitor.updateGPU(0, device.memoryUsed, device.memoryTotal);
            }
        }
        
        try {
            com.sun.management.OperatingSystemMXBean osBean = 
                (com.sun.management.OperatingSystemMXBean) java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            
            float cpuUtil = (float) osBean.getProcessCpuLoad() * 100;
            long memUsed = osBean.getTotalPhysicalMemorySize() - osBean.getFreePhysicalMemorySize();
            long memTotal = osBean.getTotalPhysicalMemorySize();
            
            hardwareMonitor.updateCPU(cpuUtil, memUsed, memTotal);
        } catch (Exception e) {
            // Ignore
        }
    }

    public synchronized void recordStep(float loss, float gradNorm, float learningRate, long stepTimeMs) {
        int step = globalStep.incrementAndGet();
        totalStepTime.addAndGet(stepTimeMs);
        
        currentLoss = loss;
        currentGradNorm = gradNorm;
        currentLearningRate = learningRate;
        
        lossCurve.add(step, loss);
        gradNormTracker.add(step, gradNorm);
        
        MetricSnapshot snapshot = new MetricSnapshot(
            System.currentTimeMillis(),
            step,
            loss,
            gradNorm,
            learningRate,
            stepTimeMs,
            hardwareMonitor.getGpuUtilization(),
            hardwareMonitor.getGpuMemoryUsed(),
            hardwareMonitor.getGpuMemoryTotal(),
            hardwareMonitor.getCpuUtilization(),
            hardwareMonitor.getMemoryUsed(),
            hardwareMonitor.getMemoryTotal(),
            hardwareMonitor.getThroughput(),
            hardwareMonitor.getSamplesProcessed(),
            epoch.get()
        );
        
        recentSnapshots.add(snapshot);
        if (recentSnapshots.size() > 100) {
            recentSnapshots.remove(0);
        }
        
        if (step % config.consoleOutputInterval == 0 && config.enableConsoleOutput) {
            System.out.println(snapshot.toString());
        }
        
        if (config.enableAnomalyDetection) {
            checkAnomalies(snapshot);
        }
    }

    public void recordEpoch(long epochNum) {
        epoch.set(epochNum);
    }

    public void recordSamples(long samples) {
        hardwareMonitor.updateThroughput(samples);
    }

    private void checkAnomalies(MetricSnapshot snapshot) {
        if (snapshot.loss > config.alertThresholdLoss) {
            System.err.println("[Alert] Loss exceeded threshold: " + snapshot.loss);
        }
        
        if (snapshot.gradNorm > config.alertThresholdGradNorm) {
            System.err.println("[Alert] Gradient norm exceeded threshold: " + snapshot.gradNorm);
        }
        
        if (gradNormTracker.isExploding()) {
            System.err.println("[Alert] Potential gradient explosion detected!");
        }
    }

    private void logStats() {
        if (!running) return;
        
        long elapsed = System.currentTimeMillis() - startTime.get();
        int steps = globalStep.get();
        
        double avgStepTime = steps > 0 ? (totalStepTime.get() / (double) steps) : 0;
        double throughput = steps > 0 ? (steps * 1000.0 / elapsed) : 0;
        
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("=").append("=".repeat(70)).append("\n");
        sb.append("              TRAINING STATISTICS SUMMARY\n");
        sb.append("=").append("=".repeat(70)).append("\n");
        sb.append(String.format("  Epoch: %d | Global Step: %d\n", epoch.get(), steps));
        sb.append(String.format("  Elapsed Time: %s\n", formatTime(elapsed)));
        sb.append(String.format("  Current Loss: %.6f\n", currentLoss));
        sb.append(String.format("  Average Loss (recent): %.6f\n", lossCurve.getAvgLoss()));
        sb.append(String.format("  Min Loss: %.6f\n", lossCurve.getMinLoss()));
        sb.append(String.format("  Current Gradient Norm: %.4f\n", currentGradNorm));
        sb.append(String.format("  Average Gradient Norm: %.4f\n", gradNormTracker.getAvgGradNorm()));
        sb.append(String.format("  Learning Rate: %.6f\n", currentLearningRate));
        sb.append(String.format("  Average Step Time: %.2f ms\n", avgStepTime));
        sb.append(String.format("  Throughput: %.2f steps/s\n", throughput));
        sb.append("\n");
        sb.append("  Hardware Utilization:\n");
        sb.append(String.format("    GPU: %.1f%% | Memory: %d/%d MB\n", 
            hardwareMonitor.getGpuUtilization(),
            hardwareMonitor.getGpuMemoryUsed() / 1024 / 1024,
            hardwareMonitor.getGpuMemoryTotal() / 1024 / 1024));
        sb.append(String.format("    CPU: %.1f%% | Memory: %d/%d MB\n", 
            hardwareMonitor.getCpuUtilization(),
            hardwareMonitor.getMemoryUsed() / 1024 / 1024,
            hardwareMonitor.getMemoryTotal() / 1024 / 1024));
        sb.append("=").append("=".repeat(70)).append("\n");
        
        System.out.println(sb.toString());
    }

    private String formatTime(long ms) {
        long seconds = ms / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        
        if (hours > 0) {
            return String.format("%d h %d m %d s", hours, minutes % 60, seconds % 60);
        } else if (minutes > 0) {
            return String.format("%d m %d s", minutes, seconds % 60);
        } else {
            return String.format("%d s", seconds);
        }
    }

    public void printProgressBar(int currentStep, int totalSteps, int epoch, int totalEpochs) {
        if (!config.enableProgressBar) return;
        
        int barWidth = 50;
        float progress = (float) currentStep / totalSteps;
        int filledWidth = (int) (progress * barWidth);
        
        StringBuilder bar = new StringBuilder();
        bar.append("\r[Epoch ").append(epoch + 1).append("/").append(totalEpochs).append("] ");
        
        bar.append("[");
        for (int i = 0; i < barWidth; i++) {
            if (i < filledWidth) {
                bar.append("=");
            } else if (i == filledWidth) {
                bar.append(">");
            } else {
                bar.append(" ");
            }
        }
        bar.append("] ");
        
        bar.append(String.format("%.1f%%", progress * 100));
        bar.append(String.format(" | Loss: %.4f", currentLoss));
        bar.append(String.format(" | GradNorm: %.3f", currentGradNorm));
        
        System.out.print(bar.toString());
    }

    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("# Training Report\n\n");
        sb.append("## Overview\n\n");
        sb.append(String.format("- Total Steps: %d\n", globalStep.get()));
        sb.append(String.format("- Total Epochs: %d\n", epoch.get()));
        sb.append(String.format("- Total Time: %s\n", formatTime(System.currentTimeMillis() - startTime.get())));
        sb.append(String.format("- Samples Processed: %d\n", hardwareMonitor.getSamplesProcessed()));
        
        sb.append("\n## Loss Statistics\n\n");
        sb.append(String.format("- Final Loss: %.6f\n", currentLoss));
        sb.append(String.format("- Average Loss: %.6f\n", lossCurve.getAvgLoss()));
        sb.append(String.format("- Minimum Loss: %.6f\n", lossCurve.getMinLoss()));
        sb.append(String.format("- Maximum Loss: %.6f\n", lossCurve.getMaxLoss()));
        
        sb.append("\n## Gradient Statistics\n\n");
        sb.append(String.format("- Final Gradient Norm: %.4f\n", currentGradNorm));
        sb.append(String.format("- Average Gradient Norm: %.4f\n", gradNormTracker.getAvgGradNorm()));
        sb.append(String.format("- Maximum Gradient Norm: %.4f\n", gradNormTracker.getMaxGradNorm()));
        sb.append(String.format("- Exploding Detected: %s\n", gradNormTracker.isExploding() ? "Yes" : "No"));
        
        sb.append("\n## Hardware Performance\n\n");
        sb.append(String.format("- GPU Utilization: %.1f%%\n", hardwareMonitor.getGpuUtilization()));
        sb.append(String.format("- GPU Memory Used: %d MB / %d MB\n", 
            hardwareMonitor.getGpuMemoryUsed() / 1024 / 1024,
            hardwareMonitor.getGpuMemoryTotal() / 1024 / 1024));
        sb.append(String.format("- CPU Utilization: %.1f%%\n", hardwareMonitor.getCpuUtilization()));
        sb.append(String.format("- System Memory Used: %d MB / %d MB\n", 
            hardwareMonitor.getMemoryUsed() / 1024 / 1024,
            hardwareMonitor.getMemoryTotal() / 1024 / 1024));
        
        sb.append("\n## Throughput\n\n");
        sb.append(String.format("- Current Throughput: %.2f samples/s\n", hardwareMonitor.getThroughput()));
        
        sb.append("\n## Loss Curve Data (Last 100 Points)\n\n");
        float[] losses = lossCurve.getLossArray();
        int[] steps = lossCurve.getStepArray();
        
        sb.append("| Step | Loss |\n");
        sb.append("|------|------|\n");
        for (int i = Math.max(0, losses.length - 100); i < losses.length; i++) {
            sb.append(String.format("| %d | %.6f |\n", steps[i], losses[i]));
        }
        
        return sb.toString();
    }

    public void exportReport(String filePath) {
        try {
            String report = generateReport();
            java.io.PrintWriter writer = new java.io.PrintWriter(filePath);
            writer.write(report);
            writer.close();
            System.out.println("[Monitor] Report exported to: " + filePath);
        } catch (Exception e) {
            System.err.println("[Monitor] Failed to export report: " + e.getMessage());
        }
    }

    public LossCurve getLossCurve() { return lossCurve; }
    public GradientNormTracker getGradNormTracker() { return gradNormTracker; }
    public HardwareMonitor getHardwareMonitor() { return hardwareMonitor; }
    public int getGlobalStep() { return globalStep.get(); }
    public long getEpoch() { return epoch.get(); }
    public float getCurrentLoss() { return currentLoss; }
    public float getCurrentGradNorm() { return currentGradNorm; }
    
    public List<MetricSnapshot> getRecentSnapshots() {
        return Collections.unmodifiableList(recentSnapshots);
    }

    public void integrateWithDistributedFramework(AutogradDistributedFramework framework) {
        monitoringPool.scheduleAtFixedRate(() -> {
            if (!running) return;
            
            AutogradDistributedFramework.TrainingMetrics metrics = framework.getMetrics();
            
            if (metrics != null) {
                currentLoss = metrics.currentLoss;
                currentGradNorm = metrics.currentGradNorm;
                globalStep.set(metrics.globalStep);
                
                lossCurve.add(metrics.globalStep, metrics.currentLoss);
                gradNormTracker.add(metrics.globalStep, metrics.currentGradNorm);
                hardwareMonitor.updateThroughput(metrics.samplesProcessed - hardwareMonitor.getSamplesProcessed());
            }
        }, 1, 1, TimeUnit.SECONDS);
        
        System.out.println("[Monitor] Integrated with distributed framework");
    }
}