package com.qingxiaotuan.simpcore.monitoring;

import java.util.*;
import java.util.concurrent.*;
import java.lang.management.*;

public class Dashboard {
    
    public static class Metric {
        public final String name;
        public final double value;
        public final String unit;
        public final long timestamp;
        
        public Metric(String name, double value, String unit) {
            this.name = name;
            this.value = value;
            this.unit = unit;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    public static class TrainingMetrics {
        public double loss;
        public double accuracy;
        public double learningRate;
        public int epoch;
        public long tokensProcessed;
        public double gradientNorm;
    }
    
    private final ConcurrentHashMap<String, List<Metric>> metricsHistory;
    private final ScheduledExecutorService scheduler;
    private final TrainingMetrics trainingMetrics;
    private final MemoryMXBean memoryBean;
    private final OperatingSystemMXBean osBean;
    private volatile boolean enabled;
    private final int historySize;
    
    public Dashboard() {
        this.metricsHistory = new ConcurrentHashMap<>();
        this.scheduler = Executors.newScheduledThreadPool(2);
        this.trainingMetrics = new TrainingMetrics();
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.enabled = true;
        this.historySize = 100;
        startMonitoring();
    }
    
    private void startMonitoring() {
        scheduler.scheduleAtFixedRate(() -> {
            if (enabled) {
                collectSystemMetrics();
            }
        }, 0, 5, TimeUnit.SECONDS);
    }
    
    private void collectSystemMetrics() {
        addMetric("memory.used", memoryBean.getHeapMemoryUsage().getUsed() / (1024.0 * 1024.0), "MB");
        addMetric("memory.max", memoryBean.getHeapMemoryUsage().getMax() / (1024.0 * 1024.0), "MB");
        addMetric("cpu.load", osBean.getSystemLoadAverage(), "%");
        addMetric("process.cpu", getProcessCpuLoad(), "%");
    }
    
    private double getProcessCpuLoad() {
        try {
            com.sun.management.OperatingSystemMXBean bean = 
                (com.sun.management.OperatingSystemMXBean) osBean;
            return bean.getProcessCpuLoad() * 100;
        } catch (Exception e) {
            return 0;
        }
    }
    
    public void addMetric(String name, double value, String unit) {
        metricsHistory.computeIfAbsent(name, k -> new ArrayList<>());
        
        List<Metric> history = metricsHistory.get(name);
        synchronized (history) {
            history.add(new Metric(name, value, unit));
            while (history.size() > historySize) {
                history.remove(0);
            }
        }
    }
    
    public void updateTrainingMetrics(double loss, double accuracy, double learningRate, 
                                      int epoch, long tokensProcessed, double gradientNorm) {
        trainingMetrics.loss = loss;
        trainingMetrics.accuracy = accuracy;
        trainingMetrics.learningRate = learningRate;
        trainingMetrics.epoch = epoch;
        trainingMetrics.tokensProcessed = tokensProcessed;
        trainingMetrics.gradientNorm = gradientNorm;
        
        addMetric("training.loss", loss, "");
        addMetric("training.accuracy", accuracy * 100, "%");
        addMetric("training.lr", learningRate, "");
        addMetric("training.gradientNorm", gradientNorm, "");
    }
    
    public List<Metric> getMetricHistory(String name) {
        return metricsHistory.getOrDefault(name, Collections.emptyList());
    }
    
    public TrainingMetrics getTrainingMetrics() {
        return trainingMetrics;
    }
    
    public Map<String, Double> getCurrentMetrics() {
        Map<String, Double> result = new HashMap<>();
        
        for (Map.Entry<String, List<Metric>> entry : metricsHistory.entrySet()) {
            List<Metric> history = entry.getValue();
            if (!history.isEmpty()) {
                result.put(entry.getKey(), history.get(history.size() - 1).value);
            }
        }
        
        return result;
    }
    
    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("========================================\n");
        sb.append("          SC Architecture Dashboard\n");
        sb.append("========================================\n\n");
        
        sb.append("--- System Metrics ---\n");
        sb.append(String.format("Memory Used: %.2f MB / %.2f MB\n",
            getMetricValue("memory.used"), getMetricValue("memory.max")));
        sb.append(String.format("CPU Load: %.2f%%\n", getMetricValue("cpu.load")));
        sb.append(String.format("Process CPU: %.2f%%\n", getMetricValue("process.cpu")));
        sb.append("\n");
        
        sb.append("--- Training Metrics ---\n");
        sb.append(String.format("Epoch: %d\n", trainingMetrics.epoch));
        sb.append(String.format("Loss: %.6f\n", trainingMetrics.loss));
        sb.append(String.format("Accuracy: %.4f%%\n", trainingMetrics.accuracy * 100));
        sb.append(String.format("Learning Rate: %.2e\n", trainingMetrics.learningRate));
        sb.append(String.format("Tokens Processed: %,d\n", trainingMetrics.tokensProcessed));
        sb.append(String.format("Gradient Norm: %.4f\n", trainingMetrics.gradientNorm));
        sb.append("\n");
        
        sb.append("--- Status ---\n");
        sb.append(String.format("Enabled: %s\n", enabled));
        sb.append(String.format("Metrics tracked: %d\n", metricsHistory.size()));
        sb.append(String.format("Timestamp: %s\n", new Date()));
        
        return sb.toString();
    }
    
    private double getMetricValue(String name) {
        List<Metric> history = metricsHistory.get(name);
        return history != null && !history.isEmpty() ? history.get(history.size() - 1).value : 0;
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public void shutdown() {
        scheduler.shutdown();
    }
    
    public String getStatus() {
        return String.format(
            "Dashboard: enabled=%s, metrics=%d, training_epoch=%d",
            enabled, metricsHistory.size(), trainingMetrics.epoch
        );
    }
}