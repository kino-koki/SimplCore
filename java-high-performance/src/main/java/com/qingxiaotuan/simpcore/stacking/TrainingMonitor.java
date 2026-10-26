package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.evaluation.BenchmarkEngine;
import com.qingxiaotuan.simpcore.evaluation.ModelEvaluator;
import com.qingxiaotuan.simpcore.inference.InferenceEngine;
import com.qingxiaotuan.simpcore.inference.AntiHallucinationEngine;

import java.util.*;
import java.util.concurrent.*;

/**
 * 训练监控器
 * 功能：
 * 1. 实时监控训练进度
 * 2. 计算剩余时间
 * 3. 训练完成后自动启动跑分对比
 */
public class TrainingMonitor {
    
    private final StackedTrainer trainer;
    private final StackedTrainer smallModelTrainer; // 20个小模型对比
    private final ExecutorService monitorExecutor;
    private volatile boolean trainingComplete;
    private volatile boolean benchmarkDone;
    
    private long startTime;
    private long lastProgressTime;
    private int lastStep;
    private double lastLoss;
    
    public TrainingMonitor(StackedTrainer trainer) {
        this.trainer = trainer;
        this.smallModelTrainer = null;
        this.monitorExecutor = Executors.newSingleThreadExecutor();
        this.trainingComplete = false;
        this.benchmarkDone = false;
    }
    
    public TrainingMonitor(StackedTrainer trainer, StackedTrainer smallModelTrainer) {
        this.trainer = trainer;
        this.smallModelTrainer = smallModelTrainer;
        this.monitorExecutor = Executors.newSingleThreadExecutor();
        this.trainingComplete = false;
        this.benchmarkDone = false;
    }
    
    /**
     * 开始监控
     */
    public void startMonitoring() {
        startTime = System.currentTimeMillis();
        lastProgressTime = startTime;
        lastStep = 0;
        lastLoss = Double.MAX_VALUE;
        
        monitorExecutor.submit(() -> {
            while (!trainingComplete) {
                try {
                    Thread.sleep(5000); // 每5秒检查一次
                    printProgress();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            
            // 训练完成，自动启动跑分
            System.out.println("\n=========================================");
            System.out.println("训练完成！开始跑分对比...");
            System.out.println("=========================================\n");
            
            runBenchmarkComparison();
            benchmarkDone = true;
        });
    }
    
    /**
     * 打印进度和预计剩余时间
     */
    private void printProgress() {
        StackedTrainer.TrainingStats stats = trainer.getStats();
        
        if (stats.totalSteps == lastStep && stats.averageLoss == lastLoss) {
            return; // 没有变化，跳过打印
        }
        
        long currentTime = System.currentTimeMillis();
        long elapsed = currentTime - startTime;
        long timeSinceLast = currentTime - lastProgressTime;
        
        // 计算速度（步/秒）
        double stepsPerSecond = (stats.totalSteps - lastStep) / (timeSinceLast / 1000.0);
        if (stepsPerSecond <= 0) stepsPerSecond = 0.1; // 避免除零
        
        // 计算剩余步数和时间
        int totalSteps = 3000; // 预计训练步数
        int remainingSteps = totalSteps - (int) stats.totalSteps;
        double remainingSeconds = remainingSteps / stepsPerSecond;
        
        // 格式化时间
        String remainingTime = formatTime((long) remainingSeconds);
        String elapsedTime = formatTime(elapsed / 1000);
        
        // 计算完成百分比
        double progress = (stats.totalSteps * 100.0) / totalSteps;
        
        // 打印进度条
        System.out.printf("\r[监控] 进度: %.1f%% | 步数: %,d/%d | 损失: %.6f | 速度: %.2f步/秒 | 已用: %s | 剩余: %s",
            progress, stats.totalSteps, totalSteps, stats.averageLoss, 
            stepsPerSecond, elapsedTime, remainingTime);
        System.out.flush();
        
        lastStep = (int) stats.totalSteps;
        lastLoss = stats.averageLoss;
        lastProgressTime = currentTime;
        
        // 如果损失变化明显，额外打印一行
        if (Math.abs(stats.averageLoss - lastLoss) > 0.001) {
            System.out.println();
        }
    }
    
    /**
     * 格式化时间
     */
    private String formatTime(long seconds) {
        if (seconds < 60) {
            return seconds + "秒";
        } else if (seconds < 3600) {
            long minutes = seconds / 60;
            long secs = seconds % 60;
            return minutes + "分" + secs + "秒";
        } else {
            long hours = seconds / 3600;
            long minutes = (seconds % 3600) / 60;
            long secs = seconds % 60;
            return hours + "时" + minutes + "分" + secs + "秒";
        }
    }
    
    /**
     * 标记训练完成
     */
    public void markTrainingComplete() {
        this.trainingComplete = true;
    }
    
    /**
     * 等待监控和跑分完成
     */
    public void waitForCompletion() throws InterruptedException {
        while (!benchmarkDone) {
            Thread.sleep(1000);
        }
        monitorExecutor.shutdown();
    }
    
    /**
     * 运行跑分对比（青小团-7B vs 20个小模型）
     */
    public void runBenchmarkComparison() {
        System.out.println("\n=== 青小团-7B 参数堆叠模型 ===");
        System.out.println("专家数量: " + trainer.getExperts().getNExperts());
        System.out.println("总参数: ~" + (trainer.getExperts().getNExperts() * 0.1) + "B");
        System.out.println();
        
        // 创建评估器
        ModelEvaluator evaluator = new ModelEvaluator();
        
        // 评估青小团-7B
        System.out.println("[1/2] 正在评估青小团-7B...");
        ModelEvaluator.BenchmarkResult qxtResult = evaluator.runFullBenchmark(trainer);
        System.out.println(evaluator.generateReport(qxtResult));
        
        // 如果有小模型训练器，也进行评估
        if (smallModelTrainer != null) {
            System.out.println("\n=== 20个小模型（对比组） ===");
            System.out.println("专家数量: " + smallModelTrainer.getExperts().getNExperts());
            System.out.println("总参数: ~" + (smallModelTrainer.getExperts().getNExperts() * 0.1) + "B");
            System.out.println();
            
            System.out.println("[2/2] 正在评估20个小模型...");
            ModelEvaluator.BenchmarkResult smallResult = evaluator.runFullBenchmark(smallModelTrainer);
            System.out.println(evaluator.generateReport(smallResult));
            
            // 生成对比报告
            generateComparisonReport(qxtResult, smallResult);
        }
        
        evaluator.shutdown();
    }
    
    /**
     * 生成对比报告
     */
    private void generateComparisonReport(ModelEvaluator.BenchmarkResult qxtResult, 
                                          ModelEvaluator.BenchmarkResult smallResult) {
        System.out.println("\n=========================================");
        System.out.println("          青小团-7B vs 20个小模型");
        System.out.println("=========================================\n");
        
        System.out.println("| 数据集 | 青小团-7B | 20个小模型 | 差值 |");
        System.out.println("|--------|-----------|-------------|------|");
        
        float qxtTotal = 0, smallTotal = 0;
        int count = 0;
        
        for (ModelEvaluator.EvaluationResult qxtEval : qxtResult.results) {
            for (ModelEvaluator.EvaluationResult smallEval : smallResult.results) {
                if (qxtEval.dataset.equals(smallEval.dataset)) {
                    float diff = qxtEval.accuracy - smallEval.accuracy;
                    System.out.printf("| %-8s | %7.2f%% | %9.2f%% | %+6.2f%% |\n",
                        qxtEval.dataset, qxtEval.accuracy * 100, 
                        smallEval.accuracy * 100, diff * 100);
                    qxtTotal += qxtEval.accuracy;
                    smallTotal += smallEval.accuracy;
                    count++;
                    break;
                }
            }
        }
        
        float qxtAvg = count > 0 ? qxtTotal / count : 0;
        float smallAvg = count > 0 ? smallTotal / count : 0;
        float avgDiff = qxtAvg - smallAvg;
        
        System.out.println("|--------|-----------|-------------|------|");
        System.out.printf("| 平均分 | %7.2f%% | %9.2f%% | %+6.2f%% |\n",
            qxtAvg * 100, smallAvg * 100, avgDiff * 100);
        
        System.out.println("\n结论：");
        if (avgDiff > 0.05) {
            System.out.println("  ✓ 青小团-7B在综合性能上显著优于20个小模型！");
        } else if (avgDiff > 0) {
            System.out.println("  ~ 青小团-7B略优于20个小模型");
        } else {
            System.out.println("  ✗ 青小团-7B与20个小模型相当或略差");
        }
        
        // 参数效率对比
        double qxtParams = trainer.getExperts().getNExperts() * 0.1; // B
        double smallParams = smallModelTrainer.getExperts().getNExperts() * 0.1; // B
        double paramsRatio = qxtParams / smallParams;
        
        System.out.println("\n参数效率分析：");
        System.out.printf("  青小团-7B参数: %.1fB | 20个小模型参数: %.1fB | 比例: %.1f倍\n",
            qxtParams, smallParams, paramsRatio);
        System.out.printf("  每B参数得分: 青小团=%.2f%%/B | 小模型=%.2f%%/B | 效率提升: %.1f倍\n",
            qxtAvg * 100 / qxtParams, smallAvg * 100 / smallParams,
            (qxtAvg / qxtParams) / (smallAvg / smallParams));
    }
    
    /**
     * 获取当前状态
     */
    public String getStatus() {
        return String.format("TrainingMonitor: training=%s, benchmark=%s", 
            !trainingComplete, benchmarkDone);
    }
}