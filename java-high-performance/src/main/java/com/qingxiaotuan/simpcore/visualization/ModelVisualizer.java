package com.qingxiaotuan.simpcore.visualization;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.training.TrainingLoop;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class ModelVisualizer {

    private final SimpCoreModel model;
    private final ConcurrentHashMap<String, List<Float>> metrics = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, float[][]> layerActivations = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, float[]> attentionMaps = new ConcurrentHashMap<>();
    private boolean enabled = true;

    public ModelVisualizer(SimpCoreModel model) {
        this.model = model;
        initMetrics();
    }

    private void initMetrics() {
        metrics.put("loss", new ArrayList<>());
        metrics.put("learningRate", new ArrayList<>());
        metrics.put("accuracy", new ArrayList<>());
        metrics.put("gradientNorm", new ArrayList<>());
    }

    public void recordMetric(String name, float value) {
        if (!enabled) return;
        metrics.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
    }

    public void recordLayerActivation(int layerIndex, float[][] activations) {
        if (!enabled) return;
        String key = "layer_" + layerIndex;
        layerActivations.put(key, activations);
    }

    public void recordAttentionMap(int layerIndex, float[] attention) {
        if (!enabled) return;
        String key = "attention_" + layerIndex;
        attentionMaps.put(key, attention);
    }

    public void visualizeTraining(TrainingLoop trainer, String[] texts, int iterations) {
        System.out.println("\n=== Training Visualization ===");
        System.out.println("Model: SimpCore");
        System.out.println("Layers: " + model.getNLayers());
        System.out.println("dModel: " + model.getDModel());
        System.out.println("VocabSize: " + model.getVocabSize());
        System.out.println("=============================\n");

        for (int iter = 0; iter < iterations; iter++) {
            String text = texts[iter % texts.length];
            float loss = trainer.trainStep(text, 1);
            
            recordMetric("loss", loss);
            
            if (iter % 10 == 0) {
                printProgress(iter, iterations, loss);
            }
            
            if (iter % 100 == 0) {
                printMetricsSummary();
            }
        }
        
        printFinalReport();
    }

    private void printProgress(int iteration, int total, float loss) {
        int progress = (iteration * 50) / total;
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < 50; i++) {
            bar.append(i < progress ? "=" : " ");
        }
        System.out.printf("\r[%s] Iteration %d/%d | Loss: %.6f", 
            bar.toString(), iteration, total, loss);
        System.out.flush();
    }

    private void printMetricsSummary() {
        System.out.println("\n\n--- Metrics Summary ---");
        
        for (String name : metrics.keySet()) {
            List<Float> values = metrics.get(name);
            if (values.isEmpty()) continue;
            
            float sum = 0;
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            
            for (float v : values) {
                sum += v;
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            
            float avg = sum / values.size();
            float last = values.get(values.size() - 1);
            
            System.out.printf("%-20s | Last: %.6f | Avg: %.6f | Min: %.6f | Max: %.6f%n",
                name, last, avg, min, max);
        }
        System.out.println("----------------------");
    }

    private void printFinalReport() {
        System.out.println("\n\n=== Training Complete ===");
        printMetricsSummary();
        System.out.println("\n--- Model Architecture ---");
        
        System.out.println("Total Parameters: " + model.getTotalParameters());
        System.out.println("Embedding: " + model.getVocabSize() + " x " + model.getDModel());
        
        for (int l = 0; l < model.getNLayers(); l++) {
            System.out.printf("Layer %d: ResonanceField + DynamicMemory + TopologyRouter + CausalIntervention + InfoBottleneck + FFN + MetaDistiller + SpontaneousPlasticity%n", l);
        }
        
        System.out.println("Output: " + model.getDModel() + " x " + model.getVocabSize());
        System.out.println("==========================");
    }

    public String generateHTMLReport() {
        StringBuilder html = new StringBuilder();
        
        html.append("<!DOCTYPE html>\n");
        html.append("<html lang=\"zh-CN\">\n");
        html.append("<head>\n");
        html.append("    <meta charset=\"UTF-8\">\n");
        html.append("    <title>SimpCore Training Report</title>\n");
        html.append("    <style>\n");
        html.append("        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }\n");
        html.append("        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }\n");
        html.append("        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }\n");
        html.append("        h2 { color: #555; margin-top: 30px; }\n");
        html.append("        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }\n");
        html.append("        .metric-card { background: #f9f9f9; padding: 15px; border-radius: 6px; }\n");
        html.append("        .metric-card h3 { margin-top: 0; color: #4CAF50; }\n");
        html.append("        .progress-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }\n");
        html.append("        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }\n");
        html.append("        table { width: 100%; border-collapse: collapse; margin-top: 10px; }\n");
        html.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.append("        th { background: #f2f2f2; }\n");
        html.append("        .loss-chart { height: 300px; border: 1px solid #ddd; margin-top: 10px; position: relative; }\n");
        html.append("    </style>\n");
        html.append("</head>\n");
        html.append("<body>\n");
        html.append("<div class=\"container\">\n");
        html.append("    <h1>SimpCore Training Report</h1>\n");
        
        html.append("    <h2>Model Information</h2>\n");
        html.append("    <table>\n");
        html.append("        <tr><th>Property</th><th>Value</th></tr>\n");
        html.append("        <tr><td>Vocabulary Size</td><td>" + model.getVocabSize() + "</td></tr>\n");
        html.append("        <tr><td>Model Dimension</td><td>" + model.getDModel() + "</td></tr>\n");
        html.append("        <tr><td>Number of Layers</td><td>" + model.getNLayers() + "</td></tr>\n");
        html.append("        <tr><td>Max Sequence Length</td><td>" + model.getMaxSeqLen() + "</td></tr>\n");
        html.append("        <tr><td>Total Parameters</td><td>" + model.getTotalParameters() + "</td></tr>\n");
        html.append("    </table>\n");
        
        html.append("    <h2>Training Metrics</h2>\n");
        html.append("    <div class=\"metrics\">\n");
        
        for (String name : metrics.keySet()) {
            List<Float> values = metrics.get(name);
            if (values.isEmpty()) continue;
            
            float last = values.get(values.size() - 1);
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            for (float v : values) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            
            html.append("        <div class=\"metric-card\">\n");
            html.append("            <h3>" + name + "</h3>\n");
            html.append("            <p>Last: " + String.format("%.6f", last) + "</p>\n");
            html.append("            <p>Min: " + String.format("%.6f", min) + "</p>\n");
            html.append("            <p>Max: " + String.format("%.6f", max) + "</p>\n");
            html.append("            <div class=\"progress-bar\">\n");
            html.append("                <div class=\"progress-fill\" style=\"width: " + ((last - min) / (max - min) * 100) + "%\"></div>\n");
            html.append("            </div>\n");
            html.append("        </div>\n");
        }
        
        html.append("    </div>\n");
        
        if (metrics.containsKey("loss") && !metrics.get("loss").isEmpty()) {
            html.append("    <h2>Loss Curve</h2>\n");
            html.append("    <div class=\"loss-chart\">\n");
            html.append("        <svg width=\"100%\" height=\"100%\">\n");
            
            List<Float> losses = metrics.get("loss");
            float minLoss = Float.MAX_VALUE;
            float maxLoss = Float.MIN_VALUE;
            for (float l : losses) {
                minLoss = Math.min(minLoss, l);
                maxLoss = Math.max(maxLoss, l);
            }
            
            float range = maxLoss - minLoss;
            if (range == 0) range = 1;
            
            html.append("            <polyline fill=\"none\" stroke=\"#4CAF50\" stroke-width=\"2\" points=\"");
            for (int i = 0; i < losses.size(); i++) {
                float x = (i * 100.0f) / (losses.size() - 1);
                float y = 100 - ((losses.get(i) - minLoss) / range * 90 + 5);
                html.append(x + "," + y + " ");
            }
            html.append("\"/>\n");
            html.append("        </svg>\n");
            html.append("    </div>\n");
        }
        
        html.append("</div>\n");
        html.append("</body>\n");
        html.append("</html>");
        
        return html.toString();
    }

    public void saveHTMLReport(String filePath) {
        String html = generateHTMLReport();
        try (java.io.PrintWriter writer = new java.io.PrintWriter(filePath)) {
            writer.write(html);
        } catch (java.io.IOException e) {
            System.err.println("Failed to save HTML report: " + e.getMessage());
        }
    }

    public List<Float> getMetric(String name) {
        return metrics.getOrDefault(name, new ArrayList<>());
    }

    public float[][] getLayerActivation(int layerIndex) {
        return layerActivations.get("layer_" + layerIndex);
    }

    public float[] getAttentionMap(int layerIndex) {
        return attentionMaps.get("attention_" + layerIndex);
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void clearMetrics() {
        metrics.clear();
        initMetrics();
    }
}