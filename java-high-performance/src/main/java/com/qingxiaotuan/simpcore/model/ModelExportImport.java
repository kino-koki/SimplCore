package com.qingxiaotuan.simpcore.model;

import java.io.*;
import java.util.*;

public class ModelExportImport {
    
    public static class ModelData {
        String modelName;
        String architecture;
        int dModel;
        int nLayers;
        int nHeads;
        int vocabSize;
        long parameterCount;
        Map<String, float[]> weights;
        Map<String, float[]> biases;
        Map<String, Object> metadata;
        
        ModelData() {
            weights = new HashMap<>();
            biases = new HashMap<>();
            metadata = new HashMap<>();
        }
    }
    
    public static class ExportOptions {
        boolean includeWeights = true;
        boolean includeBiases = true;
        boolean includeMetadata = true;
        String format = "binary";
        float quantize = 0;
        
        public ExportOptions includeWeights(boolean val) { includeWeights = val; return this; }
        public ExportOptions includeBiases(boolean val) { includeBiases = val; return this; }
        public ExportOptions includeMetadata(boolean val) { includeMetadata = val; return this; }
        public ExportOptions format(String val) { format = val; return this; }
        public ExportOptions quantize(float bits) { quantize = bits; return this; }
    }
    
    public static boolean exportModel(ModelData model, String path, ExportOptions options) {
        try {
            FileOutputStream fos = new FileOutputStream(path);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);
            
            dos.writeUTF("SC_MODEL");
            dos.writeUTF(model.modelName);
            dos.writeUTF(model.architecture);
            dos.writeInt(model.dModel);
            dos.writeInt(model.nLayers);
            dos.writeInt(model.nHeads);
            dos.writeInt(model.vocabSize);
            dos.writeLong(model.parameterCount);
            
            if (options.includeMetadata) {
                dos.writeInt(model.metadata.size());
                for (Map.Entry<String, Object> entry : model.metadata.entrySet()) {
                    dos.writeUTF(entry.getKey());
                    String valueStr = entry.getValue().toString();
                    dos.writeUTF(valueStr);
                }
            } else {
                dos.writeInt(0);
            }
            
            if (options.includeWeights) {
                dos.writeInt(model.weights.size());
                for (Map.Entry<String, float[]> entry : model.weights.entrySet()) {
                    dos.writeUTF(entry.getKey());
                    float[] w = entry.getValue();
                    dos.writeInt(w.length);
                    for (float v : w) {
                        dos.writeFloat(v);
                    }
                }
            } else {
                dos.writeInt(0);
            }
            
            if (options.includeBiases) {
                dos.writeInt(model.biases.size());
                for (Map.Entry<String, float[]> entry : model.biases.entrySet()) {
                    dos.writeUTF(entry.getKey());
                    float[] b = entry.getValue();
                    dos.writeInt(b.length);
                    for (float v : b) {
                        dos.writeFloat(v);
                    }
                }
            } else {
                dos.writeInt(0);
            }
            
            dos.flush();
            dos.close();
            
            File file = new File(path);
            System.out.printf("Model exported successfully to %s (%.2f MB)%n", 
                path, file.length() / (1024.0 * 1024.0));
            return true;
            
        } catch (IOException e) {
            System.err.println("Failed to export model: " + e.getMessage());
            return false;
        }
    }
    
    public static ModelData importModel(String path) {
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);
            
            String magic = dis.readUTF();
            if (!"SC_MODEL".equals(magic)) {
                System.err.println("Invalid model file format");
                return null;
            }
            
            ModelData model = new ModelData();
            model.modelName = dis.readUTF();
            model.architecture = dis.readUTF();
            model.dModel = dis.readInt();
            model.nLayers = dis.readInt();
            model.nHeads = dis.readInt();
            model.vocabSize = dis.readInt();
            model.parameterCount = dis.readLong();
            
            int metaSize = dis.readInt();
            for (int i = 0; i < metaSize; i++) {
                String key = dis.readUTF();
                String value = dis.readUTF();
                model.metadata.put(key, value);
            }
            
            int weightsSize = dis.readInt();
            for (int i = 0; i < weightsSize; i++) {
                String key = dis.readUTF();
                int len = dis.readInt();
                float[] w = new float[len];
                for (int j = 0; j < len; j++) {
                    w[j] = dis.readFloat();
                }
                model.weights.put(key, w);
            }
            
            int biasesSize = dis.readInt();
            for (int i = 0; i < biasesSize; i++) {
                String key = dis.readUTF();
                int len = dis.readInt();
                float[] b = new float[len];
                for (int j = 0; j < len; j++) {
                    b[j] = dis.readFloat();
                }
                model.biases.put(key, b);
            }
            
            dis.close();
            
            System.out.printf("Model imported successfully from %s%n", path);
            System.out.printf("  Name: %s%n", model.modelName);
            System.out.printf("  Architecture: %s%n", model.architecture);
            System.out.printf("  Parameters: %.2fM%n", model.parameterCount / 1_000_000.0);
            return model;
            
        } catch (IOException e) {
            System.err.println("Failed to import model: " + e.getMessage());
            return null;
        }
    }
    
    public static ModelData createDummyModel(String name) {
        ModelData model = new ModelData();
        model.modelName = name;
        model.architecture = "SC-Transformer";
        model.dModel = 64;
        model.nLayers = 2;
        model.nHeads = 8;
        model.vocabSize = 35;
        
        Random r = new Random(42);
        
        for (int i = 0; i < model.nLayers; i++) {
            String prefix = "layer_" + i + "_";
            
            float[] w1 = new float[model.dModel * model.dModel];
            float[] b1 = new float[model.dModel];
            float[] w2 = new float[model.dModel * model.dModel];
            float[] b2 = new float[model.dModel];
            
            for (int j = 0; j < w1.length; j++) w1[j] = (float) r.nextGaussian() * 0.01f;
            for (int j = 0; j < b1.length; j++) b1[j] = 0;
            for (int j = 0; j < w2.length; j++) w2[j] = (float) r.nextGaussian() * 0.01f;
            for (int j = 0; j < b2.length; j++) b2[j] = 0;
            
            model.weights.put(prefix + "attn_w", w1);
            model.biases.put(prefix + "attn_b", b1);
            model.weights.put(prefix + "ffn_w", w2);
            model.biases.put(prefix + "ffn_b", b2);
        }
        
        float[] embW = new float[model.vocabSize * model.dModel];
        float[] embB = new float[model.dModel];
        float[] outW = new float[model.dModel * model.vocabSize];
        float[] outB = new float[model.vocabSize];
        
        for (int j = 0; j < embW.length; j++) embW[j] = (float) r.nextGaussian() * 0.01f;
        for (int j = 0; j < embB.length; j++) embB[j] = 0;
        for (int j = 0; j < outW.length; j++) outW[j] = (float) r.nextGaussian() * 0.01f;
        for (int j = 0; j < outB.length; j++) outB[j] = 0;
        
        model.weights.put("embedding_w", embW);
        model.biases.put("embedding_b", embB);
        model.weights.put("output_w", outW);
        model.biases.put("output_b", outB);
        
        long paramCount = 0;
        for (float[] w : model.weights.values()) paramCount += w.length;
        for (float[] b : model.biases.values()) paramCount += b.length;
        model.parameterCount = paramCount;
        
        model.metadata.put("created", new Date().toString());
        model.metadata.put("framework", "SC Architecture");
        model.metadata.put("version", "1.0");
        
        return model;
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Model Export/Import");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        String modelPath = "models/test-export.model";
        
        System.out.println("[1/3] Creating dummy model...");
        ModelData model = createDummyModel("Test-SC-Model");
        
        System.out.println("\n[2/3] Exporting model...");
        ExportOptions options = new ExportOptions()
            .includeWeights(true)
            .includeBiases(true)
            .includeMetadata(true)
            .format("binary");
        exportModel(model, modelPath, options);
        
        System.out.println("\n[3/3] Importing model...");
        ModelData imported = importModel(modelPath);
        
        if (imported != null) {
            System.out.println("\nModel verification:");
            System.out.printf("  Name matches: %s%n", imported.modelName.equals(model.modelName));
            System.out.printf("  Architecture matches: %s%n", imported.architecture.equals(model.architecture));
            System.out.printf("  Parameter count matches: %s%n", imported.parameterCount == model.parameterCount);
        }
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Export/Import Complete!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}