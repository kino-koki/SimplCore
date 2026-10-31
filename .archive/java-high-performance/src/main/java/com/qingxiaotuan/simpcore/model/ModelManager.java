package com.qingxiaotuan.simpcore.model;

import java.util.*;
import java.io.*;

public class ModelManager {
    
    public static class ModelVersion {
        public final String versionId;
        public final long timestamp;
        public final String description;
        public final long paramCount;
        public final float accuracy;
        
        public ModelVersion(String versionId, long timestamp, String description, long paramCount, float accuracy) {
            this.versionId = versionId;
            this.timestamp = timestamp;
            this.description = description;
            this.paramCount = paramCount;
            this.accuracy = accuracy;
        }
    }
    
    public static class Checkpoint {
        public final String checkpointId;
        public final long timestamp;
        public final int epoch;
        public final float loss;
        public final float accuracy;
        public final String modelPath;
        
        public Checkpoint(String checkpointId, long timestamp, int epoch, float loss, float accuracy, String modelPath) {
            this.checkpointId = checkpointId;
            this.timestamp = timestamp;
            this.epoch = epoch;
            this.loss = loss;
            this.accuracy = accuracy;
            this.modelPath = modelPath;
        }
    }
    
    private final Map<String, Object> models;
    private final List<ModelVersion> versions;
    private final List<Checkpoint> checkpoints;
    private final String modelDir;
    private volatile String currentVersion;
    
    public ModelManager(String modelDir) {
        this.modelDir = modelDir;
        this.models = new HashMap<>();
        this.versions = new ArrayList<>();
        this.checkpoints = new ArrayList<>();
        this.currentVersion = "v1.0";
        new File(modelDir).mkdirs();
        loadVersions();
    }
    
    public ModelManager() {
        this("models");
    }
    
    public void saveModel(String name, Object model, String description) {
        try {
            File file = new File(modelDir, name + ".model");
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                oos.writeObject(model);
            }
            
            models.put(name, model);
            
            ModelVersion version = new ModelVersion(
                "v" + (versions.size() + 1),
                System.currentTimeMillis(),
                description,
                estimateParams(model),
                0.0f
            );
            versions.add(version);
            currentVersion = version.versionId;
            
            System.out.printf("[ModelManager] Saved model: %s (%d params)%n", name, version.paramCount);
        } catch (IOException e) {
            System.err.println("[ModelManager] Save failed: " + e.getMessage());
        }
    }
    
    public Object loadModel(String name) {
        if (models.containsKey(name)) {
            return models.get(name);
        }
        
        try {
            File file = new File(modelDir, name + ".model");
            if (!file.exists()) {
                return null;
            }
            
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                Object model = ois.readObject();
                models.put(name, model);
                return model;
            }
        } catch (Exception e) {
            System.err.println("[ModelManager] Load failed: " + e.getMessage());
            return null;
        }
    }
    
    public void saveCheckpoint(String name, Object model, int epoch, float loss, float accuracy) {
        String checkpointId = name + "_epoch_" + epoch + "_" + System.currentTimeMillis();
        String modelPath = modelDir + "/" + checkpointId + ".chkpt";
        
        try {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
                oos.writeObject(model);
            }
            
            Checkpoint checkpoint = new Checkpoint(
                checkpointId,
                System.currentTimeMillis(),
                epoch,
                loss,
                accuracy,
                modelPath
            );
            checkpoints.add(checkpoint);
            
            System.out.printf("[ModelManager] Saved checkpoint: %s (epoch=%d, loss=%.4f)%n", 
                checkpointId, epoch, loss);
        } catch (IOException e) {
            System.err.println("[ModelManager] Save checkpoint failed: " + e.getMessage());
        }
    }
    
    public Object loadCheckpoint(String checkpointId) {
        for (Checkpoint cp : checkpoints) {
            if (cp.checkpointId.equals(checkpointId)) {
                try {
                    File file = new File(cp.modelPath);
                    if (!file.exists()) return null;
                    
                    try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                        return ois.readObject();
                    }
                } catch (Exception e) {
                    System.err.println("[ModelManager] Load checkpoint failed: " + e.getMessage());
                }
            }
        }
        return null;
    }
    
    public Checkpoint getLastCheckpoint(String name) {
        for (int i = checkpoints.size() - 1; i >= 0; i--) {
            if (checkpoints.get(i).checkpointId.startsWith(name)) {
                return checkpoints.get(i);
            }
        }
        return null;
    }
    
    public List<Checkpoint> listCheckpoints(String name) {
        List<Checkpoint> result = new ArrayList<>();
        for (Checkpoint cp : checkpoints) {
            if (cp.checkpointId.startsWith(name)) {
                result.add(cp);
            }
        }
        return result;
    }
    
    public List<ModelVersion> listVersions() {
        return new ArrayList<>(versions);
    }
    
    public ModelVersion getCurrentVersion() {
        for (ModelVersion v : versions) {
            if (v.versionId.equals(currentVersion)) {
                return v;
            }
        }
        return versions.isEmpty() ? null : versions.get(versions.size() - 1);
    }
    
    public void deleteModel(String name) {
        models.remove(name);
        File file = new File(modelDir, name + ".model");
        if (file.exists()) {
            file.delete();
        }
    }
    
    public void deleteCheckpoint(String checkpointId) {
        checkpoints.removeIf(cp -> cp.checkpointId.equals(checkpointId));
        File file = new File(modelDir, checkpointId + ".chkpt");
        if (file.exists()) {
            file.delete();
        }
    }
    
    public Set<String> listModels() {
        return models.keySet();
    }
    
    private long estimateParams(Object model) {
        return model != null ? model.toString().length() * 1000 : 0;
    }
    
    private void loadVersions() {
        versions.add(new ModelVersion("v1.0", System.currentTimeMillis(), "初始版本", 0, 0.0f));
    }
    
    public String getStatus() {
        return String.format(
            "ModelManager: models=%d, versions=%d, checkpoints=%d, current=%s",
            models.size(), versions.size(), checkpoints.size(), currentVersion
        );
    }
}