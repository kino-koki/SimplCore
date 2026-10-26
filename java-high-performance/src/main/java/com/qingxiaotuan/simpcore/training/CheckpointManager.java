package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;
import java.util.*;

public class CheckpointManager {

    private final Path checkpointDir;
    private final int maxCheckpoints;
    private final String prefix;

    public CheckpointManager(String checkpointDir) {
        this(checkpointDir, 5, "checkpoint");
    }

    public CheckpointManager(String checkpointDir, int maxCheckpoints, String prefix) {
        this.checkpointDir = Paths.get(checkpointDir);
        this.maxCheckpoints = maxCheckpoints;
        this.prefix = prefix;
        try {
            Files.createDirectories(this.checkpointDir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create checkpoint directory", e);
        }
    }

    public void save(SimpCoreModel model, Optimizer optimizer, int step, float loss) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s_step_%d_loss_%.4f_%s.bin", prefix, step, loss, timestamp);
        Path filepath = checkpointDir.resolve(filename);

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(filepath)))) {
            dos.writeInt(step);
            dos.writeFloat(loss);
            dos.writeLong(System.currentTimeMillis());

            model.save(dos);

            dos.writeFloat(optimizer.getCurrentLearningRate());
            dos.writeFloat(optimizer.getWeightDecay());
            dos.writeFloat(optimizer.getMomentum());
            dos.writeFloat(optimizer.getGradClip());
            dos.writeInt(optimizer.getGradientAccumulationSteps());

            System.out.printf("Checkpoint saved: %s (step=%d, loss=%.4f)%n", filepath, step, loss);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save checkpoint", e);
        }

        cleanupOldCheckpoints();
    }

    public Checkpoint load(String filename) {
        Path filepath = checkpointDir.resolve(filename);
        if (!Files.exists(filepath)) {
            throw new IllegalArgumentException("Checkpoint not found: " + filepath);
        }

        return loadFromPath(filepath);
    }

    public Checkpoint loadLatest() {
        List<Path> checkpoints = listCheckpoints();
        if (checkpoints.isEmpty()) {
            return null;
        }

        Path latest = checkpoints.get(checkpoints.size() - 1);
        return loadFromPath(latest);
    }

    public Checkpoint loadBest() {
        List<Path> checkpoints = listCheckpoints();
        if (checkpoints.isEmpty()) {
            return null;
        }

        Path best = null;
        float bestLoss = Float.MAX_VALUE;

        for (Path cp : checkpoints) {
            float loss = extractLossFromFilename(cp.getFileName().toString());
            if (loss < bestLoss) {
                bestLoss = loss;
                best = cp;
            }
        }

        if (best == null) {
            return null;
        }

        return loadFromPath(best);
    }

    private Checkpoint loadFromPath(Path filepath) {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(filepath)))) {
            int step = dis.readInt();
            float loss = dis.readFloat();
            long timestamp = dis.readLong();

            SimpCoreModel model = SimpCoreModel.loadFromStream(dis);

            float lr = dis.readFloat();
            float weightDecay = dis.readFloat();
            float momentum = dis.readFloat();
            float gradClip = dis.readFloat();
            int gradAccumSteps = dis.readInt();

            Optimizer optimizer = new Optimizer(lr, weightDecay, momentum, gradClip);
            optimizer.setGradientAccumulationSteps(gradAccumSteps);

            System.out.printf("Checkpoint loaded: %s (step=%d, loss=%.4f)%n", filepath, step, loss);

            return new Checkpoint(model, optimizer, step, loss, timestamp, filepath.toString());
        } catch (IOException e) {
            throw new RuntimeException("Failed to load checkpoint", e);
        }
    }

    private List<Path> listCheckpoints() {
        List<Path> checkpoints = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(checkpointDir, prefix + "_*")) {
            for (Path path : stream) {
                if (Files.isRegularFile(path)) {
                    checkpoints.add(path);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to list checkpoints", e);
        }

        checkpoints.sort(Comparator.comparing(this::extractStepFromFilename));
        return checkpoints;
    }

    private void cleanupOldCheckpoints() {
        List<Path> checkpoints = listCheckpoints();
        if (checkpoints.size() <= maxCheckpoints) {
            return;
        }

        int toRemove = checkpoints.size() - maxCheckpoints;
        for (int i = 0; i < toRemove; i++) {
            try {
                Files.delete(checkpoints.get(i));
                System.out.printf("Removed old checkpoint: %s%n", checkpoints.get(i));
            } catch (IOException e) {
                System.err.printf("Failed to remove old checkpoint: %s%n", checkpoints.get(i));
            }
        }
    }

    private int extractStepFromFilename(Path path) {
        String name = path.getFileName().toString();
        try {
            int start = name.indexOf("step_") + 5;
            int end = name.indexOf("_loss");
            return Integer.parseInt(name.substring(start, end));
        } catch (Exception e) {
            return 0;
        }
    }

    private float extractLossFromFilename(String name) {
        try {
            int start = name.indexOf("loss_") + 5;
            int end = name.indexOf("_", start);
            return Float.parseFloat(name.substring(start, end));
        } catch (Exception e) {
            return Float.MAX_VALUE;
        }
    }

    public List<String> listCheckpointNames() {
        List<Path> checkpoints = listCheckpoints();
        List<String> names = new ArrayList<>();
        for (Path cp : checkpoints) {
            names.add(cp.getFileName().toString());
        }
        return names;
    }

    public static class Checkpoint {
        public final SimpCoreModel model;
        public final Optimizer optimizer;
        public final int step;
        public final float loss;
        public final long timestamp;
        public final String path;

        public Checkpoint(SimpCoreModel model, Optimizer optimizer, int step, float loss, long timestamp, String path) {
            this.model = model;
            this.optimizer = optimizer;
            this.step = step;
            this.loss = loss;
            this.timestamp = timestamp;
            this.path = path;
        }

        @Override
        public String toString() {
            return String.format(
                "Checkpoint{step=%d, loss=%.4f, timestamp=%d, path='%s'}",
                step, loss, timestamp, path
            );
        }
    }
}