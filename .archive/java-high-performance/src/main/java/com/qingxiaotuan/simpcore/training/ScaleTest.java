package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.io.PrintWriter;
import java.io.FileOutputStream;

/**
 * 规模测试 — 直接写入文件避免缓存问题
 */
public class ScaleTest {
    static final int VOCAB_SIZE = 128;
    static final int D_MODEL = 64;
    static final int N_LAYERS = 2;
    static final int N_HEADS = 2;
    static final int N_BASES = 1;
    static final int N_CHANNELS = 1;
    static final int N_PATHS = 1;
    static final int TOP_K = 1;
    static final int MAX_SEQ_LEN = 64;

    static final String[] DATA = {
        "hello world this is a test of the ai system",
        "the cat sat on the mat and looked around",
        "deep learning changes everything we know about",
        "AI is the future of technology in the modern",
        "the quick brown fox jumps over the lazy dog"
    };

    public static void main(String[] args) throws Exception {
        PrintWriter pw = new PrintWriter(new FileOutputStream("scale_result.txt"));
        MatrixOperations.matmul(new float[4], new float[4], 2, 2, 2);
        
        pw.println("=== SC架构 规模扩展测试 ===");
        pw.println("模型: D_MODEL=" + D_MODEL + ", LAYERS=" + N_LAYERS + ", VOCAB=" + VOCAB_SIZE);
        pw.flush();

        long t0 = System.nanoTime();
        pw.println("[1] 初始化...");
        SimpCoreModel model = new SimpCoreModel(VOCAB_SIZE, D_MODEL, N_LAYERS, N_BASES, N_HEADS,
            N_CHANNELS, N_PATHS, TOP_K, MAX_SEQ_LEN);
        Tokenizer tokenizer = new Tokenizer(VOCAB_SIZE);
        pw.println("  Tokenizer: " + tokenizer.getVocabSize() + " tokens, "
            + (System.nanoTime() - t0) / 1e9 + "s");

        pw.println("[2] 训练20步...");
        t0 = System.nanoTime();
        Optimizer optimizer = new Optimizer(0.005f, 0.01f, 0.9f, 1.0f);
        optimizer.applyCosineSchedule(1, 20, 5, 0.005f, 0.0005f);
        TrainingLoop loop = new TrainingLoop(model, tokenizer, optimizer);
        float fL = 0, lL = 0;
        for (int s = 1; s <= 20; s++) {
            float loss = loop.trainStep(DATA[(s - 1) % DATA.length], 1);
            optimizer.stepScheduler();
            if (s == 1) fL = loss;
            if (s == 20) lL = loss;
            if (s % 10 == 0) pw.println("  步" + s + ": loss=" + loss);
        }
        double drop = ((fL - lL) / fL) * 100;
        pw.println("  损失: " + fL + " -> " + lL + " (下降" + String.format("%.1f", drop) + "%)");
        pw.println("  耗时: " + (System.nanoTime() - t0) / 1e9 + "s");

        pw.println("[3] 生成测试...");
        t0 = System.nanoTime();
        int ok = 0;
        for (String p : new String[]{"hello", "the", "AI", "cat"}) {
            try {
                int[] ids = tokenizer.encode(p);
                int[] out = model.generate(ids, 8, tokenizer.getEosTokenId());
                String gen = tokenizer.decode(out);
                boolean clean = !gen.contains("<unk>") && !gen.contains("<extra_");
                pw.println("  '" + p + "' -> \"" + gen + "\"" + (clean ? "" : " (has unk)"));
                if (clean) ok++;
            } catch (Exception e) {
                pw.println("  '" + p + "' -> " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
        }
        pw.println("  无乱码: " + ok + "/4, " + (System.nanoTime() - t0) / 1e9 + "s");

        pw.println("[4] 在线学习...");
        t0 = System.nanoTime();
        try {
            OnlineLearningManager olm = new OnlineLearningManager(model, 5e-5f, 1);
            for (int l = 0; l < N_LAYERS; l++)
                model.getSpmLayers()[l].setSupervisedMode(true);
            // 12条输入 × 3轮 = 36次在线学习步
            String[] olData = {"hello world this is a test", "the cat sat on the mat",
                "deep learning changes everything", "AI is the future of technology",
                "the quick brown fox jumps", "i like to play with my dog",
                "the sun is shining bright", "she went to the store to buy milk",
                "this technology is truly innovative", "a model that gets smarter over time",
                "continuous learning never stops", "smart systems learn from data"};
            for (int r = 0; r < 3; r++)
                for (String in : olData)
                    olm.processInteraction(tokenizer.encode(in));
            pw.println("  总在线步数: " + olm.getTotalOnlineSteps());
            pw.println("  平均损失: " + String.format("%.4f", olm.getAverageOnlineLoss()));
            pw.println("  改进指数: " + String.format("%.2f", olm.getImprovementRatio() * 100) + "%");
            pw.println("  耗时: " + (System.nanoTime() - t0) / 1e9 + "s");
        } catch (Exception e) {
            pw.println("  跳过: " + e.getMessage());
        }

        // ── 5. 在线学习后生成 ──
        pw.println("[5] 在线学习后生成测试...");
        ok = 0;
        for (String p : new String[]{"hello", "the", "AI", "cat", "deep"}) {
            try {
                int[] ids = tokenizer.encode(p);
                int[] out = model.generate(ids, 8, tokenizer.getEosTokenId());
                String gen = tokenizer.decode(out);
                boolean clean = !gen.contains("<unk>") && !gen.contains("<extra_");
                pw.println("  '" + p + "' -> \"" + gen + "\"" + (clean ? "" : " (has unk)"));
                if (clean) ok++;
            } catch (Exception e) {
                pw.println("  '" + p + "' -> " + e.getMessage());
            }
        }
        pw.println("  无乱码: " + ok + "/5");

        pw.println();
        pw.println("=== 完成! ===");
        pw.close();
        System.err.println("结果已写入 scale_result.txt");
    }
}
