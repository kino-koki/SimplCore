package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;

public class TimeoutTrainTest {
    public static void main(String[] args) throws Exception {
        System.out.println("Creating model...");
        SimpCoreModel model = new SimpCoreModel(64, 128, 4, 4, 4, 4, 2, 4, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        Optimizer optimizer = new Optimizer(0.005f, 0.01f, 0.9f, 1.0f);
        TrainingLoop loop = new TrainingLoop(model, tokenizer, optimizer);
        
        System.out.println("Testing FORWARD only (no backward)...");
        System.out.flush();
        int[] ids = tokenizer.encode("hello world");
        System.out.println("Tokens: " + ids.length + " ids: " + ids[0] + "," + ids[1]);
        System.out.flush();
        
        // Forward only
        long t0 = System.currentTimeMillis();
        float[] logits = model.forward(ids, 1, ids.length);
        long t1 = System.currentTimeMillis();
        System.out.println("Forward took " + (t1-t0) + "ms");
        System.out.println("Logits length: " + logits.length);
        System.out.flush();
        
        System.out.println("Testing FORWARD+BACKWARD (trainStep)...");
        System.out.flush();
        long t2 = System.currentTimeMillis();
        float loss = loop.trainStep("hello world", 1);
        long t3 = System.currentTimeMillis();
        System.out.println("trainStep took " + (t3-t2) + "ms, loss=" + loss);
        System.out.flush();
        
        System.out.println("ALL DONE");
    }
}
