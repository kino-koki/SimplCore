package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;

public class SimpleTrainTest {
    public static void main(String[] args) throws Exception {
        System.out.println("Creating model...");
        SimpCoreModel model = new SimpCoreModel(64, 128, 4, 4, 4, 2, 4, 2, 32);
        Tokenizer tokenizer = new Tokenizer(64);
        Optimizer optimizer = new Optimizer(0.005f, 0.01f, 0.9f, 1.0f);
        TrainingLoop loop = new TrainingLoop(model, tokenizer, optimizer);
        
        String text = "hello world this is a test";
        System.out.println("Training step 1...");
        float loss1 = loop.trainStep(text, 1);
        System.out.println("Loss step 1: " + loss1);
        
        System.out.println("Training step 2...");
        float loss2 = loop.trainStep(text, 1);
        System.out.println("Loss step 2: " + loss2);
        
        System.out.println("Training step 3...");
        float loss3 = loop.trainStep(text, 1);
        System.out.println("Loss step 3: " + loss3);
        
        System.out.println("Test completed successfully!");
    }
}
