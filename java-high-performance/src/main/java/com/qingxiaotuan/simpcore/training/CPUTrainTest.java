package com.qingxiaotuan.simpcore.training;

public class CPUTrainTest {
    public static void main(String[] args) throws Exception {
        System.out.println("START");
        // Simulate a simple training step to verify pure CPU path
        com.qingxiaotuan.simpcore.model.SimpCoreModel model = 
            new com.qingxiaotuan.simpcore.model.SimpCoreModel(8, 32, 2, 2, 2, 1, 2, 1, 16);
        com.qingxiaotuan.simpcore.model.Tokenizer tokenizer = 
            new com.qingxiaotuan.simpcore.model.Tokenizer(8);
        Optimizer optimizer = new Optimizer(0.01f, 0.001f, 0.9f, Float.MAX_VALUE);
        TrainingLoop loop = new TrainingLoop(model, tokenizer, optimizer);
        
        String text = "hello";
        System.out.println("step 1");
        float l1 = loop.trainStep(text, 1);
        System.out.println("L1=" + l1);
        System.out.println("step 2");
        float l2 = loop.trainStep(text, 1);
        System.out.println("L2=" + l2);
        System.out.println("DONE");
    }
}
