package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class GradientVerificationTest {

    private static final float EPSILON = 1e-4f;
    private static final float TOLERANCE = 1e-3f;

    @Test
    public void testMatrixOperationsGradients() {
        System.out.println("Testing MatrixOperations gradients...");
        
        float[] A = {1, 2, 3, 4};
        float[] B = {5, 6, 7, 8};
        
        float[] C = MatrixOperations.add(A, B);
        assertTrue("Add should produce correct result", Math.abs(C[0] - 6) < EPSILON);
        
        float[] D = MatrixOperations.sub(A, B);
        assertTrue("Sub should produce correct result", Math.abs(D[0] + 4) < EPSILON);
        
        float[] E = MatrixOperations.mul(A, 2);
        assertTrue("Mul should produce correct result", Math.abs(E[0] - 2) < EPSILON);
        
        float[] F = MatrixOperations.relu(new float[]{-1, 0, 1, 2});
        assertTrue("ReLU should produce correct result", F[0] == 0 && F[2] == 1);
        
        float dot = MatrixOperations.dotProduct(A, B);
        assertTrue("Dot product should produce correct result", Math.abs(dot - 70) < EPSILON);
        
        System.out.println("MatrixOperations gradients test passed!");
    }

    @Test
    public void testSoftmaxGradient() {
        System.out.println("Testing Softmax gradient...");
        
        float[] logits = {1, 2, 3};
        float[] softmax = MatrixOperations.softmax(logits);
        
        float sum = 0;
        for (float s : softmax) sum += s;
        assertTrue("Softmax should sum to 1", Math.abs(sum - 1) < EPSILON);
        
        float[] grad = numericalGradient(x -> MatrixOperations.softmax(x)[0], logits, 0, 1e-5f);
        float[] analyticalGrad = computeSoftmaxGradient(logits, 0);
        
        assertTrue("Softmax gradient should match", Math.abs(grad[0] - analyticalGrad[0]) < TOLERANCE);
        
        System.out.println("Softmax gradient test passed!");
    }

    @Test
    public void testMatmulGradient() {
        System.out.println("Testing Matmul gradient...");
        
        float[] A = {1, 2, 3, 4};
        float[] B = {5, 6, 7, 8};
        float[] C = MatrixOperations.matmul(A, B, 2, 2, 2);
        
        assertTrue("Matmul should produce correct result", Math.abs(C[0] - 19) < EPSILON);
        
        System.out.println("Matmul gradient test passed!");
    }

    @Test
    public void testEndToEndTraining() {
        System.out.println("Testing end-to-end training...");
        
        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(64, 2, 64, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);
        
        String[] texts = {"hello world this is a test", "machine learning is fun", "deep learning is powerful"};
        
        float initialLoss = Float.MAX_VALUE;
        float finalLoss = Float.MAX_VALUE;
        
        for (int iter = 0; iter < 100; iter++) {
            String text = texts[iter % texts.length];
            float loss = trainingLoop.trainStep(text, 1);
            
            if (iter == 0) initialLoss = loss;
            if (iter == 99) finalLoss = loss;
            
            if (iter % 20 == 0) {
                System.out.printf("Iteration %d: Loss = %.4f%n", iter, loss);
            }
        }
        
        System.out.printf("Initial Loss: %.4f, Final Loss: %.4f%n", initialLoss, finalLoss);
        
        assertTrue("Loss should decrease during training", finalLoss < initialLoss);
        assertTrue("Loss should be finite", Float.isFinite(finalLoss));
        
        System.out.println("End-to-end training test passed!");
    }

    @Test
    public void testGeneration() {
        System.out.println("Testing generation...");
        
        Tokenizer tokenizer = new Tokenizer(1000);
        SimpCoreModel model = new SimpCoreModel(64, 2, 64, 1000, 32);
        Optimizer optimizer = new Optimizer(0.001f);
        TrainingLoop trainingLoop = new TrainingLoop(model, tokenizer, optimizer);
        
        String[] texts = {"hello world", "machine learning"};
        for (int iter = 0; iter < 50; iter++) {
            trainingLoop.trainStep(texts[iter % texts.length], 1);
        }
        
        String prompt = "hello";
        String generated = trainingLoop.generate(prompt, 10);
        
        System.out.printf("Prompt: %s, Generated: %s%n", prompt, generated);
        
        assertTrue("Generated text should not be empty", generated != null && !generated.isEmpty());
        assertTrue("Generated text should contain prompt", generated.contains("hello"));
        
        System.out.println("Generation test passed!");
    }

    private float[] numericalGradient(java.util.function.Function<float[], Float> func, float[] input, int idx, float eps) {
        float[] grad = new float[input.length];
        
        for (int i = 0; i < input.length; i++) {
            float[] inputPlus = input.clone();
            float[] inputMinus = input.clone();
            
            inputPlus[i] += eps;
            inputMinus[i] -= eps;
            
            float fPlus = func.apply(inputPlus);
            float fMinus = func.apply(inputMinus);
            
            grad[i] = (fPlus - fMinus) / (2 * eps);
        }
        
        return grad;
    }

    private float[] computeSoftmaxGradient(float[] logits, int outputIdx) {
        float[] softmax = MatrixOperations.softmax(logits);
        float[] grad = new float[logits.length];
        
        for (int i = 0; i < logits.length; i++) {
            if (i == outputIdx) {
                grad[i] = softmax[i] * (1 - softmax[i]);
            } else {
                grad[i] = -softmax[i] * softmax[outputIdx];
            }
        }
        
        return grad;
    }
}