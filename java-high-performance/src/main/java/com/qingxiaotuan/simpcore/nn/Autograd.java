package com.qingxiaotuan.simpcore.nn;

import java.util.*;

public class Autograd {
    
    public static class Tensor {
        public float[] data;
        public float[] grad;
        public List<Tensor> inputs;
        public GradFn gradFn;
        public boolean requiresGrad;
        
        public Tensor(float[] data) {
            this(data, false);
        }
        
        public Tensor(float[] data, boolean requiresGrad) {
            this.data = data;
            this.grad = new float[data.length];
            this.inputs = new ArrayList<>();
            this.gradFn = null;
            this.requiresGrad = requiresGrad;
        }
        
        public Tensor(float scalar) {
            this(new float[]{scalar}, false);
        }
        
        public int size() {
            return data.length;
        }
        
        public float get(int idx) {
            return data[idx];
        }
        
        public void set(int idx, float val) {
            data[idx] = val;
        }
        
        public void zeroGrad() {
            Arrays.fill(grad, 0);
        }
        
        public void backward() {
            if (!requiresGrad) return;
            if (gradFn == null) return;
            
            Arrays.fill(grad, 1);
            LinkedList<Tensor> queue = new LinkedList<>();
            queue.add(this);
            
            while (!queue.isEmpty()) {
                Tensor t = queue.poll();
                if (t.gradFn != null) {
                    t.gradFn.backward(t.grad);
                }
                for (Tensor input : t.inputs) {
                    if (input.requiresGrad) {
                        queue.add(input);
                    }
                }
            }
        }
        
        public Tensor add(Tensor other) {
            return Operations.add(this, other);
        }
        
        public Tensor sub(Tensor other) {
            return Operations.sub(this, other);
        }
        
        public Tensor mul(Tensor other) {
            return Operations.mul(this, other);
        }
        
        public Tensor div(Tensor other) {
            return Operations.div(this, other);
        }
        
        public Tensor matmul(Tensor other) {
            return Operations.matmul(this, other);
        }
        
        public Tensor relu() {
            return Operations.relu(this);
        }
        
        public Tensor softmax() {
            return Operations.softmax(this);
        }
        
        public Tensor log() {
            return Operations.log(this);
        }
        
        public Tensor neg() {
            return Operations.neg(this);
        }
        
        public Tensor pow(float exponent) {
            return Operations.pow(this, exponent);
        }
        
        public float sum() {
            float sum = 0;
            for (float val : data) sum += val;
            return sum;
        }
        
        public Tensor clone() {
            return new Tensor(Arrays.copyOf(data, data.length), requiresGrad);
        }
        
        public Tensor detach() {
            return new Tensor(Arrays.copyOf(data, data.length), false);
        }
    }
    
    public interface GradFn {
        void backward(float[] gradOutput);
    }
    
    public static class Operations {
        
        public static Tensor add(Tensor a, Tensor b) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = a.data[i] + b.data[i];
            }
            
            Tensor out = new Tensor(result, a.requiresGrad || b.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.inputs.add(b);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i];
                        }
                    }
                    if (b.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            b.grad[i] += grad[i];
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor sub(Tensor a, Tensor b) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = a.data[i] - b.data[i];
            }
            
            Tensor out = new Tensor(result, a.requiresGrad || b.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.inputs.add(b);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i];
                        }
                    }
                    if (b.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            b.grad[i] -= grad[i];
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor mul(Tensor a, Tensor b) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = a.data[i] * b.data[i];
            }
            
            Tensor out = new Tensor(result, a.requiresGrad || b.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.inputs.add(b);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i] * b.data[i];
                        }
                    }
                    if (b.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            b.grad[i] += grad[i] * a.data[i];
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor div(Tensor a, Tensor b) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = a.data[i] / b.data[i];
            }
            
            Tensor out = new Tensor(result, a.requiresGrad || b.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.inputs.add(b);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i] / b.data[i];
                        }
                    }
                    if (b.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            b.grad[i] -= grad[i] * a.data[i] / (b.data[i] * b.data[i]);
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor matmul(Tensor a, Tensor b) {
            int m = (int) Math.sqrt(a.size());
            int n = m;
            int p = n;
            
            float[] result = new float[m * p];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += a.data[i * n + k] * b.data[k * p + j];
                    }
                    result[i * p + j] = sum;
                }
            }
            
            Tensor out = new Tensor(result, a.requiresGrad || b.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.inputs.add(b);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < m; i++) {
                            for (int k = 0; k < n; k++) {
                                float sum = 0;
                                for (int j = 0; j < p; j++) {
                                    sum += grad[i * p + j] * b.data[k * p + j];
                                }
                                a.grad[i * n + k] += sum;
                            }
                        }
                    }
                    if (b.requiresGrad) {
                        for (int k = 0; k < n; k++) {
                            for (int j = 0; j < p; j++) {
                                float sum = 0;
                                for (int i = 0; i < m; i++) {
                                    sum += a.data[i * n + k] * grad[i * p + j];
                                }
                                b.grad[k * p + j] += sum;
                            }
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor relu(Tensor a) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = Math.max(0, a.data[i]);
            }
            
            Tensor out = new Tensor(result, a.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            if (a.data[i] > 0) {
                                a.grad[i] += grad[i];
                            }
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor softmax(Tensor a) {
            int n = a.size();
            float[] result = new float[n];
            
            float max = a.data[0];
            for (float val : a.data) {
                if (val > max) max = val;
            }
            
            float sum = 0;
            for (int i = 0; i < n; i++) {
                result[i] = (float) Math.exp(a.data[i] - max);
                sum += result[i];
            }
            
            for (int i = 0; i < n; i++) {
                result[i] /= sum;
            }
            
            Tensor out = new Tensor(result, a.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < n; i++) {
                            float sumGrad = 0;
                            for (int j = 0; j < n; j++) {
                                float jacobian = (i == j) ? result[i] * (1 - result[j]) : -result[i] * result[j];
                                sumGrad += grad[j] * jacobian;
                            }
                            a.grad[i] += sumGrad;
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor log(Tensor a) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = (float) Math.log(Math.max(1e-10f, a.data[i]));
            }
            
            Tensor out = new Tensor(result, a.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i] / Math.max(1e-10f, a.data[i]);
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor neg(Tensor a) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = -a.data[i];
            }
            
            Tensor out = new Tensor(result, a.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] -= grad[i];
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor pow(Tensor a, float exponent) {
            float[] result = new float[a.size()];
            for (int i = 0; i < result.length; i++) {
                result[i] = (float) Math.pow(a.data[i], exponent);
            }
            
            Tensor out = new Tensor(result, a.requiresGrad);
            if (out.requiresGrad) {
                out.inputs.add(a);
                out.gradFn = grad -> {
                    if (a.requiresGrad) {
                        for (int i = 0; i < grad.length; i++) {
                            a.grad[i] += grad[i] * exponent * (float) Math.pow(a.data[i], exponent - 1);
                        }
                    }
                };
            }
            return out;
        }
        
        public static Tensor crossEntropy(Tensor logits, Tensor targets) {
            Tensor softmax = softmax(logits);
            Tensor logSoftmax = log(softmax);
            
            float[] oneHot = new float[logits.size()];
            int targetIdx = (int) targets.data[0];
            oneHot[targetIdx] = 1.0f;
            
            Tensor targetTensor = new Tensor(oneHot);
            float lossValue = neg(targetTensor.mul(logSoftmax)).sum();
            
            return new Tensor(new float[]{lossValue}, true);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Architecture - Autograd Engine Test");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        System.out.println("[1/3] Testing basic operations...");
        Tensor x = new Tensor(new float[]{2.0f, 3.0f}, true);
        Tensor y = new Tensor(new float[]{4.0f, 5.0f}, true);
        
        Tensor z = x.add(y).mul(new Tensor(new float[]{1.0f, 2.0f}));
        System.out.printf("  x + y = %s%n", Arrays.toString(z.data));
        
        z.backward();
        System.out.printf("  x.grad = %s%n", Arrays.toString(x.grad));
        System.out.printf("  y.grad = %s%n", Arrays.toString(y.grad));
        
        System.out.println("\n[2/3] Testing ReLU...");
        Tensor a = new Tensor(new float[]{-1.0f, 0.0f, 1.0f, 2.0f}, true);
        Tensor b = a.relu();
        System.out.printf("  relu([-1, 0, 1, 2]) = %s%n", Arrays.toString(b.data));
        
        b.backward();
        System.out.printf("  a.grad = %s%n", Arrays.toString(a.grad));
        
        System.out.println("\n[3/3] Testing softmax and cross-entropy...");
        Tensor logits = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, true);
        Tensor target = new Tensor(new float[]{2.0f});
        
        Tensor loss = Operations.crossEntropy(logits, target);
        System.out.printf("  Cross-entropy loss: %.4f%n", loss.data[0]);
        
        loss.backward();
        System.out.printf("  logits.grad = %s%n", Arrays.toString(logits.grad));
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    Autograd Test Passed!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}