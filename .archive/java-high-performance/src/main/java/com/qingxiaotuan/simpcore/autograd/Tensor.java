package com.qingxiaotuan.simpcore.autograd;

import java.util.*;
import java.util.concurrent.atomic.*;

public class Tensor {
    
    public final float[] data;
    public float[] grad;
    public boolean requiresGrad;
    public FunctionNode gradFn;
    public final int[] shape;
    public final int ndim;
    public final int numel;
    
    private static final AtomicLong idCounter = new AtomicLong(0);
    public final long id;
    
    public Tensor(float[] data, int... shape) {
        this.data = data;
        this.shape = shape;
        this.ndim = shape.length;
        this.numel = computeNumel(shape);
        this.requiresGrad = false;
        this.grad = null;
        this.gradFn = null;
        this.id = idCounter.incrementAndGet();
        
        if (data.length != numel) {
            throw new IllegalArgumentException("Data length " + data.length + " does not match shape " + Arrays.toString(shape));
        }
    }
    
    public Tensor(float[] data, boolean requiresGrad, int... shape) {
        this(data, shape);
        this.requiresGrad = requiresGrad;
        if (requiresGrad) {
            this.grad = new float[numel];
        }
    }
    
    private int computeNumel(int[] shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        return numel;
    }
    
    public static Tensor zeros(int... shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        return new Tensor(new float[numel], shape);
    }
    
    public static Tensor zeros(boolean requiresGrad, int... shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        return new Tensor(new float[numel], requiresGrad, shape);
    }
    
    public static Tensor ones(int... shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        float[] data = new float[numel];
        Arrays.fill(data, 1.0f);
        return new Tensor(data, shape);
    }
    
    public static Tensor randn(int... shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        float[] data = new float[numel];
        Random random = new Random();
        for (int i = 0; i < numel; i++) {
            data[i] = (float) random.nextGaussian();
        }
        return new Tensor(data, shape);
    }
    
    public static Tensor randn(boolean requiresGrad, int... shape) {
        int numel = 1;
        for (int dim : shape) {
            numel *= dim;
        }
        float[] data = new float[numel];
        Random random = new Random();
        for (int i = 0; i < numel; i++) {
            data[i] = (float) random.nextGaussian();
        }
        return new Tensor(data, requiresGrad, shape);
    }
    
    public Tensor clone() {
        float[] newData = Arrays.copyOf(data, data.length);
        Tensor result = new Tensor(newData, shape);
        result.requiresGrad = this.requiresGrad;
        if (this.grad != null) {
            result.grad = Arrays.copyOf(this.grad, this.grad.length);
        }
        result.gradFn = this.gradFn;
        return result;
    }
    
    public Tensor detach() {
        float[] newData = Arrays.copyOf(data, data.length);
        return new Tensor(newData, shape);
    }
    
    public void backward() {
        backward(null);
    }
    
    public void backward(Tensor gradient) {
        if (!requiresGrad) {
            return;
        }
        
        if (gradient == null) {
            if (numel == 1) {
                gradient = ones(shape);
            } else {
                throw new IllegalArgumentException("Gradient must be provided for non-scalar tensor");
            }
        }
        
        if (grad == null) {
            grad = new float[numel];
        }
        System.arraycopy(gradient.data, 0, grad, 0, numel);
        
        List<Tensor> visited = new ArrayList<>();
        Stack<Tensor> stack = new Stack<>();
        stack.push(this);
        
        while (!stack.isEmpty()) {
            Tensor t = stack.peek();
            
            if (t.gradFn == null || visited.contains(t)) {
                stack.pop();
                continue;
            }
            
            visited.add(t);
            
            for (Tensor input : t.gradFn.inputs) {
                if (input.requiresGrad && !visited.contains(input)) {
                    stack.push(input);
                }
            }
        }
        
        for (int i = visited.size() - 1; i >= 0; i--) {
            Tensor t = visited.get(i);
            if (t.gradFn != null) {
                t.gradFn.backward(t.grad);
            }
        }
    }
    
    public void zeroGrad() {
        if (grad != null) {
            Arrays.fill(grad, 0.0f);
        }
    }
    
    public float norm() {
        float sum = 0;
        for (float val : data) {
            sum += val * val;
        }
        return (float) Math.sqrt(sum);
    }
    
    public float gradNorm() {
        if (grad == null) return 0;
        float sum = 0;
        for (float val : grad) {
            sum += val * val;
        }
        return (float) Math.sqrt(sum);
    }
    
    public float mean() {
        float sum = 0;
        for (float val : data) {
            sum += val;
        }
        return sum / numel;
    }
    
    public float sum() {
        float sum = 0;
        for (float val : data) {
            sum += val;
        }
        return sum;
    }
    
    public float max() {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : data) {
            max = Math.max(max, val);
        }
        return max;
    }
    
    public float min() {
        float min = Float.POSITIVE_INFINITY;
        for (float val : data) {
            min = Math.min(min, val);
        }
        return min;
    }
    
    public Tensor reshape(int... newShape) {
        int newNumel = 1;
        for (int dim : newShape) {
            if (dim == -1) continue;
            newNumel *= dim;
        }
        
        int negIdx = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                negIdx = i;
                break;
            }
        }
        
        if (negIdx != -1) {
            int[] finalShape = new int[newShape.length];
            System.arraycopy(newShape, 0, finalShape, 0, newShape.length);
            finalShape[negIdx] = numel / newNumel;
            return new Tensor(data, requiresGrad, finalShape);
        }
        
        if (newNumel != numel) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + numel + " to " + Arrays.toString(newShape));
        }
        
        return new Tensor(data, requiresGrad, newShape);
    }
    
    public Tensor view(int... newShape) {
        return reshape(newShape);
    }
    
    public Tensor transpose(int dim0, int dim1) {
        if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
            throw new IllegalArgumentException("Invalid dimensions");
        }
        
        int[] newShape = Arrays.copyOf(shape, shape.length);
        int temp = newShape[dim0];
        newShape[dim0] = newShape[dim1];
        newShape[dim1] = temp;
        
        float[] newData = new float[numel];
        
        int[] strides = computeStrides(shape);
        int[] newStrides = computeStrides(newShape);
        
        for (int i = 0; i < numel; i++) {
            int[] indices = unravelIndex(i, shape);
            int[] transposedIndices = Arrays.copyOf(indices, indices.length);
            int tempIdx = transposedIndices[dim0];
            transposedIndices[dim0] = transposedIndices[dim1];
            transposedIndices[dim1] = tempIdx;
            
            int newIndex = ravelIndex(transposedIndices, newStrides);
            newData[newIndex] = data[i];
        }
        
        return new Tensor(newData, requiresGrad, newShape);
    }
    
    private int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[strides.length - 1] = 1;
        for (int i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    
    private int[] unravelIndex(int index, int[] shape) {
        int[] indices = new int[shape.length];
        int[] strides = computeStrides(shape);
        for (int i = 0; i < shape.length; i++) {
            indices[i] = index / strides[i];
            index %= strides[i];
        }
        return indices;
    }
    
    private int ravelIndex(int[] indices, int[] strides) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }
    
    public Tensor permute(int... dims) {
        if (dims.length != ndim) {
            throw new IllegalArgumentException("Number of dimensions must match");
        }
        
        int[] newShape = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            newShape[i] = shape[dims[i]];
        }
        
        float[] newData = new float[numel];
        int[] strides = computeStrides(shape);
        int[] newStrides = computeStrides(newShape);
        
        for (int i = 0; i < numel; i++) {
            int[] indices = unravelIndex(i, shape);
            int[] permutedIndices = new int[ndim];
            for (int j = 0; j < ndim; j++) {
                permutedIndices[j] = indices[dims[j]];
            }
            int newIndex = ravelIndex(permutedIndices, newStrides);
            newData[newIndex] = data[i];
        }
        
        return new Tensor(newData, requiresGrad, newShape);
    }
    
    public Tensor squeeze() {
        List<Integer> newShapeList = new ArrayList<>();
        for (int dim : shape) {
            if (dim != 1) {
                newShapeList.add(dim);
            }
        }
        
        int[] newShape = new int[newShapeList.size()];
        for (int i = 0; i < newShape.length; i++) {
            newShape[i] = newShapeList.get(i);
        }
        
        return new Tensor(data, requiresGrad, newShape);
    }
    
    public Tensor squeeze(int dim) {
        if (dim < 0) {
            dim += ndim;
        }
        
        if (shape[dim] != 1) {
            return this;
        }
        
        int[] newShape = new int[ndim - 1];
        int idx = 0;
        for (int i = 0; i < ndim; i++) {
            if (i != dim) {
                newShape[idx++] = shape[i];
            }
        }
        
        return new Tensor(data, requiresGrad, newShape);
    }
    
    public Tensor unsqueeze(int dim) {
        if (dim < 0) {
            dim += ndim + 1;
        }
        
        int[] newShape = new int[ndim + 1];
        int idx = 0;
        for (int i = 0; i < ndim + 1; i++) {
            if (i == dim) {
                newShape[i] = 1;
            } else {
                newShape[i] = shape[idx++];
            }
        }
        
        return new Tensor(data, requiresGrad, newShape);
    }
    
    public Tensor expand(int... newShape) {
        if (newShape.length < ndim) {
            throw new IllegalArgumentException("New shape must have at least " + ndim + " dimensions");
        }
        
        int[] expandedShape = new int[newShape.length];
        int offset = newShape.length - ndim;
        
        for (int i = 0; i < ndim; i++) {
            int originalDim = shape[i];
            int newDim = newShape[i + offset];
            
            if (originalDim != 1 && originalDim != newDim) {
                throw new IllegalArgumentException("Cannot expand dimension " + i + " from " + originalDim + " to " + newDim);
            }
            
            expandedShape[i + offset] = newDim;
        }
        
        for (int i = 0; i < offset; i++) {
            expandedShape[i] = newShape[i];
        }
        
        int expandedNumel = 1;
        for (int dim : expandedShape) {
            expandedNumel *= dim;
        }
        
        float[] newData = new float[expandedNumel];
        int[] strides = computeStrides(shape);
        int[] expandedStrides = computeStrides(expandedShape);
        
        for (int i = 0; i < expandedNumel; i++) {
            int[] indices = unravelIndex(i, expandedShape);
            int[] originalIndices = new int[ndim];
            
            for (int j = 0; j < ndim; j++) {
                originalIndices[j] = indices[j + offset] % shape[j];
            }
            
            int originalIndex = ravelIndex(originalIndices, strides);
            newData[i] = data[originalIndex];
        }
        
        return new Tensor(newData, requiresGrad, expandedShape);
    }
    
    public int get(int... indices) {
        int[] strides = computeStrides(shape);
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            index += indices[i] * strides[i];
        }
        return index;
    }
    
    public float getValue(int... indices) {
        return data[get(indices)];
    }
    
    public void setValue(float value, int... indices) {
        data[get(indices)] = value;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(id=").append(id).append(", shape=").append(Arrays.toString(shape));
        sb.append(", requiresGrad=").append(requiresGrad).append(")\n");
        
        int displaySize = Math.min(20, numel);
        sb.append("Data: [");
        for (int i = 0; i < displaySize; i++) {
            sb.append(String.format("%.4f", data[i]));
            if (i < displaySize - 1) {
                sb.append(", ");
            }
        }
        if (numel > displaySize) {
            sb.append(", ...");
        }
        sb.append("]");
        
        if (grad != null) {
            sb.append("\nGrad: [");
            for (int i = 0; i < displaySize; i++) {
                sb.append(String.format("%.4f", grad[i]));
                if (i < displaySize - 1) {
                    sb.append(", ");
                }
            }
            if (numel > displaySize) {
                sb.append(", ...");
            }
            sb.append("]");
        }
        
        return sb.toString();
    }
}