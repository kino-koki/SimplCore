package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.util.*;

/**
 * 思维树搜索 — 模拟人类的发散思维和收敛决策过程
 * 
 * 人类解决复杂问题时，不是线性思考，而是：
 * 1. 发散阶段：产生多个可能的思路（树枝）
 * 2. 评估阶段：评估每个思路的可行性和质量
 * 3. 收敛阶段：选择最优路径继续深入
 * 4. 回溯阶段：如果当前路径走不通，回到上一层尝试其他分支
 * 
 * 参考论文：Tree of Thoughts: Deliberate Problem Solving with Large Language Models
 */
public class TreeOfThought {

    private final int dModel;
    private final int maxDepth;
    private final int branchFactor;
    private final float pruningThreshold;
    
    /** 思路扩展权重 */
    private final float[] wExpand;
    
    /** 思路评估权重 */
    private final float[] wEvaluate;
    
    /** 状态转换权重 */
    private final float[] wTransition;
    
    /** 思维树节点 */
    private static class TreeNode {
        float[] state;
        float score;
        int depth;
        List<TreeNode> children;
        TreeNode parent;
        int visitCount;
        
        TreeNode(float[] state, int depth, TreeNode parent) {
            this.state = state;
            this.score = 0.0f;
            this.depth = depth;
            this.children = new ArrayList<>();
            this.parent = parent;
            this.visitCount = 0;
        }
    }
    
    private final Random random;
    private TreeNode root;
    private final List<float[]> reasoningTrajectory;

    public TreeOfThought(int dModel) {
        this(dModel, 5, 3, 0.5f);
    }

    public TreeOfThought(int dModel, int maxDepth, int branchFactor, float pruningThreshold) {
        this.dModel = dModel;
        this.maxDepth = maxDepth;
        this.branchFactor = branchFactor;
        this.pruningThreshold = pruningThreshold;
        this.random = new Random(42);
        this.reasoningTrajectory = new ArrayList<>();
        
        this.wExpand = new float[dModel * dModel];
        this.wEvaluate = new float[dModel];
        this.wTransition = new float[2 * dModel * dModel];
        
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < wExpand.length; i++) {
            wExpand[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wEvaluate.length; i++) {
            wEvaluate[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        for (int i = 0; i < wTransition.length; i++) {
            wTransition[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
    }

    /**
     * 执行思维树搜索
     * 1. 构建思维树：从根节点开始，逐层扩展
     * 2. 评估节点：使用评估函数给每个节点打分
     * 3. 剪枝：移除低质量分支
     * 4. 回溯搜索：选择最优路径
     */
    public float[] search(float[] initialState, int iterations) {
        reasoningTrajectory.clear();
        
        root = new TreeNode(initialState, 0, null);
        
        for (int iter = 0; iter < iterations; iter++) {
            TreeNode leaf = selectNode(root);
            
            if (leaf.depth < maxDepth) {
                expandNode(leaf);
            }
            
            float value = evaluateNode(leaf);
            backpropagate(leaf, value);
            
            pruning();
        }
        
        float[] bestPath = extractBestPath();
        
        return bestPath;
    }

    /**
     * 选择节点 — 使用UCB-like策略平衡探索和利用
     */
    private TreeNode selectNode(TreeNode node) {
        if (node.children.isEmpty() || node.depth >= maxDepth) {
            return node;
        }
        
        TreeNode bestChild = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        
        for (TreeNode child : node.children) {
            float ucbScore = child.score + 
                2.0f * (float) Math.sqrt(Math.log(node.visitCount + 1) / (child.visitCount + 1));
            if (ucbScore > bestScore) {
                bestScore = ucbScore;
                bestChild = child;
            }
        }
        
        if (bestChild == null) {
            return node;
        }
        
        return selectNode(bestChild);
    }

    /**
     * 扩展节点 — 产生多个分支（思路）
     */
    private void expandNode(TreeNode node) {
        for (int i = 0; i < branchFactor; i++) {
            float[] perturbation = new float[dModel];
            for (int j = 0; j < dModel; j++) {
                perturbation[j] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            
            float[] expanded = MatrixOperations.matmul(node.state, wExpand, 1, dModel, dModel);
            for (int j = 0; j < dModel; j++) {
                expanded[j] = node.state[j] + expanded[j] + perturbation[j];
            }
            
            float[] normalized = layerNorm(expanded);
            
            TreeNode child = new TreeNode(normalized, node.depth + 1, node);
            node.children.add(child);
        }
    }

    /**
     * 评估节点 — 评估当前思路的质量
     */
    private float evaluateNode(TreeNode node) {
        float score = 0;
        for (int i = 0; i < dModel; i++) {
            score += node.state[i] * wEvaluate[i];
        }
        
        float norm = 0;
        for (float v : node.state) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        
        score = (float) Math.tanh(score / (norm + 1e-8f));
        
        node.score = score;
        node.visitCount++;
        
        return score;
    }

    /**
     * 反向传播 — 更新路径上所有节点的分数
     */
    private void backpropagate(TreeNode node, float value) {
        TreeNode current = node;
        while (current != null) {
            current.score = (current.score * current.visitCount + value) / (current.visitCount + 1);
            current.visitCount++;
            current = current.parent;
        }
    }

    /**
     * 剪枝 — 移除低质量分支
     */
    private void pruning() {
        pruneNode(root);
    }

    private void pruneNode(TreeNode node) {
        List<TreeNode> toRemove = new ArrayList<>();
        for (TreeNode child : node.children) {
            if (child.score < pruningThreshold && child.depth > 1) {
                toRemove.add(child);
            } else {
                pruneNode(child);
            }
        }
        node.children.removeAll(toRemove);
    }

    /**
     * 提取最优路径
     */
    private float[] extractBestPath() {
        TreeNode current = root;
        reasoningTrajectory.add(Arrays.copyOf(current.state, dModel));
        
        while (!current.children.isEmpty()) {
            TreeNode bestChild = null;
            float bestScore = Float.NEGATIVE_INFINITY;
            
            for (TreeNode child : current.children) {
                if (child.score > bestScore) {
                    bestScore = child.score;
                    bestChild = child;
                }
            }
            
            if (bestChild != null) {
                current = bestChild;
                reasoningTrajectory.add(Arrays.copyOf(current.state, dModel));
            } else {
                break;
            }
        }
        
        float[] finalState = new float[dModel];
        for (float[] state : reasoningTrajectory) {
            for (int i = 0; i < dModel; i++) {
                finalState[i] += state[i];
            }
        }
        for (int i = 0; i < dModel; i++) {
            finalState[i] /= reasoningTrajectory.size();
        }
        
        return finalState;
    }

    /**
     * 执行多步思维链（简化版，类似CoT但支持分支）
     */
    public float[] chainReasoning(float[] input, int steps) {
        reasoningTrajectory.clear();
        
        float[] state = Arrays.copyOf(input, dModel);
        reasoningTrajectory.add(Arrays.copyOf(state, dModel));
        
        for (int step = 0; step < steps; step++) {
            float[] nextState = new float[dModel];
            
            for (int i = 0; i < dModel; i++) {
                float sum = 0;
                for (int j = 0; j < dModel; j++) {
                    sum += state[j] * wExpand[j * dModel + i];
                }
                nextState[i] = state[i] + 0.1f * sum;
            }
            
            state = layerNorm(nextState);
            reasoningTrajectory.add(Arrays.copyOf(state, dModel));
        }
        
        return state;
    }

    private float[] layerNorm(float[] x) {
        float mean = 0;
        for (float v : x) {
            mean += v;
        }
        mean /= dModel;
        
        float var = 0;
        for (float v : x) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= dModel;
        
        float invStd = 1.0f / (float) Math.sqrt(var + 1e-5f);
        float[] output = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            output[i] = (x[i] - mean) * invStd;
        }
        return output;
    }

    public List<float[]> getReasoningTrajectory() {
        return reasoningTrajectory;
    }

    public int getTreeDepth() {
        return computeDepth(root);
    }

    private int computeDepth(TreeNode node) {
        if (node == null || node.children.isEmpty()) {
            return node != null ? node.depth : 0;
        }
        int maxChildDepth = 0;
        for (TreeNode child : node.children) {
            maxChildDepth = Math.max(maxChildDepth, computeDepth(child));
        }
        return maxChildDepth;
    }

    public int getNodeCount() {
        return countNodes(root);
    }

    private int countNodes(TreeNode node) {
        if (node == null) return 0;
        int count = 1;
        for (TreeNode child : node.children) {
            count += countNodes(child);
        }
        return count;
    }

    public float getBestScore() {
        return root != null ? root.score : 0.0f;
    }

    public int getMaxDepth() { return maxDepth; }
    public int getBranchFactor() { return branchFactor; }
    public float getPruningThreshold() { return pruningThreshold; }
    public float[] getWExpand() { return wExpand; }
    public float[] getWEvaluate() { return wEvaluate; }
    public float[] getWTransition() { return wTransition; }
}