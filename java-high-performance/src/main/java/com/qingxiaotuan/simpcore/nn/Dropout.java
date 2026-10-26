package com.qingxiaotuan.simpcore.nn;

import java.util.Random;

/**
 * Dropout正则化层 - 通过随机置零神经元输出防止过拟合。
 *
 * <p>工作原理：训练时以概率p将神经元输出置零，其余输出按1/(1-p)缩放以保持期望不变。
 * 推理时不执行任何操作（直接透传）。</p>
 *
 * <p>复杂度分析：
 * <ul>
 *   <li>前向传播: O(n), n为输入元素数</li>
 *   <li>反向传播: O(n), 使用缓存的mask直接乘以梯度</li>
 *   <li>内存开销: O(n), 需存储mask用于反向传播</li>
 * </ul></p>
 *
 * @author SC Architecture Team
 */
public class Dropout {

    private final float dropoutRate;
    private final Random random;
    private boolean trainingMode;
    private float[] mask;

    /**
     * 创建Dropout层。
     *
     * @param dropoutRate dropout概率，范围[0.0, 1.0)。常用值: 0.1~0.5
     * @param seed 随机种子，用于可复现性
     * @throws IllegalArgumentException 如果dropoutRate不在[0, 1)范围内
     */
    public Dropout(float dropoutRate, long seed) {
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException(
                "dropoutRate must be in [0.0, 1.0), got: " + dropoutRate);
        }
        this.dropoutRate = dropoutRate;
        this.random = new Random(seed);
        this.trainingMode = true;
    }

    /**
     * 创建Dropout层（使用默认种子）。
     *
     * @param dropoutRate dropout概率
     */
    public Dropout(float dropoutRate) {
        this(dropoutRate, System.currentTimeMillis());
    }

    /**
     * 前向传播。
     * 训练模式：以dropoutRate概率置零，其余按1/(1-p)缩放（inverted dropout）。
     * 推理模式：直接返回输入副本。
     *
     * @param input 输入数组，形状为[seqLen * dModel]
     * @return 经过dropout处理后的输出
     */
    public float[] forward(float[] input) {
        if (!trainingMode || dropoutRate == 0.0f) {
            return input.clone();
        }

        float[] output = new float[input.length];
        mask = new float[input.length];
        float keepProb = 1.0f - dropoutRate;
        float scale = 1.0f / keepProb;

        for (int i = 0; i < input.length; i++) {
            if (random.nextFloat() < keepProb) {
                mask[i] = scale;
                output[i] = input[i] * scale;
            } else {
                mask[i] = 0.0f;
                output[i] = 0.0f;
            }
        }

        return output;
    }

    /**
     * 反向传播。
     * 梯度通过mask传回：被dropout的位置梯度为0，其余位置梯度乘以缩放因子。
     *
     * @param gradOutput 上游传来的梯度
     * @return 经过mask处理后的梯度
     */
    public float[] backward(float[] gradOutput) {
        if (!trainingMode || dropoutRate == 0.0f || mask == null) {
            return gradOutput.clone();
        }

        float[] gradInput = new float[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = gradOutput[i] * mask[i];
        }
        return gradInput;
    }

    /**
     * 设置训练/推理模式。
     *
     * @param training true=训练模式（执行dropout），false=推理模式（透传）
     */
    public void setTraining(boolean training) {
        this.trainingMode = training;
        if (!training) {
            this.mask = null;
        }
    }

    /**
     * 获取当前是否处于训练模式。
     *
     * @return true如果在训练模式
     */
    public boolean isTraining() {
        return trainingMode;
    }

    /**
     * 获取dropout率。
     *
     * @return dropout概率
     */
    public float getDropoutRate() {
        return dropoutRate;
    }

    /**
     * 清除缓存的mask，释放内存。
     */
    public void clearCache() {
        this.mask = null;
    }
}
