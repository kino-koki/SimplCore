package com.qingxiaotuan.simpcore.training;

/**
 * 混合精度训练管理器 - 支持FP16/FP32混合精度运算和自动损失缩放。
 *
 * <p>混合精度训练通过在部分计算中使用FP16（半精度浮点）来加速训练并减少内存占用，
 * 同时在关键计算节点保持FP32（单精度浮点）以确保数值稳定性。</p>
 *
 * <p>自动损失缩放（Automatic Loss Scaling）流程：
 * <ol>
 *   <li>前向传播前，将损失乘以缩放因子scale</li>
 *   <li>反向传播得到缩放后的梯度</li>
 *   <li>检查梯度是否溢出（NaN/Inf）</li>
 *   <li>若溢出：减小scale，跳过本次更新</li>
 *   <li>若正常：梯度除以scale恢复，可选地增大scale</li>
 * </ol></p>
 *
 * <p>复杂度分析：
 * <ul>
 *   <li>FP16转换开销: O(n)，但FP16计算本身可加速1.5-3x</li>
 *   <li>内存节省: 约40-50%（权重主副本FP32 + FP32梯度 + FP16副本）</li>
 * </ul></p>
 *
 * @author SC Architecture Team
 */
public class MixedPrecision {

    /** 默认初始缩放因子 */
    private static final float DEFAULT_INIT_SCALE = 65536.0f;  // 2^16
    /** 缩放因子增长因子 */
    private static final float SCALE_GROWTH = 2.0f;
    /** 缩放因子缩减因子 */
    private static final float SCALE_SHRINK = 0.5f;
    /** 连续无溢出步数后增大scale的阈值 */
    private static final int GROWTH_INTERVAL = 2000;

    private float lossScale;
    private int consecutiveGoodSteps;
    private long totalSteps;
    private long overflowSteps;
    private boolean enabled;

    /** 内存使用统计（字节） */
    private long fp32MemoryBytes;
    private long fp16MemoryBytes;

    /**
     * 创建混合精度管理器。
     *
     * @param enabled 是否启用混合精度
     */
    public MixedPrecision(boolean enabled) {
        this.enabled = enabled;
        this.lossScale = DEFAULT_INIT_SCALE;
        this.consecutiveGoodSteps = 0;
        this.totalSteps = 0;
        this.overflowSteps = 0;
        this.fp32MemoryBytes = 0;
        this.fp16MemoryBytes = 0;
    }

    /**
     * 创建混合精度管理器（默认启用）。
     */
    public MixedPrecision() {
        this(true);
    }

    /**
     * 将FP32数组转换为FP16表示（用short存储以模拟FP16）。
     * 注意：Java原生不支持FP16，这里使用float存储但模拟FP16的精度范围。
     *
     * @param fp32Array FP32浮点数组
     * @return 模拟FP16精度的数组
     */
    public float[] toFP16(float[] fp32Array) {
        if (!enabled) return fp32Array;

        float[] fp16Array = new float[fp32Array.length];
        for (int i = 0; i < fp32Array.length; i++) {
            fp16Array[i] = floatToHalf(fp32Array[i]);
        }
        return fp16Array;
    }

    /**
     * 将FP16表示转换回FP32。
     *
     * @param fp16Array FP16精度数组
     * @return FP32浮点数组
     */
    public float[] toFP32(float[] fp16Array) {
        if (!enabled) return fp16Array;
        // FP16到FP32的转换在Java中是隐式的（因为我们都用float存储）
        // 这里只需返回副本
        return fp16Array.clone();
    }

    /**
     * 模拟FP16精度：截断到FP16的有效范围和精度。
     * FP16范围: ±65504, 最小正规数: ~6.1e-5
     *
     * @param f FP32值
     * @return 模拟FP16精度的值
     */
    private float floatToHalf(float f) {
        if (f == 0.0f) return 0.0f;
        if (Float.isNaN(f) || Float.isInfinite(f)) return f;

        // 检查是否超出FP16范围
        float absF = Math.abs(f);
        if (absF > 65504.0f) {
            return f > 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        if (absF < 6.1e-5f && absF > 0) {
            // 下溢，可能变为非正规数或零
            return 0.0f;
        }

        // 模拟FP16精度：保留10位尾数（FP16有10位尾数）
        int bits = Float.floatToIntBits(f);
        int sign = bits & 0x80000000;
        int exp = (bits >> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        // 截断尾数到FP16精度（保留高10位）
        mantissa = mantissa & 0xFFC00000;  // 保留高10位

        int halfBits = sign | (exp << 23) | mantissa;
        return Float.intBitsToFloat(halfBits);
    }

    /**
     * 获取当前损失缩放因子。
     *
     * @return 缩放因子
     */
    public float getLossScale() {
        return lossScale;
    }

    /**
     * 缩放损失值。
     *
     * @param loss 原始损失
     * @return 缩放后的损失
     */
    public float scaleLoss(float loss) {
        if (!enabled) return loss;
        return loss * lossScale;
    }

    /**
     * 检查梯度是否溢出（包含NaN或Inf）。
     *
     * @param gradients 梯度数组
     * @return true如果检测到溢出
     */
    public boolean checkGradientOverflow(float[] gradients) {
        if (!enabled) return false;

        for (float g : gradients) {
            if (Float.isNaN(g) || Float.isInfinite(g)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 反缩放梯度（除以损失缩放因子）。
     *
     * @param gradients 缩放后的梯度
     * @return 反缩放后的梯度（原地修改）
     */
    public float[] unscaleGradients(float[] gradients) {
        if (!enabled) return gradients;

        float invScale = 1.0f / lossScale;
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] *= invScale;
        }
        return gradients;
    }

    /**
     * 更新缩放因子（在每个训练步后调用）。
     *
     * @param gradientOverflow 本次梯度是否溢出
     */
    public void updateScale(boolean gradientOverflow) {
        totalSteps++;

        if (gradientOverflow) {
            // 溢出：减小缩放因子，重置计数器
            lossScale *= SCALE_SHRINK;
            lossScale = Math.max(lossScale, 1.0f);  // 最小为1
            consecutiveGoodSteps = 0;
            overflowSteps++;
        } else {
            consecutiveGoodSteps++;
            // 连续无溢出达到阈值：增大缩放因子
            if (consecutiveGoodSteps >= GROWTH_INTERVAL) {
                lossScale *= SCALE_GROWTH;
                lossScale = Math.min(lossScale, DEFAULT_INIT_SCALE * 8);  // 上限
                consecutiveGoodSteps = 0;
            }
        }
    }

    /**
     * 记录内存使用情况。
     *
     * @param paramCount 参数总数
     */
    public void recordMemoryUsage(long paramCount) {
        // FP32: 权重 + 梯度 + 优化器状态(2x) = 4x参数 * 4字节
        this.fp32MemoryBytes = paramCount * 4 * 4;
        // 混合精度: FP32主副本(4字节) + FP16副本(2字节) + FP32梯度(4字节) + FP32优化器(8字节) = 18字节/参数
        this.fp16MemoryBytes = paramCount * 18;
    }

    /**
     * 获取内存节省百分比。
     *
     * @return 节省的内存百分比（0-100）
     */
    public float getMemorySavings() {
        if (fp32MemoryBytes == 0) return 0;
        return (1.0f - (float) fp16MemoryBytes / fp32MemoryBytes) * 100;
    }

    /**
     * 获取溢出率。
     *
     * @return 梯度溢出步数占总步数的比例
     */
    public float getOverflowRate() {
        if (totalSteps == 0) return 0;
        return (float) overflowSteps / totalSteps;
    }

    /**
     * 获取统计报告。
     *
     * @return 格式化的统计字符串
     */
    public String getStatsReport() {
        return String.format(
            "MixedPrecision[enabled=%b, scale=%.1f, steps=%d, overflows=%d (%.2f%%), mem_saved=%.1f%%]",
            enabled, lossScale, totalSteps, overflowSteps,
            getOverflowRate() * 100, getMemorySavings());
    }

    /**
     * 是否启用混合精度。
     *
     * @return true如果启用
     */
    public boolean isEnabled() { return enabled; }

    /**
     * 获取FP32内存使用（字节）。
     *
     * @return FP32内存字节数
     */
    public long getFp32MemoryBytes() { return fp32MemoryBytes; }

    /**
     * 获取FP16内存使用（字节）。
     *
     * @return FP16内存字节数
     */
    public long getFp16MemoryBytes() { return fp16MemoryBytes; }

    /**
     * 重置统计计数器。
     */
    public void reset() {
        consecutiveGoodSteps = 0;
        totalSteps = 0;
        overflowSteps = 0;
        lossScale = DEFAULT_INIT_SCALE;
    }
}
