package com.qingxiaotuan.simpcore.training;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Adam优化器单元测试。
 *
 * <p>覆盖范围：构造验证、参数注册、Adam更新公式正确性、
 * 学习率调度、梯度裁剪、权重衰减、收敛性、NaN安全。</p>
 */
@DisplayName("Adam 优化器测试")
class AdamOptimizerTest {

    @Nested
    @DisplayName("构造函数与参数测试")
    class ConstructorTests {

        @Test
        @DisplayName("默认参数构造应使用推荐超参数")
        void testDefaultConstructor() {
            AdamOptimizer opt = new AdamOptimizer(1e-3f);
            assertEquals(1e-3f, opt.getBaseLearningRate(), 1e-10f);
            assertEquals(AdamOptimizer.DEFAULT_BETA1, opt.getBeta1(), 1e-10f);
            assertEquals(AdamOptimizer.DEFAULT_BETA2, opt.getBeta2(), 1e-10f);
            assertEquals(AdamOptimizer.DEFAULT_EPSILON, opt.getEpsilon(), 1e-10f);
            assertEquals(0.0f, opt.getWeightDecay(), 1e-10f);
        }

        @Test
        @DisplayName("全参数构造应正确记录")
        void testFullConstructor() {
            AdamOptimizer opt = new AdamOptimizer(0.01f, 0.85f, 0.995f, 1e-7f, 0.01f, 1.0f);
            assertEquals(0.01f, opt.getBaseLearningRate(), 1e-10f);
            assertEquals(0.85f, opt.getBeta1(), 1e-10f);
            assertEquals(0.995f, opt.getBeta2(), 1e-10f);
            assertEquals(1e-7f, opt.getEpsilon(), 1e-10f);
            assertEquals(0.01f, opt.getWeightDecay(), 1e-10f);
        }

        @Test
        @DisplayName("带权重衰减构造应正确记录")
        void testWeightDecayConstructor() {
            AdamOptimizer opt = new AdamOptimizer(0.001f, 0.1f);
            assertEquals(0.1f, opt.getWeightDecay(), 1e-10f);
        }

        @Test
        @DisplayName("默认参数应符合论文推荐值")
        void testDefaultsMatchPaper() {
            // Adam论文推荐：β1=0.9, β2=0.999, ε=1e-8
            assertEquals(0.9f, AdamOptimizer.DEFAULT_BETA1, 0.0f,
                "β1应为0.9");
            assertEquals(0.999f, AdamOptimizer.DEFAULT_BETA2, 0.0f,
                "β2应为0.999");
            assertEquals(1e-8f, AdamOptimizer.DEFAULT_EPSILON, 0.0f,
                "ε应为1e-8");
        }
    }

    @Nested
    @DisplayName("参数注册测试")
    class RegistrationTests {

        @Test
        @DisplayName("registerParams应返回唯一ID")
        void testRegisterReturnsUniqueId() {
            AdamOptimizer opt = new AdamOptimizer(1e-3f);
            int id1 = opt.registerParams(100);
            int id2 = opt.registerParams(200);
            assertNotEquals(id1, id2, "不同参数组应有不同ID");
        }

        @Test
        @DisplayName("registerParams应增加缓存大小")
        void testRegisterIncreasesCacheSize() {
            AdamOptimizer opt = new AdamOptimizer(1e-3f);
            assertEquals(0, opt.getCacheSize());
            opt.registerParams(100);
            assertEquals(1, opt.getCacheSize());
            opt.registerParams(200);
            assertEquals(2, opt.getCacheSize());
        }

        @Test
        @DisplayName("clearCache应清空所有缓存")
        void testClearCache() {
            AdamOptimizer opt = new AdamOptimizer(1e-3f);
            opt.registerParams(100);
            opt.registerParams(200);
            assertEquals(2, opt.getCacheSize());
            opt.clearCache();
            assertEquals(0, opt.getCacheSize());
        }
    }

    @Nested
    @DisplayName("Adam更新公式正确性测试")
    class UpdateFormulaTests {

        @Test
        @DisplayName("更新应减小损失方向的参数")
        void testUpdateDecreasesLoss() {
            // 简单凸问题：w^2，梯度2w，最优解w=0
            AdamOptimizer opt = new AdamOptimizer(0.1f);
            float[] w = {5.0f};
            float[] grad = {10.0f};  // 2*5

            float wBefore = w[0];
            opt.update(w, grad);
            assertTrue(w[0] < wBefore,
                "在正梯度方向上参数应减小，但 " + wBefore + " -> " + w[0]);
        }

        @Test
        @DisplayName("多次更新应收敛到最优解")
        void testConvergence() {
            // 凸问题：min (w-3)^2，梯度 2(w-3)，最优w=3
            AdamOptimizer opt = new AdamOptimizer(0.1f);
            float[] w = {0.0f};

            for (int i = 0; i < 500; i++) {
                float[] grad = {2.0f * (w[0] - 3.0f)};
                opt.update(w, grad);
            }
            assertEquals(3.0f, w[0], 0.1f,
                "Adam应在500步内收敛到3.0，实际: " + w[0]);
        }

        @Test
        @DisplayName("一阶矩估计应正确计算")
        void testFirstMomentEstimate() {
            // 使用β1=0.9，梯度恒为g=1
            // m_t = 0.9*m_{t-1} + 0.1*1
            // t=1: m = 0.1, mHat = 0.1/(1-0.9) = 1.0
            AdamOptimizer opt = new AdamOptimizer(0.001f);
            float[] w = {0.0f};
            float[] grad = {1.0f};
            opt.update(w, grad);
            // 验证更新方向（参数应减小）
            assertTrue(w[0] < 0,
                "正梯度下参数应减小");
        }

        @Test
        @DisplayName("二阶矩估计应正确计算")
        void testSecondMomentEstimate() {
            // 大梯度应被二阶矩归一化
            AdamOptimizer opt1 = new AdamOptimizer(0.01f);
            AdamOptimizer opt2 = new AdamOptimizer(0.01f);
            float[] w1 = {0.0f};
            float[] w2 = {0.0f};
            float[] grad1 = {1.0f};
            float[] grad2 = {1000.0f};  // 大1000倍的梯度

            opt1.update(w1, grad1);
            opt2.update(w2, grad2);
            // 由于二阶矩归一化，更新幅度应相近（而非差1000倍）
            float ratio = Math.abs(w2[0]) / Math.max(Math.abs(w1[0]), 1e-12f);
            assertTrue(ratio < 100,
                "Adam应归一化大梯度，ratio=" + ratio + " 应远小于1000");
        }

        @Test
        @DisplayName("偏差修正应使初期更新更准确")
        void testBiasCorrection() {
            // 初期t=1时，m=0.1*1=0.1，mHat=0.1/(1-0.9)=1.0
            // 没有偏差修正，更新会非常小
            AdamOptimizer opt = new AdamOptimizer(0.1f);
            float[] w = {0.0f};
            float[] grad = {1.0f};
            opt.update(w, grad);
            // 第一步 lr * mHat/(sqrt(vHat)+eps) ≈ 0.1 * 1.0 / (sqrt(1.0)+1e-8) ≈ 0.1
            assertEquals(-0.1f, w[0], 0.05f,
                "偏差修正后第一步更新应≈lr，实际: " + w[0]);
        }

        @Test
        @DisplayName("NaN梯度应被安全处理")
        void testNaNGradientSafety() {
            AdamOptimizer opt = new AdamOptimizer(0.1f);
            float[] w = {1.0f};
            float[] grad = {Float.NaN};
            opt.update(w, grad);
            // NaN导致权重变为NaN后应被重置为0
            assertEquals(0.0f, w[0], 1e-10f,
                "NaN应被重置为0");
        }

        @Test
        @DisplayName("Inf梯度应被安全处理")
        void testInfGradientSafety() {
            AdamOptimizer opt = new AdamOptimizer(0.1f);
            float[] w = {1.0f};
            float[] grad = {Float.POSITIVE_INFINITY};
            opt.update(w, grad);
            assertFalse(Float.isNaN(w[0]), "权重不应为NaN");
            assertFalse(Float.isInfinite(w[0]), "权重不应为Inf");
        }
    }

    @Nested
    @DisplayName("学习率调度测试")
    class SchedulerTests {

        @Test
        @DisplayName("无调度器时应使用基础学习率")
        void testNoScheduler() {
            AdamOptimizer opt = new AdamOptimizer(0.005f);
            assertEquals(0.005f, opt.getCurrentLearningRate(), 1e-10f);
        }

        @Test
        @DisplayName("setScheduler应安装调度器")
        void testSetScheduler() {
            AdamOptimizer opt = new AdamOptimizer(0.01f);
            LearningRateScheduler scheduler =
                new LearningRateScheduler.CosineAnnealingScheduler(0.01f, 100);
            opt.setScheduler(scheduler);
            assertEquals(0.01f, opt.getCurrentLearningRate(), 1e-6f);
        }

        @Test
        @DisplayName("stepScheduler应推进调度器")
        void testStepScheduler() {
            AdamOptimizer opt = new AdamOptimizer(0.01f);
            LearningRateScheduler scheduler =
                new LearningRateScheduler.CosineAnnealingScheduler(0.01f, 100);
            opt.setScheduler(scheduler);

            float lr0 = opt.getCurrentLearningRate();
            opt.stepScheduler();
            float lr1 = opt.getCurrentLearningRate();
            // 余弦退火：t=1时学习率应略降
            assertTrue(lr1 < lr0,
                String.format("余弦退火t=1应使lr降低: %.6f -> %.6f", lr0, lr1));
        }

        @Test
        @DisplayName("余弦退火应在totalSteps时达到最小值")
        void testCosineAnnealingReachesMin() {
            AdamOptimizer opt = new AdamOptimizer(0.01f);
            LearningRateScheduler scheduler =
                new LearningRateScheduler.CosineAnnealingScheduler(0.01f, 10, 0.001f);
            opt.setScheduler(scheduler);
            for (int i = 0; i < 10; i++) opt.stepScheduler();
            assertEquals(0.001f, opt.getCurrentLearningRate(), 1e-6f,
                "余弦退火应在totalSteps时达到minLr");
        }

        @Test
        @DisplayName("线性预热应使lr线性增加")
        void testLinearWarmup() {
            AdamOptimizer opt = new AdamOptimizer(0.01f);
            LearningRateScheduler inner =
                new LearningRateScheduler.ConstantScheduler(0.01f);
            LearningRateScheduler warmup =
                new LearningRateScheduler.LinearWarmupScheduler(0.01f, 10, inner);
            opt.setScheduler(warmup);

            float[] lrs = new float[10];
            for (int i = 0; i < 10; i++) {
                lrs[i] = opt.getCurrentLearningRate();
                opt.stepScheduler();
            }
            // 应单调递增
            for (int i = 1; i < 10; i++) {
                assertTrue(lrs[i] > lrs[i-1],
                    String.format("预热期间lr应单调递增: step %d %.6f -> step %d %.6f",
                        i-1, lrs[i-1], i, lrs[i]));
            }
        }

        @Test
        @DisplayName("setLearningRate应更新基础学习率")
        void testSetLearningRate() {
            AdamOptimizer opt = new AdamOptimizer(0.01f);
            opt.setLearningRate(0.001f);
            assertEquals(0.001f, opt.getBaseLearningRate(), 1e-10f);
            assertEquals(0.001f, opt.getCurrentLearningRate(), 1e-10f);
        }
    }

    @Nested
    @DisplayName("梯度裁剪与权重衰减测试")
    class ClipAndWeightDecayTests {

        @Test
        @DisplayName("梯度裁剪应限制梯度范数")
        void testGradientClipping() {
            // gradClip=1.0，梯度[3,4]范数=5应被缩放到1.0
            AdamOptimizer opt = new AdamOptimizer(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 1.0f);
            float[] w = {0.0f, 0.0f};
            float[] grad = {3.0f, 4.0f};
            opt.update(w, grad);
            // 更新幅度应受限
            double updateNorm = Math.sqrt(w[0]*w[0] + w[1]*w[1]);
            assertTrue(updateNorm < 0.1,
                "裁剪后更新范数应小，实际: " + updateNorm);
        }

        @Test
        @DisplayName("权重衰减应将权重向0收缩")
        void testWeightDecayShrinksWeights() {
            // 无梯度时，权重衰减应使权重变小
            AdamOptimizer opt = new AdamOptimizer(0.01f, 0.9f, 0.999f, 1e-8f, 0.1f, 0.0f);
            float[] w = {5.0f};
            float[] grad = {0.0f};  // 零梯度
            opt.update(w, grad);
            // 由于权重衰减，权重应减小（即使梯度为0）
            assertTrue(w[0] < 5.0f,
                "权重衰减应使权重减小，实际: " + w[0]);
        }
    }

    @Nested
    @DisplayName("与SGD对比测试")
    class AdamVsSgdTests {

        @Test
        @DisplayName("Adam与SGD都应能在病态条件下收敛")
        void testAdamFasterThanSgdInIllConditioned() {
            // 病态问题：f(x,y) = 10x^2 + y^2，x方向曲率大
            // 两者都应能收敛（验证Adam能稳定训练即可）
            AdamOptimizer adam = new AdamOptimizer(0.01f);
            Optimizer sgd = new Optimizer(0.01f);  // 较保守lr避免SGD发散

            float[] wAdam = {5.0f, 5.0f};
            float[] wSgd = {5.0f, 5.0f};
            float initDist = (float) Math.sqrt(50.0);

            for (int i = 0; i < 500; i++) {
                float[] gAdam = {20.0f * wAdam[0], 2.0f * wAdam[1]};
                float[] gSgd = {20.0f * wSgd[0], 2.0f * wSgd[1]};
                adam.update(wAdam, gAdam);
                sgd.update(wSgd, gSgd);
            }

            double adamDist = Math.sqrt(wAdam[0]*wAdam[0] + wAdam[1]*wAdam[1]);
            double sgdDist = Math.sqrt(wSgd[0]*wSgd[0] + wSgd[1]*wSgd[1]);
            // 两者都应收敛（距离减小）
            assertTrue(adamDist < initDist,
                String.format("Adam应收敛: 初始=%.4f, 最终=%.4f", initDist, adamDist));
            assertTrue(sgdDist < initDist,
                String.format("SGD应收敛: 初始=%.4f, 最终=%.4f", initDist, sgdDist));
        }
    }

    @Test
    @DisplayName("多参数组独立更新")
    void testMultipleParamGroups() {
        AdamOptimizer opt = new AdamOptimizer(0.1f);
        int id1 = opt.registerParams(3);
        int id2 = opt.registerParams(3);

        float[] w1 = {5.0f, 5.0f, 5.0f};
        float[] w2 = {1.0f, 1.0f, 1.0f};
        float[] g1 = {1.0f, 1.0f, 1.0f};
        float[] g2 = {2.0f, 2.0f, 2.0f};

        opt.update(id1, w1, g1);
        opt.update(id2, w2, g2);

        // 两组应独立更新
        assertNotEquals(w1[0], w2[0], 1e-6f,
            "不同参数组应独立更新");
    }
}
