package com.qingxiaotuan.simpcore.nn;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Dropout正则化层单元测试。
 *
 * <p>覆盖范围：构造验证、前向传播、反向传播、训练/推理模式切换、
 * 可复现性、边界条件、复杂度验证。</p>
 */
@DisplayName("Dropout 正则化层测试")
class DropoutTest {

    @Nested
    @DisplayName("构造函数测试")
    class ConstructorTests {

        @Test
        @DisplayName("合法dropout率应正确创建")
        void testValidDropoutRates() {
            for (float p : new float[]{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.9f}) {
                Dropout d = new Dropout(p, 42L);
                assertEquals(p, d.getDropoutRate(), 1e-6f,
                    "合法率 " + p + " 应被正确记录");
                assertTrue(d.isTraining(), "默认应为训练模式");
            }
        }

        @Test
        @DisplayName("负数dropout率应抛出IllegalArgumentException")
        void testNegativeRateThrows() {
            assertThrows(IllegalArgumentException.class,
                () -> new Dropout(-0.1f, 42L),
                "负数率应被拒绝");
        }

        @Test
        @DisplayName("大于等于1的dropout率应抛出IllegalArgumentException")
        void testRateGEOneThrows() {
            assertThrows(IllegalArgumentException.class,
                () -> new Dropout(1.0f, 42L),
                "1.0 应被拒绝");
            assertThrows(IllegalArgumentException.class,
                () -> new Dropout(1.5f, 42L),
                "1.5 应被拒绝");
        }

        @Test
        @DisplayName("无种子构造函数应正常工作")
        void testNoSeedConstructor() {
            Dropout d = new Dropout(0.3f);
            assertEquals(0.3f, d.getDropoutRate(), 1e-6f);
        }
    }

    @Nested
    @DisplayName("前向传播测试")
    class ForwardTests {

        @Test
        @DisplayName("推理模式应原样返回输入（透传）")
        void testInferenceModeIsIdentity() {
            Dropout d = new Dropout(0.5f, 42L);
            d.setTraining(false);
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] output = d.forward(input);
            assertArrayEquals(input, output, 1e-6f,
                "推理模式应直接透传");
            assertNotSame(input, output, "应返回副本避免外部修改");
        }

        @Test
        @DisplayName("dropout率=0时训练模式也应透传")
        void testZeroDropoutIsIdentity() {
            Dropout d = new Dropout(0.0f, 42L);
            d.setTraining(true);
            float[] input = {1.0f, -2.0f, 3.5f};
            float[] output = d.forward(input);
            assertArrayEquals(input, output, 1e-6f);
        }

        @Test
        @DisplayName("训练模式下应部分置零并按1/(1-p)缩放")
        void testTrainingModeDropsAndScales() {
            // 使用足够大的数组和固定seed使统计稳定
            Dropout d = new Dropout(0.5f, 1234L);
            int n = 10000;
            float[] input = new float[n];
            for (int i = 0; i < n; i++) input[i] = 1.0f;

            float[] output = d.forward(input);
            int zeros = 0;
            float sum = 0;
            for (float v : output) {
                if (v == 0.0f) zeros++;
                sum += v;
            }

            // 约一半被置零
            float dropRate = (float) zeros / n;
            assertTrue(dropRate > 0.4f && dropRate < 0.6f,
                "dropout=0.5 时实际置零率应在 [0.4, 0.6]，实际: " + dropRate);
            // 期望保持: E[output] = E[input] = 1
            float mean = sum / n;
            assertEquals(1.0f, mean, 0.1f,
                "缩放后输出期望应≈输入期望，实际: " + mean);
            // 被保留的位置应为 1/(1-0.5) = 2.0
            boolean foundScaled = false;
            for (float v : output) {
                if (v > 0) {
                    assertEquals(2.0f, v, 1e-6f, "保留值应为2.0");
                    foundScaled = true;
                }
            }
            assertTrue(foundScaled, "应至少有一个保留值");
        }

        @Test
        @DisplayName("不同dropout率应产生不同的置零比例")
        void testDifferentRatesProduceDifferentDropRates() {
            int n = 5000;
            float[] input = new float[n];
            for (int i = 0; i < n; i++) input[i] = 1.0f;

            float[] rates = {0.2f, 0.3f, 0.4f};
            float[] observedRates = new float[3];
            for (int i = 0; i < 3; i++) {
                Dropout d = new Dropout(rates[i], 42L + i);
                float[] out = d.forward(input);
                int zeros = 0;
                for (float v : out) if (v == 0.0f) zeros++;
                observedRates[i] = (float) zeros / n;
            }

            // 观察到的置零率应与配置近似单调递增
            assertTrue(observedRates[0] < observedRates[1],
                String.format("0.2 应比 0.3 置零少: %.3f vs %.3f",
                    observedRates[0], observedRates[1]));
            assertTrue(observedRates[1] < observedRates[2],
                String.format("0.3 应比 0.4 置零少: %.3f vs %.3f",
                    observedRates[1], observedRates[2]));
        }

        @Test
        @DisplayName("相同seed应产生可复现的mask")
        void testReproducibilityWithSameSeed() {
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                             6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
            Dropout d1 = new Dropout(0.3f, 999L);
            Dropout d2 = new Dropout(0.3f, 999L);
            assertArrayEquals(d1.forward(input), d2.forward(input), 1e-6f,
                "相同seed应产生相同输出");
        }

        @Test
        @DisplayName("输出长度应等于输入长度")
        void testOutputLength() {
            Dropout d = new Dropout(0.4f, 42L);
            for (int len : new int[]{1, 10, 100, 256}) {
                float[] input = new float[len];
                float[] output = d.forward(input);
                assertEquals(len, output.length, "长度应保持");
            }
        }
    }

    @Nested
    @DisplayName("反向传播测试")
    class BackwardTests {

        @Test
        @DisplayName("推理模式反向传播应返回梯度副本")
        void testInferenceBackward() {
            Dropout d = new Dropout(0.5f, 42L);
            d.setTraining(false);
            float[] grad = {1.0f, 2.0f, 3.0f};
            float[] out = d.backward(grad);
            assertArrayEquals(grad, out, 1e-6f);
            assertNotSame(grad, out, "应返回副本");
        }

        @Test
        @DisplayName("训练模式反向传播应使用前向mask")
        void testTrainingBackwardUsesMask() {
            Dropout d = new Dropout(0.5f, 42L);
            float[] input = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f};
            float[] fwd = d.forward(input);
            float[] gradIn = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                              0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            float[] gradOut = d.backward(gradIn);

            for (int i = 0; i < input.length; i++) {
                if (fwd[i] == 0.0f) {
                    assertEquals(0.0f, gradOut[i], 1e-6f,
                        "前向置零位置的反向梯度应为0");
                } else {
                    // 前向是 input * 2.0（scale=1/(1-0.5)），梯度应乘以2.0
                    assertEquals(1.0f, gradOut[i], 1e-6f,
                        "前向保留位置的反向梯度应乘以scale");
                }
            }
        }

        @Test
        @DisplayName("未调用forward直接backward应返回梯度副本")
        void testBackwardWithoutForward() {
            Dropout d = new Dropout(0.4f, 42L);
            float[] grad = {1.0f, 2.0f, 3.0f};
            float[] out = d.backward(grad);
            assertArrayEquals(grad, out, 1e-6f,
                "无mask时应透传梯度");
        }
    }

    @Nested
    @DisplayName("模式切换测试")
    class ModeSwitchTests {

        @Test
        @DisplayName("setTraining应正确切换模式")
        void testSetTrainingMode() {
            Dropout d = new Dropout(0.5f, 42L);
            assertTrue(d.isTraining(), "初始应为训练模式");

            d.setTraining(false);
            assertFalse(d.isTraining(), "切换后应为推理模式");

            d.setTraining(true);
            assertTrue(d.isTraining(), "再切换回训练模式");
        }

        @Test
        @DisplayName("推理模式应清除mask缓存")
        void testInferenceClearsMask() {
            Dropout d = new Dropout(0.5f, 42L);
            d.forward(new float[]{1.0f, 2.0f, 3.0f});
            // mask应已建立
            float[] out1 = d.backward(new float[]{1.0f, 1.0f, 1.0f});
            // 切换到推理
            d.setTraining(false);
            float[] out2 = d.backward(new float[]{1.0f, 1.0f, 1.0f});
            assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f}, out2, 1e-6f,
                "推理模式反向应透传");
        }

        @Test
        @DisplayName("clearCache应释放mask内存")
        void testClearCache() {
            Dropout d = new Dropout(0.5f, 42L);
            d.forward(new float[]{1.0f, 2.0f});
            // 应能正常反向
            float[] grad = d.backward(new float[]{1.0f, 1.0f});
            assertNotNull(grad);
            // 清除缓存后再反向应透传
            d.clearCache();
            float[] grad2 = d.backward(new float[]{1.0f, 1.0f});
            assertArrayEquals(new float[]{1.0f, 1.0f}, grad2, 1e-6f);
        }
    }

    @Nested
    @DisplayName("PyTorch接口兼容性测试")
    class InterfaceCompatibilityTests {

        @Test
        @DisplayName("应支持train()/eval()语义")
        void testTrainEvalSemantics() {
            Dropout d = new Dropout(0.5f, 42L);
            // train()
            d.setTraining(true);
            assertTrue(d.isTraining());
            // eval()
            d.setTraining(false);
            assertFalse(d.isTraining());
        }

        @Test
        @DisplayName("不应修改输入数组")
        void testDoesNotModifyInput() {
            Dropout d = new Dropout(0.5f, 42L);
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] inputCopy = input.clone();
            d.forward(input);
            assertArrayEquals(inputCopy, input, 1e-6f,
                "输入数组不应被修改");
        }
    }

    @RepeatedTest(5)
    @DisplayName("多次运行统计应保持稳定")
    void testStatisticalStability() {
        Dropout d = new Dropout(0.3f, System.currentTimeMillis());
        int n = 2000;
        float[] input = new float[n];
        for (int i = 0; i < n; i++) input[i] = 1.0f;
        float[] out = d.forward(input);
        float mean = 0;
        for (float v : out) mean += v;
        mean /= n;
        // 期望保持
        assertEquals(1.0f, mean, 0.15f,
            "dropout=0.3 时期望应≈1.0，实际: " + mean);
    }
}
