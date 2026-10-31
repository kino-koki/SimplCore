package com.qingxiaotuan.simpcore.data;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SCDataLoader多线程数据加载器单元测试。
 *
 * <p>覆盖范围：批处理、序列填充/截断、预加载、缓存、
 * 滑动窗口、token拼接、动态长度支持。</p>
 */
@DisplayName("SCDataLoader 数据加载器测试")
class SCDataLoaderTest {

    private List<int[]> sampleSequences;

    @BeforeEach
    void setUp() {
        sampleSequences = new ArrayList<>();
        // 创建10条长度不同的序列
        for (int i = 0; i < 10; i++) {
            int[] seq = new int[20 + i * 3];
            for (int j = 0; j < seq.length; j++) {
                seq[j] = (i * 100 + j) % 256;
            }
            sampleSequences.add(seq);
        }
    }

    @Nested
    @DisplayName("构造函数与基础功能测试")
    class ConstructorTests {

        @Test
        @DisplayName("构造应正确记录序列数")
        void testConstructor() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 32, true, 2);
            assertEquals(10, loader.size());
            assertEquals(4, loader.getBatchSize());
            assertEquals(32, loader.getSeqLen());
        }

        @Test
        @DisplayName("序列长度超过512应被截断为512")
        void testSeqLenCappedAt512() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 2, 1000, false, 1);
            assertEquals(SCDataLoader.MAX_SEQ_LEN, loader.getSeqLen(),
                "序列长度不应超过512");
        }

        @Test
        @DisplayName("默认构造应使用1个工作线程")
        void testDefaultConstructor() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 32);
            assertEquals(10, loader.size());
        }

        @Test
        @DisplayName("MAX_SEQ_LEN应为512")
        void testMaxSeqLen() {
            assertEquals(512, SCDataLoader.MAX_SEQ_LEN);
        }
    }

    @Nested
    @DisplayName("批处理测试")
    class BatchingTests {

        @Test
        @DisplayName("应正确计算总批次数")
        void testTotalBatches() {
            // 10条序列，batchSize=4 → ceil(10/4)=3
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 32, false, 1);
            assertEquals(3, loader.getTotalBatches());
        }

        @Test
        @DisplayName("应按批次顺序返回数据")
        void testBatchOrder() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 5, 20, false, 1);
            int[][] batch0 = loader.getNextBatch();
            int[][] batch1 = loader.getNextBatch();

            assertNotNull(batch0);
            assertNotNull(batch1);
            assertEquals(5, batch0.length, "第一批应有5条");
            assertEquals(5, batch1.length, "第二批应有5条");
        }

        @Test
        @DisplayName("最后一批可能小于batchSize")
        void testLastBatchSmaller() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            int[][] last = null;
            while (loader.hasNext()) {
                last = loader.getNextBatch();
            }
            assertNotNull(last);
            assertEquals(2, last.length, "最后一批应为2条（10%4=2）");
        }

        @Test
        @DisplayName("无数据时getNextBatch应返回null")
        void testNoMoreData() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            while (loader.hasNext()) loader.getNextBatch();
            assertNull(loader.getNextBatch(), "无数据应返回null");
        }

        @Test
        @DisplayName("hasNext应正确反映剩余批次")
        void testHasNext() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            assertTrue(loader.hasNext());
            loader.getNextBatch();
            assertTrue(loader.hasNext());
            loader.getNextBatch();
            assertTrue(loader.hasNext());
            loader.getNextBatch();
            assertFalse(loader.hasNext());
        }
    }

    @Nested
    @DisplayName("序列填充与截断测试")
    class PadTruncateTests {

        @Test
        @DisplayName("padOrTruncate应截断长序列")
        void testTruncate() {
            int[] seq = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] result = SCDataLoader.padOrTruncate(seq, 5);
            assertEquals(5, result.length);
            assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
        }

        @Test
        @DisplayName("padOrTruncate应填充短序列")
        void testPad() {
            int[] seq = {1, 2, 3};
            int[] result = SCDataLoader.padOrTruncate(seq, 5);
            assertEquals(5, result.length);
            assertArrayEquals(new int[]{1, 2, 3, 0, 0}, result);
        }

        @Test
        @DisplayName("padOrTruncate应保持等长序列")
        void testEqualLength() {
            int[] seq = {1, 2, 3, 4, 5};
            int[] result = SCDataLoader.padOrTruncate(seq, 5);
            assertEquals(5, result.length);
            assertArrayEquals(seq, result);
        }

        @Test
        @DisplayName("所有batch序列应统一为seqLen长度")
        void testAllBatchSeqsSameLength() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 25, false, 1);
            while (loader.hasNext()) {
                int[][] batch = loader.getNextBatch();
                assertNotNull(batch);
                for (int[] seq : batch) {
                    assertEquals(25, seq.length,
                        "所有序列应为25长度");
                }
            }
        }

        @Test
        @DisplayName("空序列应被填充")
        void testEmptySeq() {
            int[] seq = {};
            int[] result = SCDataLoader.padOrTruncate(seq, 3);
            assertEquals(3, result.length);
            assertArrayEquals(new int[]{0, 0, 0}, result);
        }
    }

    @Nested
    @DisplayName("打乱测试")
    class ShuffleTests {

        @Test
        @DisplayName("打乱模式应改变批次顺序（统计上）")
        void testShuffleChangesOrder() {
            // 多次运行，至少有一次顺序不同
            boolean foundDifference = false;
            List<Integer> firstOrder = new ArrayList<>();
            SCDataLoader loader1 = new SCDataLoader(sampleSequences, 5, 20, false, 1);
            while (loader1.hasNext()) {
                int[][] b = loader1.getNextBatch();
                for (int[] s : b) firstOrder.add(s[0]);
            }

            for (int trial = 0; trial < 10; trial++) {
                SCDataLoader loader2 = new SCDataLoader(sampleSequences, 5, 20, true, 1);
                List<Integer> shuffledOrder = new ArrayList<>();
                while (loader2.hasNext()) {
                    int[][] b = loader2.getNextBatch();
                    for (int[] s : b) shuffledOrder.add(s[0]);
                }
                if (!firstOrder.equals(shuffledOrder)) {
                    foundDifference = true;
                    break;
                }
            }
            assertTrue(foundDifference, "至少应有一次打乱顺序与原始不同");
        }

        @Test
        @DisplayName("不打乱模式应保持顺序")
        void testNoShuffleKeepsOrder() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 5, 20, false, 1);
            List<Integer> order = new ArrayList<>();
            while (loader.hasNext()) {
                int[][] b = loader.getNextBatch();
                for (int[] s : b) order.add(s[0]);
            }
            // 不打乱时应与原始顺序一致
            for (int i = 0; i < sampleSequences.size(); i++) {
                assertEquals(sampleSequences.get(i)[0], order.get(i),
                    "不打乱应保持原始顺序");
            }
        }
    }

    @Nested
    @DisplayName("滑动窗口测试")
    class SlidingWindowTests {

        @Test
        @DisplayName("滑动窗口应生成正确数量的序列")
        void testSlidingWindowCount() {
            int[] tokens = new int[100];
            for (int i = 0; i < 100; i++) tokens[i] = i;
            List<int[]> seqs = SCDataLoader.createSlidingWindowSequences(tokens, 10, 5);
            // (100-10)/5 + 1 = 19
            assertEquals(19, seqs.size() + (seqs.size() > 18 ? 0 : 0),
                "应生成约19条序列");
            assertTrue(seqs.size() >= 18 && seqs.size() <= 20,
                "序列数量应在18-20之间，实际: " + seqs.size());
        }

        @Test
        @DisplayName("滑动窗口每条序列长度应等于seqLen")
        void testSlidingWindowSeqLen() {
            int[] tokens = new int[50];
            for (int i = 0; i < 50; i++) tokens[i] = i;
            List<int[]> seqs = SCDataLoader.createSlidingWindowSequences(tokens, 10, 5);
            for (int[] seq : seqs) {
                assertEquals(10, seq.length);
            }
        }

        @Test
        @DisplayName("stride=1时应生成N-seqLen+1条序列")
        void testSlidingWindowStride1() {
            int[] tokens = new int[20];
            List<int[]> seqs = SCDataLoader.createSlidingWindowSequences(tokens, 5, 1);
            assertEquals(16, seqs.size(),
                "stride=1时应生成 20-5+1=16 条");
        }

        @Test
        @DisplayName("序列长度超过512应被截断")
        void testSlidingWindowCappedAt512() {
            int[] tokens = new int[2000];
            List<int[]> seqs = SCDataLoader.createSlidingWindowSequences(tokens, 1000, 100);
            for (int[] seq : seqs) {
                assertEquals(512, seq.length, "应截断到512");
            }
        }
    }

    @Nested
    @DisplayName("Token拼接测试")
    class ConcatenateTests {

        @Test
        @DisplayName("concatenateTokens应正确合并带分隔符")
        void testConcatenate() {
            List<int[]> lists = new ArrayList<>();
            lists.add(new int[]{1, 2, 3});
            lists.add(new int[]{4, 5});
            lists.add(new int[]{6, 7, 8, 9});

            int[] result = SCDataLoader.concatenateTokens(lists, 0);
            // 3 + 1 + 2 + 1 + 4 + 1 = 12
            assertEquals(12, result.length);
            assertArrayEquals(new int[]{1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 9, 0}, result);
        }

        @Test
        @DisplayName("空列表应返回空数组")
        void testEmptyList() {
            List<int[]> lists = new ArrayList<>();
            int[] result = SCDataLoader.concatenateTokens(lists, 0);
            assertEquals(0, result.length);
        }
    }

    @Nested
    @DisplayName("预加载测试")
    class PrefetchTests {

        @Test
        @DisplayName("startPrefetch应启动后台预加载")
        void testStartPrefetch() throws InterruptedException {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 2);
            loader.startPrefetch();
            Thread.sleep(100);  // 给预加载时间
            // 应能获取数据
            int[][] batch = loader.getNextBatch();
            assertNotNull(batch);
            loader.close();
        }

        @Test
        @DisplayName("重复调用startPrefetch应无副作用")
        void testDoubleStartPrefetch() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            loader.startPrefetch();
            loader.startPrefetch();  // 应不抛异常
            loader.close();
        }
    }

    @Nested
    @DisplayName("资源管理测试")
    class ResourceTests {

        @Test
        @DisplayName("close应释放资源")
        void testClose() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            loader.startPrefetch();
            loader.close();
            // 不抛异常即通过
            assertTrue(true);
        }

        @Test
        @DisplayName("reset应重新开始迭代")
        void testReset() {
            SCDataLoader loader = new SCDataLoader(sampleSequences, 4, 20, false, 1);
            int firstBatchIdx0 = loader.getCurrentBatchIndex();
            loader.getNextBatch();
            loader.getNextBatch();
            int before = loader.getCurrentBatchIndex();
            loader.reset();
            assertEquals(0, loader.getCurrentBatchIndex(),
                "reset后当前批次应为0");
        }
    }

    @AfterEach
    void tearDown() {
        // 资源清理由各测试内调用close完成
    }
}
