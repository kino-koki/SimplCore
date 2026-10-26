# 学习记录：2026-10-26 架构对话

> 本文件是"对话的同时进行深度学习"的落点。架构决策的**理由**沉淀在这里，
> **规格**在 `docs/SGC_SPEC.md`，**路线**在 `docs/ARCHITECTURE_PIVOT.md`。

## 触发

指令：完善 SimplCore 到世界顶级；彻底抛弃 PyTorch 与 Transformer；多模态从架构钉住；
优化训练 HBM；持续优化 SGC；全量删除 v2.0 等后缀；对话同时深度学习；四层闸门抗幻觉。

## 关键发现（改变了问题的性质）

指令的前提"仓库还在用 PyTorch"**只对一半**：

1. `simplcore/tensor/` 已是**自研张量引擎**，含反向模式自动微分（拓扑排序 + 梯度累积）、
   AMP、梯度检查点、集合通信、显存预估，后端为 NumPy/CuPy/OpenCL，**代码里显式拒绝
   torch 后端**。零依赖是**既成设计**，不是待办。
2. `simplcore/mix/` 走的正是 `import simplcore.tensor as torch` 的非 PyTorch 路线。
3. 真正的差距只有一条：**最强的架构（SGC）跑在 PyTorch 上；零依赖的那条路不是 SGC，
   且被刻意限定为纯文本**。

因此正确动作是**移植 + 钉桩**，不是重写。

## 四个值得记住的技术结论

### 1. 零依赖 ≠ 零成本（最重要）

自研引擎的自动微分是 **Python 层 tape**。1.3B/seq4096 单步算子调用 1.5e4~5e4 次，
每次 Python+dispatch 5~50 μs → **纯 Python 开销 0.4~2.5 s/step**，GPU 利用率会被
压到个位数百分比。PyTorch 的价值不在"能算"，而在**把算子调度下沉到 C++/CUDA 之后
仍然能自动微分**。

→ 结论：自研化 = "把热路径编译掉之后重新实现自动微分"。`simplcore/engine/csimplcore.cpp`
是唯一路径；`simplcore/tensor` 与 `simplcore/engine` 两套引擎必须合并。

### 2. 闸门要保住对偶形式，统计量必须是因果前缀量

SGC 的核心资产是 **FFT（训练）↔ 二阶 IIR（推理）** 的严格对偶。任何"随内容变化"的
调制若在平行路径与流式路径算法不同，就会打掉这份资产。

- L2 证据门最初设计为**逐谱基**的动态权重（`e_m` 依赖输入统计）→ **行不通**：
  核变成位置相关，FFT 单次卷积无法表达；流式路径也没有对应实现。
- 改为**逐位置标量** + **因果前缀平均能量**（平行用 `cumsum`、流式用运行和）→
  两者一致，闸门不破坏对偶。**这是本次设计的关键取舍。**

### 3. 零初始化会静默废掉一个闸门

`evidence_proj` 初版权重全零 → 闸门对输入完全不敏感，恒为 sigmoid(0)=0.5，形同虚设；
更隐蔽的是 `SGCModel._init_weights` 会对**所有** `nn.Linear` 重施 `normal(0, init_std)`，
把精心设的初值冲掉。

→ 结论：闸门用**标量参数**（强度/偏置）而非 Linear。响应形状是结构性的，只有强度可学；
既避免零初始化退化，也不受模型级重初始化影响。

### 4. 解析式参数计数是"架构契约"，改结构必须同步

新增闸门标量后，`analytic_params` / `analytic_hybrid_params` 与真实模型差了 8
（2 层 × 4 标量），8 项测试立刻报警。**这类解析式函数是把架构写成公式的守护装置**，
任何结构变更都要同步 —— 它的失败不是麻烦，是它在工作。

## 已交付

- 去版本化重构：包 `sgc2`→`sgc`，`SGC2`→`SGC`，预设 `sgc2-*`→`sgc-*`，
  规格文档 `SGC2_SPEC.md`→`SGC_SPEC.md`，63 个文件 + 路径级 `git mv`；**511 项测试全绿**。
- 四层闸门 L1~L4：`simplcore/sgc/mixer.py`（L2/L3）+ `simplcore/sgc/abstain.py`（L4），
  18 项守护测试（对偶形式 / 因果性 / 梯度可达 / 可消融 / 行为）。
- 架构蓝图 `docs/ARCHITECTURE_PIVOT.md`（含能力矩阵、算子清单、显存预算、路线图、风险登记）。

## 待办（下一轮）

1. `simplcore/tensor` 补可微 `rfft/irfft`（反向用重算法：梯度 cotangent 再做一次正变换）。
2. SGC mixer 移植到自研引擎，逐算子对齐 PyTorch 参考（`max|Δ| ≤ 1e-5`）。
3. `GatedSpectralConv2`（`gate_ab` 分支）补 L2/L3，抹平闸门能力的不一致。
4. 多模态钉桩：`MultimodalConfig` 冻结 + 统一 token 化 + 模态路由门。
5. HBM 实测归档（`scripts/probe_batch.py` 是唯一来源，不许算术填表）。

---

## 补充：仓库清理（同日）

### 根因不是"文件多"，是四类东西混在一起

| 类别 | 例子 | 体积 |
|---|---|---|
| vendored 第三方依赖 | `libs/` 里塞了 2.4 GB 的 torch/sympy/numpy 源码 | 2.4 GB |
| 多个并行虚拟环境 | `.venv` `.venv21` `.venv311` `.venv_sc`（4 个） | 12 GB |
| 可重建的数据 | `data/` 语料、`runs/`、`tokenized_cache/` | 6.6 GB |
| 无 `.gitignore` | 253 个未跟踪文件，每次 `git status` 都是噪音 | — |

**`libs/` 是最隐蔽的一个**：它看起来像"运行时依赖"，但仓库里引用它的 25 个文件
用的都是**失效的绝对路径**（指向早已不存在的旧项目目录
`G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0`）。真正的 torch 来自 venv。
它 2.4 GB，零有效引用。

### 一条容易被忽略的因果链

旧项目目录改名/迁移 → 代码里的硬编码绝对路径失效 → `sys.path.insert` 变成空操作 →
`libs/` 失去存在理由却没人删 → 仓库体积失控。
**目录搬家时，硬编码路径是首要清理项**，比整理文件优先级高。

### 三类"死文件"（语法都过不了，等于从未被 import 过）

1. **双 BOM**：`EF BB BF EF BB BF` 开头 → `SyntaxError: invalid non-printable character`
2. **截断残片**：只剩 `sys.exit(0 if success else 1)` 一行的 `test_*.py`
3. **未转义引号**：中文内容里直接写 `"过去的过去"` → 字符串提前闭合

这类文件**全部未被 git 跟踪**，是某次写入事故的产物。判断依据：
`git ls-files` 查不到 + 体积异常小 + 内容是片段。

### 慢盘上删大目录的工程教训（G: 盘）

| 方法 | 速度（134 文件样本） | 备注 |
|---|---|---|
| `rm -rf`（Git Bash） | 2113 ms（16 ms/文件） | 超深目录树会**中途失败**，需重试 |
| `cmd /c rd /s /q` | 651 ms（5 ms/文件） | 快 3.2×，但路径含非 ASCII（`α`）时编码会坏 |
| `find -depth -delete` | 中等 | 最稳，**自底向上**逐层删，失败可重试 |

三条纪律：
* **删前先 `du` 拿基线，但不要在删除路径里 `du`** —— 那一遍遍历会白白吃掉一倍时间
* **深目录树用 `find -depth -delete` 并带重试循环**，不要指望 `rm -rf` 一次到位
* **8.3 短路径在路径含非 ASCII 时同样不可用**，不要指望用它绕开编码问题

### 删之前必须回答的三个问题

1. **它能被重建吗？** 重建命令写进文档（`docs/CLEANUP_2026-10-26.md`）
2. **它被 git 跟踪吗？** 未跟踪 = 纯垃圾；已跟踪 = 删除是历史变更，要留痕
3. **它是别的脚本的依赖吗？** `libs/` 的案例：引用它的路径全是失效的 → 零有效依赖


---

## 补记：删完之后才发现的第四个问题——**悬挂对象**

删完工作区文件、体积从 15.8 GB 降到 2.16 GB，看起来结束了。但 `git count-objects -vH`
显示 `size-pack: 716 MB`，而历史里最大的 blob 只有几十 KB。

原因：早前某次 `git add` 把 `data/`、`libs/` 的内容写进了对象库。即使**从未提交**，
这些 blob 仍然存在于 pack 中，`git status` 看不到它们，`git ls-files` 也查不到。

一个 `git gc --prune=now`：**716 MB → 6.45 MB**。

### 教训

* **工作区体积 ≠ 仓库体积。** 双份：`du` 看工作区，`git count-objects -vH` 看 `.git`。
* **`git add` 会把 blob 落地，哪怕你马上 `git reset`。** 大目录先写 `.gitignore`，
  再 `git add -A`。
* **清理的最后一公里是 `git gc`。** 不做，前 90% 的工作只回收了空间的一半。

### 判断标准（下次直接照做）

```
du -sm .                          # 工作区
git count-objects -vH | grep size-pack   # 仓库
```
两者差一个数量级 → 有悬挂对象，跑 `git gc --prune=now`。
