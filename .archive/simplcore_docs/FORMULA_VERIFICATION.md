# SimplCore 公式正确性验证方案

> 目的：对 SimplCore 各核心模块中涉及的数学公式（尤其是"上下文记忆遗忘程度"
> 与"O(1) 计算/空间特性"）进行逻辑与实现层面的核验，给出可重复的验证方法、
> 指标、预期输出与未决风险。

---

## 0. 待验证公式清单（来源文件）

| # | 模块 | 公式 / 声明 | 位置 |
|---|------|------------|------|
| F1 | persistent_memory | 遗忘价值 `value = access_count / (1 + decay_rate·age)` | `forget_step()` |
| F2 | persistent_memory | 驱逐掩码 `mask = value < 1/(1 + min_access_to_keep)` | `forget_step()` |
| F3 | persistent_memory | 门控写入 `erase = sigmoid(gate)·slot_strength; new = mem·(1−erase) + erase·v` | `write()` |
| F4 | persistent_memory | 空间复杂度 **O(1)**（形状 `(C,S,D)` 与 n 无关） | `__init__` / `stats()` |
| F5 | lattice_model | 谐振耦合 FLOPs `≈ B·D·(L−K+1)·K·Bn`（O(n)） | `ResonantCoupling.complexity` |
| F6 | lattice_model | 哈希记忆 FLOPs `≈ B·L·C·S·D`（O(n)，**非 O(1)**） | `HashMemory.complexity` |
| F7 | lattice_model | 参数计数 `per_block` 与 totals 公式 | `parameter_count()` |

> ⚠️ 修正点：原 `HashMemory` 将复杂度标注为 `"O(1) wrt n"`，已修正为
> `"O(n) wrt n; capacity O(1) wrt n"`。其**单步计算**随序列长度 L 线性增长，
> 不存在 n² 项；真正相对 n 不变的是**记忆容量**（形状不含 L）。

---

## 1. 验证方法总览

| 方法 | 用途 | 工具 |
|------|------|------|
| 理论推导 / 量纲检查 | 确认公式在数学上自洽、量纲正确、边界合理 | 纸面 + 本文件推导 |
| 单元测试（pytest） | 对函数/类行为做确定性断言 | `verify_formulas.py`（tensor 集成测试） |
| 数值对比 | 固定规模 vs 增长规模，验证缩放律符合声明复杂度 | 纯 Python 数值脚本（无需 numpy，本环境可跑） |
| 边界/符号测试 | 极端输入（空、零、极大 age）下公式行为 | 单元测试 |

---

## 2. 逐项验证设计

### F1 — 遗忘价值公式
**验证指标**
- 单调性：`value` 随 `age` 单调非增；`age→∞` 时 `value→0`。
- 单调性：`value` 随 `access_count` 单调非减；`access_count=0` 时 `value=0`。
- 边界：`age<0` 取 `max(age,0)` 防护；`decay_rate=0` 时退化为纯频率。
- 量纲：`value` 无量纲（频率 / 无量纲比值），与 `min_access_to_keep` 同量级可比。

**方法**：构造 `[access_count, age]` 网格，断言上述三项；并验证 `value < 1/(1+min_access_to_keep)` 的命中阈值确实把"久未访问且低频"slot 判为可衰减。

**预期输出**：网格测试中至少 12 组断言全过；`age→large` 时 `value≈0`。

### F2 — 驱逐掩码
**验证指标**：`mask` 仅当 `value` 低于阈值时为真；参数敏感性——`min_access_to_keep` 越大越易驱逐。

**方法**：给定 `value` 序列 `[0.1, 0.5, 1.2]`（`min_access_to_keep=1.0` → 阈值 0.5）：预期 `[True, False, False]`。

**预期输出**：断言完全匹配；`decay_factor` 在 mask 处为 0.95，否则 1.0。

### F3 — 门控写入 (NTM 擦除-添加)
**验证指标**
- 无擦除（`gate=0` 或 `slot_strength=0`）：`new == mem`。
- 满擦除（`erase=1`）：`new == v`。
- 部分擦除：`new == mem·(1−e) + e·v` 逐元素成立。
- 写入后 `clamp(-10,10)` 不溢出；命中 slot 的 `access_count += 1`。

**方法**：用 numpy/tensor 构造小规模 `(C,S,D)`，直接复算并比对。

### F4 — 持久记忆空间 O(1)
**验证指标**：写入任意数量 token 后，`stats()['params_memory_bytes']` 恒定 = `C·S·D·4`。

**方法**（纯逻辑，本环境可跑）：形状由 `(num_channels, slot_size, dim)` 决定，与 L 无关——直接断言 `len` 不随写入次数变化（见脚本 `verify_o1_space`）。

### F5/F6 — FLOPs 缩放律
**验证指标**：`rc_flops` 应是 L 的**线性**函数（二阶差分 ≈ 0）；`hm_flops` 同。
二者相对 attention 的 `speedup = attn_flops / x_flops` 应随 L **递增**（因 attn 是 n²，x 是 n）。

**方法**（纯 Python，本环境可跑）：取 L∈{256,1024,4096,16384,65536}，断言
`rc_flops[L2]-2*rc_flops[L1]+rc_flops[L0] ≈ 0`（线性）；并断言 `speedup(65536) > speedup(256)`。

**预期输出**：二阶差分均 < 1e-6（相对）；加速比单调递增。

### F7 — 参数计数
**验证指标**：用极小配置（HIDDEN=16, LAYERS=2）手算 `per_block` 与 totals，与 `parameter_count` 输出比对。

**预期输出**：误差为 0（整数精确匹配）。

---

## 3. 验证脚本

`verify_formulas.py` 提供两层：
- **纯 Python 层**（`verify_*` 函数）：不依赖 numpy，**当前环境可直接运行**，输出 PASS/FAIL。
- **Tensor 集成层**（`verify_tensor_*`）：使用 `simplcore.tensor`，需联网安装 numpy 后运行，做真实 reshape/einsum 断言。

运行：
```bash
python verify_formulas.py            # 纯 Python 验证（无需 numpy）
python verify_formulas.py --torch    # 含 tensor 集成测试（需 numpy）
```

---

## 4. 未决风险（必须披露）

1. **persistent_memory 读取仍含 softmax 注意力**：`read()` 用
   `F.softmax(scores, dim=-1)` 做内容寻址加权，本质是**注意力变体**，与
   "彻底脱离 Transformer/注意力"目标冲突。建议在 `O(1) 容量 + 确定性寻址`
   方向上重写为 argmax/gather（类似 HashMemory），或将其从"非注意力"声明中移除。
2. **HM 复杂度口径**：已修正标注；但文档/报告历史中若出现过 "O(1) 时间复杂度"
   的表述，需同步更正为 "容量 O(1)，计算 O(n)"。
3. **运行环境**：当前沙箱无法安装 numpy，tensor 集成测试未实际执行；纯逻辑验证已通过可视为弱验证，正式验收需在可装 numpy 的环境复跑 tensor 层。

---

## 5. 预期结果汇总（形式）

最终交付：
- `FORMULA_VERIFICATION.md`（本文件）：验证规范与结论。
- `verify_formulas.py`：可执行验证脚本（纯 Python 部分已在本环境跑通）。
- 若 tensor 层通过：打印 `ALL TESTS PASSED`；否则列出具体 FAIL 行与期望/实际差值。
