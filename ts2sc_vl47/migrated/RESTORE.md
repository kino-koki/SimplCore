# 分片说明

本包 **111 个分片都是独立合法的 safetensors 文件**，不需要 `cat` 重组。
按 `model.safetensors.index.json` 的 `weight_map` 按 key 定位即可，
`scripts/load_vl1b.py` 已实现（零 torch 依赖）。

有 4 个张量因超过单分片上限（24 MiB）被按 axis 0 切开，见 `slices.json`：

| 原张量 | 原 shape | 块数 | 还原 |
|---|---|---:|---|
| `model.embed_tokens.weight` | `[151936, 1024]` | 16 | `np.concatenate(parts, axis=0)` |
| `vision_tower.vit_merger.linear_1.weight` | `[17216, 4608]` | 8 | 同上 |
| `vision_tower.vit_merger.linear_2.weight` | `[1152, 17216]` | 2 | 同上 |
| `merger.mlp.0.linear_1.weight` | `[4608, 4608]` | 2 | 同上 |

拼接为同参数拆分，**不增加参数量**。

`lm_head.weight` 与 `model.embed_tokens.weight` tie（逐位相同），
不在 index 里，加载时直接指向 embedding。

若确实需要拼回单一文件（例如在支持大文件的机器上）：

```bash
python3 - <<'PY'
from pathlib import Path
import sys; sys.path.insert(0, "scripts")
from load_vl1b import VL1B
m = VL1B(".")
# 逐张量写出由调用方自行实现；通常不需要。
print(m.summary())
PY
```
