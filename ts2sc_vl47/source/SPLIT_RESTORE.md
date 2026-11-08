# MiniCPM-V-4.7-1B 模型文件重组说明

模型权重 `model.safetensors` 已按 95MB 分片存储为 `vpart_00` ~ `vpart_31`（共 32 片），以绕过 GitHub 单文件 100MB 限制。

## 重组方法

在包含分片的目录下执行（Windows Git Bash / Linux / macOS）：

```bash
cat vpart_* > model.safetensors
```

## 校验

重组后的 `model.safetensors` 应为 **3,109,054,480 字节**（3.11GB）。

```bash
ls -l model.safetensors   # 应为 3109054480
```

或校验所有分片总大小：

```bash
cat vpart_* | wc -c        # 应为 3109054480
```

## 分片清单

`vpart_00` ~ `vpart_31`，每片 99,614,720 字节（95MB），末片为余数。
