# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore Tensor 版本信息 (torch.version 兼容占位).

提供 ``version`` 字符串与 ``hip`` 占位, 用于兼容 torch.version.hip 检测逻辑.
"""

# 引擎版本
version = "1.0.0-simplcore-native"

# HIP / ROCm 检测占位 (原生引擎无 GPU, 恒为空)
hip = ""

# 后端标识
__backend__ = "numpy"
