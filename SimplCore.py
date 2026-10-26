# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""兼容旧代码的 `SimplCore` 大写包名。

真实包名为 `simplcore`。本文件让历史脚本中的
`import SimplCore...` 在大小写敏感的 Python 导入系统中继续可用。
"""

from __future__ import annotations

import sys

import simplcore as _simplcore

globals().update(_simplcore.__dict__)
__path__ = _simplcore.__path__
__package__ = "SimplCore"

sys.modules.setdefault("SimplCore", sys.modules[__name__])
