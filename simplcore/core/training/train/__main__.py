# Copyright 2026 (C) 六安市青奈科技有限责任公司
"""``python -m simplcore.core.train`` / ``torchrun -m ...`` 兼容入口。"""
from __future__ import annotations

from .main import main

raise SystemExit(main())
