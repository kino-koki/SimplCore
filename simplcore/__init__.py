# 必须在任何子模块导入 torch / tempfile 之前执行：C 盘只剩 7.8G，
# 缓存一律写 E 盘。详见 simplcore/_cache_env.py 里的原因说明。
from . import _cache_env as _cache_env

_cache_env.apply()

__author__ = "青奈科技青小团团队"
