"""simplcore.core.eval 子包（0.1-α 结构化拆分）。

同名旧模块 simplcore.core.eval 的全部公共 API 在此平铺 re-export，
使 ``from simplcore.core.eval import X`` 在新旧两种布局下语义一致。
"""
from . import eval as _impl

_globals = globals()
for _k, _v in vars(_impl).items():
    if not (_k.startswith('__') and _k.endswith('__')):
        _globals.setdefault(_k, _v)
del _globals, _k, _v, _impl
