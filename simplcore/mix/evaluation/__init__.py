# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 评估系统包。

重新导出 evaluation.py 中定义的 Evaluator 和 FactualEvaluator。
"""
# 当 evaluation/ 目录存在时，它会遮蔽 evaluation.py 文件。
# 这里通过 importlib 显式加载 evaluation.py 的内容。
import importlib.util
from pathlib import Path

_eval_py = Path(__file__).parent.parent / "evaluation.py"
if _eval_py.exists():
    _spec = importlib.util.spec_from_file_location(
        "simplcore.mix.evaluation_py", _eval_py
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Evaluator = _mod.Evaluator
    FactualEvaluator = _mod.FactualEvaluator
else:
    # 回退：如果 evaluation.py 不存在，提供基础实现
    class Evaluator:
        pass
    class FactualEvaluator:
        pass

__all__ = ["Evaluator", "FactualEvaluator"]
