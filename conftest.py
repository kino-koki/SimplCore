# 让 pytest 能将工作区根目录加入 sys.path，从而 `import simplcore_pg` 可用。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
