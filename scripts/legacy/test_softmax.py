# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SimplCore.engine import randn

print("Testing softmax...")
x = randn(32, 1024)
print(f"x type: {type(x)}")
print(f"x has _data: {hasattr(x, '_data')}")

try:
    y = x.softmax(dim=-1)
    print(f"y shape: {y.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("Testing softmax function directly...")
from SimplCore.engine.tensor import softmax
try:
    y = softmax(x, dim=-1)
    print(f"y shape: {y.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
