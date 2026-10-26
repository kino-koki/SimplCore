# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from SimplCore.engine import randn, LSTM

print("Debugging LSTM...")
x = randn(32, 10, 256)
print(f"x type: {type(x)}")
print(f"x._data type: {type(x._data)}")
print(f"x._data shape: {x._data.shape}")
print(f"x._data ndim: {x._data.ndim}")

try:
    transposed = x._data.transpose(0, 1)
    print(f"Transposed shape: {transposed.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\nChecking if _data is actually a numpy array...")
print(f"Is numpy array: {isinstance(x._data, np.ndarray)}")
print(f"_data dtype: {x._data.dtype}")
print(f"_data flags: {x._data.flags}")

print("\nChecking randn implementation...")
from SimplCore.engine.tensor import randn as randn_fn
x2 = randn_fn(32, 10, 256)
print(f"x2._data type: {type(x2._data)}")
print(f"x2._data shape: {x2._data.shape}")

try:
    transposed2 = x2._data.transpose(0, 1)
    print(f"x2 transposed shape: {transposed2.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
