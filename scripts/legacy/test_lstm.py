# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SimplCore.engine import randn, LSTM

print("Testing LSTM with single layer...")
lstm = LSTM(256, 256, num_layers=1, batch_first=True)
x = randn(32, 10, 256)
print(f"Input shape: {x.shape}")

try:
    y, _ = lstm(x)
    print(f"Output shape: {y.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\nTesting LSTM with num_layers=1, hidden_size=256, bidirectional=False...")
lstm2 = LSTM(256, 256, num_layers=1, bidirectional=False, batch_first=True)
try:
    y, _ = lstm2(x)
    print(f"Output shape: {y.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\nTesting LSTM with num_layers=2...")
lstm3 = LSTM(256, 128, num_layers=2, batch_first=True)
x2 = randn(32, 10, 256)
try:
    y, _ = lstm3(x2)
    print(f"Output shape: {y.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
