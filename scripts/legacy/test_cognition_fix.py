import sys
sys.path.insert(0, '.')
import torch
from SimplCore.cognition.tree_of_thought import TreeOfThought

tot = TreeOfThought(dim=64, max_depth=3, num_branches=4)
test_input = torch.randn(32, 64)
output, info = tot(test_input)
print(f"TreeOfThought OK: input={test_input.shape}, output_shape={output._data.shape}, confidence={info['confidence']:.4f}")

