import torch
import pickle

with open('data/cpp_conv1.pkl', 'rb') as f:
    cpp_conv1 = pickle.load(f)

with open('data/tdefault_conv1.pkl', 'rb') as f:
    conv1 = pickle.load(f)

# Define epsilon
epsilon = 1e-5

# Detach the tensors from the computation graph
cpp_conv1 = cpp_conv1.detach()
conv1 = conv1.detach()

# Wiping the 63rd channel from both
# cpp_conv1[:, 63, :, :] = 0
# conv1[:, 63, :, :] = 0

# Compute the absolute difference and create a mask
difference_mask = torch.abs(cpp_conv1 - conv1) > epsilon
different_indices = torch.nonzero(difference_mask)

for idx in different_indices:
    batch, channel, row, col = idx
    print(f"Difference at index (batch={batch}, channel={channel}, row={row}, col={col}):")
    print(f"cpp_conv1: {cpp_conv1[batch, channel, row, col]}")
    print(f"conv1: {conv1[batch, channel, row, col]}")
