import torch
import torch.nn.functional as F
import numpy as np

# Create a random input tensor

# Setting the seed for reproducibility
torch.manual_seed(42)
size = 25
# Determine how many elements should be NaN
num_nans = size // 3

# Create a random permutation of the indices
indices = torch.randperm(size)
input_tensor = torch.arange(1,26)
# Convert the integer tensor to float
input_tensor = input_tensor.float()
# Set the elements at the indices to NaN
input_tensor[indices[:num_nans]] = float('nan')

# Reshape the tensor back to its original shape
input_tensor = input_tensor.reshape(1, 1, 5, 5)

print("Input Tensor: ")
print(input_tensor)
print()

# Create a random weights tensor of integers
weight = torch.ones(1, 1, 2, 2)

# Convert the integer tensor to float
weight = weight.float()

print("Weight Tensor: ")
print(weight)
print()

## Printing hardware device
# print("Printing hardware device: ")
# print(input_tensor.device)
# print(weight.device)
# print()

# Apply the convolution operation
output_tensor = F.conv2d(input_tensor, weight, padding=0)

# Print the shape of the output tensor
print("Output after convolution: ")
print(output_tensor)
print(output_tensor.shape)