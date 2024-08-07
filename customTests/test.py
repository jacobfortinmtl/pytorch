import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

# Set the number of threads to 1
torch.set_num_threads(1)

# Setting the seed for reproducibility
torch.manual_seed(42)
channels = 1
batches = 1
size = 4**2
sqrt_size = int(math.sqrt(size))
# Determine how many elements should be NaN
num_nans = int(size * 0.33)

# Create a random permutation of the indices
indices = torch.randperm(size)
input_tensor = torch.arange(1, size*channels*batches + 1).float()

# Reshape the tensor to the desired shape
input_tensor = input_tensor.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

# Set the elements at the indices to NaN for each channel
for c in range(input_tensor.shape[1]):
    input_tensor[0, c].view(-1)[indices[:num_nans]] = float('nan')

print("Input Tensor: ")
print(input_tensor)

# Create a random weights tensor of integers
weight = torch.ones(1, 1*channels, 2, 2).float()

# Apply the convolution operation
print("Size: ", sqrt_size**2)
start_time = time.time()
output_tensor = F.conv2d(input_tensor, weight, padding=0)
end_time = time.time()
elapsed_time = end_time - start_time

# Print the shape of the output tensor
print("Output after convolution: ")
print(output_tensor)

# Print elapsed time
print(f"Time taken for convolution: {elapsed_time} seconds")
print("End of call ---- \n")