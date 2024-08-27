import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

# input = torch.tensor([[[[[-0.2303, -0.3918,  0.5433, -0.3952,  0.2055],
#            [-0.4503, -0.5731, -0.5554, -1.5312, -1.2341],
#            [ 1.8197, -0.5515, -1.3253,  0.1886, -0.0691],
#            [-0.4949, -1.4782,  2.5672, -0.4731,  0.3356],
#            [ 1.5091,  2.0820,  1.7067,  2.3804, -1.0670]],

#           [[ 1.1149, -0.1407,  0.8058,  0.3276, -0.7607],
#            [-1.5991,  0.0185,  0.8419, -0.4000, -0.2282],
#            [ 0.2800,  0.0732,  1.1133,  0.2823,  0.4342],
#            [-0.8025, -1.2952,  0.7813, -0.9268,  0.2064],
#            [-0.3334, -0.4288,  0.2329,  0.7969, -0.1848]]]]])

# kernel = torch.tensor([[[[[-1.1258, -1.1524, -0.2506],
#            [-0.4339,  0.8487,  0.6920],
#            [-0.3160, -2.1152,  0.3223]],

#           [[-1.2633,  0.3500,  0.3081],
#            [ 0.1198,  1.2377,  1.1168],
#            [-0.2473, -1.3527, -1.6959]]],


#          [[[ 0.5667,  0.7935,  0.4397],
#            [ 0.1124,  0.6408,  0.4412],
#            [-0.2159, -0.7425,  0.5627]],

#           [[ 0.2596,  0.5229,  2.3022],
#            [-1.4689, -1.5867,  1.2032],
#            [ 0.0845, -1.2001, -0.0048]]]]])

# print(input.shape)
# print(kernel.shape)
# output = F.conv2d(input[0], kernel[0], padding=1)
# print(output)


# # Set the number of threads to 1
# torch.set_num_threads(1)

# Setting the seed for reproducibility
torch.manual_seed(42)
channels = 2
batches = 1
size = 4**2
sqrt_size = int(math.sqrt(size))
# Determine how many elements should be NaN
num_nans = int(size * 0.33)

# Create a random permutation of the indices
indices = torch.randperm(size)
# input_tensor = torch.arange(1, size*channels*batches + 1).float()
input_tensor = torch.zeros(1*batches, 1*channels, sqrt_size, sqrt_size).float()

# Reshape the tensor to the desired shape
# input_tensor = input_tensor.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

# Set the elements at the indices to NaN for each channel
for c in range(input_tensor.shape[1]):
    input_tensor[0, c].view(-1)[indices[:num_nans]] = float('nan')

print("Input Tensor: ")
print(input_tensor)

# Create a random weights tensor of integers
weight = torch.ones(1*batches, 1*channels, 2, 2).float()

# Apply the convolution operation
print("Size: ", sqrt_size**2)
start_time = time.time()
output_tensor = F.conv2d(input_tensor, weight, padding=1)
end_time = time.time()
elapsed_time = end_time - start_time

# Print the shape of the output tensor
print("Output after convolution: ")
print(output_tensor)

# Print elapsed time
print(f"Time taken for convolution: {elapsed_time} seconds")
print("End of call ---- \n")