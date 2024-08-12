import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle

# kernel = torch.randn(2, 2, 3, 3)
# inputs = torch.randn(1, 2, 5, 5)
# inputs[:] = float('nan')

# output = torch.zeros((1,2,5,5))
# output[:,:, 1:-1, 1:-1] = float('nan')
# mask = torch.isnan(output)
# output[mask] = 1

# nanoutput = F.conv2d(inputs, kernel, stride=1, padding=0)
# print(inputs)
# print(kernel)
# print(nanoutput)

data= torch.tensor([[[[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.1093]]],


        [[[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]]]])

kernel = torch.tensor([[[[-1.6383e-02, -4.6532e-02,  2.7559e-02],
          [ 1.2042e-02, -7.8450e-02,  4.4842e-02],
          [ 2.9786e-03, -3.0273e-02, -2.7136e-02]],

         [[-2.0751e-02, -2.3382e-03, -1.2575e-02],
          [-9.4025e-03, -4.7025e-02,  3.5216e-03],
          [ 3.1999e-02,  1.9897e-02,  1.9879e-02]],

         [[ 1.6916e-03,  6.8086e-02,  2.2535e-02],
          [-7.2501e-03, -1.3292e-01, -2.3621e-02],
          [-1.2451e-02, -4.6858e-02, -3.5012e-03]],

         [[ 2.5113e-02, -4.2966e-02,  3.3374e-03],
          [-8.0081e-02,  4.8635e-02,  7.5314e-02],
          [ 7.2055e-02, -1.2497e-01, -4.4701e-02]],

         [[-1.3323e-02,  1.6450e-02, -1.4601e-02],
          [ 8.1429e-02,  1.3582e-01, -8.2534e-03],
          [-4.4671e-02, -1.7206e-02,  4.1633e-02]],

         [[-6.8414e-03, -2.1228e-02,  1.1259e-04],
          [ 5.8388e-03, -4.6491e-02, -4.3068e-03],
          [-5.3104e-03, -4.9091e-02,  5.8290e-03]],

         [[ 3.8621e-03,  1.0240e-02,  1.6323e-02],
          [-1.0641e-03, -2.3820e-02,  6.1980e-03],
          [ 2.3955e-02, -3.1064e-02,  8.3065e-03]]]])

biases = torch.tensor([0.0127])

print(data.shape)
print(kernel.shape)
output = F.conv2d(data, kernel, bias = biases, padding=0)
print(output[0, :, :, :])


# # Set the number of threads to 1
# torch.set_num_threads(1)

# # Setting the seed for reproducibility
# torch.manual_seed(42)
# channels = 2
# batches = 2
# size = 4**2
# sqrt_size = int(math.sqrt(size))
# # Determine how many elements should be NaN
# num_nans = int(size * 0.33)

# # Create a random permutation of the indices
# indices = torch.randperm(size)
# input_tensor = torch.arange(1, size*channels*batches + 1).float()

# # Reshape the tensor to the desired shape
# input_tensor = input_tensor.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

# # Set the elements at the indices to NaN for each channel
# for c in range(input_tensor.shape[1]):
#     input_tensor[0, c].view(-1)[indices[:num_nans]] = float('nan')

# print("Input Tensor: ")
# print(input_tensor)

# # Create a random weights tensor of integers
# weight = torch.ones(1*batches, 1*channels, 2, 2).float()

# # Apply the convolution operation
# print("Size: ", sqrt_size**2)
# start_time = time.time()
# output_tensor = F.conv2d(input_tensor, weight, padding=0)
# end_time = time.time()
# elapsed_time = end_time - start_time

# # Print the shape of the output tensor
# print("Output after convolution: ")
# print(output_tensor)

# # Print elapsed time
# print(f"Time taken for convolution: {elapsed_time} seconds")
# print("End of call ---- \n")