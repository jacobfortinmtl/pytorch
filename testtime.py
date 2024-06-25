import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

# Function to perform convolution and measure time
# def measure_convolution_time(size):
#     sqrt_size = int(math.sqrt(size))
#     if sqrt_size * sqrt_size != size:
#         return None
    
#     # Determine how many elements should be NaN
#     num_nans = size // 3

#     # Create a random permutation of the indices
#     indices = torch.randperm(size)
#     input_tensor = torch.arange(1, size + 1)
#     # Convert the integer tensor to float
#     input_tensor = input_tensor.float()
#     # Set the elements at the indices to NaN
#     input_tensor[indices[:num_nans]] = float('nan')

#     # Reshape the tensor back to its original shape
#     input_tensor = input_tensor.reshape(1, 1, sqrt_size, sqrt_size)

#     # Create a random weights tensor of integers
#     weight = torch.ones(1, 1, 2, 2).float()

#     # Apply the convolution operation
#     start_time = time.time()
#     output_tensor = F.conv2d(input_tensor, weight, padding=0)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     return elapsed_time

# # Write the time taken for each size to a file
# with open('..//plots//convolution_times_custom.txt', 'w') as f:
#     for size in range(5**5, 1000*1000 + 1):
#         elapsed_time = measure_convolution_time(size)
#         if elapsed_time is not None:
#             f.write(f"Size: {size}, Time taken: {elapsed_time} seconds\n")
#             print(f"Size: {size}, Time taken: {elapsed_time} seconds")


# uncomment above to write to the files
# uncomment below to plot the files

def read_data_from_file(file_path):
    sizes = []
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            size_part = parts[0].split(': ')[1]
            time_part = parts[1].split(': ')[1].split(' ')[0]
            sizes.append(int(size_part))
            times.append(float(time_part))
    return sizes, times

# Read data from both files
sizes1, times1 = read_data_from_file('..//plots/convolution_times_custom.txt')
sizes2, times2 = read_data_from_file('..//plots/convolution_times_torch.txt')
sizes3, times3 = read_data_from_file('..//plots/convolution_times_no_insert.txt')

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(sizes1, times1, marker='o', label='Squash - Insert')
plt.plot(sizes3, times3, marker='s', label='Squas - No Insert')
plt.plot(sizes2, times2, marker='x', label='Torch')
plt.title('Time taken for convolution vs. Size')
plt.xlabel('Size')
plt.ylabel('Time taken (seconds)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation on the x-axis
plt.savefig('..//plots/convolution_time_comparison.png')
plt.show()