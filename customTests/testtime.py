import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Function to perform convolution and measure time
def measure_convolution_time(size):
    sqrt_size = int(math.sqrt(size))
    if sqrt_size * sqrt_size != size:
        return None
    
    # Determine how many elements should be NaN
    num_nans = size // 3

    # Create a random permutation of the indices
    indices = torch.randperm(size)
    input_tensor = torch.arange(1, size + 1)
    # Convert the integer tensor to float
    input_tensor = input_tensor.float()
    # Set the elements at the indices to NaN
    input_tensor[indices[:num_nans]] = float('nan')

    # Reshape the tensor back to its original shape
    input_tensor = input_tensor.reshape(1, 1, sqrt_size, sqrt_size)

    # Create a random weights tensor of integers
    weight = torch.ones(1, 1, 2, 2).float()

    # Apply the convolution operation
    start_time = time.time()
    output_tensor = F.conv2d(input_tensor, weight, padding=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time

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


# The rest of your code follows
if (os.environ.get('WRITE') == '1'):
    # Specify path from environment variable
    filepath = '..//..//plots//'
    os_filepath = os.environ.get('FILENAME')
    if os_filepath is None:
        print("No FILENAME specified. Exiting the program.")
        sys.exit(1)  # Exit with a non-zero value to indicate error
    else:
        filepath += os_filepath + '.txt'
        
    with open(filepath, 'w') as f:  # Use the modified filepath variable
        for size in range(5**5, 1000*1000 + 1):
            elapsed_time = measure_convolution_time(size)
            if elapsed_time is not None:
                f.write(f"Size: {size}, Time taken: {elapsed_time} seconds\n")
                print(f"Size: {size}, Time taken: {elapsed_time} seconds")
else:
    # Read data from both files
    sizes2, times2 = read_data_from_file('..//..//plots/convolution_times_torch.txt')
    sizes1, times1 = read_data_from_file('..//..//plots/convolution_times_parallel.txt')
    sizes3, times3 = read_data_from_file('..//..//plots/convolution_times_no_insert.txt')
    sizes4, times4 = read_data_from_file('..//..//plots/convolution_times_sequential.txt')

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(sizes1, times1, marker='o', label='Squash - Insert')
    plt.plot(sizes3, times3, marker='s', label='Squash - No Insert')
    plt.plot(sizes4, times4, marker='d', label='Squash - Sequential')
    plt.plot(sizes2, times2, marker='x', label='Torch')
    plt.title('Time taken for convolution vs. Size')
    plt.xlabel('Size')
    plt.ylabel('Time taken (seconds)')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation on the x-axis
    plt.savefig('..//..//plots/convolution_time_comparison.png')
    plt.show()

