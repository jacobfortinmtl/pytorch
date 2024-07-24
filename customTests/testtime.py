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
    square = size
    size = size**2
    # Determine how many elements should be NaN
    fraction_of_image_with_nans = os.environ.get('FRACTION')
    if fraction_of_image_with_nans is None:
        fraction_of_image_with_nans = 0.33
    #print(fraction_of_image_with_nans)
    num_nans = int(size*(float(fraction_of_image_with_nans)))

    # Create a random permutation of the indices
    indices = torch.randperm(size)
    input_tensor = torch.arange(1, size + 1)
    # Convert the integer tensor to float
    input_tensor = input_tensor.float()
    # Set the elements at the indices to NaN
    input_tensor[indices[:num_nans]] = float('nan')

    # Reshape the tensor back to its original shape
    input_tensor = input_tensor.reshape(1, 1, square, square)

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
        for size in np.logspace(np.log10(10), np.log10(3162), num=100, dtype=int):
            elapsed_time = measure_convolution_time(size)
            if elapsed_time is not None:
                f.write(f"Size: {size}, Time taken: {elapsed_time} seconds\n")
                # print(f"Size: {size}, Time taken: {elapsed_time} seconds")

else:
    file_prefixes = [
        'convolution_cores_', 'convolution_nan_ratio_', 'convolution_nan_ratio_no_insert','convolution_torch_default'
    ]

    # Read data for NaN ratio variations
    nan_data = {}
    for fraction in [0, 0.25, 0.5, 0.75, 0.99]:
        file_path = f'../../plots/{file_prefixes[1]}{fraction}.txt'
        if os.path.exists(file_path):
            sizes, times = read_data_from_file(file_path)
            nan_data[f'NaN Ratio - {fraction}'] = (sizes, times)

    for fraction in [0.99]:
        file_path = f'../../plots/{file_prefixes[2]}{fraction}.txt'
        if os.path.exists(file_path):
            sizes, times = read_data_from_file(file_path)
            nan_data[f'NaN Ratio NO REINSERT- {fraction}'] = (sizes, times)

    # Read data for core variations
    core_data = {}
    for i in range(1, os.cpu_count() + 1):
        file_path = f'../../plots/{file_prefixes[0]}{i}.txt'
        if os.path.exists(file_path):
            sizes, times = read_data_from_file(file_path)
            core_data[f'Threads - {i}'] = (sizes, times)

    # Read data for default torch setting
    file_path = f'../../plots/{file_prefixes[3]}.txt'
    if os.path.exists(file_path):
        sizes, times = read_data_from_file(file_path)
        core_data['Torch Default'] = (sizes, times)
        nan_data['Torch Default'] = (sizes, times)

    # Plotting the data in subplots
    fig, axs = plt.subplots(1, 2, figsize=(24, 14))

    # Plotting NaN ratio variations
    distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
    ]

    for i, (label, (sizes, times)) in enumerate(nan_data.items()):
        axs[0].plot(sizes, times, 'o-', label=label, markersize=4, color=distinct_colors[i % len(distinct_colors)])

    axs[0].set_title('Time taken for convolution vs. Size (NaN Ratio Variations)')
    axs[0].set_xlabel('Size of side of input Matrix')
    axs[0].set_ylabel('Time taken (seconds)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].ticklabel_format(style='plain', axis='x')  # Disable scientific notation on the x-axis

    # Plotting core variations
    distinct_colors = [
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#00b3be', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', 
        '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
    ]

    for i, (label, (sizes, times)) in enumerate(core_data.items()):
        axs[1].plot(sizes, times, 'o-', label=label, markersize=4, color=distinct_colors[i % len(distinct_colors)])

    axs[1].set_title('Time taken for convolution vs. Size (Thread Variations) - Nan is at default 0.33')
    axs[1].set_xlabel('Size of side of input Matrix')
    axs[1].set_ylabel('Time taken (seconds)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].ticklabel_format(style='plain', axis='x')  # Disable scientific notation on the x-axis

        # For the first subplot
    axs[0].set_xlim(0, 500)  # Set maximum x-axis value to 500
    axs[0].set_ylim(0, 0.01)  # Set maximum y-axis value to 1000

    # For the second subplot
    axs[1].set_xlim(0, 500)  # Set maximum x-axis value to 500
    axs[1].set_ylim(0, 0.01)  # Set maximum y-axis value to 1000

    plt.savefig('../../plots/convolution_time_comparison_subplots_row_major.png')
    plt.show()