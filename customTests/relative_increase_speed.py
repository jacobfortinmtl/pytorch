import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle
import sys
import os

# open file for writing

# Open file for writing
file_path = '../../plots/relative_increase.txt'

choice = int(os.getenv('DEFAULT', '0'))

if choice == 0:
    torch.manual_seed(42)
    # Get size from environment variable, with a default value if not set
    size = int(os.getenv('SIZE', '4'))**2  # Use the environment variable SIZE

    sqrt_size = int(math.sqrt(size))
    num_nans = int(size * 0.33)

    indices = torch.randperm(size)
    input_tensor = torch.arange(1, size + 1)
    input_tensor = input_tensor.float()
    input_tensor[indices[:num_nans]] = float('nan')

    input_tensor = input_tensor.reshape(1, 1, sqrt_size, sqrt_size)

    weight = torch.ones(1, 1, 2, 2)
    weight = weight.float()

    print("Size: ", sqrt_size**2)
    start_time = time.time()
    output_tensor = torch.nn.functional.conv2d(input_tensor, weight, padding=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print("Time taken for whole default", elapsed_time)
elif choice == 1:
    torch.manual_seed(42)
    # Get size from environment variable, with a default value if not set
    size = int(os.getenv('SIZE', '4'))**2  # Use the environment variable SIZE

    sqrt_size = int(math.sqrt(size))
    num_nans = int(size * 0.33)

    indices = torch.randperm(size)
    input_tensor = torch.arange(1, size + 1)
    input_tensor = input_tensor.float()
    input_tensor[indices[:num_nans]] = float('nan')

    input_tensor = input_tensor.reshape(1, 1, sqrt_size, sqrt_size)

    weight = torch.ones(1, 1, 2, 2)
    weight = weight.float()

    print("Size: ", sqrt_size**2)
    start_time = time.time()
    output_tensor = torch.nn.functional.conv2d(input_tensor, weight, padding=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken for whole default", elapsed_time)
