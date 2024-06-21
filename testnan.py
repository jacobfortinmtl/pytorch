import torch
import torch.nn.functional as F
# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle
# import numpy as np
import time
from nan_ops import NaNPool2d, NormalPool2d, NaNConv2d, NormalConv2d

# todo uncomment if you want to get the nans
# FILENAME = "..//sample_data//outp_bn3_1.pkl"
FILENAME = "..//sample_data//maxpool_output.pkl" # output from maxpool
# Load the data from the pickle file
data = pickle.load(open(FILENAME, 'rb'))

if isinstance(data, tuple):
    a = data[0]
else:
    a = data

# Todo only uncomment for nans
# nanpoolPy = NaNPool2d(max_threshold=1)
# nanoutputPy, nanindicesPy = nanpoolPy(a, pool_size=(2,2), strides=(2,2))
# with open('..//sample_data//maxpool_output.pkl', 'wb') as f:
#     pickle.dump(nanoutputPy, f)

# Apply convolution with the following
weight = torch.ones(1, 71, 2, 2)
output_tensor = F.conv2d(data, weight, padding=0)