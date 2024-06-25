import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import time
from nan_ops import NaNPool2d, NormalPool2d, NaNConv2d, NormalConv2d

# todo uncomment if you want to get the nans
# FILENAME = "..//sample_data//outp_bn3_1.pkl"
FILENAME = "..//sample_data//0.5_decoder_output2_3.pkl" # output from maxpool
# Load the data from the pickle file
data = pickle.load(open(FILENAME, 'rb'))

if isinstance(data, tuple):
    a = data[0]
else:
    a = data

# Printing the shape of the data
print(a.shape)

# Todo only uncomment for nans
# nanpoolPy = NaNPool2d(max_threshold=1)
# nanoutputPy, nanindicesPy = nanpoolPy(a, pool_size=(2,2), strides=(2,2))
# with open('..//sample_data//maxpool_output.pkl', 'wb') as f:
#     pickle.dump(nanoutputPy, f)

# Apply convolution with the following
weight = torch.ones(1, 71, 2, 2)
output_tensor = F.conv2d(data, weight, padding=0)

# Squeeze the output tensor
output_tensor_squeezed = output_tensor.detach().numpy().squeeze()

# Visualize the output
fig, axes = plt.subplots(1, 1, figsize=(7, 5))

sns.heatmap(output_tensor_squeezed, ax=axes)
axes.set_title('Convolution After NaN Pooling')

plt.savefig('../plots/output_comparison.png')  # Save the plot
plt.close(fig)