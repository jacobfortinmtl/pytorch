import torch

import torch.nn as nn

# Create a random input tensor
input_tensor = torch.randn(1, 3, 32, 32)

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Apply the convolution operation
output_tensor = conv_layer(input_tensor)

# Print the shape of the output tensor
print(output_tensor.shape)