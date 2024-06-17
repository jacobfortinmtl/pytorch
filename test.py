import torch
import torch.nn.functional as F

# Create a random input tensor

# Setting the seed for reproducibility
torch.manual_seed(42)
# Create a random input tensor of integers
input_tensor = torch.arange(1,17).reshape(1, 1, 4, 4)  # generates integers between 0 and 99

# Convert the integer tensor to float
input_tensor = input_tensor.float()

print("Input Tensor: ")
print(input_tensor)
print()

# Create a random weights tensor of integers
weight = torch.ones(1, 1, 2, 2)  # generates integers between -10 and 9

# Convert the integer tensor to float
weight = weight.float()

print("Weight Tensor: ")
print(weight)
print()

## Printing hardware device
# print("Printing hardware device: ")
# print(input_tensor.device)
# print(weight.device)
# print()

# Apply the convolution operation
output_tensor = F.conv2d(input_tensor, weight, padding=0)

# Print the shape of the output tensor
print("Output after convolution: ")
print(output_tensor)
print(output_tensor.shape)