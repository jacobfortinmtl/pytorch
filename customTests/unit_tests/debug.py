import torch.nn.functional as F
import pickle
import torch

# TODO UNCOMMENT START HERE
# Load input_data
with open('data/input.pkl', 'rb') as f:
    input_data = pickle.load(f)
batches_in, channels_in, rows_in, columns_in = input_data.shape
print("Input shape", input_data.shape)

# Importing the weights and biases
with open('data/weight.pkl', 'rb') as f:
    weights = pickle.load(f)
print("Weights Shape: ", weights.shape)

with open('data/bias.pkl', 'rb') as f:
    biases = pickle.load(f)

# My custom output
custom_conv_output = F.conv2d(input_data, weights, bias=biases, padding=1)
print(f"Custom Conv Output Shape: {custom_conv_output.shape}")
print(custom_conv_output[0, 63, 22, 88])


# Saving custom output
with open('data/cpp_conv1.pkl', 'wb') as f:
    pickle.dump(custom_conv_output, f)
# TODO UNCOMMENT UNTIL HERE TO GENERATE THE DATA

# Using only 1 thread in torch
# torch.set_num_threads(1)

# # TRYING TO REGET THE MAX:
# # Load input_data
# with open('data/input.pkl', 'rb') as f:
#     input_data = pickle.load(f)
# batches_in, channels_in, rows_in, columns_in = input_data.shape
# # print("Input shape", input_data.shape)

# # Importing the weights and biases
# with open('data/weight.pkl', 'rb') as f:
#     weights = pickle.load(f)
# # print("Weights Shape: ", weights.shape)

# with open('data/bias.pkl', 'rb') as f:
#     biases = pickle.load(f)

# # Load conv_output
# with open('data/conv1.pkl', 'rb') as f:
#     conv_output = pickle.load(f)
# # print("Convolution Output: ", conv_output.shape)


# input_problem = input_data[:, :, 21:24, 87:90]
# print("Input Problem Shape: ", input_problem.shape)
# # print("Input Problem: ", input_problem)
# # Keeping only from the weight the channel 63
# weights_sliced = weights[63, :, :, :].unsqueeze(0)
# print("Weights Shape: ", weights_sliced.shape)
# # print("Weights: ", weights_sliced)
# biases_sliced = biases[63].unsqueeze(0)
# print("Biases Shape: ", biases_sliced.shape)
# # print("Biases: ", biases_sliced)

# # print("Conv output at 63: ", F.conv2d(input_data, weights, bias = biases, padding=0)[0, 63, 22, 88])

# # print(input_problem)
# # print(weights_sliced)
# # print(biases_sliced)
# print("Conv output at 63: ", F.conv2d(input_problem, weights_sliced, bias = biases_sliced, padding=0))
