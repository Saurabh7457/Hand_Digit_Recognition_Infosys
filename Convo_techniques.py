import torch
import torch.nn.functional as F

# 1. Create an Input Tensor
# Construct a 6x6 tensor: first 3 columns [0:3] filled with 1s, last 3 columns [3:6] filled with 0s
input_tensor = torch.tensor([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

print("Input Tensor:")
print(input_tensor)

# 2. Create a Kernel Tensor
# Define a 3x3 kernel tensor for detecting vertical edges
kernel = torch.tensor([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions

print("\nKernel Tensor:")
print(kernel)

# 3. Convolve with Strides
def convolve_with_stride(input_tensor, kernel, stride):
    # Perform convolution using F.conv2d
    output = F.conv2d(input_tensor, kernel, stride=stride, padding=0)
    return output

# Perform convolution with different strides
stride_1 = convolve_with_stride(input_tensor, kernel, stride=1)
stride_2 = convolve_with_stride(input_tensor, kernel, stride=2)

print("\nOutput with Stride = 1:")
print(stride_1)

print("\nOutput with Stride = 2:")
print(stride_2)

# 4. Analyze the Output
print("\nAnalysis:")
print("1. With Stride = 1: The output is dense, capturing all vertical edge changes.")
print("2. With Stride = 2: The output is sparse, as the kernel skips positions, reducing spatial resolution.")
