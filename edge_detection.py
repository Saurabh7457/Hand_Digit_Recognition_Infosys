# Define the image array (2D list) representing a grayscale image
image = [
    [10, 10, 10, 10, 10],
    [10, 100, 100, 100, 10],
    [10, 100, 100, 100, 10],
    [10, 100, 100, 100, 10],
    [10, 10, 10, 10, 10]
]

# Define vertical and horizontal edge detection kernels (Sobel filters)
vertical_kernel = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

horizontal_kernel = [
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
]

# Function to perform convolution
def convolve(image, kernel):
    rows = len(image)
    cols = len(image[0])
    k_size = len(kernel)
    k_half = k_size // 2
    
    # Initialize the output array with zeros
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Perform convolution
    for i in range(k_half, rows - k_half):
        for j in range(k_half, cols - k_half):
            weighted_sum = 0
            for ki in range(k_size):
                for kj in range(k_size):
                    pixel = image[i + ki - k_half][j + kj - k_half]
                    weight = kernel[ki][kj]
                    weighted_sum += pixel * weight
            output[i][j] = abs(weighted_sum)  # Taking absolute value for edge intensity
    return output

# Apply vertical and horizontal kernels
vertical_edges = convolve(image, vertical_kernel)
horizontal_edges = convolve(image, horizontal_kernel)

# Display the resulting arrays
print("Original Image:")
for row in image:
    print(row)

print("\nVertical Edges Detected:")
for row in vertical_edges:
    print(row)

print("\nHorizontal Edges Detected:")
for row in horizontal_edges:
    print(row)
