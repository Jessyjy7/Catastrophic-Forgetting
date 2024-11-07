import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
# from rp_fast import encoding_rp
# from hadamardHD import binding, unbinding, bundling, unbundling
# from PYTORCHCNNS.model_zoo.utils import *
# from PYTORCHCNNS.model_zoo import datasets
# from PYTORCHCNNS.model_zoo import models
from PYTORCHCNNS.model_zoo.datasets.digit_loader import get_digit_loader

parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="mnist", help="Dataset name")
parser.add_argument('-model', type=str, default="lenet", help="Model name")
parser.add_argument('-checkpoint', type=str, default="weights_lenet", help="Checkpoint name")
parser.add_argument('-train-ratio', type=float, default=0.9, help="Training data ratio")
parser.add_argument('-batch-size', type=int, default=64, help="Batch size")
parser.add_argument('-test-batch-size', type=int, default=500, help="Test batch size")
parser.add_argument('-epochs', type=int, default=20, help="Number of epochs")
args = parser.parse_args()

# Step 1: Create a dictionary to store loaders for each digit
digit_loaders = {}
for digit in range(10):
    digit_loaders[digit] = get_digit_loader(digit, batch_size=64, train=True)

# Step 2: Create a dictionary to store one batch of data for each digit
digit_samples = {}

# Retrieve one batch per digit and store it in digit_samples
for digit in range(10):
    print(f"Accessing one batch of data for digit {digit} outside the loop")
    for batch in digit_loaders[digit]:
        images, labels = batch
        images = images.view(images.size(0), -1)  # Flatten each image to 1024x1
        digit_samples[digit] = (images, labels)  # Store the batch in the dictionary
        print(f"Digit {digit} - Batch shape: {images.shape}")
        break
    
# Step 3: Generate the random projection base matrix    
D = 4096  # Target dimensionality
vector_len = 1024  # Dimension of the flattened image vectors
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)  # Binarize the matrix to -1 and 1
pseudo_inverse = np.linalg.pinv(base_matrix)

def binarize(arr):
    return np.where(arr < 0, -1, 1)

def encoding_rp(X_data, base_matrix, binary=False):
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = binarize(enc_hvs)
    return enc_hvs.T

# Step 4: Encode each digit's batch using encoding_rp and store the encoded results
encoded_digit_samples = {}
for digit in range(10):
    images, labels = digit_samples[digit]
    # Apply random projection encoding to the batch of images
    encoded_images = encoding_rp(images.numpy(), base_matrix, binary=False)  # Assuming encoding_rp expects numpy array
    encoded_digit_samples[digit] = (encoded_images, labels)  # Store the encoded vectors

    # Print encoded vector shape for verification
    print(f"Random projected vectors for digit {digit} - Shape: {encoded_images.shape}")
    
# Step 5: Decode each digit's encoded vectors using the pseudo-inverse
decoded_digit_samples = {}
for digit in range(10):
    encoded_images, labels = encoded_digit_samples[digit]
    # Apply the pseudo-inverse to decode
    decoded_images = np.dot(pseudo_inverse, encoded_images.T).T
    decoded_digit_samples[digit] = (decoded_images, labels)  # Store the decoded vectors

    # Print decoded vector shape for verification
    print(f"Decoded vectors for digit {digit} - Shape: {decoded_images.shape}")
    
# Initialize a figure for displaying the images
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
fig.suptitle("Original vs Decoded Images (by Digit)")

for digit in range(10):
    # Retrieve the first image for each digit from both original and decoded samples
    original_image, _ = digit_samples[digit]  # Shape: [batch_size, 1024]
    decoded_image, _ = decoded_digit_samples[digit]  # Shape: [batch_size, 1024]

    # Get the first image from the batch
    original_image = original_image[0].reshape(32, 32)  # Reshape to 32x32 for visualization
    decoded_image = decoded_image[0].reshape(32, 32)    # Reshape to 32x32 for visualization

    # Plot original image (top row)
    axes[0, digit].imshow(original_image, cmap="gray")
    axes[0, digit].axis("off")
    axes[0, digit].set_title(f"Original {digit}")

    # Plot decoded image (bottom row)
    axes[1, digit].imshow(decoded_image, cmap="gray")
    axes[1, digit].axis("off")
    axes[1, digit].set_title(f"Decoded {digit}")

plt.show()
