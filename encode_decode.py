import numpy as np
import argparse as ap
import pickle
import matplotlib.pyplot as plt
from hadamardHD import kronecker_hadamard, binding, unbinding, bundling, calculate_similarity
from PYTORCHCNNS.model_zoo.datasets.digit_loader import get_digit_loader

# Setup arguments
parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="mnist", help="Dataset name")
parser.add_argument('-n', type=int, default=5, help="Number of encoded samples to bundle per digit")
args = parser.parse_args()

# Step 1: Load data and encode as before
digit_loaders = {digit: get_digit_loader(digit, batch_size=64, train=True) for digit in range(10)}
digit_samples = {digit: next(iter(digit_loaders[digit])) for digit in range(10)}

D = 4096  # Encoded dimensionality
vector_len = 1024
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)
pseudo_inverse = np.linalg.pinv(base_matrix)

# Helper functions
def encoding_rp(X_data, base_matrix, binary=False):
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = np.where(enc_hvs < 0, -1, 1)
    return enc_hvs.T

# Encode digit samples and store first n encoded samples for each digit
n = args.n
encoded_digit_samples = {}
for digit in range(10):
    images, labels = digit_samples[digit]
    images = images.view(images.size(0), -1)  # Flatten
    encoded_images = encoding_rp(images.numpy(), base_matrix, binary=False)
    encoded_digit_samples[digit] = encoded_images[:n]  # Take first n samples

# Step 2: Bundle n encoded samples per digit
bundled_hvs = {}
for digit in range(10):
    bundled_HV = np.zeros(D)
    for i in range(n):
        hadamard_row = kronecker_hadamard(D, i)
        binded_HV = binding(D, i, encoded_digit_samples[digit][i])
        bundled_HV += binded_HV
    bundled_hvs[digit] = bundled_HV
    print(f"Bundled HV for digit {digit} - Shape: {bundled_HV.shape}")

# Step 3: Unbundle and decode
unbundled_digit_samples = {}
decoded_digit_samples = {}
for digit in range(10):
    unbundled_samples = []
    decoded_samples = []
    for i in range(n):
        hadamard_row = kronecker_hadamard(D, i)
        unbound_HV = unbinding(D, i, bundled_hvs[digit])
        unbundled_samples.append(unbound_HV)

        # Decode each unbound HV to get the image approximation
        decoded_image = np.dot(pseudo_inverse, unbound_HV).reshape(32, 32)
        decoded_samples.append(decoded_image)

    unbundled_digit_samples[digit] = unbundled_samples
    decoded_digit_samples[digit] = decoded_samples

# Step 4: Calculate similarity between original encoded samples and unbundled samples
# similarity_scores = {}
# for digit in range(10):
#     similarities = []
#     for i in range(n):
#         original_encoded = encoded_digit_samples[digit][i]
#         unbundled_encoded = unbundled_digit_samples[digit][i]
#         similarity = calculate_similarity(original_encoded, unbundled_encoded)
#         similarities.append(similarity)
#     similarity_scores[digit] = similarities
#     print(f"Similarity scores for digit {digit}: {similarities}")

# Step 5: Visualize original and decoded images for each digit
def plot_original_and_decoded_grouped(digit_samples, decoded_digit_samples, n):
    fig, axes = plt.subplots(2 * n, 10, figsize=(20, 4 * n))
    fig.suptitle("Original and Decoded Images by Group (n={} per row)".format(n))

    for j in range(n):
        for digit in range(10):
            # Original image for each n
            original_image = digit_samples[digit][0][j].reshape(32, 32)
            axes[2 * j, digit].imshow(original_image, cmap="gray")
            axes[2 * j, digit].axis("off")
            axes[2 * j, digit].set_title(f"Original {digit} (n={j+1})", fontsize=7, pad=3)

            # Decoded image for each n
            decoded_image = decoded_digit_samples[digit][j].reshape(32, 32)
            axes[2 * j + 1, digit].imshow(decoded_image, cmap="gray")
            axes[2 * j + 1, digit].axis("off")
            axes[2 * j + 1, digit].set_title(f"Decoded {digit} (n={j+1})", fontsize=7, pad=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to avoid overlap
    plt.show()
    
# plot_original_and_decoded_grouped(digit_samples, decoded_digit_samples, n)

# Save decoded_digit_samples to a file
with open("replay_buffer.pkl", "wb") as f:
    pickle.dump(decoded_digit_samples, f)
print("Replay buffer saved to replay_buffer.pkl")