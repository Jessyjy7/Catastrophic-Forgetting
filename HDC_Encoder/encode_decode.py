import numpy as np
import argparse as ap
import pickle
import matplotlib.pyplot as plt
from hadamardHD import kronecker_hadamard, binding, unbinding, bundling, calculate_similarity
from PYTORCHCNNS.model_zoo.datasets.digit_loader import get_digit_loader

parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="mnist", help="Dataset name")
parser.add_argument('-n', type=int, default=10, help="Number of encoded samples to bundle per digit")
args = parser.parse_args()

# 1) Load digit loaders (no 32x32 resize now)
digit_loaders = {digit: get_digit_loader(digit, batch_size=64, train=True) for digit in range(10)}
# Grab one batch per digit
digit_samples = {digit: next(iter(digit_loaders[digit])) for digit in range(10)}

# 2) Hyperdimensional parameters
D = 131072  # Encoded dimensionality
vector_len = 784  # 28*28

# 3) Build a random bipolar matrix of shape [D, 784]
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)
pseudo_inverse = np.linalg.pinv(base_matrix)

def encoding_rp(X_data, base_matrix, binary=False):
    """
    X_data: shape [batch_size, 784]
    base_matrix: shape [D, 784]
    """
    enc_hvs = np.matmul(base_matrix, X_data.T)  # => shape [D, batch_size]
    if binary:
        enc_hvs = np.where(enc_hvs < 0, -1, 1)
    return enc_hvs.T  # => shape [batch_size, D]

# 4) Encode digit samples and store first n encoded samples for each digit
n = args.n
encoded_digit_samples = {}
for digit in range(10):
    images, labels = digit_samples[digit]
    # images shape: [batch_size, 1, 28, 28]
    images = images.view(images.size(0), -1)  # Flatten to [batch_size, 784]
    encoded_images = encoding_rp(images.numpy(), base_matrix, binary=False)
    encoded_digit_samples[digit] = encoded_images[:n] 

# 5) Bundle n encoded samples per digit
bundled_hvs = {}
for digit in range(10):
    bundled_HV = np.zeros(D)
    for i in range(n):
        binded_HV = binding(D, i, encoded_digit_samples[digit][i])
        bundled_HV += binded_HV
    bundled_hvs[digit] = bundled_HV
    print(f"Bundled HV for digit {digit} - Shape: {bundled_HV.shape}")

# 6) Unbundle and decode
unbundled_digit_samples = {}
decoded_digit_samples = {}
for digit in range(10):
    unbundled_samples = []
    decoded_samples = []
    for i in range(n):
        unbound_HV = unbinding(D, i, bundled_hvs[digit])
        unbundled_samples.append(unbound_HV)

        # Decode each unbound HV => shape [784]
        decoded_image = np.dot(pseudo_inverse, unbound_HV).reshape(28, 28)
        decoded_samples.append(decoded_image)

    unbundled_digit_samples[digit] = unbundled_samples
    decoded_digit_samples[digit] = decoded_samples

# 7) Visualization
def plot_original_and_decoded_grouped(digit_samples, decoded_digit_samples, n):
    fig, axes = plt.subplots(2 * n, 10, figsize=(20, 4 * n))
    fig.suptitle("Original and Decoded Images by Group (n={} per row)".format(n))

    for j in range(n):
        for digit in range(10):
            # Original image: shape [1,28,28] => flatten => [784], reshape => [28,28]
            original_image = digit_samples[digit][0][j].reshape(28, 28)
            axes[2 * j, digit].imshow(original_image, cmap="gray")
            axes[2 * j, digit].axis("off")

            # Decoded image: shape [28,28]
            decoded_image = decoded_digit_samples[digit][j]
            axes[2 * j + 1, digit].imshow(decoded_image, cmap="gray")
            axes[2 * j + 1, digit].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    
# Optionally call the grouped plot
# plot_original_and_decoded_grouped(digit_samples, decoded_digit_samples, n)

def plot_original_and_decoded_per_digit(digit_samples, decoded_digit_samples, n):
    for digit in range(10):
        fig, axes = plt.subplots(2, n, figsize=(15, 5))
        fig.suptitle(f"Original and Decoded Images for Digit {digit} (n={n})", fontsize=16)

        for j in range(n):
            # Original: shape [28,28]
            original_image = digit_samples[digit][0][j].reshape(28, 28)
            axes[0, j].imshow(original_image, cmap="gray")
            axes[0, j].axis("off")

            # Decoded: shape [28,28]
            decoded_image = decoded_digit_samples[digit][j]
            axes[1, j].imshow(decoded_image, cmap="gray")
            axes[1, j].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

plot_original_and_decoded_per_digit(digit_samples, decoded_digit_samples, n)

# 8) Save decoded_digit_samples to a file
with open("replay_buffer.pkl", "wb") as f:
    pickle.dump(decoded_digit_samples, f)
print("Replay buffer saved to replay_buffer.pkl")
