import sys
import os
import torch
import numpy as np
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hadamardHD import kronecker_hadamard, binding, unbinding

sys.path.append(os.path.abspath("PYTORCHCNNS"))
sys.path.append(os.path.abspath("PYTORCHCNNS/model_zoo/models"))

from PYTORCHCNNS.model_zoo.utils import *
from PYTORCHCNNS.model_zoo import datasets, models
from PYTORCHCNNS.model_zoo.datasets.digit_loader import get_digit_loader
from PYTORCHCNNS.model_zoo.models.lenet import LeNet_Pre

# Hyperparameters
D = 16384  # Hypervector dimensionality
vector_len = 84  # FC2 output size
n = 10       # Number of encoded samples to bundle per digit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a base matrix for projection (fixed seed for consistency)
np.random.seed(42)
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
pseudo_inverse = np.linalg.pinv(base_matrix)  # Store its inverse for decoding

lenet_pre = LeNet_Pre().to(device).eval()

decoded_replay_buffer = {}

print("\nExtracting FC2 Features, Encoding, Bundling, Unbundling, and Decoding...")

def encode_fc2(feature, base_matrix, D):
    """Encodes FC2 feature into a hypervector using random projection."""
    projected_hv = np.matmul(base_matrix, feature)  # Project FC2 feature
    return projected_hv  # Keep continuous values for better reconstruction

def bundle_encoded_samples(encoded_samples, D, n):
    """Bundles n encoded samples per digit."""
    bundled_hvs = {}
    for digit, encoded in encoded_samples.items():
        bundled_HV = np.zeros(D)
        for i in range(n):
            binded_HV = binding(D, i, encoded[i])  # Bind with unique Hadamard key
            bundled_HV += binded_HV
        bundled_hvs[digit] = bundled_HV
        print(f"Bundled HV for digit {digit} - Shape: {bundled_HV.shape}")
    return bundled_hvs

def unbundle_and_decode(bundled_hvs, pseudo_inverse, D, n):
    """Unbundles and decodes hypervectors back to FC2-like features."""
    decoded_digit_samples = {}

    for digit, bundled_HV in bundled_hvs.items():
        decoded_samples = []
        for i in range(n):
            unbound_HV = unbinding(D, i, bundled_HV)  # Correct unbinding using the same keys
            decoded_features = np.dot(pseudo_inverse, unbound_HV)  # Decode to FC2 space
            decoded_samples.append(decoded_features)
        
        decoded_digit_samples[digit] = np.array(decoded_samples)
    
    return decoded_digit_samples

for digit in range(10):
    print(f"Processing digit {digit}...")
    digit_loader = get_digit_loader(digit, batch_size=64, train=True)

    digit_fc2_features = []
    with torch.no_grad():
        for data, _ in digit_loader:
            data = data.to(device)
            features = lenet_pre(data)  # Extract FC2
            digit_fc2_features.extend(features.cpu().numpy())

    digit_fc2_features = np.array(digit_fc2_features)[:n]  # Take only n samples per digit

    # Encode FC2 features into hypervectors
    encoded_digit_samples = {digit: [encode_fc2(f, base_matrix, D) for f in digit_fc2_features]}

    # Bundle multiple encoded HVs into a single stored HV per digit
    bundled_hvs = bundle_encoded_samples(encoded_digit_samples, D, n)

    # Unbundle and decode back to FC2 features
    decoded_fc2 = unbundle_and_decode(bundled_hvs, pseudo_inverse, D, n)

    # Store decoded FC2 features
    decoded_replay_buffer[digit] = decoded_fc2[digit]  # Shape (n, 84)

# Save replay buffer
with open("replay_buffer_fc2.pkl", "wb") as f:
    pickle.dump(decoded_replay_buffer, f)

print("\nDecoded FC2 replay buffer saved to decoded_replay_buffer_fc2.pkl")
