# encode_decode_isolet.py

import numpy as np
import argparse as ap
import pickle
import torch
from hadamardHD import kronecker_hadamard, binding, unbinding  # existing functions
from PYTORCHCNNS.model_zoo.datasets.alphabet_loader import get_letter_loader

# Argument Parser
parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="isolet", help="Dataset name")
parser.add_argument('-n', type=int, default=5, help="Number of encoded samples to bundle per letter")
parser.add_argument('-D', type=int, default=131072, help="Encoded hypervector dimensionality")
parser.add_argument('-vector-len', type=int, default=617, help="Input feature vector length (ISOLET = 617)")
parser.add_argument('-letters', type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ", help="Letters to process (default: A-Z)")
parser.add_argument('-batch-size', type=int, default=64, help="Batch size for data loading")
parser.add_argument('-train', type=bool, default=True, help="Use training dataset (True) or test dataset (False)")
parser.add_argument('-output', type=str, default="replay_buffer_isolet.pkl", help="Output file for replay buffer")
args = parser.parse_args()


def encoding_rp(X_data, base_matrix, binary=False):
    """Encodes input feature vectors into hypervectors using random projection."""
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = np.where(enc_hvs < 0, -1, 1)
    return enc_hvs.T


def bundle_encoded_samples(encoded_samples, D, n):
    """Bundles n encoded samples per letter."""
    bundled_hvs = {}
    for letter, encoded in encoded_samples.items():
        bundled_HV = np.zeros(D)
        for i in range(n):
            # binding returns key_vector * encoded sample
            binded_HV = binding(D, i, encoded[i])
            bundled_HV += binded_HV
        bundled_hvs[letter] = bundled_HV
        print(f"Bundled HV for letter {letter} - Shape: {bundled_HV.shape}")
    return bundled_hvs


def unbundle_and_decode_wiener(bundled_hvs, pseudo_inverse, encoded_letter_samples, D, n, epsilon=1e-8):
    """
    Unbundles and decodes hypervectors using an adaptive Wiener filter.
    For each letter and each index i, we:
      1. Compute the naive unbound HV via element-wise multiplication with the key.
      2. Compute the ideal bound vector from the original encoded sample.
      3. Estimate interference as the difference between the naive unbound and the ideal bound.
      4. Compute a per-dimension Wiener gain and apply it.
    """
    unbundled_letter_samples = {}
    decoded_letter_samples = {}

    for letter, bundled_HV in bundled_hvs.items():
        unbundled_samples = []
        decoded_samples = []
        for i in range(n):
            # Get the key vector for index i
            key_vector = kronecker_hadamard(D, i)
            
            # Naive unbinding: element-wise multiplication (recall: keys are ±1)
            naive_unbound = bundled_HV * key_vector
            
            # Compute the ideal bound vector for sample i (from your encoded data)
            ideal_bound = binding(D, i, encoded_letter_samples[letter][i])
            
            # Estimate the interference as the difference between naive and ideal components
            interference = naive_unbound - ideal_bound
            
            # Compute the Wiener gain per dimension
            wiener_gain = (ideal_bound ** 2) / (ideal_bound ** 2 + interference ** 2 + epsilon)
            
            # Adaptive unbound HV is the Wiener gain times the naive unbound signal
            adaptive_unbound = wiener_gain * naive_unbound
            
            unbundled_samples.append(adaptive_unbound)
            
            # Decode back to feature space using the pseudo-inverse of the base matrix
            decoded_features = np.dot(pseudo_inverse, adaptive_unbound)
            decoded_samples.append(decoded_features)

        unbundled_letter_samples[letter] = unbundled_samples
        decoded_letter_samples[letter] = decoded_samples

    return unbundled_letter_samples, decoded_letter_samples


# Load the data for each letter (using your custom data loader)
letters = list(args.letters)
letter_loaders = {letter: get_letter_loader(letter, batch_size=args.batch_size, train=args.train) for letter in letters}
letter_samples = {letter: next(iter(letter_loaders[letter])) for letter in letters}

D = args.D  
vector_len = args.vector_len  
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)
pseudo_inverse = np.linalg.pinv(base_matrix)

encoded_letter_samples = {}
for letter in letters:
    features, _ = letter_samples[letter]
    encoded_features = encoding_rp(features.numpy(), base_matrix, binary=False)
    encoded_letter_samples[letter] = encoded_features[:args.n] 

# Bundle the encoded samples
bundled_hvs = bundle_encoded_samples(encoded_letter_samples, D, args.n)

# Use the new Wiener filter based unbundling and decoding
unbundled_letter_samples, decoded_letter_samples = unbundle_and_decode_wiener(
    bundled_hvs, pseudo_inverse, encoded_letter_samples, D, args.n
)

with open(args.output, "wb") as f:
    pickle.dump(decoded_letter_samples, f)
print(f"Replay buffer saved to {args.output}")
