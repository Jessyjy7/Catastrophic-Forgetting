import numpy as np
import argparse as ap
import pickle
import torch
from hadamardHD import kronecker_hadamard, binding, unbinding
from PYTORCHCNNS.model_zoo.datasets.alphabet_loader import get_letter_loader

# Argument Parser
parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="isolet", help="Dataset name")
parser.add_argument('-n', type=int, default=5, help="Number of encoded samples to bundle per letter")
parser.add_argument('-D', type=int, default=262144, help="Encoded hypervector dimensionality")
parser.add_argument('-vector-len', type=int, default=617, help="Input feature vector length (ISOLET = 617)")
parser.add_argument('-letters', type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ", help="Letters to process (default: A-Z)")
parser.add_argument('-batch-size', type=int, default=64, help="Batch size for data loading")
parser.add_argument('-train', type=bool, default=True, help="Use training dataset (True) or test dataset (False)")
parser.add_argument('-output', type=str, default="replay_buffer_isolet.pkl", help="Output file for replay buffer")
args = parser.parse_args()

def encoding_rp(X_data, base_matrix, binary=False):
    """
    Encodes input feature vectors into hypervectors using random projection.
    """
    enc_hvs = np.matmul(base_matrix, X_data.T)
    if binary:
        enc_hvs = np.where(enc_hvs < 0, -1, 1)
    return enc_hvs.T

def bundle_encoded_samples(encoded_samples, D, n):
    """
    Bundles n encoded samples per letter by binding each sample with a unique key
    using circular convolution, then summing the bound vectors.
    """
    bundled_hvs = {}
    for letter, encoded in encoded_samples.items():
        bundled_HV = np.zeros(D)
        for i in range(n):
            binded_HV = binding(D, i, encoded[i])
            bundled_HV += binded_HV
        bundled_hvs[letter] = bundled_HV
        print(f"Bundled HV for letter {letter} - Shape: {bundled_HV.shape}")
    return bundled_hvs

def unbundle_and_decode(bundled_hvs, pseudo_inverse, D, n):
    """
    Unbundles and decodes hypervectors. For each letter, each of the n bound items is
    recovered using circular correlation (the inverse of circular convolution) and then
    decoded via multiplication with the pseudo-inverse.
    """
    unbundled_letter_samples = {}
    decoded_letter_samples = {}

    for letter, bundled_HV in bundled_hvs.items():
        unbundled_samples = []
        decoded_samples = []
        for i in range(n):
            unbound_HV = unbinding(D, i, bundled_HV)
            unbundled_samples.append(unbound_HV)

            decoded_features = np.dot(pseudo_inverse, unbound_HV)
            decoded_samples.append(decoded_features)

        unbundled_letter_samples[letter] = unbundled_samples
        decoded_letter_samples[letter] = decoded_samples
    
    return unbundled_letter_samples, decoded_letter_samples

# Load data for each letter using the provided loader
letters = list(args.letters)
letter_loaders = {letter: get_letter_loader(letter, batch_size=args.batch_size, train=args.train)
                  for letter in letters}
letter_samples = {letter: next(iter(letter_loaders[letter])) for letter in letters}

# Set up the random projection base matrix (using bipolar values)
D = args.D  
vector_len = args.vector_len  
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)
pseudo_inverse = np.linalg.pinv(base_matrix)

# Encode the letter samples
encoded_letter_samples = {}
for letter in letters:
    features, _ = letter_samples[letter]
    encoded_features = encoding_rp(features.numpy(), base_matrix, binary=False)
    # Use only the first n samples per letter
    encoded_letter_samples[letter] = encoded_features[:args.n]

# Bundle the encoded samples using circular convolution binding
bundled_hvs = bundle_encoded_samples(encoded_letter_samples, D, args.n)

# Unbundle and decode the bundled hypervectors using circular correlation
unbundled_letter_samples, decoded_letter_samples = unbundle_and_decode(bundled_hvs, pseudo_inverse, D, args.n)

# Save the decoded replay buffer to a pickle file
with open(args.output, "wb") as f:
    pickle.dump(decoded_letter_samples, f)
print(f"Replay buffer saved to {args.output}")
