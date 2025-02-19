import numpy as np
import argparse as ap
import pickle
import torch
from hadamardHD import kronecker_hadamard, binding, unbinding
from PYTORCHCNNS.model_zoo.datasets.alphabet_loader import get_letter_loader

# -----------------------------
# Argument Parser
# -----------------------------
parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="isolet", help="Dataset name")
parser.add_argument('-n', type=int, default=5, help="Number of encoded samples to bundle per letter")
parser.add_argument('-D', type=int, default=262144, help="Encoded hypervector dimensionality")
parser.add_argument('-vector-len', type=int, default=617, help="Input feature vector length (ISOLET = 617)")
parser.add_argument('-letters', type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ", help="Letters to process (default: A-Z)")
parser.add_argument('-batch-size', type=int, default=64, help="Batch size for data loading")
parser.add_argument('-train', type=bool, default=True, help="Use training dataset (True) or test dataset (False)")
# New arguments for iterative unbundling:
parser.add_argument('--iter-unbundle', action='store_true', help="Use iterative unbundling to improve decoding quality")
parser.add_argument('--iterations', type=int, default=5, help="Number of iterations for iterative unbundling")
parser.add_argument('--alpha', type=float, default=0.1, help="Learning rate for iterative unbundling updates")
parser.add_argument('-output', type=str, default="replay_buffer_isolet.pkl", help="Output file for replay buffer")
args = parser.parse_args()

# -----------------------------
# Random Projection Encoding (without extra binarization)
# -----------------------------
def encoding_rp(X_data, base_matrix):
    """
    Encodes input feature vectors into hypervectors using a random projection.
    (We do not binarize here so as to preserve amplitude information from the projection.)
    """
    enc_hvs = np.matmul(base_matrix, X_data.T)
    return enc_hvs.T

# -----------------------------
# Bundling of Encoded Samples
# -----------------------------
def bundle_encoded_samples(encoded_samples, D, n):
    """
    For each letter, bind each of the first n encoded samples with a distinct Hadamard key,
    and then sum them to form a single bundled hypervector.
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

# -----------------------------
# Iterative Unbundling (with Normalization)
# -----------------------------
def iterative_unbundle(bundled_HV, D, n, pseudo_inverse, iterations=5, alpha=0.1):
    """
    Attempts to recover each of the n bound hypervectors from the bundled hypervector
    using an iterative residual correction procedure.
    
    After each update, the estimate is normalized to approximately the norm of its initial value.
    """
    # Step 1: Initial estimates via basic unbinding.
    estimates = []
    for i in range(n):
        est = unbinding(D, i, bundled_HV)
        estimates.append(est.copy())
    
    # Compute target norms (we use the initial norm for each estimate)
    target_norms = [np.linalg.norm(est) for est in estimates]
    
    for it in range(iterations):
        # Reconstruct the bundled hypervector from current estimates.
        H_recon = np.zeros(D)
        for i in range(n):
            H_recon += binding(D, i, estimates[i])
        # Compute the reconstruction error.
        error = bundled_HV - H_recon
        # Update each estimate by unbinding the error and applying a correction.
        for i in range(n):
            correction = unbinding(D, i, error)
            estimates[i] += alpha * correction
            # Normalize the estimate to keep its norm close to the initial target norm.
            norm_val = np.linalg.norm(estimates[i])
            if norm_val > 0:
                estimates[i] = estimates[i] * (target_norms[i] / norm_val)
        err_norm = np.linalg.norm(error)
        print(f"Iteration {it+1}/{iterations}: reconstruction error norm = {err_norm:.4f}")
    
    # Decode the refined estimates back to the original feature space.
    decoded_samples = [np.dot(pseudo_inverse, est) for est in estimates]
    return decoded_samples

# -----------------------------
# Basic Unbundling (Single-step)
# -----------------------------
def basic_unbundle(bundled_HV, D, n, pseudo_inverse):
    """
    Basic unbundling: for each index, unbind the bundled HV once and decode.
    """
    decoded_samples = []
    for i in range(n):
        unbound = unbinding(D, i, bundled_HV)
        decoded_features = np.dot(pseudo_inverse, unbound)
        decoded_samples.append(decoded_features)
    return decoded_samples

# -----------------------------
# Data Loading
# -----------------------------
letters = list(args.letters)
letter_loaders = {letter: get_letter_loader(letter, batch_size=args.batch_size, train=args.train) for letter in letters}
letter_samples = {letter: next(iter(letter_loaders[letter])) for letter in letters}

D = args.D  
vector_len = args.vector_len  

# -----------------------------
# Base Matrix Creation
# -----------------------------
# Here we use a binary (±1) projection matrix to be consistent with Hadamard-based binding.
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)

# -----------------------------
# Regularized Pseudo-Inverse Calculation
# -----------------------------
lambda_reg = 1e-3  # You may adjust this value.
I = np.eye(vector_len)
pseudo_inverse = np.linalg.inv(base_matrix.T @ base_matrix + lambda_reg * I) @ base_matrix.T

# -----------------------------
# Encoding and Bundling
# -----------------------------
encoded_letter_samples = {}
for letter in letters:
    features, _ = letter_samples[letter]
    features_np = features.numpy() if isinstance(features, torch.Tensor) else features
    encoded_features = encoding_rp(features_np, base_matrix)
    # Take the first n samples for bundling.
    encoded_letter_samples[letter] = encoded_features[:args.n]

bundled_hvs = bundle_encoded_samples(encoded_letter_samples, D, args.n)

# -----------------------------
# Unbundling and Decoding
# -----------------------------
decoded_letter_samples = {}
for letter, bundled_HV in bundled_hvs.items():
    print(f"Decoding for letter {letter}")
    if args.iter_unbundle:
        # Use iterative unbundling (with normalization) to (hopefully) reduce interference.
        decoded_samples = iterative_unbundle(bundled_HV, D, args.n, pseudo_inverse,
                                             iterations=args.iterations, alpha=args.alpha)
    else:
        decoded_samples = basic_unbundle(bundled_HV, D, args.n, pseudo_inverse)
    decoded_letter_samples[letter] = decoded_samples

# -----------------------------
# Save the Replay Buffer
# -----------------------------
with open(args.output, "wb") as f:
    pickle.dump(decoded_letter_samples, f)
print(f"Replay buffer saved to {args.output}")
