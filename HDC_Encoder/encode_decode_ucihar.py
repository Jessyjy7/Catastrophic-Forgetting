import numpy as np
import argparse as ap
import pickle
import torch
from hadamardHD import binding, unbinding
from PYTORCHCNNS.model_zoo.datasets.ucihar_loader import get_activity_loader

parser = ap.ArgumentParser()
parser.add_argument('-dataset', type=str, default="ucihar", help="Dataset name")
parser.add_argument('-n', type=int, default=5, help="Number of encoded samples to bundle per activity")
parser.add_argument('-D', type=int, default=262144, help="Encoded hypervector dimensionality")
parser.add_argument('-vector-len', type=int, default=561, help="Input feature vector length (UCI HAR = 561)")
parser.add_argument('-activities', type=str, default="WALKING,WALKING_UPSTAIRS,WALKING_DOWNSTAIRS,SITTING,STANDING,LAYING", help="Comma-separated list of activities to process")
parser.add_argument('-batch-size', type=int, default=64, help="Batch size for data loading")
parser.add_argument('-train', type=bool, default=True, help="Use training dataset (True) or test dataset (False)")
parser.add_argument('-output', type=str, default="replay_buffer_ucihar.pkl", help="Output file for replay buffer")
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
    Bundles n encoded samples per activity by binding each sample with a unique key
    using circular convolution, then summing the bound vectors.
    """
    bundled_hvs = {}
    for activity, encoded in encoded_samples.items():
        bundled_HV = np.zeros(D)
        for i in range(n):
            binded_HV = binding(D, i, encoded[i])
            bundled_HV += binded_HV
        bundled_hvs[activity] = bundled_HV
        print(f"Bundled HV for activity {activity} - Shape: {bundled_HV.shape}")
    return bundled_hvs

def unbundle_and_decode(bundled_hvs, pseudo_inverse, D, n):
    """
    Unbundles and decodes hypervectors. For each activity, each of the n bound items is
    recovered using circular correlation (the inverse of circular convolution) and then
    decoded via multiplication with the pseudo-inverse.
    """
    unbundled_activity_samples = {}
    decoded_activity_samples = {}

    for activity, bundled_HV in bundled_hvs.items():
        unbundled_samples = []
        decoded_samples = []
        for i in range(n):
            unbound_HV = unbinding(D, i, bundled_HV)
            unbundled_samples.append(unbound_HV)

            decoded_features = np.dot(pseudo_inverse, unbound_HV)
            decoded_samples.append(decoded_features)

        unbundled_activity_samples[activity] = unbundled_samples
        decoded_activity_samples[activity] = decoded_samples
    
    return unbundled_activity_samples, decoded_activity_samples

activities = args.activities.split(",")
activity_loaders = {activity: get_activity_loader(activity, batch_size=args.batch_size, train=args.train)
                    for activity in activities}
activity_samples = {activity: next(iter(activity_loaders[activity])) for activity in activities}

D = args.D  
vector_len = args.vector_len  
base_matrix = np.random.uniform(-1, 1, (D, vector_len))
base_matrix = np.where(base_matrix >= 0, 1, -1)
pseudo_inverse = np.linalg.pinv(base_matrix)

encoded_activity_samples = {}
for activity in activities:
    features, _ = activity_samples[activity]
    encoded_features = encoding_rp(features.numpy(), base_matrix, binary=False)
    encoded_activity_samples[activity] = encoded_features[:args.n]

bundled_hvs = bundle_encoded_samples(encoded_activity_samples, D, args.n)

unbundled_activity_samples, decoded_activity_samples = unbundle_and_decode(bundled_hvs, pseudo_inverse, D, args.n)

with open(args.output, "wb") as f:
    pickle.dump(decoded_activity_samples, f)
print(f"Replay buffer saved to {args.output}")

for activity in activities:
    features, _ = activity_samples[activity]
    original_samples = features.numpy()[:args.n]
    
    print(f"\nActivity: {activity}")
    for i in range(args.n):
        original = original_samples[i]
        decoded = decoded_activity_samples[activity][i]
        diff_norm = np.linalg.norm(original - decoded)
        norm = np.linalg.norm(original)
        relative_error = diff_norm / norm
        print(f"Sample {i}:")
        print("L2 Original: ", norm)
        print("L2 Difference Norm: ", diff_norm)
        print("Relative Error: ", relative_error)
