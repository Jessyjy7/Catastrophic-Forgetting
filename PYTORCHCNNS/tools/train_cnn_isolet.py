import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse as ap
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from model_zoo.utils import *
from model_zoo import datasets
from model_zoo import models
from model_zoo.models import mlp
from model_zoo.datasets.alphabet_loader import get_letter_loader
from model_zoo.datasets.isolet import load_train_val_data, load_test_data
import time
import copy
import numpy as np
import matplotlib.pyplot as plt


parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True, help='prefix name for the checkpoints')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=500, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs per letter')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
print("Device: {}".format(device))

if args.dataset == 'isolet':  
    input_channels = 617  
    out_classes = 26  
    test_loader = load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = load_train_val_data(batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda)

if args.model == "mlp": 
    model = mlp.MLP(input_dim=input_channels, output_dim=out_classes)

# if args.model == "lenet":
#     model = models.LeNet(input_channels=input_channels, out_classes=out_classes)
# elif args.model == "resnet8":
#     model = models.ResNet8(input_channels=input_channels, out_classes=out_classes)
# model.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Helper function for evaluation
def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    letter_correct = [0] * 26
    letter_total = [0] * 26
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            for i in range(26):
                letter_correct[i] += ((predicted == target) & (target == i)).sum().item()
                letter_total[i] += (target == i).sum().item()
    overall_accuracy = correct / total * 100
    per_letter_accuracy = [letter_correct[i] / letter_total[i] * 100 if letter_total[i] > 0 else 0 for i in range(26)]
    return overall_accuracy, per_letter_accuracy


def sequential_train_without_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training without replay buffer")
    for letter in range(26):
        print(f"\nTraining on letter {chr(65+letter)} only for {epochs} epochs")
        letter_loader = get_letter_loader(chr(65+letter), batch_size=args.batch_size, train=True)
        model.train()
        for epoch in range(epochs):
            for data, target in letter_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        overall_acc, per_letter_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on letter {chr(65+letter)}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_letter_acc):
            print(f"Accuracy on Letter {chr(65+i)}: {acc:.2f}%")


def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training with incremental replay buffer")
    replay_buffer = {}

    for letter_idx in range(26):  
        letter = chr(65 + letter_idx)  
        print(f"\nTraining on letter {letter} with replay buffer for {epochs} epochs")

        # Load current alphabet data
        letter_loader = get_letter_loader(letter, batch_size=args.batch_size, train=True)
        current_letter_samples = [(data, target) for data, target in letter_loader]

        # Add 50 samples per seen alphabet
        buffer_samples = []
        for seen_letter in replay_buffer:
            buffer_samples.extend(replay_buffer[seen_letter])

        # Store only 50 samples from the current letter
        replay_buffer[letter] = current_letter_samples[:10]  

        buffer_images = torch.cat([x if x.shape[0] == args.batch_size else x[:args.batch_size] for x, _ in buffer_samples + current_letter_samples])
        buffer_labels = torch.cat([y if y.shape[0] == args.batch_size else y[:args.batch_size] for _, y in buffer_samples + current_letter_samples])

        combined_dataset = TensorDataset(buffer_images, buffer_labels)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for data, target in combined_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        overall_acc, per_letter_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on letter {letter}:")  
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_letter_acc):
            letter_char = chr(65 + i) 
            print(f"Accuracy on Letter {letter_char}: {acc:.2f}%")


import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

def sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, epochs, test_loader, args):
    print("\nStarting sequential training with decoded replay buffer")

    buffer_path = "../../replay_buffer_isolet.pkl"

    # Load the replay buffer
    with open(buffer_path, "rb") as f:
        decoded_replay_buffer = pickle.load(f)
    print(f"Loaded buffer with {len(decoded_replay_buffer)} classes.")

    # Iterate through letters sequentially (A-Z)
    seen_letters = []  # Keep track of seen letters
    for letter_idx in range(26):  
        letter_char = chr(65 + letter_idx)  # Convert index to letter (0 -> 'A', 1 -> 'B', etc.)
        print(f"\nTraining on letter {letter_char} with replay buffer for {epochs} epochs")

        # Load current training letter samples
        current_letter_loader = get_letter_loader(letter_char, batch_size=args.batch_size, train=True)

        # Collect replay buffer samples (only from seen letters)
        buffer_samples = []
        for seen_letter in seen_letters:  # Only use previously seen letters
            buffer_samples.extend(decoded_replay_buffer[seen_letter][:5])  

        # Convert lists to tensors (Ensure non-empty tensors)
        buffer_samples = (
            torch.tensor(buffer_samples, dtype=torch.float32)
            if buffer_samples
            else torch.empty((0, *next(iter(current_letter_loader.dataset))[0].shape), dtype=torch.float32)
        )

        # Convert DataLoader to a Tensor (Extract images from DataLoader)
        current_samples = []
        for batch, _ in current_letter_loader:
            current_samples.append(batch)
        current_samples = torch.cat(current_samples, dim=0)  # Use all samples for the current letter

        # Stack all images together (handle empty buffer case)
        if buffer_samples.numel() == 0:
            all_images = current_samples
        else:
            all_images = torch.cat([buffer_samples, current_samples], dim=0)

        # Create labels for all data (buffer + current letter)
        buffer_labels = []
        for seen_letter in seen_letters:
            num_samples = len(decoded_replay_buffer[seen_letter][:5])
            buffer_labels.extend([ord(seen_letter) - 65] * num_samples)

        # Convert buffer_labels to tensor
        buffer_labels = torch.tensor(buffer_labels, dtype=torch.long) if buffer_labels else torch.empty(0, dtype=torch.long)

        current_labels = torch.full((len(current_samples),), letter_idx, dtype=torch.long)

        if buffer_labels.numel() == 0:
            all_labels = current_labels
        else:
            all_labels = torch.cat([buffer_labels, current_labels], dim=0)

        # Debugging prints
        print(f"Seen Letters: {seen_letters}")
        print(f"Buffer Samples Shape: {buffer_samples.shape}")
        print(f"Current Samples Shape: {current_samples.shape}")
        print(f"Buffer Labels Length: {buffer_labels.shape}")
        print(f"Current Labels Length: {current_labels.shape}")
        print(f"Final Dataset Shapes - Images: {all_images.shape}, Labels: {all_labels.shape}")

        # Check for mismatches before dataset creation
        assert all_images.shape[0] == all_labels.shape[0], "Error: Mismatch between images and labels!"

        # Create DataLoader
        combined_dataset = TensorDataset(all_images, all_labels)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # Model Training
        model.train()
        for epoch in range(epochs):
            for data, target in combined_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Add the current letter to seen letters
        seen_letters.append(letter_char)

        # Evaluate
        overall_acc, per_letter_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on letter {letter_char}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_letter_acc):
            print(f"Accuracy on Letter {chr(65 + i)}: {acc:.2f}%")


def train_with_decoded_buffer_only(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting training with only decoded buffer")

    with open("replay_buffer_isolet.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    buffer_dataset = TensorDataset(torch.tensor([item for sublist in decoded_replay_buffer.values() for item in sublist], dtype=torch.float32),
                                   torch.tensor([letter for letter, samples in decoded_replay_buffer.items() for _ in samples], dtype=torch.long))
    buffer_loader = DataLoader(buffer_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for data, target in buffer_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    overall_acc, _ = evaluate_accuracy(model, test_loader, device)
    print(f"\nFinal accuracy after training only on decoded buffer: {overall_acc:.2f}%")


def reset_model():
    if args.model == "mlp":
        model = mlp.MLP(input_dim=input_channels, output_dim=out_classes).to(device)
    elif args.model == "lenet":
        model = models.LeNet(input_channels=input_channels, out_classes=out_classes).to(device)
    elif args.model == "resnet8":
        model = models.ResNet8(input_channels=input_channels, out_classes=out_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


print("===== Running Experiments on ISOLET =====")

# Sequential Training Without Buffer
# print("\n===== Training Without Buffer =====")
# model, optimizer = reset_model()
# sequential_train_without_buffer(model, device, nn.CrossEntropyLoss().to(device), optimizer, args.epochs, test_loader)

# Sequential Training With Replay Buffer
# print("\n===== Training With Replay Buffer =====")
# model, optimizer = reset_model()
# sequential_train_with_buffer(model, device, nn.CrossEntropyLoss().to(device), optimizer, args.epochs, test_loader)

# Sequential Training With Decoded Replay Buffer
print("\n===== Training With Decoded Replay Buffer =====")
model, optimizer = reset_model()
sequential_train_with_buffer_using_decoded(model, device, nn.CrossEntropyLoss().to(device), optimizer, args.epochs, test_loader, args)


# Training Using Only Decoded Buffer
# print("\n===== Training With Only Decoded Buffer =====")
# model, optimizer = reset_model()
# train_with_decoded_buffer_only(model, device, nn.CrossEntropyLoss().to(device), optimizer, args.epochs, test_loader)

print("\n===== All Experiments Completed Successfully! =====")