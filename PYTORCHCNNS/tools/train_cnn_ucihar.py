import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse as ap
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model_zoo.utils import *
from model_zoo import datasets
from model_zoo import models
from model_zoo.models import mlp
# Import UCI HAR loaders – ensure these functions exist (or create equivalents)
from model_zoo.datasets.ucihar import load_train_val_data, load_test_data
from model_zoo.datasets.ucihar_loader import get_activity_loader
import numpy as np

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name (e.g., mlp)')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (should be "ucihar")')
parser.add_argument('--checkpoint', type=str, required=True, help='Prefix name for the checkpoints')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=500, help='Input batch size for testing')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per activity')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
print("Device: {}".format(device))

# UCI HAR settings
if args.dataset == 'ucihar':  
    input_channels = 561  
    out_classes = 6  
    test_loader = load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = load_train_val_data(batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda)

if args.model == "mlp": 
    model = mlp.MLP(input_dim=input_channels, output_dim=out_classes).to(device)
# You can uncomment or add more model options as needed

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Define the list of activities in UCI HAR
activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    activity_correct = [0] * out_classes
    activity_total = [0] * out_classes
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            for i in range(out_classes):
                activity_correct[i] += ((predicted == target) & (target == i)).sum().item()
                activity_total[i] += (target == i).sum().item()
    overall_accuracy = correct / total * 100
    per_activity_accuracy = [activity_correct[i] / activity_total[i] * 100 if activity_total[i] > 0 else 0 for i in range(out_classes)]
    return overall_accuracy, per_activity_accuracy

def sequential_train_without_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training without replay buffer")
    for idx, activity in enumerate(activities):
        print(f"\nTraining on activity {activity} only for {epochs} epochs")
        activity_loader = get_activity_loader(activity, batch_size=args.batch_size, train=True)
        model.train()
        for epoch in range(epochs):
            for data, target in activity_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        overall_acc, per_activity_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on activity {activity}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_activity_acc):
            print(f"Accuracy on {activities[i]}: {acc:.2f}%")

def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training with incremental replay buffer")
    replay_buffer = {}

    for idx, activity in enumerate(activities):  
        print(f"\nTraining on activity {activity} with replay buffer for {epochs} epochs")

        # Load current activity data
        activity_loader = get_activity_loader(activity, batch_size=args.batch_size, train=True)
        current_activity_samples = [(data, target) for data, target in activity_loader]

        # Collect replay samples from previously seen activities
        buffer_samples = []
        for seen_activity in replay_buffer:
            buffer_samples.extend(replay_buffer[seen_activity])

        # Store a fixed number of samples from the current activity
        replay_buffer[activity] = current_activity_samples[:10]  

        # Concatenate samples ensuring batch consistency
        combined_data = []
        combined_labels = []
        for batch, labels in buffer_samples + current_activity_samples:
            # Adjust each batch if necessary to have a consistent size
            combined_data.append(batch if batch.shape[0] == args.batch_size else batch[:args.batch_size])
            combined_labels.append(labels if labels.shape[0] == args.batch_size else labels[:args.batch_size])
        buffer_images = torch.cat(combined_data)
        buffer_labels = torch.cat(combined_labels)

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

        overall_acc, per_activity_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on activity {activity}:")  
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_activity_acc):
            print(f"Accuracy on {activities[i]}: {acc:.2f}%")

def sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, epochs, test_loader, args):
    print("\nStarting sequential training with decoded replay buffer")
    buffer_path = "../../replay_buffer_ucihar.pkl"  # Adjust the path as needed

    with open(buffer_path, "rb") as f:
        decoded_replay_buffer = pickle.load(f)
    print(f"Loaded buffer with {len(decoded_replay_buffer)} activities.")

    seen_activities = []  # Keep track of seen activities
    for idx, activity in enumerate(activities):  
        print(f"\nTraining on activity {activity} with replay buffer for {epochs} epochs")

        # Load current training activity samples
        current_activity_loader = get_activity_loader(activity, batch_size=args.batch_size, train=True)

        # Collect replay buffer samples from previously seen activities
        buffer_samples = []
        for seen_activity in seen_activities:
            # Use a fixed number of samples per seen activity
            buffer_samples.extend(decoded_replay_buffer[seen_activity][:5])  

        # Convert lists to tensors
        buffer_samples = (
            torch.tensor(buffer_samples, dtype=torch.float32)
            if buffer_samples
            else torch.empty((0, *next(iter(current_activity_loader.dataset))[0].shape), dtype=torch.float32)
        )

        # Gather current activity samples from the loader
        current_samples = []
        for batch, _ in current_activity_loader:
            current_samples.append(batch)
        current_samples = torch.cat(current_samples, dim=0)

        if buffer_samples.numel() == 0:
            all_images = current_samples
        else:
            all_images = torch.cat([buffer_samples, current_samples], dim=0)

        # Create labels: buffer samples from seen activities and current activity labels
        buffer_labels = []
        for seen_activity in seen_activities:
            num_samples = len(decoded_replay_buffer[seen_activity][:5])
            # Label based on the index of the activity in the list
            buffer_labels.extend([activities.index(seen_activity)] * num_samples)

        buffer_labels = torch.tensor(buffer_labels, dtype=torch.long) if buffer_labels else torch.empty(0, dtype=torch.long)
        current_labels = torch.full((len(current_samples),), idx, dtype=torch.long)

        if buffer_labels.numel() == 0:
            all_labels = current_labels
        else:
            all_labels = torch.cat([buffer_labels, current_labels], dim=0)

        # Debug prints
        print(f"Seen Activities: {seen_activities}")
        print(f"Buffer Samples Shape: {buffer_samples.shape}")
        print(f"Current Samples Shape: {current_samples.shape}")
        print(f"Final Dataset Shapes - Images: {all_images.shape}, Labels: {all_labels.shape}")

        assert all_images.shape[0] == all_labels.shape[0], "Error: Mismatch between images and labels!"

        combined_dataset = TensorDataset(all_images, all_labels)
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

        seen_activities.append(activity)

        overall_acc, per_activity_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on activity {activity}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_activity_acc):
            print(f"Accuracy on {activities[i]}: {acc:.2f}%")

def train_with_decoded_buffer_only(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting training with only decoded buffer")

    with open("replay_buffer_ucihar.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    # Create a dataset from the decoded buffer
    images = []
    labels = []
    for activity, samples in decoded_replay_buffer.items():
        images.extend(samples)
        labels.extend([activities.index(activity)] * len(samples))
    buffer_dataset = TensorDataset(torch.tensor(images, dtype=torch.float32),
                                   torch.tensor(labels, dtype=torch.long))
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

print("===== Running Experiments on UCI HAR =====")

# Uncomment the desired training experiment

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
