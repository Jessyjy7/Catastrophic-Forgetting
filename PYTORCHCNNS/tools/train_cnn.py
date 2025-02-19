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
from model_zoo.datasets.digit_loader import get_digit_loader
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import adjust_contrast

###############################################################################
# 1. Helper: replicate_channels
###############################################################################
def replicate_channels(data):
    """
    For MobileNet (which expects 3 channels), we replicate single-channel MNIST
    data from [1, H, W] to [3, H, W].
    
    If data is [batch_size, 1, H, W], this becomes [batch_size, 3, H, W].
    """
    if data.dim() == 4 and data.shape[1] == 1:
        data = data.repeat(1, 3, 1, 1)  # replicate channel dimension
    return data

###############################################################################
# 2. Evaluate Accuracy
###############################################################################
def evaluate_accuracy(model, test_loader, device, model_type):
    model.eval()
    correct = 0
    total = 0
    digit_correct = [0] * 10
    digit_total = [0] * 10
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # If MobileNet, replicate channels
            if model_type == 'mobilenet':
                data = replicate_channels(data)

            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            for i in range(10):
                digit_correct[i] += ((predicted == target) & (target == i)).sum().item()
                digit_total[i] += (target == i).sum().item()

    overall_accuracy = correct / total * 100
    per_digit_accuracy = [
        digit_correct[i] / digit_total[i] * 100 if digit_total[i] > 0 else 0
        for i in range(10)
    ]
    return overall_accuracy, per_digit_accuracy

###############################################################################
# 3. Training Methods
###############################################################################
def sequential_train_without_buffer(model, device, criterion, optimizer, epochs, test_loader, args):
    print("\n=== Starting sequential training WITHOUT replay buffer ===")
    for digit in range(10):
        print(f"\nTraining on digit {digit} for {epochs} epochs")
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in digit_loader:
                data, target = data.to(device), target.to(device)
                # If MobileNet, replicate channels
                if args.model == 'mobilenet':
                    data = replicate_channels(data)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(digit_loader):.4f}")

        # Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader, args):
    print("\n=== Starting sequential training WITH incremental replay buffer ===")
    replay_buffer = []

    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {epochs} epochs")

        # 1. Add e.g. 50 original images for each previously seen digit to the replay buffer
        for seen_digit in range(digit):
            seen_digit_loader = get_digit_loader(seen_digit, batch_size=args.batch_size, train=True)
            seen_digit_data = []
            seen_digit_labels = []
            for data, target in seen_digit_loader:
                data, target = data.to(device), target.to(device)
                # If MobileNet, replicate channels
                if args.model == 'mobilenet':
                    data = replicate_channels(data)

                seen_digit_data.extend(data[:50])
                seen_digit_labels.extend(target[:50])
                if len(seen_digit_data) >= 50:
                    break
            # Store them in the replay buffer
            replay_buffer.extend(zip(seen_digit_data, seen_digit_labels))

        # 2. Gather original images for the current digit
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            data, target = data.to(device), target.to(device)
            if args.model == 'mobilenet':
                data = replicate_channels(data)

            current_digit_images.extend(data)
            current_digit_labels.extend(target)

        # Limit to 5000 if you want
        current_digit_images = current_digit_images[:5000]
        current_digit_labels = current_digit_labels[:5000]

        # Combine buffer + current
        buffer_images = [img.clone().unsqueeze(0) if img.dim() == 3 else img.clone() for img, _ in replay_buffer]
        buffer_labels = [lbl.clone() for _, lbl in replay_buffer]
        current_digit_images = [img.unsqueeze(0) if img.dim() == 3 else img for img in current_digit_images]

        combined_images = buffer_images + current_digit_images
        combined_labels = buffer_labels + current_digit_labels

        if len(combined_images) == 0:
            print("No data to train on. Skipping.")
            continue

        combined_images_tensor = torch.cat(combined_images, dim=0)
        combined_labels_tensor = torch.stack(combined_labels) if len(buffer_labels) > 0 else torch.tensor([], dtype=torch.long)
        if len(current_digit_labels) > 0:
            curr_labels_tensor = torch.stack(current_digit_labels)
            if combined_labels_tensor.numel() == 0:
                combined_labels_tensor = curr_labels_tensor
            else:
                combined_labels_tensor = torch.cat([combined_labels_tensor, curr_labels_tensor], dim=0)

        combined_dataset = TensorDataset(combined_images_tensor, combined_labels_tensor)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 3. Train
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in combined_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader):.4f}")

        # 4. Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit} with replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

def sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, epochs, test_loader, args):
    print("\n=== Starting sequential training with DECODED replay buffer ===")
    
    with open("/Users/jessyjy7/Desktop/Catastrophic-Forgetting/replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    # For each digit in the buffer, we have a list of decoded images
    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images, Example shape: {images[0].shape if images else None}")

    for digit in range(10):
        print(f"\nTraining on digit {digit} with decoded replay buffer for {epochs} epochs")

        # 1. Collect decoded images for previously seen digits
        replay_images = []
        replay_labels = []
        for seen_digit in range(digit):
            decoded_images = decoded_replay_buffer[seen_digit]
            for img_np in decoded_images:
                # shape might be [H, W] or [1, H, W]
                img_t = torch.tensor(img_np, dtype=torch.float32)
                if img_t.dim() == 2:
                    img_t = img_t.unsqueeze(0)  # => [1, H, W]
                # If MobileNet, replicate channels
                if args.model == 'mobilenet' and img_t.shape[0] == 1:
                    img_t = img_t.repeat(3, 1, 1)
                replay_images.append(img_t)
                replay_labels.append(torch.tensor(seen_digit, dtype=torch.long))

        # 2. Gather original images for current digit
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            data, target = data.to(device), target.to(device)
            # replicate channels if mobilenet
            if args.model == 'mobilenet':
                data = replicate_channels(data)
            current_digit_images.extend(data)
            current_digit_labels.extend(target)

        # Combine
        # replay_images are on CPU, current_digit_images on GPU => we can keep them on CPU for cat
        # let's just gather them all on CPU for uniform
        combined_images = []
        combined_labels = []

        for i, img in enumerate(replay_images):
            combined_images.append(img.unsqueeze(0))  # => [1, C, H, W]
            combined_labels.append(replay_labels[i])

        for i, img in enumerate(current_digit_images):
            if img.dim() == 3:
                combined_images.append(img.unsqueeze(0).cpu())
            else:
                combined_images.append(img.cpu())
            combined_labels.append(current_digit_labels[i].cpu())

        if len(combined_images) == 0:
            print("No data to train on for digit", digit)
            continue

        combined_images_tensor = torch.cat(combined_images, dim=0)
        combined_labels_tensor = torch.stack(combined_labels)

        combined_dataset = TensorDataset(combined_images_tensor, combined_labels_tensor)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 3. Train
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in combined_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader):.4f}")

        # 4. Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit} with decoded replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

def train_with_decoded_buffer_only(model, device, criterion, optimizer, epochs, batch_size, args):
    print("\n=== Starting training with decoded buffer ONLY ===")

    with open("replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    print("Replay Buffer Keys:", decoded_replay_buffer.keys())
    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images")

    # Gather all images from the decoded buffer
    buffer_images = []
    buffer_labels = []
    for digit in range(10):
        decoded_images = decoded_replay_buffer[digit]
        for img_np in decoded_images:
            img_t = torch.tensor(img_np, dtype=torch.float32)
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)  # => [1, H, W]
            # If MobileNet, replicate
            if args.model == 'mobilenet' and img_t.shape[0] == 1:
                img_t = img_t.repeat(3, 1, 1)
            buffer_images.append(img_t)
            buffer_labels.append(torch.tensor(digit, dtype=torch.long))

    buffer_images = [img.unsqueeze(0) if img.dim() == 3 else img for img in buffer_images]
    if len(buffer_images) == 0:
        print("No decoded images found!")
        return

    buffer_images_tensor = torch.cat(buffer_images, dim=0)
    buffer_labels_tensor = torch.stack(buffer_labels)

    buffer_dataset = TensorDataset(buffer_images_tensor, buffer_labels_tensor)
    buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=True)

    # Train
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in buffer_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(buffer_loader):.4f}")

    # Evaluate on buffer
    print("\nEvaluating model performance on decoded buffer (itself)...")
    overall_acc, per_digit_acc = evaluate_accuracy(model, buffer_loader, device, args.model)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    for digit, acc in enumerate(per_digit_acc):
        print(f"Accuracy on Digit {digit}: {acc:.2f}%")

###############################################################################
# MAIN
###############################################################################
parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True, help='prefix name for checkpoints')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=500, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
print("Device:", device)

if args.dataset == 'mnist':
    input_channels = 1
    out_classes = 10
    test_loader = datasets.mnist.load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = datasets.mnist.load_train_val_data(
        batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda
    )
elif args.dataset == 'cifar10':
    input_channels = 3
    out_classes = 10
    test_loader = datasets.cifar10.load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = datasets.cifar10.load_train_val_data(
        batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda
    )
else:
    raise ValueError("Unknown dataset")

# Instantiate model
if args.model == "lenet":
    model = models.LeNet(input_channels=input_channels, out_classes=out_classes)
elif args.model == "resnet8":
    model = models.ResNet8(input_channels=input_channels, out_classes=out_classes)
elif args.model == "mobilenet":
    from model_zoo.models import mobilenet
    model = mobilenet.MobileNetV2(config='base', num_classes=out_classes)
else:
    raise ValueError(f"Unknown model: {args.model}")

model.to(device)

# Weight initialization
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n===== Running Experiments =====")

# Example usage: pick whichever function you want to test:
# sequential_train_without_buffer(model, device, criterion, optimizer, args.epochs, test_loader, args)
# sequential_train_with_buffer(model, device, criterion, optimizer, args.epochs, test_loader, args)
sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, args.epochs, test_loader, args)
# train_with_decoded_buffer_only(model, device, criterion, optimizer, args.epochs, args.batch_size, args)

print("\n===== All Experiments Completed Successfully! =====")

