import os
import sys
import argparse
import pickle
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_zoo import models
from model_zoo.models import mlp 


##############################################################################
# 1. Replicate Channels for MobileNet
##############################################################################
def replicate_channels(data):
    """
    If using MobileNet, we need 3 channels. MNIST is 1 channel by default.
    This function repeats the single channel to produce [batch, 3, H, W].
    """
    if data.dim() == 4 and data.shape[1] == 1:
        data = data.repeat(1, 3, 1, 1)
    return data

##############################################################################
# 2. Evaluate Accuracy
##############################################################################
def evaluate_accuracy(model, test_loader, device, model_type):
    """
    Evaluates overall and per-digit accuracy on a test loader.
    Expects test_loader to contain 10 classes (digits 0..9).
    """
    model.eval()
    correct = 0
    total = 0
    digit_correct = [0] * 10
    digit_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Replicate channels if using MobileNet
            if model_type == 'mobilenet':
                data = replicate_channels(data)

            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            for i in range(10):
                digit_correct[i] += ((predicted == target) & (target == i)).sum().item()
                digit_total[i] += (target == i).sum().item()

    overall_accuracy = 100.0 * correct / total
    per_digit_accuracy = [
        (100.0 * digit_correct[i] / digit_total[i]) if digit_total[i] > 0 else 0
        for i in range(10)
    ]
    return overall_accuracy, per_digit_accuracy

##############################################################################
# 3. Data Helpers: MNIST with TorchVision
##############################################################################
# We'll load the entire train/test sets with torchvision,
# then define get_digit_loader to filter a single digit.
def get_mnist_datasets():
    """
    Returns the (train_set, test_set) for MNIST using torchvision.
    """
    transform = transforms.ToTensor()

    train_set = torchvision.datasets.MNIST(
        root='mnist_data',
        train=True,
        download=True,
        transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root='mnist_data',
        train=False,
        download=True,
        transform=transform
    )
    return train_set, test_set

def get_digit_loader(digit, dataset, batch_size=64, train=True):
    """
    Filters the given MNIST dataset for samples with label == digit,
    and returns a DataLoader of those samples.
    If train=True, we assume dataset is the train set; otherwise it can be test set.
    """
    indices = [i for i, label in enumerate(dataset.targets) if label == digit]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

##############################################################################
# 4. Training Routines
##############################################################################
def sequential_train_without_buffer(model, device, criterion, optimizer, epochs, test_loader, args, train_set):
    """
    Train digit by digit (0..9) without any replay buffer.
    """
    print("\n=== Starting sequential training WITHOUT replay buffer ===")
    for digit in range(10):
        print(f"\nTraining on digit {digit} for {epochs} epochs")
        digit_loader = get_digit_loader(digit, train_set, batch_size=args.batch_size, train=True)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in digit_loader:
                data, target = data.to(device), target.to(device)
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

def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader, args, train_set):
    """
    Sequential training with an incremental replay buffer of original images.
    """
    print("\n=== Starting sequential training WITH incremental replay buffer ===")
    replay_buffer = []  # will store (image_tensor, label) pairs

    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {epochs} epochs")

        # 1. Add e.g. 50 images for each previously seen digit to the buffer
        for seen_digit in range(digit):
            seen_digit_loader = get_digit_loader(seen_digit, train_set, batch_size=args.batch_size, train=True)
            seen_digit_data = []
            seen_digit_labels = []
            for data, target in seen_digit_loader:
                data, target = data.to(device), target.to(device)
                if args.model == 'mobilenet':
                    data = replicate_channels(data)

                seen_digit_data.extend(data[:50])
                seen_digit_labels.extend(target[:50])
                if len(seen_digit_data) >= 50:
                    break
            replay_buffer.extend(zip(seen_digit_data, seen_digit_labels))

        # 2. Gather current digit images
        digit_loader = get_digit_loader(digit, train_set, batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            data, target = data.to(device), target.to(device)
            if args.model == 'mobilenet':
                data = replicate_channels(data)

            current_digit_images.extend(data)
            current_digit_labels.extend(target)

        # Combine buffer + current
        buffer_images = [img.clone().unsqueeze(0) if img.dim() == 3 else img.clone() for img, _ in replay_buffer]
        buffer_labels = [lbl.clone() for _, lbl in replay_buffer]
        current_digit_images = [img.unsqueeze(0) if img.dim() == 3 else img for img in current_digit_images]

        combined_images = buffer_images + current_digit_images
        combined_labels = buffer_labels + current_digit_labels

        if len(combined_images) == 0:
            print("No data to train on. Skipping digit", digit)
            continue

        combined_images_tensor = torch.cat(combined_images, dim=0)
        combined_labels_tensor = torch.stack(combined_labels) if len(buffer_labels) > 0 else torch.tensor([], dtype=torch.long)
        if len(current_digit_labels) > 0:
            curr_labels_tensor = torch.stack(current_digit_labels)
            if combined_labels_tensor.numel() == 0:
                combined_labels_tensor = curr_labels_tensor
            else:
                combined_labels_tensor = torch.cat([combined_labels_tensor, curr_labels_tensor], dim=0)

        # Create dataset
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


def sequential_train_with_buffer_using_decoded(
    model, device, criterion, optimizer,
    epochs, test_loader, args, train_set
):
    """
    Class‐incremental on 0..9.  For each digit d, we build a
    single DataLoader containing:
      - all decoded buffer imgs for labels < d
      - all raw MNIST imgs for label == d
    and train for `epochs` epochs over that loader.
    """
    print("\n=== Starting sequential training with DECODED replay buffer ===")

    # 1) Load & unpack replay_buffer.pkl
    with open("../../replay_buffer.pkl", "rb") as f:
        replay_dict = pickle.load(f)

    buf_imgs, buf_lbls = [], []
    for lbl, arrs in replay_dict.items():
        for arr in arrs:
            t = torch.tensor(arr, dtype=torch.float32)
            if t.ndim == 2:    # MNIST gray
                t = t.unsqueeze(0)
            buf_imgs.append(t)
            buf_lbls.append(int(lbl))

    buf_tensor = torch.stack(buf_imgs, dim=0)   # [N_buf, C, H, W]
    lbl_tensor = torch.tensor(buf_lbls, dtype=torch.long)
    full_buffer_ds = TensorDataset(buf_tensor, lbl_tensor)

    # 2) A tiny Subset wrapper that converts its label to a Tensor
    class LabelSubset(Subset):
        def __init__(self, dataset, indices):
            super().__init__(dataset, indices)
        def __getitem__(self, idx):
            img, lbl = super().__getitem__(idx)
            return img, torch.tensor(lbl, dtype=torch.long)

    # 3) Loop over digits
    for digit in range(10):
        print(f"\n--- Training on digit {digit} with buffer for {epochs} epochs ---")

        # 3.1 Build buffer subset of only classes < digit
        if digit > 0:
            mask = (lbl_tensor < digit)
            idxs = mask.nonzero(as_tuple=False).view(-1).tolist()
            seen_ds = Subset(full_buffer_ds, idxs)
        else:
            seen_ds = None

        # 3.2 Build new‐digit subset (and wrap its labels)
        new_idxs = [i for i, lab in enumerate(train_set.targets) if int(lab) == digit]
        new_ds   = LabelSubset(train_set, new_idxs)

        # 3.3 Concat old + new
        if seen_ds is not None:
            combined_ds = ConcatDataset([seen_ds, new_ds])
        else:
            combined_ds = new_ds

        combined_loader = DataLoader(
            combined_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0    # ← single‐process loader
        )

        # 4) Train
        model.train()
        for ep in range(epochs):
            running_loss = 0.0
            for x_cpu, y_cpu in combined_loader:
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                if args.model == 'mobilenet':
                    x = replicate_channels(x)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg = running_loss / len(combined_loader)
            print(f"Digit {digit}  Ep {ep+1}/{epochs}  loss={avg:.4f}")

        # 5) Evaluate on full test set
        overall_acc, per_digit_acc = evaluate_accuracy(
            model, test_loader, device, args.model
        )
        print(f"\nAfter digit {digit} → Overall: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"  Digit {i}: {acc:.2f}%")


        
def lifelong_learning(model, device, criterion, optimizer, test_loader, args):
    """
    Sequential training on digits 0..9 WITHOUT a replay buffer.
    Data is loaded digit by digit, 
    then in the final loop we move it to GPU and replicate channels if needed.
    """
    print("\nStarting sequential training without replay buffer")
    for digit in range(10):
        print(f"\nTraining on digit {digit} only for {args.epochs} epochs")

        # 1) Load digit data (CPU)
        digit_loader = get_digit_loader(digit, train_set,batch_size=args.batch_size, train=True)
        
        # 2) Train for 'epochs' epochs
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for data, target in digit_loader:
                # Only now do we move data to GPU (and replicate if mobilenet).
                if args.model == 'mobilenet':
                    data = replicate_channels(data)
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Digit {digit}, Epoch {epoch+1}/{args.epochs}, Loss: {running_loss / len(digit_loader):.4f}")

        # 3) Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")



def lifelong_learning_with_buffer(model, device, criterion, optimizer, test_loader, args):
    """
    Sequential training with an incremental replay buffer of original images.
    We'll store replay images on CPU. 
    Only in the final training loop do we move them to GPU and replicate channels if needed.
    """
    print("\nStarting sequential training with incremental replay buffer")
    replay_buffer = []  # will store (img_cpu, label)

    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {args.epochs} epochs")
        
        # 1) Add some images for each previously seen digit to the replay buffer (CPU)
        for seen_digit in range(digit):  
            seen_digit_loader = get_digit_loader(seen_digit, train_set, batch_size=args.batch_size, train=True)
            # We collect e.g. 5 images from the first batch
            for data, target in seen_digit_loader:
                # 'data' is on CPU by default
                # Just store them directly. We do not .to(device) yet.
                # We can store fewer images if we want, e.g. data[:5]
                for i in range(min(5, data.shape[0])):
                    replay_buffer.append((data[i], target[i]))
                break  # only the first batch for that digit

        # 2) Load the current digit's images
        digit_loader = get_digit_loader(digit, train_set, batch_size=args.batch_size, train=True)
        current_digit_data = []
        current_digit_labels = []
        for data, target in digit_loader:
            # also on CPU
            for i in range(data.shape[0]):
                current_digit_data.append(data[i])
                current_digit_labels.append(target[i])
        # Now we have CPU-based lists for current digit

        # 3) Combine replay buffer + current digit
        combined_data = []
        combined_labels = []
        # replay buffer
        for (img_cpu, lbl_cpu) in replay_buffer:
            combined_data.append(img_cpu.unsqueeze(0) if img_cpu.dim() == 3 else img_cpu)
            combined_labels.append(lbl_cpu)
        # current digit
        for i in range(len(current_digit_data)):
            img_cpu = current_digit_data[i]
            lbl_cpu = current_digit_labels[i]
            combined_data.append(img_cpu.unsqueeze(0) if img_cpu.dim() == 3 else img_cpu)
            combined_labels.append(lbl_cpu)

        if len(combined_data) == 0:
            print("No data to train on. Skipping digit", digit)
            continue

        # 4) Stack into a single CPU dataset
        combined_images_tensor = torch.cat(combined_data, dim=0)  # shape [N, C, H, W]
        combined_labels_tensor = torch.stack(combined_labels)     # shape [N]
        combined_dataset = TensorDataset(combined_images_tensor, combined_labels_tensor)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 5) Train for 'epochs' epochs
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for data, target in combined_loader:
                # Now do device + replicate
                if args.model == 'mobilenet':
                    data = replicate_channels(data)
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit}, Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(combined_loader):.4f}")

        # 6) Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit} with replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")




def lifelong_learning_with_buffer_using_decoded(model, device, criterion, optimizer, test_loader, args):
    """
    Sequential training with a decoded replay buffer (replay_buffer.pkl).
    We'll unify everything on CPU, then in final training loop, move to GPU & replicate if needed.
    """
    print("\nStarting sequential training with decoded replay buffer")
    
    with open("../../replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images")

    for digit in range(10):
        print(f"\nTraining on digit {digit} with decoded buffer for {args.epochs} epochs")

        # 1) Gather decoded images for previously seen digits (CPU)
        replay_images = []
        replay_labels = []
        for seen_digit in range(digit):
            decoded_images = decoded_replay_buffer[seen_digit]
            for img_np in decoded_images:
                # Create a CPU tensor
                img_t = torch.tensor(img_np, dtype=torch.float32)
                if img_t.dim() == 2:
                    img_t = img_t.unsqueeze(0)  # => [1,H,W]
                replay_images.append(img_t)
                replay_labels.append(torch.tensor(seen_digit, dtype=torch.long))

        # 2) Gather the current digit's images
        digit_loader = get_digit_loader(digit, train_set,batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            # data is on CPU
            for i in range(data.shape[0]):
                current_digit_images.append(data[i])
                current_digit_labels.append(target[i])

        # 3) Combine CPU replay + current
        combined_images = []
        combined_labels = []

        # replay
        for i, img in enumerate(replay_images):
            combined_images.append(img.unsqueeze(0) if img.dim() == 3 else img)
            combined_labels.append(replay_labels[i])
        # current
        for i, img in enumerate(current_digit_images):
            combined_images.append(img.unsqueeze(0) if img.dim() == 3 else img)
            combined_labels.append(current_digit_labels[i])

        if len(combined_images) == 0:
            print("No data for digit", digit)
            continue

        combined_images_tensor = torch.cat(combined_images, dim=0)   # CPU
        combined_labels_tensor = torch.stack(combined_labels)        # CPU
        combined_dataset = TensorDataset(combined_images_tensor, combined_labels_tensor)
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 4) Training
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for data, target in combined_loader:
                # Now move to GPU + replicate if mobilenet
                if args.model == 'mobilenet':
                    data = replicate_channels(data)
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Digit {digit}, Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(combined_loader):.4f}")

        # 5) Evaluate
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device, args.model)
        print(f"After training on digit {digit} with decoded buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")


##############################################################################
# MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='prefix name for the checkpoints')
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=500, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print("Device:", device)

    # Load dataset
    if args.dataset == 'mnist':
        train_set, test_set = get_mnist_datasets()
        out_classes = 10
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    elif args.dataset == 'cifar10':
        out_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        train_set = torchvision.datasets.CIFAR10(root='cifar_data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='cifar_data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    else:
        raise ValueError("Unknown dataset: " + args.dataset)

    # Create model
    if args.model == "mlp":
        from model_zoo.models import mlp
        # Adjust input_dim if you flatten images, etc.
        model = mlp.MLP(input_dim=28*28, output_dim=out_classes)
    elif args.model == "lenet":
        model = models.LeNet(input_channels=1 if args.dataset=='mnist' else 3, out_classes=out_classes)
    elif args.model == "resnet8":
        model = models.ResNet8(input_channels=1 if args.dataset=='mnist' else 3, out_classes=out_classes)
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

    # Example usage: pick whichever training function you want:
    # sequential_train_without_buffer(model, device, criterion, optimizer, args.epochs, test_loader, args, train_set)
    # sequential_train_with_buffer(model, device, criterion, optimizer, args.epochs, test_loader, args, train_set)
    sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, args.epochs, test_loader, args, train_set)
    # train_with_decoded_buffer_only(model, device, criterion, optimizer, args.epochs, args.batch_size, args)
    
    # lifelong_learning(model, device, criterion, optimizer, test_loader, args)
    # lifelong_learning_with_buffer(model, device, criterion, optimizer, test_loader, args)
    # lifelong_learning_with_buffer_using_decoded(model, device, criterion, optimizer, test_loader, args)


    print("\n===== All Experiments Completed Successfully! =====")


