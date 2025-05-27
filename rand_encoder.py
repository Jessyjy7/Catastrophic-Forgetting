import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as functional
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as compute_ssim

def load_full_dataset(dataset_name: str, train: bool = True):
    if dataset_name == "MNIST":
        return MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
    if dataset_name == "CIFAR10":
        return CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first_convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.first_batch_norm = nn.BatchNorm2d(channels)
        self.second_convolution = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.second_batch_norm = nn.BatchNorm2d(channels)

    def forward(self, input_tensor: torch.Tensor):
        identity = input_tensor
        out = functional.relu(self.first_batch_norm(self.first_convolution(input_tensor)))
        out = self.second_batch_norm(self.second_convolution(out)) + identity
        return functional.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention_convolution = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input_tensor: torch.Tensor):
        attention_map = torch.sigmoid(self.attention_convolution(input_tensor))
        return input_tensor * attention_map

class EncoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.encoding_convolution = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(output_channels)
        )
        self.downsampling = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_tensor: torch.Tensor):
        skip_tensor = self.encoding_convolution(input_tensor)
        downsampled_tensor = self.downsampling(skip_tensor)
        return skip_tensor, downsampled_tensor

class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.upsampling = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(output_channels)
        self.attention_block = AttentionBlock(output_channels)
        self.decoding_convolution = nn.Conv2d(output_channels + skip_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor: torch.Tensor, skip_tensor: torch.Tensor):
        x = self.upsampling(input_tensor)
        x = self.residual_block(x)
        x = self.attention_block(x)
        x = torch.cat([x, skip_tensor], dim=1)
        return functional.relu(self.decoding_convolution(x))

class UNetAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        latent_dimension: int,
        image_height: int,
        image_width: int
    ):
        super().__init__()
        levels = 0
        h, w = image_height, image_width
        while h >= 8 and w >= 8:
            h //= 2
            w //= 2
            levels += 1

        encoder_channel_list = [base_channels * (2**i) for i in range(levels)]
        self.encoder_blocks = nn.ModuleList()
        in_c = input_channels
        for out_c in encoder_channel_list:
            self.encoder_blocks.append(EncoderBlock(in_c, out_c))
            in_c = out_c

        bottleneck_channels = encoder_channel_list[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(bottleneck_channels)
        )

        self.flat_height, self.flat_width = h, w
        flat_channels = bottleneck_channels * h * w
        self.to_latent = nn.Linear(flat_channels, latent_dimension)
        self.from_latent = nn.Linear(latent_dimension, flat_channels)

        self.decoder_blocks = nn.ModuleList()
        in_c = encoder_channel_list[-1]
        for skip_c in reversed(encoder_channel_list):
            out_c = skip_c // 2
            self.decoder_blocks.append(DecoderBlock(in_c, skip_c, out_c))
            in_c = out_c

        self.output_convolution = nn.Conv2d(in_c, input_channels, kernel_size=1)

    def encode(self, input_tensor: torch.Tensor):
        skip_tensors = []
        x = input_tensor
        for block in self.encoder_blocks:
            skip, x = block(x)
            skip_tensors.append(skip)
        x = self.bottleneck(x)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        latent_vector = self.to_latent(x_flat)
        return latent_vector, skip_tensors

    def decode(self, latent_vector: torch.Tensor, skip_tensors: list):
        batch_size = latent_vector.size(0)
        x = self.from_latent(latent_vector).view(batch_size, -1, self.flat_height, self.flat_width)
        for block, skip in zip(self.decoder_blocks, reversed(skip_tensors)):
            x = block(x, skip)
        return torch.sigmoid(self.output_convolution(x))


def train_decoder_without_replay(
    model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
):
    for parameter in model.encoder_blocks.parameters():
        parameter.requires_grad = False
    for parameter in model.bottleneck.parameters():
        parameter.requires_grad = False
    for parameter in model.to_latent.parameters():
        parameter.requires_grad = False

    optimizer = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoder_blocks.parameters()) +
        list(model.output_convolution.parameters()),
        lr=learning_rate
    )
    loss_function = nn.MSELoss()
    model.train()

    for _ in range(num_epochs):
        for images, _ in data_loader:
            images = images.to(device)
            with torch.no_grad():
                latent_vectors, skip_tensors = model.encode(images)
            reconstructions = model.decode(latent_vectors, skip_tensors)
            loss = loss_function(reconstructions, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate_reconstruction_ssim(
    model: nn.Module,
    dataset,
    number_of_classes: int,
    evaluation_samples: int,
    output_directory: str,
    device: torch.device
):
    model.eval()
    per_class_scores = {}

    for class_index in range(number_of_classes):
        collected_images = []
        for image, label in dataset:
            if label == class_index:
                collected_images.append(image)
                if len(collected_images) >= evaluation_samples:
                    break
        batch = torch.stack(collected_images).to(device)
        with torch.no_grad():
            latent_vectors, skip_tensors = model.encode(batch)
            reconstructions = model.decode(latent_vectors, skip_tensors)
        score = compute_ssim(batch, reconstructions, data_range=1.0, size_average=True).item()
        per_class_scores[class_index] = score
        print(f"Class {class_index}: SSIM = {score:.4f}")

        originals = batch.cpu().numpy()
        reconstructed = reconstructions.cpu().numpy()
        figure, axes = plt.subplots(2, evaluation_samples, figsize=(2 * evaluation_samples, 4))
        for i in range(evaluation_samples):
            axes[0, i].imshow(originals[i].transpose(1, 2, 0), cmap='gray' if originals.shape[1] == 1 else None)
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i].transpose(1, 2, 0), cmap='gray' if reconstructed.shape[1] == 1 else None)
            axes[1, i].axis('off')
        axes[0, 0].set_title('Original')
        axes[1, 0].set_title('Reconstructed')
        plt.tight_layout()
        figure.savefig(f"{output_directory}/reconstruction_class_{class_index}.png", dpi=150)
        plt.close(figure)

    mean_score = sum(per_class_scores.values()) / len(per_class_scores)
    print(f"Mean SSIM across classes: {mean_score:.4f}")
    return per_class_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="CIFAR10")
    parser.add_argument("--latent-dimension", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-samples", type=int, default=10)
    parser.add_argument("--output-directory", type=str, default="./")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset = load_full_dataset(args.dataset, train=True)
    testing_dataset = load_full_dataset(args.dataset, train=False)

    sample_image, _ = training_dataset[0]
    input_channels, image_height, image_width = sample_image.shape

    model = UNetAutoencoder(
        input_channels=input_channels,
        base_channels=args.base_channels,
        latent_dimension=args.latent-dimension,
        image_height=image_height,
        image_width=image_width
    ).to(device)

    total_parameters = sum(param.numel() for param in model.parameters())
    model_size_bytes = total_parameters * 4
    model_size_kilobytes = model_size_bytes / 1024
    print(f"Model parameters: {total_parameters:,} ({total_parameters/1e6:.2f}M)")
    print(f"Approximate model size: {model_size_kilobytes:.2f} KB")

    number_of_classes = len(np.unique(np.array(training_dataset.targets)))
    for class_index in range(number_of_classes):
        print(f"\n=== Training on class {class_index} ===")
        indices = np.where(np.array(training_dataset.targets) == class_index)[0]
        data_loader = DataLoader(
            Subset(training_dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        train_decoder_without_replay(
            model,
            data_loader,
            args.epochs,
            args.learning_rate,
            device
        )

        with torch.no_grad():
            encoder_weight_tensor = model.encoder_blocks[0].encoding_convolution[0].weight.data.cpu().flatten()
            print(f"Encoder first-layer weights[:5] after class {class_index}: {encoder_weight_tensor[:5].tolist()}")

        print(f"\n--- Evaluating after training class {class_index} ---")
        evaluate_reconstruction_ssim(
            model,
            testing_dataset,
            number_of_classes,
            args.evaluation_samples,
            args.output-directory,
            device
        )

if __name__ == '__main__':
    main()

