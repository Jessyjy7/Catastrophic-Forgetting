import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Subset
from pytorch_msssim import ssim as ssim_func
import matplotlib.pyplot as plt

def get_full_dataset(dataset_name):
    if dataset_name == "MNIST":
        tf = transforms.ToTensor()
        return MNIST("./data", train=True, download=True, transform=tf)
    elif dataset_name == "CIFAR10":
        tf = transforms.ToTensor()
        return CIFAR10("./data", train=True, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        res = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return nn.functional.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock(out_ch)
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res  = ResidualBlock(out_ch)
        self.att  = AttentionBlock(out_ch)
        self.conv = nn.Conv2d(out_ch*2, out_ch, 3, padding=1)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        x = torch.cat([x, skip], dim=1)
        return nn.functional.relu(self.conv(x))

class UNetAutoencoder(nn.Module):
    def __init__(self, in_ch, base_ch, latent_dim, height, width):
        super().__init__()
        levels = 0
        h, w = height, width
        while h >= 8 and w >= 8:
            h //= 2; w //= 2; levels += 1
        self.encoders = nn.ModuleList()
        ch = base_ch; in_c = in_ch
        for _ in range(levels):
            self.encoders.append(EncoderBlock(in_c, ch))
            in_c = ch; ch *= 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            ResidualBlock(in_c)
        )
        self.flat_h, self.flat_w = h, w
        flat_ch = in_c * h * w
        self.to_latent   = nn.Linear(flat_ch, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_ch)
        self.decoders = nn.ModuleList()
        ch //= 2
        for _ in range(levels):
            self.decoders.append(DecoderBlock(in_c, ch))
            in_c = ch; ch //= 2
        self.final_conv = nn.Conv2d(in_c, in_ch, 1)

    def encode(self, x):
        skips, out = [], x
        for enc in self.encoders:
            skip, out = enc(out)
            skips.append(skip)
        out = self.bottleneck(out)
        B, C, H, W = out.shape
        flat = out.view(B, -1)
        z    = self.to_latent(flat)
        return z, skips

    def decode(self, z, skips):
        B = z.size(0)
        flat = self.from_latent(z)
        out  = flat.view(B, -1, self.flat_h, self.flat_w)
        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)
        return torch.sigmoid(self.final_conv(out))

def train_decoder_only(model, loader, epochs, lr, device):
    for p in model.encoders.parameters():    p.requires_grad = False
    for p in model.bottleneck.parameters():   p.requires_grad = False
    for p in model.to_latent.parameters():    p.requires_grad = False
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()),
        lr=lr
    )
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                z, skips = model.encode(images)
            recon = model.decode(z, skips)
            loss  = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch}/{epochs}] MSE: {total_loss/len(loader):.6f}")

def integration_pipeline(model, loader, num_samples, device):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)[:num_samples]
    with torch.no_grad():
        z, skips = model.encode(images)
        decoded = model.decode(z, skips)
        score   = ssim_func(images, decoded, data_range=1.0, size_average=True).item()
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10"], default="CIFAR10")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = get_full_dataset(args.dataset)
    labels = torch.tensor(full_ds.targets)
    num_classes = len(torch.unique(labels))
    split = num_classes // 2
    idx1 = torch.where(labels < split)[0].tolist()
    idx2 = torch.where(labels >= split)[0].tolist()
    ds1 = Subset(full_ds, idx1); ds2 = Subset(full_ds, idx2)
    loader1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    model = UNetAutoencoder(full_ds[0][0].shape[0], args.base_ch, args.latent_dim,
                            full_ds[0][0].shape[1], full_ds[0][0].shape[2]).to(device)
    train_decoder_only(model, loader1, args.epochs, args.lr, device)
    score1 = integration_pipeline(model, loader1, args.num_samples, device)
    print(f"Task1 SSIM: {score1:.4f}")
    train_decoder_only(model, loader2, args.epochs, args.lr, device)
    score1_post = integration_pipeline(model, loader1, args.num_samples, device)
    score2      = integration_pipeline(model, loader2, args.num_samples, device)
    print(f"After Task2 SSIM1: {score1_post:.4f}, SSIM2: {score2:.4f}")

if __name__ == "__main__":
    main()
