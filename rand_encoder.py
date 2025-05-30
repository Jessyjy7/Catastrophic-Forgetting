import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_func

def get_full_dataset(name, train=True):
    if name == "MNIST":
        return MNIST("./data", train=train, download=True, transform=transforms.ToTensor())
    if name == "CIFAR10":
        return CIFAR10("./data", train=train, download=True, transform=transforms.ToTensor())
    if name == "CIFAR100":
        return CIFAR100("./data", train=train, download=True, transform=transforms.ToTensor())
    raise ValueError("Unsupported dataset")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)

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
        return down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res  = ResidualBlock(out_ch)
        self.att  = AttentionBlock(out_ch)
        self.conv = nn.Conv2d(out_ch*2, out_ch, 3, padding=1)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        return F.relu(self.conv(x))

class UNetAutoencoder(nn.Module):
    def __init__(self, in_ch, base_ch, latent_dim, height, width):
        super().__init__()
        levels = 0
        h, w = height, width
        while h >= 8 and w >= 8:
            h //= 2; w //= 2; levels += 1
        self.levels = levels

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

        # latent projections
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
        out = x
        for enc in self.encoders:
            out = enc(out)
        out = self.bottleneck(out)
        B, C, H, W = out.shape
        flat = out.view(B, -1)
        z    = self.to_latent(flat)
        return z

    def decode(self, z):
        B = z.size(0)
        flat = self.from_latent(z)
        out  = flat.view(B, -1, self.flat_h, self.flat_w)
        for dec in self.decoders:
            out = dec(out)
        return torch.sigmoid(self.final_conv(out))

def train_decoder_without_hdc(model, loader, epochs, lr, dev):
    for p in model.encoders.parameters(): p.requires_grad = False
    for p in model.bottleneck.parameters(): p.requires_grad = False
    for p in model.to_latent.parameters(): p.requires_grad = False
    opt = optim.Adam(list(model.from_latent.parameters()) + 
                     list(model.decoders.parameters()) + 
                     list(model.final_conv.parameters()), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(dev)
            opt.zero_grad()
            zc = model.encode(x)
            dec = model.decode(zc)
            loss = criterion(dec, x)
            loss.backward()
            opt.step()

def evaluate_ssim_no_hdc(model, test_loaders, num_classes, num_samples, out_dir, dev):
    model.eval()
    per_class_ssim = {}
    for cls in range(num_classes):
        loader = test_loaders[cls]
        s, i = 0, 0
        for x, y in loader:
            with torch.no_grad():
                z = model.encode(x)
                dec = model.decode(z)
                s = s + ssim_func(x, dec, data_range=1.0, size_average=True).item()
                i = i + 1
        per_class_ssim[cls] = s / 
        print(f"Class {cls}: SSIM = {per_class_ssim[cls]:.4f}")
    mean_ssim = sum(per_class_ssim.values()) / len(per_class_ssim)
    print(f"Mean SSIM: {mean_ssim:.4f}")
    return per_class_ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10", "CIFAR100"], default="CIFAR10")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./")
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)
    if args.dataset == "MNIST":
        train_ds = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_ds = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    if args.dataset == "CIFAR10":
        train_ds = CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
        test_ds = CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    if args.dataset == "CIFAR100":
        train_ds = CIFAR100("./data", train=True, download=True, transform=transforms.ToTensor())
        test_ds = CIFAR100("./data", train=False, download=True, transform=transforms.ToTensor())

    in_ch = train_ds[0][0].shape[0]
    H, W = train_ds[0][0].shape[1:]
    model = UNetAutoencoder(in_ch, args.base_ch, args.latent_dim, H, W).to(dev)
    num_params = sum(p.numel() for p in model.parameters())
    size_kb = num_params * 4 / 1024
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Approximate model size: {size_kb:.2f} KB")
    num_classes = len(np.unique(np.array(train_ds.targets)))
    test_loaders = []
    for i in range(num_classes):
        idxs = np.where(np.array(test_ds.targets)==i)[0]
        loader = DataLoader(Subset(test_ds, idxs), batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loaders.append(loader)
    print(f"\n--- Pre-Training Eval ---")
    evaluate_ssim_no_hdc(model, test_loaders, num_classes, args.num_samples, args.out_dir, dev)
    for c in range(num_classes):
        print(f"\n=== Training on class {c} ===")
        idxs = np.where(np.array(train_ds.targets)==c)[0]
        loader_c = DataLoader(Subset(train_ds, idxs), batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_decoder_without_hdc(model, loader_c, args.epochs, args.lr, dev)
        print(f"\n--- Eval after class {c} ---")
        evaluate_ssim_no_hdc(model, test_loaders, num_classes, args.num_samples, args.out_dir, dev)

if __name__ == "__main__":
    main()
