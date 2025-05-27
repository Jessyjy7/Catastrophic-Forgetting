import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_func

def get_full_dataset(name, train=True):
    if name == "MNIST":
        return MNIST("./data", train=train, download=True, transform=transforms.ToTensor())
    if name == "CIFAR10":
        return CIFAR10("./data", train=train, download=True, transform=transforms.ToTensor())
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

def train_decoder_without_hdc(model, loader, epochs, lr, dev):
    for p in model.encoders.parameters(): p.requires_grad = False
    for p in model.bottleneck.parameters(): p.requires_grad = False
    for p in model.to_latent.parameters(): p.requires_grad = False
    opt = optim.Adam(list(model.from_latent.parameters()) + list(model.decoders.parameters()) + list(model.final_conv.parameters()), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(dev)
            with torch.no_grad(): zc, skips = model.encode(x)
            dec = model.decode(zc, skips)
            loss = criterion(dec, x)
            opt.zero_grad(); loss.backward(); opt.step()

def evaluate_ssim_no_hdc(model, dataset, num_classes, num_samples, out_dir, dev):
    model.eval()
    per_class_ssim = {}
    for cls in range(num_classes):
        imgs = []
        for x, y in dataset:
            if y == cls:
                imgs.append(x)
                if len(imgs) >= num_samples:
                    break
        batch = torch.stack(imgs).to(dev)
        with torch.no_grad(): z, skips = model.encode(batch); dec = model.decode(z, skips)
        s = ssim_func(batch, dec, data_range=1.0, size_average=True).item()
        per_class_ssim[cls] = s
        print(f"Class {cls}: SSIM = {s:.4f}")
        orig = batch.cpu().numpy(); recon = dec.cpu().numpy()
        fig, ax = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        for i in range(num_samples):
            ax[0,i].imshow(orig[i].transpose(1,2,0), cmap='gray' if orig.shape[1]==1 else None); ax[0,i].axis('off')
            ax[1,i].imshow(recon[i].transpose(1,2,0), cmap='gray' if recon.shape[1]==1 else None); ax[1,i].axis('off')
        ax[0,0].set_title('Orig'); ax[1,0].set_title('Recon')
        plt.tight_layout(); plt.savefig(f"{out_dir}/recon_class_{cls}.png", dpi=150); plt.close(fig)
    mean_ssim = sum(per_class_ssim.values()) / len(per_class_ssim)
    print(f"Mean SSIM: {mean_ssim:.4f}")
    return per_class_ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10"], default="CIFAR10")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./")
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = (MNIST if args.dataset=="MNIST" else CIFAR10)("./data", train=True, download=True, transform=transforms.ToTensor())
    test_ds  = (MNIST if args.dataset=="MNIST" else CIFAR10)("./data", train=False, download=True, transform=transforms.ToTensor())
    in_ch = train_ds[0][0].shape[0]
    H, W = train_ds[0][0].shape[1:]
    model = UNetAutoencoder(in_ch, args.base_ch, args.latent_dim, H, W).to(dev)
    num_params = sum(p.numel() for p in model.parameters())
    size_kb = num_params * 4 / 1024
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Approximate model size: {size_kb:.2f} KB")
    num_classes = len(np.unique(np.array(train_ds.targets)))
    for c in range(num_classes):
        print(f"\n=== Training on class {c} ===")
        idxs = np.where(np.array(train_ds.targets)==c)[0]
        loader_c = DataLoader(Subset(train_ds, idxs), batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_decoder_without_hdc(model, loader_c, args.epochs, args.lr, dev)
        with torch.no_grad():
            w = model.encoders[0].conv[0].weight.data.cpu().flatten()
            print(f"Enc weights[:5] after class {c}: {w[:5].tolist()}")
        print(f"\n--- Eval after class {c} ---")
        evaluate_ssim_no_hdc(model, test_ds, num_classes, args.num_samples, args.out_dir, dev)

if __name__ == "__main__":
    main()
