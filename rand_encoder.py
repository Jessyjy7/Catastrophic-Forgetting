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
        return MNIST(
            "./data", train=train, download=True,
            transform=transforms.ToTensor()
        )
    if name == "CIFAR10":
        return CIFAR10(
            "./data", train=train, download=True,
            transform=transforms.ToTensor()
        )
    raise ValueError(f"Unsupported dataset {name}")

def generate_hadamard(n, device):
    H = torch.tensor([[1.]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + res
        return F.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class EncoderBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(True),
            ResidualBlock(oc)
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)

class DecoderBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up = nn.ConvTranspose2d(ic, oc, 2, stride=2)
        self.res = ResidualBlock(oc)
        self.att = AttentionBlock(oc)
        self.conv = nn.Conv2d(oc + sc, oc, 3, padding=1)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        x = torch.cat([x, skip], dim=1)
        return F.relu(self.conv(x))

class UNetAutoencoder(nn.Module):
    def __init__(self, inc, bc, ld, H, W):
        super().__init__()
        lvl, h, w = 0, H, W
        while h >= 8 and w >= 8:
            h //= 2; w //= 2; lvl += 1
        self.encoders = nn.ModuleList()
        scs = [bc * (2**i) for i in range(lvl)]
        ic = inc
        for sc in scs:
            self.encoders.append(EncoderBlock(ic, sc))
            ic = sc
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ic, ic, 3, padding=1),
            nn.BatchNorm2d(ic),
            nn.ReLU(True),
            ResidualBlock(ic)
        )
        self.flat_h, self.flat_w = h, w
        flat_ch = ic * h * w
        self.to_latent = nn.Linear(flat_ch, ld)
        self.from_latent = nn.Linear(ld, flat_ch)
        self.decoders = nn.ModuleList()
        din = scs[-1]
        for sc in reversed(scs):
            oc = sc // 2
            self.decoders.append(DecoderBlock(din, sc, oc))
            din = oc
        self.final_conv = nn.Conv2d(din, inc, 1)

    def encode(self, x):
        skips, out = [], x
        for enc in self.encoders:
            s, out = enc(out); skips.append(s)
        out = self.bottleneck(out)
        B, C, hh, ww = out.shape
        flat = out.view(B, -1)
        return self.to_latent(flat), skips

    def decode(self, z, skips):
        B = z.size(0)
        out = self.from_latent(z).view(B, -1, self.flat_h, self.flat_w)
        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)
        return torch.sigmoid(self.final_conv(out))


def train_decoder_without_hdc(model, loader, epochs, lr, dev):
    for p in model.encoders.parameters(): p.requires_grad = False
    for p in model.bottleneck.parameters(): p.requires_grad = False
    for p in model.to_latent.parameters(): p.requires_grad = False

    opt = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()), lr=lr
    )
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for x, _ in loader:
            x = x.to(dev)
            with torch.no_grad():
                zc, skips = model.encode(x)
            dec = model.decode(zc, skips)
            loss = criterion(dec, x)
            opt.zero_grad()
            loss.backward()
            opt.step()


def evaluate_ssim_no_hdc(model, dataset, num_classes, num_samples, dev):
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
        with torch.no_grad():
            z, skips = model.encode(batch)
            dec = model.decode(z, skips)
        s = ssim_func(batch, dec, data_range=1.0, size_average=True).item()
        per_class_ssim[cls] = s
        print(f"Class {cls}: SSIM = {s:.4f}")
        orig = batch.cpu().numpy()
        recon = dec.cpu().numpy()
        fig, ax = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        for i in range(num_samples):
            ax[0,i].imshow(orig[i].transpose(1,2,0),
                           cmap='gray' if orig.shape[1]==1 else None)
            ax[0,i].axis('off')
            ax[1,i].imshow(recon[i].transpose(1,2,0),
                           cmap='gray' if recon.shape[1]==1 else None)
            ax[1,i].axis('off')
        ax[0,0].set_title('Original'); ax[1,0].set_title('Reconstructed')
        plt.tight_layout()
        plt.show()
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
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = (MNIST if args.dataset=="MNIST" else CIFAR10)(
        "./data", train=True, download=True,
        transform=transforms.ToTensor())
    test_ds = (MNIST if args.dataset=="MNIST" else CIFAR10)(
        "./data", train=False, download=True,
        transform=transforms.ToTensor())

    in_ch = train_ds[0][0].shape[0]
    H, W = train_ds[0][0].shape[1:]
    model = UNetAutoencoder(in_ch, args.base_ch,
                            args.latent_dim, H, W).to(dev)

    num_classes = len(np.unique(np.array(train_ds.targets)))
    encoder_history = [] 

    for c in range(num_classes):
        print(f"\n=== Training on class {c} (no HDC) ===")
        idxs = np.where(np.array(train_ds.targets)==c)[0]
        loader_c = DataLoader(Subset(train_ds, idxs),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=4)

        train_decoder_without_hdc(model, loader_c,
                                   args.epochs, args.lr, dev)

        with torch.no_grad():
            w = model.encoders[0].conv[0].weight.data.cpu().flatten()
            first5 = w[:5].tolist()
            encoder_history.append(first5)
            print(f"Encoder weights after class {c}: {first5}")

        print(f"\n--- Evaluation after training class {c} ---")
        evaluate_ssim_no_hdc(model, test_ds, num_classes,
                             args.num_samples, dev)

    print("\n=== Encoder weight changes over classes ===")
    for i, ws in enumerate(encoder_history):
        print(f"Class {i} weights: {ws}")


if __name__ == "__main__":
    main()
