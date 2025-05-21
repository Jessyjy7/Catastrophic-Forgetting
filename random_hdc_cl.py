# experiment2_fixed.py

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Subset
from pytorch_msssim import ssim as ssim_func
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_full_dataset(name):
    if name == "MNIST":
        return MNIST("./data", train=True, download=True,
                     transform=transforms.ToTensor())
    if name == "CIFAR10":
        return CIFAR10("./data", train=True, download=True,
                       transform=transforms.ToTensor())
    raise ValueError(f"Unsupported dataset {name}")

def generate_hadamard(n, device):
    H = torch.tensor([[1.]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(c)
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
        self.up   = nn.ConvTranspose2d(ic, oc, 2, stride=2)
        self.res  = ResidualBlock(oc)
        self.att  = AttentionBlock(oc)
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
        self.to_latent   = nn.Linear(flat_ch, ld)
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

def train_decoder_with_hdc(model, loader, d, g, epochs, lr, dev):
    for p in model.encoders.parameters():    p.requires_grad = False
    for p in model.bottleneck.parameters():   p.requires_grad = False
    for p in model.to_latent.parameters():    p.requires_grad = False
    Hm = generate_hadamard(d, dev)
    opt = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()), lr=lr
    )
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                zc, skips = model.encode(x)
            B, ld = zc.shape
            u, cnt = torch.unique(y, return_counts=True)
            v = u[cnt >= g]
            if v.numel() == 0: continue
            cls = v[torch.randint(len(v), (1,)).item()]
            idx = (y == cls).nonzero(as_tuple=False).view(-1)
            sel = idx[torch.randperm(idx.size(0))[:g]]
            grp = zc[sel]
            if ld < d:
                pad = torch.zeros((g, d-ld), device=dev)
                grp = torch.cat([grp, pad], dim=1)
            ids = torch.arange(g, device=dev) % d
            ks  = Hm[ids]
            bundle = (ks * grp).sum(0)
            rec    = bundle.unsqueeze(0) * ks
            zn     = rec[:, :ld]
            skips_sel = [s[sel] for s in skips]
            dec = model.decode(zn, skips_sel)
            loss = nn.MSELoss()(dec, x[sel])
            opt.zero_grad(); loss.backward(); opt.step()

def evaluate_ssim_hdc(model, dataset_name, seen, num_samples, d, dev):
    model.eval()
    Hm = generate_hadamard(d, dev)
    ds = get_full_dataset(dataset_name)
    tot = 0.0
    for cls in seen:
        imgs = []
        for x, y in ds:
            if y == cls:
                imgs.append(x)
                if len(imgs) >= num_samples:
                    break
        batch = torch.stack(imgs).to(dev)
        with torch.no_grad():
            z, skips = model.encode(batch)
        B, ld = z.shape
        if ld < d:
            pad = torch.zeros((B, d-ld), device=dev)
            z   = torch.cat([z, pad], dim=1)
        key    = Hm[cls % d].unsqueeze(0)
        bundle = z * key
        rec    = bundle * key
        z_rec  = rec[:, :ld]
        skips2 = [s[:B] for s in skips]
        with torch.no_grad():
            dec = model.decode(z_rec, skips2)
            s   = ssim_func(batch, dec, data_range=1.0, size_average=True).item()
        tot += s
    avg = tot / len(seen)
    print(f"{seen}: AvgSSIM {avg:.4f}")

def plot_reconstructions_hdc(model, dataset_name, cls, num_samples, d, dev, out_path):
    model.eval()
    Hm = generate_hadamard(d, dev)
    ds = get_full_dataset(dataset_name)
    imgs = []
    for x, y in ds:
        if y == cls:
            imgs.append(x)
            if len(imgs) >= num_samples:
                break
    batch = torch.stack(imgs).to(dev)
    with torch.no_grad():
        z, skips = model.encode(batch)
    B, ld = z.shape
    if ld < d:
        pad = torch.zeros((B, d-ld), device=dev)
        z   = torch.cat([z, pad], dim=1)
    key    = Hm[cls % d].unsqueeze(0)
    bundle = z * key
    rec    = bundle * key
    z_rec  = rec[:, :ld]
    skips2 = [s[:B] for s in skips]
    with torch.no_grad():
        dec = model.decode(z_rec, skips2)
    orig = batch.cpu().numpy()
    recon = dec.cpu().numpy()
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    for i in range(num_samples):
        axes[0, i].imshow(orig[i].transpose(1,2,0),
                          cmap='gray' if orig.shape[1]==1 else None)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].transpose(1,2,0),
                          cmap='gray' if recon.shape[1]==1 else None)
        axes[1, i].axis('off')
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["MNIST","CIFAR10"], default="CIFAR10")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--base_ch",    type=int, default=16)
    parser.add_argument("--group_size", type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--num_samples",type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = get_full_dataset(args.dataset)
    labels = np.array(ds.targets)
    num_classes = len(np.unique(labels))

    in_ch = ds[0][0].shape[0]
    H = ds[0][0].shape[1]; W = ds[0][0].shape[2]
    model = UNetAutoencoder(in_ch, args.base_ch, args.latent_dim, H, W).to(device)

    seen = []
    for c in range(num_classes):
        seen.append(c)
        idx = np.where(labels == c)[0]
        sub = Subset(ds, idx)
        loader = DataLoader(sub, batch_size=args.batch_size,
                            shuffle=True, pin_memory=True, num_workers=4)

        train_decoder_with_hdc(model, loader,
            args.latent_dim, args.group_size, args.epochs, args.lr, device)

        evaluate_ssim_hdc(model, args.dataset,
                          seen, args.num_samples,
                          args.latent_dim, device)

        plot_reconstructions_hdc(model, args.dataset,
                                 c, args.num_samples,
                                 args.latent_dim, device,
                                 f"recon_class_{c}.png")

if __name__ == "__main__":
    main()
