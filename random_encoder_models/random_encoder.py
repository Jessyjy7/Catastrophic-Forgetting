
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_func
import matplotlib.pyplot as plt

def get_loader(dataset_name, batch_size, shuffle=True):
    if dataset_name == "MNIST":
        in_ch, H, W = 1, 28, 28
        tf = transforms.ToTensor()
        ds = MNIST("./data", train=True, download=True, transform=tf)
    elif dataset_name == "CIFAR10":
        in_ch, H, W = 3, 32, 32
        tf = transforms.ToTensor()
        ds = CIFAR10("./data", train=True, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return loader, in_ch, H, W

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

def integration_pipeline(model, loader, num_samples, device, out_path):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)[:num_samples]
    with torch.no_grad():
        z, skips = model.encode(images)
        decoded = model.decode(z, skips)
        score   = ssim_func(images, decoded, data_range=1.0, size_average=True).item()
    print(f"SSIM: {score:.4f}")
    orig, recon = images.cpu().numpy(), decoded.cpu().numpy()
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples,4))
    for i in range(num_samples):
        axes[0,i].imshow(orig[i].transpose(1,2,0), cmap='gray' if orig.shape[1]==1 else None)
        axes[0,i].axis('off')
        axes[1,i].imshow(recon[i].transpose(1,2,0), cmap='gray' if recon.shape[1]==1 else None)
        axes[1,i].axis('off')
    axes[0,0].set_title("Orig"); axes[1,0].set_title("Recon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10"], default="MNIST")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--base_ch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_path", type=str, default="recon.png")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, in_ch, H, W = get_loader(args.dataset, args.batch_size)
    model = UNetAutoencoder(in_ch, args.base_ch, args.latent_dim, H, W).to(device)
    train_decoder_only(model, loader, args.epochs, args.lr, device)
    integration_pipeline(model, loader, args.num_samples, device, args.out_path)


if __name__ == "__main__":
    main()