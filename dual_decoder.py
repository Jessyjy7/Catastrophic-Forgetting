import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_func

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,  3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

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
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        return F.relu(self.conv(x))

def train_decoder_without_hdc(model, loader, epochs, lr, dev):
    for p in model.encoder.parameters(): p.requires_grad = False
    opt = optim.Adam(
        list(model.decoder.parameters()),
        lr=lr
    )
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

def train_decoder_on_latents(model, latent_loader, epochs, lr, dev):
    for p in model.encoder.parameters(): p.requires_grad = False
    opt = optim.Adam(
        list(model.decoder.parameters()),
        lr=lr
    )
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for z_batch, x_batch in latent_loader:
            z_batch = z_batch.to(dev)
            x_batch = x_batch.to(dev)
            opt.zero_grad()
            dec = model.decode(z_batch)
            loss = criterion(dec, x_batch)
            loss.backward()
            opt.step()

def evaluate_ssim_no_hdc(model, test_loaders, num_classes, num_samples, out_dir, dev):
    model.eval()
    per_class_ssim = {}
    for cls in range(num_classes):
        loader = test_loaders[cls]
        s, i = 0.0, 0
        for x, y in loader:
            x = x.to(dev)
            with torch.no_grad():
                z = model.encode(x)
                dec = model.decode(z)
                s += ssim_func(x, dec, data_range=1.0, size_average=True).item()
                i += 1
        per_class_ssim[cls] = s / (i if i > 0 else 1)
        print(f"Class {cls}: SSIM = {per_class_ssim[cls]:.4f}")
    mean_ssim = sum(per_class_ssim.values()) / len(per_class_ssim)
    print(f"Mean SSIM: {mean_ssim:.4f}")
    return per_class_ssim

def prepare_buffer_for_class(model, loader_c, buffer_size, dev):
    buffer_list = []
    count = 0
    with torch.no_grad():
        for x_batch, _ in loader_c:
            x_batch = x_batch.to(dev)
            z_batch = model.encode(x_batch)
            z_cpu = z_batch.detach().cpu()
            x_cpu = x_batch.detach().cpu()
            for zi, xi in zip(z_cpu, x_cpu):
                buffer_list.append((zi, xi))
                count += 1
                if count >= buffer_size:
                    break
            if count >= buffer_size:
                break
    return buffer_list  

def train_class0(model, loader_c, epochs, lr, buffer_size, dev):
    train_decoder_without_hdc(model, loader_c, epochs, lr, dev)
    buffer0 = prepare_buffer_for_class(model, loader_c, buffer_size, dev)
    return buffer0

def train_incremental_class(model, c, loader_c, replay_buffer, args, dev):
    delta_model = copy.deepcopy(model)
    for p in delta_model.encoder.parameters(): p.requires_grad = False
    latent_list = []
    for old_cls in range(c):
        latent_list.extend(replay_buffer[old_cls])
    new_list = prepare_buffer_for_class(model, loader_c, args.buffer_size, dev)
    latent_list.extend(new_list)
    replay_buffer[c] = new_list
    all_z = torch.stack([pair[0] for pair in latent_list])
    all_x = torch.stack([pair[1] for pair in latent_list])
    latent_ds = TensorDataset(all_z, all_x)
    latent_loader = DataLoader(latent_ds, batch_size=args.batch_size, shuffle=True)
    train_decoder_on_latents(delta_model, latent_loader, args.epochs, args.lr, dev)
    dec_dict = {
        k: v
        for k, v in delta_model.state_dict().items()
        if k.startswith("decoder")
    }
    model.load_state_dict(dec_dict, strict=False)
    return new_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST","CIFAR10", "CIFAR100"], default="CIFAR10")
    parser.add_argument("--latent_dim", type=int, default=256, help="(ignored)")
    parser.add_argument("--base_ch", type=int, default=16, help="(ignored)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--buffer_size", type=int, default=100,
                        help="How many (z,x) pairs to store per class in the replay buffer")
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", dev)
    if args.dataset == "MNIST":
        train_ds = MNIST("./data", train=True,  download=True, transform=transforms.ToTensor())
        test_ds  = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        in_ch, H, W = 1, 28, 28
        raise ValueError("SimpleAutoencoder is built for 3×32×32 (CIFAR). Use CIFAR10 or CIFAR100.")
    elif args.dataset == "CIFAR10":
        train_ds = CIFAR10("./data", train=True,  download=True, transform=transforms.ToTensor())
        test_ds  = CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == "CIFAR100":
        train_ds = CIFAR100("./data", train=True,  download=True, transform=transforms.ToTensor())
        test_ds  = CIFAR100("./data", train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError("Unsupported dataset")
    model = SimpleAutoencoder().to(dev)
    num_params = sum(p.numel() for p in model.parameters())
    size_kb = num_params * 4 / 1024
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Approximate model size: {size_kb:.2f} KB")
    num_classes = len(np.unique(np.array(train_ds.targets)))
    test_loaders = []
    for i in range(num_classes):
        idxs = np.where(np.array(test_ds.targets) == i)[0]
        loader = DataLoader(
            Subset(test_ds, idxs),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        test_loaders.append(loader)
    print("\n--- Pre‐Training Eval (SSIM before any decoder training) ---")
    evaluate_ssim_no_hdc(model, test_loaders, num_classes, args.num_samples, args.out_dir, dev)
    replay_buffer = {}
    for c in range(num_classes):
        print(f"\n=== Training on class {c} ===")
        idxs_c = np.where(np.array(train_ds.targets) == c)[0]
        loader_c = DataLoader(
            Subset(train_ds, idxs_c),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        if c == 0:
            replay_buffer[0] = train_class0(
                model, loader_c,
                epochs      = args.epochs,
                lr          = args.lr,
                buffer_size = args.buffer_size,
                dev         = dev
            )
        else:
            replay_buffer[c] = train_incremental_class(
                model, c, loader_c, replay_buffer, args, dev
            )
        print(f"\n--- Eval after class {c} ---")
        evaluate_ssim_no_hdc(model, test_loaders, num_classes, args.num_samples, args.out_dir, dev)
    print("\n=== Finished all classes! ===")

if __name__ == "__main__":
    main()
