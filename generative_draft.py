import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_func
from tqdm import tqdm

from torchvision.datasets import MNIST, CIFAR10, Flowers102

##############################################
# 0) Dataset loader
##############################################
def get_loader(dataset_name, batch_size, shuffle=True):
    if dataset_name == "MNIST":
        in_ch, H, W = 1, 28, 28
        tf = transforms.ToTensor()
        ds = MNIST("./data", train=True, download=True, transform=tf)

    elif dataset_name == "CIFAR10":
        in_ch, H, W = 3, 32, 32
        tf = transforms.Compose([ transforms.ToTensor() ])
        ds = CIFAR10("./data", train=True, download=True, transform=tf)

    elif dataset_name == "Flowers102":
        in_ch, H, W = 3, 64, 64
        tf = transforms.Compose([ transforms.Resize((H, W)), transforms.ToTensor() ])
        ds = Flowers102("./data", split="train", download=True, transform=tf)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=4
    )
    return loader, in_ch, H, W, ds

##############################################
# 1) HDC helper
##############################################
def generate_hadamard(n: int, device: torch.device) -> torch.Tensor:
    assert n > 0 and (n & (n-1)) == 0, "hdc_dim must be power-of-two"
    H = torch.tensor([[1.]], dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H

##############################################
# 2) U-Net / Residual / Attention blocks
##############################################
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

##############################################
# 3) HDC-augmented decoder training
##############################################
def train_decoder_with_hdc(model, loader, hdc_dim, group_size, epochs, lr, device):
    for p in model.encoders.parameters():      p.requires_grad = False
    for p in model.bottleneck.parameters():     p.requires_grad = False
    for p in model.to_latent.parameters():      p.requires_grad = False

    H = generate_hadamard(hdc_dim, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                z_clean, skips = model.encode(images)
            B, ld = z_clean.shape

            uniq, cnts = torch.unique(labels, return_counts=True)
            valid = uniq[cnts >= group_size]
            if valid.numel() == 0:
                continue
            cls  = valid[torch.randint(len(valid), (1,)).item()]
            idxs = (labels == cls).nonzero(as_tuple=False).view(-1)
            sel  = idxs[torch.randperm(idxs.size(0))[:group_size]]

            grp = z_clean[sel]
            if ld < hdc_dim:
                pad = torch.zeros((group_size, hdc_dim-ld), device=device)
                grp = torch.cat([grp, pad], dim=1)

            idx    = torch.arange(group_size, device=device) % hdc_dim
            keys   = H[idx]
            bundle = (keys * grp).sum(dim=0)
            rec    = bundle.unsqueeze(0) * keys
            z_noisy= rec[:, :ld]

            skips_sel = [s[sel] for s in skips]
            decoded = model.decode(z_noisy, skips_sel)
            loss    = criterion(decoded, images[sel])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}/{epochs}] Loss: {total_loss/len(loader):.4f}")

##############################################
# 4) Integration + SSIM
##############################################
def integration_pipeline(model, loader, hdc_dim, num_samples, device, out_path):
    model.eval()
    H = generate_hadamard(hdc_dim, device)

    images, _ = next(iter(loader))
    images = images.to(device)[:num_samples]

    with torch.no_grad():
        z, skips = model.encode(images)
    B, ld = z.shape
    if ld < hdc_dim:
        pad = torch.zeros((num_samples, hdc_dim-ld), device=device)
        z   = torch.cat([z, pad], dim=1)

    idx    = torch.arange(num_samples, device=device) % hdc_dim
    keys   = H[idx]
    bundle = (keys * z).sum(dim=0)
    rec    = bundle.unsqueeze(0) * keys
    z_rec  = rec[:, :ld]
    skips_s= [s[:num_samples] for s in skips]

    with torch.no_grad():
        decoded = model.decode(z_rec, skips_s)
        score   = ssim_func(images, decoded, data_range=1.0, size_average=True).item()
    print(f"Test SSIM: {score:.4f}")

    orig, recon = images.cpu().numpy(), decoded.cpu().numpy()
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples,4))
    for i in range(num_samples):
        axes[0,i].imshow(orig[i].transpose(1,2,0)); axes[0,i].axis('off')
        axes[1,i].imshow(recon[i].transpose(1,2,0)); axes[1,i].axis('off')
    axes[0,0].set_title("Orig"); axes[1,0].set_title("Recon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return score

##############################################
# 5) Create buffer: 50 decoded images/class
##############################################
def create_buffer(model, dataset_name, hdc_dim, device,
                  per_class=50, output_file= "replay_buffer.pkl"):
    loader, in_ch, H, W, ds = get_loader(dataset_name, batch_size=1, shuffle=False)
    Hmat = generate_hadamard(hdc_dim, device)
    model.eval()

    if hasattr(ds, "classes"):
        num_classes = len(ds.classes)
    else:
        num_classes = int(max(ds.targets)) + 1

    buffer = {c: [] for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for img, label in loader:
            c = int(label.item())
            if counts[c] >= per_class:
                if all(counts[k] >= per_class for k in counts):
                    break
                continue

            x = img.to(device)
            z, skips = model.encode(x)
            if z.size(1) < hdc_dim:
                pad = torch.zeros((1, hdc_dim - z.size(1)), device=device)
                z = torch.cat([z, pad], dim=1)

            key    = Hmat[:1]           
            bundle = (key * z).sum(dim=1, keepdim=True)
            rec_h  = bundle * key
            z_rec  = rec_h[:, :z.size(1)]
            decoded = model.decode(z_rec, skips)

            buffer[c].append(decoded.cpu().squeeze().numpy())
            counts[c] += 1

    with open(output_file, "wb") as f:
        pickle.dump(buffer, f)
    print(f"Saved {per_class} samples/class to {output_file}")

##############################################
# 6) Sweep experiment
##############################################
def run_experiment(dataset_name, args):
    loader, in_ch, H, W, _ = get_loader(dataset_name, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims, scores = [2**k for k in range(3,15)], []

    for dim in tqdm(dims, desc=f"Sweeping {dataset_name}"):
        model = UNetAutoencoder(in_ch, args.base_ch, dim, H, W).to(device)
        train_decoder_with_hdc(model, loader, dim,
                               args.group_size, args.epochs, args.lr, device)
        out_path = f"{dataset_name}_recon_dim_{dim}.png"
        score = integration_pipeline(model, loader, dim,
                                     args.num_samples, device, out_path)
        scores.append(score)

    plt.figure(figsize=(8,5))
    plt.plot(dims, scores, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('latent_dim')
    plt.ylabel('SSIM')
    plt.ylim(0, 1)
    plt.title(f'{dataset_name}: latent_dim vs SSIM')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_ssim_vs_latent_dim.png", dpi=150)
    print(f"→ Saved {dataset_name}_ssim_vs_latent_dim.png")

##############################################
# 7) Main & Argparse
##############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["MNIST","CIFAR10","Flowers102"],
                        default="MNIST")
    parser.add_argument("--mode",
                        choices=["single","sweep"],
                        default="single",
                        help="single: one run; sweep: latent_dim sweep")
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--base_ch",     type=int,   default=8)
    parser.add_argument("--group_size",  type=int,   default=10)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int,   default=10)
    parser.add_argument("--out_path",    type=str,   default="reconstruction.png")
    parser.add_argument("--create_buffer", action="store_true",
                        help="Run HDC pipeline and save 50 decoded images/class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with dataset={args.dataset}")

    if args.mode == "single":
        loader, in_ch, H, W, ds = get_loader(args.dataset, args.batch_size)
        model = UNetAutoencoder(in_ch, args.base_ch,
                                args.latent_dim, H, W).to(device)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        size_mb = sum(p.numel()*p.element_size() for p in model.parameters())/(1024**2)
        print(f"Model size: {size_mb:.2f} MB")

        print("→ Training HDC-augmented decoder …")
        train_decoder_with_hdc(model, loader,
                               args.latent_dim, args.group_size,
                               args.epochs, args.lr, device)

        if args.create_buffer:
            create_buffer(model, args.dataset,
                          args.latent_dim, device,
                          per_class=50,
                          output_file="replay_buffer.pkl")
        else:
            print("→ Running integration pipeline …")
            integration_pipeline(model, loader,
                                 args.latent_dim, args.num_samples,
                                 device, args.out_path)

    else:
        run_experiment(args.dataset, args)

if __name__ == "__main__":
    main()
