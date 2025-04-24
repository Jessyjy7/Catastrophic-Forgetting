import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_func
from tqdm import tqdm


##############################################
# helper: build a power-of-2 Hadamard on GPU
##############################################
def generate_hadamard(n: int, device: torch.device) -> torch.Tensor:
    """
    Sylvester’s construction of an n×n Hadamard matrix,
    with entries ±1. Requires n to be a power of 2.
    """
    assert n > 0 and (n & (n-1)) == 0, "hdc_dim must be power-of-two"
    H = torch.tensor([[1.]], dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H  # shape [n,n]

##############################################
# (your U-Net / Residual / Attention blocks)
# ———————— UNCHANGED ————————
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
        skips = []
        out = x
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
        out = flat.view(B, -1, self.flat_h, self.flat_w)
        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)
        return torch.sigmoid(self.final_conv(out))

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips)

##############################################
# 4) HDC-Augmented Decoder Training (all on GPU)
##############################################
def train_decoder_with_hdc(model, loader, hdc_dim, group_size, epochs, lr, device):
    # freeze encoder & to_latent
    for p in model.encoders.parameters():  p.requires_grad = False
    for p in model.bottleneck.parameters(): p.requires_grad = False
    for p in model.to_latent.parameters(): p.requires_grad = False

    # precompute Hadamard keys on GPU
    H = generate_hadamard(hdc_dim, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # get clean latents & skips
            with torch.no_grad():
                z_clean, skips = model.encode(images)
            B, ld = z_clean.shape

            # pick a class with ≥ group_size examples
            uniq, cnts = torch.unique(labels, return_counts=True)
            valid = uniq[cnts >= group_size]
            if valid.numel() == 0:
                continue
            cls  = valid[torch.randint(len(valid), (1,)).item()]
            idxs = (labels == cls).nonzero(as_tuple=False).view(-1)
            sel  = idxs[torch.randperm(idxs.size(0))[:group_size]]

            group = z_clean[sel]  # [G, ld]

            # pad to hdc_dim if needed
            if ld < hdc_dim:
                pad = torch.zeros((group_size, hdc_dim-ld), device=device)
                group = torch.cat([group, pad], dim=1)

            # binding & bundling
            keys   = H[:group_size]           # [G, hdc_dim]
            bound  = keys * group             # [G, hdc_dim]
            bundle = bound.sum(dim=0)         # [hdc_dim]

            # unbinding & recover
            rec    = bundle.unsqueeze(0) * keys   # [G, hdc_dim]
            z_noisy = rec[:, :ld]                 # [G, ld]

            # slice skips
            sliced_skips = [skip[sel] for skip in skips]

            # decode & optimize
            decoded = model.decode(z_noisy, sliced_skips)
            target  = images[sel]
            loss    = criterion(decoded, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        print(f"[Epoch {epoch}/{epochs}] HDC-train loss: {avg:.4f}")

##############################################
# 5) Integration + SSIM + Plot (all on GPU)
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

    keys   = H[:num_samples]          # [S, hdc_dim]
    bound  = keys * z                 # [S, hdc_dim]
    bundle = bound.sum(dim=0)         # [hdc_dim]

    rec    = bundle.unsqueeze(0) * keys  # [S, hdc_dim]
    z_rec  = rec[:, :ld]                 # [S, ld]

    sliced_skips = [skip[:num_samples] for skip in skips]
    with torch.no_grad():
        decoded = model.decode(z_rec, sliced_skips)
        score   = ssim_func(images, decoded, data_range=1.0, size_average=True)
    print(f"Test SSIM: {score.item():.4f}")

    # save comparison plot
    orig  = images.cpu().numpy()
    recon = decoded.cpu().numpy()
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    for i in range(num_samples):
        axes[0,i].imshow(orig[i].squeeze(), cmap='gray');   axes[0,i].axis('off')
        axes[1,i].imshow(recon[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
    axes[0,0].set_title("Original"); axes[1,0].set_title("Decoded")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

##############################################
# 6) Main & Argparse
##############################################
def main():
    parser = argparse.ArgumentParser(description="HDC-augmented U-Net Autoencoder")
    parser.add_argument("--latent_dim",  type=int, default=128,
                        help="latent vector size (power-of-two!)")
    parser.add_argument("--base_ch",     type=int, default=8)
    parser.add_argument("--group_size",  type=int, default=10)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_path",    type=str, default="reconstruction.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Running on", device)

    # data
    transform = transforms.ToTensor()
    ds    = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader= torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                        pin_memory=True, num_workers=4)

    # model
    model = UNetAutoencoder(1, args.base_ch, args.latent_dim, 28, 28).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # train HDC decoder
    print("→ Starting HDC-augmented decoder training …")
    train_decoder_with_hdc(model, loader,
                           hdc_dim    = args.latent_dim,
                           group_size = args.group_size,
                           epochs     = args.epochs,
                           lr         = args.lr,
                           device     = device)

    # run integration
    print("→ Running integration pipeline …")
    integration_pipeline(model, loader,
                         hdc_dim     = args.latent_dim,
                         num_samples = args.num_samples,
                         device      = device,
                         out_path    = args.out_path)
    
def run_experiment():
    # fixed params
    base_ch    = 8
    group_size = 10
    epochs     = 10
    batch_size = 64
    lr         = 1e-3
    num_samples= 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("→ Running on", device)

    # data
    ds     = torchvision.datasets.MNIST("./data", train=True, download=True,
                                        transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=True, pin_memory=True,
                                         num_workers=4)

    dims, scores = [2**k for k in range(4,15)], []
    for dim in tqdm(dims, desc="Sweeping latent_dim"):
        # build model
        model = UNetAutoencoder(1, base_ch, dim, 28, 28).to(device)

        # train decoder w/ HDC
        print(f"\n→ Training (latent_dim={dim})")
        train_decoder_with_hdc(model, loader, dim, group_size, epochs, lr, device)

        # test & save reconstructions
        print(f"→ Testing (latent_dim={dim})")
        H = generate_hadamard(dim, device)
        images, _ = next(iter(loader))
        images = images.to(device)[:num_samples]

        with torch.no_grad():
            z, skips = model.encode(images)
        B, ld = z.shape
        if ld < dim:
            pad = torch.zeros((num_samples, dim-ld), device=device)
            z   = torch.cat([z, pad], dim=1)

        keys   = H[:num_samples]
        bound  = keys * z
        bundle = bound.sum(dim=0)
        rec    = bundle.unsqueeze(0) * keys
        z_rec  = rec[:, :ld]
        skips_s= [s[:num_samples] for s in skips]

        with torch.no_grad():
            decoded = model.decode(z_rec, skips_s)
            score   = ssim_func(images, decoded, data_range=1.0, size_average=True).item()
        print(f"  SSIM @ {dim}: {score:.4f}")
        scores.append(score)

        # save this reconstruction
        orig, recon = images.cpu().numpy(), decoded.cpu().numpy()
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples,4))
        for i in range(num_samples):
            axes[0,i].imshow(orig[i].squeeze(), cmap='gray'); axes[0,i].axis('off')
            axes[1,i].imshow(recon[i].squeeze(), cmap='gray'); axes[1,i].axis('off')
        axes[0,0].set_title("Orig"); axes[1,0].set_title("Recon")
        plt.tight_layout()
        plt.savefig(f"reconstruction_dim_{dim}.png", dpi=150)
        plt.close(fig)

    # final plot
    plt.figure(figsize=(8,5))
    plt.plot(dims, scores, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('latent_dim (power of 2)')
    plt.ylabel('SSIM')
    plt.title('latent_dim vs SSIM')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("ssim_vs_latent_dim.png", dpi=150)
    print("→ Saved ssim_vs_latent_dim.png")

if __name__ == "__main__":
    # main()
    run_experiment()
