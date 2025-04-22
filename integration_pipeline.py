# hdc_unet_autoencoder.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from hadamardHD import binding, unbinding
from pytorch_msssim import ssim as ssim_func

##############################################
# 1) Building Blocks: Residual + Attention
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return torch.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
    def forward(self, x):
        w = torch.sigmoid(self.conv(x))
        return x * w

##############################################
# 2) Encoder & Decoder Blocks for U‑Net
##############################################
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
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResidualBlock(out_ch)
        self.att = AttentionBlock(out_ch)
        self.conv = nn.Conv2d(out_ch*2, out_ch, 3, padding=1)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        x = torch.cat([x, skip], dim=1)
        return torch.relu(self.conv(x))

##############################################
# 3) Full U‑Net Autoencoder with Latent Vector
##############################################
class UNetAutoencoder(nn.Module):
    def __init__(self, in_ch, base_ch, latent_dim, height, width):
        super().__init__()
        # compute downsampling levels
        levels = 0
        h, w = height, width
        while h >= 8 and w >= 8:
            h //= 2; w //= 2; levels += 1
        self.levels = levels

        # encoders
        self.encoders = nn.ModuleList()
        ch = base_ch
        in_c = in_ch
        for _ in range(levels):
            self.encoders.append(EncoderBlock(in_c, ch))
            in_c = ch; ch *= 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            ResidualBlock(in_c)
        )

        # flatten dims for latent
        self.flat_h, self.flat_w = h, w
        flat_ch = in_c * h * w
        self.to_latent   = nn.Linear(flat_ch, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_ch)

        # decoders
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
# 4) HDC‑Augmented Decoder Training
##############################################
def train_decoder_with_hdc(model, loader, hdc_dim, group_size, epochs, lr, device):
    # freeze encoder + bottleneck + to_latent
    for p in model.encoders.parameters():      p.requires_grad = False
    for p in model.bottleneck.parameters():     p.requires_grad = False
    for p in model.to_latent.parameters():      p.requires_grad = False

    criterion = nn.MSELoss()
    opt = optim.Adam(
        list(model.from_latent.parameters()) +
        list(model.decoders.parameters()) +
        list(model.final_conv.parameters()),
        lr=lr
    )

    for ep in range(1, epochs+1):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs  = imgs.to(device)
            labels= labels.to(device)

            # encode once
            with torch.no_grad():
                z_clean, skips = model.encode(imgs)
            z_np  = z_clean.cpu().numpy()
            labs  = labels.cpu().numpy()
            ld    = z_np.shape[1]

            # pick a class with enough samples
            u, cts = np.unique(labs, return_counts=True)
            valid  = u[cts >= group_size]
            if len(valid)==0: continue
            cls = np.random.choice(valid)
            idxs = np.where(labs==cls)[0]
            sel  = np.random.choice(idxs, group_size, False)
            group= z_np[sel]

            # bind & bundle
            bound = []
            for i, v in enumerate(group):
                vec = v if ld>=hdc_dim else np.pad(v, (0,hdc_dim-ld), 'constant')
                bound.append(binding(hdc_dim, i, vec))
            bundle = np.sum(bound, axis=0)

            # unbind
            recs = []
            for i in range(group_size):
                uvec = unbinding(hdc_dim, i, bundle)
                recs.append(uvec[:ld])
            z_noisy = torch.tensor(np.stack(recs), dtype=torch.float32, device=device)

            # slice skips to match sel
            new_skips = [skip_tensor[sel] for skip_tensor in skips]

            # decode & loss
            dec = model.decode(z_noisy, new_skips)
            tgt = imgs[sel]
            loss = criterion(dec, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"[Epoch {ep}/{epochs}] HDC‑train loss: {avg:.4f}")

##############################################
# 5) Integration + SSIM + Save
##############################################
def integration_pipeline(model, loader, hdc_dim, num_samples, device, out_path):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)[:num_samples]

    with torch.no_grad():
        z, skips = model.encode(imgs)
    z_np = z.cpu().numpy()
    ld   = z_np.shape[1]

    # bind & bundle all num_samples
    bound = []
    for i in range(num_samples):
        v = z_np[i]
        vec = v if ld>=hdc_dim else np.pad(v, (0,hdc_dim-ld), 'constant')
        bound.append(binding(hdc_dim, i, vec))
    bundle = np.sum(bound, axis=0)

    # unbind
    recs = []
    for i in range(num_samples):
        uvec = unbinding(hdc_dim, i, bundle)
        recs.append(uvec[:ld])
    z_rec = torch.tensor(np.stack(recs), dtype=torch.float32, device=device)

    # slice skips for test set
    new_skips = [skip_tensor[:num_samples] for skip_tensor in skips]

    with torch.no_grad():
        dec = model.decode(z_rec, new_skips)
        s   = ssim_func(imgs, dec, data_range=1.0, size_average=True)
    print(f"Test SSIM: {s.item():.4f}")

    # save plot
    o = imgs.cpu().numpy()
    d = dec.cpu().numpy()
    fig, ax = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    for i in range(num_samples):
        ax[0,i].imshow(o[i].squeeze(), cmap='gray'); ax[0,i].axis('off')
        ax[1,i].imshow(d[i].squeeze(), cmap='gray'); ax[1,i].axis('off')
    ax[0,0].set_title("Original"); ax[1,0].set_title("Decoded")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

##############################################
# 6) Main & Args
##############################################
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dim",  type=int, default=2048,
                   help="latent and HDC dimension (must match)")
    p.add_argument("--base_ch",     type=int, default=16,
                   help="base number of channels in U‑Net")
    p.add_argument("--group_size",  type=int, default=10,
                   help="how many latents to bundle each HDC training step")
    p.add_argument("--epochs",      type=int, default=5,
                   help="HDC training epochs")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="MNIST dataloader batch size")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="learning rate for HDC decoder training")
    p.add_argument("--num_samples", type=int, default=10,
                   help="samples in final integration test")
    p.add_argument("--out_path",    type=str,   default="reconstruction.png",
                   help="where to save the final comparison plot")
    args = p.parse_args()

    assert args.latent_dim > 0, "latent_dim must be positive"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # Data
    transform = transforms.ToTensor()
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Model
    latent_dim = args.latent_dim
    model = UNetAutoencoder(1, args.base_ch, latent_dim, height=28, width=28).to(device)

    # HDC‑augmented decoder training
    print("→ Starting HDC‑augmented decoder training …")
    train_decoder_with_hdc(
        model, loader,
        hdc_dim    = latent_dim,
        group_size = args.group_size,
        epochs     = args.epochs,
        lr         = args.lr,
        device     = device
    )

    # Integration + SSIM + save
    print("→ Running integration pipeline …")
    integration_pipeline(
        model, loader,
        hdc_dim     = latent_dim,
        num_samples = args.num_samples,
        device      = device,
        out_path    = args.out_path
    )

if __name__ == "__main__":
    main()
