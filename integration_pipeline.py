# hdc_autoencoder_cuda.py

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
# 1. Simple Autoencoder (same as before)
##############################################
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 32, 7, 7)
        return self.decoder_conv(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))

##############################################
# 2. HDC‑Augmented Decoder Training
##############################################
def train_decoder_with_hdc(model, dataloader, hdc_dim, group_size, epochs, lr, device):
    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.decoder_fc.parameters()) + list(model.decoder_conv.parameters()),
        lr=lr
    )
    
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # encode & detach
            with torch.no_grad():
                z_clean = model.encode(images)  # [B, latent_dim]
            z_np = z_clean.cpu().numpy()
            labels_np = labels.cpu().numpy()
            latent_dim = z_np.shape[1]
            
            # pick a class with enough samples
            unique, counts = np.unique(labels_np, return_counts=True)
            valid = unique[counts >= group_size]
            if len(valid) == 0:
                continue
            cls = np.random.choice(valid)
            
            idxs = np.where(labels_np == cls)[0]
            sel = np.random.choice(idxs, group_size, replace=False)
            selected = z_np[sel]
            
            # bind & bundle
            bound = []
            for i, vec in enumerate(selected):
                v = vec
                if latent_dim < hdc_dim:
                    v = np.pad(v, (0, hdc_dim - latent_dim), 'constant')
                else:
                    v = v[:hdc_dim]
                bound.append(binding(hdc_dim, i, v))
            bundle = np.sum(bound, axis=0)
            
            # unbind
            recs = []
            for i in range(group_size):
                u = unbinding(hdc_dim, i, bundle)
                recs.append(u[:latent_dim])
            z_noisy = torch.tensor(np.stack(recs), dtype=torch.float32, device=device)
            
            # decode & loss
            outputs = model.decode(z_noisy)
            target = images[sel]
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch}/{epochs}] HDC‑train loss: {avg:.4f}")

##############################################
# 3. Integration + SSIM + Save Plot
##############################################
def integration_pipeline(model, dataloader, hdc_dim, num_samples, device, out_path):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)[:num_samples]
    
    # encode
    with torch.no_grad():
        z = model.encode(images).cpu().numpy()
    latent_dim = z.shape[1]
    
    # bind & bundle
    bound = []
    for i in range(num_samples):
        v = z[i]
        if latent_dim < hdc_dim:
            v = np.pad(v, (0, hdc_dim - latent_dim), 'constant')
        hv = binding(hdc_dim, i, v)
        bound.append(hv)
    bundle = np.sum(bound, axis=0)
    
    # unbind
    recs = []
    for i in range(num_samples):
        u = unbinding(hdc_dim, i, bundle)
        recs.append(u[:latent_dim])
    z_rec = torch.tensor(np.stack(recs), dtype=torch.float32, device=device)
    
    # decode + SSIM
    with torch.no_grad():
        dec = model.decode(z_rec)
        s = ssim_func(images, dec, data_range=1.0, size_average=True)
    print(f"Test SSIM: {s.item():.4f}")
    
    # save comparison plot
    imgs = images.cpu().numpy()
    decs = dec.cpu().numpy()
    fig, axs = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    for i in range(num_samples):
        axs[0,i].imshow(imgs[i].squeeze(), cmap='gray'); axs[0,i].axis('off')
        axs[1,i].imshow(decs[i].squeeze(), cmap='gray'); axs[1,i].axis('off')
    axs[0,0].set_title("Original")
    axs[1,0].set_title("Decoded")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved reconstruction plot to {out_path}")

##############################################
# 4. Main & Argparse
##############################################
def main():
    p = argparse.ArgumentParser(description="HDC‑Autoencoder on MNIST (CUDA enabled)")
    p.add_argument("--latent_dim",  type=int,   default=2048)
    p.add_argument("--hdc_dim",     type=int,   default=2048)
    p.add_argument("--group_size",  type=int,   default=10)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_samples", type=int,   default=10)
    p.add_argument("--out_path",    type=str,   default="reconstruction.png",
                   help="where to save the final comparison plot")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data loader
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size, shuffle=True)

    # Model
    model = SimpleAutoencoder(latent_dim=args.latent_dim).to(device)

    # 1) Train decoder with HDC noise
    print("→ Training decoder with HDC noise…")
    train_decoder_with_hdc(
        model, loader,
        hdc_dim=args.hdc_dim,
        group_size=args.group_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    # 2) Integration + SSIM + save plot
    print("→ Running integration pipeline…")
    integration_pipeline(
        model, loader,
        hdc_dim=args.hdc_dim,
        num_samples=args.num_samples,
        device=device,
        out_path=args.out_path
    )

if __name__ == "__main__":
    main()
