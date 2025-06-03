#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict

import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from pytorch_msssim import ssim as ssim_func
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_loader(name, batch_size, shuffle=True):
    """
    MNIST → 3×32×32, CIFAR10 → 3×32×32, normalized into [-1,1].
    """
    if name == "MNIST":
        tf = T.Compose([
            T.Resize((32,32)),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        ds = dsets.MNIST("data", train=True, download=True, transform=tf)

    elif name == "CIFAR10":
        tf = T.Compose([
            T.Resize((32,32)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        ds = dsets.CIFAR10("data", train=True, download=True, transform=tf)

    else:
        raise ValueError(f"Unsupported dataset {name}")

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=shuffle, num_workers=0, pin_memory=True)
    return loader, ds


def baseline_recon_ssim(vae, loader, device, max_samples=200):
    """
    Compute SSIM of VAE reconstructions (no HDC) over up to max_samples images.
    """
    vae.eval()
    total_ssim, count = 0.0, 0
    origs, recs = [], []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            enc = vae.encode(x)
            z   = enc.latent_dist.sample()
            dec = vae.decode(z)
            x_rec = ((dec.sample) * 0.5 + 0.5).clamp(0,1)
            x_den = (x * 0.5 + 0.5).clamp(0,1)

            scores = ssim_func(
                x_den, x_rec,
                data_range=1.0,
                size_average=False
            ).cpu()
            total_ssim += scores.sum().item()
            count      += scores.numel()

            if len(origs) < 5:
                origs.append(x_den[0].cpu().permute(1,2,0).numpy())
                recs.append(x_rec[0].cpu().permute(1,2,0).numpy())

            if count >= max_samples:
                break

    avg = total_ssim / count
    print(f"\n📊 Baseline VAE SSIM ({count} samples): {avg:.4f}\n")

    # visualize first few
    if origs:
        fig, ax = plt.subplots(2, len(origs), figsize=(2*len(origs),4))
        for i in range(len(origs)):
            ax[0,i].imshow(origs[i]); ax[0,i].axis("off")
            ax[1,i].imshow(recs[i]); ax[1,i].axis("off")
        ax[0,0].set_title("Orig")
        ax[1,0].set_title("Recon")
        plt.tight_layout()
        plt.show()


def generate_hadamard(n, device):
    """
    Sylvester Hadamard matrix of size n×n (n must be a power of 2).
    """
    assert n>0 and (n&(n-1))==0, "hdc_dim must be power-of-two"
    H = torch.tensor([[1.]], device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def create_buffer_pretrained(
    vae, dataset, hdc_dim, group_size,
    per_class, output_file, device
):
    """
    Build a replay buffer of `per_class` decoded images per class,
    using HDC bundling/unbundling on the VAE latents.
    """
    vae.eval()
    Hmat = generate_hadamard(hdc_dim, device)

    _, ds = get_loader(dataset, batch_size=1, shuffle=False)
    cls2idx = defaultdict(list)
    for idx, lbl in enumerate(ds.targets):
        cls2idx[int(lbl)].append(idx)

    buffer = {c: [] for c in cls2idx}

    for c, idxs in cls2idx.items():
        sel = idxs[:per_class]
        for i in range(0, len(sel), group_size):
            grp = sel[i:i+group_size]
            if len(grp) < group_size:
                break

            # load & batch
            x = torch.cat([ ds[j][0].unsqueeze(0) for j in grp ], dim=0).to(device)

            # encode
            enc = vae.encode(x)
            z   = enc.latent_dist.sample()        # [G, Cz, Hz, Wz]
            G, Cz, Hz, Wz = z.shape
            z_flat = z.view(G, -1)                # [G, hdc_dim]
            assert z_flat.shape[1] == hdc_dim

            # bind & bundle
            keys   = Hmat[:G]                     # [G, hdc_dim]
            bundle = (keys * z_flat).sum(dim=0)   # [hdc_dim]

            # unbind
            recf   = bundle.unsqueeze(0) * keys   # [G, hdc_dim]
            z_rec  = recf.view(G, Cz, Hz, Wz)

            # decode
            dec    = vae.decode(z_rec)
            x_rec  = ((dec.sample) * 0.5 + 0.5).clamp(0,1)

            # detach & store
            for k in range(G):
                if len(buffer[c]) < per_class:
                    buffer[c].append(x_rec[k].detach().cpu().numpy())
                else:
                    break
            if len(buffer[c]) >= per_class:
                break

    with open(output_file, "wb") as f:
        pickle.dump(buffer, f)
    print(f"✅ Saved replay buffer → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["MNIST","CIFAR10"], required=True)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--per_class",  type=int, default=50)
    parser.add_argument("--output",     type=str, default="replay_buffer.pkl")
    parser.add_argument("--device",     choices=["cpu","cuda"], default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"▶ Running on device: {device}")

    print("⏳ Loading pretrained SD‐VAE (AutoencoderKL)…")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", subfolder="vae"
    ).to(device)

    # determine hdc_dim from the model’s latent size
    with torch.no_grad():
        dummy = torch.zeros((1,3,32,32), device=device)
        enc = vae.encode(dummy)
        z   = enc.latent_dist.sample()
        hdc_dim = z.view(1,-1).shape[1]
    print(f"ℹ️  Using hdc_dim = {hdc_dim}")

    # 1) baseline SSIM check
    loader, _ = get_loader(args.dataset, batch_size=32, shuffle=True)
    baseline_recon_ssim(vae, loader, device)

    # 2) create HDC buffer
    create_buffer_pretrained(
        vae,
        args.dataset,
        hdc_dim,
        args.group_size,
        args.per_class,
        args.output,
        device
    )

