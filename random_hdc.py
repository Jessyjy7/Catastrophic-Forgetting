import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_func
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_loader(dataset_name, batch_size):
    if dataset_name == "MNIST":
        in_ch, H, W = 1, 28, 28
        ds = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == "CIFAR10":
        in_ch, H, W = 3, 32, 32
        ds = CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return loader, in_ch, H, W

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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            ResidualBlock(out_c)
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.res  = ResidualBlock(out_c)
        self.att  = AttentionBlock(out_c)
        self.conv = nn.Conv2d(out_c + skip_c, out_c, 3, padding=1)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.res(x)
        x = self.att(x)
        x = torch.cat([x, skip], dim=1)
        return F.relu(self.conv(x))

class UNetAutoencoder(nn.Module):
    def __init__(self, in_ch, base_ch, latent_dim, H, W):
        super().__init__()
        levels = 0; h, w = H, W
        while h >= 8 and w >= 8:
            h//=2; w//=2; levels+=1
        skip_chs = [base_ch*(2**i) for i in range(levels)]
        self.encoders = nn.ModuleList()
        in_c, ch = in_ch, base_ch
        for sc in skip_chs:
            self.encoders.append(EncoderBlock(in_c, sc))
            in_c = sc
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            ResidualBlock(in_c)
        )
        self.flat_h, self.flat_w = h, w
        flat_c = in_c*h*w
        self.to_latent   = nn.Linear(flat_c, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_c)
        self.decoders = nn.ModuleList()
        dec_in = skip_chs[-1]
        for skip_c in reversed(skip_chs):
            out_c = skip_c//2
            self.decoders.append(DecoderBlock(dec_in, skip_c, out_c))
            dec_in = out_c
        self.final_conv = nn.Conv2d(dec_in, in_ch, 1)

    def encode(self, x):
        skips, out = [], x
        for enc in self.encoders:
            skip, out = enc(out); skips.append(skip)
        out = self.bottleneck(out)
        B, C, h, w = out.shape
        flat = out.view(B, -1)
        z = self.to_latent(flat)
        return z, skips

    def decode(self, z, skips):
        B = z.size(0)
        flat = self.from_latent(z)
        out = flat.view(B, -1, self.flat_h, self.flat_w)
        for dec, skip in zip(self.decoders, reversed(skips)):
            out = dec(out, skip)
        return torch.sigmoid(self.final_conv(out))

def train_decoder_with_hdc(m, loader, d, g, e, lr, dev):
    for p in m.encoders.parameters(): p.requires_grad=False
    for p in m.bottleneck.parameters(): p.requires_grad=False
    for p in m.to_latent.parameters(): p.requires_grad=False
    Hm = generate_hadamard(d, dev)
    opt = optim.Adam(
        list(m.from_latent.parameters())+
        list(m.decoders.parameters())+
        list(m.final_conv.parameters()), lr=lr
    )
    for epoch in range(1, e+1):
        tl=0
        for x,y in loader:
            x,y = x.to(dev), y.to(dev)
            with torch.no_grad(): zc,sk = m.encode(x)
            B,ld = zc.shape
            u,c = torch.unique(y, return_counts=True)
            v = u[c>=g]
            if v.numel()==0: continue
            cls = v[torch.randint(len(v),(1,)).item()]
            idx = (y==cls).nonzero().view(-1)
            sel = idx[torch.randperm(idx.size(0))[:g]]
            grp = zc[sel]
            if ld<d: grp=torch.cat([grp,torch.zeros(g,d-ld,device=dev)],1)
            ids = torch.arange(g,device=dev)%d; ks=Hm[ids]
            bnd = (ks*grp).sum(0)
            rec = bnd.unsqueeze(0)*ks; zn = rec[:,:ld]
            sks = [s[sel] for s in sk]
            dec = m.decode(zn, sks)
            loss = nn.MSELoss()(dec, x[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()
        print(f"{epoch}/{e} loss {tl/len(loader):.4f}")

def integration_pipeline(m, loader, d, ns, dev, op):
    m.eval(); Hm=generate_hadamard(d,dev)
    x,_=next(iter(loader)); x=x.to(dev)[:ns]
    with torch.no_grad(): z,sk=m.encode(x)
    B,ld=z.shape
    if ld<d: z=torch.cat([z,torch.zeros(ns,d-ld,device=dev)],1)
    ids=torch.arange(ns,device=dev)%d; ks=Hm[ids]
    bnd=(ks*z).sum(0); rec=bnd.unsqueeze(0)*ks; zr=rec[:,:ld]
    sks=[s[:ns] for s in sk]
    with torch.no_grad(): dec=m.decode(zr,sks)
    sc=ssim_func(x,dec,data_range=1.0,size_average=True).item()
    print(f"SSIM {sc:.4f}")
    o, r = x.cpu().numpy(), dec.cpu().numpy()
    fig,ax=plt.subplots(2,ns,figsize=(2*ns,4))
    for i in range(ns):
        ax[0,i].imshow(o[i].transpose(1,2,0),cmap='gray' if o.shape[1]==1 else None); ax[0,i].axis('off')
        ax[1,i].imshow(r[i].transpose(1,2,0),cmap='gray' if r.shape[1]==1 else None); ax[1,i].axis('off')
    ax[0,0].set_title("Orig"); ax[1,0].set_title("Recon")
    plt.tight_layout(); plt.savefig(op,dpi=150); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["MNIST","CIFAR10"],default="MNIST")
    p.add_argument("--latent_dim",type=int,default=128)
    p.add_argument("--base_ch",type=int,default=8)
    p.add_argument("--group_size",type=int,default=10)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--num_samples",type=int,default=10)
    p.add_argument("--out_path",type=str,default="recon.png")
    a=p.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L,in_ch,H,W=get_loader(a.dataset,a.batch_size)
    m=UNetAutoencoder(in_ch,a.base_ch,a.latent_dim,H,W).to(dev)
    print(sum(pt.numel() for pt in m.parameters()))
    train_decoder_with_hdc(m,L,a.latent_dim,a.group_size,a.epochs,a.lr,dev)
    integration_pipeline(m,L,a.latent_dim,a.num_samples,dev,a.out_path)

if __name__=="__main__":
    main()