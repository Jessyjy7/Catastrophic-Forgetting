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

def get_full_dataset(name):
    if name=="MNIST": return MNIST("./data",train=True,download=True,transform=transforms.ToTensor())
    if name=="CIFAR10": return CIFAR10("./data",train=True,download=True,transform=transforms.ToTensor())
    raise ValueError

def generate_hadamard(n, dev):
    H=torch.tensor([[1.]],device=dev)
    while H.shape[0]<n:
        H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H

class ResidualBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.conv1=nn.Conv2d(c,c,3,padding=1); self.bn1=nn.BatchNorm2d(c); self.conv2=nn.Conv2d(c,c,3,padding=1); self.bn2=nn.BatchNorm2d(c)
    def forward(self,x): r=x; o=F.relu(self.bn1(self.conv1(x))); o=self.bn2(self.conv2(o))+r; return F.relu(o)

class AttentionBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.conv=nn.Conv2d(c,1,1)
    def forward(self,x): return x*torch.sigmoid(self.conv(x))

class EncoderBlock(nn.Module):
    def __init__(self,ic,oc): super().__init__(); self.conv=nn.Sequential(nn.Conv2d(ic,oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(True),ResidualBlock(oc)); self.pool=nn.MaxPool2d(2)
    def forward(self,x): s=self.conv(x); return s,self.pool(s)

class DecoderBlock(nn.Module):
    def __init__(self,ic,sc,oc): super().__init__(); self.up=nn.ConvTranspose2d(ic,oc,2,stride=2); self.res=ResidualBlock(oc); self.att=AttentionBlock(oc); self.conv=nn.Conv2d(oc+sc,oc,3,padding=1)
    def forward(self,x,skip): x=self.up(x); x=self.res(x); x=self.att(x); x=torch.cat([x,skip],1); return F.relu(self.conv(x))

class UNetAutoencoder(nn.Module):
    def __init__(self,inc,bc,ld,H,W):
        super().__init__()
        lvl, h, w = 0, H, W
        while h>=8 and w>=8: h//=2; w//=2; lvl+=1
        self.encoders=nn.ModuleList()
        scs=[bc*(2**i) for i in range(lvl)]
        ic, ch = inc, bc
        for sc in scs: self.encoders.append(EncoderBlock(ic,sc)); ic=sc
        self.bottleneck=nn.Sequential(nn.Conv2d(ic,ic,3,padding=1),nn.BatchNorm2d(ic),nn.ReLU(True),ResidualBlock(ic))
        self.flat_h, self.flat_w, flat_ch = h, w, ic*h*w
        self.to_latent=nn.Linear(flat_ch,ld); self.from_latent=nn.Linear(ld,flat_ch)
        self.decoders=nn.ModuleList(); din=scs[-1]
        for sc in reversed(scs):
            oc=sc//2
            self.decoders.append(DecoderBlock(din,sc,oc))
            din=oc
        self.final_conv=nn.Conv2d(din,inc,1)

    def encode(self,x):
        skips, out = [], x
        for e in self.encoders: s,out=e(out); skips.append(s)
        out=self.bottleneck(out)
        B,C,h,w=out.shape; z=self.to_latent(out.view(B,-1))
        return z, skips

    def decode(self,z,skips):
        B=z.size(0); out=self.from_latent(z).view(B,-1,self.flat_h,self.flat_w)
        for d,s in zip(self.decoders, reversed(skips)): out=d(out,s)
        return torch.sigmoid(self.final_conv(out))

def train_decoder_with_hdc(m,loader,d,g,e,lr,dev):
    for p in m.encoders.parameters():p.requires_grad=False
    for p in m.bottleneck.parameters():p.requires_grad=False
    for p in m.to_latent.parameters():p.requires_grad=False
    Hm=generate_hadamard(d,dev)
    opt=optim.Adam(list(m.from_latent.parameters())+list(m.decoders.parameters())+list(m.final_conv.parameters()),lr=lr)
    for _ in range(e):
        for x,y in loader:
            x,y=x.to(dev),y.to(dev)
            with torch.no_grad(): zc,sk=m.encode(x)
            B,ld=zc.shape; u,c=torch.unique(y,return_counts=True); v=u[c>=g]
            if not v.numel(): continue
            cls=v[torch.randint(len(v),(1,)).item()]; idx=(y==cls).nonzero().view(-1)
            sel=idx[torch.randperm(idx.size(0))[:g]]; grp=zc[sel]
            if ld<d: grp=torch.cat([grp,torch.zeros(g,d-ld,device=dev)],1)
            ids=torch.arange(g,device=dev)%d; ks=Hm[ids]
            b=(ks*grp).sum(0); rec=b.unsqueeze(0)*ks; zn=rec[:,:ld]
            sks=[s[sel] for s in sk]; dec=m.decode(zn,sks)
            loss=nn.MSELoss()(dec,x[sel]); opt.zero_grad(); loss.backward(); opt.step()

def evaluate_ssim(m,name,seen,ns,dev):
    ds=get_full_dataset(name); tot=0
    for c in seen:
        cnt=0; imgs=[]
        for x,y in ds:
            if y==c: imgs.append(x); cnt+=1
            if cnt>=ns: break
        b=torch.stack(imgs).to(dev)
        with torch.no_grad(): z,s=m.encode(b); dc=m.decode(z,s)
        tot+=ssim_func(b,dc,data_range=1.0,size_average=True).item()
    print(f"{seen}: AvgSSIM {tot/len(seen):.4f}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["MNIST","CIFAR10"],default="CIFAR10")
    p.add_argument("--latent_dim",type=int,default=256)
    p.add_argument("--base_ch",type=int,default=16)
    p.add_argument("--group_size",type=int,default=10)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--num_samples",type=int,default=10)
    a=p.parse_args(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=get_full_dataset(a.dataset); labels=torch.tensor(ds.targets).numpy()
    seen=[]; in_ch=ds[0][0].shape[0]; H=ds[0][0].shape[1]; W=ds[0][0].shape[2]
    m=UNetAutoencoder(in_ch,a.base_ch,a.latent_dim,H,W).to(dev)
    for c in range(len(np.unique(labels))):
        seen.append(c)
        idx=np.where(labels==c)[0]
        sub=Subset(ds,idx)
        loader=DataLoader(sub,batch_size=a.batch_size,shuffle=True,pin_memory=True,num_workers=4)
        train_decoder_with_hdc(m,loader,a.latent_dim,a.group_size,a.epochs,a.lr,dev)
        evaluate_ssim(m,a.dataset,seen,a.num_samples,dev)

if __name__=="__main__":
    main()