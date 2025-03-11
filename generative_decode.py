import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_func
import psutil

# If Flowers102 is not in your torchvision, you may need a newer version:
# from torchvision.datasets import Flowers102
from torchvision.datasets import MNIST, CIFAR10

###############################################################################
# 1) Data Loaders for MNIST, CIFAR-10, Flowers102
###############################################################################
def get_dataloaders(dataset_name, batch_size):
    dataset_name = dataset_name.upper()
    if dataset_name == "MNIST":
        transform = transforms.ToTensor()
        train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset  = MNIST(root="./data", train=False, download=True, transform=transform)

        in_channels = 1
        height, width = 28, 28

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Optionally normalize: transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset  = CIFAR10(root="./data", train=False, download=True, transform=transform)

        in_channels = 3
        height, width = 32, 32

    elif dataset_name == "FLOWERS102":
        # Make sure your torchvision supports Flowers102 (>=0.13).
        from torchvision.datasets import Flowers102
        transform = transforms.Compose([
            transforms.Resize((64,64)),  # resizing all images to 64x64
            transforms.ToTensor()
        ])
        train_data = Flowers102(root="./data", split='train', download=True, transform=transform)
        val_data   = Flowers102(root="./data", split='val', download=True, transform=transform)
        test_data  = Flowers102(root="./data", split='test', download=True, transform=transform)

        # Combine train+val
        import torch.utils.data as data
        train_dataset = data.ConcatDataset([train_data, val_data])
        test_dataset  = test_data

        in_channels = 3
        height, width = 64, 64

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_channels, (height, width)

###############################################################################
# 2) Utility to Determine Number of Downsampling Levels
###############################################################################
def calc_num_levels(height, width, min_size=4):
    """
    Counts how many times we can apply a 2x2 pooling before (H,W) < min_size.
    For example, if H=W=28, we can do:
      28->14->7->3
    which is 3 levels if min_size=4 (since 3 < 4).
    If H=W=32, we get:
      32->16->8->4
    which is 3 levels if min_size=4 (4 is == min_size, so we stop).
    If H=W=64, we get 64->32->16->8->4->2 (5 levels if min_size=2, or 4 levels if min_size=4).
    """
    levels = 0
    while height >= min_size*2 and width >= min_size*2:
        height //= 2
        width //= 2
        levels += 1
    return levels

###############################################################################
# 3) Model Components (ResidualBlock, AttentionBlock, etc.)
###############################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # skip connection, pooled output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_layer=None):
        super(DecoderBlock, self).__init__()
        if upsample_layer is None:
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.upsample = upsample_layer

        self.res_block = ResidualBlock(out_channels)
        self.attn = AttentionBlock(out_channels)
        self.conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.res_block(x)
        x = self.attn(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv(x))
        return x

###############################################################################
# 4) Dynamic U-Net: Builds 'n' levels based on calc_num_levels
###############################################################################
class DynamicUNet(nn.Module):
    """
    Automatically builds a U-Net with 'n' downsampling levels based on input size.
    Each level keeps the same number of channels or can optionally double them.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, num_levels=2):
        super(DynamicUNet, self).__init__()
        self.num_levels = num_levels

        # Build encoders
        self.encoders = nn.ModuleList()
        c_in = in_channels
        c_out = base_channels
        for lvl in range(num_levels):
            enc_block = EncoderBlock(c_in, c_out)
            self.encoders.append(enc_block)
            # Optionally keep c_out the same or double it each time:
            # c_in = c_out
            # c_out *= 2
            # For simplicity, let's keep them the same:
            c_in = c_out

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            ResidualBlock(c_in)
        )

        # Build decoders (reverse order)
        self.decoders = nn.ModuleList()
        for lvl in range(num_levels):
            dec_block = DecoderBlock(c_in, c_in)  # same channels in/out
            self.decoders.append(dec_block)

        self.final_conv = nn.Conv2d(c_in, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skips = []
        out = x
        for enc in self.encoders:
            s, out = enc(out)
            skips.append(s)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder path (reverse the skip list)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i+1)]  # from the end
            out = dec(out, skip)

        out = self.final_conv(out)
        return out

###############################################################################
# 5) Build a model for each dataset
###############################################################################
def build_model_for_dataset(dataset_name, in_channels, height, width, base_channels=16):
    """
    1) Compute how many levels we can do with 2x2 pooling.
    2) Build a DynamicUNet with that many levels.
    3) Return the model.
    """
    # Decide a min_size. E.g., if we don't want to go below 4x4
    min_size = 4
    num_levels = calc_num_levels(height, width, min_size=min_size)
    print(f"Dataset {dataset_name}: image size {height}x{width}, -> building {num_levels} levels U-Net")

    model = DynamicUNet(
        in_channels=in_channels,
        out_channels=in_channels,  # reconstruct same # of channels
        base_channels=base_channels,
        num_levels=num_levels
    )
    return model

###############################################################################
# 6) Training, SSIM, Visualization
###############################################################################
def train_autoencoder(model, device, train_loader, epochs=5, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    print("Training complete!")

def compute_ssim(model, device, test_loader):
    model.eval()
    ssim_scores = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            batch_ssim = ssim_func(images, outputs, data_range=1.0, size_average=True)
            ssim_scores.append(batch_ssim.item())
    return sum(ssim_scores) / len(ssim_scores)

def show_original_decoded(model, device, test_loader, num_images=8, in_channels=1):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        decoded = model(images)
    decoded = torch.clamp(decoded, 0.0, 1.0)

    images = images.cpu()
    decoded = decoded.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
    for i in range(num_images):
        if in_channels == 3:
            # (C,H,W) -> (H,W,C)
            axes[0, i].imshow(images[i].permute(1,2,0).numpy())
            axes[1, i].imshow(decoded[i].permute(1,2,0).numpy())
        else:
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            axes[1, i].imshow(decoded[i].squeeze(), cmap='gray')

        axes[0, i].axis('off')
        axes[1, i].axis('off')
        if i == (num_images // 2):
            axes[0, i].set_title("Original", fontsize=12)
            axes[1, i].set_title("Decoded", fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.savefig("decoded_comparison.png")
    print("Plot saved as decoded_comparison.png")

###############################################################################
# 7) Memory Tracking (Optional)
###############################################################################
def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")
    print(f"Total Model Memory Usage: {total_memory / 1024**2:.2f} MB")

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Current Allocated GPU Memory: {allocated:.2f} MB")
        print(f"Max Allocated GPU Memory: {max_allocated:.2f} MB")
    else:
        print("CUDA is not available.")

def get_cpu_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current Process Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def get_activation_memory(model, in_channels=1, height=28, width=28):
    input_tensor = torch.randn((1, in_channels, height, width)).to(next(model.parameters()).device)
    activation_mem = 0
    def hook_fn(module, inp, output):
        nonlocal activation_mem
        if isinstance(output, torch.Tensor):
            activation_mem += output.element_size() * output.numel()
        elif isinstance(output, (list, tuple)):
            for out in output:
                if isinstance(out, torch.Tensor):
                    activation_mem += out.element_size() * out.numel()

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    print(f"Total Activation Memory: {activation_mem / 1024**2:.2f} MB")

###############################################################################
# 8) Main Script
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net style autoencoder on MNIST, CIFAR10, or FLOWERS102.")
    parser.add_argument("--dataset", type=str, default="MNIST", help="MNIST, CIFAR10, FLOWERS102")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--base_channels", type=int, default=16, help="Number of base channels in the model.")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Get data
    train_loader, test_loader, in_channels, (height, width) = get_dataloaders(args.dataset, args.batch_size)

    # 2) Build model dynamically
    model = build_model_for_dataset(
        args.dataset,
        in_channels,
        height,
        width,
        base_channels=args.base_channels
    ).to(device)

    # 3) Memory usage before training
    print(f"=== {args.dataset} dataset: input {height}x{width}, in_channels={in_channels} ===")
    print("=== Memory Usage Before Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    get_activation_memory(model, in_channels=in_channels, height=height, width=width)

    # 4) Train
    print("\nTraining the autoencoder...")
    train_autoencoder(model, device, train_loader, epochs=args.epochs, lr=args.lr)

    # 5) Memory usage after training
    print("\n=== Memory Usage After Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    get_activation_memory(model, in_channels=in_channels, height=height, width=width)

    # 6) Evaluate SSIM
    print("\nEvaluating SSIM on test set...")
    avg_ssim = compute_ssim(model, device, test_loader)
    print(f"Average SSIM on {args.dataset} test set: {avg_ssim:.4f}")

    # 7) Visualize
    show_original_decoded(model, device, test_loader, num_images=args.num_images, in_channels=in_channels)
