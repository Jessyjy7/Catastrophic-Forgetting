import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_func
import psutil  # For CPU memory tracking

# If Flowers102 is not included in your torchvision version, you may need an updated PyTorch.
# from torchvision.datasets import Flowers102
from torchvision.datasets import MNIST, CIFAR10

###############################################################################
# 1) Dataset Loader: MNIST, CIFAR-10, Flowers102
###############################################################################

def get_dataloaders(dataset_name, batch_size):
    """
    Returns (train_loader, test_loader, input_channels, is_color, image_size).
    - is_color: bool, whether images are RGB (3-channel).
    - image_size: tuple (H, W) for the dataset.
    """

    dataset_name = dataset_name.upper()

    if dataset_name == "MNIST":
        transform = transforms.ToTensor()
        train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset  = MNIST(root="./data", train=False, download=True, transform=transform)

        input_channels = 1
        is_color = False
        image_size = (28, 28)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Optionally normalize: transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset  = CIFAR10(root="./data", train=False, download=True, transform=transform)

        input_channels = 3
        is_color = True
        image_size = (32, 32)

    elif dataset_name == "FLOWERS102":
        # Make sure your torchvision supports Flowers102 (added in torchvision>=0.13).
        # We'll resize to 64x64 for a manageable input size.
        from torchvision.datasets import Flowers102
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            # Optionally normalize or augment further
        ])
        # Flowers102 splits: 'train', 'val', 'test'
        # We'll combine 'train' + 'val' for training, 'test' for testing
        train_dataset = Flowers102(root="./data", split='train', download=True, transform=transform)
        val_dataset   = Flowers102(root="./data", split='val', download=True, transform=transform)
        test_dataset  = Flowers102(root="./data", split='test', download=True, transform=transform)

        # Combine train+val into one training set
        import torch.utils.data as data
        train_dataset = data.ConcatDataset([train_dataset, val_dataset])

        input_channels = 3
        is_color = True
        image_size = (64, 64)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from MNIST, CIFAR10, FLOWERS102.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, input_channels, is_color, image_size

###############################################################################
# 2) Define Model Components
###############################################################################

class ResidualBlock(nn.Module):
    """Simple residual block with two convolutional layers."""
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
    """Attention mechanism to emphasize important features."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class EncoderBlock(nn.Module):
    """
    Single encoder block: 
    (Conv + ResidualBlock) -> MaxPool(2x2).
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )
        self.pool = nn.MaxPool2d(2)  # halves the spatial dimension

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # 'x' for skip connection, 'p' for pooled output

class DecoderBlock(nn.Module):
    """
    Single decoder block:
    Upsample -> ResidualBlock -> Attention -> Concatenate skip -> Conv.

    Allows a custom 'upsample_layer' if you need something other than the default
    2x2 stride=2 transposed convolution.
    """
    def __init__(self, in_channels, out_channels, upsample_layer=None):
        super(DecoderBlock, self).__init__()
        # If no custom upsample layer is given, default to a 2x2 stride=2 transpose conv
        if upsample_layer is None:
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.upsample = upsample_layer

        self.res_block = ResidualBlock(out_channels)
        self.attn = AttentionBlock(out_channels)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.res_block(x)
        x = self.attn(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv(x))
        return x

###############################################################################
# 3) 3-Level U-Net (example for 28->14->7->3) but can adapt to other sizes
###############################################################################

class EncoderDecoderNet(nn.Module):
    """
    A 3-level U-Net–style network. By default, it's configured for a small input 
    (like MNIST 28x28), but it can handle larger images too; it just won't necessarily 
    compress all the way to 3x3 if the input is bigger.
    
    For CIFAR10 (32x32), you'll get 32->16->8->4 if you add an extra level, etc.
    For Flowers102 resized to 64x64, you might want more levels or bigger 'base_channels'.
    """
    def __init__(self, input_channels=1, output_channels=1, base_channels=16):
        super(EncoderDecoderNet, self).__init__()
        
        # --- Encoder ---
        self.enc1 = EncoderBlock(input_channels, base_channels)   
        self.enc2 = EncoderBlock(base_channels, base_channels)    
        self.enc3 = EncoderBlock(base_channels, base_channels)    

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        
        # --- Decoder ---
        custom_upsample_3to7 = nn.ConvTranspose2d(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.dec3 = DecoderBlock(base_channels, base_channels, upsample_layer=custom_upsample_3to7)
        self.dec2 = DecoderBlock(base_channels, base_channels)
        self.dec1 = DecoderBlock(base_channels, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        
        # Bottleneck
        bn = self.bottleneck(p3)
        
        # Decode
        d3 = self.dec3(bn, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        
        out = self.final_conv(d1)
        return out

###############################################################################
# 4) Training, Evaluation, and Visualization
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
            # Ensure values are in [0,1] for SSIM
            outputs = torch.clamp(outputs, 0.0, 1.0)

            # If using pytorch_msssim, it handles NxCxHxW (including 3 channels)
            batch_ssim = ssim_func(images, outputs, data_range=1.0, size_average=True)
            ssim_scores.append(batch_ssim.item())

    return sum(ssim_scores) / len(ssim_scores)

def show_original_decoded(model, device, test_loader, num_images=8, is_color=False):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)  # just in case batch>num_images
    with torch.no_grad():
        decoded = model(images)
    decoded = torch.clamp(decoded, 0.0, 1.0)

    images = images.cpu()
    decoded = decoded.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
    for i in range(num_images):
        if is_color:
            # Convert (C,H,W) -> (H,W,C) for plt.imshow
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
# 5) Memory Tracking Utilities
###############################################################################

def get_model_memory(model):
    """Calculate memory used by model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")
    print(f"Total Model Memory Usage: {total_memory / 1024**2:.2f} MB")

def get_gpu_memory_usage():
    """Display current and maximum GPU memory usage (if CUDA is available)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Current Allocated GPU Memory: {allocated:.2f} MB")
        print(f"Max Allocated GPU Memory: {max_allocated:.2f} MB")
    else:
        print("CUDA is not available.")

def get_cpu_memory_usage():
    """Display the current process CPU memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current Process Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def get_activation_memory(model, input_shape=(1, 1, 28, 28)):
    """
    Estimate memory used by activations during a forward pass.
    Note: This only tracks layers that are instances of common modules.
    """
    input_tensor = torch.randn(input_shape).to(next(model.parameters()).device)
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
# 6) Main Script with argparse
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net style autoencoder on MNIST, CIFAR10, or FLOWERS102.")
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        help="Which dataset to use: MNIST, CIFAR10, FLOWERS102")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--base_channels", type=int, default=16, help="Number of base channels in the model.")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load data for chosen dataset
    train_loader, test_loader, in_channels, is_color, image_size = get_dataloaders(
        args.dataset, args.batch_size
    )

    # 2) Build model for the correct # of channels
    model = EncoderDecoderNet(
        input_channels=in_channels,
        output_channels=in_channels,  # reconstruct the same # of channels
        base_channels=args.base_channels
    ).to(device)

    # 3) Memory tracking before training
    print(f"=== Using {args.dataset} dataset ===")
    print("=== Memory Usage Before Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    # For activation memory, adjust input_shape to match dataset if desired
    # e.g., if is_color and image_size=(32,32), input_shape=(1,3,32,32)
    test_input_shape = (1, in_channels, image_size[0], image_size[1])
    get_activation_memory(model, input_shape=test_input_shape)

    # 4) Train the model
    print("\nTraining the autoencoder...")
    train_autoencoder(model, device, train_loader, epochs=args.epochs, lr=args.lr)

    # 5) Memory tracking after training
    print("\n=== Memory Usage After Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    get_activation_memory(model, input_shape=test_input_shape)

    # 6) Evaluate SSIM
    print("\nEvaluating SSIM on test set...")
    avg_ssim = compute_ssim(model, device, test_loader)
    print(f"Average SSIM on {args.dataset} test set: {avg_ssim:.4f}")

    # 7) Show original vs decoded images
    show_original_decoded(model, device, test_loader, num_images=args.num_images, is_color=is_color)
