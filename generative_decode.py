import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_func
import psutil  # For CPU memory tracking


# -------------------------------
# 1) Define Model Components
# -------------------------------

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

# -------------------------------
# 2) 3-Level U-Net (28->14->7->3)
# -------------------------------

class EncoderDecoderNet(nn.Module):
    """
    A 3-level U-Net–style network for 28×28 MNIST images,
    compressing the spatial dimensions down to 3×3 at the deepest part.
    """
    def __init__(self, input_channels=1, output_channels=1, base_channels=16):
        super(SmallEncoderDecoderNet, self).__init__()
        
        # --- Encoder ---
        self.enc1 = EncoderBlock(input_channels, base_channels)   # 28->14
        # Instead of doubling channels, keep the same:
        self.enc2 = EncoderBlock(base_channels, base_channels)    # 14->7
        self.enc3 = EncoderBlock(base_channels, base_channels)    # 7->3

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)  # optional
        )
        
        # --- Decoder ---
        # 3->7 with a custom upsample to match the 3->7 shape
        custom_upsample_3to7 = nn.ConvTranspose2d(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.dec3 = DecoderBlock(base_channels, base_channels, upsample_layer=custom_upsample_3to7)
        
        self.dec2 = DecoderBlock(base_channels, base_channels)  # 7->14
        self.dec1 = DecoderBlock(base_channels, base_channels)  # 14->28
        
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        
        bn = self.bottleneck(p3)
        
        d3 = self.dec3(bn, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        
        out = self.final_conv(d1)
        return out

# -------------------------------
# 2) Training & Evaluation
# -------------------------------

def train_autoencoder(model, device, epochs=5, batch_size=64, lr=1e-3):
    """
    Train the autoencoder on MNIST using MSE loss.
    """
    # 1. MNIST training data
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Compute reconstruction loss
            loss = criterion(outputs, images)

            # Backprop & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")

def compute_ssim(model, device, batch_size=64):
    """
    Compute the average SSIM on the MNIST test set.
    """
    transform = transforms.ToTensor()
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    ssim_scores = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Ensure values are in [0,1] for SSIM
            outputs = torch.clamp(outputs, 0.0, 1.0)

            # ssim_func returns a single scalar if shape is NxCxHxW
            batch_ssim = ssim_func(images, outputs, data_range=1.0, size_average=True)
            ssim_scores.append(batch_ssim.item())

    return sum(ssim_scores) / len(ssim_scores)

def show_original_decoded(model, device, num_images=8):
    """
    Fetch a batch of MNIST images, run them through the model, 
    and plot original vs decoded side by side.
    """
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=num_images, shuffle=True)

    images, _ = next(iter(test_loader))
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        decoded = model(images)

    # Clamp outputs to [0,1] for display
    decoded = torch.clamp(decoded, 0.0, 1.0)

    images = images.cpu()
    decoded = decoded.cpu()

    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == (num_images // 2):
            axes[0, i].set_title("Original", fontsize=12)

        # Decoded
        axes[1, i].imshow(decoded[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == (num_images // 2):
            axes[1, i].set_title("Decoded", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Save figure to file if you want
    plt.savefig("decoded_comparison.png")
    print("Plot saved as decoded_comparison.png")

# -------------------------------
# Memory Tracking Utilities
# -------------------------------

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
        activation_mem += output.element_size() * output.numel()

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    print(f"Total Activation Memory: {activation_mem / 1024**2:.2f} MB")

# -------------------------------
# 3) Main Script with argparse
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net style autoencoder on MNIST.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--base_channels", type=int, default=32, help="Number of base channels in the model.")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoderNet(
        input_channels=1,
        output_channels=1,
        base_channels=args.base_channels
    ).to(device)

    # --- Memory Tracking Before Training ---
    print("=== Memory Usage Before Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    get_activation_memory(model)

    # 1. Train the model
    print("\nTraining the autoencoder...")
    train_autoencoder(model, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # --- Memory Tracking After Training ---
    print("\n=== Memory Usage After Training ===")
    get_model_memory(model)
    get_cpu_memory_usage()
    get_gpu_memory_usage()
    # Optionally, you can check activation memory again:
    get_activation_memory(model)

    # 2. Compute SSIM on the test set
    print("\nEvaluating SSIM on test set...")
    avg_ssim = compute_ssim(model, device, batch_size=args.batch_size)
    print(f"Average SSIM on MNIST test set: {avg_ssim:.4f}")

    # 3. Show original vs decoded images
    show_original_decoded(model, device, num_images=args.num_images)

