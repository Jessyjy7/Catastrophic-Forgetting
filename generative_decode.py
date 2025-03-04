import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# For SSIM computation
from pytorch_msssim import ssim as ssim_func

# -------------------------------
# 1) Define the Model
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
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # skip connection (x) and pooled output (p)

class DecoderBlock(nn.Module):
    """
    Single decoder block:
    Upsample -> ResidualBlock -> Attention -> Concatenate skip -> Conv.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
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

class EncoderDecoderNet(nn.Module):
    """
    A 2-level U-Net–style network tailored for 28×28 MNIST images.
    """
    def __init__(self, input_channels=1, output_channels=1, base_channels=32):
        super(EncoderDecoderNet, self).__init__()
        
        # ---------- Encoder ----------
        self.enc1 = EncoderBlock(input_channels, base_channels)        # 28 -> 14
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)     # 14 -> 7
        
        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4)
        )
        
        # ---------- Decoder ----------
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2)  # 7 -> 14
        self.dec1 = DecoderBlock(base_channels * 2, base_channels)      # 14 -> 28
        
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        s1, p1 = self.enc1(x)  
        s2, p2 = self.enc2(p1)
        
        # Bottleneck
        bn = self.bottleneck(p2)
        
        # Decoder path
        d2 = self.dec2(bn, s2)
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
            # If your model output is already in [0,1], this is not strictly necessary
            outputs = torch.clamp(outputs, 0.0, 1.0)

            # Compute SSIM for the batch; shape is [N, 1, H, W]
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

    plt.savefig("decoded_comparison.png")


# -------------------------------
# 3) Main Script
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoderNet(input_channels=1, output_channels=1, base_channels=32).to(device)

    # 1. Train the model
    print("Training the autoencoder...")
    train_autoencoder(model, device, epochs=5, batch_size=64, lr=1e-3)

    # 2. Compute SSIM on the test set
    print("Evaluating SSIM on test set...")
    avg_ssim = compute_ssim(model, device, batch_size=64)
    print(f"Average SSIM on MNIST test set: {avg_ssim:.4f}")

    # 3. Show original vs decoded images
    show_original_decoded(model, device, num_images=8)
