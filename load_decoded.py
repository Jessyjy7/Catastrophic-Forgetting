import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def view_decoded_images(buffer_file="../../decoded_buffer_MNIST.pt", num_samples=10):
    # 1) Load the buffer
    decoded_buffer = torch.load(buffer_file)  
    # e.g., a list of (decoded_img_tensor, label)

    # 2) Build separate lists
    images = []
    labels = []
    for (img, lbl) in decoded_buffer[:num_samples]:
        # 'img' shape = [C,H,W], 'lbl' is int
        images.append(img.unsqueeze(0))  # shape [1,C,H,W]
        labels.append(lbl)

    if len(images) == 0:
        print("Buffer is empty or no samples found.")
        return

    # 3) Combine into a single batch
    images_tensor = torch.cat(images, dim=0)  # shape [N,C,H,W]

    # 4) Visualize
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples,3))
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        ax.axis("off")
        if images_tensor.shape[1] == 3:
            # RGB: shape [3,H,W] => [H,W,3] for imshow
            img_np = images_tensor[i].permute(1,2,0).numpy()
            ax.imshow(img_np)
        else:
            # Grayscale
            img_np = images_tensor[i].squeeze().numpy()
            ax.imshow(img_np, cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
    plt.tight_layout()
    plt.show()

# Example usage:
view_decoded_images("decoded_buffer_MNIST.pt", num_samples=0)
