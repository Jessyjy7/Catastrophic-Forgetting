from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as tvt

def get_digit_loader(digit=None, batch_size=64, train=True):
    # Load the MNIST dataset WITHOUT resizing to 32x32
    transform = tvt.Compose([
        tvt.ToTensor()  # keep original 28x28
    ])
    dataset = MNIST(root='./datasets/mnist', train=train, download=True, transform=transform)

    if digit is None:
        # Return entire dataset if no specific digit is requested
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    # Filter the dataset for only the specified digit
    indices = [i for i, label in enumerate(dataset.targets) if label == digit]
    digit_subset = Subset(dataset, indices)
    
    # Create a DataLoader for this subset
    loader = DataLoader(digit_subset, batch_size=batch_size, shuffle=True)
    return loader
