import torchhd.datasets
from torch.utils.data import DataLoader, Subset
import string

def get_letter_loader(letter, batch_size=64, train=True):
    """
    Creates a DataLoader for a specific letter (A-Z) from the ISOLET dataset.

    Args:
        letter (str): The letter to filter (A-Z).
        batch_size (int): The batch size for the DataLoader.
        train (bool): Whether to load training data (True) or test data (False).

    Returns:
        DataLoader: A PyTorch DataLoader containing only the samples for the specified letter.
    """

    letter = letter.upper()
    letter_index = ord(letter) - ord('A')

    # root_dir = "/Users/jessyjy7/Desktop/Catastrophic-Forgetting/PYTORCHCNNS/datasets" 
    root_dir = "./datasets/isolet"
    dataset = torchhd.datasets.ISOLET(root=root_dir, train=train, download=True)

    indices = [i for i, (_, label) in enumerate(dataset) if label.item() == letter_index]
    letter_subset = Subset(dataset, indices)

    loader = DataLoader(letter_subset, batch_size=batch_size, shuffle=True)
    
    return loader
